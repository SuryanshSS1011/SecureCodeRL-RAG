#!/usr/bin/env python3
"""Build a serialized BM25 + FAISS index from an exemplar_pairs.jsonl.

Inputs:
    --exemplar-pairs <path>   exemplar-pairs JSONL from the corpus build
    --output <dir>            output directory
    --embedder <name>         "hf" (default; sentence-transformers) | "mock"
    --embedder-model <id>     model id when --embedder=hf
    --mock-dim <int>          embedding dim when --embedder=mock

Outputs (in --output/):
    pairs.jsonl       deterministic re-emission of the input (sorted)
    bm25.pickle       pickled Bm25Backend ready to .search()
    faiss.index       faiss-serialized IndexFlatIP over e_pos embeddings
    embeddings.npy    raw embedding matrix (so the index can be rebuilt)
    manifest.json     hashes + embedding_dim + n_pairs + embedder metadata

Idempotent: same input + same embedder version -> byte-identical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from cargo.rag import (
    Bm25Backend,
    ExemplarPair,
    HfEmbedder,
    Language,
)


# ---------------------------------------------------------------------------
# Pair I/O
# ---------------------------------------------------------------------------


def _iter_pairs(path: Path) -> Iterator[ExemplarPair]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield ExemplarPair(
                cwe=rec["cwe"],
                task_signature=rec.get("task_signature", ""),
                e_pos=rec["e_pos"],
                e_neg=rec["e_neg"],
                language=Language(rec.get("language", "python")),
                source=rec.get("source", "unknown"),
                cve_id=rec.get("cve_id"),
                fix_commit=rec.get("fix_commit"),
                metadata=rec.get("metadata", {}),
            )


def _write_pairs(path: Path, pairs: list[ExemplarPair]) -> None:
    with open(path, "w") as fh:
        for p in pairs:
            rec = {
                "cwe": p.cwe,
                "task_signature": p.task_signature,
                "e_pos": p.e_pos,
                "e_neg": p.e_neg,
                "language": p.language.value,
                "source": p.source,
                "cve_id": p.cve_id,
                "fix_commit": p.fix_commit,
                "metadata": p.metadata,
            }
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _mock_embed_text(text: str, dim: int) -> list[float]:
    """Deterministic, hash-based embedding for fixtures.

    Produces vectors that are NOT meaningful semantically but ARE distinct
    per input, which is enough to exercise FAISS indexing + scoring.
    """
    h = hashlib.sha256(text.encode()).digest()
    raw = np.frombuffer(h, dtype=np.uint8)[:dim]
    if len(raw) < dim:
        # Pad by hashing again with a salt.
        extra = hashlib.sha256((text + "_pad").encode()).digest()
        raw = np.concatenate([raw, np.frombuffer(extra, dtype=np.uint8)])[:dim]
    vec = raw.astype(np.float32) / 255.0
    # L2-normalize so FAISS IndexFlatIP scores match cosine.
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def build(args: argparse.Namespace) -> int:
    if not args.exemplar_pairs.exists():
        print(
            f"exemplar pairs file does not exist: {args.exemplar_pairs}",
            file=sys.stderr,
        )
        return 2
    if args.output.exists() and not args.force:
        print(
            f"output directory {args.output} exists; pass --force to overwrite",
            file=sys.stderr,
        )
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    pairs = list(_iter_pairs(args.exemplar_pairs))
    # Determinism: sort pairs before indexing.
    pairs.sort(key=lambda p: (p.cwe, p.source, p.e_neg[:64], p.e_pos[:64]))

    # ---- embeddings ----

    embedder_meta: dict
    if args.embedder == "mock":
        dim = args.mock_dim
        print(f"[rag-index] using mock embedder (dim={dim})", file=sys.stderr)
        e_pos_embeddings = [
            _mock_embed_text(p.e_pos, dim) for p in pairs
        ]
        embedder_meta = {"kind": "mock", "dim": dim}
    elif args.embedder == "hf":
        print(
            f"[rag-index] loading HfEmbedder ({args.embedder_model}) ...",
            file=sys.stderr,
        )
        hfe = HfEmbedder(model_id=args.embedder_model, device=args.device)
        t0 = time.monotonic()
        e_pos_embeddings = hfe.embed_batch([p.e_pos for p in pairs])
        print(
            f"[rag-index] embedded {len(pairs)} e_pos in "
            f"{time.monotonic() - t0:.1f}s",
            file=sys.stderr,
        )
        dim = len(e_pos_embeddings[0]) if e_pos_embeddings else 0
        embedder_meta = {
            "kind": "hf",
            "dim": dim,
            "model_id": args.embedder_model,
        }
    else:
        print(f"unknown embedder: {args.embedder}", file=sys.stderr)
        return 2

    # ---- write pairs ----

    pairs_path = args.output / "pairs.jsonl"
    _write_pairs(pairs_path, pairs)

    # ---- write embeddings as .npy for re-indexing ----

    embeddings_arr = np.asarray(e_pos_embeddings, dtype=np.float32)
    np.save(args.output / "embeddings.npy", embeddings_arr)

    # ---- BM25: build a Bm25Backend and pickle it ----

    bm25 = Bm25Backend(pairs)
    try:
        bm25._ensure_loaded()
    except NotImplementedError:
        print(
            "[rag-index] rank_bm25 not installed; bm25.pickle will hold the "
            "unloaded Bm25Backend. Real use requires rank_bm25 on the "
            "training venv.",
            file=sys.stderr,
        )
    bm25_path = args.output / "bm25.pickle"
    with open(bm25_path, "wb") as fh:
        pickle.dump(bm25, fh)

    # ---- FAISS: write index file ----

    try:
        import faiss

        # Renormalize defensively (same as backend logic).
        norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_emb = embeddings_arr / norms
        index = faiss.IndexFlatIP(dim)
        index.add(norm_emb)
        faiss.write_index(index, str(args.output / "faiss.index"))
    except ImportError:
        print(
            "[rag-index] faiss not installed; writing empty faiss.index "
            "placeholder. Real use requires faiss on the training venv.",
            file=sys.stderr,
        )
        # Write a marker file so downstream sees the absence consistently.
        (args.output / "faiss.index").write_bytes(b"FAISS_PLACEHOLDER")

    # ---- manifest ----

    manifest = {
        "n_pairs": len(pairs),
        "embedding_dim": dim,
        "embedder": embedder_meta,
        "hashes": {
            "pairs.jsonl": _hash_file(pairs_path),
            "bm25.pickle": _hash_file(bm25_path),
            "faiss.index": _hash_file(args.output / "faiss.index"),
            "embeddings.npy": _hash_file(args.output / "embeddings.npy"),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"[rag-index] wrote {len(pairs)} pairs + indexes to {args.output}/",
        file=sys.stderr,
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exemplar-pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--embedder", choices=("hf", "mock"), default="hf")
    p.add_argument(
        "--embedder-model",
        default="BAAI/bge-base-en-v1.5",
        help="HF model id (when --embedder=hf)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="cuda/cpu/auto (when --embedder=hf)",
    )
    p.add_argument(
        "--mock-dim",
        type=int,
        default=768,
        help="Embedding dim when --embedder=mock",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    sys.exit(build(args))


if __name__ == "__main__":
    main()
