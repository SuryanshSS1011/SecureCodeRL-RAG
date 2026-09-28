"""Load a HybridRetriever from a built-on-disk RAG index directory.

`scripts/build_rag_index.py` writes:
  pairs.jsonl     deterministic re-emission of ExemplarPair records
  bm25.pickle     pickled Bm25Backend (search-ready)
  faiss.index     faiss-serialized IndexFlatIP (search-ready)
  embeddings.npy  raw embedding matrix (one row per pair, same order)
  manifest.json   integrity hashes + embedder metadata

This loader reconstructs a `HybridRetriever` ready to call `.retrieve(query)`.
We pull pairs from `pairs.jsonl`, unpickle the BM25 backend, and rebuild the
FAISS backend by feeding the saved embedding matrix back to `FaissBackend`
via its `pair_embeddings=` constructor arg. We do NOT load the serialized
`faiss.index` directly — `FaissBackend` rebuilds the in-memory index from
the embedding matrix at first `.search()`, which is fast (~0.5s for 12k
pairs at 768-dim) and avoids a faiss version-compat path.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .embedder import HfEmbedder
from .retriever import FaissBackend, HybridRetriever
from .schema import ExemplarPair, Language


def _iter_pairs(path: Path) -> list[ExemplarPair]:
    out: list[ExemplarPair] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(
                ExemplarPair(
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
            )
    return out


def load_retriever(
    index_dir: Path,
    mode: str = "best",
    rng_seed: int = 0,
) -> HybridRetriever:
    """Construct a HybridRetriever from a directory built by build_rag_index.py.

    mode: "best" (default, v0.1 behavior) | "adversarial" | "random".
    rng_seed: seed for the per-retriever Random when mode="random".

    Raises FileNotFoundError if any of the required artifacts are missing.
    """
    pairs_path = index_dir / "pairs.jsonl"
    bm25_path = index_dir / "bm25.pickle"
    embeddings_path = index_dir / "embeddings.npy"

    if not pairs_path.exists():
        raise FileNotFoundError(f"missing {pairs_path}")
    if not bm25_path.exists():
        raise FileNotFoundError(f"missing {bm25_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"missing {embeddings_path}")

    pairs = _iter_pairs(pairs_path)

    with open(bm25_path, "rb") as fh:
        bm25 = pickle.load(fh)

    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != len(pairs):
        raise ValueError(
            f"embeddings.npy has {embeddings.shape[0]} rows but "
            f"pairs.jsonl has {len(pairs)} pairs"
        )
    pair_embeddings = embeddings.tolist()
    dense = FaissBackend(pairs, pair_embeddings=pair_embeddings)

    retr = HybridRetriever(
        pairs, bm25_backend=bm25, dense_backend=dense, mode=mode,
    )
    if mode == "random" and hasattr(retr, "_rng"):
        import random as _random
        retr._rng = _random.Random(rng_seed)
    return retr


def load_embedder(manifest_path: Path, device: str = "cuda") -> HfEmbedder:
    """Construct the same HfEmbedder used to build the index, per manifest.

    `device` defaults to "cuda" for production training/eval; pass "cpu"
    on a login node to inspect/smoke the index without GPU access.
    """
    manifest = json.loads(manifest_path.read_text())
    embedder_meta = manifest.get("embedder", {})
    model_id = embedder_meta.get(
        "model_id",
        manifest.get("embedder_model", "BAAI/bge-base-en-v1.5"),
    )
    return HfEmbedder(model_id=model_id, device=device)
