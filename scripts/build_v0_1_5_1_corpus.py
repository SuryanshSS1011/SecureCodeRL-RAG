"""Build v0.1.5.1 corpus = v0.1.5 + authored design-pair patterns.

Extends the frozen v0.1.5 train/val splits with the authored CWE-306 +
CWE-862 patterns (from scripts/author_design_pair_corpus.py). Does NOT
re-run adapters, dedup, or RAG indexing — those are inherited from
v0.1.5. The design-pair patterns get added to BOTH:

  - train_prompts.jsonl  — for the RL trainer's batches
  - design_pair_secure_pairs.jsonl  — for the SFT warm-start trainer

A small held-out split is reserved for validation:

  - design_pair_val_prompts.jsonl  — read by build_v0_1_7_rebalance.py

Split policy: stratified by target_cwe with a 80/20 train/val ratio.
With 50 CWE-862 + 50 CWE-306 patterns → 40+40 train, 10+10 val.

Run BEFORE scripts/extract_sft_pairs.py (which consumes the
design_pair_secure_pairs.jsonl).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _stratified_split(
    patterns: list[dict], val_per_cwe: int, seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Return (train, val) with val_per_cwe records per CWE held out.

    Deterministic by seed; sorts records by id within a CWE before
    slicing so the split is reproducible across re-runs.
    """
    by_cwe: dict[str, list[dict]] = {}
    for r in patterns:
        by_cwe.setdefault(r["target_cwe"], []).append(r)
    for cwe in by_cwe:
        by_cwe[cwe].sort(key=lambda r: r["id"])

    import random
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for cwe, recs in by_cwe.items():
        rng.shuffle(recs)
        val.extend(recs[:val_per_cwe])
        train.extend(recs[val_per_cwe:])
    return train, val


def _to_sft_pair(pattern: dict) -> dict:
    """Project a design-pair pattern to the SFT consumer schema."""
    return {
        "prompt_text": pattern["prompt_text"],
        "secure_completion": pattern["metadata"]["secure_completion"],
        "target_cwe": pattern["target_cwe"],
        "language": pattern["language"],
        "source": pattern.get("source", "design_pair_authored"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v0-1-5-dir", type=Path, required=True,
                   help="Source v0.1.5 build dir (read-only).")
    p.add_argument("--patterns-jsonl", type=Path, required=True,
                   help="Authored design-pair patterns from "
                        "scripts/author_design_pair_corpus.py.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Destination v0.1.5.1 build dir.")
    p.add_argument("--val-per-cwe", type=int, default=10,
                   help="Patterns held out per CWE for the §7.4 gate set. "
                        "Default 10 (= 10+10 val, 40+40 train).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # 1. Read inputs.
    v015_train = _read_jsonl(args.v0_1_5_dir / "train_prompts.jsonl")
    v015_val = _read_jsonl(args.v0_1_5_dir / "val_prompts.jsonl")
    patterns = _read_jsonl(args.patterns_jsonl)
    print(
        f"[build] v0.1.5 train={len(v015_train)} val={len(v015_val)} "
        f"design-pair patterns={len(patterns)}",
        file=sys.stderr, flush=True,
    )

    # 2. Split the design-pair patterns into train + val.
    design_train, design_val = _stratified_split(
        patterns, args.val_per_cwe, seed=args.seed,
    )
    print(
        f"[build] design-pair: train={len(design_train)} val={len(design_val)}",
        file=sys.stderr, flush=True,
    )

    # 3. Compose the new splits. v0.1.5 prompts stay in their original
    # positions; design-pair train prompts get appended. The val set
    # for the trainer's training-time eval stays the original v0.1.5
    # val set (no design-pair pollution into the metric) — the
    # design-pair val is a SEPARATE file consumed by the §7.4 gate
    # only.
    new_train = list(v015_train) + design_train
    new_val = list(v015_val)  # unchanged
    sft_pairs = [_to_sft_pair(r) for r in design_train]

    # 4. Write outputs.
    _write_jsonl(args.output_dir / "train_prompts.jsonl", new_train)
    _write_jsonl(args.output_dir / "val_prompts.jsonl", new_val)
    _write_jsonl(args.output_dir / "design_pair_val_prompts.jsonl", design_val)
    _write_jsonl(args.output_dir / "design_pair_secure_pairs.jsonl", sft_pairs)

    # 5. Manifest.
    manifest = {
        "version": "v0.1.5.1",
        "extends": "v0.1.5",
        "n_train": len(new_train),
        "n_train_v015": len(v015_train),
        "n_train_design_pair": len(design_train),
        "n_val_trainer_eval": len(new_val),
        "n_val_design_pair_gate": len(design_val),
        "n_sft_pairs": len(sft_pairs),
        "design_pair_source": str(args.patterns_jsonl),
        "split_seed": args.seed,
        "val_per_cwe": args.val_per_cwe,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    print(
        f"[build] wrote v0.1.5.1: train={len(new_train)} val={len(new_val)} "
        f"design_pair_val={len(design_val)} sft_pairs={len(sft_pairs)}",
        file=sys.stderr, flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
