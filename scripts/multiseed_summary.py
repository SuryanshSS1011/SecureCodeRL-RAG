"""Multi-seed summary renderer (P1 #264 + #265 + original v0.1.7 cell).

Given a list of (cell_name, aggregate.json) pairs that all reproduce
the same recipe at different seeds, compute mean and std for each
metric so we can drop "headline = X ± Y%" into §6 of the paper.

Reports for each metric:
    n_seeds, mean, std, min, max, individual values

Usage:
    PYTHONPATH=src python scripts/multiseed_summary.py \\
        --paths \\
            seed42:/scratch/.../sweeps/phase1_v0_1_7_headline_step900/arm_a_grpo_rag/aggregate.json \\
            seed1337:/scratch/.../sweeps/.../seed_1337/aggregate.json \\
            seed2024:/scratch/.../sweeps/.../seed_2024/aggregate.json \\
        --output paper/results-data/multiseed_headline.json

The script doesn't care which seeds — it just reads the aggregate.json
files you list. Same approach works for sigma_floor ablation, etc.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


# Metrics we report on (extracted from aggregate["aggregate"]).
# Each is (metric_key, display_name, format).
HEADLINE_METRICS = [
    ("func_sec_at_1__compiles_and_has_tests_and_nonstub", "fs|cht_nonstub", "%.2f%%"),
    ("func_sec_at_1__compiles_and_has_tests", "fs|cht (legacy)", "%.2f%%"),
    ("secure_at_1__compiles", "sec|cmp", "%.2f%%"),
    ("compile_at_1", "compile@1", "%.2f%%"),
    ("stub_rate__compiles", "stub|cmp", "%.2f%%"),
]


def _extract_value(agg_dict: dict, key: str) -> float | None:
    """Pull `value` from the nested metric dict (if present)."""
    if key not in agg_dict:
        return None
    entry = agg_dict[key]
    if isinstance(entry, dict):
        return entry.get("value")
    return entry  # legacy scalar


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--paths", nargs="+", required=True,
        help='Pairs of "label:path" to aggregate.json files. '
             'Example: seed42:/scratch/.../arm_a_grpo_rag/aggregate.json',
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows: list[tuple[str, dict]] = []
    for token in args.paths:
        if ":" not in token:
            print(f"ERROR: expected label:path, got {token!r}", file=sys.stderr)
            return 1
        label, path_str = token.split(":", 1)
        path = Path(path_str)
        if not path.exists():
            print(f"WARN: {path} does not exist; skipping", file=sys.stderr)
            continue
        agg = json.loads(path.read_text())
        # aggregate.json from EvalReport.save() has top-level "aggregate" key
        # holding the per-spec dict.
        rows.append((label, agg.get("aggregate", agg)))

    if not rows:
        print("ERROR: no aggregates loaded", file=sys.stderr)
        return 1

    print(f"Loaded {len(rows)} seeds: {[r[0] for r in rows]}\n")

    out: dict = {
        "n_seeds": len(rows),
        "seeds": [r[0] for r in rows],
        "metrics": {},
    }

    header = f"{'metric':<35s} {'mean':>10s} {'std':>8s} {'min':>10s} {'max':>10s}"
    print(header)
    print("-" * len(header))

    for key, label, fmt in HEADLINE_METRICS:
        vals = []
        per_seed = {}
        for seed_label, agg in rows:
            v = _extract_value(agg, key)
            if v is None:
                continue
            vals.append(v)
            per_seed[seed_label] = v
        if not vals:
            print(f"  {label:<35s}  [missing in all seeds]")
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        as_pct = lambda v: v * 100  # noqa: E731
        mean_pct = as_pct(mean)
        std_pct = as_pct(std)
        print(
            f"{label:<35s} "
            f"{mean_pct:9.2f}% {std_pct:7.2f}% "
            f"{as_pct(min(vals)):9.2f}% {as_pct(max(vals)):9.2f}%"
        )
        out["metrics"][key] = {
            "label": label,
            "n": len(vals),
            "mean": mean,
            "std": std,
            "min": min(vals),
            "max": max(vals),
            "per_seed": per_seed,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
