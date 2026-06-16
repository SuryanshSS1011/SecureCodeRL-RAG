"""Single-shot training monitor for the v0.1.7 factorial + control cells.

Reads each cell's train_log.jsonl and reports:
    - progress (steps / 1000)
    - last-50-step window averages on reward components
    - canary signals: zero_variance_group_rate, stub_rate, policy_loss_nonzero_rate
    - approximate ETA based on recent step wall time

Usage on ROAR:
    PYTHONPATH=src .venv/bin/python scripts/monitor_factorial_cells.py

Designed as a quick human-readable status check between SLURM polls. Not
a daemon; run on demand. Sorts cells by recent reward so the headline
cell rises to the top once it differentiates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read_cell(out_dir: Path) -> dict:
    log = out_dir / "train_log.jsonl"
    if not log.exists():
        return {"name": out_dir.name, "status": "no log"}
    rows = []
    with log.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return {"name": out_dir.name, "status": "empty log"}

    n = len(rows)
    win = rows[-50:] if n >= 50 else rows

    def mean(key, default=0.0):
        vals = [r.get(key, default) for r in win]
        return sum(vals) / len(vals)

    pl_nz_rate = sum(1 for r in win if r.get("policy_loss_nonzero")) / len(win)
    # Recent step wall time (excludes the cold-start step 0).
    recent_walls = [r.get("wall_s_step", 0) for r in rows[-20:]]
    mean_wall = sum(recent_walls) / max(1, len(recent_walls))
    steps_left = 1000 - n
    eta_hours = (steps_left * mean_wall) / 3600 if mean_wall > 0 else 0

    return {
        "name": out_dir.name,
        "status": "ok",
        "steps": n,
        "progress_pct": 100.0 * n / 1000,
        "mean_reward": mean("mean_reward"),
        "mean_r_rag": mean("mean_r_rag"),
        "mean_r_security": mean("mean_r_security"),
        "mean_r_reliability": mean("mean_r_reliability"),
        "pl_nz_rate": pl_nz_rate,
        "stub_rate": mean("stub_rate"),
        "truncation_rate": mean("truncation_rate"),
        "zero_variance_group_rate": mean("zero_variance_group_rate"),
        "mean_completion_chars": mean("mean_completion_chars"),
        "mean_wall_s_step": mean_wall,
        "eta_hours": eta_hours,
    }


def fmt_row(c: dict) -> str:
    if c.get("status") != "ok":
        return f"  {c['name']:30s}  [{c.get('status', '?')}]"
    return (
        f"  {c['name']:30s}  "
        f"{c['steps']:4d}/1000 ({c['progress_pct']:4.1f}%)  "
        f"R={c['mean_reward']:+.3f}  rag={c['mean_r_rag']:+.3f}  "
        f"sec={c['mean_r_security']:+.3f}  "
        f"pl_nz={c['pl_nz_rate']*100:5.1f}%  "
        f"stub={c['stub_rate']*100:4.1f}%  "
        f"zvg={c['zero_variance_group_rate']*100:5.1f}%  "
        f"ETA={c['eta_hours']:5.1f}h"
    )


def main() -> int:
    roots = [
        Path("/scratch/sss6371/secure-code-rl-ictai-data/v0_1_7"),
        Path("/scratch/sss6371/secure-code-rl-ictai-data/v0_1_7_controls"),
    ]
    cells = []
    for root in roots:
        if not root.exists():
            continue
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and sub.name.startswith("arm_a_"):
                cells.append(read_cell(sub))

    if not cells:
        print("no cells found; check $SCRATCH paths")
        return 1

    # Sort by recent reward (better first), but keep "no log" cells at end.
    ok = [c for c in cells if c.get("status") == "ok"]
    bad = [c for c in cells if c.get("status") != "ok"]
    ok.sort(key=lambda c: -c["mean_reward"])

    print(f"v0.1.7 factorial + controls progress  (n={len(cells)} cells)")
    print("-" * 132)
    for c in ok + bad:
        print(fmt_row(c))

    # Aggregate canary
    rag_cells = [c for c in ok if "_rag" in c["name"] or "_uniform" in c["name"]]
    no_rag_cells = [c for c in ok if c not in rag_cells]
    if rag_cells and no_rag_cells:
        def avg(cs, key):
            return sum(c[key] for c in cs) / len(cs)
        print()
        print("RAG-on  cells (n=%d): R=%+.3f  pl_nz=%.1f%%  zvg=%.1f%%" % (
            len(rag_cells), avg(rag_cells, "mean_reward"),
            avg(rag_cells, "pl_nz_rate")*100,
            avg(rag_cells, "zero_variance_group_rate")*100,
        ))
        print("RAG-off cells (n=%d): R=%+.3f  pl_nz=%.1f%%  zvg=%.1f%%" % (
            len(no_rag_cells), avg(no_rag_cells, "mean_reward"),
            avg(no_rag_cells, "pl_nz_rate")*100,
            avg(no_rag_cells, "zero_variance_group_rate")*100,
        ))
        delta = avg(rag_cells, "mean_reward") - avg(no_rag_cells, "mean_reward")
        print("RAG-on minus RAG-off reward gap: %+.3f" % delta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
