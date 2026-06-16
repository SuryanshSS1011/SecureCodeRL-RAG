"""Per-CWE paired McNemar test (#246) for the headline RAG-on vs RAG-off claim.

Pre-registered (docs/prereg.md §1): test = paired McNemar on discordant
pairs, two-sided α=0.05, MDE=15pp per CWE. We use the secure-verdict
on each prompt as the per-prompt binary outcome, scored under both arms
of a pair (e.g. arm_a_grpo vs arm_a_grpo_rag) on the SAME eval prompt
with the SAME decoding seed.

Outputs:
    - per-CWE table: n_paired, p01, p10, mcnemar χ², p-value, BH-corrected q
    - LaTeX-ready paper table
    - JSON summary

Usage:
    PYTHONPATH=src python scripts/per_cwe_mcnemar.py \\
        --arm-off /scratch/.../sweeps/<suite>/arm_a_grpo/per_prompt_stream.jsonl \\
        --arm-on  /scratch/.../sweeps/<suite>/arm_a_grpo_rag/per_prompt_stream.jsonl \\
        --output paper/results-data/per_cwe_mcnemar.json \\
        --tex-output paper/results-data/per_cwe_mcnemar.tex

The script aligns the two streams by prompt_id, then groups by CWE and
runs McNemar per group. Discards prompts that appear in only one stream.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


def _load_stream(path: Path) -> dict[str, dict]:
    """Read per_prompt_stream.jsonl, return prompt_id -> record."""
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["prompt_id"]] = d
    return out


def _is_secure(rec: dict) -> bool:
    """Mirror metric: rec is secure iff compiles AND findings_count == 0."""
    if not rec.get("compiles"):
        return False
    return (rec.get("findings_count", 1) or 0) == 0


def _mcnemar(p01: int, p10: int) -> tuple[float, float]:
    """Exact-test χ² with continuity correction (Edwards).

    p01 = off-secure, on-insecure (n cells with this pattern)
    p10 = off-insecure, on-secure
    Returns (chi2, two-sided p-value via the chi-squared CDF with 1 df).
    """
    n_d = p01 + p10
    if n_d == 0:
        return 0.0, 1.0
    chi2 = (abs(p01 - p10) - 1.0) ** 2 / n_d
    if chi2 < 0:
        chi2 = 0.0
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p


def _bh_correct(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns q-values aligned to input."""
    n = len(pvals)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda kv: kv[1])
    qvals = [0.0] * n
    prev_q = 1.0
    for rank, (i, p) in enumerate(reversed(indexed), start=1):
        idx = n - rank + 1
        q = min(prev_q, p * n / idx)
        qvals[i] = q
        prev_q = q
    return qvals


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-off", type=Path, required=True,
                   help="per_prompt_stream.jsonl for the RAG-off arm")
    p.add_argument("--arm-on", type=Path, required=True,
                   help="per_prompt_stream.jsonl for the RAG-on arm")
    p.add_argument("--output", type=Path, required=True,
                   help="JSON summary output path")
    p.add_argument("--tex-output", type=Path, default=None,
                   help="Optional LaTeX table for the paper")
    p.add_argument("--min-pairs", type=int, default=20,
                   help="Skip CWE rows with fewer than N paired prompts")
    args = p.parse_args()

    off = _load_stream(args.arm_off)
    on = _load_stream(args.arm_on)
    common = set(off.keys()) & set(on.keys())
    only_off = set(off.keys()) - set(on.keys())
    only_on = set(on.keys()) - set(off.keys())
    print(f"[mcnemar] off={len(off)} on={len(on)} paired={len(common)}",
          file=sys.stderr)
    if only_off or only_on:
        print(f"[mcnemar] WARN: {len(only_off)} only-off, {len(only_on)} only-on",
              file=sys.stderr)

    # Group by CWE.
    per_cwe: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for pid in common:
        cwe = off[pid].get("target_cwe", "UNKNOWN")
        off_sec = _is_secure(off[pid])
        on_sec = _is_secure(on[pid])
        per_cwe[cwe].append((off_sec, on_sec))

    rows = []
    for cwe, pairs in sorted(per_cwe.items()):
        n = len(pairs)
        n_off_only = sum(1 for o, n_ in pairs if o and not n_)  # off-secure, on-insecure
        n_on_only = sum(1 for o, n_ in pairs if not o and n_)   # off-insecure, on-secure
        n_both_secure = sum(1 for o, n_ in pairs if o and n_)
        n_both_insecure = sum(1 for o, n_ in pairs if not o and not n_)
        off_rate = (n_off_only + n_both_secure) / max(1, n)
        on_rate = (n_on_only + n_both_secure) / max(1, n)
        diff_pp = (on_rate - off_rate) * 100
        chi2, pval = _mcnemar(n_off_only, n_on_only)
        rows.append({
            "cwe": cwe,
            "n_paired": n,
            "off_secure_rate": off_rate,
            "on_secure_rate": on_rate,
            "diff_pp": diff_pp,
            "discordant_off_only": n_off_only,  # off-secure, on-insecure
            "discordant_on_only": n_on_only,    # off-insecure, on-secure
            "concordant_both_secure": n_both_secure,
            "concordant_both_insecure": n_both_insecure,
            "mcnemar_chi2": chi2,
            "p_value": pval,
        })

    # Apply BH correction across CWEs with n_paired >= min_pairs.
    qualifying_idx = [i for i, r in enumerate(rows) if r["n_paired"] >= args.min_pairs]
    qpvals = [rows[i]["p_value"] for i in qualifying_idx]
    qvals = _bh_correct(qpvals)
    for j, i in enumerate(qualifying_idx):
        rows[i]["bh_qvalue"] = qvals[j]
    for r in rows:
        r.setdefault("bh_qvalue", None)
        r["significant_at_005"] = (
            r["bh_qvalue"] is not None and r["bh_qvalue"] < 0.05
        )

    summary = {
        "n_paired_total": len(common),
        "n_cwes": len(per_cwe),
        "n_cwes_qualifying": len(qualifying_idx),
        "min_pairs": args.min_pairs,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[mcnemar] wrote {args.output}", file=sys.stderr)

    # Stdout table.
    hdr = f"{'CWE':<10s} {'n_pair':>6s} {'off%':>7s} {'on%':>7s} {'Δpp':>7s} {'p01':>5s} {'p10':>5s} {'χ²':>7s} {'p':>8s} {'q':>8s} {'sig':>4s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sig = "*" if r["significant_at_005"] else ""
        q_str = f"{r['bh_qvalue']:.4f}" if r["bh_qvalue"] is not None else "  NA"
        print(f"{r['cwe']:<10s} {r['n_paired']:>6d} "
              f"{r['off_secure_rate']*100:>6.2f}% {r['on_secure_rate']*100:>6.2f}% "
              f"{r['diff_pp']:>+6.2f}pp {r['discordant_off_only']:>5d} "
              f"{r['discordant_on_only']:>5d} {r['mcnemar_chi2']:>7.3f} "
              f"{r['p_value']:>8.4f} {q_str:>8s} {sig:>4s}")

    if args.tex_output:
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Per-CWE paired McNemar test (RAG-on vs RAG-off). "
            r"$n_{paired}$ is the number of eval prompts with verdicts in both arms; "
            r"$\Delta$pp = secure-rate difference; $q$ is the BH-corrected p-value. "
            r"$^*$ denotes $q<0.05$ after BH correction across CWEs with $n \geq "
            f"{args.min_pairs}" r"$.}",
            r"\label{tab:per_cwe_mcnemar}",
            r"\begin{tabular}{lrrrrr}",
            r"\hline",
            r"CWE & $n_{paired}$ & off\% & on\% & $\Delta$pp & $q$ \\",
            r"\hline",
        ]
        for r in rows:
            if r["n_paired"] < args.min_pairs:
                continue
            sig = r"$^*$" if r["significant_at_005"] else ""
            qstr = f"{r['bh_qvalue']:.3f}" if r["bh_qvalue"] is not None else "--"
            lines.append(
                f"{r['cwe']} & {r['n_paired']} & "
                f"{r['off_secure_rate']*100:.1f} & "
                f"{r['on_secure_rate']*100:.1f} & "
                f"{r['diff_pp']:+.1f}{sig} & "
                f"{qstr} \\\\"
            )
        lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text("\n".join(lines))
        print(f"[mcnemar] wrote {args.tex_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
