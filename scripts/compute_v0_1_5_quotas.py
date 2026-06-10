"""Compute v0.1.5 per-CWE quotas under the tier policy locked in scope.md §2.

Reads docs/per_cwe_yield_v0_1_5_full.json (output of per_cwe_yield_v0_1_5.py)
and applies:
  - Tier 1 (trilingual trained): CWE-22, CWE-78, CWE-20. Floor = 20 per (CWE, lang).
  - Tier 2 (C/C++-mono trained): 787, 119, 125, 416, 476, 190. Floor = 20 per
    (CWE, lang) in {c, cpp}; Python cells dropped (not in scope).
  - Tier 3 (Python-mono trained): 89, 79, 94, 502. Floor = 20 in Python; C/C++
    cells noted descriptive-only-where-data-exists.
  - Tier 4 (descriptive-only): 327, 328, 326, 798. No floor; not trained.
  - Tier 5 (authored design pair): 306, 862. Floor = 50, Python-targeted.

Cap policy (per docs/training_spec.md §7.2):
  The cap is set by *true diverse supply* (post intra-train dedup), not by a
  multiplier × floor. After dedup, available supply per (CWE, lang) is what
  this script consumes; we cap subsample at min(supply, compute_default=200)
  for non-design CWEs and min(supply, compute_default=500) for the design pair.
  These defaults are derived in docs/training_spec.md §7.7 as compute-side
  bounds (not as imbalance bounds — that role is taken by per-CWE
  gradient-mass-share reweighting).

Synthetic cap: Juliet contribution ≤ 10% of per-(CWE, lang) pool.

Eval-side: take all available (capped at 200 per (CWE, lang, source) to
keep one source from dominating any cell).

Outputs:
  docs/v0_1_5_quotas.md
  docs/v0_1_5_quotas.json
"""

import json
from pathlib import Path

REPO = Path("/storage/home/sss6371/secure-code-rl-ictai")
if not REPO.is_dir():
    REPO = Path(__file__).resolve().parent.parent
YIELD_JSON = REPO / "docs" / "per_cwe_yield_v0_1_5_full.json"


TRILINGUAL = {"CWE-22", "CWE-78", "CWE-20"}
CPP_NATIVE = {"CWE-787", "CWE-119", "CWE-125", "CWE-416", "CWE-476", "CWE-190"}
PY_NATIVE = {"CWE-89", "CWE-79", "CWE-94", "CWE-502"}
DESCRIPTIVE_ONLY = {"CWE-327", "CWE-328", "CWE-326", "CWE-798"}
DESIGN_PAIR = {"CWE-306", "CWE-862"}

FLOOR_DEFAULT = 20
FLOOR_DESIGN = 50
SYNTHETIC_CAP = 0.10
COMPUTE_CAP_DEFAULT = 200       # per-(CWE, lang) compute-side bound
COMPUTE_CAP_DESIGN = 500        # per-(CWE, lang) compute-side bound, design pair
EVAL_PER_SOURCE_CAP = 200       # per-(CWE, lang, source) eval cap
VAL_FRACTION = 0.05             # stratified 5% val split


def in_scope_langs(cwe):
    """Return the set of languages where CWE is trained, per the tier policy.

    Tier 4 (descriptive-only) returns empty — not trained at all.
    Tier 5 (design pair) returns {python} per Phase-1 scope.
    """
    if cwe in TRILINGUAL:
        return {"python", "c", "cpp"}
    if cwe in CPP_NATIVE:
        return {"c", "cpp"}
    if cwe in PY_NATIVE:
        return {"python"}
    if cwe in DESIGN_PAIR:
        return {"python"}
    return set()  # T4 descriptive-only


def tier_of(cwe):
    if cwe in TRILINGUAL:
        return "T1 trilingual"
    if cwe in CPP_NATIVE:
        return "T2 C/C++-mono"
    if cwe in PY_NATIVE:
        return "T3 Py-mono"
    if cwe in DESCRIPTIVE_ONLY:
        return "T4 desc-only"
    if cwe in DESIGN_PAIR:
        return "T5 authored"
    return "unknown"


def main():
    y = json.loads(YIELD_JSON.read_text())
    target_cwes_path = REPO / "data" / "ictai_cwe_list.txt"
    with open(target_cwes_path) as f:
        target_cwes = [ln.strip() for ln in f
                       if ln.strip() and not ln.startswith("#")]

    cvf = y["cvefixes_distinct"]
    jul = y["juliet_distinct"]
    dv = y["diversevul_distinct"]
    scp = y["seccodeplt_distinct"]
    cwev = y["cweval_distinct"]
    cse = y["cyberseceval_distinct"]
    castle = y["castle_distinct"]
    sec = y["securityeval_distinct"]

    def lookup(d, cwe, lang):
        return d.get(f"{cwe}|{lang}", 0)

    rows = []
    total_train = 0
    total_val = 0
    total_eval = 0
    deficits = []

    for cwe in target_cwes:
        floor = FLOOR_DESIGN if cwe in DESIGN_PAIR else FLOOR_DEFAULT
        compute_cap = COMPUTE_CAP_DESIGN if cwe in DESIGN_PAIR else COMPUTE_CAP_DEFAULT
        scope_langs = in_scope_langs(cwe)

        chosen_real = {l: 0 for l in ("python", "c", "cpp")}
        chosen_synth = {l: 0 for l in ("python", "c", "cpp")}
        train_total = 0

        if cwe in DESCRIPTIVE_ONLY:
            # No training. Eval cells reported below.
            pass
        else:
            for lang in scope_langs:
                cvf_n = lookup(cvf, cwe, lang)
                dv_n = lookup(dv, cwe, lang) if lang in ("c", "cpp") else 0
                real_supply = cvf_n + dv_n
                if real_supply == 0:
                    continue
                # Pre-dedup raw supply count. Note: actual training pool will
                # be smaller after intra-train dedup; this is the upper bound.
                real_target = min(real_supply, compute_cap)
                jul_n = lookup(jul, cwe, lang) if lang in ("c", "cpp") else 0
                max_synth = int(real_target * SYNTHETIC_CAP / (1 - SYNTHETIC_CAP))
                synth_used = min(jul_n, max_synth)
                chosen_real[lang] = real_target
                chosen_synth[lang] = synth_used
                train_total += real_target + synth_used

            # Floor check per scope language
            authored_needed = 0
            authored_breakdown = {}
            for lang in scope_langs:
                cap = chosen_real[lang] + chosen_synth[lang]
                if cap < floor:
                    short = floor - cap
                    authored_needed += short
                    authored_breakdown[lang] = short
            if authored_needed > 0:
                deficits.append((cwe, floor, authored_needed, authored_breakdown))

        # Eval side
        eval_per_source = {
            "seccodeplt": {l: min(lookup(scp, cwe, l), EVAL_PER_SOURCE_CAP)
                           for l in ("python", "c", "cpp")},
            "cweval": {l: min(lookup(cwev, cwe, l), EVAL_PER_SOURCE_CAP)
                       for l in ("python", "c", "cpp")},
            "cyberseceval": {l: min(lookup(cse, cwe, l), EVAL_PER_SOURCE_CAP)
                             for l in ("python", "c", "cpp")},
            "castle": {l: min(lookup(castle, cwe, l), EVAL_PER_SOURCE_CAP)
                       for l in ("python", "c", "cpp")},
            "securityeval": {l: min(lookup(sec, cwe, l), EVAL_PER_SOURCE_CAP)
                             for l in ("python", "c", "cpp")},
        }
        eval_per_lang = {
            l: sum(eval_per_source[s][l] for s in eval_per_source)
            for l in ("python", "c", "cpp")
        }
        eval_total = sum(eval_per_lang.values())

        val = int(train_total * VAL_FRACTION + 0.5)
        train_after_val = max(train_total - val, 0)

        total_train += train_after_val
        total_val += val
        total_eval += eval_total

        rows.append({
            "cwe": cwe,
            "tier": tier_of(cwe),
            "scope_langs": sorted(scope_langs),
            "floor": floor if cwe not in DESCRIPTIVE_ONLY else None,
            "compute_cap": compute_cap if cwe not in DESCRIPTIVE_ONLY else None,
            "chosen_real": chosen_real,
            "chosen_synth": chosen_synth,
            "train_total_pre_split": train_total,
            "train": train_after_val,
            "val": val,
            "authored_needed": sum(d[2] for d in deficits if d[0] == cwe),
            "eval_per_source": eval_per_source,
            "eval_per_lang": eval_per_lang,
            "eval_total": eval_total,
        })

    # ---- markdown ----
    lines = []
    lines.append("# v0.1.5 per-CWE quotas (tier policy locked)")
    lines.append("")
    lines.append("Tier policy per `docs/scope.md §2`. Cap policy per `docs/training_spec.md §7.2`.")
    lines.append("")
    lines.append(f"Floors: standard CWE ≥ {FLOOR_DEFAULT} per (CWE, language); design pair ≥ {FLOOR_DESIGN}.")
    lines.append(f"Compute cap: {COMPUTE_CAP_DEFAULT} (standard) / {COMPUTE_CAP_DESIGN} (design pair) per (CWE, language).")
    lines.append(f"Synthetic cap (Juliet): ≤ {SYNTHETIC_CAP*100:.0f}% of per-(CWE, language) pool.")
    lines.append(f"Eval per-source cap: {EVAL_PER_SOURCE_CAP} per (CWE, language, source).")
    lines.append(f"Val fraction: {VAL_FRACTION*100:.0f}% of training pool.")
    lines.append("")
    lines.append("**Important**: training quotas in this doc are *pre-dedup* upper bounds. Intra-train near-duplicate dedup in `scripts/build_v0_1_5.py` reduces actual supply by ~30-85% in heavily-forked CVEfixes/DiverseVul cells. Final v0.1.5 train counts will be lower than the numbers in §2; the build script enforces this ordering.")
    lines.append("")

    lines.append("## 1. Training quotas (pre-dedup upper bounds)")
    lines.append("")
    lines.append("| CWE | Tier | Scope langs | Floor | py (real+synth) | c (real+synth) | cpp (real+synth) | Train total | Val | Authored deficit |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["tier"].startswith("T4"):
            cells = ["—", "—", "—"]
        else:
            cells = []
            for lang in ("python", "c", "cpp"):
                if lang in r["scope_langs"]:
                    cells.append(f"{r['chosen_real'][lang]}+{r['chosen_synth'][lang]}")
                else:
                    cells.append("n/s")
        floor_s = str(r["floor"]) if r["floor"] is not None else "—"
        authored_s = str(r["authored_needed"]) if r["authored_needed"] > 0 else "—"
        lines.append(f"| {r['cwe']} | {r['tier']} | {','.join(r['scope_langs']) or '—'} | {floor_s} | {cells[0]} | {cells[1]} | {cells[2]} | {r['train']} | {r['val']} | {authored_s} |")
    lines.append("")

    lines.append("## 2. Eval quotas")
    lines.append("")
    lines.append("| CWE | SCPLT py | CWEv py/c/cpp | CSE py/c/cpp | CASTLE c | SecEval py | Eval total py/c/cpp |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        eps = r["eval_per_source"]
        cwev_s = f"{eps['cweval']['python']}/{eps['cweval']['c']}/{eps['cweval']['cpp']}"
        cse_s = f"{eps['cyberseceval']['python']}/{eps['cyberseceval']['c']}/{eps['cyberseceval']['cpp']}"
        eval_s = f"{r['eval_per_lang']['python']}/{r['eval_per_lang']['c']}/{r['eval_per_lang']['cpp']}"
        lines.append(f"| {r['cwe']} | {eps['seccodeplt']['python'] or '—'} | {cwev_s} | {cse_s} | {eps['castle']['c'] or '—'} | {eps['securityeval']['python'] or '—'} | {eval_s} |")
    lines.append("")

    lines.append("## 3. Roll-up")
    lines.append("")
    lines.append(f"- **Train (pre-dedup)**: {total_train:,} distinct patterns")
    lines.append(f"- **Val (pre-dedup)**: {total_val:,}")
    lines.append(f"- **Eval**: {total_eval:,}")
    lines.append(f"- **Total v0.1.5 corpus (pre-dedup)**: {total_train + total_val + total_eval:,} distinct prompts")
    lines.append("")
    if deficits:
        total_authored_need = sum(d[2] for d in deficits)
        lines.append(f"- **Authored deficit**: {total_authored_need} patterns across {len(deficits)} CWEs")
        lines.append("")
        for cwe, floor, need, breakdown in deficits:
            parts = ", ".join(f"{lang}: {n}" for lang, n in breakdown.items())
            lines.append(f"  - {cwe} (floor {floor}): {need} patterns needed — {parts}")
        lines.append("")

    out_md = REPO / "docs" / "v0_1_5_quotas.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")

    summary = {
        "policy": {
            "floor_default": FLOOR_DEFAULT,
            "floor_design": FLOOR_DESIGN,
            "compute_cap_default": COMPUTE_CAP_DEFAULT,
            "compute_cap_design": COMPUTE_CAP_DESIGN,
            "synthetic_cap": SYNTHETIC_CAP,
            "eval_per_source_cap": EVAL_PER_SOURCE_CAP,
            "val_fraction": VAL_FRACTION,
        },
        "tiers": {
            "T1_trilingual": sorted(TRILINGUAL),
            "T2_cpp_mono": sorted(CPP_NATIVE),
            "T3_py_mono": sorted(PY_NATIVE),
            "T4_descriptive_only": sorted(DESCRIPTIVE_ONLY),
            "T5_authored_design": sorted(DESIGN_PAIR),
        },
        "rows": rows,
        "totals": {
            "train_pre_dedup": total_train,
            "val_pre_dedup": total_val,
            "eval": total_eval,
            "authored_deficit": sum(d[2] for d in deficits),
        },
    }
    out_json = REPO / "docs" / "v0_1_5_quotas.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    print()
    print(f"Train: {total_train:,}  Val: {total_val:,}  Eval: {total_eval:,}  Total: {total_train+total_val+total_eval:,}")
    print(f"Authored deficit: {sum(d[2] for d in deficits)} patterns across {len(deficits)} CWEs")


if __name__ == "__main__":
    main()
