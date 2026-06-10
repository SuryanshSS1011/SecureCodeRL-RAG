"""Per-CWE × per-language distinct-pattern yield v0.1.5 (7 sources).

Covers all seven sources, including SecurityEval (eval-side) and
DiverseVul (training-side), and reflects the per-language strict floor
policy locked in docs/scope.md §2.

Sources:
  Training-eligible:    CVEfixes, Juliet (10% synth cap), DiverseVul
  Eval-only:            SecCodePLT, CWEval, CyberSecEval, CASTLE, SecurityEval
  Authored design-pair: tracked separately (not yet authored)
"""

import os
import re
import sys
import json
import sqlite3
from collections import Counter
from pathlib import Path

REPO = Path("/storage/home/sss6371/secure-code-rl-ictai")
DATA = Path("/storage/home/sss6371/work/secure-code-rl-ictai-data/raw")

sys.path.insert(0, str(REPO / "src"))

from secure_code_rl_ictai.data_prep.disjointness import (
    _hash, ast_normalize_python, string_normalize,
)
from secure_code_rl_ictai.data_prep.cyberseceval import (
    CyberSecEvalAdapter, CyberSecEvalConfig,
)
from secure_code_rl_ictai.data_prep.castle import (
    CastleAdapter, CastleConfig,
)
from secure_code_rl_ictai.data_prep.securityeval import (
    SecurityEvalAdapter, SecurityEvalConfig,
)
from secure_code_rl_ictai.data_prep.diversevul import (
    DiverseVulAdapter, DiverseVulConfig,
)


SCOPE_LANGS = {"python", "c", "cpp"}
DESIGN_PAIR = {"CWE-306", "CWE-862"}
DESCRIPTIVE_ONLY = {"CWE-327", "CWE-328", "CWE-326", "CWE-798"}
TRILINGUAL = {"CWE-22", "CWE-78", "CWE-20"}
CPP_NATIVE = {"CWE-787", "CWE-119", "CWE-125", "CWE-416", "CWE-476", "CWE-190"}
PY_NATIVE = {"CWE-89", "CWE-79", "CWE-94", "CWE-502"}
FLOOR_DEFAULT = 20
FLOOR_DESIGN = 50
SYNTHETIC_CAP = 0.10


def load_target_cwes():
    out = []
    with open(REPO / "data" / "ictai_cwe_list.txt") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            out.append(ln)
    return out


def pattern_key(code, language):
    norm = ast_normalize_python(code) if language == "python" else None
    if norm is None:
        norm = string_normalize(code)
    return _hash(norm)


def normalize_cve_lang(s):
    if s is None:
        return None
    s = s.strip().lower()
    if s in ("c++", "cpp", "cplusplus"):
        return "cpp"
    if s == "c":
        return "c"
    if s in ("python", "py"):
        return "python"
    return None


def cvefixes_yield(target_cwes):
    db = DATA / "cvefixes" / "CVEfixes_v1.0.8.db"
    if not db.is_file():
        return {}, Counter(), set()
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in target_cwes)
    rows = cur.execute(f"""
        SELECT cc.cwe_id, fc.programming_language, mc.code
        FROM cve c
        JOIN fixes f          ON f.cve_id = c.cve_id
        JOIN file_change fc   ON fc.hash = f.hash
        JOIN method_change mc ON mc.file_change_id = fc.file_change_id
        JOIN cwe_classification cc ON cc.cwe_id IN ({placeholders})
        WHERE cc.cve_id = c.cve_id
          AND mc.before_change = 'True'
          AND mc.code IS NOT NULL
          AND fc.programming_language IS NOT NULL
    """, list(target_cwes))
    seen = {}
    raw = Counter()
    hashes = set()
    for cwe, lang_raw, code in rows:
        lang = normalize_cve_lang(lang_raw)
        if lang is None or lang not in SCOPE_LANGS:
            continue
        if not code or not code.strip():
            continue
        raw[(cwe, lang)] += 1
        k = pattern_key(code, lang)
        seen.setdefault((cwe, lang), set()).add(k)
        hashes.add(k)
    conn.close()
    return {k: len(v) for k, v in seen.items()}, raw, hashes


def juliet_yield(target_cwes):
    raw = Counter()
    seen = {}
    target_nums = set()
    for cwe in target_cwes:
        try:
            target_nums.add(int(cwe.split("-", 1)[1]))
        except (ValueError, IndexError):
            pass
    root = DATA / "juliet" / "C" / "testcases"
    if not root.is_dir():
        return {}, Counter()
    ext_to_lang = {".c": "c", ".cpp": "cpp"}
    for d in os.listdir(root):
        if not d.startswith("CWE"):
            continue
        m = re.match(r"CWE(\d+)_", d)
        if not m:
            continue
        cwe_num = int(m.group(1))
        if cwe_num not in target_nums:
            continue
        cwe = f"CWE-{cwe_num}"
        sub = root / d
        for dirpath, _, fnames in os.walk(sub):
            for fn in fnames:
                ext = None
                for e in ext_to_lang:
                    if fn.endswith(e):
                        ext = e
                        break
                if ext is None:
                    continue
                lang = ext_to_lang[ext]
                raw[(cwe, lang)] += 1
                try:
                    text = (Path(dirpath) / fn).read_text(errors="ignore")
                except OSError:
                    continue
                if not text.strip():
                    continue
                seen.setdefault((cwe, lang), set()).add(pattern_key(text, lang))
    return {k: len(v) for k, v in seen.items()}, raw


def seccodeplt_yield(target_cwes):
    p = DATA / "seccodeplt_jsonl" / "insecure_coding.jsonl"
    if not p.is_file():
        return {}, Counter()
    seen = {}
    raw = Counter()
    with open(p) as f:
        for ln in f:
            d = json.loads(ln)
            cwe = d.get("category", "")
            lang = (d.get("language") or "").lower()
            if cwe not in target_cwes or lang not in SCOPE_LANGS:
                continue
            md = d.get("metadata") or {}
            gt = md.get("ground_truth")
            code = None
            if isinstance(gt, str):
                try:
                    gt_d = json.loads(gt)
                    code = gt_d.get("vulnerable_code") or gt_d.get("code_before")
                except Exception:
                    pass
            if not code or not str(code).strip():
                code = d.get("task_description", "")
            raw[(cwe, lang)] += 1
            seen.setdefault((cwe, lang), set()).add(pattern_key(str(code), lang))
    return {k: len(v) for k, v in seen.items()}, raw


def cweval_yield(target_cwes):
    root = DATA / "cweval" / "benchmark" / "core"
    if not root.is_dir():
        return {}, Counter()
    fn_re = re.compile(r"cwe_(\d+)_\d+(?:_(c|cpp))?_task\.(py|c|cpp)$")
    seen = {}
    raw = Counter()
    for langd in ("py", "c", "cpp"):
        sub = root / langd
        if not sub.is_dir():
            continue
        for fn in os.listdir(sub):
            m = fn_re.match(fn)
            if not m:
                continue
            cwe = f"CWE-{int(m.group(1))}"
            if cwe not in target_cwes:
                continue
            ext = m.group(3)
            lang = "python" if ext == "py" else ext
            if lang not in SCOPE_LANGS:
                continue
            try:
                text = (sub / fn).read_text(errors="ignore")
            except OSError:
                continue
            if not text.strip():
                continue
            raw[(cwe, lang)] += 1
            seen.setdefault((cwe, lang), set()).add(pattern_key(text, lang))
    return {k: len(v) for k, v in seen.items()}, raw


def cyberseceval_yield(target_cwes):
    cfg = CyberSecEvalConfig(
        instruct_json=DATA / "instruct" / "instruct.json",
        instruct_v2_json=DATA / "instruct" / "instruct-v2.json",
        autocomplete_json=DATA / "autocomplete" / "autocomplete.json",
        target_cwes=frozenset(target_cwes),
    )
    seen = {}
    raw = Counter()
    for p in CyberSecEvalAdapter(cfg).load():
        lang = p.language.value
        cwe = p.target_cwe
        raw[(cwe, lang)] += 1
        seen.setdefault((cwe, lang), set()).add(pattern_key(p.prompt_text, lang))
    return {k: len(v) for k, v in seen.items()}, raw


def castle_yield(target_cwes):
    cfg = CastleConfig(
        json_path=DATA / "CASTLE-Benchmark" / "datasets" / "CASTLE-C250.json",
        target_cwes=frozenset(target_cwes),
    )
    seen = {}
    raw = Counter()
    for p in CastleAdapter(cfg).load():
        lang = p.language.value
        cwe = p.target_cwe
        raw[(cwe, lang)] += 1
        seen.setdefault((cwe, lang), set()).add(pattern_key(p.prompt_text, lang))
    return {k: len(v) for k, v in seen.items()}, raw


def securityeval_yield(target_cwes):
    cfg = SecurityEvalConfig(
        jsonl_path=DATA / "securityeval.jsonl",
        target_cwes=frozenset(target_cwes),
    )
    seen = {}
    raw = Counter()
    for p in SecurityEvalAdapter(cfg).load():
        lang = p.language.value
        cwe = p.target_cwe
        raw[(cwe, lang)] += 1
        seen.setdefault((cwe, lang), set()).add(pattern_key(p.prompt_text, lang))
    return {k: len(v) for k, v in seen.items()}, raw


def diversevul_yield(target_cwes):
    cfg = DiverseVulConfig(target_cwes=frozenset(target_cwes))
    seen = {}
    raw = Counter()
    hashes = set()
    for p in DiverseVulAdapter(cfg).load():
        lang = p.language.value
        cwe = p.target_cwe
        code = p.metadata.get("insecure_code_reference") or p.prompt_text
        raw[(cwe, lang)] += 1
        k = pattern_key(code, lang)
        seen.setdefault((cwe, lang), set()).add(k)
        hashes.add(k)
    return {k: len(v) for k, v in seen.items()}, raw, hashes


def cell_floor_status(cwe, lang, cvefixes_d, juliet_d, diversevul_d):
    cvf = cvefixes_d.get((cwe, lang), 0)
    dv = diversevul_d.get((cwe, lang), 0) if lang in ("c", "cpp") else 0
    real = cvf + dv
    jul = juliet_d.get((cwe, lang), 0) if lang in ("c", "cpp") else 0
    max_synth = int(real * SYNTHETIC_CAP / (1 - SYNTHETIC_CAP)) if real > 0 else 0
    synth_used = min(jul, max_synth)
    capped = real + synth_used
    if cwe in DESCRIPTIVE_ONLY:
        return f"desc ({capped})"
    floor = FLOOR_DESIGN if cwe in DESIGN_PAIR else FLOOR_DEFAULT
    if cwe in TRILINGUAL:
        in_scope = True
    elif cwe in CPP_NATIVE:
        in_scope = (lang in ("c", "cpp"))
    elif cwe in PY_NATIVE:
        in_scope = (lang == "python")
    elif cwe in DESIGN_PAIR:
        in_scope = (lang == "python")
    else:
        in_scope = True
    if not in_scope:
        return f"n/s ({capped})" if capped else "—"
    if capped >= floor:
        return f"OK ({capped})"
    else:
        return f"SHORT by {floor - capped} ({capped})"


def main():
    target_cwes = load_target_cwes()
    target_set = set(target_cwes)
    print(f"target CWEs ({len(target_cwes)}): {target_cwes}", file=sys.stderr)

    print("[v0.1.5 yield] loading 7 sources...", file=sys.stderr)
    cvefixes_d, cvefixes_r, cvefixes_h = cvefixes_yield(target_set)
    print(f"  CVEfixes: {sum(cvefixes_d.values())} distinct", file=sys.stderr)
    juliet_d, juliet_r = juliet_yield(target_set)
    print(f"  Juliet: {sum(juliet_d.values())} distinct files", file=sys.stderr)
    seccodeplt_d, seccodeplt_r = seccodeplt_yield(target_set)
    print(f"  SecCodePLT: {sum(seccodeplt_d.values())} distinct", file=sys.stderr)
    cweval_d, cweval_r = cweval_yield(target_set)
    print(f"  CWEval: {sum(cweval_d.values())} distinct", file=sys.stderr)
    cyberseceval_d, cyberseceval_r = cyberseceval_yield(target_set)
    print(f"  CyberSecEval: {sum(cyberseceval_d.values())} distinct", file=sys.stderr)
    castle_d, castle_r = castle_yield(target_set)
    print(f"  CASTLE: {sum(castle_d.values())} distinct", file=sys.stderr)
    securityeval_d, securityeval_r = securityeval_yield(target_set)
    print(f"  SecurityEval: {sum(securityeval_d.values())} distinct", file=sys.stderr)
    diversevul_d, diversevul_r, diversevul_h = diversevul_yield(target_set)
    print(f"  DiverseVul: {sum(diversevul_d.values())} distinct", file=sys.stderr)

    cvf_dv_overlap = len(cvefixes_h & diversevul_h)
    print(f"  CVEfixes ∩ DiverseVul overlap (by content hash): {cvf_dv_overlap}", file=sys.stderr)

    lines = []
    lines.append("# Per-CWE × per-language distinct-pattern yield — v0.1.5 (7 sources)")
    lines.append("")
    lines.append("Measured 2026-06-11 on ROAR. Per-language strict floor policy per `docs/scope.md §2`.")
    lines.append("")
    lines.append("Sources:")
    lines.append("- **Training-eligible**: CVEfixes (pre-fix + post-fix paired), Juliet (capped at 10% synth), DiverseVul (vulnerable-only).")
    lines.append("- **Eval-only**: SecCodePLT, CWEval, CyberSecEval, CASTLE, SecurityEval.")
    lines.append("- **Authored design-pair**: tracked separately, not in this matrix.")
    lines.append("")
    lines.append("## 1. Training-side distinct-pattern matrix")
    lines.append("")
    lines.append("| CWE | CVEf py | CVEf c | CVEf cpp | Juliet c | Juliet cpp | DV c | DV cpp | Real total py/c/cpp |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cwe in target_cwes:
        py_cvf = cvefixes_d.get((cwe, "python"), 0)
        c_cvf = cvefixes_d.get((cwe, "c"), 0)
        cpp_cvf = cvefixes_d.get((cwe, "cpp"), 0)
        c_jul = juliet_d.get((cwe, "c"), 0)
        cpp_jul = juliet_d.get((cwe, "cpp"), 0)
        c_dv = diversevul_d.get((cwe, "c"), 0)
        cpp_dv = diversevul_d.get((cwe, "cpp"), 0)
        py_total = py_cvf
        c_total = c_cvf + c_jul + c_dv
        cpp_total = cpp_cvf + cpp_jul + cpp_dv
        row = [cwe,
               f"{py_cvf}" if py_cvf else "—",
               f"{c_cvf}" if c_cvf else "—",
               f"{cpp_cvf}" if cpp_cvf else "—",
               f"{c_jul}" if c_jul else "—",
               f"{cpp_jul}" if cpp_jul else "—",
               f"{c_dv}" if c_dv else "—",
               f"{cpp_dv}" if cpp_dv else "—",
               f"{py_total} / {c_total} / {cpp_total}"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 2. Eval-side distinct-pattern matrix")
    lines.append("")
    lines.append("| CWE | SCPLT py | CWEv (py/c/cpp) | CSE (py/c/cpp) | CASTLE c | SecEval py | Eval total py/c/cpp |")
    lines.append("|---|---|---|---|---|---|---|")
    for cwe in target_cwes:
        scp = seccodeplt_d.get((cwe, "python"), 0)
        cwev_py = cweval_d.get((cwe, "python"), 0)
        cwev_c = cweval_d.get((cwe, "c"), 0)
        cwev_cpp = cweval_d.get((cwe, "cpp"), 0)
        cse_py = cyberseceval_d.get((cwe, "python"), 0)
        cse_c = cyberseceval_d.get((cwe, "c"), 0)
        cse_cpp = cyberseceval_d.get((cwe, "cpp"), 0)
        cas_c = castle_d.get((cwe, "c"), 0)
        sec_py = securityeval_d.get((cwe, "python"), 0)
        py = scp + cwev_py + cse_py + sec_py
        c_ = cwev_c + cse_c + cas_c
        cpp = cwev_cpp + cse_cpp
        row = [cwe,
               f"{scp}" if scp else "—",
               f"{cwev_py}/{cwev_c}/{cwev_cpp}",
               f"{cse_py}/{cse_c}/{cse_cpp}",
               f"{cas_c}" if cas_c else "—",
               f"{sec_py}" if sec_py else "—",
               f"{py} / {c_} / {cpp}"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 3. Per-(CWE, language) floor status under per-language strict")
    lines.append("")
    lines.append("Each cell judged independently. Status: OK = clears floor; SHORT = below floor; desc = descriptive-only tier; n/s = not in tier scope.")
    lines.append("")
    lines.append("| CWE | Tier | py status | c status | cpp status |")
    lines.append("|---|---|---|---|---|")
    for cwe in target_cwes:
        if cwe in DESCRIPTIVE_ONLY:
            tier = "T4 desc"
        elif cwe in DESIGN_PAIR:
            tier = "T5 authored"
        elif cwe in TRILINGUAL:
            tier = "T1 trilingual"
        elif cwe in CPP_NATIVE:
            tier = "T2 C/C++-mono"
        elif cwe in PY_NATIVE:
            tier = "T3 Py-mono"
        else:
            tier = "?"
        row = [cwe, tier,
               cell_floor_status(cwe, "python", cvefixes_d, juliet_d, diversevul_d),
               cell_floor_status(cwe, "c", cvefixes_d, juliet_d, diversevul_d),
               cell_floor_status(cwe, "cpp", cvefixes_d, juliet_d, diversevul_d)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 4. Cross-source overlap audit")
    lines.append("")
    lines.append(f"CVEfixes ∩ DiverseVul (content hash): **{cvf_dv_overlap}** patterns appear in both.")
    lines.append(f"- CVEfixes total distinct training patterns: {len(cvefixes_h):,}")
    lines.append(f"- DiverseVul total distinct training patterns: {len(diversevul_h):,}")
    lines.append(f"- Overlap as fraction of CVEfixes: {cvf_dv_overlap/max(1,len(cvefixes_h))*100:.1f}%")
    lines.append(f"- Overlap as fraction of DiverseVul: {cvf_dv_overlap/max(1,len(diversevul_h))*100:.1f}%")
    lines.append("")
    if cvf_dv_overlap > 0:
        lines.append("Both sources pull from public CVE-tagged commits. Intra-train near-duplicate dedup in `scripts/build_v0_1_5.py` catches these by content hash within each (CWE, language) cell.")
    lines.append("")

    out_md = REPO / "docs" / "per_cwe_yield_v0_1_5_full.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}", file=sys.stderr)

    summary = {
        "measured_at": "2026-06-11",
        "scope_languages": sorted(SCOPE_LANGS),
        "tiers": {
            "T1_trilingual_trained": sorted(TRILINGUAL),
            "T2_cpp_mono_trained": sorted(CPP_NATIVE),
            "T3_py_mono_trained": sorted(PY_NATIVE),
            "T4_descriptive_only": sorted(DESCRIPTIVE_ONLY),
            "T5_authored_design_pair": sorted(DESIGN_PAIR),
        },
        "floor_default": FLOOR_DEFAULT,
        "floor_design": FLOOR_DESIGN,
        "synthetic_cap": SYNTHETIC_CAP,
        "cvefixes_distinct": {f"{c}|{l}": n for (c, l), n in cvefixes_d.items()},
        "juliet_distinct": {f"{c}|{l}": n for (c, l), n in juliet_d.items()},
        "seccodeplt_distinct": {f"{c}|{l}": n for (c, l), n in seccodeplt_d.items()},
        "cweval_distinct": {f"{c}|{l}": n for (c, l), n in cweval_d.items()},
        "cyberseceval_distinct": {f"{c}|{l}": n for (c, l), n in cyberseceval_d.items()},
        "castle_distinct": {f"{c}|{l}": n for (c, l), n in castle_d.items()},
        "securityeval_distinct": {f"{c}|{l}": n for (c, l), n in securityeval_d.items()},
        "diversevul_distinct": {f"{c}|{l}": n for (c, l), n in diversevul_d.items()},
        "cvefixes_diversevul_overlap_hashes": cvf_dv_overlap,
    }
    out_json = REPO / "docs" / "per_cwe_yield_v0_1_5_full.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
