#!/usr/bin/env python3
"""Build v0.1.5 corpus under the tier policy + cap discipline.

Pipeline (strict ordering per docs/training_spec.md §7 header dependency block):

  1. Load all 7 source adapters.
  2. Per-(CWE, language) tier classification per docs/scope.md §2.
  3. Bucket candidates by (CWE, language). Drop out-of-tier cells.
  4. INTRA-TRAIN near-duplicate dedup within each (CWE, language) cell.
  5. Apply per-(CWE, language) compute cap from docs/v0_1_5_quotas.json.
  6. Apply synth cap (Juliet ≤ 10% per cell).
  7. Build eval pool from eval-only sources, apply per-source eval cap.
  8. Train/eval DISJOINTNESS audit: eval hashed first, training candidates
     dropped on hash collision.
  9. Stratified val split per (CWE, language) at 5% of post-quota training pool.
 10. Write manifest with SHA-256 of every file + per-prompt provenance +
     per-(CWE, language, source) histogram.

Output: build/v0.1.5/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from cargo.data_prep import (
    CvefixesAdapter, CvefixesConfig,
    CwevalAdapter, CwevalConfig,
    DisjointnessAudit,
    JulietAdapter, JulietConfig,
    SecCodePltAdapter, SecCodePltConfig,
)
from cargo.data_prep.cyberseceval import (
    CyberSecEvalAdapter, CyberSecEvalConfig,
)
from cargo.data_prep.castle import (
    CastleAdapter, CastleConfig,
)
from cargo.data_prep.securityeval import (
    SecurityEvalAdapter, SecurityEvalConfig,
)
from cargo.data_prep.diversevul import (
    DiverseVulAdapter, DiverseVulConfig,
)
from cargo.data_prep.disjointness import (
    _hash, ast_normalize_python, string_normalize,
)
from cargo.data_prep.prompt_template import (
    PromptNormalizer, PromptNormalizerConfig,
)


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_v0_1_5")


TRILINGUAL = {"CWE-22", "CWE-78", "CWE-20"}
CPP_NATIVE = {"CWE-787", "CWE-119", "CWE-125", "CWE-416", "CWE-476", "CWE-190"}
PY_NATIVE = {"CWE-89", "CWE-79", "CWE-94", "CWE-502"}
DESCRIPTIVE_ONLY = {"CWE-327", "CWE-328", "CWE-326", "CWE-798"}
DESIGN_PAIR = {"CWE-306", "CWE-862"}


def in_scope_langs(cwe):
    if cwe in TRILINGUAL:
        return {"python", "c", "cpp"}
    if cwe in CPP_NATIVE:
        return {"c", "cpp"}
    if cwe in PY_NATIVE:
        return {"python"}
    if cwe == "CWE-306":
        return {"python"}
    if cwe == "CWE-862":
        # Python primary (where eval lives via SecCodePLT 45 items + SecurityEval).
        # C is added training-only: DiverseVul supplies 47 + CVEfixes 7 = ~54 raw
        # vulnerable-function patterns. These contribute to warm-start variance
        # (failure mode A relief — lifting p_c above 0 for the SFT seed) but are
        # NOT separately evaluated, because no C authz benchmark exists in our
        # eval pool. The decision and its threats-to-validity framing live in
        # docs/scope.md §2 Tier 5.
        return {"python", "c"}
    return set()


def pattern_key(code, language):
    norm = ast_normalize_python(code) if language == "python" else None
    if norm is None:
        norm = string_normalize(code)
    return _hash(norm)


def _serialize_prompt(p):
    d = asdict(p)
    d["language"] = p.language.value
    d["test_spec"]["language"] = p.test_spec.language.value
    return d


def _serialize_pair(pair):
    d = asdict(pair)
    d["language"] = pair.language.value if hasattr(pair.language, "value") else str(pair.language)
    return d


def _write_jsonl(path, records):
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _hash_file(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def apply_compute_cap(prompts, per_cell_caps, seed):
    by_bucket = defaultdict(list)
    for p in prompts:
        by_bucket[(p.target_cwe, p.language.value)].append(p)
    chosen = []
    for (cwe, lang), group in by_bucket.items():
        cap = per_cell_caps.get((cwe, lang), 0)
        if cap <= 0:
            continue
        if len(group) <= cap:
            chosen.extend(group)
        else:
            group.sort(key=lambda p: p.id)
            rng = random.Random(seed ^ hash((cwe, lang)))
            indices = list(range(len(group)))
            rng.shuffle(indices)
            chosen.extend(group[i] for i in indices[:cap])
    return chosen


def _hash_dedup_by_cell(prompts):
    """Step 1 of intra-train dedup: exact content-hash dedup per cell.

    Removes prompts whose pattern_hash collides with another in the same
    (CWE, language) bucket. Conservative — catches only exact dupes after
    AST/string normalization.
    """
    seen_by_cell = defaultdict(set)
    kept = []
    for p in prompts:
        key = pattern_key(p.prompt_text, p.language.value)
        cell = (p.target_cwe, p.language.value)
        if key in seen_by_cell[cell]:
            continue
        seen_by_cell[cell].add(key)
        kept.append(p)
    return kept


def _near_dup_dedup_by_cell(prompts, threshold=0.85, top_k=5):
    """Step 2 of intra-train dedup: per-cell BM25-narrowed Levenshtein dedup.

    Within each (CWE, language) cell, build a BM25 index, find the top-K
    BM25 nearest neighbors for each prompt, and drop any prompt whose
    nearest-neighbor similarity is >= `threshold`. Uses rapidfuzz if
    available (10-50x faster than difflib).

    Threshold rationale: docs/training_spec.md §7.7's intra-train
    threshold sits in the high-sim valley near 1.0; 0.85 is the
    conservative default that catches the CVEfixes-vendoring fork-dup
    spike; >85% of CVEfixes blobs in memory CWEs are near-dups at
    sim >= 0.85.

    Order-stable: lower-id prompts are kept, higher-id duplicates dropped.
    """
    try:
        from rapidfuzz import fuzz as rf_fuzz
        def sim(a, b):
            return rf_fuzz.ratio(a, b) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        def sim(a, b):
            return SequenceMatcher(a=a, b=b).quick_ratio()

    try:
        from rank_bm25 import BM25Okapi
        def build_bm25(corpus_toks):
            return BM25Okapi(corpus_toks)
        def score_bm25(idx, query_toks):
            return idx.get_scores(query_toks)
    except ImportError:
        from collections import Counter as _C
        class _TF:
            def __init__(self, docs):
                self._tfs = [_C(d) for d in docs]
            def get_scores(self, q):
                return [sum(tf[t] for t in q) for tf in self._tfs]
        def build_bm25(corpus_toks):
            return _TF(corpus_toks)
        def score_bm25(idx, query_toks):
            return idx.get_scores(query_toks)

    import re as _re
    _strip_re = _re.compile(r"\s+")
    _ident_re = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    def _strip(text):
        return _strip_re.sub(" ", text).strip().lower()

    def _tok(text):
        return _ident_re.findall(text.lower())

    by_cell = defaultdict(list)
    for p in prompts:
        by_cell[(p.target_cwe, p.language.value)].append(p)

    kept_total = []
    dropped_total = 0
    for cell, group in by_cell.items():
        if len(group) < 2:
            kept_total.extend(group)
            continue
        # Sort by id for stable selection
        group.sort(key=lambda p: p.id)
        normed = [_strip(p.prompt_text) for p in group]
        tokens = [_tok(n) for n in normed]
        if not any(tokens):
            kept_total.extend(group)
            continue
        bm25 = build_bm25(tokens)
        dropped = set()
        for i in range(len(group)):
            if i in dropped:
                continue
            scores = list(score_bm25(bm25, tokens[i]))
            scores[i] = -1.0
            for j in range(len(group)):
                if scores[j] <= 0:
                    continue
            ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
            for j, score in ranked:
                if j == i or j in dropped:
                    continue
                if sim(normed[i], normed[j]) >= threshold:
                    dropped.add(j)
        for i, p in enumerate(group):
            if i not in dropped:
                kept_total.append(p)
        dropped_total += len(dropped)
        if len(group) >= 20:
            logger.info(
                f"  near-dup dedup {cell}: kept {len(group)-len(dropped)}/{len(group)} "
                f"(dropped {len(dropped)})"
            )
    logger.info(
        f"near-dup dedup: kept {len(kept_total)} prompts "
        f"(dropped {dropped_total} of {len(prompts)})"
    )
    return kept_total


def apply_intra_train_dedup(prompts, near_dup_threshold=0.85):
    """Two-step intra-train dedup: content-hash then BM25-narrowed near-dup.

    Step 1 (hash) is cheap and catches exact post-normalization dupes.
    Step 2 (BM25 + Levenshtein) catches CVEfixes/DiverseVul fork-dup
    near-misses at sim >= threshold within each (CWE, language) cell.
    Per docs/training_spec.md §7.7, the threshold defaults to 0.85
    (conservative; v0.1.4 audit showed >85% of memory-CWE C blobs are
    near-dups at this level).
    """
    after_hash = _hash_dedup_by_cell(prompts)
    logger.info(f"after content-hash dedup: {len(after_hash)} prompts (was {len(prompts)})")
    after_neardup = _near_dup_dedup_by_cell(after_hash, threshold=near_dup_threshold)
    return after_neardup


def build(args):
    out = args.output
    if out.exists() and not args.force:
        print(f"output {out} exists; pass --force to overwrite", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    quotas = json.loads(args.quotas.read_text())
    raw_root = args.raw_root

    real_caps = {}
    synth_caps = {}
    eval_per_source_caps = {
        s: {} for s in ("seccodeplt", "cweval", "cyberseceval", "castle", "securityeval")
    }
    for r in quotas["rows"]:
        cwe = r["cwe"]
        for lang in ("python", "c", "cpp"):
            if lang in r.get("scope_langs", []):
                real_caps[(cwe, lang)] = r["chosen_real"].get(lang, 0)
                synth_caps[(cwe, lang)] = r["chosen_synth"].get(lang, 0)
        for src in eval_per_source_caps:
            for lang in ("python", "c", "cpp"):
                eval_per_source_caps[src][(cwe, lang)] = r["eval_per_source"][src].get(lang, 0)

    target_cwes_path = Path("/storage/home/sss6371/secure-code-rl-ictai/data/ictai_cwe_list.txt")
    if not target_cwes_path.is_file():
        target_cwes_path = Path(__file__).resolve().parent.parent / "data" / "ictai_cwe_list.txt"
    with open(target_cwes_path) as f:
        target_cwes = frozenset(
            ln.strip() for ln in f if ln.strip() and not ln.startswith("#")
        )

    seed = int(hashlib.sha256(args.build_version.encode()).hexdigest()[:8], 16)

    logger.info("loading CVEfixes (training-eligible)...")
    cvefixes_prompts = []
    cvefixes_pairs = []
    cvf_jsonl_dir = raw_root / "cvefixes_jsonl"
    if cvf_jsonl_dir.exists():
        cvf_adapter = CvefixesAdapter(CvefixesConfig(
            jsonl_dir=cvf_jsonl_dir,
            target_cwes=target_cwes,
        ))
        for p in cvf_adapter.load():
            cvefixes_prompts.append(p)
        try:
            for pair in cvf_adapter.load_exemplar_pairs():
                cvefixes_pairs.append(pair)
        except (NotImplementedError, TypeError):
            pass
    logger.info(f"  CVEfixes: prompts={len(cvefixes_prompts)} pairs={len(cvefixes_pairs)}")

    logger.info("loading Juliet (training-eligible exemplars)...")
    juliet_pairs = []
    juliet_root = raw_root / "juliet"
    if juliet_root.exists():
        jul_adapter = JulietAdapter(JulietConfig(
            juliet_root=juliet_root,
            target_cwes=target_cwes,
        ))
        try:
            for pair in jul_adapter.load_exemplar_pairs():
                juliet_pairs.append(pair)
        except (NotImplementedError, TypeError):
            pass
    logger.info(f"  Juliet: pairs={len(juliet_pairs)}")

    logger.info("loading DiverseVul (training-eligible prompts)...")
    diversevul_prompts = []
    try:
        dv_adapter = DiverseVulAdapter(DiverseVulConfig(
            target_cwes=target_cwes,
        ))
        for p in dv_adapter.load():
            diversevul_prompts.append(p)
    except Exception as e:
        logger.warning(f"DiverseVul load failed: {e}")
    logger.info(f"  DiverseVul: prompts={len(diversevul_prompts)}")

    train_candidates_real = cvefixes_prompts + diversevul_prompts
    logger.info(f"combined training-candidate (real) pool: {len(train_candidates_real)}")

    in_scope_candidates = []
    for p in train_candidates_real:
        if p.target_cwe in DESCRIPTIVE_ONLY:
            continue
        if p.language.value not in in_scope_langs(p.target_cwe):
            continue
        in_scope_candidates.append(p)
    logger.info(f"after tier-scope filter: {len(in_scope_candidates)} prompts")

    deduped_real = apply_intra_train_dedup(in_scope_candidates)
    capped_real_prompts = apply_compute_cap(deduped_real, real_caps, seed)
    logger.info(f"after compute cap (real): {len(capped_real_prompts)} prompts")

    eval_prompts = []
    for src_name, factory in (
        ("seccodeplt", lambda: SecCodePltAdapter(SecCodePltConfig(
            jsonl_path=raw_root / "seccodeplt_jsonl" / "insecure_coding.jsonl",
            target_cwes=target_cwes,
        ))),
        ("cweval", lambda: CwevalAdapter(CwevalConfig(
            cweval_root=raw_root / "cweval",
            target_cwes=target_cwes,
        ))),
        ("cyberseceval", lambda: CyberSecEvalAdapter(CyberSecEvalConfig(
            instruct_json=raw_root / "instruct" / "instruct.json",
            instruct_v2_json=raw_root / "instruct" / "instruct-v2.json",
            autocomplete_json=raw_root / "autocomplete" / "autocomplete.json",
            target_cwes=target_cwes,
        ))),
        ("castle", lambda: CastleAdapter(CastleConfig(
            json_path=raw_root / "CASTLE-Benchmark" / "datasets" / "CASTLE-C250.json",
            target_cwes=target_cwes,
        ))),
        ("securityeval", lambda: SecurityEvalAdapter(SecurityEvalConfig(
            jsonl_path=raw_root / "securityeval.jsonl",
            target_cwes=target_cwes,
        ))),
    ):
        try:
            adapter = factory()
            src_prompts = list(adapter.load())
        except Exception as e:
            logger.warning(f"{src_name} adapter failed: {e}")
            continue
        capped = apply_compute_cap(src_prompts, eval_per_source_caps[src_name],
                                    seed ^ hash(src_name))
        logger.info(f"  {src_name}: loaded {len(src_prompts)} chose {len(capped)}")
        eval_prompts.extend(capped)

    logger.info("disjointness audit (eval-first)...")
    audit = DisjointnessAudit()
    for p in eval_prompts:
        audit.add_eval_blob(p.prompt_text, language_hint=p.language.value)
    surviving_train = []
    n_dropped = 0
    for p in capped_real_prompts:
        dropped = audit.audit_train_blob(
            p.prompt_text, source=p.source, language_hint=p.language.value,
        )
        if dropped:
            n_dropped += 1
        else:
            surviving_train.append(p)
    logger.info(f"disjointness: dropped {n_dropped} training prompts on collision")

    juliet_by_cell = defaultdict(list)
    for pair in juliet_pairs:
        lang_val = pair.language.value if hasattr(pair.language, "value") else str(pair.language)
        if pair.cwe in DESCRIPTIVE_ONLY:
            continue
        if lang_val not in in_scope_langs(pair.cwe):
            continue
        juliet_by_cell[(pair.cwe, lang_val)].append(pair)
    chosen_juliet = []
    for cell, group in juliet_by_cell.items():
        cap = synth_caps.get(cell, 0)
        if cap <= 0:
            continue
        group.sort(key=lambda p: (p.e_neg[:64], p.e_pos[:64]))
        rng = random.Random(seed ^ hash(("juliet", cell)))
        idx = list(range(len(group)))
        rng.shuffle(idx)
        chosen_juliet.extend(group[i] for i in idx[:cap])
    logger.info(f"Juliet pairs after synth cap: {len(chosen_juliet)} (from {len(juliet_pairs)})")

    surviving_pairs = []
    for pair in list(cvefixes_pairs) + chosen_juliet:
        lang_val = pair.language.value if hasattr(pair.language, "value") else str(pair.language)
        if pair.cwe in DESCRIPTIVE_ONLY:
            continue
        if audit.audit_train_blob(pair.e_pos, source=pair.source, language_hint=lang_val):
            continue
        if audit.audit_train_blob(pair.e_neg, source=pair.source, language_hint=lang_val):
            continue
        surviving_pairs.append(pair)
    logger.info(f"surviving exemplar pairs after eval-disjointness: {len(surviving_pairs)}")

    val_fraction = quotas["policy"]["val_fraction"]
    by_cell = defaultdict(list)
    for p in surviving_train:
        by_cell[(p.target_cwe, p.language.value)].append(p)
    final_train = []
    final_val_candidates = []
    for (cwe, lang), group in by_cell.items():
        group.sort(key=lambda p: p.id)
        n_val = int(len(group) * val_fraction + 0.5)
        final_val_candidates.extend(group[:n_val])
        final_train.extend(group[n_val:])
    logger.info(f"stratified val split (pre-leak-audit): train={len(final_train)} val_candidates={len(final_val_candidates)}")

    # Train ⊥ val per-cell near-duplicate audit. The earlier intra-train
    # dedup operates BEFORE the val split, so fork-vendor pairs at sim
    # 0.85-0.99 can survive the cell-level cut and end up split across
    # train and val (e.g., id-A in val, id-B in train, sim=0.88). This
    # audit drops val items whose nearest train neighbor in the same
    # (CWE, language) cell exceeds the intra-train threshold (0.85).
    train_val_threshold = 0.85
    try:
        from rapidfuzz import fuzz as _rf_fuzz
        def _sim(a, b):
            return _rf_fuzz.ratio(a, b) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        def _sim(a, b):
            return SequenceMatcher(a=a, b=b).quick_ratio()

    train_texts_by_cell = defaultdict(list)
    for p in final_train:
        train_texts_by_cell[(p.target_cwe, p.language.value)].append(p.prompt_text)
    final_val = []
    n_val_dropped = 0
    val_leak_log = []
    for p in final_val_candidates:
        cell = (p.target_cwe, p.language.value)
        train_texts = train_texts_by_cell.get(cell, [])
        best_sim = 0.0
        for t_text in train_texts:
            s = _sim(p.prompt_text, t_text)
            if s > best_sim:
                best_sim = s
                if best_sim >= train_val_threshold:
                    break
        if best_sim >= train_val_threshold:
            n_val_dropped += 1
            val_leak_log.append({
                "prompt_id": p.id,
                "cwe": p.target_cwe,
                "language": p.language.value,
                "max_sim_to_train": round(best_sim, 4),
            })
        else:
            final_val.append(p)
    logger.info(
        f"train ⊥ val near-dup audit: dropped {n_val_dropped} val items "
        f"at sim >= {train_val_threshold}; final val = {len(final_val)}"
    )

    final_train.sort(key=lambda p: p.id)
    final_val.sort(key=lambda p: p.id)
    eval_prompts.sort(key=lambda p: p.id)
    surviving_pairs.sort(key=lambda p: (p.cwe, p.source, p.e_neg[:64], p.e_pos[:64]))

    # Centralized prompt-format normalization. Wraps bare signatures (CVEfixes,
    # partial SecCodePLT, etc.) with instruction + code-fence directive.
    # Idempotent on already-well-formed prompts (CASTLE, DiverseVul,
    # CyberSecEval pass through). See FINDINGS_LOG 2026-06-14.
    norm_cfg = PromptNormalizerConfig()
    norm_train = PromptNormalizer(norm_cfg)
    norm_val = PromptNormalizer(norm_cfg)
    norm_eval = PromptNormalizer(norm_cfg)
    final_train = [norm_train(p) for p in final_train]
    final_val = [norm_val(p) for p in final_val]
    eval_prompts = [norm_eval(p) for p in eval_prompts]
    logger.info(
        "prompt normalize: train wrapped %d/%d (%.1f%%), val %d/%d (%.1f%%), "
        "eval %d/%d (%.1f%%)",
        norm_train.stats()["wrapped"], norm_train.stats()["total"],
        100 * norm_train.stats()["wrap_rate"],
        norm_val.stats()["wrapped"], norm_val.stats()["total"],
        100 * norm_val.stats()["wrap_rate"],
        norm_eval.stats()["wrapped"], norm_eval.stats()["total"],
        100 * norm_eval.stats()["wrap_rate"],
    )

    train_path = out / "train_prompts.jsonl"
    val_path = out / "val_prompts.jsonl"
    eval_path = out / "eval_prompts.jsonl"
    pairs_path = out / "exemplar_pairs.jsonl"

    _write_jsonl(train_path, [_serialize_prompt(p) for p in final_train])
    _write_jsonl(val_path, [_serialize_prompt(p) for p in final_val])
    _write_jsonl(eval_path, [_serialize_prompt(p) for p in eval_prompts])
    _write_jsonl(pairs_path, [_serialize_pair(p) for p in surviving_pairs])

    report = audit.report()
    src_count = defaultdict(int)
    for p in final_train:
        src_count[f"train::{p.source}"] += 1
    for p in final_val:
        src_count[f"val::{p.source}"] += 1
    for p in eval_prompts:
        src_count[f"eval::{p.source}"] += 1

    def dist(prompts):
        d = defaultdict(lambda: defaultdict(int))
        for p in prompts:
            d[p.target_cwe][p.language.value] += 1
        return {k: dict(v) for k, v in d.items()}

    def pair_dist(pairs):
        d = defaultdict(lambda: defaultdict(int))
        for pair in pairs:
            lang_val = pair.language.value if hasattr(pair.language, "value") else str(pair.language)
            d[pair.cwe][lang_val] += 1
        return {k: dict(v) for k, v in d.items()}

    manifest = {
        "build_version": args.build_version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "quota_policy": quotas["policy"],
        "tier_policy": quotas["tiers"],
        "counts": {
            "train_prompts": len(final_train),
            "val_prompts": len(final_val),
            "eval_prompts": len(eval_prompts),
            "exemplar_pairs": len(surviving_pairs),
            "by_source": dict(src_count),
        },
        "distribution": {
            "train": dist(final_train),
            "val": dist(final_val),
            "eval": dist(eval_prompts),
            "exemplar_pairs": pair_dist(surviving_pairs),
        },
        "disjointness": {
            "eval_keys": report.eval_keys,
            "dropped_per_source": report.dropped_per_source,
            "string_only_drops": report.string_only_drops,
            "ast_only_drops": report.ast_only_drops,
            "both_drops": report.both_drops,
        },
        "train_val_leak_audit": {
            "threshold": train_val_threshold,
            "method": "per-(CWE, language) cell rapidfuzz.ratio; val item dropped if any train item in same cell has sim >= threshold",
            "val_candidates": len(final_val_candidates),
            "val_dropped_on_leak": n_val_dropped,
            "final_val": len(final_val),
            "leaked_items": val_leak_log,
        },
        "hashes": {
            "train_prompts.jsonl": _hash_file(train_path),
            "val_prompts.jsonl": _hash_file(val_path),
            "eval_prompts.jsonl": _hash_file(eval_path),
            "exemplar_pairs.jsonl": _hash_file(pairs_path),
        },
        "prompt_normalize": {
            "train": norm_train.stats(),
            "val": norm_val.stats(),
            "eval": norm_eval.stats(),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(
        f"wrote train={len(final_train)} val={len(final_val)} eval={len(eval_prompts)} "
        f"pairs={len(surviving_pairs)} to {out}/"
    )
    logger.info(f"disjointness drops: {report.dropped_per_source}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotas", type=Path,
                    default=Path("/storage/home/sss6371/secure-code-rl-ictai/docs/v0_1_5_quotas.json"))
    ap.add_argument("--raw-root", type=Path,
                    default=Path("/storage/home/sss6371/work/secure-code-rl-ictai-data/raw"))
    ap.add_argument("--output", type=Path,
                    default=Path("/storage/home/sss6371/work/secure-code-rl-ictai-data/build/v0.1.5"))
    ap.add_argument("--build-version", default="v0.1.5")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
