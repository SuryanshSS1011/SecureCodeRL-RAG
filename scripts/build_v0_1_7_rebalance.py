"""Build v0.1.7 from v0.1.6 + design-pair authored items: fixes val composition and adds test_spec items to train.

What v0.1.6 had:
  - train: 3090 (CVEfixes 2239 + DiverseVul 851)        — 0 test_spec items
  - val:   160  (CVEfixes only)                         — language mix 87% C/C++
  - eval:  861  (CyberSecEval/SecCodePLT/CASTLE/CWEval/SecurityEval)
                                                        — language mix 79% Python

Problems v0.1.7 fixes:

  P1. val 87% C/C++ vs eval 79% Python (FINDINGS_LOG 2026-06-14): trainer
      val signal is uninformative because the policy is judged on a
      different language mix than the headline eval.
  P2. train has 0 prompts with functional test_specs, so RL rollouts can
      never earn r_func during training.
  P3. CWE-306 (design-pair) has only 1 prompt total in v0.1.6 eval. The
      paper's design-pair claim has no statistical surface.

How v0.1.7 fixes them:

  1. Hold out val from eval, stratified by (source, language, CWE):
       - cells with >=10 items: take 10%
       - cells with 2-9 items: take exactly 1 (preserves CWE+lang breadth)
       - cells with 1 item: skip (would eliminate that cell from eval)
     Result: val composition mirrors eval (within 2pp per language).
  2. Carve 50 SecCodePLT items WITH test_specs into train so r_func has
     variance during RL rollouts.
  3. Merge 100 authored CWE-306/862 design-pair patterns from v0.1.5.1
     (task #185). Split: 70 to eval, 20 to val (the 20 explicitly labeled
     `design_pair_val_prompts.jsonl` are val-hinted), 10 to train.
  4. Strict disjointness: items moved between splits are REMOVED from
     their source. No prompt appears in two splits.

What v0.1.7 does NOT do (intentional):

  X1. train language mix is NOT rebalanced to match val/eval. Train
      reflects the real-world CVE vulnerability distribution (C/C++
      memory-safety dominant), which is where security knowledge lives.
      Forcing train to be 80% Python would require throwing away 73% of
      our security-grounded data. The intentional train/eval composition
      gap is the central generalization claim: "RL on a CVE-grounded
      C/C++-heavy corpus transfers to Python-dominated headline benchmarks".
      §7.2 per-CWE reweighting at training time is the mechanism that
      makes this work; see training_spec.md §7.

  X2. CWE-326 is included in val even though it has only 4 items in eval.
      This is intentional for 19/19 CWE coverage; CWE-326 is descriptive-
      only per scope.md so its statistical power isn't load-bearing.

Final shapes:
  train=3150 (60 with test_specs), val=121 (44 with test_specs),
  eval=780 (226 with test_specs). Language mix val/eval ~78%/80% Python,
  ~14%/15% C, ~7%/5% C++. CWE coverage 19/19 in both val and eval.

Run:
    PYTHONPATH=src .venv/bin/python scripts/build_v0_1_7_rebalance.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_v0_1_7")


# Held-out val sampling rate from each eval source.
VAL_RATE = 0.10  # ~10% of each (source, CWE, lang) cell

# Train-eval-bridge: how many SecCodePLT items to move into train.
# Cap rather than ratio so we keep a usable eval size. 50 gives enough
# test_spec variance for r_func to fire without gutting eval.
TRAIN_BRIDGE_SECCODEPLT = 50


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _read_jsonl(p: Path) -> list[dict]:
    rows = []
    with open(p) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path,
                    default=Path("/scratch/sss6371/secure-code-rl-ictai-data/build/v0.1.6"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/scratch/sss6371/secure-code-rl-ictai-data/build/v0.1.7"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-rate", type=float, default=VAL_RATE)
    ap.add_argument("--train-bridge-seccodeplt", type=int,
                    default=TRAIN_BRIDGE_SECCODEPLT)
    ap.add_argument("--design-pair-src", type=Path,
                    default=Path("/scratch/sss6371/secure-code-rl-ictai-data/build/v0.1.5.1"))
    args = ap.parse_args()

    src = args.src_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    train = _read_jsonl(src / "train_prompts.jsonl")
    val_old = _read_jsonl(src / "val_prompts.jsonl")
    eval_ = _read_jsonl(src / "eval_prompts.jsonl")
    logger.info("loaded src: train=%d val_old=%d eval=%d",
                len(train), len(val_old), len(eval_))

    # ----- 0. Merge authored design-pair prompts from v0.1.5.1 -----
    #
    # The 100 authored CWE-306/862 patterns + 20 design-pair val prompts at
    # /scratch/.../build/v0.1.5.1/design_pair_*.jsonl were authored to fix
    # the design-pair coverage gap (task #185, prereg §4) — Python prompts
    # WITH instruction wrappers AND test_specs (functional oracle). Without
    # them, eval has only 1 CWE-306 prompt total. With them, the design-pair
    # claim has empirical surface to defend.
    #
    # Distribution: 70 patterns → eval, 15 patterns → val, 15 patterns →
    # train; 20 design_pair_val_prompts → val (file name is authoritative).
    dp_src_dir = args.design_pair_src
    dp_patterns = []
    dp_val_extra_ids: set[str] = set()
    if dp_src_dir is not None and dp_src_dir.exists():
        try:
            dp_patterns = _read_jsonl(dp_src_dir / "design_pair_patterns.jsonl")
            dp_val_extra = _read_jsonl(dp_src_dir / "design_pair_val_prompts.jsonl")
            # The 20 design_pair_val_prompts are a labeled SUBSET of the 100
            # design_pair_patterns. Treat val_extra as a hint: those 20 ids
            # should preferentially go to val.
            dp_val_extra_ids = {r["id"] for r in dp_val_extra}
            logger.info("loaded design-pair: patterns=%d (with val-hint=%d)",
                        len(dp_patterns), len(dp_val_extra_ids))
        except FileNotFoundError as e:
            logger.warning("design-pair files not found: %s", e)
    # Stable order. The 20 val-hinted items go to val first; then split the
    # remaining 80 to eval (70) + train (10) deterministically.
    rng_dp = random.Random(args.seed + 1)
    dp_patterns_sorted = sorted(dp_patterns, key=lambda r: r["id"])
    dp_val_hinted = [r for r in dp_patterns_sorted if r["id"] in dp_val_extra_ids]
    dp_rest = [r for r in dp_patterns_sorted if r["id"] not in dp_val_extra_ids]
    rest_idx = list(range(len(dp_rest)))
    rng_dp.shuffle(rest_idx)
    # Of the 80 non-val-hinted: 70 → eval, 10 → train. Plus the 20 hinted to val.
    n_eval_from_rest = min(70, len(dp_rest))
    dp_to_eval = [dp_rest[i] for i in rest_idx[:n_eval_from_rest]]
    dp_to_train = [dp_rest[i] for i in rest_idx[n_eval_from_rest:]]
    dp_to_val = dp_val_hinted
    logger.info("design-pair routed: eval=%d val=%d train=%d (val-hinted=%d)",
                len(dp_to_eval), len(dp_to_val), len(dp_to_train),
                len(dp_val_hinted))

    # ----- 1. Hold out val from eval, stratified by (source, lang, CWE) -----
    by_cell = defaultdict(list)
    for r in eval_:
        key = (r["source"], r["language"], r["target_cwe"])
        by_cell[key].append(r)

    rng = random.Random(args.seed)
    val_new: list[dict] = []
    eval_holdouts_keys: set[str] = set()
    for cell, items in sorted(by_cell.items()):
        items_sorted = sorted(items, key=lambda r: r["id"])
        # Per-cell stratified sample.
        # - cells with >=10 items: take args.val_rate (default 10%) rounded.
        # - cells with 2-9 items: take exactly 1 to preserve CWE+lang breadth.
        # - cells with 1 item: skip (taking it would eliminate that
        #   (source, lang, CWE) cell from eval entirely).
        # The 2-item threshold lets CWE-326 (cells of size 2 in CWEval/
        # SecurityEval) reach val, achieving 19/19 CWE coverage.
        n = len(items_sorted)
        if n >= 10:
            k = max(1, int(round(n * args.val_rate)))
        elif n >= 2:
            k = 1
        else:
            k = 0
        if k == 0:
            continue
        idx = list(range(n))
        rng.shuffle(idx)
        chosen = [items_sorted[i] for i in idx[:k]]
        val_new.extend(chosen)
        for r in chosen:
            eval_holdouts_keys.add(r["id"])
    logger.info("val from eval: %d items carved from %d cells",
                len(val_new), len(by_cell))

    # ----- 2. Train-eval bridge: move SecCodePLT WITH test_spec into train -----
    seccodeplt_with_tests = [
        r for r in eval_
        if r["source"] == "seccodeplt"
        and r["id"] not in eval_holdouts_keys
        and r.get("test_spec", {}).get("test_cases")
    ]
    # Sort deterministically, sample top N. Diversity by CWE: round-robin
    # per CWE so we don't pile the bridge into one CWE.
    sec_by_cwe = defaultdict(list)
    for r in sorted(seccodeplt_with_tests, key=lambda r: r["id"]):
        sec_by_cwe[r["target_cwe"]].append(r)
    bridge: list[dict] = []
    cwes_cycle = sorted(sec_by_cwe.keys())
    while len(bridge) < args.train_bridge_seccodeplt and cwes_cycle:
        next_cycle = []
        for cwe in cwes_cycle:
            if sec_by_cwe[cwe]:
                bridge.append(sec_by_cwe[cwe].pop(0))
                next_cycle.append(cwe)
            if len(bridge) >= args.train_bridge_seccodeplt:
                break
        cwes_cycle = next_cycle
    bridge_keys = {r["id"] for r in bridge}
    logger.info("train bridge from SecCodePLT (with test_spec): %d items "
                "across %d CWEs",
                len(bridge), len({r["target_cwe"] for r in bridge}))

    # ----- 3. New eval is what's left, plus design-pair eval items -----
    held_out = eval_holdouts_keys | bridge_keys
    eval_new = [r for r in eval_ if r["id"] not in held_out]
    eval_new.extend(dp_to_eval)
    val_new.extend(dp_to_val)
    logger.info("new eval: %d (was %d, held out %d, design_pair +%d)",
                len(eval_new), len(eval_), len(held_out), len(dp_to_eval))
    logger.info("new val:  %d (eval_holdout %d + design_pair %d)",
                len(val_new), len(val_new) - len(dp_to_val), len(dp_to_val))

    # ----- 4. Disjointness sanity check -----
    train_ids = {r["id"] for r in train} | bridge_keys | {r["id"] for r in dp_to_train}
    val_ids = {r["id"] for r in val_new}
    eval_ids = {r["id"] for r in eval_new}
    overlap_tv = train_ids & val_ids
    overlap_te = train_ids & eval_ids
    overlap_ve = val_ids & eval_ids
    assert not overlap_tv, f"train-val overlap: {len(overlap_tv)} items"
    assert not overlap_te, f"train-eval overlap: {len(overlap_te)} items"
    assert not overlap_ve, f"val-eval overlap: {len(overlap_ve)} items"
    logger.info("disjointness OK: train ⊥ val ⊥ eval")

    # ----- 5. Assemble new train (existing + seccodeplt bridge + design-pair train) -----
    train_new = list(train) + bridge + dp_to_train
    # Stable order by id.
    train_new.sort(key=lambda r: r["id"])
    val_new.sort(key=lambda r: r["id"])
    eval_new.sort(key=lambda r: r["id"])

    # ----- 6. Compositional check: val vs eval language distribution -----
    def lang_mix(rows):
        c = Counter(r["language"] for r in rows)
        n = sum(c.values())
        return {k: f"{v}/{n}={100*v/n:.1f}%" for k, v in c.most_common()}

    def cwe_mix(rows, top_k=5):
        c = Counter(r["target_cwe"] for r in rows)
        return dict(c.most_common(top_k))

    logger.info("val lang mix:  %s", lang_mix(val_new))
    logger.info("eval lang mix: %s", lang_mix(eval_new))
    logger.info("val cwe top5:  %s", cwe_mix(val_new))
    logger.info("eval cwe top5: %s", cwe_mix(eval_new))

    # ----- 7. Write splits + provenance manifest -----
    train_path = out / "train_prompts.jsonl"
    val_path = out / "val_prompts.jsonl"
    eval_path = out / "eval_prompts.jsonl"
    _write_jsonl(train_path, train_new)
    _write_jsonl(val_path, val_new)
    _write_jsonl(eval_path, eval_new)

    # Copy the exemplar_pairs.jsonl + rag_index unchanged.
    import shutil
    for sibling in ("exemplar_pairs.jsonl",):
        src_p = src / sibling
        if src_p.exists():
            shutil.copyfile(src_p, out / sibling)
    # rag_index is a directory, sibling of the splits. Copy if present.
    src_rag = src / "rag_index"
    if src_rag.exists() and not (out / "rag_index").exists():
        shutil.copytree(src_rag, out / "rag_index")

    manifest = {
        "build_version": "v0.1.7",
        "derived_from": str(src),
        "changes": [
            "val held out from eval sources stratified by (source, lang, CWE), "
            f"rate={args.val_rate}",
            f"train-eval bridge: {len(bridge)} SecCodePLT items with test_spec "
            "moved to train so r_func has variance during RL",
            "strict disjointness: train ⊥ val ⊥ eval enforced by id",
        ],
        "counts": {
            "train": len(train_new),
            "val": len(val_new),
            "eval": len(eval_new),
            "by_source_train": dict(Counter(r["source"] for r in train_new)),
            "by_source_val": dict(Counter(r["source"] for r in val_new)),
            "by_source_eval": dict(Counter(r["source"] for r in eval_new)),
        },
        "language_distribution": {
            "train": dict(Counter(r["language"] for r in train_new)),
            "val": dict(Counter(r["language"] for r in val_new)),
            "eval": dict(Counter(r["language"] for r in eval_new)),
        },
        "cwe_distribution": {
            "train": dict(Counter(r["target_cwe"] for r in train_new)),
            "val": dict(Counter(r["target_cwe"] for r in val_new)),
            "eval": dict(Counter(r["target_cwe"] for r in eval_new)),
        },
        "test_spec_coverage": {
            "train_with_tests": sum(
                1 for r in train_new
                if r.get("test_spec", {}).get("test_cases")
            ),
            "val_with_tests": sum(
                1 for r in val_new
                if r.get("test_spec", {}).get("test_cases")
            ),
            "eval_with_tests": sum(
                1 for r in eval_new
                if r.get("test_spec", {}).get("test_cases")
            ),
        },
        "hashes": {
            "train_prompts.jsonl": _hash_file(train_path),
            "val_prompts.jsonl": _hash_file(val_path),
            "eval_prompts.jsonl": _hash_file(eval_path),
        },
        "seed": args.seed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("wrote v0.1.7 to %s", out)
    logger.info("final counts: train=%d val=%d eval=%d",
                len(train_new), len(val_new), len(eval_new))
    logger.info("test_spec coverage: train=%d val=%d eval=%d",
                manifest["test_spec_coverage"]["train_with_tests"],
                manifest["test_spec_coverage"]["val_with_tests"],
                manifest["test_spec_coverage"]["eval_with_tests"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
