#!/usr/bin/env python3
"""Measure the near-duplicate similarity distribution between the
training-index source set and the eval set.

Per `docs/scope.md` §9 / §10 (T2): v0.1 skips the Levenshtein near-
duplicate pass in the disjointness audit. But we **measure-and-report**
the distribution so the elbow is visible in the appendix and a later
threshold choice is data-justified rather than guessed.

This script:
  1. Reads `exemplar_pairs.jsonl` (train-index source) and
     `eval_prompts.jsonl` (eval set) from a build directory.
  2. For each eval blob, finds the nearest train-side blob under three
     metrics: BM25 (sparse), normalized Levenshtein (edit distance), and
     AST-token Jaccard (Python only).
  3. Reports the histogram of best-match similarities across all eval
     blobs.
  4. Heuristically picks an elbow at the largest gap in the sorted
     similarity values and reports it as the recommended-cutoff hint.
     The recommendation is **not** applied; the disjointness audit
     still uses exact + AST-exact for v0.1.

Output: JSON artifact suitable for appendix Threats-to-Validity §T2.

Usage:
    python scripts/measure_near_duplicate_distribution.py \\
        --build-dir data/build/v0.1/ \\
        --output data/near_duplicate_distribution_v0.1.json \\
        --bm25-top 5 \\
        --max-eval 0   # 0 = all
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("near_dup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Normalization (same as the disjointness audit)
# ---------------------------------------------------------------------------


_PY_COMMENT_RE = re.compile(r"#[^\n]*")
_C_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_C_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _strip_normalize(text: str) -> str:
    t = _C_BLOCK_COMMENT_RE.sub(" ", text)
    t = _C_LINE_COMMENT_RE.sub("", t)
    t = _PY_COMMENT_RE.sub("", t)
    t = _WHITESPACE_RE.sub(" ", t)
    return t.strip().lower()


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


def _bm25_top_k_indices(
    eval_doc: str, train_corpus: list[list[str]], k: int
) -> list[int]:
    """Crude BM25 top-K — used to narrow the expensive Levenshtein search.

    Lazy-imports rank_bm25. Falls back to a TF-only score if rank_bm25
    is unavailable (still useful as a candidate pre-filter, just less
    accurate ranking).
    """
    query_tokens = _tokenize(eval_doc)
    if not query_tokens:
        return list(range(min(k, len(train_corpus))))
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        # TF fallback. Per-doc scores are sum of query-term frequencies.
        doc_tfs: list[Counter[str]] = [Counter(d) for d in train_corpus]
        scores = [sum(tf[t] for t in query_tokens) for tf in doc_tfs]
    else:
        scores = list(BM25Okapi(train_corpus).get_scores(query_tokens))
    # Top-K indices by score (descending).
    indexed = sorted(enumerate(scores), key=lambda x: -x[1])
    return [i for i, _ in indexed[:k]]


def _tokenize(text: str) -> list[str]:
    # Lowercase identifier-aware split with camelCase splitting.
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    flat = re.sub(r"[^a-zA-Z0-9]+", " ", camel)
    return [tok for tok in flat.lower().split() if tok]


def _levenshtein_ratio(a: str, b: str) -> float:
    """Normalized similarity in [0, 1]. 1.0 = identical, 0.0 = no overlap.

    Uses difflib.SequenceMatcher.ratio() which is O(n*m) Levenshtein-like.
    For very long blobs we cap input length so this stays bounded.
    """
    a = a[:4000]
    b = b[:4000]
    return SequenceMatcher(None, a, b).ratio()


def _ast_jaccard_python(a: str, b: str) -> float | None:
    """Jaccard over Python AST node-name multisets. None if either fails to parse."""
    try:
        nodes_a = Counter(type(n).__name__ for n in ast.walk(ast.parse(a)))
        nodes_b = Counter(type(n).__name__ for n in ast.walk(ast.parse(b)))
    except SyntaxError:
        return None
    if not nodes_a and not nodes_b:
        return 1.0
    inter = sum((nodes_a & nodes_b).values())
    union = sum((nodes_a | nodes_b).values())
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _iter_train_blobs(pairs_path: Path) -> Iterable[tuple[str, str]]:
    """Yield (e_pos, e_neg) text pairs from exemplar_pairs.jsonl."""
    with open(pairs_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec.get("e_pos", ""), rec.get("e_neg", "")


def _iter_eval_blobs(prompts_path: Path) -> Iterable[tuple[str, str]]:
    """Yield (prompt_id, prompt_text) from eval_prompts.jsonl."""
    with open(prompts_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec.get("id", "?"), rec.get("prompt_text", "")


# ---------------------------------------------------------------------------
# Elbow heuristic
# ---------------------------------------------------------------------------


def _suggest_elbow(values: list[float]) -> float | None:
    """Suggest a threshold at the largest gap in the sorted descending values.

    The intuition: if there's a clear separation between "actually near-
    duplicates" and "merely similar," it shows up as a step in the
    sorted similarity curve. The elbow is just below that step.
    Returns None for fewer than 3 values.
    """
    if len(values) < 3:
        return None
    sorted_desc = sorted(values, reverse=True)
    gaps = [
        (sorted_desc[i] - sorted_desc[i + 1], (sorted_desc[i] + sorted_desc[i + 1]) / 2)
        for i in range(len(sorted_desc) - 1)
    ]
    biggest = max(gaps, key=lambda g: g[0])
    return round(biggest[1], 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def measure(args: argparse.Namespace) -> int:
    pairs_path = args.build_dir / "exemplar_pairs.jsonl"
    eval_path = args.build_dir / "eval_prompts.jsonl"
    for required in (pairs_path, eval_path):
        if not required.exists():
            print(f"missing: {required}", file=sys.stderr)
            return 2

    # Load train-side blobs (use e_pos by default; the index e_pos is what
    # contamination on the eval set most often comes from).
    train_texts: list[str] = []
    for e_pos, e_neg in _iter_train_blobs(pairs_path):
        if e_pos.strip():
            train_texts.append(e_pos)
    logger.info("loaded %d train-side blobs", len(train_texts))

    if not train_texts:
        print("no train-side blobs found", file=sys.stderr)
        return 1

    # Tokenize train corpus once.
    train_tokenized = [_tokenize(t) for t in train_texts]

    # Iterate eval blobs.
    levenshtein_best: list[float] = []
    ast_best: list[float] = []
    per_eval_rows: list[dict] = []
    n_eval = 0
    for prompt_id, text in _iter_eval_blobs(eval_path):
        n_eval += 1
        if args.max_eval > 0 and n_eval > args.max_eval:
            break
        eval_norm = _strip_normalize(text)
        candidates = _bm25_top_k_indices(eval_norm, train_tokenized, args.bm25_top)
        if not candidates:
            continue
        # Levenshtein over the BM25-narrowed candidate set.
        best_lev = 0.0
        best_lev_idx = -1
        for idx in candidates:
            sim = _levenshtein_ratio(eval_norm, _strip_normalize(train_texts[idx]))
            if sim > best_lev:
                best_lev = sim
                best_lev_idx = idx
        # AST Jaccard (Python only; None if either side doesn't parse).
        best_ast: float | None = None
        for idx in candidates:
            j = _ast_jaccard_python(text, train_texts[idx])
            if j is not None and (best_ast is None or j > best_ast):
                best_ast = j

        levenshtein_best.append(best_lev)
        if best_ast is not None:
            ast_best.append(best_ast)
        per_eval_rows.append(
            {
                "prompt_id": prompt_id,
                "levenshtein_best": round(best_lev, 4),
                "ast_jaccard_best": round(best_ast, 4) if best_ast is not None else None,
                "best_lev_train_idx": best_lev_idx,
            }
        )
        if n_eval % 200 == 0:
            logger.info("processed %d eval blobs", n_eval)

    logger.info("done: %d eval blobs", len(per_eval_rows))

    # Histograms.
    bins = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0]

    def _hist(values: list[float]) -> dict[str, int]:
        h: dict[str, int] = {}
        for lo, hi in zip(bins, bins[1:]):
            label = f"[{lo:.2f}, {hi:.2f})"
            h[label] = sum(1 for v in values if lo <= v < hi)
        h[f"== 1.00"] = sum(1 for v in values if v >= 1.0)
        return h

    report = {
        "_meta": {
            "n_train_blobs": len(train_texts),
            "n_eval_blobs": len(per_eval_rows),
            "bm25_top_k": args.bm25_top,
            "spec_version": "scope.md v0.1 §10 T2 (Threats-to-Validity)",
        },
        "levenshtein": {
            "histogram": _hist(levenshtein_best),
            "max": round(max(levenshtein_best), 4) if levenshtein_best else None,
            "elbow_suggested": _suggest_elbow(levenshtein_best),
            "exact_count": sum(1 for v in levenshtein_best if v >= 1.0),
        },
        "ast_jaccard_python": {
            "histogram": _hist(ast_best),
            "max": round(max(ast_best), 4) if ast_best else None,
            "elbow_suggested": _suggest_elbow(ast_best),
            "exact_count": sum(1 for v in ast_best if v >= 1.0),
            "n_python_parseable": len(ast_best),
        },
        "per_eval_top_n": per_eval_rows[: args.max_per_eval_in_report]
        if args.max_per_eval_in_report > 0
        else per_eval_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[near-dup] wrote {args.output}")
    print(f"[near-dup] elbow (Levenshtein): {report['levenshtein']['elbow_suggested']}")
    print(f"[near-dup] elbow (AST Jaccard): {report['ast_jaccard_python']['elbow_suggested']}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bm25-top", type=int, default=5)
    p.add_argument("--max-eval", type=int, default=0, help="0 = all")
    p.add_argument(
        "--max-per-eval-in-report",
        type=int,
        default=100,
        help="Keep N per-eval rows in the report for inspection (0 = all)",
    )
    args = p.parse_args()
    sys.exit(measure(args))


if __name__ == "__main__":
    main()
