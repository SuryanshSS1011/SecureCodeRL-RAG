"""Extract (prompt, secure_completion) pairs for SFT-only baseline (#262).

Reads the same source-data the v0.1.7 corpus build read, and for every
train prompt that has an extractable "gold" completion in the source,
emits a SFT pair to train_sft_pairs.jsonl. Sources without a gold field
(DiverseVul, SecCodePLT) are skipped.

Design (#262 reviewer-attack defense): SFT-only must use ONLY gold data
that ships in the source corpus. No model-generated distillation. No
hand-authored fixes for prompts that didn't have them. The dropped
27%+3% = 30% of train prompts that lack a gold are simply omitted; we
report the reduced n in the paper.

Eligible sources:
    CVEfixes: post_fix is the gold; prompt = signature              [70%]
    design_pair_authored: shipped with both pre and post            [<1%]
    DiverseVul: vulnerable-only, no gold available                  [skipped]
    SecCodePLT: tests-only, no gold completion                      [skipped]
    Juliet: included via design_pair_authored or skipped

Usage:
    PYTHONPATH=src python scripts/extract_sft_pairs.py \\
        --train-prompts /scratch/.../v0.1.7/train_prompts.jsonl \\
        --cvefixes-jsonl-dir /scratch/.../cvefixes_export \\
        --design-pair-jsonl /scratch/.../design_pair_secure_pairs.jsonl \\
        --output /scratch/.../v0.1.7/train_sft_pairs.jsonl

Output schema (per line):
    {"id": "...", "source": "...", "language": "...", "target_cwe": "...",
     "prompt_text": "...", "secure_completion": "...",
     "task_signature": "...", "metadata": {...}}

The `id` matches train_prompts.jsonl so the SFT trainer can verify the
sub-corpus is a strict subset of the RL training corpus (load-bearing
for the reviewer-attack defense).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def _load_train_prompt_ids(train_jsonl: Path) -> dict[str, dict]:
    """Map train prompt id -> the full row dict.

    Used to verify each SFT pair we emit corresponds to a row in the RL
    training corpus. SFT-only must not introduce prompts the RL run
    didn't see.
    """
    out: dict[str, dict] = {}
    with train_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = row
    return out


def _iter_cvefixes_pairs(
    cvefixes_dir: Path,
    train_ids: dict[str, dict],
) -> Iterator[dict]:
    """Yield SFT pairs from CVEfixes-shaped JSONL files.

    We need this to be import-light so we can run on ROAR without
    bringing in the full data_prep adapter chain. We do our own
    iteration + filtering and use train_ids to confirm the prompt was
    in the RL training set.
    """
    from secure_code_rl_ictai.data_prep import normalize_cwe, normalize_language
    from secure_code_rl_ictai.data_prep.cvefixes import make_prompt_id

    for jsonl in sorted(cvefixes_dir.glob("*.jsonl")):
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pre = (rec.get("pre_fix") or "").strip()
                post = (rec.get("post_fix") or "").strip()
                signature = (rec.get("signature") or "").strip()
                if not (pre and post and signature):
                    continue
                if pre == post:
                    continue
                try:
                    lang = normalize_language(rec["language"])
                    cwe = normalize_cwe(rec["cwe"])
                except (KeyError, ValueError):
                    continue
                pid = make_prompt_id(
                    "cvefixes",
                    rec.get("fix_commit", ""),
                    rec.get("file_path", ""),
                    rec.get("function_name", ""),
                )
                if pid not in train_ids:
                    # CVEfixes row didn't land in the RL training set
                    # (filtered out by the corpus builder for CWE balance
                    # or near-dup reasons). Drop.
                    continue
                t_row = train_ids[pid]
                yield {
                    "id": pid,
                    "source": "cvefixes",
                    "language": lang,
                    "target_cwe": cwe,
                    "prompt_text": t_row["prompt_text"],
                    "secure_completion": post,
                    "task_signature": signature,
                    "metadata": {
                        "cve_id": rec.get("cve_id"),
                        "fix_commit": rec.get("fix_commit"),
                        "file_path": rec.get("file_path"),
                        "function_name": rec.get("function_name"),
                        "gold_source": "cvefixes.post_fix",
                    },
                }


def _iter_design_pair_pairs(
    design_pair_jsonl: Optional[Path],
    train_ids: dict[str, dict],
) -> Iterator[dict]:
    """Yield SFT pairs from the design_pair_secure_pairs.jsonl artifact.

    This artifact is produced by build_v0_1_5_1_corpus.py specifically
    for SFT warm-start. Each row has `prompt_text` + `secure_completion`
    already paired. We just verify the prompt id is in the train set.
    """
    if design_pair_jsonl is None or not design_pair_jsonl.exists():
        return
    with design_pair_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("id")
            if pid not in train_ids:
                continue
            t_row = train_ids[pid]
            yield {
                "id": pid,
                "source": "design_pair_authored",
                "language": t_row["language"],
                "target_cwe": t_row["target_cwe"],
                "prompt_text": t_row["prompt_text"],
                "secure_completion": row["secure_completion"],
                "task_signature": t_row.get("task_signature", ""),
                "metadata": {
                    "gold_source": "design_pair_authored.secure",
                    "design_pair_id": row.get("pair_id"),
                },
            }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-prompts", type=Path, required=True,
        help="Path to v0.1.7 train_prompts.jsonl (the RL training corpus)."
    )
    p.add_argument(
        "--cvefixes-jsonl-dir", type=Path, default=None,
        help="Directory of CVEfixes JSONL exports (with pre_fix/post_fix).",
    )
    p.add_argument(
        "--design-pair-jsonl", type=Path, default=None,
        help="Path to design_pair_secure_pairs.jsonl (from build_v0_1_5_1_corpus.py).",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    train_ids = _load_train_prompt_ids(args.train_prompts)
    logger.info("loaded %d train prompts from %s", len(train_ids), args.train_prompts)

    sources_present = {row["source"] for row in train_ids.values()}
    logger.info("train sources: %s", sorted(sources_present))

    n_written = 0
    by_source: dict[str, int] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        if args.cvefixes_jsonl_dir is not None:
            for row in _iter_cvefixes_pairs(args.cvefixes_jsonl_dir, train_ids):
                out.write(json.dumps(row) + "\n")
                n_written += 1
                by_source["cvefixes"] = by_source.get("cvefixes", 0) + 1
        if args.design_pair_jsonl is not None:
            for row in _iter_design_pair_pairs(args.design_pair_jsonl, train_ids):
                out.write(json.dumps(row) + "\n")
                n_written += 1
                by_source["design_pair_authored"] = (
                    by_source.get("design_pair_authored", 0) + 1
                )

    skipped_sources = sources_present - {"cvefixes", "design_pair_authored"}
    n_skipped_no_gold = sum(
        1 for row in train_ids.values() if row["source"] in skipped_sources
    )

    print(json.dumps({
        "n_train_prompts": len(train_ids),
        "n_sft_pairs_written": n_written,
        "by_source": by_source,
        "n_skipped_no_gold_field_in_source": n_skipped_no_gold,
        "skipped_sources": sorted(skipped_sources),
        "coverage_pct": round(100 * n_written / max(1, len(train_ids)), 2),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
