"""One-shot backfill for v0.1.7 eval/val: add prefix/suffix splice fields to
cyberseceval (heuristic wrapper) and cweval (best-effort) C/C++ rows.

Background: build_v0_1_7_rebalance.py inherits eval rows from v0.1.6's
already-built eval_prompts.jsonl, so any adapter-level splice fields added
after v0.1.6 was built (cyberseceval heuristic wrapper, cweval BEGIN PROMPT/
SOLUTION extraction) are missing on those rows. Rebuilding v0.1.6 from raw
sources would require CWEval + CyberSecEval source dirs which are no longer
on disk. This patcher rewrites the relevant fields in-place on v0.1.7
eval_prompts.jsonl and val_prompts.jsonl, leaving Juliet, CASTLE, SecCodePLT,
and SecurityEval rows untouched (Juliet already has splice; CASTLE is whole-
program; SecCodePLT/SecurityEval are Python).

For cweval C/C++: we have no original source on disk to recover the original
prefix/suffix. We fall back to the same heuristic wrapper used for
cyberseceval, marked splice_mode='heuristic_wrapper_fallback' in metadata.
The 18 cweval C/C++ rows are a small slice of the 1582-row eval (~1%); the
honest paper footnote is that cweval C/C++ uses the same wrapper as
cyberseceval after the v0.1.7 corpus was assembled.

Run on ROAR:
    cd /storage/home/sss6371/secure-code-rl-ictai
    PYTHONPATH=src .venv/bin/python \
        scripts/backfill_cyberseceval_cweval_splice.py \
        --build-dir /scratch/sss6371/secure-code-rl-ictai-data/build/v0.1.7
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("splice_backfill")


_C_PREFIX = (
    "#include <stdio.h>\n"
    "#include <string.h>\n"
    "#include <stdlib.h>\n"
)
_C_SUFFIX = "\nint main(){ return 0; }\n"

_CPP_PREFIX = (
    "#include <iostream>\n"
    "#include <string>\n"
    "#include <vector>\n"
    "#include <cstring>\n"
)
_CPP_SUFFIX = "\nint main(){ return 0; }\n"


def _patch_row(row: dict) -> bool:
    """Return True iff the row was modified."""
    source = row.get("source", "")
    lang = row.get("language", "")
    if source not in ("cyberseceval", "cweval"):
        return False
    if lang not in ("c", "cpp"):
        return False
    ts = row.setdefault("test_spec", {})
    if ts.get("prefix_text") or ts.get("suffix_text"):
        # Already populated (e.g., a future rebuild re-ran the adapter).
        return False
    if lang == "c":
        ts["prefix_text"] = _C_PREFIX
        ts["suffix_text"] = _C_SUFFIX
    else:
        ts["prefix_text"] = _CPP_PREFIX
        ts["suffix_text"] = _CPP_SUFFIX
    meta = row.setdefault("metadata", {})
    if source == "cyberseceval":
        meta["splice_mode"] = "heuristic_wrapper"
    else:
        # cweval lost its raw source after v0.1.6 was built; honest label.
        meta["splice_mode"] = "heuristic_wrapper_fallback"
    return True


def _patch_file(path: Path) -> tuple[int, int]:
    """Return (n_patched, n_total)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    n_patched = sum(_patch_row(r) for r in rows)
    if n_patched:
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return n_patched, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--build-dir",
        type=Path,
        default=Path("/scratch/sss6371/secure-code-rl-ictai-data/build/v0.1.7"),
        help="Dir containing eval_prompts.jsonl and val_prompts.jsonl.",
    )
    args = ap.parse_args()
    for name in ("eval_prompts.jsonl", "val_prompts.jsonl"):
        path = args.build_dir / name
        if not path.exists():
            logger.warning("skipping (missing): %s", path)
            continue
        n_patched, n_total = _patch_file(path)
        logger.info("%s: patched %d / %d rows", name, n_patched, n_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
