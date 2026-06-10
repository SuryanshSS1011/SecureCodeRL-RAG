#!/usr/bin/env python3
"""Export CVEfixes v1.0.8 SQLite database to JSONL records.

The CVEfixes dataset (Bhandari et al., MSR 2021; Zenodo record 13118970)
ships as a ~3 GB SQLite database. CvefixesAdapter reads JSONL per the
schema documented in docs/data_prep_spec.md §4.1. This script bridges
the two: SQL JOINs over CVEfixes tables → per-function JSONL records.

We do the JOIN here (not in the adapter) for three reasons:
  1. The full DB is too big to ship in the repo or hold in memory.
  2. The exporter can be re-run when CVEfixes ships a new version;
     CvefixesAdapter's JSONL contract stays stable.
  3. SQLite isn't installed on every training environment; JSONL is.

Per-record output schema (one JSON object per .jsonl line):
    {
      "fix_commit": "abc123...",
      "cve_id": "CVE-2023-12345",
      "project": "https://github.com/x/y",
      "license": "BSD-3-Clause",
      "language": "Python",
      "cwe": "CWE-89",
      "file_path": "lib/orm.py",
      "function_name": "get_user",
      "signature": "def get_user(uid):",
      "pre_fix": "<vulnerable function body>",
      "post_fix": "<fixed function body>"
    }

We shard output by language: `python.jsonl`, `c.jsonl`, `cpp.jsonl`,
etc. Idempotent on the same input.

Usage:
    python scripts/export_cvefixes_to_jsonl.py \\
        --db data/raw/cvefixes/CVEfixes_v1.0.8.sqlite \\
        --output-dir data/raw/cvefixes_jsonl/ \\
        --languages Python C C++ \\
        --max-rows 0

Pass `--max-rows N` (N > 0) for a sample export when testing the
downstream pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Iterator, TextIO

logger = logging.getLogger("export_cvefixes")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)


# Default language set: Python + C + C++ per scope.md §1.
_DEFAULT_LANGUAGES = ("Python", "C", "C++")

# CVEfixes uses inconsistent capitalization in some columns. Canonicalize
# to the values CvefixesAdapter expects (it then normalizes via the
# data_prep schema helpers, but we'd rather not pass garbage).
_LANG_NORMALIZE = {
    "python": "Python",
    "py": "Python",
    "c": "C",
    "cpp": "C++",
    "c++": "C++",
    "cxx": "C++",
}


# The SQL we run. References to CVEfixes columns are documented in the
# project's `Doc/` directory. We join six tables to get from
# method_change → file_change → fixes → cve → cwe_classification (the
# separate junction table for CWE assignments) → repository.
#
# `mc.code` is the post-fix function body; `mc.before_change` is the
# pre-fix function body. CVEfixes stores both as TEXT.
#
# IMPORTANT (verified against real CVEfixes v1.0.8 schema, 2026-06-10):
# CWE is NOT a column on `cve`. It lives on `cwe_classification(cve_id,
# cwe_id)`, allowing multi-CWE CVEs to have multiple rows. We join through
# it and filter to CWE-* values. Multi-CWE CVEs produce one output row
# per (CWE, method) pair, which the adapter then dedups downstream by
# sha256(pre_fix + "|" + post_fix).
_EXPORT_SQL = """
SELECT
    f.hash AS fix_commit,
    c.cve_id AS cve_id,
    f.repo_url AS project,
    NULL AS license,
    fc.programming_language AS language,
    cc.cwe_id AS cwe,
    fc.filename AS file_path,
    mc.name AS function_name,
    mc.signature AS signature,
    mc.before_change AS pre_fix,
    mc.code AS post_fix
FROM method_change mc
JOIN file_change fc ON mc.file_change_id = fc.file_change_id
JOIN fixes f ON fc.hash = f.hash
JOIN cve c ON f.cve_id = c.cve_id
JOIN cwe_classification cc ON c.cve_id = cc.cve_id
WHERE mc.before_change IS NOT NULL
  AND mc.code IS NOT NULL
  AND cc.cwe_id IS NOT NULL
  AND cc.cwe_id LIKE 'CWE-%'
"""
# NB: No ORDER BY. Sorting the full result set (hundreds of thousands of
# rows × pre/post-fix function bodies that can each be many KB) at the
# SQL layer materializes a multi-GB sort buffer that OOM-killed our
# 32-GB SLURM jobs even with indexes. The streaming output writes rows
# in nested-loop order, which is deterministic within a single run for
# a given database but not byte-stable across separate database
# snapshots. Downstream determinism is restored by
# `scripts/build_v0_1_5.py`'s own sort step before writing manifest
# hashes; the exporter's JSONL files are an intermediate artifact that
# doesn't need byte-stable ordering at the source.


def _iter_rows(conn: sqlite3.Connection, languages: set[str]) -> Iterator[dict]:
    """Yield records that pass the language filter. Normalizes language."""
    cur = conn.cursor()
    cur.execute(_EXPORT_SQL)
    cols = [d[0] for d in cur.description]
    n_kept = 0
    n_dropped_lang = 0
    n_dropped_other = 0
    for row in cur:
        rec = dict(zip(cols, row))
        lang_raw = (rec.get("language") or "").strip()
        if not lang_raw:
            n_dropped_other += 1
            continue
        canonical = _LANG_NORMALIZE.get(lang_raw.lower(), lang_raw)
        if canonical not in languages:
            n_dropped_lang += 1
            continue
        rec["language"] = canonical
        yield rec
        n_kept += 1
    logger.info(
        "rows kept=%d dropped(language)=%d dropped(other)=%d",
        n_kept, n_dropped_lang, n_dropped_other,
    )


def _shard_path(out_dir: Path, language: str) -> Path:
    safe = language.lower().replace("+", "p").replace("#", "sharp")
    return out_dir / f"{safe}.jsonl"


def export(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"db does not exist: {args.db}", file=sys.stderr)
        return 2
    if args.output_dir.exists() and not args.force:
        print(
            f"output directory {args.output_dir} exists; pass --force to overwrite",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    languages = set(args.languages)
    logger.info("exporting CVEfixes from %s; languages=%s", args.db, sorted(languages))

    conn = sqlite3.connect(args.db)

    # Create indexes on the JOIN columns if they don't exist. Without
    # these, SQLite's planner falls back to nested-loop joins with a
    # huge intermediate sort buffer, which OOM-killed our jobs at 16 GB.
    # Indexing on the join columns drops peak memory to a few hundred MB.
    #
    # The CREATE INDEX IF NOT EXISTS calls are idempotent and take a
    # few minutes on the first run only. Subsequent exporter runs see
    # indexes already present and skip the cost.
    cur = conn.cursor()
    for index_sql in (
        "CREATE INDEX IF NOT EXISTS idx_method_change_file ON method_change(file_change_id)",
        "CREATE INDEX IF NOT EXISTS idx_file_change_hash ON file_change(hash)",
        "CREATE INDEX IF NOT EXISTS idx_fixes_cve ON fixes(cve_id)",
        "CREATE INDEX IF NOT EXISTS idx_fixes_hash ON fixes(hash)",
        "CREATE INDEX IF NOT EXISTS idx_cwe_class_cve ON cwe_classification(cve_id)",
    ):
        idx_name = index_sql.split("INDEX IF NOT EXISTS ", 1)[1].split(" ON ", 1)[0]
        logger.info("ensuring index: %s", idx_name)
        cur.execute(index_sql)
    conn.commit()
    # PRAGMAs to reduce intermediate-buffer footprint during the JOIN.
    cur.execute("PRAGMA temp_store = MEMORY")
    cur.execute("PRAGMA cache_size = -262144")  # 256 MB cache (negative = KB)

    # Streaming output: open per-language file handles, write records as
    # they arrive. The SQL has no ORDER BY (removed because sort buffers
    # OOM'd at 32 GB); rows arrive in nested-loop order from the indexes.
    # Determinism is restored by build_v0_1_5.py's sort before manifest.
    file_handles: dict[str, TextIO] = {}
    counts: dict[str, int] = {lang: 0 for lang in languages}
    # Pre-create empty files for every requested language so downstream
    # consumers see the expected shard layout even when a language has
    # no matching rows.
    for lang in languages:
        out_path = _shard_path(args.output_dir, lang)
        file_handles[lang] = open(out_path, "w")

    try:
        n_total = 0
        for rec in _iter_rows(conn, languages):
            lang = rec["language"]
            fh = file_handles.get(lang)
            if fh is None:
                # New language not in the initial set; open a handle.
                fh = open(_shard_path(args.output_dir, lang), "w")
                file_handles[lang] = fh
                counts.setdefault(lang, 0)
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            counts[lang] = counts.get(lang, 0) + 1
            n_total += 1
            if args.max_rows > 0 and n_total >= args.max_rows:
                logger.info("stopping at --max-rows=%d", args.max_rows)
                break
            # Periodic progress log so long runs are observable.
            if n_total % 50000 == 0:
                logger.info("  ...wrote %d records so far", n_total)
    finally:
        for fh in file_handles.values():
            fh.close()
        conn.close()

    for lang, n in counts.items():
        logger.info("  %s -> %s (%d rows)", lang, _shard_path(args.output_dir, lang), n)
    logger.info("export complete: %d records total", sum(counts.values()))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--languages",
        nargs="+",
        default=list(_DEFAULT_LANGUAGES),
        help="Language whitelist (default: %(default)s).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Stop after N rows total. 0 = no limit (default).",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    sys.exit(export(args))


if __name__ == "__main__":
    main()
