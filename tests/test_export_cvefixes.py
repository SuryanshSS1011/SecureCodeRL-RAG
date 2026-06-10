"""Tests for scripts/export_cvefixes_to_jsonl.py.

Builds a synthetic SQLite database matching CVEfixes v1.0.8's documented
schema (https://github.com/secureIT-project/CVEfixes/blob/main/Doc/) and
verifies the exporter:
  - Filters to Python/C/C++ at export time.
  - Emits the per-function record schema CvefixesAdapter consumes.
  - Joins method_change before/after rows correctly.
  - Drops malformed rows without crashing.
  - Is idempotent on re-run.

We do not download the real ~3 GB CVEfixes database for these tests;
the synthetic fixture is enough to pin the exporter's contract.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "export_cvefixes_to_jsonl.py"


def _run_script(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _build_fixture_db(path: Path) -> None:
    """Build a tiny SQLite DB matching the CVEfixes v1.0.8 schema we use.

    Schema (subset; real CVEfixes has more columns we don't read):
      cve(cve_id, severity, published_date)            # NO cwe_id here
      cwe_classification(cve_id, cwe_id)               # junction table
      fixes(cve_id, hash, repo_url)
      repository(repo_url, language, license)
      file_change(file_change_id, hash, filename, programming_language)
      method_change(
          method_change_id, file_change_id, name, signature,
          code, before_change
      )

    Verified against real CVEfixes v1.0.8 on 2026-06-10: CWE is NOT a
    column on `cve`; it lives on the `cwe_classification` junction table.
    """
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE cve (
            cve_id TEXT PRIMARY KEY,
            severity REAL,
            published_date TEXT
        );
        CREATE TABLE cwe_classification (
            cve_id TEXT,
            cwe_id TEXT
        );
        CREATE TABLE fixes (
            cve_id TEXT,
            hash TEXT,
            repo_url TEXT
        );
        CREATE TABLE repository (
            repo_url TEXT PRIMARY KEY,
            language TEXT,
            license TEXT
        );
        CREATE TABLE file_change (
            file_change_id INTEGER PRIMARY KEY,
            hash TEXT,
            filename TEXT,
            programming_language TEXT
        );
        CREATE TABLE method_change (
            method_change_id INTEGER PRIMARY KEY,
            file_change_id INTEGER,
            name TEXT,
            signature TEXT,
            code TEXT,
            before_change TEXT
        );
        """
    )

    cur.executemany(
        "INSERT INTO cve VALUES (?, ?, ?)",
        [
            ("CVE-2023-00001", 8.8, "2023-01-01"),
            ("CVE-2023-00002", 7.5, "2023-02-01"),
            ("CVE-2023-00003", 6.5, "2023-03-01"),
        ],
    )
    cur.executemany(
        "INSERT INTO cwe_classification VALUES (?, ?)",
        [
            ("CVE-2023-00001", "CWE-89"),
            ("CVE-2023-00002", "CWE-787"),
            ("CVE-2023-00003", "CWE-79"),
        ],
    )
    cur.executemany(
        "INSERT INTO fixes VALUES (?, ?, ?)",
        [
            ("CVE-2023-00001", "abc123", "https://github.com/x/py-proj"),
            ("CVE-2023-00002", "def456", "https://github.com/x/c-proj"),
            ("CVE-2023-00003", "ghi789", "https://github.com/x/js-proj"),
        ],
    )
    cur.executemany(
        "INSERT INTO repository VALUES (?, ?, ?)",
        [
            ("https://github.com/x/py-proj", "Python", "BSD-3-Clause"),
            ("https://github.com/x/c-proj", "C", "MIT"),
            ("https://github.com/x/js-proj", "JavaScript", "MIT"),
        ],
    )
    cur.executemany(
        "INSERT INTO file_change VALUES (?, ?, ?, ?)",
        [
            (1, "abc123", "orm.py", "Python"),
            (2, "def456", "copy.c", "C"),
            (3, "ghi789", "render.js", "JavaScript"),
        ],
    )
    # method_change: one row per function with `code` = post-fix and
    # `before_change` = pre-fix (matches CVEfixes convention).
    cur.executemany(
        "INSERT INTO method_change VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                10, 1, "get_user", "def get_user(uid):",
                "def get_user(uid):\n    return db.query('SELECT * WHERE id=?', [uid])\n",
                "def get_user(uid):\n    return db.query('SELECT * WHERE id=' + uid)\n",
            ),
            (
                11, 2, "copy", "void copy(char *dst, const char *src, int n)",
                "void copy(char *dst, const char *src, int n) {\n    for (int i = 0; i < n; ++i) dst[i] = src[i];\n}",
                "void copy(char *dst, const char *src, int n) {\n    for (int i = 0; i <= n; ++i) dst[i] = src[i];\n}",
            ),
            (
                12, 3, "render", "function render(name)",
                "function render(name) { return escape(name); }",
                "function render(name) { return name; }",
            ),
        ],
    )
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_export_emits_one_jsonl_record_per_method(tmp_path: Path):
    db = tmp_path / "cvefixes.db"
    _build_fixture_db(db)
    out_dir = tmp_path / "export"

    result = _run_script(
        ["--db", str(db), "--output-dir", str(out_dir)],
        cwd=tmp_path,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"

    # Output JSONL files land in out_dir.
    files = sorted(out_dir.glob("*.jsonl"))
    assert files, f"no .jsonl files in {out_dir}; stderr:\n{result.stderr}"

    records: list[dict] = []
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # 3 method_change rows but JS one filtered out by default language set.
    assert len(records) == 2

    cves = [r["cve_id"] for r in records]
    assert "CVE-2023-00001" in cves  # Python
    assert "CVE-2023-00002" in cves  # C
    assert "CVE-2023-00003" not in cves  # JavaScript filtered

    # Per-record schema matches CvefixesAdapter's expectation.
    py_rec = next(r for r in records if r["cve_id"] == "CVE-2023-00001")
    for key in (
        "fix_commit", "cve_id", "project", "license", "language", "cwe",
        "file_path", "function_name", "signature", "pre_fix", "post_fix",
    ):
        assert key in py_rec, f"missing key {key} in {py_rec}"

    # pre_fix and post_fix are distinct.
    assert py_rec["pre_fix"] != py_rec["post_fix"]
    assert "WHERE id=' + uid" in py_rec["pre_fix"]   # vulnerable concat
    assert "WHERE id=?" in py_rec["post_fix"]         # parameterized


# ----------------------------------------------------------------------
# Language filter
# ----------------------------------------------------------------------


def test_export_filters_to_only_explicit_languages(tmp_path: Path):
    db = tmp_path / "cvefixes.db"
    _build_fixture_db(db)
    out_dir = tmp_path / "export"

    # Restrict to Python only.
    result = _run_script(
        ["--db", str(db), "--output-dir", str(out_dir), "--languages", "Python"],
        cwd=tmp_path,
    )
    assert result.returncode == 0

    records: list[dict] = []
    for f in out_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 1
    assert records[0]["language"] == "Python"


# ----------------------------------------------------------------------
# Malformed rows
# ----------------------------------------------------------------------


def test_export_drops_rows_with_missing_before_change(tmp_path: Path):
    """If before_change (pre-fix) is NULL, skip the row — we need both halves
    to emit an ExemplarPair downstream."""
    db = tmp_path / "cvefixes.db"
    _build_fixture_db(db)
    # Null out before_change on one row.
    conn = sqlite3.connect(db)
    conn.execute("UPDATE method_change SET before_change = NULL WHERE method_change_id = 10")
    conn.commit()
    conn.close()

    out_dir = tmp_path / "export"
    result = _run_script(
        ["--db", str(db), "--output-dir", str(out_dir)],
        cwd=tmp_path,
    )
    assert result.returncode == 0

    records: list[dict] = []
    for f in out_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    # The Python row dropped; only the C row survives.
    assert len(records) == 1
    assert records[0]["language"] == "C"


# ----------------------------------------------------------------------
# Idempotence
# ----------------------------------------------------------------------


def test_export_is_idempotent_as_set(tmp_path: Path):
    """The exporter's JSONL output is an intermediate artifact; we removed
    the SQL ORDER BY because sorting hundreds of thousands of rows × many-KB
    function bodies OOM'd a 32 GB SLURM job. Without ORDER BY the row order
    within a file depends on SQLite's nested-loop join order, which is
    deterministic for a given database but not guaranteed byte-stable
    across separate database snapshots. Downstream determinism is restored
    by build_v0_1_5.py's own sort before computing manifest hashes.

    We pin the weaker idempotence guarantee: same set of records across runs.
    """
    db = tmp_path / "cvefixes.db"
    _build_fixture_db(db)

    out1 = tmp_path / "export1"
    out2 = tmp_path / "export2"
    args = ["--db", str(db)]

    r1 = _run_script([*args, "--output-dir", str(out1)], cwd=tmp_path)
    r2 = _run_script([*args, "--output-dir", str(out2)], cwd=tmp_path)
    assert r1.returncode == 0 and r2.returncode == 0

    files1 = sorted(p.name for p in out1.glob("*.jsonl"))
    files2 = sorted(p.name for p in out2.glob("*.jsonl"))
    assert files1 == files2
    for name in files1:
        recs1 = sorted(
            json.loads(line) for line in (out1 / name).read_text().splitlines() if line.strip()
        )
        recs2 = sorted(
            json.loads(line) for line in (out2 / name).read_text().splitlines() if line.strip()
        )
        assert recs1 == recs2, f"{name} record sets differ between runs"


# ----------------------------------------------------------------------
# CLI hygiene
# ----------------------------------------------------------------------


def test_export_rejects_missing_db(tmp_path: Path):
    result = _run_script(
        ["--db", str(tmp_path / "nope.db"), "--output-dir", str(tmp_path / "out")],
        cwd=tmp_path,
    )
    assert result.returncode != 0


def test_export_refuses_overwrite_without_force(tmp_path: Path):
    db = tmp_path / "cvefixes.db"
    _build_fixture_db(db)
    out_dir = tmp_path / "export"

    r1 = _run_script(["--db", str(db), "--output-dir", str(out_dir)], cwd=tmp_path)
    assert r1.returncode == 0

    r2 = _run_script(["--db", str(db), "--output-dir", str(out_dir)], cwd=tmp_path)
    assert r2.returncode != 0
    assert "exists" in r2.stderr.lower() or "force" in r2.stderr.lower()

    r3 = _run_script(
        ["--db", str(db), "--output-dir", str(out_dir), "--force"],
        cwd=tmp_path,
    )
    assert r3.returncode == 0


# ----------------------------------------------------------------------
# Output is consumable by CvefixesAdapter
# ----------------------------------------------------------------------


def test_export_output_is_consumable_by_adapter(tmp_path: Path):
    """Round-trip: export to JSONL, then load through CvefixesAdapter.
    Verifies the exporter's schema matches the adapter's expectations."""
    from secure_code_rl_ictai.data_prep import CvefixesAdapter, CvefixesConfig

    db = tmp_path / "cvefixes.db"
    _build_fixture_db(db)
    out_dir = tmp_path / "export"

    result = _run_script(
        ["--db", str(db), "--output-dir", str(out_dir)],
        cwd=tmp_path,
    )
    assert result.returncode == 0

    adapter = CvefixesAdapter(CvefixesConfig(jsonl_dir=out_dir))
    prompts = list(adapter.load())
    pairs = list(adapter.load_exemplar_pairs())

    assert len(pairs) == 2  # Python + C
    assert len(prompts) >= 1  # at least the Python one with a usable signature
