"""Tests for scripts/convert_seccodeplt_parquet_to_jsonl.py.

The script reads SecCodePLT's HF parquet and emits the JSONL schema
SecCodePltAdapter consumes (id / category / language / task_description
/ signature / tests / metadata).

We don't have access to the real SecCodePLT schema until the download
lands; this fixture writes a synthetic parquet matching the NeurIPS'25
release's documented columns. The real-data smoke test runs on ROAR
after the actual parquet is present.

Locally these tests skip when pyarrow isn't installed. Marker
`real_seccodeplt` is for the on-ROAR smoke against the real file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "convert_seccodeplt_parquet_to_jsonl.py"


pytest.importorskip("pyarrow", reason="pyarrow not installed (runs on ROAR)")


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


def _write_synthetic_parquet(path: Path) -> None:
    """Build a synthetic SecCodePLT parquet matching the NeurIPS'25 schema.

    Columns we expect (verified against real on-ROAR after download):
      id (str), CWE_ID (str e.g. "CWE-89"), language (str), description (str),
      task_description (str), capability (str), prompt (str),
      vulnerable_code (str), ground_truth (str), unittest (str|dict|list),
      install_requires (list[str]).
    The script normalizes column names case-insensitively and best-effort.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {
        "id": ["sc-001", "sc-002", "sc-003"],
        "CWE_ID": ["CWE-89", "CWE-79", "CWE-22"],
        "language": ["python", "python", "python"],
        "task_description": [
            "Write a get_user function...",
            "Render a name as HTML...",
            "Open a file safely...",
        ],
        "capability": ["SQL Injection", "XSS", "Path Traversal"],
        "prompt": [
            "def get_user(uid):",
            "def render_name(name):",
            "def open_safe(path):",
        ],
        "ground_truth": [
            "def get_user(uid):\n    return db.query('SELECT * FROM users WHERE id = ?', [uid])",
            "import html\ndef render_name(name):\n    return html.escape(name)",
            "def open_safe(path):\n    safe = os.path.normpath(path)\n    return open(safe)",
        ],
        "unittest": [
            '{"setup": "", "testcases": [["get_user(1)", "alice"]]}',
            '{"setup": "", "testcases": [["render_name(\'<x>\')", "&lt;x&gt;"]]}',
            '{"setup": "", "testcases": [["open_safe(\'/etc/passwd\')", "..."]]}',
        ],
    }
    table = pa.Table.from_pydict(rows)
    pq.write_table(table, path)


def test_converter_emits_jsonl_with_expected_keys(tmp_path: Path):
    parquet_path = tmp_path / "in.parquet"
    _write_synthetic_parquet(parquet_path)
    output_path = tmp_path / "out.jsonl"

    result = _run_script(
        ["--parquet", str(parquet_path), "--output", str(output_path)],
        cwd=tmp_path,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"

    records = [
        json.loads(line)
        for line in output_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 3

    # Schema check: every record has the keys SecCodePltAdapter expects.
    for rec in records:
        for required in ("id", "category", "language", "task_description", "tests"):
            assert required in rec, f"missing {required} in {rec}"

    # Specific value check.
    sql_rec = next(r for r in records if r["id"] == "sc-001")
    assert sql_rec["category"] == "CWE-89"
    assert "get_user" in sql_rec["task_description"]


def test_converter_filters_by_language(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "in.parquet"
    rows = {
        "id": ["sc-001", "sc-002"],
        "CWE_ID": ["CWE-89", "CWE-787"],
        "language": ["python", "c"],
        "task_description": ["py task", "c task"],
        "capability": ["SQLi", "OOB"],
        "prompt": ["def f():", "int g();"],
        "ground_truth": ["pass", "return 0;"],
        "unittest": ["", ""],
    }
    pq.write_table(pa.Table.from_pydict(rows), parquet_path)
    output_path = tmp_path / "out.jsonl"

    result = _run_script(
        [
            "--parquet", str(parquet_path),
            "--output", str(output_path),
            "--languages", "python",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0
    records = [
        json.loads(line)
        for line in output_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["language"] == "python"


def test_converter_rejects_missing_parquet(tmp_path: Path):
    result = _run_script(
        ["--parquet", str(tmp_path / "nope.parquet"), "--output", str(tmp_path / "out.jsonl")],
        cwd=tmp_path,
    )
    assert result.returncode != 0


def test_converter_output_consumable_by_adapter(tmp_path: Path):
    """Round-trip: convert parquet -> JSONL, then load through SecCodePltAdapter."""
    from secure_code_rl_ictai.data_prep import SecCodePltAdapter, SecCodePltConfig

    parquet_path = tmp_path / "in.parquet"
    _write_synthetic_parquet(parquet_path)
    output_path = tmp_path / "out.jsonl"
    result = _run_script(
        ["--parquet", str(parquet_path), "--output", str(output_path)],
        cwd=tmp_path,
    )
    assert result.returncode == 0

    adapter = SecCodePltAdapter(SecCodePltConfig(jsonl_path=output_path))
    prompts = list(adapter.load())
    assert len(prompts) == 3
    cwes = {p.target_cwe for p in prompts}
    assert "CWE-89" in cwes
    assert "CWE-79" in cwes
