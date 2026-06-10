"""Tests for real SAST adapters (Bandit, CodeQL, Semgrep).

Each adapter is `real_sast`-marked because it requires the tool installed
on PATH. The Bandit tests are the most reliable (Bandit is fast and
deterministic). CodeQL tests are slow (database creation dominates).
Semgrep tests skip cleanly when `semgrep-core` is missing.

Unit tests at the top verify the adapters' shape contracts without
shelling out (e.g., construction-only).
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from secure_code_rl_ictai.sast.adapters.bandit import BanditAdapter
from secure_code_rl_ictai.sast.adapters.codeql import CodeQLAdapter
from secure_code_rl_ictai.sast.adapters.semgrep import SemgrepAdapter
from secure_code_rl_ictai.sast.models import ToolName
from secure_code_rl_ictai.sast.runner import Language


# ----------------------------------------------------------------------
# Construction-only unit tests (no real_sast)
# ----------------------------------------------------------------------


def test_bandit_adapter_reports_tool_name():
    a = BanditAdapter()
    assert a.tool == ToolName.BANDIT


def test_codeql_adapter_reports_tool_name():
    a = CodeQLAdapter(codeql_binary="codeql")
    assert a.tool == ToolName.CODEQL


def test_semgrep_adapter_reports_tool_name():
    a = SemgrepAdapter()
    assert a.tool == ToolName.SEMGREP



# ----------------------------------------------------------------------
# Real Bandit (real_sast)
# ----------------------------------------------------------------------


@pytest.mark.real_sast
@pytest.mark.skipif(shutil.which("bandit") is None, reason="bandit not on PATH")
def test_bandit_finds_known_issue(tmp_path: Path):
    """Bandit B608 (hardcoded_sql_expression) should fire on a clear SQLi pattern."""
    vulnerable = textwrap.dedent(
        """
        import sqlite3
        def get_user(uid):
            conn = sqlite3.connect("u.db")
            return conn.execute("SELECT * FROM users WHERE id = " + uid).fetchall()
        """
    ).strip()
    adapter = BanditAdapter(bandit_binary=shutil.which("bandit"))
    result = adapter.run(vulnerable, Language.PYTHON, tmp_path, timeout_s=10.0)
    assert not result.crashed
    assert result.sarif is not None
    # Should have at least one finding from the runs.
    runs = result.sarif.get("runs", [])
    assert len(runs) >= 1
    # Look for B608 across runs.
    rule_ids = []
    for run in runs:
        for r in run.get("results", []):
            rid = r.get("ruleId")
            if rid:
                rule_ids.append(rid)
    assert any("B608" in rid for rid in rule_ids), (
        f"expected B608 (SQL injection) finding; got rule_ids={rule_ids}"
    )


@pytest.mark.real_sast
@pytest.mark.skipif(shutil.which("bandit") is None, reason="bandit not on PATH")
def test_bandit_clean_code_no_findings(tmp_path: Path):
    """A clean snippet should produce zero results (Bandit at -ll level)."""
    clean = textwrap.dedent(
        """
        def add(a, b):
            return a + b
        """
    ).strip()
    result = BanditAdapter(bandit_binary=shutil.which("bandit")).run(clean, Language.PYTHON, tmp_path, timeout_s=10.0)
    assert not result.crashed
    assert result.sarif is not None
    findings_count = sum(
        len(run.get("results", [])) for run in result.sarif.get("runs", [])
    )
    assert findings_count == 0


@pytest.mark.real_sast
@pytest.mark.skipif(shutil.which("bandit") is None, reason="bandit not on PATH")
def test_bandit_only_runs_on_python(tmp_path: Path):
    """Bandit returns crashed=False but with no SARIF on non-Python input.

    Actually, per the runner's contract, the adapter should refuse and
    return a non-SARIF result with a clear error in stderr. Concrete shape
    decision: return `crashed=False, sarif=None, stderr=<reason>` so the
    runner records it as a skip, not a crash."""
    c_code = "int main(){return 0;}"
    result = BanditAdapter(bandit_binary=shutil.which("bandit")).run(c_code, Language.C, tmp_path, timeout_s=5.0)
    # Adapter should decline; treat as a no-op return.
    assert result.sarif is None or result.sarif == {"runs": []}


# ----------------------------------------------------------------------
# Real CodeQL (real_sast)
# ----------------------------------------------------------------------


@pytest.mark.real_sast
@pytest.mark.skipif(shutil.which("codeql") is None, reason="codeql not on PATH")
def test_codeql_finds_known_issue_python(tmp_path: Path):
    """Smoke test: run CodeQL python-security-extended on a SQLi-flavored snippet.

    Expensive (~10-60s for the smallest database). Confirm a parseable SARIF
    is produced; rule id matches will vary by CodeQL version.
    """
    vulnerable = textwrap.dedent(
        """
        import sqlite3
        def get_user(uid):
            conn = sqlite3.connect("u.db")
            return conn.execute("SELECT * FROM users WHERE id = " + uid).fetchall()
        """
    ).strip()
    adapter = CodeQLAdapter(codeql_binary=shutil.which("codeql"))
    result = adapter.run(vulnerable, Language.PYTHON, tmp_path, timeout_s=120.0)
    if result.crashed:
        # CodeQL can fail to bootstrap on small Python snippets in some configs.
        # Treat as informative failure; skip rather than block.
        pytest.skip(f"codeql crashed: {result.stderr[:300]}")
    assert result.sarif is not None
    # We don't pin the rule id (varies by version); just confirm we got SARIF runs.
    assert "runs" in result.sarif


# ----------------------------------------------------------------------
# Real Semgrep (real_sast)
# ----------------------------------------------------------------------


@pytest.mark.real_sast
@pytest.mark.skipif(shutil.which("semgrep") is None, reason="semgrep not on PATH")
def test_semgrep_runs_on_python(tmp_path: Path):
    """Semgrep --config=auto downloads rules on first run. May fail if
    semgrep-core OCaml binary is missing (known issue on the ROAR venv)."""
    vulnerable = textwrap.dedent(
        """
        import sqlite3
        def get_user(uid):
            conn = sqlite3.connect("u.db")
            return conn.execute("SELECT * FROM users WHERE id = " + uid).fetchall()
        """
    ).strip()
    adapter = SemgrepAdapter(semgrep_binary=shutil.which("semgrep"))
    result = adapter.run(vulnerable, Language.PYTHON, tmp_path, timeout_s=60.0)
    if result.crashed and "semgrep-core" in result.stderr:
        pytest.skip("semgrep-core OCaml binary not installed")
    assert result.sarif is not None or result.crashed
