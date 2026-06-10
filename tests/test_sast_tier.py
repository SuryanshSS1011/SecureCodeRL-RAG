"""Tests for sequential SAST tier knob (cheap / cheap+periodic / all).

Per docs/sast_pipeline_spec.md §1 routing + the tier extension:
  - tier="all" (default): run every tool in the language routing.
  - tier="cheap": run only Bandit + Semgrep + Cppcheck (no CodeQL).
  - tier="cheap+periodic": run cheap every step; run deep every N steps.

Eval-time runs always use tier="all".
"""

from __future__ import annotations

from pathlib import Path

from secure_code_rl_ictai.sast.models import ToolName
from secure_code_rl_ictai.sast.runner import (
    Language,
    MockAdapter,
    SastRunner,
    SastTier,
)


def _all_mock_adapters() -> dict[ToolName, MockAdapter]:
    return {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, {"runs": []}),
        ToolName.SEMGREP: MockAdapter(ToolName.SEMGREP, {"runs": []}),
        ToolName.BANDIT: MockAdapter(ToolName.BANDIT, {"runs": []}),
        ToolName.CPPCHECK: MockAdapter(ToolName.CPPCHECK, {"runs": []}),
    }


def test_tier_all_runs_every_tool_python(tmp_path: Path):
    runner = SastRunner(_all_mock_adapters(), tier=SastTier.ALL)
    summary, _ = runner.run("import os", Language.PYTHON, tmp_path)
    invoked = set(summary.per_tool.keys())
    # Python routing: CodeQL, Semgrep, Bandit. All run under tier=all.
    assert invoked == {ToolName.CODEQL, ToolName.SEMGREP, ToolName.BANDIT}


def test_tier_cheap_skips_codeql_python(tmp_path: Path):
    runner = SastRunner(_all_mock_adapters(), tier=SastTier.CHEAP)
    summary, _ = runner.run("import os", Language.PYTHON, tmp_path)
    invoked = set(summary.per_tool.keys())
    # Tier-cheap drops CodeQL on Python.
    assert ToolName.CODEQL not in invoked
    assert ToolName.SEMGREP in invoked
    assert ToolName.BANDIT in invoked


def test_tier_cheap_skips_codeql_c(tmp_path: Path):
    runner = SastRunner(_all_mock_adapters(), tier=SastTier.CHEAP)
    summary, _ = runner.run("int main(){}", Language.C, tmp_path)
    invoked = set(summary.per_tool.keys())
    assert ToolName.CODEQL not in invoked
    assert ToolName.SEMGREP in invoked
    assert ToolName.CPPCHECK in invoked


def test_tier_cheap_plus_periodic_first_step_skips_deep(tmp_path: Path):
    """Step 0 of a cheap+periodic run: cheap only, no CodeQL."""
    runner = SastRunner(
        _all_mock_adapters(),
        tier=SastTier.CHEAP_PLUS_PERIODIC,
        deep_period=50,
    )
    summary, _ = runner.run("import os", Language.PYTHON, tmp_path)
    invoked = set(summary.per_tool.keys())
    assert ToolName.CODEQL not in invoked


def test_tier_cheap_plus_periodic_every_n_runs_deep(tmp_path: Path):
    """On step N (one-indexed), the deep tools fire alongside the cheap ones."""
    runner = SastRunner(
        _all_mock_adapters(),
        tier=SastTier.CHEAP_PLUS_PERIODIC,
        deep_period=3,
    )
    # Three runs to bring us up to step 3.
    for _ in range(3):
        summary, _ = runner.run("import os", Language.PYTHON, tmp_path)
    invoked = set(summary.per_tool.keys())
    # On the third call (deep_period=3), CodeQL should fire.
    assert ToolName.CODEQL in invoked


def test_tier_cheap_plus_periodic_between_deep_runs_skips_deep(tmp_path: Path):
    runner = SastRunner(
        _all_mock_adapters(),
        tier=SastTier.CHEAP_PLUS_PERIODIC,
        deep_period=5,
    )
    # Calls 1 and 2 are between deep runs.
    runner.run("a", Language.PYTHON, tmp_path)
    summary, _ = runner.run("b", Language.PYTHON, tmp_path)
    assert ToolName.CODEQL not in summary.per_tool


def test_tier_string_constructor_accepts_legacy_string():
    """Backward compatibility: tier may be passed as 'all' / 'cheap' / 'cheap+periodic'."""
    runner = SastRunner(_all_mock_adapters(), tier="cheap")
    assert runner.tier == SastTier.CHEAP


def test_legacy_no_tier_kwarg_defaults_to_all(tmp_path: Path):
    """Construction without the tier kwarg keeps the old behaviour."""
    runner = SastRunner(_all_mock_adapters())
    assert runner.tier == SastTier.ALL
    summary, _ = runner.run("import os", Language.PYTHON, tmp_path)
    assert ToolName.CODEQL in summary.per_tool
