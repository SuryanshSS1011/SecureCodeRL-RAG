"""Tests for the corroboration-aware severity sourcing pipeline.

Pins the four-way source-path semantics from docs/reward_spec.md §3 and
the decision recorded in OPEN_ADVISER_ITEMS.md §3 Caveat B:

    sarif:primary       - primary detecting tool emits security-severity.
    sarif:corroborator  - primary tool does not emit it; a corroborating
                          tool in the same dedup cluster does. The
                          normalizer's merge() carries that severity
                          into Finding.security_severity.
    nvdlib_median       - no tool in the cluster emits severity; fall
                          back to per-CWE empirical median.
    tier_midpoint       - the CWE is not in the median table.

These tests pin the sourcing logic; the corroboration logic itself
(merge step) is tested in test_sast.py.
"""

from __future__ import annotations

import json
from pathlib import Path


from cargo.sast.models import Finding, Location, Tier, ToolName
from cargo.sast.severity import SeveritySource


def _finding(
    *,
    cwe: str = "CWE-89",
    tool: ToolName = ToolName.CODEQL,
    sev: float | None = 8.8,
    conf: float = 0.8,
) -> Finding:
    """Build a Finding with sensible defaults; tests override the
    interesting fields (tool, security_severity, cwe)."""
    return Finding(
        cwe=cwe,
        tier=Tier.HIGH if (sev is not None and sev >= 7.0) else Tier.MEDIUM,
        tool=tool,
        security_severity=sev,
        confidence=conf,
        location=Location(file="x.py", start_line=1),
        rule_id="r",
    )


def _medians_file(tmp_path: Path, medians: dict[str, float]) -> Path:
    p = tmp_path / "medians.json"
    p.write_text(
        json.dumps(
            {
                "_meta": {"status": "populated", "spec_version": "test"},
                "medians": {k: {"median": v, "n_cves": 10} for k, v in medians.items()},
            }
        )
    )
    return p


# ----------------------------------------------------------------------
# sarif:primary — primary tool emits severity
# ----------------------------------------------------------------------


def test_sarif_primary_when_codeql_is_primary_tool():
    src = SeveritySource()
    f = _finding(tool=ToolName.CODEQL, sev=8.8)
    sev, source = src.severity_with_source(f)
    assert sev == 8.8
    assert source == "sarif:primary"
    diag = src.diagnostics()
    assert diag["sarif_primary"] == 1
    assert diag["sarif_corroborator"] == 0
    assert diag["nvdlib_median"] == 0


def test_sarif_primary_when_semgrep_is_primary_tool():
    src = SeveritySource()
    f = _finding(tool=ToolName.SEMGREP, sev=7.5)
    sev, source = src.severity_with_source(f)
    assert source == "sarif:primary"


# ----------------------------------------------------------------------
# sarif:corroborator — primary tool does not emit; corroborator did
# ----------------------------------------------------------------------


def test_sarif_corroborator_when_bandit_primary_but_security_severity_set():
    """If the primary tool is Bandit (a non-emitter) but Finding.security_severity
    is set, the merge() step pulled it from a corroborating tool. Source label
    should be sarif:corroborator."""
    src = SeveritySource()
    f = _finding(tool=ToolName.BANDIT, sev=8.8)
    sev, source = src.severity_with_source(f)
    assert sev == 8.8
    assert source == "sarif:corroborator"
    diag = src.diagnostics()
    assert diag["sarif_corroborator"] == 1
    assert diag["sarif_primary"] == 0




# ----------------------------------------------------------------------
# nvdlib_median — no tool in the cluster emits; fall back to median
# ----------------------------------------------------------------------


def test_nvdlib_median_when_no_security_severity(tmp_path: Path):
    medians = _medians_file(tmp_path, {"CWE-89": 7.3})
    src = SeveritySource(medians)
    f = _finding(tool=ToolName.BANDIT, sev=None, cwe="CWE-89")
    sev, source = src.severity_with_source(f)
    assert sev == 7.3
    assert source == "nvdlib_median"
    diag = src.diagnostics()
    assert diag["nvdlib_median"] == 1
    assert diag["fallback_total"] == 1


# ----------------------------------------------------------------------
# tier_midpoint — final defensive fallback
# ----------------------------------------------------------------------


def test_tier_midpoint_when_cwe_not_in_medians_table(tmp_path: Path):
    medians = _medians_file(tmp_path, {})  # empty
    src = SeveritySource(medians)
    f = _finding(tool=ToolName.BANDIT, sev=None, cwe="CWE-99999")
    sev, source = src.severity_with_source(f)
    # CWE-99999 not in table; tier=HIGH (default) -> tier midpoint 7.95
    # But our test default has tier=MEDIUM for sev=None; check the right midpoint.
    # The Finding helper sets tier=MEDIUM when sev is None. Tier-MEDIUM midpoint = 5.45.
    assert source == "tier_midpoint"
    assert sev == 5.45


# ----------------------------------------------------------------------
# Diagnostics fan-out: stratified counts
# ----------------------------------------------------------------------


def test_diagnostics_split_across_paths(tmp_path: Path):
    medians = _medians_file(tmp_path, {"CWE-89": 7.3})
    src = SeveritySource(medians)
    # One sarif:primary (CodeQL with severity)
    src.severity_with_source(_finding(tool=ToolName.CODEQL, sev=8.0))
    # One sarif:corroborator (Bandit primary but severity set)
    src.severity_with_source(_finding(tool=ToolName.BANDIT, sev=8.0))
    # One nvdlib_median
    src.severity_with_source(_finding(tool=ToolName.BANDIT, sev=None, cwe="CWE-89"))
    # One tier_midpoint
    src.severity_with_source(_finding(tool=ToolName.BANDIT, sev=None, cwe="CWE-99999"))

    diag = src.diagnostics()
    assert diag["sarif_primary"] == 1
    assert diag["sarif_corroborator"] == 1
    assert diag["nvdlib_median"] == 1
    assert diag["tier_midpoint"] == 1
    # Legacy keys still populated for back-compat.
    assert diag["fallback_total"] == 3  # everything except sarif:primary
    assert diag["cwe_median_fallback"] == 1
    assert diag["tier_midpoint_fallback"] == 1



# ----------------------------------------------------------------------
# Calculator surfaces severity_source per finding
# ----------------------------------------------------------------------


def test_calculator_per_finding_includes_severity_source():
    """Reward calculator's per_finding records should carry the source label
    so the trainer log can stratify per-finding-vs-fallback."""
    from cargo.reward.calculator import (
        ReliabilitySignals,
        RewardCalculator,
        RewardConfig,
    )

    calc = RewardCalculator(RewardConfig(alpha=1.0), SeveritySource())
    f_primary = _finding(tool=ToolName.CODEQL, sev=8.8)
    f_corr = _finding(tool=ToolName.BANDIT, sev=8.0)
    breakdown = calc.compute(
        ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
        findings=[f_primary, f_corr],
    )
    sources = [e["severity_source"] for e in breakdown.per_finding]
    assert sources == ["sarif:primary", "sarif:corroborator"]
