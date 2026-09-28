"""Tests for the reward calculator (paper Eq. 4 and Table I)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cargo.reward.calculator import (
    ReliabilitySignals,
    RewardCalculator,
    RewardConfig,
)
from cargo.sast.models import Finding, Location, Tier, ToolName
from cargo.sast.severity import SeveritySource


def _finding(
    cwe: str = "CWE-89",
    severity: float | None = 8.8,
    confidence: float = 0.8,
    tool: ToolName = ToolName.CODEQL,
    line: int = 1,
) -> Finding:
    tier_lookup = {
        9.0: Tier.CRITICAL,
        8.8: Tier.HIGH,
        7.0: Tier.HIGH,
        5.5: Tier.MEDIUM,
        2.0: Tier.LOW,
    }
    if severity is None:
        tier = Tier.MEDIUM
    else:
        tier = tier_lookup.get(severity, Tier.HIGH if severity >= 7 else Tier.MEDIUM)
    return Finding(
        cwe=cwe,
        tier=tier,
        tool=tool,
        security_severity=severity,
        confidence=confidence,
        location=Location(file="x.py", start_line=line),
        rule_id="r",
    )


# ----------------------------------------------------------------------
# Reliability component
# ----------------------------------------------------------------------


def test_reliability_zero_when_nothing_works():
    calc = RewardCalculator(RewardConfig())
    b = calc.compute(ReliabilitySignals(), findings=[])
    assert b.r_reliability == 0.0
    assert b.r_total == 0.0


def test_reliability_full_credit_when_all_tests_pass():
    calc = RewardCalculator(RewardConfig())
    b = calc.compute(
        ReliabilitySignals(
            compiles=True, runs=True, produces_output=True,
            tests_passed=5, tests_total=5,
        ),
        findings=[],
    )
    # 0.2 + 0.2 + 0.2 + 0.4 * 1.0 = 1.0
    assert b.r_reliability == pytest.approx(1.0)


def test_reliability_partial_credit_proportional_to_tests():
    calc = RewardCalculator(RewardConfig())
    b = calc.compute(
        ReliabilitySignals(
            compiles=True, runs=True, produces_output=True,
            tests_passed=3, tests_total=5,
        ),
        findings=[],
    )
    # 0.6 + 0.4 * 3/5 = 0.84
    assert b.r_reliability == pytest.approx(0.84)


def test_reliability_renormalizes_on_no_test_prompts():
    """When tests_total == 0, redistribute the r_func budget across the
    three remaining components so max-reachable r_reliability is still 1.0.

    96.6% of v0.1.7 train prompts have no tests. Without this, the model
    would systematically receive weaker per-prompt reward on no-test
    prompts, biasing the gradient toward the rare tested prompts and
    leaving most of the corpus under-utilized."""
    calc = RewardCalculator(RewardConfig())
    full = calc.compute(
        ReliabilitySignals(
            compiles=True, runs=True, produces_output=True,
            tests_passed=0, tests_total=0,
        ),
        findings=[],
    )
    # Compile + run + output, each (0.2 + 0.4/3) = 0.3333, sum = 1.0
    assert full.r_reliability == pytest.approx(1.0)
    only_compile = calc.compute(
        ReliabilitySignals(
            compiles=True, runs=False, produces_output=False,
            tests_passed=0, tests_total=0,
        ),
        findings=[],
    )
    assert only_compile.r_reliability == pytest.approx(1.0 / 3.0)


# ----------------------------------------------------------------------
# Functionality gate (diagnostic)
# ----------------------------------------------------------------------


def test_gate_zero_when_no_tests_pass_even_if_compiles():
    """The gate requires compile AND >=1 test pass. Compile alone is not enough."""
    sigs = ReliabilitySignals(compiles=True, tests_passed=0, tests_total=5)
    assert sigs.gate == 0


def test_gate_zero_when_compiles_but_no_tests_run():
    """If no tests are configured (tests_total=0), gate is 0."""
    sigs = ReliabilitySignals(compiles=True, tests_passed=0, tests_total=0)
    assert sigs.gate == 0


def test_gate_one_when_compiles_and_at_least_one_test_passes():
    sigs = ReliabilitySignals(compiles=True, tests_passed=1, tests_total=10)
    assert sigs.gate == 1


def test_non_parsing_code_gets_zero_security():
    """Non-parsing code has no findings but must not share the clean-code
    reward of 1 (Section II-B)."""
    calc = RewardCalculator(RewardConfig())
    b = calc.compute(ReliabilitySignals(compiles=False), findings=[])
    assert b.r_security == 0.0
    assert b.r_total == 0.0


def test_clean_compiling_code_gets_full_security():
    calc = RewardCalculator(RewardConfig())
    b = calc.compute(ReliabilitySignals(compiles=True), findings=[])
    assert b.r_security == 1.0
    assert b.clipped is False


# ----------------------------------------------------------------------
# CVSS-weighted security score
# ----------------------------------------------------------------------


def test_single_finding_penalty_is_severity_times_confidence_over_10():
    calc = RewardCalculator(RewardConfig())
    b = calc.compute(
        ReliabilitySignals(compiles=True),
        findings=[_finding(severity=7.5, confidence=0.8)],
    )
    # 1 - 7.5/10 * 0.8 = 0.4
    assert b.r_security == pytest.approx(0.4)


def test_multi_finding_penalties_sum():
    calc = RewardCalculator(RewardConfig())
    findings = [
        _finding(severity=5.5, confidence=0.6, line=1),  # 0.33
        _finding(severity=2.0, confidence=0.5, line=2),  # 0.10
    ]
    b = calc.compute(ReliabilitySignals(compiles=True), findings=findings)
    assert b.r_security == pytest.approx(0.57)


def test_security_floors_at_zero():
    calc = RewardCalculator(RewardConfig())
    findings = [_finding(severity=10.0, confidence=1.0, line=i) for i in range(5)]
    b = calc.compute(ReliabilitySignals(compiles=True), findings=findings)
    assert b.r_security == 0.0
    assert b.clipped is True


def test_severity_sourced_from_nvdlib_when_sarif_absent(tmp_path: Path):
    medians = tmp_path / "medians.json"
    medians.write_text(
        json.dumps(
            {
                "_meta": {"status": "populated", "spec_version": "test"},
                "medians": {"CWE-787": {"median": 7.5, "n_cves": 50}},
            }
        )
    )
    calc = RewardCalculator(RewardConfig(), SeveritySource(medians))
    b = calc.compute(
        ReliabilitySignals(compiles=True),
        findings=[
            Finding(
                cwe="CWE-787",
                tier=Tier.HIGH,
                tool=ToolName.CPPCHECK,
                security_severity=None,  # tool didn't emit
                confidence=1.0,
                location=Location(file="x.c", start_line=1),
                rule_id="r",
            )
        ],
    )
    # NVDLib median 7.5 -> penalty 0.75 -> R_sec 0.25
    assert b.r_security == pytest.approx(0.25)


# ----------------------------------------------------------------------
# Composition into r (Eq. 4)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "sigs,findings,r_rel,r_sec,r",
    [
        (ReliabilitySignals(), [], 0.0, 0.0, 0.0),
        (ReliabilitySignals(compiles=True), [], 1 / 3, 1.0, 0.53),
        (
            ReliabilitySignals(compiles=True),
            [_finding(severity=8.8, confidence=0.8, line=i) for i in range(2)],
            1 / 3, 0.0, 0.23,
        ),
        (
            ReliabilitySignals(compiles=True, runs=True, produces_output=True),
            [_finding(severity=8.8, confidence=0.8)],
            1.0, 0.296, 0.79,
        ),
        (ReliabilitySignals(compiles=True, runs=True, produces_output=True), [], 1.0, 1.0, 1.0),
    ],
)
def test_reward_by_rollout_state_matches_table_one(sigs, findings, r_rel, r_sec, r):
    calc = RewardCalculator(RewardConfig())
    b = calc.compute(sigs, findings=findings)
    assert b.r_reliability == pytest.approx(r_rel)
    assert b.r_security == pytest.approx(r_sec)
    assert b.r_total == pytest.approx(r, abs=0.005)


def test_r_rag_added_to_total():
    calc = RewardCalculator(RewardConfig())
    b = calc.compute(ReliabilitySignals(compiles=True), findings=[], r_rag=0.05)
    # 0.3 * 1 + 0.7 * (1/3) + 0.05
    assert b.r_total == pytest.approx(0.3 + 0.7 / 3 + 0.05)


def test_stub_penalty_subtracted():
    calc = RewardCalculator(RewardConfig())
    b = calc.compute(ReliabilitySignals(), findings=[], is_stub=True)
    assert b.r_total == pytest.approx(-1.5)


# ----------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------


def test_cache_roundtrip(tmp_path: Path):
    calc = RewardCalculator(RewardConfig(alpha=0.5, cache_dir=tmp_path))
    sigs = ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1)
    findings = [_finding(severity=8.0, confidence=1.0)]
    h = calc.canonicalize_completion("def f():\n    return 1\n")
    b1 = calc.compute(sigs, findings, completion_hash=h)
    # second call should hit cache; result must match
    b2 = calc.compute(sigs, findings, completion_hash=h)
    assert b1.r_total == b2.r_total
    assert b1.r_security == b2.r_security


def test_canonicalize_normalizes_trailing_whitespace():
    a = RewardCalculator.canonicalize_completion("x = 1\n")
    b = RewardCalculator.canonicalize_completion("x = 1\n\n\n")
    c = RewardCalculator.canonicalize_completion("x = 1")
    d = RewardCalculator.canonicalize_completion("x = 1\r\n")
    assert a == b == c == d


def test_canonicalize_does_not_collapse_internal_whitespace():
    a = RewardCalculator.canonicalize_completion("x = 1\ny = 2\n")
    b = RewardCalculator.canonicalize_completion("x = 1\n\ny = 2\n")
    assert a != b


# ----------------------------------------------------------------------
# Per-finding breakdown (diagnostics)
# ----------------------------------------------------------------------


def test_per_finding_breakdown_records_severity_source_outcome():
    calc = RewardCalculator(RewardConfig(alpha=1.0))
    b = calc.compute(
        ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
        findings=[_finding(severity=8.8, confidence=0.8)],
    )
    assert len(b.per_finding) == 1
    entry = b.per_finding[0]
    assert entry["cwe"] == "CWE-89"
    assert entry["severity"] == 8.8
    assert entry["base"] == pytest.approx(0.88, rel=1e-3)
    assert entry["confidence"] == 0.8
    assert entry["penalty"] == pytest.approx(0.704, rel=1e-3)
