"""Tests for SARIF data models, severity sourcing, rule mapping, and the
normalizer's confidence-floor + corroboration logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from secure_code_rl_ictai.sast.models import (
    Finding,
    Location,
    Tier,
    ToolName,
    tier_from_severity,
)
from secure_code_rl_ictai.sast.normalizer import SarifNormalizer
from secure_code_rl_ictai.sast.rule_map import rule_to_cwe
from secure_code_rl_ictai.sast.severity import SeveritySource


# ----------------------------------------------------------------------
# Tier banding
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "severity,expected",
    [
        (10.0, Tier.CRITICAL),
        (9.0, Tier.CRITICAL),
        (8.9, Tier.HIGH),
        (7.0, Tier.HIGH),
        (6.9, Tier.MEDIUM),
        (4.0, Tier.MEDIUM),
        (3.9, Tier.LOW),
        (0.1, Tier.LOW),
        (0.0, Tier.LOW),
    ],
)
def test_tier_from_severity_band_cutoffs(severity, expected):
    assert tier_from_severity(severity) == expected


# ----------------------------------------------------------------------
# Finding validation
# ----------------------------------------------------------------------


def test_finding_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        Finding(
            cwe="CWE-787",
            tier=Tier.HIGH,
            tool=ToolName.CODEQL,
            security_severity=8.5,
            confidence=1.5,
            location=Location(file="a.py", start_line=1),
            rule_id="r",
        )


def test_finding_rejects_out_of_range_severity():
    with pytest.raises(ValueError):
        Finding(
            cwe="CWE-787",
            tier=Tier.HIGH,
            tool=ToolName.CODEQL,
            security_severity=11.0,
            confidence=0.8,
            location=Location(file="a.py", start_line=1),
            rule_id="r",
        )


def test_finding_rejects_malformed_cwe():
    with pytest.raises(ValueError):
        Finding(
            cwe="787",
            tier=Tier.HIGH,
            tool=ToolName.CODEQL,
            security_severity=8.5,
            confidence=0.8,
            location=Location(file="a.py", start_line=1),
            rule_id="r",
        )


def test_location_normalized_key_drops_column():
    loc = Location(file="src/foo.py", start_line=42, start_column=10)
    assert loc.normalized_key() == ("src/foo.py", 42)


# ----------------------------------------------------------------------
# Rule -> CWE mapping
# ----------------------------------------------------------------------


def test_rule_to_cwe_prefers_sarif_taxa():
    """SARIF standards-compliant path: relationships -> taxa -> CWE component."""
    sarif_rule = {
        "id": "py/sql-injection",
        "relationships": [
            {
                "target": {
                    "id": "89",
                    "toolComponent": {"name": "CWE"},
                }
            }
        ],
        # tag fallback would say CWE-79; taxa should win
        "properties": {"tags": ["CWE-79"]},
    }
    assert rule_to_cwe(ToolName.CODEQL, "py/sql-injection", sarif_rule) == "CWE-89"


def test_rule_to_cwe_falls_through_to_tags():
    sarif_rule = {"properties": {"tags": ["security", "CWE-78"]}}
    assert rule_to_cwe(ToolName.SEMGREP, "shell-injection", sarif_rule) == "CWE-78"


def test_rule_to_cwe_falls_through_to_properties_cwe():
    sarif_rule = {"properties": {"cwe": "476"}}
    assert rule_to_cwe(ToolName.CPPCHECK, "nullPointer", sarif_rule) == "CWE-476"


def test_rule_to_cwe_returns_none_when_unmappable():
    # Use a fabricated rule_id that is NOT in the _BANDIT_TO_CWE static map
    # and has no CWE in SARIF metadata. "B999_unknown" is intentionally not
    # a real Bandit rule; mapping must return None.
    sarif_rule = {"properties": {"tags": ["style"]}}
    assert rule_to_cwe(ToolName.BANDIT, "B999_unknown", sarif_rule) is None


def test_rule_to_cwe_uses_static_map_for_bandit_when_sarif_lacks_cwe():
    """bandit-sarif-formatter 1.x emits empty rule properties (no CWE in
    SARIF). We resolve via the static _BANDIT_TO_CWE map, sourced from the
    Bandit rule docs. Regression for the empty-properties case observed on
    ROAR 2026-06-10."""
    assert rule_to_cwe(ToolName.BANDIT, "B602", {"properties": {}}) == "CWE-78"
    assert rule_to_cwe(ToolName.BANDIT, "B608", {}) == "CWE-89"
    assert rule_to_cwe(ToolName.BANDIT, "B324", None) == "CWE-327"


def test_rule_to_cwe_uses_static_map_for_cppcheck_when_sarif_lacks_cwe():
    """Cppcheck 2.18 SARIF emits no CWE on rules — only `tags: ["security"]`
    when severity=error and id is non-critical. The internal CWE per checker
    isn't serialized. We resolve via the static _CPPCHECK_TO_CWE map sourced
    from upstream `lib/check*.cpp` `CWE(NNN)` calls."""
    assert rule_to_cwe(ToolName.CPPCHECK, "memleak", {}) == "CWE-401"
    assert rule_to_cwe(ToolName.CPPCHECK, "deallocuse", {}) == "CWE-416"   # use-after-free
    assert rule_to_cwe(ToolName.CPPCHECK, "doubleFree", {}) == "CWE-415"
    assert rule_to_cwe(ToolName.CPPCHECK, "nullPointer", {}) == "CWE-476"
    assert rule_to_cwe(ToolName.CPPCHECK, "integerOverflow", {}) == "CWE-190"


# ----------------------------------------------------------------------
# Severity sourcing
# ----------------------------------------------------------------------


def test_severity_uses_sarif_score_when_present():
    src = SeveritySource()
    f = Finding(
        cwe="CWE-787",
        tier=Tier.HIGH,
        tool=ToolName.CODEQL,
        security_severity=8.4,
        confidence=0.8,
        location=Location(file="a.c", start_line=1),
        rule_id="cpp/oob-write",
    )
    assert src.severity_with_source(f)[0] == 8.4
    assert src.diagnostics()["fallback_total"] == 0


def test_severity_falls_back_to_nvdlib_median(tmp_path: Path):
    medians_path = tmp_path / "medians.json"
    medians_path.write_text(
        json.dumps(
            {
                "_meta": {"status": "populated", "spec_version": "test"},
                "medians": {"CWE-787": {"median": 8.1, "n_cves": 100}},
            }
        )
    )
    src = SeveritySource(medians_path)
    f = Finding(
        cwe="CWE-787",
        tier=Tier.HIGH,
        tool=ToolName.CPPCHECK,
        security_severity=None,  # tool didn't emit
        confidence=0.7,
        location=Location(file="a.c", start_line=1),
        rule_id="bufferAccessOutOfBounds",
    )
    assert src.severity_with_source(f)[0] == 8.1
    diag = src.diagnostics()
    assert diag["fallback_total"] == 1
    assert diag["cwe_median_fallback"] == 1
    assert diag["tier_midpoint_fallback"] == 0


def test_severity_final_fallback_is_tier_midpoint(tmp_path: Path):
    """No SARIF score, no NVDLib entry -> tier midpoint."""
    medians_path = tmp_path / "medians.json"
    medians_path.write_text(
        json.dumps(
            {
                "_meta": {"status": "populated", "spec_version": "test"},
                "medians": {},  # empty
            }
        )
    )
    src = SeveritySource(medians_path)
    f = Finding(
        cwe="CWE-999",
        tier=Tier.HIGH,
        tool=ToolName.BANDIT,
        security_severity=None,
        confidence=0.6,
        location=Location(file="a.py", start_line=1),
        rule_id="B999",
    )
    assert src.severity_with_source(f)[0] == 7.95  # High midpoint
    diag = src.diagnostics()
    assert diag["tier_midpoint_fallback"] == 1


def test_severity_handles_placeholder_medians(tmp_path: Path):
    """A placeholder medians file should not crash; treat as empty."""
    medians_path = tmp_path / "medians.json"
    medians_path.write_text(
        json.dumps({"_meta": {"status": "placeholder"}, "medians": {"CWE-787": {"median": 8.1}}})
    )
    src = SeveritySource(medians_path)
    f = Finding(
        cwe="CWE-787",
        tier=Tier.HIGH,
        tool=ToolName.CPPCHECK,
        security_severity=None,
        confidence=0.6,
        location=Location(file="a.c", start_line=1),
        rule_id="r",
    )
    # placeholder should be ignored -> tier midpoint
    assert src.severity_with_source(f)[0] == 7.95


# ----------------------------------------------------------------------
# Normalizer: SARIF parsing
# ----------------------------------------------------------------------


def _sarif_with_one_result(
    rule_id: str,
    rule_props: dict,
    result_props: dict | None = None,
    level: str | None = None,
    file: str = "src/foo.py",
    line: int = 10,
) -> dict:
    """Construct a minimal SARIF v2.1.0 document with one result."""
    result = {
        "ruleId": rule_id,
        "message": {"text": "test finding"},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": file},
                    "region": {"startLine": line},
                }
            }
        ],
    }
    if level is not None:
        result["level"] = level
    if result_props is not None:
        result["properties"] = result_props
    return {
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "test-tool",
                        "rules": [{"id": rule_id, **rule_props}],
                    }
                },
                "results": [result],
            }
        ]
    }


def test_normalizer_parses_codeql_style_severity():
    """CodeQL emits security-severity in rule.properties."""
    sarif = _sarif_with_one_result(
        rule_id="py/sql-injection",
        rule_props={
            "relationships": [
                {"target": {"id": "89", "toolComponent": {"name": "CWE"}}}
            ],
            "properties": {"security-severity": "8.8"},
        },
        level="error",
    )
    findings = SarifNormalizer().parse_sarif(ToolName.CODEQL, sarif)
    assert len(findings) == 1
    f = findings[0]
    assert f.cwe == "CWE-89"
    assert f.security_severity == 8.8
    assert f.tier == Tier.HIGH
    assert f.confidence == 0.8  # SARIF level=error -> 0.8


def test_normalizer_drops_findings_below_confidence_floor():
    """A LOW-severity finding with a tool-emitted note level (conf=0.3)
    against LOW tier (floor 0.3) sits at the floor. Dial conf to 0.2 by
    using a non-standard level and confirm it drops.

    Note: this test does NOT use a high security-severity because the
    cross-signal boost (sev>=7 -> conf>=0.8) would override the low
    level. The drop semantics are still tested below — they just need a
    finding without a high severity to trigger the floor path."""
    sarif = _sarif_with_one_result(
        rule_id="r1",
        rule_props={
            "properties": {"tags": ["CWE-79"], "security-severity": "2.0"},
        },
        # level: not "error"/"warning"/"note" -> _coerce_confidence returns
        # default=0.5. We bypass that by passing the level=None and a
        # numeric confidence under the floor.
        level=None,
        result_props={"confidence": 0.2},
    )
    findings = SarifNormalizer().parse_sarif(ToolName.SEMGREP, sarif)
    assert findings == []


def test_normalizer_boosts_confidence_when_security_severity_is_high():
    """Cppcheck emits SARIF `level: warning` (0.5) on findings with
    security-severity=9.9 — without a boost, every cppcheck CRITICAL
    finding gets dropped by the per-tier floor (CRITICAL=0.6). Verify
    sev>=7 promotes confidence to >=0.8."""
    sarif = _sarif_with_one_result(
        rule_id="memleak",
        rule_props={
            "properties": {"security-severity": "9.9"},
        },
        level="warning",  # -> 0.5; boost to 0.8 because sev=9.9 >= 7
    )
    findings = SarifNormalizer().parse_sarif(ToolName.CPPCHECK, sarif)
    assert len(findings) == 1
    assert findings[0].confidence >= 0.8
    assert findings[0].cwe == "CWE-401"


def test_normalizer_keeps_findings_at_or_above_floor():
    sarif = _sarif_with_one_result(
        rule_id="r1",
        rule_props={
            "properties": {"tags": ["CWE-79"], "security-severity": "5.0"},
        },
        level="warning",  # -> 0.5; Medium floor is 0.4
    )
    findings = SarifNormalizer().parse_sarif(ToolName.SEMGREP, sarif)
    assert len(findings) == 1
    assert findings[0].confidence == 0.5


def test_normalizer_drops_unmappable_findings():
    # B999_unknown is intentionally not in the static map; with empty SARIF
    # CWE metadata it should be dropped.
    sarif = _sarif_with_one_result(
        rule_id="B999_unknown",
        rule_props={"properties": {"tags": ["style"]}},
        level="warning",
    )
    findings = SarifNormalizer().parse_sarif(ToolName.BANDIT, sarif)
    assert findings == []


# ----------------------------------------------------------------------
# Normalizer: cross-tool merge
# ----------------------------------------------------------------------


def _f(cwe: str, tool: ToolName, conf: float, sev: float | None, line: int = 10) -> Finding:
    return Finding(
        cwe=cwe,
        tier=tier_from_severity(sev) if sev is not None else Tier.MEDIUM,
        tool=tool,
        security_severity=sev,
        confidence=conf,
        location=Location(file="x.py", start_line=line),
        rule_id=f"rule-{tool.value}",
    )


def test_merge_corroboration_boost_two_tools():
    a = _f("CWE-89", ToolName.CODEQL, conf=0.7, sev=8.8)
    b = _f("CWE-89", ToolName.SEMGREP, conf=0.6, sev=8.5)
    merged = SarifNormalizer().merge([[a], [b]])
    assert len(merged) == 1
    m = merged[0]
    # primary is the higher-confidence one
    assert m.tool == ToolName.CODEQL
    # one extra corroborator -> +0.15
    assert m.confidence == pytest.approx(0.85)
    assert m.corroborating_tools == frozenset({ToolName.CODEQL, ToolName.SEMGREP})
    # best severity
    assert m.security_severity == 8.8


def test_merge_corroboration_boost_capped():
    a = _f("CWE-89", ToolName.CODEQL, conf=0.95, sev=8.8)
    b = _f("CWE-89", ToolName.SEMGREP, conf=0.9, sev=8.5)
    c = _f("CWE-89", ToolName.BANDIT, conf=0.85, sev=None)
    merged = SarifNormalizer().merge([[a], [b], [c]])
    assert merged[0].confidence == 1.0  # capped


def test_merge_keeps_distinct_findings_separate():
    a = _f("CWE-89", ToolName.CODEQL, conf=0.7, sev=8.8, line=10)
    b = _f("CWE-79", ToolName.SEMGREP, conf=0.6, sev=6.0, line=10)  # different CWE
    c = _f("CWE-89", ToolName.SEMGREP, conf=0.6, sev=8.5, line=20)  # different line
    merged = SarifNormalizer().merge([[a], [b, c]])
    assert len(merged) == 3


def test_merge_severity_is_none_when_no_tool_emits():
    a = _f("CWE-476", ToolName.SEMGREP, conf=0.7, sev=None)
    b = _f("CWE-476", ToolName.CPPCHECK, conf=0.6, sev=None)
    merged = SarifNormalizer().merge([[a], [b]])
    assert merged[0].security_severity is None
