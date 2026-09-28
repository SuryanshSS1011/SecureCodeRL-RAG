"""Tests for the CWE parent/child hierarchy module and the normalizer's
parent-collapsing logic.

The hierarchy is loaded from data/cwe_hierarchy.json (a static, hand-curated
file derived from MITRE's CWE database).
"""

from __future__ import annotations


import pytest

from cargo.sast.cwe_hierarchy import (
    CweHierarchy,
    is_related,
    load_default_hierarchy,
)
from cargo.sast.models import Finding, Location, Tier, ToolName
from cargo.sast.normalizer import SarifNormalizer


# ----------------------------------------------------------------------
# CweHierarchy
# ----------------------------------------------------------------------


def test_hierarchy_loads_from_default_path():
    h = load_default_hierarchy()
    # Sanity: known parents from data/cwe_hierarchy.json.
    assert "CWE-119" in h.parents_of("CWE-787")
    assert "CWE-74" in h.parents_of("CWE-89")


def test_hierarchy_returns_empty_for_unknown_cwe():
    h = load_default_hierarchy()
    assert h.parents_of("CWE-99999") == set()


def test_ancestors_traverses_transitively():
    h = load_default_hierarchy()
    # CWE-89 -> CWE-74 -> CWE-707
    anc = h.ancestors_of("CWE-89")
    assert "CWE-74" in anc
    assert "CWE-707" in anc


def test_is_related_detects_parent():
    h = load_default_hierarchy()
    assert is_related("CWE-787", "CWE-119", h)  # parent
    assert is_related("CWE-119", "CWE-787", h)  # child (symmetric)


def test_is_related_detects_transitive_ancestor():
    h = load_default_hierarchy()
    assert is_related("CWE-89", "CWE-707", h)  # grandparent
    assert is_related("CWE-707", "CWE-89", h)


def test_is_related_returns_true_for_self():
    h = load_default_hierarchy()
    assert is_related("CWE-89", "CWE-89", h)


def test_is_related_unrelated_cwes_false():
    h = load_default_hierarchy()
    # CWE-89 (SQLi, injection) vs CWE-787 (OOB write, memory): no relation in our map.
    assert not is_related("CWE-89", "CWE-787", h)


def test_hierarchy_from_dict():
    h = CweHierarchy({"CWE-A": ["CWE-B"], "CWE-B": ["CWE-C"]})
    assert h.parents_of("CWE-A") == {"CWE-B"}
    assert h.ancestors_of("CWE-A") == {"CWE-B", "CWE-C"}


def test_hierarchy_handles_cycles_defensively():
    """Pathological input: A -> B -> A. ancestors_of should terminate."""
    h = CweHierarchy({"CWE-A": ["CWE-B"], "CWE-B": ["CWE-A"]})
    # Both should be reachable; terminating without infinite loop is the test.
    anc = h.ancestors_of("CWE-A")
    assert "CWE-B" in anc


# ----------------------------------------------------------------------
# Normalizer integration: parent/child collapse during merge
# ----------------------------------------------------------------------


def _f(cwe: str, tool: ToolName, conf: float, sev: float | None, line: int = 10) -> Finding:
    if sev is not None:
        tier = Tier.HIGH if sev >= 7.0 else Tier.MEDIUM
    else:
        tier = Tier.MEDIUM
    return Finding(
        cwe=cwe,
        tier=tier,
        tool=tool,
        security_severity=sev,
        confidence=conf,
        location=Location(file="x.py", start_line=line),
        rule_id=f"r-{tool.value}",
    )


def test_merge_collapses_parent_and_child_at_same_location():
    """Two tools at the same location find CWE-87 and CWE-74 (parent of CWE-87
    in our hierarchy). The merged result should treat them as one finding
    and apply the corroboration boost."""
    normalizer = SarifNormalizer(
        cwe_hierarchy=load_default_hierarchy(),
    )
    a = _f("CWE-89", ToolName.CODEQL, conf=0.7, sev=8.8, line=10)
    b = _f("CWE-74", ToolName.SEMGREP, conf=0.6, sev=7.0, line=10)
    merged = normalizer.merge([[a], [b]])
    assert len(merged) == 1
    m = merged[0]
    # The more specific CWE wins (CWE-89 is a descendant of CWE-74).
    assert m.cwe == "CWE-89"
    # Corroboration boost applied: +0.15 for the extra tool.
    assert m.confidence == pytest.approx(0.85)
    assert ToolName.CODEQL in m.corroborating_tools
    assert ToolName.SEMGREP in m.corroborating_tools


def test_merge_does_not_collapse_unrelated_cwes_at_same_location():
    """CWE-89 (injection) and CWE-787 (memory) at the same location are
    distinct findings; should NOT collapse."""
    normalizer = SarifNormalizer(
        cwe_hierarchy=load_default_hierarchy(),
    )
    a = _f("CWE-89", ToolName.CODEQL, conf=0.7, sev=8.8, line=10)
    b = _f("CWE-787", ToolName.CPPCHECK, conf=0.6, sev=7.0, line=10)
    merged = normalizer.merge([[a], [b]])
    assert len(merged) == 2


def test_merge_collapses_transitive_ancestor():
    """CWE-89 and CWE-707 (its grandparent in the hierarchy)."""
    normalizer = SarifNormalizer(
        cwe_hierarchy=load_default_hierarchy(),
    )
    a = _f("CWE-89", ToolName.CODEQL, conf=0.7, sev=8.8, line=10)
    b = _f("CWE-707", ToolName.SEMGREP, conf=0.6, sev=7.0, line=10)
    merged = normalizer.merge([[a], [b]])
    assert len(merged) == 1
    # More specific descendant is the canonical CWE.
    assert merged[0].cwe == "CWE-89"


def test_merge_without_hierarchy_uses_exact_match_only():
    """Without a hierarchy, the parent/child collapse must NOT fire.
    This is the legacy behaviour and the default; consumers opt in by
    passing a hierarchy."""
    normalizer = SarifNormalizer()  # no hierarchy
    a = _f("CWE-89", ToolName.CODEQL, conf=0.7, sev=8.8, line=10)
    b = _f("CWE-74", ToolName.SEMGREP, conf=0.6, sev=7.0, line=10)
    merged = normalizer.merge([[a], [b]])
    assert len(merged) == 2  # not collapsed
