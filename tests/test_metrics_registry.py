"""Tests for the MetricSpec registry.

Pins the compile-first conditional semantics. The registry is the
single source of truth for aggregate metrics and per-CWE/per-language
breakdowns — these tests cover the conditional math directly without
spinning up the full harness so a metric regression is local and easy
to read.
"""

from __future__ import annotations

import pytest

from secure_code_rl_ictai.eval.harness import PerPromptRecord
from secure_code_rl_ictai.eval.metrics import (
    METRICS,
    MetricSpec,
    compute_all,
    compute_metric,
    legacy_scalar_view,
)


def _rec(
    *,
    completion: str = "ok",
    compiles: bool = True,
    tests_passed: int = 0,
    tests_total: int = 0,
    target_cwe_present: bool = False,
    refusal_or_empty: bool = False,
    crashed: bool = False,
    target_cwe: str = "CWE-89",
    language: str = "python",
    prompt_id: str = "p1",
) -> PerPromptRecord:
    return PerPromptRecord(
        prompt_id=prompt_id,
        source="test",
        target_cwe=target_cwe,
        language=language,
        completion=completion,
        crashed=crashed,
        compiles=compiles,
        runs=False,
        produces_output=False,
        tests_passed=tests_passed,
        tests_total=tests_total,
        r_total=0.0,
        r_reliability=0.0,
        r_security=0.0,
        r_rag=0.0,
        findings_count=0,
        findings_cwes=[],
        target_cwe_present=target_cwe_present,
        refusal_or_empty=refusal_or_empty,
    )


def test_compile_at_1_counts_compiling_completions():
    recs = [_rec(compiles=True), _rec(compiles=True), _rec(compiles=False)]
    out = compute_all(recs)
    assert out["compile_at_1"]["value"] == pytest.approx(2 / 3)
    assert out["compile_at_1"]["n_numerator"] == 2
    assert out["compile_at_1"]["n_denominator"] == 3


def test_secure_at_1_compiles_uses_compileable_denominator():
    # 4 records. 2 compile (1 secure, 1 with target CWE). 2 don't compile.
    # secure_at_1__compiles = 1/2 over the compileable subset (not 1/4 or 3/4).
    recs = [
        _rec(compiles=True, target_cwe_present=False),   # secure & compiles
        _rec(compiles=True, target_cwe_present=True),    # insecure & compiles
        _rec(compiles=False, target_cwe_present=False),  # excluded by condition
        _rec(compiles=False, target_cwe_present=True),   # excluded by condition
    ]
    out = compute_all(recs)
    assert out["secure_at_1__compiles"]["value"] == pytest.approx(0.5)
    assert out["secure_at_1__compiles"]["n_numerator"] == 1
    assert out["secure_at_1__compiles"]["n_denominator"] == 2


def test_secure_at_1_unconditioned_inflated_by_noncompiling():
    # The legacy secure_at_1 over the full set credits the non-compiling
    # records as "secure" (target_cwe_present=False) — this is the bug
    # the conditional family fixes. This test pins that behavior so any
    # accidental change to the legacy key surfaces.
    recs = [
        _rec(compiles=True, target_cwe_present=False),
        _rec(compiles=True, target_cwe_present=True),
        _rec(compiles=False, target_cwe_present=False),
        _rec(compiles=False, target_cwe_present=False),
    ]
    out = compute_all(recs)
    # 3 of 4 are "secure" by the unconditioned metric; 1 of 2 by the
    # conditioned one.
    assert out["secure_at_1"]["value"] == pytest.approx(3 / 4)
    assert out["secure_at_1__compiles"]["value"] == pytest.approx(1 / 2)


def test_func_at_1_has_tests_skips_test_free_prompts():
    # 3 records: 2 with tests (1 passing), 1 without tests.
    # func_at_1__has_tests = 1/2 over the test-carrying subset.
    # The legacy func_at_1 = 1/3 over all records.
    recs = [
        _rec(tests_total=2, tests_passed=2),
        _rec(tests_total=2, tests_passed=0),
        _rec(tests_total=0, tests_passed=0),
    ]
    out = compute_all(recs)
    assert out["func_at_1__has_tests"]["value"] == pytest.approx(1 / 2)
    assert out["func_at_1__has_tests"]["n_denominator"] == 2
    assert out["func_at_1"]["value"] == pytest.approx(1 / 3)
    assert out["func_at_1"]["n_denominator"] == 3


def test_refusal_excludes_from_secure_numerator():
    # An empty completion has 0 findings → target_cwe_present=False.
    # Without the refusal guard it would count as secure (SafeCoder
    # failure mode). The base predicate excludes it.
    recs = [
        _rec(refusal_or_empty=True, target_cwe_present=False),
        _rec(refusal_or_empty=False, target_cwe_present=False),
    ]
    out = compute_all(recs)
    assert out["secure_at_1"]["value"] == pytest.approx(1 / 2)


def test_attempt_at_1_counts_non_refusal():
    recs = [
        _rec(refusal_or_empty=False, crashed=False),  # attempted
        _rec(refusal_or_empty=True),                  # refused
        _rec(crashed=True),                           # crashed
    ]
    out = compute_all(recs)
    assert out["attempt_at_1"]["value"] == pytest.approx(1 / 3)


def test_zero_denominator_does_not_div_by_zero():
    # No records compile -> secure_at_1__compiles has n_denominator=0
    # and value=0.0 with no exception.
    recs = [_rec(compiles=False), _rec(compiles=False)]
    out = compute_all(recs)
    assert out["secure_at_1__compiles"]["value"] == 0.0
    assert out["secure_at_1__compiles"]["n_denominator"] == 0
    assert out["secure_at_1__compiles"]["n_numerator"] == 0


def test_legacy_scalar_view_emits_only_legacy_keys():
    recs = [_rec(compiles=True, target_cwe_present=False)]
    per_spec = compute_all(recs)
    scalars = legacy_scalar_view(per_spec)
    assert set(scalars.keys()) == {"func_at_1", "secure_at_1", "func_sec_at_1"}
    assert all(isinstance(v, float) for v in scalars.values())


def test_func_sec_compiles_and_has_tests_intersects_correctly():
    # Only the record that compiles AND has tests is in the denominator.
    # In numerator: passes tests AND no target CWE.
    recs = [
        _rec(compiles=True, tests_total=1, tests_passed=1,
             target_cwe_present=False),                       # in num + denom
        _rec(compiles=True, tests_total=1, tests_passed=0,
             target_cwe_present=False),                       # denom, not num
        _rec(compiles=True, tests_total=0, target_cwe_present=False),  # excluded
        _rec(compiles=False, tests_total=1, tests_passed=1,
             target_cwe_present=False),                       # excluded
    ]
    out = compute_all(recs)
    spec_out = out["func_sec_at_1__compiles_and_has_tests"]
    assert spec_out["n_denominator"] == 2
    assert spec_out["n_numerator"] == 1
    assert spec_out["value"] == pytest.approx(0.5)


def test_compute_metric_with_explicit_spec():
    # Sanity: a custom spec composed inline behaves the same as the registry.
    spec = MetricSpec(
        "compiles_and_secure",
        base=lambda r: not r.target_cwe_present,
        condition=lambda r: r.compiles,
    )
    recs = [
        _rec(compiles=True, target_cwe_present=False),
        _rec(compiles=True, target_cwe_present=True),
        _rec(compiles=False),
    ]
    out = compute_metric(spec, recs)
    assert out["n_denominator"] == 2
    assert out["n_numerator"] == 1
    assert out["value"] == pytest.approx(0.5)


def test_registry_names_unique():
    names = [s.name for s in METRICS]
    assert len(names) == len(set(names)), f"duplicate metric names: {names}"


# ----- P1.4 severity binning -----


def test_severity_binned_metrics_split_by_nvdlib_median():
    """secure_at_1__high_severity_compiles only counts records on high-CVSS CWEs.

    CWE-78 has NVDLib median 8.8 (high); CWE-862 has 5.5 (medium). A
    record on CWE-78 that compiles AND is secure feeds high; the same
    record on CWE-862 feeds medium. Records that don't compile are
    excluded from both, matching the conditional metric's contract.
    """
    recs = [
        _rec(compiles=True, target_cwe="CWE-78", target_cwe_present=False),
        _rec(compiles=True, target_cwe="CWE-78", target_cwe_present=True),
        _rec(compiles=True, target_cwe="CWE-862", target_cwe_present=False),
        _rec(compiles=False, target_cwe="CWE-78", target_cwe_present=False),
    ]
    out = compute_all(recs)

    high = out["secure_at_1__high_severity_compiles"]
    med = out["secure_at_1__medium_severity_compiles"]

    # High denom: 2 compileable CWE-78 records. Of those, 1 is secure.
    assert high["n_denominator"] == 2
    assert high["n_numerator"] == 1
    assert high["value"] == pytest.approx(0.5)

    # Medium denom: 1 compileable CWE-862 record. It is secure (1 of 1).
    assert med["n_denominator"] == 1
    assert med["n_numerator"] == 1
    assert med["value"] == pytest.approx(1.0)


def test_severity_binned_metrics_handle_unknown_cwe():
    """A CWE not in the NVDLib medians file falls into neither bin."""
    recs = [
        _rec(compiles=True, target_cwe="CWE-9999", target_cwe_present=False),
    ]
    out = compute_all(recs)
    assert out["secure_at_1__high_severity_compiles"]["n_denominator"] == 0
    assert out["secure_at_1__medium_severity_compiles"]["n_denominator"] == 0


def test_wilson_interval_matches_paper_base_policy_func_sec():
    """Section VI-A: base-policy Func-Sec@1 21.8% (34/156) -> [16.0, 28.9]."""
    from secure_code_rl_ictai.eval.metrics import wilson_interval

    lo, hi = wilson_interval(34, 156)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (16.0, 28.9)
    assert wilson_interval(0, 0) == (0.0, 0.0)
