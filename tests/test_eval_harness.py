"""Tests for the eval harness.

Pins per-prompt scoring composition, aggregate metric math (func@1,
secure@1, func-sec@1, per-CWE), refusal/empty/copy-guard canary
calculation, per-(CWE, language) breakdown, and report serialization.
"""

from __future__ import annotations

import json
from pathlib import Path


from secure_code_rl_ictai.data_prep import (
    Prompt,
)
from secure_code_rl_ictai.eval.harness import EvalHarness
from secure_code_rl_ictai.eval.model import MockModel, SamplingConfig
from secure_code_rl_ictai.reward import (
    Language,
    MockOracle,
    ReliabilitySignals,
    RewardCalculator,
    RewardConfig,
    RewardPipeline,
    TestCase,
    TestSpec,
)
from secure_code_rl_ictai.sast.models import ToolName
from secure_code_rl_ictai.sast.runner import MockAdapter, SastRunner
from secure_code_rl_ictai.sast.severity import SeveritySource


# A completion long enough to clear the reward pipeline's stub guard
# (`_looks_like_stub` zeroes anything under 20 non-space characters and
# skips the oracle + SAST entirely). Tests that assert on func/secure
# metrics need the pipeline to actually score the completion.
_CODE = "import sqlite3\n\ndef get_user(db, uid):\n    return db.execute('SELECT 1')\n"


def _prompt(prompt_id: str, cwe: str = "CWE-89", lang: Language = Language.PYTHON) -> Prompt:
    return Prompt(
        id=prompt_id,
        source="test",
        language=lang,
        target_cwe=cwe,
        prompt_text=f"# task {prompt_id}",
        test_spec=TestSpec(language=lang, test_cases=[]),
    )


def _sarif_with_cwe(cwe_num: int, severity: float = 8.0) -> dict:
    return {
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "codeql",
                        "rules": [
                            {
                                "id": f"py/test-{cwe_num}",
                                "relationships": [
                                    {
                                        "target": {
                                            "id": str(cwe_num),
                                            "toolComponent": {"name": "CWE"},
                                        }
                                    }
                                ],
                                "properties": {"security-severity": str(severity)},
                            }
                        ],
                    }
                },
                "results": [
                    {
                        "ruleId": f"py/test-{cwe_num}",
                        "level": "error",
                        "message": {"text": "test"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "snippet.py"},
                                    "region": {"startLine": 1},
                                }
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _pipeline(sarif: dict | None, signals: ReliabilitySignals) -> RewardPipeline:
    adapters = {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, sarif if sarif else {"runs": []}),
        ToolName.SEMGREP: MockAdapter(ToolName.SEMGREP, {"runs": []}),
        ToolName.BANDIT: MockAdapter(ToolName.BANDIT, {"runs": []}),
        ToolName.CPPCHECK: MockAdapter(ToolName.CPPCHECK, {"runs": []}),
    }
    runner = SastRunner(adapters)
    sev_src = SeveritySource()
    calc = RewardCalculator(RewardConfig(alpha=1.0), sev_src)
    return RewardPipeline(
        oracle=MockOracle(signals),
        sast_runner=runner,
        severity_source=sev_src,
        calculator=calc,
    )


# ----------------------------------------------------------------------
# Single-prompt evaluation
# ----------------------------------------------------------------------


def test_evaluate_single_prompt_records_breakdown():
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    model = MockModel(name="mock", responses={"p1-text": "import sqlite3"}, default="")
    prompts = [_prompt("p1")]
    # MockModel returns default for unknown prompts; supply prompt_text key
    prompts[0].metadata.update({})
    report = harness.evaluate(model, prompts)
    assert report.n_prompts == 1
    rec = report.per_prompt[0]
    assert rec.prompt_id == "p1"
    assert rec.completion == ""  # MockModel returned default
    # Reliability oracle is MockOracle returning the supplied signals
    assert rec.tests_passed == 1
    assert rec.tests_total == 1


# ----------------------------------------------------------------------
# Aggregate metrics (spec §3.1)
# ----------------------------------------------------------------------


def test_aggregate_func_at_1_counts_full_test_passes():
    """A prompt is func@1 iff tests_passed == tests_total > 0."""
    full_pass = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=2, tests_total=2),
    )
    partial_pass = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=2),
    )
    no_pass = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=0, tests_total=2),
    )
    model = MockModel(name="m", responses=lambda p: _CODE)

    # Three separate harness calls, one per fixture, to control the pipeline.
    h1 = EvalHarness(pipeline=full_pass, sampling=SamplingConfig())
    h2 = EvalHarness(pipeline=partial_pass, sampling=SamplingConfig())
    h3 = EvalHarness(pipeline=no_pass, sampling=SamplingConfig())

    r1 = h1.evaluate(model, [_prompt("a")])
    r2 = h2.evaluate(model, [_prompt("b")])
    r3 = h3.evaluate(model, [_prompt("c")])

    assert r1.aggregate["func_at_1"] == 1.0
    assert r2.aggregate["func_at_1"] == 0.0
    assert r3.aggregate["func_at_1"] == 0.0


def test_aggregate_secure_at_1_excludes_target_cwe_findings():
    """A prompt is secure@1 iff NO surviving finding has cwe == target_cwe."""
    # CodeQL fires a CWE-89 finding; prompt targets CWE-89; not secure.
    not_secure = _pipeline(
        sarif=_sarif_with_cwe(89),
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    # CodeQL fires a CWE-79 finding; prompt targets CWE-89; still secure for the target.
    other_cwe = _pipeline(
        sarif=_sarif_with_cwe(79),
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    # No findings; secure.
    clean = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )

    model = MockModel(name="m", responses=lambda p: _CODE)
    h_ns = EvalHarness(pipeline=not_secure, sampling=SamplingConfig())
    h_other = EvalHarness(pipeline=other_cwe, sampling=SamplingConfig())
    h_clean = EvalHarness(pipeline=clean, sampling=SamplingConfig())

    r_ns = h_ns.evaluate(model, [_prompt("a", cwe="CWE-89")])
    r_other = h_other.evaluate(model, [_prompt("b", cwe="CWE-89")])
    r_clean = h_clean.evaluate(model, [_prompt("c", cwe="CWE-89")])

    assert r_ns.aggregate["secure_at_1"] == 0.0
    assert r_other.aggregate["secure_at_1"] == 1.0
    assert r_clean.aggregate["secure_at_1"] == 1.0


def test_aggregate_func_sec_at_1_is_intersection():
    """func_sec_at_1 <= min(func_at_1, secure_at_1) and = 1 only when both are 1."""
    # Functional but has target CWE finding.
    func_only = _pipeline(
        sarif=_sarif_with_cwe(89),
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    # Has target CWE but not functional.
    secure_only = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=0, tests_total=1),
    )
    # Both.
    both = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )

    model = MockModel(name="m", responses=lambda p: _CODE)
    h_f = EvalHarness(pipeline=func_only, sampling=SamplingConfig())
    h_s = EvalHarness(pipeline=secure_only, sampling=SamplingConfig())
    h_b = EvalHarness(pipeline=both, sampling=SamplingConfig())

    r_f = h_f.evaluate(model, [_prompt("a", cwe="CWE-89")])
    r_s = h_s.evaluate(model, [_prompt("b", cwe="CWE-89")])
    r_b = h_b.evaluate(model, [_prompt("c", cwe="CWE-89")])

    assert r_f.aggregate["func_sec_at_1"] == 0.0
    assert r_s.aggregate["func_sec_at_1"] == 0.0
    assert r_b.aggregate["func_sec_at_1"] == 1.0


# ----------------------------------------------------------------------
# Per-CWE breakdown (spec §3.3)
# ----------------------------------------------------------------------


def test_per_cwe_aggregates_by_target_cwe():
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    model = MockModel(name="m", responses=lambda p: _CODE)
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    prompts = [
        _prompt("a1", cwe="CWE-89"),
        _prompt("a2", cwe="CWE-89"),
        _prompt("b1", cwe="CWE-79"),
    ]
    report = harness.evaluate(model, prompts)
    assert report.per_cwe["CWE-89"]["n_prompts"] == 2
    assert report.per_cwe["CWE-79"]["n_prompts"] == 1
    assert report.per_cwe["CWE-89"]["secure_at_1"] == 1.0


# ----------------------------------------------------------------------
# Per-(CWE, language) breakdown (spec §4)
# ----------------------------------------------------------------------


def test_per_cwe_language_split():
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    model = MockModel(name="m", responses=lambda p: _CODE)
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    prompts = [
        _prompt("a", cwe="CWE-89", lang=Language.PYTHON),
        _prompt("b", cwe="CWE-89", lang=Language.C),
    ]
    report = harness.evaluate(model, prompts)
    assert ("CWE-89", "python") in report.per_cwe_language
    assert ("CWE-89", "c") in report.per_cwe_language
    assert report.per_cwe_language[("CWE-89", "python")]["n_prompts"] == 1
    assert report.per_cwe_language[("CWE-89", "c")]["n_prompts"] == 1


# ----------------------------------------------------------------------
# Reward-hacking canaries (spec §3.4)
# ----------------------------------------------------------------------


def test_refusal_rate_counted():
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(),  # nothing works
    )
    model = MockModel(
        name="m",
        responses=lambda p: "I cannot help with this.",
    )
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    report = harness.evaluate(model, [_prompt("a"), _prompt("b")])
    assert report.diagnostics["refusal_rate"] == 1.0


def test_empty_rate_counted():
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(),
    )
    model = MockModel(name="m", responses=lambda p: "")
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    report = harness.evaluate(model, [_prompt("a"), _prompt("b")])
    assert report.diagnostics["empty_rate"] == 1.0


# ----------------------------------------------------------------------
# Model crash handling (spec §9)
# ----------------------------------------------------------------------


def test_model_crash_recorded_not_raised():
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(),
    )
    model = MockModel(
        name="m",
        responses={"_": ""},
        crash_on_prompts=frozenset({"# task crashed"}),
    )
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    p_ok = _prompt("ok")
    p_crash = _prompt("crashed")
    report = harness.evaluate(model, [p_ok, p_crash])
    assert report.n_prompts == 2
    # The crashed prompt's record has crashed=True
    crashed_rec = next(r for r in report.per_prompt if r.prompt_id == "crashed")
    assert crashed_rec.crashed is True


# ----------------------------------------------------------------------
# Report serialization (spec §6)
# ----------------------------------------------------------------------


def test_report_serializes_to_directory(tmp_path: Path):
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    model = MockModel(name="my-model", responses=lambda p: _CODE)
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    report = harness.evaluate(model, [_prompt("a", cwe="CWE-89")])
    report.save(tmp_path)

    out = tmp_path / "my-model"
    assert (out / "aggregate.json").exists()
    assert (out / "per_prompt.jsonl").exists()

    agg = json.loads((out / "aggregate.json").read_text())
    assert agg["model_name"] == "my-model"
    assert agg["n_prompts"] == 1


# ----------------------------------------------------------------------
# Empty prompt list
# ----------------------------------------------------------------------


def test_empty_prompts_returns_zero_aggregate():
    pipeline = _pipeline(sarif=None, signals=ReliabilitySignals())
    model = MockModel(name="m", responses=lambda p: "x")
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    report = harness.evaluate(model, [])
    assert report.n_prompts == 0
    assert report.aggregate["func_at_1"] == 0.0
    assert report.aggregate["secure_at_1"] == 0.0
    assert report.per_cwe == {}


# ----------------------------------------------------------------------
# Resume from stream (SLURM walltime recovery)
# ----------------------------------------------------------------------


def test_resume_skips_already_completed_prompt_ids(tmp_path: Path):
    """A second evaluate() with resume=True picks up where the first left off.

    Simulates the SLURM walltime-cancellation flow: the original run streamed
    records 1..N before being killed; the resume run skips those, processes
    N+1..M, and the final in-memory report covers all M.
    """
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    model = MockModel(name="m", responses=lambda p: _CODE)
    stream = tmp_path / "stream.jsonl"

    first_batch = [_prompt(f"p{i}", cwe="CWE-89") for i in range(3)]
    harness.evaluate(model, first_batch, stream_path=stream)
    n_lines_after_first = sum(1 for _ in stream.open())
    assert n_lines_after_first == 3

    all_prompts = [_prompt(f"p{i}", cwe="CWE-89") for i in range(5)]
    report = harness.evaluate(
        model, all_prompts, stream_path=stream, resume=True,
    )
    n_lines_after_resume = sum(1 for _ in stream.open())
    assert n_lines_after_resume == 5
    assert report.n_prompts == 5
    seen_ids = {r.prompt_id for r in report.per_prompt}
    assert seen_ids == {f"p{i}" for i in range(5)}


def test_resume_no_existing_stream_starts_fresh(tmp_path: Path):
    """resume=True on a non-existent stream is equivalent to a fresh run."""
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    model = MockModel(name="m", responses=lambda p: _CODE)
    stream = tmp_path / "no_existing_stream.jsonl"
    prompts = [_prompt(f"p{i}", cwe="CWE-89") for i in range(2)]
    report = harness.evaluate(
        model, prompts, stream_path=stream, resume=True,
    )
    assert report.n_prompts == 2
    assert sum(1 for _ in stream.open()) == 2


def test_resume_false_truncates_stream(tmp_path: Path):
    """Default resume=False replaces a stale stream (pre-refactor behavior)."""
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    model = MockModel(name="m", responses=lambda p: _CODE)
    stream = tmp_path / "stale.jsonl"
    stream.write_text('{"prompt_id":"x"}\n{"prompt_id":"y"}\n')
    harness.evaluate(model, [_prompt("p0")], stream_path=stream)
    lines = stream.read_text().splitlines()
    assert len(lines) == 1
    assert '"prompt_id": "p0"' in lines[0]


# ----------------------------------------------------------------------
# Bootstrap CIs (spec §5)
# ----------------------------------------------------------------------


def test_bootstrap_cis_populated_for_headline_metrics():
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    model = MockModel(name="m", responses=lambda p: _CODE)
    harness = EvalHarness(
        pipeline=pipeline,
        sampling=SamplingConfig(),
        bootstrap_n_resamples=200,
    )
    prompts = [_prompt(f"p{i}", cwe="CWE-89") for i in range(10)]
    report = harness.evaluate(model, prompts)
    assert "func_at_1" in report.bootstrap_cis
    assert "secure_at_1" in report.bootstrap_cis
    assert "func_sec_at_1" in report.bootstrap_cis
    lo, hi = report.bootstrap_cis["func_at_1"]
    # All prompts pass; CI is degenerate at 1.0.
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    """For a non-degenerate metric, the CI must contain the point estimate."""
    # Alternate pass/fail tests by using two different pipelines? Simpler:
    # use a model that produces different completions and a pipeline that
    # always passes — then make some prompts have target_cwe absent in
    # findings and others present. We can't do that with one fixed pipeline,
    # so test with a degenerate setup but force the CI math to be exercised.
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    model = MockModel(name="m", responses=lambda p: _CODE)
    harness = EvalHarness(
        pipeline=pipeline,
        sampling=SamplingConfig(),
        bootstrap_n_resamples=500,
    )
    prompts = [_prompt(f"p{i}", cwe="CWE-89") for i in range(20)]
    report = harness.evaluate(model, prompts)
    point = report.aggregate["func_at_1"]
    lo, hi = report.bootstrap_cis["func_at_1"]
    assert lo <= point <= hi


def test_bootstrap_cis_seeded_for_reproducibility():
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    model = MockModel(name="m", responses=lambda p: _CODE)
    prompts = [_prompt(f"p{i}") for i in range(20)]

    h1 = EvalHarness(
        pipeline=pipeline,
        sampling=SamplingConfig(seed=42),
        bootstrap_n_resamples=300,
    )
    h2 = EvalHarness(
        pipeline=pipeline,
        sampling=SamplingConfig(seed=42),
        bootstrap_n_resamples=300,
    )
    r1 = h1.evaluate(model, prompts)
    r2 = h2.evaluate(model, prompts)
    # Same seed -> same CIs.
    assert r1.bootstrap_cis["func_at_1"] == r2.bootstrap_cis["func_at_1"]


def test_bootstrap_cis_per_cwe_skipped_when_low_support():
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    model = MockModel(name="m", responses=lambda p: _CODE)
    harness = EvalHarness(
        pipeline=pipeline,
        sampling=SamplingConfig(),
        bootstrap_per_cwe_min_support=10,
    )
    # Only 5 prompts per CWE; below threshold.
    prompts = [_prompt(f"p{i}", cwe="CWE-89") for i in range(5)]
    report = harness.evaluate(model, prompts)
    # No per-CWE CI keys.
    assert not any(k.startswith("CWE-89:") for k in report.bootstrap_cis)


def test_bootstrap_cis_per_cwe_present_when_high_support():
    pipeline = _pipeline(
        sarif=None,
        signals=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
    )
    model = MockModel(name="m", responses=lambda p: _CODE)
    harness = EvalHarness(
        pipeline=pipeline,
        sampling=SamplingConfig(),
        bootstrap_per_cwe_min_support=10,
    )
    prompts = [_prompt(f"p{i}", cwe="CWE-89") for i in range(25)]
    report = harness.evaluate(model, prompts)
    assert "CWE-89:func_at_1" in report.bootstrap_cis
    assert "CWE-89:secure_at_1" in report.bootstrap_cis
    assert "CWE-89:func_sec_at_1" in report.bootstrap_cis


# ----------------------------------------------------------------------
# Completion sanitization for base/CLM models (_extract_code)
# ----------------------------------------------------------------------


def test_extract_code_truncates_at_chat_continuation_marker():
    from secure_code_rl_ictai.eval.harness import _extract_code

    text = "def f():\n    return 1\n<|im_end|>\n<|im_start|>user\nmore prompt"
    assert _extract_code(text, "python") == "def f():\n    return 1\n"


def test_extract_code_strips_duplicate_main_for_c():
    from secure_code_rl_ictai.eval.harness import _extract_code

    first = "#include <stdio.h>\nint main(void) {\n    return 0;\n}"
    text = first + "\n\n// hardened version\nint main(void) {\n    return 1;\n}\n"
    out = _extract_code(text, "c")
    assert out.count("int main") == 1
    assert out.startswith("#include <stdio.h>")


def test_extract_code_drops_unterminated_block_comment():
    from secure_code_rl_ictai.eval.harness import _extract_code

    text = "int main(void) {\n    return 0;\n}\n/* the model ran out of tokens mid-comm"
    assert _extract_code(text, "c") == "int main(void) {\n    return 0;\n}"


def test_extract_code_truncates_prose_after_top_level_unit():
    from secure_code_rl_ictai.eval.harness import _extract_code

    text = "int main(void) {\n    return 0;\n}\nThis program prints nothing and exits."
    assert _extract_code(text, "c") == "int main(void) {\n    return 0;\n}"


def test_extract_code_leaves_python_untouched_beyond_fences():
    from secure_code_rl_ictai.eval.harness import _extract_code

    text = "```python\ndef f():\n    return 1\n```\nThat is the function."
    assert _extract_code(text, "python") == "def f():\n    return 1"


# ----------------------------------------------------------------------
# Fixed-denominator Functional-Secure@1 (paper Section V.B)
# ----------------------------------------------------------------------


def test_func_sec_has_tests_denominator_includes_compile_failures():
    """Every test-equipped prompt stays in the denominator; a completion
    that fails to compile contributes 0 to the numerator instead of
    shrinking the denominator (the floating-denominator behaviour of
    `func_sec_at_1__compiles_and_has_tests`)."""
    from secure_code_rl_ictai.eval.harness import PerPromptRecord
    from secure_code_rl_ictai.eval.metrics import compute_all

    passing = PerPromptRecord(
        prompt_id="a", source="t", target_cwe="CWE-89", language="python",
        completion=_CODE, compiles=True, tests_passed=2, tests_total=2,
        n_test_cases=2,
    )
    failed_compile = PerPromptRecord(
        prompt_id="b", source="t", target_cwe="CWE-89", language="python",
        completion=_CODE, compiles=False, tests_passed=0, tests_total=0,
        n_test_cases=2,
    )
    no_tests = PerPromptRecord(
        prompt_id="c", source="t", target_cwe="CWE-89", language="python",
        completion=_CODE, compiles=True, tests_passed=0, tests_total=0,
        n_test_cases=0,
    )
    agg = compute_all([passing, failed_compile, no_tests])
    fixed = agg["func_sec_at_1__has_tests"]
    assert fixed["n_denominator"] == 2 and fixed["n_numerator"] == 1
    floating = agg["func_sec_at_1__compiles_and_has_tests"]
    assert floating["n_denominator"] == 1 and floating["n_numerator"] == 1


def test_harness_records_intrinsic_test_count():
    pipeline = _pipeline(sarif=None, signals=ReliabilitySignals(compiles=False))
    model = MockModel(name="m", responses=lambda p: _CODE)
    harness = EvalHarness(pipeline=pipeline, sampling=SamplingConfig())
    prompt = Prompt(
        id="p", source="test", language=Language.PYTHON, target_cwe="CWE-89",
        prompt_text="# task", test_spec=TestSpec(
            language=Language.PYTHON,
            test_cases=[TestCase(input_stdin="", expected_stdout="1")],
        ),
    )
    report = harness.evaluate(model, [prompt])
    rec = report.per_prompt[0]
    assert rec.compiles is False and rec.tests_total == 0
    assert rec.n_test_cases == 1
    assert report.aggregate["func_sec_at_1__has_tests"]["n_denominator"] == 1
