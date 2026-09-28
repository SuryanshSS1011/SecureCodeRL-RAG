"""End-to-end reward pipeline tests with all mocks.

Verifies that the integration layer composes R_total correctly by feeding
it a planted SARIF, a planted oracle outcome, and hand-built embeddings.
The math is identical to the unit-level calculator + R_RAG tests; this
suite catches wiring bugs (missing field passthrough, language enum
mismatch, refusal detection, etc.).
"""

from __future__ import annotations


import pytest

from cargo.rag import (
    ExemplarPair,
    HybridRetriever,
    Language as RagLanguage,
    StubBackend,
)
from cargo.reward import (
    Language,
    MockOracle,
    PromptContext,
    ReliabilitySignals,
    RewardConfig,
    RewardCalculator,
    RewardPipeline,
    TestSpec,
)
from cargo.sast.models import ToolName
from cargo.sast.runner import MockAdapter, SastRunner
from cargo.sast.severity import SeveritySource


# Completions long enough to clear the pipeline's stub guard (< 20
# non-space characters short-circuits the oracle, SAST and RAG).
_CLEAN_CODE = "import sys\n\nprint('hello world', file=sys.stdout)\n"
_QUERY_CODE = "import sqlite3\n\nquery = 'SELECT * FROM t WHERE id = ' + uid\n"


def _sarif_codeql_one_finding(cwe_num: int, severity: float) -> dict:
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


class _FakeEmbedder:
    """Deterministic 'embedder' that returns text-keyed unit vectors.

    Each input text maps to a unit vector spelled out by characters,
    padded with zeros. Two different texts ALWAYS produce different
    embeddings. Used to control sim_pos / sim_neg in tests.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed(self, text: str) -> list[float]:
        if text in self._table:
            return self._table[text]
        # Fallback: hash-derived vector (not norm 1, but consistent).
        return [float(sum(map(ord, text)) % 10) / 10.0, 0.5, 0.5]


def _python_pipeline_no_rag(
    sarif: dict | None,
    oracle_signals: ReliabilitySignals,
    alpha: float = 1.0,
) -> RewardPipeline:
    adapters = {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, sarif if sarif else {"runs": []}),
        ToolName.SEMGREP: MockAdapter(ToolName.SEMGREP, {"runs": []}),
        ToolName.BANDIT: MockAdapter(ToolName.BANDIT, {"runs": []}),
        ToolName.CPPCHECK: MockAdapter(ToolName.CPPCHECK, {"runs": []}),
    }
    runner = SastRunner(adapters)
    sev_src = SeveritySource()
    calc = RewardCalculator(RewardConfig(alpha=alpha), sev_src)
    return RewardPipeline(
        oracle=MockOracle(oracle_signals),
        sast_runner=runner,
        severity_source=sev_src,
        calculator=calc,
    )


# ----------------------------------------------------------------------
# Reward composition without RAG
# ----------------------------------------------------------------------


def test_pipeline_clean_code_passing_tests():
    """Clean compiling code with passing tests scores R_sec = R_rel = r = 1."""
    pipeline = _python_pipeline_no_rag(
        sarif=None,
        oracle_signals=ReliabilitySignals(
            compiles=True, runs=True, produces_output=True,
            tests_passed=1, tests_total=1,
        ),
    )
    ctx = PromptContext(
        target_cwe="CWE-89",
        language="python",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
    )
    out = pipeline.evaluate("prompt", _CLEAN_CODE, ctx)
    assert out.breakdown.r_reliability == pytest.approx(1.0)
    assert out.breakdown.r_security == 1.0
    assert out.breakdown.gate == 1
    assert out.breakdown.r_total == pytest.approx(1.0)


def test_pipeline_dirty_code_passing_tests_penalizes():
    pipeline = _python_pipeline_no_rag(
        sarif=_sarif_codeql_one_finding(89, 8.8),
        oracle_signals=ReliabilitySignals(
            compiles=True, runs=True, produces_output=True,
            tests_passed=1, tests_total=1,
        ),
        alpha=1.0,
    )
    ctx = PromptContext(
        target_cwe="CWE-89",
        language="python",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
    )
    out = pipeline.evaluate("prompt", _QUERY_CODE, ctx)
    # one finding: severity 8.8, confidence 0.8 (SARIF level=error)
    # penalty = 0.88 * 0.8 = 0.704; R_sec = 1 - 0.704 = 0.296
    assert out.breakdown.r_security == pytest.approx(0.296, rel=1e-3)
    assert out.breakdown.gate == 1
    # alpha = 1: r = R_sec
    assert out.breakdown.r_total == pytest.approx(0.296, rel=1e-3)


def test_pipeline_dirty_code_failing_tests_still_includes_security():
    """The gate is a diagnostic flag, not a multiplier: when tests fail,
    the security score still enters r (Eq. 4)."""
    pipeline = _python_pipeline_no_rag(
        sarif=_sarif_codeql_one_finding(89, 8.8),
        oracle_signals=ReliabilitySignals(
            compiles=True, runs=True, produces_output=True,
            tests_passed=0, tests_total=5,  # gate=0
        ),
        alpha=1.0,
    )
    ctx = PromptContext(
        target_cwe="CWE-89",
        language="python",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
    )
    out = pipeline.evaluate("prompt", _CLEAN_CODE, ctx)
    assert out.breakdown.gate == 0
    # alpha = 1: r = R_sec = 1 - 0.704
    assert out.breakdown.r_total == pytest.approx(0.296, rel=1e-3)


# ----------------------------------------------------------------------
# Refusal / empty short-circuit
# ----------------------------------------------------------------------


def test_pipeline_refusal_is_detected_and_security_suppressed():
    pipeline = _python_pipeline_no_rag(
        sarif=None,
        oracle_signals=ReliabilitySignals(),  # empty -> all false
    )
    ctx = PromptContext(
        target_cwe="CWE-89",
        language="python",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
    )
    out = pipeline.evaluate("prompt", "I cannot help with that.\n", ctx)
    assert out.diagnostics.refusal_or_empty is True
    assert out.breakdown.gate == 0
    assert out.breakdown.r_total == 0.0


def test_pipeline_empty_completion_is_detected():
    pipeline = _python_pipeline_no_rag(
        sarif=None,
        oracle_signals=ReliabilitySignals(),
    )
    ctx = PromptContext(
        target_cwe="CWE-89",
        language="python",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
    )
    out = pipeline.evaluate("prompt", "   \n  \n", ctx)
    assert out.diagnostics.refusal_or_empty is True


def test_pipeline_code_with_apologetic_comment_is_not_refusal():
    """A comment starting with 'I cannot' should NOT be flagged as a refusal
    if the surrounding text contains real code markers."""
    pipeline = _python_pipeline_no_rag(
        sarif=None,
        oracle_signals=ReliabilitySignals(
            compiles=True, tests_passed=1, tests_total=1,
        ),
    )
    ctx = PromptContext(
        target_cwe="CWE-89",
        language="python",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
    )
    out = pipeline.evaluate(
        "prompt",
        "# I cannot do this without a library, so:\nimport sqlite3\n",
        ctx,
    )
    # The first non-blank line starts with '#', not the refusal pattern
    # directly. Refusal detection should not fire here. We also have
    # 'import ' marker.
    assert out.diagnostics.refusal_or_empty is False


# ----------------------------------------------------------------------
# RAG composition (end-to-end with stubbed retriever and fake embedder)
# ----------------------------------------------------------------------


def test_pipeline_rag_added_to_total_when_retriever_hits():
    pair = ExemplarPair(
        cwe="CWE-89",
        task_signature="get_user",
        e_pos="SECURE_CODE",
        e_neg="VULN_CODE",
        language=RagLanguage.PYTHON,
        source="test",
    )
    # Hand-craft embeddings so sim(y, e_pos) is a known value.
    embedder = _FakeEmbedder({
        # 'y' (completion)
        _QUERY_CODE: [0.5, 0.5, 0.0],
        "SECURE_CODE": [1.0, 0.0, 0.0],   # dot = 0.5
        "VULN_CODE": [0.2, 0.0, 0.0],      # dot = 0.1
    })
    bm25 = StubBackend([(0, 5.0)])
    dense = StubBackend([(0, 0.9)])
    retriever = HybridRetriever([pair], bm25, dense, cwe_parent_fallback=False)

    adapters = {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, {"runs": []}),
        ToolName.SEMGREP: MockAdapter(ToolName.SEMGREP, {"runs": []}),
        ToolName.BANDIT: MockAdapter(ToolName.BANDIT, {"runs": []}),
    }
    sev_src = SeveritySource()
    calc = RewardCalculator(RewardConfig(alpha=0.0), sev_src)
    pipeline = RewardPipeline(
        oracle=MockOracle(ReliabilitySignals(
            compiles=True, tests_passed=1, tests_total=1
        )),
        sast_runner=SastRunner(adapters),
        severity_source=sev_src,
        retriever=retriever,
        embedder=embedder,
        calculator=calc,
        lambda_rag=0.5,
        copy_guard_threshold=0.95,
    )
    ctx = PromptContext(
        target_cwe="CWE-89",
        language="python",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
        query_embedding=[0.5, 0.5, 0.0],
        task_signature="get_user",
    )
    out = pipeline.evaluate("prompt", _QUERY_CODE, ctx)
    # r_rag = lambda * cos(y, e+) = 0.5 * 0.5
    assert out.breakdown.r_rag == pytest.approx(0.25, rel=1e-3)
    assert out.diagnostics.rag_used is True
    assert out.diagnostics.rag_copy_guard_hit is False


def test_pipeline_rag_miss_records_diagnostic():
    pair = ExemplarPair(
        cwe="CWE-79",  # wrong CWE for query
        task_signature="x",
        e_pos="A",
        e_neg="B",
        language=RagLanguage.PYTHON,
        source="test",
    )
    embedder = _FakeEmbedder({})
    bm25 = StubBackend([(0, 5.0)])
    dense = StubBackend([(0, 0.9)])
    retriever = HybridRetriever([pair], bm25, dense, cwe_parent_fallback=False)
    adapters = {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, {"runs": []}),
        ToolName.SEMGREP: MockAdapter(ToolName.SEMGREP, {"runs": []}),
        ToolName.BANDIT: MockAdapter(ToolName.BANDIT, {"runs": []}),
    }
    sev_src = SeveritySource()
    calc = RewardCalculator(RewardConfig(alpha=0.0), sev_src)
    pipeline = RewardPipeline(
        oracle=MockOracle(ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1)),
        sast_runner=SastRunner(adapters),
        severity_source=sev_src,
        retriever=retriever,
        embedder=embedder,
        calculator=calc,
    )
    ctx = PromptContext(
        target_cwe="CWE-9999",  # unknown
        language="python",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
        query_embedding=[1.0, 0.0, 0.0],
    )
    out = pipeline.evaluate("prompt", _CLEAN_CODE, ctx)
    assert out.diagnostics.rag_missing is True
    assert out.breakdown.r_rag == 0.0


def test_pipeline_pre_supplied_pair_embeddings_skip_retrieval():
    """When the trainer caches retrieval across rollouts in a group, the
    PromptContext can carry pre-computed e_pos/e_neg embeddings. The
    pipeline should use them directly."""
    embedder = _FakeEmbedder({
        _CLEAN_CODE: [1.0, 0.0],
    })
    # Retriever and FakeEmbedder are needed because the pipeline requires
    # both, but they should be unused for this test.
    bm25 = StubBackend([])
    dense = StubBackend([])
    retriever = HybridRetriever([], bm25, dense)
    adapters = {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, {"runs": []}),
        ToolName.SEMGREP: MockAdapter(ToolName.SEMGREP, {"runs": []}),
        ToolName.BANDIT: MockAdapter(ToolName.BANDIT, {"runs": []}),
    }
    sev_src = SeveritySource()
    calc = RewardCalculator(RewardConfig(alpha=0.0), sev_src)
    pipeline = RewardPipeline(
        oracle=MockOracle(ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1)),
        sast_runner=SastRunner(adapters),
        severity_source=sev_src,
        retriever=retriever,
        embedder=embedder,
        calculator=calc,
        lambda_rag=1.0,
    )
    ctx = PromptContext(
        target_cwe="CWE-89",
        language="python",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
        e_pos_embedding=[1.0, 0.0],  # sim(y, e_pos) = 1.0
        e_neg_embedding=[0.0, 1.0],  # sim(y, e_neg) = 0.0
    )
    out = pipeline.evaluate("prompt", _CLEAN_CODE, ctx)
    # sim_pos = 1.0 > 0.95 -> copy guard fires
    assert out.diagnostics.rag_copy_guard_hit is True
    assert out.breakdown.r_rag == 0.0


# ----------------------------------------------------------------------
# Mixed retriever / embedder construction
# ----------------------------------------------------------------------


def test_pipeline_rejects_only_retriever_without_embedder():
    bm25 = StubBackend([])
    dense = StubBackend([])
    retriever = HybridRetriever([], bm25, dense)
    adapters = {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, {"runs": []}),
    }
    sev_src = SeveritySource()
    with pytest.raises(ValueError):
        RewardPipeline(
            oracle=MockOracle(ReliabilitySignals()),
            sast_runner=SastRunner(adapters),
            severity_source=sev_src,
            retriever=retriever,
            embedder=None,  # mismatch
        )
