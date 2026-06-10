"""Tests for the RAG module: schema validation, R_RAG math with copy-guard,
RRF fusion, CWE filtering, and parent-CWE fallback."""

from __future__ import annotations

import math

import pytest

from secure_code_rl_ictai.rag import (
    ExemplarPair,
    HybridRetriever,
    Language,
    RetrievalQuery,
    StubBackend,
    compute_r_rag,
    r_rag_missing,
)


# ----------------------------------------------------------------------
# ExemplarPair validation (schema)
# ----------------------------------------------------------------------


def test_exemplar_pair_rejects_malformed_cwe():
    with pytest.raises(ValueError):
        ExemplarPair(
            cwe="89",  # missing CWE- prefix
            task_signature="get_user",
            e_pos="def f(): pass",
            e_neg="def f(): pass",
            language=Language.PYTHON,
            source="cvefixes",
        )


def test_exemplar_pair_rejects_empty_exemplar():
    with pytest.raises(ValueError):
        ExemplarPair(
            cwe="CWE-89",
            task_signature="get_user",
            e_pos="   \n",
            e_neg="def f(): pass",
            language=Language.PYTHON,
            source="cvefixes",
        )


# ----------------------------------------------------------------------
# R_RAG math (spec §6)
# ----------------------------------------------------------------------


def _unit_vec(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values] if norm > 0 else values


def test_r_rag_is_scaled_cosine_to_e_pos():
    """Eq. 2: R_RAG = cos(y, e+); e- does not enter the continuous reward."""
    completion = _unit_vec([1.0, 0.0, 0.0])
    e_pos = _unit_vec([0.7, 0.7, 0.0])
    e_neg = _unit_vec([0.7, -0.7, 0.0])
    value, diag = compute_r_rag(completion, e_pos, e_neg, lambda_rag=0.5)
    assert value == pytest.approx(0.5 * diag.sim_pos)
    assert diag.copy_guard_hit is False


def test_binary_control_uses_contrastive_indicator():
    completion = _unit_vec([1.0, 0.2, 0.0])
    e_pos = _unit_vec([0.7, 0.7, 0.0])
    e_neg = _unit_vec([0.7, -0.7, 0.0])
    value, _ = compute_r_rag(completion, e_pos, e_neg, lambda_rag=0.1, binary=True)
    assert value == pytest.approx(0.1)
    value, _ = compute_r_rag(completion, e_neg, e_pos, lambda_rag=0.1, binary=True)
    assert value == 0.0


def test_r_rag_positive_when_closer_to_e_pos():
    completion = _unit_vec([1.0, 0.5, 0.0])
    e_pos = _unit_vec([1.0, 0.5, 0.0])  # identical -> sim_pos = 1
    e_neg = _unit_vec([1.0, -0.5, 0.0])
    value, diag = compute_r_rag(completion, e_pos, e_neg, lambda_rag=0.5)
    # sim_pos = 1.0 > 0.95 -> copy-guard fires -> value = 0
    assert diag.copy_guard_hit is True
    assert value == 0.0
    # raw was still computed and recorded
    assert diag.raw > 0


def test_r_rag_negative_when_anticorrelated_with_e_pos():
    completion = _unit_vec([-1.0, 0.2, 0.0])
    e_pos = _unit_vec([1.0, 0.0, 0.0])
    e_neg = _unit_vec([0.0, 1.0, 0.0])
    value, diag = compute_r_rag(completion, e_pos, e_neg, lambda_rag=0.5)
    assert value < 0
    assert diag.copy_guard_hit is False


def test_copy_guard_threshold_is_inclusive_above():
    """sim_pos > threshold fires, sim_pos == threshold does NOT fire."""
    # Construct embeddings so sim_pos is exactly the threshold.
    completion = [1.0, 0.0]
    e_pos = [0.95, math.sqrt(1 - 0.95**2)]  # cos(y, e_pos) = 0.95
    e_neg = [0.0, 1.0]
    value, diag = compute_r_rag(
        completion, e_pos, e_neg, lambda_rag=0.5, copy_guard_threshold=0.95
    )
    assert diag.sim_pos == pytest.approx(0.95, rel=1e-6)
    assert diag.copy_guard_hit is False
    assert value != 0.0


def test_copy_guard_fires_just_above_threshold():
    completion = [1.0, 0.0]
    e_pos = [0.96, math.sqrt(1 - 0.96**2)]
    e_neg = [0.0, 1.0]
    value, diag = compute_r_rag(
        completion, e_pos, e_neg, lambda_rag=0.5, copy_guard_threshold=0.95
    )
    assert diag.copy_guard_hit is True
    assert value == 0.0


def test_r_rag_dim_mismatch_raises():
    with pytest.raises(ValueError):
        compute_r_rag([1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0])


def test_r_rag_missing_returns_zero_with_sentinel_diag():
    value, diag = r_rag_missing()
    assert value == 0.0
    assert diag.sim_pos == 0.0
    assert diag.sim_neg == 0.0
    assert diag.copy_guard_hit is False


# ----------------------------------------------------------------------
# Hybrid retriever: RRF fusion (spec §5 step 3)
# ----------------------------------------------------------------------


def _pair(idx: int, cwe: str, language=Language.PYTHON) -> ExemplarPair:
    return ExemplarPair(
        cwe=cwe,
        task_signature=f"task_{idx}",
        e_pos=f"# secure {idx}\npass\n",
        e_neg=f"# vuln {idx}\npass\n",
        language=language,
        source="test",
    )


def test_rrf_fusion_combines_ranks():
    """Pair appearing in both lists should outrank a pair in only one."""
    pairs = [_pair(i, "CWE-89") for i in range(5)]
    bm25 = StubBackend([(0, 5.0), (1, 4.0), (2, 3.0), (3, 2.0)])
    dense = StubBackend([(3, 0.9), (0, 0.8), (4, 0.7)])
    retriever = HybridRetriever(pairs, bm25, dense, rrf_k=60)
    hit = retriever.retrieve(RetrievalQuery(cwe="CWE-89"))
    # Pair 0 appears in BM25 rank 1 and dense rank 2.
    # Pair 3 appears in BM25 rank 4 and dense rank 1.
    # RRF: pair 0 = 1/61 + 1/62; pair 3 = 1/64 + 1/61. Pair 0 wins.
    assert hit is not None
    assert hit.pair.task_signature == "task_0"


def test_cwe_filter_excludes_wrong_cwe_pairs():
    pairs = [_pair(0, "CWE-89"), _pair(1, "CWE-79"), _pair(2, "CWE-89")]
    # Both backends rank the WRONG-CWE pair first.
    bm25 = StubBackend([(1, 10.0), (0, 5.0), (2, 4.0)])
    dense = StubBackend([(1, 0.95), (2, 0.7), (0, 0.5)])
    retriever = HybridRetriever(pairs, bm25, dense)
    hit = retriever.retrieve(RetrievalQuery(cwe="CWE-89"))
    assert hit is not None
    assert hit.pair.cwe == "CWE-89"
    # Pair 1 (CWE-79) was filtered out.
    assert hit.pair.task_signature in ("task_0", "task_2")


def test_adversarial_mode_picks_lowest_ranked_within_cwe():
    pairs = [_pair(0, "CWE-89"), _pair(1, "CWE-79"), _pair(2, "CWE-89")]
    bm25 = StubBackend([(0, 5.0), (1, 4.0), (2, 3.0)])
    dense = StubBackend([(0, 0.9), (1, 0.8), (2, 0.7)])
    retriever = HybridRetriever(pairs, bm25, dense, mode="adversarial")
    hit = retriever.retrieve(RetrievalQuery(cwe="CWE-89"))
    assert hit is not None
    assert hit.pair.task_signature == "task_2"


def test_random_mode_samples_any_cwe():
    pairs = [_pair(0, "CWE-89")] + [_pair(i, "CWE-79") for i in range(1, 20)]
    retriever = HybridRetriever(
        pairs, StubBackend([]), StubBackend([]), mode="random", cwe_parent_fallback=False,
    )
    cwes = {retriever.retrieve(RetrievalQuery(cwe="CWE-89")).pair.cwe for _ in range(50)}
    assert "CWE-79" in cwes


def test_no_match_returns_none_when_no_fallback():
    pairs = [_pair(0, "CWE-79")]
    bm25 = StubBackend([(0, 5.0)])
    dense = StubBackend([(0, 0.5)])
    # Query for a CWE absent from the index and not in the parent map.
    retriever = HybridRetriever(
        pairs, bm25, dense, cwe_parent_fallback=True
    )
    hit = retriever.retrieve(RetrievalQuery(cwe="CWE-9999"))
    assert hit is None


def test_parent_cwe_fallback_when_exact_missing():
    """CWE-87 has parent CWE-74 in the hand-coded map (actually CWE-89 ->
    CWE-74). Construct a case where exact match fails but parent matches."""
    pairs = [_pair(0, "CWE-74"), _pair(1, "CWE-22")]
    bm25 = StubBackend([(0, 5.0), (1, 4.0)])
    dense = StubBackend([(0, 0.8), (1, 0.7)])
    retriever = HybridRetriever(pairs, bm25, dense, cwe_parent_fallback=True)
    # CWE-89's parent is CWE-74 per the hand-coded map.
    hit = retriever.retrieve(RetrievalQuery(cwe="CWE-89"))
    assert hit is not None
    assert hit.pair.cwe == "CWE-74"


def test_parent_fallback_disabled_returns_none():
    pairs = [_pair(0, "CWE-74")]
    bm25 = StubBackend([(0, 5.0)])
    dense = StubBackend([(0, 0.8)])
    retriever = HybridRetriever(pairs, bm25, dense, cwe_parent_fallback=False)
    hit = retriever.retrieve(RetrievalQuery(cwe="CWE-89"))
    assert hit is None
