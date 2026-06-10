"""Tests for the real BM25 + FAISS backends and the HfEmbedder.

Bm25 / Faiss / HfEmbedder all require external packages (rank_bm25,
faiss, sentence-transformers). The unit tests gate on the `real_rag`
marker; they pass when those packages are importable AND skip
gracefully when they're not.

The unit-test pattern for each backend:
  1. Build a tiny ExemplarPair list (3-5 entries).
  2. Construct the backend.
  3. Issue a known-good RetrievalQuery.
  4. Assert the expected top hit and ordering.
"""

from __future__ import annotations

import sys
import types

import pytest

from secure_code_rl_ictai.rag import (
    Bm25Backend,
    ExemplarPair,
    FaissBackend,
    HfEmbedder,
    Language,
    MockEmbedder,
    RetrievalQuery,
)


def _pair(idx: int, cwe: str = "CWE-89", sig: str = "", e_pos: str = "") -> ExemplarPair:
    return ExemplarPair(
        cwe=cwe,
        task_signature=sig or f"def task_{idx}():",
        e_pos=e_pos or f"# secure code {idx}\npass\n",
        e_neg=f"# vuln {idx}\npass\n",
        language=Language.PYTHON,
        source="test",
    )


# ----------------------------------------------------------------------
# BM25 backend (real_rag - needs rank_bm25)
# ----------------------------------------------------------------------


@pytest.mark.real_rag
def test_bm25_backend_ranks_signature_match_first():
    pairs = [
        _pair(0, sig="def get_user(uid):"),
        _pair(1, sig="def render_template(html):"),
        _pair(2, sig="def hash_password(pw):"),
    ]
    backend = Bm25Backend(pairs)
    results = backend.search(
        RetrievalQuery(cwe="CWE-89", task_signature="get_user", top_k_per_backend=3)
    )
    assert len(results) > 0
    # pair 0's signature contains "get_user"; expect it ranked first.
    top_idx, _ = results[0]
    assert top_idx == 0


@pytest.mark.real_rag
def test_bm25_backend_returns_empty_for_empty_query():
    pairs = [_pair(0)]
    backend = Bm25Backend(pairs)
    results = backend.search(
        RetrievalQuery(cwe="CWE-89", task_signature="", top_k_per_backend=3)
    )
    assert results == []


@pytest.mark.real_rag
def test_bm25_backend_returns_empty_when_no_signature():
    pairs = [_pair(0)]
    backend = Bm25Backend(pairs)
    results = backend.search(
        RetrievalQuery(cwe="CWE-89", task_signature=None, top_k_per_backend=3)
    )
    assert results == []


def test_bm25_tokenizer_splits_identifiers():
    """camelCase and snake_case should both tokenize into constituent words.
    Does not require the real_rag marker — pure-Python regex, no deps."""
    tokens = Bm25Backend._tokenize("getUserById and get_user_by_id")
    assert "get" in tokens
    assert "user" in tokens
    # camelCase split
    assert "by" in tokens
    assert "id" in tokens


def test_bm25_tokenizer_preserves_cwe_cve_identifiers():
    """CWE and CVE identifiers must survive tokenization as atomic tokens.

    Without this, the most lexically informative tokens in the security
    domain (CVE-2023-12345, CWE-78) get fragmented into 'cwe' + '78' or
    worse. Adapted from the medsingla/security-rag tokenizer survey
    2026-06-11; their BM25 used the same atomic-CWE/CVE preservation."""
    tokens = Bm25Backend._tokenize(
        "CWE-78 OS command injection in CVE-2023-12345 strcpy(buf, src)"
    )
    assert "cwe-78" in tokens, f"CWE-78 lost in {tokens}"
    assert "cve-2023-12345" in tokens, f"CVE-2023-12345 lost in {tokens}"
    # Should NOT have the fragmented versions.
    assert "78" not in tokens or tokens.count("78") == 0
    # Regular identifiers should still survive.
    assert "strcpy" in tokens
    assert "buf" in tokens


def test_bm25_tokenizer_lowercases_cwe_cve():
    """Both 'CWE-78' and 'cwe-78' should tokenize to the same form."""
    upper = Bm25Backend._tokenize("CWE-78")
    lower = Bm25Backend._tokenize("cwe-78")
    assert upper == lower
    assert upper == ["cwe-78"]


# ----------------------------------------------------------------------
# FAISS backend (real_rag - needs faiss)
# ----------------------------------------------------------------------


@pytest.mark.real_rag
def test_faiss_backend_returns_nearest_by_cosine():
    """Hand-craft 3-dim embeddings so the nearest neighbor is predictable."""
    pairs = [_pair(0), _pair(1), _pair(2)]
    pair_embeddings = [
        [1.0, 0.0, 0.0],  # pair 0: along +x
        [0.0, 1.0, 0.0],  # pair 1: along +y
        [0.0, 0.0, 1.0],  # pair 2: along +z
    ]
    backend = FaissBackend(pairs, pair_embeddings=pair_embeddings)
    # Query closest to pair 0:
    results = backend.search(
        RetrievalQuery(cwe="CWE-89", query_embedding=[0.9, 0.1, 0.0], top_k_per_backend=3)
    )
    assert len(results) == 3
    top_idx, top_score = results[0]
    assert top_idx == 0
    assert top_score > 0.9  # cosine with [1,0,0] should be ~0.99


@pytest.mark.real_rag
def test_faiss_backend_returns_empty_when_no_query_embedding():
    pairs = [_pair(0)]
    backend = FaissBackend(
        pairs, pair_embeddings=[[1.0, 0.0]]
    )
    results = backend.search(
        RetrievalQuery(cwe="CWE-89", query_embedding=None, top_k_per_backend=3)
    )
    assert results == []


@pytest.mark.real_rag
def test_faiss_backend_dim_mismatch_raises():
    pairs = [_pair(0)]
    backend = FaissBackend(pairs, pair_embeddings=[[1.0, 0.0]])
    with pytest.raises(ValueError):
        backend.search(
            RetrievalQuery(cwe="CWE-89", query_embedding=[1.0, 0.0, 0.0], top_k_per_backend=1)
        )


# ----------------------------------------------------------------------
# HfEmbedder (real_rag - needs sentence-transformers)
# ----------------------------------------------------------------------


@pytest.mark.real_rag
def test_hf_embedder_constructor_does_not_load():
    e = HfEmbedder()
    assert e._model is None  # not loaded


def test_hf_embedder_resolves_auto_device(monkeypatch):
    """`device='auto'` must resolve to a concrete cuda/cpu/mps string
    before SentenceTransformer.__init__ calls torch.to(device) — torch
    rejects 'auto' as an invalid device. Regression for the OOM-free
    RAG index job that crashed at runtime with:
        RuntimeError: Expected one of cpu, cuda, ... at start of device
        string: auto
    """
    captured = {}

    class _FakeST:
        def __init__(self, model_id, device):
            captured["device"] = device

    fake_st_module = types.SimpleNamespace(SentenceTransformer=_FakeST)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

    e = HfEmbedder(device="auto")
    e._ensure_loaded()
    assert captured["device"] in {"cuda", "cpu", "mps"}
    assert captured["device"] != "auto"


@pytest.mark.real_rag
def test_hf_embedder_embed_returns_normalized_vector():
    """Bge-base-en-v1.5 returns L2-normalized 768-dim vectors when
    normalize=True. We don't load the actual model in this test on the
    local box; this asserts via the lazy-load contract, which the real_hf
    smoke run on ROAR will exercise end-to-end."""
    e = HfEmbedder()
    try:
        vec = e.embed("def get_user(uid): pass")
    except NotImplementedError:
        pytest.skip("sentence-transformers not installed locally")
    assert len(vec) == 768
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-3


# ----------------------------------------------------------------------
# MockEmbedder (unit-level; no external deps)
# ----------------------------------------------------------------------


def test_mock_embedder_dict_lookup():
    e = MockEmbedder({"hello": [1.0, 0.0], "world": [0.0, 1.0]})
    assert e.embed("hello") == [1.0, 0.0]
    assert e.embed("world") == [0.0, 1.0]


def test_mock_embedder_callable():
    e = MockEmbedder(lambda t: [len(t) / 10.0, 0.0])
    v = e.embed("abc")
    assert v[0] == 0.3


def test_mock_embedder_default_for_missing():
    e = MockEmbedder({"x": [1.0]}, default=[0.0, 0.0])
    assert e.embed("not in dict") == [0.0, 0.0]


def test_mock_embedder_no_default_raises():
    e = MockEmbedder({"x": [1.0]})
    with pytest.raises(KeyError):
        e.embed("not in dict")
