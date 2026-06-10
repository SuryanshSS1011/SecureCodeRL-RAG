"""Retrieval-augmented generation per docs/rag_spec.md v0.1."""

from .embedder import HfEmbedder, MockEmbedder, cosine_similarity
from .loader import load_embedder, load_retriever
from .r_rag import RagDiagnostics, compute_r_rag, r_rag_missing
from .retriever import (
    Bm25Backend,
    FaissBackend,
    HybridRetriever,
    IndexBackend,
    RetrievalQuery,
    StubBackend,
)
from .schema import ExemplarPair, Language, RetrievalHit

__all__ = [
    "Bm25Backend",
    "ExemplarPair",
    "FaissBackend",
    "HfEmbedder",
    "HybridRetriever",
    "IndexBackend",
    "Language",
    "MockEmbedder",
    "RagDiagnostics",
    "RetrievalHit",
    "RetrievalQuery",
    "StubBackend",
    "compute_r_rag",
    "cosine_similarity",
    "load_embedder",
    "load_retriever",
    "r_rag_missing",
]
