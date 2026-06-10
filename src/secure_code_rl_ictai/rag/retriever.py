"""Hybrid BM25 + dense retriever with RRF fusion.

Implements docs/rag_spec.md §5. The retriever is index-agnostic at the
type level: an `IndexBackend` interface abstracts over the actual BM25
and FAISS implementations, so unit tests can supply hand-built ranked
lists without bringing in dependencies.

`Bm25Backend` (bm25s, rank_bm25 fallback) and `FaissBackend` (bge-base
embeddings) are the real backends; `rag.loader` builds them from an index
written by scripts/build_rag_index.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .schema import ExemplarPair, RetrievalHit


@dataclass
class RetrievalQuery:
    """A query against the retrieval index.

    Either or both of `task_signature` and `query_embedding` may be
    provided; BM25 backends need the signature, dense backends need the
    embedding. A CWE filter narrows candidates before fusion.
    """

    cwe: str
    task_signature: Optional[str] = None
    query_embedding: Optional[list[float]] = None
    top_k_per_backend: int = 20


class IndexBackend(ABC):
    """Returns a ranked list of (pair_id, score) for the query.

    Implementations:
      - Bm25Backend: sparse over `task_signature + e_pos[:1024]`.
      - FaissBackend: dense over `embed(e_pos)`.
      - StubBackend (tests): caller supplies the ranked list directly.
    """

    @abstractmethod
    def search(self, query: RetrievalQuery) -> list[tuple[int, float]]: ...


class StubBackend(IndexBackend):
    """Returns a pre-supplied ranked list. For unit tests."""

    def __init__(self, ranked_list: list[tuple[int, float]]) -> None:
        self._ranked = list(ranked_list)

    def search(self, query: RetrievalQuery) -> list[tuple[int, float]]:
        return list(self._ranked[: query.top_k_per_backend])


class Bm25Backend(IndexBackend):
    """BM25 over `task_signature + e_pos[:1024]` per pair.

    Uses `bm25s` (Lú & Bonet 2024) by default — ~500x faster than
    `rank_bm25` at our 41k-pair corpus scale, vectorized scipy sparse
    matrices for both indexing and querying. Drops back to `rank_bm25`
    when bm25s isn't available.

    Tokenization preserves CWE/CVE identifiers as atomic tokens
    (`CWE-78` and `CVE-2023-12345` are single tokens, not split into
    `cwe`+`78`). Without this, BM25 loses an important security-domain
    signal because the most lexically informative tokens get fragmented.

    camelCase + snake_case + dotted identifiers split into constituents.
    """

    def __init__(self, pairs: list[ExemplarPair]) -> None:
        self.pairs = pairs
        self._bm25 = None
        self._backend_kind = None  # "bm25s" or "rank_bm25"

    def _ensure_loaded(self) -> None:
        if self._bm25 is not None:
            return

        corpus_tokens = [self._tokenize(self._document(p)) for p in self.pairs]

        # Prefer bm25s; fall back to rank_bm25 for forward-compatibility
        # with older environments.
        try:
            import bm25s
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            self._bm25 = retriever
            self._backend_kind = "bm25s"
            return
        except ImportError:
            pass

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise NotImplementedError(
                f"Bm25Backend requires bm25s or rank_bm25 ({exc}). "
                "Install with `pip install bm25s` (preferred) or "
                "`pip install rank_bm25`."
            ) from exc

        self._bm25 = BM25Okapi(corpus_tokens)
        self._backend_kind = "rank_bm25"

    @staticmethod
    def _document(pair: ExemplarPair) -> str:
        """Per spec §3, the BM25 document is task_signature + first 1024
        chars of e_pos. We treat empty signatures as e_pos-only."""
        sig = (pair.task_signature or "").strip()
        body = pair.e_pos[:1024]
        return f"{sig}\n{body}" if sig else body

    # Single regex that preserves CWE-NNN, CVE-NNNN-NNNN, and identifiers.
    # Order in the alternation matters: longer-prefix patterns first so the
    # CWE/CVE forms aren't pre-empted by the identifier rule.
    _TOKEN_RE = None  # lazy-compiled below

    @classmethod
    def _get_token_re(cls):
        if cls._TOKEN_RE is None:
            import re
            # cwe-NNN | cve-NNNN-NNNN(-NNNN...) | identifier | bare-number
            # All lowercased before matching.
            cls._TOKEN_RE = re.compile(
                r"cwe-\d+|cve-\d+(?:-\d+)+|[a-z_][a-z0-9_]*|\d+"
            )
        return cls._TOKEN_RE

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """Lowercase + identifier-aware split, with CWE/CVE preservation.

        - `CWE-78` and `CVE-2023-12345` survive as single tokens.
        - camelCase boundaries split (so `getUserId` -> `getuserid` after
          lowercase, but the camel split runs *before* lowercase to
          preserve word boundaries: `get user id`).
        - snake_case and dotted/punctuated names split.
        - Punctuation (other than the CVE/CWE dash) is dropped.
        """
        import re

        # Step 1: insert spaces at camelCase boundaries while still mixed case.
        camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Step 2: lowercase everything.
        lowered = camel.lower()
        # Step 3: regex-match the preserved-token forms.
        return cls._get_token_re().findall(lowered)

    def search(self, query: RetrievalQuery) -> list[tuple[int, float]]:
        if query.task_signature is None:
            return []
        self._ensure_loaded()
        assert self._bm25 is not None
        query_tokens = self._tokenize(query.task_signature)
        if not query_tokens:
            return []

        k = min(query.top_k_per_backend, len(self.pairs))
        if k <= 0:
            return []

        import numpy as np

        if self._backend_kind == "bm25s":
            # bm25s API: .retrieve returns (results, scores) as parallel arrays.
            results, scores = self._bm25.retrieve(
                [query_tokens], k=k, show_progress=False
            )
            ranked = [
                (int(results[0, i]), float(scores[0, i]))
                for i in range(results.shape[1])
            ]
            return ranked
        else:
            # rank_bm25 fallback.
            scores = self._bm25.get_scores(query_tokens)
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            return [(int(i), float(scores[i])) for i in idx]


class FaissBackend(IndexBackend):
    """Dense FAISS-flat index over `embed(e_pos)` for each pair.

    Construction requires either pre-computed `pair_embeddings` (one per
    pair, in the same order as `pairs`) OR an `embedder` that produces
    them on demand. Embeddings are L2-normalized; the index uses
    `IndexFlatIP` (inner product = cosine when normalized).

    Lazy-imports `faiss`. The corpus embedding pass is the expensive part;
    for ~40k pairs at 768-dim with a CPU encoder it's a few minutes. The
    typical pipeline pre-computes embeddings during `scripts/build_rag_index.py`
    and serializes them with the index.
    """

    def __init__(
        self,
        pairs: list[ExemplarPair],
        *,
        pair_embeddings: Optional[list[list[float]]] = None,
        embedder=None,  # HfEmbedder or compatible (.embed_batch + .embed)
    ) -> None:
        self.pairs = pairs
        self._provided_embeddings = pair_embeddings
        self.embedder = embedder
        self._index = None
        self._dim = None

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        try:
            import faiss
        except ImportError as exc:
            raise NotImplementedError(
                f"FaissBackend requires faiss ({exc}). "
                "Install `faiss-cpu` or `faiss-gpu`; "
                "or use StubBackend for unit tests."
            ) from exc

        import numpy as np

        if self._provided_embeddings is not None:
            vecs = np.asarray(self._provided_embeddings, dtype=np.float32)
        elif self.embedder is not None:
            texts = [p.e_pos for p in self.pairs]
            vecs = np.asarray(
                self.embedder.embed_batch(texts), dtype=np.float32
            )
        else:
            raise ValueError(
                "FaissBackend needs either pair_embeddings or an embedder"
            )

        if vecs.ndim != 2 or vecs.shape[0] != len(self.pairs):
            raise ValueError(
                f"embedding matrix shape {vecs.shape} does not match "
                f"len(pairs)={len(self.pairs)}"
            )
        self._dim = vecs.shape[1]
        # Re-normalize defensively. Sentence-Transformers already does this
        # when normalize_embeddings=True, but external embedders may not.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        index = faiss.IndexFlatIP(self._dim)
        index.add(vecs)
        self._index = index

    def search(self, query: RetrievalQuery) -> list[tuple[int, float]]:
        if query.query_embedding is None:
            return []
        self._ensure_loaded()
        assert self._index is not None and self._dim is not None

        import numpy as np

        q = np.asarray(query.query_embedding, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self._dim:
            raise ValueError(
                f"query embedding dim {q.shape[1]} != index dim {self._dim}"
            )
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        k = min(query.top_k_per_backend, len(self.pairs))
        if k <= 0:
            return []
        distances, indices = self._index.search(q, k)
        # IndexFlatIP returns inner-product scores; higher is better.
        return [
            (int(indices[0, i]), float(distances[0, i]))
            for i in range(k)
            if indices[0, i] >= 0  # FAISS returns -1 for missing
        ]


class HybridRetriever:
    """Two-stage retrieval: CWE pre-filter -> hybrid BM25 + dense -> RRF.

    Per docs/rag_spec.md §5:
        1. Filter pairs by exact CWE match (or parent CWE as fallback).
        2. Get top-K from each backend.
        3. Fuse via modified Reciprocal Rank Fusion with k = 60.
        4. Return top-1.

    The retriever does NOT own the index data. It is constructed with the
    pair list and the two backends; it does not pre-filter the backends
    by CWE itself (that would mean re-indexing per query). Instead, the
    fusion stage drops candidates whose CWE doesn't match.
    """

    def __init__(
        self,
        pairs: list[ExemplarPair],
        bm25_backend: IndexBackend,
        dense_backend: IndexBackend,
        rrf_k: int = 60,
        cwe_parent_fallback: bool = True,
        mode: str = "best",
    ) -> None:
        """mode (Section IV-A controls):
            "best"        -> top-ranked exemplar for the prompt's CWE (CARGO)
            "adversarial" -> lowest-ranked exemplar within the prompt's CWE
            "random"      -> exemplar sampled uniformly from the whole index,
                             any CWE
        """
        self.pairs = pairs
        self.bm25 = bm25_backend
        self.dense = dense_backend
        self.rrf_k = rrf_k
        self.cwe_parent_fallback = cwe_parent_fallback
        if mode not in ("best", "adversarial", "random"):
            raise ValueError(f"unknown retriever mode: {mode!r}")
        self.mode = mode
        if mode == "random":
            import random as _random
            # Per-retriever RNG so cells with --seed produce reproducible
            # random retrieval. The trainer's seed propagates via this
            # constructor; default 0 is harmless.
            self._rng = _random.Random(0)

    @staticmethod
    def _rrf_combine(
        bm25_ranked: list[tuple[int, float]],
        dense_ranked: list[tuple[int, float]],
        k: int,
    ) -> dict[int, tuple[float, Optional[int], Optional[int]]]:
        """Reciprocal Rank Fusion. Returns {pair_id: (rrf_score, bm25_rank, dense_rank)}."""
        combined: dict[int, tuple[float, Optional[int], Optional[int]]] = {}
        for rank, (pair_id, _score) in enumerate(bm25_ranked, start=1):
            existing = combined.get(pair_id, (0.0, None, None))
            new_rrf = existing[0] + 1.0 / (k + rank)
            combined[pair_id] = (new_rrf, rank, existing[2])
        for rank, (pair_id, _score) in enumerate(dense_ranked, start=1):
            existing = combined.get(pair_id, (0.0, None, None))
            new_rrf = existing[0] + 1.0 / (k + rank)
            combined[pair_id] = (new_rrf, existing[1], rank)
        return combined

    def retrieve(self, query: RetrievalQuery) -> Optional[RetrievalHit]:
        """Return the top-1 hit, or None if no candidates match the CWE."""
        if self.mode == "random":
            if not self.pairs:
                return None
            pid = self._rng.randrange(len(self.pairs))
            return RetrievalHit(
                pair=self.pairs[pid], rrf_score=0.0, bm25_rank=None, dense_rank=None,
            )

        bm25_ranked = self.bm25.search(query)
        dense_ranked = self.dense.search(query)

        combined = self._rrf_combine(bm25_ranked, dense_ranked, self.rrf_k)

        # CWE filter: drop candidates whose pair.cwe != query.cwe.
        def cwe_ok(pid: int) -> bool:
            pair = self.pairs[pid]
            return pair.cwe == query.cwe

        filtered = {pid: v for pid, v in combined.items() if cwe_ok(pid)}

        if not filtered and self.cwe_parent_fallback:
            # Parent-CWE fallback is a hierarchy lookup. For v0.1 we use a
            # minimal hand-coded subset; full CWE hierarchy traversal is a
            # TODO and lives in data/cwe_hierarchy.json once built.
            parent = _CWE_PARENTS.get(query.cwe)
            if parent is not None:
                def parent_ok(pid: int) -> bool:
                    return self.pairs[pid].cwe == parent

                filtered = {pid: v for pid, v in combined.items() if parent_ok(pid)}

        if not filtered:
            return None

        if self.mode == "best":
            best_pid, (rrf, bm25_rank, dense_rank) = max(
                filtered.items(), key=lambda kv: kv[1][0]
            )
        else:  # "adversarial"
            best_pid, (rrf, bm25_rank, dense_rank) = min(
                filtered.items(), key=lambda kv: kv[1][0]
            )
        return RetrievalHit(
            pair=self.pairs[best_pid],
            rrf_score=rrf,
            bm25_rank=bm25_rank,
            dense_rank=dense_rank,
        )


# Minimal CWE parent map for v0.1. Expanded to the full hierarchy when
# data/cwe_hierarchy.json is built from MITRE.
_CWE_PARENTS: dict[str, str] = {
    "CWE-787": "CWE-119",  # OOB Write -> Memory Boundary
    "CWE-125": "CWE-119",  # OOB Read -> Memory Boundary
    "CWE-416": "CWE-119",  # UAF -> Memory Boundary
    "CWE-78": "CWE-77",    # OS Command Injection -> Command Injection
    "CWE-79": "CWE-74",    # XSS -> Injection
    "CWE-89": "CWE-74",    # SQL Injection -> Injection
    "CWE-94": "CWE-74",    # Code Injection -> Injection
    "CWE-862": "CWE-284",  # Missing Authorization -> Improper Access Control
    "CWE-306": "CWE-287",  # Missing Auth -> Improper Authentication
}
