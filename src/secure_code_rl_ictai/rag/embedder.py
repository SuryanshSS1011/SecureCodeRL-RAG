"""Embedder wrapper for the RAG retriever.

Per docs/rag_spec.md §4: bge-base-en-v1.5, 768-dim, L2-normalized output.

Two implementations:
  - `HfEmbedder` lazy-loads sentence-transformers on first call.
  - `MockEmbedder` returns deterministic hand-built vectors for unit tests.

The pipeline takes a `CompletionEmbedder` (one `.embed(text)` method) so
either implementation slots in. HfEmbedder constructor is cheap; load
happens on first `.embed()` call.
"""

from __future__ import annotations

import math
from typing import Callable, Optional


class MockEmbedder:
    """Returns canned embeddings per input. For unit tests.

    `responses` is either a callable `text -> list[float]` or a dict
    `text -> list[float]`. Unknown text gets `default` (an empty list ->
    raises in downstream R_RAG).
    """

    def __init__(
        self,
        responses: dict[str, list[float]] | Callable[[str], list[float]],
        default: Optional[list[float]] = None,
    ) -> None:
        self._responses = responses
        self._default = default

    def embed(self, text: str) -> list[float]:
        if callable(self._responses):
            return self._responses(text)
        if text in self._responses:
            return self._responses[text]
        if self._default is not None:
            return list(self._default)
        raise KeyError(f"no embedding for {text!r}; supply default or add to dict")


class HfEmbedder:
    """Lazy-loaded sentence-transformers embedder.

    Default model: `BAAI/bge-base-en-v1.5` per docs/rag_spec.md §4.
    Output is L2-normalized so cosine similarity == dot product.

    Construction is cheap (records model_id only); the model loads on
    first `.embed()` call. Same pattern as HfBaselineModel.
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda",
        normalize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.normalize = normalize
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise NotImplementedError(
                f"HfEmbedder requires sentence-transformers ({exc}). "
                "Install in the training venv, or use MockEmbedder for unit tests."
            ) from exc

        self._model = SentenceTransformer(self.model_id, device=self.device)

    def embed(self, text: str) -> list[float]:
        self._ensure_loaded()
        assert self._model is not None
        vec = self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding for indexing. Same model; lazy-loads if needed."""
        self._ensure_loaded()
        assert self._model is not None
        vecs = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. Convenience helper."""
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
