"""R_RAG: continuous retrieval-grounded reward with copy guard (paper Eq. 2).

    R_RAG(y, e+) = cos(phi(y), phi(e+)) * 1[cos(phi(y), phi(e+)) <= tau_copy]

scaled by lambda_rag when composed into the reward (Eq. 4). The binary
control of Section IV-A replaces it with 1[cos(y, e+) > cos(y, e-)],
where e- is the pre-fix version of the retrieved exemplar.

This module is pure math: embedding generation, retrieval, and pair
selection are upstream.

All embeddings are assumed L2-normalized; cosine similarity = dot product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class RagDiagnostics:
    sim_pos: float
    sim_neg: float
    raw: float
    copy_guard_hit: bool
    lambda_rag: float
    copy_guard_threshold: float


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding dim mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def compute_r_rag(
    completion_embedding: Sequence[float],
    e_pos_embedding: Sequence[float],
    e_neg_embedding: Sequence[float],
    *,
    lambda_rag: float = 0.1,
    copy_guard_threshold: float = 0.95,
    binary: bool = False,
) -> tuple[float, RagDiagnostics]:
    """Compute lambda_rag * R_RAG.

    Args:
        completion_embedding: L2-normalized embedding of `y`.
        e_pos_embedding: retrieved secure exemplar embedding.
        e_neg_embedding: pre-fix (vulnerable) version of the exemplar;
            read only by the binary control.
        lambda_rag: reward weight from Eq. 4.
        copy_guard_threshold: tau_copy; cos(y, e+) above it zeroes R_RAG.

    Returns:
        (value, diagnostics). `value` is in [-lambda_rag, lambda_rag * tau_copy].
    """
    sim_pos = _dot(completion_embedding, e_pos_embedding)
    sim_neg = _dot(completion_embedding, e_neg_embedding)
    if binary:
        raw = lambda_rag if sim_pos > sim_neg else 0.0
    else:
        raw = lambda_rag * sim_pos
    copy_guard_hit = sim_pos > copy_guard_threshold
    value = 0.0 if copy_guard_hit else raw
    diag = RagDiagnostics(
        sim_pos=sim_pos,
        sim_neg=sim_neg,
        raw=raw,
        copy_guard_hit=copy_guard_hit,
        lambda_rag=lambda_rag,
        copy_guard_threshold=copy_guard_threshold,
    )
    return value, diag


def r_rag_missing() -> tuple[float, RagDiagnostics]:
    """Returned when the index has no matching pair for the prompt's CWE.

    R_RAG = 0 on a retrieval miss (Section IV-A). Diagnostics
    are filled with sentinel values so downstream logging can detect
    misses by `copy_guard_hit=False, sim_pos==sim_neg==0`.
    """
    return 0.0, RagDiagnostics(
        sim_pos=0.0,
        sim_neg=0.0,
        raw=0.0,
        copy_guard_hit=False,
        lambda_rag=0.0,
        copy_guard_threshold=1.0,
    )
