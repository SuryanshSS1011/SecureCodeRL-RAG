"""Retrieval-index schema per docs/rag_spec.md §2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Language(str, Enum):
    PYTHON = "python"
    C = "c"
    CPP = "cpp"


@dataclass
class ExemplarPair:
    """A paired secure/vulnerable code example for one CWE.

    See docs/rag_spec.md §2 for the rationale on pairing at construction
    time rather than retrieving positives and negatives separately.
    """

    cwe: str
    task_signature: str
    e_pos: str
    e_neg: str
    language: Language
    source: str
    cve_id: Optional[str] = None
    fix_commit: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cwe.startswith("CWE-"):
            raise ValueError(f"cwe must look like 'CWE-NNN', got {self.cwe!r}")
        if not self.e_pos.strip():
            raise ValueError("e_pos must be non-empty")
        if not self.e_neg.strip():
            raise ValueError("e_neg must be non-empty")


@dataclass
class RetrievalHit:
    """A single retrieval result."""

    pair: ExemplarPair
    rrf_score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
