"""Canonical Prompt schema and DataAdapter base interface.

Adapters normalize source corpora (CVEfixes, SecCodePLT, CWEval, Juliet)
into a uniform `Prompt` shape consumed by the trainer and the reliability
oracle. See docs/data_prep_spec.md §2 for the schema rationale.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

# Reuse the language enum from the reliability oracle to avoid drift.
# Prompt.language is the same type as TestSpec.language.
from ..reward.reliability_oracle import Language, TestSpec
from ..rag.schema import ExemplarPair  # re-exported for convenience

__all__ = [
    "DataAdapter",
    "ExemplarPair",
    "Language",
    "Prompt",
    "TestSpec",
    "normalize_cwe",
    "normalize_language",
]


_CWE_RE = re.compile(r"^CWE-(\d+)$")


def normalize_cwe(raw: str) -> str:
    """Coerce common CWE encodings to 'CWE-NNN'. Raises on un-parseable."""
    s = str(raw).strip()
    if _CWE_RE.match(s):
        return s
    if s.upper().startswith("CWE-"):
        candidate = "CWE-" + s.split("-", 1)[1]
        if _CWE_RE.match(candidate):
            return candidate
    if s.isdigit():
        return f"CWE-{s}"
    raise ValueError(f"cannot normalize {raw!r} to 'CWE-NNN'")


_LANGUAGE_ALIASES: dict[str, Language] = {
    "python": Language.PYTHON,
    "py": Language.PYTHON,
    "python3": Language.PYTHON,
    "c": Language.C,
    "cpp": Language.CPP,
    "c++": Language.CPP,
    "cxx": Language.CPP,
}


def normalize_language(raw: str) -> Language:
    """Map common language strings to the Language enum. Raises on unknown."""
    key = str(raw).strip().lower()
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    raise ValueError(f"unsupported language: {raw!r}")


@dataclass
class Prompt:
    """A training- or evaluation-time prompt.

    `prompt_text` is exactly what the model sees. The adapter decides
    whether to include the function signature, docstring, I/O contract,
    etc. (Most adapters include all three.)

    `test_spec` is the `TestSpec` consumed by the reliability oracle to
    score the model's completion. For retrieval-index-only data (Juliet),
    `test_spec` may have an empty `test_cases` list — the oracle will then
    only score compile/run/output, not test-pass.
    """

    id: str
    source: str
    language: Language
    target_cwe: str
    prompt_text: str
    test_spec: TestSpec
    task_signature: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_text.strip():
            raise ValueError("prompt_text must be non-empty")
        # Normalize the CWE (raises if malformed).
        self.target_cwe = normalize_cwe(self.target_cwe)
        # Sanity-check that the adapter set a consistent id.
        if not self.id:
            raise ValueError("Prompt.id must be non-empty")
        if not self.source:
            raise ValueError("Prompt.source must be non-empty")


def make_prompt_id(source: str, *parts: str) -> str:
    """Construct a stable Prompt id from a source name and free-form parts.

    The id is `<source>:<short-hash-of-parts>` so it is unique within an
    adapter's output and human-readable in logs. Re-running the adapter
    on the same source data produces the same ids (idempotence).
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return f"{source}:{h.hexdigest()[:16]}"


class DataAdapter(ABC):
    """Normalizes a raw corpus into `Prompt` and (optionally) `ExemplarPair`."""

    @property
    @abstractmethod
    def source_name(self) -> str: ...

    @abstractmethod
    def load(self) -> Iterator[Prompt]: ...

    def load_exemplar_pairs(self) -> Iterator[ExemplarPair]:
        """Optional: paired (secure, vulnerable) data for the retrieval index.

        Eval-only adapters (SecCodePLT, CWEval) raise NotImplementedError.
        Adapters with paired data (CVEfixes, Juliet) override this.
        """
        raise NotImplementedError(
            f"{self.source_name} does not produce exemplar pairs"
        )
