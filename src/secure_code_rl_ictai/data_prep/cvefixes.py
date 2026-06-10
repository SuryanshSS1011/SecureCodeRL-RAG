"""CVEfixes (v1.0.8) adapter.

Reads CVEfixes-shaped records from a directory of JSONL files. Each line
is one fix-touched function with both pre-fix and post-fix code.

Real CVEfixes is a SQLite database; in v0.1 we operate on a JSONL export
produced by `scripts/export_cvefixes_to_jsonl.py` (TBD). This decoupling
keeps the adapter unit-testable with synthetic fixtures and keeps the
heavy SQL out of the training-time path entirely.

JSONL record shape (one per line):
    {
        "fix_commit": "abc123...",
        "cve_id": "CVE-2023-12345",
        "project": "django",
        "license": "BSD-3-Clause",
        "language": "Python",          # case-insensitive
        "cwe": "CWE-89",
        "file_path": "django/db/orm.py",
        "function_name": "get_user",
        "signature": "def get_user(user_id):",
        "pre_fix": "...vulnerable code...",
        "post_fix": "...fixed code..."
    }

The adapter:
  - filters by language (config) and CWE (config).
  - emits one ExemplarPair per record.
  - emits one Prompt per record IF a clean signature can be extracted
    AND a heuristic-built TestSpec is plausible (see _build_test_spec).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from ..rag.schema import ExemplarPair
from ..rag.schema import Language as RagLanguage
from ..reward.reliability_oracle import TestSpec
from .schema import (
    DataAdapter,
    Language,
    Prompt,
    make_prompt_id,
    normalize_cwe,
    normalize_language,
)

logger = logging.getLogger(__name__)


@dataclass
class CvefixesConfig:
    jsonl_dir: Path
    languages: frozenset[Language] = field(
        default_factory=lambda: frozenset(
            {Language.PYTHON, Language.C, Language.CPP}
        )
    )
    target_cwes: Optional[frozenset[str]] = None  # None = accept any CWE
    dataset_version: str = "cvefixes-v1.0.8"
    adapter_version: str = "0.1"


_REQUIRED_FIELDS = (
    "language", "cwe", "pre_fix", "post_fix", "function_name", "signature",
)


class CvefixesAdapter(DataAdapter):
    def __init__(self, config: CvefixesConfig) -> None:
        self.config = config

    @property
    def source_name(self) -> str:
        return "cvefixes"

    # ---- iteration helpers ----

    def _iter_records(self) -> Iterator[dict]:
        """Yield raw record dicts from every .jsonl file in the configured dir."""
        if not self.config.jsonl_dir.exists():
            raise FileNotFoundError(
                f"CVEfixes JSONL dir does not exist: {self.config.jsonl_dir}"
            )
        files = sorted(self.config.jsonl_dir.glob("*.jsonl"))
        if not files:
            logger.warning("no .jsonl files in %s", self.config.jsonl_dir)
        for path in files:
            with open(path) as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "skipping malformed JSON at %s:%d (%s)",
                            path, lineno, exc,
                        )
                        continue
                    yield rec

    def _record_passes_filters(self, rec: dict) -> Optional[tuple[Language, str]]:
        """Apply language + CWE filters. Return (lang, cwe) if record passes, else None.

        Records with missing required fields are dropped (with a log).
        """
        missing = [f for f in _REQUIRED_FIELDS if not rec.get(f)]
        if missing:
            logger.debug("dropping record missing fields: %s", missing)
            return None

        try:
            lang = normalize_language(rec["language"])
        except ValueError:
            return None
        if lang not in self.config.languages:
            return None

        try:
            cwe = normalize_cwe(rec["cwe"])
        except ValueError:
            return None
        if self.config.target_cwes is not None and cwe not in self.config.target_cwes:
            return None

        return lang, cwe

    # ---- emit Prompt ----

    def load(self) -> Iterator[Prompt]:
        """Emit Prompts for records with a usable signature.

        Records that fail signature extraction or test-spec construction
        are skipped silently (they're still available via
        load_exemplar_pairs()). The current implementation only emits
        prompts when the record has a non-empty `signature` field; richer
        prompt construction (docstring extraction, I/O contracts) is a
        TODO that will plug in via a separate template module.
        """
        for rec in self._iter_records():
            filtered = self._record_passes_filters(rec)
            if filtered is None:
                continue
            lang, cwe = filtered

            signature = rec.get("signature", "").strip()
            if not signature:
                continue

            prompt_text = signature  # v0.1: signature is the prompt
            test_spec = self._build_test_spec(rec, lang)

            yield Prompt(
                id=make_prompt_id(
                    self.source_name,
                    rec.get("fix_commit", ""),
                    rec.get("file_path", ""),
                    rec.get("function_name", ""),
                ),
                source=self.source_name,
                language=lang,
                target_cwe=cwe,
                prompt_text=prompt_text,
                test_spec=test_spec,
                task_signature=signature,
                metadata={
                    "cve_id": rec.get("cve_id"),
                    "fix_commit": rec.get("fix_commit"),
                    "project": rec.get("project"),
                    "license": rec.get("license"),
                    "file_path": rec.get("file_path"),
                    "function_name": rec.get("function_name"),
                    "dataset_version": self.config.dataset_version,
                    "adapter_version": self.config.adapter_version,
                },
            )

    # ---- emit ExemplarPair ----

    def load_exemplar_pairs(self) -> Iterator[ExemplarPair]:
        """Emit one ExemplarPair per fix-touched function.

        Deduplicates by sha256(pre_fix + "|" + post_fix) within a single
        adapter run. Cross-run dedup is the dataset-build script's job.
        """
        seen: set[str] = set()
        for rec in self._iter_records():
            filtered = self._record_passes_filters(rec)
            if filtered is None:
                continue
            lang, cwe = filtered

            pre = (rec.get("pre_fix") or "").strip()
            post = (rec.get("post_fix") or "").strip()
            if not pre or not post:
                continue
            if pre == post:
                # No real fix; skip.
                continue

            dedup_key = _hash_pair(pre, post)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            yield ExemplarPair(
                cwe=cwe,
                task_signature=rec.get("signature", ""),
                e_pos=post,
                e_neg=pre,
                language=_rag_language(lang),
                source=self.source_name,
                cve_id=rec.get("cve_id"),
                fix_commit=rec.get("fix_commit"),
                metadata={
                    "project": rec.get("project"),
                    "license": rec.get("license"),
                    "file_path": rec.get("file_path"),
                    "function_name": rec.get("function_name"),
                    "dataset_version": self.config.dataset_version,
                    "adapter_version": self.config.adapter_version,
                },
            )

    # ---- helpers ----

    def _build_test_spec(self, rec: dict, lang: Language) -> TestSpec:
        """Construct a TestSpec from the record.

        CVEfixes does not ship functional tests with its fixes, so v0.1
        emits an empty test_cases list. The reliability oracle then scores
        compile/run/output but not test-pass. Better: pair these Prompts
        with synthetic tests via a template module (TODO).
        """
        return TestSpec(language=lang, test_cases=[])


def _hash_pair(pre: str, post: str) -> str:
    h = hashlib.sha256()
    h.update(pre.encode("utf-8"))
    h.update(b"|")
    h.update(post.encode("utf-8"))
    return h.hexdigest()


def _rag_language(lang: Language) -> RagLanguage:
    """Bridge from oracle Language enum to RAG schema Language enum."""
    return {
        Language.PYTHON: RagLanguage.PYTHON,
        Language.C: RagLanguage.C,
        Language.CPP: RagLanguage.CPP,
    }[lang]
