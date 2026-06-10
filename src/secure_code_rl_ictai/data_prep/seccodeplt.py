"""SecCodePLT adapter (eval-only).

Reads SecCodePLT-shaped records from a JSON or JSONL file. SecCodePLT
ships dynamic / unit-test oracles; we lower its richer test format into
the TestCase (stdin -> expected stdout) shape consumed by the reliability
oracle.

In v0.1 we accept a JSONL file with one record per line. Each record:

    {
        "id": "seccodeplt-001",
        "category": "CWE-89",         # SecCodePLT calls them "categories"
        "language": "python",
        "task_description": "Write a function get_user(uid)...",
        "signature": "def get_user(uid):",
        "tests": [
            {"stdin": "1\n", "expected_stdout": "alice\n", "timeout_s": 5.0},
            ...
        ],
        "metadata": {...}             # optional source metadata
    }

The "tests" field is the simplified form; SecCodePLT's native format
(function-call assertions) is converted upstream by
`scripts/build_seccodeplt_eval.py` (TBD).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from ..reward.reliability_oracle import TestCase, TestSpec
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
class SecCodePltConfig:
    jsonl_path: Path
    languages: frozenset[Language] = field(
        default_factory=lambda: frozenset(
            {Language.PYTHON, Language.C, Language.CPP}
        )
    )
    target_cwes: Optional[frozenset[str]] = None
    dataset_version: str = "seccodeplt-neurips25"
    adapter_version: str = "0.1"


class SecCodePltAdapter(DataAdapter):
    def __init__(self, config: SecCodePltConfig) -> None:
        self.config = config

    @property
    def source_name(self) -> str:
        return "seccodeplt"

    def load(self) -> Iterator[Prompt]:
        if not self.config.jsonl_path.exists():
            raise FileNotFoundError(
                f"SecCodePLT JSONL does not exist: {self.config.jsonl_path}"
            )

        with open(self.config.jsonl_path) as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "skipping malformed JSON at line %d (%s)", lineno, exc
                    )
                    continue

                prompt = self._record_to_prompt(rec)
                if prompt is not None:
                    yield prompt

    def _record_to_prompt(self, rec: dict) -> Optional[Prompt]:
        # Language filter.
        try:
            lang = normalize_language(rec.get("language", ""))
        except ValueError:
            return None
        if lang not in self.config.languages:
            return None

        # CWE filter.
        try:
            cwe = normalize_cwe(rec.get("category", ""))
        except ValueError:
            return None
        if self.config.target_cwes is not None and cwe not in self.config.target_cwes:
            return None

        description = rec.get("task_description", "").strip()
        signature = rec.get("signature", "").strip()
        if not description and not signature:
            return None

        prompt_text = description if description else signature
        if description and signature and signature not in description:
            # Append signature if the description doesn't already mention it.
            prompt_text = f"{description}\n\n{signature}"

        test_cases = self._parse_tests(rec.get("tests", []))
        test_spec = TestSpec(
            language=lang,
            test_cases=test_cases,
            extra_files=dict(rec.get("extra_files", {}) or {}),
            entry_module=rec.get("entry_module"),
        )

        return Prompt(
            id=make_prompt_id(
                self.source_name,
                rec.get("id") or f"line:{rec}",
            ),
            source=self.source_name,
            language=lang,
            target_cwe=cwe,
            prompt_text=prompt_text,
            test_spec=test_spec,
            task_signature=signature or None,
            metadata={
                "seccodeplt_id": rec.get("id"),
                "category": cwe,
                **rec.get("metadata", {}),
                "dataset_version": self.config.dataset_version,
                "adapter_version": self.config.adapter_version,
            },
        )

    @staticmethod
    def _parse_tests(raw: list) -> list[TestCase]:
        out: list[TestCase] = []
        for t in raw:
            if not isinstance(t, dict):
                continue
            stdin = str(t.get("stdin", ""))
            expected = str(t.get("expected_stdout", ""))
            timeout = float(t.get("timeout_s", 5.0))
            out.append(
                TestCase(
                    input_stdin=stdin, expected_stdout=expected, timeout_s=timeout
                )
            )
        return out
