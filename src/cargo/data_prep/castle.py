"""CASTLE adapter (eval-only).

Reads the CASTLE-C250 benchmark from CASTLE-Benchmark repo
(github.com/CASTLE-Benchmark/CASTLE-Benchmark), arxiv:2503.09433.

CASTLE ships a single JSON file `datasets/CASTLE-C250.json` plus the raw
`.c` files under `datasets/CASTLE-C250/`. Each test entry has full code
inline plus a `lines` field with vulnerable line numbers (ground truth)
and a `vulnerable: bool` flag.

Schema (one record per test, accessed via top-level `tests` list):
    {
        "name": "CASTLE-22-1.c",
        "version": 1.1,
        "compile": "gcc CASTLE-22-1.c -o CASTLE-22-1",
        "vulnerable": true,
        "description": "Improper limitation of a pathname leads to ...",
        "cwe": 22,
        "lines": [16, 19],      # ground-truth vulnerable lines (integer)
        "id": "22-1",
        "hash": "82ffd8...",
        "code": "#include <stdio.h>\n..."
    }

We model each entry as a Prompt where:
  - prompt_text = the `description` plus instruction to produce a fixed version
  - target_cwe = "CWE-{cwe}" (numeric CWE -> string)
  - language = Language.C (CASTLE is C-only; cpp/python = 0 entries)
  - test_spec = empty test_cases (SAST scoring path); compile command is
    preserved in metadata for future SAST harness wiring.

CASTLE is licensed under CC-BY-4.0 (joint research project, paper under
peer-review). Cite per `cite_castle_v0.1.bib` in `docs/reference/`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

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
class CastleConfig:
    json_path: Path
    languages: frozenset[Language] = field(
        default_factory=lambda: frozenset({Language.C})
    )
    target_cwes: Optional[frozenset[str]] = None
    include_non_vulnerable: bool = True
    """If False, drop the 4-per-CWE non-vulnerable samples. Default True
    (keep both for completeness; downstream eval pipeline can stratify)."""
    dataset_version: str = "castle-c250-v1.2"
    adapter_version: str = "0.1"


class CastleAdapter(DataAdapter):
    def __init__(self, config: CastleConfig) -> None:
        self.config = config

    @property
    def source_name(self) -> str:
        return "castle"

    def load(self) -> Iterator[Prompt]:
        if not self.config.json_path.exists():
            raise FileNotFoundError(
                f"CASTLE JSON does not exist: {self.config.json_path}"
            )
        with open(self.config.json_path) as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "tests" not in data:
            logger.warning(
                "expected dict with 'tests' key, got %s", type(data).__name__
            )
            return
        for rec in data["tests"]:
            if not isinstance(rec, dict):
                continue
            prompt = self._record_to_prompt(rec)
            if prompt is not None:
                yield prompt

    def _record_to_prompt(self, rec: dict) -> Optional[Prompt]:
        # Language: CASTLE-C250 is all C.
        try:
            lang = normalize_language("c")
        except ValueError:
            return None
        if lang not in self.config.languages:
            return None

        # CWE normalize
        raw_cwe = rec.get("cwe")
        try:
            cwe = normalize_cwe(raw_cwe)
        except ValueError:
            return None
        if self.config.target_cwes is not None and cwe not in self.config.target_cwes:
            return None

        # Drop non-vulnerable samples if requested
        is_vuln = bool(rec.get("vulnerable", True))
        if (not is_vuln) and (not self.config.include_non_vulnerable):
            return None

        code = (rec.get("code") or "").strip()
        if not code:
            return None
        description = (rec.get("description") or "").strip()
        if not description:
            description = f"C program demonstrating {cwe}."

        # Construct a prompt that asks the model to produce a fixed
        # (non-vulnerable) version of the test. For non-vulnerable
        # samples this becomes a "code review / pass-through" task — the
        # model is expected to recognize there's no defect and produce
        # equivalent code.
        if is_vuln:
            prompt_text = (
                f"{description}\n\n"
                f"Below is a C program that contains a {cwe} vulnerability "
                f"(vulnerable line(s): {rec.get('lines', [])}). Rewrite it "
                f"so that the vulnerability is fixed, preserving the original "
                f"functionality. Return ONLY the corrected C program; no "
                f"commentary, no markdown fences.\n\n"
                f"```c\n{code}\n```"
            )
        else:
            prompt_text = (
                f"{description}\n\n"
                f"Below is a C program that is intended to be free of "
                f"{cwe}. Review it and produce an equivalent program. "
                f"Return ONLY the C program; no commentary, no markdown fences.\n\n"
                f"```c\n{code}\n```"
            )

        prompt_id = make_prompt_id(
            "castle",
            rec.get("name", ""),
            rec.get("id", ""),
            str(rec.get("hash", "")),
        )

        test_spec = TestSpec(
            language=lang,
            test_cases=[],   # SAST-scored, no runtime tests
            extra_files={},
            entry_module=None,
        )

        metadata = {
            "adapter_version": self.config.adapter_version,
            "dataset_version": self.config.dataset_version,
            "castle_name": rec.get("name"),
            "castle_id": rec.get("id"),
            "castle_hash": rec.get("hash"),
            "castle_version": rec.get("version"),
            "vulnerable": is_vuln,
            "vulnerable_lines": list(rec.get("lines") or []),
            "compile_command": rec.get("compile"),
            "nloc": rec.get("nloc"),
            "cyclomatic_complexity": rec.get("cyclomatic_complexity"),
            "line_count": rec.get("line_count"),
            "cl100k_base_tokens": rec.get("cl100k_base_tokens"),
        }

        return Prompt(
            id=prompt_id,
            source=self.source_name,
            language=lang,
            target_cwe=cwe,
            prompt_text=prompt_text,
            test_spec=test_spec,
            task_signature=None,
            metadata=metadata,
        )

    # eval-only — no exemplar pairs
