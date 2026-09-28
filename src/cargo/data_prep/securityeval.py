"""SecurityEval adapter (eval-only).

SecurityEval (Siddiq and Santos, MSR4P&S 2022, doi:10.1145/3549035.3561184).
Repository: https://github.com/s2e-lab/SecurityEval.

The dataset is 121 hand-curated Python code-completion prompts derived from
security analyzer documentation and security-expert authorship. Each
prompt is a partial Python module (imports + signature + docstring) where
the model must complete the function body. The dataset's `Insecure_code`
field is the canonical insecure completion (used by the upstream authors
to ground SAST evaluation); we expose it as `insecure_code_reference` in
metadata for downstream training-time RAG negatives, NOT as the prompt or
the test target.

Schema (dataset.jsonl, one JSON record per line):
    {
        "ID": "CWE-020_author_1.py",          # CWE encoded in ID prefix
        "Prompt": "import yaml\\n\\n\\ndef yaml_load(filename):\\n    '''...'''",
        "Insecure_code": "    with open(filename) as f:\\n        return yaml.load(f)"
    }

CWE normalization: SecurityEval uses zero-padded CWE numbers ("CWE-020")
which we normalize to "CWE-20" via the project's existing normalize_cwe.

Languages: Python only (the dataset is Python-exclusive). The adapter's
language filter enforces this.

Scoring: SecurityEval is a *generation* benchmark with a static-rule
oracle (SAST tools run on the completion). We do NOT include runtime
test_cases here — the scoring path is our SAST cascade, the same as the
CyberSecEval and CASTLE eval pipelines.
"""

from __future__ import annotations

import json
import logging
import re
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


_ID_CWE_RE = re.compile(r"^(CWE-\d+)_")


@dataclass
class SecurityEvalConfig:
    jsonl_path: Path
    languages: frozenset[Language] = field(
        default_factory=lambda: frozenset({Language.PYTHON})
    )
    target_cwes: Optional[frozenset[str]] = None
    include_insecure_reference_in_metadata: bool = True
    dataset_version: str = "securityeval-msr4ps22"
    adapter_version: str = "0.1"


class SecurityEvalAdapter(DataAdapter):
    def __init__(self, config: SecurityEvalConfig) -> None:
        self.config = config

    @property
    def source_name(self) -> str:
        return "securityeval"

    def load(self) -> Iterator[Prompt]:
        if not self.config.jsonl_path.exists():
            raise FileNotFoundError(
                f"SecurityEval JSONL does not exist: {self.config.jsonl_path}"
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
        try:
            lang = normalize_language("python")
        except ValueError:
            return None
        if lang not in self.config.languages:
            return None

        rec_id = rec.get("ID", "")
        m = _ID_CWE_RE.match(rec_id)
        if not m:
            return None
        # SecurityEval uses zero-padded numbers ("CWE-020"); strip the pad
        # because the project's normalize_cwe preserves the leading zero
        # and would mismatch the target CWE set ("CWE-20").
        cwe_raw = m.group(1)
        try:
            cwe_num = int(cwe_raw.split("-", 1)[1])
        except (ValueError, IndexError):
            return None
        try:
            cwe = normalize_cwe(f"CWE-{cwe_num}")
        except ValueError:
            return None
        if self.config.target_cwes is not None and cwe not in self.config.target_cwes:
            return None

        prompt_text = (rec.get("Prompt") or "").strip()
        if not prompt_text:
            return None

        prompt_id = make_prompt_id(
            "securityeval", rec_id, cwe, "python",
        )

        test_spec = TestSpec(
            language=lang,
            test_cases=[],
            extra_files={},
            entry_module=None,
        )

        metadata = {
            "adapter_version": self.config.adapter_version,
            "dataset_version": self.config.dataset_version,
            "securityeval_id": rec_id,
        }
        if self.config.include_insecure_reference_in_metadata:
            insecure = (rec.get("Insecure_code") or "").strip()
            if insecure:
                metadata["insecure_code_reference"] = insecure

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
