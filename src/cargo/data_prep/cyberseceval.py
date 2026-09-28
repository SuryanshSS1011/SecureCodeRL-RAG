"""CyberSecEval adapter (eval-only).

Reads Meta PurpleLlama's CybersecurityBenchmarks instruct + autocomplete
datasets. Two source files share the same prompt pool (autocomplete is the
prefix-style variant of instruct); we deduplicate by `prompt_id` across both
plus the instruct-v2.json patch.

Schema (each JSON file is a list of records):
    {
        "prompt_id": 0,
        "cwe_identifier": "CWE-680",
        "language": "c",                # c | cpp | python | java | rust | php | ...
        "test_case_prompt": "Write a function in C that ...",
        "origin_code": "...",            # the vulnerable function from upstream
        "file_path": "path/in/origin/repo.c",
        "repo": "upstream/repo",
        "variant": "instruct" | "autocomplete",
        "pattern_desc": "...",
        "pattern_id": "C-W-004",
        "analyzer": "weggli",
        ...
    }

We treat CyberSecEval as eval-only (per the disjointness audit). The
upstream `origin_code` is real CVE-derived code, so cross-source overlap
with CVEfixes is checked separately (see scripts/audit_cyberseceval_overlap.py).
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
class CyberSecEvalConfig:
    """Paths to the three JSON files shipped by PurpleLlama.

    All three are optional individually; pass only those that exist on
    disk. We dedupe across `(file_basename, prompt_id)` so the same
    prompt isn't yielded twice from instruct.json and autocomplete.json.
    """

    instruct_json: Optional[Path] = None
    instruct_v2_json: Optional[Path] = None
    autocomplete_json: Optional[Path] = None
    languages: frozenset[Language] = field(
        default_factory=lambda: frozenset(
            {Language.PYTHON, Language.C, Language.CPP}
        )
    )
    target_cwes: Optional[frozenset[str]] = None
    dataset_version: str = "cyberseceval-purplellama-2024"
    adapter_version: str = "0.1"


class CyberSecEvalAdapter(DataAdapter):
    def __init__(self, config: CyberSecEvalConfig) -> None:
        self.config = config

    @property
    def source_name(self) -> str:
        return "cyberseceval"

    def load(self) -> Iterator[Prompt]:
        # Dedup across the three source files by content (origin_code, cwe,
        # language). The instruct + autocomplete files share `prompt_id`s
        # for the same upstream record (different rendering style); the
        # instruct-v2 file overlaps with instruct by `prompt_id` for
        # ~60-70% of records. Deduping by content is the safe path that
        # works regardless of how the upstream files share ids.
        seen_content: set[tuple[str, str, str]] = set()
        for label, path in (
            ("instruct", self.config.instruct_json),
            ("instruct-v2", self.config.instruct_v2_json),
            ("autocomplete", self.config.autocomplete_json),
        ):
            if path is None:
                continue
            if not path.exists():
                logger.warning("CyberSecEval source not found: %s", path)
                continue
            with open(path) as fh:
                try:
                    data = json.load(fh)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "skipping malformed JSON in %s (%s)", path, exc
                    )
                    continue
            if not isinstance(data, list):
                logger.warning(
                    "expected JSON array in %s, got %s", path, type(data).__name__
                )
                continue
            for rec in data:
                if not isinstance(rec, dict):
                    continue
                content_key = (
                    rec.get("origin_code", "") or "",
                    rec.get("cwe_identifier", "") or "",
                    rec.get("language", "") or "",
                )
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)
                prompt = self._record_to_prompt(rec, label)
                if prompt is not None:
                    yield prompt

    def _record_to_prompt(self, rec: dict, source_label: str) -> Optional[Prompt]:
        # Language filter
        raw_lang = rec.get("language", "")
        try:
            lang = normalize_language(raw_lang)
        except ValueError:
            return None
        if lang not in self.config.languages:
            return None

        # CWE filter
        raw_cwe = rec.get("cwe_identifier", "")
        try:
            cwe = normalize_cwe(raw_cwe)
        except ValueError:
            return None
        if self.config.target_cwes is not None and cwe not in self.config.target_cwes:
            return None

        prompt_text = (rec.get("test_case_prompt") or "").strip()
        if not prompt_text:
            return None

        # CyberSecEval prompts include the trailing
        #   "Only return the code, don't include any other information,
        #    such as a preamble or suffix."
        # which is the prompt's instruction-following discipline. Keep it
        # verbatim so retrieval and scoring align with how the upstream
        # CyberSecEval evaluation does it.

        pid_int = int(rec.get("prompt_id", 0))
        prompt_id = make_prompt_id(
            "cyberseceval",
            source_label,
            str(pid_int),
            rec.get("language", ""),
            rec.get("cwe_identifier", ""),
        )

        # CyberSecEval is a *generation* benchmark, not a dynamic-oracle
        # benchmark. Their evaluation pipeline runs the generated code
        # through ICD (Insecure Code Detector — a weggli/semgrep rule
        # bundle). We do NOT include a runtime test_spec here because the
        # right scoring path is our SAST cascade, not pytest.
        #
        # However, model completions to CyberSecEval prompts are function-
        # bodies (the prompt asks for "a function in C/C++"), so compiling
        # them as standalone .c/.cpp files fails to link (no includes, no
        # main). The oracle previously reported Compile@1 = 0 across the
        # board. We don't have a clean prefix/suffix from CyberSecEval
        # (origin_code is the reference function, not its TU), so we wrap
        # the completion with minimal default includes plus an empty main.
        # This won't fix every prompt — e.g., snippets that need <netdb.h>
        # or POSIX-specific headers will still fail — but it lifts the
        # floor materially. Metadata flag splice_mode='heuristic_wrapper'
        # so the paper can footnote this honestly.
        prefix_text: Optional[str] = None
        suffix_text: Optional[str] = None
        if lang == Language.C:
            prefix_text = (
                "#include <stdio.h>\n"
                "#include <string.h>\n"
                "#include <stdlib.h>\n"
            )
            suffix_text = "\nint main(){ return 0; }\n"
        elif lang == Language.CPP:
            prefix_text = (
                "#include <iostream>\n"
                "#include <string>\n"
                "#include <vector>\n"
                "#include <cstring>\n"
            )
            suffix_text = "\nint main(){ return 0; }\n"

        test_spec = TestSpec(
            language=lang,
            test_cases=[],  # no dynamic tests — SAST scoring only
            extra_files={},
            entry_module=None,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
        )

        metadata = {
            "adapter_version": self.config.adapter_version,
            "dataset_version": self.config.dataset_version,
            "source_file": source_label,
            "upstream_prompt_id": pid_int,
            "upstream_file_path": rec.get("file_path"),
            "upstream_repo": rec.get("repo"),
            "pattern_id": rec.get("pattern_id"),
            "pattern_desc": rec.get("pattern_desc"),
            "analyzer": rec.get("analyzer"),
            "variant": rec.get("variant", source_label),
            "origin_code_present": bool(rec.get("origin_code")),
        }
        if prefix_text is not None or suffix_text is not None:
            # CyberSecEval ships no clean prefix/suffix; we wrap the model
            # completion with minimal default includes + empty main so
            # body-only completions can compile. This is a heuristic that
            # won't help every prompt (e.g., snippets needing <netdb.h>),
            # but it lifts Compile@1 above 0. The paper footnotes this.
            metadata["splice_mode"] = "heuristic_wrapper"

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
