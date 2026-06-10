"""DiverseVul adapter (training-side).

DiverseVul (Chen, Ding, Alowain, X. Chen, Wagner; RAID 2023,
arXiv:2304.00409). Repository: https://github.com/wagner-group/diversevul.
HuggingFace mirror: https://huggingface.co/datasets/claudios/DiverseVul.

The dataset is **18,945 vulnerable C/C++ functions** plus 311,547
non-vulnerable functions across 7,514 commits and ~700+ projects.
Vulnerable functions are labeled with one or more CWEs (16,109 of 18,945
have non-empty CWE labels; ~85% coverage of vulnerable functions).
Non-vulnerable functions are not used here.

For our training pipeline DiverseVul plays a different role than CVEfixes:
CVEfixes ships pre-fix AND post-fix pairs, so it emits ExemplarPair
(e_pos=patched, e_neg=vulnerable) for the RAG index. DiverseVul ships only
the vulnerable function source — there is no companion patched function
in the dataset (only commit IDs we'd have to scrape upstream to resolve).

Therefore DiverseVul is treated as **Prompt-only training data**: each
vulnerable function becomes a training prompt where the task is "given
this signature and context, produce a secure implementation," and the
original vulnerable function lives in `metadata.insecure_code_reference`
for downstream RAG-negative use. The training reward path is the SAST
cascade on the model's completion (same as CyberSecEval / CASTLE eval).

Cross-source duplication concern: DiverseVul partially overlaps CVEfixes
upstream (both pull from public CVE-tagged commits). The intra-train
near-duplicate dedup (`scripts/build_v0_1_5.py`) runs after
DiverseVul ingestion and catches functional duplicates by content hash,
within each (CWE, language) cell.

Schema (HF version, `claudios/DiverseVul` split=`test`):
    {
        "func": "int foo(...) { ... }",       # vulnerable function source
        "target": 1,                          # 1 = vulnerable, 0 = not
        "cwe": ["CWE-787", "CWE-119"],        # one or more CWE labels
        "project": "linux",
        "commit_id": "abc123...",
        "hash": <float>,                      # content hash (not for our use)
        "size": <int>,                        # function size in some unit
        "message": "commit message ..."
    }

Language detection: DiverseVul doesn't ship an explicit language field.
We infer from the function source: 'cpp' if class/namespace/template/std::
or scope-resolution syntax is present, 'c' otherwise.

Multi-CWE handling: a single function may have multiple CWE labels. We
emit one Prompt per (function × in-scope CWE) pair so the per-CWE quota
machinery can balance independently.
"""

from __future__ import annotations

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


_CPP_HINTS = (
    re.compile(r"\bclass\s+\w+"),
    re.compile(r"\bnamespace\s+\w+"),
    re.compile(r"\btemplate\s*<"),
    re.compile(r"::"),
    re.compile(r"\bstd::"),
)


def _infer_cpp_vs_c(func_text: str) -> str:
    for pat in _CPP_HINTS:
        if pat.search(func_text):
            return "cpp"
    return "c"


_FN_SIG_RE = re.compile(
    r"^(?P<sig>[\w\*&\s:<>,]+?[\w\*&]\s+\**\w+\s*\([^)]*\))",
    re.MULTILINE,
)


def _extract_signature(func_text: str) -> Optional[str]:
    head = func_text[:1024]
    m = _FN_SIG_RE.match(head.lstrip())
    if m:
        return m.group("sig").strip()
    return None


def _cwe_short_description(cwe: str) -> str:
    descs = {
        "CWE-787": "out-of-bounds write",
        "CWE-119": "memory boundary error",
        "CWE-125": "out-of-bounds read",
        "CWE-416": "use after free",
        "CWE-476": "NULL pointer dereference",
        "CWE-190": "integer overflow",
        "CWE-20": "improper input validation",
        "CWE-89": "SQL injection",
        "CWE-78": "OS command injection",
        "CWE-79": "cross-site scripting",
        "CWE-94": "code injection",
        "CWE-327": "broken or risky cryptographic algorithm",
        "CWE-328": "weak hash",
        "CWE-326": "inadequate cryptographic strength",
        "CWE-798": "use of hard-coded credentials",
        "CWE-502": "deserialization of untrusted data",
        "CWE-22": "path traversal",
        "CWE-306": "missing authentication for critical function",
        "CWE-862": "missing authorization",
    }
    return descs.get(cwe, "vulnerability")


@dataclass
class DiverseVulConfig:
    """Configuration for the DiverseVul adapter.

    Either `jsonl_path` (a local .jsonl export, one record per line) OR
    the HuggingFace mirror is used. Local file takes precedence.
    """

    jsonl_path: Optional[Path] = None
    hf_dataset_id: str = "claudios/DiverseVul"
    hf_split: str = "test"
    languages: frozenset[Language] = field(
        default_factory=lambda: frozenset({Language.C, Language.CPP})
    )
    target_cwes: Optional[frozenset[str]] = None
    min_function_chars: int = 64
    max_function_chars: int = 8192
    dataset_version: str = "diversevul-raid23"
    adapter_version: str = "0.1"


class DiverseVulAdapter(DataAdapter):
    def __init__(self, config: DiverseVulConfig) -> None:
        self.config = config

    @property
    def source_name(self) -> str:
        return "diversevul"

    def _iter_records(self) -> Iterator[dict]:
        if self.config.jsonl_path is not None:
            path = self.config.jsonl_path
            if not path.exists():
                raise FileNotFoundError(f"DiverseVul JSONL does not exist: {path}")
            import json as _json
            with open(path) as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield _json.loads(line)
                    except _json.JSONDecodeError as exc:
                        logger.warning(
                            "skipping malformed JSON in %s at line %d (%s)",
                            path, lineno, exc,
                        )
                        continue
            return

        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "DiverseVul HF mode requires `datasets` package; "
                "install via pip or set `jsonl_path` for offline loading."
            ) from e

        logger.info(
            "loading DiverseVul from HF: %s split=%s",
            self.config.hf_dataset_id, self.config.hf_split,
        )
        ds = load_dataset(self.config.hf_dataset_id, split=self.config.hf_split)
        for rec in ds:
            yield rec

    def load(self) -> Iterator[Prompt]:
        n_total = 0
        n_emitted = 0
        for rec in self._iter_records():
            n_total += 1
            for prompt in self._record_to_prompts(rec):
                n_emitted += 1
                yield prompt
        logger.info(
            "DiverseVul: scanned %d records, emitted %d in-scope prompts",
            n_total, n_emitted,
        )

    def _record_to_prompts(self, rec: dict) -> Iterator[Prompt]:
        if rec.get("target") != 1:
            return
        cwes_raw = rec.get("cwe") or []
        if not cwes_raw:
            return

        func = (rec.get("func") or "").strip()
        if not func:
            return
        if len(func) < self.config.min_function_chars:
            return
        if len(func) > self.config.max_function_chars:
            return

        lang_str = _infer_cpp_vs_c(func)
        try:
            lang = normalize_language(lang_str)
        except ValueError:
            return
        if lang not in self.config.languages:
            return

        signature = _extract_signature(func)
        project = (rec.get("project") or "").strip()
        commit_id = (rec.get("commit_id") or "").strip()[:12]

        for raw_cwe in cwes_raw:
            try:
                cwe = normalize_cwe(raw_cwe)
            except ValueError:
                continue
            if self.config.target_cwes is not None and cwe not in self.config.target_cwes:
                continue

            project_clause = f" Project: {project}." if project else ""
            sig_clause = f" Signature: {signature}." if signature else ""
            prompt_text = (
                f"Implement the following {lang_str.upper()} function such "
                f"that it is free of {cwe} ({_cwe_short_description(cwe)}). "
                f"Return ONLY the complete function definition; no commentary, "
                f"no markdown fences.{project_clause}{sig_clause}"
            )

            prompt_id = make_prompt_id(
                "diversevul",
                commit_id, project, cwe, lang_str,
                signature or func[:64],
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
                "project": project,
                "commit_id": commit_id,
                "size_field": rec.get("size"),
                "commit_message_head": (rec.get("message") or "")[:200],
                "cwes_all_labels": list(cwes_raw),
                "insecure_code_reference": func,
            }

            yield Prompt(
                id=prompt_id,
                source=self.source_name,
                language=lang,
                target_cwe=cwe,
                prompt_text=prompt_text,
                test_spec=test_spec,
                task_signature=signature,
                metadata=metadata,
            )
