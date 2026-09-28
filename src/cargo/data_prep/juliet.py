"""Juliet / NIST SARD C/C++ Test Suite v1.3 adapter.

Real Juliet ships as:

    juliet/C/testcases/CWE<NUM>_<desc>/[s<NN>/]CWE<NUM>__*_NN.c
    juliet/Cpp/testcases/CWE<NUM>_<desc>/[s<NN>/]CWE<NUM>__*_NN.cpp

The CWE id is in the parent directory name. Each .c/.cpp file contains:
  - A `<filename>_bad()` function under `#ifndef OMITBAD`.
  - A `goodG2B()` or `goodB2G()` static function under `#ifndef OMITGOOD`.

We extract both function bodies and emit an ExemplarPair per file:
  - e_neg = bad function body
  - e_pos = good function body

Juliet is index-only (synthetic test cases, not real prompts). `load()`
yields nothing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from ..rag.schema import ExemplarPair
from ..rag.schema import Language as RagLanguage
from .schema import DataAdapter, Language, Prompt, normalize_cwe

logger = logging.getLogger(__name__)


_DIR_LANGUAGE: dict[str, Language] = {
    "C": Language.C,
    "Cpp": Language.CPP,
}
_LANG_EXTENSIONS: dict[Language, tuple[str, ...]] = {
    Language.C: (".c",),
    Language.CPP: (".cpp", ".cxx", ".cc"),
}


# Parent directory matches CWE<NUM>_<desc>, e.g. CWE121_Stack_Based_Buffer_Overflow.
_CWE_DIR_RE = re.compile(r"^CWE(?P<cwe>\d+)_")

# Function signatures we look for. Juliet conventions:
#   void <prefix>_bad()
#   static void goodG2B()  or  static void goodB2G()
# `<prefix>` is the file's base name; we don't try to match it.
_BAD_FN_RE = re.compile(
    r"\bvoid\s+\w*?_?bad\s*\([^)]*\)\s*\{", re.MULTILINE
)
_GOOD_FN_RE = re.compile(
    r"\bstatic\s+void\s+(good(?:G2B|B2G))\s*\([^)]*\)\s*\{", re.MULTILINE
)


@dataclass
class JulietConfig:
    juliet_root: Path
    languages: frozenset[Language] = field(
        default_factory=lambda: frozenset({Language.C, Language.CPP})
    )
    target_cwes: Optional[frozenset[str]] = None
    dataset_version: str = "juliet-1.3"
    adapter_version: str = "0.2"  # bumped after real on-disk-format rework


class JulietAdapter(DataAdapter):
    def __init__(self, config: JulietConfig) -> None:
        self.config = config

    @property
    def source_name(self) -> str:
        return "juliet"

    def load(self) -> Iterator[Prompt]:
        """Juliet is index-only; never emits Prompts."""
        return iter(())

    def load_exemplar_pairs(self) -> Iterator[ExemplarPair]:
        if not self.config.juliet_root.exists():
            raise FileNotFoundError(
                f"Juliet root does not exist: {self.config.juliet_root}"
            )

        for lang_dir_name, language in _DIR_LANGUAGE.items():
            if language not in self.config.languages:
                continue
            lang_root = self.config.juliet_root / lang_dir_name / "testcases"
            if not lang_root.exists():
                continue

            extensions = _LANG_EXTENSIONS[language]

            for cwe_dir in sorted(lang_root.iterdir()):
                if not cwe_dir.is_dir():
                    continue
                m = _CWE_DIR_RE.match(cwe_dir.name)
                if m is None:
                    continue
                cwe = normalize_cwe(str(int(m.group("cwe"))))
                if (
                    self.config.target_cwes is not None
                    and cwe not in self.config.target_cwes
                ):
                    continue

                # Walk recursively to pick up subshards (s03, s09, ...).
                for source_file in sorted(cwe_dir.rglob("*")):
                    if not source_file.is_file():
                        continue
                    if source_file.suffix.lower() not in extensions:
                        continue
                    pair = self._parse_file(
                        source_file, cwe=cwe, language=language,
                        cwe_dir_name=cwe_dir.name,
                    )
                    if pair is not None:
                        yield pair

    def _parse_file(
        self, path: Path, cwe: str, language: Language, cwe_dir_name: str,
    ) -> Optional[ExemplarPair]:
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            logger.warning("failed to read %s: %s", path, exc)
            return None

        bad_body = _extract_function_body(text, _BAD_FN_RE)
        good_body = _extract_function_body(text, _GOOD_FN_RE)

        if bad_body is None or good_body is None:
            return None
        if not bad_body.strip() or not good_body.strip():
            return None

        return ExemplarPair(
            cwe=cwe,
            task_signature=_extract_signature_hint(text, language),
            e_pos=good_body,
            e_neg=bad_body,
            language=_rag_language(language),
            source=self.source_name,
            metadata={
                "synthetic": True,
                "case_file": path.name,
                "cwe_dir": cwe_dir_name,
                "dataset_version": self.config.dataset_version,
                "adapter_version": self.config.adapter_version,
            },
        )


def _extract_function_body(text: str, signature_re: re.Pattern) -> Optional[str]:
    """Find the first match of `signature_re` (which ends at the opening
    brace of a function body) and return the body up to the matching
    closing brace. None if no match or brace matching fails.
    """
    m = signature_re.search(text)
    if m is None:
        return None
    # Body starts after the `{` of the function header.
    start = m.end() - 1  # index of the `{`
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                # Body is text between the opening { and this closing }.
                return text[start + 1 : i].strip("\n")
        i += 1
    return None


def _extract_signature_hint(text: str, language: Language) -> str:
    """A short hint for the retriever's BM25 key. Use the first non-blank
    non-comment line that contains an identifier — typically a function
    declaration or definition."""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("/*") or s.startswith("*") or s.startswith("//"):
            continue
        if s.startswith("#"):
            continue
        if "(" in s:
            return s.rstrip("{").strip()
    return ""


def _rag_language(lang: Language) -> RagLanguage:
    return {
        Language.PYTHON: RagLanguage.PYTHON,
        Language.C: RagLanguage.C,
        Language.CPP: RagLanguage.CPP,
    }[lang]
