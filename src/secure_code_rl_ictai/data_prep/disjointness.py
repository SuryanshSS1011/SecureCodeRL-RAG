"""Disjointness audit: drop train/index blobs equivalent to eval blobs.

Implements docs/dataset_build_spec.md §3. Two-pass:

    1. String normalization (cheap; comments stripped, whitespace collapsed,
       lowercased). Hash-compared.
    2. AST normalization (Python only in v0.1; locals renamed to v0/v1/...).
       Hash-compared.

A train-side blob is dropped iff it is string-equivalent OR AST-equivalent
to any eval blob.

The audit is *additive*: callers add every eval blob first, then audit each
train-side blob in turn. No ordering coupling between sources.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---- normalization helpers ----


_PY_COMMENT_RE = re.compile(r"#[^\n]*")
_C_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_C_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def string_normalize(text: str) -> str:
    """Strip comments, collapse whitespace, lowercase. Used by every blob.

    Strips both Python (`#`) and C-style (`//`, `/* */`) comments because
    we don't know the language at this layer. False positives (e.g., `#`
    inside a string literal) would matter for code execution but are
    inert for hash matching: as long as we apply the same transform to
    both sides, the comparison is consistent.
    """
    t = _C_BLOCK_COMMENT_RE.sub(" ", text)
    t = _C_LINE_COMMENT_RE.sub("", t)
    t = _PY_COMMENT_RE.sub("", t)
    t = _WHITESPACE_RE.sub(" ", t)
    return t.strip().lower()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ast_normalize_python(text: str) -> Optional[str]:
    """Parse, rename locals to canonical placeholders, re-serialize.

    Returns None if parsing fails. The caller (DisjointnessAudit) falls
    back to string normalization when this returns None.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    name_map: dict[str, str] = {}

    def canonical(name: str) -> str:
        if name not in name_map:
            name_map[name] = f"v{len(name_map)}"
        return name_map[name]

    class Renamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.name = canonical(node.name)
            for arg in node.args.args:
                arg.arg = canonical(arg.arg)
            self.generic_visit(node)
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            return self.visit_FunctionDef(node)  # type: ignore[arg-type]

        def visit_Name(self, node: ast.Name) -> ast.AST:
            node.id = canonical(node.id)
            return node

        def visit_arg(self, node: ast.arg) -> ast.AST:
            node.arg = canonical(node.arg)
            return node

    Renamer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


# ---- audit ----


@dataclass
class DisjointnessReport:
    eval_keys: int = 0
    dropped_per_source: dict[str, int] = field(default_factory=dict)
    string_only_drops: int = 0
    ast_only_drops: int = 0
    both_drops: int = 0


class DisjointnessAudit:
    """Tracks eval-side keys and audits train-side blobs against them.

    Usage:
        audit = DisjointnessAudit()
        for blob in eval_blobs:
            audit.add_eval_blob(blob)
        for blob, src in train_blobs:
            audit.audit_train_blob(blob, source=src)
        report = audit.report()
    """

    def __init__(self) -> None:
        self._eval_string_keys: set[str] = set()
        self._eval_ast_keys: set[str] = set()
        self._eval_count = 0
        self._dropped_per_source: dict[str, int] = defaultdict(int)
        self._string_only = 0
        self._ast_only = 0
        self._both = 0

    def add_eval_blob(
        self, blob: str, *, language_hint: Optional[str] = None
    ) -> None:
        self._eval_count += 1
        self._eval_string_keys.add(_hash(string_normalize(blob)))
        if language_hint is None or language_hint.lower() in ("python", "py"):
            ast_form = ast_normalize_python(blob)
            if ast_form is not None:
                self._eval_ast_keys.add(_hash(ast_form))

    def audit_train_blob(
        self,
        blob: str,
        *,
        source: str,
        language_hint: Optional[str] = None,
    ) -> bool:
        """Audit one train-side blob. Returns True iff dropped.

        Updates per-source drop counts and the string-only / ast-only /
        both breakdown so the report can identify which normalization
        pass is contributing.
        """
        str_match = _hash(string_normalize(blob)) in self._eval_string_keys
        ast_match = False
        if language_hint is None or language_hint.lower() in ("python", "py"):
            ast_form = ast_normalize_python(blob)
            if ast_form is not None:
                ast_match = _hash(ast_form) in self._eval_ast_keys

        if not str_match and not ast_match:
            return False

        self._dropped_per_source[source] += 1
        if str_match and ast_match:
            self._both += 1
        elif str_match:
            self._string_only += 1
        else:
            self._ast_only += 1
        return True

    def report(self) -> DisjointnessReport:
        return DisjointnessReport(
            eval_keys=self._eval_count,
            dropped_per_source=dict(self._dropped_per_source),
            string_only_drops=self._string_only,
            ast_only_drops=self._ast_only,
            both_drops=self._both,
        )
