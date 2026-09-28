"""Tests for the disjointness audit.

Pins the spec's two-pass normalization:
  - String normalization (comments stripped, whitespace collapsed, lowercased).
  - AST normalization (Python: ast.dump after renaming locals to v0/v1/...).

The audit drops any train-side blob string- OR AST-equivalent to an eval blob.
"""

from __future__ import annotations


from cargo.data_prep.disjointness import (
    DisjointnessAudit,
    ast_normalize_python,
    string_normalize,
)


# ----------------------------------------------------------------------
# String normalization
# ----------------------------------------------------------------------


def test_string_normalize_strips_python_comments():
    a = "def f():\n    # a comment\n    return 1\n"
    b = "def f():\n    return 1\n"
    assert string_normalize(a) == string_normalize(b)


def test_string_normalize_strips_c_line_comments():
    a = "int main() {\n    // line comment\n    return 0;\n}"
    b = "int main() { return 0; }"
    assert string_normalize(a) == string_normalize(b)


def test_string_normalize_strips_c_block_comments():
    a = "int main() {\n    /* block\n    comment */\n    return 0;\n}"
    b = "int main() { return 0; }"
    assert string_normalize(a) == string_normalize(b)


def test_string_normalize_collapses_whitespace():
    a = "def    f():\treturn   1"
    b = "def f(): return 1"
    assert string_normalize(a) == string_normalize(b)


def test_string_normalize_is_case_insensitive():
    a = "def F(): return 1"
    b = "def f(): return 1"
    assert string_normalize(a) == string_normalize(b)


def test_string_normalize_distinguishes_different_code():
    a = string_normalize("return 1")
    b = string_normalize("return 2")
    assert a != b


# ----------------------------------------------------------------------
# AST normalization (Python)
# ----------------------------------------------------------------------


def test_ast_normalize_python_treats_renamed_locals_as_equal():
    a = "def f(x):\n    y = x + 1\n    return y\n"
    b = "def g(a):\n    b = a + 1\n    return b\n"
    assert ast_normalize_python(a) == ast_normalize_python(b)


def test_ast_normalize_python_distinguishes_different_structure():
    a = "def f(x):\n    return x + 1\n"
    b = "def f(x):\n    return x - 1\n"
    assert ast_normalize_python(a) != ast_normalize_python(b)


def test_ast_normalize_python_returns_none_on_syntax_error():
    # Caller (DisjointnessAudit) should fall back to string normalization
    # when AST parsing fails.
    assert ast_normalize_python("def f(:") is None


# ----------------------------------------------------------------------
# DisjointnessAudit: integration
# ----------------------------------------------------------------------


def test_audit_drops_string_equivalent_train_blob():
    audit = DisjointnessAudit()
    audit.add_eval_blob("def get_user(uid):\n    return db.find(uid)\n")
    drop = audit.audit_train_blob(source="test", blob="def get_user(uid):\n    # extra comment\n    return db.find(uid)\n")
    assert drop is True


def test_audit_drops_ast_equivalent_train_blob_python():
    audit = DisjointnessAudit()
    audit.add_eval_blob("def f(x):\n    y = x + 1\n    return y\n")
    drop = audit.audit_train_blob(source="test", blob="def g(a):\n    b = a + 1\n    return b\n")
    assert drop is True


def test_audit_keeps_distinct_train_blob():
    audit = DisjointnessAudit()
    audit.add_eval_blob("def get_user(uid):\n    return db.find(uid)\n")
    drop = audit.audit_train_blob(source="test", blob="def get_product(pid):\n    return db.lookup(pid)\n")
    assert drop is False


def test_audit_distinguishes_constant_changes():
    """Renaming a function or variable should NOT distinguish (renaming is
    normalized). But changing a constant SHOULD distinguish."""
    audit = DisjointnessAudit()
    audit.add_eval_blob("def f(x):\n    return x + 1\n")
    assert audit.audit_train_blob(source="test", blob="def f(x):\n    return x + 1\n") is True
    assert audit.audit_train_blob(source="test", blob="def f(x):\n    return x + 2\n") is False


def test_audit_report_counts_drops_per_source():
    audit = DisjointnessAudit()
    audit.add_eval_blob("def f(x): return x + 1")
    audit.add_eval_blob("def g(y): return y * 2")

    # Two train blobs equivalent to eval, one distinct.
    audit.audit_train_blob("def h(a): return a + 1", source="cvefixes")  # match by AST
    audit.audit_train_blob("def k(b): return b * 2", source="cvefixes")  # match by AST
    audit.audit_train_blob("def m(c): return c - 3", source="juliet")    # distinct

    report = audit.report()
    assert report.eval_keys == 2
    assert report.dropped_per_source["cvefixes"] == 2
    assert report.dropped_per_source.get("juliet", 0) == 0


def test_audit_handles_c_code_via_string_normalize_only():
    """For C code we don't AST-normalize in v0.1; string match must catch."""
    audit = DisjointnessAudit()
    audit.add_eval_blob(
        "int main() {\n    return 0;\n}",
        language_hint="c",
    )
    drop = audit.audit_train_blob(source="test", blob=
        "int main() {\n    // a comment\n    return 0;\n}",
        language_hint="c",
    )
    assert drop is True


def test_audit_recovers_from_python_syntax_error_on_eval_side():
    """Malformed eval blob: AST normalization returns None; string still works."""
    audit = DisjointnessAudit()
    audit.add_eval_blob("def f(:")  # syntax error; only string key indexed
    # Identical broken string -> still matches via string normalization.
    assert audit.audit_train_blob(source="test", blob="def f(:") is True
    # Different broken string -> no match.
    assert audit.audit_train_blob(source="test", blob="class X(") is False


def test_audit_does_not_match_substring_equivalents():
    """A train blob that contains the eval blob as a substring is not equivalent."""
    audit = DisjointnessAudit()
    audit.add_eval_blob("return 1")
    # Larger blob that contains "return 1" in its body shouldn't be flagged.
    assert audit.audit_train_blob(source="test", blob="def f():\n    return 1\nclass Y: pass\n") is False
