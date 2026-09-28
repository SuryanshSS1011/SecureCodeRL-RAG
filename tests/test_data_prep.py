"""Tests for the data-prep adapters.

Fixture data is written to tmp_path per test. Real corpus data is never
required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cargo.data_prep import (
    CvefixesAdapter,
    CvefixesConfig,
    Language,
    Prompt,
    SecCodePltAdapter,
    SecCodePltConfig,
    make_prompt_id,
    normalize_cwe,
    normalize_language,
)
from cargo.reward.reliability_oracle import TestSpec


# ----------------------------------------------------------------------
# Schema normalization
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CWE-89", "CWE-89"),
        ("89", "CWE-89"),
        ("cwe-89", "CWE-89"),
        ("CWE-787", "CWE-787"),
    ],
)
def test_normalize_cwe_accepts_common_forms(raw, expected):
    assert normalize_cwe(raw) == expected


@pytest.mark.parametrize("raw", ["not-a-cwe", "CWE-", "CWEabc", ""])
def test_normalize_cwe_rejects_garbage(raw):
    with pytest.raises(ValueError):
        normalize_cwe(raw)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("python", Language.PYTHON),
        ("Python", Language.PYTHON),
        ("PY", Language.PYTHON),
        ("c", Language.C),
        ("C++", Language.CPP),
        ("cpp", Language.CPP),
        ("cxx", Language.CPP),
    ],
)
def test_normalize_language_aliases(raw, expected):
    assert normalize_language(raw) == expected


def test_normalize_language_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_language("rust")


# ----------------------------------------------------------------------
# Prompt validation
# ----------------------------------------------------------------------


def test_prompt_rejects_empty_text():
    with pytest.raises(ValueError):
        Prompt(
            id="x:1",
            source="x",
            language=Language.PYTHON,
            target_cwe="CWE-89",
            prompt_text="   ",
            test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
        )


def test_prompt_normalizes_cwe_on_construction():
    p = Prompt(
        id="x:1",
        source="x",
        language=Language.PYTHON,
        target_cwe="89",  # accepted, normalized
        prompt_text="def f(): pass",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
    )
    assert p.target_cwe == "CWE-89"


def test_make_prompt_id_is_stable():
    a = make_prompt_id("src", "a", "b", "c")
    b = make_prompt_id("src", "a", "b", "c")
    c = make_prompt_id("src", "a", "b", "d")
    assert a == b
    assert a != c
    assert a.startswith("src:")


# ----------------------------------------------------------------------
# CVEfixes adapter
# ----------------------------------------------------------------------


def _write_cvefixes_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    d = tmp_path / "cvefixes_jsonl"
    d.mkdir()
    fh = d / "shard_000.jsonl"
    with open(fh, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return d


def test_cvefixes_adapter_emits_prompt_and_pair_for_clean_record(tmp_path: Path):
    d = _write_cvefixes_jsonl(
        tmp_path,
        [
            {
                "fix_commit": "abc123",
                "cve_id": "CVE-2023-12345",
                "project": "test-proj",
                "license": "BSD-3-Clause",
                "language": "Python",
                "cwe": "CWE-89",
                "file_path": "lib/orm.py",
                "function_name": "get_user",
                "signature": "def get_user(uid):",
                "pre_fix": "def get_user(uid):\n    return db.query('SELECT * FROM u WHERE id=' + uid)\n",
                "post_fix": "def get_user(uid):\n    return db.query('SELECT * FROM u WHERE id=?', [uid])\n",
            }
        ],
    )
    adapter = CvefixesAdapter(CvefixesConfig(jsonl_dir=d))
    prompts = list(adapter.load())
    pairs = list(adapter.load_exemplar_pairs())

    assert len(prompts) == 1
    p = prompts[0]
    assert p.source == "cvefixes"
    assert p.language == Language.PYTHON
    assert p.target_cwe == "CWE-89"
    assert p.task_signature == "def get_user(uid):"
    assert p.metadata["cve_id"] == "CVE-2023-12345"

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.cwe == "CWE-89"
    assert pair.e_neg.startswith("def get_user(uid):")
    assert "?" in pair.e_pos and "[uid]" in pair.e_pos
    assert pair.cve_id == "CVE-2023-12345"


def test_cvefixes_adapter_filters_by_language(tmp_path: Path):
    d = _write_cvefixes_jsonl(
        tmp_path,
        [
            _record(language="Python", cwe="CWE-89"),
            _record(language="JavaScript", cwe="CWE-89"),
        ],
    )
    adapter = CvefixesAdapter(
        CvefixesConfig(jsonl_dir=d, languages=frozenset({Language.PYTHON}))
    )
    prompts = list(adapter.load())
    assert len(prompts) == 1
    assert prompts[0].language == Language.PYTHON


def test_cvefixes_adapter_filters_by_target_cwe(tmp_path: Path):
    d = _write_cvefixes_jsonl(
        tmp_path,
        [
            _record(cwe="CWE-89"),
            _record(cwe="CWE-999"),
        ],
    )
    adapter = CvefixesAdapter(
        CvefixesConfig(jsonl_dir=d, target_cwes=frozenset({"CWE-89"}))
    )
    assert len(list(adapter.load())) == 1
    assert len(list(adapter.load_exemplar_pairs())) == 1


def test_cvefixes_adapter_drops_records_missing_required_fields(tmp_path: Path):
    d = _write_cvefixes_jsonl(
        tmp_path,
        [
            # missing pre_fix
            {
                "language": "Python", "cwe": "CWE-89",
                "function_name": "f", "signature": "def f():",
                "post_fix": "def f(): pass",
            },
        ],
    )
    adapter = CvefixesAdapter(CvefixesConfig(jsonl_dir=d))
    assert list(adapter.load()) == []
    assert list(adapter.load_exemplar_pairs()) == []


def test_cvefixes_adapter_deduplicates_identical_pairs(tmp_path: Path):
    rec = _record(cwe="CWE-89")
    d = _write_cvefixes_jsonl(tmp_path, [rec, rec])  # same fix appears twice
    adapter = CvefixesAdapter(CvefixesConfig(jsonl_dir=d))
    pairs = list(adapter.load_exemplar_pairs())
    assert len(pairs) == 1


def test_cvefixes_adapter_skips_no_op_fixes(tmp_path: Path):
    """A 'fix' where pre_fix == post_fix is not informative; drop the pair."""
    d = _write_cvefixes_jsonl(
        tmp_path,
        [
            _record(cwe="CWE-89", pre_fix="def f(): pass\n", post_fix="def f(): pass\n"),
        ],
    )
    adapter = CvefixesAdapter(CvefixesConfig(jsonl_dir=d))
    assert list(adapter.load_exemplar_pairs()) == []


def test_cvefixes_adapter_handles_malformed_lines(tmp_path: Path):
    d = tmp_path / "cvefixes_jsonl"
    d.mkdir()
    fh = d / "shard.jsonl"
    with open(fh, "w") as f:
        f.write("{not valid json\n")
        f.write(json.dumps(_record(cwe="CWE-89")) + "\n")
    adapter = CvefixesAdapter(CvefixesConfig(jsonl_dir=d))
    # The malformed line is skipped; the valid record still emits.
    assert len(list(adapter.load())) == 1


def test_cvefixes_adapter_raises_when_dir_missing(tmp_path: Path):
    adapter = CvefixesAdapter(CvefixesConfig(jsonl_dir=tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        list(adapter.load())


# ----------------------------------------------------------------------
# SecCodePLT adapter
# ----------------------------------------------------------------------


def _write_seccodeplt_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    fh = tmp_path / "seccodeplt.jsonl"
    with open(fh, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return fh


def test_seccodeplt_adapter_emits_prompt(tmp_path: Path):
    path = _write_seccodeplt_jsonl(
        tmp_path,
        [
            {
                "id": "seccodeplt-001",
                "category": "CWE-89",
                "language": "python",
                "task_description": "Write a function get_user(uid) that ...",
                "signature": "def get_user(uid):",
                "tests": [
                    {"stdin": "1\n", "expected_stdout": "alice\n", "timeout_s": 5.0},
                    {"stdin": "2\n", "expected_stdout": "bob\n"},
                ],
                "metadata": {"source_split": "test"},
            }
        ],
    )
    adapter = SecCodePltAdapter(SecCodePltConfig(jsonl_path=path))
    prompts = list(adapter.load())
    assert len(prompts) == 1
    p = prompts[0]
    assert p.source == "seccodeplt"
    assert p.target_cwe == "CWE-89"
    assert "Write a function" in p.prompt_text
    assert "def get_user(uid):" in p.prompt_text  # signature appended
    assert len(p.test_spec.test_cases) == 2
    assert p.test_spec.test_cases[0].expected_stdout == "alice\n"
    assert p.test_spec.test_cases[1].timeout_s == 5.0  # default applied
    assert p.metadata["source_split"] == "test"


def test_seccodeplt_adapter_filters_by_cwe(tmp_path: Path):
    path = _write_seccodeplt_jsonl(
        tmp_path,
        [
            _seccodeplt_record(cwe="CWE-89", record_id="a"),
            _seccodeplt_record(cwe="CWE-22", record_id="b"),
        ],
    )
    adapter = SecCodePltAdapter(
        SecCodePltConfig(jsonl_path=path, target_cwes=frozenset({"CWE-89"}))
    )
    prompts = list(adapter.load())
    assert len(prompts) == 1
    assert prompts[0].target_cwe == "CWE-89"


def test_seccodeplt_adapter_raises_not_implemented_for_pairs(tmp_path: Path):
    path = _write_seccodeplt_jsonl(tmp_path, [])
    adapter = SecCodePltAdapter(SecCodePltConfig(jsonl_path=path))
    with pytest.raises(NotImplementedError):
        list(adapter.load_exemplar_pairs())


def test_seccodeplt_adapter_skips_records_missing_text(tmp_path: Path):
    path = _write_seccodeplt_jsonl(
        tmp_path,
        [
            {"id": "x", "category": "CWE-89", "language": "python"}
            # missing both task_description and signature
        ],
    )
    adapter = SecCodePltAdapter(SecCodePltConfig(jsonl_path=path))
    assert list(adapter.load()) == []


# ----------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------


def _record(**overrides) -> dict:
    """Default CVEfixes record; override any field for a test."""
    base = {
        "fix_commit": "abc",
        "cve_id": "CVE-2023-99999",
        "project": "p",
        "license": "MIT",
        "language": "Python",
        "cwe": "CWE-89",
        "file_path": "a.py",
        "function_name": "f",
        "signature": "def f(x):",
        "pre_fix": "def f(x):\n    return 'SELECT * FROM t WHERE x=' + x\n",
        "post_fix": "def f(x):\n    return ('SELECT * FROM t WHERE x=?', [x])\n",
    }
    base.update(overrides)
    return base


def _seccodeplt_record(cwe: str = "CWE-89", record_id: str = "sec-x", **overrides) -> dict:
    """Default SecCodePLT record; override any field for a test.

    `cwe` populates the dict's `category` key (SecCodePLT's name for the
    CWE field). `record_id` populates `id`.
    """
    base = {
        "id": record_id,
        "category": cwe,
        "language": "python",
        "task_description": "Task description goes here.",
        "signature": "def f():",
        "tests": [],
    }
    base.update(overrides)
    return base
