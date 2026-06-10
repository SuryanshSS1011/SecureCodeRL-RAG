"""Tests for the Juliet adapter.

Real Juliet (NIST SARD C/C++ Test Suite v1.3) ships as:

    juliet/C/testcases/CWE<NUM>_<desc>/[s<NN>/]CWE<NUM>__*_NN.c

Each .c file contains functions named `*_bad()` (under #ifndef OMITBAD)
and `*_goodG2B()` / `*_goodB2G()` (under #ifndef OMITGOOD). The adapter
extracts (bad, good) function-body pairs as ExemplarPairs.

The fixture writes a tiny tree that matches this layout precisely so the
tests pin the same regex + brace-matching code that will run on the real
data.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from secure_code_rl_ictai.data_prep import Language
from secure_code_rl_ictai.data_prep.juliet import JulietAdapter, JulietConfig


def _juliet_c_file(bad_body: str, good_body: str, base_name: str = "CWE121__test") -> str:
    """Build a Juliet-shaped C file with bad/good function pairs."""
    return textwrap.dedent(f'''\
        /* TEMPLATE GENERATED TESTCASE FILE
        Filename: {base_name}.c
        */
        #ifndef OMITBAD

        void {base_name}_bad()
        {{
        {bad_body}
        }}

        #endif /* OMITBAD */

        #ifndef OMITGOOD

        static void goodG2B()
        {{
        {good_body}
        }}

        #endif /* OMITGOOD */
        ''')


def _write_juliet_file(
    root: Path,
    cwe_num: int,
    cwe_desc: str,
    base_name: str,
    bad_body: str,
    good_body: str,
    subshard: str = "",
    language: str = "C",
) -> None:
    """Place a Juliet .c file in the canonical layout."""
    lang_dir = "C" if language == "C" else "Cpp"
    ext = "c" if language == "C" else "cpp"
    cwe_dir_name = f"CWE{cwe_num}_{cwe_desc}"
    parts = [root, lang_dir, "testcases", cwe_dir_name]
    if subshard:
        parts.append(subshard)
    dir_path = Path(*[str(p) for p in parts])
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{base_name}.{ext}"
    file_path.write_text(_juliet_c_file(bad_body, good_body, base_name))


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_juliet_emits_one_pair_per_file_c(tmp_path: Path):
    _write_juliet_file(
        tmp_path,
        cwe_num=121,
        cwe_desc="Stack_Based_Buffer_Overflow",
        base_name="CWE121_Stack_Based_Buffer_Overflow__CWE193_wchar_t_declare_loop_01",
        bad_body="    char buf[10];\n    memcpy(buf, src, 16);",
        good_body="    char buf[10];\n    memcpy(buf, src, 10);",
    )
    adapter = JulietAdapter(JulietConfig(juliet_root=tmp_path))
    pairs = list(adapter.load_exemplar_pairs())
    assert len(pairs) == 1
    p = pairs[0]
    assert p.cwe == "CWE-121"
    assert "memcpy(buf, src, 16)" in p.e_neg
    assert "memcpy(buf, src, 10)" in p.e_pos
    assert p.language == Language.C
    assert p.metadata["synthetic"] is True


def test_juliet_walks_subshards(tmp_path: Path):
    """CWE dirs often have s03/, s09/, etc. subshards. Adapter walks them."""
    _write_juliet_file(
        tmp_path, 121, "Stack_Based_Buffer_Overflow",
        "CWE121__test_01",
        "    /* bad */",
        "    /* good */",
        subshard="s03",
    )
    _write_juliet_file(
        tmp_path, 121, "Stack_Based_Buffer_Overflow",
        "CWE121__test_02",
        "    /* bad */",
        "    /* good */",
        subshard="s09",
    )
    adapter = JulietAdapter(JulietConfig(juliet_root=tmp_path))
    pairs = list(adapter.load_exemplar_pairs())
    assert len(pairs) == 2


def test_juliet_filters_by_target_cwe(tmp_path: Path):
    _write_juliet_file(
        tmp_path, 121, "Stack_Based_Buffer_Overflow",
        "x", "/* bad */", "/* good */",
    )
    _write_juliet_file(
        tmp_path, 89, "SQL_Injection",
        "y", "/* bad */", "/* good */",
    )
    adapter = JulietAdapter(
        JulietConfig(juliet_root=tmp_path, target_cwes=frozenset({"CWE-121"}))
    )
    pairs = list(adapter.load_exemplar_pairs())
    assert len(pairs) == 1
    assert pairs[0].cwe == "CWE-121"


# ----------------------------------------------------------------------
# Variants of the "good" function
# ----------------------------------------------------------------------


def test_juliet_recognizes_goodB2G_variant(tmp_path: Path):
    """Juliet sometimes uses goodB2G in place of goodG2B."""
    content = textwrap.dedent('''\
        #ifndef OMITBAD
        void CWE121__test_bad() {
            /* bad path */
        }
        #endif

        #ifndef OMITGOOD
        static void goodB2G() {
            /* good B2G path */
        }
        #endif
    ''')
    cwe_dir = tmp_path / "C" / "testcases" / "CWE121_Stack_Based_Buffer_Overflow"
    cwe_dir.mkdir(parents=True)
    (cwe_dir / "CWE121__test.c").write_text(content)

    adapter = JulietAdapter(JulietConfig(juliet_root=tmp_path))
    pairs = list(adapter.load_exemplar_pairs())
    assert len(pairs) == 1
    assert "good B2G" in pairs[0].e_pos
    assert "bad path" in pairs[0].e_neg


# ----------------------------------------------------------------------
# Malformed / partial
# ----------------------------------------------------------------------


def test_juliet_skips_files_with_only_bad(tmp_path: Path):
    """No good() function -> no pair; drop the file."""
    cwe_dir = tmp_path / "C" / "testcases" / "CWE121_Stack_Based_Buffer_Overflow"
    cwe_dir.mkdir(parents=True)
    (cwe_dir / "CWE121__broken.c").write_text(
        "#ifndef OMITBAD\nvoid bad() { /* bad */ }\n#endif\n"
    )
    adapter = JulietAdapter(JulietConfig(juliet_root=tmp_path))
    assert list(adapter.load_exemplar_pairs()) == []


def test_juliet_skips_files_with_only_good(tmp_path: Path):
    cwe_dir = tmp_path / "C" / "testcases" / "CWE121_Stack_Based_Buffer_Overflow"
    cwe_dir.mkdir(parents=True)
    (cwe_dir / "CWE121__broken.c").write_text(
        "#ifndef OMITGOOD\nstatic void goodG2B() {}\n#endif\n"
    )
    adapter = JulietAdapter(JulietConfig(juliet_root=tmp_path))
    assert list(adapter.load_exemplar_pairs()) == []


def test_juliet_ignores_non_c_files(tmp_path: Path):
    cwe_dir = tmp_path / "C" / "testcases" / "CWE121_Stack_Based_Buffer_Overflow"
    cwe_dir.mkdir(parents=True)
    (cwe_dir / "README.md").write_text("not a test case")
    (cwe_dir / "manifest.xml").write_text("<xml/>")
    adapter = JulietAdapter(JulietConfig(juliet_root=tmp_path))
    assert list(adapter.load_exemplar_pairs()) == []


def test_juliet_handles_missing_root(tmp_path: Path):
    adapter = JulietAdapter(JulietConfig(juliet_root=tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        list(adapter.load_exemplar_pairs())


def test_juliet_handles_root_with_no_testcases(tmp_path: Path):
    """Root exists but no C/testcases/ — gracefully empty."""
    adapter = JulietAdapter(JulietConfig(juliet_root=tmp_path))
    assert list(adapter.load_exemplar_pairs()) == []


# ----------------------------------------------------------------------
# Language: C and C++
# ----------------------------------------------------------------------


def test_juliet_loads_cpp_files(tmp_path: Path):
    _write_juliet_file(
        tmp_path,
        cwe_num=787,
        cwe_desc="Out_of_bounds_Write",
        base_name="CWE787__test",
        bad_body="    /* bad */",
        good_body="    /* good */",
        language="C++",
    )
    adapter = JulietAdapter(
        JulietConfig(
            juliet_root=tmp_path,
            languages=frozenset({Language.C, Language.CPP}),
        )
    )
    pairs = list(adapter.load_exemplar_pairs())
    assert len(pairs) == 1
    assert pairs[0].language == Language.CPP


def test_juliet_filters_by_language(tmp_path: Path):
    """Restrict to C++ only."""
    _write_juliet_file(
        tmp_path, 121, "Stack_Based_Buffer_Overflow",
        "x", "/* bad */", "/* good */", language="C",
    )
    _write_juliet_file(
        tmp_path, 787, "Out_of_bounds_Write",
        "y", "/* bad */", "/* good */", language="C++",
    )
    adapter = JulietAdapter(
        JulietConfig(
            juliet_root=tmp_path,
            languages=frozenset({Language.CPP}),
        )
    )
    pairs = list(adapter.load_exemplar_pairs())
    assert len(pairs) == 1
    assert pairs[0].language == Language.CPP


# ----------------------------------------------------------------------
# Adapter doesn't emit prompts (index-only)
# ----------------------------------------------------------------------


def test_juliet_emits_no_prompts(tmp_path: Path):
    _write_juliet_file(
        tmp_path, 121, "Stack_Based_Buffer_Overflow",
        "x", "/* bad */", "/* good */",
    )
    adapter = JulietAdapter(JulietConfig(juliet_root=tmp_path))
    assert list(adapter.load()) == []
