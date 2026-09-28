"""Tests for the CWEval adapter.

CWEval (Co1lin/CWEval, paper arXiv:2501.08200) ships as a flat per-task-
pair layout under `benchmark/core/<lang>/`:

    benchmark/core/py/cwe_020_0_task.py
    benchmark/core/py/cwe_020_0_test.py
    benchmark/core/py/cwe_022_0_task.py
    benchmark/core/py/cwe_022_0_test.py
    ...

Each `_task.py` is a Python function with a docstring (the prompt) and a
`# BEGIN SOLUTION` marker that delimits the reference implementation.
The CWE id is encoded in the filename (`cwe_NNN_M`). The matching
`_test.py` is the oracle script run by the reliability oracle at eval
time.

We DON'T parse the reference implementation here — the prompt is the
function signature + docstring; the model generates a completion.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from cargo.data_prep import Language
from cargo.data_prep.cweval import CwevalAdapter, CwevalConfig


def _write_cweval_pair(
    root: Path,
    language: str,
    cwe_num: int,
    variant: int,
    signature: str,
    docstring: str,
    solution: str = "    return None\n",
    test_body: str = "def test_x():\n    assert True\n",
) -> None:
    """Write a (task, test) file pair matching CWEval's on-disk layout.

    CWEval conventions (verified against the real repo):
      - Python: `<root>/benchmark/core/py/cwe_<NUM>_<M>_task.py`
        and matching `_test.py`. Task file ends with `# BEGIN SOLUTION`.
      - C: `<root>/benchmark/core/c/cwe_<NUM>_<M>_c_task.c` (note the
        `_c` suffix) and `_c_test.py` (test driver is Python; runs the
        compiled binary). C task files have `// BEGIN PROMPT` marking
        the start of the function the model fills in, and
        `// BEGIN SOLUTION` marking the function body.
      - C++: `<root>/benchmark/core/cpp/cwe_<NUM>_<M>_cpp_task.cpp`
        and `_cpp_test.py`. Same PROMPT / SOLUTION marker convention.

    root = the cweval *clone* root.
    """
    if language == "py":
        ext = "py"
        task_name = f"cwe_{cwe_num:03d}_{variant}_task.py"
        test_name = f"cwe_{cwe_num:03d}_{variant}_test.py"
        task_body = textwrap.dedent(
            f'''\
            {signature}
                """{docstring}"""
                # BEGIN SOLUTION
            {solution}'''
        )
    elif language in ("c", "cpp"):
        ext = language
        # CWEval suffix: cwe_NNN_M_c_task.c or cwe_NNN_M_cpp_task.cpp.
        task_name = f"cwe_{cwe_num:03d}_{variant}_{language}_task.{ext}"
        test_name = f"cwe_{cwe_num:03d}_{variant}_{language}_test.py"
        # CWEval uses BEGIN PROMPT (start of the model-filled fn) and
        # BEGIN SOLUTION (start of the body). The text BEFORE PROMPT is
        # helper / includes context that should NOT appear in the prompt.
        task_body = (
            f"#include <stdio.h>\n\n"
            f"// Helper code; should NOT appear in the prompt.\n"
            f"int helper() {{ return 0; }}\n\n"
            f"// BEGIN PROMPT\n"
            f"/*\n{docstring}\n*/\n"
            f"{signature} {{\n"
            f"    // BEGIN SOLUTION\n"
            f"{solution}}}\n"
        )
    else:
        raise ValueError(f"unsupported language for fixture: {language}")

    lang_dir = root / "benchmark" / "core" / language
    lang_dir.mkdir(parents=True, exist_ok=True)
    (lang_dir / task_name).write_text(task_body)
    (lang_dir / test_name).write_text(test_body)


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_cweval_emits_prompt_per_task_python(tmp_path: Path):
    _write_cweval_pair(
        tmp_path,
        language="py",
        cwe_num=89,
        variant=0,
        signature="def get_user(uid: str) -> str:",
        docstring="Fetch a user record by id safely.",
    )
    adapter = CwevalAdapter(CwevalConfig(cweval_root=tmp_path))
    prompts = list(adapter.load())
    assert len(prompts) == 1
    p = prompts[0]
    assert p.source == "cweval"
    assert p.target_cwe == "CWE-89"
    assert p.language == Language.PYTHON
    # prompt_text contains the function signature and docstring.
    assert "get_user" in p.prompt_text
    assert "safely" in p.prompt_text
    assert p.task_signature == "def get_user(uid: str) -> str:"
    assert p.metadata["cweval_task_id"] == "cwe_089_0"
    # oracle_script points at the matching test file.
    assert p.metadata["oracle_script"].endswith("cwe_089_0_test.py")


def test_cweval_emits_prompt_per_task_c(tmp_path: Path):
    _write_cweval_pair(
        tmp_path,
        language="c",
        cwe_num=787,
        variant=0,
        signature="int copy(char *dst, const char *src, int n)",
        docstring="Copy n bytes from src to dst safely.",
        solution="    for (int i = 0; i < n; ++i) dst[i] = src[i];\n    return 0;\n",
    )
    adapter = CwevalAdapter(CwevalConfig(cweval_root=tmp_path))
    prompts = list(adapter.load())
    assert len(prompts) == 1
    p = prompts[0]
    assert p.target_cwe == "CWE-787"
    assert p.language == Language.C
    assert "copy" in p.prompt_text
    # CWEval C convention: BEGIN PROMPT marks the start of the model's
    # prompt; the helper function before PROMPT must NOT leak in.
    assert "helper()" not in p.prompt_text
    assert "Copy n bytes" in p.prompt_text


# ----------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------


def test_cweval_filters_by_cwe(tmp_path: Path):
    _write_cweval_pair(tmp_path, "py", 89, 0, "def f():", "x")
    _write_cweval_pair(tmp_path, "py", 79, 0, "def g():", "y")
    adapter = CwevalAdapter(
        CwevalConfig(cweval_root=tmp_path, target_cwes=frozenset({"CWE-89"}))
    )
    prompts = list(adapter.load())
    assert len(prompts) == 1
    assert prompts[0].target_cwe == "CWE-89"


def test_cweval_filters_by_language(tmp_path: Path):
    _write_cweval_pair(tmp_path, "py", 89, 0, "def f():", "x")
    _write_cweval_pair(
        tmp_path, "c", 787, 0,
        "int f(char *p)", "x", solution="    return 0;\n",
    )
    adapter = CwevalAdapter(
        CwevalConfig(cweval_root=tmp_path, languages=frozenset({Language.C}))
    )
    prompts = list(adapter.load())
    assert len(prompts) == 1
    assert prompts[0].language == Language.C


def test_cweval_default_drops_go_and_js(tmp_path: Path):
    """Default config: only Python, C, C++. Skip Go and JS dirs even if present."""
    _write_cweval_pair(tmp_path, "py", 89, 0, "def f():", "x")
    # Manually create a `go/` task to verify it's skipped under defaults.
    (tmp_path / "benchmark" / "core" / "go").mkdir(parents=True)
    (tmp_path / "benchmark" / "core" / "go" / "cwe_089_0_task.go").write_text("// task")
    (tmp_path / "benchmark" / "core" / "go" / "cwe_089_0_test.go").write_text("// test")

    adapter = CwevalAdapter(CwevalConfig(cweval_root=tmp_path))
    prompts = list(adapter.load())
    assert len(prompts) == 1
    assert prompts[0].language == Language.PYTHON


# ----------------------------------------------------------------------
# Malformed / missing
# ----------------------------------------------------------------------


def test_cweval_skips_task_with_no_matching_test_file(tmp_path: Path):
    """If `cwe_089_0_task.py` has no matching `cwe_089_0_test.py`, drop it."""
    lang_dir = tmp_path / "benchmark" / "core" / "py"
    lang_dir.mkdir(parents=True)
    (lang_dir / "cwe_089_0_task.py").write_text("def f():\n    pass\n")
    # No test file.
    adapter = CwevalAdapter(CwevalConfig(cweval_root=tmp_path))
    assert list(adapter.load()) == []


def test_cweval_skips_files_with_bad_naming(tmp_path: Path):
    """Files that don't match the cwe_NNN_M pattern are ignored."""
    lang_dir = tmp_path / "benchmark" / "core" / "py"
    lang_dir.mkdir(parents=True)
    (lang_dir / "random_helper.py").write_text("def helper(): pass\n")
    (lang_dir / "__init__.py").write_text("")
    adapter = CwevalAdapter(CwevalConfig(cweval_root=tmp_path))
    assert list(adapter.load()) == []


def test_cweval_raises_not_implemented_for_pairs(tmp_path: Path):
    adapter = CwevalAdapter(CwevalConfig(cweval_root=tmp_path))
    with pytest.raises(NotImplementedError):
        list(adapter.load_exemplar_pairs())


def test_cweval_raises_when_root_missing(tmp_path: Path):
    adapter = CwevalAdapter(CwevalConfig(cweval_root=tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        list(adapter.load())


def test_cweval_handles_missing_benchmark_dir(tmp_path: Path):
    """Root exists but no benchmark/core/ — gracefully empty, no crash."""
    adapter = CwevalAdapter(CwevalConfig(cweval_root=tmp_path))
    assert list(adapter.load()) == []
