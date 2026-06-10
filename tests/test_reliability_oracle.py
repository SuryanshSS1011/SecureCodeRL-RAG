"""Tests for the reliability oracle.

The mock-mode tests are deterministic. The real-mode tests are marked with
`@pytest.mark.real_oracle` and skipped by default; run with
`pytest -m real_oracle` to exercise them. They require python3 (and for the
C tests, clang) on PATH.
"""

from __future__ import annotations

import shutil

import pytest

from secure_code_rl_ictai.reward import (
    Language,
    MockOracle,
    RealOracle,
    ReliabilitySignals,
    TestCase,
    TestSpec,
)


# ----------------------------------------------------------------------
# Mock oracle
# ----------------------------------------------------------------------


def test_mock_oracle_returns_supplied_signals():
    signals = ReliabilitySignals(
        compiles=True, runs=True, produces_output=True,
        tests_passed=3, tests_total=5,
    )
    oracle = MockOracle(signals)
    out = oracle.evaluate(
        "def f(): pass",
        TestSpec(language=Language.PYTHON, test_cases=[]),
    )
    assert out == signals


# ----------------------------------------------------------------------
# Real oracle - Python
# ----------------------------------------------------------------------


@pytest.mark.real_oracle
def test_real_oracle_python_syntax_error_blocks_compile():
    oracle = RealOracle()
    out = oracle.evaluate(
        "def f(:\n    pass\n",  # syntax error
        TestSpec(language=Language.PYTHON, test_cases=[]),
    )
    assert out.compiles is False
    assert out.runs is False


@pytest.mark.real_oracle
def test_real_oracle_python_runs_and_outputs():
    oracle = RealOracle()
    out = oracle.evaluate(
        'print("hello")\n',
        TestSpec(language=Language.PYTHON, test_cases=[]),
    )
    assert out.compiles is True
    assert out.runs is True
    assert out.produces_output is True


@pytest.mark.real_oracle
def test_real_oracle_python_test_passes():
    code = (
        "import sys\n"
        "n = int(sys.stdin.read().strip())\n"
        "print(n * 2)\n"
    )
    spec = TestSpec(
        language=Language.PYTHON,
        test_cases=[
            TestCase(input_stdin="5\n", expected_stdout="10\n", timeout_s=5.0),
            TestCase(input_stdin="0\n", expected_stdout="0\n", timeout_s=5.0),
        ],
    )
    out = RealOracle().evaluate(code, spec)
    assert out.compiles is True
    assert out.runs is True
    assert out.tests_passed == 2
    assert out.tests_total == 2


@pytest.mark.real_oracle
def test_real_oracle_python_partial_test_pass():
    """Code that doubles input correctly only for non-empty stdin."""
    code = (
        "import sys\n"
        "data = sys.stdin.read().strip()\n"
        'if data:\n'
        "    print(int(data) * 2)\n"
    )
    spec = TestSpec(
        language=Language.PYTHON,
        test_cases=[
            TestCase(input_stdin="5\n", expected_stdout="10\n", timeout_s=5.0),
            TestCase(input_stdin="", expected_stdout="0\n", timeout_s=5.0),  # fails
        ],
    )
    out = RealOracle().evaluate(code, spec)
    assert out.compiles is True
    assert out.tests_passed == 1
    assert out.tests_total == 2


# ----------------------------------------------------------------------
# Real oracle - C
# ----------------------------------------------------------------------


@pytest.mark.real_oracle
@pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang not on PATH"
)
def test_real_oracle_c_compile_failure():
    oracle = RealOracle()
    out = oracle.evaluate(
        "int main() { return undefined_symbol; }\n",
        TestSpec(language=Language.C, test_cases=[]),
    )
    assert out.compiles is False


@pytest.mark.real_oracle
@pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang not on PATH"
)
def test_real_oracle_c_runs_and_outputs():
    code = '#include <stdio.h>\nint main(){ printf("hi\\n"); return 0; }\n'
    out = RealOracle().evaluate(
        code,
        TestSpec(language=Language.C, test_cases=[]),
    )
    assert out.compiles is True
    assert out.runs is True
    assert out.produces_output is True


# ----------------------------------------------------------------------
# Splice (prefix_text / suffix_text)
# ----------------------------------------------------------------------


def test_test_spec_splice_fields_default_to_none():
    """Schema regression: existing call-sites construct TestSpec without the
    new splice fields, so prefix_text / suffix_text must default to None
    (i.e., the oracle's behavior is unchanged for legacy specs)."""
    spec = TestSpec(language=Language.C, test_cases=[])
    assert spec.prefix_text is None
    assert spec.suffix_text is None


@pytest.mark.real_oracle
@pytest.mark.skipif(
    shutil.which("gcc") is None and shutil.which("clang") is None,
    reason="no C compiler on PATH",
)
def test_real_oracle_c_splice_recovers_body_only_completion():
    """Splice fix regression: a model that emits a function body only
    (no includes, no main, no closing brace context) MUST still compile
    when the prompt's TestSpec carries the original prefix_text +
    suffix_text. Without splice, this fails to compile (no printf decl,
    no main, dangling body). With splice, it compiles, runs, and
    prints '5' to stdout.

    Mirrors the LCTES/Juliet C function-completion eval shape: the
    model returns body content; the oracle reattaches the surrounding
    TU before invoking gcc.
    """
    # Body the "model" emitted (no includes, no surrounding `int bad()`).
    body = "    return 5;\n"
    spec = TestSpec(
        language=Language.C,
        test_cases=[],
        prefix_text="#include <stdio.h>\nint bad() {\n",
        suffix_text="}\nint main(){ printf(\"%d\\n\", bad()); return 0; }\n",
    )
    out = RealOracle().evaluate(body, spec)
    assert out.compiles is True, (
        f"splice should produce a compilable TU; got compiles={out.compiles}"
    )
    assert out.runs is True
    assert out.produces_output is True


@pytest.mark.real_oracle
def test_real_oracle_python_splice_concatenates_around_body():
    """Splice fix regression (Python): when prefix/suffix are set, the
    oracle stitches them around the model's snippet before AST-parsing
    and executing. Verifies the Python path uses the same splice as C/C++.
    """
    body = "    return x * 2\n"
    spec = TestSpec(
        language=Language.PYTHON,
        test_cases=[],
        prefix_text="def f(x):\n",
        suffix_text="print(f(7))\n",
    )
    out = RealOracle().evaluate(body, spec)
    assert out.compiles is True
    assert out.runs is True
    assert out.produces_output is True


# ----------------------------------------------------------------------
# compile_mode (paper Section V.B Compile@1: syntax-only for C/C++)
# ----------------------------------------------------------------------


def test_real_oracle_rejects_unknown_compile_mode():
    with pytest.raises(ValueError):
        RealOracle(compile_mode="bogus")


def _capture_compile_calls(monkeypatch):
    """Stub subprocess.run so no compiler or binary actually executes."""
    import subprocess

    from secure_code_rl_ictai.reward import reliability_oracle as ro

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append([str(c) for c in cmd])
        return subprocess.CompletedProcess(cmd, 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(ro.subprocess, "run", fake_run)
    return calls


def test_syntax_only_mode_uses_fsyntax_only_and_never_runs(monkeypatch):
    calls = _capture_compile_calls(monkeypatch)
    oracle = RealOracle(compile_mode="syntax_only")
    spec = TestSpec(
        language=Language.C,
        test_cases=[TestCase(input_stdin="", expected_stdout="ok")],
    )
    out = oracle.evaluate("int main(void) { return 0; }\n", spec)
    # One compiler invocation, no binary execution.
    assert len(calls) == 1
    assert "-fsyntax-only" in calls[0]
    assert "-o" not in calls[0]
    assert out.compiles is True
    assert out.runs is False and out.produces_output is False
    assert out.tests_passed == 0 and out.tests_total == 1


def test_link_and_run_mode_links_then_executes(monkeypatch):
    calls = _capture_compile_calls(monkeypatch)
    oracle = RealOracle(compile_mode="link_and_run")
    spec = TestSpec(
        language=Language.C,
        test_cases=[TestCase(input_stdin="", expected_stdout="ok")],
    )
    out = oracle.evaluate("int main(void) { return 0; }\n", spec)
    assert len(calls) == 2  # compile+link, then run the binary
    assert "-o" in calls[0] and "-fsyntax-only" not in calls[0]
    assert out.compiles is True and out.runs is True
    assert out.tests_passed == 1


def test_default_compile_mode_is_link_and_run():
    assert RealOracle().compile_mode == "link_and_run"
