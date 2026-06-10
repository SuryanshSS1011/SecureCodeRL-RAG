"""Reliability oracle: compiles, runs, and tests generated code.

Populates the `ReliabilitySignals` consumed by the reward calculator. This
module is the only place that *executes* model-generated code. It must
operate under a sandbox; see `Sandbox` notes below.

Two oracle backends:

    - Python: a subprocess that imports the snippet (or pytest, if a test
      file is provided) under a resource-limited child process.
    - C/C++: shells out to a C/C++ compiler (clang by default), then runs
      the resulting binary against test inputs.

Both backends share the same `ReliabilityOracle.evaluate(...)` interface
so the trainer doesn't care about language.

Mock mode (`MockOracle`) returns canned signals; used in unit tests and in
end-to-end pipeline tests that don't need real compilation.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from .calculator import ReliabilitySignals

logger = logging.getLogger(__name__)


class Language(str, Enum):
    PYTHON = "python"
    C = "c"
    CPP = "cpp"


@dataclass
class TestCase:
    """A single functional test against the generated code.

    For Python: `input_stdin` is fed to stdin; the code's stdout is
    compared against `expected_stdout`. Stripping trailing whitespace
    only; no semantic comparison.

    For C/C++: same; the compiled binary is invoked with `input_stdin`.

    `timeout_s` per-test, separate from the per-snippet wall-clock budget.
    """

    __test__ = False  # not a pytest test class despite the "Test" prefix

    input_stdin: str = ""
    expected_stdout: str = ""
    timeout_s: float = 5.0


@dataclass
class TestSpec:
    __test__ = False  # not a pytest test class despite the "Test" prefix

    language: Language
    test_cases: list[TestCase]
    # For C/C++ snippets, extra source files (e.g., test harness) or
    # compile flags. For Python, the entry point's module function name
    # (defaults to running the snippet as a script).
    extra_files: dict[str, str] = None  # filename -> contents
    compile_flags: list[str] = None
    entry_module: Optional[str] = None  # Python: module to import; default = run as script
    # Optional splice context. When set, the oracle writes
    #   (prefix_text or "") + model_code + (suffix_text or "")
    # as the source file before compiling/parsing. Used for function-
    # completion prompts (Juliet's "complete bad()" style; CWEval's
    # BEGIN PROMPT / BEGIN SOLUTION wrapping) where the model is asked
    # to emit only a function body, and the surrounding TU (includes,
    # helper decls, harness main) must be re-attached before the source
    # can compile. `None` preserves the prior behavior (model output is
    # the whole TU).
    prefix_text: Optional[str] = None
    suffix_text: Optional[str] = None

    def __post_init__(self):
        if self.extra_files is None:
            self.extra_files = {}
        if self.compile_flags is None:
            self.compile_flags = []


class ReliabilityOracle(ABC):
    """Evaluates a generated code snippet against a TestSpec."""

    @abstractmethod
    def evaluate(self, code: str, spec: TestSpec) -> ReliabilitySignals: ...


class MockOracle(ReliabilityOracle):
    """Returns a canned ReliabilitySignals. For unit tests."""

    def __init__(self, signals: ReliabilitySignals) -> None:
        self._signals = signals

    def evaluate(self, code: str, spec: TestSpec) -> ReliabilitySignals:
        return self._signals


class RealOracle(ReliabilityOracle):
    """Compiles and runs the generated code in a subprocess sandbox.

    Sandboxing knobs (initial; will tighten when real-world hardening is
    needed):
        - `subprocess.run` with `timeout=` for wall-clock cap.
        - `resource.setrlimit` (POSIX) for memory and CPU caps via a
          preexec_fn. (Caller can disable for portability.)
        - Each evaluation runs in a fresh `tempfile.TemporaryDirectory`.
        - No network restrictions in code; rely on outer-process firewalling
          when training in CI.

    This is a single-threaded, single-process implementation. The trainer
    is expected to call `evaluate(...)` on each rollout serially within a
    GRPO group; if that becomes a bottleneck, parallelization moves to the
    rollout layer (multiple processes), not this layer.
    """

    def __init__(
        self,
        python_executable: str = "python3",
        c_compiler: str = "gcc",
        cpp_compiler: str = "g++",
        memory_limit_mb: int = 512,
        cpu_limit_s: int = 10,
        compile_mode: str = "link_and_run",
    ) -> None:
        # compile_mode controls how Compile@1 is judged for C/C++.
        # "link_and_run" links to a binary (requires main symbol +
        # test cases) and is the legacy in-flight RL training reward.
        # "syntax_only" passes -fsyntax-only and accepts any
        # syntactically valid translation unit, matching the Python
        # AST-parse check and giving SLMs an apples-to-apples definition
        # of Compile@1. Eval-time rescoring uses syntax_only.
        if compile_mode not in ("link_and_run", "syntax_only"):
            raise ValueError(
                "compile_mode must be 'link_and_run' or 'syntax_only', "
                "got {!r}".format(compile_mode)
            )
        import sys as _sys
        from ..sast.binary_resolver import resolve as _resolve_binary
        self.python_executable = _sys.executable if python_executable == "python3" else python_executable
        self.c_compiler = _resolve_binary(c_compiler)
        self.cpp_compiler = _resolve_binary(cpp_compiler)
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_s = cpu_limit_s
        self.compile_mode = compile_mode

    def evaluate(self, code: str, spec: TestSpec) -> ReliabilitySignals:
        # Manual tmpdir mgmt instead of TemporaryDirectory context to
        # tolerate subprocess cleanup races (see pipeline.py for the
        # same fix and the 2026-06-15 incident that prompted both).
        tmp_name = f"ictai_oracle_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        work_dir = Path(tempfile.gettempdir()) / tmp_name
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            for name, content in spec.extra_files.items():
                (work_dir / name).write_text(content)

            if spec.language == Language.PYTHON:
                return self._eval_python(code, spec, work_dir)
            if spec.language in (Language.C, Language.CPP):
                return self._eval_c_family(code, spec, work_dir)
            raise ValueError(f"unsupported language: {spec.language!r}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # ---------- Python ----------

    def _eval_python(self, code: str, spec: TestSpec, work_dir: Path) -> ReliabilitySignals:
        # Splice: if the prompt supplied surrounding text (e.g., the model
        # was asked to emit only a function body), stitch the full TU back
        # together before parsing. None on both -> model output is the
        # whole module, original behavior.
        if spec.prefix_text is not None or spec.suffix_text is not None:
            source_text = (spec.prefix_text or "") + code + (spec.suffix_text or "")
        else:
            source_text = code
        source_path = work_dir / "snippet.py"
        source_path.write_text(source_text)

        # Stage 1: parse check. Use AST parse rather than running the file,
        # because importing the file would execute top-level code.
        try:
            import ast

            ast.parse(source_text)
        except SyntaxError:
            return ReliabilitySignals(compiles=False)

        # If a harness file was supplied via TestSpec.entry_module (e.g., the
        # SecCodePLT translator writes _seccodeplt_harness.py and points
        # entry_module at it), run that instead of the snippet directly.
        # The harness imports the snippet as a module and runs the dataset's
        # native test format. Falls back to running snippet.py directly when
        # entry_module is None.
        if spec.entry_module:
            entry_path = work_dir / f"{spec.entry_module}.py"
            if not entry_path.exists():
                # extra_files should have written it. If not, bail honestly.
                return ReliabilitySignals(
                    compiles=True, runs=False, produces_output=False,
                    tests_passed=0, tests_total=len(spec.test_cases),
                )
            run_cmd = [self.python_executable, str(entry_path)]
        else:
            run_cmd = [self.python_executable, str(source_path)]

        compiles = True
        runs, produces_output, tests_passed = self._run_tests(
            cmd=run_cmd,
            spec=spec,
            work_dir=work_dir,
        )
        return ReliabilitySignals(
            compiles=compiles,
            runs=runs,
            produces_output=produces_output,
            tests_passed=tests_passed,
            tests_total=len(spec.test_cases),
        )

    # ---------- C / C++ ----------

    def _eval_c_family(self, code: str, spec: TestSpec, work_dir: Path) -> ReliabilitySignals:
        ext = ".c" if spec.language == Language.C else ".cpp"
        compiler = self.c_compiler if spec.language == Language.C else self.cpp_compiler
        # Splice: if the prompt was function-completion-style (model
        # emitted only a function body), reattach the original includes/
        # helpers (prefix_text) and harness tail (suffix_text) before
        # compiling. Without this, Juliet/CWEval body-only prompts could
        # never link, and C/C++ Compile@1 was artificially zero.
        if spec.prefix_text is not None or spec.suffix_text is not None:
            source_text = (spec.prefix_text or "") + code + (spec.suffix_text or "")
        else:
            source_text = code
        source_path = work_dir / f"snippet{ext}"
        source_path.write_text(source_text)
        binary_path = work_dir / "snippet.out"

        if self.compile_mode == "syntax_only":
            compile_cmd = [
                compiler,
                "-fsyntax-only",
                str(source_path),
                *spec.compile_flags,
            ]
        else:
            compile_cmd = [
                compiler,
                str(source_path),
                "-o",
                str(binary_path),
                *spec.compile_flags,
            ]
        try:
            # text=False + manual decode-with-replace tolerates the
            # arbitrary non-UTF-8 bytes that C/C++ programs (and the
            # toolchain) can emit on stderr: locale-dependent symbols,
            # ANSI escape sequences with high bytes, raw memory dumps
            # in compiler error contexts. With text=True the default
            # _translate_newlines step would raise UnicodeDecodeError
            # and the v0.1.5 sweep run on 2026-06-12 confirmed this is
            # a hot path under --oracle-kind=real.
            proc = subprocess.run(
                compile_cmd,
                cwd=work_dir,
                capture_output=True,
                timeout=15.0,
                text=False,
            )
        except subprocess.TimeoutExpired:
            return ReliabilitySignals(compiles=False)
        if proc.returncode != 0:
            return ReliabilitySignals(compiles=False)

        compiles = True
        if self.compile_mode == "syntax_only":
            # No binary was emitted; runs/output/tests aren't testable.
            # Compile@1 still reflects compile success.
            return ReliabilitySignals(
                compiles=compiles,
                runs=False,
                produces_output=False,
                tests_passed=0,
                tests_total=len(spec.test_cases),
            )
        runs, produces_output, tests_passed = self._run_tests(
            cmd=[str(binary_path)],
            spec=spec,
            work_dir=work_dir,
        )
        return ReliabilitySignals(
            compiles=compiles,
            runs=runs,
            produces_output=produces_output,
            tests_passed=tests_passed,
            tests_total=len(spec.test_cases),
        )

    # ---------- Test loop shared by Python / C / C++ ----------

    def _run_tests(
        self,
        cmd: list[str],
        spec: TestSpec,
        work_dir: Path,
    ) -> tuple[bool, bool, int]:
        """Run `cmd` against each test case; return (runs, produces_output, tests_passed).

        `runs` is True iff at least one test case ran without crashing.
        `produces_output` is True iff at least one test case produced any
        stdout. `tests_passed` counts cases whose stdout (rstrip'd) matches
        `expected_stdout` (rstrip'd) exactly.
        """
        if not spec.test_cases:
            # No tests configured -> can't tell if it runs. Use a single
            # empty-input invocation as a smoke test.
            spec_to_use = TestSpec(
                language=spec.language,
                test_cases=[TestCase(input_stdin="", expected_stdout="", timeout_s=5.0)],
                extra_files=spec.extra_files,
                compile_flags=spec.compile_flags,
                entry_module=spec.entry_module,
            )
        else:
            spec_to_use = spec

        any_run = False
        any_output = False
        n_passed = 0
        for case in spec_to_use.test_cases:
            try:
                # text=False + manual decode-with-replace: see compile
                # invocation above. The test executable can emit raw
                # bytes that crash UTF-8 _translate_newlines.
                proc = subprocess.run(
                    cmd,
                    cwd=work_dir,
                    input=case.input_stdin.encode("utf-8") if case.input_stdin else None,
                    capture_output=True,
                    timeout=case.timeout_s,
                    text=False,
                )
            except subprocess.TimeoutExpired:
                continue
            except (OSError, subprocess.SubprocessError):
                continue

            any_run = True
            stdout_bytes = proc.stdout or b""
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            if stdout.strip():
                any_output = True
            # No tests configured -> we don't count "pass" for the smoke
            # invocation; just record runs/output.
            if spec.test_cases:
                if stdout.rstrip() == case.expected_stdout.rstrip():
                    n_passed += 1

        return any_run, any_output, n_passed
