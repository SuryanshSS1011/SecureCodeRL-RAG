"""Real Cppcheck adapter — shells out to the `cppcheck` CLI.

C/C++ only. Cppcheck 2.13+ supports native SARIF v2.1.0 output via
`--output-format=sarif`.

Known quirk (cppcheck 2.18 verified 2026-06-10): the SARIF emitter does
**not** populate CWE on rules — it only adds `tags: ["security"]` when
the finding's severity is "error" and it's not in the critical-id list.
The CWE is known internally per checker but isn't serialized. Our
`rule_map._CPPCHECK_TO_CWE` static table fills the gap (see rule_map.py).

Standard usage (`--enable=warning,style,performance,portability,information`
covers the security-relevant checks; `--inline-suppr` lets the policy code
override findings via `// cppcheck-suppress` comments if needed):

    cppcheck --enable=warning,style,performance,portability,information \\
        --output-format=sarif --inline-suppr --quiet snippet.c
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ..binary_resolver import resolve as _resolve_binary
from ..models import ToolName
from ..runner import Language, ToolAdapter, ToolRunResult


_LANG_EXT: dict[Language, str] = {
    Language.C: ".c",
    Language.CPP: ".cpp",
}


class CppcheckAdapter(ToolAdapter):
    def __init__(self, cppcheck_binary: str = "cppcheck") -> None:
        self._binary = _resolve_binary(cppcheck_binary) if cppcheck_binary == "cppcheck" else cppcheck_binary

    @property
    def tool(self) -> ToolName:
        return ToolName.CPPCHECK

    def run(
        self,
        code: str,
        language: Language,
        work_dir: Path,
        timeout_s: float,
    ) -> ToolRunResult:
        if language not in _LANG_EXT:
            return ToolRunResult(
                tool=self.tool,
                sarif={"runs": []},
                stderr=f"CppcheckAdapter does not support {language.value}",
                exit_code=0,
                duration_s=0.0,
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        source = work_dir / f"snippet{_LANG_EXT[language]}"
        source.write_text(code)

        # Cppcheck writes SARIF to stdout, not a file. We capture stdout.
        # `--enable=warning,style,performance,portability,information` covers
        # the practical security checkers. `--inline-suppr` honors per-line
        # suppressions; harmless when there are none.
        cmd = [
            self._binary,
            "--enable=warning,style,performance,portability,information",
            "--output-format=sarif",
            "--inline-suppr",
            "--quiet",
            str(source),
        ]
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                timeout=timeout_s,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            return ToolRunResult(
                tool=self.tool,
                sarif=None,
                stderr=f"timeout: {exc}",
                exit_code=-1,
                duration_s=time.monotonic() - start,
                timed_out=True,
            )

        duration = time.monotonic() - start

        # Cppcheck writes SARIF to STDERR by default (errors and SARIF go
        # to the error stream; stdout is reserved for the progress banner
        # under --quiet that gets suppressed). Try stdout first, fall back
        # to stderr — if the first parses as JSON we use it, otherwise the
        # other.
        sarif = None
        parse_target = proc.stdout if proc.stdout and proc.stdout.lstrip().startswith("{") else proc.stderr
        if parse_target:
            try:
                sarif = json.loads(parse_target)
            except json.JSONDecodeError as exc:
                return ToolRunResult(
                    tool=self.tool,
                    sarif=None,
                    stderr=(
                        f"failed to parse cppcheck SARIF: {exc}\n"
                        f"stdout head: {proc.stdout[:200]}\n"
                        f"stderr head: {proc.stderr[:400]}"
                    ),
                    exit_code=proc.returncode,
                    duration_s=duration,
                )

        # The error stream now also held the SARIF; surface only the
        # non-SARIF prefix (if any) in our `stderr` field.
        residual_stderr = proc.stderr if parse_target is proc.stdout else ""

        return ToolRunResult(
            tool=self.tool,
            sarif=sarif,
            stderr=residual_stderr,
            exit_code=proc.returncode,
            duration_s=duration,
        )
