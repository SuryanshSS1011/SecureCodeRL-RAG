"""Real Semgrep adapter — shells out to the `semgrep` CLI.

Per docs/sast_pipeline_spec.md §3 (revised 2026-06-10):
    semgrep scan --config=p/security-audit --sarif --output=$OUT \\
        --metrics=off --no-rewrite-rule-ids --quiet $SOURCE

Why `p/security-audit` and not `auto`: `--config=auto` requires metrics
on (sends telemetry to semgrep.dev to fetch a relevant rule set), which
is incompatible with `--metrics=off`. We pin to `p/security-audit`, an
offline-friendly multi-language rule pack covering Python + C + C++ with
~250 rules across CWEs we care about (CWE-78, 89, 415, 79, 22, 327, ...).

`--metrics=off` keeps us off the network at scan time. `--no-rewrite-rule-ids`
keeps rule ids stable across versions for cache keys + regression tests.

`semgrep-core` (OCaml binary) ships with the Python package in the manylinux
wheel; ROAR venv uses semgrep 1.157 with the bundled binary at
`.venv/lib/python3.11/site-packages/semgrep/bin/semgrep-core`.
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
    Language.PYTHON: ".py",
    Language.C: ".c",
    Language.CPP: ".cpp",
}


class SemgrepAdapter(ToolAdapter):
    def __init__(
        self,
        semgrep_binary: str = "semgrep",
        config: str = "p/security-audit",
    ) -> None:
        self._binary = _resolve_binary(semgrep_binary) if semgrep_binary == "semgrep" else semgrep_binary
        self._config = config

    @property
    def tool(self) -> ToolName:
        return ToolName.SEMGREP

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
                stderr=f"SemgrepAdapter does not support {language.value}",
                exit_code=0,
                duration_s=0.0,
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        source = work_dir / f"snippet{_LANG_EXT[language]}"
        source.write_text(code)
        out_path = work_dir / "semgrep.sarif"

        cmd = [
            self._binary, "scan",
            f"--config={self._config}",
            "--sarif",
            f"--output={out_path}",
            "--metrics=off",
            "--no-rewrite-rule-ids",
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

        sarif = None
        if out_path.exists():
            try:
                sarif = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass

        # Semgrep exits 0 on success regardless of findings; nonzero is a
        # real failure.
        return ToolRunResult(
            tool=self.tool,
            sarif=sarif,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            duration_s=duration,
        )
