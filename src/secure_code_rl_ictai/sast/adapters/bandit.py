"""Real Bandit adapter — shells out to the `bandit` CLI.

Python-only. Returns SARIF v2.1.0. Bandit's `-ll` flag matches the LCTES
choice: report MEDIUM and HIGH severity only; LOW findings on `input()`
etc. would otherwise dominate Python code.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ..binary_resolver import resolve as _resolve_binary
from ..models import ToolName
from ..runner import Language, ToolAdapter, ToolRunResult


class BanditAdapter(ToolAdapter):
    def __init__(self, bandit_binary: str = "bandit") -> None:
        self._binary = _resolve_binary(bandit_binary) if bandit_binary == "bandit" else bandit_binary

    @property
    def tool(self) -> ToolName:
        return ToolName.BANDIT

    def run(
        self,
        code: str,
        language: Language,
        work_dir: Path,
        timeout_s: float,
    ) -> ToolRunResult:
        if language != Language.PYTHON:
            # Bandit is Python-only. Return a no-op result rather than crashing.
            return ToolRunResult(
                tool=self.tool,
                sarif={"runs": []},
                stderr=f"BanditAdapter declined non-Python language: {language.value}",
                exit_code=0,
                duration_s=0.0,
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        source = work_dir / "snippet.py"
        source.write_text(code)
        out_path = work_dir / "bandit.sarif"

        cmd = [
            self._binary,
            "-f", "sarif",
            "-ll",  # MEDIUM and HIGH only
            "-o", str(out_path),
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

        # Bandit exits 1 when findings are present; 0 means clean. Both are
        # success from our perspective.
        sarif = None
        if out_path.exists():
            try:
                sarif = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                return ToolRunResult(
                    tool=self.tool,
                    sarif=None,
                    stderr=f"failed to parse bandit SARIF: {exc}\nstderr:\n{proc.stderr}",
                    exit_code=proc.returncode,
                    duration_s=duration,
                )

        return ToolRunResult(
            tool=self.tool,
            sarif=sarif,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            duration_s=duration,
        )
