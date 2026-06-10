"""Real CodeQL adapter — shells out to the `codeql` CLI.

Per docs/sast_pipeline_spec.md §3, the flow is:
    codeql database create $DB --language=$LANG --source-root=$WORK_DIR
    codeql database analyze $DB $QUERY_SUITE --format=sarif-latest --output=$OUT

Database creation is the dominant cost (~10-60s on small Python snippets,
longer for C/C++). Per-run wall-clock is dominated by the create step;
cache the database across runs of the same snippet hash if you can.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from ..binary_resolver import resolve as _resolve_binary
from ..models import ToolName
from ..runner import Language, ToolAdapter, ToolRunResult


_QUERY_SUITES: dict[Language, str] = {
    Language.PYTHON: "codeql/python-queries:codeql-suites/python-security-extended.qls",
    Language.C: "codeql/cpp-queries:codeql-suites/cpp-security-extended.qls",
    Language.CPP: "codeql/cpp-queries:codeql-suites/cpp-security-extended.qls",
}

_LANG_NAME: dict[Language, str] = {
    Language.PYTHON: "python",
    Language.C: "cpp",  # CodeQL groups C and C++ under "cpp"
    Language.CPP: "cpp",
}

_LANG_EXT: dict[Language, str] = {
    Language.PYTHON: ".py",
    Language.C: ".c",
    Language.CPP: ".cpp",
}


class CodeQLAdapter(ToolAdapter):
    def __init__(self, codeql_binary: str = "codeql") -> None:
        self._binary = _resolve_binary(codeql_binary) if codeql_binary == "codeql" else codeql_binary

    @property
    def tool(self) -> ToolName:
        return ToolName.CODEQL

    def run(
        self,
        code: str,
        language: Language,
        work_dir: Path,
        timeout_s: float,
    ) -> ToolRunResult:
        if language not in _QUERY_SUITES:
            return ToolRunResult(
                tool=self.tool,
                sarif={"runs": []},
                stderr=f"CodeQLAdapter does not support {language.value}",
                exit_code=0,
                duration_s=0.0,
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        # CodeQL wants a source root *directory* that contains the snippet.
        source_root = work_dir / "source"
        source_root.mkdir(exist_ok=True)
        snippet = source_root / f"snippet{_LANG_EXT[language]}"
        snippet.write_text(code)
        db_path = work_dir / "db"
        sarif_path = work_dir / "codeql.sarif"

        # Stale DB from a previous call in the same work_dir? CodeQL `database
        # create` refuses to overwrite. Delete pre-emptively.
        if db_path.exists():
            shutil.rmtree(db_path, ignore_errors=True)

        # Step 1: create database
        create_cmd = [
            self._binary, "database", "create",
            str(db_path),
            f"--language={_LANG_NAME[language]}",
            f"--source-root={source_root}",
            "--quiet",
        ]
        if language in (Language.C, Language.CPP):
            # CodeQL needs a build command for C/C++. The simplest is a
            # one-shot compile via gcc/clang; we let CodeQL infer via its
            # default --command. If neither is available the create step
            # will fail; we surface that in stderr.
            create_cmd.extend(["--command", "gcc -c snippet.c"])

        start = time.monotonic()
        # The timeout applies to each CodeQL invocation (database create,
        # then analyze); cold-start query loading alone needs most of it.
        create_timeout = timeout_s
        analyze_timeout = timeout_s

        try:
            create_proc = subprocess.run(
                create_cmd,
                cwd=work_dir,
                capture_output=True,
                timeout=create_timeout,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return ToolRunResult(
                tool=self.tool,
                sarif=None,
                stderr="codeql database create timed out",
                exit_code=-1,
                duration_s=time.monotonic() - start,
                timed_out=True,
            )

        if create_proc.returncode != 0:
            return ToolRunResult(
                tool=self.tool,
                sarif=None,
                stderr=f"codeql database create failed: {create_proc.stderr}",
                exit_code=create_proc.returncode,
                duration_s=time.monotonic() - start,
            )

        # Step 2: analyze.
        # --threads=1 to keep CPU contention down across the 16 GRPO rollouts
        # called sequentially within the trainer. RAM left at default; codeql
        # autoresolves to ~8 GB heap which fits comfortably in the 80 GB SLURM
        # allocation (only one analyze runs at a time per trainer process).
        analyze_cmd = [
            self._binary, "database", "analyze",
            str(db_path),
            _QUERY_SUITES[language],
            "--format=sarif-latest",
            f"--output={sarif_path}",
            "--quiet",
            "--threads=1",
        ]
        try:
            analyze_proc = subprocess.run(
                analyze_cmd,
                cwd=work_dir,
                capture_output=True,
                timeout=analyze_timeout,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return ToolRunResult(
                tool=self.tool,
                sarif=None,
                stderr="codeql database analyze timed out",
                exit_code=-1,
                duration_s=time.monotonic() - start,
                timed_out=True,
            )

        duration = time.monotonic() - start

        sarif = None
        if sarif_path.exists():
            try:
                sarif = json.loads(sarif_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                return ToolRunResult(
                    tool=self.tool,
                    sarif=None,
                    stderr=f"failed to parse codeql SARIF: {exc}",
                    exit_code=analyze_proc.returncode,
                    duration_s=duration,
                )

        return ToolRunResult(
            tool=self.tool,
            sarif=sarif,
            stderr=analyze_proc.stderr,
            exit_code=analyze_proc.returncode,
            duration_s=duration,
        )
