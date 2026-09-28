"""Resolve SAST tool binaries to absolute paths.

Cluster compute nodes may not have .venv/bin or ~/.local/share/codeql on
PATH, so a bare `subprocess.run(["codeql", ...])` fails with
FileNotFoundError on every step. The resolver checks $CODEQL_BINARY /
$CPPCHECK_BINARY and known install paths first, then PATH, then the
interpreter's venv bin.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_CODEQL_FIXED_PATHS: tuple[str, ...] = (
    str(Path.home() / ".local" / "share" / "codeql" / "codeql"),
)


def resolve(name: str) -> str:
    """Return an absolute path to the named tool, or the bare name as a
    last resort so subprocess raises a clear FileNotFoundError.

    Tool-specific overrides:
        codeql:   honor $CODEQL_BINARY, then ~/.local/share/codeql
        cppcheck: honor $CPPCHECK_BINARY
    """
    if name == "codeql":
        env_override = os.environ.get("CODEQL_BINARY")
        if env_override and Path(env_override).exists():
            return env_override
        for cand in _CODEQL_FIXED_PATHS:
            if Path(cand).exists():
                return cand
    if name == "cppcheck":
        env_override = os.environ.get("CPPCHECK_BINARY")
        if env_override and Path(env_override).exists():
            return env_override

    via_path = shutil.which(name)
    if via_path:
        return via_path

    venv_bin = Path(sys.executable).parent / name
    if venv_bin.exists():
        return str(venv_bin)

    return name
