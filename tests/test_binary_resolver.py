"""SAST binary resolution honors env overrides before PATH."""

from __future__ import annotations

from pathlib import Path

import pytest

from secure_code_rl_ictai.sast.binary_resolver import resolve


@pytest.mark.parametrize("tool,env", [("codeql", "CODEQL_BINARY"), ("cppcheck", "CPPCHECK_BINARY")])
def test_env_override_wins(tool: str, env: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    binary = tmp_path / tool
    binary.write_text("")
    monkeypatch.setenv(env, str(binary))
    assert resolve(tool) == str(binary)


def test_missing_override_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CPPCHECK_BINARY", str(tmp_path / "absent"))
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve("cppcheck") != str(tmp_path / "absent")
