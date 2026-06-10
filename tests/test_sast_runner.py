"""Tests for the SAST runner orchestration (mock mode).

Real-mode tests that shell out to actual SAST tools are marked
@pytest.mark.real_sast and skipped by default.
"""

from __future__ import annotations

from pathlib import Path

from secure_code_rl_ictai.sast.models import ToolName
from secure_code_rl_ictai.sast.runner import (
    LANGUAGE_ROUTING,
    Language,
    MockAdapter,
    SastRunner,
)


def _canned_sarif_codeql(cwe_num: int, severity: float, line: int = 10) -> dict:
    return {
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "codeql",
                        "rules": [
                            {
                                "id": f"py/test-{cwe_num}",
                                "relationships": [
                                    {
                                        "target": {
                                            "id": str(cwe_num),
                                            "toolComponent": {"name": "CWE"},
                                        }
                                    }
                                ],
                                "properties": {"security-severity": str(severity)},
                            }
                        ],
                    }
                },
                "results": [
                    {
                        "ruleId": f"py/test-{cwe_num}",
                        "level": "error",
                        "message": {"text": "test finding"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "snippet.py"},
                                    "region": {"startLine": line},
                                }
                            }
                        ],
                    }
                ],
            }
        ]
    }


def test_python_routing_invokes_three_tools():
    """Language routing for Python should call CodeQL, Semgrep, Bandit."""
    adapters = {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, _canned_sarif_codeql(89, 8.8)),
        ToolName.SEMGREP: MockAdapter(ToolName.SEMGREP, {"runs": []}),
        ToolName.BANDIT: MockAdapter(ToolName.BANDIT, {"runs": []}),
        ToolName.CPPCHECK: MockAdapter(ToolName.CPPCHECK, {"runs": []}),  # should not be called
    }
    runner = SastRunner(adapters)
    summary, findings = runner.run("import os\n", Language.PYTHON, work_dir=Path("/tmp"))
    assert set(summary.per_tool.keys()) == {ToolName.CODEQL, ToolName.SEMGREP, ToolName.BANDIT}
    assert len(findings) == 1
    assert findings[0].cwe == "CWE-89"


def test_c_routing_invokes_four_tools():
    adapters = {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, {"runs": []}),
        ToolName.SEMGREP: MockAdapter(ToolName.SEMGREP, {"runs": []}),
        ToolName.BANDIT: MockAdapter(ToolName.BANDIT, {"runs": []}),  # not called
        ToolName.CPPCHECK: MockAdapter(ToolName.CPPCHECK, {"runs": []}),
    }
    runner = SastRunner(adapters)
    summary, findings = runner.run("int main(){}\n", Language.C, work_dir=Path("/tmp"))
    assert set(summary.per_tool.keys()) == {
        ToolName.CODEQL, ToolName.SEMGREP, ToolName.CPPCHECK
    }
    assert findings == []


def test_crashed_tool_recorded_but_run_proceeds():
    adapters = {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, _canned_sarif_codeql(89, 8.8)),
        ToolName.SEMGREP: MockAdapter(ToolName.SEMGREP, None),  # crashes (no canned SARIF)
        ToolName.BANDIT: MockAdapter(ToolName.BANDIT, {"runs": []}),
    }
    runner = SastRunner(adapters)
    summary, findings = runner.run("import os\n", Language.PYTHON, work_dir=Path("/tmp"))
    assert ToolName.SEMGREP in summary.crashed_tools
    # CodeQL still produced a finding
    assert len(findings) == 1


def test_snippet_hash_stable_and_distinct_per_language():
    h_py = SastRunner.snippet_hash("x = 1\n", Language.PYTHON)
    h_py_again = SastRunner.snippet_hash("x = 1", Language.PYTHON)  # trailing ws ignored
    h_c = SastRunner.snippet_hash("x = 1\n", Language.C)
    assert h_py == h_py_again
    assert h_py != h_c


def test_routing_table_is_consistent_with_tool_enum():
    """Every tool in LANGUAGE_ROUTING must exist in ToolName."""
    for lang, tools in LANGUAGE_ROUTING.items():
        for tool in tools:
            assert isinstance(tool, ToolName)
