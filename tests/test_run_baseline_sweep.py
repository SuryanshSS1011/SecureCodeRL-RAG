"""Tests for scripts/run_baseline_sweep.py CLI.

The sweep runs the eval harness against multiple baselines and produces
one report dir per baseline. We exercise the mock-baseline path: a JSON
config maps baseline names to mock-response JSONLs. Real-baseline rows
that can't load (cite-only, our-checkpoints) are recorded as errors but
don't crash the sweep.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_baseline_sweep.py"


def _run_script(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _write_eval_jsonl(path: Path, n: int = 2) -> None:
    records = [
        {
            "id": f"sweep:p{i}",
            "source": "test",
            "language": "python",
            "target_cwe": "CWE-89",
            "prompt_text": f"# task {i}",
            "test_spec": {"language": "python", "test_cases": []},
            "task_signature": f"def task_{i}():",
            "metadata": {},
        }
        for i in range(n)
    ]
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _write_mock_responses(path: Path, *, completion: str = "pass") -> None:
    """Mock responses keyed by prompt_text from the eval JSONL above."""
    records = [
        {"prompt": "# task 0", "completion": completion},
        {"prompt": "# task 1", "completion": completion},
    ]
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ----------------------------------------------------------------------
# Mock-baselines path
# ----------------------------------------------------------------------


def test_sweep_runs_multiple_mock_baselines(tmp_path: Path):
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_jsonl(eval_path)
    mock_a = tmp_path / "mock_a.jsonl"
    mock_b = tmp_path / "mock_b.jsonl"
    _write_mock_responses(mock_a, completion="def f(): return 1")
    _write_mock_responses(mock_b, completion="def f(): return 2")

    sweep_config = tmp_path / "sweep.json"
    sweep_config.write_text(
        json.dumps(
            {
                "baselines": [
                    {"name": "mock-a", "mock_from_jsonl": str(mock_a)},
                    {"name": "mock-b", "mock_from_jsonl": str(mock_b)},
                ]
            }
        )
    )

    output_dir = tmp_path / "runs"

    result = _run_script(
        [
            "--eval-jsonl", str(eval_path),
            "--config", str(sweep_config),
            "--output", str(output_dir),
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert (output_dir / "mock-a" / "aggregate.json").exists()
    assert (output_dir / "mock-b" / "aggregate.json").exists()
    assert (output_dir / "sweep_summary.json").exists()


def test_sweep_summary_records_per_baseline_metrics(tmp_path: Path):
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_jsonl(eval_path)
    mock_path = tmp_path / "mock.jsonl"
    _write_mock_responses(mock_path)

    sweep_config = tmp_path / "sweep.json"
    sweep_config.write_text(
        json.dumps({"baselines": [{"name": "m", "mock_from_jsonl": str(mock_path)}]})
    )
    output_dir = tmp_path / "runs"

    result = _run_script(
        ["--eval-jsonl", str(eval_path), "--config", str(sweep_config), "--output", str(output_dir)],
        cwd=tmp_path,
    )
    assert result.returncode == 0
    summary = json.loads((output_dir / "sweep_summary.json").read_text())
    assert "m" in summary["baselines"]
    bm = summary["baselines"]["m"]
    assert "status" in bm and bm["status"] == "ok"
    assert "aggregate" in bm
    assert "func_at_1" in bm["aggregate"]


# ----------------------------------------------------------------------
# Real-baseline path failures recorded, not crashing
# ----------------------------------------------------------------------


def test_sweep_records_unrunnable_baseline_without_crashing(tmp_path: Path):
    """Cite-only baselines raise NotImplementedError from their factory.
    The sweep should record an error entry and move on, not crash."""
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_jsonl(eval_path)
    mock_path = tmp_path / "mock.jsonl"
    _write_mock_responses(mock_path)

    sweep_config = tmp_path / "sweep.json"
    sweep_config.write_text(
        json.dumps(
            {
                "baselines": [
                    {"name": "mock-ok", "mock_from_jsonl": str(mock_path)},
                    {"name": "seccoderx", "baseline": "seccoderx"},  # cite-only
                ]
            }
        )
    )
    output_dir = tmp_path / "runs"

    result = _run_script(
        ["--eval-jsonl", str(eval_path), "--config", str(sweep_config), "--output", str(output_dir)],
        cwd=tmp_path,
    )
    # Sweep itself succeeds (exit 0) even with one row failing.
    assert result.returncode == 0
    summary = json.loads((output_dir / "sweep_summary.json").read_text())
    assert summary["baselines"]["mock-ok"]["status"] == "ok"
    assert summary["baselines"]["seccoderx"]["status"] == "error"
    assert "error" in summary["baselines"]["seccoderx"]


def test_sweep_rejects_missing_config(tmp_path: Path):
    result = _run_script(
        ["--eval-jsonl", str(tmp_path / "e.jsonl"),
         "--config", str(tmp_path / "missing.json"),
         "--output", str(tmp_path / "out")],
        cwd=tmp_path,
    )
    assert result.returncode != 0
