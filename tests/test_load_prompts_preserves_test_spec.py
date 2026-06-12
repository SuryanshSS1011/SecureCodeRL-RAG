"""Regression: every script that reads eval_prompts.jsonl must preserve
TestSpec.extra_files and TestSpec.entry_module.

Background (2026-06-11):
  the prompt loaders in scripts/run_baseline_sweep.py and scripts/train_method.py had _load_prompts implementations that
  constructed TestSpec without extra_files/entry_module. The harness wireup
  for SecCodePLT and Python-CWEval prompts lives in those fields, so
  dropping them caused RealOracle to run snippet.py directly, miss the
  expected_stdout='PASS' contract, and silently report tests_passed=0
  on every SecCodePLT/CWEval prompt — making the entire MULTI_LLM
  sweep return func@1=0 universally. Burned ~2 a40-hr of calibration
  before the diagnosis.

This test imports each script's `_load_prompts` and feeds it a
SecCodePLT-style fixture; if any script drops the fields, the test
fails loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _fixture_record() -> dict:
    """One SecCodePLT-style prompt with the v0.1.1+ harness wired."""
    return {
        "id": "seccodeplt:test_fixture_001",
        "source": "seccodeplt",
        "language": "python",
        "target_cwe": "CWE-22",
        "prompt_text": "Implement path_check that validates URLs.",
        "task_signature": None,
        "test_spec": {
            "language": "python",
            "test_cases": [{
                "input_stdin": "",
                "expected_stdout": "PASS",
                "timeout_s": 10.0,
            }],
            "extra_files": {
                "_seccodeplt_harness.py": "print('PASS')\n",
            },
            "entry_module": "_seccodeplt_harness",
            "compile_flags": [],
        },
        "metadata": {"seccodeplt_fn": "path_check"},
    }


def _write_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "eval.jsonl"
    p.write_text(json.dumps(_fixture_record()) + "\n")
    return p


def _import_script_module(name: str):
    """Import a scripts/<name>.py module via spec_from_file_location.

    Adds src to sys.path so the script's `from secure_code_rl_ictai...`
    imports resolve.
    """
    import importlib.util

    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    script_path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_test_spec_complete(prompts, source_name: str) -> None:
    assert len(prompts) == 1, f"{source_name}: expected 1 prompt, got {len(prompts)}"
    prompt = prompts[0]
    ts = prompt.test_spec
    assert ts.test_cases and ts.test_cases[0].expected_stdout == "PASS", (
        f"{source_name}: TestCase.expected_stdout missing or wrong"
    )
    assert ts.entry_module == "_seccodeplt_harness", (
        f"{source_name}: TestSpec.entry_module was dropped "
        f"(got {ts.entry_module!r})"
    )
    assert "_seccodeplt_harness.py" in ts.extra_files, (
        f"{source_name}: TestSpec.extra_files was dropped "
        f"(got keys {list(ts.extra_files.keys())})"
    )


def test_run_baseline_sweep_load_prompts_preserves_test_spec(tmp_path: Path):
    eval_jsonl = _write_fixture(tmp_path)
    mod = _import_script_module("run_baseline_sweep")
    prompts = mod._load_prompts(eval_jsonl)
    _assert_test_spec_complete(prompts, "run_baseline_sweep")



def test_train_method_load_prompts_preserves_test_spec(tmp_path: Path):
    eval_jsonl = _write_fixture(tmp_path)
    mod = _import_script_module("train_method")
    prompts = mod._load_prompts(eval_jsonl, n=0)
    _assert_test_spec_complete(prompts, "train_method")
