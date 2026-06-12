#!/usr/bin/env python3
"""Run the eval harness against multiple baselines and aggregate the results.

Config JSON shape:

    {
      "baselines": [
        {"name": "qwen-1.5b", "baseline": "qwen2.5-coder-1.5b"},
        {"name": "mock-fixture", "mock_from_jsonl": "responses.jsonl"},
        {"name": "seccoderx",  "baseline": "seccoderx"}    // cite-only -> error
      ]
    }

Each entry is either a real baseline (`baseline:` name from the registry)
or a mock (`mock_from_jsonl:` path). The sweep:
  - Loads the eval set once.
  - For each entry, runs the eval harness and writes a report directory.
  - Catches NotImplementedError (cite-only or unrunnable baselines) and
    records `status: error` in the summary; does NOT propagate the failure.
  - Writes `output/sweep_summary.json` with per-baseline metrics for the
    paper-table generator.

Usage:
    python scripts/run_baseline_sweep.py \\
        --eval-jsonl data/build/v0.1/eval_prompts.jsonl \\
        --config sweep.json \\
        --output runs/sweep_v0/
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from secure_code_rl_ictai.data_prep.schema import Prompt
from secure_code_rl_ictai.data_prep import (
    normalize_cwe,
    normalize_language,
)
from secure_code_rl_ictai.eval import (
    EvalHarness,
    MockModel,
    SamplingConfig,
    get_baseline_spec,
)
from secure_code_rl_ictai.reward import (
    MockOracle,
    RealOracle,
    ReliabilitySignals,
    RewardCalculator,
    RewardConfig,
    RewardPipeline,
    TestCase,
    TestSpec,
)
from secure_code_rl_ictai.sast.adapters.bandit import BanditAdapter
from secure_code_rl_ictai.sast.adapters.codeql import CodeQLAdapter
from secure_code_rl_ictai.sast.adapters.cppcheck import CppcheckAdapter
from secure_code_rl_ictai.sast.adapters.semgrep import SemgrepAdapter
from secure_code_rl_ictai.sast.models import ToolName
from secure_code_rl_ictai.sast.runner import MockAdapter, SastRunner
from secure_code_rl_ictai.sast.severity import SeveritySource


def _load_prompts(path: Path) -> list[Prompt]:
    if not path.exists():
        print(f"eval JSONL does not exist: {path}", file=sys.stderr)
        sys.exit(2)
    prompts: list[Prompt] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lang = normalize_language(rec["language"])
            ts_raw = rec.get("test_spec", {})
            test_spec = TestSpec(
                language=lang,
                test_cases=[
                    TestCase(
                        input_stdin=str(tc.get("input_stdin", "")),
                        expected_stdout=str(tc.get("expected_stdout", "")),
                        timeout_s=float(tc.get("timeout_s", 5.0)),
                    )
                    for tc in ts_raw.get("test_cases", [])
                ],
                # Preserve harness wireup (SecCodePLT/CWEval). Dropping
                # these causes RealOracle to run snippet.py directly, which
                # never matches the expected_stdout='PASS' contract -> all
                # SecCodePLT / CWEval prompts silently get tests_passed=0.
                extra_files=dict(ts_raw.get("extra_files") or {}),
                entry_module=ts_raw.get("entry_module"),
                prefix_text=ts_raw.get("prefix_text"),
                suffix_text=ts_raw.get("suffix_text"),
            )
            prompts.append(
                Prompt(
                    id=rec["id"],
                    source=rec.get("source", "unknown"),
                    language=lang,
                    target_cwe=normalize_cwe(rec["target_cwe"]),
                    prompt_text=rec["prompt_text"],
                    test_spec=test_spec,
                    task_signature=rec.get("task_signature"),
                    metadata=rec.get("metadata", {}),
                )
            )
    return prompts


def _load_mock(path: Path, name: str) -> MockModel:
    responses: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            responses[rec["prompt"]] = rec["completion"]
    return MockModel(name=name, responses=responses, default="")


def _build_pipeline(
    oracle_kind: str = "mock", compile_mode: str = "syntax_only",
) -> RewardPipeline:
    """Construct a RewardPipeline for the sweep.

    `oracle_kind`:
      - "mock":  MockOracle (always-pass), all MockAdapters (empty SARIF).
        Smoke-test the harness; R_total collapses to the trivial value.
      - "real":  RealOracle (compiles+runs code in a subprocess sandbox)
        + RealBanditAdapter for Python; remaining SAST tools stay
        MockAdapter until their real adapters land. R_security for
        Python prompts is now actually informed by SAST findings; for
        C/C++ it still collapses to no-findings.

    Requires for `oracle_kind="real"`: python3 on PATH (and clang,
    clang++ for C/C++ test_spec compilation), bandit + bandit-sarif-formatter
    importable in the venv. Per-prompt test_spec must be in the eval JSONL.
    """
    if oracle_kind == "real":
        # Resolve CLI binaries explicitly. On compute nodes the venv bin/
        # is not on PATH, so the default name lookups fail. Try
        # shutil.which first; fall back to sys.executable.parent.
        import shutil
        import sys as _sys

        def _resolve(name: str) -> str:
            via_path = shutil.which(name)
            if via_path:
                return via_path
            candidate = Path(_sys.executable).parent / name
            if candidate.exists():
                return str(candidate)
            return name  # let subprocess raise FileNotFoundError with a clear message

        bandit_adapter = BanditAdapter(bandit_binary=_resolve("bandit"))
        semgrep_adapter = SemgrepAdapter(semgrep_binary=_resolve("semgrep"))
        # CodeQL binary is installed at a fixed path on ROAR. If it's not
        # there, fall back to PATH lookup; if that fails too let it become
        # MockAdapter so the sweep can still run on hosts without CodeQL.
        import os
        codeql_bin = os.environ.get("CODEQL_BINARY") or _resolve("codeql")
        codeql_at_oss = "/storage/home/sss6371/work/oss/codeql-cli/codeql/codeql"
        if codeql_bin == "codeql" and Path(codeql_at_oss).exists():
            codeql_bin = codeql_at_oss
        if codeql_bin != "codeql" or shutil.which("codeql"):
            codeql_adapter = CodeQLAdapter(codeql_binary=codeql_bin)
        else:
            codeql_adapter = MockAdapter(ToolName.CODEQL, {"runs": []})

        # Cppcheck: built from source at /storage/.../cppcheck_build/. If
        # the binary is on PATH or at the known location use it, else fall
        # back to MockAdapter.
        cppcheck_bin = os.environ.get("CPPCHECK_BINARY") or _resolve("cppcheck")
        cppcheck_at_build = "/storage/home/sss6371/work/cppcheck_build/cppcheck-2.18.0/cppcheck"
        if cppcheck_bin == "cppcheck" and Path(cppcheck_at_build).exists():
            cppcheck_bin = cppcheck_at_build
        if cppcheck_bin != "cppcheck" or shutil.which("cppcheck"):
            cppcheck_adapter = CppcheckAdapter(cppcheck_binary=cppcheck_bin)
        else:
            cppcheck_adapter = MockAdapter(ToolName.CPPCHECK, {"runs": []})
    else:
        bandit_adapter = MockAdapter(ToolName.BANDIT, {"runs": []})
        semgrep_adapter = MockAdapter(ToolName.SEMGREP, {"runs": []})
        codeql_adapter = MockAdapter(ToolName.CODEQL, {"runs": []})
        cppcheck_adapter = MockAdapter(ToolName.CPPCHECK, {"runs": []})

    adapters = {
        ToolName.CODEQL: codeql_adapter,
        ToolName.SEMGREP: semgrep_adapter,
        ToolName.BANDIT: bandit_adapter,
        ToolName.CPPCHECK: cppcheck_adapter,
    }
    runner = SastRunner(adapters)
    sev_src = SeveritySource(Path("data/nvdlib_cwe_medians.json"))
    calc = RewardCalculator(RewardConfig(alpha=1.0), sev_src)

    if oracle_kind == "real":
        # Resolve C/C++ compilers per host. ROAR has gcc/g++ but not clang;
        # other hosts may have either. Mirror the resolution logic in
        # scripts/train_method.py.
        import shutil as _sh
        c_cc = _resolve("clang") if _sh.which("clang") else _resolve("gcc")
        cpp_cc = _resolve("clang++") if _sh.which("clang++") else _resolve("g++")
        # Prefer sys.executable for python — that's the venv python, which
        # has pytest (needed by the CWEval harness). _resolve("python3")
        # would find /usr/bin/python3 on ROAR which lacks pytest.
        oracle = RealOracle(
            python_executable=sys.executable,
            c_compiler=c_cc,
            cpp_compiler=cpp_cc,
            compile_mode=compile_mode,
        )
    elif oracle_kind == "mock":
        oracle = MockOracle(
            ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1)
        )
    else:
        raise ValueError(f"unknown oracle_kind: {oracle_kind!r}")

    return RewardPipeline(
        oracle=oracle,
        sast_runner=runner,
        severity_source=sev_src,
        calculator=calc,
    )


def _run_one(
    entry: dict,
    prompts: list[Prompt],
    output: Path,
    sampling: SamplingConfig,
    oracle_kind: str = "mock",
    resume: bool = False,
    compile_mode: str = "syntax_only",
) -> dict:
    """Run the harness for one config entry. Returns a summary dict."""
    name = entry["name"]
    pipeline = _build_pipeline(oracle_kind=oracle_kind, compile_mode=compile_mode)
    harness = EvalHarness(pipeline=pipeline, sampling=sampling)

    try:
        if "mock_from_jsonl" in entry:
            model = _load_mock(Path(entry["mock_from_jsonl"]), name=name)
        elif "baseline" in entry:
            spec = get_baseline_spec(entry["baseline"])
            model = spec.factory()
        else:
            return {
                "status": "error",
                "error": "config entry needs either mock_from_jsonl or baseline",
            }

        # Stream per-prompt records to per_prompt_stream.jsonl so a SLURM
        # timeout doesn't lose all in-flight data. At completion we also
        # write the conventional per_prompt.jsonl via report.save().
        baseline_dir = output / name
        baseline_dir.mkdir(parents=True, exist_ok=True)
        stream_path = baseline_dir / "per_prompt_stream.jsonl"
        report = harness.evaluate(
            model, prompts, stream_path=stream_path, resume=resume,
        )
        report.save(output)
        return {
            "status": "ok",
            "aggregate": report.aggregate,
            "per_cwe": report.per_cwe,
            "diagnostics": report.diagnostics,
            "n_prompts": report.n_prompts,
        }
    except NotImplementedError as exc:
        return {"status": "error", "error": f"NotImplementedError: {exc}"}
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def run(args: argparse.Namespace) -> int:
    if not args.config.exists():
        print(f"config does not exist: {args.config}", file=sys.stderr)
        return 2
    config = json.loads(args.config.read_text())

    prompts = _load_prompts(args.eval_jsonl)
    args.output.mkdir(parents=True, exist_ok=True)

    sampling = SamplingConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    summary: dict = {
        "n_prompts": len(prompts),
        "sampling": {
            "temperature": sampling.temperature,
            "max_new_tokens": sampling.max_new_tokens,
            "seed": sampling.seed,
        },
        "oracle_kind": args.oracle_kind,
        "compile_mode": args.compile_mode,
        "baselines": {},
    }

    for entry in config.get("baselines", []):
        name = entry["name"]
        print(f"[sweep] running {name} ...", file=sys.stderr)
        result = _run_one(
            entry, prompts, args.output, sampling, args.oracle_kind,
            resume=args.resume_from_stream,
            compile_mode=args.compile_mode,
        )
        summary["baselines"][name] = result
        if result["status"] == "ok":
            print(
                f"[sweep]   {name}: ok "
                f"(func@1={result['aggregate']['func_at_1']:.3f}, "
                f"secure@1={result['aggregate']['secure_at_1']:.3f})",
                file=sys.stderr,
            )
        else:
            print(f"[sweep]   {name}: ERROR - {result['error']}", file=sys.stderr)

    (args.output / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-jsonl", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--oracle-kind",
        choices=("mock", "real"),
        default="mock",
        help="ReliabilityOracle kind. 'mock' = always-pass (smoke test); "
        "'real' = compiles+runs code in a subprocess sandbox (v0.1 MULTI_LLM table). "
        "Default 'mock'.",
    )
    p.add_argument(
        "--compile-mode",
        choices=("syntax_only", "link_and_run"),
        default="syntax_only",
        help="C/C++ Compile@1 judgement under --oracle-kind real. "
        "syntax_only = gcc -fsyntax-only (the paper's metric definition); "
        "link_and_run = link + execute (training reward path).",
    )
    p.add_argument(
        "--resume-from-stream",
        action="store_true",
        help="Append to per_prompt_stream.jsonl if it exists and skip "
             "prompts already recorded. Used to finish baselines that were "
             "SLURM-cancelled at the walltime cap. Without this flag, the "
             "stream is truncated and the run starts from prompt 0.",
    )
    args = p.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
