"""Rescore per-prompt eval streams with the current oracle.

Reads `per_prompt_stream.jsonl` files (which carry the model's stored
`completion` text plus prompt metadata), re-runs the real oracle on each
completion, and writes a fresh `aggregate.json` to a sibling directory.

This exists because the 2026-06-22 fix to `_eval_c_family` (compile-only
first, link+run only when main+tests are present) changes the `compiles`
field for C/C++ on every prior eval. Re-running the LLM is unnecessary
and prohibitive; we only need to re-execute the oracle.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/rescore_aggregates.py \\
        --eval-jsonl /scratch/.../build/v0.1.7/eval_prompts.jsonl \\
        --stream /scratch/.../sweeps/.../per_prompt_stream.jsonl \\
        --out /scratch/.../sweeps/.../aggregate_rescored.json

    # Or batch over a directory tree:
    PYTHONPATH=src .venv/bin/python scripts/rescore_aggregates.py \\
        --eval-jsonl /scratch/.../build/v0.1.7/eval_prompts.jsonl \\
        --sweeps-root /scratch/.../sweeps/

Per-prompt stream lookup is keyed by `prompt_id`; the eval JSONL supplies
the TestSpec (language, test cases, extra files, compile flags) so the
oracle behaves identically to a fresh run.

Output schema mirrors what `eval.harness` writes for `aggregate.json`:
  - n_prompts
  - aggregate.compile_at_1 = {value, n_numerator, n_denominator}
  - aggregate.secure_at_1__compiles (security only re-extracted from stored fields)
  - aggregate.func_sec_at_1__compiles_and_has_tests (likewise)
  - aggregate.func_sec_at_1__has_tests: the paper's Functional-Secure@1,
    whose denominator is the test-equipped subset of the eval set for
    every system (compile failures count as non-pass, not as dropped).
The security/func components are NOT recomputed: they come from stored
per_prompt fields (`target_cwe_present`, `tests_passed`). Only compile
status is recomputed. Functional pass means every test case passed,
matching `eval.metrics._func_predicate`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Repo's local secure_code_rl_ictai import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from secure_code_rl_ictai.eval.harness import _extract_code  # noqa: E402
from secure_code_rl_ictai.reward.reliability_oracle import (  # noqa: E402
    Language,
    RealOracle,
    TestCase,
    TestSpec,
)


def extract_code(text: str, language: str) -> str:
    """Same extractor the live harness uses (fences + base-model tail
    sanitization), so a rescore reproduces a fresh eval byte-for-byte."""
    if not isinstance(text, str):
        return ""
    return _extract_code(text, language)


_LANG_MAP = {
    "python": Language.PYTHON,
    "py": Language.PYTHON,
    "c": Language.C,
    "cpp": Language.CPP,
    "c++": Language.CPP,
}


def _testspec_from_eval_row(row: dict) -> TestSpec | None:
    """Reconstruct a TestSpec from an eval prompts.jsonl row."""
    lang_str = (row.get("language") or "").lower()
    lang = _LANG_MAP.get(lang_str)
    if lang is None:
        return None
    raw_spec = row.get("test_spec") or {}
    test_cases = []
    for tc in raw_spec.get("test_cases", []) or []:
        test_cases.append(TestCase(
            input_stdin=tc.get("input_stdin", "") or "",
            expected_stdout=tc.get("expected_stdout", "") or "",
            timeout_s=float(tc.get("timeout_s", 5.0)),
        ))
    return TestSpec(
        language=lang,
        test_cases=test_cases,
        extra_files=raw_spec.get("extra_files") or {},
        compile_flags=raw_spec.get("compile_flags") or [],
        entry_module=raw_spec.get("entry_module"),
        prefix_text=raw_spec.get("prefix_text"),
        suffix_text=raw_spec.get("suffix_text"),
    )


@dataclass
class RescoreResult:
    prompt_id: str
    language: str
    new_compiles: bool
    old_compiles: bool
    secure: bool
    has_tests: bool
    tests_passed_old: int
    tests_total: int


def _rescore_one(prompt_id: str, completion: str, spec_json: str,
                 old_compiles: bool, secure: bool,
                 tests_passed_old: int, has_tests: bool) -> RescoreResult:
    """Worker for ProcessPoolExecutor. Spec is passed as JSON to be pickle-safe."""
    spec_dict = json.loads(spec_json)
    test_cases = [
        TestCase(input_stdin=tc.get("input_stdin", "") or "",
                 expected_stdout=tc.get("expected_stdout", "") or "",
                 timeout_s=float(tc.get("timeout_s", 5.0)))
        for tc in spec_dict.get("test_cases", []) or []
    ]
    spec = TestSpec(
        language=Language(spec_dict["language"]),
        test_cases=test_cases,
        extra_files=spec_dict.get("extra_files") or {},
        compile_flags=spec_dict.get("compile_flags") or [],
        entry_module=spec_dict.get("entry_module"),
        prefix_text=spec_dict.get("prefix_text"),
        suffix_text=spec_dict.get("suffix_text"),
    )
    oracle = RealOracle(compile_mode="syntax_only")
    code = extract_code(completion, spec.language.value)
    sig = oracle.evaluate(code, spec)
    return RescoreResult(
        prompt_id=prompt_id,
        language=spec.language.value,
        new_compiles=bool(sig.compiles),
        old_compiles=bool(old_compiles),
        secure=bool(secure),
        has_tests=bool(has_tests),
        tests_passed_old=int(tests_passed_old),
        tests_total=len(spec.test_cases),
    )


def rescore_stream(stream_path: Path, eval_index: dict[str, dict],
                   workers: int) -> dict:
    """Run oracle on every row in per_prompt_stream.jsonl. Returns aggregate dict."""
    rows = []
    with stream_path.open() as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Build worker args, skipping rows whose prompt_id is not in the eval index.
    work_items = []
    missing = 0
    for r in rows:
        pid = r.get("prompt_id")
        if not pid:
            missing += 1
            continue
        eval_row = eval_index.get(pid)
        if eval_row is None:
            missing += 1
            continue
        spec = _testspec_from_eval_row(eval_row)
        if spec is None:
            missing += 1
            continue
        spec_json = json.dumps({
            "language": spec.language.value,
            "test_cases": [
                {"input_stdin": tc.input_stdin,
                 "expected_stdout": tc.expected_stdout,
                 "timeout_s": tc.timeout_s}
                for tc in spec.test_cases
            ],
            "extra_files": spec.extra_files,
            "compile_flags": spec.compile_flags,
            "entry_module": spec.entry_module,
            "prefix_text": spec.prefix_text,
            "suffix_text": spec.suffix_text,
        })
        completion = r.get("completion", "")
        old_compiles = bool(r.get("compiles", False))
        secure = bool(r.get("secure", False))
        tests_passed_old = int(r.get("tests_passed", 0) or 0)
        has_tests = len(spec.test_cases) > 0
        work_items.append((pid, completion, spec_json, old_compiles, secure,
                           tests_passed_old, has_tests))

    print(f"  [{stream_path.parent.name}] {len(work_items)} prompts to rescore "
          f"({missing} skipped: no eval-jsonl match)", file=sys.stderr, flush=True)

    results: list[RescoreResult] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_rescore_one, *args) for args in work_items]
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if i % 200 == 0:
                    print(f"    {i}/{len(work_items)} done",
                          file=sys.stderr, flush=True)
    else:
        for i, args in enumerate(work_items, 1):
            results.append(_rescore_one(*args))
            if i % 100 == 0:
                print(f"    {i}/{len(work_items)} done",
                      file=sys.stderr, flush=True)

    # ---------- aggregates ----------
    n = len(results)
    n_compile_new = sum(1 for r in results if r.new_compiles)
    n_compile_old = sum(1 for r in results if r.old_compiles)
    n_secure_given_cmp_new = sum(1 for r in results
                                 if r.new_compiles and r.secure)
    has_test_pool_new = sum(1 for r in results
                            if r.new_compiles and r.has_tests)

    def _func_pass(r: RescoreResult) -> bool:
        # All test cases passed (eval.metrics._func_predicate). A row that
        # did not compile has tests_passed_old == 0 and therefore fails.
        return r.has_tests and r.tests_passed_old >= r.tests_total > 0

    n_funcsec_new = sum(1 for r in results
                        if r.new_compiles and r.has_tests
                        and r.secure and _func_pass(r))
    # Fixed-denominator variant: every test-equipped prompt is in the
    # denominator regardless of compile status.
    has_test_pool_fixed = sum(1 for r in results if r.has_tests)
    n_funcsec_fixed = sum(1 for r in results
                          if r.new_compiles and r.secure and _func_pass(r))

    by_lang_compile_new = defaultdict(lambda: [0, 0])
    by_lang_compile_old = defaultdict(lambda: [0, 0])
    for r in results:
        by_lang_compile_new[r.language][1] += 1
        if r.new_compiles:
            by_lang_compile_new[r.language][0] += 1
        by_lang_compile_old[r.language][1] += 1
        if r.old_compiles:
            by_lang_compile_old[r.language][0] += 1

    def _metric(num, den):
        return {
            "value": (num / den) if den else 0.0,
            "n_numerator": num,
            "n_denominator": den,
        }

    return {
        "n_prompts": n,
        "rescored_with_fixed_oracle": True,
        "rescored_from_stream": str(stream_path),
        "aggregate": {
            "compile_at_1": _metric(n_compile_new, n),
            "compile_at_1_old": _metric(n_compile_old, n),
            "secure_at_1__compiles": _metric(n_secure_given_cmp_new,
                                             n_compile_new),
            "func_sec_at_1__compiles_and_has_tests": _metric(
                n_funcsec_new, has_test_pool_new),
            "func_sec_at_1__has_tests": _metric(
                n_funcsec_fixed, has_test_pool_fixed),
        },
        "per_language": {
            lang: {
                "compile_at_1": _metric(by_lang_compile_new[lang][0],
                                        by_lang_compile_new[lang][1]),
                "compile_at_1_old": _metric(by_lang_compile_old[lang][0],
                                            by_lang_compile_old[lang][1]),
            }
            for lang in by_lang_compile_new
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-jsonl", type=Path, required=True,
                   help="Path to v0.1.7 eval_prompts.jsonl (source of TestSpecs)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--stream", type=Path,
                     help="Single per_prompt_stream.jsonl to rescore")
    src.add_argument("--sweeps-root", type=Path,
                     help="Walk this dir, rescore every per_prompt_stream.jsonl found")
    p.add_argument("--out-suffix", type=str, default="aggregate_rescored.json",
                   help="Output filename written next to each stream")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Only rescore the first N streams (for testing)")
    args = p.parse_args()

    print(f"Loading eval prompts index from {args.eval_jsonl} ...",
          file=sys.stderr, flush=True)
    eval_index: dict[str, dict] = {}
    with args.eval_jsonl.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("id") or row.get("prompt_id")
            if pid:
                eval_index[pid] = row
    print(f"  loaded {len(eval_index)} prompts", file=sys.stderr, flush=True)

    if args.stream:
        streams = [args.stream]
    else:
        streams = sorted(args.sweeps_root.rglob("per_prompt_stream.jsonl"))

    if args.limit:
        streams = streams[: args.limit]

    print(f"Rescoring {len(streams)} stream files with {args.workers} workers",
          file=sys.stderr, flush=True)

    for sp in streams:
        out_path = sp.parent / args.out_suffix
        try:
            agg = rescore_stream(sp, eval_index, workers=args.workers)
        except Exception as e:
            print(f"  ERROR on {sp}: {e}", file=sys.stderr, flush=True)
            continue
        out_path.write_text(json.dumps(agg, indent=2))
        old_cmp = agg["aggregate"]["compile_at_1_old"]
        new_cmp = agg["aggregate"]["compile_at_1"]
        print(f"  wrote {out_path}", file=sys.stderr, flush=True)
        print(f"    compile_at_1: {old_cmp['n_numerator']}/{old_cmp['n_denominator']} "
              f"({old_cmp['value']*100:.1f}%) -> "
              f"{new_cmp['n_numerator']}/{new_cmp['n_denominator']} "
              f"({new_cmp['value']*100:.1f}%)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
