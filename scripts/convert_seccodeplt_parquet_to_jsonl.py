#!/usr/bin/env python3
"""Convert SecCodePLT's HuggingFace parquet to the JSONL schema
SecCodePltAdapter consumes.

SecCodePLT (Yang et al., NeurIPS 2025; HF dataset `Virtue-AI-HUB/SecCodePLT`)
ships as a parquet file at `data/insecure_coding-00000-of-00001.parquet`.
The columns vary in capitalization across snapshots; we normalize on read.

Output JSONL schema (one record per line) matches
`docs/data_prep_spec.md` §4.3 and `SecCodePltAdapter`:

    {
        "id": "...",
        "category": "CWE-89",
        "language": "python",
        "task_description": "...",
        "signature": "...",
        "tests": [
            {"stdin": "...", "expected_stdout": "...", "timeout_s": 5.0},
            ...
        ],
        "metadata": {<source-specific extras>}
    }

The script does NOT execute the unit tests; it just transcribes them
into the simplified stdin/expected_stdout shape. SecCodePLT's native
unit tests are richer (function-call assertions); for v0.1 we either
leave `tests` empty (when no conversion is obvious) or attempt a
best-effort lowering. The reliability oracle's SecCodePLT execution
path (deferred) is the authoritative test runner; this converter exists
so the dataset-build pipeline has a uniform JSONL contract.

Usage:
    python scripts/convert_seccodeplt_parquet_to_jsonl.py \\
        --parquet data/raw/seccodeplt/insecure_coding.parquet \\
        --output data/raw/seccodeplt_jsonl/insecure_coding.jsonl \\
        --languages python c c++
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("convert_seccodeplt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Column name aliases we accept (real SecCodePLT snapshots have used
# different capitalization across releases). Each entry maps canonical
# output key → list of plausible input column names; we pick the first
# present.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "ID", "Id", "task_id"),
    "category": ("CWE_ID", "cwe_id", "category", "CWE", "cwe"),
    "language": ("language", "Language", "lang"),
    "task_description": (
        "task_description", "description", "task", "Task_Description",
        "prompt_description",
    ),
    "signature": ("signature", "prompt", "function_signature", "func_sig"),
    "ground_truth": ("ground_truth", "fixed_code", "secure_code", "GroundTruth"),
    "vulnerable_code": ("vulnerable_code", "insecure_code", "VulnerableCode"),
    "capability": ("capability", "Capability"),
    "unittest": ("unittest", "tests", "test_cases", "TestCases"),
    "install_requires": ("install_requires", "InstallRequires"),
}


def _pick(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for name in aliases:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _stringify(val: Any) -> str:
    """Coerce arbitrary parquet values to a stable string.

    SecCodePLT's `task_description`, `ground_truth`, and `unittest` are
    dict-typed (nested structs); numpy strings need decoding; bytes need
    decoding too. For dicts we JSON-serialize so the JSONL is loadable
    later without surprises.
    """
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace")
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True, default=str)
    if isinstance(val, (list, tuple)):
        # numpy ndarray or pyarrow ListArray-derived; serialize as JSON.
        return json.dumps(list(val), sort_keys=True, default=str)
    # Catch numpy scalars (np.bool_, np.str_, np.int64, ...).
    return str(val)


def _stringify_for_prompt(val: Any) -> str:
    """Coerce to a string suitable for the prompt body.

    SecCodePLT's `task_description` is a dict like
    {"arguments": "...", "description": "...", "examples": "..."}. We
    join the human-readable subfields when present.
    """
    if isinstance(val, dict):
        parts = []
        for key in ("description", "task_description", "arguments", "examples"):
            sub = val.get(key)
            if isinstance(sub, str) and sub.strip():
                parts.append(sub.strip())
        if parts:
            return "\n\n".join(parts)
    return _stringify(val)


def _normalize_cwe_id(raw: Any) -> str:
    """SecCodePLT stores CWE as a bare numeric string ('22'); we want 'CWE-22'."""
    s = _stringify(raw).strip()
    if not s:
        return ""
    if s.upper().startswith("CWE-"):
        return s.upper()
    if s.isdigit():
        return f"CWE-{s}"
    # Already in some other form ('CWE_22', 'cwe-22'); best-effort.
    return f"CWE-{s.lstrip('CWEcwe-_')}"


def _normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    """Map a raw parquet row to the SecCodePltAdapter JSONL schema."""
    record: dict[str, Any] = {}

    raw_id = _pick(row, _COLUMN_ALIASES["id"])
    record["id"] = _stringify(raw_id)

    record["category"] = _normalize_cwe_id(_pick(row, _COLUMN_ALIASES["category"]))

    language = _pick(row, _COLUMN_ALIASES["language"])
    if language is None:
        # SecCodePLT v0.1: language column absent; dataset is Python-only.
        record["language"] = "python"
    else:
        record["language"] = _stringify(language).lower()

    desc = _pick(row, _COLUMN_ALIASES["task_description"])
    record["task_description"] = _stringify_for_prompt(desc).strip()

    sig = _pick(row, _COLUMN_ALIASES["signature"])
    record["signature"] = _stringify(sig).strip()

    # Test-spec extraction. SecCodePLT's native unittest is a dict like:
    #   {"setup": "import re", "testcases": "<python source defining
    #    a `testcases` dict with capability/safety keys>"}
    # We translate this into:
    #   - One TestCase with input_stdin="", expected_stdout="PASS",
    #     timeout_s based on capability+safety count.
    #   - extra_files["_seccodeplt_harness.py"]: a harness that imports
    #     the model's snippet.py, applies setup, evaluates the
    #     `testcases` dict, and prints "PASS" iff every (args, expected)
    #     pair matches.
    #   - entry_module="_seccodeplt_harness": RealOracle runs the harness
    #     instead of the snippet directly.
    unittest = _pick(row, _COLUMN_ALIASES["unittest"])
    record["tests"] = []

    metadata: dict[str, Any] = {}
    for key in ("capability", "ground_truth", "vulnerable_code", "install_requires"):
        val = _pick(row, _COLUMN_ALIASES[key])
        if val is not None:
            metadata[key] = _stringify(val)
    if unittest is not None:
        metadata["unittest_raw"] = _stringify(unittest)

        # Attempt the harness translation. If unittest is a dict with the
        # expected shape, emit the harness + one TestCase + entry_module.
        ground_truth = _pick(row, _COLUMN_ALIASES["ground_truth"])
        harness, fn_name = _build_harness(
            unittest, record.get("signature", ""), ground_truth
        )
        if harness is not None:
            record["tests"] = [{
                "stdin": "",
                "expected_stdout": "PASS",
                "timeout_s": 10.0,
            }]
            record["extra_files"] = {"_seccodeplt_harness.py": harness}
            record["entry_module"] = "_seccodeplt_harness"
            metadata["seccodeplt_fn"] = fn_name

    # Also carry rule/use_rule (SecCodePLT v0.1 specific columns) if present.
    for extra in ("rule", "use_rule"):
        if extra in row and row[extra] is not None:
            metadata[extra] = _stringify(row[extra])
    record["metadata"] = metadata

    return record


_HARNESS_TEMPLATE = '''"""SecCodePLT harness.

Loads the model's snippet.py as a module, executes the dataset's `setup`
block in that module's namespace (so names defined there are visible to
the snippet's functions), evaluates the `testcases` dict
({{"capability": [(args, expected), ...], "safety": [...]}}), and prints
"PASS" iff every pair matches.

The harness is generated by scripts/convert_seccodeplt_parquet_to_jsonl.py;
edit there, not here.
"""

import sys
import importlib.util
import traceback


def _load_snippet():
    spec = importlib.util.spec_from_file_location("snippet", "snippet.py")
    if spec is None or spec.loader is None:
        print("FAIL: cannot load snippet.py")
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("FAIL: snippet.py raised during import")
        sys.exit(1)
    return mod


def main():
    snippet = _load_snippet()

    # Dataset setup: exec into the snippet module's namespace so that names
    # like ALLOWED_COMMANDS defined in setup are visible to functions inside
    # the snippet. This matches SecCodePLT's evaluation model where setup
    # provides the runtime context the user-code expects.
    setup_src = {setup_src!r}
    if setup_src.strip():
        try:
            exec(compile(setup_src, "<seccodeplt-setup>", "exec"),
                 snippet.__dict__, snippet.__dict__)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            print("FAIL: setup raised")
            sys.exit(1)

    # The dataset's testcases definition: a string of Python source that
    # defines `testcases = {{"capability": [...], "safety": [...]}}`.
    testcases_src = {testcases_src!r}
    _ns = {{}}
    try:
        exec(compile(testcases_src, "<seccodeplt-testcases>", "exec"), _ns, _ns)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("FAIL: testcases source raised")
        sys.exit(1)
    testcases = _ns.get("testcases", {{}})

    # Locate the target function. Prefer the dataset-supplied name; fall
    # back to the first non-underscore callable in the snippet.
    fn_name_hint = {function_name_lit!r}
    fn = getattr(snippet, fn_name_hint, None) if fn_name_hint else None
    if fn is None:
        for name in dir(snippet):
            if name.startswith("_"):
                continue
            obj = getattr(snippet, name)
            if callable(obj) and not isinstance(obj, type):
                fn = obj
                break
    if fn is None:
        print("FAIL: no callable found in snippet")
        sys.exit(1)

    ok = True
    failures = []
    for group in ("capability", "safety"):
        for pair in testcases.get(group, []):
            try:
                args, expected = pair
            except Exception:
                continue
            try:
                if isinstance(args, dict):
                    actual = fn(**args)
                elif isinstance(args, (list, tuple)):
                    actual = fn(*args)
                else:
                    actual = fn(args)
            except Exception:
                actual = "<exception>"
            if actual != expected:
                ok = False
                failures.append((group, args, expected, actual))

    if ok:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        for g, a, e, x in failures[:3]:
            print(f"  {{g}}: args={{a!r}} expected={{e!r}} actual={{x!r}}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


def _build_harness(
    unittest_val: Any, signature: str, ground_truth: Any = None
) -> tuple[str | None, str | None]:
    """Translate a SecCodePLT unittest dict into a harness script + the
    inferred function name. Returns (None, None) if the unittest doesn't
    have the expected shape (we still keep `unittest_raw` in metadata for
    later inspection)."""
    if not isinstance(unittest_val, dict):
        return None, None
    setup = unittest_val.get("setup", "")
    testcases_src = unittest_val.get("testcases", "")
    if not isinstance(setup, str) or not isinstance(testcases_src, str):
        return None, None
    if not testcases_src.strip():
        return None, None

    # Infer the function name. Try the signature first; if empty, fall back
    # to ground_truth.code_before which often contains the actual `def`.
    fn_name = _infer_fn_name(signature)
    if not fn_name and isinstance(ground_truth, dict):
        fn_name = _infer_fn_name(str(ground_truth.get("code_before", "")))

    harness = _HARNESS_TEMPLATE.format(
        setup_src=setup,
        testcases_src=testcases_src,
        function_name_lit=fn_name or "",
    )
    return harness, fn_name


def _infer_fn_name(signature: str) -> str | None:
    """Best-effort extraction of the function name from a Python signature
    string. Returns None if no name is found."""
    if not signature:
        return None
    m = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", signature)
    if m:
        return m.group(1)
    return None


def convert(args: argparse.Namespace) -> int:
    if not args.parquet.exists():
        print(f"parquet does not exist: {args.parquet}", file=sys.stderr)
        return 2

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "pyarrow is required; install with `pip install pyarrow`",
            file=sys.stderr,
        )
        return 2

    table = pq.read_table(args.parquet)
    df = table.to_pandas() if hasattr(table, "to_pandas") else None
    if df is None:
        print("failed to convert parquet to pandas", file=sys.stderr)
        return 2

    logger.info(
        "loaded %d rows from %s; columns=%s",
        len(df), args.parquet, list(df.columns),
    )

    languages_lower = {l.lower() for l in args.languages} if args.languages else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.output, "w") as fh:
        for _, row in df.iterrows():
            rec = _normalize_record(row.to_dict())
            if languages_lower is not None and rec["language"] not in languages_lower:
                continue
            # Skip records with no usable text.
            if not rec["task_description"] and not rec["signature"]:
                continue
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            n_written += 1

    logger.info("wrote %d records to %s", n_written, args.output)
    if n_written == 0:
        print(
            "WARNING: 0 records written. Check column-alias coverage in "
            "this script against the parquet's actual schema:\n"
            f"  schema columns: {list(df.columns)}",
            file=sys.stderr,
        )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Language whitelist (lowercased on compare). Omit for no filter.",
    )
    args = p.parse_args()
    sys.exit(convert(args))


if __name__ == "__main__":
    main()
