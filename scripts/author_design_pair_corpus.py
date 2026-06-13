"""Author the CWE-306 + CWE-862 design-pair corpus.

Produces ~50 CWE-862 (missing authorization) + ~50 CWE-306 (missing
authentication) patterns. Each pattern combines three approaches:

  1. Template-seeded (Approach 3): structural scaffold from a
     SecCodePLT/SecurityEval-style framework + endpoint shape (Flask,
     FastAPI, Django, Express, plain-Python CLI, etc.).
  2. Greenfield diversified (Approach 1): the scenario is varied per
     pattern — different domain (admin/banking/healthcare/CMS/CI), role
     model (role-based, ABAC, token-scope, owner-only), and operation
     (read/update/delete/grant) — so no two patterns are syntactic
     copies of each other.
  3. Runtime-verified (Approach 2): each pattern emits an executable
     reference vulnerable_completion + secure_completion + stdin/stdout
     test cases. The builder runs both completions against the probe;
     the vulnerable version MUST fail (auth bypass succeeds) and the
     secure version MUST pass (auth bypass blocked). Patterns that
     don't verify are dropped from the output JSONL.

Output schema matches data_prep.schema.Prompt:
    {
      id, source="design_pair_authored",
      language, target_cwe,
      prompt_text,
      test_spec: {language, test_cases: [{input_stdin, expected_stdout, timeout_s}]},
      task_signature, metadata
    }

Plus, for the SFT warm-start consumer:
    metadata.vulnerable_completion   (drops would go here)
    metadata.secure_completion       (extract_sft_pairs.py's target)
    metadata.template_source         (which Approach-3 scaffold)
    metadata.verified_at_authoring   (runtime-probe pass/fail at build)
    metadata.approach                (always "1+2+3 combined")

Usage:
    PYTHONPATH=src .venv/bin/python scripts/author_design_pair_corpus.py \\
        --output data/v0.1.5.1/design_pair_patterns.jsonl \\
        --n-per-cwe 50
"""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from hashlib import md5
from pathlib import Path

# Sentinel strings the test cases compare against. Each pattern's
# completion must follow this output convention so the stdin/stdout
# oracle can verify auth without parsing HTTP responses.
ALLOW = "ALLOW"
DENY = "DENY"


@dataclass
class AuthorScenario:
    """An Approach-3 template + Approach-1 diversification slot."""
    template_source: str       # which scaffold inspired this pattern
    framework: str             # flask | fastapi | django | express | cli
    domain: str                # admin | banking | healthcare | cms | ci | files
    operation: str             # read | update | delete | grant | reset
    role_model: str            # role_based | abac | token_scope | owner_only
    target_cwe: str            # CWE-862 | CWE-306
    language: str = "python"


# ---- Approach-3 template registry ----
#
# Each scenario lists (framework, domain, operation, role_model). We
# pair each with a CWE intent. The Cartesian space is much larger than
# 100; we sample for diversity below.

SCENARIOS_CWE_862 = [
    # (template_source, framework, domain, operation, role_model)
    ("seccodeplt:admin-delete",        "flask",   "admin",      "delete",  "role_based"),
    ("seccodeplt:admin-update",        "flask",   "admin",      "update",  "role_based"),
    ("seccodeplt:user-profile",        "flask",   "cms",        "update",  "owner_only"),
    ("seccodeplt:banking-transfer",    "flask",   "banking",    "transfer","abac"),
    ("seccodeplt:healthcare-records",  "fastapi", "healthcare", "read",    "abac"),
    ("seccodeplt:permission-grant",    "fastapi", "admin",      "grant",   "role_based"),
    ("seccodeplt:file-download",       "fastapi", "files",      "read",    "owner_only"),
    ("seccodeplt:file-delete",         "fastapi", "files",      "delete",  "owner_only"),
    ("securityeval:cms-edit",          "django",  "cms",        "update",  "owner_only"),
    ("securityeval:cms-publish",       "django",  "cms",        "grant",   "role_based"),
    ("securityeval:ci-trigger",        "django",  "ci",         "update",  "token_scope"),
    ("securityeval:ci-deploy",         "django",  "ci",         "delete",  "token_scope"),
    ("seccodeplt:account-close",       "express", "banking",    "delete",  "owner_only"),
    ("seccodeplt:account-statement",   "express", "banking",    "read",    "owner_only"),
    ("seccodeplt:settings-modify",     "express", "cms",        "update",  "role_based"),
    ("seccodeplt:cli-vault-read",      "cli",     "files",      "read",    "owner_only"),
    ("seccodeplt:cli-vault-delete",    "cli",     "files",      "delete",  "owner_only"),
    ("seccodeplt:cli-admin-grant",     "cli",     "admin",      "grant",   "role_based"),
    ("securityeval:hospital-prescribe","fastapi", "healthcare", "update",  "abac"),
    ("securityeval:hospital-view",     "fastapi", "healthcare", "read",    "abac"),
    ("seccodeplt:tx-cancel",           "express", "banking",    "delete",  "abac"),
    ("seccodeplt:role-revoke",         "flask",   "admin",      "delete",  "role_based"),
    ("seccodeplt:file-share",          "fastapi", "files",      "grant",   "owner_only"),
    ("seccodeplt:doc-archive",         "django",  "cms",        "update",  "role_based"),
    ("seccodeplt:ci-secrets-rotate",   "django",  "ci",         "update",  "token_scope"),
]

SCENARIOS_CWE_306 = [
    # CWE-306 is missing AUTHENTICATION (no login check at all),
    # vs CWE-862 which is missing AUTHORIZATION (logged in, wrong role).
    ("seccodeplt:public-debug",        "flask",   "admin",      "read",    "any_authenticated"),
    ("seccodeplt:metrics-endpoint",    "flask",   "admin",      "read",    "any_authenticated"),
    ("seccodeplt:health-check",        "fastapi", "admin",      "read",    "any_authenticated"),
    ("seccodeplt:user-list",           "fastapi", "cms",        "read",    "any_authenticated"),
    ("seccodeplt:internal-api",        "django",  "ci",         "update",  "any_authenticated"),
    ("securityeval:internal-status",   "django",  "ci",         "read",    "any_authenticated"),
    ("seccodeplt:webhook-endpoint",    "flask",   "ci",         "update",  "token_required"),
    ("seccodeplt:job-queue",           "fastapi", "ci",         "update",  "any_authenticated"),
    ("seccodeplt:export-endpoint",     "express", "cms",        "read",    "any_authenticated"),
    ("seccodeplt:public-userinfo",     "flask",   "cms",        "read",    "any_authenticated"),
    ("seccodeplt:db-stats",            "fastapi", "admin",      "read",    "any_authenticated"),
    ("seccodeplt:cache-purge",         "django",  "admin",      "delete",  "any_authenticated"),
    ("seccodeplt:rate-limit-bypass",   "express", "admin",      "update",  "any_authenticated"),
    ("seccodeplt:legacy-shim",         "flask",   "cms",        "update",  "any_authenticated"),
    ("seccodeplt:cron-trigger",        "django",  "ci",         "update",  "any_authenticated"),
    ("seccodeplt:cli-config-read",     "cli",     "admin",      "read",    "any_authenticated"),
    ("seccodeplt:cli-status-server",   "cli",     "admin",      "read",    "any_authenticated"),
    ("seccodeplt:cli-deploy-hook",     "cli",     "ci",         "update",  "any_authenticated"),
    ("securityeval:patient-portal",    "fastapi", "healthcare", "read",    "any_authenticated"),
    ("securityeval:claim-submit",      "django",  "healthcare", "update",  "any_authenticated"),
    ("seccodeplt:tx-history",          "express", "banking",    "read",    "any_authenticated"),
    ("seccodeplt:atm-withdraw",        "express", "banking",    "update",  "any_authenticated"),
    ("seccodeplt:tellerlog-read",      "flask",   "banking",    "read",    "any_authenticated"),
    ("seccodeplt:file-upload-public",  "fastapi", "files",      "update",  "any_authenticated"),
    ("seccodeplt:share-link",          "django",  "files",      "grant",   "any_authenticated"),
]


def _make_scenarios(target_cwe: str, n: int) -> list[AuthorScenario]:
    """Pick n scenarios with diversification across (framework, domain, role)."""
    pool = SCENARIOS_CWE_862 if target_cwe == "CWE-862" else SCENARIOS_CWE_306
    # Each scenario template is reused once if n > len(pool); diversify
    # within each reuse by appending a salt to the scenario_id below.
    scenarios = []
    for i in range(n):
        tmpl, framework, domain, op, role = pool[i % len(pool)]
        scenarios.append(AuthorScenario(
            template_source=tmpl,
            framework=framework,
            domain=domain,
            operation=op,
            role_model=role,
            target_cwe=target_cwe,
        ))
    return scenarios


# ---- Approach-1 pattern generator ----
#
# Each generator builds the (vulnerable, secure, prompt, probe) tuple
# from an AuthorScenario, with surface-level details varied per
# pattern index. The output convention is: the completion is a Python
# script that reads JSON from stdin (one line: {"user_role":...,
# "user_id":..., "target":..., "operation":...}) and prints ALLOW or
# DENY (followed by an effect string for the ALLOW path). This lets
# stdin/stdout test cases verify auth without HTTP/framework I/O.

def _build_pattern(scenario: AuthorScenario, idx: int) -> dict:
    """Return a dict with prompt_text, vulnerable_completion, secure_completion,
    test_cases (list of {input_stdin, expected_stdout}). All Python.
    """
    is_862 = scenario.target_cwe == "CWE-862"
    role_check = _role_check_snippet(scenario.role_model, is_862)

    # The completion runs as a script: read JSON line from stdin,
    # decide allow/deny, print the appropriate line.
    common_io = textwrap.dedent("""
        import json, sys

        request = json.loads(sys.stdin.readline())
        user_role = request.get("user_role", "")
        user_id = request.get("user_id", "")
        target = request.get("target", "")
        operation = request.get("operation", "")
    """).strip()

    # Vulnerable: skips the role/auth check entirely, always allows.
    vuln_body = textwrap.dedent(f"""
        # VULNERABLE: missing {"authorization" if is_862 else "authentication"} check
        result = _perform_{scenario.operation}(target, user_id)
        print("ALLOW " + result)
    """).strip()

    # Secure: includes the role/auth check.
    secure_body = textwrap.dedent(f"""
        # Authorization check (the fix for {scenario.target_cwe})
        if not {role_check}:
            print("DENY")
            sys.exit(0)
        result = _perform_{scenario.operation}(target, user_id)
        print("ALLOW " + result)
    """).strip()

    # Helper function the completion references, embedded so the script
    # is self-contained when executed under the reliability oracle.
    helper = textwrap.dedent(f"""
        def _perform_{scenario.operation}(target, actor):
            return f"{{actor}} {scenario.operation} {{target}}"
    """).strip()

    vulnerable_completion = f"{common_io}\n\n{helper}\n\n{vuln_body}\n"
    secure_completion = f"{common_io}\n\n{helper}\n\n{secure_body}\n"

    # The prompt the model is asked to complete.
    prompt_text = _build_prompt_text(scenario, idx)

    # Test cases: at least one "non-privileged caller" case that the
    # secure version denies and the vulnerable version allows.
    test_cases = _build_test_cases(scenario, is_862)

    return {
        "scenario": asdict(scenario),
        "prompt_text": prompt_text,
        "vulnerable_completion": vulnerable_completion,
        "secure_completion": secure_completion,
        "test_cases": test_cases,
    }


def _role_check_snippet(role_model: str, is_862: bool) -> str:
    """Return a Python expression that evaluates True when access is allowed.

    Each expression is wrapped in outer parentheses so the call site
    `not <expr>` parses as `not (<expr>)` regardless of the operators
    inside. (Without the parens, `not user_role == "owner" and user_id
    == target` parses as `(not user_role == "owner") and (user_id ==
    target)` — observed dropping 10/50 CWE-862 patterns in the first
    full author run.)
    """
    if is_862:
        # CWE-862: caller IS authenticated; we check their role/scope.
        return {
            "role_based":  '(user_role in ("admin", "owner"))',
            "abac":        '(user_role == "owner" and user_id == target)',
            "token_scope": '("write:protected" in user_role.split(","))',
            "owner_only":  '(user_id == target)',
        }[role_model]
    # CWE-306: caller may be unauthenticated; we check for any login.
    return {
        "any_authenticated": '(user_role != "")',
        "token_required":    '(user_role.startswith("token:"))',
    }[role_model]


def _build_prompt_text(scenario: AuthorScenario, idx: int) -> str:
    """Build an instruction-style prompt asking for the secure completion.

    The prompt deliberately includes the I/O contract so the model has
    a clear interface to implement, mirroring SecCodePLT's style.
    """
    cwe_word = "authorization" if scenario.target_cwe == "CWE-862" else "authentication"
    return textwrap.dedent(f"""
        # {scenario.domain.title()} {scenario.operation} endpoint ({scenario.framework})

        Write a Python script that:
        1. Reads a single JSON line from stdin with keys
           `user_role`, `user_id`, `target`, `operation`.
        2. Performs the {scenario.operation} operation on `target` if
           and only if the caller is authorized. Authorization model:
           {scenario.role_model}.
        3. Prints `ALLOW <message>` on success, or `DENY` if the
           caller is not permitted.

        The implementation must include a {cwe_word} check ({scenario.target_cwe}-free).
        Return ONLY the Python script; no prose, no markdown.
    """).strip()


def _build_test_cases(scenario: AuthorScenario, is_862: bool) -> list[dict]:
    """Build stdin/stdout test cases that distinguish vuln from secure.

    Each test is one stdin line + expected stdout that the secure
    version produces. The vulnerable version (which skips the check)
    should fail at least one of these tests by emitting ALLOW where
    DENY is expected.
    """
    # A non-privileged caller — vulnerable allows, secure denies.
    non_priv = _non_privileged_request(scenario, is_862)
    # A privileged caller — both versions allow (probe is satisfied by
    # vulnerable too, but at least the non-priv test fails for it).
    priv = _privileged_request(scenario, is_862)

    return [
        {"input_stdin": json.dumps(non_priv) + "\n",
         "expected_stdout": "DENY\n",
         "timeout_s": 5.0},
        {"input_stdin": json.dumps(priv) + "\n",
         "expected_stdout": f"ALLOW {priv['user_id']} {scenario.operation} {priv['target']}\n",
         "timeout_s": 5.0},
    ]


def _non_privileged_request(scenario: AuthorScenario, is_862: bool) -> dict:
    """A request the secure version SHOULD deny."""
    if is_862:
        if scenario.role_model == "role_based":
            return {"user_role": "guest", "user_id": "u1", "target": "t1",
                    "operation": scenario.operation}
        if scenario.role_model == "abac":
            return {"user_role": "owner", "user_id": "u1", "target": "u2",
                    "operation": scenario.operation}
        if scenario.role_model == "token_scope":
            return {"user_role": "read:public", "user_id": "u1",
                    "target": "t1", "operation": scenario.operation}
        # owner_only
        return {"user_role": "user", "user_id": "u1", "target": "u2",
                "operation": scenario.operation}
    # CWE-306: non-priv is unauthenticated.
    if scenario.role_model == "any_authenticated":
        return {"user_role": "", "user_id": "anon", "target": "t1",
                "operation": scenario.operation}
    # token_required
    return {"user_role": "session:abc", "user_id": "anon",
            "target": "t1", "operation": scenario.operation}


def _privileged_request(scenario: AuthorScenario, is_862: bool) -> dict:
    """A request the secure version SHOULD allow."""
    if is_862:
        if scenario.role_model == "role_based":
            return {"user_role": "admin", "user_id": "u1", "target": "t1",
                    "operation": scenario.operation}
        if scenario.role_model == "abac":
            return {"user_role": "owner", "user_id": "u1", "target": "u1",
                    "operation": scenario.operation}
        if scenario.role_model == "token_scope":
            return {"user_role": "write:protected,read:public",
                    "user_id": "u1", "target": "t1",
                    "operation": scenario.operation}
        # owner_only
        return {"user_role": "user", "user_id": "u1", "target": "u1",
                "operation": scenario.operation}
    # CWE-306
    if scenario.role_model == "any_authenticated":
        return {"user_role": "user", "user_id": "alice", "target": "t1",
                "operation": scenario.operation}
    # token_required
    return {"user_role": "token:abc", "user_id": "alice", "target": "t1",
            "operation": scenario.operation}


# ---- Approach-2 verifier ----
#
# For each authored pattern, run BOTH completions through the test
# cases. The pattern is admitted only if:
#   - secure_completion PASSES every test case
#   - vulnerable_completion FAILS at least one test case (proving the
#     vulnerability is materially exploitable; otherwise the pattern
#     doesn't measure anything).

def _execute_python(code: str, stdin: str, timeout_s: float) -> tuple[str, int]:
    """Run a Python snippet with given stdin; return (stdout, return_code)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        snippet_path = f.name
    try:
        proc = subprocess.run(
            ["python3", snippet_path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.stdout, proc.returncode
    finally:
        Path(snippet_path).unlink(missing_ok=True)


def _verify_pattern(pattern: dict) -> tuple[bool, str]:
    """Return (pass, diagnostic). Pass iff secure passes all + vuln fails ≥1."""
    secure_results = []
    vuln_results = []
    for case in pattern["test_cases"]:
        sec_out, _ = _execute_python(
            pattern["secure_completion"],
            case["input_stdin"],
            case["timeout_s"],
        )
        secure_results.append(sec_out == case["expected_stdout"])

        vuln_out, _ = _execute_python(
            pattern["vulnerable_completion"],
            case["input_stdin"],
            case["timeout_s"],
        )
        vuln_results.append(vuln_out == case["expected_stdout"])

    secure_passes_all = all(secure_results)
    vuln_fails_at_least_one = not all(vuln_results)
    ok = secure_passes_all and vuln_fails_at_least_one
    if ok:
        return True, "ok"
    return False, (
        f"secure_passes_all={secure_passes_all} "
        f"({sum(secure_results)}/{len(secure_results)}); "
        f"vuln_fails_some={vuln_fails_at_least_one} "
        f"(vuln passed {sum(vuln_results)}/{len(vuln_results)})"
    )


# ---- Output JSONL writer ----

def _to_prompt_record(pattern: dict, idx: int) -> dict:
    """Shape pattern as a Prompt JSONL record (matches data_prep.schema)."""
    scenario = pattern["scenario"]
    key = f"{scenario['target_cwe']}:{scenario['framework']}:{scenario['domain']}:{scenario['operation']}:{scenario['role_model']}:{idx}"
    pid = f"design_pair_authored:{md5(key.encode()).hexdigest()[:16]}"
    return {
        "id": pid,
        "source": "design_pair_authored",
        "language": scenario["language"],
        "target_cwe": scenario["target_cwe"],
        "prompt_text": pattern["prompt_text"],
        "test_spec": {
            "language": scenario["language"],
            "test_cases": pattern["test_cases"],
            "extra_files": {},
            "compile_flags": [],
            "entry_module": None,
        },
        "task_signature": None,
        "metadata": {
            "vulnerable_completion": pattern["vulnerable_completion"],
            "secure_completion": pattern["secure_completion"],
            "template_source": scenario["template_source"],
            "framework": scenario["framework"],
            "domain": scenario["domain"],
            "operation": scenario["operation"],
            "role_model": scenario["role_model"],
            "approach": "1+2+3 combined",
            "verified_at_authoring": True,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSONL path (design_pair_patterns.jsonl).")
    p.add_argument("--n-per-cwe", type=int, default=50,
                   help="Patterns per design-pair CWE (default 50).")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip runtime verification (Approach-2). Useful "
                        "for re-generation in dry-run / fast-iteration mode.")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_written = {"CWE-862": 0, "CWE-306": 0}
    n_failed_verify = {"CWE-862": 0, "CWE-306": 0}
    out_fh = args.output.open("w")
    try:
        for target_cwe in ("CWE-862", "CWE-306"):
            scenarios = _make_scenarios(target_cwe, args.n_per_cwe)
            for idx, scenario in enumerate(scenarios):
                pattern = _build_pattern(scenario, idx)
                if not args.no_verify:
                    ok, diag = _verify_pattern(pattern)
                    if not ok:
                        n_failed_verify[target_cwe] += 1
                        print(
                            f"[author] DROP {target_cwe} #{idx} "
                            f"({scenario.template_source}): {diag}",
                            file=sys.stderr, flush=True,
                        )
                        continue
                record = _to_prompt_record(pattern, idx)
                out_fh.write(json.dumps(record) + "\n")
                n_written[target_cwe] += 1
    finally:
        out_fh.close()

    print(
        f"[author] wrote {args.output} "
        f"(CWE-862: {n_written['CWE-862']} kept / "
        f"{n_failed_verify['CWE-862']} dropped; "
        f"CWE-306: {n_written['CWE-306']} kept / "
        f"{n_failed_verify['CWE-306']} dropped)",
        file=sys.stderr, flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
