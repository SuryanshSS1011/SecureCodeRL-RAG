"""Tool-rule-id to CWE mapping for the ICTAI SAST pipeline.

The SARIF spec emits CWE assignments in `taxa` references on each rule's
`relationships` field, but coverage is inconsistent across tools. This
module provides a normalized lookup that falls back to tool-specific
heuristics when SARIF taxa references are absent.

Lookup order in `rule_to_cwe(tool, rule_id, sarif_rule)`:
    1. SARIF rule's `relationships` -> taxa references with toolComponent
       name "CWE" (the standards-compliant path).
    2. SARIF rule's `properties.tags` containing "CWE-NNN" (Semgrep,
       Bandit format).
    3. SARIF rule's `properties.cwe` field (Cppcheck convention).
    4. Substring match on rule_id (last resort, tool-specific).

This module is intentionally a thin wrapper: it does not encode every
known mapping for every tool, because doing so creates a maintenance
burden that drifts from the tools' own rule definitions. The SARIF-first
order means upgrading a tool gets the new mappings automatically.
"""

from __future__ import annotations

import re
from typing import Optional

from .models import ToolName

_CWE_PATTERN = re.compile(r"CWE-(\d+)", re.IGNORECASE)


def _from_taxa(sarif_rule: dict) -> Optional[str]:
    """Extract CWE from SARIF rule `relationships` -> taxa references.

    Per SARIF 2.1.0, a rule's relationships point at taxonomy entries. For
    CWE, the toolComponent name is "CWE" and the `id` is the numeric CWE
    identifier as a string.
    """
    for rel in sarif_rule.get("relationships", []):
        target = rel.get("target", {})
        toolcomp = target.get("toolComponent", {})
        if toolcomp.get("name", "").upper() == "CWE":
            cwe_id = target.get("id")
            if cwe_id:
                cwe_id = str(cwe_id).strip()
                if cwe_id.isdigit():
                    return f"CWE-{int(cwe_id)}"
                if cwe_id.upper().startswith("CWE-"):
                    # Strip leading zeros for consistency (CWE-078 -> CWE-78).
                    digits = cwe_id[4:]
                    if digits.isdigit():
                        return f"CWE-{int(digits)}"
                    return cwe_id.upper()
    return None


def _from_tags(sarif_rule: dict) -> Optional[str]:
    """Extract CWE from `properties.tags`.

    Semgrep emits tags like "cwe: CWE-79"; Bandit emits "cwe-78". We
    canonicalize to "CWE-NNN".
    """
    tags = sarif_rule.get("properties", {}).get("tags", [])
    for tag in tags:
        match = _CWE_PATTERN.search(str(tag))
        if match:
            return f"CWE-{int(match.group(1))}"
    return None


def _from_properties_cwe(sarif_rule: dict) -> Optional[str]:
    """Extract CWE from `properties.cwe` (Cppcheck convention)."""
    cwe = sarif_rule.get("properties", {}).get("cwe")
    if cwe is None:
        return None
    match = _CWE_PATTERN.search(str(cwe))
    if match:
        return f"CWE-{int(match.group(1))}"
    if str(cwe).strip().isdigit():
        return f"CWE-{int(str(cwe).strip())}"
    return None


# Bandit rule-id → CWE static map. bandit-sarif-formatter 1.x emits empty
# rule `properties`, so the taxa/tags/properties.cwe resolvers all miss for
# Bandit findings. Without this map every Bandit finding would be dropped
# by the normalizer. Sourced from the upstream Bandit rule docs at
# https://bandit.readthedocs.io/en/1.9.4/plugins/ and the blacklist tables;
# each entry is the CWE that the rule's official documentation cites.
# Maintain this list when bumping Bandit; missing rules silently drop.
_BANDIT_TO_CWE: dict[str, str] = {
    # B1xx — misc
    "B101": "CWE-703",   # assert_used
    "B102": "CWE-78",    # exec_used
    "B103": "CWE-732",   # set_bad_file_permissions
    "B104": "CWE-605",   # hardcoded_bind_all_interfaces
    "B105": "CWE-259",   # hardcoded_password_string
    "B106": "CWE-259",   # hardcoded_password_funcarg
    "B107": "CWE-259",   # hardcoded_password_default
    "B108": "CWE-377",   # hardcoded_tmp_directory
    "B110": "CWE-703",   # try_except_pass
    "B112": "CWE-703",   # try_except_continue
    # B2xx — flask/jinja2/etc
    "B201": "CWE-94",    # flask_debug_true
    "B202": "CWE-22",    # tarfile_unsafe_members
    # B3xx — blacklists (calls)
    "B301": "CWE-502",   # pickle / cPickle.load*
    "B302": "CWE-502",   # marshal.load*
    "B303": "CWE-327",   # md5 / sha1 (insecure hash)
    "B304": "CWE-327",   # insecure ciphers
    "B305": "CWE-327",   # insecure cipher mode
    "B306": "CWE-377",   # mktemp_q (insecure tempfile)
    "B307": "CWE-78",    # eval
    "B308": "CWE-79",    # mark_safe (xss)
    "B309": "CWE-295",   # HTTPSConnection (deprecated, no cert checks)
    "B310": "CWE-22",    # urllib_urlopen (allows file:// schemes)
    "B311": "CWE-330",   # random (insufficient randomness)
    "B312": "CWE-319",   # telnetlib (cleartext)
    "B313": "CWE-20",    # xml.etree (insecure XML)
    "B314": "CWE-20",    # xml.etree (alias)
    "B315": "CWE-20",    # xml.expat
    "B316": "CWE-20",    # xml.expat (alias)
    "B317": "CWE-20",    # xml.sax
    "B318": "CWE-20",    # xml.minidom
    "B319": "CWE-20",    # xml.pulldom
    "B320": "CWE-20",    # lxml etree
    "B321": "CWE-319",   # ftplib (cleartext)
    "B322": "CWE-20",    # input (py2 compat)
    "B323": "CWE-295",   # unverified_context (ssl)
    "B324": "CWE-327",   # hashlib weak (md5 etc.)
    "B325": "CWE-377",   # tempnam (insecure tempfile)
    # B4xx — blacklisted imports
    "B401": "CWE-319",   # telnetlib import
    "B402": "CWE-319",   # ftplib import
    "B403": "CWE-502",   # pickle import
    "B404": "CWE-78",    # subprocess import
    "B405": "CWE-20",    # xml.etree import
    "B406": "CWE-20",    # xml.sax import
    "B407": "CWE-20",    # xml.expat import
    "B408": "CWE-20",    # xml.minidom import
    "B409": "CWE-20",    # xml.pulldom import
    "B410": "CWE-20",    # lxml import
    "B411": "CWE-319",   # xmlrpclib import
    "B412": "CWE-319",   # httpoxy
    "B413": "CWE-327",   # pycrypto import (deprecated)
    "B414": "CWE-327",   # pycryptodome import
    # B5xx — crypto/auth misuse
    "B501": "CWE-295",   # request_with_no_cert_validation
    "B502": "CWE-327",   # ssl_with_bad_version
    "B503": "CWE-327",   # ssl_with_bad_defaults
    "B504": "CWE-327",   # ssl_with_no_version
    "B505": "CWE-326",   # weak_cryptographic_key
    "B506": "CWE-20",    # yaml_load (unsafe)
    "B507": "CWE-295",   # ssh_no_host_key_verification
    "B508": "CWE-319",   # snmp_insecure_version
    "B509": "CWE-327",   # snmp_weak_cryptography
    # B6xx — injection
    "B601": "CWE-78",    # paramiko_calls (shell injection via cmd)
    "B602": "CWE-78",    # subprocess_popen_with_shell_equals_true
    "B603": "CWE-78",    # subprocess_without_shell_equals_true
    "B604": "CWE-78",    # any_other_function_with_shell_equals_true
    "B605": "CWE-78",    # start_process_with_a_shell
    "B606": "CWE-78",    # start_process_with_no_shell
    "B607": "CWE-78",    # start_process_with_partial_path
    "B608": "CWE-89",    # hardcoded_sql_expressions (sql injection)
    "B609": "CWE-78",    # linux_commands_wildcard_injection
    "B610": "CWE-89",    # django_extra_used (sql injection)
    "B611": "CWE-89",    # django_rawsql_used
    "B612": "CWE-20",    # logging_config_insecure_listen
    # B7xx — jinja2/mako/django templates
    "B701": "CWE-79",    # jinja2_autoescape_false
    "B702": "CWE-79",    # use_of_mako_templates
    "B703": "CWE-79",    # django_mark_safe
    "B704": "CWE-79",    # markupsafe_mark_safe
}


# Cppcheck rule-id → CWE static map. Cppcheck 2.18 SARIF emits no CWE on
# rules (the internal CWE() per checker is not serialized to SARIF; only
# `tags: ["security"]` is added when severity=error and id is non-critical).
# Sourced from cppcheck/lib/check*.cpp `CWE(NNN)` constructor calls and the
# cppcheck rule documentation. Maintain when bumping cppcheck.
_CPPCHECK_TO_CWE: dict[str, str] = {
    # Buffer / array
    "arrayIndexOutOfBounds": "CWE-119",
    "arrayIndexOutOfBoundsCond": "CWE-119",
    "bufferAccessOutOfBounds": "CWE-119",
    "negativeIndex": "CWE-786",
    "negativeMemoryAllocationSize": "CWE-131",
    "objectIndex": "CWE-758",
    "outOfBounds": "CWE-788",
    "pointerOutOfBounds": "CWE-823",
    "pointerOutOfBoundsCond": "CWE-823",
    "stringIndexOutOfBounds": "CWE-119",
    # Memory
    "memleak": "CWE-401",
    "memleakOnRealloc": "CWE-401",
    "resourceLeak": "CWE-775",
    "doubleFree": "CWE-415",
    "deallocuse": "CWE-416",   # use-after-free
    "uninitvar": "CWE-457",
    "uninitdata": "CWE-457",
    "uninitMemberVar": "CWE-908",
    "uninitstring": "CWE-457",
    "uninitStructMember": "CWE-457",
    "mismatchAllocDealloc": "CWE-762",
    "leakReturnValNotUsed": "CWE-771",
    "nullPointer": "CWE-476",
    "nullPointerArithmetic": "CWE-682",
    "nullPointerDefaultArg": "CWE-476",
    "nullPointerRedundantCheck": "CWE-476",
    # Integer / arithmetic
    "integerOverflow": "CWE-190",
    "signConversion": "CWE-195",
    "shiftNegative": "CWE-758",
    "shiftNegativeLHS": "CWE-758",
    "shiftTooManyBits": "CWE-758",
    "shiftTooManyBitsSigned": "CWE-758",
    "zerodiv": "CWE-369",
    "zerodivcond": "CWE-369",
    # Format strings / I/O
    "invalidPrintfArgType_s": "CWE-686",
    "invalidPrintfArgType_int": "CWE-686",
    "invalidPrintfArgType_uint": "CWE-686",
    "invalidPrintfArgType_sint": "CWE-686",
    "invalidPrintfArgType_float": "CWE-686",
    "invalidScanfArgType_int": "CWE-686",
    "invalidScanfArgType_s": "CWE-686",
    "invalidScanfArgType_float": "CWE-686",
    "wrongPrintfScanfArgNum": "CWE-685",
    "wrongPrintfScanfParameterPositionError": "CWE-685",
    "scanfFormatStringWidth": "CWE-687",
    "IOWithoutPositioning": "CWE-664",
    "useClosedFile": "CWE-910",
    "writeReadOnlyFile": "CWE-664",
    # Strings
    "bufferAccessOutOfBoundsCond": "CWE-119",
    "danglingTemporaryLifetime": "CWE-562",
    "danglingLifetime": "CWE-562",
    "danglingReference": "CWE-562",
    "deallocret": "CWE-672",
    "deallocuseuninit": "CWE-457",
    "sprintfOverlappingData": "CWE-628",
    # Concurrency / signals
    "unsafeClassDivZero": "CWE-369",
    # Class / inheritance
    "virtualCallInConstructor": "CWE-1037",
    "pureVirtualCall": "CWE-664",
    "duplInheritedMember": "CWE-694",
    "noConstructor": "CWE-665",
    "noExplicitConstructor": "CWE-665",
    "uninitMember": "CWE-908",
    # Inputs / cast / portability
    "invalidPointerCast": "CWE-704",
    "invalidFunctionArg": "CWE-628",
    "invalidFunctionArgBool": "CWE-628",
    "invalidFunctionArgStr": "CWE-628",
    "uselessAssignmentArg": "CWE-563",
    "uselessAssignmentPtrArg": "CWE-563",
    "unreadVariable": "CWE-563",
    # Generic / unspecified — mapped to closest CWE for tracking
    "syntaxError": "CWE-758",
}


def _from_rule_id_heuristic(tool: ToolName, rule_id: str) -> Optional[str]:
    """Last-resort substring match on rule_id.

    For Bandit, applies the curated `_BANDIT_TO_CWE` map (rule docs cite a
    CWE per rule; bandit-sarif-formatter does not emit it in SARIF
    properties). For CodeQL, rule ids like "py/sql-injection" embed no CWE
    in the id string; CodeQL emits CWE via SARIF taxa/tags, handled earlier.
    Generic substring match catches the rare "CWE-NNN" in rule_id.
    """
    if tool == ToolName.BANDIT and rule_id in _BANDIT_TO_CWE:
        return _BANDIT_TO_CWE[rule_id]
    if tool == ToolName.CPPCHECK and rule_id in _CPPCHECK_TO_CWE:
        return _CPPCHECK_TO_CWE[rule_id]

    match = _CWE_PATTERN.search(rule_id)
    if match:
        return f"CWE-{int(match.group(1))}"
    return None


def rule_to_cwe(
    tool: ToolName, rule_id: str, sarif_rule: Optional[dict] = None
) -> Optional[str]:
    """Resolve a rule id to a normalized CWE string of the form 'CWE-NNN'.

    Returns None if no CWE assignment can be found. Callers should drop
    findings with no CWE (they cannot be reward-graded against the ICTAI
    CWE taxonomy) and log the rule_id for taxonomy expansion review.
    """
    if sarif_rule is None:
        sarif_rule = {}

    for resolver in (_from_taxa, _from_tags, _from_properties_cwe):
        cwe = resolver(sarif_rule)
        if cwe is not None:
            return cwe

    return _from_rule_id_heuristic(tool, rule_id)
