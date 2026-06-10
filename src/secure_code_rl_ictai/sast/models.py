"""SARIF v2.1.0 data models for the ICTAI SAST pipeline.

Derived from the OASIS SARIF v2.1.0 specification. The Tier enum aligns to
FIRST's CVSS v3.1 severity bands; tiers are used for curriculum gating and
reporting only, NOT as reward penalty inputs (see docs/reward_spec.md §3).

This module is intentionally tool-agnostic. Tool-specific parsing lives in
sast/normalizer.py; rule-to-CWE mapping lives in sast/rule_map.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Tier(str, Enum):
    """CVSS v3.1 severity band, per FIRST.

    Band cutoffs:
        Critical: 9.0 - 10.0
        High:     7.0 - 8.9
        Medium:   4.0 - 6.9
        Low:      0.1 - 3.9

    Used for curriculum phase gating and per-tier detection-rate reporting.
    The reward penalty is computed from the underlying continuous severity,
    not from the tier (docs/reward_spec.md §3).
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def tier_from_severity(severity: float) -> Tier:
    """Map a CVSS-scale severity (0.0-10.0) to its FIRST band."""
    if severity >= 9.0:
        return Tier.CRITICAL
    if severity >= 7.0:
        return Tier.HIGH
    if severity >= 4.0:
        return Tier.MEDIUM
    return Tier.LOW


class ToolName(str, Enum):
    """SAST tools in the ICTAI pipeline.

    Membership is closed: adding a tool here is a deliberate scope expansion
    requiring a CWE-coverage audit and a SARIF rule-map extension.
    """

    CODEQL = "codeql"
    SEMGREP = "semgrep"
    BANDIT = "bandit"
    CPPCHECK = "cppcheck"


@dataclass(frozen=True)
class Location:
    """A finding's source location, normalized to (file, start_line)."""

    file: str
    start_line: int
    start_column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def normalized_key(self) -> tuple[str, int]:
        """Key for dedup-by-location across tools.

        Column is intentionally not in the key: tools disagree on column
        conventions (1-based vs 0-based, byte vs character) and a 5-column
        delta on the same line is the same finding for reward purposes.
        """
        return (self.file, self.start_line)


@dataclass
class Finding:
    """A single SAST finding, normalized across tools.

    `security_severity` is the raw 0.0-10.0 score as emitted by the tool
    (SARIF `security-severity` property) or `None` if the tool does not emit
    one. The severity-sourcing pipeline (sast/severity.py) resolves `None`
    via NVDLib fallback at reward-computation time, not here.

    `confidence` is the post-corroboration confidence in [0, 1]. Findings
    below the per-tier floor are dropped before they reach this dataclass.

    `corroborating_tools` lists every tool that independently reported this
    finding at the same (CWE, normalized location). Set has at least the
    primary tool; len > 1 means multi-tool corroboration.
    """

    cwe: str  # e.g. "CWE-787"
    tier: Tier
    tool: ToolName  # primary tool (the one with the highest raw confidence)
    security_severity: Optional[float]  # 0.0-10.0 if emitted by tool, else None
    confidence: float  # in [0, 1], post-corroboration
    location: Location
    rule_id: str  # tool-native rule id, for traceability
    message: str = ""
    corroborating_tools: frozenset[ToolName] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.security_severity is not None:
            if not 0.0 <= self.security_severity <= 10.0:
                raise ValueError(
                    f"security_severity must be in [0, 10], got {self.security_severity}"
                )
        if not self.cwe.startswith("CWE-"):
            raise ValueError(f"cwe must look like 'CWE-NNN', got {self.cwe!r}")


@dataclass(frozen=True)
class CodeFlowStep:
    """A single step in a SARIF codeFlow (data-flow trace).

    Preserved for tools that emit data-flow information (CodeQL). Not consumed
    by the reward calculator; useful for diagnostics and for the per-finding
    breakdown shown in training logs.
    """

    location: Location
    message: str = ""
