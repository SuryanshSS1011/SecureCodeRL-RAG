"""Severity sourcing for SAST findings.

Implements the corroboration-aware sourcing pipeline from
docs/reward_spec.md §3:

    1. SARIF `security-severity` on the primary tool (sarif:primary).
    2. SARIF `security-severity` on a corroborating reporting tool
       (sarif:corroborator). This is implicit in how the normalizer
       populates `Finding.security_severity` — it carries the max
       emitted severity across all tools in the corroboration cluster —
       so if security_severity is not None, the source label is
       "sarif:primary" when the primary tool emits it, else
       "sarif:corroborator".
    3. Per-CWE NVDLib empirical median (nvdlib_median).
    4. Per-tier midpoint over 10 (tier_midpoint; logged).

`severity_with_source(f)` returns `(severity, source_label)` so callers
can stratify diagnostics by which path each finding took.

The fallback split between sarif:primary and sarif:corroborator is the
"corroboration severity sourcing" decision recorded in
`OPEN_ADVISER_ITEMS.md` §3 Caveat B: when a finding is corroborated by a
severity-reporting tool, source severity from that tool even if a
non-reporting tool is the primary detector. Settled at-desk before any
measurement runs, so the reported per-finding-vs-fallback fraction
reflects the decision.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .models import Finding, Tier

logger = logging.getLogger(__name__)

# Per-tier midpoint over 10, used as the final fallback.
# Critical 9.5/10 = 0.95, High 7.95/10 = 0.795, Medium 5.45/10 = 0.545,
# Low 2.0/10 = 0.20. Multiplied by 10 here for consistency (severity scale
# is 0-10, base scale 0-1 is computed at reward time).
_TIER_MIDPOINT_SEVERITY: dict[Tier, float] = {
    Tier.CRITICAL: 9.5,
    Tier.HIGH: 7.95,
    Tier.MEDIUM: 5.45,
    Tier.LOW: 2.0,
}


class SeveritySource:
    """Loads the NVDLib median table once and resolves severities.

    Instantiate once per training run; reuse across many findings.
    """

    def __init__(self, medians_path: Optional[Path] = None) -> None:
        self._medians: dict[str, float] = {}
        # Four-way source-path counters; reported per training run via
        # `diagnostics()` and surfaced in `RewardBreakdown.per_finding[i]`.
        self._sarif_primary_count: int = 0
        self._sarif_corroborator_count: int = 0
        self._nvdlib_median_count: int = 0
        self._tier_midpoint_count: int = 0

        if medians_path is not None:
            self._load(medians_path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning(
                "NVDLib medians file missing at %s; all unscored findings "
                "will fall through to per-tier midpoint.",
                path,
            )
            return
        with open(path) as fh:
            data = json.load(fh)
        if data.get("_meta", {}).get("status") == "placeholder":
            logger.warning(
                "NVDLib medians file at %s is a placeholder; populate via "
                "scripts/build_nvdlib_medians.py before running training.",
                path,
            )
            return
        for cwe, entry in data.get("medians", {}).items():
            median = entry.get("median")
            if median is not None:
                self._medians[cwe] = float(median)

    def severity_with_source(self, finding: Finding) -> tuple[float, str]:
        """Resolve severity AND report which sourcing path it came from.

        Returns:
            (severity in [0, 10], source_label in
             {"sarif:primary", "sarif:corroborator", "nvdlib_median",
              "tier_midpoint"}).

        Source labels:
            - "sarif:primary": the primary detecting tool emitted the
              SARIF security-severity. Reliable per-finding signal.
            - "sarif:corroborator": the primary tool did not emit
              security-severity, but a corroborating tool in the same
              dedup cluster did. The normalizer's merge() puts the max
              emitted severity onto Finding.security_severity, so we
              detect this case by checking whether the primary tool is
              in the known severity-emitting set.
            - "nvdlib_median": no tool in the cluster emitted severity;
              falling back to the NVDLib per-CWE empirical median. Still
              externally grounded but per-CWE rather than per-finding.
            - "tier_midpoint": even the NVDLib median is unavailable
              (CWE not in the table). Defensive default, logged.
        """
        if finding.security_severity is not None:
            if finding.tool in _SARIF_SEVERITY_EMITTING_TOOLS:
                self._sarif_primary_count += 1
                return finding.security_severity, "sarif:primary"
            else:
                self._sarif_corroborator_count += 1
                return finding.security_severity, "sarif:corroborator"

        median = self._medians.get(finding.cwe)
        if median is not None:
            self._nvdlib_median_count += 1
            return median, "nvdlib_median"

        self._tier_midpoint_count += 1
        logger.debug(
            "Falling back to tier midpoint for %s (tool=%s, rule=%s); "
            "this branch should be rare in steady state.",
            finding.cwe,
            finding.tool.value,
            finding.rule_id,
        )
        return _TIER_MIDPOINT_SEVERITY[finding.tier], "tier_midpoint"

    def diagnostics(self) -> dict[str, int]:
        """Return per-source-path counts for this source's lifetime.

        Keys:
            - sarif_primary: primary tool emitted severity.
            - sarif_corroborator: only a corroborating tool emitted it.
            - nvdlib_median: NVDLib per-CWE empirical-median fallback.
            - tier_midpoint: even the NVDLib median was unavailable.
            - fallback_total: legacy alias for the sum of the last three.
            - cwe_median_fallback / tier_midpoint_fallback: legacy names
              for backward compatibility with existing tests.
        """
        fallback_total = (
            self._sarif_corroborator_count
            + self._nvdlib_median_count
            + self._tier_midpoint_count
        )
        return {
            "sarif_primary": self._sarif_primary_count,
            "sarif_corroborator": self._sarif_corroborator_count,
            "nvdlib_median": self._nvdlib_median_count,
            "tier_midpoint": self._tier_midpoint_count,
            # Legacy keys, kept for the existing test_sast.py assertions.
            "fallback_total": fallback_total,
            "cwe_median_fallback": self._nvdlib_median_count,
            "tier_midpoint_fallback": self._tier_midpoint_count,
        }


# Tools known to emit `security-severity` in their SARIF output. Per
# docs/sast_pipeline_spec.md §1: CodeQL and Semgrep emit it natively;
# Bandit and Cppcheck largely do not.
#
# Used by severity_with_source to distinguish sarif:primary vs
# sarif:corroborator. If a non-emitting primary tool returns a Finding
# with non-None security_severity, that means the merge step pulled the
# severity from a corroborating tool.
from .models import ToolName  # noqa: E402  (imported at module bottom to avoid an early cycle)

_SARIF_SEVERITY_EMITTING_TOOLS: frozenset[ToolName] = frozenset(
    {ToolName.CODEQL, ToolName.SEMGREP}
)
