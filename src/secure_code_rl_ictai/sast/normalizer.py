"""SARIF v2.1.0 normalizer.

Per docs/reward_spec.md §6, the normalizer runs three stages:

    1. Per-tool ingestion: parse each tool's SARIF, extract raw findings
       with tool-emitted security-severity and confidence.
    2. Cross-tool dedup: group findings by (CWE, normalized location) and
       merge into one finding per group.
    3. Confidence shaping: apply per-tier floors (drop below) and the
       corroboration boost.

The normalizer does NOT compute reward penalties. It produces a list of
`Finding`s that the reward calculator consumes. Severity sourcing
(SARIF-first, NVDLib fallback, tier midpoint) is also deferred to the
reward calculator via `sast.severity.SeveritySource`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .models import Finding, Location, Tier, ToolName, tier_from_severity
from .rule_map import rule_to_cwe


_DEFAULT_CONFIDENCE_FLOORS: dict[Tier, float] = {
    Tier.CRITICAL: 0.6,
    Tier.HIGH: 0.5,
    Tier.MEDIUM: 0.4,
    Tier.LOW: 0.3,
}


def _coerce_confidence(raw: object, default: float) -> float:
    """Map tool-emitted confidence onto [0, 1].

    Tools encode confidence inconsistently:
      - SARIF `level` ("error"/"warning"/"note"): mapped to 0.8/0.5/0.3.
      - Numeric in [0, 1]: passed through.
      - Numeric in [0, 100]: scaled to [0, 1].
      - Anything else: default (typically 0.5 for tools that don't emit).
    """
    if raw is None:
        return default
    if isinstance(raw, str):
        return {"error": 0.8, "warning": 0.5, "note": 0.3}.get(raw.lower(), default)
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        val = float(raw)
        if 0.0 <= val <= 1.0:
            return val
        if 0.0 <= val <= 100.0:
            return val / 100.0
        return max(0.0, min(1.0, val))
    return default


class SarifNormalizer:
    def __init__(
        self,
        confidence_floors: Optional[dict[Tier, float]] = None,
        corroboration_boost: float = 0.15,
        confidence_cap: float = 1.0,
        cwe_hierarchy=None,  # CweHierarchy | None; opt-in
    ) -> None:
        self.confidence_floors: dict[Tier, float] = (
            confidence_floors or dict(_DEFAULT_CONFIDENCE_FLOORS)
        )
        self.corroboration_boost = corroboration_boost
        self.confidence_cap = confidence_cap
        self.cwe_hierarchy = cwe_hierarchy

    # ---------- Per-tool ingestion ----------

    def parse_sarif(self, tool: ToolName, sarif: dict) -> list[Finding]:
        findings: list[Finding] = []
        for run in sarif.get("runs", []):
            rules_index = self._index_rules(run)
            for result in run.get("results", []):
                finding = self._result_to_finding(tool, result, rules_index)
                if finding is not None:
                    findings.append(finding)
        return findings

    @staticmethod
    def _index_rules(run: dict) -> dict[str, dict]:
        index: dict[str, dict] = {}
        driver = run.get("tool", {}).get("driver", {})
        for rule in driver.get("rules", []):
            rule_id = rule.get("id")
            if rule_id:
                index[rule_id] = rule
        return index

    def _result_to_finding(
        self,
        tool: ToolName,
        result: dict,
        rules_index: dict[str, dict],
    ) -> Optional[Finding]:
        rule_id = result.get("ruleId") or result.get("rule", {}).get("id")
        if not rule_id:
            return None

        sarif_rule = rules_index.get(rule_id, {})
        cwe = rule_to_cwe(tool, rule_id, sarif_rule)
        if cwe is None:
            return None  # un-mappable; drop and let the caller log unknowns

        # SARIF security-severity lives in `properties.security-severity` on
        # the rule (CodeQL) or on the result (Semgrep variant). Try both.
        sev = sarif_rule.get("properties", {}).get("security-severity")
        if sev is None:
            sev = result.get("properties", {}).get("security-severity")
        try:
            sev_f = float(sev) if sev is not None else None
        except (TypeError, ValueError):
            sev_f = None

        # Tier from severity if we have it; else default to MEDIUM as a
        # neutral placeholder. The reward calculator's severity sourcing
        # will replace this with NVDLib median or tier-midpoint at reward
        # time; the tier here is only used for confidence-floor lookup.
        tier = tier_from_severity(sev_f) if sev_f is not None else Tier.MEDIUM

        # Confidence: SARIF `level` on the result, or tool-specific properties.
        # CodeQL emits neither — it uses `security-severity` ([0,10]) as a
        # combined severity+confidence signal. When neither standard confidence
        # source is present but we have a security-severity, derive confidence
        # from it (≥7 → 0.8, ≥4 → 0.5, else 0.3) matching CodeQL's documented
        # "high-quality findings" threshold.
        raw_conf: object = result.get("level")
        if raw_conf is None:
            raw_conf = result.get("properties", {}).get("confidence")
        if raw_conf is None and sev_f is not None:
            if sev_f >= 7.0:
                raw_conf = 0.8
            elif sev_f >= 4.0:
                raw_conf = 0.5
            else:
                raw_conf = 0.3
        conf = _coerce_confidence(raw_conf, default=0.5)

        # Cross-signal boost: if security-severity is high but the SARIF
        # `level` mapped to a low confidence (e.g., cppcheck emits
        # `level: warning` => 0.5 even when sev=9.9), upgrade. Otherwise
        # the per-tier floor drops every high-severity cppcheck finding.
        # The rule: when sev>=7, confidence is at least 0.8 (matching the
        # CodeQL "high-quality findings" threshold). When sev>=4, at least
        # 0.5. Tools that disagree by emitting a low level alongside a
        # high security-severity have a self-inconsistent SARIF; we trust
        # the severity signal because it's the cross-tool standard.
        if sev_f is not None:
            if sev_f >= 7.0 and conf < 0.8:
                conf = 0.8
            elif sev_f >= 4.0 and conf < 0.5:
                conf = 0.5

        # Apply per-tier confidence floor (pre-corroboration).
        if conf < self.confidence_floors.get(tier, 0.0):
            return None

        loc = self._extract_location(result)
        if loc is None:
            return None

        return Finding(
            cwe=cwe,
            tier=tier,
            tool=tool,
            security_severity=sev_f,
            confidence=conf,
            location=loc,
            rule_id=rule_id,
            message=self._extract_message(result),
            corroborating_tools=frozenset({tool}),
        )

    @staticmethod
    def _extract_location(result: dict) -> Optional[Location]:
        for loc in result.get("locations", []):
            phys = loc.get("physicalLocation", {})
            art = phys.get("artifactLocation", {})
            region = phys.get("region", {})
            file = art.get("uri")
            start_line = region.get("startLine")
            if file and start_line is not None:
                return Location(
                    file=file,
                    start_line=int(start_line),
                    start_column=region.get("startColumn"),
                    end_line=region.get("endLine"),
                    end_column=region.get("endColumn"),
                )
        return None

    @staticmethod
    def _extract_message(result: dict) -> str:
        msg = result.get("message", {})
        if isinstance(msg, dict):
            return str(msg.get("text", ""))
        return str(msg)

    # ---------- Cross-tool merge ----------

    def merge(self, per_tool_findings: list[list[Finding]]) -> list[Finding]:
        """Group findings across tools by (CWE, normalized location).

        For each group:
          - Pick the primary tool: the one with the highest raw confidence.
          - Boost confidence: +corroboration_boost per additional tool,
            capped at confidence_cap.
          - Carry the highest tool-emitted security-severity (or None if
            no tool emitted one).
          - Collect all corroborating tools.

        If `cwe_hierarchy` is set, also collapse pairs of groups whose
        locations match and whose CWEs are parent/child (the more
        specific CWE — the descendant — becomes canonical).
        """
        # Stage 1: exact-CWE-match grouping.
        groups: dict[tuple[str, tuple[str, int]], list[Finding]] = defaultdict(list)
        for findings in per_tool_findings:
            for f in findings:
                key = (f.cwe, f.location.normalized_key())
                groups[key].append(f)

        # Stage 2: parent/child collapse (when a hierarchy is supplied).
        if self.cwe_hierarchy is not None:
            groups = self._collapse_parent_child_groups(groups)

        merged: list[Finding] = []
        for group in groups.values():
            primary = max(group, key=lambda f: f.confidence)
            n_extra = len(group) - 1
            boosted = min(
                self.confidence_cap,
                primary.confidence + self.corroboration_boost * n_extra,
            )
            sev_values = [f.security_severity for f in group if f.security_severity is not None]
            best_sev = max(sev_values) if sev_values else None
            corroborators = frozenset({f.tool for f in group})
            # Canonical CWE: when parent/child collapse fired, primary may
            # be the descendant or the ancestor depending on which tool
            # had the highest confidence. We prefer the descendant (more
            # specific) for reporting; recompute canonical CWE here.
            canonical_cwe = self._canonical_cwe(group)
            merged.append(
                Finding(
                    cwe=canonical_cwe,
                    tier=primary.tier,
                    tool=primary.tool,
                    security_severity=best_sev,
                    confidence=boosted,
                    location=primary.location,
                    rule_id=primary.rule_id,
                    message=primary.message,
                    corroborating_tools=corroborators,
                )
            )
        return merged

    # --- private helpers for parent/child collapse ---

    def _collapse_parent_child_groups(
        self,
        groups: dict[tuple[str, tuple[str, int]], list[Finding]],
    ) -> dict[tuple[str, tuple[str, int]], list[Finding]]:
        """Merge groups at the same location whose CWEs are parent/child.

        Strategy:
          - Bucket groups by location.
          - Within a location bucket, merge any two groups whose CWEs are
            in an ancestor/descendant relation per `self.cwe_hierarchy`.
          - Canonical key for merged groups: (descendant_cwe, location).
        """
        from .cwe_hierarchy import is_related

        by_location: dict[tuple[str, int], list[tuple[str, list[Finding]]]] = defaultdict(list)
        for (cwe, loc_key), group in groups.items():
            by_location[loc_key].append((cwe, group))

        collapsed: dict[tuple[str, tuple[str, int]], list[Finding]] = {}
        for loc_key, cwe_groups in by_location.items():
            if len(cwe_groups) == 1:
                cwe, group = cwe_groups[0]
                collapsed[(cwe, loc_key)] = group
                continue

            # Union-find style merging among related CWE groups.
            # Each entry starts as its own cluster; if two are related, merge.
            n = len(cwe_groups)
            parent_idx = list(range(n))

            def find(i: int) -> int:
                while parent_idx[i] != i:
                    parent_idx[i] = parent_idx[parent_idx[i]]
                    i = parent_idx[i]
                return i

            for i in range(n):
                for j in range(i + 1, n):
                    if is_related(cwe_groups[i][0], cwe_groups[j][0], self.cwe_hierarchy):
                        parent_idx[find(j)] = find(i)

            clusters: dict[int, list[int]] = defaultdict(list)
            for i in range(n):
                clusters[find(i)].append(i)

            for cluster_indices in clusters.values():
                merged_findings: list[Finding] = []
                cluster_cwes: list[str] = []
                for idx in cluster_indices:
                    cwe_i, group_i = cwe_groups[idx]
                    cluster_cwes.append(cwe_i)
                    merged_findings.extend(group_i)
                # Canonical CWE for the cluster: most-specific (descendant)
                # among the cluster's CWEs. Pick whichever is NOT an
                # ancestor of any other.
                canonical = self._most_specific_cwe(cluster_cwes)
                collapsed[(canonical, loc_key)] = merged_findings
        return collapsed

    def _most_specific_cwe(self, cwes: list[str]) -> str:
        """Among a list of CWEs that are all parent/child related, return
        the most specific (descendant). If none, return the first."""
        if not cwes:
            return ""
        if len(cwes) == 1:
            return cwes[0]
        assert self.cwe_hierarchy is not None
        for candidate in cwes:
            # `candidate` is most-specific iff none of the other cwes
            # appear as candidate's descendants. Equivalently: candidate
            # is not an ancestor of any other in the list.
            is_ancestor_of_any = any(
                candidate in self.cwe_hierarchy.ancestors_of(other)
                for other in cwes
                if other != candidate
            )
            if not is_ancestor_of_any:
                return candidate
        return cwes[0]

    def _canonical_cwe(self, group: list[Finding]) -> str:
        """When stage-2 collapse merged distinct CWEs into one group, the
        canonical CWE is the most-specific one (descendant in hierarchy)."""
        cwes = sorted({f.cwe for f in group})
        if len(cwes) == 1:
            return cwes[0]
        if self.cwe_hierarchy is None:
            # All exact-match; the caller's grouping ensures this case.
            return cwes[0]
        return self._most_specific_cwe(cwes)
