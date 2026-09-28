"""Reward calculator for the combined training objective (paper Eq. 4).

    r = alpha_mix * R_sec + (1 - alpha_mix) * R_rel + lambda_rag * R_RAG
        - beta * 1[stub]

    - R_sec in [0, 1] is the SAST-derived security score: 0 for code that
      does not parse, otherwise 1 minus the CVSS-weighted penalty of every
      finding (severity / 10, scaled by the finding's confidence), floored
      at 0.
    - R_rel in [0, 1] is the staged reliability reward: graded credit for
      compiling, running, producing output, and passing unit tests. On
      prompts without tests the test budget is spread over the first
      three stages, so each earns one third.
    - lambda_rag * R_RAG arrives precomputed from rag.r_rag.
    - beta is the anti-Goodhart stub penalty.

The calculator does NOT compute severity from CWE; it consumes a
SeveritySource (see sast/severity.py) which encapsulates the SARIF-first
NVDLib-fallback chain. This keeps the calculator free of I/O.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ..sast.models import Finding
from ..sast.severity import SeveritySource

logger = logging.getLogger(__name__)


@dataclass
class ReliabilitySignals:
    """Inputs needed to compute R_reliability and the functionality gate."""

    compiles: bool = False
    runs: bool = False
    produces_output: bool = False
    tests_passed: int = 0
    tests_total: int = 0

    @property
    def gate(self) -> int:
        """Functionality gate from docs/reward_spec.md §1.

        Returns 1 iff the code compiles AND at least one test passes.
        Both conditions are required. `runs` and `produces_output` are
        partial-credit signals for R_reliability but do NOT relax the gate.
        """
        return int(self.compiles and self.tests_passed >= 1)


@dataclass
class RewardConfig:
    """Reward weights; defaults are the paper's values (Section IV-D)."""

    # alpha_mix: weight on R_sec; R_rel gets (1 - alpha).
    alpha: float = 0.3

    # R_reliability component weights.
    r_comp: float = 0.2
    r_run: float = 0.2
    r_out: float = 0.2
    r_func_weight: float = 0.4

    # beta: subtracted from r_total when the pipeline flags an empty-body
    # or return-only stub, so stubs cannot win by satisfying the
    # analyzers' no-finding state.
    stub_penalty: float = 1.5

    # Optional cache directory for reward determinism across reruns.
    cache_dir: Optional[Path] = None

    # Spec version this config targets; logged for traceability.
    spec_version: str = "cargo-eq4"


@dataclass
class RewardBreakdown:
    r_total: float = 0.0
    r_reliability: float = 0.0
    r_security: float = 0.0
    r_rag: float = 0.0
    gate: int = 0
    r_comp: float = 0.0
    r_run: float = 0.0
    r_out: float = 0.0
    r_func: float = 0.0
    findings_count: int = 0
    clipped: bool = False
    per_finding: list[dict] = field(default_factory=list)
    rag_diagnostics: Optional[dict] = None
    spec_version: str = ""


class RewardCalculator:
    """Computes R_total per the reward spec.

    Usage:
        sev_source = SeveritySource(Path("data/nvdlib_cwe_medians.json"))
        calc = RewardCalculator(RewardConfig(), sev_source)
        breakdown = calc.compute(
            reliability=signals,
            findings=merged_findings,
            r_rag=0.12,
            rag_diagnostics={"copy_guard_hit": False},
            completion_hash=calc.canonicalize_completion(text),
        )

    R_RAG is computed upstream (sast.rag.r_rag) and passed in. The
    calculator only composes it into R_total.
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        severity_source: Optional[SeveritySource] = None,
    ) -> None:
        self.config = config or RewardConfig()
        self.severity_source = severity_source or SeveritySource()

    def compute(
        self,
        reliability: ReliabilitySignals,
        findings: list[Finding],
        *,
        r_rag: float = 0.0,
        rag_diagnostics: Optional[dict] = None,
        completion_hash: Optional[str] = None,
        is_stub: bool = False,
    ) -> RewardBreakdown:
        cached = self._read_cache(completion_hash) if completion_hash else None
        if cached is not None:
            return cached

        # --- R_reliability ---
        # When the prompt has no tests (tests_total == 0), the r_func
        # component is unreachable. Without compensation, no-test prompts
        # would have a 0.6 reward ceiling vs 1.0 for tested prompts,
        # introducing a per-prompt scale bias. Only 3.44% of v0.1.7
        # train prompts have tests, so 96.6% of training would receive
        # weaker reward signal across prompts even though within-group
        # advantages were intact. We renormalize: the r_func_weight
        # budget is redistributed proportionally across the three
        # remaining components so max-reachable r_reliability is 1.0
        # for both cases.
        has_tests = reliability.tests_total > 0
        if has_tests:
            r_comp_w = self.config.r_comp
            r_run_w = self.config.r_run
            r_out_w = self.config.r_out
            r_func_w = self.config.r_func_weight
        else:
            base = self.config.r_comp + self.config.r_run + self.config.r_out
            scale = (
                (base + self.config.r_func_weight) / base
                if base > 0 else 1.0
            )
            r_comp_w = self.config.r_comp * scale
            r_run_w = self.config.r_run * scale
            r_out_w = self.config.r_out * scale
            r_func_w = 0.0
        r_comp = r_comp_w if reliability.compiles else 0.0
        r_run = r_run_w if reliability.runs else 0.0
        r_out = r_out_w if reliability.produces_output else 0.0
        if has_tests:
            r_func = r_func_w * (
                reliability.tests_passed / reliability.tests_total
            )
        else:
            r_func = 0.0
        r_reliability = r_comp + r_run + r_out + r_func

        # --- R_security ---
        per_finding: list[dict] = []
        total_penalty = 0.0
        for f in findings:
            severity, severity_source = self.severity_source.severity_with_source(f)
            base = severity / 10.0
            penalty = base * f.confidence
            total_penalty += penalty
            per_finding.append(
                {
                    "cwe": f.cwe,
                    "tier": f.tier.value,
                    "tool": f.tool.value,
                    "severity": round(severity, 3),
                    "severity_source": severity_source,
                    "base": round(base, 4),
                    "confidence": round(f.confidence, 4),
                    "penalty": round(penalty, 4),
                    "corroborators": sorted(t.value for t in f.corroborating_tools),
                    "location": {
                        "file": f.location.file,
                        "line": f.location.start_line,
                    },
                }
            )

        # Non-parsing code scores 0 so it cannot share the analyzer-clean
        # reward of 1 (Table I).
        clipped = total_penalty > 1.0
        r_security = max(0.0, 1.0 - total_penalty) if reliability.compiles else 0.0

        # Diagnostic only; not a multiplier on R_security.
        gate = reliability.gate

        stub_penalty_applied = self.config.stub_penalty if is_stub else 0.0
        alpha = self.config.alpha
        r_total = (
            alpha * r_security + (1.0 - alpha) * r_reliability + r_rag
            - stub_penalty_applied
        )

        breakdown = RewardBreakdown(
            r_total=round(r_total, 6),
            r_reliability=round(r_reliability, 6),
            r_security=round(r_security, 6),
            r_rag=round(r_rag, 6),
            gate=gate,
            r_comp=r_comp,
            r_run=r_run,
            r_out=r_out,
            r_func=round(r_func, 6),
            findings_count=len(findings),
            clipped=clipped,
            per_finding=per_finding,
            rag_diagnostics=rag_diagnostics,
            spec_version=self.config.spec_version,
        )

        if completion_hash:
            self._write_cache(completion_hash, breakdown)
        return breakdown

    @staticmethod
    def canonicalize_completion(text: str) -> str:
        """Stable hash key for caching, ignoring trailing whitespace differences."""
        normalized = text.replace("\r\n", "\n").rstrip() + "\n"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _cache_path(self, completion_hash: str) -> Optional[Path]:
        if self.config.cache_dir is None:
            return None
        return self.config.cache_dir / f"{completion_hash}.json"

    def _read_cache(self, completion_hash: str) -> Optional[RewardBreakdown]:
        path = self._cache_path(completion_hash)
        if path is None or not path.exists():
            return None
        with open(path) as fh:
            data = json.load(fh)
        return RewardBreakdown(**data)

    def _write_cache(self, completion_hash: str, breakdown: RewardBreakdown) -> None:
        path = self._cache_path(completion_hash)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(breakdown), fh)
