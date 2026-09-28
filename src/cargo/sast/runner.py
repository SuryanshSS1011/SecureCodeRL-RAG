"""SAST runner: orchestrates tool invocations per docs/sast_pipeline_spec.md.

Two modes:

    - **Mock mode** (default for tests): adapters return canned SARIF. No
      shell-out. Used for unit tests and for testing downstream consumers
      without the tool environment installed.
    - **Real mode**: adapters shell out to the actual tools. Requires the
      tools to be installed and on PATH. Wired in `RealCodeQLAdapter`,
      `RealSemgrepAdapter`, etc. — left unimplemented until the SAST
      environment is set up (Phase C of the proceed plan).

The runner is structured so adding a tool means writing one adapter and
registering it in `DEFAULT_REGISTRY`. The normalizer downstream is already
tool-agnostic.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from .models import ToolName
from .normalizer import SarifNormalizer

logger = logging.getLogger(__name__)


class Language(str, Enum):
    PYTHON = "python"
    C = "c"
    CPP = "cpp"


# Language routing per docs/sast_pipeline_spec.md §1.
LANGUAGE_ROUTING: dict[Language, tuple[ToolName, ...]] = {
    Language.PYTHON: (ToolName.CODEQL, ToolName.SEMGREP, ToolName.BANDIT),
    Language.C: (ToolName.CODEQL, ToolName.SEMGREP, ToolName.CPPCHECK),
    Language.CPP: (ToolName.CODEQL, ToolName.SEMGREP, ToolName.CPPCHECK),
}


# Tools classified as "deep" (expensive; database create / formal verification).
# Tier=cheap drops these; tier=cheap+periodic runs them every `deep_period` steps.
_DEEP_TOOLS: frozenset[ToolName] = frozenset({ToolName.CODEQL})


class SastTier(str, Enum):
    """Cost-based tier controlling which tools fire per step.

    See docs/sast_pipeline_spec.md (sequential-gating section) and the
    `Add sequential SAST tier knob` finding in FINDINGS_LOG.md.
    """

    ALL = "all"  # default: run every tool every step (eval-time setting)
    CHEAP = "cheap"  # skip the deep tools entirely (fast iteration)
    CHEAP_PLUS_PERIODIC = "cheap+periodic"  # cheap every step + deep every N


# Per-tool timeouts in seconds (paper Section V-D): 60 s per analyzer
# invocation, 120 s for CodeQL on C and C++.
DEFAULT_TIMEOUTS: dict[ToolName, float] = {
    ToolName.CODEQL: 60.0,
    ToolName.SEMGREP: 60.0,
    ToolName.BANDIT: 60.0,
    ToolName.CPPCHECK: 60.0,
}
CODEQL_C_CPP_TIMEOUT = 120.0


@dataclass
class ToolRunResult:
    tool: ToolName
    sarif: Optional[dict]  # None if crashed or timed out
    stderr: str = ""
    exit_code: int = 0
    duration_s: float = 0.0
    timed_out: bool = False

    @property
    def crashed(self) -> bool:
        return self.sarif is None and not self.timed_out


@dataclass
class SastRunSummary:
    snippet_hash: str
    language: Language
    per_tool: dict[ToolName, ToolRunResult] = field(default_factory=dict)
    total_findings: int = 0  # populated after normalization
    duration_s: float = 0.0
    crashed_tools: list[ToolName] = field(default_factory=list)
    timed_out_tools: list[ToolName] = field(default_factory=list)


class ToolAdapter(ABC):
    """Per-tool invocation contract. See docs/sast_pipeline_spec.md §2."""

    @property
    @abstractmethod
    def tool(self) -> ToolName: ...

    @abstractmethod
    def run(
        self,
        code: str,
        language: Language,
        work_dir: Path,
        timeout_s: float,
    ) -> ToolRunResult: ...


class MockAdapter(ToolAdapter):
    """Returns canned SARIF. Used for unit tests and for downstream-consumer
    development before the real tools are installed.

    The `canned_sarif` is yielded as-is. If `canned_sarif` is None the run
    is reported as a crash (sarif=None, exit_code=1).
    """

    def __init__(self, tool: ToolName, canned_sarif: Optional[dict] = None) -> None:
        self._tool = tool
        self._canned = canned_sarif

    @property
    def tool(self) -> ToolName:
        return self._tool

    def run(
        self,
        code: str,
        language: Language,
        work_dir: Path,
        timeout_s: float,
    ) -> ToolRunResult:
        if self._canned is None:
            return ToolRunResult(
                tool=self._tool,
                sarif=None,
                stderr="mock: no canned sarif provided",
                exit_code=1,
            )
        return ToolRunResult(tool=self._tool, sarif=self._canned, exit_code=0)


class SastRunner:
    """Orchestrates the per-language tool set on a code snippet.

    Caching: per-(tool, snippet_hash, language) cache of SARIF results in
    `cache_dir/sarif/`. Tool version is NOT yet in the cache key; that's a
    TODO once real adapters land (the version is read from `tool --version`
    at adapter init time).
    """

    def __init__(
        self,
        adapters: dict[ToolName, ToolAdapter],
        normalizer: Optional[SarifNormalizer] = None,
        timeouts: Optional[dict[ToolName, float]] = None,
        cache_dir: Optional[Path] = None,
        run_in_parallel: bool = False,
        *,
        tier: SastTier | str = SastTier.ALL,
        deep_period: int = 50,
    ) -> None:
        self.adapters = adapters
        self.normalizer = normalizer or SarifNormalizer()
        self.timeouts = timeouts or dict(DEFAULT_TIMEOUTS)
        self.cache_dir = cache_dir
        self.run_in_parallel = run_in_parallel
        # Accept either the enum or the equivalent legacy string.
        if isinstance(tier, str) and not isinstance(tier, SastTier):
            tier = SastTier(tier)
        self.tier: SastTier = tier
        self.deep_period = deep_period
        self._run_counter: int = 0
        if run_in_parallel:
            logger.info("parallel SAST runs requested but not yet implemented; running serially")

    @staticmethod
    def snippet_hash(code: str, language: Language) -> str:
        """Stable hash for cache keying. Includes language so the same code
        under different language assumptions is cached separately."""
        h = hashlib.sha256()
        h.update(language.value.encode("utf-8"))
        h.update(b"\x00")
        normalized = code.replace("\r\n", "\n").rstrip() + "\n"
        h.update(normalized.encode("utf-8"))
        return h.hexdigest()

    def _tools_for_this_run(self, language: Language) -> tuple[ToolName, ...]:
        """Apply the tier policy to the language routing for this run.

        Increments the internal run counter. The result is the tuple of
        tools that should actually fire for this call.
        """
        routed = LANGUAGE_ROUTING.get(language, ())
        self._run_counter += 1

        if self.tier == SastTier.ALL:
            return routed
        if self.tier == SastTier.CHEAP:
            return tuple(t for t in routed if t not in _DEEP_TOOLS)
        # cheap+periodic: deep tools fire every `deep_period` runs.
        if self._run_counter % self.deep_period == 0:
            return routed
        return tuple(t for t in routed if t not in _DEEP_TOOLS)

    def run(self, code: str, language: Language, work_dir: Path) -> SastRunSummary:
        snippet = self.snippet_hash(code, language)
        summary = SastRunSummary(snippet_hash=snippet, language=language)
        start = time.monotonic()

        routed_tools = self._tools_for_this_run(language)
        for tool in routed_tools:
            adapter = self.adapters.get(tool)
            if adapter is None:
                logger.warning(
                    "tool %s in routing for %s but no adapter registered; skipping",
                    tool.value,
                    language.value,
                )
                continue
            timeout = self.timeouts.get(tool, 60.0)
            if tool == ToolName.CODEQL and language in (Language.C, Language.CPP):
                timeout = max(timeout, CODEQL_C_CPP_TIMEOUT)
            result = adapter.run(code, language, work_dir, timeout)
            summary.per_tool[tool] = result
            if result.timed_out:
                summary.timed_out_tools.append(tool)
            elif result.crashed:
                summary.crashed_tools.append(tool)

        # Normalize per-tool SARIF and merge across tools.
        per_tool_findings = []
        for tool, result in summary.per_tool.items():
            if result.sarif is None:
                continue
            findings = self.normalizer.parse_sarif(tool, result.sarif)
            per_tool_findings.append(findings)

        merged = self.normalizer.merge(per_tool_findings)
        summary.total_findings = len(merged)
        summary.duration_s = time.monotonic() - start

        return summary, merged
