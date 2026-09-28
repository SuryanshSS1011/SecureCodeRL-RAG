"""End-to-end per-rollout reward pipeline.

One entry point — `RewardPipeline.evaluate(prompt, completion, language)`
— wires together every component the reward calculator needs:

    1. Reliability oracle: compiles + runs + tests the completion.
    2. SAST runner: invokes static-analysis tools on the completion.
    3. SARIF normalizer: applies confidence floors, dedup, corroboration.
    4. RAG retriever: picks (e+, e-) for the prompt's CWE family.
    5. Embedder: embeds completion (e+ and e- pre-embedded at index build).
    6. R_RAG: contrastive score with copy-guard.
    7. Reward calculator: composes R_total.

The pipeline is the *only* place that knows about all these subsystems.
The trainer talks to the pipeline; the pipeline talks to everything else.
This is the seam where mock-mode and real-mode swap in/out cleanly.

A `PromptContext` carries the per-prompt metadata the pipeline needs:
target CWE, language, test spec, query embedding for retrieval. The
trainer assembles this from the dataset.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from ..rag.r_rag import RagDiagnostics, compute_r_rag, r_rag_missing
from ..rag.retriever import HybridRetriever, RetrievalQuery
from ..sast.normalizer import SarifNormalizer
from ..sast.runner import Language as SastLanguage
from ..sast.runner import SastRunner
from ..sast.severity import SeveritySource
from .calculator import (
    ReliabilitySignals,
    RewardBreakdown,
    RewardCalculator,
    RewardConfig,
)
from .reliability_oracle import ReliabilityOracle, TestSpec

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """Per-prompt context the pipeline needs to evaluate a completion."""

    target_cwe: str
    language: str  # "python" | "c" | "cpp"
    test_spec: TestSpec
    # Query embedding for retrieval (768-dim, L2-normalized). May be None
    # if retrieval is disabled for this prompt (e.g., warm-up phase).
    query_embedding: Optional[list[float]] = None
    task_signature: Optional[str] = None
    # Optional pre-supplied (e_pos, e_neg) embeddings to skip retrieval
    # — used when the trainer caches retrieval across rollouts in a group.
    e_pos_embedding: Optional[list[float]] = None
    e_neg_embedding: Optional[list[float]] = None


class CompletionEmbedder(Protocol):
    """The trainer supplies an embedder. The pipeline doesn't import any
    sentence-transformers code itself; that's deferred to the trainer."""

    def embed(self, text: str) -> list[float]: ...


@dataclass
class PipelineDiagnostics:
    """Everything the pipeline learned about a completion, beyond R_total.

    The trainer surfaces these in training logs. High refusal rate, high
    copy-guard hit rate, high SAST crash rate are all reward-hacking
    canaries.
    """

    reliability: ReliabilitySignals = field(
        default_factory=lambda: ReliabilitySignals()
    )
    sast_total_findings: int = 0
    sast_crashed_tools: list[str] = field(default_factory=list)
    sast_timed_out_tools: list[str] = field(default_factory=list)
    rag_used: bool = False
    rag_hit_pair_cwe: Optional[str] = None
    rag_copy_guard_hit: bool = False
    rag_missing: bool = False
    refusal_or_empty: bool = False
    stub_detected: bool = False
    stub_reason: str = ""


@dataclass
class PipelineOutput:
    breakdown: RewardBreakdown
    diagnostics: PipelineDiagnostics


class RewardPipeline:
    """Composes the reward from reliability + SAST + RAG.

    Args:
        oracle: ReliabilityOracle (Real or Mock).
        sast_runner: SastRunner with adapters wired.
        severity_source: SeveritySource with NVDLib medians loaded.
        retriever: HybridRetriever, or None to disable RAG.
        embedder: CompletionEmbedder, or None to disable RAG.
        calculator: optional pre-configured RewardCalculator; default uses
            the paper's RewardConfig() weights.
        normalizer: optional; if None, the runner's normalizer is used.
        lambda_rag, copy_guard_threshold: passed through to compute_r_rag.

    Either both retriever and embedder are set, or both are None. Mixed
    setting raises at construction.
    """

    def __init__(
        self,
        oracle: ReliabilityOracle,
        sast_runner: SastRunner,
        severity_source: SeveritySource,
        retriever: Optional[HybridRetriever] = None,
        embedder: Optional[CompletionEmbedder] = None,
        calculator: Optional[RewardCalculator] = None,
        normalizer: Optional[SarifNormalizer] = None,
        lambda_rag: float = 0.1,
        copy_guard_threshold: float = 0.95,
        rag_binary: bool = False,
    ) -> None:
        if (retriever is None) != (embedder is None):
            raise ValueError(
                "retriever and embedder must both be set or both be None"
            )
        self.oracle = oracle
        self.sast_runner = sast_runner
        self.severity_source = severity_source
        self.retriever = retriever
        self.embedder = embedder
        self.calculator = calculator or RewardCalculator(
            RewardConfig(), severity_source
        )
        self.normalizer = normalizer or sast_runner.normalizer
        self.lambda_rag = lambda_rag
        self.copy_guard_threshold = copy_guard_threshold
        self.rag_binary = rag_binary

    def evaluate(self, prompt: str, completion: str, context: PromptContext) -> PipelineOutput:
        diag = PipelineDiagnostics()
        _ = prompt  # currently unused; reserved for retrieval-query construction

        # ---- 0. Refusal / empty short-circuit ----
        if not completion.strip() or _looks_like_refusal(completion):
            diag.refusal_or_empty = True
            # Run the oracle so we have well-defined reliability signals.
            # An empty completion won't compile; the gate stays 0 and
            # R_security contributes nothing.
            sigs = self.oracle.evaluate(completion, context.test_spec)
            diag.reliability = sigs
            breakdown = self.calculator.compute(
                reliability=sigs,
                findings=[],
                r_rag=0.0,
            )
            return PipelineOutput(breakdown=breakdown, diagnostics=diag)

        # ---- 0.5. Stub / triviality guard ----
        # Anti-Goodhart: an empty-body or return-only stub skips the oracle
        # and analyzers and takes the stub penalty (beta in Eq. 4).
        is_stub, stub_reason = _looks_like_stub(completion, context.language)
        if is_stub:
            diag.stub_detected = True
            diag.stub_reason = stub_reason
            sigs = ReliabilitySignals()
            diag.reliability = sigs
            breakdown = self.calculator.compute(
                reliability=sigs,
                findings=[],
                r_rag=0.0,
                is_stub=True,
            )
            return PipelineOutput(breakdown=breakdown, diagnostics=diag)

        # ---- 1. Reliability ----
        sigs = self.oracle.evaluate(completion, context.test_spec)
        diag.reliability = sigs

        # ---- 2-3. SAST + normalize ----
        # SAST language enum is distinct from oracle Language enum; map.
        sast_lang = _to_sast_language(context.language)
        # Manual tmpdir management instead of TemporaryDirectory context
        # because the latter raises OSError ([Errno 39] Directory not
        # empty) when SAST subprocesses leave residual files at the moment
        # of __exit__. Observed in production: job 53656148 (qwen-7b-
        # securecode) failed at prompt 350 with this exact race
        # (FINDINGS_LOG 2026-06-15). Per-call uniqueness via PID + uuid
        # eliminates collisions. shutil.rmtree(..., ignore_errors=True)
        # tolerates residual subprocess artifacts.
        # Tmpdir base selection mirrors reliability_oracle.evaluate (ROAR
        # /tmp is small; under concurrent jobs it fills, causing codeql
        # ENOSPC crashes seen in v0.1.7 RL cells).
        tmp_base = (
            os.environ.get("SLURM_TMPDIR")
            or os.environ.get("ICTAI_PIPELINE_TMP")
            or tempfile.gettempdir()
        )
        tmp_name = f"ictai_pipeline_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        tmp = Path(tmp_base) / tmp_name
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            summary, merged_findings = self.sast_runner.run(completion, sast_lang, tmp)
        finally:
            # Even if SAST raised, scrub the tmpdir. ignore_errors=True so
            # a subprocess holding a fd doesn't kill the whole evaluate().
            shutil.rmtree(tmp, ignore_errors=True)
        diag.sast_total_findings = summary.total_findings
        diag.sast_crashed_tools = [t.value for t in summary.crashed_tools]
        diag.sast_timed_out_tools = [t.value for t in summary.timed_out_tools]

        # ---- 4-6. RAG ----
        r_rag_value = 0.0
        rag_diag: Optional[RagDiagnostics] = None
        if self.retriever is not None and self.embedder is not None:
            r_rag_value, rag_diag = self._compute_rag(completion, context)
            diag.rag_used = True
            if rag_diag is not None:
                diag.rag_copy_guard_hit = rag_diag.copy_guard_hit
                # Detect miss sentinel from r_rag_missing(): lambda_rag set to 0.
                if rag_diag.lambda_rag == 0.0:
                    diag.rag_missing = True

        # ---- 7. Compose R_total ----
        completion_hash = self.calculator.canonicalize_completion(completion)
        breakdown = self.calculator.compute(
            reliability=sigs,
            findings=merged_findings,
            r_rag=r_rag_value,
            rag_diagnostics=_rag_diag_to_dict(rag_diag),
            completion_hash=completion_hash,
        )
        return PipelineOutput(breakdown=breakdown, diagnostics=diag)

    def _compute_rag(
        self, completion: str, context: PromptContext
    ) -> tuple[float, RagDiagnostics]:
        """Retrieve a pair (or use the supplied ones) and compute R_RAG."""
        assert self.retriever is not None and self.embedder is not None

        # If the caller pre-supplied pair embeddings, skip retrieval.
        if context.e_pos_embedding is not None and context.e_neg_embedding is not None:
            comp_emb = self.embedder.embed(completion)
            return compute_r_rag(
                comp_emb,
                context.e_pos_embedding,
                context.e_neg_embedding,
                lambda_rag=self.lambda_rag,
                copy_guard_threshold=self.copy_guard_threshold,
                binary=self.rag_binary,
            )

        # Otherwise, retrieve from the index.
        if context.query_embedding is None:
            logger.debug("no query embedding for %s; RAG miss", context.target_cwe)
            return r_rag_missing()

        query = RetrievalQuery(
            cwe=context.target_cwe,
            task_signature=context.task_signature,
            query_embedding=context.query_embedding,
            top_k_per_backend=20,
        )
        hit = self.retriever.retrieve(query)
        if hit is None:
            return r_rag_missing()

        # The pipeline needs (e_pos, e_neg) embeddings to compute R_RAG.
        # In the production wiring these come from the index (pre-computed
        # at build time). For now we embed on the fly; the trainer can
        # supply a faster path via context.e_pos/neg_embedding.
        comp_emb = self.embedder.embed(completion)
        pos_emb = self.embedder.embed(hit.pair.e_pos)
        neg_emb = self.embedder.embed(hit.pair.e_neg)
        return compute_r_rag(
            comp_emb,
            pos_emb,
            neg_emb,
            lambda_rag=self.lambda_rag,
            copy_guard_threshold=self.copy_guard_threshold,
            binary=self.rag_binary,
        )


# ---------- helpers ----------


_REFUSAL_PATTERNS: tuple[str, ...] = (
    "i cannot",
    "i can't",
    "i will not",
    "i won't",
    "i'm not able to",
    "i am not able to",
    "as an ai",
    "sorry, but",
    "i'm sorry, but",
)


def _looks_like_refusal(text: str) -> bool:
    """Detect natural-language refusals masquerading as code.

    Conservative: only fires on text whose first non-blank line starts
    with a refusal pattern AND lacks any common code markers. False
    positives here would suppress legitimate code with apologetic comments.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    first = lines[0].strip().lower()
    has_code_marker = any(
        m in text for m in ("def ", "class ", "import ", "#include", "int main")
    )
    if has_code_marker:
        return False
    return any(first.startswith(p) for p in _REFUSAL_PATTERNS)


_PY_TRIVIAL_BODIES: tuple[str, ...] = (
    "pass",
    "...",
    "return",
    "return none",
    "return 0",
    'return ""',
    "return ''",
    "return []",
    "return {}",
    "raise notimplementederror",
    "raise notimplementederror()",
)

_C_TRIVIAL_BODIES: tuple[str, ...] = (
    "",
    "return 0;",
    "return;",
    "return 1;",
    "return -1;",
    "exit(0);",
)


def _looks_like_stub(text: str, language: str) -> tuple[bool, str]:
    """Detect degenerate completions that game the reliability reward.

    Returns (True, reason) if the code is trivially short, an empty
    function body, or a known no-op pattern. Conservative: false negatives
    are fine (model can still learn), false positives kill real code so
    thresholds are tight. The reward calculator zeros r_reliability when
    True; r_security and r_rag still flow.
    """
    stripped = "\n".join(
        ln for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith(("#", "//"))
    )
    if len(stripped.replace(" ", "").replace("\t", "")) < 20:
        return True, "len<20"

    lang = (language or "").lower()
    body = stripped.lower().strip()

    if lang == "python":
        # Strip a single leading `def ...:` or `class ...:` line; the
        # rest must contain real logic.
        body_lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
        if body_lines and (
            body_lines[0].startswith("def ") or body_lines[0].startswith("class ")
        ):
            body_inner = " ".join(body_lines[1:]).lower().strip()
        else:
            body_inner = body
        for triv in _PY_TRIVIAL_BODIES:
            if body_inner == triv or body_inner.endswith(triv):
                if len(body_inner) <= len(triv) + 4:
                    return True, f"py_trivial:{triv!r}"

    if lang in ("c", "cpp", "c++"):
        # Strip a single leading signature ending in `{`; check the body.
        start = body.find("{")
        end = body.rfind("}")
        if 0 <= start < end:
            body_inner = body[start + 1 : end].strip()
        else:
            body_inner = body
        for triv in _C_TRIVIAL_BODIES:
            if body_inner == triv:
                return True, f"c_trivial:{triv!r}"

    return False, ""


def _to_sast_language(s: str) -> SastLanguage:
    s = s.lower()
    if s == "python":
        return SastLanguage.PYTHON
    if s == "c":
        return SastLanguage.C
    if s in ("cpp", "c++"):
        return SastLanguage.CPP
    raise ValueError(f"unsupported language string: {s!r}")


def _rag_diag_to_dict(d: Optional[RagDiagnostics]) -> Optional[dict]:
    if d is None:
        return None
    return {
        "sim_pos": d.sim_pos,
        "sim_neg": d.sim_neg,
        "raw": d.raw,
        "copy_guard_hit": d.copy_guard_hit,
        "lambda_rag": d.lambda_rag,
        "copy_guard_threshold": d.copy_guard_threshold,
    }
