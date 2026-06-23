"""EvalHarness: single entry point for ablation arms.

Per docs/eval_harness_spec.md, the harness:
    1. Iterates Prompts.
    2. Calls the model to generate a completion.
    3. Runs the RewardPipeline on each completion.
    4. Aggregates per-CWE, per-(CWE, language), and overall.
    5. Computes reward-hacking canaries (refusal/empty/copy-guard rates).
    6. Bootstrap CIs on the headline metrics.
    7. Returns an EvalReport with serialization support.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from ..data_prep.schema import Prompt
from ..reward.pipeline import PromptContext, RewardPipeline
from .model import BaselineModel, SamplingConfig

import re as _re

_CODE_FENCE_RE = _re.compile(
    r"```(?:python|py|c|cpp|c\+\+)?\s*\n(.*?)```",
    _re.DOTALL | _re.IGNORECASE,
)
_OPEN_FENCE_RE = _re.compile(
    r"```(?:python|py|c|cpp|c\+\+)?\s*\n",
    _re.IGNORECASE,
)

# Tokens emitted by base/CLM models after they finish the task and
# start regenerating the prompt or chat scaffolding. Truncating at the
# first occurrence prevents oracle failures driven by trailing
# prose/template noise (observed on starcoder2-3b and sven-codegen-2.7b
# in the v0.1.7 baseline sweep).
_CONTINUATION_MARKERS = (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|file_separator|>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|endofcompletion|>", "<|end|>", "<|user|>", "<|assistant|>",
    "</s>", "<eos>", "<|EOT|>",
    "/* Original program",
    "# Original program",
    "// Original program",
    "/* The above program",
)


def _truncate_at_continuation(text: str) -> str:
    """Cut at the first chat/prompt-continuation marker."""
    cut = len(text)
    for m in _CONTINUATION_MARKERS:
        i = text.find(m)
        if i != -1 and i < cut:
            cut = i
    return text[:cut]


def _strip_unterminated_comment(text: str) -> str:
    """If the text ends with an unclosed /* ... block-comment, drop it.

    Base models that run out of tokens mid-comment leave the trailing
    /* ... open, which makes gcc/g++ refuse to parse the entire file.
    Stripping the orphan comment lets the preceding valid code parse.
    """
    last_open = text.rfind("/*")
    if last_open == -1:
        return text
    last_close = text.rfind("*/")
    if last_close > last_open:
        return text
    return text[:last_open].rstrip()


def _truncate_after_top_level_unit(text: str, language: str) -> str:
    """For C/C++: truncate after the last balanced-brace top-level unit.

    Base CLMs frequently emit a complete function or program and then
    keep generating prose, a hardened-rewrite, or a duplicate main.
    Walk the source tracking depth ignoring single-line, block, and
    string contexts; if depth returns to 0 and the rest of the source
    is dominated by non-code, cut there. The check is conservative:
    we only truncate when the tail's first non-whitespace token is a
    comment opener or a CWE/natural-language marker, so qwen-style
    instruct output with multiple genuine functions isn't damaged.
    """
    if language not in ("c", "cpp"):
        return text
    depth = 0
    i = 0
    n = len(text)
    last_brace_close_at_depth_zero = -1
    in_line_comment = False
    in_block_comment = False
    in_string: str | None = None  # holds the opening quote char
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_string is not None:
            if ch == "\\" and nxt:
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in ('"', "'"):
            in_string = ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_brace_close_at_depth_zero = i
        i += 1
    if last_brace_close_at_depth_zero == -1:
        return text
    # What follows the last top-level closing brace?
    tail = text[last_brace_close_at_depth_zero + 1 :].lstrip()
    if not tail:
        return text
    # Truncate when the tail starts with a comment, preprocessor directive
    # that's actually English prose, or a clear natural-language marker.
    suspicious_starts = (
        "/*",        # block-comment commentary
        "//",        # line-comment commentary
        "/* ",
        "// ",
        "#Implementation",
        "# Implementation",
        "# Original",
        "# Hardened",
        "# The ",
        "/CWE-",     # CASTLE file-path-style header
    )
    if any(tail.startswith(s) for s in suspicious_starts):
        return text[: last_brace_close_at_depth_zero + 1]
    # Also truncate if the tail starts with a capital letter followed by
    # a lowercase letter (prose), with no semicolons or braces for a long
    # run — heuristic for "the model started narrating".
    head_64 = tail[:64]
    if (
        head_64
        and head_64[0].isupper()
        and len(head_64) > 4
        and head_64[1:4].islower()
        and ";" not in head_64
        and "{" not in head_64
        and "}" not in head_64
    ):
        return text[: last_brace_close_at_depth_zero + 1]
    return text


def _strip_duplicate_main(text: str, language: str) -> str:
    """Drop everything from the second `int main(` (or `void main(`) onward.

    Base models on CASTLE-style prompts emit a working program, then a
    'hardened version' of the same program with its own main. The
    duplicate-symbol error wrecks compile; the first program is fine.
    """
    if language not in ("c", "cpp"):
        return text
    pat = _re.compile(r"\b(?:int|void)\s+main\s*\(", _re.MULTILINE)
    matches = list(pat.finditer(text))
    if len(matches) >= 2:
        return text[: matches[1].start()].rstrip()
    return text


def _extract_code(text: str, language: str) -> str:
    """Strip chat-style prose around a generated completion.

    Instruction-tuned models (Qwen, DeepSeek-Coder, etc.) often wrap code
    in markdown fences (```python ... ``` or just ``` ... ```). Without
    extraction, the pipeline tries to compile/run/analyze the prose,
    producing universally func@1=0 and meaningless SAST signals.

    Strategy (in priority order):
      1. If text contains one or more fenced code blocks, concatenate
         all of them with double newlines. Most models emit one block;
         some emit imports + main in separate blocks.
      2. Else if there's an opening fence but no closing one (truncated
         output at max_new_tokens), take everything after the opening
         fence. Without this, the bare language-name line "c" or "python"
         leaks into the saved snippet and breaks gcc / python parsing.
      3. Else if text looks like raw code (starts with `def `, `import `,
         `class `, `#include`, `int main`, etc.), use it as-is.
      4. Else return text unchanged; the downstream parser will catch it.

    Language hint is reserved for future per-language heuristics (e.g.
    extracting bare snippets between BEGIN/END markers); v0.1 ignores it.
    """
    text = _truncate_at_continuation(text)
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        code = "\n\n".join(m.strip() for m in matches)
    else:
        open_match = _OPEN_FENCE_RE.search(text)
        if open_match:
            code = text[open_match.end():].rstrip().removesuffix("```").rstrip()
        else:
            code = text
    # Post-extraction sanitization. These passes are language-aware but
    # safe to run unconditionally: Python branches early-return when the
    # heuristic doesn't apply.
    lang_lower = (language or "").lower()
    code = _truncate_after_top_level_unit(code, lang_lower)
    code = _strip_duplicate_main(code, lang_lower)
    code = _strip_unterminated_comment(code)
    return code


@dataclass
class PerPromptRecord:
    """One row in the eval report.

    Captures everything a downstream analysis needs without forcing
    consumers to re-run the pipeline. The completion text is preserved so
    we can re-tokenize / re-analyze later.
    """

    prompt_id: str
    source: str
    target_cwe: str
    language: str
    completion: str
    crashed: bool = False
    compiles: bool = False
    runs: bool = False
    produces_output: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    # Intrinsic test count from the prompt's TestSpec, independent of
    # whether the completion compiled. `tests_total` is populated by the
    # oracle only when the code compiles, so it cannot define a
    # per-system-invariant denominator; this field can (see
    # metrics.py `func_sec_at_1__has_tests`). Old streams lack it and
    # default to 0; `_has_test_spec` falls back to `tests_total`.
    n_test_cases: int = 0
    r_total: float = 0.0
    r_reliability: float = 0.0
    r_security: float = 0.0
    r_rag: float = 0.0
    findings_count: int = 0
    findings_cwes: list[str] = field(default_factory=list)
    target_cwe_present: bool = False
    refusal_or_empty: bool = False
    copy_guard_hit: bool = False
    rag_missing: bool = False
    generation_duration_s: float = 0.0
    # Per-tool SAST availability on this prompt. Empty lists when all 4
    # tools ran cleanly. Reviewer-disclosed in the paper appendix per T6.
    sast_crashed_tools: list[str] = field(default_factory=list)
    sast_timed_out_tools: list[str] = field(default_factory=list)
    # Inference-time prompt-prepend RAG (paper cell 3). When the harness
    # is invoked with a `prompt_transform`, this dict records whether the
    # exemplar was retrieved and prepended for this prompt, plus the
    # top-1 pair identifier and RRF score. Empty dict when no transform
    # was used.
    rag_diagnostics: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    model_name: str
    sampling_config: SamplingConfig
    n_prompts: int
    aggregate: dict = field(default_factory=dict)
    per_cwe: dict[str, dict] = field(default_factory=dict)
    per_cwe_language: dict[tuple[str, str], dict] = field(default_factory=dict)
    bootstrap_cis: dict[str, tuple[float, float]] = field(default_factory=dict)
    per_prompt: list[PerPromptRecord] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def save(self, output_dir: Path) -> Path:
        """Write per_prompt.jsonl + aggregate.json under output_dir/<model_name>/."""
        out = output_dir / self.model_name
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "per_prompt.jsonl", "w") as fh:
            for rec in self.per_prompt:
                fh.write(json.dumps(asdict(rec)) + "\n")

        agg_payload = {
            "model_name": self.model_name,
            "n_prompts": self.n_prompts,
            "sampling_config": asdict(self.sampling_config),
            "aggregate": self.aggregate,
            "per_cwe": self.per_cwe,
            "per_cwe_language": {
                f"{cwe}|{lang}": v for (cwe, lang), v in self.per_cwe_language.items()
            },
            "bootstrap_cis": {k: list(v) for k, v in self.bootstrap_cis.items()},
            "diagnostics": self.diagnostics,
            "timestamp": self.timestamp,
        }
        with open(out / "aggregate.json", "w") as fh:
            json.dump(agg_payload, fh, indent=2)
        return out


class EvalHarness:
    """Drives one model through a set of prompts and produces an EvalReport."""

    def __init__(
        self,
        pipeline: RewardPipeline,
        sampling: SamplingConfig,
        *,
        bootstrap_n_resamples: int = 1000,
        bootstrap_ci_level: float = 0.95,
        bootstrap_per_cwe_min_support: int = 20,
    ) -> None:
        self.pipeline = pipeline
        self.sampling = sampling
        self.bootstrap_n_resamples = bootstrap_n_resamples
        self.bootstrap_ci_level = bootstrap_ci_level
        self.bootstrap_per_cwe_min_support = bootstrap_per_cwe_min_support

    def evaluate(
        self,
        model: BaselineModel,
        prompts: Iterable[Prompt],
        *,
        stream_path: Optional[Path] = None,
        resume: bool = False,
        prompt_transform: Optional[
            Callable[[Prompt], tuple[str, dict]]
        ] = None,
        checkpoint_output_dir: Optional[Path] = None,
        checkpoint_every: int = 200,
    ) -> EvalReport:
        """Run model + pipeline on each prompt.

        When `stream_path` is provided, every PerPromptRecord is appended to
        that JSONL file as it's produced, with `fh.flush()` after each line.
        That way a SLURM timeout / OOM / manual cancel preserves all
        completed prompts instead of losing the in-memory buffer.

        When `resume=True` AND `stream_path` already exists, the existing
        records are loaded into memory, their prompt_ids skipped in the
        iteration loop, and new records appended to the same file. The
        sweep was originally cancelled-at-time-limit on the v0.1.5 main
        run; resume lets the cancelled baselines complete to n=861 with
        ~2hr of incremental compute instead of re-running the full ~10hr.
        Without `resume=True`, the existing stream is truncated as before
        so a re-run starts fresh.
        """
        records: list[PerPromptRecord] = []
        completed_ids: set[str] = set()

        import time as _time
        import sys as _sys
        prompts = list(prompts)
        n_total = len(prompts)
        t_eval_start = _time.perf_counter()

        stream_fh = None
        if stream_path is not None:
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            if resume and stream_path.exists():
                # Re-load completed records so the in-memory report is
                # consistent (final aggregate covers all n_total prompts,
                # not just the new ones), and build the skip set.
                # Deduplicate by prompt_id (keep LAST occurrence: latest
                # streaming write wins) AND filter to the current eval set
                # so that records from older eval-set versions don't inflate
                # the aggregate. Streams from interrupted+resumed runs over
                # weeks of corpus iteration accumulate both kinds of debris:
                # duplicate records AND stale prompt_ids from prior corpora.
                eval_prompt_ids = {p.id for p in prompts}
                pending: dict[str, PerPromptRecord] = {}
                with open(stream_path) as fh_in:
                    for line in fh_in:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        pid = raw["prompt_id"]
                        if pid not in eval_prompt_ids:
                            continue
                        completed_ids.add(pid)
                        pending[pid] = PerPromptRecord(**{
                            k: raw.get(k, getattr(PerPromptRecord, k, None))
                            for k in PerPromptRecord.__dataclass_fields__
                        })
                records.extend(pending.values())
                print(
                    f"[eval] resuming: {len(completed_ids)} prompts already "
                    f"in {stream_path}",
                    file=_sys.stderr, flush=True,
                )
                stream_fh = open(stream_path, "a")
            else:
                stream_fh = open(stream_path, "w")

        def _flush_record(rec):
            records.append(rec)
            if stream_fh is not None:
                stream_fh.write(json.dumps(asdict(rec), default=str) + "\n")
                stream_fh.flush()

        try:
            for prompt_idx, prompt in enumerate(prompts, start=1):
                if prompt.id in completed_ids:
                    continue
                # Per-50-prompts progress log so that long sweeps (typical: 1411
                # prompts x ~20-60s/prompt = 7-24 hr) are observable mid-stream
                # rather than a silent ~hr-scale wait until the report dumps.
                if prompt_idx > 1 and (prompt_idx - 1) % 50 == 0:
                    elapsed = _time.perf_counter() - t_eval_start
                    rate = (prompt_idx - 1) / max(1.0, elapsed)
                    eta_s = (n_total - prompt_idx + 1) / max(rate, 1e-6)
                    print(
                        f"[eval] {prompt_idx-1}/{n_total} prompts in {elapsed:.0f}s "
                        f"({rate:.2f}/s; ETA {eta_s:.0f}s = {eta_s/60:.1f}min)",
                        file=_sys.stderr, flush=True,
                    )

                # Periodic aggregate.json checkpoint. Without this, a SLURM
                # walltime kill at prompt N<n_total leaves only the per-prompt
                # stream and no aggregated metrics. We rebuild and save the
                # full EvalReport every `checkpoint_every` prompts so the
                # downstream paper-table consumer sees a usable file even on
                # partial runs. The final .save() at the end overwrites the
                # last checkpoint with the full-coverage version.
                if (checkpoint_output_dir is not None
                        and prompt_idx > 1
                        and (prompt_idx - 1) % checkpoint_every == 0
                        and records):
                    try:
                        partial_report = self._build_report(model.name, list(records))
                        partial_report.save(checkpoint_output_dir)
                        print(
                            f"[eval] checkpoint aggregate.json saved at prompt "
                            f"{prompt_idx-1}/{n_total} ({len(records)} records)",
                            file=_sys.stderr, flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"[eval] WARNING: checkpoint save at prompt "
                            f"{prompt_idx-1} failed: {exc}",
                            file=_sys.stderr, flush=True,
                        )

                # Inference-time prompt-prepend RAG (paper cell 3). The
                # transform returns the text to send to the model plus a
                # diagnostics dict recording whether the prepend fired,
                # the pair identifier, and the RRF score. We do NOT alter
                # `prompt.prompt_text` itself — the reward pipeline still
                # scores against the original prompt so reliability and
                # SAST signals stay comparable to the no-prepend cell.
                if prompt_transform is not None:
                    gen_input, rag_diag = prompt_transform(prompt)
                else:
                    gen_input, rag_diag = prompt.prompt_text, {}

                comp_result = model.generate(
                    gen_input, sampling=self.sampling
                )

                # Build the PromptContext from the Prompt.
                ctx = PromptContext(
                    target_cwe=prompt.target_cwe,
                    language=prompt.language.value,
                    test_spec=prompt.test_spec,
                    task_signature=prompt.task_signature,
                )

                if comp_result.crashed:
                    rec = PerPromptRecord(
                        prompt_id=prompt.id,
                        source=prompt.source,
                        target_cwe=prompt.target_cwe,
                        language=prompt.language.value,
                        completion="",
                        crashed=True,
                        refusal_or_empty=True,
                        n_test_cases=len(prompt.test_spec.test_cases),
                        generation_duration_s=comp_result.duration_s,
                        rag_diagnostics=rag_diag,
                    )
                    _flush_record(rec)
                    continue

                out = self.pipeline.evaluate(
                    prompt.prompt_text, _extract_code(comp_result.text, prompt.language.value), ctx
                )

                findings_cwes = [
                    f["cwe"] for f in out.breakdown.per_finding
                ]
                target_present = prompt.target_cwe in findings_cwes

                rec = PerPromptRecord(
                    prompt_id=prompt.id,
                    source=prompt.source,
                    target_cwe=prompt.target_cwe,
                    language=prompt.language.value,
                    completion=comp_result.text,
                    crashed=False,
                    compiles=out.diagnostics.reliability.compiles,
                    runs=out.diagnostics.reliability.runs,
                    produces_output=out.diagnostics.reliability.produces_output,
                    tests_passed=out.diagnostics.reliability.tests_passed,
                    tests_total=out.diagnostics.reliability.tests_total,
                    n_test_cases=len(prompt.test_spec.test_cases),
                    r_total=out.breakdown.r_total,
                    r_reliability=out.breakdown.r_reliability,
                    r_security=out.breakdown.r_security,
                    r_rag=out.breakdown.r_rag,
                    findings_count=out.breakdown.findings_count,
                    findings_cwes=findings_cwes,
                    target_cwe_present=target_present,
                    refusal_or_empty=out.diagnostics.refusal_or_empty,
                    copy_guard_hit=out.diagnostics.rag_copy_guard_hit,
                    rag_missing=out.diagnostics.rag_missing,
                    generation_duration_s=comp_result.duration_s,
                    sast_crashed_tools=list(out.diagnostics.sast_crashed_tools),
                    sast_timed_out_tools=list(out.diagnostics.sast_timed_out_tools),
                    rag_diagnostics=rag_diag,
                )
                _flush_record(rec)
        finally:
            if stream_fh is not None:
                stream_fh.close()

        return self._build_report(model.name, records)

    # ------------------------------------------------------------------

    def _build_report(
        self, model_name: str, records: list[PerPromptRecord]
    ) -> EvalReport:
        n = len(records)
        if n == 0:
            return EvalReport(
                model_name=model_name,
                sampling_config=self.sampling,
                n_prompts=0,
                aggregate={
                    "func_at_1": 0.0,
                    "secure_at_1": 0.0,
                    "func_sec_at_1": 0.0,
                    "mean_r_total": 0.0,
                    "mean_r_security": 0.0,
                    "mean_r_reliability": 0.0,
                    "mean_r_rag": 0.0,
                    "mean_findings_per_completion": 0.0,
                },
                per_cwe={},
                per_cwe_language={},
                diagnostics={
                    "refusal_rate": 0.0,
                    "empty_rate": 0.0,
                    "copy_guard_hit_rate": 0.0,
                    "crash_rate": 0.0,
                },
                per_prompt=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        agg = self._aggregate(records)
        per_cwe = self._aggregate_per_cwe(records)
        per_cwe_lang = self._aggregate_per_cwe_language(records)
        diag = self._diagnostics(records)
        cis = self._bootstrap_cis(records)

        return EvalReport(
            model_name=model_name,
            sampling_config=self.sampling,
            n_prompts=n,
            aggregate=agg,
            per_cwe=per_cwe,
            per_cwe_language=per_cwe_lang,
            bootstrap_cis=cis,
            per_prompt=records,
            diagnostics=diag,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ---- bootstrap CIs (spec §5) ----

    def _bootstrap_cis(
        self, records: list[PerPromptRecord]
    ) -> dict[str, tuple[float, float]]:
        """Compute (low, high) 95% bootstrap CIs on the headline metrics.

        Procedure (spec §5):
            1. Per metric, build a 0/1 indicator array over `records`.
            2. Resample indices with replacement `n_resamples` times.
            3. Compute the metric (mean of indicators) per resample.
            4. Return the (alpha/2, 1-alpha/2) percentiles, alpha = 1 - ci_level.

        Seeded by self.sampling.seed so re-runs are reproducible.
        """
        if not records:
            return {}

        rng = np.random.default_rng(self.sampling.seed)
        alpha = 1.0 - self.bootstrap_ci_level
        low_q = alpha / 2.0
        high_q = 1.0 - alpha / 2.0

        # Per-spec bootstrap. Each MetricSpec defines a (base, condition)
        # pair; the conditional metric subsets the records before
        # resampling so a metric "secure_at_1 | compiles" has its own
        # denominator (compileable records) bootstrapped against itself.
        # Per-CWE CIs only when support meets the spec §5 threshold.
        from .metrics import METRICS

        cis: dict[str, tuple[float, float]] = {}
        records_arr = list(records)

        def ci_for_spec(spec, recs):
            subset = (
                recs if spec.condition is None
                else [r for r in recs if spec.condition(r)]
            )
            m = len(subset)
            if m == 0:
                return None
            indicator = np.array([spec.base(r) for r in subset], dtype=float)
            sample = rng.integers(0, m, size=(self.bootstrap_n_resamples, m))
            samples = indicator[sample].mean(axis=1)
            return (
                float(np.quantile(samples, low_q)),
                float(np.quantile(samples, high_q)),
            )

        for spec in METRICS:
            ci = ci_for_spec(spec, records_arr)
            if ci is not None:
                cis[spec.name] = ci

        by_cwe: dict[str, list[PerPromptRecord]] = defaultdict(list)
        for r in records_arr:
            by_cwe[r.target_cwe].append(r)
        for cwe, recs in by_cwe.items():
            if len(recs) < self.bootstrap_per_cwe_min_support:
                continue
            for spec in METRICS:
                ci = ci_for_spec(spec, recs)
                if ci is not None:
                    cis[f"{cwe}:{spec.name}"] = ci

        return cis

    def _aggregate(self, records: list[PerPromptRecord]) -> dict:
        """Aggregate every MetricSpec plus reward/finding/diagnostic means.

        Each MetricSpec produces a `{value, n_numerator, n_denominator}`
        dict under its name. The three legacy keys (`func_at_1`,
        `secure_at_1`, `func_sec_at_1`) are also hoisted to scalars at
        the top level for backward-compat with the existing renderer.

        Additional diagnostics emitted alongside the metric registry:
          - mean_findings_per_compileable_completion (A3): findings rate
            among code that actually compiled — paper-defensible "given
            the model wrote real code, how many findings does SAST raise?"
          - refusal_rate, crash_rate (A4): per-prompt diagnostics over the
            full sample. refusal_rate is the share of (empty or refusal)
            completions; crash_rate is the share of generation crashes.
          - completion_length_{mean, median, p95} (B6): non-refusal
            completion length distribution. Long completions often
            disagree with the reliability oracle's tests; reporting the
            shape helps reviewers reason about per-baseline behavior.
          - mean_r_{total, security, reliability, rag} (B7): reward
            decomposition. Lets the paper claim "our reward signal isn't
            dominated by reliability" via the relative magnitudes.
        """
        from .metrics import compute_all, legacy_scalar_view

        per_spec = compute_all(records)
        agg: dict = dict(per_spec)
        agg.update(legacy_scalar_view(per_spec))
        n = len(records)
        agg.update({
            "mean_r_total": sum(r.r_total for r in records) / n,
            "mean_r_security": sum(r.r_security for r in records) / n,
            "mean_r_reliability": sum(r.r_reliability for r in records) / n,
            "mean_r_rag": sum(r.r_rag for r in records) / n,
            "mean_findings_per_completion": (
                sum(r.findings_count for r in records) / n
            ),
        })

        # A3: findings per *compileable* completion. Aligns with the
        # compile-first interpretation: if a baseline doesn't compile,
        # SAST has nothing to find, so the full-sample mean understates
        # "what does the model produce when it does produce code?"
        compileable = [
            r for r in records
            if r.compiles and not r.refusal_or_empty and not r.crashed
        ]
        agg["mean_findings_per_compileable_completion"] = (
            sum(r.findings_count for r in compileable) / len(compileable)
            if compileable else 0.0
        )
        agg["n_compileable_completions"] = len(compileable)

        # A4: refusal and crash rates.
        agg["refusal_rate"] = sum(
            r.refusal_or_empty for r in records
        ) / n
        agg["crash_rate"] = sum(r.crashed for r in records) / n

        # B6: completion-length distribution over non-refusal records.
        # Reported as mean, median, and p95. Refusals are excluded so the
        # distribution reflects "what the model actually wrote."
        non_refusal_lens = [
            len(r.completion)
            for r in records
            if not r.refusal_or_empty and not r.crashed
        ]
        if non_refusal_lens:
            import statistics
            sorted_lens = sorted(non_refusal_lens)
            p95_idx = max(0, int(0.95 * len(sorted_lens)) - 1)
            agg["completion_length_mean"] = (
                sum(non_refusal_lens) / len(non_refusal_lens)
            )
            agg["completion_length_median"] = statistics.median(non_refusal_lens)
            agg["completion_length_p95"] = sorted_lens[p95_idx]
            agg["n_non_refusal"] = len(non_refusal_lens)
        else:
            agg["completion_length_mean"] = 0.0
            agg["completion_length_median"] = 0.0
            agg["completion_length_p95"] = 0.0
            agg["n_non_refusal"] = 0

        return agg

    def _aggregate_per_cwe(
        self, records: list[PerPromptRecord]
    ) -> dict[str, dict]:
        from .metrics import compute_all, legacy_scalar_view

        by_cwe: dict[str, list[PerPromptRecord]] = defaultdict(list)
        for r in records:
            by_cwe[r.target_cwe].append(r)
        out: dict[str, dict] = {}
        for cwe, recs in by_cwe.items():
            per_spec = compute_all(recs)
            cell: dict = dict(per_spec)
            cell["n_prompts"] = len(recs)
            cell.update(legacy_scalar_view(per_spec))
            out[cwe] = cell
        return out

    def _aggregate_per_cwe_language(
        self, records: list[PerPromptRecord]
    ) -> dict[tuple[str, str], dict]:
        from .metrics import compute_all, legacy_scalar_view

        by_key: dict[tuple[str, str], list[PerPromptRecord]] = defaultdict(list)
        for r in records:
            by_key[(r.target_cwe, r.language)].append(r)
        out: dict[tuple[str, str], dict] = {}
        for key, recs in by_key.items():
            per_spec = compute_all(recs)
            cell: dict = dict(per_spec)
            cell["n_prompts"] = len(recs)
            cell.update(legacy_scalar_view(per_spec))
            out[key] = cell
        return out

    @staticmethod
    def _diagnostics(records: list[PerPromptRecord]) -> dict[str, float]:
        n = len(records)
        rag_used = [r for r in records if not r.rag_missing and not r.crashed]
        rag_used_n = len(rag_used) if rag_used else 1

        # Per-tool SAST timeout + crash rates. Critical for the appendix
        # disclosure: when CodeQL times out on a prompt, we lose the
        # signal it would have provided (notably on design-level CWEs).
        # Reported per tool so reviewers can read "CodeQL timeout rate
        # was X%" and weigh the bias.
        from collections import Counter
        timeout_counter: Counter[str] = Counter()
        crash_counter: Counter[str] = Counter()
        for r in records:
            for tool in r.sast_timed_out_tools:
                timeout_counter[tool] += 1
            for tool in r.sast_crashed_tools:
                crash_counter[tool] += 1

        diag = {
            "refusal_rate": sum(r.refusal_or_empty for r in records) / n,
            "empty_rate": sum(
                not r.completion.strip() and not r.crashed for r in records
            ) / n,
            "copy_guard_hit_rate": (
                sum(r.copy_guard_hit for r in rag_used) / rag_used_n
            ),
            "crash_rate": sum(r.crashed for r in records) / n,
        }
        for tool, count in timeout_counter.items():
            diag[f"sast_timeout_rate_{tool}"] = count / n
        for tool, count in crash_counter.items():
            diag[f"sast_crash_rate_{tool}"] = count / n
        return diag
