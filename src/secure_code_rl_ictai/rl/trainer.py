"""Trainer orchestration.

Glues the model, reward pipeline, schedule, and algorithm step into the
per-step training loop. The trainer does NOT own the model implementation
— the `PolicyProtocol` is a four-method interface that the torch glue
class (TorchPolicy, TBD) implements. This module is torch-free.

Per docs/training_spec.md §1, one step:

    1. Sample B prompts from the dataset.
    2. For each prompt, generate G completions (group).
    3. Score every completion via the reward pipeline.
    4. Compute log-probs under current and reference policies.
    5. Run algorithm.step(rewards, log_probs, ref_log_probs) -> loss.
    6. policy.step(loss) -> grad_norm.
    7. Aggregate per-step metrics and emit a TrainStepRecord.

The trainer writes a JSONL log if `config.log_path` is set. Otherwise
records are returned only from `Trainer.run()`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ..data_prep.schema import Prompt
from ..eval.harness import _extract_code
from ..eval.model import SamplingConfig
from ..reward.pipeline import PromptContext, RewardPipeline
from .grpo import GrpoStepOutput
from .metrics import StepMetricsAggregator, TrainStepRecord
from .registry import AlgorithmStep
from .reweight import Reweighter
from .schedule import PhaseSchedule


@runtime_checkable
class PolicyProtocol(Protocol):
    """Policy interface. Torch glue (TBD) implements this against an
    HF model + LoRA adapter + optimizer.

    `step()` now accepts `advantages`, `clip_epsilon`, and `kl_beta` from
    the algorithm step so the policy can rebuild the autograd surrogate
    from cached log-probs. Trainers that don't need this (mocks) can
    ignore the extras.
    """

    def generate(
        self, prompts: list[str], sampling: SamplingConfig
    ) -> list[str]: ...

    def log_probs(
        self, prompts: list[str], completions: list[str]
    ) -> np.ndarray: ...

    def ref_log_probs(
        self, prompts: list[str], completions: list[str]
    ) -> np.ndarray: ...

    def step(
        self,
        loss: float,
        step_idx: int,
        *,
        advantages: np.ndarray | None = None,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.0,
        entropy_coef: float = 0.0,
        sample_weights: np.ndarray | None = None,
    ) -> float: ...


class EarlyStopTriggered(RuntimeError):
    """Raised when the val metric has degraded past patience.

    Caught by run() so the training loop exits cleanly with the existing
    checkpoint-best in place. We deliberately treat this as a controlled
    early exit, not an error — the SLURM job should mark COMPLETED.
    """


@dataclass
class TrainerConfig:
    total_steps: int = 5000
    batch_prompts: int = 4
    group_size: int = 16
    seed: int = 42
    log_path: Optional[Path] = None
    max_new_tokens: int = 512
    temperature: float = 0.7

    # Checkpoint cadence + retention. SLURM 11hr cap on Arm A's 42hr run
    # means we need periodic saves; save_every=0 disables saves (still
    # writes a final checkpoint at end of run for backwards-compat).
    # keep_last_k rotates intermediate checkpoints; the final-step one is
    # never rotated out. output_dir is where checkpoint-{step}/ lands and
    # must be set when save_every > 0.
    save_every: int = 0
    keep_last_k: int = 3
    output_dir: Optional[Path] = None

    # Training-time eval (§9 / P0.3). When eval_prompts is set, the
    # trainer runs the policy against it every eval_every steps,
    # producing eval_log.jsonl. The per-eval best by
    # func_sec_at_1__compiles_and_has_tests is symlinked to
    # output_dir/checkpoint-best/ for downstream consumers (P0.4).
    # eval_every=0 disables, even when eval_prompts is set, so smoke
    # tests stay cheap.
    eval_every: int = 0
    eval_prompts: Optional[list] = None  # list[Prompt]; typed at runtime
    eval_temperature: float = 0.0  # determinism per spec §9
    eval_max_prompts: int = 0       # 0 = all val prompts

    # Early-stop tripwire. After each eval, if val_metric drops by at
    # least `early_stop_drop_pp / 100` from the best-so-far for
    # `early_stop_patience` consecutive evals, the trainer aborts and
    # leaves checkpoint-best in place. Default disabled (patience=0). For
    # the headline (1000 steps, eval-every-100) we use patience=4 (a
    # 400-step window). Matches the Open-RS / Dang & Ngo finding that
    # GRPO collapse at 1.5B becomes irrecoverable past a few hundred
    # steps of degradation.
    early_stop_patience: int = 0
    early_stop_drop_pp: float = 5.0

    # Training-time prompt-prepend RAG (P1.3 cell 3/4, distinct from
    # reward-time RAG). When enabled, before each rollout the trainer
    # retrieves the top-1 secure exemplar for the prompt's CWE+lang and
    # prepends it as a reference block to prompt_text. The model SEES
    # the exemplar at generation time. Requires the reward_pipeline to
    # already have a retriever+embedder loaded (paths are shared).
    prepend_rag_on: bool = False

    # Adaptive σ_floor schedule (Dr.GRPO §7.6). When
    # sigma_floor_anneal_steps > 0, the algorithm's σ_floor is linearly
    # interpolated from sigma_floor_initial at step 0 to sigma_floor_final
    # at sigma_floor_anneal_steps; held at sigma_floor_final after. The
    # trainer mutates self.algorithm.config.sigma_floor each step (the
    # algorithm step reads it fresh). When sigma_floor_anneal_steps == 0,
    # the algorithm's σ_floor is left at whatever it was constructed with
    # (a constant) — preserves pre-schedule behavior.
    sigma_floor_initial: float = 0.0
    sigma_floor_final: float = 0.0
    sigma_floor_anneal_steps: int = 0

    # Entropy bonus with linear decay. When entropy_coef_initial > 0, a
    # policy-gradient entropy regularizer `H ≈ -mean(log_probs)` is added
    # to the policy step's objective with coefficient that decays linearly
    # from entropy_coef_initial at step 0 to entropy_coef_final at
    # entropy_decay_steps. Default 0.0 disables the bonus (pre-schedule
    # behavior). The bonus is computed in TorchPolicy.step using cached
    # log-probs already on the autograd graph.
    entropy_coef_initial: float = 0.0
    entropy_coef_final: float = 0.0
    entropy_decay_steps: int = 0


class Trainer:
    """Drives the per-step loop. Torch-free; uses PolicyProtocol.

    Optional `reweighter` supplies the CWE-aware per-prompt weights w(x)
    of paper Eq. 3; each prompt's weight scales its rollouts' surrogate
    loss. None trains with uniform weights.
    """

    def __init__(
        self,
        policy: PolicyProtocol,
        reward_pipeline: RewardPipeline,
        schedule: PhaseSchedule,
        algorithm: AlgorithmStep,
        config: TrainerConfig,
        *,
        reweighter: Optional[Reweighter] = None,
    ) -> None:
        self.policy = policy
        self.reward_pipeline = reward_pipeline
        self.schedule = schedule
        self.algorithm = algorithm
        self.config = config
        self.reweighter = reweighter
        self.aggregator = StepMetricsAggregator()
        self._rng = np.random.default_rng(config.seed)
        # Best eval state for P0.4 (best-checkpoint-by-eval-metric symlink).
        # Updated each time _run_eval reports a higher func_sec metric than
        # the running maximum; the symlink is rewritten by
        # _link_best_checkpoint. -inf means "no eval seen yet."
        self._best_eval_metric: float = float("-inf")
        # Early-stop tripwire. _early_stop_strikes increments on each eval
        # that scores below (best - drop_pp/100). Reset to 0 on a new best.
        # When strikes >= early_stop_patience, _run_step raises
        # EarlyStopTriggered.
        self._early_stop_strikes: int = 0

    # ---- main loop ----

    def run(
        self, prompts: list[Prompt], *, start_step: int = 0,
    ) -> list[TrainStepRecord]:
        """Run for `config.total_steps` steps. Returns per-step records.

        If `config.log_path` is set, the records are also written to a
        JSONL file (one record per line, in step order).

        `start_step` lets a resumed run pick up from a saved checkpoint:
        the loop iterates from `start_step` to `config.total_steps`, the
        log file is opened in append mode (so the previous run's records
        are preserved), and the RNG is fast-forwarded by replaying
        `start_step` draws (the same sequence the original run consumed
        before being interrupted). This keeps the train_log monotonic in
        step and reproducible under resume.
        """
        # Append-mode when resuming so previous train_log records
        # survive the SLURM kill that triggered the resume.
        log_mode = "a" if start_step > 0 else "w"
        log_fh = None
        eval_log_fh = None
        if self.config.log_path is not None:
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(self.config.log_path, log_mode)
            if self.config.eval_every > 0 and self.config.eval_prompts:
                eval_log_path = self.config.log_path.parent / "eval_log.jsonl"
                eval_log_fh = open(eval_log_path, log_mode)

        # Fast-forward the RNG so the post-resume batch sampling matches
        # what a fresh run would draw. Each step consumes one `integers`
        # call of size `batch_prompts`; replaying them yields an identical
        # state. Cheap (microseconds per step).
        for _ in range(start_step):
            self._rng.integers(
                0, max(1, len(prompts)), size=self.config.batch_prompts
            )

        # Reliability-signal health guard: hard-fail at step 50 if
        # cumulative mean_r_reliability is exactly 0.0 across the window.
        # The 2026-06-19 incident was a silent zero across 1000 steps
        # because rollout completions were fenced and ast.parse rejected
        # them, producing compiles=False / r_rel=0 uniformly. Catches
        # code-path bugs where the oracle never runs at all.
        #
        # The prior sustained-low rolling-mean variant was removed on
        # 2026-06-23 after it false-positived on the v0.1.7 corpus where
        # only ~1.6% of training prompts have unit tests, so steady-state
        # r_rel sits around 0.01-0.04, well below any "5% over 50 steps"
        # threshold. Strict-zero is the right check for "oracle never
        # runs"; sparse-but-real signal must be allowed.
        _RREL_EARLY_WINDOW = 50
        _r_rel_sum_window = 0.0
        try:
            records: list[TrainStepRecord] = []
            for step in range(start_step, self.config.total_steps):
                record = self._run_step(step, prompts)
                records.append(record)
                # Accumulate within the early window. Skip the resume case:
                # if we resumed past the guard window the check is moot
                # because the saved checkpoint already passed it.
                if step < _RREL_EARLY_WINDOW:
                    _r_rel_sum_window += record.mean_r_reliability or 0.0
                elif step == _RREL_EARLY_WINDOW:
                    if _r_rel_sum_window == 0.0:
                        raise RuntimeError(
                            f"r_reliability stayed at exactly 0.0 across all "
                            f"{_RREL_EARLY_WINDOW} initial steps. The reward "
                            f"pipeline is not seeing functional signal. Most "
                            f"likely cause: rollout completions are not being "
                            f"fence-stripped before reaching the oracle "
                            f"(see commit history around 2026-06-19 for "
                            f"context). Inspect trainer._run_step's call to "
                            f"reward_pipeline.evaluate(...) — the completion "
                            f"argument must pass through _extract_code first."
                        )
                if log_fh is not None:
                    log_fh.write(json.dumps(record.to_dict()) + "\n")
                    log_fh.flush()

                # Periodic checkpoint save (§P0.1). Skips when save_every=0
                # OR no output_dir set OR step is the first (we already have
                # the start_step state from resume / fresh init). The final
                # checkpoint is saved by train_method.py after run() returns.
                if (
                    self.config.save_every > 0
                    and self.config.output_dir is not None
                    and (step + 1) % self.config.save_every == 0
                    and (step + 1) < self.config.total_steps
                ):
                    self._save_checkpoint(step + 1)

                # Training-time eval (§P0.3). Runs after the save above so
                # checkpoint-best/ can symlink the checkpoint just written
                # if the eval was a new best.
                if (
                    self.config.eval_every > 0
                    and self.config.eval_prompts is not None
                    and (step + 1) % self.config.eval_every == 0
                ):
                    eval_agg = self._run_eval(step + 1)
                    if eval_agg is not None and eval_log_fh is not None:
                        eval_log_fh.write(json.dumps(eval_agg) + "\n")
                        eval_log_fh.flush()
                        # Promote checkpoint-best by the func+sec composite
                        # — the same metric we'd report in the headline paper
                        # table. Only fires if a checkpoint was just saved
                        # this step (the symlink target must exist).
                        best_metric = eval_agg.get(
                            "func_sec_at_1__compiles_and_has_tests", 0.0
                        )
                        if (
                            self.config.save_every > 0
                            and (step + 1) % self.config.save_every == 0
                        ):
                            self._link_best_checkpoint(step + 1, best_metric)

                        # Early-stop tripwire. Compare to best seen so
                        # far; tally a strike if we're meaningfully below
                        # it. patience consecutive strikes => abort. We
                        # only count strikes AFTER we have a best (i.e.
                        # at least one eval has happened), so the first
                        # eval can never trigger.
                        if self.config.early_stop_patience > 0:
                            drop = self._best_eval_metric - best_metric
                            if (
                                self._best_eval_metric > float("-inf")
                                and drop * 100.0 >= self.config.early_stop_drop_pp
                            ):
                                self._early_stop_strikes += 1
                            else:
                                # Either a new best or within tolerance.
                                self._early_stop_strikes = 0
                            if (
                                self._early_stop_strikes
                                >= self.config.early_stop_patience
                            ):
                                raise EarlyStopTriggered(
                                    f"val metric dropped >= "
                                    f"{self.config.early_stop_drop_pp}pp "
                                    f"from best ({self._best_eval_metric:.4f}) "
                                    f"for {self._early_stop_strikes} "
                                    f"consecutive evals; current="
                                    f"{best_metric:.4f} at step {step + 1}. "
                                    "Leaving checkpoint-best in place."
                                )

            return records
        except EarlyStopTriggered as e:
            # Controlled abort. Log the reason; downstream sees a normal
            # return with the records collected so far.
            print(f"[trainer] EARLY STOP: {e}", flush=True)
            return records
        finally:
            if log_fh is not None:
                log_fh.close()
            if eval_log_fh is not None:
                eval_log_fh.close()

    def _run_eval(self, step: int) -> Optional[dict]:
        """Run the policy against eval_prompts and return aggregate metrics.

        Generates one completion per val prompt at temperature 0, scores
        through the same reward_pipeline used for rollouts, and feeds the
        per-prompt records through eval.metrics.compute_all() to get the
        compile-first family. Returns the aggregate dict or None if eval
        is disabled. The trainer writes this dict to eval_log.jsonl.

        This is §9 of the launch plan ("eval at training time"). It runs
        on the same compute budget as the rollout step (rough math:
        n_val_prompts forward passes at temperature 0). With n_val ~ 30
        and one forward pass per prompt, ~5 min wall every eval_every
        steps — cheap compared to 250-step training windows.
        """
        if (
            self.config.eval_every <= 0
            or self.config.eval_prompts is None
            or len(self.config.eval_prompts) == 0
        ):
            return None

        from ..eval.harness import PerPromptRecord
        from ..eval.metrics import compute_all, METRICS

        sampling = SamplingConfig(
            temperature=self.config.eval_temperature,
            max_new_tokens=self.config.max_new_tokens,
            n_samples=1,
            seed=self.config.seed + step,  # deterministic per (config, step)
        )

        eval_prompts = list(self.config.eval_prompts)
        if self.config.eval_max_prompts > 0:
            eval_prompts = eval_prompts[: self.config.eval_max_prompts]

        records: list[PerPromptRecord] = []
        for prompt in eval_prompts:
            try:
                comps = self.policy.generate(
                    [prompt.prompt_text], sampling=sampling
                )
                completion = comps[0] if comps else ""
            except Exception:
                # Skip prompts that hit a generation error; record an
                # empty/crashed entry so the denominators still match.
                records.append(PerPromptRecord(
                    prompt_id=prompt.id,
                    source=prompt.source,
                    target_cwe=prompt.target_cwe,
                    language=prompt.language.value,
                    completion="",
                    crashed=True,
                    refusal_or_empty=True,
                    n_test_cases=len(prompt.test_spec.test_cases),
                ))
                continue

            query_embedding = None
            if (
                self.reward_pipeline.retriever is not None
                and self.reward_pipeline.embedder is not None
            ):
                src = prompt.task_signature or prompt.prompt_text
                query_embedding = self.reward_pipeline.embedder.embed(src)
            ctx = PromptContext(
                target_cwe=prompt.target_cwe,
                language=prompt.language.value,
                test_spec=prompt.test_spec,
                task_signature=prompt.task_signature,
                query_embedding=query_embedding,
            )
            # Strip fences before scoring (same as rollout path / eval harness).
            clean = _extract_code(completion, prompt.language.value)
            out = self.reward_pipeline.evaluate(
                prompt.prompt_text, clean, ctx
            )
            findings_cwes = [
                f["cwe"] for f in out.breakdown.per_finding
            ]
            target_present = prompt.target_cwe in findings_cwes

            records.append(PerPromptRecord(
                prompt_id=prompt.id,
                source=prompt.source,
                target_cwe=prompt.target_cwe,
                language=prompt.language.value,
                completion=completion,
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
                sast_crashed_tools=list(out.diagnostics.sast_crashed_tools),
                sast_timed_out_tools=list(out.diagnostics.sast_timed_out_tools),
            ))

        per_spec = compute_all(records)
        aggregate = {
            "step": step,
            "n_prompts": len(records),
        }
        # Flatten per-spec dicts to scalar value + counts for the eval_log row.
        for spec in METRICS:
            s = per_spec[spec.name]
            aggregate[spec.name] = s["value"]
            aggregate[spec.name + ".n_numerator"] = s["n_numerator"]
            aggregate[spec.name + ".n_denominator"] = s["n_denominator"]
        return aggregate

    def _link_best_checkpoint(self, step: int, metric: float) -> None:
        """Update <output>/checkpoint-best to point at the current step.

        P0.4: the trainer remembers the best
        `func_sec_at_1__compiles_and_has_tests` seen so far in
        self._best_eval_metric and symlinks checkpoint-best when a new
        best lands. Symlink (not copy) so disk use is bounded; the
        underlying checkpoint-{step}/ is exempt from rotation while
        checkpoint-best points at it.
        """
        if self.config.output_dir is None:
            return
        if metric <= self._best_eval_metric:
            return
        self._best_eval_metric = metric
        link_path = self.config.output_dir / "checkpoint-best"
        target = self.config.output_dir / f"checkpoint-{step}"
        if not target.exists():
            return
        import os as _os
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        # Relative symlink so the link survives directory moves.
        _os.symlink(target.name, link_path)

    def _save_checkpoint(self, step: int) -> None:
        """Save a recoverable training-state snapshot to <output>/checkpoint-{step}/.

        Layout under the checkpoint dir:
            adapter/                LoRA adapter via peft.save_pretrained
            adapter/tokenizer/      tokenizer files
            optimizer.pt            torch.save of policy optimizer state
            trainer_state.json      step, rng state
        """
        ckpt_dir = self.config.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = ckpt_dir / "adapter"

        # Adapter + tokenizer. Done via the policy's save method since
        # the trainer is torch-free; the policy is the side that owns
        # the model.
        if hasattr(self.policy, "save_adapter"):
            self.policy.save_adapter(adapter_dir)
        else:
            # Backwards-compat for older policy stubs: hit save_pretrained
            # directly if the policy exposes _model.
            _model = getattr(self.policy, "_model", None)
            _tokenizer = getattr(self.policy, "_tokenizer", None)
            if _model is not None and hasattr(_model, "save_pretrained"):
                adapter_dir.mkdir(parents=True, exist_ok=True)
                _model.save_pretrained(str(adapter_dir))
                if _tokenizer is not None:
                    _tokenizer.save_pretrained(str(adapter_dir))

        # Optimizer state. Skipped silently if torch isn't available
        # (mock policy in tests). Stored separately from the adapter so
        # the same checkpoint dir can be loaded for inference-only use
        # without dragging the optimizer pickle into a paper artifact.
        if hasattr(self.policy, "save_optimizer_state"):
            self.policy.save_optimizer_state(ckpt_dir / "optimizer.pt")
        if hasattr(self.policy, "save_scheduler_state"):
            self.policy.save_scheduler_state(ckpt_dir / "scheduler.pt")

        # Trainer state: step counter (for resume start_step) + RNG
        # state (for reproducibility).
        trainer_state = {
            "step": step,
            "rng_state": self._rng.bit_generator.state,
            "config": {
                "total_steps": self.config.total_steps,
                "batch_prompts": self.config.batch_prompts,
                "group_size": self.config.group_size,
                "seed": self.config.seed,
            },
        }
        (ckpt_dir / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2, default=str)
        )

        # Rotation: keep the latest keep_last_k checkpoint-{step}/ dirs
        # (skipping checkpoint-{total_steps} which is the final one written
        # outside this method by train_method.py). Older ones are removed
        # to bound disk usage during long Arm A runs.
        self._rotate_checkpoints()

    def _rotate_checkpoints(self) -> None:
        """Keep only the last config.keep_last_k periodic checkpoints."""
        if self.config.output_dir is None or self.config.keep_last_k <= 0:
            return
        import re
        import shutil
        pat = re.compile(r"checkpoint-(\d+)$")
        ckpts: list[tuple[int, Path]] = []
        for child in self.config.output_dir.iterdir():
            m = pat.match(child.name)
            if m and child.is_dir():
                ckpts.append((int(m.group(1)), child))
        ckpts.sort(key=lambda t: t[0])
        n_to_keep = self.config.keep_last_k
        if len(ckpts) > n_to_keep:
            for _, doomed in ckpts[: len(ckpts) - n_to_keep]:
                shutil.rmtree(doomed, ignore_errors=True)

    @staticmethod
    def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
        """Return the most-advanced checkpoint-{N}/ in output_dir, or None.

        Used by train_method.py's --resume-from auto-discovery (`--resume-from
        <output_dir>` finds the latest checkpoint by step number, not by
        mtime, so a partial write during a kill can't mask an earlier
        completed checkpoint).
        """
        import re
        pat = re.compile(r"checkpoint-(\d+)$")
        candidates: list[tuple[int, Path]] = []
        if not output_dir.exists():
            return None
        for child in output_dir.iterdir():
            m = pat.match(child.name)
            if m and child.is_dir() and (child / "trainer_state.json").exists():
                candidates.append((int(m.group(1)), child))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[-1][1]

    @staticmethod
    def load_trainer_state(ckpt_dir: Path) -> dict:
        """Read trainer_state.json from a checkpoint dir."""
        return json.loads((ckpt_dir / "trainer_state.json").read_text())

    def _prepend_exemplar(self, prompt: Prompt) -> str:
        """Build a prompt-prepend-RAG variant for a Prompt.

        Returns the prepended text if a top-1 secure exemplar is found
        for the prompt's CWE; otherwise returns the bare prompt_text
        unchanged (silently falls back to no prepend).

        Distinct from reward-time RAG: here the model SEES the exemplar
        during generation. See P1.3 cell 3/4.
        """
        if (
            self.reward_pipeline.retriever is None
            or self.reward_pipeline.embedder is None
        ):
            return prompt.prompt_text
        from ..rag.retriever import RetrievalQuery
        src = prompt.task_signature or prompt.prompt_text
        query = RetrievalQuery(
            cwe=prompt.target_cwe,
            task_signature=src,
            query_embedding=self.reward_pipeline.embedder.embed(src),
        )
        hit = self.reward_pipeline.retriever.retrieve(query)
        if hit is None:
            return prompt.prompt_text
        return (
            f"// Reference secure implementation for {prompt.target_cwe}:\n"
            f"{hit.pair.e_pos}\n\n"
            f"// Now complete:\n{prompt.prompt_text}"
        )

    # ---- step ----

    def _run_step(
        self, step: int, prompts: list[Prompt]
    ) -> TrainStepRecord:
        t0 = time.monotonic()
        phase = self.schedule.for_step(step)

        # 1. Sample prompts.
        idx = self._rng.integers(0, len(prompts), size=self.config.batch_prompts)
        batch = [prompts[i] for i in idx]
        group_cwes = [p.target_cwe for p in batch]

        # 2. For each prompt, generate `group_size` rollouts.
        sampling = SamplingConfig(
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            n_samples=self.config.group_size,
            seed=self.config.seed + step,
        )
        per_prompt_completions: list[list[str]] = []
        for prompt in batch:
            # The policy receives `group_size` copies of the same prompt
            # text so the generate() implementation can batch within the
            # group. Test mock returns the same completion for each.
            # When prepend_rag_on, _prepend_exemplar wraps prompt_text
            # with the top-1 secure exemplar as a reference block.
            text = (
                self._prepend_exemplar(prompt)
                if self.config.prepend_rag_on
                else prompt.prompt_text
            )
            comps = self.policy.generate(
                [text] * self.config.group_size, sampling=sampling
            )
            per_prompt_completions.append(comps)

        # 3. Score every completion via the reward pipeline.
        # Layout: rewards.shape = (n_groups, group_size).
        all_outputs = []  # flat list for metrics aggregator
        rewards = np.zeros(
            (self.config.batch_prompts, self.config.group_size), dtype=float
        )
        for g, (prompt, comps) in enumerate(zip(batch, per_prompt_completions)):
            # When RAG is wired in the reward pipeline, embed the
            # task_signature once per group (16 rollouts share a prompt)
            # so the dense backend has a query vector. Without this the
            # FaissBackend.search short-circuits to [] on
            # `query.query_embedding is None` and R_RAG drops to the
            # `r_rag_missing()` sentinel.
            query_embedding = None
            if (
                self.reward_pipeline.retriever is not None
                and self.reward_pipeline.embedder is not None
            ):
                # Prefer task_signature (terse, focused) over full
                # prompt_text (noisy, may exceed embedder context).
                src = prompt.task_signature or prompt.prompt_text
                query_embedding = self.reward_pipeline.embedder.embed(src)

            ctx = PromptContext(
                target_cwe=prompt.target_cwe,
                language=prompt.language.value,
                test_spec=prompt.test_spec,
                task_signature=prompt.task_signature,
                query_embedding=query_embedding,
            )
            for i, completion in enumerate(comps):
                # Strip markdown fences before scoring, matching the eval
                # harness. Without this, fenced completions fail ast.parse,
                # forcing compiles=False and r_reliability=0 — the policy
                # then sees zero functional signal and optimizes only against
                # security + RAG.
                clean = _extract_code(completion, prompt.language.value)
                out = self.reward_pipeline.evaluate(
                    prompt.prompt_text, clean, ctx
                )
                rewards[g, i] = out.breakdown.r_total
                all_outputs.append(out)

        # 4. Log-probs under current and reference policies.
        # Each group's rollouts come from the same prompt; stack the
        # log-probs per group.
        log_probs = np.zeros_like(rewards)
        ref_log_probs = np.zeros_like(rewards)
        for g, (prompt, comps) in enumerate(zip(batch, per_prompt_completions)):
            log_probs[g] = self.policy.log_probs(
                [prompt.prompt_text] * self.config.group_size, comps
            )
            ref_log_probs[g] = self.policy.ref_log_probs(
                [prompt.prompt_text] * self.config.group_size, comps
            )

        # 4b. CWE-aware per-prompt weights w(x) (Eq. 3). None = uniform.
        per_group_weight = None
        if self.reweighter is not None:
            per_group_weight = self.reweighter.weights_for_batch(group_cwes)

        # 4c. PPO value head: when the algorithm declares
        # `needs_value_head=True` AND the policy exposes a `values()`
        # method (i.e. enable_value_head was called), compute per-sample
        # values now so the algorithm can do A = R - V instead of falling
        # back to REINFORCE. See FINDINGS_LOG 2026-06-14: silent REINFORCE
        # fallback was the original PPO bug.
        values = None
        algo_needs_values = getattr(self.algorithm, "needs_value_head", False)
        policy_has_values = hasattr(self.policy, "values")
        if algo_needs_values and policy_has_values:
            values = np.zeros_like(rewards)
            for g, (prompt, comps) in enumerate(zip(batch, per_prompt_completions)):
                values[g] = self.policy.values(
                    [prompt.prompt_text] * self.config.group_size, comps
                )

        # 4d. Adaptive σ_floor schedule (TrainerConfig fields). When
        # sigma_floor_anneal_steps > 0, override the algorithm's σ_floor
        # for this step. The algorithm's `.config.sigma_floor` is read
        # inside its `.step()` so mutating it here takes effect this call.
        if (
            self.config.sigma_floor_anneal_steps > 0
            and hasattr(self.algorithm, "config")
            and hasattr(self.algorithm.config, "sigma_floor")
        ):
            t = min(1.0, step / float(self.config.sigma_floor_anneal_steps))
            sigma_floor_eff = (
                self.config.sigma_floor_initial
                + t * (self.config.sigma_floor_final - self.config.sigma_floor_initial)
            )
            self.algorithm.config.sigma_floor = sigma_floor_eff

        # 5. Algorithm step -> loss.
        algo_step_kwargs = {
            "rewards": rewards,
            "log_probs": log_probs,
            "ref_log_probs": ref_log_probs,
            "per_group_weight": per_group_weight,
        }
        if values is not None:
            algo_step_kwargs["values"] = values
        algo_out: GrpoStepOutput = self.algorithm.step(**algo_step_kwargs)

        # 5b. Entropy coefficient schedule. Linearly interpolate between
        # initial and final over entropy_decay_steps; hold final after.
        # Default 0 → no entropy bonus; passed to policy.step which adds
        # `entropy_coef * log_probs.mean()` to the surrogate (the standard
        # policy-gradient entropy proxy at the sampled actions).
        entropy_coef_eff = 0.0
        if self.config.entropy_decay_steps > 0:
            t = min(1.0, step / float(self.config.entropy_decay_steps))
            entropy_coef_eff = (
                self.config.entropy_coef_initial
                + t * (self.config.entropy_coef_final - self.config.entropy_coef_initial)
            )
        elif self.config.entropy_coef_initial > 0.0:
            entropy_coef_eff = self.config.entropy_coef_initial

        # 6. Policy step -> grad_norm. Hand the algorithm's advantage
        # tensor + clip + kl_beta to the policy so it can reconstruct the
        # autograd surrogate (see TorchPolicy.step for the math). For PPO
        # we also pass `rewards` so the value head can be regressed via
        # MSE(V_i, R_i).
        step_kwargs = {
            "advantages": algo_out.advantages,
            "clip_epsilon": algo_out.clip_epsilon,
            "kl_beta": algo_out.kl_beta,
            "entropy_coef": entropy_coef_eff,
        }
        if per_group_weight is not None:
            # Algorithm 1 step 11: the optimizer steps on w(x) * grad L.
            step_kwargs["sample_weights"] = np.repeat(
                per_group_weight, self.config.group_size
            )
        if values is not None:
            step_kwargs["rewards"] = rewards
        grad_norm = self.policy.step(
            algo_out.loss,
            step,
            **step_kwargs,
        )

        # 8. Aggregate per-step metrics. Pass the flat completions so the
        # aggregator can compute mean length + truncation rate (Open-RS-
        # style "responses ballooning to max_new_tokens" canary).
        flat_completions: list[str] = []
        for comps in per_prompt_completions:
            flat_completions.extend(comps)
        # kl_mean is the unweighted per-sample KL divergence in nats.
        # kl_loss is beta * kl_mean (the loss contribution); recover the
        # raw KL by dividing out beta. Guards against beta=0 phases.
        kl_mean_value = (
            algo_out.kl_loss / phase.beta if phase.beta > 0 else 0.0
        )
        record = self.aggregator.aggregate(
            step=step,
            phase=phase.number,
            alpha=phase.alpha,
            beta=phase.beta,
            outputs=all_outputs,
            policy_loss=algo_out.policy_loss,
            kl_loss=algo_out.kl_loss,
            grad_norm=grad_norm,
            wall_s_step=time.monotonic() - t0,
            n_groups_masked=algo_out.n_groups_masked,
            n_groups_total=algo_out.n_groups_total,
            kl_mean=kl_mean_value,
            completions=flat_completions,
            max_new_tokens=self.config.max_new_tokens,
            group_stds=[
                g.get("std_reward", 0.0) for g in algo_out.per_group_diagnostics
            ],
        )
        return record
