"""Per-step metrics aggregation for the trainer log.

Implements docs/training_spec.md §4. The aggregator takes per-rollout
PipelineOutputs plus per-step loss + wall-clock + group-mask numbers
from the algorithm step and produces one TrainStepRecord per step.

TrainStepRecord is intentionally pure data: serializable to JSON so the
training log is a flat JSONL stream. Downstream dashboards (wandb, etc.)
consume the same record without re-running anything.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from ..reward.pipeline import PipelineOutput

# Within-group reward std below which a group counts as zero-variance for
# the paper's reporting definition (Section II.C). Distinct from the GRPO
# masking threshold in rl/grpo.py (1e-6), which decides whether a group
# contributes gradient at all.
ZVG_STD_THRESHOLD = 0.01


@dataclass
class TrainStepRecord:
    """One per-step row in the training log. See docs/training_spec.md §4."""

    step: int
    phase: int = 1
    alpha: float = 0.0
    beta: float = 0.0
    mean_reward: float = 0.0
    mean_r_reliability: float = 0.0
    mean_r_security: float = 0.0
    mean_r_rag: float = 0.0
    gate_rate: float = 0.0
    refusal_rate: float = 0.0
    clip_floor_saturation_rate: float = 0.0
    copy_guard_hit_rate: float = 0.0
    rag_miss_rate: float = 0.0
    sast_crash_rate_per_tool: dict[str, float] = field(default_factory=dict)
    zero_variance_group_rate: float = 0.0
    # Paper-definition zero-variance-group rate (Section II.C): fraction
    # of groups whose within-group reward std is below 0.01. The field
    # above is the algorithm's masking rate (std < 1e-6, GRPO only);
    # this one is algorithm-agnostic and computed from the per-group
    # reward stds every algorithm reports.
    zero_variance_group_rate_0p01: float = 0.0
    kl_mean: float = 0.0
    policy_loss: float = 0.0
    kl_loss: float = 0.0
    total_loss: float = 0.0
    grad_norm: float = 0.0
    # Algorithm-agnostic "did we actually update?" flag. True when
    # |policy_loss| > 1e-6. Catches the RLOO-style silent collapse where
    # zero_variance_group_rate stays 0 but the algorithm still produces
    # zero policy gradient (rloo.py:12-15). Computed post-hoc as
    # rolling_window_100 fraction for early-warning.
    policy_loss_nonzero: bool = False
    wall_s_step: float = 0.0
    n_rollouts: int = 0
    # Length / stub telemetry. mean_completion_chars surfaces the
    # Open-RS-style "responses ballooning to max_new_tokens" pattern;
    # truncation_rate is fraction of completions at or near max_new_tokens;
    # stub_rate is fraction of completions flagged by _looks_like_stub in
    # the reward pipeline (zero r_reliability). Stub_rate climbing is the
    # Goodhart canary for secure-code RL — model learns to emit `pass`.
    mean_completion_chars: float = 0.0
    truncation_rate: float = 0.0
    stub_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class StepMetricsAggregator:
    """Folds rollout outputs + algorithm-step numbers into a TrainStepRecord."""

    def aggregate(
        self,
        *,
        step: int,
        phase: int,
        alpha: float,
        beta: float,
        outputs: Sequence[PipelineOutput],
        policy_loss: float,
        kl_loss: float,
        grad_norm: float,
        wall_s_step: float,
        n_groups_masked: int,
        n_groups_total: int,
        kl_mean: float = 0.0,
        completions: Optional[Sequence[str]] = None,
        max_new_tokens: int = 0,
        group_stds: Optional[Sequence[float]] = None,
    ) -> TrainStepRecord:
        n = len(outputs)
        zvg_0p01 = (
            sum(1 for s in group_stds if s < ZVG_STD_THRESHOLD) / len(group_stds)
            if group_stds
            else 0.0
        )
        if n == 0:
            return TrainStepRecord(
                step=step,
                phase=phase,
                alpha=alpha,
                beta=beta,
                policy_loss=policy_loss,
                kl_loss=kl_loss,
                total_loss=policy_loss + kl_loss,
                grad_norm=grad_norm,
                wall_s_step=wall_s_step,
                zero_variance_group_rate=(
                    n_groups_masked / n_groups_total if n_groups_total else 0.0
                ),
                zero_variance_group_rate_0p01=zvg_0p01,
                kl_mean=kl_mean,
                policy_loss_nonzero=abs(policy_loss) > 1e-6,
                n_rollouts=0,
            )

        mean_reward = sum(o.breakdown.r_total for o in outputs) / n
        mean_r_rel = sum(o.breakdown.r_reliability for o in outputs) / n
        mean_r_sec = sum(o.breakdown.r_security for o in outputs) / n
        mean_r_rag = sum(o.breakdown.r_rag for o in outputs) / n
        gate_rate = sum(o.breakdown.gate for o in outputs) / n
        refusal_rate = sum(o.diagnostics.refusal_or_empty for o in outputs) / n
        clip_rate = sum(o.breakdown.clipped for o in outputs) / n

        rag_used = [o for o in outputs if not o.diagnostics.rag_missing]
        rag_used_n = len(rag_used)
        copy_guard_rate = (
            sum(o.diagnostics.rag_copy_guard_hit for o in rag_used) / rag_used_n
            if rag_used_n
            else 0.0
        )
        rag_miss_rate = sum(o.diagnostics.rag_missing for o in outputs) / n

        # Length / stub canaries. mean_completion_chars + truncation_rate
        # require the raw completion strings (PipelineOutput doesn't carry
        # them). When unavailable (older call sites that don't pass them)
        # we report 0.0 / 0.0 for those two metrics. stub_rate comes from
        # the pipeline diagnostics regardless.
        stub_rate = sum(
            getattr(o.diagnostics, "stub_detected", False) for o in outputs
        ) / n
        if completions is not None and len(completions) == n:
            char_lengths = [len(c) for c in completions]
            mean_completion_chars = sum(char_lengths) / n
            # Heuristic: completion is "truncated" if char length is in the
            # top 5% near max_new_tokens worth of chars. We don't have token
            # counts here so use 3 chars / token as a conservative proxy
            # (Qwen tokenizer averages ~3.5 chars/token on code). Skip when
            # max_new_tokens unset.
            if max_new_tokens > 0:
                trunc_threshold = int(0.95 * max_new_tokens * 3)
                truncation_rate = sum(
                    1 for L in char_lengths if L >= trunc_threshold
                ) / n
            else:
                truncation_rate = 0.0
        else:
            mean_completion_chars = 0.0
            truncation_rate = 0.0

        # SAST per-tool crash rate: count crashes per tool across outputs / n.
        tool_crash_counts: dict[str, int] = defaultdict(int)
        for o in outputs:
            for tool in o.diagnostics.sast_crashed_tools:
                tool_crash_counts[tool] += 1
        per_tool_rate = {tool: c / n for tool, c in tool_crash_counts.items()}

        zero_var_rate = (
            n_groups_masked / n_groups_total if n_groups_total else 0.0
        )

        return TrainStepRecord(
            step=step,
            phase=phase,
            alpha=alpha,
            beta=beta,
            mean_reward=mean_reward,
            mean_r_reliability=mean_r_rel,
            mean_r_security=mean_r_sec,
            mean_r_rag=mean_r_rag,
            gate_rate=gate_rate,
            refusal_rate=refusal_rate,
            clip_floor_saturation_rate=clip_rate,
            copy_guard_hit_rate=copy_guard_rate,
            rag_miss_rate=rag_miss_rate,
            sast_crash_rate_per_tool=per_tool_rate,
            zero_variance_group_rate=zero_var_rate,
            zero_variance_group_rate_0p01=zvg_0p01,
            kl_mean=kl_mean,
            policy_loss=policy_loss,
            kl_loss=kl_loss,
            total_loss=policy_loss + kl_loss,
            grad_norm=grad_norm,
            policy_loss_nonzero=abs(policy_loss) > 1e-6,
            wall_s_step=wall_s_step,
            n_rollouts=n,
            mean_completion_chars=mean_completion_chars,
            truncation_rate=truncation_rate,
            stub_rate=stub_rate,
        )
