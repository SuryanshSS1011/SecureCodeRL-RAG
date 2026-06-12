"""Tests for TrainStepRecord + the per-step metrics aggregator.

Pins docs/training_spec.md §4. The aggregator takes per-rollout pipeline
outputs and produces one TrainStepRecord with all canaries rolled up.
"""

from __future__ import annotations

import pytest

from secure_code_rl_ictai.reward import (
    PipelineDiagnostics,
    PipelineOutput,
    ReliabilitySignals,
    RewardBreakdown,
)
from secure_code_rl_ictai.rl.metrics import (
    StepMetricsAggregator,
    TrainStepRecord,
)


def _pipeline_output(
    *,
    r_total: float = 0.5,
    r_reliability: float = 0.6,
    r_security: float = -0.1,
    r_rag: float = 0.0,
    gate: int = 1,
    findings_count: int = 0,
    clipped: bool = False,
    refusal: bool = False,
    copy_guard: bool = False,
    sast_crashed_tools: list[str] | None = None,
) -> PipelineOutput:
    return PipelineOutput(
        breakdown=RewardBreakdown(
            r_total=r_total,
            r_reliability=r_reliability,
            r_security=r_security,
            r_rag=r_rag,
            gate=gate,
            findings_count=findings_count,
            clipped=clipped,
        ),
        diagnostics=PipelineDiagnostics(
            reliability=ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1),
            sast_crashed_tools=sast_crashed_tools or [],
            refusal_or_empty=refusal,
            rag_copy_guard_hit=copy_guard,
        ),
    )


# ----------------------------------------------------------------------
# TrainStepRecord dataclass
# ----------------------------------------------------------------------


def test_train_step_record_defaults():
    rec = TrainStepRecord(step=0)
    assert rec.step == 0
    assert rec.mean_reward == 0.0
    assert rec.refusal_rate == 0.0


# ----------------------------------------------------------------------
# Aggregator math
# ----------------------------------------------------------------------


def test_aggregator_means_over_rollouts():
    outs = [
        _pipeline_output(r_total=0.6, r_reliability=0.8, r_security=-0.2, r_rag=0.0),
        _pipeline_output(r_total=0.4, r_reliability=0.7, r_security=-0.3, r_rag=0.0),
    ]
    agg = StepMetricsAggregator()
    rec = agg.aggregate(
        step=10,
        phase=1,
        alpha=0.3,
        beta=0.01,
        outputs=outs,
        policy_loss=0.05,
        kl_loss=0.01,
        grad_norm=0.5,
        wall_s_step=12.3,
        n_groups_masked=0,
        n_groups_total=2,
    )
    assert rec.step == 10
    assert rec.phase == 1
    assert rec.alpha == 0.3
    assert rec.mean_reward == 0.5
    assert rec.mean_r_reliability == 0.75
    assert rec.mean_r_security == -0.25
    assert rec.mean_r_rag == 0.0
    assert rec.policy_loss == 0.05
    assert rec.kl_loss == 0.01
    assert rec.total_loss == pytest.approx(0.06)
    assert rec.grad_norm == 0.5
    assert rec.wall_s_step == 12.3


def test_aggregator_gate_rate_counts_one_gates():
    outs = [
        _pipeline_output(gate=1),
        _pipeline_output(gate=1),
        _pipeline_output(gate=0),
        _pipeline_output(gate=0),
    ]
    rec = StepMetricsAggregator().aggregate(
        step=0, phase=1, alpha=0.3, beta=0.01,
        outputs=outs,
        policy_loss=0, kl_loss=0, grad_norm=0, wall_s_step=0,
        n_groups_masked=0, n_groups_total=1,
    )
    assert rec.gate_rate == 0.5


def test_aggregator_refusal_rate():
    outs = [
        _pipeline_output(refusal=True),
        _pipeline_output(refusal=True),
        _pipeline_output(refusal=False),
    ]
    rec = StepMetricsAggregator().aggregate(
        step=0, phase=1, alpha=0.3, beta=0.01,
        outputs=outs,
        policy_loss=0, kl_loss=0, grad_norm=0, wall_s_step=0,
        n_groups_masked=0, n_groups_total=1,
    )
    assert rec.refusal_rate == 2 / 3


def test_aggregator_clip_floor_saturation_rate():
    outs = [
        _pipeline_output(clipped=True),
        _pipeline_output(clipped=False),
        _pipeline_output(clipped=True),
        _pipeline_output(clipped=False),
    ]
    rec = StepMetricsAggregator().aggregate(
        step=0, phase=1, alpha=0.3, beta=0.01,
        outputs=outs,
        policy_loss=0, kl_loss=0, grad_norm=0, wall_s_step=0,
        n_groups_masked=0, n_groups_total=1,
    )
    assert rec.clip_floor_saturation_rate == 0.5


def test_aggregator_copy_guard_hit_rate():
    outs = [
        _pipeline_output(copy_guard=True),
        _pipeline_output(copy_guard=False),
    ]
    rec = StepMetricsAggregator().aggregate(
        step=0, phase=1, alpha=0.3, beta=0.01,
        outputs=outs,
        policy_loss=0, kl_loss=0, grad_norm=0, wall_s_step=0,
        n_groups_masked=0, n_groups_total=1,
    )
    assert rec.copy_guard_hit_rate == 0.5


def test_aggregator_sast_crash_rate_per_tool():
    outs = [
        _pipeline_output(sast_crashed_tools=["codeql"]),
        _pipeline_output(sast_crashed_tools=["codeql", "semgrep"]),
        _pipeline_output(sast_crashed_tools=[]),
    ]
    rec = StepMetricsAggregator().aggregate(
        step=0, phase=1, alpha=0.3, beta=0.01,
        outputs=outs,
        policy_loss=0, kl_loss=0, grad_norm=0, wall_s_step=0,
        n_groups_masked=0, n_groups_total=1,
    )
    # codeql crashed 2 of 3 rollouts; semgrep 1 of 3.
    assert rec.sast_crash_rate_per_tool["codeql"] == 2 / 3
    assert rec.sast_crash_rate_per_tool["semgrep"] == 1 / 3
    # Tools not seen in any crash list don't appear.
    assert "bandit" not in rec.sast_crash_rate_per_tool


def test_aggregator_zero_variance_group_rate():
    rec = StepMetricsAggregator().aggregate(
        step=0, phase=1, alpha=0.3, beta=0.01,
        outputs=[_pipeline_output()],
        policy_loss=0, kl_loss=0, grad_norm=0, wall_s_step=0,
        n_groups_masked=2, n_groups_total=8,
    )
    assert rec.zero_variance_group_rate == 0.25


def test_aggregator_empty_rollouts_returns_zeros():
    rec = StepMetricsAggregator().aggregate(
        step=0, phase=1, alpha=0.3, beta=0.01,
        outputs=[],
        policy_loss=0, kl_loss=0, grad_norm=0, wall_s_step=0,
        n_groups_masked=0, n_groups_total=0,
    )
    assert rec.mean_reward == 0.0
    assert rec.gate_rate == 0.0
    assert rec.refusal_rate == 0.0
    assert rec.sast_crash_rate_per_tool == {}


# ----------------------------------------------------------------------
# Serialization (for jsonl training log)
# ----------------------------------------------------------------------


def test_train_step_record_to_dict_is_json_safe():
    import json

    rec = TrainStepRecord(
        step=5,
        phase=2,
        alpha=0.5,
        beta=0.02,
        sast_crash_rate_per_tool={"codeql": 0.1, "semgrep": 0.0},
    )
    payload = rec.to_dict()
    # Round-trip through json without TypeError.
    json.dumps(payload)
    assert payload["step"] == 5
    assert payload["phase"] == 2


def test_aggregator_zero_variance_rate_at_paper_threshold():
    """Paper Section II.C: a group is zero-variance when its reward std is
    below 0.01. Independent of the GRPO masking rate (std < 1e-6)."""
    agg = StepMetricsAggregator()
    rec = agg.aggregate(
        step=0, phase=1, alpha=0.3, beta=0.01,
        outputs=[_pipeline_output()],
        policy_loss=0.1, kl_loss=0.0, grad_norm=1.0, wall_s_step=1.0,
        n_groups_masked=0, n_groups_total=4,
        group_stds=[0.0, 0.005, 0.02, 0.5],
    )
    assert rec.zero_variance_group_rate == 0.0
    assert rec.zero_variance_group_rate_0p01 == pytest.approx(0.5)
    assert "zero_variance_group_rate_0p01" in rec.to_dict()
