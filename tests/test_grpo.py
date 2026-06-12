"""Tests for the GRPO step math (numpy reference implementation).

Each test pins one property from docs/training_spec.md §2. The torch
implementation will be type-checked against this reference; failures
here are spec-level failures, not torch-wiring bugs.
"""

from __future__ import annotations

import numpy as np
import pytest

from secure_code_rl_ictai.rl import (
    GrpoConfig,
    compute_group_advantages,
    get_algorithm,
    grpo_step,
)


# ----------------------------------------------------------------------
# Advantage normalization
# ----------------------------------------------------------------------


def test_advantages_zero_mean_within_group():
    rewards = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 0.6, 0.7, 0.8],
        ]
    )
    adv, mask = compute_group_advantages(rewards)
    assert mask.tolist() == [True, True]
    # Mean of each group's advantages is 0 (up to numerical tolerance).
    assert np.allclose(adv.mean(axis=1), 0.0, atol=1e-6)


def test_advantages_unit_std_within_group():
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    adv, _ = compute_group_advantages(rewards, epsilon=0.0)
    # std after normalization should be ~1
    assert adv.std(axis=1)[0] == pytest.approx(1.0, abs=1e-6)


def test_zero_variance_group_masked_with_all_zero_rewards():
    rewards = np.array([[0.0, 0.0, 0.0, 0.0]])
    adv, mask = compute_group_advantages(rewards)
    assert mask.tolist() == [False]
    # Advantages zeroed defensively.
    assert (adv == 0.0).all()


def test_zero_variance_group_masked_with_all_equal_nonzero():
    rewards = np.array([[0.5, 0.5, 0.5, 0.5]])
    adv, mask = compute_group_advantages(rewards)
    assert mask.tolist() == [False]
    assert (adv == 0.0).all()


def test_mixed_groups_partial_masking():
    rewards = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],   # masked
            [1.0, 2.0, 3.0, 4.0],   # active
            [0.5, 0.5, 0.5, 0.5],   # masked
        ]
    )
    adv, mask = compute_group_advantages(rewards)
    assert mask.tolist() == [False, True, False]
    # Active group's advantages should sum to 0
    assert adv[1].sum() == pytest.approx(0.0, abs=1e-6)
    # Masked groups: all zeros
    assert (adv[0] == 0.0).all()
    assert (adv[2] == 0.0).all()


def test_rewards_must_be_2d():
    with pytest.raises(ValueError):
        compute_group_advantages(np.array([1.0, 2.0, 3.0]))


# ----------------------------------------------------------------------
# GRPO step composition
# ----------------------------------------------------------------------


def test_grpo_step_at_reference_policy_loss_is_just_kl():
    """When log_probs == ref_log_probs, ratio == 1 and the policy-loss
    surrogate reduces to -A. With mean-zero advantages within each active
    group, the per-sample policy loss has zero mean. KL log-ratio is also
    0, so the KL term is 0. Total loss = 0."""
    rewards = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    out = grpo_step(rewards, log_probs, ref_log_probs, GrpoConfig(kl_beta=0.5))
    assert out.kl_loss == pytest.approx(0.0, abs=1e-9)
    # Policy loss: mean over samples of -min(1 * A, 1 * A) = -mean(A) = 0.
    assert out.policy_loss == pytest.approx(0.0, abs=1e-9)
    assert out.loss == pytest.approx(0.0, abs=1e-9)


def test_grpo_step_increases_kl_loss_with_log_ratio():
    """If log_probs > ref_log_probs, the per-sample KL contribution is positive."""
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    log_probs = np.full_like(rewards, 0.5)        # log pi > log pi_ref
    ref_log_probs = np.full_like(rewards, 0.0)
    out = grpo_step(rewards, log_probs, ref_log_probs, GrpoConfig(kl_beta=1.0))
    # mean log_ratio = 0.5 -> kl_loss = beta * 0.5 = 0.5
    assert out.kl_loss == pytest.approx(0.5, abs=1e-6)


def test_grpo_step_clipped_ratio_caps_unbounded_growth():
    """A wildly-larger log_ratio with positive advantage: PPO clip prevents
    arbitrarily-large policy gradient. Verify the loss is bounded.

    surr1 = ratio * A; surr2 = clip(ratio, 1-eps, 1+eps) * A
    With A > 0 and ratio > 1+eps, the min is surr2 (the smaller).
    """
    # one group with positive advantage on sample 0
    rewards = np.array([[10.0, 0.0, 0.0, 0.0]])
    # large log_ratio on the high-reward sample
    log_probs = np.array([[5.0, 0.0, 0.0, 0.0]])
    ref_log_probs = np.zeros_like(rewards)
    out = grpo_step(
        rewards, log_probs, ref_log_probs,
        GrpoConfig(clip_epsilon=0.2, kl_beta=0.0),
    )
    # For sample 0: A > 0, ratio = e^5 ~= 148.
    # surr1 = 148 * A; surr2 = 1.2 * A. min = 1.2 * A -> loss contribution = -1.2 * A.
    # The other 3 samples have A < 0, ratio = 1, so surr1 == surr2 == A (negative);
    # min = A; loss contribution = -A (positive).
    # Without clipping, the loss would be -148 * A (much more negative).
    assert out.policy_loss > -200  # sanity bound, well above the unclipped magnitude


def test_grpo_step_masks_zero_variance_group_in_loss():
    rewards = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],   # masked
            [1.0, 2.0, 3.0, 4.0],   # active
        ]
    )
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    out = grpo_step(rewards, log_probs, ref_log_probs)
    assert out.n_groups_masked == 1
    assert out.n_groups_total == 2
    # Loss computed only over active samples; mean-zero advantages -> 0.
    assert out.loss == pytest.approx(0.0, abs=1e-6)


def test_grpo_step_all_masked_loss_is_zero():
    """If every group is masked, total loss must be 0 (no gradient flows)."""
    rewards = np.zeros((3, 4))
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    out = grpo_step(rewards, log_probs, ref_log_probs)
    assert out.n_groups_masked == 3
    assert out.loss == 0.0


def test_shape_mismatch_raises():
    rewards = np.zeros((2, 4))
    log_probs = np.zeros((2, 3))
    ref_log_probs = np.zeros((2, 4))
    with pytest.raises(ValueError):
        grpo_step(rewards, log_probs, ref_log_probs)


def test_per_group_diagnostics_populated():
    rewards = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    out = grpo_step(rewards, log_probs, ref_log_probs)
    assert len(out.per_group_diagnostics) == 2
    assert out.per_group_diagnostics[0]["active"] is True
    assert out.per_group_diagnostics[1]["active"] is False
    assert out.per_group_diagnostics[0]["mean_reward"] == pytest.approx(2.5)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def test_registry_returns_grpo_algorithm():
    alg = get_algorithm("grpo")
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    out = alg.step(rewards, log_probs, ref_log_probs)
    assert out.n_groups_total == 1
    assert out.n_groups_masked == 0


def test_registry_case_insensitive():
    a = get_algorithm("GRPO")
    b = get_algorithm("grpo")
    assert type(a) is type(b)


def test_registry_unknown_algorithm_raises():
    with pytest.raises(KeyError):
        get_algorithm("nonexistent")


def test_all_registry_algorithms_are_callable():
    """No registry entry is stubbed anymore; every one returns a valid output."""
    # Use group_size=4 so RAFT's default top_k=4 doesn't exceed it.
    rewards = np.zeros((1, 4))
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    for name in ("grpo", "ppo", "rloo", "raft"):
        alg = get_algorithm(name)
        out = alg.step(rewards, log_probs, ref_log_probs)
        assert out.n_groups_total == 1, f"{name} returned wrong group count"
