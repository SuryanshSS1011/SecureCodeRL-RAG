"""Tests for the RLOO (Leave-One-Out) algorithm step.

RLOO uses an unbiased leave-one-out baseline:

    A_i = R_i - mean(R_{-i})

where `mean(R_{-i})` is the mean over the i-th sample's group excluding
sample i. With group size G, this is equivalent to

    A_i = (G * R_i - sum(R_group)) / (G - 1)

Tests verify:
  - LOO formula across hand-computed groups.
  - For G=1 the formula degenerates (no leave-one-out possible); we
    fall back to A_i = R_i.
  - LOO advantages have mean 0 across the group (no bias term).
"""

from __future__ import annotations

import numpy as np
import pytest

from secure_code_rl_ictai.rl import GrpoConfig, get_algorithm
from secure_code_rl_ictai.rl.rloo import compute_loo_advantages, rloo_step


# ----------------------------------------------------------------------
# Advantage math
# ----------------------------------------------------------------------


def test_loo_advantage_matches_hand_computation():
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    # For sample 0: R_0 - mean(R_1,R_2,R_3) = 1 - 3 = -2
    # For sample 1: R_1 - mean(R_0,R_2,R_3) = 2 - 2.67 = -0.67
    # For sample 2: R_2 - mean(R_0,R_1,R_3) = 3 - 2.33 = 0.67
    # For sample 3: R_3 - mean(R_0,R_1,R_2) = 4 - 2 = 2
    adv = compute_loo_advantages(rewards)
    assert adv[0, 0] == pytest.approx(-2.0)
    assert adv[0, 1] == pytest.approx(-2/3, abs=1e-6)
    assert adv[0, 2] == pytest.approx(2/3, abs=1e-6)
    assert adv[0, 3] == pytest.approx(2.0)


def test_loo_advantages_sum_to_zero_per_group():
    """LOO advantages have mean 0 within each group (unbiased baseline)."""
    rewards = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 0.0, 5.0, 5.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )
    adv = compute_loo_advantages(rewards)
    assert np.allclose(adv.sum(axis=1), 0.0, atol=1e-9)


def test_loo_degenerate_group_size_one():
    """G=1: no other samples to leave out; fall back to A_i = R_i (no baseline)."""
    rewards = np.array([[2.5]])
    adv = compute_loo_advantages(rewards)
    assert adv[0, 0] == pytest.approx(2.5)


def test_loo_constant_rewards_yield_zero_advantage():
    """All samples equal -> mean(others) == R_i -> advantage = 0."""
    rewards = np.array([[3.0, 3.0, 3.0, 3.0]])
    adv = compute_loo_advantages(rewards)
    assert np.allclose(adv, 0.0)


# ----------------------------------------------------------------------
# Step composition
# ----------------------------------------------------------------------


def test_rloo_step_at_reference_yields_zero_loss():
    """When log_probs == ref_log_probs, ratio = 1 and the surrogate reduces
    to -A. LOO advantages have mean 0 -> loss = 0."""
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    out = rloo_step(
        rewards, log_probs, ref_log_probs,
        config=GrpoConfig(kl_beta=0.0),
    )
    assert out.policy_loss == pytest.approx(0.0, abs=1e-9)
    assert out.kl_loss == pytest.approx(0.0)
    assert out.loss == pytest.approx(0.0, abs=1e-9)


def test_rloo_step_responds_to_kl():
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    log_probs = np.full_like(rewards, 0.5)
    ref_log_probs = np.zeros_like(rewards)
    out = rloo_step(
        rewards, log_probs, ref_log_probs,
        config=GrpoConfig(kl_beta=1.0),
    )
    assert out.kl_loss == pytest.approx(0.5)


def test_rloo_step_clips_extreme_ratio():
    rewards = np.array([[10.0, 0.0, 0.0, 0.0]])
    log_probs = np.array([[5.0, 0.0, 0.0, 0.0]])
    ref_log_probs = np.zeros_like(rewards)
    out = rloo_step(
        rewards, log_probs, ref_log_probs,
        config=GrpoConfig(clip_epsilon=0.2, kl_beta=0.0),
    )
    assert out.policy_loss > -200  # well above unclipped


# ----------------------------------------------------------------------
# Registry wiring
# ----------------------------------------------------------------------


def test_registry_rloo_now_callable():
    alg = get_algorithm("rloo")
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    out = alg.step(rewards, log_probs, ref_log_probs)
    assert out.n_groups_total == 1
    # LOO advantages sum to 0 -> mean 0.
    assert out.mean_advantage == pytest.approx(0.0, abs=1e-9)
