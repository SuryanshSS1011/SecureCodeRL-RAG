"""Tests for the PPO algorithm step (numpy reference implementation).

PPO differs from GRPO by using a value-function baseline rather than the
group mean. Per `docs/training_spec.md` §3, our PPO baseline runs at the
same reward and KL structure as GRPO; the difference is the advantage:

    A_i = R_i - V(x_i)   (PPO with a value model)
    A_i = (R_i - mean(R_group)) / (std(R_group) + eps)   (GRPO)

The trainer supplies `values` alongside `rewards`. When `values` is None
PPO reduces to plain REINFORCE (advantage = reward); that's documented
but not the path we take in practice.
"""

from __future__ import annotations

import numpy as np
import pytest

from secure_code_rl_ictai.rl import (
    GrpoConfig,
    get_algorithm,
)
from secure_code_rl_ictai.rl.ppo import ppo_step


# ----------------------------------------------------------------------
# Advantage math
# ----------------------------------------------------------------------


def test_ppo_advantage_is_reward_minus_value():
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    values = np.array([[2.0, 2.0, 2.0, 2.0]])
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    out = ppo_step(
        rewards, log_probs, ref_log_probs, values=values,
        config=GrpoConfig(clip_epsilon=0.2, kl_beta=0.0),
    )
    # advantages: -1, 0, 1, 2 -> mean 0.5
    assert out.mean_advantage == pytest.approx(0.5)


def test_ppo_no_value_model_reduces_to_reinforce():
    """With values=None, A_i = R_i (no baseline). Verifies the fallback."""
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    out = ppo_step(
        rewards, log_probs, ref_log_probs, values=None,
        config=GrpoConfig(clip_epsilon=0.2, kl_beta=0.0),
    )
    # advantages == rewards -> mean 2.5
    assert out.mean_advantage == pytest.approx(2.5)


# ----------------------------------------------------------------------
# Clip + KL composition (mirrors GRPO tests)
# ----------------------------------------------------------------------


def test_ppo_clip_caps_unbounded_growth():
    rewards = np.array([[10.0, 0.0, 0.0, 0.0]])
    values = np.array([[5.0, 5.0, 5.0, 5.0]])  # so A = [5, -5, -5, -5]
    log_probs = np.array([[5.0, 0.0, 0.0, 0.0]])
    ref_log_probs = np.zeros_like(rewards)
    out = ppo_step(
        rewards, log_probs, ref_log_probs, values=values,
        config=GrpoConfig(clip_epsilon=0.2, kl_beta=0.0),
    )
    assert out.policy_loss > -200  # well above the unclipped magnitude


def test_ppo_kl_loss_responds_to_log_ratio():
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    values = np.zeros_like(rewards)
    log_probs = np.full_like(rewards, 0.5)
    ref_log_probs = np.zeros_like(rewards)
    out = ppo_step(
        rewards, log_probs, ref_log_probs, values=values,
        config=GrpoConfig(kl_beta=1.0),
    )
    # mean log_ratio = 0.5 -> kl_loss = 1.0 * 0.5 = 0.5
    assert out.kl_loss == pytest.approx(0.5)


def test_ppo_shape_mismatch_raises():
    rewards = np.zeros((2, 4))
    log_probs = np.zeros((2, 3))
    ref_log_probs = np.zeros((2, 4))
    with pytest.raises(ValueError):
        ppo_step(rewards, log_probs, ref_log_probs)


def test_ppo_value_shape_must_match_rewards():
    rewards = np.zeros((2, 4))
    log_probs = np.zeros((2, 4))
    ref_log_probs = np.zeros((2, 4))
    values = np.zeros((2, 3))  # wrong shape
    with pytest.raises(ValueError):
        ppo_step(rewards, log_probs, ref_log_probs, values=values)


# ----------------------------------------------------------------------
# Registry wiring
# ----------------------------------------------------------------------


def test_registry_ppo_now_callable():
    """After implementing ppo_step, PpoAlgorithm.step should work."""
    alg = get_algorithm("ppo")
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    log_probs = np.zeros_like(rewards)
    ref_log_probs = np.zeros_like(rewards)
    out = alg.step(rewards, log_probs, ref_log_probs)
    assert out.n_groups_total == 1
    # advantages == rewards in the no-value-model fallback
    assert out.mean_advantage == pytest.approx(2.5)
