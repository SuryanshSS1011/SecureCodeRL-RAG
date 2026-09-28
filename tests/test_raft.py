"""Tests for the RAFT (Reward-Ranked Fine-Tuning) algorithm step.

RAFT is structurally different from PPO/GRPO/RLOO:
  - Top-k selection by reward within each group.
  - Loss = -mean(log_probs over selected samples).
  - No ratio, no clip, no KL term.

RAFT in the ablation table answers "why not just filter and SFT?" It's
expected to underperform because it binarizes a graded reward, throwing
away dense signal from the moderate-reward samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from cargo.rl import get_algorithm
from cargo.rl.raft import (
    RaftConfig,
    raft_step,
    select_top_k_per_group,
)


# ----------------------------------------------------------------------
# Top-k selection
# ----------------------------------------------------------------------


def test_top_k_selects_highest_reward_per_group():
    rewards = np.array(
        [
            [1.0, 4.0, 2.0, 3.0],
            [10.0, 0.0, 5.0, 5.0],
        ]
    )
    selected = select_top_k_per_group(rewards, k=2)
    # Group 0: top 2 are indices 1 (R=4) and 3 (R=3) -> selected mask True there.
    assert selected[0].tolist() == [False, True, False, True]
    # Group 1: top 2 are indices 0 (R=10) and either 2 or 3 (both R=5).
    # We require deterministic tie-breaking: argsort returns earliest first
    # on ties; "top-2" of [10, 0, 5, 5] is indices [0, 2] (after np.argsort
    # is stable / np.argpartition order).
    assert selected[1, 0] is np.True_ or selected[1, 0]  # the 10 is selected
    assert selected[1].sum() == 2


def test_top_k_equals_group_size_selects_all():
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    selected = select_top_k_per_group(rewards, k=4)
    assert selected.sum() == 4


def test_top_k_larger_than_group_raises():
    rewards = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        select_top_k_per_group(rewards, k=4)


# ----------------------------------------------------------------------
# Loss math
# ----------------------------------------------------------------------


def test_raft_loss_is_neg_mean_log_prob_over_selected():
    """Loss = -mean(log_probs[selected]) when kl_beta=0 (pure-RAFT mode)."""
    rewards = np.array([[1.0, 4.0, 2.0, 3.0]])
    # selected = indices 1 (R=4) and 3 (R=3). log_probs[selected] = [-0.5, -1.0]
    log_probs = np.array([[-2.0, -0.5, -2.0, -1.0]])
    ref_log_probs = np.zeros_like(rewards)
    # kl_beta=0.0 forces pure-RAFT (no KL term); the default kl_beta=0.01
    # adds a small KL contribution for algorithm-ablation parity with
    # GRPO/PPO/RLOO (FINDINGS_LOG 2026-06-14).
    out = raft_step(
        rewards, log_probs, ref_log_probs,
        config=RaftConfig(top_k=2, kl_beta=0.0),
    )
    # -mean([-0.5, -1.0]) = -(-0.75) = 0.75
    assert out.policy_loss == pytest.approx(0.75)
    assert out.kl_loss == 0.0  # pure-RAFT has no KL term
    assert out.loss == pytest.approx(0.75)


def test_raft_kl_term_default_matches_grpo_for_algorithm_parity():
    """With the new default kl_beta=0.01, RAFT applies β·mean(log_ratio).

    Added 2026-06-14: aligns RAFT's KL strength with GRPO/PPO/RLOO so the
    algorithm-ablation comparison isn't confounded by KL-strength differences.
    """
    rewards = np.array([[1.0, 4.0, 2.0, 3.0]])
    log_probs = np.array([[-2.0, -0.5, -2.0, -1.0]])
    ref_log_probs = np.array([[-1.0, -0.0, -1.0, -0.5]])
    # log_ratio = [-1, -0.5, -1, -0.5]; mean = -0.75; kl_loss = 0.01 * -0.75
    out = raft_step(
        rewards, log_probs, ref_log_probs,
        config=RaftConfig(top_k=2, kl_beta=0.01),
    )
    assert out.kl_loss == pytest.approx(0.01 * -0.75)
    assert out.kl_beta == 0.01
    # Total loss includes both terms.
    assert out.loss == pytest.approx(out.policy_loss + out.kl_loss)


def test_raft_step_selects_all_when_top_k_equals_group():
    rewards = np.array([[1.0, 2.0]])
    log_probs = np.array([[-1.0, -2.0]])
    ref_log_probs = np.zeros_like(rewards)
    out = raft_step(
        rewards, log_probs, ref_log_probs,
        config=RaftConfig(top_k=2),
    )
    # loss = -mean([-1, -2]) = 1.5
    assert out.policy_loss == pytest.approx(1.5)


# ----------------------------------------------------------------------
# Diagnostic fields
# ----------------------------------------------------------------------


def test_raft_step_no_clipping_no_masking():
    rewards = np.array([[1.0, 2.0, 3.0, 4.0]])
    log_probs = np.array([[-1.0, -1.0, -1.0, -1.0]])
    ref_log_probs = np.zeros_like(rewards)
    out = raft_step(rewards, log_probs, ref_log_probs, config=RaftConfig(top_k=2))
    # RAFT doesn't use group masking
    assert out.n_groups_masked == 0
    # mean_ratio is reported as 1 (no ratio computed; default)
    assert out.mean_ratio == 1.0


# ----------------------------------------------------------------------
# Registry wiring
# ----------------------------------------------------------------------


def test_registry_raft_now_callable():
    """Default top_k from registry's RaftAlgorithm constructor."""
    alg = get_algorithm("raft")
    rewards = np.array([[1.0, 4.0, 2.0, 3.0]])
    log_probs = np.array([[-2.0, -0.5, -2.0, -1.0]])
    ref_log_probs = np.zeros_like(rewards)
    out = alg.step(rewards, log_probs, ref_log_probs)
    assert out.policy_loss > 0  # at least the math ran without raising
