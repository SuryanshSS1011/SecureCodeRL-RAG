"""RLOO step (numpy reference) — leave-one-out baseline.

Per docs/training_spec.md §3, RLOO is an unbiased estimator alternative
to GRPO's group-mean baseline. The advantage is

    A_i = R_i - mean(R_{-i})

with `mean(R_{-i})` = the mean over the i-th sample's group excluding
sample i. Equivalent to `(G * R_i - sum(R_group)) / (G - 1)` for G > 1;
degenerates to A_i = R_i for G == 1.

LOO advantages have mean 0 per group (unbiased baseline). No zero-
variance group masking is needed: when all rewards are equal, the LOO
advantages are zero and the surrogate is zero, so the gradient is zero
without any explicit mask.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .grpo import GrpoConfig, GrpoStepOutput


def compute_loo_advantages(rewards: np.ndarray) -> np.ndarray:
    """A_i = R_i - mean(R over the group, excluding sample i).

    For group size G == 1, degenerates to A_i = R_i (no other samples).
    """
    if rewards.ndim != 2:
        raise ValueError(
            f"rewards must be 2-D (n_groups, group_size), got {rewards.shape}"
        )
    G = rewards.shape[1]
    if G == 1:
        return rewards.copy()
    total = rewards.sum(axis=1, keepdims=True)  # (n_groups, 1)
    # mean over the others = (total - R_i) / (G - 1)
    leave_one_out_mean = (total - rewards) / (G - 1)
    return rewards - leave_one_out_mean


def rloo_step(
    rewards: np.ndarray,
    log_probs: np.ndarray,
    ref_log_probs: np.ndarray,
    config: Optional[GrpoConfig] = None,
) -> GrpoStepOutput:
    cfg = config or GrpoConfig()

    if not (rewards.shape == log_probs.shape == ref_log_probs.shape):
        raise ValueError(
            f"shape mismatch: rewards {rewards.shape}, "
            f"log_probs {log_probs.shape}, ref_log_probs {ref_log_probs.shape}"
        )

    advantages = compute_loo_advantages(rewards)

    log_ratio = log_probs - ref_log_probs
    log_ratio_clipped = np.clip(log_ratio, -20.0, 20.0)
    ratio = np.exp(log_ratio_clipped)

    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages
    policy_loss_per_sample = -np.minimum(surr1, surr2)
    kl_loss_per_sample = cfg.kl_beta * log_ratio

    n_samples = int(rewards.size)
    if n_samples == 0:
        policy_loss = 0.0
        kl_loss = 0.0
    else:
        policy_loss = float(policy_loss_per_sample.sum() / n_samples)
        kl_loss = float(kl_loss_per_sample.sum() / n_samples)

    loss = policy_loss + kl_loss

    n_groups_total = rewards.shape[0]
    per_group = []
    for g in range(n_groups_total):
        per_group.append(
            {
                "group_id": g,
                "active": True,  # RLOO has no group masking
                "mean_reward": float(rewards[g].mean()),
                "std_reward": float(rewards[g].std()),
                "mean_advantage": float(advantages[g].mean()),
                "mean_ratio": float(ratio[g].mean()),
            }
        )

    return GrpoStepOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        mean_advantage=float(advantages.mean()) if advantages.size else 0.0,
        mean_ratio=float(ratio.mean()) if ratio.size else 1.0,
        n_groups_masked=0,
        n_groups_total=n_groups_total,
        per_group_diagnostics=per_group,
        advantages=advantages,
        clip_epsilon=cfg.clip_epsilon,
        kl_beta=cfg.kl_beta,
    )
