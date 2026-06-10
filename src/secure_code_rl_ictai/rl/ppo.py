"""PPO step (numpy reference) with value-function baseline.

Per docs/training_spec.md §3, PPO is run as the LCTES-continuity baseline.
The only structural difference from GRPO is the advantage:

    A_i = R_i - V(x_i)

When `values` is None we degenerate to A_i = R_i (REINFORCE). This is
documented but not the recommended path; the trainer should always
supply a value model output.

The clip + KL composition mirrors `grpo_step` so the two algorithms are
directly comparable in the ablation table.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .grpo import GrpoConfig, GrpoStepOutput


def ppo_step(
    rewards: np.ndarray,            # (n_groups, group_size)
    log_probs: np.ndarray,
    ref_log_probs: np.ndarray,
    *,
    values: Optional[np.ndarray] = None,    # (n_groups, group_size) or None
    config: Optional[GrpoConfig] = None,
) -> GrpoStepOutput:
    """PPO step. Same return type as GRPO so the registry stays uniform."""
    cfg = config or GrpoConfig()

    if not (rewards.shape == log_probs.shape == ref_log_probs.shape):
        raise ValueError(
            f"shape mismatch: rewards {rewards.shape}, "
            f"log_probs {log_probs.shape}, ref_log_probs {ref_log_probs.shape}"
        )
    if values is not None and values.shape != rewards.shape:
        raise ValueError(
            f"values shape {values.shape} must match rewards shape {rewards.shape}"
        )

    if values is None:
        advantages = rewards.copy()
    else:
        advantages = rewards - values

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
    per_group: list[dict] = []
    for g in range(n_groups_total):
        per_group.append(
            {
                "group_id": g,
                "active": True,  # PPO has no group masking
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
