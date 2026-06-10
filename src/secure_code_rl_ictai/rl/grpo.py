"""GRPO (Group Relative Policy Optimization) step — algorithm-only.

This module is **algorithm math only**. It does not own the model, the
tokenizer, or the optimizer. The trainer hands in pre-computed rewards
and log-probs; this module returns the loss and per-step metrics.

The reference implementation uses numpy so it can be unit-tested without
torch. The trainer's torch path will mirror these formulas exactly; the
unit tests pin the algorithm shape (zero-variance masking, ratio
clipping, KL composition), and the torch wrapper is type-checked against
this reference.

See docs/training_spec.md §2 for the full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GrpoConfig:
    """GRPO hyperparameters (see docs/training_spec.md §2.5)."""

    clip_epsilon: float = 0.2
    kl_beta: float = 0.05
    advantage_epsilon: float = 1e-8  # numerical-stability term in std division
    zero_variance_threshold: float = 1e-6  # |std| below this triggers group mask
    # sigma_min of paper Section IV-C: the std divisor becomes
    # max(std, sigma_floor). 0.0 disables the floor.
    sigma_floor: float = 0.05


@dataclass
class GrpoStepOutput:
    """Per-step output from a GRPO step.

    `advantages` is the per-sample advantage array, same shape as
    `rewards`. The trainer carries this to `policy.step()` so the policy
    can rebuild the surrogate with autograd-enabled tensors (the
    algorithm step itself is numpy-only).
    """

    loss: float
    policy_loss: float
    kl_loss: float
    mean_advantage: float
    mean_ratio: float
    n_groups_masked: int = 0
    n_groups_total: int = 0
    per_group_diagnostics: list[dict] = field(default_factory=list)
    advantages: "np.ndarray | None" = None
    clip_epsilon: float = 0.2
    kl_beta: float = 0.0


def compute_group_advantages(
    rewards: np.ndarray,  # shape (n_groups, group_size)
    *,
    epsilon: float = 1e-8,
    zero_variance_threshold: float = 1e-6,
    sigma_floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GRPO advantages with zero-variance group masking.

    For each group of `G` rollouts on the same prompt:
        A_i = (R_i - mean(R_group)) / max(std(R_group), sigma_floor) + epsilon

    Setting `sigma_floor > 0` enables the Dr.GRPO floor from
    training_spec.md §7.6: a 1-of-G success in a high-stakes group
    (typically design-pair early-RL) gives tiny std → enormous normalized
    advantage → outsized high-variance update on the most fragile CWE.
    The floor bounds the divisor without altering the zero-variance
    masking (groups with std below `zero_variance_threshold` are still
    masked entirely, regardless of `sigma_floor`).

    Returns:
        advantages: same shape as rewards, with masked groups set to 0.
        group_active_mask: shape (n_groups,), True where the group's
            gradient should flow. Groups with std below threshold AND
            all-zero rewards are masked out.

    See docs/training_spec.md §2.1 for the zero-variance rationale and
    §7.6 for the sigma_floor mechanism.
    """
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be 2-D (n_groups, group_size), got {rewards.shape}")
    if sigma_floor < 0.0:
        raise ValueError(f"sigma_floor must be >= 0, got {sigma_floor}")

    means = rewards.mean(axis=1, keepdims=True)
    stds = rewards.std(axis=1, keepdims=True)

    # Apply the Dr.GRPO floor (§7.6) to the divisor. When sigma_floor == 0,
    # this is identical to the pre-§7.6 behavior. When sigma_floor > 0,
    # we take max(std, sigma_floor) before adding the numerical-stability
    # epsilon, bounding the 1-of-G advantage blowup.
    divisor = np.maximum(stds, sigma_floor) + epsilon
    advantages = (rewards - means) / divisor

    # Mask groups where std is effectively zero. The "all-zero rewards"
    # condition is the load-bearing case (early training, no checker fires),
    # but we also mask all-equal-nonzero groups defensively.
    zero_variance = stds.squeeze(axis=1) < zero_variance_threshold
    group_active_mask = ~zero_variance

    # Zero out masked groups' advantages so they contribute nothing
    # downstream even if the loss code forgets to apply the mask.
    advantages = np.where(group_active_mask[:, None], advantages, 0.0)

    return advantages, group_active_mask


def grpo_step(
    rewards: np.ndarray,           # (n_groups, G)
    log_probs: np.ndarray,         # (n_groups, G) — sum over tokens
    ref_log_probs: np.ndarray,     # (n_groups, G)
    config: Optional[GrpoConfig] = None,
    *,
    per_group_weight: Optional[np.ndarray] = None,  # (n_groups,), default 1.0
) -> GrpoStepOutput:
    """One GRPO step's loss + diagnostics.

    Inputs are sequence-level log-probs (sum over completion tokens).
    The torch implementation will mirror this exactly with `.sum(dim=-1)`
    over the token axis upstream.

    Loss form (with per-group reweighting per training_spec.md §7.2):
        ratio_i      = exp(log_probs_i - ref_log_probs_i)
        L_pol_i      = -w_g · min(ratio_i * A_i, clip(ratio_i, 1±eps) * A_i)
        L_kl_i       =  w_g · beta * (log_probs_i - ref_log_probs_i)
                       (i.e. unbiased KL estimator at sample i, weighted by w_g)
        L_i          = L_pol_i + L_kl_i
        L            = mean over active samples

    where `w_g` is the per-group weight from `per_group_weight[g]`, or 1.0
    when `per_group_weight is None`. Weights are applied per sample (group
    membership broadcast across the G axis); this is equivalent to
    weighting whole groups' losses.

    Setting `per_group_weight = None` preserves pre-§7.2 behavior. When
    used, weights should come from the trainer-side per-CWE
    gradient-mass-share reweighter (`reweight.py`), clipped to [0.25, 4.0]
    per §7.2's derived stability bounds.

    Note on KL estimator: we use the simple unbiased single-sample
    estimator `log(pi/pi_ref) = log_p - log_p_ref` averaged over samples,
    rather than the closed-form full-distribution KL. This matches GRPO's
    standard form and is what the LCTES / causality-aware lineage used.

    Masked groups contribute 0 to the loss; the mean is computed over
    active samples only. Reweighting does not interact with masking —
    a masked group is zeroed regardless of its weight.
    """
    cfg = config or GrpoConfig()

    if not (rewards.shape == log_probs.shape == ref_log_probs.shape):
        raise ValueError(
            f"shape mismatch: rewards {rewards.shape}, "
            f"log_probs {log_probs.shape}, ref_log_probs {ref_log_probs.shape}"
        )

    advantages, group_active = compute_group_advantages(
        rewards,
        epsilon=cfg.advantage_epsilon,
        zero_variance_threshold=cfg.zero_variance_threshold,
        sigma_floor=cfg.sigma_floor,
    )

    log_ratio = log_probs - ref_log_probs
    # Clip log_ratio to prevent exp overflow. Symmetric large bound;
    # torch's implementation will use the same trick.
    log_ratio_clipped = np.clip(log_ratio, -20.0, 20.0)
    ratio = np.exp(log_ratio_clipped)

    # Per-sample PPO surrogate.
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages
    policy_loss_per_sample = -np.minimum(surr1, surr2)

    # KL term: unbiased estimator at sample i.
    kl_loss_per_sample = cfg.kl_beta * log_ratio  # not clipped — this is the actual KL contribution

    # Per-group reweighting (training_spec.md §7.2). per_group_weight has
    # shape (n_groups,); broadcast across the G axis so each sample in a
    # group carries that group's weight. Default 1.0 preserves pre-§7.2
    # behavior and is uniform-weighting (the §7.5 ablation arm).
    if per_group_weight is not None:
        if per_group_weight.shape != (rewards.shape[0],):
            raise ValueError(
                f"per_group_weight must have shape (n_groups={rewards.shape[0]},), "
                f"got {per_group_weight.shape}"
            )
        weight_2d = per_group_weight[:, None]  # (n_groups, 1) → broadcasts to (n_groups, G)
        policy_loss_per_sample = policy_loss_per_sample * weight_2d
        kl_loss_per_sample = kl_loss_per_sample * weight_2d

    # Per-group masking: zero out the masked groups entirely.
    # Masking happens AFTER reweighting so a masked group is zeroed
    # regardless of its weight (a non-zero weight on a zero loss is
    # still zero, but explicit is better).
    group_mask_2d = group_active[:, None]  # (n_groups, 1)
    policy_loss_per_sample = np.where(group_mask_2d, policy_loss_per_sample, 0.0)
    kl_loss_per_sample = np.where(group_mask_2d, kl_loss_per_sample, 0.0)

    n_active_samples = int(group_active.sum() * rewards.shape[1])
    if n_active_samples == 0:
        # All groups masked. Loss is 0; gradient won't flow.
        policy_loss = 0.0
        kl_loss = 0.0
    else:
        policy_loss = float(policy_loss_per_sample.sum() / n_active_samples)
        kl_loss = float(kl_loss_per_sample.sum() / n_active_samples)

    loss = policy_loss + kl_loss

    # Diagnostics.
    n_groups_total = rewards.shape[0]
    n_groups_masked = int((~group_active).sum())

    per_group: list[dict] = []
    for g in range(n_groups_total):
        per_group.append(
            {
                "group_id": g,
                "active": bool(group_active[g]),
                "mean_reward": float(rewards[g].mean()),
                "std_reward": float(rewards[g].std()),
                "mean_advantage": (
                    float(advantages[g].mean()) if group_active[g] else 0.0
                ),
                "mean_ratio": float(ratio[g].mean()),
            }
        )

    return GrpoStepOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        mean_advantage=float(advantages[group_active].mean()) if group_active.any() else 0.0,
        mean_ratio=float(ratio[group_active[:, None].repeat(rewards.shape[1], axis=1)].mean())
        if group_active.any()
        else 1.0,
        n_groups_masked=n_groups_masked,
        n_groups_total=n_groups_total,
        per_group_diagnostics=per_group,
        advantages=advantages,
        clip_epsilon=cfg.clip_epsilon,
        kl_beta=cfg.kl_beta,
    )
