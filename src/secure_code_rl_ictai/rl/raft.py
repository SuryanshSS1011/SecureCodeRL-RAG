"""RAFT step (numpy reference) — reward-ranked fine-tuning.

Per docs/training_spec.md §3:
    - Select top-k completions per group by reward.
    - Loss = -mean(log_probs over selected) + kl_beta * mean(log_pi - log_pi_ref).

For algorithm-ablation parity (FINDINGS_LOG 2026-06-14): GRPO, PPO, and
RLOO all include a β·KL(π‖π_ref) term against the reference policy. RAFT
originally omitted this, which confounded the algorithm choice with KL
strength. We now apply the same β so the four algorithms differ ONLY in
their advantage estimator (top-k selection mask vs group-relative vs LOO
vs value-baseline).

RAFT remains the "why not just filter and SFT?" ablation arm. It binarizes
a graded reward — selected vs not-selected — and is expected to
underperform GRPO/PPO/RLOO on dense reward landscapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grpo import GrpoStepOutput


@dataclass
class RaftConfig:
    """Tunables for the RAFT step.

    top_k: number of completions per group kept for the SFT loss.
    kl_beta: KL penalty weight against the reference policy. Defaults to
        the GRPO default (0.05) for algorithm-ablation parity. Set to 0.0
        to recover the original "pure RAFT" behavior.
    """

    top_k: int = 4
    kl_beta: float = 0.05


def select_top_k_per_group(rewards: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask selecting the top-k rewards per group.

    For ties, `argpartition` semantics decide; this is deterministic for
    any given input but not guaranteed to match a specific tie-breaking
    rule across numpy versions. Tests assert on cardinality and on the
    presence of the unambiguous maxima, not on the specific tied indices.
    """
    if rewards.ndim != 2:
        raise ValueError(
            f"rewards must be 2-D (n_groups, group_size), got {rewards.shape}"
        )
    G = rewards.shape[1]
    if k > G:
        raise ValueError(f"top_k={k} exceeds group_size={G}")
    if k == G:
        return np.ones_like(rewards, dtype=bool)

    # argpartition: indices of the k highest along axis 1.
    # We want the highest, so negate.
    top_idx = np.argpartition(-rewards, k - 1, axis=1)[:, :k]  # (n_groups, k)
    mask = np.zeros_like(rewards, dtype=bool)
    np.put_along_axis(mask, top_idx, True, axis=1)
    return mask


def raft_step(
    rewards: np.ndarray,
    log_probs: np.ndarray,
    ref_log_probs: np.ndarray,
    config: Optional[RaftConfig] = None,
) -> GrpoStepOutput:
    cfg = config or RaftConfig()

    if not (rewards.shape == log_probs.shape == ref_log_probs.shape):
        raise ValueError(
            f"shape mismatch: rewards {rewards.shape}, "
            f"log_probs {log_probs.shape}, ref_log_probs {ref_log_probs.shape}"
        )

    selected = select_top_k_per_group(rewards, cfg.top_k)
    n_selected = int(selected.sum())

    if n_selected == 0:
        policy_loss = 0.0
    else:
        policy_loss = float(-log_probs[selected].sum() / n_selected)

    # KL penalty for algorithm-ablation parity (FINDINGS_LOG 2026-06-14).
    # Mirrors GRPO/PPO/RLOO's kl_beta * mean(log_ratio). Computed over ALL
    # samples (not just selected) so the reference-anchoring effect is the
    # same magnitude as the other algorithms.
    n_samples = int(rewards.size)
    if n_samples == 0 or cfg.kl_beta == 0.0:
        kl_loss = 0.0
    else:
        log_ratio = log_probs - ref_log_probs
        kl_loss = float(cfg.kl_beta * log_ratio.sum() / n_samples)

    total_loss = policy_loss + kl_loss

    n_groups_total = rewards.shape[0]
    per_group = []
    for g in range(n_groups_total):
        per_group.append(
            {
                "group_id": g,
                "active": True,
                "mean_reward": float(rewards[g].mean()),
                "std_reward": float(rewards[g].std()),
                "mean_advantage": 0.0,  # RAFT has no advantage
                "mean_ratio": 1.0,      # no ratio computed
                "n_selected": int(selected[g].sum()),
            }
        )

    # RAFT carries the boolean selection mask as a *float* advantage proxy
    # (1 for selected samples, 0 for rest). The trainer's policy.step()
    # treats this as advantages; for RAFT the reconstructed surrogate
    # reduces to `-mean(selected log_probs)` + the KL term.
    raft_advantage_proxy = selected.astype(float)
    return GrpoStepOutput(
        loss=total_loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        mean_advantage=0.0,
        mean_ratio=1.0,
        n_groups_masked=0,
        n_groups_total=n_groups_total,
        per_group_diagnostics=per_group,
        advantages=raft_advantage_proxy,
        clip_epsilon=0.0,
        kl_beta=cfg.kl_beta,  # surfaced so policy.step() applies it
    )
