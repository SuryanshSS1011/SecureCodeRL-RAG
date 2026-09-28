"""RL algorithm registry.

Algorithms implement a common `AlgorithmStep` protocol. The trainer picks
by config string and calls `.step(...)` with pre-computed rewards and
log-probs.

GRPO, PPO (value-baselined), RLOO, and RAFT are the four algorithms of
the paper's 4x2 factorial.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .grpo import GrpoConfig, GrpoStepOutput, grpo_step
from .ppo import ppo_step
from .raft import RaftConfig, raft_step
from .rloo import rloo_step


class AlgorithmStep(Protocol):
    """Common interface for one RL algorithm step.

    Inputs are sequence-level (sum over completion tokens). The trainer
    is responsible for summation upstream.

    Returns a `GrpoStepOutput`-like dataclass; we use GRPO's output as the
    common shape since RLOO / PPO produce nearly identical fields.

    `per_group_weight` is the CWE-aware per-prompt weight w(x) (Eq. 3).
    The trainer applies it to the torch surrogate for every algorithm;
    GRPO also folds it into its reported numpy loss, the others ignore it
    here.
    """

    def step(
        self,
        rewards: np.ndarray,
        log_probs: np.ndarray,
        ref_log_probs: np.ndarray,
        *,
        per_group_weight: np.ndarray | None = None,
    ) -> GrpoStepOutput: ...


class GrpoAlgorithm:
    def __init__(self, config: GrpoConfig | None = None) -> None:
        self.config = config or GrpoConfig()

    def step(
        self,
        rewards: np.ndarray,
        log_probs: np.ndarray,
        ref_log_probs: np.ndarray,
        *,
        per_group_weight: np.ndarray | None = None,
    ) -> GrpoStepOutput:
        return grpo_step(
            rewards, log_probs, ref_log_probs, self.config,
            per_group_weight=per_group_weight,
        )


class PpoAlgorithm:
    """PPO with a value-function baseline.

    The trainer computes values via TorchPolicy.values() (which requires
    enable_value_head() to have been called) and passes them in via
    `step(..., values=...)`. The advantage is then A = R - V.

    For backward compatibility with tests that don't wire a value head,
    `values=None` falls back to REINFORCE (advantages = rewards). The
    fallback is logged once per run (see `needs_value_head` below) so the
    trainer can hard-error when the user requested PPO but no head is
    enabled.

    `per_group_weight` is ignored here; the trainer applies w(x) to the
    per-prompt clipped surrogate.
    """

    needs_value_head: bool = True

    def __init__(self, config: GrpoConfig | None = None) -> None:
        self.config = config or GrpoConfig()

    def step(
        self,
        rewards: np.ndarray,
        log_probs: np.ndarray,
        ref_log_probs: np.ndarray,
        *,
        values: np.ndarray | None = None,
        per_group_weight: np.ndarray | None = None,
    ) -> GrpoStepOutput:
        del per_group_weight
        return ppo_step(
            rewards, log_probs, ref_log_probs,
            values=values, config=self.config,
        )


class RlooAlgorithm:
    """Leave-one-out baseline: A_i = R_i - mean(R_{-i})."""

    def __init__(self, config: GrpoConfig | None = None) -> None:
        self.config = config or GrpoConfig()

    def step(
        self,
        rewards: np.ndarray,
        log_probs: np.ndarray,
        ref_log_probs: np.ndarray,
        *,
        per_group_weight: np.ndarray | None = None,
    ) -> GrpoStepOutput:
        del per_group_weight
        return rloo_step(rewards, log_probs, ref_log_probs, self.config)


class RaftAlgorithm:
    """Reward-ranked fine-tuning: select top-k and SFT-train on them.

    The trainer is expected to set top_k via RaftConfig. Default top_k=4
    matches the typical "keep best 25% of a 16-rollout group" pattern.
    """

    def __init__(self, config: RaftConfig | None = None) -> None:
        self.config = config or RaftConfig()

    def step(
        self,
        rewards: np.ndarray,
        log_probs: np.ndarray,
        ref_log_probs: np.ndarray,
        *,
        per_group_weight: np.ndarray | None = None,
    ) -> GrpoStepOutput:
        del per_group_weight
        return raft_step(rewards, log_probs, ref_log_probs, self.config)


_REGISTRY: dict[str, type] = {
    "grpo": GrpoAlgorithm,
    "ppo": PpoAlgorithm,
    "rloo": RlooAlgorithm,
    "raft": RaftAlgorithm,
}


def get_algorithm(name: str, config: GrpoConfig | None = None) -> AlgorithmStep:
    """Look up an algorithm by name.

    Names are case-insensitive. Unknown names raise KeyError with a
    helpful message listing the available algorithms.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown RL algorithm {name!r}; available: {sorted(_REGISTRY)}"
        )
    klass = _REGISTRY[key]
    if klass is RaftAlgorithm:
        return klass(RaftConfig(kl_beta=config.kl_beta) if config else None)
    return klass(config)
