"""RL algorithms + trainer orchestration per docs/training_spec.md v0.1."""

from .grpo import GrpoConfig, GrpoStepOutput, compute_group_advantages, grpo_step
from .metrics import StepMetricsAggregator, TrainStepRecord
from .ppo import ppo_step
from .raft import RaftConfig, raft_step, select_top_k_per_group
from .registry import (
    AlgorithmStep,
    GrpoAlgorithm,
    PpoAlgorithm,
    RaftAlgorithm,
    RlooAlgorithm,
    get_algorithm,
)
from .rloo import compute_loo_advantages, rloo_step
from .schedule import DEFAULT_PHASES, PhaseSchedule, PhaseSpec
from .torch_policy import TorchPolicy, TorchPolicyConfig
from .trainer import PolicyProtocol, Trainer, TrainerConfig

__all__ = [
    "AlgorithmStep",
    "DEFAULT_PHASES",
    "GrpoAlgorithm",
    "GrpoConfig",
    "GrpoStepOutput",
    "PhaseSchedule",
    "PhaseSpec",
    "PolicyProtocol",
    "PpoAlgorithm",
    "RaftAlgorithm",
    "RaftConfig",
    "RlooAlgorithm",
    "StepMetricsAggregator",
    "TorchPolicy",
    "TorchPolicyConfig",
    "TrainStepRecord",
    "Trainer",
    "TrainerConfig",
    "compute_group_advantages",
    "compute_loo_advantages",
    "get_algorithm",
    "grpo_step",
    "ppo_step",
    "raft_step",
    "rloo_step",
    "select_top_k_per_group",
]
