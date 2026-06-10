"""Reward calculator implementing docs/reward_spec.md v0.1."""

from .calculator import (
    ReliabilitySignals,
    RewardBreakdown,
    RewardCalculator,
    RewardConfig,
)
from .pipeline import (
    CompletionEmbedder,
    PipelineDiagnostics,
    PipelineOutput,
    PromptContext,
    RewardPipeline,
)
from .reliability_oracle import (
    Language,
    MockOracle,
    RealOracle,
    ReliabilityOracle,
    TestCase,
    TestSpec,
)

__all__ = [
    "CompletionEmbedder",
    "Language",
    "MockOracle",
    "PipelineDiagnostics",
    "PipelineOutput",
    "PromptContext",
    "RealOracle",
    "ReliabilityOracle",
    "ReliabilitySignals",
    "RewardBreakdown",
    "RewardCalculator",
    "RewardConfig",
    "RewardPipeline",
    "TestCase",
    "TestSpec",
]
