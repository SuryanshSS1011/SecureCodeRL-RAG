"""Evaluation harness, metric registry, and Table II baselines."""

from .baselines import (
    BASELINE_REGISTRY,
    BaselineCategory,
    BaselineSpec,
    get_baseline_spec,
    list_baselines,
)
from .harness import EvalHarness, EvalReport, PerPromptRecord
from .model import (
    BaselineModel,
    CompletionResult,
    HfBaselineModel,
    MockModel,
    SamplingConfig,
)

__all__ = [
    "BASELINE_REGISTRY",
    "BaselineCategory",
    "BaselineModel",
    "BaselineSpec",
    "CompletionResult",
    "EvalHarness",
    "EvalReport",
    "HfBaselineModel",
    "MockModel",
    "PerPromptRecord",
    "SamplingConfig",
    "get_baseline_spec",
    "list_baselines",
]
