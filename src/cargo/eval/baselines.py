"""Baseline registry for Table II of the paper.

Two categories:
    UNTRAINED         - base policies evaluated zero-shot.
    SECURITY_TRAINED  - released security-trained checkpoints.

Each entry is a BaselineSpec whose factory builds an HfBaselineModel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .model import BaselineModel, HfBaselineModel


class BaselineCategory(str, Enum):
    UNTRAINED = "untrained"
    SECURITY_TRAINED = "security-trained"


@dataclass
class BaselineSpec:
    name: str
    category: BaselineCategory
    description: str
    model_id: str  # Hugging Face model id
    factory: Callable[[], BaselineModel]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BaselineSpec.name must be non-empty")
        if not isinstance(self.category, BaselineCategory):
            raise ValueError(f"category must be BaselineCategory, got {type(self.category)}")


# ---- factory helpers ----


def _hf_factory(name: str, model_id: str) -> Callable[[], BaselineModel]:
    """Factory that returns an HfBaselineModel."""

    def factory() -> BaselineModel:
        return HfBaselineModel(name=name, model_id=model_id)

    return factory


# ---- registry ----


_SPECS: list[BaselineSpec] = [
    BaselineSpec(
        name="qwen2.5-coder-1.5b",
        category=BaselineCategory.UNTRAINED,
        description="Qwen2.5-Coder-1.5B-Instruct, zero-shot. The CARGO base policy.",
        model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        factory=_hf_factory("qwen2.5-coder-1.5b", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    ),
    BaselineSpec(
        name="qwen2.5-coder-3b",
        category=BaselineCategory.UNTRAINED,
        description="Qwen2.5-Coder-3B-Instruct, zero-shot.",
        model_id="Qwen/Qwen2.5-Coder-3B-Instruct",
        factory=_hf_factory("qwen2.5-coder-3b", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    ),
    BaselineSpec(
        name="qwen2.5-coder-7b",
        category=BaselineCategory.UNTRAINED,
        description="Qwen2.5-Coder-7B-Instruct, zero-shot.",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        factory=_hf_factory("qwen2.5-coder-7b", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ),
    BaselineSpec(
        name="starcoder2-3b",
        category=BaselineCategory.UNTRAINED,
        description="StarCoder2-3B, zero-shot. Different model family.",
        model_id="bigcode/starcoder2-3b",
        factory=_hf_factory("starcoder2-3b", "bigcode/starcoder2-3b"),
    ),
    BaselineSpec(
        name="sven-codegen-2.7b",
        category=BaselineCategory.SECURITY_TRAINED,
        description="SVEN (He and Vechev, CCS 2023) on a CodeGen-2.7B base.",
        model_id="Salesforce/codegen-2B-multi",
        factory=_hf_factory("sven-codegen-2.7b", "Salesforce/codegen-2B-multi"),
        metadata={"reference_arxiv": "2302.05319", "note": "SVEN prefix-tuning weights are distributed in the SVEN repository, not on the Hugging Face Hub; this entry loads the CodeGen base."},
    ),
    BaselineSpec(
        name="seccoderx-qwen2.5-coder-3b",
        category=BaselineCategory.SECURITY_TRAINED,
        description="SecCoder-X aligned on Qwen2.5-Coder-3B.",
        model_id="SecCoderX/Qwen2.5_Coder_3B_SecCoderX_aligned",
        factory=_hf_factory("seccoderx-qwen2.5-coder-3b", "SecCoderX/Qwen2.5_Coder_3B_SecCoderX_aligned"),
        metadata={"reference_arxiv": "2602.07422"},
    ),
    BaselineSpec(
        name="seccoderx-qwen2.5-coder-7b",
        category=BaselineCategory.SECURITY_TRAINED,
        description="SecCoder-X aligned on Qwen2.5-Coder-7B.",
        model_id="SecCoderX/Qwen2.5_Coder_7B_SecCoderX_aligned",
        factory=_hf_factory("seccoderx-qwen2.5-coder-7b", "SecCoderX/Qwen2.5_Coder_7B_SecCoderX_aligned"),
        metadata={"reference_arxiv": "2602.07422"},
    ),
    BaselineSpec(
        name="starcoderbase-3b-safecoder",
        category=BaselineCategory.SECURITY_TRAINED,
        description="SafeCoder release on StarCoderBase-3B. Omitted from Table II: it emits end-of-sequence at the first position for every prompt.",
        model_id="k1h0/starcoderbase-3b-safecoder",
        factory=_hf_factory("starcoderbase-3b-safecoder", "k1h0/starcoderbase-3b-safecoder"),
        metadata={"reference_arxiv": "2402.09497"},
    ),
]


BASELINE_REGISTRY: dict[str, BaselineSpec] = {s.name: s for s in _SPECS}


def get_baseline_spec(name: str) -> BaselineSpec:
    if name not in BASELINE_REGISTRY:
        raise KeyError(
            f"unknown baseline {name!r}; available: {sorted(BASELINE_REGISTRY)}"
        )
    return BASELINE_REGISTRY[name]


def list_baselines(category: Optional[BaselineCategory] = None) -> list[str]:
    if category is None:
        return list(BASELINE_REGISTRY.keys())
    return [n for n, s in BASELINE_REGISTRY.items() if s.category == category]
