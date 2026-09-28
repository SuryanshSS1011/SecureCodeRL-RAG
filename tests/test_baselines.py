"""Tests for the Table II baseline registry."""

from __future__ import annotations

import pytest

from cargo.eval.baselines import (
    BASELINE_REGISTRY,
    BaselineCategory,
    BaselineSpec,
    get_baseline_spec,
    list_baselines,
)


def test_registry_has_both_categories():
    categories = {spec.category for spec in BASELINE_REGISTRY.values()}
    assert categories == {BaselineCategory.UNTRAINED, BaselineCategory.SECURITY_TRAINED}


def test_get_baseline_spec_returns_spec():
    spec = get_baseline_spec("seccoderx-qwen2.5-coder-3b")
    assert spec.category == BaselineCategory.SECURITY_TRAINED


def test_get_baseline_spec_raises_on_unknown():
    with pytest.raises(KeyError):
        get_baseline_spec("not-a-baseline")


def test_list_baselines_returns_all_names():
    names = list_baselines()
    assert "qwen2.5-coder-1.5b" in names
    assert len(names) == len(BASELINE_REGISTRY)


def test_list_baselines_filtered_by_category():
    names = list_baselines(category=BaselineCategory.UNTRAINED)
    assert "starcoder2-3b" in names
    assert all(BASELINE_REGISTRY[n].category == BaselineCategory.UNTRAINED for n in names)


def test_baseline_factory_for_real_model_construct_then_generate():
    """A real HF baseline can always be *constructed* (cheap; no model load).
    Its `.generate()` raises NotImplementedError when torch+transformers are
    absent; when they're present, generation actually runs (covered by the
    real_hf tests in test_hf_baseline.py)."""
    from cargo.eval.model import SamplingConfig

    spec = get_baseline_spec("qwen2.5-coder-1.5b")
    model = spec.factory()
    assert model.name == "qwen2.5-coder-1.5b"

    try:
        import torch  # noqa: F401
        # torch is available; the .generate() path would try to download
        # the model. That's the real_hf test's job; here we just confirmed
        # constructor cheapness.
        return
    except ImportError:
        pass
    with pytest.raises(NotImplementedError):
        model.generate("x", sampling=SamplingConfig())


def test_baseline_spec_validates_name_and_category():
    with pytest.raises(ValueError):
        BaselineSpec(
            name="",
            category=BaselineCategory.UNTRAINED,
            description="x",
            model_id="x",
            factory=lambda: None,
        )


def test_security_trained_baselines_have_citation():
    for spec in BASELINE_REGISTRY.values():
        if spec.category == BaselineCategory.SECURITY_TRAINED:
            assert "reference_arxiv" in spec.metadata, spec.name
