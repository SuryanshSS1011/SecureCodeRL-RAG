"""Tests for HfBaselineModel.

Unit tests (no real_hf marker) verify lazy-load semantics and error
messages when torch is absent. The real_hf-marked tests load a small
real model (Qwen2.5-Coder-1.5B-Instruct) and run end-to-end generation;
they download weights on first run and are slow. Skipped by default.
"""

from __future__ import annotations

import pytest

from cargo.eval.model import HfBaselineModel, SamplingConfig


# ----------------------------------------------------------------------
# Unit tests (no torch required)
# ----------------------------------------------------------------------


def test_hf_baseline_constructor_does_not_load():
    """Cheap constructor: just records the model id; no torch import."""
    m = HfBaselineModel(name="x", model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    assert m.name == "x"
    assert m.model_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert m._model is None  # not loaded


def test_hf_baseline_generate_raises_without_torch():
    """If torch isn't importable, generate() should surface a clear error.

    On the dev box (no torch), the load attempt fails with NotImplementedError.
    On ROAR (with torch), this test won't be representative — the
    `real_hf` marker is the path for actual integration testing.
    """
    try:
        import torch  # noqa: F401
        pytest.skip("torch available; this test only validates the no-torch path")
    except ImportError:
        pass

    m = HfBaselineModel(name="x", model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    with pytest.raises(NotImplementedError):
        m.generate("def f():", sampling=SamplingConfig())


# ----------------------------------------------------------------------
# real_hf integration tests
# ----------------------------------------------------------------------


@pytest.mark.real_hf
def test_hf_baseline_generates_text_greedy():
    """End-to-end generation against a small real model.

    Downloads weights on first run (~3GB for Qwen2.5-Coder-1.5B).
    """
    m = HfBaselineModel(
        name="qwen-1.5b",
        model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        torch_dtype="bfloat16",
        device="cuda",  # falls back to cpu inside if cuda absent
    )
    result = m.generate(
        "Write a Python function add(a, b) that returns a + b.",
        sampling=SamplingConfig(temperature=0.0, max_new_tokens=64),
    )
    assert result.text  # non-empty
    assert result.n_input_tokens is not None and result.n_input_tokens > 0
    assert result.n_output_tokens is not None and result.n_output_tokens > 0
    assert result.duration_s > 0
    assert not result.crashed


@pytest.mark.real_hf
def test_hf_baseline_greedy_is_deterministic():
    """Two identical greedy calls produce the same output."""
    m = HfBaselineModel(
        name="qwen-1.5b",
        model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    )
    sampling = SamplingConfig(temperature=0.0, max_new_tokens=32)
    r1 = m.generate("def square(x):", sampling=sampling)
    r2 = m.generate("def square(x):", sampling=sampling)
    assert r1.text == r2.text
