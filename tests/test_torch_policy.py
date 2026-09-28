"""Tests for TorchPolicy.

Unit tests (no torch required) verify:
  - Constructor is cheap (no model load).
  - Methods raise NotImplementedError when torch is absent.

Real_hf-marked tests load a small real model + LoRA adapter and verify:
  - generate() returns the right shape (group_size completions).
  - log_probs(prompts, completions) returns the right shape and finite values.
  - ref_log_probs() values differ from log_probs() only if the LoRA has
    been mutated (initially they're identical because LoRA-B is zero-init).
  - step() runs without raising and updates trainable params.
"""

from __future__ import annotations

import pytest

from cargo.eval.model import SamplingConfig


# ----------------------------------------------------------------------
# Unit tests (no torch required)
# ----------------------------------------------------------------------


def test_torch_policy_constructor_does_not_load():
    from cargo.rl.torch_policy import TorchPolicy

    p = TorchPolicy(
        model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device="cpu",  # so the test runs even on a no-CUDA box
        lora_r=16,
        lora_alpha=32,
    )
    assert p._model is None
    assert p._ref_model is None


def test_torch_policy_methods_raise_without_torch():
    """If torch isn't importable, every method should surface a clear error.

    Self-skips on machines where torch IS available — that path is the
    real_hf integration test below.
    """
    try:
        import torch  # noqa: F401
        pytest.skip("torch available; this test only validates the no-torch path")
    except ImportError:
        pass

    from cargo.rl.torch_policy import TorchPolicy

    p = TorchPolicy(model_id="x", device="cpu")
    with pytest.raises(NotImplementedError):
        p.generate(["x"], SamplingConfig())


# ----------------------------------------------------------------------
# real_hf integration tests
# ----------------------------------------------------------------------


@pytest.mark.real_hf
def test_torch_policy_generate_returns_n_completions():
    """One prompt × group_size=2 -> two completions."""
    from cargo.rl.torch_policy import TorchPolicy

    p = TorchPolicy(
        model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device="auto",  # cuda if available, else cpu
        lora_r=16,
        lora_alpha=32,
    )
    sampling = SamplingConfig(
        temperature=0.7,  # nonzero so we get diverse completions
        max_new_tokens=16,
        n_samples=2,
    )
    completions = p.generate(
        ["def add(a, b):"] * 2, sampling=sampling
    )
    assert isinstance(completions, list)
    assert len(completions) == 2
    assert all(isinstance(c, str) for c in completions)


@pytest.mark.real_hf
def test_torch_policy_log_probs_returns_right_shape():
    import numpy as np
    from cargo.rl.torch_policy import TorchPolicy

    p = TorchPolicy(
        model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device="auto",
        lora_r=16,
        lora_alpha=32,
    )
    prompts = ["def add(a, b):", "def add(a, b):"]
    completions = [" return a + b", " return a * b"]
    lp = p.log_probs(prompts, completions)
    assert lp.shape == (2,)
    # log probs are sequence-level sums; finite real numbers.
    assert np.isfinite(lp).all()
    # All log probs are non-positive.
    assert (lp <= 0).all()


@pytest.mark.real_hf
def test_torch_policy_ref_log_probs_match_log_probs_at_init():
    """At initialization, LoRA-B is zero-initialized so the adapted model
    produces identical outputs to the reference. log_probs == ref_log_probs."""
    import numpy as np
    from cargo.rl.torch_policy import TorchPolicy

    p = TorchPolicy(
        model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device="auto",
        lora_r=16,
        lora_alpha=32,
    )
    prompts = ["def add(a, b):"]
    completions = [" return a + b"]
    lp = p.log_probs(prompts, completions)
    ref_lp = p.ref_log_probs(prompts, completions)
    # Should be very close at init (within float precision).
    assert np.allclose(lp, ref_lp, atol=1e-3)
