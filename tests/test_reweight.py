"""CWE-aware per-prompt reweighting (paper Eq. 3)."""

from __future__ import annotations

import numpy as np
import pytest

from cargo.rl.reweight import ReweightConfig, Reweighter


def test_raw_weights_are_dampened_inverse_frequency():
    rw = Reweighter(ReweightConfig(alpha_cwe=0.5), {"CWE-787": 100, "CWE-862": 1})
    assert rw.weights["CWE-787"] == pytest.approx((2 / 100) ** 0.5)
    assert rw.weights["CWE-862"] == pytest.approx(2**0.5)


def test_batch_weights_have_mean_one():
    rw = Reweighter.from_cwes(ReweightConfig(), ["CWE-79"] * 90 + ["CWE-306"] * 10)
    w = rw.weights_for_batch(["CWE-79", "CWE-79", "CWE-306", "CWE-79"])
    assert w.mean() == pytest.approx(1.0)
    assert w[2] > w[0]


def test_alpha_zero_and_disabled_give_uniform_weights():
    counts = {"CWE-79": 50, "CWE-306": 5}
    batch = ["CWE-79", "CWE-306"]
    assert np.allclose(Reweighter(ReweightConfig(alpha_cwe=0.0), counts).weights_for_batch(batch), 1.0)
    assert np.allclose(Reweighter(ReweightConfig(enabled=False), counts).weights_for_batch(batch), 1.0)


def test_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        Reweighter(ReweightConfig(alpha_cwe=1.5), {"CWE-79": 1})
