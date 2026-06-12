"""CWE-aware per-prompt gradient reweighting (paper Section IV-B, Eq. 3).

    w(x) = (|C| / n_c(x))^alpha_cwe / Z

`c(x)` is the CWE label of prompt x, `|C|` the number of trained CWEs,
`n_c` the number of training prompts labeled c, and `Z` normalizes the
weights so that their mean over each sampled batch is 1. The exponent
`alpha_cwe` in [0, 1] dampens raw inverse-frequency reweighting so the
smallest categories are not over-amplified.

The weights scale each prompt's loss contribution without touching the
within-group advantage normalization, so they compose with every
algorithm in the registry. The trainer broadcasts `weights_for_batch`
across each group's rollouts and hands the per-sample weights to the
policy's surrogate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

DEFAULT_ALPHA_CWE = 0.5


@dataclass
class ReweightConfig:
    enabled: bool = True
    alpha_cwe: float = DEFAULT_ALPHA_CWE


class Reweighter:
    """Per-prompt weights from training-pool CWE counts (Eq. 3)."""

    def __init__(self, cfg: ReweightConfig, cwe_counts: Mapping[str, int]) -> None:
        if not 0.0 <= cfg.alpha_cwe <= 1.0:
            raise ValueError(f"alpha_cwe must be in [0, 1], got {cfg.alpha_cwe}")
        if not cwe_counts or any(n <= 0 for n in cwe_counts.values()):
            raise ValueError("cwe_counts must be non-empty with positive counts")
        self.cfg = cfg
        n_cwes = len(cwe_counts)
        self._raw: dict[str, float] = {
            cwe: (n_cwes / n) ** cfg.alpha_cwe for cwe, n in cwe_counts.items()
        }

    @classmethod
    def from_cwes(cls, cfg: ReweightConfig, cwes: Iterable[str]) -> Reweighter:
        """Build from the CWE label of every training prompt."""
        return cls(cfg, Counter(cwes))

    @property
    def trained_cwes(self) -> list[str]:
        return sorted(self._raw)

    @property
    def weights(self) -> dict[str, float]:
        """Unnormalized per-CWE weights (|C| / n_c)^alpha_cwe."""
        return dict(self._raw)

    def weights_for_batch(self, group_cwes: list[str]) -> np.ndarray:
        """Per-prompt weights for one batch, normalized to mean 1 (Z)."""
        if not self.cfg.enabled:
            return np.ones(len(group_cwes), dtype=float)
        raw = np.array([self._raw[c] for c in group_cwes], dtype=float)
        return raw / raw.mean()
