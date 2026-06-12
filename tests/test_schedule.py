"""Tests for the reward-mix / KL schedule.

The paper's configuration is one phase: alpha_mix = 0.3, beta_KL = 0.05.
Multi-phase boundary semantics are pinned with explicit phase lists.
"""

from __future__ import annotations

import pytest

from secure_code_rl_ictai.rl.schedule import (
    DEFAULT_PHASES,
    PhaseSchedule,
    PhaseSpec,
)


def test_default_is_single_phase_with_paper_values():
    assert len(DEFAULT_PHASES) == 1
    assert DEFAULT_PHASES[0].alpha == 0.3
    assert DEFAULT_PHASES[0].beta == 0.05


def test_schedule_returns_phase_1_at_step_0():
    sched = PhaseSchedule(DEFAULT_PHASES)
    phase = sched.for_step(0)
    assert phase.number == 1


def test_schedule_returns_phase_1_just_before_boundary():
    """Boundary semantics: phase k applies for `start_step <= s < next.start_step`."""
    phases = [
        PhaseSpec(number=1, start_step=0, alpha=0.3, beta=0.01),
        PhaseSpec(number=2, start_step=1000, alpha=0.5, beta=0.02),
    ]
    sched = PhaseSchedule(phases)
    assert sched.for_step(999).number == 1
    assert sched.for_step(1000).number == 2


def test_schedule_returns_last_phase_for_large_step():
    sched = PhaseSchedule(DEFAULT_PHASES)
    huge = DEFAULT_PHASES[-1].start_step + 1_000_000
    assert sched.for_step(huge).number == DEFAULT_PHASES[-1].number


def test_schedule_rejects_unsorted_phases():
    with pytest.raises(ValueError):
        PhaseSchedule(
            [
                PhaseSpec(number=2, start_step=1000, alpha=0.5, beta=0.02),
                PhaseSpec(number=1, start_step=0, alpha=0.3, beta=0.01),
            ]
        )


def test_schedule_rejects_empty():
    with pytest.raises(ValueError):
        PhaseSchedule([])


def test_schedule_rejects_phase_starting_after_first_step():
    """Phase 1 must start at step 0 so every step is covered."""
    with pytest.raises(ValueError):
        PhaseSchedule(
            [PhaseSpec(number=1, start_step=10, alpha=0.3, beta=0.01)]
        )


def test_schedule_iterates_in_phase_order():
    phases = [
        PhaseSpec(number=1, start_step=0, alpha=0.3, beta=0.05),
        PhaseSpec(number=2, start_step=500, alpha=0.5, beta=0.05),
        PhaseSpec(number=3, start_step=800, alpha=0.7, beta=0.05),
    ]
    sched = PhaseSchedule(phases)
    nums = [p.number for p in sched.phases]
    assert nums == [1, 2, 3]
