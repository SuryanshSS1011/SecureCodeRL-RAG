"""Step-indexed reward-mix / KL schedule.

The paper's configuration is a single phase: alpha_mix = 0.3 (Eq. 4) and
KL coefficient beta_KL = 0.05 against the frozen base policy for the whole
run. Multi-phase schedules remain supported for experimentation. The
schedule is step-indexed and deterministic. Phase k applies for steps
in `[phases[k].start_step, phases[k+1].start_step)`. Phase 1 must start
at step 0 so every step is covered.

Step counts (not online metrics) drive transitions to keep the schedule
re-runnable and to avoid spurious mid-training resets if a metric
oscillates around a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseSpec:
    """One schedule phase: reward mix `alpha` and KL coefficient `beta`."""

    number: int
    start_step: int
    alpha: float
    beta: float


DEFAULT_PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec(number=1, start_step=0, alpha=0.3, beta=0.05),
)


class PhaseSchedule:
    """Look up the current phase for a given training step."""

    def __init__(self, phases: list[PhaseSpec] | tuple[PhaseSpec, ...]) -> None:
        if not phases:
            raise ValueError("PhaseSchedule needs at least one phase")
        # Validate sortedness and starting-at-zero.
        last_start = -1
        for phase in phases:
            if phase.start_step <= last_start:
                raise ValueError(
                    f"phases must be strictly increasing by start_step; got "
                    f"phase {phase.number} starting at {phase.start_step} after {last_start}"
                )
            last_start = phase.start_step
        if phases[0].start_step != 0:
            raise ValueError(
                f"first phase must start at step 0; got {phases[0].start_step}"
            )
        self.phases: tuple[PhaseSpec, ...] = tuple(phases)

    def for_step(self, step: int) -> PhaseSpec:
        """Return the phase applicable at `step`.

        Phase k applies for `phases[k].start_step <= step < phases[k+1].start_step`.
        The last phase applies for all step >= phases[-1].start_step.
        """
        chosen = self.phases[0]
        for phase in self.phases:
            if phase.start_step <= step:
                chosen = phase
            else:
                break
        return chosen
