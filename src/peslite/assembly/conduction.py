"""Ideal-diode conduction handling around an otherwise generic ODE solver."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["DiodeConduction"]


class DiodeConduction:
    """Apply ideal-diode current blocking and DCM resolution for one assembled model.

    A plant participates by declaring a zero lower state bound and providing
    ``diode_blocks_reverse_current()`` for an exact-switching diode path, or
    ``diode_maximum_step()`` when its averaged diode law needs a local resolution limit.
    Solvers receive that limit through their generic per-call ``advance`` interface.
    """

    def __init__(self, model: Any, solver: Any = None) -> None:
        self.model = model
        self.solver = solver
        self.current_states = tuple(
            (subsystem, model.state_slice(subsystem, state).start)
            for subsystem in model.subsystems
            if callable(getattr(subsystem, "diode_blocks_reverse_current", None))
            for state, (lower, _upper) in getattr(subsystem, "state_bounds", {}).items()
            if lower == 0.0
        )
        self.resolution_subsystems = tuple(
            subsystem for subsystem in model.subsystems
            if callable(getattr(subsystem, "diode_maximum_step", None))
        )
        validate = getattr(solver, "validate_local_step", None)
        if validate is not None:
            validate(self.resolution_subsystems)

    def __bool__(self) -> bool:
        return bool(self.current_states or self.resolution_subsystems)

    def advance(
            self, t0: float, t1: float, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Advance while enforcing this model's ideal-diode conduction transitions."""
        if self.solver is None:
            raise RuntimeError("diode conduction cannot advance without a solver")
        if not self.current_states:
            return self._advance_with_resolution(t0, t1, state)

        t = t0
        while t < t1 - 1e-15:
            initial = state.copy()
            trial = self._advance_with_resolution(t, t1, initial)
            guarded = tuple(
                index for subsystem, index in self.current_states
                if subsystem.diode_blocks_reverse_current()
            )
            crossing = tuple(
                index for index in guarded
                if initial[index] > 0.0 and trial[index] < 0.0
            )
            if not crossing:
                return trial

            roots: list[tuple[float, int]] = []
            for index in crossing:
                lower, upper = t, t1
                for _ in range(24):
                    middle = lower + 0.5 * (upper - lower)
                    value = self._advance_with_resolution(t, middle, initial)[index]
                    if value > 0.0:
                        lower = middle
                    else:
                        upper = middle
                roots.append((lower + 0.5 * (upper - lower), index))

            first = min(at for at, _index in roots)
            state = self._advance_with_resolution(t, first, initial)
            for at, index in roots:
                if abs(at - first) <= 2e-14:
                    state[index] = 0.0
            t = first
        return state

    def _advance_with_resolution(
            self, t0: float, t1: float, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Advance with any local resolution required by an averaged diode law."""
        if not self.resolution_subsystems:
            return self.solver(self.model.rhs, t0, t1, state).y

        self.model.sync(t0, state)
        maximum_step = min(
            subsystem.diode_maximum_step()
            for subsystem in self.resolution_subsystems
        )
        advance = getattr(self.solver, "advance", None)
        if advance is not None:
            return advance(
                self.model.rhs, t0, t1, state, max_step=maximum_step,
            ).y

        t = t0
        while t < t1 - 1e-15:
            self.model.sync(t, state)
            maximum_step = min(
                subsystem.diode_maximum_step()
                for subsystem in self.resolution_subsystems
            )
            target = min(t1, t + maximum_step)
            if target <= t:
                raise FloatingPointError("diode conduction step is below time resolution")
            state = self.solver(self.model.rhs, t, target, state).y
            t = target
        return state
