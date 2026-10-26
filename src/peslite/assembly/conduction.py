"""Ideal-diode conduction and discrete commutation around an otherwise generic ODE solver."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .params import ConfigError

__all__ = ["DiodeConduction", "CommutationError"]

_TIME_TOL = 1e-15  # absolute/relative time resolution of a located commutation
_MAX_COMMUTATIONS = 64  # commutations allowed at one instant before the model is rejected


class CommutationError(RuntimeError):
    """Commutating subsystems kept switching without time advancing (a Zeno condition)."""


class DiodeConduction:
    """Apply ideal-diode blocking and exact discrete commutations for one assembled model.

    A plant participates in one of three ways:

    * an ideal diode in series with one current state declares a zero lower state bound and
      ``diode_blocks_reverse_current()``; the state is clamped at its located zero;
    * an averaged diode law declares ``diode_maximum_step()``, a local resolution limit passed to
      the solver through its generic per-call ``advance`` interface;
    * a *commutating* subsystem holds a discrete state (for example the conducting pair of a diode
      bridge or a switch position) that is valid while ``commutation_margin()`` is nonnegative.
      ``commutate(t)`` changes it at the located instant where the margin becomes negative, and
      an optional ``next_commutation_time()`` names a scheduled instant at which ``commutate`` is
      called exactly. ``commutate`` may reinitialize the subsystem's own continuous states; their
      values are written back into the state vector.

    The margin is checked against the state *after* every probe interval, so a margin which an
    outer event (a bridge edge, a register load) makes negative at the start of an interval is
    handled exactly like one which crosses zero inside it. Between commutations the model is
    smooth and the solver keeps its order. The probe interval is the fixed step ``dt`` or, for an
    adaptive solver, its ``max_step``, which must then be finite: the checks must not depend on how
    far apart the outer events (output, control and PWM instants) happen to be.
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
        self.commutating = tuple(
            subsystem for subsystem in model.subsystems
            if callable(getattr(subsystem, "commutation_margin", None))
            and callable(getattr(subsystem, "commutate", None))
        )
        self._scheduled = tuple(
            subsystem for subsystem in self.commutating
            if callable(getattr(subsystem, "next_commutation_time", None))
        )
        if self.commutating and getattr(solver, "outer", None) is not None:
            raise ConfigError(
                "simulation.solver.subsystems: commutating subsystems "
                f"{sorted(model.name_of(sub) for sub in self.commutating)} need a single-rate "
                "solver; a windowed subsystem group cannot locate their commutation instants"
            )
        self._probe = None
        if self.commutating and solver is not None:
            dt = getattr(solver, "dt", None)
            probe = dt if dt is not None and dt > 0.0 else getattr(solver, "max_step", math.inf)
            if not math.isfinite(probe) or probe <= 0.0:
                raise ConfigError(
                    "simulation.solver.max_step must be finite with an adaptive solver: it is the "
                    "interval at which commutating subsystems "
                    f"{sorted(model.name_of(sub) for sub in self.commutating)} are checked"
                )
            self._probe = float(probe)
        validate = getattr(solver, "validate_local_step", None)
        if validate is not None:
            validate(self.resolution_subsystems)

    def __bool__(self) -> bool:
        return bool(self.current_states or self.resolution_subsystems or self.commutating)

    def synchronize(self, t: float, state: NDArray[np.float64]) -> None:
        """Resolve boundary modes before measurements, controls or observations read them."""
        self._settle(t, state)

    def advance(
            self, t0: float, t1: float, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Advance all conduction guards through one locator and one boundary dispatcher."""
        if self.solver is None:
            raise RuntimeError("diode conduction cannot advance without a solver")
        if not self.current_states and not self.commutating:
            return self._advance_with_resolution(t0, t1, state)
        if self.commutating:
            state = self._settle(t0, state)
        t = t0
        end = t1 - _TIME_TOL * max(1.0, abs(t1))
        stalled = 0
        while t < end:
            target = min(t1, t + self._probe) if self._probe is not None else t1
            scheduled = math.inf
            if self._scheduled:
                scheduled = min(float(sub.next_commutation_time()) for sub in self._scheduled)
                target = min(target, scheduled)
            guarded = tuple(index for sub, index in self.current_states
                            if sub.diode_blocks_reverse_current() and state[index] > 0.0)
            trial = self._advance_with_resolution(t, target, state.copy())
            upper_margin = self._margin(target, trial, guarded)
            if upper_margin < 0.0:
                lower, upper = t, target
                lower_margin = self._margin(t, state, guarded)
                bisect_next = False
                for _ in range(80):
                    tolerance = _TIME_TOL * max(1.0, abs(upper))
                    width = upper - lower
                    if width <= tolerance:
                        break
                    middle = lower + 0.5 * width
                    interpolated = False
                    if not bisect_next and lower_margin > 0.0 and upper_margin < 0.0:
                        candidate = lower + width * lower_margin / (lower_margin - upper_margin)
                        candidate = max(lower + tolerance, min(upper - tolerance, candidate))
                        if lower < candidate < upper:
                            middle = candidate
                            interpolated = True
                    # Alternation guarantees bracket contraction even if a secant hugs an end.
                    bisect_next = interpolated
                    value = self._advance_with_resolution(t, middle, state.copy())
                    margin = self._margin(middle, value, guarded)
                    invalid = margin < 0.0
                    if invalid:
                        upper, upper_margin, trial = middle, margin, value
                    else:
                        lower, lower_margin = middle, margin
                    if interpolated:
                        # Verify the opposite side of an interpolated zero in time, rather than
                        # accepting a small residual (a valid guard may have a zero plateau).
                        adjacent = middle - tolerance if invalid else middle + tolerance
                        if lower < adjacent < upper:
                            value = self._advance_with_resolution(t, adjacent, state.copy())
                            margin = self._margin(adjacent, value, guarded)
                            if margin < 0.0:
                                upper, upper_margin, trial = adjacent, margin, value
                            else:
                                lower, lower_margin = adjacent, margin
                target = upper
                for index in guarded:
                    if trial[index] < 0.0:
                        trial[index] = 0.0
            state = trial
            if self.commutating and (upper_margin < 0.0 or scheduled <= target):
                state = self._settle(target, state)
            if target == t1:
                return state
            stalled = stalled + 1 if target - t <= _TIME_TOL * max(1.0, abs(t)) else 0
            if stalled >= _MAX_COMMUTATIONS:
                raise CommutationError(f"t = {t:.12g} s: conduction events cannot advance time")
            t = target
        return state

    def _margin(self, t: float, state: NDArray[np.float64], guarded=()) -> float:
        """Minimum guard margin; no owner lists are needed to bracket a boundary."""
        if self.commutating:
            self.model.sync(t, state)
        margin = math.inf
        for sub in self.commutating:
            margin = min(margin, sub.commutation_margin())
        for index in guarded:
            if state[index] < margin:
                margin = state[index]
        return float(margin)

    def _settle(self, t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Dispatch every due schedule and invalid mode at the same physical boundary.

        Scheduled and located guards share this path, including the initial boundary and events
        exposed by another commutation. Each owner is synchronized before its state transition.
        """
        if not self.commutating:
            return state
        tolerance = _TIME_TOL * max(1.0, abs(t))
        for _ in range(_MAX_COMMUTATIONS):
            self.model.sync(t, state)
            pending = {sub for sub in self.commutating if sub.commutation_margin() < 0.0}
            pending.update(sub for sub in self._scheduled
                           if float(sub.next_commutation_time()) <= t + tolerance)
            if not pending:
                return state
            for sub in self.commutating:
                if sub not in pending:
                    continue
                self.model.sync(t, state)
                scheduled = (callable(getattr(sub, "next_commutation_time", None))
                             and float(sub.next_commutation_time()) <= t + tolerance)
                if not scheduled and sub.commutation_margin() >= 0.0:
                    continue
                sub.commutate(t)
                state[:] = self.model.get_initial_values()
        names = sorted(self.model.name_of(sub) for sub in pending)
        raise CommutationError(
            f"t = {t:.12g} s: {names} commutated {_MAX_COMMUTATIONS} times without time "
            "advancing; the configured circuit has no consistent conduction state here"
        )

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
