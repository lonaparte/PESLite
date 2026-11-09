"""Ideal-diode conduction and discrete commutation around an otherwise generic ODE solver."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .params import ConfigError
from ..solver.integrators import (_accepted_checkpoint, _accepted_checkpoint_at,
                                  _accepted_restore)
from ..solver.model import commutation_tolerance

__all__ = ["DiodeConduction", "CommutationError"]

_MAX_COMMUTATIONS = 64  # commutations allowed at one instant before the model is rejected
_LOCATOR_ITERATIONS = 80


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
    smooth and the solver keeps its order. Built-in solvers observe the guard at their own accepted
    steps; a user solver without that interface needs a finite local check interval.
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
            and getattr(subsystem, "commutation_active", True)
        )
        self.located = tuple(
            subsystem for subsystem in self.commutating
            if getattr(subsystem, "locate_commutations", True)
        )
        self._boundary_only = tuple(
            subsystem for subsystem in self.commutating if subsystem not in self.located
        )
        self.state_guard_subsystems = tuple(
            subsystem for subsystem in self.located
            if callable(getattr(subsystem, "commutation_state_guards", None))
        )
        self._topology_owners = tuple(
            subsystem for subsystem in model.subsystems
            if getattr(subsystem, "requires_topology_freeze", False)
            if callable(getattr(subsystem, "freeze_topology", None))
        )
        self._scheduled = tuple(
            subsystem for subsystem in self.commutating
            if callable(getattr(subsystem, "next_commutation_time", None))
        )
        if self.located and getattr(solver, "outer", None) is not None:
            raise ConfigError(
                "simulation.solver.subsystems: commutating subsystems "
                f"{sorted(model.name_of(sub) for sub in self.located)} need a single-rate "
                "solver; a windowed subsystem group cannot locate their commutation instants"
            )
        self._probe = None
        if self.located and solver is not None:
            dt = getattr(solver, "dt", None)
            probe = dt if dt is not None and dt > 0.0 else getattr(solver, "max_step", math.inf)
            if math.isfinite(probe) and probe > 0.0:
                self._probe = float(probe)
            elif not callable(getattr(solver, "advance_guarded", None)):
                raise ConfigError(
                    "simulation.solver.max_step must be finite with this solver: it is the "
                    "interval at which commutating subsystems "
                    f"{sorted(model.name_of(sub) for sub in self.commutating)} are checked"
                )
        guarded = getattr(solver, "advance_guarded", None)
        self._guarded_advance = guarded if callable(guarded) else None
        validate = getattr(solver, "validate_local_step", None)
        if validate is not None:
            validate(self.resolution_subsystems)

    def __bool__(self) -> bool:
        return bool(self.current_states or self.resolution_subsystems
                    or self.commutating or self._topology_owners)

    def synchronize(self, t: float, state: NDArray[np.float64]) -> None:
        """Resolve boundary modes before measurements, controls or observations read them."""
        self._settle(t, state)

    def advance(
            self, t0: float, t1: float, state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Advance all conduction guards through one locator and one boundary dispatcher."""
        if self.solver is None:
            raise RuntimeError("diode conduction cannot advance without a solver")
        if not (self.commutating or self.current_states or self.located or self._scheduled):
            return self._smooth(t0, t1, state)  # no guard to plan or observe
        initial_guarded, initial_located = self._guard_plan(state)
        scheduled_due = any(
            float(subsystem.next_commutation_time()) <= t0 + commutation_tolerance(t0)
            for subsystem in self._scheduled
        )
        state_proves_mode = (
            not self._boundary_only
            and not initial_located
            and all(direction * state[index] > 0.0
                    for index, direction, commutates in initial_guarded if commutates)
        )
        if self.commutating and (scheduled_due or not state_proves_mode):
            state = self._settle(t0, state)
            initial_guarded = initial_located = None
        if not self.current_states and not self.located and not self._scheduled:
            return self._smooth(t0, t1, state)
        t = t0
        end = t1 - commutation_tolerance(t1)
        stalled = 0
        while t < end:
            guarded_advance = self._guarded_advance
            if initial_guarded is None:
                guarded, located = self._guard_plan(state)
            else:
                guarded, located = initial_guarded, initial_located
                initial_guarded = initial_located = None
            observe_steps = guarded_advance is not None and bool(guarded or located)
            target = (
                t1 if guarded_advance is not None or self._probe is None
                else min(t1, t + self._probe)
            )
            scheduled = math.inf
            if self._scheduled:
                scheduled = min(float(sub.next_commutation_time()) for sub in self._scheduled)
                target = min(target, scheduled)
            origin_t, origin_state = t, state
            origin_checkpoint = _accepted_checkpoint()
            lower_margin = None
            if observe_steps:
                step = self._guarded(t, target, state.copy(), guarded, located)
                trial, upper_margin = step.y, step.margin
                if step.crossed:
                    origin_t, origin_state = step.previous_t, step.previous_y
                    origin_checkpoint = _accepted_checkpoint_at(origin_t)
                    lower_margin = step.previous_margin
                    target = step.t
            else:
                trial, upper_margin = self._trial(t, target, state.copy(), guarded, located)
            if upper_margin < 0.0:
                upper_checkpoint = _accepted_checkpoint()
                lower, upper = origin_t, target
                if lower_margin is None:
                    lower_margin = self._margin(lower, origin_state, guarded, located)
                bisect_next = False
                for _ in range(_LOCATOR_ITERATIONS):
                    tolerance = commutation_tolerance(upper)
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
                    _accepted_restore(origin_checkpoint)
                    value, margin = self._trial(
                        origin_t, middle, origin_state.copy(), guarded, located
                    )
                    candidate_checkpoint = _accepted_checkpoint()
                    invalid = margin < 0.0
                    if invalid:
                        upper, upper_margin, trial = middle, margin, value
                        upper_checkpoint = candidate_checkpoint
                    else:
                        lower, lower_margin = middle, margin
                    if interpolated:
                        # Verify the opposite side of an interpolated zero in time, rather than
                        # accepting a small residual (a valid guard may have a zero plateau).
                        adjacent = middle - tolerance if invalid else middle + tolerance
                        if lower < adjacent < upper:
                            _accepted_restore(origin_checkpoint)
                            value, margin = self._trial(
                                origin_t, adjacent, origin_state.copy(), guarded, located
                            )
                            candidate_checkpoint = _accepted_checkpoint()
                            if margin < 0.0:
                                upper, upper_margin, trial = adjacent, margin, value
                                upper_checkpoint = candidate_checkpoint
                            else:
                                lower, lower_margin = adjacent, margin
                _accepted_restore(upper_checkpoint)
                target = upper
                for index, direction, commutates in guarded:
                    if not commutates and direction * trial[index] < 0.0:
                        trial[index] = 0.0
            state = trial
            if self.commutating and (upper_margin < 0.0 or scheduled <= target):
                state = self._settle(target, state)
            if target == t1:
                return state
            stalled = stalled + 1 if target - t <= commutation_tolerance(t) else 0
            if stalled >= _MAX_COMMUTATIONS:
                raise CommutationError(f"t = {t:.12g} s: conduction events cannot advance time")
            t = target
        return state

    def _guard_plan(self, state: NDArray[np.float64]) -> tuple[tuple, tuple]:
        """Replace located subsystem margins with their proved state guards."""
        guarded = tuple((index, 1.0, False) for sub, index in self.current_states
                        if sub.diode_blocks_reverse_current() and state[index] > 0.0)
        located = list(self.located)
        for subsystem in self.state_guard_subsystems:
            replacements = subsystem.commutation_state_guards()
            if replacements is None:
                continue
            located.remove(subsystem)
            guarded += tuple((index, direction, True) for index, direction in replacements)
        return guarded, tuple(located)

    def _guarded(self, t0: float, t1: float, state: NDArray[np.float64], guarded, located):
        """Observe margins inside one solver call while reusing a fixed topology."""
        maximum_step = self._probe if self._probe is not None else math.inf
        if self.resolution_subsystems:
            self.model.sync(t0, state)
            maximum_step = min(
                maximum_step,
                *(subsystem.diode_maximum_step() for subsystem in self.resolution_subsystems),
            )
        for subsystem in self._topology_owners:
            subsystem.freeze_topology()
        try:
            assert self._guarded_advance is not None
            return self._guarded_advance(
                self.model.rhs, t0, t1, state,
                lambda t, y: self._margin(t, y, guarded, located),
                max_step=maximum_step,
                synchronized_guard=lambda t, y: self._margin(
                    t, y, guarded, located, synchronized=True
                ),
            )
        finally:
            for subsystem in self._topology_owners:
                subsystem.release_topology()

    def _trial(self, t0: float, t1: float, state: NDArray[np.float64], guarded, located
               ) -> tuple[NDArray[np.float64], float]:
        """Integrate and test one smooth interval while reusing its compiled topology."""
        value = self._smooth(t0, t1, state)
        return value, self._margin(t1, value, guarded, located)

    def _smooth(self, t0: float, t1: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Integrate one interval while every topology owner reuses its boundary mode."""
        for subsystem in self._topology_owners:
            subsystem.freeze_topology()
        try:
            return self._advance_with_resolution(t0, t1, state)
        finally:
            for subsystem in self._topology_owners:
                subsystem.release_topology()

    def _margin(self, t: float, state: NDArray[np.float64], guarded=(), subsystems=None, *,
                synchronized: bool = False) -> float:
        """Minimum guard margin; no owner lists are needed to bracket a boundary."""
        subsystems = self.commutating if subsystems is None else subsystems
        if subsystems and not synchronized:
            self.model.sync(t, state)
        margin = math.inf
        for sub in subsystems:
            margin = min(margin, sub.commutation_margin())
        for index, direction, _commutates in guarded:
            value = direction * state[index]
            if value < margin:
                margin = value
        return float(margin)

    def _settle(self, t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Dispatch every due schedule and invalid mode at the same physical boundary.

        Scheduled and located guards share this path, including the initial boundary and events
        exposed by another commutation. Each owner is synchronized before its state transition.
        """
        if not self.commutating:
            return state
        due = t + commutation_tolerance(t)
        commutated = False
        for _ in range(_MAX_COMMUTATIONS):
            self.model.sync(t, state)
            pending = {sub for sub in self.commutating if sub.commutation_margin() < 0.0}
            pending.update(sub for sub in self._scheduled
                           if float(sub.next_commutation_time()) <= due)
            if not pending:
                if commutated:
                    invalidate = getattr(self.model, "invalidate_equations", None)
                    if invalidate is not None:
                        invalidate()
                return state
            changed = False
            for sub in self.commutating:
                if sub not in pending:
                    continue
                if changed:
                    self.model.sync(t, state)
                scheduled = sub in self._scheduled and float(sub.next_commutation_time()) <= due
                if changed and not scheduled and sub.commutation_margin() >= 0.0:
                    continue
                sub.commutate(t)
                state[:] = self.model.get_initial_values()
                changed = True
                commutated = True
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
