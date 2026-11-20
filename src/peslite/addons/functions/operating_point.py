"""Nonintrusive operating-point classification using PESLite's restartable loop map."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...solver.linearization import compare_loop_states
from ...solver.simulation import SimulationTransition
from ...solver.timing import TIME_EPS

if TYPE_CHECKING:
    from ...solver.simulation import Simulation

__all__ = ["OperatingPointResult", "detect_operating_point"]


@dataclass(frozen=True)
class OperatingPointResult:
    """A check of one completed state, with no eigenvalue calculation."""

    at: float
    kind: str  # dc, periodic, or unconverged
    period: float | None
    periodic_error: float | None
    intra_period_error: float | None
    evaluations: int
    diagnostics: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.kind != "unconverged"

    def to_dict(self) -> dict:
        return {
            "at": {"value": self.at, "unit": "s"},
            "kind": self.kind,
            "valid": self.valid,
            "period": None if self.period is None else {"value": self.period, "unit": "s"},
            "periodic_error": (self.periodic_error if self.periodic_error is not None
                               and math.isfinite(self.periodic_error) else None),
            "intra_period_error": (self.intra_period_error if self.intra_period_error is not None
                                   and math.isfinite(self.intra_period_error) else None),
            "evaluations": self.evaluations,
            "diagnostics": list(self.diagnostics),
        }


def _period_error(loop: SimulationTransition, initial: dict, final: dict,
                  start: float, end: float) -> tuple[float, str | None]:
    comparison = compare_loop_states(loop, initial, final, start, end)
    if comparison.changed_modes:
        name = comparison.changed_modes[0]
        return math.inf, f"discrete state {name!r} changes within the check window"
    return comparison.error, comparison.worst_coordinate


def detect_operating_point(sim: Simulation, *, period: float | None = None,
                           tolerance: float = 1e-3) -> OperatingPointResult:
    """Classify the final saved operating point through one period and interior points.

    Unlike :func:`small_signal`, this does not calculate a transition matrix. The
    configured control, PWM, source and forcing schedules determine the period.
    Autonomous models with no schedule use the configured output sample period.
    Interior checks distinguish a static DC state from an orbit that returns
    to the same end state. Failure to establish recurrence is never called stable.
    """
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if period is not None and (isinstance(period, bool) or not math.isfinite(period)
                               or period <= TIME_EPS):
        raise ValueError("period must be finite and exceed the simulation's time resolution")
    if sim.result is None:
        raise ValueError("detect_operating_point requires a completed Simulation.run()")
    row = sim.result.final_states()
    at = float(row.pop("t"))

    def fail(message: str, *, span: float | None = None, error: float | None = None,
             evaluations: int = 0) -> OperatingPointResult:
        return OperatingPointResult(at, "unconverged", span, error, None,
                                    evaluations, (message,))

    if sim.result.summary.get("stop_reason"):
        return fail(f"simulation stopped early: {sim.result.summary['stop_reason']}")
    if sim.result.tripped:
        return fail("operating point is tripped")
    if not sim._rebuildable:
        return fail("operating point cannot be rebuilt from part instances")
    if any(getattr(unit.ctrl, "startup", None) is not None and unit.ctrl.startup.in_progress
           for unit in sim.units.values()):
        return fail("startup ramp has not completed")

    try:
        loop = SimulationTransition(sim, period=period, frame_frequency=0.0, repeat_forcing=True)
        if loop.period is None and not loop.periods:
            sample = float(sim.p.simulation.output.period)
            if not math.isfinite(sample) or sample <= TIME_EPS:
                return fail("no valid autonomous-state comparison interval")
            loop = SimulationTransition(sim, period=sample, frame_frequency=0.0,
                                        repeat_forcing=True)
        if loop.period is None:
            return fail("no common loop period; supply a valid period explicitly")
        span = loop.period
        if any(at < change.t <= at + span + TIME_EPS for change in sim.p.changes):
            return fail("check interval crosses a non-repeating simulation event", span=span)
        final = loop.advance(dict(row), at)
        error, worst = _period_error(loop, row, final, at, at + span)
        if error > tolerance:
            detail = (f"nonrepeating state {worst!r}: scaled residual {error:.3g} "
                      f"exceeds {tolerance:.3g}") if math.isfinite(error) else str(worst)
            return fail(detail, span=span, error=error, evaluations=1)
        # Interior advances need not repeat a schedule: their purpose is only
        # to reveal motion hidden by identical full-period endpoints. A quarter
        # and a half period avoid mistaking a zero crossing for a static state.
        interior = 0.0
        for fraction in (0.25, 0.5):
            loop.period = span * fraction
            try:
                observed = loop.advance(dict(row), at)
            finally:
                loop.period = span
            deviation, _ = _period_error(loop, row, observed, at, at + span * fraction)
            interior = max(interior, deviation)
        midway = interior
        if midway <= tolerance:
            kind = "dc"
        else:
            kind = "periodic"
        return OperatingPointResult(at, kind, span, error, midway, 3)
    except (ValueError, FloatingPointError, ArithmeticError, np.linalg.LinAlgError) as exc:
        return fail(str(exc))
