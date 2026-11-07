"""Shared periodic-event grids used by solvers, controllers and hardware models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TIME_EPS = 1e-10


@dataclass(frozen=True)
class PeriodicClock:
    """One periodic event grid described by its period and absolute-time offset."""

    period: float
    offset: float = 0.0

    def at(self, index: int) -> float:
        return self.offset + index * self.period

    def index_after(self, t: float):
        """Index of the first event strictly after ``t`` within :data:`TIME_EPS`."""
        return np.floor((t - self.offset + TIME_EPS) / self.period) + 1

    def index_at_or_after(self, t: float):
        """Index of the first event at or after ``t`` within :data:`TIME_EPS`."""
        return np.ceil((t - self.offset - TIME_EPS) / self.period)

    def start_index(self, t: float, continued: bool):
        """Index at which a fresh or continued run starts processing this clock."""
        return self.index_after(t) if continued else self.index_at_or_after(t)

    def due(self, t: float, index: int) -> bool:
        return abs(t - self.at(index)) < TIME_EPS


def resolve_clock_offset(period: float, offset: float | None, default: float = 0.0) -> float:
    """Return the canonical offset within one period, inheriting ``default`` when omitted."""
    return default % period if offset is None else offset


def control_period(control, pwm) -> float:
    """Return the configured control period or its carrier-period default."""
    return pwm.switching_period if control.period is None else control.period


def control_computation(control) -> float:
    """Return the configured computation delay or its sampled-control default."""
    return 1.0e-6 if control.computation is None else control.computation


def control_clock_offset(control, pwm, *, pwm_timed: bool = True) -> float:
    """Resolve a controller's clock position from the ordinary control and PWM fields."""
    period = control_period(control, pwm)
    return resolve_clock_offset(period, control.offset, pwm.grid_offset if pwm_timed else 0.0)


def control_actuation_clock(control, pwm, *, pwm_timed: bool = True) -> PeriodicClock:
    """Return the grid on which a sampled command becomes electrically active."""
    if pwm_timed:
        return PeriodicClock(pwm.load_period, pwm.grid_offset)
    period = control_period(control, pwm)
    offset = (control_clock_offset(control, pwm, pwm_timed=False)
              + control_computation(control))
    return PeriodicClock(period, offset % period)
