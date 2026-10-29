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

    def due(self, t: float, index: int) -> bool:
        return abs(t - self.at(index)) < TIME_EPS


def resolve_clock_offset(period: float, offset: float | None, default: float = 0.0) -> float:
    """Return the canonical offset within one period, inheriting ``default`` when omitted."""
    return default % period if offset is None else offset


def control_clock_offset(control, pwm) -> float:
    """Resolve a controller's clock position from the ordinary control and PWM fields."""
    period = pwm.switching_period if control.period is None else control.period
    return resolve_clock_offset(period, control.offset, pwm.grid_offset)
