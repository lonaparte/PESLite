"""Accepted-step histories for delayed physical relations."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any

from .timing import TIME_EPS


class DelayHistory:
    """Piecewise-linear history over one fixed delay.

    Values are committed only after a solver step is accepted. Checkpoints make the same history
    safe for rejected steps and event-location retries. Samples older than the interpolation
    bracket of the next delayed read are discarded.
    """

    def __init__(self, delay: float, initial: complex | float) -> None:
        if not math.isfinite(delay) or delay <= 0.0:
            raise ValueError(f"delay must be finite and > 0, got {delay}")
        self.delay = float(delay)
        self.initial = initial
        self.times: list[float] = []
        self.values: list[complex | float] = []

    def __call__(self, t: float) -> complex | float:
        """Return the linearly interpolated value at ``t``."""
        if not self.times or t < self.times[0]:
            return self.initial
        after = bisect.bisect_right(self.times, t)
        if after == len(self.times):
            return self.values[-1]
        before = after - 1
        t0, t1 = self.times[before], self.times[after]
        fraction = (t - t0) / (t1 - t0)
        return self.values[before] + fraction * (self.values[after] - self.values[before])

    def accept(self, t: float, value: complex | float) -> None:
        """Commit one chronological sample, replacing a sample at the same instant."""
        if self.times and t < self.times[-1] - TIME_EPS:
            raise RuntimeError("delay-history samples must be chronological")
        if self.times and abs(t - self.times[-1]) <= TIME_EPS:
            self.times[-1] = float(t)
            self.values[-1] = value
        else:
            self.times.append(float(t))
            self.values.append(value)
        cutoff = t - self.delay
        while len(self.times) > 2 and self.times[1] < cutoff - TIME_EPS:
            del self.times[0]
            del self.values[0]

    def derivative(self, t: float) -> complex | float:
        """Return the slope of the committed interpolation at ``t``.

        At a committed knot the left slope is used, so a method-of-steps evaluation never reads
        an uncommitted value beyond the current history boundary.
        """
        if len(self.times) < 2 or not self.times[0] < t <= self.times[-1]:
            return 0.0
        after = bisect.bisect_left(self.times, t)
        before = after - 1
        return ((self.values[after] - self.values[before])
                / (self.times[after] - self.times[before]))

    def checkpoint(self) -> tuple[tuple[float, ...], tuple[complex | float, ...]]:
        """Return a restorable immutable snapshot."""
        return tuple(self.times), tuple(self.values)

    def restore(self, checkpoint: tuple[tuple[float, ...], tuple[complex | float, ...]]) -> None:
        """Restore a snapshot made by :meth:`checkpoint`."""
        times, values = checkpoint
        self.times[:] = times
        self.values[:] = values

    def integral_squared(self, stop: float) -> float:
        """Integrate ``abs(value)**2`` over ``[stop - delay, stop]``.

        The history is linear between samples, so each segment is integrated analytically.
        """
        start = stop - self.delay
        knots = [start]
        knots.extend(t for t in self.times if start < t < stop)
        knots.append(stop)
        total = 0.0
        for t0, t1 in zip(knots, knots[1:]):
            a, b = self(t0), self(t1)
            cross = (a * complex(b).conjugate()).real
            total += (t1 - t0) * (abs(a) ** 2 + abs(b) ** 2 + cross) / 3.0
        return total


@dataclass(frozen=True)
class AcceptedSignal:
    """A record field committed to a history after each accepted physical state."""

    history: DelayHistory
    record: Any
    field: str

    def value(self) -> complex | float:
        return getattr(self.record, self.field)


@dataclass(frozen=True)
class DelayedSignal:
    """A scaled delayed-history source shared by Python and native equation lowering."""

    history: DelayHistory
    scale: complex | float = 1.0

    def __call__(self, t: float) -> complex | float:
        return self.scale * self.history(t - self.history.delay)

    def derivative(self, t: float) -> complex | float:
        """Return the derivative of the delayed committed interpolation."""
        return self.scale * self.history.derivative(t - self.history.delay)


__all__ = ["AcceptedSignal", "DelayedSignal", "DelayHistory"]
