"""Discrete-time filters, timers, a moving window and scalar helpers.

Filters accept real or complex signals; filters and timers expose named states.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

__all__ = ["clamp", "smoothstep", "peak_abs", "LowPass1", "HighPass1", "HoldTimer", "MovingWindow"]


class LowPass1:
    """First-order low-pass filter: ``y += (1 - exp(-2*pi*bw_hz*T))*(x - y)``.

    ``bw_hz`` in Hz (``<= 0`` passes the input through), ``T`` in s. ``init_on_first``
    copies the first sample to the output; ``y0`` is the initial output.
    Named state: ``y``, NaN while not yet seeded.
    """

    def __init__(self, bw_hz: float, T: float, y0=0.0, init_on_first: bool = True) -> None:
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * bw_hz * T) if bw_hz > 0.0 else 1.0
        self.y = y0
        self._y0 = y0
        self._init_on_first = init_on_first
        self._first = init_on_first

    def update(self, x):
        if self._first:
            self.y = x
            self._first = False
        else:
            self.y = self.y + self.alpha * (x - self.y)
        return self.y

    @property
    def seeded(self) -> bool:
        """Whether the output has been initialised from a sample."""
        return not self._first

    def get_state(self) -> dict[str, Any]:
        if self._first:
            return {"": complex(math.nan, math.nan) if isinstance(self._y0, complex) else math.nan}
        return {"": self.y}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if "" not in values:
            return
        y = values[""]
        if (y.real != y.real) or (isinstance(y, complex) and y.imag != y.imag):  # nan: not seeded
            self.y, self._first = self._y0, self._init_on_first
        else:
            self.y, self._first = y, False


class HighPass1:
    """First-order high-pass filter ``y = x - LowPass1(x)`` with corner ``bw_hz`` (Hz)."""

    def __init__(self, bw_hz: float, T: float, init_on_first: bool = True, y0=0.0) -> None:
        self.lpf = LowPass1(bw_hz, T, y0, init_on_first=init_on_first)

    def update(self, x):
        return x - self.lpf.update(x)

    def get_state(self) -> dict[str, Any]:
        return self.lpf.get_state()  # the low-passed signal

    def set_state(self, values: Mapping[str, Any]) -> None:
        self.lpf.set_state(values)


class HoldTimer:
    """Return how long (s) a condition has held continuously; resets when it clears."""

    def __init__(self, T: float) -> None:
        self.T = T
        self.held = 0.0

    def update(self, condition: bool) -> float:
        self.held = self.held + self.T if condition else 0.0
        return self.held

    def get_state(self) -> dict[str, Any]:
        return {"": self.held}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if "" in values:
            self.held = float(values[""])


class MovingWindow:
    """Ring buffer whose ``push`` returns the sample ``n`` steps ago (``None`` until full)."""

    def __init__(self, n: int) -> None:
        self.n = max(1, n)
        self.buf = [0.0] * self.n
        self.idx = 0
        self.full = False

    def push(self, x: float):
        old = self.buf[self.idx] if self.full else None
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.n
        if self.idx == 0:
            self.full = True
        return old


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def smoothstep(x: float) -> float:
    """Smooth ramp from 0 to 1 on [0, 1], constant outside."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)

def peak_abs(values) -> float:
    """Return the largest absolute value in ``values`` (NaN if any value is NaN)."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    it = iter(values)
    peak = abs(next(it))
    for v in it:
        a = abs(v)
        if a > peak or a != a:
            peak = a
    return float(peak)
