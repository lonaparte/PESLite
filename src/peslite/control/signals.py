"""Deterministic time tables used by controller reference inputs."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

from ..solver.model import ConfigError
from ..solver.timing import TIME_EPS


def _finite_sequence(values: Any, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ConfigError(f"{name} must be a sequence of finite numbers")
    try:
        items = tuple(values)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a sequence of finite numbers") from None
    if any(isinstance(value, bool) for value in items):
        raise ConfigError(f"{name} must be a sequence of finite numbers")
    try:
        result = tuple(float(value) for value in items)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a sequence of finite numbers") from None
    if not result or not all(math.isfinite(value) for value in result):
        raise ConfigError(f"{name} must be a nonempty sequence of finite numbers")
    return result


@dataclass(frozen=True)
class TableSignal:
    """A scalar signal defined on absolute simulation time.

    Linear and zero-order interpolation need no numerical package. Cubic and PCHIP interpolation
    use SciPy once during construction and retain only their piecewise-polynomial coefficients.
    """

    time: tuple[float, ...]
    value: tuple[float, ...]
    repeat: bool = False
    interpolation: str = "linear"
    extrapolation: str = "hold"
    type: str = "table"
    _coefficients: tuple[tuple[float, ...], ...] = field(
        init=False, default=(), repr=False, compare=False,
    )
    _left_slope: float = field(init=False, default=0.0, repr=False, compare=False)
    _right_slope: float = field(init=False, default=0.0, repr=False, compare=False)

    def __post_init__(self) -> None:
        time = _finite_sequence(self.time, "time")
        value = _finite_sequence(self.value, "value")
        if len(time) != len(value):
            raise ConfigError("time and value must have the same length")
        if any(right <= left for left, right in zip(time, time[1:])):
            raise ConfigError("time must be strictly increasing")
        if self.type != "table":
            raise ConfigError("type must be 'table'")
        if not isinstance(self.repeat, bool):
            raise ConfigError("repeat must be true or false")
        if self.interpolation not in ("zero_order_hold", "linear", "cubic", "pchip"):
            raise ConfigError(
                "interpolation must be 'zero_order_hold', 'linear', 'cubic' or 'pchip'"
            )
        if self.extrapolation not in ("hold", "zero", "linear", "error"):
            raise ConfigError("extrapolation must be 'hold', 'zero', 'linear' or 'error'")
        if self.repeat and len(time) < 2:
            raise ConfigError("repeat requires at least two time points")
        if self.interpolation == "cubic" and len(time) < 3:
            raise ConfigError("cubic interpolation requires at least three time points")
        if self.interpolation == "pchip" and len(time) < 2:
            raise ConfigError("PCHIP interpolation requires at least two time points")

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "value", value)
        if len(time) == 1:
            return

        if self.interpolation in ("cubic", "pchip"):
            try:
                from scipy.interpolate import CubicSpline, PchipInterpolator
            except ImportError as exc:  # pragma: no cover - SciPy is a core dependency
                raise ConfigError(
                    f"{self.interpolation} interpolation requires SciPy"
                ) from exc
            polynomial = (CubicSpline(time, value, extrapolate=False)
                          if self.interpolation == "cubic"
                          else PchipInterpolator(time, value, extrapolate=False))
            coefficients = tuple(
                tuple(float(item) for item in polynomial.c[:, index])
                for index in range(len(time) - 1)
            )
            left_slope = float(polynomial(time[0], 1))
            right_slope = float(polynomial(time[-1], 1))
        else:
            slopes = tuple(
                (right - left) / (t_right - t_left)
                for t_left, t_right, left, right in zip(time, time[1:], value, value[1:])
            )
            coefficients = tuple((slope, left) for slope, left in zip(slopes, value))
            left_slope, right_slope = slopes[0], slopes[-1]
        object.__setattr__(self, "_coefficients", coefficients)
        object.__setattr__(self, "_left_slope", left_slope)
        object.__setattr__(self, "_right_slope", right_slope)

    def _aligned_position(self, t: float) -> tuple[float, int]:
        """Return an interval position after applying the event-grid time tolerance."""
        position = bisect_right(self.time, t)
        if position < len(self.time) and self.time[position] - t < TIME_EPS:
            return self.time[position], position + 1
        if position and t - self.time[position - 1] < TIME_EPS:
            return self.time[position - 1], position
        return t, position

    def _outside(self, t: float, *, left: bool) -> float:
        if self.extrapolation == "hold":
            return self.value[0 if left else -1]
        if self.extrapolation == "zero":
            return 0.0
        if self.extrapolation == "linear":
            edge = 0 if left else -1
            slope = self._left_slope if left else self._right_slope
            return self.value[edge] + slope * (t - self.time[edge])
        side = "before" if left else "after"
        raise ValueError(f"time {t:.12g} is {side} the table interval")

    def __call__(self, t: float) -> float:
        """Evaluate the table at absolute time ``t``."""
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("signal time must be finite")
        t, position = self._aligned_position(t)
        first, last = self.time[0], self.time[-1]
        if t < first:
            return self._outside(t, left=True)
        if t > last or (self.repeat and t == last):
            if not self.repeat:
                return self._outside(t, left=False)
            t, position = self._aligned_position(
                first + math.fmod(t - first, last - first)
            )
        if len(self.time) == 1 or t == last:
            return self.value[-1]
        index = position - 1
        if self.interpolation == "zero_order_hold":
            return self.value[index]
        x = t - self.time[index]
        result = 0.0
        for coefficient in self._coefficients[index]:
            result = result * x + coefficient
        return result


@dataclass(frozen=True)
class ReferenceBinding:
    """One controller-reference binding with its ordinary SI/pu conversion."""

    name: str
    signal: TableSignal
    input_name: str
    scale: float = 1.0
    integer: bool = False

    def __call__(self, t: float) -> float | int:
        value = self.scale * self.signal(t)
        return int(round(value)) if self.integer else value


__all__ = ["TableSignal"]
