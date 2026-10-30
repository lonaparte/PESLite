"""Building blocks of the controller: transforms, dynamic blocks, timers and helpers.

Space vectors are peak-value, ``u = (2/3)(u_a + a u_b + a^2 u_c)`` with ``a = exp(j 2 pi/3)``, the
zero sequence dropped; ``u_dq = u_ab * exp(-j theta)``. Filters accept real or complex signals;
filters and timers expose named states.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..solver.model import ConfigError, _real

__all__ = ["A120", "abc2complex", "complex2abc", "phases", "complex_power",
           "unit_delay", "clamp", "smoothstep", "peak_abs",
           "Integrator", "PI", "OutputLimit", "Filter", "HoldTimer", "MovingWindow"]


# ------------------------------------------------------------------ space vectors

A120 = cmath.exp(2j * math.pi / 3.0)  # 120-degree rotation
_A240 = A120 * A120
_A120_CONJ = A120.conjugate()


def abc2complex(u_abc) -> complex:
    """Convert phase quantities to a peak-scaled space vector (zero sequence dropped)."""
    u_a = float(u_abc[0])
    u_b = float(u_abc[1])
    u_c = float(u_abc[2])
    return (2.0 / 3.0) * (u_a + A120 * u_b + _A240 * u_c)


def phases(u: complex) -> tuple[float, float, float]:
    """Convert a space vector to three phase quantities (floats)."""
    return u.real, (u * _A120_CONJ).real, (u * A120).real


def complex2abc(u: complex) -> NDArray[np.float64]:
    """Convert a space vector to a phase-quantity array."""
    return np.array(phases(u))


def complex_power(voltage, current):
    """Return ``p + j q`` from voltage and current space vectors."""
    return voltage * current.conjugate()


def unit_delay(value, previous):
    """Return ``(previous output, next state)`` for a one-sample delay."""
    return previous, value


# ------------------------------------------------------------------ scalar helpers

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


# ------------------------------------------------------------------ dynamic control blocks

class Integrator:
    """Integrator with one user-facing call and construction-time execution binding.

    ``period=None`` selects the continuous implementation: calling the block returns its present
    state and records the supplied derivative for the ODE solver.  A positive ``period`` selects
    the sampled implementation, discretised by the trapezoidal rule.  Mode selection therefore
    adds no branch to the block's hot path.
    """

    def __new__(cls, period: float | None, initial=0.0):
        if cls is Integrator:
            impl = _ContinuousIntegrator if period is None else _SampledIntegrator
            return object.__new__(impl)
        return object.__new__(cls)

    def __init__(self, period: float | None, initial=0.0) -> None:
        if period is not None and (not math.isfinite(period) or period <= 0.0):
            raise ValueError(f"integrator period must be finite and positive, got {period}")
        self.T = None if period is None else _real(period)
        self.initial = initial
        self.value = initial
        self.derivative = initial * 0

    def __call__(self, derivative):
        raise NotImplementedError

    def reset(self) -> None:
        self.value = self.initial
        self.derivative = self.initial * 0

    def get_state(self) -> dict[str, Any]:
        return {"": self.value}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if "" in values:
            self.value = values[""]

    def state_derivatives(self) -> dict[str, Any]:
        return {"": self.derivative}

    def state_bindings(self):
        return {"": (self, "value")}


class _SampledIntegrator(Integrator):
    """Tustin realisation of ``1/s`` with one stored state."""

    def __call__(self, derivative):
        half_step = 0.5 * self.T * derivative
        output = self.value + half_step
        self.value = output + half_step
        return output


class _ContinuousIntegrator(Integrator):
    def __call__(self, derivative):
        self.derivative = derivative
        return self.value

class PI:
    """PI law with sampled and continuous implementations behind one call.

    The sampled integral branch is the Tustin realisation of ``1/s``.  The continuous branch
    records ``r - fb`` as its state derivative.  Output constraints are owned by the controller;
    tracking anti-windup (``antiwindup``) remains local to the PI state in either branch.
    """

    def __new__(cls, kp: float, ki: float, period: float | None, *,
                antiwindup: bool = True, initial=0.0):
        if cls is PI:
            impl = _ContinuousPI if period is None else _SampledPI
            return object.__new__(impl)
        return object.__new__(cls)

    def __init__(self, kp: float, ki: float, period: float | None, *,
                 antiwindup: bool = True, initial=0.0) -> None:
        if period is not None and (not math.isfinite(period) or period <= 0.0):
            raise ValueError(f"PI period must be finite and positive, got {period}")
        self.kp, self.ki = _real(kp), _real(ki)
        self.T = None if period is None else _real(period)
        self.antiwindup_enabled = bool(antiwindup)
        self.initial = initial
        self.integral = initial
        self.derivative = initial * 0

    def __call__(self, r, fb, ff=0.0):
        raise NotImplementedError

    def reset(self) -> None:
        self.integral = self.initial
        self.derivative = self.initial * 0

    def get_state(self) -> dict[str, Any]:
        return {"": self.integral}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if "" in values:
            self.integral = values[""]

    def state_derivatives(self) -> dict[str, Any]:
        return {"": self.derivative}

    def state_bindings(self):
        return {"": (self, "integral")}


class _SampledPI(PI):
    def __call__(self, r, fb, ff=0.0):
        error = r - fb
        half_step = 0.5 * self.T * error
        integrated = self.integral + half_step
        self.integral = integrated + half_step
        return ff + self.kp * error + self.ki * integrated

    def antiwindup(self, intended, output) -> None:
        """Track a constrained output by correcting the integral."""
        if self.antiwindup_enabled:
            self.integral += self.T * (output - intended) / self.kp


class _ContinuousPI(PI):
    def __call__(self, r, fb, ff=0.0):
        error = r - fb
        self.derivative = error
        return ff + self.kp * error + self.ki * self.integral

    def antiwindup(self, intended, output) -> None:
        """Track a constrained output by correcting the integral's derivative."""
        if not self.antiwindup_enabled or output == intended:
            return
        self.derivative += (output - intended) / self.kp


class OutputLimit:
    """Scalar or magnitude constraint for sampled and continuous loops."""

    def __init__(self, limit, *, lower=None, pi=None, active=None):
        self.pi = pi
        self.limit = _real(limit)
        self.lower = lower
        self.active = active
        self.saturated = False
        if not math.isfinite(self.limit) or self.limit <= 0.0:
            raise ConfigError("a controller output limit must be finite and positive")
        if lower is not None and (not math.isfinite(lower) or lower >= self.limit):
            raise ConfigError("a controller output lower limit must be finite and below its upper limit")
        if pi is not None and pi.antiwindup_enabled and pi.kp == 0.0:
            raise ConfigError("tracking anti-windup requires a nonzero proportional gain")

    def _apply(self, intended):
        if self.lower is not None:
            return min(self.limit, max(self.lower, intended))
        magnitude = abs(intended)
        return intended * (self.limit / magnitude) if magnitude > self.limit else intended

    def sample(self, t, intended):
        """Constrain ``intended`` at ``t``; an active PI tracks a saturated output in either mode."""
        output = self._apply(intended)
        self.saturated = output != intended
        if self.saturated and self.pi is not None and (self.active is None or self.active()):
            self.pi.antiwindup(intended, output)
        return output


def _coefficients(values, name: str) -> tuple[float, ...]:
    try:
        result = tuple(_real(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"filter {name} must be a finite coefficient sequence") from exc
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"filter {name} must be a non-empty finite coefficient sequence")
    return result


def _filter_order(den) -> int:
    coefficients = _coefficients(den, "denominator")
    if coefficients[0] == 0.0:
        raise ValueError("filter denominator leading coefficient must be nonzero")
    return len(coefficients) - 1


class Filter:
    r"""Continuous transfer function with a mode-specific runtime implementation.

    ``num`` and ``den`` contain coefficients in descending powers of :math:`s`, matching the
    Simulink Transfer Fcn convention.  The transfer function must be proper.  ``period=None``
    exposes its observable-canonical continuous state and derivatives; a positive ``period`` constructs a
    discrete Tustin (bilinear/trapezoidal) realisation once and runs only that recurrence.

    Initial values are the states of the observable canonical realisation.  As with a transfer
    function block, the default is zero state.  A scalar initial value is accepted for first-order
    filters; higher-order filters require one value per denominator order.
    """

    def __new__(cls, num, den, period: float | None, *, initial=0.0):
        if cls is Filter:
            order = _filter_order(den)
            if period is None:
                impl = (_ContinuousGain if order == 0 else
                        _ContinuousFilter1 if order == 1 else _ContinuousFilterN)
            else:
                impl = (_SampledGain if order == 0 else
                        _SampledFilter1 if order == 1 else _SampledFilterN)
            return object.__new__(impl)
        return object.__new__(cls)

    def __init__(self, num, den, period: float | None, *, initial=0.0) -> None:
        numerator = _coefficients(num, "numerator")
        denominator = _coefficients(den, "denominator")
        if denominator[0] == 0.0:
            raise ValueError("filter denominator leading coefficient must be nonzero")
        while len(numerator) > 1 and numerator[0] == 0.0:
            numerator = numerator[1:]
        order = len(denominator) - 1
        if len(numerator) - 1 > order:
            raise ValueError(
                "filter must be proper: denominator order must be at least numerator order"
            )
        if period is not None and (not math.isfinite(period) or period <= 0.0):
            raise ValueError(f"filter period must be finite and positive, got {period}")

        scale = denominator[0]
        den_n = tuple(value / scale for value in denominator)
        num_n = (0.0,) * (order + 1 - len(numerator)) + tuple(
            value / scale for value in numerator
        )
        self.num, self.den = numerator, denominator
        self.order = order
        self.T = None if period is None else _real(period)
        self.derivative = ()

        if order == 0:
            self.gain = num_n[0]
            return

        # Observable canonical form.  It makes the state of a first-order low-pass filter equal
        # to its output, while remaining a general proper transfer-function realisation.
        if order == 1:
            a, d = -den_n[1], num_n[0]
            b = num_n[1] + a * d
            values = (initial,) if not isinstance(initial, (tuple, list, np.ndarray)) else tuple(initial)
            if len(values) != 1:
                raise ValueError("first-order filter initial state must contain one value")
            self.x0 = values[0]
            self._initial = values
            if self.T is None:
                self._a, self._b, self._c, self._d = a, b, 1.0, d
            else:
                h = 0.5 * self.T
                inv = 1.0 / (1.0 - h * a)
                self._ad = inv * (1.0 + h * a)
                self._bd = inv * self.T * b
                self._cd = inv
                self._dd = d + h * inv * b
            return

        a = np.asarray(den_n[1:], dtype=float)
        b0 = num_n[0]
        B = np.asarray(num_n[1:], dtype=float) - a * b0
        A = np.zeros((order, order), dtype=float)
        A[:, 0] = -a
        A[:-1, 1:] += np.eye(order - 1)
        C = np.zeros(order, dtype=float)
        C[0] = 1.0
        D = b0

        if isinstance(initial, (tuple, list, np.ndarray)):
            values = tuple(initial)
        elif initial == 0 or initial == 0.0 or initial == 0j:
            values = (initial,) * order
        else:
            raise ValueError(f"order-{order} filter initial state needs {order} values")
        if len(values) != order:
            raise ValueError(f"order-{order} filter initial state needs {order} values")
        self._initial = values
        self._x = np.asarray(values)
        self._A, self._B, self._C, self._D = A, B, C, D
        if self.T is not None:
            identity = np.eye(order)
            inverse = np.linalg.solve(identity - 0.5 * self.T * A, identity)
            self._Ad = inverse @ (identity + 0.5 * self.T * A)
            self._Bd = inverse @ (self.T * B)
            self._Cd = C @ inverse
            self._Dd = D + 0.5 * self.T * (C @ inverse @ B)

    def __call__(self, value):
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        if self.order == 0:
            return {}
        if self.order == 1:
            return {"": self.x0}
        return {str(index): value.item() if hasattr(value, "item") else value
                for index, value in enumerate(self._x)}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if self.order == 1:
            if "" in values:
                self.x0 = values[""]
            return
        if any(isinstance(values.get(str(index)), complex) for index in range(self.order)):
            self._x = self._x.astype(complex)
        for index in range(self.order):
            key = str(index)
            if key in values:
                self._x[index] = values[key]

    def state_derivatives(self) -> dict[str, Any]:
        if self.order == 0:
            return {}
        if self.order == 1:
            return {"": self.derivative[0]}
        return {str(index): value for index, value in enumerate(self.derivative)}

    def state_bindings(self):
        if self.order == 0:
            return {}
        if self.order == 1:
            return {"": (self, "x0")}
        # Higher-order filters keep a compact ndarray. The graph falls back to set_state() when
        # binding individual array entries into the packed ODE state.
        return None

    def reset(self) -> None:
        if self.order == 1:
            self.x0 = self._initial[0]
        elif self.order > 1:
            self._x = np.asarray(self._initial)
        self.derivative = ()


class _SampledGain(Filter):
    def __call__(self, value):
        return self.gain * value


class _ContinuousGain(_SampledGain):
    pass


class _SampledFilter1(Filter):
    def __call__(self, value):
        output = self._cd * self.x0 + self._dd * value
        self.x0 = self._ad * self.x0 + self._bd * value
        return output


class _ContinuousFilter1(Filter):
    def __call__(self, value):
        output = self._c * self.x0 + self._d * value
        self.derivative = (self._a * self.x0 + self._b * value,)
        return output


class _SampledFilterN(Filter):
    def __call__(self, value):
        output = self._Cd @ self._x + self._Dd * value
        self._x = self._Ad @ self._x + self._Bd * value
        return output.item() if hasattr(output, "item") else output


class _ContinuousFilterN(Filter):
    def __call__(self, value):
        output = self._C @ self._x + self._D * value
        self.derivative = tuple(self._A @ self._x + self._B * value)
        return output.item() if hasattr(output, "item") else output


# ------------------------------------------------------------------ timers


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
