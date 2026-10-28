"""The output stage of the controller: from the dq voltage command to the duty ratios of the three phases.

The command is rotated into alpha-beta, turned into modulating signals by the PWM method (``spwm``,
``svpwm`` or a custom one), and limited. Controller-level command constraints and PI anti-windup
are handled before this final output stage.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from .blocks import peak_abs, phases

__all__ = ["PWMMethod", "SignalLimiter", "PWM_METHODS", "spwm", "svpwm", "ModulationLimiter", "OutputStage"]


class PWMMethod(Protocol):
    """Map an alpha-beta voltage command (V) and a positive dc voltage (V) to modulating signals ``m_abc``.

    Must be deterministic and return three finite real values, shape ``(3,)``, possibly outside [-1, 1].
    """

    def __call__(self, u_ab: complex, u_dc: float) -> NDArray[np.float64]: ...


class SignalLimiter(Protocol):
    """Map three modulating signals to three finite limited signals, shape ``(3,)``.

    Must be deterministic; it may be called on tentative commands and keeps no saturation state.
    """

    def __call__(self, m_abc: NDArray[np.float64]) -> NDArray[np.float64]: ...


def spwm(u_ab: complex, u_dc: float) -> NDArray[np.float64]:
    """Sinusoidal PWM: return ``2 * u_abc / u_dc`` (no zero sequence)."""
    a, b, c = phases(u_ab)
    return np.array([2.0 * a / u_dc, 2.0 * b / u_dc, 2.0 * c / u_dc])


def svpwm(u_ab: complex, u_dc: float) -> NDArray[np.float64]:
    """Continuous space-vector PWM: SPWM signals with min-max zero-sequence injection."""
    m = spwm(u_ab, u_dc)
    maximum = np.maximum(m[0], np.maximum(m[1], m[2]))
    minimum = np.minimum(m[0], np.minimum(m[1], m[2]))
    return m - 0.5 * (maximum + minimum)


PWM_METHODS = {"spwm": spwm, "svpwm": svpwm}


class ModulationLimiter:
    """Return a copy of ``m_abc`` scaled uniformly so that ``max(abs(m_abc)) <= limit``."""

    def __init__(self, limit: float) -> None:
        self.limit = limit

    def __call__(self, m_abc: NDArray[np.float64]) -> NDArray[np.float64]:
        m = np.array(m_abc, dtype=float, copy=True)
        max_abs = peak_abs(m)
        if max_abs > self.limit:
            m *= self.limit / max_abs
        return m


class _ConfiguredLimiter:
    """Distinguish an omitted limiter (use the configuration) from explicit None."""


CONFIGURED = _ConfiguredLimiter()


def _signals(value: Any, component: str) -> np.ndarray:
    """``value`` as three finite real modulating signals, or a ValueError naming ``component``."""
    message = f"{component} must return three finite real modulating signals with shape (3,)"
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError
        m = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if m.shape != (3,) or not all(map(math.isfinite, m.tolist())):
        raise ValueError(message)
    return m


class OutputStage:
    """Duty ratios from the dq voltage command of a unit's controller.

    pwm_method: ``None`` uses ``cfg.pwm.method``.
    limiter: omitted uses ``ModulationLimiter(cfg.pwm.modulation_limit)``; ``None`` disables the limit
    and its anti-windup.
    """

    def __init__(self, cfg: Any, pwm_method: Optional[PWMMethod] = None, *,
                 limiter: SignalLimiter | None | _ConfiguredLimiter = CONFIGURED) -> None:
        self.v_base, self.v_dc_base = cfg.base.v_phase_peak, cfg.dc_base.v
        self.pwm_method = PWM_METHODS[cfg.pwm.method] if pwm_method is None else pwm_method
        self.limiter = ModulationLimiter(cfg.pwm.modulation_limit) if limiter is CONFIGURED else limiter
        if not callable(self.pwm_method):
            raise TypeError("pwm_method must be callable")
        if self.limiter is not None and not callable(self.limiter):
            raise TypeError("limiter must be callable or None")
        self._trusted_pwm = any(self.pwm_method is method for method in PWM_METHODS.values())
        self._standard_limiter = type(self.limiter) is ModulationLimiter
        self.n_updates = 0
        self.n_saturated = 0
        self.first_saturation_t: float | None = None
        self.saturated = False
        self.m_abc = np.zeros(3)
        self._command_saturated = False

    def command_limit(self, u_dc_pu: float) -> float | None:
        """Circular dq-command limit for the configured standard PWM, otherwise ``None``."""
        if not self._trusted_pwm or not self._standard_limiter:
            return None
        factor = 1.0 / math.sqrt(3.0) if self.pwm_method is svpwm else 0.5
        return self.limiter.limit * max(0.0, u_dc_pu) * self.v_dc_base / self.v_base * factor

    def equation(self, command: complex, rot: complex, u_dc: float) -> tuple[tuple[Any, ...], bool]:
        """Pure output-stage equation shared by Python execution and code generation."""
        u_ab = command * rot * self.v_base
        if u_dc > 0.0:
            raw = self.pwm_method(u_ab, u_dc)
            m = tuple(raw if self._trusted_pwm else _signals(raw, "pwm_method"))
        else:
            u = phases(u_ab)
            peak = np.maximum(abs(u[0]), np.maximum(abs(u[1]), abs(u[2])))
            m = tuple(value / peak for value in u) if peak > 0.0 else (0.0, 0.0, 0.0)
        saturated = False
        if self._standard_limiter:
            max_abs = np.maximum(abs(m[0]), np.maximum(abs(m[1]), abs(m[2])))
            if max_abs > self.limiter.limit:
                scale = self.limiter.limit / max_abs
                m = tuple(value * scale for value in m)
                saturated = True
            elif u_dc <= 0.0 and u_ab != 0j:
                saturated = True
        elif self.limiter is not None:
            original = np.asarray(m, dtype=float)
            limited = _signals(self.limiter(original.copy()), "limiter")
            saturated = limited.tolist() != original.tolist() or (u_dc <= 0.0 and u_ab != 0j)
            m = tuple(limited)
        return m, saturated

    def modulate(self, t: float, u_cmd_dq: complex, theta: float, u_dc: float, *,
                 count: bool = True) -> np.ndarray:
        """Convert a dq voltage command to duty ratios.

        ``u_cmd_dq`` is in pu on the AC base; ``theta`` is in rad and ``u_dc`` is in pu on the
        DC base. ``count=False`` leaves the controller-update and saturation counters unchanged.
        """
        rot = np.exp(1j * theta)
        u_dc = u_dc * self.v_dc_base  # PWM receives DC volts; loop feedback stays pu
        values, saturated = self.equation(u_cmd_dq, rot, u_dc)
        saturated = saturated or self._command_saturated
        self.m_abc, self.saturated = np.asarray(values, dtype=float), saturated
        duty = 0.5 * (1.0 + self.m_abc)
        if count:
            self.n_updates += 1
            if saturated:
                self.n_saturated += 1
                if self.first_saturation_t is None:
                    self.first_saturation_t = t
        return duty
