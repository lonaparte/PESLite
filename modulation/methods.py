"""PWM methods mapping a voltage reference to three-phase modulating signals.

Interface: ``method(u_ab, u_dc) -> m_abc`` with ``u_ab`` (complex, V) and ``u_dc`` (V).
Custom methods with this signature can be passed as ``pwm_method=``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..firmware.transforms import phases

__all__ = ["PWM_METHODS", "make_pwm_method", "spwm", "svpwm"]


def spwm(u_ab: complex, u_dc: float) -> NDArray[np.float64]:
    """Sinusoidal PWM: return ``2 * u_abc / u_dc`` (no zero sequence)."""
    a, b, c = phases(u_ab)
    return np.array([2.0 * a / u_dc, 2.0 * b / u_dc, 2.0 * c / u_dc])


def svpwm(u_ab: complex, u_dc: float) -> NDArray[np.float64]:
    """Continuous space-vector PWM: SPWM signals with min-max zero-sequence injection."""
    m = spwm(u_ab, u_dc)
    return m - 0.5 * (float(m.max()) + float(m.min()))


PWM_METHODS = {"spwm": spwm, "svpwm": svpwm}


def make_pwm_method(name: str):
    """Return the built-in PWM method ``name`` (``"spwm"`` or ``"svpwm"``)."""
    try:
        return PWM_METHODS[name]
    except KeyError:
        raise KeyError(f"unknown pwm.method {name!r}; known: {sorted(PWM_METHODS)}") from None
