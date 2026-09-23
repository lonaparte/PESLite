"""Space-vector transforms.

Conventions: peak-value space vectors ``u = (2/3)(u_a + a u_b + a^2 u_c)``, ``a = exp(j 2 pi/3)``,
zero sequence dropped; ``u_dq = u_ab * exp(-j theta)``; ``S = 1.5 * u * conj(i)``.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["A120", "abc2complex", "complex2abc", "phases", "rotate", "power"]


A120 = cmath.exp(2j * math.pi / 3.0)  # 120-degree rotation
_A240 = A120 * A120
_A120_CONJ = A120.conjugate()


def abc2complex(u_abc) -> complex:
    """Convert phase quantities to a peak-scaled space vector (zero sequence dropped)."""
    return (2.0 / 3.0) * (u_abc[0] + A120 * u_abc[1] + _A240 * u_abc[2])

def phases(u: complex) -> tuple[float, float, float]:
    """Convert a space vector to three phase quantities (floats)."""
    return u.real, (u * _A120_CONJ).real, (u * A120).real

def complex2abc(u: complex) -> NDArray[np.float64]:
    """Convert a space vector to a phase-quantity array."""
    return np.array(phases(u))

def rotate(u: complex, theta: float) -> complex:
    """Rotate ``u`` by ``theta`` radians (``theta > 0`` counter-clockwise)."""
    return u * cmath.exp(1j * theta)

def power(u: complex, i: complex) -> tuple[float, float]:
    """Return active and reactive power ``(p, q)`` of peak-scaled space vectors."""
    s = 1.5 * u * i.conjugate()
    return s.real, s.imag
