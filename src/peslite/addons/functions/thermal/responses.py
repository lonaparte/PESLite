"""Analytical response helpers for the thermal addon."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...components.thermalnetwork import FosterImpedance, _finite

__all__ = ["foster_step_response"]


def foster_step_response(
    time_s: Sequence[float],
    heat_W: float,
    resistances_K_W: Sequence[float],
    capacitances_J_K: Sequence[float],
    *,
    ambient_K: float = 298.15,
) -> np.ndarray:
    """Return the exact zero-initial-state response of a Foster impedance."""
    network = FosterImpedance(
        resistances_K_W,
        capacitances_J_K,
        ambient_K=ambient_K,
    )
    times = np.asarray(time_s, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("time_s must be finite nonnegative observations")
    heat = _finite(heat_W, "heat_W")
    if heat < 0:
        raise ValueError("step heat must be nonnegative")
    impedance = network.resistances_K_W[None, :] * heat * (
        1
        - np.exp(
            -times[:, None]
            / (network.resistances_K_W * network.capacitances_J_K)
        )
    )
    return ambient_K + np.sum(impedance, axis=1)
