"""Integrated energy closure of a completed simulation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["energy_closure"]


def energy_closure(result: Any) -> dict[str, float]:
    """Return stored, supplied, dissipated, impulsive and residual energy in joules."""
    table: Mapping[str, Any] = getattr(result, "energy", result)
    required = ("t", "energy_J", "power_supplied_W", "power_dissipated_W")
    try:
        t, energy, supplied_power, dissipated_power = (
            np.asarray(table[name], dtype=float) for name in required
        )
    except KeyError as exc:
        raise ValueError(
            "energy_closure requires an energy table; enable simulation.output.energy"
        ) from exc
    sizes = {array.size for array in (t, energy, supplied_power, dissipated_power)}
    if sizes != {t.size} or t.size < 2:
        raise ValueError("energy_closure requires equally sized columns with at least two rows")
    step = np.diff(t)
    if np.any(step <= 0.0):
        raise ValueError("energy rows must be strictly increasing in time")
    impulsive = sum(float(column[-1] - column[0])
                    for name in table if str(name).endswith(".dissipated_impulsive_J")
                    for column in [np.asarray(table[name], dtype=float)])
    stored = float(energy[-1] - energy[0])
    supplied = float(np.sum(0.5 * step * (supplied_power[1:] + supplied_power[:-1])))
    dissipated = float(np.sum(0.5 * step * (dissipated_power[1:] + dissipated_power[:-1])))
    residual = supplied - dissipated - impulsive - stored
    scale = max(abs(stored), abs(supplied), abs(dissipated), abs(impulsive), 1e-300)
    return {
        "stored": stored,
        "supplied": supplied,
        "dissipated": dissipated,
        "impulsive": impulsive,
        "residual": residual,
        "relative": residual / scale,
    }
