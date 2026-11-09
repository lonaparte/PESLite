"""Integrated energy closure of a completed simulation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["energy_closure"]


def energy_closure(result: Any) -> dict[str, float]:
    """Return stored, supplied, dissipated, impulsive and residual energy in joules."""
    summary = getattr(result, "summary", {})
    if isinstance(summary, Mapping) and summary.get("energy_closure", ...) is None:
        raise ValueError("accepted-step energy closure is unavailable because an energy read failed")
    accumulated = summary.get("energy_closure") if isinstance(summary, Mapping) else None
    if isinstance(accumulated, Mapping):
        return {name: float(value) for name, value in accumulated.items()}
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
    def jumps(suffix: str) -> float:
        return sum(float(column[-1] - column[0])
                   for name in table if str(name).endswith(suffix)
                   for column in [np.asarray(table[name], dtype=float)])

    impulsive = jumps(".dissipated_impulsive_J")
    impulsive_supply = jumps(".supplied_impulsive_J")
    stored = float(energy[-1] - energy[0])
    supplied = impulsive_supply + float(
        np.sum(0.5 * step * (supplied_power[1:] + supplied_power[:-1]))
    )
    dissipated = float(np.sum(0.5 * step * (dissipated_power[1:] + dissipated_power[:-1])))
    residual = supplied - dissipated - impulsive - stored
    scale = max(abs(stored), abs(supplied), abs(dissipated), abs(impulsive), 1e-300)
    return {
        "stored": stored,
        "supplied": supplied,
        "dissipated": dissipated,
        "impulsive": impulsive,
        "impulsive_supply": impulsive_supply,
        "residual": residual,
        "relative": residual / scale,
    }
