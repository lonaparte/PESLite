"""Thermal reports from the existing restartable PESLite result histories."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["thermal_report", "compare_thermal_reports"]


def _array(name: str, value: Any, size: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be numeric") from None
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a one-dimensional finite array")
    if size is not None and len(result) != size:
        raise ValueError(f"{name} must match the observation count")
    return result


def thermal_report(
    result: Any, name: str, *, maximum_junction_K: float,
    cooling_column: str | None = None,
    output_power_W: float | None = None,
) -> dict[str, float | bool | None]:
    """Summarize an electrothermal element without changing the electrical run.

    The input power is the already-connected electrical heat signal. Thermal
    energy closure is separate from PESLite's electrical port-Hamiltonian audit.
    Optional electrical efficiency is calculated only with explicitly supplied
    delivered output power, not inferred from gross circuit supply.
    """
    plant, states = result.plant, result.states
    if not isinstance(plant, Mapping) or not isinstance(states, Mapping):
        raise TypeError("result needs streamed plant and state mappings")
    t = _array("time", result.t)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("thermal report requires two increasing recorded times")
    max_temp = float(maximum_junction_K)
    if not np.isfinite(max_temp) or max_temp <= 0:
        raise ValueError("maximum_junction_K must be finite positive K")
    junction = _array("temperature_K", plant[f"{name}.temperature_K"], len(t))
    case = _array("case_K", plant[f"{name}.case_K"], len(t))
    heat = _array("heat_W", plant[f"{name}.heat_W"], len(t))
    cooling_key = cooling_column or f"{name}.cooling_W"
    rejected = _array("cooling_W", plant[cooling_key], len(t))
    duration = float(t[-1] - t[0])
    dt = np.diff(t)

    def integrate(x):
        return float(np.dot(dt, 0.5 * (x[1:] + x[:-1])))

    heat_j = integrate(heat)
    cooling_j = integrate(rejected)
    peak = float(np.max(junction))
    peak_index = int(np.argmax(junction))
    # Report physical temperature, never an unvalidated extrapolated steady state.
    tail = max(2, len(t) // 10)
    steady = float(np.mean(junction[-tail:]))
    if not np.all(np.isfinite(case)):
        raise ValueError("case temperature is unavailable; a Cauer network is required")
    if not np.all(np.isfinite(rejected)):
        raise ValueError("cooling heat is unavailable; a Cauer network is required")

    power = None
    efficiency = None
    if output_power_W is not None:
        power = float(output_power_W)
        if not np.isfinite(power) or power < 0:
            raise ValueError("output_power_W must be nonnegative and finite")
        efficiency = power / (power + heat_j / duration) if power + heat_j > 0 else None

    return {
        "duration_s": duration,
        "junction_peak_K": peak,
        "junction_peak_time_s": float(t[peak_index]),
        "junction_final_K": float(junction[-1]),
        "junction_tail_mean_K": steady,
        "case_peak_K": float(np.max(case)),
        "case_final_K": float(case[-1]),
        "junction_limit_K": max_temp,
        "safety_margin_K": max_temp - peak,
        "within_rating": bool(peak <= max_temp),
        "heat_integrated_J": heat_j,
        "heat_average_W": heat_j / duration,
        "cooling_integrated_J": cooling_j,
        "cooling_average_W": cooling_j / duration,
        "output_power_W": power,
        "estimated_device_efficiency": efficiency,
    }


def compare_thermal_reports(
    cases: Mapping[str, Mapping[str, float | bool | None]],
) -> dict[str, dict]:
    """Rank already-simulated device/cooling candidates by thermal headroom.

    This does not infer that different electrical operating conditions are equal:
    callers must simulate all candidates with matching circuit parameters, time
    ranges, heat mechanisms and device-current assignments.
    """
    if not cases:
        raise ValueError("at least one thermal case is required")
    result = {}
    for name, entry in cases.items():
        if not {"junction_peak_K", "safety_margin_K", "heat_average_W"} <= entry.keys():
            raise ValueError(f"thermal case {name!r} lacks required report metrics")
        result[name] = dict(entry)
    return dict(sorted(result.items(), key=lambda pair: (
        -float(pair[1]["safety_margin_K"]), float(pair[1]["heat_average_W"]), pair[0]
    )))
