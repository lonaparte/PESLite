"""Step-response and tracking metrics from recorded transient waveforms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = ["DynamicMetrics", "dynamic_performance"]


def _vector(name: str, values: Iterable[float]) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
    return result


def _mean(t: np.ndarray, values: np.ndarray) -> float:
    if len(t) == 1:
        return float(values[0])
    width = float(t[-1] - t[0])
    return float(np.sum(np.diff(t) * (values[1:] + values[:-1]) / 2.0) / width)


def _crossing(t: np.ndarray, progress: np.ndarray, threshold: float) -> float | None:
    if progress[0] >= threshold:
        return None  # The threshold preceded the first observation.
    for index in range(1, len(t)):
        if progress[index] >= threshold and progress[index - 1] < threshold:
            fraction = ((threshold - progress[index - 1]) /
                        (progress[index] - progress[index - 1]))
            return float(t[index - 1] + fraction * (t[index] - t[index - 1]))
    return None


@dataclass(frozen=True)
class DynamicMetrics:
    """Numerical metrics, measurement conditions and explicitly missing results."""

    rise_time: float | None
    overshoot_percent: float | None
    settling_time: float | None
    steady_state_error: float | None
    ripple_peak_to_peak: float
    tracking_rmse: float | None
    tracking_max_abs: float | None
    final_value: float
    target: float
    unit: str
    conditions: dict
    flags: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.flags

    def to_dict(self) -> dict:
        values = {
            "rise_time": (self.rise_time, "s"),
            "overshoot": (self.overshoot_percent, "%"),
            "settling_time": (self.settling_time, "s"),
            "steady_state_error": (self.steady_state_error, self.unit),
            "ripple_peak_to_peak": (self.ripple_peak_to_peak, self.unit),
            "tracking_rmse": (self.tracking_rmse, self.unit),
            "tracking_max_abs": (self.tracking_max_abs, self.unit),
            "final_value": (self.final_value, self.unit),
            "target": (self.target, self.unit),
        }
        return {
            "metrics": {name: {"value": value, "unit": unit, "valid": value is not None}
                        for name, (value, unit) in values.items()},
            "valid": self.valid, "flags": list(self.flags),
            "conditions": dict(self.conditions),
        }


def dynamic_performance(time: Iterable[float], response: Iterable[float], *,
                        reference: Iterable[float] | float | None = None,
                        target: float | None = None, step_time: float | None = None,
                        initial: float | None = None, settling_tolerance: float = 0.02,
                        absolute_tolerance: float = 0.0, tail_fraction: float = 0.1,
                        unit: str = "") -> DynamicMetrics:
    """Calculate step dynamics and tracking error on an observed time interval.

    ``step_time`` is the beginning of the commanded step (defaults to the first
    sample). The target is an explicit ``target``, the final reference value, or
    the mean of the last ``tail_fraction`` of the response. Rise time uses the
    first 10% and 90% crossings with linear interpolation; settling time is the
    earliest post-step sample after which all samples remain within the stated
    relative/absolute band. Missing or unresolved metrics are returned as None.
    """
    t, y = _vector("time", time), _vector("response", response)
    if t.size < 3 or t.size != y.size or np.any(np.diff(t) <= 0):
        raise ValueError("time and response need at least three increasing, paired samples")
    if not isinstance(unit, str):
        raise ValueError("unit must be a string")
    for name, value in (("settling_tolerance", settling_tolerance),
                        ("absolute_tolerance", absolute_tolerance),
                        ("tail_fraction", tail_fraction)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if settling_tolerance <= 0 or absolute_tolerance < 0 or not 0 < tail_fraction <= 1:
        raise ValueError("invalid settling tolerances or tail_fraction")
    step_time = float(t[0]) if step_time is None else float(step_time)
    if not math.isfinite(step_time) or not t[0] <= step_time < t[-1]:
        raise ValueError("step_time must occur within the observed interval")
    start_index = int(np.searchsorted(t, step_time, side="left"))
    if t[start_index] != step_time:
        raise ValueError("step_time must coincide with a recorded sample")
    if t.size - start_index < 3:
        raise ValueError("at least three post-step samples are required")

    r = None
    if reference is not None:
        if isinstance(reference, (int, float)) and not isinstance(reference, bool):
            if not math.isfinite(reference):
                raise ValueError("reference must be finite")
            r = np.full_like(t, float(reference))
        else:
            r = _vector("reference", reference)
            if r.size != t.size:
                raise ValueError("reference must match the time array")
    for name, value in (("target", target), ("initial", initial)):
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError(f"{name} must be finite")

    tail_start = t[-1] - tail_fraction * (t[-1] - step_time)
    tail_index = min(int(np.searchsorted(t, tail_start)), t.size - 2)
    final_value = _mean(t[tail_index:], y[tail_index:])
    goal = float(target if target is not None else r[-1] if r is not None else final_value)
    origin = float(initial if initial is not None else y[start_index])
    delta = goal - origin
    flags: list[str] = []
    if r is None and target is None:
        flags.append("target_estimated_from_response")
    if r is not None and np.max(np.abs(r[start_index:] - goal)) > max(
            absolute_tolerance, settling_tolerance * max(abs(delta), 1e-12)):
        flags.append("reference_changes_after_step")
    steady_error = final_value - goal if r is not None or target is not None else None
    ripple = float(np.ptp(y[tail_index:]))

    tracking_rmse = tracking_max = None
    if r is not None:
        e = y[start_index:] - r[start_index:]
        tracking_rmse = math.sqrt(_mean(t[start_index:], e * e))
        tracking_max = float(np.max(np.abs(e)))

    rise = overshoot = settle = None
    if delta == 0.0:
        flags.append("zero_step_amplitude")
    elif "reference_changes_after_step" not in flags:
        progress = (y[start_index:] - origin) / delta
        t10 = _crossing(t[start_index:], progress, 0.1)
        t90 = _crossing(t[start_index:], progress, 0.9)
        if t10 is not None and t90 is not None and t90 >= t10:
            rise = t90 - t10
        else:
            flags.append("rise_time_unresolved")
        overshoot = 100.0 * max(0.0, float(np.max((y[start_index:] - goal) / delta)))
        band = max(absolute_tolerance, settling_tolerance * abs(delta))
        violations = np.flatnonzero(np.abs(y[start_index:] - goal) > band)
        if not len(violations):
            settle = 0.0
        elif violations[-1] < len(t) - start_index - 1:
            settle = float(t[start_index + violations[-1] + 1] - step_time)
        else:
            flags.append("not_settled")
    return DynamicMetrics(
        rise, overshoot, settle, steady_error, ripple, tracking_rmse,
        tracking_max, final_value, goal, unit,
        {"step_time_s": step_time, "tail_start_s": float(t[tail_index]),
         "sample_count": int(t.size), "settling_tolerance": settling_tolerance,
         "absolute_tolerance": absolute_tolerance},
        tuple(flags),
    )
