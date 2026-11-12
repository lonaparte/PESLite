"""Feedback return-ratio and classical crossover margins from real simulation data.

A virtual injection block records the signal on each side of the injection
point. A closed feedback connection has a return ratio given by the quotient
of the return and forward signals at that cut, not by return/disturbance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from .frequency import FrequencyMeasurement, _complex
from .port_measurement import measure_recorded_transfer

__all__ = ["measure_return_ratio", "loop_gain_sweep", "LoopGainResult"]


def measure_return_ratio(result, *, injection: str, frequency: float,
                         component: str = "d", feedback_sign: float = -1.0,
                         window=None, **quality_options) -> FrequencyMeasurement:
    """Read before/after observations from a typed virtual feedback injection.

    ``feedback_sign=-1`` is for ordinary negative feedback: L = -before/after.
    Do not divide the return by the *added disturbance*: that is a closed-loop
    sensitivity, not the return ratio. The user must verify the edge's sign.
    """
    if component not in ("d", "q"):
        raise ValueError("component must be 'd' or 'q'")
    if isinstance(feedback_sign, bool) or feedback_sign not in (-1.0, 1.0):
        raise ValueError("feedback_sign must be +1 or -1")
    before = f"{injection}.before_{component}"
    after = f"{injection}.after_{component}"
    observed = measure_recorded_transfer(result, after, before, frequency=frequency,
                                         window=window, **quality_options)
    value = observed.gain * feedback_sign if observed.gain is not None else None
    return replace(observed, gain=value, conditions={**observed.conditions,
                     "feedback_sign": feedback_sign, "virtual_cut": injection,
                     "definition": "feedback_sign * before / after"})


def _crossings(logf, samples, level, accepted):
    """Interpolate only between consecutive valid samples; never across rejected gaps."""
    crossing = []
    for i in range(len(samples) - 1):
        if not (accepted[i] and accepted[i + 1]):
            continue
        a, b = float(samples[i] - level), float(samples[i + 1] - level)
        if a == 0.0:
            crossing.append((float(logf[i]), i, 0.0))
        if a * b < 0.0:
            alpha = -a / (b - a)
            crossing.append((float(logf[i] + alpha * (logf[i + 1] - logf[i])), i, alpha))
    if accepted[-1] and samples[-1] == level:
        crossing.append((float(logf[-1]), len(samples) - 2, 1.0))
    return crossing


@dataclass(frozen=True)
class LoopGainResult:
    """Measured complex return ratios and interpolated classical stability margins."""

    frequencies_hz: np.ndarray
    gains: np.ndarray
    valid: tuple[bool, ...]
    flags: tuple[tuple[str, ...], ...]
    conditions: tuple[dict, ...]
    gain_margin_db: float | None
    phase_margin_deg: float | None
    gain_crossover_hz: float | None
    phase_crossover_hz: float | None
    diagnostics: tuple[str, ...]

    def to_dict(self):
        return {
            "frequency": {"value": self.frequencies_hz.tolist(), "unit": "Hz"},
            "return_ratio": {"value": [_complex(value) for value in self.gains],
                             "unit": "1"},
            "validity": [{"valid": good, "flags": list(flags), "conditions": dict(conditions)}
                         for good, flags, conditions in zip(self.valid, self.flags,
                                                             self.conditions)],
            "margins": {
                "gain_margin": {"value": self.gain_margin_db, "unit": "dB"},
                "phase_margin": {"value": self.phase_margin_deg, "unit": "deg"},
                "gain_crossover": {"value": self.gain_crossover_hz, "unit": "Hz"},
                "phase_crossover": {"value": self.phase_crossover_hz, "unit": "Hz"},
            },
            "diagnostics": list(self.diagnostics),
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path


def loop_gain_sweep(measurements: Mapping[float, FrequencyMeasurement]) -> LoopGainResult:
    """Find gain/phase crossovers without extrapolating or using failed measurements.

    Log-frequency interpolation is a descriptive measurement, not a certificate
    of closed-loop stability. All intermediate samples must be independently
    validated with sufficient excitation and an appropriate feedback cut.
    """
    if not measurements:
        raise ValueError("measurements must be a nonempty frequency mapping")
    points = sorted(measurements.items())
    f = np.asarray([item[0] for item in points], dtype=float)
    if (not np.all(np.isfinite(f)) or np.any(f <= 0)
            or np.any(np.diff(f) <= 0)):
        raise ValueError("measurement frequencies must be positive and strictly increasing")
    records = [item[1] for item in points]
    if any(not isinstance(item, FrequencyMeasurement) for item in records):
        raise ValueError("every measurement must be FrequencyMeasurement")
    if any(not math.isclose(float(freq), item.frequency_hz, rel_tol=1e-8)
           for freq, item in points):
        raise ValueError("measurement frequency keys must match their measurements")
    accepted = np.array([item.valid and item.gain is not None for item in records], dtype=bool)
    gains = np.asarray([item.gain if good else complex(math.nan, math.nan)
                        for item, good in zip(records, accepted)], dtype=complex)
    # Unwrap only contiguous valid islands; there is no phase bridge over a
    # missing or unreliable point.
    db = np.full(len(f), math.nan)
    phase = np.full(len(f), math.nan)
    i = 0
    while i < len(f):
        if not accepted[i]:
            i += 1
            continue
        j = i + 1
        while j < len(f) and accepted[j]:
            j += 1
        island = gains[i:j]
        magnitude = np.abs(island)
        if np.any(magnitude <= 0):
            accepted[i:j] &= magnitude > 0
        else:
            db[i:j] = 20 * np.log10(magnitude)
            phase[i:j] = np.rad2deg(np.unwrap(np.angle(island)))
        i = j
    accepted &= np.isfinite(db) & np.isfinite(phase)
    crossing_magnitude = _crossings(np.log10(f), db, 0.0, accepted)
    crossing_phase = _crossings(np.log10(f), phase, -180.0, accepted)
    phase_margins = [(180.0 + np.interp(log_at, np.log10(f)[good], phase[good]),
                      10 ** log_at)
                     for log_at, _i, _a in crossing_magnitude
                     for good in [accepted]]
    gain_margins = [(-float(np.interp(log_at, np.log10(f)[accepted], db[accepted])),
                     10 ** log_at)
                    for log_at, _i, _a in crossing_phase]
    # A multi-crossover response has multiple classical margins: report the
    # smallest absolute margin and retain the ambiguity as a diagnostic.
    pm, fg = (min(phase_margins, key=lambda a: abs(a[0]))
              if phase_margins else (None, None))
    gm, fp = (min(gain_margins, key=lambda a: abs(a[0]))
              if gain_margins else (None, None))
    diagnostics = []
    if not phase_margins:
        diagnostics.append("gain crossover not observed within the validated frequency range")
    if not gain_margins:
        diagnostics.append("-180-degree phase crossover not observed within the validated frequency range")
    if len(phase_margins) > 1 or len(gain_margins) > 1:
        diagnostics.append("multiple crossovers: nearest-to-zero margin selected")
    if any(not good for good in accepted):
        diagnostics.append("unreliable or zero-gain measurements excluded")
    return LoopGainResult(f, gains, tuple(bool(x) for x in accepted),
                          tuple(record.flags for record in records),
                          tuple(record.conditions for record in records),
                          gm, pm, fg, fp, tuple(diagnostics))
