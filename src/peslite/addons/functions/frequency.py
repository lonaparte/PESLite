"""Single-tone response validation and structured multiport impedance estimates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

__all__ = ["FrequencyMeasurement", "FrequencyResponseResult", "MatrixResponseResult",
           "frequency_response", "frequency_sweep", "impedance_matrix"]


def _complex(value: complex | None) -> dict | None:
    if value is None:
        return None
    z = complex(value)
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        return None
    return {"real": z.real, "imag": z.imag}


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _write(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _fit(t: np.ndarray, y: np.ndarray, f: float, harmonics: int, origin: float):
    angle = math.tau * f * (t - origin)
    terms = [np.ones(len(t))]
    for h in range(1, harmonics + 1):
        terms.extend((np.cos(h * angle), np.sin(h * angle)))
    design = np.column_stack(terms)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    phasors = np.asarray([coefficients[2 * i - 1] - 1j * coefficients[2 * i]
                          for i in range(1, harmonics + 1)])
    residual = y - design @ coefficients
    signal_rms = abs(phasors[0]) / math.sqrt(2)
    noise_rms = float(np.sqrt(np.mean(residual * residual)))
    snr = math.inf if noise_rms == 0 else (20 * math.log10(signal_rms / noise_rms)
                                           if signal_rms else -math.inf)
    thd = (float(np.linalg.norm(phasors[1:]) / abs(phasors[0]))
           if abs(phasors[0]) else math.inf)
    return complex(phasors[0]), float(coefficients[0]), thd, snr


@dataclass(frozen=True)
class FrequencyMeasurement:
    """Fundamental gain, quality evidence and explicit measurement validity."""

    frequency_hz: float
    gain: complex | None
    excitation_phasor: complex
    response_phasor: complex
    excitation_amplitude: float
    distortion: float | None
    excitation_snr_db: float | None
    response_snr_db: float | None
    conditions: dict
    flags: tuple[str, ...]
    input_unit: str = ""
    output_unit: str = ""

    @property
    def valid(self) -> bool:
        return not self.flags

    def to_dict(self) -> dict:
        def snr(value):
            return {"value": _finite(value) if value is not None else None,
                    "unit": "dB", "infinite": value is not None and math.isinf(value)}

        return {
            "frequency": {"value": self.frequency_hz, "unit": "Hz"},
            "gain": {"value": _complex(self.gain),
                     "unit": f"{self.output_unit}/{self.input_unit}".strip("/")},
            "excitation": {"phasor": _complex(self.excitation_phasor),
                           "amplitude": self.excitation_amplitude,
                           "unit": self.input_unit},
            "response": {"phasor": _complex(self.response_phasor), "unit": self.output_unit},
            "distortion": {"value": self.distortion, "unit": "ratio"},
            "excitation_snr": snr(self.excitation_snr_db),
            "response_snr": snr(self.response_snr_db),
            "valid": self.valid, "flags": list(self.flags),
            "conditions": dict(self.conditions),
        }

    def write_json(self, path: str | Path) -> Path:
        return _write(self.to_dict(), path)


def frequency_response(time: Iterable[float], excitation: Iterable[float],
                       response: Iterable[float], *, frequency: float,
                       min_cycles: float = 6.0, min_excitation_amplitude: float = 1e-6,
                       max_steady_drift: float = 0.1, max_distortion: float = 0.1,
                       min_snr_db: float = 20.0, max_sample_jitter: float = 0.01,
                       input_unit: str = "", output_unit: str = "") -> FrequencyMeasurement:
    """Measure a sinusoidal gain and reject unreliable transient frequency estimates.

    Fundamental and harmonic amplitudes are least-squares phasors, referenced to
    the first sample, avoiding a requirement for integer-bin FFT alignment.
    First/second-half phasor and offset differences check convergence. The
    maximum sample gap supplies a conservative Nyquist bound; the window length
    supplies frequency resolution. Invalid gains are None, never silent numbers.
    """
    try:
        t, x, y = (np.asarray(values, dtype=float)
                   for values in (time, excitation, response))
    except (ValueError, TypeError) as exc:
        raise ValueError("measurement arrays must contain numbers") from exc
    if (any(a.ndim != 1 for a in (t, x, y)) or t.size < 16 or
            t.size != x.size or t.size != y.size or
            not all(np.all(np.isfinite(a)) for a in (t, x, y)) or
            np.any(np.diff(t) <= 0)):
        raise ValueError("measurement arrays need at least 16 finite, increasing paired samples")
    parameters = {
        "frequency": frequency, "min_cycles": min_cycles,
        "min_excitation_amplitude": min_excitation_amplitude,
        "max_steady_drift": max_steady_drift, "max_distortion": max_distortion,
        "min_snr_db": min_snr_db, "max_sample_jitter": max_sample_jitter,
    }
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in parameters.values()):
        raise ValueError("frequency analysis options must be finite numbers")
    if (frequency <= 0 or min_cycles <= 0 or min_excitation_amplitude <= 0 or
            max_steady_drift < 0 or max_distortion < 0 or max_sample_jitter < 0):
        raise ValueError("invalid frequency analysis limits")
    if not isinstance(input_unit, str) or not isinstance(output_unit, str):
        raise ValueError("physical units must be strings")

    delta = np.diff(t)
    duration = float(t[-1] - t[0])
    sample_gap = float(np.max(delta))
    nominal_gap = float(np.median(delta))
    nyquist = 0.5 / sample_gap
    resolution = 1.0 / duration
    jitter = float(np.max(np.abs(delta - nominal_gap)) / nominal_gap)
    cycles = frequency * duration
    flags = []
    if jitter > max_sample_jitter:
        flags.append("nonuniform_sampling")
    if frequency >= nyquist:
        flags.append("sampling_aliasing")
    if cycles < min_cycles:
        flags.append("insufficient_frequency_resolution")
    harmonics = max(1, min(5, int(math.floor(nyquist * (1.0 - 1e-10) / frequency))))
    if harmonics < 3:
        flags.append("insufficient_harmonic_bandwidth")
    harmonics = min(harmonics, (len(t) - 2) // 2)
    fx, cx, _, sx = _fit(t, x, frequency, harmonics, float(t[0]))
    fy, cy, thd, sy = _fit(t, y, frequency, harmonics, float(t[0]))
    if abs(fx) < min_excitation_amplitude:
        flags.append("insufficient_excitation_amplitude")
    if thd > max_distortion:
        flags.append("nonlinear_distortion")
    if sx < min_snr_db or sy < min_snr_db:
        flags.append("low_signal_to_noise_ratio")
    midpoint = t.size // 2
    if midpoint >= 4 and t.size - midpoint >= 4:
        fx1, cx1, *_ = _fit(t[:midpoint], x[:midpoint], frequency, 1, float(t[0]))
        fx2, cx2, *_ = _fit(t[midpoint:], x[midpoint:], frequency, 1, float(t[0]))
        fy1, cy1, *_ = _fit(t[:midpoint], y[:midpoint], frequency, 1, float(t[0]))
        fy2, cy2, *_ = _fit(t[midpoint:], y[midpoint:], frequency, 1, float(t[0]))
        drift = max(
            abs(fx2 - fx1) / max(abs(fx), min_excitation_amplitude),
            abs(fy2 - fy1) / max(abs(fy), min_excitation_amplitude),
            abs(cx2 - cx1) / max(abs(fx), min_excitation_amplitude),
            abs(cy2 - cy1) / max(abs(fy), min_excitation_amplitude),
        )
        if drift > max_steady_drift:
            flags.append("steady_state_not_converged")
    gain = fy / fx if abs(fx) >= min_excitation_amplitude else None
    if flags:
        gain = None
    return FrequencyMeasurement(
        float(frequency), gain, fx, fy, abs(fx), thd if math.isfinite(thd) else None,
        sx, sy, {"sample_count": int(t.size), "duration_s": duration,
                 "sample_gap_max_s": sample_gap, "sample_jitter": jitter,
                 "nyquist_hz": nyquist, "frequency_resolution_hz": resolution,
                 "cycles": cycles, "harmonics_evaluated": harmonics,
                 "min_cycles": min_cycles, "max_steady_drift": max_steady_drift,
                 "max_distortion": max_distortion, "min_snr_db": min_snr_db},
        tuple(flags), input_unit, output_unit,
    )


@dataclass(frozen=True)
class FrequencyResponseResult:
    """A frequency-ordered series of independently validated complex gains."""

    frequencies_hz: np.ndarray
    gains: np.ndarray
    valid: tuple[bool, ...]
    flags: tuple[tuple[str, ...], ...]
    conditions: tuple[dict, ...]
    input_unit: str
    output_unit: str

    def to_dict(self) -> dict:
        return {
            "frequency": {"value": self.frequencies_hz.tolist(), "unit": "Hz"},
            "transfer": {
                "value": [_complex(value) for value in self.gains],
                "unit": f"{self.output_unit}/{self.input_unit}".strip("/"),
                "input_unit": self.input_unit, "output_unit": self.output_unit,
            },
            "validity": [{"valid": valid, "flags": list(flags), "conditions": dict(conditions)}
                         for valid, flags, conditions in zip(self.valid, self.flags,
                                                            self.conditions)],
        }

    def write_json(self, path: str | Path) -> Path:
        return _write(self.to_dict(), path)


def frequency_sweep(measurements: Iterable[FrequencyMeasurement]) -> FrequencyResponseResult:
    """Combine independently measured tones without discarding invalid frequencies."""
    records = tuple(measurements)
    if not records or any(not isinstance(record, FrequencyMeasurement) for record in records):
        raise ValueError("measurements must contain FrequencyMeasurement objects")
    if any(b.frequency_hz <= a.frequency_hz for a, b in zip(records, records[1:])):
        raise ValueError("measurement frequencies must be strictly increasing")
    if any((record.input_unit, record.output_unit) !=
           (records[0].input_unit, records[0].output_unit) for record in records):
        raise ValueError("measurement units must agree across the frequency sweep")
    gains = np.asarray([record.gain if record.valid and record.gain is not None
                        else complex(math.nan, math.nan) for record in records], dtype=complex)
    return FrequencyResponseResult(
        np.asarray([record.frequency_hz for record in records]), gains,
        tuple(record.valid for record in records),
        tuple(record.flags for record in records),
        tuple(record.conditions for record in records),
        records[0].input_unit, records[0].output_unit,
    )


@dataclass(frozen=True)
class MatrixResponseResult:
    """Complex transfer matrices ordered by frequency, output and input."""

    frequencies_hz: np.ndarray
    values: np.ndarray  # [frequency, output, input]
    quantity: str
    unit: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    valid: tuple[bool, ...]
    flags: tuple[tuple[str, ...], ...]
    conditions: tuple[dict, ...]
    frame: str | None = None

    def to_dict(self) -> dict:
        response = {
            "value": [[[_complex(z) for z in row] for row in matrix]
                      for matrix in self.values],
            "unit": self.unit, "outputs": list(self.output_names),
            "inputs": list(self.input_names),
        }
        if self.frame is not None:
            response["frame"] = self.frame
        return {
            "frequency": {"value": self.frequencies_hz.tolist(), "unit": "Hz"},
            self.quantity: response,
            "validity": [{"valid": good, "flags": list(reasons), "conditions": dict(conditions)}
                         for good, reasons, conditions in zip(self.valid, self.flags,
                                                              self.conditions)],
        }

    def write_json(self, path: str | Path) -> Path:
        return _write(self.to_dict(), path)


def impedance_matrix(frequencies: Iterable[float], current_phasors: np.ndarray,
                     voltage_phasors: np.ndarray, *, input_names: Iterable[str] | None = None,
                     output_names: Iterable[str] | None = None,
                     measurement_valid: Iterable[Iterable[bool]] | None = None,
                     max_condition_number: float = 1e8,
                     max_relative_residual: float = 0.05) -> MatrixResponseResult:
    """Fit ``V = Z I`` from independent complex-current injection experiments.

    Arrays have shape [frequency, experiment, input/output port]; a single
    frequency can use two-dimensional arrays. Rank-deficient, ill-conditioned,
    untrusted or poorly fitted points retain invalid/NaN matrices for inspection.
    """
    f = np.asarray(frequencies, dtype=float)
    current = np.asarray(current_phasors, dtype=complex)
    voltage = np.asarray(voltage_phasors, dtype=complex)
    if f.ndim != 1 or not len(f) or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequencies must be a nonempty vector of positive finite Hz values")
    if len(f) > 1 and np.any(np.diff(f) <= 0):
        raise ValueError("frequencies must be strictly increasing")
    if len(f) == 1 and current.ndim == voltage.ndim == 2:
        current, voltage = current[None, ...], voltage[None, ...]
    if (current.ndim != 3 or voltage.ndim != 3 or current.shape[:2] != voltage.shape[:2]
            or current.shape[0] != len(f) or not current.size or not voltage.size or
            not np.all(np.isfinite(current)) or not np.all(np.isfinite(voltage))):
        raise ValueError("phasors must be finite arrays matching [frequency, experiment, port]")
    for name, value in (("max_condition_number", max_condition_number),
                        ("max_relative_residual", max_relative_residual)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    nfreq, nexperiments, ninputs = current.shape
    noutputs = voltage.shape[-1]
    names_in = (tuple(f"i{j+1}" for j in range(ninputs)) if input_names is None
                else tuple(input_names))
    names_out = (tuple(f"v{j+1}" for j in range(noutputs)) if output_names is None
                 else tuple(output_names))
    if (len(names_in) != ninputs or len(names_out) != noutputs or
            any(not isinstance(name, str) or not name for name in (*names_in, *names_out)) or
            len(set(names_in)) != len(names_in) or len(set(names_out)) != len(names_out)):
        raise ValueError("port names must be distinct nonempty strings matching the array widths")
    trusted = (np.ones((nfreq, nexperiments), dtype=bool) if measurement_valid is None
               else np.asarray(measurement_valid))
    if trusted.shape != (nfreq, nexperiments) or trusted.dtype != np.dtype(bool):
        raise ValueError("measurement_valid must be a boolean [frequency, experiment] array")

    matrices = np.full((nfreq, noutputs, ninputs), complex(np.nan, np.nan))
    good = []
    issues = []
    conditions = []
    for k in range(nfreq):
        flags = []
        if not np.all(trusted[k]):
            flags.append("unreliable_input_measurements")
        selected = trusted[k]
        a, b = current[k, selected], voltage[k, selected]
        if len(a) < ninputs:
            flags.append("insufficient_independent_excitations")
        condition = float(np.linalg.cond(a)) if len(a) >= ninputs else math.inf
        if condition > max_condition_number:
            flags.append("ill_conditioned_excitation")
        if len(a):
            coefficients, _residuals, rank, _ = np.linalg.lstsq(a, b, rcond=None)
            if rank < ninputs:
                flags.append("rank_deficient_excitation")
            residual = float(np.linalg.norm(a @ coefficients - b) /
                             max(float(np.linalg.norm(b)), np.finfo(float).tiny))
            if residual > max_relative_residual:
                flags.append("poor_matrix_fit")
            if not flags:
                matrices[k] = coefficients.T
        else:
            residual = math.inf
        good.append(not flags)
        issues.append(tuple(flags))
        conditions.append({"experiments": int(len(a)),
                           "condition_number": _finite(condition),
                           "relative_residual": _finite(residual),
                           "max_condition_number": float(max_condition_number),
                           "max_relative_residual": float(max_relative_residual)})
    return MatrixResponseResult(f, matrices, "impedance", "ohm", names_in, names_out,
                                tuple(good), tuple(issues), tuple(conditions))
