"""Offline analysis of deliberately excited simulations and recorded waveforms."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "FrequencyMeasurement", "FrequencyResponseResult", "MatrixResponseResult",
    "frequency_response", "frequency_sweep", "impedance_matrix",
    "recorded_signal", "measure_recorded_transfer", "measure_recorded_sweep",
    "measure_multitone", "measure_port_matrix", "run_port_sweep", "run_port_matrix",
    "measure_return_ratio", "loop_gain_sweep", "LoopGainResult",
]


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


# Recorded-signal identification

def _table(result, name: str):
    """Find one original recorder and its own sampling clock, not an assumed common clock."""
    for kind in ("plant", "ctrl", "states"):
        table = getattr(result, kind)
        if name in table or all(f"{name}_{phase}" in table for phase in "abc"):
            if kind == "ctrl":
                owner = name.rpartition(".")[0]
                while owner:
                    clock = f"{owner}.t"
                    if clock in table:
                        return np.asarray(table[clock], dtype=float), table
                    owner = owner.rpartition(".")[0]
                raise ValueError(f"controller observation {name!r} has no sampling clock")
            if kind == "states":
                return np.asarray(table["t"], dtype=float), table
            return np.asarray(result.t, dtype=float), table
    raise ValueError(f"{name!r} was not recorded; enable simulation.output.signals "
                     "or simulation.output.states")


def _recorded_values(t, table, name, component, frame_frequency):
    """Project one located recording using its column structure."""
    if component is None:
        if name not in table:
            raise ValueError(f"{name!r} is a three-phase signal; select a component")
        y = np.asarray(table[name], dtype=float)
    else:
        if component not in ("alpha", "beta", "d", "q"):
            raise ValueError("component must be alpha, beta, d, or q")
        columns = [f"{name}_{phase}" for phase in "abc"]
        if not all(column in table for column in columns):
            raise ValueError(f"{name!r} has no recorded three-phase columns")
        a, b, c = (np.asarray(table[column], dtype=float) for column in columns)
        vector = (2.0 / 3.0) * (a + np.exp(2j * np.pi / 3) * b
                                + np.exp(-2j * np.pi / 3) * c)
        if component in ("d", "q"):
            vector *= np.exp(-1j * math.tau * frame_frequency * t)
        y = vector.real if component in ("alpha", "d") else vector.imag
    if (t.ndim != 1 or y.ndim != 1 or len(t) != len(y) or not len(t)
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y))
            or np.any(np.diff(t) <= 0)):
        raise ValueError(f"{name!r} must have finite, strictly increasing recorded samples")
    return t, y


def recorded_signal(result, name: str, *, component: str | None = None,
                    frame_frequency: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Read one scalar or three-phase signal without inventing another sample history.

    Three-phase sources use their standard ``_a/_b/_c`` columns. Set component
    to ``alpha``, ``beta``, ``d`` or ``q``; d/q rotates the space vector by the
    specified frame frequency. Scalar signals require ``component=None``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a nonempty recorded signal path")
    if (isinstance(frame_frequency, bool) or not isinstance(frame_frequency, (int, float))
            or not math.isfinite(frame_frequency)):
        raise ValueError("frame_frequency must be finite")
    t, table = _table(result, name)
    return _recorded_values(t, table, name, component, frame_frequency)


def _window(t, y, window):
    if window is None:
        return t, y
    if (len(window) != 2 or any(isinstance(x, bool) or not math.isfinite(x) for x in window)
            or window[0] >= window[1]):
        raise ValueError("window must contain two finite, increasing times")
    take = (t >= window[0]) & (t <= window[1])
    return t[take], y[take]


def _pair(result, first, second, *, window=None, first_component=None,
          second_component=None, frame_frequency=0.0, align="exact"):
    t1, x = recorded_signal(result, first, component=first_component,
                            frame_frequency=frame_frequency)
    t2, y = recorded_signal(result, second, component=second_component,
                            frame_frequency=frame_frequency)
    t1, x = _window(t1, x, window)
    t2, y = _window(t2, y, window)
    if t1.shape == t2.shape and np.array_equal(t1, t2):
        return t1, x, y
    if align != "linear":
        raise ValueError("recorded signals have different sample times; explicitly select "
                         "align='linear' to interpolate after verifying their bandwidth")
    if len(t2) < 2 or len(t1) < 2 or t1[0] < t2[0] or t1[-1] > t2[-1]:
        raise ValueError("the response does not cover the excitation's selected interval")
    return t1, x, np.interp(t1, t2, y)


def measure_recorded_transfer(result, excitation: str, response: str, *, frequency: float,
                              window: tuple[float, float] | None = None,
                              excitation_component: str | None = None,
                              response_component: str | None = None,
                              frame_frequency: float = 0.0, align: str = "exact",
                              input_unit: str = "", output_unit: str = "", **options):
    """Validate one physical or control-signal transfer using the shared single-tone fit.

    The caller chooses physical signal orientation and the relative coordinate
    frame explicitly. The resulting gain is rejected on any quality failure.
    """
    if align not in ("exact", "linear"):
        raise ValueError("align must be 'exact' or 'linear'")
    t, x, y = _pair(result, excitation, response, window=window,
                    first_component=excitation_component,
                    second_component=response_component,
                    frame_frequency=frame_frequency, align=align)
    result = frequency_response(t, x, y, frequency=frequency,
                                input_unit=input_unit, output_unit=output_unit, **options)
    return replace(result, conditions={**result.conditions, "excitation": excitation,
                                       "response": response, "window_s": [float(t[0]), float(t[-1])],
                                       "frame_frequency_hz": frame_frequency,
                                       "excitation_component": excitation_component,
                                       "response_component": response_component,
                                       "alignment": align})


def measure_recorded_sweep(runs: Mapping[float, object], excitation: str,
                           response: str, **options):
    """Analyze separately settled frequency runs in increasing frequency order."""
    if not runs:
        raise ValueError("runs must contain at least one frequency and completed result")
    records = [measure_recorded_transfer(result, excitation, response,
                                         frequency=float(f), **options)
               for f, result in sorted(runs.items())]
    return frequency_sweep(records)


def measure_multitone(result, excitation: str, response: str,
                      frequencies: Iterable[float], *, window=None,
                      input_unit: str = "", output_unit: str = "",
                      min_spacing_bins: float = 1.0, **options):
    """Use a joint coherent-tone fit before the shared single-frequency validity checks.

    Tones closer than one observation-window bin, or harmonically overlapping
    tones, cannot be disentangled reliably and are rejected explicitly.
    """
    freqs = np.asarray(tuple(frequencies), dtype=float)
    if (freqs.ndim != 1 or not len(freqs) or not np.all(np.isfinite(freqs))
            or np.any(freqs <= 0) or np.any(np.diff(freqs) <= 0)):
        raise ValueError("frequencies must be strictly increasing positive Hz values")
    if not math.isfinite(min_spacing_bins) or min_spacing_bins < 1:
        raise ValueError("min_spacing_bins must be at least one")
    t, x, y = _pair(result, excitation, response, window=window)
    if len(t) < max(16, 2 * len(freqs) + 5):
        raise ValueError("multitone identification needs more observed samples")
    duration = t[-1] - t[0]
    if len(freqs) > 1 and np.min(np.diff(freqs)) * duration < min_spacing_bins:
        raise ValueError("tones are too close for the available frequency resolution")
    for f in freqs:
        if any(abs(other / f - round(other / f)) < 1.0 / (duration * f)
               and other > f for other in freqs):
            raise ValueError("harmonically overlapping tones cannot be validated separately")
    theta = math.tau * (t - t[0])[:, None] * freqs[None, :]
    terms = np.column_stack([np.ones(len(t)), *(term for j in range(len(freqs))
                      for term in (np.cos(theta[:, j]), np.sin(theta[:, j])))])
    if np.linalg.cond(terms) > 1e8:
        raise ValueError("multitone experiment is poorly conditioned")
    co_x, *_ = np.linalg.lstsq(terms, x, rcond=None)
    co_y, *_ = np.linalg.lstsq(terms, y, rcond=None)
    measurements = []
    for i, f in enumerate(freqs):
        # Remove only other *deliberate* frequencies; retain target distortion
        # and all unmodelled residuals for the common quality checks.
        others = [j for j in range(len(freqs)) if j != i]
        remainder_x = x.copy()
        remainder_y = y.copy()
        for j in others:
            remainder_x -= (co_x[2*j+1] * np.cos(theta[:, j])
                            + co_x[2*j+2] * np.sin(theta[:, j]))
            remainder_y -= (co_y[2*j+1] * np.cos(theta[:, j])
                            + co_y[2*j+2] * np.sin(theta[:, j]))
        measured = frequency_response(t, remainder_x, remainder_y, frequency=float(f),
                                      input_unit=input_unit, output_unit=output_unit, **options)
        measurements.append(replace(measured, conditions={**measured.conditions,
             "joint_tones_hz": freqs.tolist(), "excitation": excitation,
             "response": response, "frequency_separation_bins": float(min_spacing_bins)}))
    return frequency_sweep(measurements)


def _phasor(t, y, frequency):
    """The same least-squares phasor convention used by frequency_response."""
    return _fit(t, y, frequency, 1, float(t[0]))[0]


def _complex_signal(result, name, *, frame_frequency):
    t, table = _table(result, name)
    channels = tuple(_recorded_values(t, table, name, component, frame_frequency)[1]
                     for component in ("d", "q"))
    return t, channels


def _source_quality(result, name, frequency, window, frame_frequency, options):
    """Validate the imposed waveform, including an intentional zero cross-coupling."""
    t, table = _table(result, name)
    if name in table:
        channels = (_recorded_values(t, table, name, None, frame_frequency)[1],)
    else:
        channels = tuple(_recorded_values(t, table, name, component, frame_frequency)[1]
                         for component in ("d", "q"))
    selected = [_window(t, channel, window)[1] for channel in channels]
    t = _window(t, channels[0], window)[0]
    strongest = selected[int(np.argmax([abs(_phasor(t, s, frequency))
                                        for s in selected]))]
    check = frequency_response(t, strongest, strongest,
                               frequency=frequency, **options)
    return t, check


def measure_port_matrix(experiments: Mapping[float, Sequence[object]], *,
                        voltage: str, current: str, injection: str,
                        kind: str = "impedance", frame: str = "dq",
                        frame_frequency: float = 0.0, window=None,
                        current_sign: float = 1.0, **quality_options) -> MatrixResponseResult:
    """Identify scalar or complete dq/modified-sequence Z/Y from independent runs.

    Each frequency needs enough independent injected directions to span every
    axis. A shunt source's physical ``port.current`` is into the source, so
    ``current_sign=-1`` expresses current entering the measured network.
    """
    if kind not in ("impedance", "admittance"):
        raise ValueError("kind must be 'impedance' or 'admittance'")
    if frame not in ("scalar", "dq", "sequence"):
        raise ValueError("frame must be 'scalar', 'dq', or 'sequence'")
    if (isinstance(current_sign, bool) or not math.isfinite(current_sign)
            or current_sign not in (-1.0, 1.0)):
        raise ValueError("current_sign must be +1 or -1")
    if (isinstance(frame_frequency, bool) or not isinstance(frame_frequency, (int, float))
            or not math.isfinite(frame_frequency)):
        raise ValueError("frame_frequency must be finite")
    if not experiments:
        raise ValueError("experiments must contain at least one frequency")
    frequencies = np.asarray(sorted(experiments), dtype=float)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("frequencies must be positive finite numbers")
    inputs, outputs, valid_runs = [], [], []
    nports = 1 if frame == "scalar" else 2
    for f in frequencies:
        group = tuple(experiments[float(f)])
        if not group:
            raise ValueError(f"frequency {f:g} has no injection experiments")
        currents, voltages, validity = [], [], []
        for result in group:
            t, source_check = _source_quality(result, injection, float(f), window,
                                              frame_frequency, quality_options)
            def columns(name):
                if frame == "scalar":
                    t_raw, scalar = recorded_signal(result, name)
                    fields = (scalar,)
                else:
                    t_raw, fields = _complex_signal(result, name,
                                                   frame_frequency=frame_frequency)
                sliced = [_window(t_raw, field, window)[1] for field in fields]
                selected_times = _window(t_raw, fields[0], window)[0]
                if not np.array_equal(t, selected_times):
                    raise ValueError("source, voltage and current must share recorded sample times")
                return [_phasor(t, field, float(f)) for field in sliced]

            currents.append([current_sign * value for value in columns(current)])
            voltages.append(columns(voltage))
            validity.append(source_check.valid)
        inputs.append(np.asarray(currents) if kind == "impedance" else np.asarray(voltages))
        outputs.append(np.asarray(voltages) if kind == "impedance" else np.asarray(currents))
        valid_runs.append(validity)
    # Each frequency may use its own number of experiments. Fit separately
    # instead of padding invalid measurements with fictional zero injections.
    matrices, quality, flags, conditions = [], [], [], []
    for idx, f in enumerate(frequencies):
        fit = impedance_matrix([float(f)], inputs[idx], outputs[idx],
                               input_names=("scalar",) if nports == 1 else ("d", "q"),
                               output_names=("scalar",) if nports == 1 else ("d", "q"),
                               measurement_valid=[valid_runs[idx]])
        matrices.append(fit.values[0])
        quality.append(fit.valid[0])
        flags.append(fit.flags[0])
        conditions.append({**fit.conditions[0], "frame_frequency_hz": frame_frequency,
                           "current_sign": current_sign, "injection": injection,
                           "voltage": voltage, "current": current,
                           "accepted_source_measurements": sum(valid_runs[idx])})
    matrices = np.asarray(matrices, dtype=complex)
    if frame == "sequence":
        # Modified-sequence coordinates: x+ = (xd + j*xq)/2,
        # x- = (xd - j*xq)/2. Both cross-couplings remain explicit.
        T = np.array([[1, 1j], [1, -1j]]) / 2
        inv = np.array([[1, 1], [-1j, 1j]])
        matrices = np.asarray([T @ matrix @ inv for matrix in matrices])
    names = (("scalar",) if frame == "scalar" else
             ("positive", "negative") if frame == "sequence" else ("d", "q"))
    return MatrixResponseResult(
        frequencies, matrices, kind, "ohm" if kind == "impedance" else "S",
        names, names, tuple(quality), tuple(flags), tuple(conditions), frame,
    )


def _run_cases(params, injection, cases, directory):
    """Build fresh simulations for independent tests; never reuse a prior final state."""
    from ...solver.simulation import Simulation

    if injection not in params.elements or params.elements[injection].type != "port_injection":
        raise ValueError(f"{injection!r} must be a configured port_injection element")
    directory = Path(directory)
    outputs = []
    for index, changes in enumerate(cases):
        overrides = {f"elements.{injection}.{name}": value
                     for name, value in changes.items()}
        overrides["simulation.output.signals"] = True
        configured = params.replace(**overrides)
        # Separate directories avoid replacing earlier run records.
        outputs.append(Simulation(configured).run(out_dir=directory / f"experiment-{index:03d}"))
    return outputs


def run_port_sweep(params, *, injection: str, frequencies: Iterable[float],
                   response: str, directory: str | Path, component: str | None = None,
                   response_component: str | None = None, frame_frequency: float = 0.0,
                   window=None, **quality_options):
    """Run one fresh native circuit simulation per tone and measure its transfer.

    The required ``directory`` retains each run's ordinary CSV and resolved PES
    configuration. Caller provides a settled recording window and a compatible
    injection/response projection (d or q for three-phase signals).
    """
    f = tuple(float(value) for value in frequencies)
    if not f or any(not math.isfinite(value) or value <= 0 for value in f):
        raise ValueError("frequencies must be positive finite values")
    if any(right <= left for left, right in zip(f, f[1:])):
        raise ValueError("frequencies must be strictly increasing")
    if not isinstance(directory, (str, Path)):
        raise ValueError("directory must name a writable output directory")
    records = _run_cases(params, injection, [{"frequency": value} for value in f], directory)
    analyses = [measure_recorded_transfer(
        result, f"{injection}.injection", response,
        frequency=frequency, excitation_component=component,
        response_component=response_component, frame_frequency=frame_frequency,
        window=window, **quality_options)
        for frequency, result in zip(f, records)]
    return frequency_sweep(analyses)


def run_port_matrix(params, *, injection: str, frequencies: Iterable[float],
                    voltage: str, current: str, directory: str | Path,
                    frame: str = "dq", frame_frequency: float = 0.0,
                    **options) -> MatrixResponseResult:
    """Run independent scalar/d/q electrical injections and fit the response matrix."""
    frequencies = tuple(float(f) for f in frequencies)
    if (not frequencies or any(not math.isfinite(f) or f <= 0 for f in frequencies)
            or any(b <= a for a, b in zip(frequencies, frequencies[1:]))):
        raise ValueError("frequencies must be strictly increasing positive Hz values")
    if frame not in ("scalar", "dq", "sequence"):
        raise ValueError("frame must be scalar, dq or sequence")
    axes = ("alpha",) if frame == "scalar" else ("d", "q")
    cases = [{"frequency": f, "axis": axis, "frame_frequency": frame_frequency}
             for f in frequencies for axis in axes]
    records = _run_cases(params, injection, cases, directory)
    n = len(axes)
    groups = {f: records[k*n:(k+1)*n] for k, f in enumerate(frequencies)}
    return measure_port_matrix(groups, voltage=voltage, current=current,
                               injection=f"{injection}.injection", frame=frame,
                               frame_frequency=frame_frequency, **options)


# Feedback return ratio

def _signed_return_ratio(observed: FrequencyMeasurement, feedback_sign: float,
                         before: str, after: str) -> FrequencyMeasurement:
    """Apply one feedback convention to a measured cut transfer."""
    if isinstance(feedback_sign, bool) or feedback_sign not in (-1.0, 1.0):
        raise ValueError("feedback_sign must be +1 or -1")
    value = observed.gain * feedback_sign if observed.gain is not None else None
    return replace(observed, gain=value, conditions={
        **observed.conditions,
        "feedback_sign": feedback_sign,
        "before": before,
        "after": after,
        "definition": "feedback_sign * before / after",
    })


def measure_return_ratio(result, *, before: str, after: str, frequency: float,
                         feedback_sign: float = -1.0,
                         window=None, **quality_options) -> FrequencyMeasurement:
    """Measure a return ratio from signals recorded on both sides of a feedback cut.

    ``feedback_sign=-1`` is for ordinary negative feedback: L = -before/after.
    Do not divide the return by the *added disturbance*: that is a closed-loop
    sensitivity, not the return ratio. The user must verify the edge's sign.
    """
    observed = measure_recorded_transfer(result, after, before, frequency=frequency,
                                         window=window, **quality_options)
    return _signed_return_ratio(observed, feedback_sign, before, after)


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
        return _write(self.to_dict(), path)


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
    # Unwrap only contiguous nonzero valid islands; there is no phase bridge
    # over a missing, zero-gain or unreliable point.
    db = np.full(len(f), math.nan)
    phase = np.full(len(f), math.nan)
    accepted &= np.abs(gains) > 0
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
        db[i:j] = 20 * np.log10(magnitude)
        phase[i:j] = np.rad2deg(np.unwrap(np.angle(island)))
        i = j
    accepted &= np.isfinite(db) & np.isfinite(phase)
    crossing_magnitude = _crossings(np.log10(f), db, 0.0, accepted)
    finite_phase = phase[accepted]
    crossing_phase = []
    if len(finite_phase):
        first = math.ceil((float(np.min(finite_phase)) - 180.0) / 360.0)
        last = math.floor((float(np.max(finite_phase)) - 180.0) / 360.0)
        for branch in range(first, last + 1):
            crossing_phase.extend(
                _crossings(np.log10(f), phase, 180.0 + 360.0 * branch, accepted)
            )
    valid_logf = np.log10(f)[accepted]
    phase_margins = []
    for log_at, _index, _fraction in crossing_magnitude:
        phase_at = float(np.interp(log_at, valid_logf, phase[accepted]))
        phase_margins.append((math.remainder(phase_at + 180.0, 360.0), 10 ** log_at))
    gain_margins = []
    for log_at, _index, _fraction in crossing_phase:
        gain_at = float(np.interp(log_at, valid_logf, db[accepted]))
        gain_margins.append((-gain_at, 10 ** log_at))
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
