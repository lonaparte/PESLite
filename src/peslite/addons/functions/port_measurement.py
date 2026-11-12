"""Opt-in electrical-port identification from native, time-aligned simulation records.

The network solver performs every injection. This module only reads the ordinary
CSV-backed result columns; it neither replaces circuit equations nor creates an
alternative frequency-response estimator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .frequency import (_complex, _fit, frequency_response, frequency_sweep,
                        impedance_matrix)

__all__ = ["recorded_signal", "measure_recorded_transfer", "measure_recorded_sweep",
           "measure_multitone", "measure_port_matrix", "PortMatrixResult"]


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


def recorded_signal(result, name: str, *, component: str | None = None,
                    frame_frequency: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Read one scalar or three-phase signal without inventing another sample history.

    Three-phase sources use their standard ``_a/_b/_c`` columns. Set component
    to ``alpha``, ``beta``, ``d`` or ``q``; d/q rotates the space vector by the
    specified frame frequency. Scalar signals require ``component=None``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a nonempty recorded signal path")
    if isinstance(frame_frequency, bool) or not math.isfinite(frame_frequency):
        raise ValueError("frame_frequency must be finite")
    t, table = _table(result, name)
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


@dataclass(frozen=True)
class PortMatrixResult:
    """SI complex multiport response with explicit axes, frame and validity."""

    frequencies_hz: np.ndarray
    values: np.ndarray  # [frequency, output component, input component]
    kind: str
    frame: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    valid: tuple[bool, ...]
    flags: tuple[tuple[str, ...], ...]
    conditions: tuple[dict, ...]

    def to_dict(self):
        return {
            "frequency": {"value": self.frequencies_hz.tolist(), "unit": "Hz"},
            self.kind: {"value": [[[_complex(z) for z in row] for row in matrix]
                                  for matrix in self.values],
                        "unit": "ohm" if self.kind == "impedance" else "S",
                        "frame": self.frame, "inputs": list(self.input_names),
                        "outputs": list(self.output_names)},
            "validity": [{"valid": valid, "flags": list(flags), "conditions": dict(info)}
                         for valid, flags, info in zip(self.valid, self.flags, self.conditions)],
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), allow_nan=False, indent=2) + "\n", encoding="utf-8")
        return path


def _complex_signal(result, name, *, frame_frequency):
    t, d = recorded_signal(result, name, component="d", frame_frequency=frame_frequency)
    tq, q = recorded_signal(result, name, component="q", frame_frequency=frame_frequency)
    if not np.array_equal(t, tq):
        raise ValueError("the d and q components must share a sampling clock")
    return t, (d, q)


def _source_quality(result, name, frequency, window, frame_frequency, options):
    """Validate the imposed waveform, including an intentional zero cross-coupling."""
    try:
        t, channels = _complex_signal(result, name, frame_frequency=frame_frequency)
    except ValueError as exc:
        if "three-phase columns" not in str(exc):
            raise
        t, signal = recorded_signal(result, name)
        channels = (signal,)
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
                        current_sign: float = 1.0, **quality_options) -> PortMatrixResult:
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
    if not math.isfinite(frame_frequency):
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
    return PortMatrixResult(frequencies, matrices, kind, frame, names, names,
                            tuple(quality), tuple(flags), tuple(conditions))


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
                    **options) -> PortMatrixResult:
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


__all__ += ["run_port_sweep", "run_port_matrix"]
