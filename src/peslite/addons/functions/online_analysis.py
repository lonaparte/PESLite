"""Online analysis of ordinary simulations and their recorded responses."""

from __future__ import annotations

import heapq
import json
import math
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from ...defaults import (DEFAULT_LINEARIZATION_RELATIVE_STEP,
                         DEFAULT_STABILITY_MULTIPLIER_TOLERANCE,
                         DEFAULT_STABILITY_PERIODIC_TOLERANCE)
from ...solver.linearization import active_coordinates, compare_loop_states, linearize_loop
from ...solver.simulation import SimulationTransition
from ...solver.timing import TIME_EPS

if TYPE_CHECKING:
    from ...solver.simulation import Simulation

__all__ = [
    "SmallSignalResult", "SmallSignalReport", "small_signal", "small_signal_report",
    "OperatingPointResult", "detect_operating_point",
    "DynamicMetrics", "dynamic_performance",
    "energy_closure",
]


@dataclass(frozen=True)
class SmallSignalResult:
    """A local period-map matrix and modes in the returned named coordinates.

    ``exponents`` uses the principal logarithm: frequency is ambiguous modulo ``1 / period``.
    A zero multiplier has growth rate ``-inf``. ``near-unit-circle`` is deliberately not a
    binary stability verdict and does not discard genuine neutral modes.
    """

    at: float
    period: float
    phi: np.ndarray
    coordinates: tuple[str, ...]
    multipliers: np.ndarray
    periodic_error: float
    evaluations: int
    wall_time: float
    multiplier_tolerance: float
    diagnostics: tuple[str, ...]
    periodic: bool = True
    amplification: float = math.nan

    @property
    def exponents(self) -> np.ndarray:
        values = np.empty(self.multipliers.shape, dtype=complex)
        with np.errstate(divide="ignore"):
            values.real = np.log(np.abs(self.multipliers)) / self.period
        values.imag = np.angle(self.multipliers) / self.period
        return values

    @property
    def stability(self) -> str:
        if not self.periodic:
            return "transient"
        magnitude = np.abs(self.multipliers)
        if np.any(magnitude > 1.0 + self.multiplier_tolerance):
            return "unstable"
        if np.all(magnitude < 1.0 - self.multiplier_tolerance):
            return "stable"
        return "near-unit-circle"

    @property
    def mode_table(self) -> str:
        if not self.periodic:
            return "No stability analysis: a periodic operating state has not been established."
        lines = ["mode  multiplier                  growth (1/s)  frequency (Hz)"]
        for index, (mu, exponent) in enumerate(zip(self.multipliers, self.exponents)):
            lines.append(f"{index:4d}  {mu.real:+.8g}{mu.imag:+.8g}j"
                         f"  {exponent.real:13.6g}  {exponent.imag / math.tau:14.6g}")
        return "\n".join(lines)


def small_signal(
    sim: Simulation, *, period: float | None = None,
    relative_step: float = DEFAULT_LINEARIZATION_RELATIVE_STEP,
    periodic_tolerance: float = DEFAULT_STABILITY_PERIODIC_TOLERANCE,
    multiplier_tolerance: float = DEFAULT_STABILITY_MULTIPLIER_TOLERANCE,
) -> SmallSignalResult:
    """Linearize the completed run's final state through the shared one-period simulation.

    The default period repeats declared control/PWM and source schedules in stationary
    coordinates. Supply ``period`` for an autonomous continuous model or additional custom
    periodic forcing. It must repeat every declared schedule. The state must already be
    approximately periodic; this function does not trim it or replace the configured bridge.

    Complete named-state restart must be supported. No CSV history or new recording is needed.
    The simulation transition uses unsplit reference integration for a multirate configuration.
    """
    _validate_options(relative_step, periodic_tolerance, multiplier_tolerance, period)
    if sim.result is None:
        raise ValueError("small_signal requires a completed Simulation.run()")
    if not sim._rebuildable:
        raise ValueError("small_signal requires rebuildable parts; use parameter factories, not instances")
    if sim.result.summary.get("stop_reason"):
        raise ValueError("small_signal cannot analyze a run that stopped early: "
                         + sim.result.summary["stop_reason"])
    if sim.result.tripped:
        raise ValueError("small_signal cannot analyze a tripped final operating state")
    row = sim.result.final_states()
    t0 = float(row.pop("t"))
    loop = SimulationTransition(sim, period=period, frame_frequency=0.0, repeat_forcing=True)
    if loop.period is None:
        raise ValueError("no common loop period was found; supply a valid period explicitly")
    if any(t0 < change.t <= t0 + loop.period + TIME_EPS for change in sim.p.changes):
        raise ValueError("the analysis period contains a non-repeating simulation-file event")
    if any(getattr(unit.ctrl, "startup", None) is not None and unit.ctrl.startup.in_progress
           for unit in sim.units.values()):
        raise ValueError("the final operating state is still in a startup ramp")

    return _analyze(loop, row, t0, relative_step, periodic_tolerance, multiplier_tolerance)


def _validate_options(relative_step, periodic_tolerance, multiplier_tolerance, period):
    for name, value in (("relative_step", relative_step),
                        ("periodic_tolerance", periodic_tolerance),
                        ("multiplier_tolerance", multiplier_tolerance), ("period", period)):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if period is not None and period <= TIME_EPS:
        raise ValueError("period must exceed the simulation's time resolution")


def _analyze(loop, row, t0, relative_step, periodic_tolerance, multiplier_tolerance, *,
             baseline=None, require_periodic=True, repeating=True):
    """One calculation for final-state and time-resolved analyses; no new execution path."""
    started = time.perf_counter()
    cols = loop.coordinates(list(row))
    active = active_coordinates(cols)
    if not active:
        raise ValueError("the loop has no differentiable state coordinates")
    coordinate_names = tuple(name for kind, group, _scale in cols if kind not in ("frozen", "mode")
                             for name in group)
    baseline_evaluations = int(baseline is None)
    baseline = loop.advance(dict(row), t0) if baseline is None else baseline
    comparison = compare_loop_states(loop, row, baseline, t0, t0 + loop.period)
    error = comparison.error
    periodic = repeating and not comparison.changed_modes and error <= periodic_tolerance
    if not periodic:
        if not repeating:
            reason = "the analysis period contains a non-repeating simulation-file event"
        elif comparison.changed_modes:
            reason = f"the analysis period changes the discrete state {comparison.changed_modes[0]!r}"
        else:
            reason = (f"the final state is not periodic over {loop.period:.8g} s "
                      f"({comparison.worst_coordinate}: scaled residual {error:.3g} "
                      f"> {periodic_tolerance:.3g}); settle the run or supply the correct period")
        if require_periodic:
            raise ValueError(reason)
        return SmallSignalResult(t0, loop.period, np.empty((0, 0)), (),
                                 np.array([], dtype=complex), error, baseline_evaluations,
                                 time.perf_counter() - started, multiplier_tolerance,
                                 (reason,), periodic=False)

    linear = linearize_loop(loop, row, t0, cols, relative_step=relative_step, baseline=baseline)
    if linear.changed_modes:
        raise ValueError("a perturbation changes discrete operating modes: "
                         + ", ".join(linear.changed_modes))
    phi = linear.phi[np.ix_(active, active)]
    multipliers = np.linalg.eigvals(phi).astype(complex)
    order = np.argsort(-np.abs(multipliers), kind="stable")
    diagnostics = ["Frequencies use the principal logarithm and are ambiguous modulo 1 / period."]
    # A single norm convention is used at both window boundaries; it is not an energy norm.
    scales_by_name = dict(zip(comparison.coordinates, comparison.scales))
    scale = np.array([scales_by_name[name] for name in coordinate_names])
    amplification = float(np.linalg.norm(phi * scale[None, :] / scale[:, None], ord=2))
    if getattr(loop, "split_reference", False):
        diagnostics.append("The transition uses unsplit reference integration.")
    return SmallSignalResult(t0, loop.period, phi, coordinate_names, multipliers[order], error,
                             linear.evaluations + baseline_evaluations, time.perf_counter() - started,
                             multiplier_tolerance, tuple(diagnostics), periodic, amplification)


@dataclass
class SmallSignalReport:
    """Sparse, time-ordered changes, plus the cost of explicitly requested analyses.

    ``evaluations`` counts period advances in successful checks; it is ``None`` when a failed
    check could have performed additional advances. No matrices or full state history are kept.
    """

    changes: tuple[dict, ...]
    checks: int
    evaluations: int | None
    reused_baselines: int
    wall_time: float
    files: tuple[Path, ...] = ()

    def write(self, directory: str | Path, *, name: str = "simulation") -> tuple[Path, ...]:
        """Write this analysis as one section of the simulation's JSON report."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        data_path = directory / f"{name}_report.json"
        payload = {"checks": self.checks, "evaluations": self.evaluations,
                   "reused_baselines": self.reused_baselines, "wall_time": self.wall_time,
                   "changes": self.changes}
        data_path.write_text(json.dumps({"small_signal_stability": payload}, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
        self.files = (data_path,)
        return self.files


@dataclass
class _ReportPoint:
    at: float
    row: dict
    loop: SimulationTransition


def _report_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None if math.isnan(value) else str(float(value))
    if value is not None and not isinstance(value, (bool, int, str)):
        raise ValueError("report watch items must be scalar signals, states, or parameter paths")
    return value


def small_signal_report(sim: Simulation, *, times: Iterable[float] = (),
                        interval: float | None = None, period: float | None = None,
                        watch: Iterable[str] = (),
                        relative_step: float = DEFAULT_LINEARIZATION_RELATIVE_STEP,
                        periodic_tolerance: float = DEFAULT_STABILITY_PERIODIC_TOLERANCE,
                        multiplier_tolerance: float = DEFAULT_STABILITY_MULTIPLIER_TOLERANCE,
                        change_tolerance: float = 1e-6,
                        out_dir: str | Path | None = "output/run", name: str = "simulation",
                        info: Mapping[str, object] | None = None, timeout: float | None = None,
                        max_rhs: int | None = None) -> SmallSignalReport:
    """Run a fresh simulation and report only changes at selected analysis times.

    Start, end, and file-event times are always checked. Add explicit ``times`` or an
    ``interval`` measured from the start; neither means every PWM edge. ``watch`` accepts the
    existing watch signal/state names or full parameter paths. Event writes are included
    automatically. Numerical report changes use the requested absolute/relative tolerance.

    Reuse unperturbed main-run endpoints when they were observed at exactly ``t + period``.
    Other endpoints use the existing advance once. Only periodic operating states start
    perturbation runs; earlier checks report transient without a matrix or modal values.
    Unchanged parameters do not justify reusing an old transition matrix. Multirate reference
    maps cannot use endpoints from the split main run.
    """
    _validate_options(relative_step, periodic_tolerance, multiplier_tolerance, period)
    if not math.isfinite(change_tolerance) or change_tolerance < 0.0:
        raise ValueError("change_tolerance must be finite and nonnegative")
    if sim.result is not None:
        raise ValueError("small_signal_report runs a fresh Simulation; it must not have run already")
    if not sim._rebuildable:
        raise ValueError("small_signal_report requires rebuildable parts; use factories, not instances")
    start, end = sim.p.simulation.initial.t, sim.p.simulation.t_end
    selected = set(times)
    if interval is not None:
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("interval must be finite and positive")
        selected.update(start + i * interval for i in range(1, math.ceil((end - start) / interval))
                        if start + i * interval < end)
    if any(not math.isfinite(t) or t < start or t > end for t in selected):
        raise ValueError("analysis times must be finite and within the simulation interval")
    changes = sim.p.changes
    selected.update(change.t for change in changes if start <= change.t <= end)
    selected.update((start, end))
    targets = deque(sorted(selected))
    watched = tuple(watch)
    if any(name.startswith("analysis.") for name in watched):
        raise ValueError("analysis.* is reserved for report metrics, not watched signal aliases")
    records: dict[float, dict] = {}
    waiting: list[tuple[float, int, _ReportPoint]] = []
    was_tripped = False
    reported = tuple(dict.fromkeys([*(path for change in changes if change.t <= end
                                     for path in change.paths), *watched]))
    evaluations, reused = 0, 0
    analysis_wall = 0.0

    def analyze(point, baseline=None):
        nonlocal evaluations, reused, analysis_wall
        entry = records[point.at]
        tick = time.perf_counter()
        try:
            repeating = not any(point.at < change.t <= point.at + point.loop.period + TIME_EPS
                                for change in changes)
            analysis = _analyze(point.loop, point.row, point.at, relative_step, periodic_tolerance,
                                multiplier_tolerance, baseline=baseline,
                                require_periodic=False, repeating=repeating)
            if evaluations is not None:
                evaluations += analysis.evaluations
            reused += int(baseline is not None)
            exponent = analysis.exponents[0] if analysis.periodic else None
            entry.update({"analysis.stability": analysis.stability,
                          "analysis.reason": None if analysis.periodic else analysis.diagnostics[0],
                          "analysis.period": analysis.period,
                          "analysis.periodic_error": analysis.periodic_error,
                          "analysis.amplification": analysis.amplification if analysis.periodic else None,
                          "analysis.growth_rate": float(exponent.real) if exponent is not None else None,
                          "analysis.frequency": (float(abs(exponent.imag) / math.tau)
                                                 if exponent is not None else None)})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            evaluations = None
            entry.update({"analysis.stability": "unavailable", "analysis.reason": str(exc)})
        finally:
            analysis_wall += time.perf_counter() - tick

    def next_endpoint():
        return waiting[0][0] if waiting else None

    def observe(t, states, read_values, *, final, stop_reason):
        nonlocal was_tripped
        while waiting and (final or waiting[0][0] <= t + TIME_EPS):
            endpoint, _index, point = heapq.heappop(waiting)
            analyze(point, states if abs(endpoint - t) <= TIME_EPS else None)
        tripped = any(unit.tripped for unit in sim.units.values())
        due = bool(targets and targets[0] <= t + TIME_EPS)
        if not (due or final or (tripped and not was_tripped)):
            return next_endpoint()
        was_tripped = tripped
        while targets and targets[0] <= t + TIME_EPS:
            targets.popleft()
        entry = dict(read_values())
        records[t] = entry
        if stop_reason:
            entry.update({"analysis.stability": "unavailable", "analysis.reason": stop_reason})
            return None
        if tripped:
            entry.update({"analysis.stability": "unavailable", "analysis.reason": "tripped operating state"})
            return None if final else next_endpoint()
        try:
            loop = SimulationTransition(sim, period=period, frame_frequency=0.0,
                                        repeat_forcing=True)
            if loop.period is None:
                raise ValueError("no common analysis period; supply period explicitly")
        except ValueError as exc:
            entry.update({"analysis.stability": "unavailable", "analysis.reason": str(exc)})
            return None if final else next_endpoint()
        point = _ReportPoint(t, dict(states), loop)
        entry["analysis.integration"] = ("unsplit reference" if loop.split_reference
                                         else "configured solver")
        if not loop.split_reference and t + loop.period <= end:
            heapq.heappush(waiting, (t + loop.period, len(records), point))
        else:
            analyze(point)
        return None if final else next_endpoint()

    result = sim.run(out_dir=out_dir, name=name, info=info, observe=observe, times=selected,
                     observe_values=reported, timeout=timeout, max_rhs=max_rhs)
    previous, timeline = {}, []
    metric_names = sorted({name for entry in records.values() for name in entry
                           if name.startswith("analysis.")})
    for t, entry in sorted(records.items()):
        # Clear values from a preceding valid check when a later check is unavailable.
        entry = {**dict.fromkeys(metric_names), **entry}
        delta = {}
        for key, raw in entry.items():
            value = _report_value(raw)
            old = previous.get(key)
            equal = value == old
            if type(value) in (int, float) and type(old) in (int, float):
                equal = math.isclose(value, old, rel_tol=change_tolerance, abs_tol=change_tolerance)
            if key not in previous or not equal:
                delta[key] = value
                previous[key] = value
        if delta:
            timeline.append({"t": t, "changes": delta})
    report = SmallSignalReport(tuple(timeline), len(records), evaluations, reused, analysis_wall)
    if result.out_dir is not None:
        result.files.extend(report.write(result.out_dir, name=name))
    return report


# Operating-point classification

@dataclass(frozen=True)
class OperatingPointResult:
    """A check of one completed state, with no eigenvalue calculation."""

    at: float
    kind: str  # dc, periodic, or unconverged
    period: float | None
    periodic_error: float | None
    intra_period_error: float | None
    evaluations: int
    diagnostics: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.kind != "unconverged"

    def to_dict(self) -> dict:
        return {
            "at": {"value": self.at, "unit": "s"},
            "kind": self.kind,
            "valid": self.valid,
            "period": None if self.period is None else {"value": self.period, "unit": "s"},
            "periodic_error": (self.periodic_error if self.periodic_error is not None
                               and math.isfinite(self.periodic_error) else None),
            "intra_period_error": (self.intra_period_error if self.intra_period_error is not None
                                   and math.isfinite(self.intra_period_error) else None),
            "evaluations": self.evaluations,
            "diagnostics": list(self.diagnostics),
        }


def _period_error(loop: SimulationTransition, initial: dict, final: dict,
                  start: float, end: float) -> tuple[float, str | None]:
    comparison = compare_loop_states(loop, initial, final, start, end)
    if comparison.changed_modes:
        name = comparison.changed_modes[0]
        return math.inf, f"discrete state {name!r} changes within the check window"
    return comparison.error, comparison.worst_coordinate


def detect_operating_point(sim: Simulation, *, period: float | None = None,
                           tolerance: float = 1e-3) -> OperatingPointResult:
    """Classify the final saved operating point through one period and interior points.

    Unlike :func:`small_signal`, this does not calculate a transition matrix. The
    configured control, PWM, source and forcing schedules determine the period.
    Autonomous models with no schedule use the configured output sample period.
    Interior checks distinguish a static DC state from an orbit that returns
    to the same end state. Failure to establish recurrence is never called stable.
    """
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if period is not None and (isinstance(period, bool) or not math.isfinite(period)
                               or period <= TIME_EPS):
        raise ValueError("period must be finite and exceed the simulation's time resolution")
    if sim.result is None:
        raise ValueError("detect_operating_point requires a completed Simulation.run()")
    row = sim.result.final_states()
    at = float(row.pop("t"))

    def fail(message: str, *, span: float | None = None, error: float | None = None,
             evaluations: int = 0) -> OperatingPointResult:
        return OperatingPointResult(at, "unconverged", span, error, None,
                                    evaluations, (message,))

    if sim.result.summary.get("stop_reason"):
        return fail(f"simulation stopped early: {sim.result.summary['stop_reason']}")
    if sim.result.tripped:
        return fail("operating point is tripped")
    if not sim._rebuildable:
        return fail("operating point cannot be rebuilt from part instances")
    if any(getattr(unit.ctrl, "startup", None) is not None and unit.ctrl.startup.in_progress
           for unit in sim.units.values()):
        return fail("startup ramp has not completed")

    try:
        loop = SimulationTransition(sim, period=period, frame_frequency=0.0, repeat_forcing=True)
        if loop.period is None and not loop.periods:
            sample = float(sim.p.simulation.output.period)
            if not math.isfinite(sample) or sample <= TIME_EPS:
                return fail("no valid autonomous-state comparison interval")
            loop = SimulationTransition(sim, period=sample, frame_frequency=0.0,
                                        repeat_forcing=True)
        if loop.period is None:
            return fail("no common loop period; supply a valid period explicitly")
        span = loop.period
        if any(at < change.t <= at + span + TIME_EPS for change in sim.p.changes):
            return fail("check interval crosses a non-repeating simulation event", span=span)
        final = loop.advance(dict(row), at)
        error, worst = _period_error(loop, row, final, at, at + span)
        if error > tolerance:
            detail = (f"nonrepeating state {worst!r}: scaled residual {error:.3g} "
                      f"exceeds {tolerance:.3g}") if math.isfinite(error) else str(worst)
            return fail(detail, span=span, error=error, evaluations=1)
        # Interior advances need not repeat a schedule: their purpose is only
        # to reveal motion hidden by identical full-period endpoints. A quarter
        # and a half period avoid mistaking a zero crossing for a static state.
        interior = 0.0
        for fraction in (0.25, 0.5):
            loop.period = span * fraction
            try:
                observed = loop.advance(dict(row), at)
            finally:
                loop.period = span
            deviation, _ = _period_error(loop, row, observed, at, at + span * fraction)
            interior = max(interior, deviation)
        midway = interior
        if midway <= tolerance:
            kind = "dc"
        else:
            kind = "periodic"
        return OperatingPointResult(at, kind, span, error, midway, 3)
    except (ValueError, FloatingPointError, ArithmeticError, np.linalg.LinAlgError) as exc:
        return fail(str(exc))


# Time-domain response metrics

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


# Energy closure

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
