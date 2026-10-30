"""Opt-in small-signal analysis of the existing complete simulation loop map."""

from __future__ import annotations

import heapq
import json
import math
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...solver.linearization import (DEFAULT_STEP, active_coordinates, coordinate_difference,
                                     coordinate_width, linearize_loop)
from ...solver.simulation import SystemLoop
from ...solver.timing import TIME_EPS

if TYPE_CHECKING:
    from ...solver.simulation import Simulation

__all__ = ["SmallSignalResult", "SmallSignalReport", "small_signal", "small_signal_report"]


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


def small_signal(sim: Simulation, *, period: float | None = None,
                 relative_step: float = DEFAULT_STEP, periodic_tolerance: float = 1e-3,
                 multiplier_tolerance: float = 1e-5) -> SmallSignalResult:
    """Linearize the completed run's final state through the shared one-period simulation.

    The default period repeats declared control/PWM and source schedules in stationary
    coordinates. Supply ``period`` for an autonomous continuous model or additional custom
    periodic forcing. It must repeat every declared schedule. The state must already be
    approximately periodic; this function does not trim it or replace the configured bridge.

    Complete named-state restart must be supported. No CSV history or new recording is needed.
    The existing SystemLoop uses unsplit reference integration for a multirate configuration.
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
    loop = SystemLoop(sim, period=period, frame_frequency=0.0, repeat_forcing=True)
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
    check_row = loop.periodic_state(row)
    check_cols = loop.coordinates(list(check_row), linearization=False)
    check_active = active_coordinates(check_cols)
    check_names = tuple(name for _kind, group, _scale in check_cols for name in group)
    initial = loop.to_frame(check_row, t0, check_cols)
    baseline_evaluations = int(baseline is None)
    baseline = loop.advance(dict(row), t0) if baseline is None else baseline
    final = loop.to_frame(loop.periodic_state(baseline), t0 + loop.period, check_cols)
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(final)):
        raise ValueError("small_signal requires finite initial and final states")
    floors = np.concatenate([np.full(coordinate_width(kind), scale)
                             for kind, _names, scale in check_cols])
    scales = np.maximum(floors, np.maximum(np.abs(initial), np.abs(final)))
    j = 0
    evolved_modes = []
    for kind, names, scale in check_cols:
        if kind == "angle":
            scales[j] = scale
        elif kind in ("frozen", "mode") and initial[j] != final[j]:
            evolved_modes.append(names[0])
        j += coordinate_width(kind)
    residual = (np.abs(coordinate_difference(final, initial, check_cols))[list(check_active)]
                / scales[list(check_active)])
    worst = int(np.argmax(residual))
    error = float(residual[worst])
    periodic = repeating and not evolved_modes and error <= periodic_tolerance
    if not periodic:
        if not repeating:
            reason = "the analysis period contains a non-repeating simulation-file event"
        elif evolved_modes:
            reason = f"the analysis period changes the discrete state {evolved_modes[0]!r}"
        else:
            reason = (f"the final state is not periodic over {loop.period:.8g} s "
                      f"({check_names[check_active[worst]]}: scaled residual {error:.3g} "
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
    scales_by_name = dict(zip(check_names, scales))
    scale = np.array([scales_by_name[name] for name in coordinate_names])
    amplification = float(np.linalg.norm(phi * scale[None, :] / scale[:, None], ord=2))
    if getattr(loop, "split_reference", False):
        diagnostics.append("The map uses SystemLoop's unsplit reference integration.")
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
    loop: SystemLoop


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
                        watch: Iterable[str] = (), relative_step: float = DEFAULT_STEP,
                        periodic_tolerance: float = 1e-3, multiplier_tolerance: float = 1e-5,
                        change_tolerance: float = 1e-6,
                        out_dir: str | Path | None = "output/run", name: str = "simulation",
                        info: Mapping[str, object] | None = None) -> SmallSignalReport:
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
            loop = SystemLoop(sim, period=period, frame_frequency=0.0, repeat_forcing=True)
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
                     observe_values=reported)
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
