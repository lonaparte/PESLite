"""Simulation event loop and command-line interface (run through ``peslite.py``).

Order at a coincident instant: over-current check, ADC samples, loop updates, PWM publication,
window opening, switching instants, snapshot.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .phs.protocols import Solver
from .phs.solvers import make_solver
from .phs.splitbound import split_error_bound
from .phs.states import expand_aliases, flatten, gather, resolve, scatter
from .params import ConfigError, Params, load
from .assembly.protocols import Controller, Delay, Modulator
from .results import Recorder, SimulationResult
from .firmware.transforms import abc2complex, complex2abc

from .assembly.system import System
from .assembly.unit import Unit

__all__ = ["Simulation", "main"]


_EPS = 1e-10  # seconds; intervals shorter than this are not integrated separately


def _finite(y: np.ndarray) -> bool:
    """Return True when every entry of ``y`` is finite."""
    return all(map(math.isfinite, y.tolist()))


class _UnitLoop:
    """Event-loop bookkeeping for one converter unit."""

    __slots__ = ("unit", "name", "ctrl", "delay", "modulator", "T_s", "n_samples", "T_samp", "k", "d",
                 "sync", "peeks", "n_samp", "schedule", "next_switch", "start", "period_end", "tripped",
                 "T_avg", "window_open", "latest_meas")

    def __init__(self, unit: Unit, k0: int) -> None:
        self.unit, self.name = unit, unit.name
        self.ctrl, self.delay, self.modulator = unit.ctrl, unit.delay, unit.modulator
        self.T_s = unit.T_s
        self.n_samples = unit.samples_per_update
        self.T_samp = unit.T_samp
        self.k = k0
        self.d = np.zeros(3)
        self.sync: tuple[float | None, float | None] = (
            unit.ctrl.initial_sync() if hasattr(unit.ctrl, "initial_sync") else (None, None))
        self.peeks: list[Any] = []
        self.n_samp = 1  # sample 0 of a period is its control instant
        self.T_avg = unit.T_avg  # ADC averaging window (s); None for instantaneous sampling
        # a full-period window opens at the control instant itself
        self.window_open = self.T_avg is None or self.T_avg >= unit.T_s * (1.0 - 1e-12)
        self.schedule: list[tuple[float, Any]] = []  # remaining switching instants of this period
        self.next_switch = math.inf
        self.start = k0 * unit.T_s  # start of the current period
        self.period_end = math.inf
        self.tripped = bool(getattr(unit.ctrl, "tripped", False)) or unit.breaker_open
        self.latest_meas = None

    @property
    def t_control(self) -> float:
        return self.k * self.T_s

    @property
    def t_window(self) -> float:
        """Opening time of the next averaging window; ``inf`` when none is pending."""
        if self.window_open:
            return math.inf
        return self.t_control - self.T_avg

    @property
    def t_sample(self) -> float:
        return (self.start + self.n_samp * self.T_samp) if self.n_samp < self.n_samples else math.inf


class Simulation:
    """Assemble the system and solver from parameters and run the event loop once."""

    def __init__(self, p: Params, system: Any = None, solver: Solver | None = None,
                 parts: Mapping[str, Callable[..., Any]] | None = None,
                 instances: Mapping[str, Any] | None = None) -> None:
        """Build every part from ``p`` unless replaced through ``parts`` or ``instances``.

        parts: factories keyed ``"system"`` or ``"solver"`` (called with ``p``), or ``"<unit>"``,
        ``"<unit>.ctrl"``, ``"<unit>.modulator"``, ``"<unit>.delay"`` (called with the unit's section).
        instances: ready objects under the same keys (also ``system=``, ``solver=``); with any
        instance the split error bound is not computed.
        """
        self.p = p
        self._factories: dict[str, Callable[..., Any]] = dict(parts or {})
        given: dict[str, Any] = dict(instances or {})
        if system is not None:
            given["system"] = system
        if solver is not None:
            given["solver"] = solver
        known = {"system", "solver"}
        for name in p.units:
            known |= {name, f"{name}.ctrl", f"{name}.modulator", f"{name}.delay"}
        for label, what in (("parts", self._factories), ("instances", given)):
            unknown = set(what) - known
            if unknown:
                raise ConfigError(f"{label}: unknown part(s) {sorted(unknown)}; known: {sorted(known)}")
        both = set(self._factories) & set(given)
        if both:
            raise ConfigError(f"parts: {sorted(both)} given both as a factory and as an instance")
        self._rebuildable = not given

        unit_parts = {}
        for name, cfg in p.units.items():
            for key in (name, f"{name}.ctrl", f"{name}.modulator", f"{name}.delay"):
                if key in self._factories:
                    unit_parts[key] = self._factories[key](cfg)
                elif key in given:
                    unit_parts[key] = given[key]
        if "system" in self._factories:
            self.system = self._factories["system"](p)
        elif "system" in given:
            self.system = given["system"]
        else:
            self.system = System(p, parts=unit_parts)
        if "solver" in self._factories:
            self.solver = self._factories["solver"](p)
        elif "solver" in given:
            self.solver = given["solver"]
        else:
            self.solver = make_solver(p.simulation.solver, model=self.system.model,
                                      ratings=self._ratings())
        for unit in self.system.units.values():
            for obj, proto in ((unit.ctrl, Controller), (unit.modulator, Modulator), (unit.delay, Delay)):
                if not isinstance(obj, proto):
                    raise TypeError(f"{unit.name}: {type(obj).__name__} does not satisfy the "
                                    f"{proto.__name__} protocol")
        if not isinstance(self.solver, Solver):
            raise TypeError(f"{type(self.solver).__name__} does not satisfy the Solver protocol")
        self.result: SimulationResult | None = None
        self.energy_problems: list[str] = []
        self.ph_report = None  # port-Hamiltonian structure report
        self._check_energy()

    @property
    def units(self) -> dict[str, Unit]:
        return self.system.units

    def unit(self, name: Optional[str] = None) -> Unit:
        """Return the named unit, or the only unit when ``name`` is omitted."""
        if name is None:
            if len(self.system.units) != 1:
                raise ConfigError(f"this system has {len(self.system.units)} units "
                                  f"{sorted(self.system.units)}: name the one you mean")
            name = next(iter(self.system.units))
        return self.system.units[name]

    def _ratings(self):
        """Return a function giving the rated (effort, flow) of a storage, or ``None``."""
        base = self.p.base
        system_ac = (base.v_phase_peak, base.i_phase_peak)
        units = self.p.units

        def rating(name: str, storage: Any) -> Optional[tuple[float, float]]:
            cfg = units.get(name.partition(".")[0])
            if storage.scale == 1.0:  # dc quantities: that unit's dc base
                if cfg is None:
                    return None
                dc = cfg.dc_base
                return (dc.v, dc.i) if dc.v > 0 else None
            return (cfg.base.v_phase_peak, cfg.base.i_phase_peak) if cfg is not None else system_ac

        return rating

    def _check_energy(self) -> None:
        """Build the structure report and apply ``simulation.energy_check`` (warn, strict or off)."""
        mode = self.p.simulation.energy_check
        model = getattr(self.system, "model", None)
        if model is None or not hasattr(model, "ph_report"):
            return
        groups = None
        solver = self.solver
        if hasattr(solver, "inner_names") and hasattr(solver, "outer_names"):
            assignment = {n: "inner" for n in solver.inner_names}
            assignment.update({n: "outer" for n in solver.outer_names})
            groups = model.groups(assignment) if assignment else None
        hold = None
        if hasattr(solver, "window") and hasattr(solver, "dt"):
            hold = {"step": solver.dt, "window": solver.window}
        zoh = {unit.zoh: 0.7 + 0.2j for unit in self.system.units.values()}
        self.ph_report = report = model.ph_report(zoh=zoh, groups=groups, hold=hold)
        problems = list(report.problems)
        if report.defaulted:
            problems.append(f"default energy declaration (black box, net power inferred from the neighbours) "
                            f"applied to: {report.defaulted}")
        bad_cuts = [c for c in report.cuts if not c.admissible]
        if bad_cuts:
            problems.append("subsystems stepped on their own whose interface cannot be accounted for: "
                            + "; ".join(f"{c.group} {c.members} holds {c.unaccounted}" for c in bad_cuts))
        problems += report.warnings
        self.energy_problems = problems
        if mode == "off":
            return
        if problems and mode == "strict":
            raise ConfigError("energy check: " + "; ".join(problems))
        for msg in problems:
            warnings.warn("energy check: " + msg, stacklevel=3)

    # ------------------------------------------------------------ states
    def _state(self, duty: Mapping[str, np.ndarray]) -> dict[str, Any]:
        s = gather({"plant": self.system})
        for name, unit in self.system.units.items():
            s.update(gather({f"ctrl.{name}": unit.ctrl}))
            for k, ph in enumerate("abc"):
                s[f"pwm.{name}.d_{ph}"] = float(duty[name][k])
            s.update(gather({f"delay.{name}": unit.delay}))
            if unit.window is not None:
                s.update(gather({f"meas.{name}": unit.window}))
        s.update(gather({"solver": self.solver}))
        return s

    def _initial_duty(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(unit.ctrl.initial_duty(), dtype=float)
                for name, unit in self.system.units.items()}

    def state_names(self) -> list[str]:
        """Return the state-table column names after ``t``, i.e. the valid ``initial.states`` keys."""
        if self.result is None:
            for unit in self.system.units.values():
                unit.delay.reset(unit.ctrl.initial_duty())
        return list(flatten(self._state(self._initial_duty())))

    def _apply_initial(self, t0: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load ``initial.states`` into the parts; return the solver vector and the first duty ratios."""
        system = self.system
        aliases = {f"plant.{a}": f"plant.{c}" for a, c in getattr(system, "state_aliases", {}).items()}
        presets_of = getattr(system, "state_presets", None)
        presets = presets_of(t0) if presets_of is not None else {}
        if callable(presets):  # state-dependent keywords: strip the "plant." prefix
            inner = presets
            presets = lambda key, word: inner(key[6:] if key.startswith("plant.") else key, word)  # noqa: E731
        for unit in system.units.values():
            unit.delay.reset(unit.ctrl.initial_duty())  # fill the pipeline so its states exist
        template = self._state(self._initial_duty())
        try:
            values = resolve(template, expand_aliases(self.p.initial.states, aliases), presets)
        except (KeyError, ValueError) as exc:
            msg = exc.args[0] if exc.args else str(exc)
            if aliases:
                msg += f". Aliases: {', '.join(f'{a} -> {c}' for a, c in aliases.items())}"
            raise ConfigError(msg) from None
        # 1. power stage (unset states stay zero), then the solver vector
        scatter({"plant": system}, values)
        y = system.model.get_initial_values()
        # 2. start-up modulation matching each unit's terminal voltage at t0
        system.model.sync(t0, y)
        duty: dict[str, np.ndarray] = {}
        for name, unit in system.units.items():
            align = getattr(unit.ctrl, "align_startup", None)
            if align is not None:
                align(unit.ports.u_g(), unit.ports.u_dc())
            unit.seed_window()  # averaging window starts at t0
            pwm = np.asarray(unit.ctrl.initial_duty(), dtype=float).copy()
            for k, ph in enumerate("abc"):
                if f"pwm.{name}.d_{ph}" in values:
                    pwm[k] = values[f"pwm.{name}.d_{ph}"]
            duty[name] = pwm
        # 3. controllers, delay pipelines, measurement windows, solver
        for name, unit in system.units.items():
            scatter({f"ctrl.{name}": unit.ctrl}, values)
            unit.delay.reset(duty[name])
            scatter({f"delay.{name}": unit.delay}, values)
            if unit.window is not None:
                scatter({f"meas.{name}": unit.window}, values)
        scatter({"solver": self.solver}, values)
        return y, duty

    # ------------------------------------------------------------ the loop
    def run(self, t_end: float | None = None) -> SimulationResult:
        if self.result is not None:
            raise RuntimeError("a Simulation runs once; build a new one for another run")
        p, system = self.p, self.system
        mdl, solver = system.model, self.solver
        t_end = p.simulation.t_end if t_end is None else t_end
        log_period = p.simulation.log.plant_period
        control_every = max(1, p.simulation.log.control_every)
        stop_on_trip = p.simulation.stop_on_trip
        progress_every = p.simulation.progress_every
        rec = Recorder(keep_states=p.output.states)
        wall0 = time.time()

        periods = [u.T_s for u in system.units.values()]
        t_start = min(round(p.initial.t / T) * T for T in periods)
        if t_end <= t_start + _EPS:
            raise ValueError(f"t_end = {t_end} must be after the start time {t_start}")
        # the run ends at the earliest period boundary at or after t_end
        t_final = min(math.ceil((t_end - _EPS) / T) * T for T in periods)
        loops = [_UnitLoop(u, int(round(t_start / u.T_s))) for u in system.units.values()]
        averaging = [lp for lp in loops if lp.unit.window is not None]  # units whose ADC averages
        for loop in loops:
            if hasattr(loop.ctrl, "reset_clocks"):
                loop.ctrl.reset_clocks(t_start)
        y, duty = self._apply_initial(t_start)
        for loop in loops:
            loop.d = duty[loop.name]
            loop.latest_meas = loop.unit.measure(t_start)
        n_log = max(0, math.ceil(t_start / log_period - 1e-9))
        next_report = t_start + progress_every if progress_every > 0 else math.inf
        t_local = t_start
        n_rhs0 = getattr(solver, "n_rhs", 0)

        mode = p.simulation.energy_check
        energy_on = mode != "off" and hasattr(mdl, "energy_balance")
        worst = {"tellegen": 0.0, "balance": 0.0}
        stop_reason = ""
        coarse_reported = False
        stopped = False

        def snapshot(t_now: float) -> None:
            try:
                with np.errstate(over="raise"):
                    mdl.sync(t_now, y)
                    rec.plant_snapshot(t_now, system.signals(t_now))
                    rec.state_row(t_now, flatten(self._state({lp.name: lp.d for lp in loops})))
                    if energy_on and _finite(y):
                        rep = mdl.energy_balance(t_now, y)
                        rec.energy_row(t_now, rep.columns())
                        scale = rep.scale
                        worst["tellegen"] = max(worst["tellegen"], abs(rep.tellegen) / scale)
                        worst["balance"] = max(worst["balance"], rep.max_residual / scale)
            except (OverflowError, FloatingPointError):
                pass  # diverged state: not recorded

        @np.errstate(over="raise")  # overflow counts as divergence
        def integrate(t_a: float, t_b: float, y: np.ndarray) -> np.ndarray:
            return solver(mdl.rhs, t_a, t_b, y).y

        def advance(t_a: float, t_b: float) -> bool:
            """Integrate over ``[t_a, t_b]``; return True when the run must stop (divergence, strict coarse window)."""
            nonlocal y, t_local, stop_reason, coarse_reported
            try:
                y = integrate(t_a, t_b, y)
            except (OverflowError, FloatingPointError):
                stop_reason = f"diverged: the states overflowed between t = {t_a:.6g} s and {t_b:.6g} s"
                t_local = t_a
                return True
            t_local = t_b
            finite = _finite(y)
            if averaging and finite:
                mdl.sync(t_b, y)
                for lp in averaging:
                    lp.unit.accumulate(t_b - t_a)
            if not finite:
                stop_reason = f"diverged: non-finite states at t = {t_b:.6g} s"
                return True
            coarse = getattr(solver, "coarse_hold", None)
            if coarse is not None and not coarse_reported:
                coarse_reported = True
                t_c, label, rel = coarse
                msg = (f"the window holding {label} (closed at t = {t_c:.6g} s) bounds its change by {rel:.2f} of "
                       f"the rated effort: the window is too long for this storage")
                if mode == "strict":
                    stop_reason = "coarse window: " + msg
                    return True
                if mode == "warn":
                    warnings.warn("energy check: " + msg, stacklevel=3)
            return False

        def do_trip(loop: _UnitLoop, t_now: float) -> None:
            nonlocal y
            loop.tripped = True
            mdl.set_states(y)
            loop.unit.trip()
            y = mdl.get_initial_values()

        def control_step(loop: _UnitLoop, t_k: float) -> bool:
            """Sample, run the controller and advance the delay; return True if a new trip stops the run."""
            unit = loop.unit
            mdl.sync(t_k, y)
            meas = unit.measure(t_k)
            loop.latest_meas = meas
            if loop.T_avg is not None:  # rearm the averaging window
                loop.window_open = loop.T_avg >= loop.T_s * (1.0 - 1e-12)
                if loop.window_open:    # full-period window opens now
                    unit.open_window()
            if loop.n_samples > 1:  # oversampling: this update's samples, oldest first
                meas = dataclasses.replace(meas, samples=tuple(loop.peeks) + (meas,))
            loop.peeks, loop.n_samp = [], 1
            out = loop.ctrl(t_k, meas)
            if out.log is not None:
                rec.last_ctrl_log[loop.name] = out.log
                if loop.k % control_every == 0:
                    rec.control_sample(loop.name, t_k, out.log)
            new_trip = out.tripped and not loop.tripped
            if new_trip:
                do_trip(loop, t_k)
            loop.d = loop.delay(out.d_abc)
            loop.sync = (out.theta, out.omega)
            return new_trip and stop_on_trip

        def schedule(loop: _UnitLoop, t_k: float) -> None:
            """Get the next period's switching sequence from the modulator and schedule its instants."""
            seq = loop.modulator(t_k, loop.T_s, loop.d, *loop.sync)
            dt = np.asarray(seq.dt, dtype=float).tolist()
            n = len(dt)
            loop.start, loop.period_end = t_k, t_k + loop.T_s
            times, t = [], t_k
            for i in range(n - 1):
                t = t + dt[i]
                times.append(t)
            # switching states as Python complex numbers
            q =[abc2complex(row) for row in np.asarray(seq.q_abc, dtype=float).tolist()]
            loop.schedule = [(times[i], q[i + 1]) for i in range(n - 1)]
            mdl.set_zoh_input(loop.unit.zoh, q[0])
            loop.next_switch = loop.schedule[0][0] if loop.schedule else math.inf

        # ---------------------------------------------------------------- the event grid
        for loop in loops:  # first period: start-up duty ratios, no controller call
            schedule(loop, loop.t_control)
            loop.k += 1
        while True:
            t_log = n_log * log_period
            t_stop = t_log if t_log < t_final - _EPS else t_final  # the earliest event of all
            for lp in loops:
                t_c = lp.t_control
                t_stop = min(t_stop, t_c if t_c < t_final - _EPS else t_final, lp.next_switch,
                             lp.t_window, lp.t_sample, getattr(lp.ctrl, "next_event", math.inf))
            if t_stop > t_local + _EPS and advance(t_local, t_stop):
                stopped = True
                snapshot(t_local)
                break
            # 0. over-current check of units whose switching interval ends here
            for loop in loops:
                ends = abs(loop.next_switch - t_stop) < _EPS or abs(loop.period_end - t_stop) < _EPS
                check = getattr(loop.ctrl, "fast_check", None)
                if ends and check is not None and not loop.tripped:
                    mdl.set_states(y)
                    if check(t_local, loop.unit.phase_currents()):
                        do_trip(loop, t_local)
                        if stop_on_trip:
                            stopped = True
            if stopped:
                snapshot(t_local)
                break
            # ADC samples due here, before any loop update
            for loop in loops:
                if abs(loop.t_sample - t_stop) < _EPS:
                    mdl.sync(t_local, y)
                    sample = loop.unit.measure(t_local)
                    loop.peeks.append(sample)
                    loop.latest_meas = sample
                    loop.n_samp += 1
            # loop-only updates: latest sample, no PWM/delay advance
            for loop in loops:
                if (getattr(loop.ctrl, "next_event", math.inf) <= t_stop + _EPS
                        and abs(loop.t_control - t_stop) >= _EPS and t_stop < t_final - _EPS):
                    loop.ctrl.update(t_stop, loop.latest_meas)
            # 1. PWM publications due here (with any due loop updates)
            for loop in loops:
                if abs(loop.t_control - t_stop) < _EPS and t_stop < t_final - _EPS:
                    if control_step(loop, t_stop):
                        stopped = True
                    schedule(loop, t_stop)
                    loop.k += 1
            if stopped:
                snapshot(t_local)
                break
            # 1b. an averaging window shorter than the PWM update period opens here
            for loop in loops:
                if abs(loop.t_window - t_stop) < _EPS:
                    mdl.sync(t_local, y)
                    loop.unit.open_window()
                    loop.window_open = True
            # 2. switching instants due here
            for loop in loops:
                while loop.schedule and abs(loop.schedule[0][0] - t_stop) < _EPS:
                    mdl.set_zoh_input(loop.unit.zoh, loop.schedule.pop(0)[1])
                loop.next_switch = loop.schedule[0][0] if loop.schedule else math.inf
            # 3. snapshots
            if abs(t_log - t_stop) < _EPS and t_stop < t_final - _EPS:
                snapshot(t_local)
                n_log += 1
            if t_stop >= t_final - _EPS:
                break
            if t_local >= next_report:
                print(f"  t = {t_local:8.4f} s   wall {time.time() - wall0:8.1f} s", flush=True)
                next_report += progress_every

        if not stopped:
            # final instant: only the loop/PWM events due here
            for loop in loops:
                if abs(loop.t_control - t_local) < _EPS:
                    control_step(loop, t_local)
                elif getattr(loop.ctrl, "next_event", math.inf) <= t_local + _EPS:
                    loop.ctrl.update(t_local, loop.latest_meas)
        if not rec.t or rec.t[-1] < t_local - _EPS:
            snapshot(t_local)
        if stop_reason:
            warnings.warn(f"simulation stopped at t = {t_local:.6g} s: {stop_reason}", stacklevel=2)
        t_arr, plant_arrays, ctrl_arrays, state_arrays, energy_arrays = rec.arrays()
        summary: dict[str, Any] = {}
        for name, unit in system.units.items():
            if hasattr(unit.ctrl, "summary"):
                summary.update({f"{name}.{k}": v for k, v in unit.ctrl.summary().items()})
        summary["tripped"] = float(any(lp.tripped for lp in loops))
        summary["t_start_s"] = t_start
        summary["t_stop_s"] = float(t_arr[-1]) if len(t_arr) else t_start
        if stop_reason:
            summary["stop_reason"] = stop_reason
        coarse = getattr(solver, "coarse_hold", None)
        if coarse is not None:
            summary["interface_coarse_hold"] = f"{coarse[1]} at t = {coarse[0]:.6g} s: {coarse[2]:.3g} of rating"
        if self.ph_report is not None:
            summary["ph_verdict"] = self.ph_report.verdict
            summary["ph_defaulted"] = "|".join(self.ph_report.defaulted) if self.ph_report.defaulted else "none"
            summary["ph_report"] = self.ph_report.to_dict()
        if energy_on:
            summary["energy_tellegen_max_rel"] = worst["tellegen"]
            summary["energy_balance_max_rel"] = worst["balance"]
            summary["energy_check"] = "|".join(self.energy_problems) if self.energy_problems else "ok"
        interface = getattr(solver, "interface", None)
        if interface:  # split-interface indicators
            summary.update({f"interface_{k}": v for k, v in interface.items()})
        if getattr(solver, "window_log", None) and p.simulation.solver.linearisations > 0:
            if self._rebuildable:
                loop = SystemLoop(self)
                if loop.period is None:  # no common control period
                    summary["split_bound"] = (f"the converters' control periods {loop.periods} have no "
                                              f"common multiple within 64 updates, so the loop has no "
                                              f"period to linearise: no bound")
                else:
                    summary.update(split_error_bound(
                        solver, loop, t_arr, state_arrays, p.simulation.solver.linearisations,
                        labels=mdl.state_labels(), prefix="plant."))
            else:  # parts given as instances
                summary["split_bound"] = ("the loop has custom parts given as instances, so it cannot be "
                                          "rebuilt for the linearisation: pass them as factories of the "
                                          "parameter tree for a bound")
        self.result = SimulationResult(
            params=p, t=t_arr, plant=plant_arrays, control=ctrl_arrays, states=state_arrays,
            energy=energy_arrays, summary=summary, wall_time=time.time() - wall0,
            n_rhs=getattr(solver, "n_rhs", 0) - n_rhs0)
        return self.result


_LOOPMAP_EPS = 1e-9

def macro_period(periods: list[float], limit: int = 64) -> float | None:
    """Return the smallest common multiple of ``periods`` (s), or ``None`` beyond ``limit`` longest periods."""
    longest = max(periods)
    for n in range(1, limit + 1):
        T = n * longest
        if all(abs(T / p - round(T / p)) <= 1e-9 * max(1.0, T / p) for p in periods):
            return T
    return None


class SystemLoop:
    """Closed-loop map of the whole system over one common control period, rebuilt from the parameters."""

    def __init__(self, sim: Any) -> None:

        self._cls = Simulation
        self._factories = dict(sim._factories)
        self.periods = [float(u.T_s) for u in sim.units.values()]
        for u in sim.units.values():
            self.periods.extend(T for T in getattr(u.ctrl, "periods", {}).values() if T)
        self.period = macro_period(self.periods)
        self.w0 = 2.0 * math.pi * sim.p.base.f0
        self._flags = {k for k, v in sim._state(sim._initial_duty()).items() if isinstance(v, bool)}
        rated = sim.solver.rated_effort
        self._rated = ({lab: float(r) for lab, r in zip(sim.system.model.state_labels(), rated)}
                       if rated is not None else {})
        p = sim.p
        sp = p.simulation
        self._p = dataclasses.replace(
            p,
            simulation=dataclasses.replace(sp, energy_check="off", stop_on_trip=False, progress_every=0.0,
                                           log=dataclasses.replace(sp.log, plant_period=self.period or 1.0),
                                           solver=dataclasses.replace(sp.solver, subsystems={},
                                                                      linearisations=0)),
            output=dataclasses.replace(p.output, states=True, energy=False, signals=False))

    # ------------------------------------------------------------ coordinates
    def coordinates(self, names: list[str]) -> list[tuple[str, list[str], float]]:
        """Group state-table columns into map coordinates ``(kind, names, scale)``."""
        cols: list[tuple[str, list[str], float]] = []
        used: set[str] = set()
        for c in names:
            if c in used or c == "t" or c.startswith("solver.") or ".clock." in c:
                continue  # solver bookkeeping, not a loop state
            if c in self._flags:
                cols.append(("frozen", [c], 1.0))
                used.add(c)
            elif c.endswith(".re") and c[:-3] + ".im" in names:
                base = c[:-3]
                turns = base.startswith("plant.")  # alpha-beta; a controller vector is already in dq
                scale = self._rated.get(base[len("plant."):] + ".re", 0.0) if turns else 0.0
                cols.append(("vector" if turns else "fixed", [c, base + ".im"], scale if scale > 0 else 1.0))
                used.update(cols[-1][1])
            elif c.endswith(".d_a") and c[:-4] + ".d_b" in names and c[:-4] + ".d_c" in names:
                cols.append(("triple", [c[:-4] + f".d_{ph}" for ph in "abc"], 1.0))
                used.update(cols[-1][1])
            else:
                scale = self._rated.get(c[len("plant."):], 0.0) if c.startswith("plant.") else 0.0
                cols.append(("scalar", [c], scale if scale > 0 else 1.0))
                used.add(c)
        return cols

    def to_frame(self, row: dict[str, float], t: float, cols: list) -> np.ndarray:
        theta = self.w0 * t
        rot = complex(math.cos(theta), -math.sin(theta))
        out: list[float] = []
        for kind, names, _s in cols:
            if kind == "vector":
                z = complex(row[names[0]], row[names[1]]) * rot
                out += [z.real, z.imag]
            elif kind == "fixed":
                out += [row[names[0]], row[names[1]]]
            elif kind == "triple":
                d = np.array([row[k] for k in names])
                z = abc2complex(d) * rot
                out += [z.real, z.imag, float(d.mean())]
            else:
                out.append(row[names[0]])
        return np.array(out)

    def from_frame(self, x: np.ndarray, t: float, cols: list, row: dict[str, float]) -> dict[str, float]:
        theta = self.w0 * t
        rot = complex(math.cos(theta), math.sin(theta))
        out = dict(row)
        i = 0
        for kind, names, _s in cols:
            if kind == "vector":
                z = complex(x[i], x[i + 1]) * rot
                out[names[0]], out[names[1]] = z.real, z.imag
            elif kind == "fixed":
                out[names[0]], out[names[1]] = float(x[i]), float(x[i + 1])
            elif kind == "triple":
                d = complex2abc(complex(x[i], x[i + 1]) * rot) + x[i + 2]
                for k, name in enumerate(names):
                    out[name] = float(d[k])
            elif kind == "scalar":
                out[names[0]] = float(x[i])
            i += 2 if kind in ("vector", "fixed") else 3 if kind == "triple" else 1
        return out

    # --------------------------------------------------------- one period of it
    def advance(self, row: dict[str, float], t: float,
                view: tuple[str, float, str] | None = None) -> dict[str, float]:
        """Return the state row one period after ``t``.

        view: optional ``(plant state label, offset, channel)``; the offset is seen by the
        integration (``"b"``) or by the end-of-period sample (``"c"``).
        """
        p = dataclasses.replace(self._p, simulation=dataclasses.replace(
            self._p.simulation, t_end=t + self.period), initial=dataclasses.replace(
            self._p.initial, t=t, states={k: v for k, v in row.items() if not k.startswith("solver.")}))
        sim = self._cls(p, parts=self._factories)
        model = sim.system.model
        if view is not None:
            label, off, channel = view
            vec = np.zeros(model.n_states)
            vec[list(model.state_labels()).index(label)] = off
            t_end = t + self.period
            if channel == "b":  # what the integration sees
                rhs0 = model.rhs
                model.rhs = lambda tt, y: rhs0(tt, y + vec)  # type: ignore[method-assign]
            else:  # what the end-of-period sample reads
                sync0 = model.sync
                model.sync = lambda tt, y: sync0(tt, y + vec if tt >= t_end - _LOOPMAP_EPS else y)  # type: ignore[method-assign]
        res = sim.run()
        out = {k: float(v[-1]) for k, v in res.states.items() if k != "t"}
        if view is not None and view[2] == "c":
            out["plant." + view[0]] -= view[1]
        return out


# --------------------------------------------------------- command-line entry

_ROOT = Path(__file__).resolve().parent  # the repository folder
_EXAMPLE_CONFIGS = _ROOT / "configs"
_RESULTS = Path.cwd() / "output"
_CONFIG_SUFFIXES = (".yaml", ".yml", ".json")


def _config_path(value: str | None) -> Path:
    if value is None:
        configs = sorted(
            p for p in _EXAMPLE_CONFIGS.iterdir()
            if p.is_file() and p.suffix.lower() in _CONFIG_SUFFIXES
        )
        if not configs:
            raise FileNotFoundError("no bundled example configurations available")
        return configs[0]
    path = Path(value).expanduser()
    candidates = [path]
    if not path.suffix:
        candidates += [path.with_suffix(s) for s in _CONFIG_SUFFIXES]
    example = _EXAMPLE_CONFIGS / path
    candidates.append(example)
    if not path.suffix:
        candidates += [example.with_suffix(s) for s in _CONFIG_SUFFIXES]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"configuration not found: {value}")

def parse_override(text: str):
    key, _, value = text.partition("=")
    if not key or not _:
        raise argparse.ArgumentTypeError("expected key.path=value")
    for cast in (int, float):
        try:
            v = cast(value)
            if cast is int and str(v) != value:
                continue
            return key, v
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return key, value.lower() == "true"
    if value.lower() in ("null", "none"):
        return key, None
    return key, value


def main(argv=None) -> int:
    """Run a converter configuration and save its states, summary and configured signals.

    With no config argument, run the first YAML/JSON file in configs/ (sorted by filename).
    """
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="configuration name or path; default: first file in configs by filename")
    ap.add_argument("--set", action="append", default=[], type=parse_override, metavar="PATH=VALUE",
                    help="override a dotted parameter path (repeatable)")
    ap.add_argument("--initial", default=None, metavar="FILE",
                    help="initial values: a states.csv (last row) or a YAML/JSON file with an 'initial' block")
    ap.add_argument("--initial-time", type=float, default=None, metavar="T",
                    help="with a states.csv: start from the row at time T instead of the last row")
    ap.add_argument("--out", default=None, help="output directory (default: output/<config name> in the repository folder)")
    ap.add_argument("--progress", type=float, default=None, metavar="SECONDS",
                    help="print a progress line every SECONDS of simulated time (overrides the config)")
    ap.add_argument("--list-states", action="store_true",
                    help="print the state names generated for this configuration and exit")
    ap.add_argument("--ph-report", action="store_true",
                    help="print the port-Hamiltonian structure report of the system (always built) and exit")
    args = ap.parse_args(argv)

    try:
        config = _config_path(args.config)
    except FileNotFoundError as exc:
        ap.error(str(exc))

    overrides = dict(args.set)
    if args.progress is not None:
        overrides["simulation.progress_every"] = args.progress
    p = load(config, initial=args.initial,
                    initial_time=args.initial_time, **overrides)
    sim = Simulation(p)
    if args.list_states:
        print("\n".join(sim.state_names()))
        return 0
    if args.ph_report:
        print(sim.ph_report)
        return 0
    units = ", ".join(f"{n} ({u.control.type}, PWM {1e-3 / u.pwm.update_period:.0f} kHz, delay {u.delay.steps})"
                      for n, u in p.units.items())
    print(f"peslite: {config}  units={units}  bridge={p.simulation.bridge}  "
          f"solver={p.simulation.solver.type}/{p.simulation.solver.method}  "
          f"t = {p.initial.t} .. {p.simulation.t_end} s  ({len(p.initial.states)} initial values given)")
    if sim.ph_report is not None:
        print(f"structure: {sim.ph_report.verdict} (state coverage {sim.ph_report.coverage:.0%}"
              f"{'; ' + '; '.join(sim.energy_problems) if sim.energy_problems else ''})")
    r = sim.run()
    s = r.summary
    print(f"done: wall {r.wall_time:.1f} s, rhs evaluations {r.n_rhs}, stop at {s.get('t_stop_s', 0):.4f} s, "
          f"tripped={bool(s.get('tripped'))}")
    for name in p.units:
        print(f"  {name}: max |i| = {s.get(f'{name}.max_current_pu', 0):.4f} pu, "
              f"trip: {s.get(f'{name}.trip_cause')}, alarms: {s.get(f'{name}.alarms')}")
    if "stop_reason" in s:
        print(f"stopped: {s['stop_reason']}")
    if "interface_kappa" in s:
        print(f"split: measured hold error {s.get('interface_max_error_rated', s['interface_max_rel_error']):.2e}, "
              f"bound kappa {s['interface_kappa']:.2e} of rating, within bound: {bool(s['interface_within_bound'])}"
              + (f"; coarse window: {s['interface_coarse_hold']}" if "interface_coarse_hold" in s else ""))
    if "split_error_bound_rated" in s:
        print(f"split error, from the linearised closed loop: <= {s['split_error_bound_rated']:.2e} of rating "
              f"({s['split_bound']})")
    out = Path(args.out) if args.out else _RESULTS / config.stem
    written = r.save(out, info={"config": str(config)})
    print("wrote", ", ".join(str(w) for w in written))
    final = r.final_states()
    if final:
        width = max(map(len, final))
        print(f"final values (last row of states.csv, t = {final['t']:.6g} s):")
        for name, value in final.items():
            if name != "t":
                print(f"  {name:<{width}}  {value: .10g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
