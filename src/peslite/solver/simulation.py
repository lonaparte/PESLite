"""The run of a simulation: the system and solver built from its parameters, the initial states, the
event loop, and the command line (the ``peslite`` command).

Between events the solver integrates the model; the events are those of the file, sampled units'
ADC samples, control interrupts, PWM loads, averaging-window openings and switching instants, and
the snapshots. Ideal averaged units have continuous controller and bridge equations instead.
Order at a coincident instant: file events,
over-current check, ADC samples, control interrupts, actuation, window opening, switching instants,
snapshot.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import sysconfig
import time
import warnings
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from ..assembly.events import EVENT_TYPES, SWITCHING, connected_at, switching_schedule
from ..assembly.params import Change, Params, dump, dumps, load
from ..assembly.system import System
from ..assembly.unit import Unit
from ..components.pwm import Modulator
from ..control.blocks import abc2complex, complex2abc
from ..control.controller import Controller
from .integrators import Solver
from .model import ConfigError, StateRegistry, expand_aliases, flatten, resolve
from .multirate import make_solver
from .splitbound import split_error_bound

__all__ = ["Simulation", "SimulationResult", "Recorder", "main", "convert_main", "read_csv", "align", "window_ptp",
           "pointwise_errors"]


_EPS = 1e-10  # seconds; intervals shorter than this are not integrated separately


def _finite(y: np.ndarray) -> bool:
    """Return True when every entry of ``y`` is finite."""
    return all(map(math.isfinite, y.tolist()))


def _plant_output_row(params: Params, t: float, plant: Mapping[str, Any],
                      ctrl_logs: Mapping[str, Mapping[str, float]],
                      ctrl_names: Mapping[str, Iterable[str]] | None = None) -> dict[str, float]:
    """Convert one raw plant snapshot into one public, real-valued output row."""
    row: dict[str, float] = {"t": t}
    for key, value in plant.items():
        head, _, what = key.rpartition(".")
        if np.iscomplexobj(value):
            stem = {"u_g": "v", "i_c": "i_conv", "u": "v"}.get(what, what)
            prefix = f"{head}." if head else ""
            for phase, phase_value in zip("abc", complex2abc(complex(value))):
                row[f"{prefix}{stem}_{phase}"] = float(phase_value)
        elif not ((head in params.units and what in ("u_dc", "i_dc"))
                  or (head in params.sources and what == "angle")):
            row[key] = float(value)
    for name in params.units:
        for what in ("u_dc", "i_dc"):
            key = f"{name}.{what}"
            if key in plant:
                row[key] = float(plant[key])
    for name in params.sources:
        key = f"{name}.angle"
        if key in plant:
            row[key] = float(plant[key])
    names_by_unit = ctrl_names or {unit: log for unit, log in ctrl_logs.items()}
    for unit, names in names_by_unit.items():
        log = ctrl_logs.get(unit, {})
        row.update({f"{unit}.{key}": float(log.get(key, math.nan)) for key in names})
    return row


# ------------------------------------------------------------------ what a run produces

@dataclass
class SimulationResult:
    params: Params
    _records: Any = field(repr=False)
    out_dir: Path | None = None
    files: list[Path] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0
    n_rhs: int = 0

    @property
    def t(self) -> np.ndarray:
        """Snapshot times, read from the streamed result when requested."""
        return self._records.t

    @property
    def plant(self) -> Mapping[str, np.ndarray]:
        """Plant output columns, read one at a time from ``plant.csv``."""
        return self._records.plant

    @property
    def ctrl(self) -> Mapping[str, np.ndarray]:
        """Controller logs, read one column at a time from the streamed result."""
        return self._records.ctrl

    @property
    def states(self) -> Mapping[str, np.ndarray]:
        """State columns, read one at a time from the streamed result."""
        return self._records.states

    @property
    def energy(self) -> Mapping[str, np.ndarray]:
        """Energy columns, read one at a time from the streamed result."""
        return self._records.energy

    @property
    def tripped(self) -> bool:
        return bool(self.summary.get("tripped", 0))

    def columns(self) -> dict[str, np.ndarray]:
        """Return all real-valued columns of ``plant.csv``."""
        return {"t": self.t, **{key: values for key, values in self.plant.items()}}

    # ------------------------------------------------------------ states
    def final_states(self) -> dict[str, float]:
        """Return the last row of the state table (``t`` included)."""
        return dict(self._records.final_states)

    def plot(self, columns: str | Iterable[str], **options: Any) -> Path:
        """Plot recorded columns from one result CSV through the optional plotting add-on."""
        from ..addons import plot_result

        path = plot_result(self, columns, **options)
        if path not in self.files:
            self.files.append(path)
        return path



class _CSVTable:
    """A fixed-width numeric table flushed to CSV every ``batch_rows`` rows."""

    def __init__(self, path: Path, batch_rows: int = 1000) -> None:
        self.path = path
        self.batch_rows = batch_rows
        self.keys: list[str] | None = None
        self.n_rows = 0
        self._pending: list[list[float]] = []
        self._fh: Any = None

    def append(self, row: Mapping[str, float]) -> None:
        if self.keys is None:
            self.keys = list(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", newline="")
            self._fh.write(",".join(self.keys) + "\n")
        elif list(row) != self.keys:
            raise RuntimeError("the set of recorded columns changed during the run")
        self._pending.append(list(row.values()))
        self.n_rows += 1
        if len(self._pending) == self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            np.savetxt(self._fh, np.asarray(self._pending), delimiter=",", fmt="%.17g")
            self._pending.clear()

    def close(self) -> None:
        self.flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def column(self, key: str) -> np.ndarray:
        if self.keys is None or key not in self.keys:
            raise KeyError(key)
        return np.loadtxt(self.path, delimiter=",", skiprows=1, usecols=self.keys.index(key),
                          dtype=float, ndmin=1)


@dataclass(frozen=True)
class _ColumnRef:
    table: _CSVTable
    key: str

    def read(self) -> np.ndarray:
        return self.table.column(self.key)


class _CSVColumns(Mapping[str, np.ndarray]):
    """Dictionary-like CSV columns; each requested column is read and then released by its caller."""

    def __init__(self, refs: Mapping[str, _ColumnRef] | None = None) -> None:
        self._refs = dict(refs or {})

    def __getitem__(self, key: str) -> np.ndarray:
        try:
            return self._refs[key].read()
        except KeyError:
            raise KeyError(key) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._refs)

    def __len__(self) -> int:
        return len(self._refs)


@dataclass
class _RecordedOutput:
    plant_table: _CSVTable | None
    ctrl_tables: dict[str, _CSVTable]
    states_table: _CSVTable | None
    energy_table: _CSVTable | None
    final_states: dict[str, float]
    t_stop: float

    def __post_init__(self) -> None:
        self._t = None
        for table in (self.plant_table, self.states_table, self.energy_table):
            if table is not None and table.n_rows:
                self._t = _ColumnRef(table, "t")
                break
        self.plant = _CSVColumns({key: _ColumnRef(self.plant_table, key)
                                  for key in (self.plant_table.keys or ()) if key != "t"}
                                 if self.plant_table is not None else {})
        ctrl: dict[str, _ColumnRef] = {}
        for unit, table in self.ctrl_tables.items():
            ctrl.update({f"{unit}.{key}": _ColumnRef(table, key)
                         for key in (table.keys or ())})
        self.ctrl = _CSVColumns(ctrl)
        self.states = _CSVColumns({key: _ColumnRef(self.states_table, key)
                                   for key in (self.states_table.keys or ())}
                                  if self.states_table is not None else {})
        self.energy = _CSVColumns({key: _ColumnRef(self.energy_table, key)
                                   for key in (self.energy_table.keys or ())}
                                  if self.energy_table is not None else {})

    @property
    def t(self) -> np.ndarray:
        return self._t.read() if self._t is not None else np.asarray([self.t_stop], dtype=float)


class Recorder:
    """Stream configured result tables directly to their final CSV files."""

    def __init__(self, params: Params, out_dir: Path | None,
                 ctrl_names: Mapping[str, Iterable[str]] | None = None,
                 keep_states: bool = True,
                 keep_signals: bool = False, keep_energy: bool = True,
                 batch_rows: int = 1000) -> None:
        self.params = params
        self.out_dir = out_dir
        self.keep_states = keep_states
        self.keep_signals = keep_signals
        self.keep_energy = keep_energy
        self.last_ctrl_log: dict[str, dict[str, float]] = {}
        self.ctrl_names = {unit: tuple(names) for unit, names in (ctrl_names or {}).items()}
        self._batch_rows = batch_rows
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            names = ["states.csv", "plant.csv", "energy.csv", "summary.json", "simulation.pes",
                     *(f"ctrl.{unit}.csv" for unit in params.units)]
            for name in names:
                (out_dir / name).unlink(missing_ok=True)
        self._plant = (_CSVTable(out_dir / "plant.csv", batch_rows)
                       if keep_signals and out_dir is not None else None)
        self._ctrl: dict[str, _CSVTable] = {}
        self._states = (_CSVTable(out_dir / "states.csv", batch_rows)
                        if keep_states and out_dir is not None else None)
        self._energy = (_CSVTable(out_dir / "energy.csv", batch_rows)
                        if keep_energy and out_dir is not None else None)
        self._last_t: float | None = None
        self._final_states: dict[str, float] = {}

    @property
    def last_t(self) -> float | None:
        return self._last_t

    def energy_row(self, t: float, columns: dict[str, float]) -> None:
        if self._energy is not None:
            self._energy.append({"t": t, **columns})

    def state_row(self, t: float, row: dict[str, float]) -> None:
        full = {"t": t, **row}
        if self._states is not None:
            self._states.append(full)
        self._final_states = full

    def plant_snapshot(self, t: float, signals: dict[str, Any]) -> None:
        if self._plant is not None:
            self._plant.append(_plant_output_row(
                self.params, t, signals, self.last_ctrl_log, self.ctrl_names))
        self._last_t = t

    def ctrl_sample(self, unit: str, t: float, log: dict[str, float]) -> None:
        """Stream one controller's log at one of its sampling instants."""
        if not self.keep_signals or self.out_dir is None:
            return
        names = self.ctrl_names.get(unit)
        if not names:
            names = self.ctrl_names[unit] = tuple(log)
        if unit not in self._ctrl:
            self._ctrl[unit] = _CSVTable(self.out_dir / f"ctrl.{unit}.csv", self._batch_rows)
        self._ctrl[unit].append(
            {"t": t, **{name: float(log.get(name, math.nan)) for name in names}})

    def finish(self) -> _RecordedOutput:
        tables = [table for table in (self._plant, self._states, self._energy) if table is not None]
        tables += list(self._ctrl.values())
        for table in tables:
            table.close()
        return _RecordedOutput(
            plant_table=self._plant,
            ctrl_tables=self._ctrl,
            states_table=self._states,
            energy_table=self._energy,
            final_states={key: float(value) for key, value in self._final_states.items()},
            t_stop=self._last_t or 0.0,
        )


# ------------------------------------------------------------------ reading results back

def read_csv(path: str | Path, skip_header: int = 0) -> np.ndarray:
    """Read a CSV file with a header row into a structured array, keeping column names verbatim (dots included)."""
    return np.genfromtxt(path, delimiter=",", names=True, skip_header=skip_header, deletechars="")


def align(a: np.ndarray, b: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``a`` and ``b`` whose times fall on the same multiple of ``period``."""
    ia = np.round(a["t"] / period).astype(int)
    ib = np.round(b["t"] / period).astype(int)
    _common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
    return a[ka], b[kb]


def window_ptp(d: np.ndarray, t0: float, t1: float, key: str) -> float:
    """Peak-to-peak of ``d[key]`` on ``[t0, t1)`` (NaN with fewer than 11 samples)."""
    m = (d["t"] >= t0) & (d["t"] < t1)
    return float(np.ptp(d[key][m])) if m.sum() > 10 else float("nan")


def pointwise_errors(a: np.ndarray, b: np.ndarray, keys: Iterable[str],
                     scale: Optional[Mapping[str, float]] = None) -> dict[str, dict[str, float]]:
    """Max and RMS of ``|a[k] - b[k]| / scale[k]`` for aligned arrays, per key present in both."""
    scale = scale or {}
    out = {}
    for key in keys:
        if key not in a.dtype.names or key not in b.dtype.names:
            continue
        s = scale.get(key, 1.0)
        x, y = a[key] / s, b[key] / s
        e = np.abs(x - y)
        out[key] = {"max": float(e.max()), "rms": float(np.sqrt(np.mean(e * e))),
                    "t_max": float(a["t"][int(e.argmax())]), "rms_ref": float(np.sqrt(np.mean(y * y)))}
    return out


# ------------------------------------------------------------------ the run

class Simulation:
    """Assemble the system and solver from parameters and run the event loop once."""

    def __init__(self, p: Params, system: Any = None, solver: Solver | None = None,
                 parts: Mapping[str, Callable[..., Any]] | None = None,
                 instances: Mapping[str, Any] | None = None) -> None:
        """Build every part from ``p`` unless replaced through ``parts`` or ``instances``.

        parts: factories keyed ``"system"`` or ``"solver"`` (called with ``p``), or ``"<unit>"``,
        ``"<unit>.ctrl"`` or ``"<unit>.modulator"`` (called with the unit's section).
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
            known |= {name, f"{name}.ctrl", f"{name}.modulator"}
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
            for key in (name, f"{name}.ctrl", f"{name}.modulator"):
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
            parts = [(unit.ctrl, Controller)]
            modulator = getattr(unit.bridge, "modulator", None)
            if modulator is not None:
                parts.append((modulator, Modulator))
            for obj, proto in parts:
                if not isinstance(obj, proto):
                    raise TypeError(f"{unit.name}: {type(obj).__name__} does not satisfy the "
                                    f"{proto.__name__} protocol")
        check_events = getattr(self.system, "check_events", None)
        if check_events is not None:
            check_events(p)
        for needed, what in (("switch", switching_schedule(p.events)), ("apply", p.changes)):
            if what and not hasattr(self.system, needed):
                raise ConfigError(f"the events need a system with {needed}(); "
                                  f"{type(self.system).__name__} has none")
        if not isinstance(self.solver, Solver):
            raise TypeError(f"{type(self.solver).__name__} does not satisfy the Solver protocol")
        self._states = StateRegistry(self._state_parts())
        self.result: SimulationResult | None = None
        self.energy_problems: list[str] = []
        self.ph_report = None  # port-Hamiltonian structure report
        self._check_energy()

    @property
    def units(self) -> dict[str, Unit]:
        return self.system.units

    def unit(self, name: Optional[str] = None) -> Unit:
        """Return the named unit, or the only unit when ``name`` is omitted."""
        self.p.unit(name)  # checks the name
        return self.system.units[name if name is not None else next(iter(self.p.units))]

    def export(self, format: str, out_dir: str | Path | None = None, *, name: str = "run",
               variables=()):
        """Export this configured simulation as a standalone implementation."""
        from ..assembly.exporter import export
        return export(self, format, out_dir, name=name, variables=variables)

    def _actions(self) -> list[tuple[float, int, Any]]:
        """Return file-event actions as ``(time, file_order, action)``."""
        order = {name: index for index, name in enumerate(self.p.events)}
        actions = [(change.t, order[change.event], change) for change in self.p.changes]
        actions += [(event.t, order[name], event) for name, event in self.p.events.items()
                    if event.type != "set"]
        return sorted(actions, key=lambda action: action[:2])

    def _act(self, what: Any, t: float) -> None:
        """Apply one file event; the caller repacks the solver vector afterwards."""
        if isinstance(what, Change):
            self.system.apply(what)
        else:
            EVENT_TYPES[what.type].apply(what, self.system, t)

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
        solver = self.solver
        groups = getattr(solver, "groups", None)
        if groups is not None and len(groups) < 2:  # not split
            groups = None
        hold = None
        if hasattr(solver, "window") and hasattr(solver, "dt"):
            hold = {"step": solver.dt, "window": solver.window}
        zoh = {unit.zoh: 0.7 + 0.2j for unit in self.system.units.values()
               if not getattr(unit, "continuous", False)}
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
    def _state_parts(self) -> list[tuple[str, Any]]:
        """State owners by fixed state-table prefix, in state-table order."""
        provide = getattr(self.system, "state_parts", None)
        if provide is not None:
            parts = list(provide())
        else:  # custom systems written for the original aggregate Stateful interface
            parts = [("", self.system)]
            for name, unit in self.system.units.items():
                bridge = unit.bridge
                parts.extend([(f"{name}.ctrl", unit.ctrl),
                              (f"{name}.{bridge.state_prefix}", bridge.state_owner)])
                if unit.adc is not None and unit.adc.averaging:
                    parts.append((f"{name}.meas", unit.adc))
        parts.append(("solver", self.solver))
        return parts

    def watch_values(self, t: float,
                     ctrl_logs: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        """Return values addressable by ``simulation.progress.watch`` at time ``t``.

        Names include state-table columns, plant-table columns, state aliases, and complex states
        without their ``.re``/``.im`` suffix, in which case the magnitude is returned.
        Public plant/controller columns take precedence over an alias with the same name.
        """
        system = self.system
        state = self._states.read()
        for alias, name in getattr(system, "state_aliases", {}).items():
            if name in state:
                state[alias] = state[name]
        values = {key: abs(value) for key, value in state.items() if isinstance(value, complex)}
        values.update(flatten(state))
        row = _plant_output_row(self.p, t, system.signals(t), ctrl_logs)
        values.update({key: value for key, value in row.items() if key != "t"})
        return values

    def state_names(self) -> list[str]:
        """Return the state-table column names after ``t``, i.e. the valid ``initial.states`` keys."""
        return list(self._states.columns)

    def _apply_initial(self, t0: float) -> np.ndarray:
        """Load ``initial.states`` into the parts; return the solver vector."""
        system = self.system
        aliases = dict(getattr(system, "state_aliases", {}))
        presets_of = getattr(system, "state_presets", None)
        presets = presets_of(t0) if presets_of is not None else {}
        template = self._states.read()
        try:
            given = expand_aliases(self.p.simulation.initial.states, aliases)
            if callable(presets):
                for bus in getattr(system, "buses", {}):
                    key = f"{bus}.u_C"
                    if key not in template or any(name == key or name.startswith(key + ".") for name in given):
                        continue
                    try:
                        given[key] = presets(key, "source")
                    except ValueError:
                        pass
            values = resolve(template, given, presets)
        except (KeyError, ValueError) as exc:
            msg = exc.args[0] if exc.args else str(exc)
            if aliases:
                msg += f". Aliases: {', '.join(f'{a} -> {c}' for a, c in aliases.items())}"
            raise ConfigError(msg) from None
        # 1. power stage (unset states stay zero), events up to t0, then the solver vector
        self._states.load(values, ["", *system.units])
        if hasattr(system, "switch"):
            for name, steps in switching_schedule(self.p.events).items():
                if not connected_at(steps, t0):
                    system.switch(name, False, t0)
        for event_t, _, what in self._actions():
            if event_t <= t0 + _EPS and getattr(what, "type", None) not in SWITCHING:
                self._act(what, event_t)
        # 2. each unit's host command as at the start
        for unit in system.units.values():
            unit.start(t0)
        # 3. controllers, PWM registers, measurement windows and solver
        self._continued = set()
        self._windows_given = set()
        for name, unit in system.units.items():
            bridge_prefix = f"{name}.{unit.bridge.state_prefix}"
            self._states.load(values, [f"{name}.ctrl", bridge_prefix])
            if any(key.startswith((f"{name}.ctrl.", bridge_prefix + ".")) for key in values):
                self._continued.add(name)
            if unit.adc is not None and unit.adc.averaging:
                self._states.load(values, [f"{name}.meas"])
                if any(key.startswith(f"{name}.meas.") for key in values):
                    self._windows_given.add(name)
            unit.states_loaded()
        self._states.load(values, ["solver"])
        return system.model.get_initial_values()

    # ------------------------------------------------------------ the loop
    def run(self, t_end: float | None = None, *, out_dir: str | Path | None = "output/run",
            info: Mapping[str, Any] | None = None) -> SimulationResult:
        """Run once, streaming configured histories directly into ``out_dir``.

        The default directory is ``output/run``. ``None`` is reserved for internal runs that
        disable every history. No complete numeric history is retained in memory; result columns
        read the final CSV files on demand.
        """
        if self.result is not None:
            raise RuntimeError("a Simulation runs once; build a new one for another run")
        p, system = self.p, self.system
        mdl, solver = system.model, self.solver
        settle = getattr(solver, "settle", None)
        parameters_changed = getattr(solver, "parameters_changed", None)
        t_end = p.simulation.t_end if t_end is None else t_end
        output = p.simulation.output
        log_period = output.period
        record_every = max(1, output.record_every)
        stop_on_trip = p.simulation.stop_on_trip
        progress = p.simulation.progress
        progress_every = progress.period if progress.enable else 0.0
        watch = list(progress.watch) if progress.enable else []
        output_dir = Path(out_dir) if out_dir is not None else None
        if output_dir is None and (output.states or output.signals or output.energy):
            raise ValueError("out_dir is required when a simulation output is enabled")
        wall0 = time.time()

        units = list(system.units.values())
        t_start = p.simulation.initial.t
        if t_end <= t_start + _EPS:
            raise ValueError(f"t_end = {t_end} must be after the start time {t_start}")
        t_final = t_end
        y = self._apply_initial(t_start)
        if parameters_changed is not None:
            parameters_changed()
        windowed = [unit for unit in units if unit.windowed]
        ctrl_names = {
            unit.name: getattr(unit.ctrl, "log_names", lambda: ())() for unit in units
        }
        for unit in units:
            set_logging = getattr(unit.ctrl, "set_logging", None)
            if set_logging is not None:
                set_logging(output.signals or bool(watch))
        rec = Recorder(p, output_dir, ctrl_names, keep_states=output.states,
                       keep_signals=output.signals, keep_energy=output.energy,
                       batch_rows=p.simulation.solver.write_length)
        n_log = max(0, math.ceil(t_start / log_period - 1e-9))
        actions = deque(action for action in self._actions() if action[0] > t_start + _EPS)
        next_report = t_start + progress_every if progress_every > 0 else math.inf
        t_local = t_start
        n_rhs0 = getattr(solver, "n_rhs", 0)

        mode = p.simulation.energy_check
        energy_on = mode != "off" and hasattr(mdl, "energy_balance")
        phs_check_step = p.simulation.solver.phs_check_step
        snapshot_index = 0
        continuous_record_index = 0
        worst = {"tellegen": 0.0, "balance": 0.0}
        stop_reason = ""
        coarse_reported = False
        stopped = False
        adc_inputs_changed = False
        synced_at: float | None = None
        zoh_changed = False

        def sync_if_needed(t_now: float) -> None:
            """Bring plant outputs to ``t_now``, refreshing only held-input descendants when possible."""
            nonlocal synced_at, zoh_changed
            if synced_at != t_now:
                mdl.sync(t_now, y)
                synced_at = t_now
            elif zoh_changed:
                mdl.sync_zoh(t_now)
            zoh_changed = False

        def snapshot(t_now: float, final: bool = False) -> None:
            nonlocal snapshot_index, continuous_record_index
            audit_due = energy_on and (snapshot_index % phs_check_step == 0 or final)
            snapshot_index += 1
            try:
                with np.errstate(over="raise"):
                    sync_if_needed(t_now)
                    for unit in units:
                        if getattr(unit, "continuous", False):
                            observe_limits = getattr(unit.ctrl, "observe_limits", None)
                            if observe_limits is not None:
                                observe_limits(t_now)
                            log = unit.ctrl.continuous_log(t_now)
                        else:
                            log = None
                        if log is not None:
                            rec.last_ctrl_log[unit.name] = log
                            if continuous_record_index % record_every == 0:
                                rec.ctrl_sample(unit.name, t_now, log)
                    continuous_record_index += 1
                    signals = system.signals(t_now) if rec.keep_signals else {}
                    rec.plant_snapshot(t_now, signals)
                    if rec.keep_states or final:
                        rec.state_row(t_now, self._states.read_flat())
                    if audit_due and _finite(y):
                        rep = mdl.energy_balance(t_now, y)
                        if rec.keep_energy:
                            rec.energy_row(t_now, rep.columns())
                        scale = rep.scale
                        worst["tellegen"] = max(worst["tellegen"], abs(rep.tellegen) / scale)
                        worst["balance"] = max(worst["balance"], rep.max_residual / scale)
            except (OverflowError, FloatingPointError):
                pass  # diverged state: not recorded

        def watched(t_now: float) -> str:
            """Format the quantities appended to a progress line."""
            sync_if_needed(t_now)
            values = self.watch_values(t_now, rec.last_ctrl_log)
            unknown = [name for name in watch if name not in values]
            if unknown:
                raise ConfigError(f"simulation.progress.watch: unknown name(s) {unknown}; "
                                  f"known: {sorted(values)}")
            return "".join(f"   {name} = {values[name]:.6g}" for name in watch)

        @np.errstate(over="raise")  # overflow counts as divergence
        def integrate(t_a: float, t_b: float, y: np.ndarray, event: bool) -> np.ndarray:
            y = solver(mdl.rhs, t_a, t_b, y).y
            return settle(t_b, y) if event and settle is not None else y

        def advance(t_a: float, t_b: float, event: bool = False) -> bool:
            """Integrate over ``[t_a, t_b]``; ``event`` means the model changes at ``t_b``."""
            nonlocal y, t_local, stop_reason, coarse_reported, synced_at
            synced_at = None
            try:
                y = integrate(t_a, t_b, y, event)
            except (OverflowError, FloatingPointError):
                stop_reason = f"diverged: the states overflowed between t = {t_a:.6g} s and {t_b:.6g} s"
                t_local = t_a
                return True
            t_local = t_b
            finite = _finite(y)
            if windowed and finite:
                sync_if_needed(t_b)
                for unit in windowed:
                    unit.accumulate(t_b - t_a)
            elif finite:
                mdl.set_states(y)
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

        def settle_now() -> None:
            """Finish deferred integration with the plant as it is, before it changes now."""
            nonlocal y
            if settle is not None:
                y = settle(t_local, y)
                mdl.set_states(y)

        # The plant operations exposed to a converter unit. The run loop owns the continuous-state vector;
        # a unit can only request current outputs, change one held input, or make a discontinuous
        # plant change inside the context manager.
        def hold(label: str, value: Any) -> None:
            nonlocal zoh_changed
            mdl.set_zoh_input(label, value)
            zoh_changed = True

        @contextmanager
        def change() -> Iterator[None]:
            nonlocal y, adc_inputs_changed, synced_at
            settle_now()
            yield
            y = mdl.get_initial_values()
            adc_inputs_changed = True
            synced_at = None

        plant = SimpleNamespace(outputs=sync_if_needed, hold=hold, change=change)

        def window_edge(t_now: float, changed: bool) -> None:
            """Restart ADC integrals from the values on the new side of a plant/input jump."""
            if windowed and (changed or zoh_changed):
                sync_if_needed(t_now)
                for unit in windowed:
                    unit.restart_windows()

        # ---------------------------------------------------------------- the event grid
        for unit in units:
            unit.begin(t_start, plant, unit.name in self._continued,
                       unit.name in self._windows_given)
        # Continuous controllers may pre-synchronise their ODE states in begin().
        y = mdl.get_initial_values()
        synced_at = None
        while True:
            adc_inputs_changed = False
            t_log = n_log * log_period
            t_stop = t_log
            for unit in units:
                t_stop = min(t_stop, unit.next_time())
            if actions:
                t_stop = min(t_stop, actions[0][0])
            if t_stop > t_final - _EPS:
                t_stop = t_final
            event_due = bool(actions and actions[0][0] <= t_stop + _EPS)
            advancing = t_stop > t_local + _EPS
            if advancing and advance(t_local, t_stop, event_due):
                stopped = True
                snapshot(t_local, final=True)
                break
            # File events run before the coincident protection/control/PWM events.
            if event_due:
                if advancing:  # advance() already settled at t_stop
                    mdl.set_states(y)
                else:
                    settle_now()
                while actions and actions[0][0] <= t_stop + _EPS:
                    self._act(actions.popleft()[2], t_stop)
                if parameters_changed is not None:
                    parameters_changed()
                y = mdl.get_initial_values()
                adc_inputs_changed = True
                synced_at = None
            # Every unit protects first, then every unit samples, then every unit actuates.  This
            # makes coincident multi-unit events independent of the units' dictionary order.
            for unit in units:
                if unit.protect(t_stop, plant) and stop_on_trip:
                    stopped = True
            if stopped:
                snapshot(t_local, final=True)
                break
            for unit in units:
                done = unit.sense(t_stop, plant)
                if done is not None:
                    k, out = done
                    if out.log is not None:
                        rec.last_ctrl_log[unit.name] = out.log
                        if k % record_every == 0:
                            rec.ctrl_sample(unit.name, t_stop, out.log)
            for unit in units:
                if unit.actuate(t_stop, plant) and stop_on_trip:
                    stopped = True
            if stopped:
                snapshot(t_local, final=True)
                break
            window_edge(t_local, adc_inputs_changed)
            if t_stop >= t_final - _EPS:
                snapshot(t_local, final=True)
                break
            if abs(t_log - t_stop) < _EPS:
                snapshot(t_local)
                n_log += 1
            if t_local >= next_report:
                suffix = watched(t_local) if watch else ""
                print(f"  t = {t_local:8.4f} s   wall {time.time() - wall0:8.1f} s{suffix}",
                      flush=True)
                next_report += progress_every

        if rec.last_t is None or rec.last_t < t_local - _EPS:
            snapshot(t_local, final=True)
        if stop_reason:
            warnings.warn(f"simulation stopped at t = {t_local:.6g} s: {stop_reason}", stacklevel=2)
        records = rec.finish()
        summary: dict[str, Any] = {}
        for name, unit in system.units.items():
            summary.update({f"{name}.{k}": v for k, v in unit.summary().items()})
        summary["tripped"] = int(any(unit.tripped for unit in units))
        summary["t_start"] = t_start
        summary["t_stop"] = rec.last_t if rec.last_t is not None else t_start
        if stop_reason:
            summary["stop_reason"] = stop_reason
        coarse = getattr(solver, "coarse_hold", None)
        if coarse is not None:
            summary["interface_coarse_hold"] = f"{coarse[1]} at t = {coarse[0]:.6g} s: {coarse[2]:.3g} of rating"
        if self.ph_report is not None:
            summary["ph_verdict"] = self.ph_report.verdict
            summary["ph_defaulted"] = list(self.ph_report.defaulted)
            summary["ph_report"] = self.ph_report.to_dict()
        if energy_on:
            summary["energy_tellegen_max_rel"] = worst["tellegen"]
            summary["energy_balance_max_rel"] = worst["balance"]
            summary["energy_problems"] = list(self.energy_problems)
        interface = getattr(solver, "interface", None)
        if interface:  # split-interface indicators
            summary.update({f"interface_{k}": v for k, v in interface.items()})
        if getattr(solver, "window_log", None) and p.simulation.solver.linearisations > 0:
            if self._rebuildable:
                loop = SystemLoop(self)
                if loop.period is None:  # no common control period
                    summary["split_bound"] = (f"the converters' periods {loop.periods} (control, PWM "
                                              f"loads, carriers and loops) have no common multiple within "
                                              f"64 of the longest, so the loop has no period to linearise: "
                                              f"no bound")
                else:
                    summary.update(split_error_bound(
                        solver, loop, records.t, records.states, p.simulation.solver.linearisations,
                        labels=mdl.state_labels(), prefix=""))
            else:  # parts given as instances
                summary["split_bound"] = ("the loop has custom parts given as instances, so it cannot be "
                                          "rebuilt for the linearisation: pass them as factories of the "
                                          "parameter tree for a bound")
        wall_time = time.time() - wall0
        n_rhs = getattr(solver, "n_rhs", 0) - n_rhs0
        files: list[Path] = []
        if output_dir is not None:
            for table in (records.states_table, records.plant_table, records.energy_table,
                          *records.ctrl_tables.values()):
                if table is not None and table.path.is_file():
                    files.append(table.path)
            payload = {**(info or {}), "wall_time": wall_time, "n_rhs": n_rhs, **summary}
            summary_path = output_dir / "summary.json"
            params_path = output_dir / "simulation.pes"
            summary_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
            dump(p, params_path)
            files += [summary_path, params_path]
        self.result = SimulationResult(
            params=p, _records=records, out_dir=output_dir, files=files, summary=summary,
            wall_time=wall_time, n_rhs=n_rhs)
        return self.result


_LOOPMAP_EPS = 1e-9

def macro_period(periods: list[float], limit: int = 64) -> float | None:
    """Return the smallest common multiple of ``periods`` (s), or ``None`` beyond ``limit`` longest periods."""
    if not periods:
        return None
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
        self.periods = sorted({period for unit in sim.units.values() for period in unit.periods})
        self.period = macro_period(self.periods)
        self.w0 = 2.0 * math.pi * sim.p.base.f0
        flags = set(sim._states.boolean_names)
        self._flags = {name for name in flags
                       if name.rpartition(".")[2] in ("tripped", "fault")}
        self._modes = (flags - self._flags) | {
            name for name in sim._states.names if ".startup." in name
        }
        self._plant_columns = set(sim._states.read_flat(["", *sim.units]))
        rated = sim.solver.rated_effort
        self._rated = ({lab: float(r) for lab, r in zip(sim.system.model.state_labels(), rated)}
                       if rated is not None else {})
        p = sim.p
        sp = p.simulation
        self._p = dataclasses.replace(
            p,
            simulation=dataclasses.replace(
                sp, energy_check="off", stop_on_trip=False,
                progress=dataclasses.replace(sp.progress, enable=False),
                                           output=dataclasses.replace(sp.output, period=self.period or 1.0,
                                                                      states=False, energy=False, signals=False),
                                           solver=dataclasses.replace(sp.solver, subsystems={},
                                                                      linearisations=0)))

    # ------------------------------------------------------------ coordinates
    def coordinates(self, names: list[str]) -> list[tuple[str, list[str], float]]:
        """Group state-table columns into map coordinates ``(kind, names, scale)``."""
        cols: list[tuple[str, list[str], float]] = []
        used: set[str] = set()
        for c in names:
            if c in used or c == "t" or c.startswith("solver."):
                continue  # solver bookkeeping, not a loop state
            if c in self._flags:
                cols.append(("frozen", [c], 1.0))
                used.add(c)
            elif c in self._modes:
                cols.append(("mode", [c], 1.0))
                used.add(c)
            elif c.endswith(".re") and c[:-3] + ".im" in names:
                base = c[:-3]
                turns = c in self._plant_columns  # physical alpha-beta; controller vectors are already dq
                scale = self._rated.get(base + ".re", 0.0) if turns else 0.0
                cols.append(("vector" if turns else "fixed", [c, base + ".im"], scale if scale > 0 else 1.0))
                used.update(cols[-1][1])
            elif c.endswith(".d_a") and c[:-4] + ".d_b" in names and c[:-4] + ".d_c" in names:
                cols.append(("triple", [c[:-4] + f".d_{ph}" for ph in "abc"], 1.0))
                used.update(cols[-1][1])
            else:
                scale = self._rated.get(c, 0.0) if c in self._plant_columns else 0.0
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
        initial = dataclasses.replace(
            self._p.simulation.initial, t=t,
            states={k: v for k, v in row.items() if not k.startswith("solver.")})
        p = dataclasses.replace(self._p, simulation=dataclasses.replace(
            self._p.simulation, t_end=t + self.period, initial=initial))
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
        res = sim.run(out_dir=None)
        out = {k: v for k, v in res.final_states().items() if k != "t"}
        if view is not None and view[2] == "c":
            out[view[0]] -= view[1]
        return out


# --------------------------------------------------------- command-line entry

def _example_configs_dir() -> Path:
    """Return the root examples directory, from a checkout or an installed wheel."""
    source = Path(__file__).resolve().parents[3] / "examples"
    if source.is_dir():
        return source
    try:
        dist = distribution("peslite")
    except PackageNotFoundError:
        return source
    # data-files live below the installation prefix, outside the import package.  A ``--target``
    # installation puts them below the metadata root; a normal environment uses sysconfig's data root.
    roots = (Path(dist.locate_file("")).resolve(), Path(sysconfig.get_path("data")).resolve())
    for root in roots:
        candidate = root / "share" / "peslite" / "examples"
        if candidate.is_dir():
            return candidate
    return source


_EXAMPLE_CONFIGS = _example_configs_dir()
_RESULTS = Path.cwd() / "output"
_CONFIG_SUFFIXES = (".pes", ".yaml", ".yml", ".json")


def _config_path(value: str | None) -> Path:
    if value is None:
        default = _EXAMPLE_CONFIGS / "gfl-example.pes"
        if default.is_file():
            return default.resolve()
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


def _solver_choice_is_set(overrides: Mapping[str, Any]) -> bool:
    """Whether ``--set`` selected a solver family or method for this run."""
    return any(
        path in ("simulation.solver.type", "simulation.solver.method")
        for path in overrides
    )


def main(argv=None) -> int:
    """Run a simulation file and save its states, summary and configured signals."""
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?",
                    help="simulation file path or bundled example name; default: first example by filename")
    ap.add_argument("--set", action="append", default=[], type=parse_override, metavar="PATH=VALUE",
                    help="override a dotted parameter path (repeatable)")
    ap.add_argument("--initial", default=None, metavar="FILE",
                    help="initial values: a states.csv row or a simulation file's simulation.initial block")
    ap.add_argument("--initial-time", type=float, default=None, metavar="T",
                    help="with a states.csv: start from the row at time T instead of the last row")
    bridge = ap.add_mutually_exclusive_group()
    bridge.add_argument("--switching", action="store_true",
                        help="exact-switching bridge with fixed-step RK4; --set has priority")
    bridge.add_argument("--pwm-averaging", action="store_true",
                        help="PWM-period-averaged bridge with fixed-step RK4; --set has priority")
    bridge.add_argument("--averaging", action="store_true",
                        help="ideal averaged bridge with adaptive DP45; --set has priority")
    ap.add_argument("--out", default=None,
                    help="output directory (default: output/<file name>, or "
                         "a bridge-mode suffix with --switching/--pwm-averaging/--averaging)")
    ap.add_argument("--export", nargs="+", default=None, metavar="FORMAT_OR_PARAM",
                    help="export a standalone simulator: FORMAT followed by optional variable parameter paths")
    ap.add_argument("--progress", type=float, default=None, metavar="SECONDS",
                    help="print a progress line every SECONDS of simulated time")
    ap.add_argument("--watch", action="append", default=None, metavar="NAME",
                    help="append a state or plant/control value to each progress line; repeatable, "
                         "or use NAME,NAME")
    ap.add_argument("--list-states", action="store_true",
                    help="print the state names generated for this configuration and exit")
    ap.add_argument("--ph-report", action="store_true",
                    help="print the port-Hamiltonian structure report of the system (always built) and exit")
    ap.add_argument("--resolved", action="store_true",
                    help="print the complete resolved simulation file and exit")
    plots = ap.add_argument_group("IEEE-style PDF plot (after simulation)")
    plots.add_argument("--plot", nargs="+", default=None, metavar="NAME",
                       help="plot these recorded columns against time; all must belong to one result CSV")
    plots.add_argument("--plot-output", default=None, metavar="PDF",
                       help="PDF path relative to the run directory (default: fig_<table>.pdf)")
    plots.add_argument("--plot-x", default="t", metavar="NAME",
                       help="horizontal-axis column (default: t)")
    plots.add_argument("--plot-time-unit", choices=("auto", "s", "ms", "us"), default="auto",
                       help="time-axis unit (default: choose s, ms, or us from the range)")
    plots.add_argument("--plot-title", default=None, metavar="TEXT")
    plots.add_argument("--plot-xlabel", default=None, metavar="TEXT")
    plots.add_argument("--plot-ylabel", default=None, metavar="TEXT")
    plots.add_argument("--plot-label", action="append", default=None, metavar="TEXT",
                       help="legend label for each waveform, in --plot order; repeatable")
    plots.add_argument("--plot-color", action="append", default=None, metavar="COLOR",
                       help="Matplotlib color for each waveform, in --plot order; repeatable")
    plots.add_argument("--plot-style", action="append", default=None, metavar="STYLE",
                       help="line style for each waveform, in --plot order; repeatable")
    plots.add_argument("--plot-width", choices=("single", "double"), default="single",
                       help="IEEE single- or double-column width (default: single)")
    plots.add_argument("--plot-height", type=float, default=None, metavar="INCHES")
    plots.add_argument("--plot-linewidth", type=float, default=1.5, metavar="POINTS")
    plots.add_argument("--plot-xlim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    plots.add_argument("--plot-ylim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    plots.add_argument("--plot-no-grid", action="store_true", help="disable the background grid")
    plots.add_argument("--plot-no-legend", action="store_true", help="disable the legend")
    args = ap.parse_args(argv)

    plot_option_without_plot = (
        args.plot_output is not None or args.plot_x != "t" or args.plot_time_unit != "auto"
        or args.plot_title is not None or args.plot_xlabel is not None or args.plot_ylabel is not None
        or args.plot_label is not None or args.plot_color is not None or args.plot_style is not None
        or args.plot_width != "single" or args.plot_height is not None or args.plot_linewidth != 1.5
        or args.plot_xlim is not None or args.plot_ylim is not None
        or args.plot_no_grid or args.plot_no_legend
    )
    if args.plot is None and plot_option_without_plot:
        ap.error("plot presentation options require --plot NAME [NAME ...]")
    if args.plot is not None and (args.export or args.list_states or args.ph_report or args.resolved):
        ap.error("--plot requires a simulation run and cannot be combined with inspection or export")

    try:
        config = _config_path(args.config)
    except FileNotFoundError as exc:
        ap.error(str(exc))

    set_overrides = dict(args.set)
    overrides = dict(set_overrides)
    if args.progress is not None:
        overrides["simulation.progress.enable"] = 1
        overrides["simulation.progress.period"] = args.progress
    if args.watch is not None:
        overrides["simulation.progress.watch"] = [name for value in args.watch
                                                   for name in value.split(",") if name]
    p = load(config, initial=args.initial,
                    initial_time=args.initial_time, **overrides)
    preset = (("switching", "fixed", "rk4") if args.switching else
              ("pwm_averaging", "fixed", "rk4") if args.pwm_averaging else
              ("averaging", "adaptive", "DP45") if args.averaging else None)
    if preset is not None:
        model, solver_type, method = preset
        changes = {f"units.{name}.bridge.model": model for name in p.units
                   if f"units.{name}.bridge.model" not in set_overrides}
        if not _solver_choice_is_set(set_overrides):
            changes.update({
                "simulation.solver.type": solver_type,
                "simulation.solver.method": method,
            })
            if solver_type == "adaptive" and not any(
                    path == "simulation.solver.subsystems"
                    or path.startswith("simulation.solver.subsystems.")
                    for path in set_overrides):
                changes["simulation.solver.subsystems"] = {}
        if changes:
            p = p.replace(**changes)
    if args.watch and not p.simulation.progress.enable:
        ap.error("--watch prints on the progress lines: add --progress SECONDS")
    if args.resolved:
        print(dumps(p), end="")
        return 0
    sim = Simulation(p)
    if args.list_states:
        print("\n".join(sim.state_names()))
        return 0
    if args.ph_report:
        print(sim.ph_report)
        return 0
    suffix = ("-switching" if args.switching else
              "-pwm-averaging" if args.pwm_averaging else
              "-averaging" if args.averaging else "")
    if args.export:
        export_format, *export_variables = args.export
        out = Path(args.out) if args.out else Path("export") / f"{config.stem}{suffix}"
        try:
            result = sim.export(export_format, out, name=f"{config.stem}{suffix}",
                                variables=export_variables)
        except (TypeError, ValueError) as exc:
            ap.error(str(exc))
        print("wrote", ", ".join(str(path) for path in result.files))
        return 0
    units = ", ".join(f"{n} ({u.ctrl.type}, {u.bridge.model.replace('_', ' ')}, "
                      f"{sim.units[n].bridge.describe()})"
                      for n, u in p.units.items())
    print(f"peslite: {config}  units={units}  "
          f"solver={p.simulation.solver.type}/{p.simulation.solver.method}  "
          f"t = {p.simulation.initial.t} .. {p.simulation.t_end} s  "
          f"({len(p.simulation.initial.states)} initial values given)")
    if sim.ph_report is not None:
        print(f"structure: {sim.ph_report.verdict} (state coverage {sim.ph_report.coverage:.0%}"
              f"{'; ' + '; '.join(sim.energy_problems) if sim.energy_problems else ''})")
    out = Path(args.out) if args.out else _RESULTS / f"{config.stem}{suffix}"
    r = sim.run(out_dir=out, info={"config": str(config)})
    s = r.summary
    print(f"done: wall {r.wall_time:.1f} s, rhs evaluations {r.n_rhs}, stop at {s.get('t_stop', 0):.4f} s, "
          f"tripped={bool(s.get('tripped'))}")
    for name in p.units:
        print(f"  {name}: max |i| = {s.get(f'{name}.max_current_pu', 0):.4f} pu, "
              f"trip: {s.get(f'{name}.trip_cause') or 'none'}, "
              f"alarms: {', '.join(s.get(f'{name}.alarms') or []) or 'none'}")
    if "stop_reason" in s:
        print(f"stopped: {s['stop_reason']}")
    if "interface_kappa" in s:
        print(f"split: measured hold error {s.get('interface_max_error_rated', s['interface_max_rel_error']):.2e}, "
              f"bound kappa {s['interface_kappa']:.2e} of rating, within bound: {bool(s['interface_within_bound'])}"
              + (f"; coarse window: {s['interface_coarse_hold']}" if "interface_coarse_hold" in s else ""))
    if "split_error_bound_rated" in s:
        print(f"split error, from the linearised closed loop: <= {s['split_error_bound_rated']:.2e} of rating "
              f"({s['split_bound']})")
    if args.plot:
        plot_output = None
        if args.plot_output:
            plot_output = Path(args.plot_output)
            if not plot_output.is_absolute():
                plot_output = out / plot_output
            if not plot_output.suffix:
                plot_output = plot_output.with_suffix(".pdf")
        try:
            r.plot(
                args.plot,
                output=plot_output,
                x=args.plot_x,
                time_unit=args.plot_time_unit,
                labels=args.plot_label,
                title=args.plot_title,
                xlabel=args.plot_xlabel,
                ylabel=args.plot_ylabel,
                width=args.plot_width,
                height=args.plot_height,
                xlim=tuple(args.plot_xlim) if args.plot_xlim else None,
                ylim=tuple(args.plot_ylim) if args.plot_ylim else None,
                colors=args.plot_color,
                linestyles=args.plot_style,
                linewidth=args.plot_linewidth,
                grid=not args.plot_no_grid,
                legend=not args.plot_no_legend,
            )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            ap.error(str(exc))
    print("wrote", ", ".join(str(w) for w in r.files))
    final = r.final_states()
    if final:
        width = max(map(len, final))
        print(f"final values (last row of states.csv, t = {final['t']:.6g} s):")
        for name, value in final.items():
            if name != "t":
                print(f"  {name:<{width}}  {value: .10g}")
    return 0


def convert_main(argv=None) -> int:
    """Export a PESLite simulation file as C++ with optional variable parameter paths."""
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in ("-h", "--help"):
        print("usage: peslite-convert PESFILE [PARAM ...] [--out DIRECTORY]")
        print("export one self-contained peslite.cpp; PARAM may be a variable path or all")
        return 0
    if not values:
        print("usage: peslite-convert PESFILE [PARAM ...]", file=sys.stderr)
        return 2
    return main([values[0], "--export", "cpp", *values[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
