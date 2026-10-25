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
import heapq
import json
import math
import sys
import sysconfig
import time
import warnings
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from ..assembly.events import Scenario, switching_schedule
from ..assembly.conduction import DiodeConduction
from ..assembly.exporter import _signal_names
from ..assembly.params import (Params, _connection_variable, _parameterized, _unit_at, _unit_items,
                               _unit_leaves, connection_defaults, dump, dumps, load)
from ..assembly.system import System, _start_connections
from ..assembly.types import unit_type
from ..assembly.unit import Plant, Unit
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
_UNAVAILABLE = object()


def _optional_output(t: float, name: str, read: Callable, *args):
    """Isolate numerical observation failures without hiding programming or I/O errors."""
    try:
        with np.errstate(over="raise"):
            return read(*args)
    except (OverflowError, FloatingPointError) as exc:
        warnings.warn(f"optional output unavailable at t = {t:.8g} s ({name}): {exc}",
                      RuntimeWarning, stacklevel=2)
        return _UNAVAILABLE


class _SnapshotRead:
    """Read one snapshot value at most once, including an unsuccessful calculation."""

    __slots__ = ("_read", "_value", "_error")

    def __init__(self, read: Callable[[], Any]) -> None:
        self._read, self._value, self._error = read, None, None

    def __call__(self):
        if self._read is not None:
            read, self._read = self._read, None
            try:
                self._value = read()
            except Exception as exc:
                self._error = exc
        if self._error is not None:
            raise self._error
        return self._value


def _finite(y: np.ndarray) -> bool:
    """Return True when every entry of ``y`` is finite."""
    return all(map(math.isfinite, y.tolist()))


def _plant_output_row(params: Params, t: float, plant: Mapping[str, Any],
                      ctrl_logs: Mapping[str, Mapping[str, float]],
                      ctrl_names: Mapping[str, Iterable[str]] | None = None) -> dict[str, float]:
    """Convert one raw plant snapshot into one public, real-valued output row."""
    row: dict[str, float] = {"t": t}
    unit_names = {name for name, _unit in _unit_items(params.units)}
    for key, value in plant.items():
        head, _, what = key.rpartition(".")
        if np.iscomplexobj(value):
            for name, phase_value in zip(_signal_names(key, 3), complex2abc(complex(value))):
                row[name] = float(phase_value)
        elif not ((head in unit_names and what in ("u_dc", "i_dc"))
                  or (head in params.sources and what == "angle")):
            row[key] = float(value)
    for name in unit_names:
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
                 batch_rows: int = 1000, name: str = "simulation") -> None:
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
                     f"{name}_report.json",
                     *(f"ctrl.{unit}.csv" for unit, _cfg in _unit_items(params.units))]
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

    def snapshot(self, t: float, states: dict[str, float] | None = None) -> None:
        """Record the physical time/state independently of optional output calculations."""
        self._last_t = t
        if states is not None:
            self.state_row(t, states)

    def plant_snapshot(self, t: float, signals: dict[str, Any]) -> None:
        if self._plant is not None:
            self._plant.append(_plant_output_row(
                self.params, t, signals, self.last_ctrl_log, self.ctrl_names))

    def ctrl_sample(self, unit: str, t: float,
                    log: Mapping[str, float] | Callable[[], Mapping[str, float] | None], *,
                    record: bool = True) -> None:
        """Read and publish one controller log independently of every other output."""
        names = self.ctrl_names.get(unit)

        def read():
            nonlocal names
            values = log() if callable(log) else log
            if values is None:
                return None
            if not names:
                names = self.ctrl_names[unit] = tuple(values)
            return {name: float(values.get(name, math.nan)) for name in names}

        values = _optional_output(t, f"controller log {unit!r}", read)
        if values is None:
            return
        if values is _UNAVAILABLE:
            values = dict.fromkeys(names or (), math.nan)
        self.last_ctrl_log[unit] = values
        if not record or not self.keep_signals or self.out_dir is None or not names:
            return
        if unit not in self._ctrl:
            self._ctrl[unit] = _CSVTable(self.out_dir / f"ctrl.{unit}.csv", self._batch_rows)
        self._ctrl[unit].append({"t": t, **values})

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
        self._parameters = p
        self._factories: dict[str, Callable[..., Any]] = dict(parts or {})
        given: dict[str, Any] = dict(instances or {})
        if system is not None:
            given["system"] = system
        if solver is not None:
            given["solver"] = solver
        known = {"system", "solver"}
        for name, _cfg in _unit_items(p.units):
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
        for name, cfg in _unit_items(p.units):
            for key in (name, f"{name}.ctrl", f"{name}.modulator"):
                if key in self._factories:
                    unit_parts[key] = _parameterized(self._factories[key], cfg)
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
        for needed, what in (("switch", switching_schedule(p.changes)), ("apply", p.changes)):
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
        if name is None:
            if len(self.system.units) != 1:
                raise ConfigError(
                    f"this system has {len(self.system.units)} assembled units "
                    f"{sorted(self.system.units)}: name the one you mean"
                )
            name = next(iter(self.system.units))
        if name not in self.system.units:
            configured = _unit_at(self.p.units, name)
            if configured is not None and unit_type(configured.type).composite:
                raise ConfigError(
                    f"unit {name!r} is composite; choose one of {list(_unit_leaves(self.p.units, name))}"
                )
            raise ConfigError(f"unknown assembled unit {name!r}; known: {sorted(self.system.units)}")
        return self.system.units[name]

    def export(self, format: str, out_dir: str | Path | None = None, *, name: str = "run",
               variables=(), **options):
        """Export this configured simulation as a standalone implementation."""
        from ..assembly.exporter import export
        return export(self, format, out_dir, name=name, variables=variables, **options)

    def _actions(self) -> list[tuple[float, int, Any]]:
        """Return file-event actions as ``(time, file_order, action)``."""
        order = {name: index for index, name in enumerate(self.p.events)}
        return [(change.t, order[change.event], change) for change in self.p.changes]

    def _ratings(self):
        """Return a function giving the rated (effort, flow) of a storage, or ``None``."""
        base = self.p.base
        system_ac = (base.v_phase_peak, base.i_phase_peak)

        def port_rating(port: Any) -> tuple[float, float]:
            cfg = self.p.unit(port.owner)
            if port.domain == "dc":
                dc = cfg.dc_base
                return dc.v, dc.i
            return cfg.base.v_phase_peak, cfg.base.i_phase_peak

        port_bindings = self.system._port_bindings
        subsystems = dict(zip(self.system.model.names, self.system.model.subsystems))

        def rating(name: str, storage: Any) -> Optional[tuple[float, float]]:
            candidates = []
            for port in port_bindings.get(name, ()):
                value = port_rating(port)
                if value not in candidates:
                    candidates.append(value)
            if candidates:
                return candidates[0] if len(candidates) == 1 else None
            if name in self.p.buses:
                if self.p.buses[name].phases == 3:
                    return system_ac
                return None
            subsystem = subsystems.get(name)
            if getattr(subsystem, "phases", None) == 3:
                return system_ac
            return None

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
        zoh = {
            unit.zoh: (0.7 + 0.2j if isinstance(unit.bridge.topology.zero, complex) else 0.7)
            for unit in self.system.units.values()
            if not getattr(unit, "continuous", False)
        }
        self.ph_report = report = model.ph_report(zoh=zoh, groups=groups, hold=hold)
        report.warnings.extend(
            f"{name}: {sub.energy_audit_note}"
            for name, sub in zip(model.names, model.subsystems)
            if getattr(sub, "energy_audit_note", None)
        )
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

    def _apply_change(self, change) -> None:
        """Apply one checked write and retain its existing effective parameter snapshot."""
        self.system.apply(change)
        if change.params is not None:
            self._parameters = change.params

    def _parameter_value(self, path: str, t: float):
        if _connection_variable(self.p, path) is not None:
            return Scenario(self.p.changes, path.removesuffix(".connected"),
                            initial=dict(connection_defaults(self.p)).get(path)).connected(t)
        value = self._parameters
        for name in path.split("."):
            value = value[name] if isinstance(value, Mapping) else getattr(value, name)
        return value

    def watch_values(self, t: float,
                     ctrl_logs: Mapping[str, Mapping[str, float]]
                     | Callable[[], Mapping[str, Mapping[str, float]]], *,
                     state: Mapping[str, Any] | None = None,
                     plant: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
                     names: Iterable[str] | None = None,
                     ctrl_names: Mapping[str, Iterable[str]] | None = None) -> dict[str, Any]:
        """Return values addressable by ``simulation.progress.watch`` at time ``t``.

        Names include state-table columns, plant-table columns, state aliases, and complex states
        without their ``.re``/``.im`` suffix, in which case the magnitude is returned.
        Public plant/controller columns take precedence over an alias with the same name.
        Explicit ``names`` can also select effective parameter paths and connection commands.
        Callable plant/log providers are selected independently from the existing column names.
        """
        selected = None if names is None else tuple(names)
        if selected == ():
            return {}
        system = self.system
        state = dict(self._states.read() if state is None else state)
        for alias, name in getattr(system, "state_aliases", {}).items():
            if name in state:
                state[alias] = state[name]
        values = {key: abs(value) for key, value in state.items() if isinstance(value, complex)}
        values.update(flatten(state))
        parameters = {}
        if selected is not None:
            for name in selected:
                try:
                    parameters[name] = self._parameter_value(name, t)
                except (KeyError, AttributeError):
                    pass  # A signal/state name is not a parameter path.
        canonical = set(self._states.columns) | set(self._states.names)
        needed = None if selected is None else set(selected) - parameters.keys() - canonical
        if needed is None or needed:
            if ctrl_names is None:
                ctrl_names = {name: getattr(unit.ctrl, "log_names", lambda: ())()
                              for name, unit in self.units.items()}
                if not callable(ctrl_logs):
                    ctrl_names = {**{name: tuple(log) for name, log in ctrl_logs.items()}, **ctrl_names}
            ctrl_names = {unit: tuple(names) for unit, names in ctrl_names.items()}
            log_columns = {f"{unit}.{key}" for unit, names in ctrl_names.items() for key in names}

            def read_plant():
                signals = (getattr(system, "signals", lambda _t: {})(t) if plant is None
                           else plant() if callable(plant) else plant)
                return _plant_output_row(self.p, t, signals, {})

            def read_logs():
                logs = ctrl_logs() if callable(ctrl_logs) else ctrl_logs
                return _plant_output_row(self.p, t, {}, logs, ctrl_names)

            for columns, read in ((None if needed is None else needed - log_columns, read_plant),
                                  (None if needed is None else needed & log_columns, read_logs)):
                if columns is not None and not columns:
                    continue
                row = read() if columns is None else _optional_output(t, "watched signals", read)
                if row is not _UNAVAILABLE:
                    values.update({key: value for key, value in row.items() if key != "t"})
                else:
                    values.update(dict.fromkeys(columns, math.nan))
        values.update(parameters)
        if selected is None:
            return values
        unknown = [name for name in selected if name not in values]
        if unknown:
            raise ValueError(f"unknown watch name(s): {unknown}")
        return {name: values[name] for name in selected}

    def state_names(self) -> list[str]:
        """Return the state-table column names after ``t``, i.e. the valid ``initial.states`` keys."""
        return list(self._states.columns)

    def _initial_values(self, t0: float) -> dict:
        """Resolve explicit initial states and automatic bus presets without loading them."""
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
        return values

    def _apply_initial(self, t0: float) -> np.ndarray:
        """Load ``initial.states`` into the parts; return the solver vector."""
        system = self.system
        values = self._initial_values(t0)
        # 1. power stage (unset states stay zero), events up to t0, then the solver vector
        self._states.load(values, ["", *system.units])
        for event_t, _, what in self._actions():
            if event_t <= t0 + _EPS:
                paths = tuple(path for path in what.paths if path not in what.connections)
                if paths:
                    self._apply_change(dataclasses.replace(what, paths=paths, connections=()))
        # 2. each connection owner establishes its initial command once
        _start_connections(system, self.p, t0)
        # 3. controllers, PWM registers, measurement windows and solver
        self._continued = set()
        self._windows_given = set()
        for name, unit in system.units.items():
            bridge_prefix = f"{name}.{unit.bridge.state_prefix}"
            self._states.load(values, [f"{name}.ctrl", bridge_prefix])
            if any(key.startswith((f"{name}.ctrl.", bridge_prefix + ".")) for key in values):
                self._continued.add(name)
            if unit.adc is not None and unit.adc.accumulating:
                self._states.load(values, [f"{name}.meas"])
                if any(key.startswith(f"{name}.meas.") for key in values):
                    self._windows_given.add(name)
            unit.states_loaded()
        self._states.load(values, ["solver"])
        return system.model.get_initial_values()

    # ------------------------------------------------------------ the loop
    def _begin_units(self, t: float, plant: Plant) -> None:
        """Initialize all units through the same state and output boundary used during a run."""
        for unit in self.system.units.values():
            unit.begin(t, plant, unit.name in self._continued,
                       unit.name in self._windows_given)
        plant.read_state()

    def run(self, t_end: float | None = None, *, out_dir: str | Path | None = "output/run",
            info: Mapping[str, Any] | None = None,
            observe: Callable[..., float | None] | None = None,
            times: Iterable[float] | None = None, observe_values: Iterable[str] | None = (),
            name: str = "simulation") -> SimulationResult:
        """Run once, streaming configured histories directly into ``out_dir``.

        The default directory is ``output/run``. ``None`` is reserved for internal runs that
        disable every history. No complete numeric history is retained in memory; result columns
        read the final CSV files on demand.

        ``observe(t, states, read_values, *, final, stop_reason)`` consumes read-only observations at exact ``times``, file
        events, protection transitions, and the final state. ``times=None`` also observes every
        output snapshot. Call ``read_values()`` within the callback to read ``observe_values``
        (empty by default; ``None`` selects all watch signals/states), using ``watch_values`` naming.
        State-only callbacks do no signal work. Explicit names enable only the needed control logs. Additional
        times do not create output rows or control/protection interrupts. Observers must not
        mutate the simulation or the supplied mappings. No observer history is retained here.
        An observer may return its next required time within the run, using this same event grid.
        Normal and early termination both deliver ``final=True``; ``stop_reason`` carries any
        runtime failure. If no valid terminal physical snapshot is available, ``states`` is
        ``None`` and ``read_values()`` is empty; the reason remains available. A terminal
        callback must not request another observation time.
        """
        if self.result is not None:
            raise RuntimeError("a Simulation runs once; build a new one for another run")
        p, system = self.p, self.system
        mdl, solver = system.model, self.solver
        settle = getattr(solver, "settle", None)
        project = mdl.project_states if mdl.has_state_bounds else None
        parameters_changed = getattr(solver, "parameters_changed", None)
        t_end = p.simulation.t_end if t_end is None else t_end
        observe_output = times is None
        requested = sorted(set(() if times is None else times))
        t_start = p.simulation.initial.t
        if any(not math.isfinite(t) or t < t_start - _EPS or t > t_end + _EPS
               for t in requested):
            raise ValueError("observation times must be finite and within the simulation interval")
        if requested and observe is None:
            raise ValueError("observation times require an observer")
        observations = requested
        pending_observations = set(requested)
        value_names = None if observe_values is None else tuple(observe_values)
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
        diode_conduction = DiodeConduction(mdl, solver)
        if t_end <= t_start + _EPS:
            raise ValueError(f"t_end = {t_end} must be after the start time {t_start}")
        t_final = t_end
        plant = Plant(mdl, self._apply_initial(t_start), project=project)
        if project is not None:
            plant.state = project(plant.state)
        if parameters_changed is not None:
            parameters_changed()
        windowed = [unit for unit in units if unit.windowed]
        ctrl_names = {
            unit.name: getattr(unit.ctrl, "log_names", lambda: ())() for unit in units
        }
        for unit in units:
            set_logging = getattr(unit.ctrl, "set_logging", None)
            if set_logging is not None:
                log_names = {f"{unit.name}.{key}" for key in ctrl_names[unit.name]}
                set_logging(output.signals or bool(watch)
                            or (observe is not None and (value_names is None
                                or bool(log_names.intersection(value_names)))))
        rec = Recorder(p, output_dir, ctrl_names, keep_states=output.states,
                       keep_signals=output.signals, keep_energy=output.energy,
                       batch_rows=p.simulation.solver.write_length, name=name)
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
        sync_if_needed = plant.outputs

        def snapshot(t_now: float, final: bool = False, record: bool = True,
                     notify: bool = True) -> None:
            nonlocal snapshot_index, continuous_record_index, stop_reason
            notifying = observe is not None and notify
            audit_due = record and energy_on and (snapshot_index % phs_check_step == 0 or final)
            snapshot_index += int(record)
            # A valid physical snapshot must survive an optional log, signal or audit failure.
            states = None
            available = True
            try:
                with np.errstate(over="raise"):
                    sync_if_needed(t_now)
                    state = self._states.read() if notifying else None
                    states = (flatten(state) if state is not None else self._states.read_flat()
                              if rec.keep_states or final else None)
                    if record:
                        rec.snapshot(t_now, states)
            except (OverflowError, FloatingPointError) as exc:
                if not final:
                    return
                available = False
                stop_reason = stop_reason or f"final physical snapshot unavailable at t = {t_now:.8g} s: {exc}"

            @_SnapshotRead
            def read_logs():
                for unit in units:
                    if getattr(unit, "continuous", False):
                        rec.ctrl_sample(unit.name, t_now,
                                        lambda unit=unit: unit.ctrl.continuous_log(t_now),
                                        record=record and continuous_record_index % record_every == 0)
                return rec.last_ctrl_log

            @_SnapshotRead
            def read_signals():
                return getattr(system, "signals", lambda _t: {})(t_now)

            @_SnapshotRead
            def read_values():
                if not available:
                    return {}
                with np.errstate(over="raise"):
                    return self.watch_values(t_now, read_logs, state=state,
                                             plant=read_signals, names=value_names,
                                             ctrl_names=ctrl_names)

            def record_energy():
                rep = mdl.energy_balance(t_now, plant.state)
                if rec.keep_energy:
                    rec.energy_row(t_now, rep.columns())
                scale = rep.scale
                worst["tellegen"] = max(worst["tellegen"], abs(rep.tellegen) / scale)
                worst["balance"] = max(worst["balance"], rep.max_residual / scale)

            if record and available:
                for unit in units:
                    if getattr(unit, "continuous", False):
                        observe_limits = getattr(unit.ctrl, "observe_limits", None)
                        if observe_limits is not None:
                            _optional_output(t_now, f"controller limits {unit.name!r}",
                                             observe_limits, t_now)
                read_logs()
                if rec.keep_signals:
                    _optional_output(t_now, "plant signals",
                                     lambda: rec.plant_snapshot(t_now, read_signals()))
            if available and audit_due and _finite(plant.state):
                _optional_output(t_now, "energy balance", record_energy)
            continuous_record_index += int(record)
            # Observer failures must not be swallowed as physical-state overflow.
            if notifying:
                next_observation = observe(t_now, states, read_values,
                                           final=final, stop_reason=stop_reason if final else "")
                if next_observation is not None:
                    if final:
                        raise ValueError("a final observation cannot request another time")
                    if (not math.isfinite(next_observation) or next_observation <= t_now + _EPS
                            or next_observation > t_final + _EPS):
                        raise ValueError("an observer's next time must be finite, later, and within the run")
                    if next_observation not in pending_observations:
                        heapq.heappush(observations, next_observation)
                        pending_observations.add(next_observation)

        def watched(t_now: float) -> str:
            """Format the quantities appended to a progress line."""
            sync_if_needed(t_now)
            values = self.watch_values(t_now, rec.last_ctrl_log)
            unknown = [name for name in watch if name not in values]
            if unknown:
                raise ConfigError(f"simulation.progress.watch: unknown name(s) {unknown}; "
                                  f"known: {sorted(values)}")
            return "".join(f"   {name} = {values[name]:.6g}" for name in watch)

        if diode_conduction:
            @np.errstate(over="raise")
            def integrate(t_a: float, t_b: float, y: np.ndarray, event: bool) -> np.ndarray:
                y = diode_conduction.advance(t_a, t_b, y)
                if event and settle is not None:
                    y = settle(t_b, y)
                return project(y) if project is not None else y
        elif project is None:
            @np.errstate(over="raise")  # overflow counts as divergence
            def integrate(t_a: float, t_b: float, y: np.ndarray, event: bool) -> np.ndarray:
                y = solver(mdl.rhs, t_a, t_b, y).y
                return settle(t_b, y) if event and settle is not None else y
        else:
            @np.errstate(over="raise")  # overflow counts as divergence
            def integrate(t_a: float, t_b: float, y: np.ndarray, event: bool) -> np.ndarray:
                y = solver(mdl.rhs, t_a, t_b, y).y
                if event and settle is not None:
                    y = settle(t_b, y)
                return project(y)

        def advance(t_a: float, t_b: float, event: bool = False) -> bool:
            """Integrate over ``[t_a, t_b]``; ``event`` means the model changes at ``t_b``."""
            nonlocal t_local, stop_reason, coarse_reported
            plant.invalidate()
            try:
                plant.state = integrate(t_a, t_b, plant.state, event)
            except (OverflowError, FloatingPointError):
                stop_reason = f"diverged: the states overflowed between t = {t_a:.6g} s and {t_b:.6g} s"
                t_local = t_a
                return True
            t_local = t_b
            finite = _finite(plant.state)
            if windowed and finite:
                sync_if_needed(t_b)
                for unit in windowed:
                    unit.accumulate(t_b - t_a)
            elif finite:
                mdl.set_states(plant.state)
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
            if settle is not None:
                plant.state = settle(t_local, plant.state)
                if project is not None:
                    plant.state = project(plant.state)
                mdl.set_states(plant.state)
                plant.invalidate()

        plant.settle = settle_now

        def window_edge(t_now: float, changed: bool) -> None:
            """Restart ADC integrals from the values on the new side of a plant/input jump."""
            if windowed and (changed or plant.zoh_changed):
                sync_if_needed(t_now)
                for unit in windowed:
                    unit.restart_windows()

        # ---------------------------------------------------------------- the event grid
        self._begin_units(t_start, plant)
        while True:
            plant.inputs_changed = False
            t_log = n_log * log_period
            t_stop = t_log
            for unit in units:
                t_stop = min(t_stop, unit.next_time())
            if actions:
                t_stop = min(t_stop, actions[0][0])
            if t_stop > t_final - _EPS:
                t_stop = t_final
            native_stop = t_stop
            if observations and observations[0] < t_stop - _EPS:
                t_stop = observations[0]
            observation_only = t_stop < native_stop - _EPS
            observation_due = bool(observations and observations[0] <= t_stop + _EPS)
            while observations and observations[0] <= t_stop + _EPS:
                pending_observations.discard(heapq.heappop(observations))
            event_due = bool(actions and actions[0][0] <= t_stop + _EPS)
            advancing = t_stop > t_local + _EPS
            if advancing and advance(t_local, t_stop, event_due):
                stopped = True
                break
            if observation_only:
                snapshot(t_local, record=False)
                continue
            # File events run before the coincident protection/control/PWM events.
            if event_due:
                if advancing:  # advance() already settled at t_stop
                    mdl.set_states(plant.state)
                else:
                    settle_now()
                while actions and actions[0][0] <= t_stop + _EPS:
                    self._apply_change(actions.popleft()[2])
                if parameters_changed is not None:
                    parameters_changed()
                plant.read_state()
                plant.inputs_changed = True
            # Every unit protects first, then every unit samples, then every unit actuates.  This
            # makes coincident multi-unit events independent of the units' dictionary order.
            protection_changed = False
            for unit in units:
                if unit.protect(t_stop, plant):
                    protection_changed = True
                    stopped = stop_on_trip
            if stopped:
                break
            for unit in units:
                done = unit.sense(t_stop, plant)
                if done is not None:
                    k, out = done
                    if out.log is not None:
                        rec.ctrl_sample(unit.name, t_stop, out.log, record=k % record_every == 0)
            for unit in units:
                if unit.actuate(t_stop, plant):
                    protection_changed = True
                    stopped = stop_on_trip
            if stopped:
                break
            window_edge(t_local, plant.inputs_changed)
            if t_stop >= t_final - _EPS:
                break
            if abs(t_log - t_stop) < _EPS:
                snapshot(t_local, notify=(observe_output or observation_due
                                          or event_due or protection_changed))
                n_log += 1
            elif observe is not None and (event_due or protection_changed or observation_due):
                snapshot(t_local, record=False)
            if t_local >= next_report:
                suffix = watched(t_local) if watch else ""
                print(f"  t = {t_local:8.4f} s   wall {time.time() - wall0:8.1f} s{suffix}",
                      flush=True)
                next_report += progress_every

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
    """Complete closed-loop map rebuilt through the existing named-state restart path.

    ``frame_frequency`` is in Hz (default: system base); zero keeps stationary coordinates.
    ``repeat_forcing`` includes waveform periods, as required for a periodic stability result.
    The default retains split-error analysis's local common-clock map. An explicit ``period``
    must repeat the selected schedules. Histories are disabled without changing their time grid.
    """

    def __init__(self, sim: Any, *, period: float | None = None,
                 frame_frequency: float | None = None, repeat_forcing: bool = False) -> None:

        self._cls = Simulation
        self._factories = dict(sim._factories)
        self.split_reference = bool(getattr(sim.solver, "groups", ()))
        frequency = sim.p.base.f0 if frame_frequency is None else frame_frequency
        self.w0 = 2.0 * math.pi * frequency
        periods = {value for unit in sim.units.values() for value in unit.periods}
        self._angles: set[str] = set()
        control_columns: set[str] = set()
        for name, unit in sim.units.items():
            ctrl = unit.ctrl
            graph = getattr(ctrl, "graph", None)
            if graph is None:
                continue
            for local in getattr(ctrl, "state_names", ()):
                control_columns.update((f"{name}.ctrl.{local}.re", f"{name}.ctrl.{local}.im"))
            measured = None
            for label, node in graph.nodes.items():
                for port, spec in node.outputs.items():
                    if spec.unit == "rad":
                        self._angles.update((f"{name}.ctrl.{label}.{port}",
                                             f"{name}.ctrl.held.{graph._state_port(label + '.' + port)}"))
                forcing = getattr(node, "forcing_periods", None)
                if repeat_forcing and forcing is not None:
                    if measured is None:
                        measured = SimpleNamespace(**{
                            key.partition(".")[2]: value
                            for key, value in unit.control_inputs().items()
                        })
                    periods.update(forcing(graph._gather(graph._wiring[label], measured)))
            self._angles.update(f"{name}.ctrl.command.{port}"
                                for port, spec in ctrl.interface.outputs.items() if spec.unit == "rad")
        if repeat_forcing:
            for source in getattr(sim.system, "sources", {}).values():
                relative = abs(source.cfg.f - frequency)
                if relative > 0.0 and source.emf.e_peak != 0.0:
                    periods.add(1.0 / relative)
        if any(not math.isfinite(value) or value <= 0.0 for value in periods):
            raise ValueError("loop periods must be finite and positive")
        self.periods = sorted(periods)
        if period is not None:
            if not math.isfinite(period) or period <= 0.0:
                raise ValueError("period must be finite and positive")
            if any(not math.isclose(period / value, round(period / value), rel_tol=1e-9, abs_tol=1e-9)
                   or period < value * (1.0 - 1e-9) for value in self.periods):
                raise ValueError(f"period must repeat the loop schedules {self.periods}")
        self.period = macro_period(self.periods) if period is None else float(period)
        flags = set(sim._states.boolean_names)
        self._flags = {name for name in flags
                       if name.rpartition(".")[2] in ("tripped", "fault")}
        self._modes = set(sim._states.discrete_names) - self._flags
        self._states = sim._states
        self._map_columns = set(sim._states.linearization_columns)
        self._plant_columns = set(sim._states.read_flat(["", *sim.units])) - control_columns
        rated = getattr(sim.solver, "rated_effort", None)
        self._rated = ({lab: float(r) for lab, r in zip(sim.system.model.state_labels(), rated)}
                       if rated is not None else {})
        p = sim.p
        sp = p.simulation
        self._p = dataclasses.replace(
            p,
            simulation=dataclasses.replace(
                sp, energy_check="off", stop_on_trip=False,
                progress=dataclasses.replace(sp.progress, enable=False),
                                           output=dataclasses.replace(sp.output, states=False,
                                                                      energy=False, signals=False),
                                           solver=dataclasses.replace(sp.solver, subsystems={},
                                                                      linearisations=0)))

    # ------------------------------------------------------------ coordinates
    def overwritten_columns(self, t: float) -> frozenset[str]:
        """Reuse owner proofs only while file events cannot change the execution schedule."""
        if any(t < change.t <= t + self.period + _EPS for change in self._p.changes):
            return frozenset()
        return self._states.overwritten_columns(self.period)

    def periodic_state(self, row: Mapping[str, float]) -> dict[str, float]:
        """Logical snapshot for recurrence checks, retaining non-modal protection history."""
        return self._states.periodic_state(row)

    def coordinates(self, names: list[str], *, linearization: bool = True
                    ) -> list[tuple[str, list[str], float]]:
        """Group state-table columns into map coordinates ``(kind, names, scale)``."""
        cols: list[tuple[str, list[str], float]] = []
        used: set[str] = set()
        selected = self._map_columns if linearization else set(self._states.columns)
        for c in names:
            if c in used or c not in selected or c.startswith("solver."):
                continue  # Select existing columns; solver bookkeeping is never a loop state.
            if c in self._flags:
                cols.append(("frozen", [c], 1.0))
                used.add(c)
            elif c in self._modes:
                cols.append(("mode", [c], 1.0))
                used.add(c)
            elif c in self._angles:
                cols.append(("angle", [c], 1.0))
                used.add(c)
            elif c.endswith(".re") and c[:-3] + ".im" in names:
                base = c[:-3]
                turns = c in self._plant_columns  # physical vectors rotate; control memory stays in its frame
                scale = self._rated.get(base + ".re", 0.0) if turns else 0.0
                cols.append(("vector" if turns else "fixed", [c, base + ".im"], scale if scale > 0 else 1.0))
                used.update(cols[-1][1])
            elif (self.w0 != 0.0 and c.endswith(".d_a")
                  and c[:-4] + ".d_b" in names and c[:-4] + ".d_c" in names):
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
            elif kind in ("scalar", "angle"):
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
        out = res.final_states()
        if res.summary.get("stop_reason") or not math.isclose(
                out.pop("t"), t + self.period, rel_tol=0.0, abs_tol=_EPS):
            raise FloatingPointError("loop advance did not complete its analysis period")
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
                    help="export an implementation: FORMAT followed by parameter paths, common or all")
    ap.add_argument("--imperix", action="store_true", help="Imperix ELF export using the official SDK")
    ap.add_argument("--imperix-config", metavar="FILE", help="ADC/PWM bindings; default: PESFILE.imperix.json")
    ap.add_argument("--imperix-sdk", metavar="DIRECTORY", help="installed official Imperix SDK")
    ap.add_argument("--imperix-source-only", action="store_true", help="generate Imperix sources without building an ELF")
    ap.add_argument("--progress", type=float, default=None, metavar="SECONDS",
                    help="print a progress line every SECONDS of simulated time")
    ap.add_argument("--watch", action="append", default=None, metavar="NAME",
                    help="append a state or plant/control value to each progress line; repeatable, "
                         "or use NAME,NAME")
    ap.add_argument("--list-states", action="store_true",
                    help="print the state names generated for this configuration and exit")
    ap.add_argument("--ph-report", action="store_true",
                    help="print the port-Hamiltonian structure report of the system (always built) and exit")
    ap.add_argument("--report", action="store_true",
                    help="generate <simulation name>_report.json using the optional analysis addon")
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
    for flag, enabled in (("--plot", args.plot is not None), ("--report", args.report)):
        if enabled and (args.export or args.imperix or args.list_states or args.ph_report or args.resolved):
            ap.error(f"{flag} requires a simulation run and cannot be combined with inspection or export")

    if args.imperix:
        if args.export and args.export[0].lower() != "imperix":
            ap.error("--imperix cannot be combined with another export format")
        args.export = args.export or ["imperix"]
    hardware = bool(args.export and args.export[0].lower() == "imperix")
    if not hardware and (args.imperix_config or args.imperix_sdk or args.imperix_source_only):
        ap.error("Imperix options require --imperix or --export imperix")

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
        changes = {f"units.{name}.bridge.model": model for name, _cfg in _unit_items(p.units)
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
        options = {}
        if hardware:
            options = dict(target_config=args.imperix_config or config.with_suffix(".imperix.json"),
                           sdk=args.imperix_sdk, build=not args.imperix_source_only)
        try:
            result = sim.export(export_format, out, name=f"{config.stem}{suffix}",
                                variables=export_variables, **options)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            ap.error(str(exc))
        for diagnostic in result.diagnostics:
            print("warning:", diagnostic["message"], file=sys.stderr)
        print("wrote", ", ".join(str(path) for path in result.files))
        return 0
    units = ", ".join(f"{n} ({u.ctrl.type}, {u.bridge.model.replace('_', ' ')}, "
                      f"{sim.units[n].bridge.describe()})"
                      for n, u in _unit_items(p.units))
    print(f"peslite: {config}  units={units}  "
          f"solver={p.simulation.solver.type}/{p.simulation.solver.method}  "
          f"t = {p.simulation.initial.t} .. {p.simulation.t_end} s  "
          f"({len(p.simulation.initial.states)} initial values given)")
    if sim.ph_report is not None:
        print(f"structure: {sim.ph_report.verdict} (state coverage {sim.ph_report.coverage:.0%}"
              f"{'; ' + '; '.join(sim.energy_problems) if sim.energy_problems else ''})")
    out = Path(args.out) if args.out else _RESULTS / f"{config.stem}{suffix}"
    reporter = None
    if args.report:
        module_name = "peslite.addons.functions.small_signal"
        try:
            addon = import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name is None or not (exc.name == module_name or module_name.startswith(exc.name + ".")):
                raise  # An installed addon's missing dependency is not an absent addon.
        else:
            reporter = getattr(addon, "small_signal_report", None)
        if not callable(reporter):
            warnings.warn("--report: the small-signal analysis addon is unavailable; "
                          "continuing the simulation without a report", RuntimeWarning, stacklevel=2)
            reporter = None
    if reporter is not None:
        reporter(sim, out_dir=out, name=config.stem, info={"config": str(config)})
        r = sim.result
    else:
        r = sim.run(out_dir=out, info={"config": str(config)}, name=config.stem)
    s = r.summary
    print(f"done: wall {r.wall_time:.1f} s, rhs evaluations {r.n_rhs}, stop at {s.get('t_stop', 0):.4f} s, "
          f"tripped={bool(s.get('tripped'))}")
    for name, _cfg in _unit_items(p.units):
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
    """Export a standalone C++ simulator or an ELF executable for Imperix hardware."""
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in ("-h", "--help"):
        print("usage: peslite-convert PESFILE [PARAM ...] [--out DIRECTORY]")
        print("export peslite.cpp, or use --imperix for Imperix ELF export")
        print("Imperix options: --imperix-config FILE, --imperix-sdk DIRECTORY, --imperix-source-only")
        print("PARAM may be a variable path, common or all")
        return 0
    if not values:
        print("usage: peslite-convert PESFILE [PARAM ...]", file=sys.stderr)
        return 2
    export_format = "imperix" if "--imperix" in values else "cpp"
    return main([values[0], "--export", export_format, *values[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
