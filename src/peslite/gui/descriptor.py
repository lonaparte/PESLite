"""Translate a PESLite configuration and its run into the frontend descriptor schema.

The PESLite frontend draws a topology of buses, devices and connections, edits flat system
parameters, plots named signals and shows small-signal modes. This module derives those views from
an ordinary :class:`~peslite.assembly.params.Params` tree and read-only run observations; it adds
no simulation behaviour.
"""

from __future__ import annotations

import cmath
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..assembly.params import Params, _unit_items, flat_paths, to_dict
from ..assembly.types import unit_type
from ..control.blocks import complex2abc

SCHEMA_VERSION = 1

# Parameter paths that are not useful or not representable as one flat override.
_HIDDEN_PREFIXES = ("meta.", "simulation.initial.", "simulation.progress.")
# ``events.<name>.set`` maps hold dotted keys themselves; they are shown, not edited.
_EVENT_SET = re.compile(r"^events\.[^.]+\.set\.")
_SLUG = re.compile(r"[^a-zA-Z0-9_-]")


def slug(text: str) -> str:
    """Return a frontend-safe identifier."""
    return _SLUG.sub("-", text).lower()


def json_safe(value: Any) -> Any:
    """Convert NumPy values, complex numbers and non-finite floats to JSON values."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"real": json_safe(value.real), "imaginary": json_safe(value.imag),
                "magnitude": json_safe(abs(value))}
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def editable(path: str, value: Any) -> bool:
    """Whether one flattened parameter can be shown as an editable dotted-path override."""
    if path.startswith(_HIDDEN_PREFIXES) or path.endswith(".type") or _EVENT_SET.match(path):
        return False
    if value is None or isinstance(value, (dict, complex)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return isinstance(value, (bool, int, float, str, list))


def _parameter(path: str, label: str, value: Any, group: str, **extra: Any) -> dict[str, Any]:
    return {"id": path, "label": label, "value": json_safe(value), "unit": "", "group": group,
            **extra}


def _source_lines(path: Path) -> dict[tuple[str, str], int]:
    """Map ``(section, name)`` to the 1-based line declaring it in a simulation file."""
    lines: dict[tuple[str, str], int] = {}
    try:
        text = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return lines
    section, indent = "", None
    for number, line in enumerate(text, 1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][\w-]*)\s*:", line)
        if match:
            section, indent = match.group(1), None
            lines.setdefault((section, ""), number)
            continue
        width = len(line) - len(stripped)
        if section and indent is None:
            indent = width
        match = re.match(r"^([^\s:#{][^:]*?)\s*:", stripped)
        if section and width == indent and match:
            lines.setdefault((section, match.group(1).strip("'\"")), number)
    return lines


@dataclass(frozen=True)
class Entity:
    """One configured network entity drawn by the frontend."""

    name: str  # bus, source, branch or element name, or a unit path
    section: str
    id: str
    owner_kind: str  # "component" or "connection"


@dataclass
class Topology:
    """Frontend topology and parameter views of one loaded configuration."""

    case: str
    path: Path
    params: Params
    components: list[dict[str, Any]] = field(default_factory=list)
    connections: list[dict[str, Any]] = field(default_factory=list)
    entities: dict[str, Entity] = field(default_factory=dict)
    # The connection carrying each device's terminal current (bus-attached devices only).
    terminals: dict[str, str] = field(default_factory=dict)
    _tree: dict[str, Any] = field(init=False, repr=False)
    _editable: dict[str, Any] = field(init=False, repr=False)
    _system: dict[str, Any] = field(init=False, repr=False)
    _lines: dict[tuple[str, str], int] = field(init=False, repr=False)
    _unit_paths: tuple[str, ...] = field(init=False, repr=False)

    @classmethod
    def build(cls, case: str, path: Path, params: Params) -> "Topology":
        topology = cls(case, path.resolve(), params)
        topology._tree = to_dict(params)
        topology._lines = _source_lines(topology.path)
        units = tuple(_unit_items(params.units, leaves_only=False))
        topology._unit_paths = tuple(sorted((name for name, _unit in units),
                                            key=len, reverse=True))
        topology._build(units)
        topology._editable = {
            name: value for name, value in flat_paths(topology._tree).items()
            if editable(name, value)
        }
        topology._system = {
            "id": topology.system_id,
            "name": params.meta.title or case,
            "className": f"peslite:{topology.path.name}",
            "parameterNames": list(topology._editable),
            "parameters": [
                _parameter(name, name, value, name.partition(".")[0], editable=True)
                for name, value in topology._editable.items()
            ],
            "source": topology._source("", ""),
        }
        return topology

    # ------------------------------------------------------------ system
    @property
    def system_id(self) -> str:
        return slug(self.case)

    def editable_parameters(self) -> dict[str, Any]:
        """The current value of every parameter the system editor may override."""
        return self._editable

    def system(self) -> dict[str, Any]:
        return self._system

    # ------------------------------------------------------------ topology
    def _source(self, section: str, name: str) -> dict[str, Any]:
        line = self._lines.get((section, name.partition(".")[0])) or self._lines.get((section, ""), 1)
        return {"file": str(self.path), "line": line, "column": 1}

    def _component(self, entity: Entity, label: str, class_name: str, kind: str, icon: str,
                   tree: Mapping[str, Any], port_type: str = "") -> None:
        prefix = f"{entity.section}.{entity.name}"
        parameters = [_parameter(f"{prefix}.{path}", path, value, entity.section)
                      for path, value in flat_paths(dict(tree)).items()
                      if not isinstance(value, dict)]
        source = self._source(entity.section, entity.name)
        self.components.append({
            "id": entity.id, "name": label, "className": class_name, "kind": kind, "icon": icon,
            "portType": port_type, "parameters": parameters,
            "classSource": source, "instanceSource": source,
        })
        self.entities.setdefault(entity.name, entity)

    def _connect(self, connection_id: str, name: str, source: str, target: str, section: str,
                 entity: str, parameters: Iterable[dict[str, Any]] = (),
                 handles: tuple[str, str] = ("bottom", "top")) -> None:
        self.connections.append({
            "id": connection_id, "name": name, "source": source, "target": target,
            "sourceHandle": handles[0], "targetHandle": handles[1],
            "parameters": list(parameters), "sourceReference": self._source(section, entity),
        })

    def _attach_point(self, bus: str | None) -> str | None:
        """The component a ``bus`` reference attaches to: a public bus or a unit's scoped bus."""
        if not bus:
            return None
        if bus in self.params.buses:
            return f"bus-{slug(bus)}"
        owner = next((path for path in self._unit_paths if bus.startswith(path + ".")), None)
        return f"unit-{slug(owner)}" if owner else None

    def _attach(self, entity: Entity, bus: str | None, label: str) -> None:
        point = self._attach_point(bus)
        if point is None:
            return
        connection_id = f"{entity.id}-terminal"
        self._connect(connection_id, label, point, entity.id, entity.section, entity.name)
        self.terminals[entity.name] = connection_id

    def _build(self, units: tuple[tuple[str, Any], ...]) -> None:
        p, tree = self.params, self._tree
        for name in p.buses:
            entity = Entity(name, "buses", f"bus-{slug(name)}", "component")
            phases = p.buses[name].phases
            self._component(entity, name, "peslite.Bus", "bus", "bus", tree["buses"][name],
                            "ac" if phases == 3 else "dc")
        for name, cfg in p.sources.items():
            entity = Entity(name, "sources", f"source-{slug(name)}", "component")
            self._component(entity, name, "peslite.Source", "grid", "grid", tree["sources"][name])
            self._attach(entity, cfg.bus, "Terminal")
        unit_tree = tree["units"]
        for path, cfg in units:
            entity = Entity(path, "units", f"unit-{slug(path)}", "component")
            node: Any = unit_tree
            for part in path.split("."):
                node = node[part]
            unit_class = unit_type(cfg.type)
            composite = unit_class.composite
            ctrl = cfg.ctrl.type if cfg.ctrl is not None else ""
            kind = {"gfm": "gfm", "gfl": "gfl"}.get(ctrl, "generic")
            icon = "device" if composite else "inverter"
            self._component(entity, path, f"peslite.{cfg.type}", kind, icon,
                            {key: value for key, value in node.items()
                             if not (composite and key in unit_class.children)},
                            cfg.type)
            parent = path.rpartition(".")[0]
            if parent:
                self._connect(f"{entity.id}-port", path.rpartition(".")[2],
                              f"unit-{slug(parent)}", entity.id, "units", path)
            self._attach(entity, cfg.bus, "Terminal")
        for name, cfg in p.branches.items():
            entity = Entity(name, "branches", f"branch-{slug(name)}", "connection")
            self.entities.setdefault(entity.name, entity)
            self._connect(entity.id, name, f"bus-{slug(cfg.bus1)}", f"bus-{slug(cfg.bus2)}",
                          "branches", name,
                          [_parameter(f"branches.{name}.r", "Resistance", cfg.r, "branches",
                                      unit="ohm"),
                           _parameter(f"branches.{name}.l", "Inductance", cfg.l, "branches",
                                      unit="H")],
                          ("right", "left"))
        for name, cfg in tree["elements"].items():
            points = [self._attach_point(cfg.get(key)) for key in ("bus1", "bus2")]
            if all(points):  # a two-terminal network element is drawn as a connection
                entity = Entity(name, "elements", f"element-{slug(name)}", "connection")
                self.entities.setdefault(entity.name, entity)
                self._connect(entity.id, name, points[0], points[1], "elements", name,
                              [_parameter(f"elements.{name}.{key}", key, value, "elements")
                               for key, value in flat_paths(cfg).items()],
                              ("right", "left"))
                continue
            entity = Entity(name, "elements", f"element-{slug(name)}", "component")
            element_type = str(cfg.get("type", ""))
            kind = ("capacitor" if "capacitor" in element_type
                    else "load" if any(word in element_type for word in ("load", "resistor"))
                    else "motor" if "motor" in element_type else "generic")
            self._component(entity, name, f"peslite.{element_type or 'element'}", kind, kind, cfg)
            self._attach(entity, cfg.get("bus"), "Terminal")

    def owner(self, signal: str) -> Entity | None:
        """The entity whose name is the longest dotted prefix of a state or signal name."""
        name = signal
        while name:
            entity = self.entities.get(name)
            if entity is not None:
                return entity
            name, separator, _part = name.rpartition(".")
            if not separator:
                break
        return None

    def descriptor(self, signals: list[dict[str, Any]] | None = None,
                   small_signal: Mapping[str, Any] | None = None,
                   layout: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The complete frontend descriptor, with unavailable analyses marked accordingly."""
        descriptor = {
            "schemaVersion": SCHEMA_VERSION,
            "system": self.system(),
            "components": self.components,
            "connections": self.connections,
            "signals": signals or [],
            "smallSignal": dict(small_signal or {"available": False}),
            "interactionAnalysis": {"available": False},
            "transientVoltage": {"available": False},
        }
        if layout:
            descriptor["layout"] = dict(layout)
        return descriptor


# ------------------------------------------------------------------ signals

@dataclass
class _Trace:
    id: str
    owner: str
    owner_kind: str
    category: str
    label: str
    unit: str
    values: list[float] = field(default_factory=list)

    def describe(self, time: list[float], start: int = 0) -> dict[str, Any]:
        return {
            "id": self.id, "componentId": self.owner, "ownerId": self.owner,
            "ownerKind": self.owner_kind, "category": self.category, "label": self.label,
            "unit": self.unit, "time": time[start:],
            "values": [value if math.isfinite(value) else None for value in self.values[start:]],
        }


class SignalRecorder:
    """Collect frontend signals from ``Simulation.run(observe=...)`` snapshots.

    The observer reads plant signals, controller logs and states only after the run has
    synchronised its outputs, and never changes simulation state. ``emit`` receives each chunk of
    new samples, already in the frontend signal schema.
    """

    def __init__(self, topology: Topology, sim: Any, *, chunk_size: int = 50,
                 emit: Callable[[list[dict[str, Any]]], None] | None = None) -> None:
        self.topology = topology
        self.system = sim.system
        self.units = sim.units
        self.base = topology.params.base
        self.chunk_size = chunk_size
        self.emit = emit
        self.observe_values = tuple(
            f"{name}.{column}" for name, unit in self.units.items() if unit.ctrl is not None
            for column in getattr(unit.ctrl, "log_names", lambda: ())()
        )
        self.traces: dict[str, _Trace] = {}
        self.time: list[float] = []
        self._emitted = 0
        self.stop_reason = ""

    # ------------------------------------------------------------ the observer
    def observe(self, t: float, states: Mapping[str, float] | None, read_values: Callable[[], Any],
                *, final: bool, stop_reason: str) -> None:
        if final and stop_reason:
            self.stop_reason = stop_reason
        if states is None or (self.time and t <= self.time[-1]):
            return None  # an unavailable terminal state, or an event already at this time
        signals = self.system.signals(t)
        logs = read_values() if self.observe_values else {}
        row: dict[tuple, float] = {}
        self._derived(row, signals)
        self._plant(row, signals)
        for name in self.observe_values:
            self._put(row, name, "output", name, _unit_of(name), logs.get(name, math.nan))
        for name, value in states.items():
            self._put(row, name, "state", name, "", value)
        self._append(t, row)
        return None

    def _put(self, row: dict, key: str, category: str, label: str, unit: str, value: float,
             owner: tuple[str, str] | None = None) -> None:
        row[(category, key)] = (label, unit, owner, float(value))

    def _append(self, t: float, row: dict) -> None:
        if not self.traces and row:
            used: set[str] = set()
            for key, (label, unit, owner, _value) in row.items():
                if owner is None:
                    entity = self.topology.owner(key[1])
                    owner = ((entity.id, entity.owner_kind) if entity is not None
                             else (self.topology.system_id, "system"))
                trace_id = slug(f"{owner[0]}-{label}")
                while trace_id in used:
                    trace_id += "_"
                used.add(trace_id)
                self.traces[key] = _Trace(trace_id, owner[0], owner[1], key[0], label, unit)
        self.time.append(float(t))
        for key, trace in self.traces.items():
            item = row.get(key)
            trace.values.append(item[3] if item is not None else math.nan)
        if self.emit is not None and len(self.time) - self._emitted >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        """Emit every sample not yet sent."""
        if self.emit is not None and len(self.time) > self._emitted:
            self.emit([trace.describe(self.time, self._emitted)
                       for trace in self.traces.values()])
            self._emitted = len(self.time)

    def signals(self) -> list[dict[str, Any]]:
        return [trace.describe(self.time) for trace in self.traces.values()]

    # ------------------------------------------------------------ signal views
    def _derived(self, row: dict, signals: Mapping[str, Any]) -> None:
        """Bus voltage, connection current, device power and bus angle for frontend plots.

        AC quantities are amplitude-invariant space vectors: magnitudes are in pu of the system
        base and ``p + jq = 1.5 u i*``. Scalar (DC) voltage and current stay in SI; power is in
        pu of the system ``s_base`` for both. Device power is positive when delivered to its bus.
        """
        topology, p, base = self.topology, self.topology.params, self.base
        v_base, i_base, s_base = base.v_phase_peak or 1.0, base.i_phase_peak or 1.0, base.s_base
        reference = next((signals[f"{name}.angle"] for name in p.sources
                          if f"{name}.angle" in signals), None)

        def bus_voltage(name: str | None):
            return signals.get(f"{name}.u") if name else None

        voltages: list[tuple] = []
        currents: list[tuple] = []
        powers: list[tuple] = []
        angles: list[tuple] = []
        for name in p.buses:
            u = signals.get(f"{name}.u")
            if u is None:
                continue
            owner = (f"bus-{slug(name)}", "component")
            if isinstance(u, complex):
                voltages.append((name, "pu", abs(u) / v_base, owner))
                if reference is not None:
                    angle = math.degrees(math.remainder(cmath.phase(u) - reference, math.tau))
                    angles.append((name, "deg", angle if abs(u) > 0.0 else math.nan, owner))
            else:
                voltages.append((name, "V", abs(float(u)), owner))
        for name in p.branches:
            i = signals.get(f"{name}.i")
            if i is not None:
                currents.append((name, *_current(i, i_base), (f"branch-{slug(name)}", "connection")))

        def device(name: str, u, i, sign: float, owner_id: str) -> None:
            terminal = topology.terminals.get(name)
            owner = (terminal, "connection") if terminal else (owner_id, "component")
            if i is not None:
                currents.append((name, *_current(i, i_base), owner))
            if u is None or i is None:
                return
            component = (owner_id, "component")
            if isinstance(u, complex) and isinstance(i, complex):
                s = sign * 1.5 * u * i.conjugate() / s_base
                powers.append(("P", name, s.real, component))
                powers.append(("Q", name, s.imag, component))
            elif not isinstance(u, complex) and not isinstance(i, complex):
                powers.append(("P", name, sign * float(u) * float(i) / s_base, component))

        for name, cfg in p.sources.items():
            # A source current enters its bus from the grid emf.
            device(name, bus_voltage(cfg.bus), signals.get(f"{name}.i"), 1.0,
                   f"source-{slug(name)}")
        for name in self.units:
            device(name, signals.get(f"{name}.u_g"), signals.get(f"{name}.i_c"), 1.0,
                   f"unit-{slug(name)}")
        for name, cfg in p.elements.items():
            entity = topology.entities.get(name)
            if entity is None:
                continue
            i = signals.get(f"{name}.i", signals.get(f"{name}.i1"))
            if entity.owner_kind == "connection":
                if i is not None:
                    currents.append((name, *_current(i, i_base), (entity.id, "connection")))
                continue
            terminal = topology.terminals.get(name)
            if i is not None:
                currents.append((name, *_current(i, i_base),
                                 (terminal, "connection") if terminal else (entity.id, "component")))
            if f"{name}.power" in signals:  # an element reports the power it absorbs
                powers.append(("P", name, -float(signals[f"{name}.power"]) / s_base,
                               (entity.id, "component")))

        for name, unit, value, owner in voltages:
            self._put(row, f"|V|-{name}", "output", f"|V|-{name}", unit, value, owner)
        for name, unit, value, owner in currents:
            self._put(row, f"I-{name}", "output", f"I-{name}", unit, value, owner)
        for which, name, value, owner in sorted(powers, key=lambda item: item[0]):
            self._put(row, f"{which}-{name}", "output", f"{which}-{name}", "pu", value, owner)
        for name, unit, value, owner in angles:
            self._put(row, f"theta-{name}", "output", f"theta-{name}", unit, value, owner)

    def _plant(self, row: dict, signals: Mapping[str, Any]) -> None:
        """Every plant signal under its ``plant.csv`` column name (SI, phases for AC vectors)."""
        for key, value in signals.items():
            if isinstance(value, complex):
                for suffix, phase in zip(("_a", "_b", "_c"), complex2abc(value)):
                    self._put(row, key + suffix, "output", key + suffix, "", float(phase))
            else:
                self._put(row, key, "output", key, _unit_of(key), float(value))


def _current(value, i_base: float) -> tuple[str, float]:
    if isinstance(value, complex):
        return "pu", abs(value) / i_base
    return "A", abs(float(value))


def _unit_of(name: str) -> str:
    return "pu" if name.endswith("_pu") else ""


# ------------------------------------------------------------------ small-signal modes

def small_signal_descriptor(result: Any) -> dict[str, Any]:
    """Serialize a :class:`SmallSignalResult` as continuous-time modes with participation.

    The period map ``phi`` has multipliers ``mu``; each mode is reported by its equivalent
    exponent ``log(mu) / period`` (principal logarithm, so frequency is ambiguous modulo
    ``1 / period``). Deadbeat multipliers (``mu == 0``, from sample-and-hold memory) have no
    finite exponent and are counted rather than plotted.
    """
    from scipy.linalg import eig

    phi = np.asarray(result.phi, dtype=float)
    if not result.periodic or phi.size == 0:
        return {"available": False, "stability": result.stability,
                "diagnostics": list(result.diagnostics)}
    names = list(result.coordinates)
    period, tolerance = float(result.period), float(result.multiplier_tolerance)
    multipliers, left, right = eig(phi, left=True, right=True)
    order = np.argsort(-np.abs(multipliers), kind="stable")
    modes, deadbeat = [], 0
    for k in order:
        mu = complex(multipliers[k])
        size = abs(mu)
        if size <= 1e-12:
            deadbeat += 1
            continue
        exponent = complex(math.log(size), cmath.phase(mu)) / period
        participation = np.abs(left[:, k].conj() * right[:, k])
        total = participation.sum()
        if total > 0.0:
            participation = participation / total
        dominant = int(np.argmax(participation))
        modes.append({
            "mode": len(modes) + 1,
            "realPart": exponent.real,
            "imaginaryPart": exponent.imag,
            "frequencyHz": abs(exponent.imag) / math.tau,
            "dampingRatio": -exponent.real / max(abs(exponent), np.finfo(float).eps),
            "stability": ("Unstable" if size > 1.0 + tolerance
                          else "Stable" if size < 1.0 - tolerance else "Marginal"),
            "dominantState": names[dominant] if names else "",
            "dominantParticipation": float(participation[dominant]),
            "multiplier": {"real": mu.real, "imaginary": mu.imag, "magnitude": size},
        })
    return {
        "available": True,
        "normalizedTime": False,
        "unstablePoleCount": sum(mode["stability"] == "Unstable" for mode in modes),
        "poleTolerance": tolerance,
        "modes": modes,
        "stability": result.stability,
        "operatingTime_s": float(result.at),
        "period_s": period,
        "periodicError": float(result.periodic_error),
        "amplification": float(result.amplification),
        "deadbeatModeCount": deadbeat,
        "coordinates": names,
        "diagnostics": list(result.diagnostics),
    }
