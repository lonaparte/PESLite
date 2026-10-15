"""The system: named buses, branches, sources, units and circuit elements wired into one model.

It also exposes the component operations used by simulation-file events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..components.network import ELEMENT_TYPES, RCNode, RLBranch, Terminal, ThreePhaseSource
from ..components.wpt import make_wpt_link
from ..solver.energy import spec_of
from ..solver.model import ConfigError, Model, gather
from .events import SWITCHING, SourceScenario
from .params import Change, Params, _unit_at, _unit_leaves
from .types import unit_type
from .unit import DCConverter, Unit, VSC

__all__ = ["System"]


@dataclass
class _UnitAssembly:
    """One configuration node expanded into runtime leaf units and shared subsystems."""

    units: dict[str, Unit] = field(default_factory=dict)
    subsystems: dict[str, Any] = field(default_factory=dict)
    connections: dict = field(default_factory=dict)
    signal_sources: tuple[Any, ...] = ()
    power_port: Any = None

    def include(self, other: "_UnitAssembly") -> None:
        """Merge one child assembly while preserving its single public unit mapping."""
        unit_overlap = set(self.units) & set(other.units)
        subsystem_overlap = set(self.subsystems) & set(other.subsystems)
        connection_overlap = set(self.connections) & set(other.connections)
        if unit_overlap:
            raise ConfigError(f"assembled unit name collision: {sorted(unit_overlap)}")
        if subsystem_overlap:
            raise ConfigError(
                f"assembled subsystem name collision: {sorted(subsystem_overlap)}"
            )
        if connection_overlap:
            raise ConfigError(
                f"assembled connection target collision: "
                f"{[repr(item) for item in connection_overlap]}"
            )
        self.units.update(other.units)
        self.subsystems.update(other.subsystems)
        self.connections.update(other.connections)
        self.signal_sources += other.signal_sources


class _WPTSignals:
    """Expose shared link quantities without making the composite a scheduled unit."""

    def __init__(self, name: str, link: Any) -> None:
        self.name = name
        self.link = link

    def signals(self) -> dict[str, float]:
        out = self.link.out
        return {
            f"{self.name}.link.i_primary": out.i_primary,
            f"{self.name}.link.i_secondary": out.i_secondary,
            f"{self.name}.link.i_primary_coil": out.i_primary_coil,
            f"{self.name}.link.i_secondary_coil": out.i_secondary_coil,
        }


class _Source:
    """Voltage source (emf) behind a series R-L branch."""

    def __init__(self, name: str, cfg, base, steps, t0: float) -> None:
        self.name, self.cfg = name, cfg
        self.scenario = SourceScenario(cfg, base.f0, steps)
        self.emf = ThreePhaseSource(base.w0, *self.scenario.law(t0))
        self.branch = RLBranch(cfg.l, cfg.r)

    def retune(self, cfg, paths: list[str], t: float) -> None:
        """Apply changed source impedance, magnitude, frequency or angle at ``t``."""
        self.cfg = cfg
        if "l" in paths or "r" in paths:
            self.branch.retune(cfg.l, cfg.r)
        if "v" in paths or "f" in paths or "angle" in paths:
            voltage, phase = self.scenario.law(t)
            if "v" in paths:
                self.emf.e_peak = voltage
            if "f" in paths or "angle" in paths:
                self.emf.phi = phase

    def subsystems(self) -> dict[str, Any]:
        return {f"{self.name}.emf": self.emf, f"{self.name}.branch": self.branch}

    def connections(self) -> dict:
        return {(self.branch, "u1"): (self.emf, "e_g"),
                (self.emf, "i"): (self.branch, "i")}  # energy accounting only

    def terminals(self) -> tuple[tuple[str, Terminal], ...]:
        return ((self.cfg.bus, self.branch.terminal2),)

    def signals(self) -> dict[str, float | complex]:
        return {f"{self.name}.i": self.branch.out.i, f"{self.name}.angle": self.emf.out.phi}


class _BranchAssembly:
    """Give a configured bare RLBranch the common network-component assembly interface."""

    def __init__(self, name: str, cfg, branch: RLBranch) -> None:
        self.name, self.cfg, self.branch = name, cfg, branch

    def subsystems(self) -> dict[str, RLBranch]:
        return {self.name: self.branch}

    @staticmethod
    def connections() -> dict:
        return {}

    def terminals(self) -> tuple[tuple[str, Terminal], ...]:
        return ((self.cfg.bus1, self.branch.terminal1),
                (self.cfg.bus2, self.branch.terminal2))


def _part(parts: Mapping[str, Any], name: str, suffix: str = "") -> Any:
    key = f"{name}.{suffix}" if suffix else name
    return parts.get(key)


def _owner(names: Mapping[str, Any], value: str, default: Optional[str] = None) -> Optional[str]:
    """Return the longest runtime name that is equal to or prefixes ``value``."""
    return next(
        (name for name in sorted(names, key=len, reverse=True)
         if value == name or value.startswith(name + ".")),
        default,
    )


def _assemble_vsc(name: str, cfg: Any, system: "System",
                  parts: Mapping[str, Any]) -> _UnitAssembly:
    unit = _part(parts, name) or VSC(
        name,
        cfg,
        system.p.simulation,
        system.buses[cfg.bus],
        ctrl=_part(parts, name, "ctrl"),
        modulator=_part(parts, name, "modulator"),
    )
    return _UnitAssembly(units={name: unit})


def _assemble_dc_converter(name: str, cfg: Any, system: "System",
                           parts: Mapping[str, Any]) -> _UnitAssembly:
    unit = _part(parts, name) or DCConverter(
        name,
        cfg,
        system.p.simulation,
        system.buses[cfg.bus] if cfg.bus is not None else None,
        ctrl=_part(parts, name, "ctrl"),
        modulator=_part(parts, name, "modulator"),
    )
    return _UnitAssembly(
        units={name: unit},
        power_port=unit.connect_power_port if cfg.bus is None else None,
    )


def _assemble_wpt(name: str, cfg: Any, system: "System",
                  parts: Mapping[str, Any]) -> _UnitAssembly:
    if _part(parts, name) is not None:
        raise ConfigError(
            f"parts.{name}: a composite WPT unit is assembled from its primary and secondary parts"
        )
    link = make_wpt_link(cfg.link)
    assembly = _UnitAssembly(
        subsystems={f"{name}.link": link},
        signal_sources=(_WPTSignals(name, link),),
    )
    for side in unit_type(cfg.type).children:
        child_name = f"{name}.{side}"
        child_cfg = getattr(cfg, side)
        child = _assemble_unit(child_name, child_cfg, system, parts)
        if child.power_port is None:
            raise ConfigError(
                f"units.{child_name}: {child_cfg.type!r} does not expose a scalar power port"
            )
        port_connections = child.power_port(link, f"u_{side}", f"i_{side}")
        overlap = set(child.connections) & set(port_connections)
        if overlap:
            raise ConfigError(
                f"units.{child_name}: power-port connection target collision: "
                f"{[repr(item) for item in overlap]}"
            )
        child.connections.update(port_connections)
        assembly.include(child)
    return assembly


def _assemble_unit(name: str, cfg: Any, system: "System",
                   parts: Mapping[str, Any]) -> _UnitAssembly:
    assembler = unit_type(cfg.type, f"units.{name}.type").assembler
    if assembler is None:
        raise ConfigError(f"units.{name}.type: {cfg.type!r} has no runtime assembler")
    return assembler(name, cfg, system, parts)


class System:
    """Buses, branches, sources, units and elements wired into one model."""

    def __init__(self, p: Params, parts: Optional[Mapping[str, Any]] = None) -> None:
        """Build the system described by ``p``, including registered ``elements`` entries.

        parts: replacement units or unit parts keyed ``"<unit>"``, ``"<unit>.ctrl"`` or
        ``"<unit>.modulator"``.
        """
        self.p = p
        parts = dict(parts or {})
        base = p.base
        # ---------------------------------------------------------- the elements
        # Three-phase buses start at one pu; scalar buses start de-energized unless overridden.
        self.buses = {name: RCNode(
                          cfg.c,
                          cfg.r_d,
                          u0=base.v_phase_peak if cfg.phases == 3 else 0.0,
                          phases=cfg.phases,
                      )
                      for name, cfg in p.buses.items()}
        self.branches = {name: RLBranch(
                             cfg.l,
                             cfg.r,
                             phases=p.buses[cfg.bus1].phases,
                         )
                         for name, cfg in p.branches.items()}
        self.sources = {name: _Source(name, cfg, base, [
                            (change.t, change.params.sources[name]) for change in p.changes
                            if set(change.touches(f"sources.{name}.")) & {"v", "f", "angle"}],
                            p.simulation.initial.t)
                        for name, cfg in p.sources.items()}
        assembled = _UnitAssembly()
        for name, cfg in p.units.items():
            assembled.include(_assemble_unit(name, cfg, self, parts))
        self.units = assembled.units
        self.composite_signal_sources = assembled.signal_sources

        # ---------------------------------------------------------- the wiring
        subsystems: dict[str, Any] = {**self.buses, **assembled.subsystems}
        connections: dict = dict(assembled.connections)
        bus_currents: dict[str, list] = {name: [] for name in self.buses}
        self.named_elements = {name: ELEMENT_TYPES[cfg.type](name, cfg, self.buses, p)
                               for name, cfg in p.elements.items()}
        branch_assemblies = tuple(_BranchAssembly(name, p.branches[name], branch)
                                  for name, branch in self.branches.items())
        elements = (*branch_assemblies, *self.sources.values(), *self.units.values(),
                    *self.named_elements.values())
        for element in elements:
            subsystems.update(element.subsystems())
            connections.update(element.connections())
        # Every network component uses the same terminal contract. Direction is positive into the
        # component, so the current injected into a bus has the opposite sign.
        for element in elements:
            for bus_name, terminal in element.terminals():
                if not isinstance(terminal, Terminal):
                    raise TypeError(
                        f"{type(element).__name__}.terminals() returned "
                        f"{type(terminal).__name__}, not Terminal"
                    )
                if bus_name not in self.buses:
                    raise ConfigError(f"terminal refers to unknown bus {bus_name!r}")
                target = (terminal.subsystem, terminal.voltage)
                if target in connections:
                    raise ConfigError(
                        f"{type(terminal.subsystem).__name__}.{terminal.voltage} is connected both "
                        "as an electrical terminal and by connections()"
                    )
                connections[target] = (self.buses[bus_name], "u")
                bus_currents[bus_name].append(
                    (terminal.subsystem, terminal.current, -terminal.direction)
                )
        for name, node in self.buses.items():
            if not bus_currents[name]:
                raise ValueError(f"bus {name!r} has nothing attached to it")
            connections[(node, "i")] = bus_currents[name]
        zoh: dict = {}
        for unit in self.units.values():
            zoh.update(unit.zoh_connections())
        self.model = Model(subsystems, connections, zoh)

        self.state_aliases: dict[str, str] = {}
        for unit in self.units.values():
            self.state_aliases.update(unit.aliases())

    # ---------------------------------------------------------------- states
    def state_parts(self) -> list[tuple[str, Any]]:
        """Return the power model and unit state owners in state-table order."""
        parts: list[tuple[str, Any]] = [("", self.model)]
        for unit in self.units.values():
            parts.extend(unit.state_parts())
        return parts

    def get_state(self) -> dict[str, Any]:
        """Return every plant state (model and units) by name; model outputs must be synced first."""
        s: dict[str, Any] = self.model.get_state()
        s.update(gather({name: unit for name, unit in self.units.items()}))
        return s

    def set_state(self, values: Mapping[str, Any]) -> None:
        """Load named states; repack the solver vector with ``model.get_initial_values()``."""
        mine: dict[str, dict] = {name: {} for name in self.units}
        rest: dict[str, Any] = {}
        for key, value in values.items():
            owner = _owner(self.units, key)
            tail = key[len(owner) + 1:] if owner is not None else ""
            if owner is not None and tail in self.units[owner].get_state():
                mine[owner][tail] = value  # unit-owned state
            else:
                rest[key] = value
        for name, own in mine.items():
            if own:
                self.units[name].set_state(own)
        self.model.set_state(rest)

    def state_presets(self, t: float):
        """Return a resolver for the ``initial.states`` keywords at time ``t`` (s).

        ``rated``: the unit's rated dc voltage (V); ``source``: the emf at the state's bus (V).
        """
        for src in self.sources.values():
            src.emf.set_outputs(t)

        def preset(key: str, word: str):
            head = _owner(self.units, key, key.partition(".")[0])
            if word == "rated":
                if head not in self.units:
                    raise ValueError(f"{key}: 'rated' is a converter's rated dc voltage, and "
                                     f"{head!r} is not a unit ({sorted(self.units)})")
                cfg = self.units[head].cfg
                return cfg.dclink.vdc_ref
            if word == "source":
                internal = self.units[head].source_preset() if head in self.units else None
                if internal is not None:
                    return internal
                bus = self.units[head].cfg.bus if head in self.units else head
                at_bus = [s.emf.out.e_g for s in self.sources.values() if s.cfg.bus == bus]
                if at_bus:
                    return at_bus[0]
                if len(self.sources) == 1:
                    return next(iter(self.sources.values())).emf.out.e_g
                raise ValueError(f"{key}: 'source' needs a source at bus {bus!r}, or exactly one "
                                 f"source in the system (there are {len(self.sources)})")
            raise ValueError(f"{key}: unknown keyword {word!r} (keywords here: rated, source)")

        return preset

    # ---------------------------------------------------------------- events
    def switch(self, name: str, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect or disconnect a named unit, source, branch or registered element."""
        configured = _unit_at(self.p.units, name)
        if configured is not None:
            for leaf in _unit_leaves(self.p.units, name):
                self.units[leaf].connect(on, t, ramp)
            return
        if name in self.sources:
            breakers = [self.sources[name].branch]
        elif name in self.branches:
            breakers = [self.branches[name]]
        elif name in self.named_elements:
            element = self.named_elements[name]
            if hasattr(element, "connect"):
                element.connect(on, t, ramp)
                return
            breakers = element.breakers
        else:
            raise ValueError(f"cannot switch {name!r}")
        for breaker in breakers:
            (breaker.close_breaker if on else breaker.open_breaker)()

    def apply(self, change: Change) -> None:
        """Apply the checked parameter state after one ``set`` event to affected components."""
        p, done = change.params, set()
        for path in change.paths:
            section, _, remainder = path.partition(".")
            if section == "units":
                name = _owner(self.units, remainder, remainder.partition(".")[0])
            else:
                name = remainder.partition(".")[0]
            if (section, name) in done:
                continue
            done.add((section, name))
            if section == "buses":
                self.buses[name].retune(p.buses[name].c, p.buses[name].r_d)
            elif section == "branches":
                self.branches[name].retune(p.branches[name].l, p.branches[name].r)
            elif section == "sources":
                self.sources[name].retune(
                    p.sources[name], change.touches(f"sources.{name}."), change.t)
            elif section == "units":
                self.units[name].retune(
                    p.unit(name), change.touches(f"units.{name}."), change.t)
            elif section == "elements":
                self.named_elements[name].retune(p.elements[name])
        model = self.model
        model.energy_specs = {name: spec_of(subsystem)
                              for name, subsystem in zip(model.names, model.subsystems)}

    def check_events(self, p: Params) -> None:
        """Check that custom elements and replacement controllers support their requested events."""
        for event in p.events.values():
            if event.type in SWITCHING and event.target in self.named_elements:
                element = self.named_elements[event.target]
                if not (hasattr(element, "breakers") or hasattr(element, "connect")):
                    raise ConfigError(
                        f"elements.{event.target}: its type has no breakers and no connect(), so "
                        "connect and disconnect events cannot switch it")
        for change in p.changes:
            for path in change.paths:
                section, _, remainder = path.partition(".")
                if section == "units":
                    name = _owner(self.units, remainder, remainder.partition(".")[0])
                    rest = remainder[len(name) + 1:]
                else:
                    name, _, rest = remainder.partition(".")
                if section == "elements" and not hasattr(self.named_elements[name], "retune"):
                    raise ConfigError(
                        f"events.{change.event}: elements.{name} has no retune(parameters), so a "
                        "set event cannot change it")
                if (section == "units"
                        and not rest.startswith(("dclink.", "protection."))
                        and not hasattr(self.units[name].ctrl, "retune")):
                    raise ConfigError(
                        f"events.{change.event}: the controller of unit {name!r} has no "
                        f"retune(cfg, paths, t), so a set event cannot change {path}")

    # ---------------------------------------------------------------- the run
    def signals(self, t: float) -> dict[str, float | complex]:
        """Return the plant signals (SI) for the recorder; outputs must be synced to ``t``."""
        out: dict[str, float | complex] = {}
        for element in (*self.sources.values(), *self.units.values(),
                        *self.composite_signal_sources, *self.named_elements.values()):
            out.update(element.signals() if hasattr(element, "signals") else {})
        for name, branch in self.branches.items():
            out[f"{name}.i"] = branch.out.i
        for name, node in self.buses.items():
            out[f"{name}.u"] = node.out.u
        return out
