"""The system: named buses, branches, sources, units and circuit elements wired into one model.

It also exposes the component operations used by simulation-file events.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..components.network import ELEMENT_TYPES, RCNode, RLBranch, Terminal, ThreePhaseSource
from ..components.adc import _PortMeasurement
from ..components.wpt import make_wpt_link
from ..solver.energy import spec_of
from ..solver.model import ConfigError, Model, gather, scatter
from .events import SWITCHABLE, SourceScenario, connected_at, switching_schedule
from .params import Change, Params, _parameterized, _unit_leaves
from .types import unit_type
from .unit import DCConverter, Unit, VSC

__all__ = ["System"]


def _start_connections(system, p: Params, t: float) -> None:
    """Initialize each connection through its owner, once, before restoring control state."""
    schedule = switching_schedule(p.changes)
    for section in SWITCHABLE:
        owners = getattr(system, "named_elements" if section == "elements" else section, {})
        for name, owner in owners.items():
            start = getattr(owner, "start", None)
            if start is not None:
                start(t)
            else:
                target = f"{section}.{name}"
                if not connected_at(schedule.get(target, ()), t):
                    system.switch(target, False, t)


def _names(values) -> list[str]:
    """Return sortable labels for configuration keys which may contain runtime objects."""
    return sorted(repr(value) for value in values)


def _check_unique(target: Mapping, values: Mapping, what: str) -> None:
    """Reject keys already owned by an assembly mapping."""
    overlap = set(target) & set(values)
    if overlap:
        raise ConfigError(f"{what} collision: {_names(overlap)}")


def _merge_unique(target: dict, values: Mapping, what: str) -> None:
    """Merge a named assembly mapping after checking ownership."""
    _check_unique(target, values, what)
    target.update(values)


def _merge_matching(target: dict, values: Mapping, what: str) -> None:
    """Merge keys whose repeated values agree, rejecting only conflicting definitions."""
    conflicts = {key for key in set(target) & set(values) if target[key] != values[key]}
    if conflicts:
        raise ConfigError(f"{what} collision: {_names(conflicts)}")
    target.update(values)


@dataclass(frozen=True)
class _UnitPort:
    """One named Unit boundary, preserving its rating owner through composition."""

    bus: str
    owner: str
    domain: str
    phases: int = 1


@dataclass
class _UnitAssembly:
    """One configuration node expanded into runtime leaf units and shared subsystems."""

    units: dict[str, Unit] = field(default_factory=dict)
    subsystems: dict[str, Any] = field(default_factory=dict)
    signal_sources: tuple[Any, ...] = ()
    terminals: tuple[tuple[str, Terminal], ...] = ()
    ports: dict[str, _UnitPort] = field(default_factory=dict)
    default_port: Optional[str] = None
    port_bindings: dict[str, tuple[_UnitPort, ...]] = field(default_factory=dict)
    endpoints: dict[str, _UnitPort] = field(default_factory=dict)

    def port(self, name: Optional[str] = None) -> _UnitPort:
        """Return a named port, or this Unit's explicitly selected default port."""
        if name is None:
            if len(self.ports) == 1:
                return next(iter(self.ports.values()))
            name = self.default_port
        if name is None or name not in self.ports:
            raise ConfigError(
                f"unit port {name!r} is unavailable; known ports: {sorted(self.ports)}"
            )
        return self.ports[name]

    def attach(self, terminal: Terminal, where: str, port: Optional[str] = None) -> None:
        """Attach one parent terminal to a named port or the Unit's default port."""
        try:
            selected = self.port(port)
        except ConfigError as exc:
            raise ConfigError(f"{where}: {exc}") from None
        self.terminals += ((selected.bus, terminal),)

    def include(self, other: "_UnitAssembly") -> None:
        """Merge one child assembly while preserving its single public unit mapping."""
        _check_unique(self.units, other.units, "assembled unit name")
        _check_unique(self.subsystems, other.subsystems, "assembled subsystem name")
        self.units.update(other.units)
        self.subsystems.update(other.subsystems)
        self.signal_sources += other.signal_sources
        self.terminals += other.terminals
        _merge_unique(self.endpoints, other.endpoints, "unit endpoint")
        for target, ports in other.port_bindings.items():
            existing = self.port_bindings.setdefault(target, ())
            self.port_bindings[target] = existing + tuple(
                port for port in ports if port not in existing
            )


class _WPTSignals:
    """Expose shared link quantities without making the composite a scheduled unit."""

    def __init__(self, name: str, link: Any) -> None:
        self.name = name
        self.link = link
        self._outputs = tuple(type(link.out).__slots__)

    def signals(self) -> dict[str, float]:
        out = self.link.out
        return {f"{self.name}.link.{name}": getattr(out, name) for name in self._outputs}


def _source_scenario(p: Params, name: str) -> SourceScenario:
    """Construct a source's complete waveform timeline from the common parameter tree."""
    paths = {f"sources.{name}.{field}" for field in ("v", "f", "angle")}
    steps = [(change.t, change.params.sources[name]) for change in p.changes
             if paths.intersection(change.values)]
    return SourceScenario(p.sources[name], p.base.f0, steps)


class _Source:
    """Voltage source (emf) behind a series R-L branch."""

    def __init__(self, name: str, p: Params, t0: float) -> None:
        cfg, base = p.sources[name], p.base
        self.name, self.cfg = name, cfg
        self.scenario = _source_scenario(p, name)
        self.emf = _parameterized(
            lambda b: ThreePhaseSource(b.w0, *self.scenario.law(t0)), base)
        self.branch = _parameterized(lambda c: RLBranch(c.l, c.r), cfg)

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
        return {
            (self.branch, "u1"): (self.emf, "e_g"),
            (self.emf, "i"): [(self.branch, "i", 1.0)],
        }

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
    ports = {
        "dc": _UnitPort(unit.dc_bus, name, "dc"),
        "power": _UnitPort(cfg.bus, name, "ac", phases=3),
    }
    return _UnitAssembly(
        units={name: unit},
        ports=ports,
        default_port=unit_type(cfg.type).default_port,
        port_bindings={
            unit.dc_bus: (ports["dc"],),
            cfg.bus: (ports["power"],),
            f"{name}.dclink": (ports["dc"],),
            f"{name}.branch_f": (ports["power"],),
        },
    )


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
    ports = {
        "dc": _UnitPort(unit.dc_bus, name, "dc"),
        "power": _UnitPort(cfg.bus or unit.power_bus, name, "dc"),
    }
    bindings = {
        unit.dc_bus: (ports["dc"],),
        ports["power"].bus: (ports["power"],),
        f"{name}.dclink": (ports["dc"],),
    }
    if unit.branch_f is not None:
        bindings[f"{name}.branch_f"] = (ports["power"],)
    if unit.conduction is not None:
        bindings[f"{name}.conduction"] = (ports["power"],)
    return _UnitAssembly(
        units={name: unit},
        ports=ports,
        default_port=unit_type(cfg.type).default_port,
        port_bindings=bindings,
    )


def _assemble_wpt(name: str, cfg: Any, system: "System",
                  parts: Mapping[str, Any]) -> _UnitAssembly:
    if _part(parts, name) is not None:
        raise ConfigError(
            f"parts.{name}: a composite WPT unit is assembled from its primary and secondary parts"
        )
    link = _parameterized(make_wpt_link, cfg.link)
    assembly = _UnitAssembly(
        subsystems={f"{name}.link": link},
        signal_sources=(_WPTSignals(name, link),),
        default_port=unit_type(cfg.type).default_port,
    )
    for side in unit_type(cfg.type).children:
        child = _assemble_unit(f"{name}.{side}", getattr(cfg, side), system, parts)
        terminal = getattr(link, f"{side}_terminal")
        child.attach(
            terminal,
            f"units.{name}.{side}",
            "power" if "power" in child.ports else None,
        )
        for scc in link.switched:
            if scc.side == side:
                port = child.port("power" if "power" in child.ports else None)
                scc.bind_bridge(child.units[port.owner].bridge)
        assembly.ports[side] = child.port()
        assembly.include(child)
    default = assembly.port()
    assembly.port_bindings[f"{name}.link"] = (default,)
    return assembly


def _assemble_unit(name: str, cfg: Any, system: "System",
                   parts: Mapping[str, Any]) -> _UnitAssembly:
    assembler = unit_type(cfg.type, f"units.{name}.type").assembler
    if assembler is None:
        raise ConfigError(f"units.{name}.type: {cfg.type!r} has no runtime assembler")
    assembly = assembler(name, cfg, system, parts)
    try:
        assembly.port()
    except ConfigError as exc:
        raise ConfigError(f"units.{name}: {exc}") from None
    assembly.endpoints[name] = assembly.port()
    assembly.endpoints.update({f"{name}.{port}": value
                               for port, value in assembly.ports.items()})
    return assembly


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
        self.buses = {name: _parameterized(lambda values: RCNode(
                          values[0].c,
                          values[0].r_d,
                          u0=values[1].v_phase_peak if values[0].phases == 3 else 0.0,
                          phases=values[0].phases,
                      ), (cfg, base))
                      for name, cfg in p.buses.items()}
        self.sources = {name: _Source(name, p, p.simulation.initial.t) for name in p.sources}
        assembled = _UnitAssembly()
        for name, cfg in p.units.items():
            assembled.include(_assemble_unit(name, cfg, self, parts))
        self.units = assembled.units
        self.composite_signal_sources = assembled.signal_sources
        self._port_bindings = assembled.port_bindings

        # ---------------------------------------------------------- the wiring
        subsystems: dict[str, Any] = dict(self.buses)
        _merge_unique(subsystems, assembled.subsystems, "subsystem name")
        connections: dict = {}
        bus_terminals: dict[str, list[Terminal]] = {
            name: [node.terminal] for name, node in self.buses.items()
        }
        network_buses = {**assembled.endpoints, **self.buses}
        for name, cfg in p.branches.items():
            if network_buses[cfg.bus1].phases != network_buses[cfg.bus2].phases:
                raise ConfigError(f"branches.{name}: both buses must have the same phase count")
        self.branches = {name: _parameterized(lambda values: RLBranch(
                             values[0].l,
                             values[0].r,
                             phases=values[1],
                         ), (cfg, network_buses[cfg.bus1].phases))
                         for name, cfg in p.branches.items()}
        self.named_elements = {name: _parameterized(
                                  lambda values, name=name: ELEMENT_TYPES[values[0].type](
                                      name, values[0], network_buses, values[1]), (cfg, p))
                               for name, cfg in p.elements.items()}

        def resolve_bus(name: str) -> str:
            port = assembled.endpoints.get(name) if name not in self.buses else None
            return port.bus if port is not None else name
        branch_assemblies = tuple(_BranchAssembly(name, p.branches[name], branch)
                                  for name, branch in self.branches.items())
        elements = (*branch_assemblies, *self.sources.values(), *self.units.values(),
                    *self.named_elements.values())
        for element in elements:
            _merge_unique(subsystems, element.subsystems(), "subsystem name")
        for bus_name, terminal in assembled.terminals:
            bus_terminals.setdefault(bus_name, []).append(terminal)
        # Every electrical connection uses the same named-terminal contract. Assembly fans one
        # driving voltage out to the followers and sums their signed currents into the driver.
        for element in elements:
            for bus, inlet in getattr(element, "series_connections", lambda: {})().items():
                terminals = bus_terminals.setdefault(resolve_bus(bus), [])
                drivers = [terminal for terminal in terminals if terminal.drives_voltage]
                if len(drivers) != 1:
                    raise ConfigError(f"series inlet {bus!r} needs one existing voltage driver")
                terminals.remove(drivers[0])
                bus_terminals.setdefault(resolve_bus(inlet), []).append(drivers[0])
            for bus_name, terminal in element.terminals():
                if not isinstance(terminal, Terminal):
                    raise TypeError(
                        f"{type(element).__name__}.terminals() returned "
                        f"{type(terminal).__name__}, not Terminal"
                    )
                bus_terminals.setdefault(resolve_bus(bus_name), []).append(terminal)
        lowered_buses: dict[str, dict] = {}
        terminal_targets = set(connections)
        for bus_name, terminals in bus_terminals.items():
            drivers = [terminal for terminal in terminals if terminal.drives_voltage]
            if len(drivers) != 1:
                raise ConfigError(
                    f"bus {bus_name!r} needs exactly one voltage-driving terminal, got "
                    f"{len(drivers)}"
                )
            driver = drivers[0]
            followers = [terminal for terminal in terminals if not terminal.drives_voltage]
            if not followers:
                if bus_name in self.buses:
                    driver.bind_current_sources([])
                    continue
                raise ConfigError(f"internal bus {bus_name!r} has no attached terminal")
            voltage_connections = {}
            for terminal in followers:
                target = (terminal.subsystem, terminal.voltage)
                if target in terminal_targets:
                    raise ConfigError(
                        f"{type(terminal.subsystem).__name__}.{terminal.voltage} is connected both "
                        "as an electrical terminal and by connections()"
                    )
                voltage_connections[target] = (
                    driver.subsystem, driver.voltage
                )
                terminal.bind_voltage_source(voltage_connections[target])
                terminal_targets.add(target)
            current_target = (driver.subsystem, driver.current)
            if current_target in terminal_targets:
                raise ConfigError(
                    f"{type(driver.subsystem).__name__}.{driver.current} is connected both as "
                    "a bus driver and by connections()"
                )
            terminal_targets.add(current_target)
            current_sources = [
                (
                    terminal.subsystem,
                    terminal.current,
                    -terminal.direction / driver.direction,
                )
                for terminal in followers
            ]
            driver.bind_current_sources(current_sources)
            lowered = lowered_buses.setdefault(bus_name, {})
            if driver.hold_voltage_input is not None:
                if len(followers) != 1:
                    raise ConfigError(
                        f"bus {bus_name!r}: blocking needs one current-owning terminal "
                        "with a zero-current voltage law")
                lowered[(driver.subsystem, driver.hold_voltage_input)] = (
                    followers[0].hold_voltage_source(
                        getattr(driver.subsystem, "current_constraint", None)))
            if bus_name not in self.buses:
                lowered[current_target] = current_sources
                lowered.update(voltage_connections)
            else:
                lowered.update(voltage_connections)
                lowered[current_target] = current_sources
        # Keep the established evaluation order: internal power buses, controller and other
        # component wiring, then the configured external network buses. This only affects the
        # construction-time ordering of an otherwise identical direct connection graph.
        for bus_name, lowered in lowered_buses.items():
            if bus_name not in self.buses:
                connections.update(lowered)
        for element in elements:
            own_connections = element.connections()
            overlap = (set(connections) | terminal_targets) & set(own_connections)
            if overlap:
                raise ConfigError(
                    f"{type(element).__name__}.connections() duplicates an electrical-terminal "
                    f"target: {[repr(item) for item in overlap]}"
                )
            connections.update(own_connections)
        for bus_name, lowered in lowered_buses.items():
            if bus_name in self.buses:
                connections.update(lowered)
        zoh: dict = {}
        for unit in self.units.values():
            own_zoh = unit.zoh_connections()
            overlap = set(connections) & set(own_zoh)
            if overlap:
                raise ConfigError(
                    "connection target is driven both continuously and through a zero-order "
                    f"hold: {_names(overlap)}"
                )
            _merge_unique(zoh, own_zoh, "zero-order-hold connection target")
        # Resolve physical measurement outputs once, before Model schedules its dependencies.
        port_measurements = {}
        for unit in self.units.values():
            for channel, source in unit.cfg.meas.sources.items():
                owner, _, output = source.rpartition(".")
                subsystem = subsystems.get(owner)
                if subsystem is None and owner in network_buses and output in (
                        "voltage", "current", "power"):
                    bus = resolve_bus(owner)
                    if bus not in port_measurements:
                        terminal, = (item for item in bus_terminals[bus] if item.drives_voltage)
                        observer = _PortMeasurement(terminal)
                        voltage, current = terminal.signal_sources()
                        # Preserve the domain in record metadata before Model packs the graph.
                        observer.inp.voltage = getattr(terminal.subsystem.out, terminal.voltage)
                        observer.inp.current = getattr(terminal.subsystem.inp, terminal.current)
                        observer.out.voltage = observer.inp.voltage
                        observer.out.current = observer.inp.current
                        _merge_unique(subsystems, {f"measurements.{owner}": observer},
                                      "subsystem name")
                        connections[(observer, "voltage")] = voltage
                        connections[(observer, "current")] = current
                        port_measurements[bus] = observer
                    subsystem = port_measurements[bus]
                if subsystem is None or not hasattr(subsystem.out, output):
                    raise ConfigError(f"units.{unit.name}.meas.sources.{channel}: "
                                      f"unknown physical output {source!r}")
                if (isinstance(getattr(subsystem.out, output), complex)
                        and not isinstance(unit.ports.value(channel), complex)):
                    raise ConfigError(f"units.{unit.name}.meas.sources.{channel}: "
                                      "a complex output cannot drive a scalar channel")
                unit.ports.readers[channel] = (
                    lambda sub=subsystem, field=output: getattr(sub.out, field)
                )
                if unit.continuous and unit.ctrl is not None:
                    if not hasattr(unit.ctrl.inp, channel):
                        raise ConfigError(f"units.{unit.name}.meas.sources.{channel}: "
                                          "controller has no matching continuous input")
                    connections[(unit.ctrl, channel)] = (subsystem, output)
        self.model = Model(subsystems, connections, zoh)

        self.state_aliases: dict[str, str] = {}
        for unit in self.units.values():
            _merge_matching(self.state_aliases, unit.aliases(), "state alias")
        self._signal_names_checked = False

    # ---------------------------------------------------------------- states
    def state_parts(self) -> list[tuple[str, Any]]:
        """Return the power model and unit state owners in state-table order."""
        parts: list[tuple[str, Any]] = [("", self.model)]
        for unit in self.units.values():
            parts.extend(unit.state_parts())
        registered = {id(part) for _name, part in parts}
        parts.extend((name, sub) for name, sub in zip(self.model.names, self.model.subsystems)
                     if id(sub) not in registered and callable(getattr(sub, "get_state", None)))
        return parts

    def get_state(self) -> dict[str, Any]:
        """Return every plant state (model and units) by name; model outputs must be synced first."""
        return gather(dict(self.state_parts()))

    def set_state(self, values: Mapping[str, Any]) -> None:
        """Load named states; repack the solver vector with ``model.get_initial_values()``."""
        parts = dict(self.state_parts())
        unknown = set(values) - set(gather(parts))
        if unknown:
            raise KeyError(f"unknown system states: {sorted(unknown)}")
        scatter(parts, values)

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
        section, _, target = name.partition(".")
        if section in SWITCHABLE and section in self.p.sections_of(target):
            sections, name = (section,), target
        else:
            sections = tuple(section for section in self.p.sections_of(name) if section in SWITCHABLE)
        if len(sections) != 1:
            detail = "unknown" if not sections else f"ambiguous across {list(sections)}"
            raise ValueError(f"cannot switch {name!r}: {detail}")
        section = sections[0]
        if section == "units":
            for leaf in _unit_leaves(self.p.units, name):
                if on and self.units[leaf].tripped:
                    warnings.warn(f"t = {t:.6g} s: unit {leaf!r} has tripped, so its connect command "
                                  "leaves it disconnected", stacklevel=3)
                self.units[leaf].connect(on, t, ramp)
            return
        if section == "sources":
            breakers = [self.sources[name].branch]
        elif section == "branches":
            breakers = [self.branches[name]]
        else:
            element = self.named_elements[name]
            if hasattr(element, "connect"):
                element.connect(on, t, ramp)
                return
            breakers = element.breakers
        for breaker in breakers:
            (breaker.close_breaker if on else breaker.open_breaker)()

    def apply(self, change: Change) -> None:
        """Write one checked event's parameters and connection commands through existing owners."""
        p, done = change.params, set()
        for path in change.paths:
            if path in change.connections:
                self.switch(path.removesuffix(".connected"), change.values[path], change.t, change.ramp)
                continue
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
        if done:
            model = self.model
            model.energy_specs = {name: spec_of(subsystem)
                                  for name, subsystem in zip(model.names, model.subsystems)}

    def check_events(self, p: Params) -> None:
        """Check that custom elements and replacement controllers support their requested events."""
        for change in p.changes:
            for path in change.paths:
                if path in change.connections:
                    if path.startswith("elements."):
                        element = self.named_elements[path[len("elements."):-len(".connected")]]
                        if not (hasattr(element, "breakers") or hasattr(element, "connect")):
                            raise ConfigError(f"{path}: its type has no breakers and no connect()")
                    continue
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
        checked = self._signal_names_checked
        owners: dict[str, Any] = {}

        def add(owner: Any, values: Mapping[str, float | complex]) -> None:
            if checked:
                out.update(values)
            else:
                conflicts = {
                    name for name in set(out) & set(values)
                    if owners[name] is not owner
                }
                if conflicts:
                    raise ConfigError(f"signal name collision: {_names(conflicts)}")
                owners.update({name: owner for name in values})
                out.update(values)

        for element in (*self.sources.values(), *self.units.values(),
                        *self.composite_signal_sources, *self.named_elements.values()):
            add(element, element.signals() if hasattr(element, "signals") else {})
        for unit in self.units.values():
            if unit.adc is not None and (unit.adc.clocked or unit.adc.rms):
                add(unit, unit.measurement_signals())
        for name, branch in self.branches.items():
            add(branch, {f"{name}.i": branch.out.i})
        for name, node in self.buses.items():
            add(node, {f"{name}.u": node.out.u})
        self._signal_names_checked = True
        return out
