"""The system: named buses, branches, sources, units and circuit elements wired into one model.

It also exposes the component operations used by simulation-file events.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping, Optional

from ..components.network import ACVoltageSource, ELEMENT_TYPES, RCNode, RLBranch, Port
from ..components.adc import _PortMeasurement
from ..components.wpt import make_wpt_link
from ..solver.energy import spec_of
from ..solver.model import ConfigError, Model, gather, scatter
from .events import SWITCHABLE, Scenario, SourceScenario
from .params import Change, Params, _parameterized, _split_path, _unit_leaves, _unit_path
from .types import unit_type
from .unit import DCConverter, Unit, VSC

__all__ = ["System"]


def _start_connections(system, p: Params, t: float) -> None:
    """Initialize each connection through its owner, once, before restoring control state."""
    for section in SWITCHABLE:
        owners = getattr(system, "named_elements" if section == "elements" else section, {})
        for name, owner in owners.items():
            start = getattr(owner, "start", None)
            if start is not None:
                start(t)
            else:
                target = f"{section}.{name}"
                if not Scenario(p.changes, target).connected(t):
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
    phases: int = 1
    electrical_domain: Optional[str] = None

    def __post_init__(self) -> None:
        if self.electrical_domain not in (None, "ac", "dc"):
            raise ValueError(
                "unit-port electrical_domain must be 'ac', 'dc' or None, got "
                f"{self.electrical_domain!r}"
            )


@dataclass
class _UnitAssembly:
    """One configuration node expanded into runtime leaf units and shared subsystems."""

    units: dict[str, Unit] = field(default_factory=dict)
    subsystems: dict[str, Any] = field(default_factory=dict)
    signal_sources: tuple[Any, ...] = ()
    attached_ports: tuple[tuple[str, Port], ...] = ()
    ports: dict[str, _UnitPort] = field(default_factory=dict)
    default_port: Optional[str] = None
    unit_port_bindings: dict[str, tuple[_UnitPort, ...]] = field(default_factory=dict)
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

    def attach(self, physical_port: Port, where: str, port: Optional[str] = None) -> None:
        """Attach one physical port to a named Unit port or its default port."""
        try:
            selected = self.port(port)
        except ConfigError as exc:
            raise ConfigError(f"{where}: {exc}") from None
        self.attached_ports += ((selected.bus, physical_port),)

    def include(self, other: "_UnitAssembly") -> None:
        """Merge one child assembly while preserving its single public unit mapping."""
        _check_unique(self.units, other.units, "assembled unit name")
        _check_unique(self.subsystems, other.subsystems, "assembled subsystem name")
        self.units.update(other.units)
        self.subsystems.update(other.subsystems)
        self.signal_sources += other.signal_sources
        self.attached_ports += other.attached_ports
        _merge_unique(self.endpoints, other.endpoints, "unit endpoint")
        for target, ports in other.unit_port_bindings.items():
            existing = self.unit_port_bindings.setdefault(target, ())
            self.unit_port_bindings[target] = existing + tuple(
                port for port in ports if port not in existing
            )


def _validate_electrical_domains(
    endpoints: Mapping[str, _UnitPort],
    passive_groups: tuple[tuple[tuple[str, Port], ...], ...],
    physical_bindings: tuple[tuple[str, Port], ...],
    resolve_bus: Callable[[str], str],
) -> None:
    """Reject only explicit AC/DC constraints joined by domain-neutral circuitry."""
    adjacent: dict[str, set[str]] = {}

    def node(name: str) -> str:
        resolved = resolve_bus(name)
        adjacent.setdefault(resolved, set())
        return resolved

    for group in passive_groups:
        buses = tuple(dict.fromkeys(node(name) for name, _terminal in group))
        if buses:
            first = buses[0]
            for other in buses[1:]:
                adjacent[first].add(other)
                adjacent[other].add(first)

    constraints: list[tuple[str, str, str]] = []
    seen_ports: set[int] = set()
    for name, port in endpoints.items():
        if id(port) in seen_ports or port.electrical_domain is None:
            continue
        seen_ports.add(id(port))
        constraints.append((node(port.bus), port.electrical_domain, f"unit port {name}"))
    for bus, port in physical_bindings:
        if port.electrical_domain is not None:
            label = f"{type(port.component).__name__}.{port.name}"
            constraints.append((node(bus), port.electrical_domain, label))

    by_bus: dict[str, list[tuple[str, str]]] = {}
    for bus, domain, label in constraints:
        by_bus.setdefault(bus, []).append((domain, label))
    visited: set[str] = set()
    for start in adjacent:
        if start in visited:
            continue
        pending = [start]
        domains: dict[str, str] = {}
        while pending:
            bus = pending.pop()
            if bus in visited:
                continue
            visited.add(bus)
            pending.extend(adjacent[bus] - visited)
            for domain, label in by_bus.get(bus, ()):
                domains.setdefault(domain, label)
        if len(domains) > 1:
            raise ConfigError(
                "electrical-domain conflict: connected network contains "
                f"AC constraint {domains['ac']!r} and DC constraint {domains['dc']!r}"
            )


class _WPTSignals:
    """Expose shared link quantities without making the composite a scheduled unit."""

    def __init__(self, name: str, link: Any) -> None:
        self.name = name
        self.link = link
        self._outputs = tuple(type(link.output).__slots__)

    def signals(self) -> dict[str, float]:
        out = self.link.output
        return {f"{self.name}.link.{name}": getattr(out, name) for name in self._outputs}


def _source_scenario(p: Params, path: str, select: Callable[[Params], Any]) -> SourceScenario:
    """Construct a source timeline using one parameter path and selector."""
    paths = {f"{path}.{field}" for field in ("v", "f", "angle")}
    steps = [(change.t, select(change.params)) for change in p.changes
             if paths.intersection(change.values)]
    return SourceScenario(select(p), p.base.f0, steps)


class _Source:
    """Voltage source (emf) behind a series R-L branch."""

    def __init__(self, name: str, p: Params, t0: float, phases: int) -> None:
        cfg, base = p.sources[name], p.base
        self.name, self.cfg = name, cfg
        self.scenario = _source_scenario(p, f"sources.{name}", lambda q: q.sources[name])
        self.emf = _parameterized(
            lambda values: ACVoltageSource(
                values[0].w0, *self.scenario.law(t0), phases=values[1]
            ), (base, phases))
        self.branch = _parameterized(
            lambda values: RLBranch(values[0].l, values[0].r, phases=values[1]),
            (cfg, phases),
        )
        self.branch.port2.electrical_domain = "ac"
        self.port = self.branch.port2

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
        return {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        internal = f"{self.name}.source"
        return ((internal, self.emf.port),
                (internal, self.branch.port1),
                (self.cfg.bus, self.port))

    def signals(self) -> dict[str, float | complex]:
        return {f"{self.name}.i": self.branch.output.i, f"{self.name}.angle": self.emf.output.phi}


class _BranchAssembly:
    """Give a configured bare RLBranch the common network-component assembly interface."""

    domain_transparent = True

    def __init__(self, name: str, cfg, branch: RLBranch) -> None:
        self.name, self.cfg, self.branch = name, cfg, branch

    def subsystems(self) -> dict[str, RLBranch]:
        return {self.name: self.branch}

    @staticmethod
    def connections() -> dict:
        return {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return ((self.cfg.bus1, self.branch.port1),
                (self.cfg.bus2, self.branch.port2))


def _part(parts: Mapping[str, Any], name: str, suffix: str = "") -> Any:
    key = f"{name}.{suffix}" if suffix else name
    return parts.get(key)


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
        "dc": _UnitPort(unit.dc_bus, name, electrical_domain="dc"),
        "power": _UnitPort(cfg.bus, name, phases=3, electrical_domain="ac"),
    }
    return _UnitAssembly(
        units={name: unit},
        ports=ports,
        default_port=unit_type(cfg.type).default_port,
        unit_port_bindings={
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
        "dc": _UnitPort(unit.dc_bus, name, electrical_domain="dc"),
        "power": _UnitPort(cfg.bus or unit.power_bus, name),
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
        unit_port_bindings=bindings,
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
        physical_port = getattr(link, f"{side}_port")
        child.attach(physical_port, f"units.{name}.{side}", "power")
        for scc in link.switched:
            if scc.side == side:
                scc.bind_bridge(child.units[child.port("power").owner].bridge)
        assembly.ports[side] = child.port()
        assembly.include(child)
    default = assembly.port()
    assembly.unit_port_bindings[f"{name}.link"] = (default,)
    return assembly


def _assemble_unit(name: str, cfg: Any, system: "System",
                   parts: Mapping[str, Any]) -> _UnitAssembly:
    assembly = unit_type(cfg.type, f"units.{name}.type").assembler(name, cfg, system, parts)
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
        assembled = _UnitAssembly()
        for name, cfg in p.units.items():
            assembled.include(_assemble_unit(name, cfg, self, parts))
        self.units = assembled.units
        self.composite_signal_sources = assembled.signal_sources
        self._unit_port_bindings = assembled.unit_port_bindings

        # ---------------------------------------------------------- the wiring
        subsystems: dict[str, Any] = dict(self.buses)
        _merge_unique(subsystems, assembled.subsystems, "subsystem name")
        connections: dict = {}
        configured_connections: list[tuple[tuple[Any, str], str, str]] = []
        connection_targets: set[tuple[Any, str]] = set()

        def claim_connections(targets, where: str) -> None:
            """Assign each model input to one connection mechanism during assembly."""
            targets = set(targets)
            unknown = [
                (type(subsystem).__name__, field)
                for subsystem, field in targets
                if not hasattr(getattr(subsystem, "input", None), field)
            ]
            if unknown:
                raise ConfigError(f"{where}: unknown input target: {_names(unknown)}")
            overlap = connection_targets & targets
            if overlap:
                raise ConfigError(f"{where}: input target is already driven: {_names(overlap)}")
            connection_targets.update(targets)

        bus_ports: dict[str, list[Port]] = {
            name: [node.port] for name, node in self.buses.items()
        }
        network_buses = {**assembled.endpoints, **self.buses}

        def resolve_bus(name: str) -> str:
            port = assembled.endpoints.get(name) if name not in self.buses else None
            return port.bus if port is not None else name

        self.sources = {
            name: _Source(
                name, p, p.simulation.initial.t, network_buses[cfg.bus].phases
            )
            for name, cfg in p.sources.items()
        }
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

        branch_assemblies = tuple(_BranchAssembly(name, p.branches[name], branch)
                                  for name, branch in self.branches.items())
        elements = (*branch_assemblies, *self.sources.values(), *self.units.values(),
                    *self.named_elements.values())
        element_ports = tuple((element, tuple(element.port_bindings())) for element in elements)
        for element in elements:
            _merge_unique(subsystems, element.subsystems(), "subsystem name")
        physical_bindings = assembled.attached_ports + tuple(
            binding for _element, bindings in element_ports for binding in bindings
        )
        passive_groups = tuple(
            bindings for element, bindings in element_ports
            if getattr(element, "domain_transparent", False)
        )
        _validate_electrical_domains(
            assembled.endpoints, passive_groups, physical_bindings, resolve_bus
        )
        for bus_name, port in assembled.attached_ports:
            bus_ports.setdefault(bus_name, []).append(port)
        for element, element_bindings in element_ports:
            for bus, inlet in getattr(element, "series_connections", lambda: {})().items():
                attached = bus_ports.setdefault(resolve_bus(bus), [])
                if len(attached) != 2:
                    raise ConfigError(
                        f"series inlet {bus!r} needs a node joining exactly two ports, "
                        f"got {len(attached)}"
                    )
                bus_ports.setdefault(resolve_bus(inlet), []).append(attached.pop())
            for bus_name, port in element_bindings:
                if not isinstance(port, Port):
                    raise TypeError(
                        f"{type(element).__name__}.port_bindings() returned "
                        f"{type(port).__name__}, not Port"
                    )
                bus_ports.setdefault(resolve_bus(bus_name), []).append(port)
        circuit_model = p.simulation.solver.circuit_model
        for element in elements:
            own_connections = element.connections()
            claim_connections(own_connections, f"{type(element).__name__}.connections()")
            for target, source in own_connections.items():
                if isinstance(source, str):
                    configured_connections.append(
                        (target, source, f"{type(element).__name__}.connections()")
                    )
                else:
                    connections[target] = source
        zoh: dict = {}
        for unit in self.units.values():
            own_zoh = unit.zoh_connections()
            claim_connections(own_zoh, f"unit {unit.name!r} zero-order hold")
            _merge_unique(zoh, own_zoh, "zero-order-hold connection target")
        # Resolve configured physical outputs once, before Model schedules its dependencies.
        port_measurements = {}

        def physical_output(source: str, where: str):
            owner, separator, output = source.rpartition(".")
            subsystem = subsystems.get(owner) if separator else None
            if subsystem is None and owner in network_buses and output in (
                    "voltage", "current", "power"):
                bus = resolve_bus(owner)
                if bus not in port_measurements:
                    port = (self.buses[bus].port if bus in self.buses
                            else bus_ports[bus][0])
                    observer = _PortMeasurement(port)
                    observer.input.voltage = port.voltage
                    observer.input.current = port.current
                    observer.output.voltage = observer.input.voltage
                    observer.output.current = observer.input.current
                    _merge_unique(subsystems, {f"measurements.{owner}": observer},
                                  "subsystem name")
                    port_measurements[bus] = observer
                subsystem = port_measurements[bus]
            if subsystem is None or not hasattr(subsystem.output, output):
                raise ConfigError(f"{where}: unknown physical output {source!r}")
            return subsystem, output

        def connect_source(target: tuple[Any, str], source: str, where: str) -> None:
            subsystem, output = physical_output(source, where)
            target_subsystem, target_input = target
            if not hasattr(target_subsystem.input, target_input):
                raise ConfigError(f"{where}: unknown input target {target_input!r}")
            if (isinstance(getattr(subsystem.output, output), complex)
                    and not isinstance(getattr(target_subsystem.input, target_input), complex)):
                raise ConfigError(f"{where}: a complex output cannot drive a scalar input")
            connections[target] = (subsystem, output)

        for target, source, where in configured_connections:
            connect_source(target, source, where)

        for unit in self.units.values():
            for channel, source in unit.cfg.meas.sources.items():
                where = f"units.{unit.name}.meas.sources.{channel}"
                subsystem, output = physical_output(source, where)
                if (isinstance(getattr(subsystem.output, output), complex)
                        and not isinstance(unit.ports.value(channel), complex)):
                    raise ConfigError(f"{where}: a complex output cannot drive a scalar channel")
                unit.ports.readers[channel] = (
                    lambda sub=subsystem, field=output: getattr(sub.output, field)
                )
                if unit.continuous and unit.ctrl is not None:
                    if not hasattr(unit.ctrl.input, channel):
                        raise ConfigError(f"units.{unit.name}.meas.sources.{channel}: "
                                          "controller has no matching continuous input")
                    target = (unit.ctrl, channel)
                    claim_connections((target,), where)
                    connections[target] = (subsystem, output)
        from .circuit import (circuit_explicit_stable, circuit_linear_blocks, circuit_mode_key,
                              circuit_structure, compile_network)
        declarations = {id(subsystem): compile_network(subsystem)
                        for subsystem in subsystems.values()}
        if circuit_model == "device":
            from .circuit import lower_devices

            subsystems, connections, zoh, self.circuits = lower_devices(
                subsystems, connections, zoh, bus_ports, declarations
            )
        else:
            from .circuit import lower

            subsystems, connections, zoh, circuit = lower(
                circuit_model, subsystems, connections, zoh, bus_ports, declarations
            )
            self.circuits = () if circuit is None else (circuit,)
        self.model = Model(
            subsystems, connections, zoh,
            linear_structure=partial(circuit_structure, circuits=self.circuits),
            linear_mode_key=partial(circuit_mode_key, circuits=self.circuits),
            linear_blocks=partial(circuit_linear_blocks, circuits=self.circuits),
            explicit_stability=partial(circuit_explicit_stable, circuits=self.circuits),
            explicit_stability_dynamic=any(
                transformer.constant is None
                for circuit in self.circuits
                for transformer in circuit.netlist._transformers
            ),
        )
        self.event_subsystems = tuple(
            (name, subsystem)
            for name, subsystem in zip(self.model.names, self.model.subsystems)
            if getattr(subsystem, "clock", None) is not None
        )
        event_methods = ("begin", "next_time", "sense", "actuate")
        for name, subsystem in self.event_subsystems:
            missing = [method for method in event_methods
                       if not callable(getattr(subsystem, method, None))]
            if missing:
                raise TypeError(f"{name}: clocked subsystem event interface is missing {missing}")

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
        if not self.circuits:
            subsystems = iter(self.model.subsystems)
        else:
            from .circuit import NetworkView

            subsystems = (
                subsystem.component if isinstance(subsystem, NetworkView) else subsystem
                for subsystem in self.model.subsystems
            )
        parts.extend((name, sub) for name, sub in zip(self.model.names, subsystems)
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
            head = _unit_path(self.p.units, key) or key.partition(".")[0]
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
                at_bus = [s.emf.output.e_g for s in self.sources.values() if s.cfg.bus == bus]
                if at_bus:
                    return at_bus[0]
                if len(self.sources) == 1:
                    return next(iter(self.sources.values())).emf.output.e_g
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
        p, done, connections_changed = change.params, set(), False
        for path in change.paths:
            if path in change.connections:
                self.switch(path.removesuffix(".connected"), change.values[path], change.t, change.ramp)
                connections_changed = True
                continue
            section, name, _ = _split_path(self.p, path)
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
            for circuit in self.circuits:
                circuit.build()
            model = self.model
            model.energy_specs = {name: spec_of(subsystem)
                                  for name, subsystem in zip(model.names, model.subsystems)}
        if done or connections_changed:
            self.reconcile_circuits(change.t)

    def reconcile_circuits(self, t: float) -> None:
        """Commit all electrical command changes at one physical event boundary."""
        for circuit in self.circuits:
            circuit.reconcile(t)
        self.refresh_circuit_plan()

    def refresh_circuit_plan(self) -> None:
        """Refresh model scheduling after circuit equations changed at a boundary."""
        changed = [circuit for circuit in self.circuits if circuit.plan_changed]
        if not changed:
            return
        self.model.rebuild_evaluation_plan()
        for circuit in changed:
            circuit.plan_changed = False

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
                section, name, rest = _split_path(p, path)
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
            add(branch, {f"{name}.i": branch.output.i})
        for name, node in self.buses.items():
            add(node, {f"{name}.u": node.output.u})
        self._signal_names_checked = True
        return out
