"""The system: named buses, branches, sources, units and circuit elements wired into one model.

It also exposes the component operations used by simulation-file events.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..components.network import ELEMENT_TYPES, RCNode, RLBranch, Terminal, ThreePhaseSource
from ..solver.energy import spec_of
from ..solver.model import ConfigError, Model, gather
from .events import SWITCHING, SourceScenario
from .params import Change, Params
from .unit import DCUnit, Unit

__all__ = ["System"]


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
        # buses start at the AC base voltage, angle zero, unless initial states override it
        self.buses = {name: RCNode(cfg.c, cfg.r_d, u0=base.v_phase_peak)
                      for name, cfg in p.buses.items()}
        self.branches = {name: RLBranch(cfg.l, cfg.r)
                         for name, cfg in p.branches.items()}
        self.sources = {name: _Source(name, cfg, base, [
                            (change.t, change.params.sources[name]) for change in p.changes
                            if set(change.touches(f"sources.{name}.")) & {"v", "f", "angle"}],
                            p.simulation.initial.t)
                        for name, cfg in p.sources.items()}
        self.units: dict[str, Any] = {}
        for name, cfg in p.units.items():
            if name in parts:
                built = parts[name]
            elif hasattr(cfg, "topology"):
                built = DCUnit(
                    name,
                    cfg,
                    p.simulation,
                    ctrl=parts.get(f"{name}.ctrl"),
                    modulator=parts.get(f"{name}.modulator"),
                )
            else:
                built = Unit(
                    name,
                    cfg,
                    p.simulation,
                    self.buses[cfg.bus],
                    ctrl=parts.get(f"{name}.ctrl"),
                    modulator=parts.get(f"{name}.modulator"),
                )
            self.units[name] = built

        # ---------------------------------------------------------- the wiring
        subsystems: dict[str, Any] = dict(self.buses)
        connections: dict = {}
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
            head, _, tail = key.partition(".")
            if head in self.units and tail in self.units[head].get_state():
                mine[head][tail] = value  # unit-owned state
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
            head = key.partition(".")[0]
            if word == "rated":
                if head not in self.units:
                    raise ValueError(f"{key}: 'rated' is a converter's rated dc voltage, and "
                                     f"{head!r} is not a unit ({sorted(self.units)})")
                cfg = self.units[head].cfg
                return (cfg.ctrl.references.voltage if hasattr(cfg, "topology")
                        else cfg.dclink.vdc_ref)
            if word == "source":
                if head in self.units and hasattr(self.units[head].cfg, "topology"):
                    return self.units[head].cfg.source_voltage
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
        if name in self.units:
            self.units[name].connect(on, t, ramp)
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
            section, name, _rest = path.split(".", 2)
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
                    p.units[name], change.touches(f"units.{name}."), change.t)
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
                section, name, rest = path.split(".", 2)
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
        for element in (*self.sources.values(), *self.units.values(), *self.named_elements.values()):
            out.update(element.signals() if hasattr(element, "signals") else {})
        for name, branch in self.branches.items():
            out[f"{name}.i"] = branch.out.i
        for name, node in self.buses.items():
            out[f"{name}.u"] = node.out.u
        return out
