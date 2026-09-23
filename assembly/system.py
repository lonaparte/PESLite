"""Assembly of named buses, R-L branches, sources and converter units into one Model.

Element names are the namespaces of their subsystems and states.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..phs.model import Model
from ..phs.states import gather, scatter
from ..params import Params
from ..power import RCNode, RLBranch, ThreePhaseSource
from .events import SourceScenario
from .unit import Unit

__all__ = ["System"]


class _Source:
    """Voltage source (emf) behind a series R-L branch."""

    def __init__(self, name: str, cfg, base, bus) -> None:
        self.name, self.cfg, self.bus = name, cfg, bus
        self.scenario = sc = SourceScenario(cfg)
        self.emf = ThreePhaseSource(base.w0, cfg.v,
                                    phi=sc.angle if sc.has_angle else None,
                                    magnitude=sc.magnitude if sc.has_voltage_step else None)
        self.branch = RLBranch(cfg.l, cfg.r)

    def subsystems(self) -> dict[str, Any]:
        return {f"{self.name}.emf": self.emf, f"{self.name}.branch": self.branch}

    def connections(self) -> dict:
        return {(self.branch, "u_from"): (self.emf, "e_g"),
                (self.branch, "u_to"): (self.bus, "u"),
                (self.emf, "i"): (self.branch, "i")}  # energy accounting only

    @property
    def bus_name(self) -> str:
        return self.cfg.bus

    @property
    def injection(self) -> tuple:
        return (self.branch, "i")

    def signals(self) -> dict[str, float | complex]:
        return {f"{self.name}.i": self.branch.out.i, f"{self.name}.angle": self.emf.out.phi}


class System:
    """Buses, branches, sources and units wired into one :class:`~peslite.phs.model.Model`."""

    def __init__(self, p: Params, parts: Optional[Mapping[str, Any]] = None,
                 elements: Sequence[Any] = ()) -> None:
        """Build the elements described by ``p`` plus any extra ``elements``.

        parts: replacement units or unit parts keyed ``"<unit>"``, ``"<unit>.ctrl"``, ``"<unit>.modulator"``,
        ``"<unit>.delay"``.
        elements: callables ``(buses, params) -> element``; an element provides ``subsystems()``,
        ``connections()``, ``bus_name``, ``injection`` (signed current into the bus, or ``None``)
        and optionally ``signals()``.
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
        self.sources = {name: _Source(name, cfg, base, self.buses[cfg.bus])
                        for name, cfg in p.sources.items()}
        self.units: dict[str, Unit] = {}
        for name, cfg in p.units.items():
            self.units[name] = parts.get(name) or Unit(
                name, cfg, p.simulation, self.buses[cfg.bus],
                ctrl=parts.get(f"{name}.ctrl"), modulator=parts.get(f"{name}.modulator"),
                delay=parts.get(f"{name}.delay"))

        # ---------------------------------------------------------- the wiring
        subsystems: dict[str, Any] = dict(self.buses)
        subsystems.update(self.branches)
        connections: dict = {}
        into: dict[str, list] = {name: [] for name in self.buses}
        self.elements = [make(self.buses, p) for make in elements]
        for element in (*self.sources.values(), *self.units.values(), *self.elements):
            subsystems.update(element.subsystems())
            connections.update(element.connections())
        # bus injections: units, sources, extra elements, then branches
        for element in (*self.units.values(), *self.sources.values(), *self.elements):
            if element.injection is not None:
                into[element.bus_name].append(element.injection)
        for name, cfg in p.branches.items():
            branch = self.branches[name]
            connections[(branch, "u_from")] = (self.buses[cfg.from_bus], "u")
            connections[(branch, "u_to")] = (self.buses[cfg.to_bus], "u")
            into[cfg.from_bus].append((branch, "i", -1.0))  # leaves the sending end
            into[cfg.to_bus].append((branch, "i", 1.0))     # arrives at the receiving end
        for name, node in self.buses.items():
            if not into[name]:
                raise ValueError(f"bus {name!r} has nothing attached to it")
            connections[(node, "i_in")] = into[name]
        zoh: dict = {}
        for unit in self.units.values():
            zoh.update(unit.zoh_connections())
        self.model = Model(subsystems, connections, zoh)

        self.state_aliases: dict[str, str] = {}
        for unit in self.units.values():
            self.state_aliases.update(unit.aliases())
        self.breakers = [b for unit in self.units.values() for b in unit.breakers]

    # ---------------------------------------------------------------- states
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
                return self.units[head].cfg.dclink.vdc_ref
            if word == "source":
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

    # ---------------------------------------------------------------- the loop
    def trip(self) -> None:
        """Trip every unit (open all breakers)."""
        for unit in self.units.values():
            unit.trip()

    @property
    def breaker_open(self) -> bool:
        return any(unit.breaker_open for unit in self.units.values())

    def signals(self, t: float) -> dict[str, float | complex]:
        """Return the plant signals (SI) for the recorder; outputs must be synced to ``t``."""
        out: dict[str, float | complex] = {}
        for element in (*self.sources.values(), *self.units.values(), *self.elements):
            out.update(element.signals() if hasattr(element, "signals") else {})
        for name, branch in self.branches.items():
            out[f"{name}.i"] = branch.out.i
        for name, node in self.buses.items():
            out[f"{name}.u"] = node.out.u
        return out
