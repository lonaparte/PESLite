"""Passive electrical branches, buses and registered loads."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from ..solver.energy import Port, StoragePort
from ..solver.model import Bag, ConfigError, Empty, OutputMethod
from .network import Element, NetworkComponent, _phase_convention, register_component_type

__all__ = ["RLBranch", "Bus", "RCNode", "Capacitor", "Load", "Resistor"]


def _node_kind(phases: int) -> tuple[int, float]:
    """Return the netlist node dimension and power scale for a phase convention."""
    return (2, 1.5) if phases == 3 else (1, 1.0)


class _BranchState(Bag):
    __slots__ = ("i",)

class _BranchInput(Bag):
    __slots__ = ("di_dt", "loss_factor")

class _BranchOutput(Bag):
    __slots__ = ("i",)

class RLBranch(NetworkComponent):
    """Series R-L branch with positive current from port 1 to port 2.

    ``phases=3`` uses the peak-scaled complex space-vector convention. ``phases=1`` uses a real
    scalar and the same port contract, so converter units do not need a second network API.
    """

    state_names: ClassVar[tuple[str, ...]] = ("i",)
    outputs_need_inputs: ClassVar[bool] = False
    netlist_jumps: bool = False
    def __init__(self, L: float, R: float, i0: complex | float = 0j, *, phases: int = 3,
                 unilateral: bool = False) -> None:
        self.power_scale, self.zero, initial = _phase_convention(phases, i0)
        if unilateral and phases != 1:
            raise ValueError("unilateral current is only available for scalar branches")
        self.L, self.R = L, R
        self.phases = phases
        self.unilateral = bool(unilateral)
        self.state = _BranchState(i=initial)
        self.input = _BranchInput(di_dt=self.zero, loss_factor=1.0)
        self.output = _BranchOutput(i=initial)
        self.port1 = Port(
            self, "1", voltage=self.zero, current=initial,
            scale=self.power_scale, standalone="voltage"
        )
        self.port2 = Port(
            self, "2", voltage=self.zero, current=-initial,
            scale=self.power_scale, standalone="voltage"
        )
        self.ports = (self.port1, self.port2)
        self.storage = (StoragePort(
            "i", L, self.power_scale,
            ((self.port1, "voltage", 1.0), (self.port2, "voltage", -1.0)),
            "inductor",
        ),)
        self.breaker_open = False
        self.external_conduction = False
        if self.unilateral:
            self.state_bounds = {"i": (0.0, None)}
            self.commutation_margin = self._device_commutation_margin
            self.commutate = self._device_commutate

    def open_breaker(self) -> None:
        self.breaker_open = True
        if not self.netlist_jumps:
            self.state.i = self.zero

    def close_breaker(self) -> None:
        """Close the breaker; current resumes from zero after an opening."""
        self.breaker_open = False

    def retune(self, L: float, R: float) -> bool:
        """Change impedance without changing current; report a network change."""
        changed = self.L != L or self.R != R
        self.L, self.R = L, R
        self.storage = (replace(self.storage[0], value=L),)
        return changed

    def use_external_conduction(self) -> None:
        """Take derivative and copper-loss factor from a connected constitutive subsystem."""
        self.external_conduction = True
        self.commutation_margin = None
        self.commutate = None
        self.energy_audit_note = (
            "cycle-averaged DCM current does not have the branch's linear instantaneous "
            "storage function"
        )

    def diode_blocks_reverse_current(self) -> bool:
        """Return whether this branch represents a diode-bounded current path."""
        return self.unilateral

    def equation_state_bindings(self) -> dict[str, tuple[Any, str]]:
        """Return discrete physical state that generated backends must preserve."""
        return {"breaker_open": (self, "breaker_open")}

    def _device_commutation_margin(self) -> float:
        from ..solver.netlist import device_commutation_margin
        return device_commutation_margin(self)

    def _device_commutate(self, t: float) -> None:
        from ..solver.netlist import device_commutate
        device_commutate(self, t)

    def network(self, nb):
        """Declare the series breaker, resistance, inductance and optional diode."""
        from ..solver import netlist as nl

        if self.external_conduction:
            branch = nb.add(nl.CurrentState(
                nb.node(self.port1), nb.node(self.port2), nb.state(self, "i"),
                nb.function(
                    lambda _t: self.zero if self.breaker_open else self.input.di_dt,
                    ("di_dt",), states=("i",),
                ),
            ))
            nb.output(self, "i", ("branch_current", branch))
            if self.R:
                nb.add(nl.Dissipation(nb.function(
                    lambda _t: self.power_scale * self.R
                    * max(0.0, self.input.loss_factor) * abs(self.state.i) ** 2,
                    ("loss_factor",),
                )))
            nb.losses(self)
            return None

        kind = _node_kind(self.phases)
        inductor = nb.internal(*kind)
        resistor = nb.internal(*kind) if self.R else inductor
        start = nb.node(self.port1)
        if self.unilateral:
            start = nb.add(nl.Diode(start, nb.internal(*kind))).ports[0][1]
        nb.add(nl.Switch(start, resistor, nb.value(self, "breaker_open", invert=True)))
        if self.R:
            nb.add(nl.Resistor(resistor, inductor, self.R))
        nb.add(nl.Inductor(
            inductor, nb.node(self.port2), self.L, nb.state(self, "i")
        ))
        nb.port_output(self, "i", self.port1, "current")
        nb.losses(self)

class _NodeState(Bag):
    __slots__ = ("u_C",)

class _NodeOutput(Bag):
    __slots__ = ("u",)


class Bus(NetworkComponent):
    """A non-loading electrical connection point with a published voltage."""

    state_names: ClassVar[tuple[str, ...]] = ()
    voltage_field: ClassVar[str] = "u"
    outputs_need_inputs: ClassVar[bool] = False
    observer: ClassVar[bool] = True

    def __init__(self, *, phases: int = 3) -> None:
        self.power_scale, zero, _initial = _phase_convention(phases)
        self.phases = phases
        self.state = Empty()
        self.input = Empty()
        self.output = _NodeOutput(u=zero)
        self.port: Port | None = Port(
            self, "power", voltage=zero, current=zero,
            scale=self.power_scale,
            electrical_domain="ac" if phases == 3 else None,
        )
        self.ports = (self.port,)

    def network(self, nb):
        """Publish the voltage of the node joined to this bus."""
        if self.port is None:
            return NotImplemented
        nb.node(self.port)
        nb.port_output(self, "u", self.port, "voltage")


class RCNode(NetworkComponent):
    """Node with a shunt capacitor ``C`` (F) in series with ``R_d`` (ohm) to ground.

    Port current is positive into the node. Output ``u = u_C + R_d i``.
    """

    state_names: ClassVar[tuple[str, ...]] = ("u_C",)
    voltage_field: ClassVar[str] = "u"
    outputs_need_inputs: ClassVar[bool] = False
    def __init__(self, C: float, R_d: float, u0: complex | float = 0j, *, phases: int = 3) -> None:
        self.power_scale, zero, initial = _phase_convention(phases, u0)
        self.C, self.R_d = C, R_d
        self.phases = phases
        self.state = _NodeState(u_C=initial)
        self.input = Empty()
        self.output = _NodeOutput(u=initial)
        self.port = Port(
            self, "power", voltage=initial, current=zero,
            scale=self.power_scale, standalone="current"
        )
        self.ports = (self.port,)
        self.storage = (StoragePort(
            "u_C", C, self.power_scale, ((self.port, "current", 1.0),)
        ),)

    def retune(self, C: float, R_d: float) -> bool:
        """Change impedance without changing voltage; report a network change."""
        changed = self.C != C or self.R_d != R_d
        self.C, self.R_d = C, R_d
        self.storage = (replace(self.storage[0], value=C),)
        return changed

    def network(self, nb):
        """Declare the damping resistance and shunt capacitor."""
        from ..solver import netlist as nl

        node = nb.node(self.port)
        capacitor = nb.internal(*_node_kind(self.phases)) if self.R_d else node
        if self.R_d:
            nb.add(nl.Resistor(node, capacitor, self.R_d))
        nb.add(nl.Capacitor(
            capacitor, None, self.C, nb.state(self, "u_C")
        ))
        nb.port_output(self, "u", self.port, "voltage")
        nb.losses(self)


@register_component_type
@dataclass(frozen=True, kw_only=True)
class Capacitor:
    """Two-terminal capacitor with optional series resistance."""

    bus1: str
    bus2: str
    c: float
    r: float = 0.0
    type: str = "capacitor"
    domain_transparent: ClassVar[bool] = True

    _quantities = {"c": "capacitance", "r": "resistance"}

    def __post_init__(self) -> None:
        if self.bus1 == self.bus2:
            raise ConfigError("bus1 and bus2 must be distinct")
        if not math.isfinite(self.c) or self.c <= 0.0:
            raise ConfigError(f"c must be finite and > 0, got {self.c}")
        if not math.isfinite(self.r) or self.r < 0.0:
            raise ConfigError(f"r must be finite and >= 0, got {self.r}")

    def network(self, nb) -> None:
        from ..solver import netlist as nl

        positive = nb.bus1
        if self.r:
            middle = nb.internal()
            nb.add(nl.Resistor(positive, middle, self.r))
            positive = middle
        capacitor = nb.add(nl.Capacitor(positive, nb.bus2, self.c, nb.state("u")))
        nb.output("u", ("branch_voltage", capacitor))
        nb.port_output("i", "bus1", "current")
        nb.losses()



@register_component_type
class Load(Element):
    """Built-in series R-L load from ``bus`` to ground; a small impedance models a fault."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "load"
        bus: str
        l: float
        r: float = 0.0

        _quantities = {"l": "inductance", "r": "resistance"}
        _input_aliases = {"x_pu": "l_pu"}

        def __post_init__(self) -> None:
            if not self.l > 0.0:
                raise ConfigError(f"l must be > 0 (x_pu > 0), got {self.l}")
            if not self.r >= 0.0:
                raise ConfigError(f"r must be >= 0, got {self.r}")

    type = "load"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        self.branch = RLBranch(cfg.l, cfg.r, phases=buses[cfg.bus].phases)
        self.branch.port2.grounded = True
        self.breakers = [self.branch]

    def subsystems(self) -> dict[str, Any]:
        return {f"{self.name}.branch": self.branch}

    def connections(self) -> dict:
        return {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return ((self.cfg.bus, self.branch.port1),)

    def signals(self) -> dict[str, complex]:
        return {f"{self.name}.i": self.branch.output.i}

    def retune(self, cfg: Any) -> bool:
        """Take new load parameters while preserving branch current."""
        self.cfg = cfg
        return self.branch.retune(cfg.l, cfg.r)


class _ResistorOutput(Bag):
    __slots__ = ("i", "u", "power")


class _ResistorModel(NetworkComponent):
    """One resistance, in shunt admittance or series voltage-drop form."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    storage: ClassVar[tuple[StoragePort, ...]] = ()

    def __init__(self, resistance: float, phases: int, series: bool = False) -> None:
        self.resistance = resistance
        self.series = series
        self.power_scale, zero, _ = _phase_convention(phases)
        self.state = Empty()
        self.input = Empty()
        self.output = _ResistorOutput(i=zero, u=zero, power=0.0)
        self.port1 = Port(
            self, "1", voltage=zero, current=zero,
            scale=self.power_scale, standalone="voltage"
        )
        self.ports = (self.port1,)
        if series:
            self.port2 = Port(
                self, "2", voltage=zero, current=zero,
                scale=self.power_scale, standalone="voltage"
            )
            self.ports += (self.port2,)
        self.output_stages = (OutputMethod(
            "set_power", (), ("power",), rhs=False
        ),)
    def set_power(self, t: float) -> None:
        self.output.power = self.power_scale * self.resistance * abs(self.output.i) ** 2

    def network(self, nb):
        """Declare the shunt or series resistance."""
        from ..solver import netlist as nl

        other = nb.node(self.port2) if self.series else None
        nb.add(nl.Resistor(nb.node(self.port1), other, self.resistance))
        nb.port_output(self, "i", self.port1, "current")
        nb.port_output(
            self, "u", self.port2 if self.series else self.port1, "voltage"
        )
        nb.losses(self)


@register_component_type
class Resistor(Element):
    """Shunt resistance, or a series resistance inserted after a bus's voltage driver."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "resistor"
        bus: str
        r: float
        series: bool = field(default=False, metadata={"runtime": False})

        _quantities = {"r": "resistance"}

        def __post_init__(self) -> None:
            if not self.r > 0.0:
                raise ConfigError(f"r must be > 0, got {self.r}")
            if type(self.series) is not bool:
                raise ConfigError("series must be bool")

    type = "resistor"
    domain_transparent = True

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        self.resistor = _ResistorModel(cfg.r, buses[cfg.bus].phases, cfg.series)

    def subsystems(self) -> dict[str, Any]:
        return {self.name: self.resistor}

    @staticmethod
    def connections() -> dict:
        return {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        if self.cfg.series:
            return ((f"{self.name}.bus.in", self.resistor.port1),
                    (self.cfg.bus, self.resistor.port2))
        return ((self.cfg.bus, self.resistor.port1),)

    def series_connections(self) -> dict[str, str]:
        return {self.cfg.bus: f"{self.name}.bus.in"} if self.cfg.series else {}

    def parameter_bindings(self) -> dict[str, tuple[Any, str]]:
        """Runtime scalar parameters, shared by Python retuning and native export."""
        return {"r": (self.resistor, "resistance")}

    def signals(self) -> dict[str, complex | float]:
        return {f"{self.name}.i": self.resistor.output.i,
                f"{self.name}.power": self.resistor.output.power}

    def retune(self, cfg: Any) -> bool:
        changed = self.cfg.r != cfg.r
        self.cfg = cfg
        for name, (subsystem, attribute) in self.parameter_bindings().items():
            setattr(subsystem, attribute, getattr(cfg, name))
        return changed



