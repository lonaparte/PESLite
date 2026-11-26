"""Series and parallel RLC networks and electrical buses."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import ClassVar

from ..solver.energy import Port, StoragePort
from ..solver.model import Empty, OutputMethod, record_type
from .network import Element, _phase_convention

__all__ = ["Passive", "SeriesRLC", "ParallelRLC", "RLBranch", "Capacitor", "Resistor", "Bus"]


class Passive(Element):
    """Common two-terminal behavior of passive electrical elements."""

    Output = record_type("PassiveOutput", ("i", "u", "power"))
    domain_transparent = True
    outputs_need_inputs: ClassVar[bool] = False
    parameter_fields = {"r": "R", "l": "L", "c": "C"}
    equation_fields = {"breaker_open": "breaker_open"}
    signal_fields = {"i": "i", "voltage": "u", "power": "power"}
    infer_netlist_storage = True
    netlist_jumps = True
    output_stages = (OutputMethod(
        "set_voltage_and_power", (), ("u", "power"), rhs=False,
    ),)

    def __init__(self, R: float | None, L: float | None, C: float | None, *,
                 i0: complex | float, u0: complex | float,
                 phases: int, breakable: bool) -> None:
        R, L, C = (None if value == 0.0 else value for value in (R, L, C))
        for name, value in (("R", R), ("L", L), ("C", C)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        self.power_scale, self.zero, initial_i = _phase_convention(phases, i0)
        _scale, _zero, initial_u = _phase_convention(phases, u0)
        self.R, self.L, self.C = R, L, C
        self.phases, self.breakable = phases, bool(breakable)
        initial = {self.inductor_state: initial_i, "u_C": initial_u}
        self.state_names = tuple(
            name for name, value in ((self.inductor_state, L), ("u_C", C))
            if value is not None
        )
        self.state = self.State(**initial)
        self.input = Empty()
        current = (self.zero if self.output_current_state is None else
                   initial[self.output_current_state])
        self.output = self.Output(i=current, u=initial_u, power=0.0)
        self.port1 = Port(
            self, "1", voltage=self.zero, current=current,
            scale=self.power_scale, standalone="voltage",
        )
        self.port2 = Port(
            self, "2", voltage=self.zero, current=-current,
            scale=self.power_scale, standalone="voltage",
        )
        self.ports = (self.port1, self.port2)
        self.storage = ()
        self.breaker_open = False

    def set_parameters(self, **values) -> bool:
        return super().set_parameters(**{
            name: None if value == 0.0 else value for name, value in values.items()
        })

    def open_breaker(self) -> None:
        self.breaker_open = True
        if not self.netlist_jumps and "i" in self.state_names:
            self.state.i = self.zero

    def close_breaker(self) -> None:
        self.breaker_open = False

    def network(self, nb):
        from ..solver import netlist as nl

        if self.R is None and self.L is None and self.C is None:
            nb.port_output(self, "i", self.port1, "current")
            return None
        start = nb.node(self.port1)
        if self.breakable:
            path = nb.internal(start.dim, start.scale)
            nb.add(nl.Switch(start, path, nb.value(self, "breaker_open", invert=True)))
        else:
            path = start
        self.declare_network(nb, path, nb.node(self.port2))
        nb.port_output(self, "i", self.port1, "current")
        nb.losses(self)


class SeriesRLC(Passive):
    """A series connection containing any nonempty combination of R, L and C."""

    State = record_type("SeriesRLCState", ("i", "u_C"))
    inductor_state = output_current_state = "i"
    structural_parameters = frozenset(("l", "c"))
    netlist_jumps = False

    def __init__(self, R: float = 0.0, L: float | None = None,
                 C: float | None = None, *, i0: complex | float = 0j,
                 u0: complex | float = 0j, phases: int = 3,
                 breakable: bool = True) -> None:
        super().__init__(R, L, C, i0=i0, u0=u0, phases=phases, breakable=breakable)

    def set_voltage_and_power(self, _t: float) -> None:
        self.output.u = self.port1.voltage - self.port2.voltage
        self.output.power = self.power_scale * (self.R or 0.0) * abs(self.output.i) ** 2

    def declare_network(self, nb, path, end) -> None:
        from ..solver import netlist as nl

        values = (("R", self.R), ("L", self.L), ("C", self.C))
        active = tuple((kind, value) for kind, value in values if value is not None)
        for index, (kind, value) in enumerate(active):
            target = end if index == len(active) - 1 else nb.internal(path.dim, path.scale)
            if kind == "R":
                nb.add(nl.Resistor(path, target, value))
            elif kind == "L":
                nb.add(nl.Inductor(path, target, value, nb.state(self, "i")))
            else:
                nb.add(nl.Capacitor(path, target, value, nb.state(self, "u_C")))
            path = target


class ParallelRLC(Passive):
    """A parallel connection containing any nonempty combination of R, L and C."""

    State = record_type("ParallelRLCState", ("i_L", "u_C"))
    inductor_state = "i_L"
    output_current_state = None
    structural_parameters = frozenset(("r", "l", "c"))

    def __init__(self, R: float | None = None, L: float | None = None,
                 C: float | None = None, *, i0: complex | float = 0j,
                 u0: complex | float = 0j, phases: int = 3,
                 breakable: bool = True) -> None:
        super().__init__(R, L, C, i0=i0, u0=u0, phases=phases, breakable=breakable)

    def set_voltage_and_power(self, _t: float) -> None:
        self.output.u = self.port1.voltage - self.port2.voltage
        self.output.power = (0.0 if self.R is None else
                             self.power_scale * abs(self.output.u) ** 2 / self.R)

    def declare_network(self, nb, start, end) -> None:
        from ..solver import netlist as nl

        if self.R is not None:
            nb.add(nl.Resistor(start, end, self.R))
        if self.L is not None:
            nb.add(nl.Inductor(start, end, self.L, nb.state(self, "i_L")))
        if self.C is not None:
            nb.add(nl.Capacitor(start, end, self.C, nb.state(self, "u_C")))


class RLBranch(SeriesRLC):
    """A series R-L branch."""

    def __init__(self, R: float = 0.0, L: float | None = None, *,
                 i0: complex | float = 0j, phases: int = 3,
                 breakable: bool = True) -> None:
        super().__init__(R=R, L=L, i0=i0, phases=phases, breakable=breakable)


class Capacitor(SeriesRLC):
    """A two-terminal capacitor with optional series resistance."""

    def __init__(self, R: float = 0.0, C: float | None = None, *,
                 u0: complex | float = 0j, phases: int = 3,
                 breakable: bool = True) -> None:
        super().__init__(R, None, C, u0=u0, phases=phases, breakable=breakable)


class Resistor(SeriesRLC):
    """A pure resistance connected to ground, between terminals or in series at a bus."""

    def __init__(self, R: float, *, phases: int = 3,
                 breakable: bool = True) -> None:
        super().__init__(R, phases=phases, breakable=breakable)


class Bus(Element):
    """Electrical connection point, optionally with a damped shunt capacitor to ground."""

    State = record_type("BusState", ("u_C",))
    Output = record_type("BusOutput", ("u",))
    parameter_fields = {"c": "C", "r_d": "R_d"}
    structural_parameters = frozenset(("c",))
    voltage_field: ClassVar[str] = "u"
    outputs_need_inputs: ClassVar[bool] = False

    def __init__(self, C: float | None = None, R_d: float = 0.0,
                 u0: complex | float = 0j, *, phases: int = 3) -> None:
        self.power_scale, zero, initial = _phase_convention(phases, u0)
        self.C, self.R_d = C, R_d
        self.phases = phases
        self.state_names = () if C is None else ("u_C",)
        self.state = Empty() if C is None else self.State(u_C=initial)
        self.input = Empty()
        self.output = self.Output(u=zero if C is None else initial)
        self.observer = C is None
        self.port: Port | None = Port(
            self, "power", voltage=self.output.u, current=zero,
            scale=self.power_scale,
            electrical_domain="ac" if C is None and phases == 3 else None,
            standalone="current" if C is not None else None,
        )
        self.ports = (self.port,)
        self.storage = (() if C is None else (StoragePort(
            "u_C", C, self.power_scale, ((self.port, "current", 1.0),)
        ),))

    def set_parameters(self, **values) -> bool:
        changed = super().set_parameters(**values)
        if self.C is not None:
            self.storage = (replace(self.storage[0], value=self.C),)
        return changed

    def network(self, nb):
        from ..solver import netlist as nl

        if self.port is None:
            return NotImplemented
        node = nb.node(self.port)
        if self.C is None:
            nb.port_output(self, "u", self.port, "voltage")
            return None
        capacitor = nb.internal(node.dim, node.scale) if self.R_d else node
        if self.R_d:
            nb.add(nl.Resistor(node, capacitor, self.R_d))
        nb.add(nl.Capacitor(capacitor, None, self.C, nb.state(self, "u_C")))
        nb.port_output(self, "u", self.port, "voltage")
        nb.losses(self)
