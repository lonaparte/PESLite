"""Series and parallel RLC networks and electrical buses."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import ClassVar

from ..solver.energy import Port, StoragePort
from ..solver.model import Empty, OutputMethod, record_type
from .network import Element, NetworkComponent, _phase_convention

__all__ = ["SeriesRLC", "RLBranch", "Capacitor", "Resistor", "ParallelRLC", "Bus"]


class SeriesRLC(Element, NetworkComponent):
    """A series connection containing any nonempty combination of R, L and C."""

    State = record_type("SeriesRLCState", ("i", "u_C"))
    Output = record_type("SeriesRLCOutput", ("i", "u", "power"))
    domain_transparent = True
    outputs_need_inputs: ClassVar[bool] = False
    parameter_fields = {"r": "R", "l": "L", "c": "C"}
    structural_parameters = frozenset(("l", "c"))
    equation_fields = {"breaker_open": "breaker_open"}
    signal_fields = {"i": "i", "voltage": "u", "power": "power"}
    infer_netlist_storage = True
    netlist_jumps = False
    output_stages = (OutputMethod(
        "set_voltage_and_power", (), ("u", "power"), rhs=False,
    ),)

    def __init__(self, R: float = 0.0, L: float | None = None,
                 C: float | None = None, *, i0: complex | float = 0j,
                 u0: complex | float = 0j, phases: int = 3,
                 breakable: bool = True) -> None:
        self.power_scale, self.zero, initial_i = _phase_convention(phases, i0)
        _scale, _zero, initial_u = _phase_convention(phases, u0)
        if not math.isfinite(R) or R < 0.0:
            raise ValueError(f"R must be finite and >= 0, got {R}")
        if L is not None and (not math.isfinite(L) or L <= 0.0):
            raise ValueError(f"L must be finite and > 0, got {L}")
        if C is not None and (not math.isfinite(C) or C <= 0.0):
            raise ValueError(f"C must be finite and > 0, got {C}")
        if R == 0.0 and L is None and C is None:
            raise ValueError("a series RLC needs at least one nonzero element")
        self.R, self.L, self.C = R, L, C
        self.phases, self.breakable = phases, bool(breakable)
        self.state_names = tuple(name for name, value in (("i", L), ("u_C", C))
                                 if value is not None)
        self.state = self.State(i=initial_i, u_C=initial_u)
        self.input = Empty()
        self.output = self.Output(i=initial_i, u=initial_u, power=0.0)
        self.port1 = Port(self, "1", voltage=self.zero, current=initial_i,
                          scale=self.power_scale, standalone="voltage")
        self.port2 = Port(self, "2", voltage=self.zero, current=-initial_i,
                          scale=self.power_scale, standalone="voltage")
        self.ports = (self.port1, self.port2)
        self.storage = ()
        self.breaker_open = False

    def open_breaker(self) -> None:
        self.breaker_open = True
        if not self.netlist_jumps and "i" in self.state_names:
            self.state.i = self.zero

    def close_breaker(self) -> None:
        self.breaker_open = False

    def set_values(self, L: float | None, R: float,
                   C: float | None = None) -> bool:
        return self.set_parameters(l=L, r=R, c=C)

    def resistive_power(self) -> float:
        return self.power_scale * self.R * abs(self.output.i) ** 2

    def set_voltage_and_power(self, _t: float) -> None:
        self.output.u = self.port1.voltage - self.port2.voltage
        self.output.power = self.resistive_power()

    def network(self, nb):
        return self._declare_network(nb, nb.node(self.port1))

    def _declare_network(self, nb, start):
        from ..solver import netlist as nl

        if self.breakable:
            path = nb.internal(start.dim, start.scale)
            nb.add(nl.Switch(start, path, nb.value(self, "breaker_open", invert=True)))
        else:
            path = start
        values = (("R", self.R), ("L", self.L), ("C", self.C))
        active = tuple((kind, value) for kind, value in values if value not in (None, 0.0))
        for index, (kind, value) in enumerate(active):
            end = (nb.node(self.port2) if index == len(active) - 1
                   else nb.internal(start.dim, start.scale))
            if kind == "R":
                nb.add(nl.Resistor(path, end, value))
            elif kind == "L":
                nb.add(nl.Inductor(path, end, value, nb.state(self, "i")))
            else:
                nb.add(nl.Capacitor(path, end, value, nb.state(self, "u_C")))
            path = end
        nb.port_output(self, "i", self.port1, "current")
        nb.losses(self)


class RLBranch(SeriesRLC):
    """A series R-L branch with optional one-way and external conduction behavior."""

    Input = record_type("RLBranchInput", ("di_dt", "loss_factor"))

    def __init__(self, R: float = 0.0, L: float | None = None, *,
                 i0: complex | float = 0j, phases: int = 3,
                 unilateral: bool = False, breakable: bool = True) -> None:
        if L is None:
            raise ValueError("an R-L branch requires L")
        if unilateral and phases != 1:
            raise ValueError("unilateral current requires a scalar R-L branch")
        super().__init__(R=R, L=L, i0=i0, phases=phases, breakable=breakable)
        self.unilateral = bool(unilateral)
        self.external_conduction = False
        self.input = self.Input(di_dt=self.zero, loss_factor=1.0)
        if self.unilateral:
            self.state_bounds = {"i": (0.0, None)}
            self.commutation_margin = self._device_commutation_margin
            self.commutate = self._device_commutate

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
        return self.unilateral

    def network(self, nb):
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

        start = nb.node(self.port1)
        if self.unilateral:
            diode = nb.add(nl.Diode(start, nb.internal(start.dim, start.scale)))
            start = diode.ports[0][1]
        return self._declare_network(nb, start)


class Capacitor(SeriesRLC):
    """A two-terminal capacitor with optional series resistance."""

    def __init__(self, R: float = 0.0, C: float | None = None, *,
                 u0: complex | float = 0j, phases: int = 3,
                 breakable: bool = True) -> None:
        if C is None:
            raise ValueError("a capacitor requires C")
        super().__init__(R, None, C, u0=u0, phases=phases, breakable=breakable)


class Resistor(SeriesRLC):
    """A pure resistance connected to ground, between terminals or in series at a bus."""

    def __init__(self, R: float, *, phases: int = 3,
                 breakable: bool = True) -> None:
        super().__init__(R, phases=phases, breakable=breakable)


class ParallelRLC(Element, NetworkComponent):
    """A parallel connection containing any nonempty combination of R, L and C."""

    State = record_type("ParallelRLCState", ("i_L", "u_C"))
    Output = record_type("ParallelRLCOutput", ("i", "u", "power"))
    domain_transparent = True
    outputs_need_inputs: ClassVar[bool] = False
    parameter_fields = {"r": "R", "l": "L", "c": "C"}
    structural_parameters = frozenset(("r", "l", "c"))
    equation_fields = {"breaker_open": "breaker_open"}
    signal_fields = {"i": "i", "voltage": "u", "power": "power"}
    infer_netlist_storage = True
    output_stages = (OutputMethod(
        "set_voltage_and_power", (), ("u", "power"), rhs=False,
    ),)

    def __init__(self, R: float | None = None, L: float | None = None,
                 C: float | None = None, *, i0: complex | float = 0j,
                 u0: complex | float = 0j, phases: int = 3,
                 breakable: bool = True) -> None:
        self.power_scale, self.zero, initial_i = _phase_convention(phases, i0)
        _scale, _zero, initial_u = _phase_convention(phases, u0)
        for name, value in (("R", R), ("L", L), ("C", C)):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if R is None and L is None and C is None:
            raise ValueError("a parallel RLC needs at least one element")
        self.R, self.L, self.C, self.phases = R, L, C, phases
        self.breakable = bool(breakable)
        self.state_names = tuple(name for name, value in (("i_L", L), ("u_C", C))
                                 if value is not None)
        self.state = self.State(i_L=initial_i, u_C=initial_u)
        self.input = Empty()
        self.output = self.Output(i=self.zero, u=initial_u, power=0.0)
        self.port1 = Port(self, "1", voltage=self.zero, current=self.zero,
                          scale=self.power_scale, standalone="voltage")
        self.port2 = Port(self, "2", voltage=self.zero, current=self.zero,
                          scale=self.power_scale, standalone="voltage")
        self.ports = (self.port1, self.port2)
        self.storage = ()
        self.breaker_open = False

    def open_breaker(self) -> None:
        self.breaker_open = True

    def close_breaker(self) -> None:
        self.breaker_open = False

    def set_values(self, L: float | None, R: float | None,
                   C: float | None = None) -> bool:
        return self.set_parameters(l=L, r=R, c=C)

    def resistive_power(self) -> float:
        return (0.0 if self.R is None else
                self.power_scale * abs(self.output.u) ** 2 / self.R)

    def set_voltage_and_power(self, _t: float) -> None:
        self.output.u = self.port1.voltage - self.port2.voltage
        self.output.power = self.resistive_power()

    def network(self, nb):
        from ..solver import netlist as nl

        start = nb.node(self.port1)
        if self.breakable:
            common = nb.internal(start.dim, start.scale)
            nb.add(nl.Switch(start, common, nb.value(self, "breaker_open", invert=True)))
        else:
            common = start
        end = nb.node(self.port2)
        if self.R is not None:
            nb.add(nl.Resistor(common, end, self.R))
        if self.L is not None:
            nb.add(nl.Inductor(common, end, self.L, nb.state(self, "i_L")))
        if self.C is not None:
            nb.add(nl.Capacitor(common, end, self.C, nb.state(self, "u_C")))
        nb.port_output(self, "i", self.port1, "current")
        nb.losses(self)


class Bus(NetworkComponent):
    """Electrical connection point, optionally with a damped shunt capacitor to ground."""

    State = record_type("BusState", ("u_C",))
    Output = record_type("BusOutput", ("u",))
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

    def retune(self, C: float | None, R_d: float) -> bool:
        if (self.C is None) != (C is None):
            raise ValueError("bus capacitance cannot be added or removed after assembly")
        if C is None:
            return False
        changed = self.C != C or self.R_d != R_d
        self.C, self.R_d = C, R_d
        self.storage = (replace(self.storage[0], value=C),)
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
