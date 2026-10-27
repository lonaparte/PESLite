"""AC network blocks and registered circuit-element types.

Quantities are SI; ac quantities are peak-scaled space vectors. Breakers expose
``open_breaker()`` and ``close_breaker()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Callable, ClassVar, Optional


from ..solver.energy import PowerPort, StoragePort, port_power
from ..solver.model import Bag, ConfigError, Empty, OutputMethod, _complex, _real

__all__ = [
    "Terminal", "ACVoltageSource", "RLBranch", "RCNode", "Element", "ELEMENT_TYPES",
    "register_element_type", "element_buses", "Load", "Resistor", "VoltageSource",
]


@dataclass(frozen=True, eq=False)
class Terminal:
    """One electrical terminal of a subsystem.

    ``direction`` is +1 when positive current enters the subsystem and -1 when it leaves.
    A voltage-following terminal receives voltage through ``inp`` and publishes current through
    ``out``. A voltage-driving terminal does the reverse. These are construction-time references;
    System lowers a named bus directly to ordinary Model connections.
    """

    subsystem: Any
    voltage: str
    current: str
    direction: int
    drives_voltage: bool = False
    hold_voltage_input: Optional[str] = None
    electrical_domain: Optional[str] = None
    current_sources: Any = field(init=False, repr=False, compare=False, default=None)
    voltage_source: Any = field(init=False, repr=False, compare=False, default=None)
    power_port: PowerPort = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.drives_voltage) is not bool:
            raise TypeError(
                f"terminal drives_voltage must be bool, got {type(self.drives_voltage).__name__}"
            )
        if self.direction not in (-1, 1) or isinstance(self.direction, bool):
            raise ValueError(f"terminal direction must be +1 or -1, got {self.direction!r}")
        if self.electrical_domain not in (None, "ac", "dc"):
            raise ValueError(
                "terminal electrical_domain must be 'ac', 'dc' or None, got "
                f"{self.electrical_domain!r}"
            )
        voltage_record = self.subsystem.out if self.drives_voltage else self.subsystem.inp
        current_record = self.subsystem.inp if self.drives_voltage else self.subsystem.out
        voltage_scope = "out" if self.drives_voltage else "inp"
        current_scope = "inp" if self.drives_voltage else "out"
        if not hasattr(voltage_record, self.voltage):
            raise ValueError(
                f"{type(self.subsystem).__name__}.{voltage_scope} has no terminal voltage "
                f"{self.voltage!r}"
            )
        if not hasattr(current_record, self.current):
            raise ValueError(
                f"{type(self.subsystem).__name__}.{current_scope} has no terminal current "
                f"{self.current!r}"
            )
        effort = f"{voltage_scope}.{self.voltage}"
        flow = f"{current_scope}.{self.current}"
        for port in getattr(self.subsystem, "ports", ()):
            if (port.effort, port.flow) == (effort, flow):
                if port.sign != self.direction:
                    raise ValueError(
                        f"terminal direction {self.direction:+d} does not match the energy-port "
                        f"direction {port.sign:+g}"
                    )
                break
        else:
            port = PowerPort(effort, flow, sign=self.direction)
        object.__setattr__(self, "power_port", port)

    def power(self):
        """Instantaneous power entering this terminal, using its energy-port convention."""
        return port_power(self.subsystem, self.power_port)

    def bind_voltage_source(self, source: tuple) -> None:
        """Remember the already-lowered voltage source for follower-side observers."""
        if self.drives_voltage:
            raise ValueError("only a voltage-following terminal receives a bus voltage")
        object.__setattr__(self, "voltage_source", source)

    def signal_sources(self) -> tuple[Any, Any]:
        """Ordinary output connections for observing the paired voltage and current."""
        voltage = ((self.subsystem, self.voltage) if self.drives_voltage
                   else self.voltage_source)
        current = (self.current_sources if self.drives_voltage
                   else (self.subsystem, self.current))
        if voltage is None or current is None:
            raise RuntimeError("the terminal's bus has not been assembled")
        return voltage, current

    def bind_current_sources(self, sources: list[tuple]) -> None:
        """Remember the already-lowered current fan-in for driver-side observers."""
        if not self.drives_voltage:
            raise ValueError("only a voltage-driving terminal owns the bus current target")
        object.__setattr__(self, "current_sources", sources)

    def hold_voltage_source(self, current_constraint=None) -> tuple:
        """Ask this current-owning component for its zero-current voltage law."""
        provider = getattr(self.subsystem, "hold_voltage_source", None)
        if self.drives_voltage or provider is None:
            raise ConfigError(
                f"{type(self.subsystem).__name__}.{self.voltage}: no zero-current voltage law")
        return provider(self, current_constraint=current_constraint)


class _BranchState(Bag):
    __slots__ = ("i",)

class _BranchInp(Bag):
    __slots__ = ("u1", "u2", "di_dt", "loss_factor")

class _BranchOut(Bag):
    __slots__ = ("i",)

class RLBranch:
    """Series R-L branch with positive current from terminal 1 to terminal 2.

    ``phases=3`` uses the peak-scaled complex space-vector convention. ``phases=1`` uses a real
    scalar and the same terminal contract, so converter units do not need a second network API.
    """

    state_names: ClassVar[tuple[str, ...]] = ("i",)
    outputs_need_inputs: ClassVar[bool] = False
    def __init__(self, L: float, R: float, i0: complex | float = 0j, *, phases: int = 3,
                 unilateral: bool = False) -> None:
        if phases not in (1, 3):
            raise ValueError(f"phases must be 1 or 3, got {phases!r}")
        if unilateral and phases != 1:
            raise ValueError("unilateral current is only available for scalar branches")
        self.L, self.R = L, R
        self.phases = phases
        self.power_scale = 1.5 if phases == 3 else 1.0
        self.zero = 0j if phases == 3 else 0.0
        self.unilateral = bool(unilateral)
        initial = _complex(i0) if phases == 3 else _real(i0.real if isinstance(i0, complex) else i0)
        self.ports = (PowerPort("inp.u1", "out.i", self.power_scale, 1.0),
                      PowerPort("inp.u2", "out.i", self.power_scale, -1.0))
        self.storage = (StoragePort(
            "i", L, self.power_scale, (("inp.u1", 1.0), ("inp.u2", -1.0)), "inductor",
        ),)
        self.state = _BranchState(i=initial)
        self.inp = _BranchInp(
            u1=self.zero, u2=self.zero, di_dt=self.zero, loss_factor=1.0,
        )
        self.out = _BranchOut(i=initial)
        self.terminal1 = Terminal(self, "u1", "i", 1)
        self.terminal2 = Terminal(self, "u2", "i", -1)
        self.breaker_open = False
        if self.unilateral:
            self.state_bounds = {"i": (0.0, None)}
        self._output_equation = (self._set_outputs_unilateral if self.unilateral
                                 else self._set_outputs_bilateral)
        self._closed_rhs = self._rhs_unilateral if self.unilateral else self._rhs_bilateral
        self._uses_default_rhs = type(self).rhs is RLBranch.rhs
        if type(self).set_outputs is RLBranch.set_outputs:
            self.set_outputs = self._output_equation
        if self._uses_default_rhs:
            self.rhs = self._closed_rhs

    def open_breaker(self) -> None:
        self.breaker_open = True
        self.state.i = self.zero
        if self._uses_default_rhs:
            self.rhs = self._rhs_open

    def close_breaker(self) -> None:
        """Close the breaker; current resumes from zero after an opening."""
        self.breaker_open = False
        if self._uses_default_rhs:
            self.rhs = self._closed_rhs

    def retune(self, L: float, R: float) -> None:
        """Change inductance and resistance without changing the current state."""
        self.L, self.R = L, R
        self.storage = (replace(self.storage[0], value=L),)

    def set_outputs(self, t: float) -> None:
        """Evaluate the output equation selected when the branch was constructed."""
        self._output_equation(t)

    def _set_outputs_bilateral(self, t: float) -> None:
        self.out.i = self.state.i

    def _set_outputs_unilateral(self, t: float) -> None:
        self.out.i = max(0.0, self.state.i)

    def rhs(self, t: float):
        """Evaluate the active state equation without duplicating its electrical law."""
        if self.breaker_open:
            return (self.zero,)
        return self._closed_rhs(t)

    def _rhs_open(self, t: float):
        return (self.zero,)

    def _rhs_bilateral(self, t: float):
        return ((self.inp.u1 - self.inp.u2 - self.R * self.state.i) / self.L,)

    def _rhs_unilateral(self, t: float):
        derivative = (self.inp.u1 - self.inp.u2 - self.R * self.state.i) / self.L
        return (0.0 if self.state.i <= 0.0 and derivative < 0.0 else derivative,)

    def _rhs_conduction(self, _t: float):
        return (self.inp.di_dt,)

    def use_external_conduction(self) -> None:
        """Take derivative and copper-loss factor from a connected constitutive subsystem."""
        self._closed_rhs = self._rhs_conduction
        self.energy_audit_note = (
            "cycle-averaged DCM current does not have the branch's linear instantaneous "
            "storage function"
        )
        if not self.breaker_open and self._uses_default_rhs:
            self.rhs = self._closed_rhs

    def diode_blocks_reverse_current(self) -> bool:
        """Return whether this branch represents a diode-bounded current path."""
        return self.unilateral

    def hold_voltage_source(self, terminal: Terminal, *, current_constraint=None) -> tuple:
        """Publish the terminal voltage at which this branch's current is constant."""
        closed = getattr(self._closed_rhs, "__func__", self._closed_rhs)
        if (self.phases != 1 or self.unilateral
                or closed is not RLBranch._rhs_bilateral):
            raise ConfigError("a blocking port needs an ordinary scalar R-L branch")
        if current_constraint is not None:
            if not hasattr(self, "_current_constraints"):
                self._current_constraints = []
                self.project_state = self._project_current
            self._current_constraints.append(current_constraint)
        name = f"hold_{terminal.voltage}"
        other = "u2" if terminal.voltage == "u1" else "u1"
        if not hasattr(self.out, name):
            names = (*type(self.out).__slots__, name)
            self.out = type("_BranchOut", (Bag,), {"__slots__": names})(
                **{key: getattr(self.out, key) for key in type(self.out).__slots__})
            stages = getattr(self, "output_stages", (
                OutputMethod("set_outputs", inputs=(), outputs=("i",)),))
            self.output_stages = (*stages, OutputMethod(
                f"set_{name}", inputs=(other,), outputs=(name,)))
        return self, name

    def _project_current(self) -> None:
        targets = {value for read in self._current_constraints if (value := read()) is not None}
        if len(targets) > 1:
            raise ConfigError("incompatible current constraints on the same R-L branch")
        if targets:
            self.state.i = targets.pop()

    def set_hold_u1(self, t: float) -> None:
        self.out.hold_u1 = self.inp.u2 + self.R * self.state.i

    def set_hold_u2(self, t: float) -> None:
        self.out.hold_u2 = self.inp.u1 - self.R * self.state.i

    def dissipated_power(self) -> float:
        return self.power_scale * self.R * self.inp.loss_factor * abs(self.state.i) ** 2

    def supplied_power(self) -> float:
        return 0.0

class _NodeState(Bag):
    __slots__ = ("u_C",)

class _NodeInp(Bag):
    __slots__ = ("i",)

class _NodeOut(Bag):
    __slots__ = ("u",)

class RCNode:
    """Node with a shunt capacitor ``C`` (F) in series with ``R_d`` (ohm) to ground.

    Input ``i`` is the net current into the node. Output ``u = u_C + R_d i``.
    """

    state_names: ClassVar[tuple[str, ...]] = ("u_C",)
    outputs_need_inputs: ClassVar[bool] = True
    def __init__(self, C: float, R_d: float, u0: complex | float = 0j, *, phases: int = 3) -> None:
        if phases not in (1, 3):
            raise ValueError(f"phases must be 1 or 3, got {phases!r}")
        self.C, self.R_d = C, R_d
        self.phases = phases
        self.power_scale = 1.5 if phases == 3 else 1.0
        initial = _complex(u0) if phases == 3 else _real(u0.real if isinstance(u0, complex) else u0)
        zero = 0j if phases == 3 else 0.0
        self.ports = (PowerPort("out.u", "inp.i", self.power_scale),)
        self.storage = (StoragePort("u_C", C, self.power_scale, (("inp.i", 1.0),)),)
        self.state = _NodeState(u_C=initial)
        self.inp = _NodeInp(i=zero)
        self.out = _NodeOut(u=initial)
        self.terminal = Terminal(self, "u", "i", 1, drives_voltage=True)

    def set_outputs(self, t: float) -> None:
        self.out.u = self.state.u_C + self.R_d * self.inp.i

    def rhs(self, t: float):
        return (self.inp.i / self.C,)

    def retune(self, C: float, R_d: float) -> None:
        """Change capacitance and damping resistance without changing voltage."""
        self.C, self.R_d = C, R_d
        self.storage = (replace(self.storage[0], value=C),)

    def dissipated_power(self) -> float:
        return self.power_scale * self.R_d * abs(self.inp.i) ** 2

    def supplied_power(self) -> float:
        return 0.0


class _SourceOut(Bag):
    __slots__ = ("e_g", "phi", "theta")


class _SourceInp(Bag):
    __slots__ = ("i",)


class ACVoltageSource:
    """Ideal single- or three-phase AC voltage source.

    Three phases use ``E exp(j (w0 t + phi(t)))``; one phase uses its real cosine.
    ``w0`` is in rad/s, ``e_peak`` in V and ``phi(t)`` is the relative angle in rad.
    """

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    storage: ClassVar[tuple[StoragePort, ...]] = ()
    has_source: ClassVar[bool] = True

    def __init__(self, w0: float, e_peak: float,
                 phi: Optional[Callable[[float], float]] = None, *, phases: int = 3) -> None:
        if phases not in (1, 3):
            raise ValueError(f"phases must be 1 or 3, got {phases!r}")
        self.w0, self.e_peak, self.phases = w0, e_peak, phases
        self.power_scale = 1.5 if phases == 3 else 1.0
        self.phi = phi
        self.state = Empty()
        self.inp = _SourceInp(i=0j if phases == 3 else 0.0)
        self.out = _SourceOut()
        self.out.e_g = 0j if phases == 3 else 0.0
        self.ports = (PowerPort("out.e_g", "inp.i", self.power_scale, -1.0),)
        self.terminal = Terminal(
            self, "e_g", "i", -1, drives_voltage=True, electrical_domain="ac"
        )
        self._output_equation = (self._set_three_phase_outputs if phases == 3
                                 else self._set_single_phase_outputs)
        if type(self).set_outputs is ACVoltageSource.set_outputs:
            self.set_outputs = self._output_equation

    def set_outputs(self, t: float) -> None:
        """Evaluate the phase representation selected when the source was constructed."""
        self._output_equation(t)

    def _set_single_phase_outputs(self, t: float) -> None:
        phi = self.phi(t) if self.phi is not None else 0.0
        theta = self.w0 * t + phi
        out = self.out
        out.phi = phi
        out.theta = theta
        out.e_g = self.e_peak * math.cos(theta)

    def _set_three_phase_outputs(self, t: float) -> None:
        phi = self.phi(t) if self.phi is not None else 0.0
        theta = self.w0 * t + phi
        out = self.out
        out.phi = phi
        out.theta = theta
        out.e_g = complex(self.e_peak * math.cos(theta), self.e_peak * math.sin(theta))

    @classmethod
    def uses_standard_outputs(cls, method: Any) -> bool:
        """Whether ``method`` is one of the built-in phase-specialized source equations."""
        return getattr(method, "__func__", None) in (
            cls.set_outputs, cls._set_single_phase_outputs, cls._set_three_phase_outputs,
        )

    def rhs(self, t: float):
        return ()

    def dissipated_power(self) -> float:
        return 0.0

    def supplied_power(self) -> float:
        return self.power_scale * (self.out.e_g * self.inp.i.conjugate()).real


# ------------------------------------------------------------------ circuit element types

class Element:
    """Base of a registered ``elements`` entry, built as ``cls(name, cfg, buses, p)``.

    A type defines ``type`` and a frozen ``Params`` dataclass with a matching ``type`` field.
    An instance supplies ``subsystems()``, ``connections()`` and ``terminals()``.  Each terminal
    binds a configured bus name to a :class:`Terminal`; one- and multi-terminal elements use the
    same interface. Optional ``breakers`` or ``connect()`` enable switching; optional ``retune()``
    enables runtime parameter changes. Optional ``series_connections()`` maps a bus to an inlet
    bus: assembly moves the original voltage driver to that inlet, inserting a two-port element
    before the bus's followers without adding an artificial energy-storage state.
    Optional ``parameter_bindings()`` maps retunable scalar fields to ``(subsystem, attribute)``
    pairs so native export uses the same parameters as Python, including energy equations.
    """

    type: ClassVar[str]
    Params: ClassVar[type]
    domain_transparent: ClassVar[bool] = False


ELEMENT_TYPES: dict[str, type] = {}


def register_element_type(cls: type) -> type:
    """Register an :class:`Element` class under ``cls.type`` and return it."""
    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not name:
        raise TypeError("an element type needs a name: the class attribute 'type'")
    if name in ELEMENT_TYPES:
        raise ValueError(f"element type {name!r} is already registered")
    params = getattr(cls, "Params", None)
    if not is_dataclass(params) or "type" not in params.__dataclass_fields__:
        raise TypeError(f"element type {name!r}: Params must be a dataclass with a type field")
    if params.__dataclass_fields__["type"].default != name:
        raise TypeError(f"element type {name!r}: Params.type must default to {name!r}")
    ELEMENT_TYPES[name] = cls
    return cls


def element_buses(cfg: Any) -> dict[str, str]:
    """Return fields named ``bus`` or ``busN`` from an element's parameters."""
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)
            if f.init and (f.name == "bus"
                           or (f.name.startswith("bus") and f.name[3:].isdigit()
                               and int(f.name[3:]) > 0))}


@register_element_type
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
        self.breakers = [self.branch]

    def subsystems(self) -> dict[str, Any]:
        return {f"{self.name}.branch": self.branch}

    def connections(self) -> dict:
        return {}

    def terminals(self) -> tuple[tuple[str, Terminal], ...]:
        return ((self.cfg.bus, self.branch.terminal1),)

    def signals(self) -> dict[str, complex]:
        return {f"{self.name}.i": self.branch.out.i}

    def retune(self, cfg: Any) -> None:
        """Take new load parameters while preserving branch current."""
        self.cfg = cfg
        self.branch.retune(cfg.l, cfg.r)


class _ResistorInp(Bag):
    __slots__ = ("u", "i")


class _ResistorOut(Bag):
    __slots__ = ("i", "u", "power")


class _ResistorModel:
    """One resistance, in shunt admittance or series voltage-drop form."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = True
    storage: ClassVar[tuple[StoragePort, ...]] = ()

    def __init__(self, resistance: float, phases: int, series: bool = False) -> None:
        self.resistance = resistance
        self.series = series
        self.power_scale = 1.5 if phases == 3 else 1.0
        zero = 0j if phases == 3 else 0.0
        self.state = Empty()
        self.inp = _ResistorInp(u=zero, i=zero)
        self.out = _ResistorOut(i=zero, u=zero, power=0.0)
        self.ports = (PowerPort("inp.u", "out.i", self.power_scale),)
        self.terminal = Terminal(self, "u", "i", 1)
        if series:
            self.ports += (PowerPort("out.u", "inp.i", self.power_scale, -1.0),)
            self.output_terminal = Terminal(self, "u", "i", -1, drives_voltage=True)
        self.output_stages = (
            OutputMethod("set_current", ("i",) if series else ("u",), ("i",)),
            OutputMethod("set_voltage", ("u", "i") if series else (), ("u",), rhs=series),
            OutputMethod("set_power", ("i",) if series else ("u",), ("power",), rhs=False),
        )
        if series:
            self._current_equation = self._set_series_current
            self._voltage_equation = self._set_series_voltage
            self._loss_equation = self._series_loss
        else:
            self._current_equation = self._set_shunt_current
            self._voltage_equation = self._set_shunt_voltage
            self._loss_equation = self._shunt_loss
        if type(self).set_current is _ResistorModel.set_current:
            self.set_current = self._current_equation
        if type(self).set_voltage is _ResistorModel.set_voltage:
            self.set_voltage = self._voltage_equation
        if type(self).dissipated_power is _ResistorModel.dissipated_power:
            self.dissipated_power = self._loss_equation

    def set_outputs(self, t: float) -> None:
        self.set_current(t)
        self.set_voltage(t)
        self.set_power(t)

    def set_current(self, t: float) -> None:
        """Evaluate the current equation selected when the resistor was constructed."""
        self._current_equation(t)

    def set_voltage(self, t: float) -> None:
        """Evaluate the voltage equation selected when the resistor was constructed."""
        self._voltage_equation(t)

    def _set_series_current(self, t: float) -> None:
        self.out.i = self.inp.i

    def _set_series_voltage(self, t: float) -> None:
        self.out.u = self.inp.u - self.resistance * self.inp.i

    def _set_shunt_current(self, t: float) -> None:
        self.out.i = self.inp.u / self.resistance

    def _set_shunt_voltage(self, t: float) -> None:
        self.out.u = 0.0

    def set_power(self, t: float) -> None:
        self.out.power = self.dissipated_power()

    def rhs(self, t: float):
        return ()

    def dissipated_power(self) -> float:
        """Evaluate the loss equation selected when the resistor was constructed."""
        return self._loss_equation()

    def _series_loss(self) -> float:
        return self.power_scale * self.resistance * abs(self.inp.i) ** 2

    def _shunt_loss(self) -> float:
        return self.power_scale * abs(self.inp.u) ** 2 / self.resistance

    def supplied_power(self) -> float:
        return 0.0


@register_element_type
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

    def terminals(self) -> tuple[tuple[str, Terminal], ...]:
        if self.cfg.series:
            return ((f"{self.name}.bus.in", self.resistor.terminal),
                    (self.cfg.bus, self.resistor.output_terminal))
        return ((self.cfg.bus, self.resistor.terminal),)

    def series_connections(self) -> dict[str, str]:
        return {self.cfg.bus: f"{self.name}.bus.in"} if self.cfg.series else {}

    def parameter_bindings(self) -> dict[str, tuple[Any, str]]:
        """Runtime scalar parameters, shared by Python retuning and native export."""
        return {"r": (self.resistor, "resistance")}

    def signals(self) -> dict[str, complex | float]:
        return {f"{self.name}.i": self.resistor.out.i,
                f"{self.name}.power": self.resistor.out.power}

    def retune(self, cfg: Any) -> None:
        self.cfg = cfg
        for name, (subsystem, attribute) in self.parameter_bindings().items():
            setattr(subsystem, attribute, getattr(cfg, name))


class _VoltageSourceInp(Bag):
    __slots__ = ("u",)


class _VoltageSourceOut(Bag):
    __slots__ = ("i",)


class _VoltageSourceModel:
    """Scalar voltage source behind a series resistance."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = True
    storage: ClassVar[tuple[StoragePort, ...]] = ()

    def __init__(self, voltage: float, resistance: float) -> None:
        self.voltage, self.resistance = voltage, resistance
        self.state = Empty()
        self.inp = _VoltageSourceInp(u=0.0)
        self.out = _VoltageSourceOut(i=0.0)
        self.ports = (PowerPort("inp.u", "out.i", 1.0),)
        self.terminal = Terminal(self, "u", "i", 1, electrical_domain="dc")

    def set_outputs(self, t: float) -> None:
        self.out.i = (self.inp.u - self.voltage) / self.resistance

    def rhs(self, t: float):
        return ()

    def dissipated_power(self) -> float:
        return self.resistance * self.out.i ** 2

    def supplied_power(self) -> float:
        return -self.voltage * self.out.i


@register_element_type
class VoltageSource(Element):
    """Scalar named-bus voltage source behind a positive series resistance."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "voltage_source"
        bus: str
        v: float
        r: float

        _quantities = {"v": "voltage", "r": "resistance"}

        def __post_init__(self) -> None:
            if not math.isfinite(self.v):
                raise ConfigError(f"v must be finite, got {self.v}")
            if not math.isfinite(self.r) or self.r <= 0.0:
                raise ConfigError(f"r must be finite and positive, got {self.r}")

    type = "voltage_source"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        if buses[cfg.bus].phases != 1:
            raise ConfigError("a scalar voltage source needs a single-phase bus")
        self.name, self.cfg = name, cfg
        self.source = _VoltageSourceModel(cfg.v, cfg.r)

    def subsystems(self) -> dict[str, Any]:
        return {self.name: self.source}

    @staticmethod
    def connections() -> dict:
        return {}

    def terminals(self) -> tuple[tuple[str, Terminal], ...]:
        return ((self.cfg.bus, self.source.terminal),)

    def signals(self) -> dict[str, float]:
        return {f"{self.name}.i": self.source.out.i}

    def retune(self, cfg: Any) -> None:
        self.cfg = cfg
        self.source.voltage, self.source.resistance = cfg.v, cfg.r
