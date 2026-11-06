"""AC network blocks and registered circuit-element types.

Quantities are SI; ac quantities are peak-scaled space vectors. Breakers expose
``open_breaker()`` and ``close_breaker()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable, ClassVar, Optional

import numpy as np

from ..solver.energy import Port, StoragePort
from ..solver.model import Bag, ConfigError, Empty, OutputMethod, _complex, _real

__all__ = [
    "Port", "ACVoltageSource", "Bus", "RLBranch", "RCNode", "register_component_type",
    "element_buses", "Load", "Resistor", "VoltageSource",
]


class NetworkComponent:
    """Common runtime adapter for a physical component declared by ``network(builder)``."""

    def set_outputs(self, t: float) -> None:
        from ..solver.netlist import device_outputs
        device_outputs(self, t)

    def rhs(self, t: float):
        from ..solver.netlist import device_rhs
        return device_rhs(self, t)

    def dissipated_power(self) -> float:
        if getattr(self, "_network_has_losses", None) is False:
            return 0.0
        from ..solver.netlist import device_dissipated_power
        return device_dissipated_power(self)

    def supplied_power(self) -> float:
        if getattr(self, "_network_has_supplies", None) is False:
            return 0.0
        from ..solver.netlist import device_supplied_power
        return device_supplied_power(self)


NetworkComponent.rhs._peslite_device_adapter = True
NetworkComponent.set_outputs._peslite_device_adapter = True


def _phase_convention(phases: int, initial: complex | float = 0.0):
    """Power scale, zero and ``initial`` of peak complex space vectors (3) or real scalars (1)."""
    if phases not in (1, 3):
        raise ValueError(f"phases must be 1 or 3, got {phases!r}")
    if phases == 3:
        return 1.5, 0j, _complex(initial)
    return 1.0, 0.0, _real(initial.real if isinstance(initial, complex) else initial)


def _node_kind(phases: int) -> tuple[int, float]:
    """Return the netlist node dimension and power scale for a phase convention."""
    return (2, 1.5) if phases == 3 else (1, 1.0)


def _extend_record(record: Bag, *names: str) -> Bag:
    """Copy ``record`` into a record type with the additional fields ``names``."""
    fields = type(record).__slots__
    return type(type(record).__name__, (Bag,), {"__slots__": (*fields, *names)})(
        **{name: getattr(record, name) for name in fields})


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


class _SourceOutput(Bag):
    __slots__ = ("e_g", "phi", "theta")


class ACVoltageSource(NetworkComponent):
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
        self.power_scale, zero, _ = _phase_convention(phases)
        self.w0, self.e_peak, self.phases = w0, e_peak, phases
        self.phi = phi
        self.state = Empty()
        self.input = Empty()
        self.output = _SourceOutput()
        self.output.e_g = zero
        self.port = Port(
            self, "power", voltage=zero, current=zero,
            scale=self.power_scale, electrical_domain="ac",
        )
        self.ports = (self.port,)
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
        out = self.output
        out.phi = phi
        out.theta = theta
        out.e_g = self.e_peak * math.cos(theta)

    def _set_three_phase_outputs(self, t: float) -> None:
        phi = self.phi(t) if self.phi is not None else 0.0
        theta = self.w0 * t + phi
        out = self.output
        out.phi = phi
        out.theta = theta
        out.e_g = complex(self.e_peak * math.cos(theta), self.e_peak * math.sin(theta))

    @classmethod
    def uses_standard_outputs(cls, method: Any) -> bool:
        """Whether ``method`` is one of the built-in phase-specialized source equations."""
        return getattr(method, "__func__", None) in (
            cls.set_outputs, cls._set_single_phase_outputs, cls._set_three_phase_outputs,
        )

    def cpp_output_stage(self, generator: Any, method: Any):
        """Use the configured native source scenario for the built-in waveform."""
        return (generator.source_output_lines(self)
                if self.uses_standard_outputs(method) else None)

    def rhs(self, t: float):
        return ()

    def voltage(self, t: float) -> complex | float:
        """Evaluate and return the source's selected output equation."""
        self._output_equation(t)
        return self.output.e_g

    def network(self, nb):
        """Declare an ideal voltage source from the port to ground."""
        from ..solver import netlist as nl

        nb.add(nl.VoltageSource(
            nb.node(self.port), None, nb.value(self.output, "e_g")
        ))
        nb.supplies(self)


# ------------------------------------------------------------------ circuit element types

class Element:
    """Base of a registered ``elements`` entry, built as ``cls(name, cfg, buses, p)``.

    A type defines ``type`` and a frozen ``Params`` dataclass with a matching ``type`` field.
    An instance supplies ``subsystems()``, ``connections()`` and ``port_bindings()``. A connection
    source is another subsystem output or a configured physical-output path resolved by assembly.
    Each binding joins a configured bus name to a :class:`Port`; one- and multi-port
    elements use the same interface. Optional ``breakers`` or ``connect()`` enable switching;
    optional ``retune()`` enables runtime parameter changes and returns whether its network
    declaration changed. Optional ``series_connections()`` maps a bus to an inlet bus: assembly
    moves the original voltage driver to that inlet, inserting a two-port element before the bus's
    followers without adding an artificial energy-storage state.
    Optional ``parameter_bindings()`` maps retunable scalar fields to ``(subsystem, attribute)``
    pairs so native export uses the same parameters as Python, including energy equations.
    """

    type: ClassVar[str]
    Params: ClassVar[type]
    domain_transparent: ClassVar[bool] = False


@dataclass(frozen=True)
class ComponentDefinition:
    """One normalized configuration type and its assembly-time constructor."""

    type: str
    params: type
    constructor: Callable[[str, Any, dict, Any], Element]

    def __call__(self, name: str, cfg: Any, buses: dict, p: Any) -> Element:
        return self.constructor(name, cfg, buses, p)


ELEMENT_TYPES: dict[str, ComponentDefinition] = {}


def _register_definition(cls: type, params: type, constructor: Callable) -> type:
    """Store one normalized component definition and return its public class."""
    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not name:
        raise TypeError("a component type needs a name: the class attribute 'type'")
    if name in ELEMENT_TYPES:
        raise ValueError(f"component type {name!r} is already registered")
    if not is_dataclass(params) or "type" not in params.__dataclass_fields__:
        raise TypeError(f"component type {name!r}: parameters need a dataclass type field")
    if params.__dataclass_fields__["type"].default != name:
        raise TypeError(f"component type {name!r}: type must default to {name!r}")
    ELEMENT_TYPES[name] = ComponentDefinition(name, params, constructor)
    return cls


def _register_assembled_type(cls: type) -> type:
    """Register a package-owned assembly adapter."""
    return _register_definition(cls, getattr(cls, "Params", None), cls)


def element_buses(cfg: Any) -> dict[str, str]:
    """Return fields named ``bus`` or ``busN`` from an element's parameters."""
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)
            if f.init and (f.name == "bus"
                           or (f.name.startswith("bus") and f.name[3:].isdigit()
                               and int(f.name[3:]) > 0))}


_DEFAULT_STATE = object()


class _ComponentNetwork:
    """Owner-bound façade used while a declarative component becomes a ``NetworkPlan``."""

    def __init__(self, component: "_DeclaredComponent", builder: Any) -> None:
        self.component, self.builder = component, builder
        self.ground = None

    def _port(self, value: str | Port) -> Port:
        if isinstance(value, Port):
            return value
        try:
            return self.component.port_by_field[value]
        except KeyError:
            raise ValueError(f"unknown component port {value!r}") from None

    def _kind(self, like: Any = None) -> tuple[int, float]:
        if isinstance(like, str):
            like = self.builder.node(self._port(like))
        if isinstance(like, Port):
            like = self.builder.node(like)
        if like is not None:
            return like.dim, like.scale
        kinds = {
            (2 if isinstance(port.voltage, complex) else 1, port.scale)
            for port in self.component.port_by_field.values()
        }
        if len(kinds) != 1:
            raise ValueError("a mixed-dimension component must specify a reference port")
        return next(iter(kinds))

    def node(self, port: str | Port):
        return self.builder.node(self._port(port))

    def internal(self, like: Any = None):
        return self.builder.internal(*self._kind(like))

    def state(self, name: str, initial: Any = _DEFAULT_STATE, *, like: Any = None):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"state name must be an identifier, got {name!r}")
        dimension, _scale = self._kind(like)
        if name not in self.component.state_names:
            value = (0j if dimension == 2 else 0.0) if initial is _DEFAULT_STATE else initial
            if dimension == 2:
                value = complex(value)
            elif isinstance(value, complex) and value.imag:
                raise ValueError(f"scalar state {name!r} cannot have a complex initial value")
            else:
                value = float(value.real if isinstance(value, complex) else value)
            setattr(self.component.state, name, value)
            self.component.state_names += (name,)
            self.component.state_dimensions[name] = dimension
        elif self.component.state_dimensions[name] != dimension:
            raise ValueError(f"state {name!r} was declared with different dimensions")
        return self.builder.state(self.component, name)

    @staticmethod
    def _quantity_zero(quantity: tuple) -> complex | float:
        target = quantity[1]
        nodes = [node for port in getattr(target, "ports", ()) for node in port if node is not None]
        return 0j if nodes and nodes[0].dim == 2 else 0.0

    def _output(self, name: str, initial: complex | float) -> None:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"output name must be an identifier, got {name!r}")
        if not hasattr(self.component.output, name):
            setattr(self.component.output, name, initial)

    def output(self, name: str, quantity: tuple) -> None:
        self._output(name, self._quantity_zero(quantity))
        self.builder.output(self.component, name, quantity)

    def linear_output(self, name: str, terms) -> None:
        terms = tuple(terms)
        initial = self._quantity_zero(terms[0][1]) if terms else 0.0
        self._output(name, initial)
        self.builder.linear_output(self.component, name, terms)

    def port_output(self, name: str, port: str | Port, quantity: str,
                    factor: float = 1.0) -> None:
        physical = self._port(port)
        initial = physical.voltage if quantity == "voltage" else physical.current
        self._output(name, initial)
        self.builder.port_output(self.component, name, physical, quantity, factor)

    def node_current_output(self, name: str, node: Any, *, exclude=()) -> None:
        self._output(name, 0j if node is not None and node.dim == 2 else 0.0)
        self.builder.node_current_output(
            self.component, name, node, exclude=exclude
        )

    def value(self, record: Any, field: str, factor: float = 1.0, *, invert: bool = False):
        return self.builder.value(record, field, factor, invert=invert)

    def function(self, callback: Callable, inputs=(), states=()):
        return self.builder.function(callback, inputs, states)

    def add(self, *elements):
        return self.builder.add(*elements)

    def losses(self) -> None:
        self.builder.losses(self.component)

    def supplies(self) -> None:
        self.component.has_source = True
        self.builder.supplies(self.component)

    def bind(self, callback: Callable, *objects: Any) -> None:
        self.builder.bind(callback, *objects)

    def mixture(self, cases: Callable, select: Callable) -> None:
        self.builder.mixture(cases, select)

    def commutation(self, **kwargs: Any) -> None:
        self.builder.commutation(self.component, **kwargs)

    def finish(self) -> None:
        """Infer ordinary scalar storage declarations from the primitives just emitted."""
        from ..solver import netlist as nl

        storage = []
        state_indices = {
            name: tuple(int(index) for index in indices)
            for (owner, name), indices in self.builder._states.items()
            if owner == id(self.component)
        }
        for primitive in self.builder.owned.get(id(self.component), ()):
            attribute = "L" if isinstance(primitive, nl.Inductor) else (
                "C" if isinstance(primitive, nl.Capacitor) else None
            )
            if attribute is None:
                continue
            indices = tuple(int(index) for index in primitive.state)
            names = [name for name, state in state_indices.items() if state == indices]
            matrix = getattr(primitive, attribute)
            diagonal = np.diag(matrix)
            if (len(names) != 1 or not np.allclose(matrix, np.diag(diagonal))
                    or not np.allclose(diagonal, diagonal[0])):
                continue
            nodes = [node for port in primitive.ports for node in port if node is not None]
            storage.append(StoragePort(
                names[0], float(diagonal[0]), nodes[0].scale if nodes else 1.0,
                kind="inductor" if attribute == "L" else "capacitor",
            ))
        self.component.storage = tuple(storage)

    def __getattr__(self, name: str):
        if name in self.component.port_by_field:
            return self.node(name)
        raise AttributeError(name)


class _DeclaredComponent(NetworkComponent):
    """Runtime physical component generated from one declarative parameter object."""

    outputs_need_inputs = False

    def __init__(self, element: "_ConfiguredComponent", buses: dict[str, Any]) -> None:
        self.element = element
        config = element.cfg
        self.state = SimpleNamespace()
        self.input = Empty()
        self.output = SimpleNamespace()
        self.state_names: tuple[str, ...] = ()
        self.state_dimensions: dict[str, int] = {}
        self.storage: tuple[StoragePort, ...] = ()
        self.has_source = False
        self.port_by_field: dict[str, Port] = {}
        for field_name, bus_name in element_buses(config).items():
            bus = buses[bus_name]
            phases = int(bus.phases)
            scale, zero, _initial = _phase_convention(phases)
            name = "power" if field_name == "bus" else field_name[3:]
            port = Port(
                self, name, voltage=zero, current=zero, scale=scale,
                electrical_domain=getattr(bus, "electrical_domain", None),
                standalone="voltage",
            )
            self.port_by_field[field_name] = port
            setattr(self, f"port_{field_name}", port)
        if not self.port_by_field:
            raise TypeError(f"{type(config).__name__} needs a bus or busN parameter")
        self.ports = tuple(self.port_by_field.values())

    def network(self, builder):
        network = _ComponentNetwork(self, builder)
        result = self.element.cfg.network(network)
        network.finish()
        return result


class _ConfiguredComponent(Element):
    """Assembly shell shared by every directly declared electrical component."""

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self.name, self.cfg = name, cfg
        self.domain_transparent = bool(getattr(type(cfg), "domain_transparent", False))
        self.model = _DeclaredComponent(self, buses)

    def subsystems(self) -> dict[str, Any]:
        return {self.name: self.model}

    @staticmethod
    def connections() -> dict:
        return {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return tuple(
            (bus, self.model.port_by_field[field])
            for field, bus in element_buses(self.cfg).items()
        )

    def signals(self) -> dict[str, complex | float]:
        outputs = {
            f"{self.name}.{field}": value
            for field, value in vars(self.model.output).items()
        }
        one = len(self.model.port_by_field) == 1
        for field, port in self.model.port_by_field.items():
            prefix = self.name if one else f"{self.name}.{field}"
            outputs[f"{prefix}.voltage"] = port.voltage
            outputs[f"{prefix}.current"] = port.current
        return outputs

    def retune(self, cfg: Any) -> bool:
        changed = self.cfg != cfg
        self.cfg = cfg
        return changed


def register_component_type(cls: type) -> type:
    """Register a frozen electrical parameter dataclass declaring ``network(builder)``."""
    if not is_dataclass(cls):
        raise TypeError("a declarative component type must be a dataclass")
    if not cls.__dataclass_params__.frozen:
        raise TypeError("a declarative component parameter dataclass must be frozen")
    if not callable(getattr(cls, "network", None)):
        raise TypeError("a declarative component type needs network(builder)")
    return _register_definition(
        cls,
        cls,
        lambda name, cfg, buses, _params: _ConfiguredComponent(name, cfg, buses, _params),
    )


@_register_assembled_type
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


@_register_assembled_type
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


class _VoltageSourceOutput(Bag):
    __slots__ = ("i",)


class _VoltageSourceModel(NetworkComponent):
    """Scalar voltage source behind a series resistance."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    storage: ClassVar[tuple[StoragePort, ...]] = ()

    def __init__(self, voltage: float, resistance: float) -> None:
        self.voltage, self.resistance = voltage, resistance
        self.state = Empty()
        self.input = Empty()
        self.output = _VoltageSourceOutput(i=0.0)
        self.port = Port(
            self, "power", voltage=0.0, current=0.0,
            electrical_domain="dc"
        )
        self.ports = (self.port,)

    def network(self, nb):
        """Declare the source and its series resistance."""
        from ..solver import netlist as nl

        inner = nb.internal()
        nb.add(nl.Resistor(nb.node(self.port), inner, self.resistance))
        nb.add(nl.VoltageSource(
            inner, None, nb.function(lambda _t: self.voltage)
        ))
        nb.port_output(self, "i", self.port, "current")
        nb.losses(self)
        nb.supplies(self)


@_register_assembled_type
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

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return ((self.cfg.bus, self.source.port),)

    def signals(self) -> dict[str, float]:
        return {f"{self.name}.i": self.source.output.i}

    def retune(self, cfg: Any) -> bool:
        changed = self.cfg.r != cfg.r
        self.cfg = cfg
        self.source.voltage, self.source.resistance = cfg.v, cfg.r
        return changed
