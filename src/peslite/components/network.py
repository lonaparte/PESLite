"""Common electrical-component connection, registration and netlist translation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional

from ..solver.energy import Port, StoragePort
from ..solver.model import ConfigError, Empty, _complex, _real, _record_fields

__all__ = [
    "Port", "ElectricalEndpoint", "Element",
    "element_buses", "resolve_endpoints", "GROUND",
]

GROUND = "ground"


@dataclass(frozen=True)
class ElectricalEndpoint:
    """One resolved electrical connection target used only during assembly."""

    name: str
    target: Any | None
    phases: int

    @property
    def grounded(self) -> bool:
        return self.target is None

    @property
    def electrical_domain(self) -> Optional[str]:
        return None if self.target is None else getattr(self.target, "electrical_domain", None)


def resolve_endpoints(
    connections: Mapping[str, str], endpoints: Mapping[str, Any], where: str, *,
    same_phases: bool = False,
) -> dict[str, ElectricalEndpoint]:
    """Resolve bus and Unit-port names, treating ``ground`` as the shared zero reference.

    Ground has no phase representation of its own. Its representation is inferred from the
    component's non-ground terminals, so a component connected only to ground is invalid and a
    ground terminal beside mixed scalar and space-vector terminals is ambiguous.
    """
    targets: dict[str, Any | None] = {}
    for field_name, endpoint_name in connections.items():
        if endpoint_name == GROUND:
            targets[field_name] = None
            continue
        try:
            targets[field_name] = endpoints[endpoint_name]
        except KeyError:
            raise ConfigError(
                f"{where}.{field_name}: unknown electrical endpoint {endpoint_name!r}; "
                f"known: {sorted(endpoints)}"
            ) from None
    phases = {int(target.phases) for target in targets.values() if target is not None}
    if not phases:
        raise ConfigError(f"{where}: an electrical component cannot connect only to ground")
    if same_phases and len(phases) != 1:
        raise ConfigError(f"{where}: all electrical endpoints must have the same phase count")
    if any(target is None for target in targets.values()) and len(phases) != 1:
        raise ConfigError(
            f"{where}: ground phase representation is ambiguous beside mixed-phase endpoints"
        )
    ground_phases = next(iter(phases)) if len(phases) == 1 else 0
    return {
        field_name: ElectricalEndpoint(
            connections[field_name], target,
            ground_phases if target is None else int(target.phases),
        )
        for field_name, target in targets.items()
    }


class Element:
    """Common assembly and runtime boundary for a PESLite element.

    An element may directly define a physical ``network(builder)`` or expose one or more runtime
    subsystems through ``subsystems()``. The solver continues to recognize physical models by the
    structural ``Subsystem`` protocol rather than by this base class.
    """

    domain_transparent = False
    signal_fields: Mapping[str, str] = {}
    parameter_fields: Mapping[str, str] = {}
    structural_parameters: frozenset[str] = frozenset()
    equation_fields: Mapping[str, str] = {}
    infer_netlist_storage = False

    def set_parameters(self, **values: Any) -> bool:
        """Update declared netlist coefficients without changing its primitive structure."""
        unknown = set(values) - set(self.parameter_fields)
        if unknown:
            raise ValueError(f"unknown component parameters: {sorted(unknown)}")
        attributes = {name: self.parameter_fields[name] for name in values}
        old = {name: getattr(self, attribute) for name, attribute in attributes.items()}
        if any((old[name] is None) != (values[name] is None)
               for name in self.structural_parameters & values.keys()):
            raise ValueError("netlist elements cannot be added or removed after assembly")
        for name, attribute in attributes.items():
            setattr(self, attribute, values[name])
        return old != values

    def parameter_bindings(self) -> dict[str, tuple[Any, str]]:
        return {
            name: (self, attribute)
            for name, attribute in self.parameter_fields.items()
            if getattr(self, attribute) is not None
        }

    def equation_state_bindings(self) -> dict[str, tuple[Any, str]]:
        return {
            name: (self, attribute)
            for name, attribute in self.equation_fields.items()
        }

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

    def _bind_netlist_devices(self, netlist, devices) -> None:
        """Retain the netlist device handles installed by the common builder."""
        self._netlist_devices = netlist, tuple(devices)

    def _device_commutation_margin(self) -> float:
        from ..solver.netlist import device_commutation_margin
        return device_commutation_margin(self)

    def _device_commutate(self, t: float) -> None:
        from ..solver.netlist import device_commutate
        device_commutate(self, t)

    @classmethod
    def from_config(cls, name: str, cfg: Any, buses: dict, params: Any) -> "Element":
        """Build an element from its resolved configuration and assembly context."""
        return cls(name, cfg, buses, params)

    def subsystems(self) -> dict[str, Any]:
        """Expose the ordinary one-model element under its configured name."""
        return {self.name: self.model}

    @staticmethod
    def connections() -> dict:
        return {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return getattr(self, "assembled_port_bindings", ())

    def series_connections(self) -> dict[str, str]:
        return getattr(self, "assembled_series_connections", {})

    def output_bindings(self) -> dict[str, tuple[Any, str]]:
        """Expose the primary model's direct outputs under this element's public name."""
        model = getattr(self, "model", None)
        output = getattr(model, "output", None)
        return ({f"{self.name}.{field}": (model, field) for field in _record_fields(output)}
                if output is not None else {})

    def signals(self) -> dict[str, complex | float]:
        output = getattr(getattr(self, "model", self), "output", None)
        return ({f"{self.name}.{name}": getattr(output, field)
                 for name, field in self.signal_fields.items()}
                if output is not None else {})


Element.rhs._peslite_device_adapter = True
Element.set_outputs._peslite_device_adapter = True


def _phase_convention(phases: int, initial: complex | float = 0.0):
    """Power scale, zero and ``initial`` of peak complex space vectors (3) or real scalars (1)."""
    if phases not in (1, 3):
        raise ValueError(f"phases must be 1 or 3, got {phases!r}")
    if phases == 3:
        return 1.5, 0j, _complex(initial)
    return 1.0, 0.0, _real(initial.real if isinstance(initial, complex) else initial)


def element_buses(cfg: Any) -> dict[str, str]:
    """Return flat bus fields and additional declared element buses."""
    buses = {f.name: value for f in fields(cfg)
             if (value := getattr(cfg, f.name)) is not None
             if f.init and (f.name == "bus"
                            or (f.name.startswith("bus") and f.name[3:].isdigit()
                                and int(f.name[3:]) > 0))}
    additional = getattr(cfg, "connection_buses", None)
    if additional is not None:
        buses.update(additional())
    return buses


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

    def __getattr__(self, name: str):
        if name in self.component.port_by_field:
            return self.node(name)
        raise AttributeError(name)


class _DeclaredComponent(Element):
    """Runtime physical component generated from one declarative parameter object."""

    outputs_need_inputs = False
    infer_netlist_storage = True

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
        self.endpoint_by_field = resolve_endpoints(
            element_buses(config), buses, f"elements.{element.name}"
        )
        ground = None
        for field_name, endpoint in self.endpoint_by_field.items():
            phases = endpoint.phases
            scale, zero, _initial = _phase_convention(phases)
            name = "power" if field_name == "bus" else field_name[3:]
            if endpoint.grounded:
                if ground is None:
                    ground = Port(
                        self, GROUND, voltage=zero, current=zero, scale=scale, grounded=True,
                    )
                port = ground
            else:
                port = Port(
                    self, name, voltage=zero, current=zero, scale=scale,
                    electrical_domain=endpoint.electrical_domain,
                    standalone="voltage",
                )
            self.port_by_field[field_name] = port
            setattr(self, f"port_{field_name}", port)
        if not self.port_by_field:
            raise TypeError(f"{type(config).__name__} needs a bus or busN parameter")
        self.ports = tuple(dict.fromkeys(self.port_by_field.values()))

    def network(self, builder):
        network = _ComponentNetwork(self, builder)
        return self.element.cfg.network(network)


class _ConfiguredComponent(Element):
    """Assembly shell shared by every directly declared electrical component."""

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self.name, self.cfg = name, cfg
        self.domain_transparent = bool(getattr(type(cfg), "domain_transparent", False))
        self.model = _DeclaredComponent(self, buses)

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return tuple(
            (endpoint.name, self.model.port_by_field[field])
            for field, endpoint in self.model.endpoint_by_field.items()
            if not endpoint.grounded
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
