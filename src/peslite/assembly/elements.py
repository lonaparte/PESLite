"""Configuration and assembly of the built-in simulation-file elements."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..components.network import element_buses, resolve_endpoints
from ..components.passive import Capacitor, ParallelRLC, Resistor, SeriesRLC
from ..components.registry import register_component
from ..solver.model import ConfigError


def validate_positive(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        raise ConfigError(f"{name} must be finite and > 0, got {value}")


def validate_connections(bus: str | None, bus1: str | None, bus2: str | None,
                         series: bool = False) -> None:
    one_terminal = bus is not None and bus1 is None and bus2 is None
    two_terminal = bus is None and bus1 is not None and bus2 is not None
    if not (one_terminal or two_terminal):
        raise ConfigError("specify either bus or both bus1 and bus2")
    if bus1 is not None and bus1 == bus2:
        raise ConfigError("bus1 and bus2 must be distinct")
    if series and not one_terminal:
        raise ConfigError("automatic series insertion requires bus")


@dataclass(frozen=True, kw_only=True)
class SeriesRLCParams:
    component_type: ClassVar[Any] = SeriesRLC
    type: str = "series_rlc"
    bus: str | None = None
    bus1: str | None = None
    bus2: str | None = None
    r: float = 0.0
    l: float | None = None
    c: float | None = None
    series: bool = field(default=False, metadata={"runtime": False})

    _quantities = {"r": "resistance", "l": "inductance", "c": "capacitance"}
    _input_aliases = {"x_pu": "l_pu"}

    def __post_init__(self) -> None:
        validate_connections(self.bus, self.bus1, self.bus2, self.series)
        if not math.isfinite(self.r) or self.r < 0.0:
            raise ConfigError(f"r must be finite and >= 0, got {self.r}")
        validate_positive("l", self.l)
        validate_positive("c", self.c)
        if self.r == 0.0 and self.l is None and self.c is None:
            raise ConfigError("a series RLC needs at least one nonzero element")
        if type(self.series) is not bool:
            raise ConfigError("series must be bool")


@dataclass(frozen=True, kw_only=True)
class CapacitorParams:
    component_type: ClassVar[Any] = Capacitor
    type: str = "capacitor"
    bus1: str
    bus2: str
    c: float
    r: float = 0.0

    _quantities = {"c": "capacitance", "r": "resistance"}

    def __post_init__(self) -> None:
        validate_connections(None, self.bus1, self.bus2)
        validate_positive("c", self.c)
        if not math.isfinite(self.r) or self.r < 0.0:
            raise ConfigError(f"r must be finite and >= 0, got {self.r}")


@dataclass(frozen=True, kw_only=True)
class ResistorParams:
    component_type: ClassVar[Any] = Resistor
    type: str = "resistor"
    bus: str | None = None
    bus1: str | None = None
    bus2: str | None = None
    r: float
    series: bool = field(default=False, metadata={"runtime": False})

    _quantities = {"r": "resistance"}

    def __post_init__(self) -> None:
        validate_connections(self.bus, self.bus1, self.bus2, self.series)
        validate_positive("r", self.r)
        if type(self.series) is not bool:
            raise ConfigError("series must be bool")


@dataclass(frozen=True, kw_only=True)
class ParallelRLCParams:
    component_type: ClassVar[Any] = ParallelRLC
    type: str = "parallel_rlc"
    bus: str | None = None
    bus1: str | None = None
    bus2: str | None = None
    r: float | None = None
    l: float | None = None
    c: float | None = None

    _quantities = {"r": "resistance", "l": "inductance", "c": "capacitance"}
    _input_aliases = {"x_pu": "l_pu"}

    def __post_init__(self) -> None:
        validate_connections(self.bus, self.bus1, self.bus2)
        validate_positive("r", self.r)
        validate_positive("l", self.l)
        validate_positive("c", self.c)
        if self.r is None and self.l is None and self.c is None:
            raise ConfigError("a parallel RLC needs at least one element")


def assemble_element(name: str, cfg: Any, buses: dict, params: Any) -> Any:
    """Construct one registered passive element and attach its resolved endpoints."""
    endpoints = resolve_endpoints(
        element_buses(cfg), buses, f"elements.{name}", same_phases=True,
    )
    phases = next(iter(endpoints.values())).phases
    path = f"elements.{name}.connected"
    breakable = any(path in change.connections for change in getattr(params, "changes", ()))
    element = cfg.component_type(
        **{
            attribute: getattr(cfg, name)
            for name, attribute in cfg.component_type.parameter_fields.items()
            if hasattr(cfg, name)
        },
        phases=phases,
        breakable=breakable,
    )
    element.name, element.cfg, element.endpoint_by_field = name, cfg, endpoints
    element.model = element
    element.breakers = [element] if element.breakable else []

    ports = {
        field_name: element.port1 if field_name in ("bus", "bus1") else element.port2
        for field_name in endpoints
    }
    for field_name, endpoint in endpoints.items():
        port = ports[field_name]
        if endpoint.grounded:
            port.grounded = True
        else:
            port.electrical_domain = endpoint.electrical_domain

    bus = getattr(cfg, "bus", None)
    series = bool(getattr(cfg, "series", False))
    if bus is not None and series:
        inlet = f"{name}.bus.in"
        element.assembled_series_connections = {bus: inlet}
        element.assembled_port_bindings = ((inlet, element.port1), (bus, element.port2))
    else:
        element.assembled_series_connections = {}
        element.assembled_port_bindings = tuple(
            (endpoint.name, ports[field_name])
            for field_name, endpoint in endpoints.items() if not endpoint.grounded
        )
        if bus is not None:
            element.port2.grounded = True
    return element


def retune_element(element: Any, cfg: Any) -> bool:
    """Apply the parameters declared by both the configuration and component."""
    changed = element.set_parameters(**{
        name: getattr(cfg, name)
        for name in element.parameter_fields if hasattr(cfg, name)
    })
    element.cfg = cfg
    return changed


for parameter_type in (SeriesRLCParams, CapacitorParams, ResistorParams, ParallelRLCParams):
    register_component(parameter_type, assemble_element, retune_element)
