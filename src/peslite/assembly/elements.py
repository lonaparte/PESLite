"""Generic assembly adapters for registered simulation-file elements."""

from __future__ import annotations

from typing import Any

from ..components.cable import CABLE_MODELS
from ..components.network import element_buses, resolve_endpoints
from ..components.onnx import ONNXModel
from ..components.passive import Capacitor, ParallelRLC, Resistor, SeriesRLC
from ..components.registry import register_component
from ..components.source import DCVoltageSource
from ..solver.model import ConfigError
from .syntax import (
    CableParams,
    CapacitorParams,
    ONNXParams,
    ParallelRLCParams,
    ResistorParams,
    SeriesRLCParams,
    VoltageSourceParams,
)


class ElementAssembly:
    """Common registration boundary between typed syntax and a physical model."""

    def register(self, params: type, *, retunable: bool = True) -> "ElementAssembly":
        register_component(params, self.construct, self.retune if retunable else None)
        return self

    def construct(self, name: str, cfg: Any, buses: dict, params: Any) -> Any:
        raise NotImplementedError

    def retune(self, element: Any, cfg: Any) -> bool:
        raise NotImplementedError


class PassiveElementAssembly(ElementAssembly):
    """Assembly shared by passive models with the ordinary two-terminal contract."""

    def __init__(self, model: type) -> None:
        self.model = model

    def construct(self, name: str, cfg: Any, buses: dict, params: Any) -> Any:
        endpoints = resolve_endpoints(
            element_buses(cfg), buses, f"elements.{name}", same_phases=True,
        )
        phases = next(iter(endpoints.values())).phases
        path = f"elements.{name}.connected"
        breakable = any(
            path in change.connections for change in getattr(params, "changes", ())
        )
        element = self.model(
            **{
                attribute: getattr(cfg, parameter)
                for parameter, attribute in self.model.parameter_fields.items()
                if hasattr(cfg, parameter)
            },
            phases=phases,
            breakable=breakable,
        )
        element.name, element.cfg, element.endpoint_by_field = name, cfg, endpoints
        element.model = element
        element.breakers = [element] if element.breakable else []

        ports = {
            field: element.port1 if field in ("bus", "bus1") else element.port2
            for field in endpoints
        }
        for field, endpoint in endpoints.items():
            port = ports[field]
            if endpoint.grounded:
                port.grounded = True
            else:
                port.electrical_domain = endpoint.electrical_domain

        bus = getattr(cfg, "bus", None)
        if bus is not None and getattr(cfg, "series", False):
            inlet = f"{name}.bus.in"
            element.assembled_series_connections = {bus: inlet}
            element.assembled_port_bindings = ((inlet, element.port1), (bus, element.port2))
        else:
            element.assembled_series_connections = {}
            element.assembled_port_bindings = tuple(
                (endpoint.name, ports[field])
                for field, endpoint in endpoints.items() if not endpoint.grounded
            )
            if bus is not None:
                element.port2.grounded = True
        return element

    def retune(self, element: Any, cfg: Any) -> bool:
        changed = element.set_parameters(**{
            parameter: getattr(cfg, parameter)
            for parameter in element.parameter_fields if hasattr(cfg, parameter)
        })
        element.cfg = cfg
        return changed


class SourceElementAssembly(ElementAssembly):
    """Assembly shared by scalar source elements connected to one named endpoint."""

    def __init__(self, model: type) -> None:
        self.model = model

    def construct(self, name: str, cfg: Any, buses: dict, _params: Any) -> Any:
        if buses[cfg.bus].phases != 1:
            raise ConfigError("a scalar voltage source needs a single-phase bus")
        source = self.model(cfg.v, R=cfg.r)
        source.name, source.bus, source.cfg = name, cfg.bus, cfg
        return source

    def retune(self, element: Any, cfg: Any) -> bool:
        element.cfg = cfg
        return element.set_parameters(voltage=cfg.v, r=cfg.r, l=None)


class ModelElementAssembly(ElementAssembly):
    """Assembly for a model that already implements the common configuration boundary."""

    def __init__(self, model: type) -> None:
        self.model = model

    def construct(self, name: str, cfg: Any, buses: dict, params: Any) -> Any:
        return self.model.from_config(name, cfg, buses, params)

    def retune(self, element: Any, cfg: Any) -> bool:
        return element.retune(cfg)


for _params, _model in (
    (SeriesRLCParams, SeriesRLC),
    (CapacitorParams, Capacitor),
    (ResistorParams, Resistor),
    (ParallelRLCParams, ParallelRLC),
):
    PassiveElementAssembly(_model).register(_params)

SourceElementAssembly(DCVoltageSource).register(VoltageSourceParams)
ModelElementAssembly(ONNXModel).register(ONNXParams, retunable=False)
register_component(CableParams, CABLE_MODELS, selector="model")
