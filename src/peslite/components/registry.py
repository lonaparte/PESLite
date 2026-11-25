"""Type registries owned by the built-in component library."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, Callable

from ..solver.model import ConfigError

__all__ = [
    "ComponentDefinition", "ELEMENT_TYPES", "register_component_type",
    "TOPOLOGY_TYPES", "register_topology", "topology_type",
    "MODULATOR_TYPES", "register_modulator_type",
]


@dataclass(frozen=True)
class ComponentDefinition:
    """One normalized configuration type and its assembly-time constructor."""

    type: str
    params: type
    constructor: Callable[[str, Any, dict, Any], Any]

    def __call__(self, name: str, cfg: Any, buses: dict, p: Any) -> Any:
        return self.constructor(name, cfg, buses, p)


ELEMENT_TYPES: dict[str, ComponentDefinition] = {}
TOPOLOGY_TYPES: dict[str, type] = {}
MODULATOR_TYPES: dict[str, type] = {}


def _register(registry: dict[str, Any], name: str, value: Any, label: str) -> Any:
    if not isinstance(name, str) or not name:
        raise TypeError(f"a {label} type needs a non-empty name")
    if name in registry:
        raise ValueError(f"{label} type {name!r} is already registered")
    registry[name] = value
    return value


def _register_component(cls: type, params: type, constructor: Callable) -> type:
    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not name:
        raise TypeError("a component type needs a name: the class attribute 'type'")
    if not is_dataclass(params) or "type" not in params.__dataclass_fields__:
        raise TypeError(f"component type {name!r}: parameters need a dataclass type field")
    if params.__dataclass_fields__["type"].default != name:
        raise TypeError(f"component type {name!r}: type must default to {name!r}")
    _register(ELEMENT_TYPES, name, ComponentDefinition(name, params, constructor), "component")
    return cls


def register_component_type(cls: type) -> type:
    """Register a declarative component or a complete ``Element`` adapter."""
    from .network import Element, _ConfiguredComponent

    if isinstance(cls, type) and issubclass(cls, Element):
        return _register_component(cls, getattr(cls, "Params", None), cls)
    if is_dataclass(cls):
        if not cls.__dataclass_params__.frozen:
            raise TypeError("a declarative component parameter dataclass must be frozen")
        if not callable(getattr(cls, "network", None)):
            raise TypeError("a declarative component type needs network(builder)")
        return _register_component(
            cls, cls,
            lambda name, cfg, buses, params: _ConfiguredComponent(name, cfg, buses, params),
        )
    raise TypeError("a component type must be a frozen parameter dataclass or Element adapter")


def register_topology(cls: type) -> type:
    """Register one bridge topology under its configuration name."""
    name = getattr(cls, "type", None)
    return _register(TOPOLOGY_TYPES, name, cls, "bridge topology")


def topology_type(name: str, where: str = "bridge.type") -> type:
    """Return a registered bridge topology class by configuration name."""
    try:
        return TOPOLOGY_TYPES[name]
    except KeyError:
        raise ConfigError(
            f"{where}: unknown bridge topology {name!r}; known: {sorted(TOPOLOGY_TYPES)}"
        ) from None


def register_modulator_type(name: str, cls: type) -> type:
    """Register one named converter modulator implementation."""
    return _register(MODULATOR_TYPES, name, cls, "modulator")
