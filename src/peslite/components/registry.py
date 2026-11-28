"""Type registries owned by the built-in component library."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from typing import Any, Callable

from ..solver.model import ConfigError

__all__ = [
    "ComponentModel", "ComponentDefinition", "ELEMENT_TYPES",
    "register_component", "register_component_type",
    "TOPOLOGY_TYPES", "register_topology", "topology_type",
    "MODULATOR_TYPES", "register_modulator_type",
]


@dataclass(frozen=True)
class ComponentModel:
    """One construction and update implementation selected at assembly boundaries."""

    constructor: Callable[[str, Any, dict, Any], Any]
    retuner: Callable[[Any, Any], bool] | None = None


@dataclass(frozen=True)
class ComponentDefinition:
    """One parameter type with either one bound model or selectable model variants."""

    type: str
    params: type
    default: ComponentModel | None
    models: Mapping[str, ComponentModel]
    selector: str | None = None

    def model(self, cfg: Any) -> ComponentModel:
        if self.default is not None:
            return self.default
        name = getattr(cfg, self.selector)
        try:
            return self.models[name]
        except KeyError:
            raise ConfigError(
                f"{self.type}.{self.selector}: unknown model {name!r}; "
                f"known: {sorted(self.models)}"
            ) from None

    def __call__(self, name: str, cfg: Any, buses: dict, p: Any) -> Any:
        return self.model(cfg).constructor(name, cfg, buses, p)

    def retune(self, element: Any, cfg: Any) -> bool:
        if self.default is not None and self.default.retuner is not None:
            return self.default.retuner(element, cfg)
        return element.retune(cfg)

    def supports_retune(self, element: Any) -> bool:
        return ((self.default is not None and self.default.retuner is not None)
                or callable(getattr(element, "retune", None)))


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


def _register_component(name: str, params: type,
                        implementation: ComponentModel | Mapping[str, type | ComponentModel],
                        selector: str | None = None) -> None:
    if not isinstance(name, str) or not name:
        raise TypeError("a component type needs a name: the class attribute 'type'")
    if not is_dataclass(params) or "type" not in params.__dataclass_fields__:
        raise TypeError(f"component type {name!r}: parameters need a dataclass type field")
    if params.__dataclass_fields__["type"].default != name:
        raise TypeError(f"component type {name!r}: type must default to {name!r}")
    if isinstance(implementation, ComponentModel):
        if selector is not None:
            raise TypeError(f"component type {name!r}: a single model has no selector")
        default, models = implementation, {}
    else:
        if not implementation:
            raise TypeError(f"component type {name!r}: models cannot be empty")
        models = {}
        for model_name, model in implementation.items():
            if not isinstance(model_name, str) or not model_name:
                raise TypeError(f"component type {name!r}: model names must be nonempty strings")
            if isinstance(model, ComponentModel):
                models[model_name] = model
                continue
            constructor = getattr(model, "from_config", None)
            if not isinstance(model, type) or not callable(constructor):
                raise TypeError(
                    f"component type {name!r}: model {model_name!r} needs from_config()"
                )
            models[model_name] = ComponentModel(constructor)
        if any(model.retuner is not None for model in models.values()):
            raise TypeError(
                f"component type {name!r}: selectable models update through their instance"
            )
        if selector not in params.__dataclass_fields__:
            raise TypeError(f"component type {name!r}: unknown model selector {selector!r}")
        selected = params.__dataclass_fields__[selector].default
        if selected not in models:
            raise TypeError(
                f"component type {name!r}: default model {selected!r} is not registered"
            )
        default = None
    _register(
        ELEMENT_TYPES, name,
        ComponentDefinition(name, params, default, models, selector),
        "component",
    )


def register_component(params: type,
                       constructor: Callable | Mapping[str, type | ComponentModel],
                       retuner: Callable | None = None, *, selector: str | None = None) -> None:
    """Register one fixed model or a set of assembly-time selectable models."""
    type_field = getattr(params, "__dataclass_fields__", {}).get("type")
    name = None if type_field is None else type_field.default
    if isinstance(constructor, Mapping):
        implementation = constructor
    elif isinstance(constructor, type):
        if retuner is not None:
            raise TypeError(f"component type {name!r}: model classes update through their instance")
        from_config = getattr(constructor, "from_config", None)
        if not callable(from_config):
            raise TypeError(f"component type {name!r}: model class needs from_config()")
        implementation = ComponentModel(from_config)
    else:
        implementation = ComponentModel(constructor, retuner)
    _register_component(name, params, implementation, selector)


def register_component_type(cls: type) -> type:
    """Register a declarative component or a complete ``Element`` adapter."""
    from .network import Element, _ConfiguredComponent

    if isinstance(cls, type) and issubclass(cls, Element):
        _register_component(
            getattr(cls, "type", None), getattr(cls, "Params", None),
            ComponentModel(cls.from_config),
        )
        return cls
    if is_dataclass(cls):
        if not cls.__dataclass_params__.frozen:
            raise TypeError("a declarative component parameter dataclass must be frozen")
        if not callable(getattr(cls, "network", None)):
            raise TypeError("a declarative component type needs network(builder)")
        _register_component(
            getattr(cls, "type", None), cls,
            ComponentModel(
                lambda name, cfg, buses, params: _ConfiguredComponent(name, cfg, buses, params)
            ),
        )
        return cls
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
