"""Registry joining unit configuration types to assembly and validation behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..solver.model import ConfigError


@dataclass(frozen=True, kw_only=True)
class UnitTypeSpec:
    """All type-specific behavior selected by one ``units.<name>.type`` value."""

    name: str
    params: type | None = None
    default_bridge: str | Callable[[Any], str] | None = None
    children: tuple[str, ...] = ()
    default_port: str | None = None
    ports: tuple[str, ...] = ()
    phases: int
    bridge_types: frozenset[str] = frozenset()
    measurement_channels: frozenset[str] = frozenset()
    current_channel: str | None = None
    protection_criteria: frozenset[str] = frozenset()
    defaults_grid_frequency: bool = False
    validator: Callable[..., None]
    assembler: Callable[..., Any]

    @property
    def composite(self) -> bool:
        return bool(self.children)

    def bridge_for(self, control: Any) -> str:
        if callable(self.default_bridge):
            return self.default_bridge(control)
        return self.default_bridge


UNIT_TYPES: dict[str, UnitTypeSpec] = {}


def register_unit_type(name: str, **values: Any) -> UnitTypeSpec:
    """Register one complete unit profile; profiles cannot be patched later."""
    if name in UNIT_TYPES:
        raise ValueError(f"unit type {name!r} is already registered")
    try:
        spec = UnitTypeSpec(name=name, **values)
    except TypeError as exc:
        raise TypeError(f"invalid unit type {name!r}: {exc}") from None
    UNIT_TYPES[name] = spec
    return spec


def unit_type(name: str, where: str = "unit.type") -> UnitTypeSpec:
    try:
        return UNIT_TYPES[name]
    except KeyError:
        raise ConfigError(f"{where}: unknown unit type {name!r}; known: {sorted(UNIT_TYPES)}") from None
