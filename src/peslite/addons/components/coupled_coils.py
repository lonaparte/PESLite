"""Concise constant-matrix configuration of the general magnetic component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...components.magnetic import (
    WindingParams,
    _MagneticElement,
    _connection_buses,
    _inductance_matrix,
    _validate_windings,
)
from ...components.network import _register_assembled_type

__all__ = ["CoupledCoils", "WindingParams"]


@_register_assembled_type
class CoupledCoils(_MagneticElement):
    """Named windings using the general magnetic component's constant matrix model."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        windings: dict[str, WindingParams] = field(metadata={"structural": True})
        inductance: list[list[float]] = field(metadata={"structural": True})
        type: str = "coupled_coils"

        def __post_init__(self) -> None:
            names = _validate_windings(self.windings)
            object.__setattr__(
                self, "inductance", _inductance_matrix(self.inductance, len(names))
            )

        def connection_buses(self) -> dict[str, str]:
            return _connection_buses(self.windings)

    type = "coupled_coils"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self._assemble(name, cfg, cfg.windings, cfg.inductance, buses)
