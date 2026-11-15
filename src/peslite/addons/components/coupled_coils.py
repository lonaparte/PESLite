"""Built-in linear coupled-coil specialization of the general magnetic component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...components.magnetic import (
    MagneticParams,
    WindingParams,
    _MagneticElement,
    _inductance_decomposition,
)
from ...components.network import register_component_type

__all__ = ["CoupledCoils", "WindingParams"]


@register_component_type
class CoupledCoils(_MagneticElement):
    """Concrete magnetic component defined by a constant winding-inductance matrix."""

    @dataclass(frozen=True, kw_only=True)
    class Params(MagneticParams):
        inductance: list[list[float]] = field(metadata={"structural": True})
        type: str = "coupled_coils"

        def __post_init__(self) -> None:
            super().__post_init__()
            names = tuple(self.windings)
            matrix, _modes, _shapes = _inductance_decomposition(
                self.inductance, len(names)
            )
            object.__setattr__(self, "inductance", matrix.tolist())

    type = "coupled_coils"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self._assemble(name, cfg, cfg.windings, tuple(cfg.windings), cfg.inductance, buses)
