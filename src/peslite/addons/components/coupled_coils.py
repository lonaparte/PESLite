"""Built-in linear coupled-coil specialization of the general magnetic component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...components.magnetic import (
    MagneticModel,
    WindingParams,
    _MagneticElement,
    _connection_buses,
    _validate_windings,
)
from ...components.network import _register_assembled_type

__all__ = ["CoupledCoils", "WindingParams"]


@_register_assembled_type
class CoupledCoils(_MagneticElement):
    """Concrete magnetic component defined by a constant winding-inductance matrix."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        windings: dict[str, WindingParams] = field(metadata={"structural": True})
        inductance: list[list[float]] = field(metadata={"structural": True})
        type: str = "coupled_coils"
        _model: MagneticModel = field(init=False, repr=False, compare=False)

        def __post_init__(self) -> None:
            names = _validate_windings(self.windings)
            model = MagneticModel.linear(names, self.inductance)
            object.__setattr__(self, "inductance", model.inductance.tolist())
            object.__setattr__(self, "_model", model)

        def connection_buses(self) -> dict[str, str]:
            return _connection_buses(self.windings)

    type = "coupled_coils"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self._assemble(name, cfg, cfg.windings, cfg._model, buses)
