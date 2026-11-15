"""File-backed adapter from external magnetic data to the core magnetic component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...components.magnetic import (WindingParams, _MagneticElement, _connection_buses,
                                    _validate_windings)
from ...components.network import _register_assembled_type
from ...solver import ConfigError
from ..functions.magport import _load

__all__ = ["MagPort"]


@_register_assembled_type
class MagPort(_MagneticElement):
    """Connect one external MagPort file to ordinary PESLite winding terminals."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        windings: dict[str, WindingParams] = field(metadata={"structural": True})
        file: str = field(metadata={"asset": True, "runtime": False})
        type: str = "magport"

        def __post_init__(self) -> None:
            _validate_windings(self.windings)
            if not isinstance(self.file, str) or not self.file:
                raise ConfigError("file must be a nonempty path")

        def connection_buses(self) -> dict[str, str]:
            return _connection_buses(self.windings)

    type = "magport"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        data = _load(cfg.file)
        self._assemble(
            name, cfg, cfg.windings, data.winding_names, data.inductance, buses
        )
