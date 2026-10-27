"""Repeating sampled mode sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from . import MODE, ConfigError, Loop, register_loop_type

__all__ = ["ModeSequence"]


@register_loop_type
class ModeSequence(Loop):
    """Emit one entry of a repeating integer-mode pattern per loop update."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        modes: list = field(default_factory=lambda: [0])
        period: Optional[float] = None
        type: str = "mode_sequence"

        def __post_init__(self) -> None:
            if not self.modes:
                raise ConfigError("ModeSequence.modes must not be empty")
            invalid = [value for value in self.modes if type(value) is not int]
            if invalid:
                raise ConfigError(
                    f"ModeSequence.modes entries must be integers, got {invalid[0]!r}"
                )

    type = "mode_sequence"
    inputs = {}
    outputs = {"mode": MODE}
    state_names = ()

    def __init__(self, cfg: Any, unit: Any, startup: Any) -> None:
        super().__init__(cfg, unit, startup)
        if self.block_period is None:
            raise ConfigError("ModeSequence requires sampled control")
        # A one-hot cursor uses only ordinary scalar state arithmetic. The same sampled-loop
        # lowering therefore works in Python and generated C++ without a backend-specific
        # sequence implementation.
        self.state_names = tuple(f"position_{index}" for index in range(len(cfg.modes)))
        for index, name in enumerate(self.state_names):
            setattr(self, name, float(index == 0))

    def initial_outputs(self) -> dict[str, Any]:
        return {"mode": float(self.cfg.modes[0])}

    def sample(self, t: float, inputs: Mapping[str, Any]) -> dict[str, Any]:
        del t
        del inputs
        positions = [getattr(self, name) for name in self.state_names]
        mode = sum(value * position for value, position in zip(self.cfg.modes, positions))
        for name, position in zip(self.state_names, positions[-1:] + positions[:-1]):
            setattr(self, name, position)
        return {"mode": mode}
