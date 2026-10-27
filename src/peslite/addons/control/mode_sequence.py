"""Repeating sampled mode, width and phase sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ...control.profiles import PHASE_SHIFT_MODES
from . import ANGLE, MODE, MODULATION, ConfigError, Loop, register_loop_type

__all__ = ["ModeSequence"]


@register_loop_type
class ModeSequence(Loop):
    """Emit one entry of a repeating bridge-mode pattern per loop update."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        modes: list = field(default_factory=lambda: [0])
        widths: Optional[list] = None
        phases: Optional[list] = None
        period: Optional[float] = None
        type: str = "mode_sequence"

        def __post_init__(self) -> None:
            if not self.modes:
                raise ConfigError("ModeSequence.modes must not be empty")
            unknown = [value for value in self.modes
                       if type(value) is not int or value not in PHASE_SHIFT_MODES]
            if unknown:
                raise ConfigError(
                    f"ModeSequence.modes: unknown mode {unknown[0]!r}; "
                    f"known: {sorted(PHASE_SHIFT_MODES)}"
                )
            for name in ("widths", "phases"):
                values = getattr(self, name)
                if values is None:
                    continue
                if len(values) != len(self.modes):
                    raise ConfigError(f"ModeSequence.{name} must have one entry per mode")
                if any(value is not None
                       and (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value)) for value in values):
                    raise ConfigError(f"ModeSequence.{name} entries must be finite or null")
            if self.widths is not None and any(
                    value is not None and not 0.0 <= float(value) <= 1.0
                    for value in self.widths):
                raise ConfigError("ModeSequence.widths entries must be in [0, 1]")

    type = "mode_sequence"
    inputs = {"width": MODULATION, "phase": ANGLE}
    outputs = {"mode": MODE, "width": MODULATION, "phase": ANGLE}
    state_names = ("index",)
    native_export = False

    def __init__(self, cfg: Any, unit: Any, startup: Any) -> None:
        super().__init__(cfg, unit, startup)
        if self.block_period is None:
            raise ConfigError("ModeSequence requires sampled control")
        self.index = 0

    def initial_outputs(self) -> dict[str, Any]:
        references = self.unit.ctrl.references
        return {"mode": float(self.cfg.modes[self.index]), "width": 0.0,
                "phase": references.phase}

    def sample(self, t: float, inputs: Mapping[str, Any]) -> dict[str, Any]:
        del t
        index = self.index
        width = inputs["width"]
        if self.cfg.widths is not None and self.cfg.widths[index] is not None:
            width = float(self.cfg.widths[index])
        phase = inputs["phase"]
        if self.cfg.phases is not None and self.cfg.phases[index] is not None:
            phase += float(self.cfg.phases[index])
        self.index = (index + 1) % len(self.cfg.modes)
        return {"mode": float(self.cfg.modes[index]), "width": width, "phase": phase}

    def get_state(self) -> dict[str, Any]:
        return {"index": self.index}

    def set_state(self, values: Mapping[str, Any]) -> None:
        unknown = set(values) - {"index"}
        if unknown:
            raise KeyError(f"ModeSequence has no state(s) {sorted(unknown)}")
        if "index" in values:
            self.index = int(round(float(values["index"]))) % len(self.cfg.modes)
