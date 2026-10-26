"""Registry for the configuration and runtime boundary of each control type."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

from ..solver.model import ConfigError

if TYPE_CHECKING:
    from .controller import ControlInterface


@dataclass(frozen=True)
class ReferenceParams:
    """VSC controller references in the unit's per-unit bases."""

    id_ref_pu: float = 0.0
    iq_ref_pu: float = 0.0
    p_ref_pu: float = 0.0
    q_ref_pu: float = 0.0
    v_ref_pu: float = 1.0
    vdc_ref_pu: float = 1.0
    theta: float = 0.0
    omega: Optional[float] = None
    zero_v_pu: float = 0.0

    _quantities = {"id_ref_pu": "current", "iq_ref_pu": "current",
                   "p_ref_pu": "power", "q_ref_pu": "power",
                   "v_ref_pu": "voltage", "vdc_ref_pu": "dc_voltage",
                   "zero_v_pu": "voltage"}


@dataclass(frozen=True)
class ScalarVoltageReferenceParams:
    """Output-voltage reference in volts for scalar converter control."""

    voltage: float


@dataclass(frozen=True)
class ResonantReferenceParams:
    """Fundamental commands and an optional signed port-power reference (W)."""

    modulation: float
    frequency: float
    phase: float = 0.0
    power: float = 0.0


@dataclass(frozen=True)
class PhaseShiftReferenceParams:
    """Square-wave commands: pulse ``width`` in half periods, ``frequency`` (Hz), ``phase``
    (rad, centre of the positive pulse), ``mode`` (0 bipolar pulses, 1 positive pulse,
    3 zero level, 4 negative pulse) and an optional signed port-power reference (W)."""

    width: float
    frequency: float
    phase: float = 0.0
    power: float = 0.0
    mode: int = 0


@dataclass(frozen=True)
class ControlTypeSpec:
    """Behavior selected by one ``ctrl.type`` value."""

    name: str
    references: type | None = None
    unit_types: frozenset[str] = field(default_factory=frozenset)
    interface: ControlInterface | None = None
    wiring: Callable[..., Any] | None = None
    log_sources: Mapping[str, str] | None = None
    grid_following: bool = False
    default_bridge: str | None = None
    controller_factory: Callable[..., Any] | None = None
    reference_validator: Callable[[Any, str], None] | None = None
    # Registered modulator (components.pwm.MODULATOR_TYPES) of a sampled bridge commanded by
    # this type's outputs; None: the carrier modulation of duty ratios.
    modulator: str | None = None


def validate_scalar_voltage_references(references: Any, where: str) -> None:
    if not math.isfinite(references.voltage) or references.voltage <= 0.0:
        raise ConfigError(f"{where}.voltage must be finite and positive")


def _validate_modulation_references(references: Any, where: str, amplitude: str) -> None:
    value = getattr(references, amplitude)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ConfigError(f"{where}.{amplitude} must be in [0, 1]")
    if not math.isfinite(references.frequency) or references.frequency <= 0.0:
        raise ConfigError(f"{where}.frequency must be finite and positive")
    for name in ("phase", "power"):
        if not math.isfinite(getattr(references, name)):
            raise ConfigError(f"{where}.{name} must be finite")


def validate_resonant_references(references: Any, where: str) -> None:
    _validate_modulation_references(references, where, "modulation")


# Each mode defines its pulses as (centre offset in rad, signed bridge level).
PHASE_SHIFT_MODES = {
    0: ((0.0, 1), (math.pi, -1)),
    1: ((0.0, 1),),
    3: (),
    4: ((math.pi, -1),),
}


def validate_phase_shift_references(references: Any, where: str) -> None:
    _validate_modulation_references(references, where, "width")
    # Keep membership as one condition when replayed with tunable parameter expressions.
    if sum(references.mode == mode for mode in PHASE_SHIFT_MODES) != 1:
        raise ConfigError(f"{where}.mode must be one of {sorted(PHASE_SHIFT_MODES)}")


CONTROL_TYPES: dict[str, ControlTypeSpec] = {}


def register_control_type(name: str, **values: Any) -> ControlTypeSpec:
    """Register one complete control profile; profiles cannot be patched later."""
    if name in CONTROL_TYPES:
        raise ValueError(f"control type {name!r} is already registered")
    try:
        spec = ControlTypeSpec(name=name, **values)
    except TypeError as exc:
        raise TypeError(f"invalid control type {name!r}: {exc}") from None
    CONTROL_TYPES[name] = spec
    return spec


def control_type(name: str, where: str = "ctrl.type") -> ControlTypeSpec:
    try:
        return CONTROL_TYPES[name]
    except KeyError:
        raise ConfigError(
            f"{where}: unknown control type {name!r}; known: {sorted(CONTROL_TYPES)}"
        ) from None
