"""Control configuration schemas, registered assembly profiles and default wiring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from ..components.pwm import PHASE_SHIFT_MODES, make_modulator
from ..control.controller import (
    CONTROL_INTERFACE,
    CURRENT_CONVERTER_INTERFACE,
    DC_CONVERTER_INTERFACE,
    OPEN_LOOP_CONVERTER_INTERFACE,
    PHASE_SHIFT_CONVERTER_INTERFACE,
    RESONANT_CONVERTER_INTERFACE,
    GraphController,
    VSCController,
)
from ..control.loops import LOOP_TYPES
from ..solver.model import ConfigError
from ..solver.signals import TableSignal


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
class ScalarCurrentReferenceParams:
    """Inductor-current reference in amperes for scalar converter control."""

    current: float


@dataclass(frozen=True)
class ResonantReferenceParams:
    """Fundamental commands and an optional signed port-power reference (W)."""

    modulation: float
    frequency: float
    phase: float = 0.0
    power: float = 0.0


@dataclass(frozen=True)
class DutyReferenceParams:
    """Open-loop duty-ratio command in [0, 1]."""

    duty: float


@dataclass(frozen=True)
class PhaseShiftReferenceParams:
    """Square-wave width, frequency, phase, power and pulse-mode references."""

    width: float
    frequency: float
    phase: float = 0.0
    power: float = 0.0
    mode: int = 0


@dataclass(frozen=True)
class ReferenceBinding:
    """One controller-reference binding with its ordinary SI/pu conversion."""

    name: str
    signal: TableSignal
    input_name: str
    scale: float = 1.0
    integer: bool = False

    def __call__(self, t: float) -> float | int:
        return self._cast(self.scale * self.signal(t))

    def bounds(self, start: float, stop: float) -> tuple[float | int, float | int]:
        low, high = sorted(self.scale * value for value in self.signal.bounds(start, stop))
        return self._cast(low), self._cast(high)

    def _cast(self, value: float) -> float | int:
        return int(round(value)) if self.integer else value


@dataclass(frozen=True)
class ControlTypeSpec:
    """Configuration and assembly policy selected by one ``ctrl.type`` value."""

    name: str
    references: type | None = None
    unit_types: frozenset[str] = field(default_factory=frozenset)
    interface: Any = None
    wiring: Callable[..., Any] | None = None
    log_sources: Mapping[str, str] | None = None
    grid_following: Optional[bool] = None
    default_bridge: str | None = None
    controller_factory: Callable[..., Any] | None = None
    reference_validator: Callable[[Any, str], None] | None = None
    options: type | None = None
    requires_loops: bool = True
    supported_models: frozenset[str] | None = None
    validator: Callable[[Any, str], None] | None = None
    modulator: str | None = None
    default_loops: Mapping[str, str] = field(default_factory=dict)
    grid_frequency_reference: str | None = None
    dc_voltage_reference: str | None = None


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


def validate_duty_references(references: Any, where: str) -> None:
    if not math.isfinite(references.duty) or not 0.0 <= references.duty <= 1.0:
        raise ConfigError(f"{where}.duty must be in [0, 1]")


def validate_phase_shift_references(references: Any, where: str) -> None:
    _validate_modulation_references(references, where, "width")
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


def control_profile(control: Any) -> ControlTypeSpec:
    """Return the profile resolved while building one control configuration."""
    profile = getattr(control, "_profile", None)
    if profile is not None:
        return profile
    kind = getattr(control, "type", None)
    if kind is not None:
        return control_type(kind)
    raise ConfigError("an explicit control graph has not been assembled")


def control_profiles(unit_type: str) -> tuple[ControlTypeSpec, ...]:
    return tuple(profile for profile in CONTROL_TYPES.values() if unit_type in profile.unit_types)


def explicit_wiring(_cfg: Any, *, continuous: bool = False) -> tuple[dict, dict]:
    return {}, {}


def _role_finder(cfg: Any, label: str):
    roles = {
        name: getattr(LOOP_TYPES.get(params.type), "role", None)
        for name, params in cfg.ctrl.loops.items()
    }

    def find(role: str) -> str | None:
        found = [name for name, value in roles.items() if value == role]
        if len(found) > 1:
            raise ConfigError(
                f"ctrl.loops: ambiguous {label} role {found}; use explicit "
                "ctrl.connections and ctrl.outputs"
            )
        return found[0] if found else None

    return find


def default_wiring(cfg: Any, family: str, *, continuous: bool = False):
    """Build the role-based default VSC graph for one registered control family."""
    loops = cfg.ctrl.loops
    roles = {name: getattr(LOOP_TYPES.get(params.type), "role", None)
             for name, params in loops.items()}
    find = _role_finder(cfg, family)
    pll, sync = find("pll"), find("sync")
    angle = pll if family == "gfl" else sync
    frame = f"{angle}.frame" if angle else "references.theta"
    theta = f"{angle}.theta" if angle else "references.theta"
    omega = f"{angle}.omega" if angle else "references.omega"
    dc, current = find("dc_voltage"), find("current")
    modulation = find("modulation")
    admittance, impedance, damping = find("admittance"), find("impedance"), find("damping")
    extra = f"{damping}.extra" if damping else "references.zero_v_pu"
    command = impedance if family == "gfm" and impedance else current
    ports_of = {
        "pll": {"v": "meas.u_g"},
        "dc_voltage": {"voltage": "meas.u_dc", "reference": "references.vdc_ref_pu"},
        "current": {"v": "meas.u_g", "i": "meas.i_c", "frame": frame, "omega": omega,
                    "extra": extra,
                    "id_ref": (f"{admittance}.id_ref" if admittance
                               else f"{dc}.current_reference" if dc
                               else "references.id_ref_pu"),
                    "iq_ref": (f"{admittance}.iq_ref" if admittance
                               else "references.iq_ref_pu")},
        "sync": {"v": "meas.u_g", "i": "meas.i_c", "u_dc": "meas.u_dc",
                 "p_ref": "references.p_ref_pu", "q_ref": "references.q_ref_pu",
                 "v_ref": "references.v_ref_pu"},
        "impedance": {"v_ref": f"{sync}.v_ref", "frame": frame, "omega": omega,
                      "i": "meas.i_c", "extra": extra},
        "admittance": {"v_ref": f"{sync}.v_ref", "frame": frame, "omega": omega,
                       "v": "meas.u_g"},
        "damping": {"i": "meas.i_c", "frame": frame},
        "modulation": {
            "u_dq": f"{command}.u_dq", "theta": theta, "u_dc": "meas.u_dc",
            "v": "meas.u_g", "i": "meas.i_c", "frame": frame, "omega": omega,
            "id_ref": (f"{admittance}.id_ref" if admittance
                       else f"{dc}.current_reference" if dc
                       else "references.id_ref_pu"),
            "iq_ref": (f"{admittance}.iq_ref" if admittance
                       else "references.iq_ref_pu"),
            "extra": extra,
        },
    }
    wires = {f"{name}.{port}": source for name, role in roles.items()
             for port, source in ports_of.get(role, {}).items()
             if port in LOOP_TYPES[loops[name].type].inputs}
    if continuous:
        return (wires, {}) if command is None else (
            wires, {"u_dq": f"{command}.u_dq", "theta": theta, "omega": omega},
        )
    if modulation is None:
        raise ConfigError(f"ctrl.loops: sampled {family} control needs a modulation loop")
    outputs = {"d_a": f"{modulation}.d_a", "d_b": f"{modulation}.d_b",
               "d_c": f"{modulation}.d_c", "theta": theta, "omega": omega}
    if "enable" in LOOP_TYPES[loops[modulation].type].outputs:
        outputs["enable"] = f"{modulation}.enable"
    return wires, outputs


def _modulation_wiring(cfg: Any, *, continuous: bool = False):
    profile = control_profile(cfg.ctrl)
    modulation = _role_finder(cfg, profile.name)("modulation")
    if modulation is None:
        return {}, {}
    loop = LOOP_TYPES[cfg.ctrl.loops[modulation].type]
    connections = {f"{modulation}.{port}": f"references.{port}" for port in loop.inputs
                   if f"references.{port}" in profile.interface.inputs}
    outputs = {port: f"{modulation}.{port}" for port in profile.interface.outputs
               if port in loop.outputs}
    return connections, outputs


def _dc_converter_wiring(cfg: Any, *, continuous: bool = False):
    profile = control_profile(cfg.ctrl)
    find = _role_finder(cfg, profile.name)
    voltage, current = find("dc_voltage"), find("current")
    wires = {}
    if voltage is not None:
        wires.update({f"{voltage}.reference": "references.voltage",
                      f"{voltage}.voltage": "meas.voltage"})
    if current is None:
        return wires, {}
    reference = (f"{voltage}.current_reference" if voltage is not None
                 else "references.current" if "references.current" in profile.interface.inputs
                 else None)
    sources = {"reference": reference, "current": "meas.current",
               "voltage": "meas.voltage", "vin": "meas.vin",
               "feedforward": "meas.duty_feedforward"}
    loop = LOOP_TYPES[cfg.ctrl.loops[current].type]
    wires.update({f"{current}.{port}": sources[port] for port in loop.inputs
                  if port in sources and sources[port] is not None})
    outputs = {port: f"{current}.{port}" for port in profile.interface.outputs
               if port in loop.outputs}
    return wires, outputs


def _converter_log_sources(outputs: tuple[str, ...] = ("duty",),
                           **specific: str) -> Mapping[str, str]:
    return MappingProxyType({
        "current": "meas.current", "voltage": "meas.voltage", "power": "meas.power",
        **specific, **{name: name for name in outputs}, "in_service": "startup.active",
    })


_DC_LOG_SOURCES = _converter_log_sources(
    voltage_reference="references.voltage", input_voltage="meas.vin",
)
_RESONANT_LOG_SOURCES = _converter_log_sources(
    input_voltage="meas.vin", modulation="references.modulation",
    frequency="references.frequency", phase="references.phase",
    power_reference="references.power",
)
_PHASE_SHIFT_LOG_SOURCES = _converter_log_sources(
    ("width", "phase", "frequency", "mode"), input_voltage="meas.vin",
    width_reference="references.width", phase_reference="references.phase",
    power_reference="references.power",
)


def resolved_control_interface(profile: ControlTypeSpec, continuous: bool):
    resolver = getattr(profile.controller_factory, "resolve_interface", None)
    return profile.interface if resolver is None else resolver(profile, continuous)


def make_controller(cfg: Any, **kwargs: Any):
    """Construct the runtime controller selected during parameter assembly."""
    factory = control_profile(cfg.ctrl).controller_factory
    if factory is None:
        raise ConfigError("the control graph has no controller factory for this Unit type")
    return factory(cfg, **kwargs)


def make_control_modulator(cfg: Any):
    """Construct the physical modulator from one already-resolved control profile."""
    if cfg.ctrl is None:
        return None
    profile = control_profile(cfg.ctrl)
    return make_modulator(
        cfg.pwm, cfg.base.f0, cfg.bridge, profile.modulator, cfg.ctrl.references,
        profile.interface.auxiliary_outputs(cfg.ctrl.outputs),
    )


for _control_name in ("gfl", "gfm"):
    register_control_type(
        _control_name, references=ReferenceParams, unit_types=frozenset(("vsc",)),
        interface=CONTROL_INTERFACE,
        wiring=lambda cfg, continuous=False, family=_control_name: default_wiring(
            cfg, family, continuous=continuous,
        ),
        grid_following=_control_name == "gfl", default_bridge="three_phase_bridge",
        controller_factory=VSCController,
        default_loops={"modulation": "ThreePhaseModulation"},
        grid_frequency_reference="omega", dc_voltage_reference="vdc_ref_pu",
    )
register_control_type(
    "dc", references=ScalarVoltageReferenceParams,
    unit_types=frozenset(("dc_converter",)), interface=DC_CONVERTER_INTERFACE,
    wiring=_dc_converter_wiring, log_sources=_DC_LOG_SOURCES,
    default_bridge="half_bridge", controller_factory=GraphController,
    reference_validator=validate_scalar_voltage_references,
)
register_control_type(
    "current", references=ScalarCurrentReferenceParams,
    unit_types=frozenset(("dc_converter",)), interface=CURRENT_CONVERTER_INTERFACE,
    wiring=_dc_converter_wiring,
    log_sources=_converter_log_sources(
        input_voltage="meas.vin", current_reference="references.current",
    ),
    default_bridge="half_bridge", controller_factory=GraphController,
)
register_control_type(
    "open_loop", references=DutyReferenceParams,
    unit_types=frozenset(("dc_converter",)), interface=OPEN_LOOP_CONVERTER_INTERFACE,
    wiring=_modulation_wiring,
    log_sources=_converter_log_sources(
        input_voltage="meas.vin", duty_reference="references.duty",
    ),
    default_bridge="half_bridge", controller_factory=GraphController,
    reference_validator=validate_duty_references,
)
register_control_type(
    "phase_shift", references=PhaseShiftReferenceParams,
    unit_types=frozenset(("dc_converter",)), interface=PHASE_SHIFT_CONVERTER_INTERFACE,
    wiring=_modulation_wiring, log_sources=_PHASE_SHIFT_LOG_SOURCES,
    default_bridge="full_bridge", controller_factory=GraphController,
    reference_validator=validate_phase_shift_references, modulator="phase_shift",
)
register_control_type(
    "resonant", references=ResonantReferenceParams,
    unit_types=frozenset(("dc_converter",)), interface=RESONANT_CONVERTER_INTERFACE,
    wiring=_modulation_wiring, log_sources=_RESONANT_LOG_SOURCES,
    default_bridge="full_bridge", controller_factory=GraphController,
    reference_validator=validate_resonant_references,
)
