"""Simulation-file parameters, pu bases, typed construction and checked event-variable writes.

Electrical inputs without suffix are SI values; ``_pu`` names are per unit. Plant quantities are
stored in SI, controller quantities in pu. A field listed in a class's ``_quantities`` may be given
in SI or pu. :func:`from_dict` builds the checked :class:`Params` from a mapping in one pass: it
converts each section's inputs on the bases in force there (the system base, or a unit's own), checks
fields, types and choices, and builds registered loop, element and event parameter types;
:func:`to_dict` converts back.

pu bases, AC: Vb = sqrt(2/3)*v_ll_rms (phase peak), Ib = S/(1.5*Vb), Zb = Vb/Ib, w0 = 2*pi*f0;
DC: Vdc = vdc_ref, Idc = S/Vdc, Zdc = Vdc**2/S. L base = Z/w0, C base = 1/(w0*Z).
``v_ll_rms`` and ``f0`` are optional until an AC quantity or frequency-based pu quantity needs
them; scalar SI models use the same base schema with only ``s_base``.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import json
import math
from collections.abc import Mapping as _Mapping
from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints

from ..components.network import ELEMENT_TYPES, element_buses
from ..solver.timing import control_clock_offset, resolve_clock_offset
from ..control.loops import LOOP_TYPES, loop_key_type, loop_type
from ..control.profiles import ReferenceParams, control_type
from ..solver.model import ConfigError
from .events import EVENT_TYPES, NAMED, SWITCHABLE, events_for
from .types import unit_type

__all__ = [
    "ConfigError", "load", "dump", "dumps", "read_tree", "read_initial", "from_dict", "to_dict",
    "BaseValues", "DCBase",
    "BusParams", "BranchParams", "SourceParams",
    "FilterParams", "DCLinkParams", "DCCapacitorParams", "DCSourceParams", "PWMParams",
    "MeasurementParams", "ReferenceParams", "ControlParams", "OvercurrentParams", "VoltageLimitParams",
    "FrequencyLimitParams", "DCVoltageLimitParams", "RocofParams", "ProtectionParams",
    "BridgeParams", "SSLinkParams", "LCCLinkParams", "LCCSLinkParams", "MultiSSLinkParams",
    "SCCParams",
    "UnitParams",
    "RUNTIME", "Change", "runtime_changeable", "event_changes",
    "SolverParams", "SimulationParams", "InitialParams", "OutputParams", "ProgressParams",
    "MetaParams", "Params",
]


_PARAMETER_RECORDER: ContextVar[Any] = ContextVar("peslite_parameter_recorder", default=None)


def _parameterized(factory, parameters):
    """Construct a part normally, optionally recording its parameter initialization.

    The callable is the construction itself, not a second binding table. Exporters can replay it
    with parameter expressions; ordinary simulations retain neither the callable nor a shadow part.
    """
    part = factory(parameters)
    recorder = _PARAMETER_RECORDER.get()
    if recorder is not None:
        recorder(part, factory, parameters)
    return part


def parameter_leaves(value, prefix="", defaulted=frozenset()):
    """Stored numerical input paths, excluding schema-declared structural fields.

    This describes parameters, not a backend's private list of supported device families.
    Collections defining layout and boolean feature switches remain structural.
    """
    defaulted = defaulted | {f"{prefix}.{path}" if prefix else path
                             for path in getattr(value, "derived_defaults", ())}
    if isinstance(value, UnitParams):
        yield from parameter_leaves(value._unit_fields, prefix, defaulted)
        items = ((item.name, getattr(value, item.name), item.metadata)
                 for item in fields(value) if item.init and not item.name.startswith("_"))
    elif is_dataclass(value):
        items = ((item.name, getattr(value, item.name), item.metadata)
                 for item in fields(value) if item.init)
    elif isinstance(value, dict):
        items = ((key, item, {}) for key, item in value.items())
    else:
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            yield prefix, value, prefix in defaulted
        return
    for name, item, metadata in items:
        if not metadata.get("structural", False):
            yield from parameter_leaves(item, f"{prefix}.{name}" if prefix else name, defaulted)


def connection_defaults(p):
    """Initial leaf-unit commands, before scheduled event writes are applied."""
    from .events import _initial_connection, switching_schedule

    if not getattr(p, "units", None):
        return
    schedule = switching_schedule(p.changes)
    for name, unit in _unit_items(p.units):
        path = f"units.{name}.connected"
        steps = schedule.get(f"units.{name}", ())
        yield path, _initial_connection(steps, unit.connected)


# ------------------------------------------------------------------ pu bases and SI/pu inputs

@dataclass(frozen=True)
class DCBase:
    """DC-side bases derived from the rated DC voltage and the power base."""

    v: float  # rated dc voltage (V)
    i: float  # s_base / v (A)
    z: float  # v / i (ohm)
    w0: Optional[float]

    def C(self, c_pu: float) -> float:
        """pu -> F."""
        if self.w0 is None:
            raise ConfigError("a per-unit DC capacitance requires base.f0")
        return c_pu / (self.w0 * self.z)


@dataclass(frozen=True)
class BaseValues:
    """Shared rating and optional AC base used by system and unit parameter scopes.

    ``s_base`` is always available. ``v_ll_rms`` and ``f0`` are required only by quantities whose
    SI/pu conversion or physical model uses them; scalar models written in SI therefore do not
    need placeholder AC values.
    """

    s_base: float
    v_ll_rms: Optional[float] = None
    f0: Optional[float] = None
    v_phase_peak: Optional[float] = field(init=False)
    i_phase_peak: Optional[float] = field(init=False)
    z_base: Optional[float] = field(init=False)
    w0: Optional[float] = field(init=False)

    def __post_init__(self) -> None:
        for name in ("s_base", "v_ll_rms", "f0"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ConfigError(f"{name} must be finite and positive")
        v = (None if self.v_ll_rms is None
             else math.sqrt(2.0 / 3.0) * self.v_ll_rms)
        i = None if v is None else self.s_base / (1.5 * v)
        object.__setattr__(self, "v_phase_peak", v)
        object.__setattr__(self, "i_phase_peak", i)
        object.__setattr__(self, "z_base", None if v is None else v / i)
        object.__setattr__(self, "w0", None if self.f0 is None else 2.0 * math.pi * self.f0)

    def require(self, *names: str, where: str) -> "BaseValues":
        """Return this base after checking fields required by the requesting model."""
        missing = [f"base.{name}" for name in names if getattr(self, name) is None]
        if missing:
            raise ConfigError(f"{where} requires {', '.join(missing)}")
        return self

    # pu -> SI
    def L(self, x_pu: float) -> float:
        self.require("v_ll_rms", "f0", where="an AC per-unit inductance")
        assert self.z_base is not None and self.w0 is not None
        return x_pu * self.z_base / self.w0

    def R(self, r_pu: float) -> float:
        self.require("v_ll_rms", where="an AC per-unit resistance")
        assert self.z_base is not None
        return r_pu * self.z_base

    def C(self, c_pu: float) -> float:
        self.require("v_ll_rms", "f0", where="an AC per-unit capacitance")
        assert self.w0 is not None and self.z_base is not None
        return c_pu / (self.w0 * self.z_base)

    def dc(self, vdc_ref: float) -> DCBase:
        """Return the DC base for the rated DC voltage ``vdc_ref`` (V)."""
        i = self.s_base / vdc_ref if vdc_ref > 0.0 else 0.0
        return DCBase(v=vdc_ref, i=i, z=vdc_ref / i if i > 0.0 else 0.0, w0=self.w0)

    def parameter_scales(self, vdc_ref: float = 1.0) -> dict[str, float]:
        """Return the SI value of one pu for every quantity defined by this base."""
        dc = self.dc(vdc_ref)
        scales = {"1": 1.0}
        if self.v_phase_peak is not None:
            scales.update(voltage=self.v_phase_peak, current=self.i_phase_peak)
        scales["power"] = self.s_base
        if self.w0 is not None:
            scales["frequency"] = self.w0
        if self.z_base is not None:
            scales["resistance"] = self.z_base
        if self.z_base is not None and self.w0 is not None:
            scales.update(inductance=self.z_base / self.w0,
                          capacitance=1.0 / (self.w0 * self.z_base))
        scales.update(dc_voltage=dc.v, dc_current=dc.i, dc_resistance=dc.z)
        if dc.w0 is not None:
            scales["dc_capacitance"] = dc.C(1.0)
        return scales


def quantity_metadata(cls, name: str) -> dict:
    """Merge the class attribute ``name`` over the class hierarchy into one dict."""
    result = {}
    for parent in reversed(cls.__mro__):
        result.update(parent.__dict__.get(name, {}))
    return result


def quantity_name(cls, name: str) -> str:
    """Return the stored field name for the SI or ``_pu`` spelling of a parameter."""
    name = quantity_metadata(cls, "_input_aliases").get(name, name)
    for target in quantity_metadata(cls, "_quantities"):
        si = target.removesuffix("_pu")
        if name in (si, si + "_pu"):
            return target
    return name


def convert_quantities(data: dict, cls, scales: dict[str, float], where: str) -> dict:
    """Convert SI/pu inputs of one section to its storage units and fill pu defaults.

    Raises ConfigError if both the SI and the pu spelling of a parameter are given.
    """
    aliases = quantity_metadata(cls, "_input_aliases")
    quantities = quantity_metadata(cls, "_quantities")
    out, seen = {}, {}
    for name, value in data.items():
        spelling = aliases.get(name, name)
        target = quantity_name(cls, name)
        if target in seen:
            raise ConfigError(f"{where}.{target}: both {seen[target]!r} and {name!r} specify the same parameter")
        seen[target] = name
        if target in quantities and value is not None:
            if isinstance(value, bool):
                raise ConfigError(f"{where}.{name}: expected a number")
            try:
                if not isinstance(value, (int, float)):
                    value = float(value)
            except (TypeError, ValueError):
                raise ConfigError(f"{where}.{name}: expected a number, got {value!r}") from None
            if spelling.endswith("_pu") != target.endswith("_pu"):
                try:
                    scale = _scale(quantities[target], scales)
                except ConfigError as exc:
                    raise ConfigError(f"{where}.{name}: {exc}") from None
                value = value * scale if spelling.endswith("_pu") else value / scale
        out[target] = value
    for name, value in quantity_metadata(cls, "_defaults_pu").items():
        if name not in out:
            out[name] = value if name.endswith("_pu") else value * _scale(quantities[name], scales)
    return out


def _scale(expression: str, scales: dict[str, float]) -> float:
    numerator, *denominators = expression.split("/")
    try:
        value = math.prod(scales[name] for name in numerator.split("*"))
        for denominator in denominators:
            value /= math.prod(scales[name] for name in denominator.split("*"))
    except KeyError as exc:
        raise ConfigError(
            f"a per-unit {exc.args[0].replace('_', ' ')} value needs the corresponding base"
        ) from None
    return value


# ------------------------------------------------------------------ the network

@dataclass(frozen=True)
class BusParams:
    """Electrical node with a shunt capacitor to ground.

    ``phases`` is 3 for the peak-scaled AC space-vector domain and 1 for a real scalar domain.
    ``c`` is capacitance in F and ``r_d`` is series damping resistance in ohm.
    """

    c: float
    r_d: float = 0.0
    phases: int = field(default=3, metadata={"structural": True})

    _quantities = {"c": "capacitance", "r_d": "resistance"}


@dataclass(frozen=True)
class BranchParams:
    """Series R-L branch between two buses; current positive from ``bus1`` to ``bus2``.

    ``l`` in H, ``r`` in ohm; pu inputs use the system base. Connect/disconnect events switch it.
    """

    bus1: str
    bus2: str
    l: float
    r: float = 0.0
    _quantities = {"l": "inductance", "r": "resistance"}
    _input_aliases = {"x_pu": "l_pu"}


@dataclass(frozen=True)
class SourceParams:
    """AC emf behind a series R-L impedance, connected to ``bus``.

    The target's phase count selects a real single-phase waveform or the three-phase complex
    space-vector representation. ``l`` is in H, ``r`` in ohm, ``v`` is peak voltage (default
    1 pu), ``f`` is in Hz (default ``base.f0``), and ``angle`` is in rad.
    """

    bus: str
    l: float
    r: float = 0.0
    v: float = 0.0  # emf magnitude, phase-peak V
    f: Optional[float] = None
    angle: float = 0.0
    derived_defaults: frozenset = field(init=False, default=frozenset(), compare=False)

    _quantities = {"l": "inductance", "r": "resistance", "v": "voltage"}
    _input_aliases = {"x_pu": "l_pu"}
    _defaults_pu = {"v": 1.0}

    def resolved(self, base: BaseValues) -> "SourceParams":
        source = replace(self, f=base.f0 if self.f is None else self.f)
        object.__setattr__(source, "derived_defaults", frozenset(("f",)) if self.f is None else frozenset())
        return source

# ------------------------------------------------------------------ a converter unit

@dataclass(frozen=True)
class FilterParams:
    """Series R-L branch at a converter terminal (``l_f`` H, ``r_f`` ohm).

    The shunt capacitor is set on the connected bus (:class:`BusParams`).
    """

    l_f: float
    r_f: float = 0.0

    _quantities = {"l_f": "inductance", "r_f": "resistance"}
    _input_aliases = {"x_f_pu": "l_f_pu"}


@dataclass(frozen=True)
class DCCapacitorParams:
    """Capacitance (F) and series ESR (ohm); pu inputs use the unit DC base."""

    c: float
    r_esr: float = 0.0

    _quantities = {"c": "dc_capacitance", "r_esr": "dc_resistance"}


@dataclass(frozen=True)
class DCSourceParams:
    """DC-link supply selected by ``type`` (pu inputs use the unit DC base)."""

    type: str = "none"  # "none" | "current" | "voltage"
    i: float = 0.0  # current source, A, positive into the DC link
    k: float = 0.0  # current-source voltage droop, A/V
    v: float = 0.0  # voltage-source emf, V
    r: float = 0.0  # voltage-source series resistance, ohm

    _choices = {"type": ("none", "current", "voltage")}

    _quantities = {"i": "dc_current", "k": "dc_current/dc_voltage",
                   "v": "dc_voltage", "r": "dc_resistance"}

    @staticmethod
    def _scaled_defaults(data: dict, scales: dict[str, float]) -> dict:
        """A voltage source's emf defaults to the rated dc voltage."""
        return ({"v": scales["dc_voltage"]}
                if data.get("type") == "voltage" else {})


@dataclass(frozen=True)
class DCLinkParams:
    """DC link of one unit: rated voltage ``vdc_ref`` (V), optional capacitor and supply."""

    vdc_ref: float
    capacitor: Optional[DCCapacitorParams] = None
    source: DCSourceParams = field(default_factory=DCSourceParams)


@dataclass(frozen=True)
class PWMParams:
    """PWM carrier, modulation method and compare-register update mode.

    ``update``: ``"single"`` loads the compare registers at carrier valleys; ``"double"``
    loads them at valleys and peaks.
    ``sync``: ``"asynchronous"`` (carrier on absolute time) or ``"synchronous"`` (carrier
    locked to the controller angle; requires an integer ``f_sw / base.f0``).
    ``pattern_sync``: ``"interval"`` applies newly loaded waveform commands
    at the load; ``"pulse"`` lets a pulse in progress finish and starts the new
    pattern at its next pulse.
    """

    f_sw: float  # carrier frequency, Hz
    modulation_limit: float = 1.0
    update: str = "single"  # "single" | "double"
    carrier_phase: float = field(default=0.0, metadata={"modulator": True})
    method: str = "spwm"  # "spwm" | "svpwm"
    sync: str = field(default="asynchronous", metadata={"modulator": True})
    pattern_sync: str = "interval"

    _choices = {"update": ("single", "double"), "method": ("spwm", "svpwm"),
                "sync": ("asynchronous", "synchronous"), "pattern_sync": ("interval", "pulse")}

    @property
    def switching_period(self) -> float:
        return 1.0 / self.f_sw

    @property
    def load_period(self) -> float:
        """Time between compare-register loads (s)."""
        return self.switching_period / 2.0 if self.update == "double" else self.switching_period

    @property
    def grid_offset(self) -> float:
        """First asynchronous carrier valley at or after t = 0; synchronous timing starts at 0."""
        if self.sync == "synchronous":
            return 0.0
        return ((-self.carrier_phase) % 1.0) * self.switching_period


@dataclass(frozen=True)
class MeasurementParams:
    """ADC measurement settings.

    ``average`` lists any of ``u_g``, ``i_c`` and ``u_dc`` which use a mean over ``window``
    (s, at most the control period). Channels not listed are instantaneous.
    ``period`` is the ADC oversampling period; it defaults to the control period.
    ``sources`` selects physical outputs for existing channels. ``rms`` selects sliding
    mean-square windows, in whole control periods (protection/output periods in ideal averaging).
    """

    average: list[str] = field(default_factory=list)
    window: Optional[float] = None  # s; default: the control period
    period: Optional[float] = None  # s; default: the control period
    offset: Optional[float] = None  # s; default: the controller clock offset
    sources: dict[str, str] = field(default_factory=dict)  # channel -> physical output name
    rms: dict[str, float] = field(default_factory=dict)  # channel -> window duration (s)

    def clock_offset(self, control_offset: float, period: float) -> float:
        """Return the ADC clock offset, inheriting the controller grid when omitted."""
        return resolve_clock_offset(period, self.offset, control_offset)


@dataclass(frozen=True, kw_only=True)
class ControlParams:
    """Digital controller fields shared by every controller architecture.

    For switching and PWM-period averaging, ``period`` defaults to one carrier period and
    ``computation`` defaults to 1 us: the time from the ADC sample until the new shadow duty
    ratios may be loaded. With the ideal averaging bridge the controller is continuous; these
    discrete timing settings are retained in the file but ignored.

    ``type`` selects the reference-field schema and default wiring. ``connections`` and ``outputs``
    override that wiring. A ``loops`` key names a registered loop class. The optional
    ``ClassName[label]`` form distinguishes repeated uses of the same class.
    """

    type: str
    period: Optional[float] = None  # s; default: one carrier period
    offset: Optional[float] = None  # s; default: the PWM load-clock offset
    computation: Optional[float] = None  # s; sampled default 1 us; ignored by continuous averaging
    loops: dict[str, Any] = field(metadata={"entries": "loop"})
    connections: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    references: Any = None  # built from the schema selected by type

    def clock_offset(self, pwm: PWMParams) -> float:
        """Return the control-clock offset, inheriting the PWM load grid when omitted."""
        return control_clock_offset(self, pwm)


@dataclass(frozen=True)
class OvercurrentParams:
    enable: bool = False
    limit_pu: Optional[float] = None

    _quantities = {"limit_pu": "current"}


@dataclass(frozen=True)
class VoltageLimitParams:
    enable: bool = False
    limit_pu: Optional[float] = None

    _quantities = {"limit_pu": "voltage"}


@dataclass(frozen=True)
class FrequencyLimitParams:
    enable: bool = False
    limit: Optional[float] = None  # Hz


@dataclass(frozen=True)
class DCVoltageLimitParams:
    enable: bool = False
    limit_pu: Optional[float] = None

    _quantities = {"limit_pu": "dc_voltage"}


@dataclass(frozen=True)
class RocofParams:
    enable: bool = False
    limit: Optional[float] = None  # Hz/s
    window: float = 0.1  # s


@dataclass(frozen=True)
class ProtectionParams:
    """Trip criteria and the ROCOF alarm, each switched by ``enable`` (1/0)."""

    overcurrent: OvercurrentParams = field(default_factory=OvercurrentParams)
    rms_current: OvercurrentParams = field(default_factory=OvercurrentParams)
    undervoltage: VoltageLimitParams = field(default_factory=VoltageLimitParams)
    overvoltage: VoltageLimitParams = field(default_factory=VoltageLimitParams)
    frequency: FrequencyLimitParams = field(default_factory=FrequencyLimitParams)
    dc_voltage: DCVoltageLimitParams = field(default_factory=DCVoltageLimitParams)
    rocof: RocofParams = field(default_factory=RocofParams)
    hold: float = 0.02  # s


@dataclass(frozen=True)
class BridgeParams:
    """Bridge model used by one unit.

    ``"switching"`` follows exact carrier-comparison edges. ``"pwm_averaging"`` keeps the
    PWM timer, duty registers and load timing but applies each active duty ratio continuously until
    the next compare-register load. ``"averaging"`` is an ideal controlled voltage source driven
    by continuous measurement and control equations, with no ADC, interrupt, computation or PWM
    events. For a unilateral half bridge, ``pwm.f_sw`` still defines the physical cycle used by
    the averaged DCM conduction law; it does not schedule carrier events.
    A controllerless diode full bridge has no PWM waveform to average, so all three selections
    retain its exactly located natural commutations while selecting the surrounding run preset.
    Optional device resistance and port capacitance use the same topology and execution form;
    nonzero capacitance requires positive device resistance.
    """

    model: str = "pwm_averaging"
    type: Optional[str] = None
    upper_controlled: bool = True
    lower_controlled: bool = True
    reverse_port: bool = False
    external_terminal: str = "midpoint"
    on_resistance: float = 0.0  # ohm, per conducting switch or diode
    parallel_capacitance: float = 0.0  # F, across the power port (per phase for three phases)

    _choices = {
        "model": ("switching", "pwm_averaging", "averaging"),
        "external_terminal": ("midpoint", "lower_rail"),
    }

    @property
    def controlled(self) -> bool:
        """Whether any device is gated; otherwise the full bridge is a diode rectifier."""
        return self.upper_controlled or self.lower_controlled

    @property
    def neutral_duty(self) -> float:
        """Duty coordinate of zero bridge voltage, from the registered topology."""
        from ..components.converter import topology_type
        return topology_type(self.type).neutral_duty


@dataclass(frozen=True, init=False)
class UnitParams:
    """Common converter-unit parameters plus one ``type``-selected field set."""

    type: str = "vsc"
    pwm: Optional[PWMParams] = None
    ctrl: Optional[ControlParams] = None
    dclink: Optional[DCLinkParams] = None
    bus: Optional[str] = None
    connected: Optional[bool] = None  # initial command; None preserves event-derived startup
    bridge: BridgeParams = field(default_factory=BridgeParams)
    meas: MeasurementParams = field(default_factory=MeasurementParams)
    protection: ProtectionParams = field(default_factory=ProtectionParams)
    s_base: Optional[float] = None
    _unit_fields: Any = field(repr=False, default=None)
    base: BaseValues = field(init=False, default=None)
    derived_defaults: frozenset = field(init=False, default=frozenset(), compare=False)
    events: tuple = field(init=False, default=(), compare=False)

    def __init__(self, *, type: str, _unit_fields: Any, **values: Any) -> None:
        schema = fields(UnitParams)
        known = {item.name for item in schema if item.init}
        changes = {name: value for name, value in values.items() if name not in known}
        selected = replace(_unit_fields, **changes) if changes else _unit_fields
        values = {**values, "type": type, "_unit_fields": selected}
        composite = unit_type(type).composite
        for item in schema:
            if item.init and item.name in values:
                value = values[item.name]
            elif item.default_factory is not dataclasses.MISSING:
                value = None if composite else item.default_factory()
            else:
                value = item.default
            object.__setattr__(self, item.name, value)

    @classmethod
    def _common_fields(cls, *, composite: bool = False) -> frozenset[str]:
        """Public common fields, interpreted from this one schema at every boundary."""
        return frozenset(item.name for item in fields(cls)
                         if item.init and item.name not in {"type", "ctrl", "_unit_fields"}
                         and (not composite or item.name == "s_base"))

    def __getattr__(self, name: str) -> Any:
        """Expose the selected schema's fields without storing a second copy."""
        return getattr(object.__getattribute__(self, "_unit_fields"), name)

    @staticmethod
    def _bases(data: dict, base: BaseValues, where: str,
               spec: Any) -> tuple[BaseValues, dict[str, float]]:
        """Resolve the common rating and the electrical scaling base of a unit of type ``spec``."""
        if data.get("s_base") is not None:
            base = _build(
                BaseValues,
                {"s_base": data["s_base"], "v_ll_rms": base.v_ll_rms, "f0": base.f0},
                where,
                None,
                None,
            )
        if spec.composite:
            return base, base.parameter_scales()
        dclink = data.get("dclink")
        voltage = _numeric_string(dclink.get("vdc_ref")) if isinstance(dclink, dict) else None
        label = "dclink.vdc_ref"
        if (isinstance(voltage, bool) or not isinstance(voltage, (int, float))
                or not math.isfinite(voltage) or voltage <= 0.0):
            raise ConfigError(f"{where}.{label} must be finite and positive")
        return base, base.parameter_scales(float(voltage))

    def resolved(self, system_base: BaseValues, events: tuple = ()) -> "UnitParams":
        derived = set()
        ctrl = self.ctrl
        if ctrl is not None:
            assert self.pwm is not None and self.bridge is not None
            continuous = self.bridge.model == "averaging"
            if not continuous and ctrl.period is None:
                ctrl = replace(ctrl, period=self.pwm.switching_period)
                derived.add("ctrl.period")
            if not continuous and ctrl.computation is None:
                ctrl = replace(ctrl, computation=1.0e-6)
                derived.add("ctrl.computation")
            if unit_type(self.type).defaults_grid_frequency and ctrl.references.omega is None:
                references = replace(ctrl.references, omega=system_base.w0)
                ctrl = replace(ctrl, references=references)
                derived.add("ctrl.references.omega")
            loops = dict(ctrl.loops)
            for name, loop in loops.items():
                if not continuous and loop.period is None:
                    loops[name] = replace(loop, period=ctrl.period)
                    derived.add(f"ctrl.loops.{name}.period")
            ctrl = replace(ctrl, loops=loops)
        unit = replace(self, ctrl=ctrl)
        base = system_base if self.s_base is None else replace(system_base, s_base=self.s_base)
        object.__setattr__(unit, "base", base)
        object.__setattr__(unit, "derived_defaults", frozenset(derived))
        object.__setattr__(unit, "events", tuple(events))
        return unit

    @functools.cached_property
    def dc_base(self) -> DCBase:
        if unit_type(self.type).composite:
            raise AttributeError("a composite WPT unit has no single DC base")
        return self.base.dc(self.dclink.vdc_ref)


def _unit_items(units: Mapping[str, UnitParams], *, leaves_only: bool = True,
                prefix: str = ""):
    """Iterate the configured unit tree without storing a second flattened mapping."""
    for name, unit in units.items():
        path = f"{prefix}.{name}" if prefix else name
        spec = unit_type(unit.type)
        if not leaves_only or not spec.composite:
            yield path, unit
        if spec.composite:
            children = {child: getattr(unit, child) for child in spec.children}
            yield from _unit_items(children, leaves_only=leaves_only, prefix=path)


def _unit_at(units: Mapping[str, UnitParams], path: str) -> Optional[UnitParams]:
    """Return one node from the configured unit tree, or ``None`` when absent."""
    parts = path.split(".")
    unit = units.get(parts[0]) if parts and parts[0] else None
    for part in parts[1:]:
        if unit is None or part not in unit_type(unit.type).children:
            return None
        unit = getattr(unit, part)
    return unit


def _unit_path(units: Mapping[str, UnitParams], value: str) -> Optional[str]:
    """Return the longest configured unit path which prefixes ``value``."""
    return next(
        (name for name, _unit in sorted(
            _unit_items(units, leaves_only=False), key=lambda item: len(item[0]), reverse=True,
        ) if value == name or value.startswith(name + ".")),
        None,
    )


def _split_path(p: Params, path: str) -> tuple[str, str, str]:
    """Split ``section.name.rest``; a unit name is the longest configured unit path."""
    section, _, rest = path.partition(".")
    name = (_unit_path(p.units, rest) if section == "units" else None) or rest.partition(".")[0]
    return section, name, rest[len(name) + 1:]


def _unit_leaves(units: Mapping[str, UnitParams], path: str) -> tuple[str, ...]:
    """Return leaf paths below ``path`` directly from the authoritative unit tree."""
    unit = _unit_at(units, path)
    if unit is None:
        return ()
    spec = unit_type(unit.type)
    if not spec.composite:
        return (path,)
    children = {child: getattr(unit, child) for child in spec.children}
    return tuple(name for name, _child in _unit_items(children, prefix=path))


@dataclass(frozen=True)
class _VSCUnitFields:
    """Fields unique to a voltage-source converter.

    Plant fields are stored in SI; controller parameters are pu on the unit's rating
    ``s_base`` (VA) and DC base (``dclink.vdc_ref``).
    """

    ac_filter: FilterParams


@dataclass(frozen=True, kw_only=True)
class _DCConverterUnitFields:
    """Fields unique to a scalar converter unit.

    A series ``filter`` exposes the bridge through the common named ``bus``. A composite may omit
    both and attach the bridge to a scoped internal bus, as WPT does.
    """

    filter: Optional[FilterParams] = None


@dataclass(frozen=True)
class SCCParams:
    """Capacitor shunted by a source-to-source MOSFET pair, with explicit gate drive.

    ``common`` drives both gates together; ``alternating`` uses overlapping pulses displaced
    by pi. ``angle`` is their nominal conduction/overlap angle. Device channels and body diodes
    determine the actual branch conduction; zero-voltage turn-on is not imposed artificially.
    """

    capacitance: float
    angle: float = 0.0
    drive: str = "common"
    sync: str = "current"
    sync_angle: float = 0.0
    command: bool = False

    _choices = {"drive": ("common", "alternating"), "sync": ("current", "bridge")}


@dataclass(frozen=True, kw_only=True)
class SSLinkParams:
    """Series-series compensated two-winding inductive link in SI units."""

    type: str = "ss"
    primary_inductance: float
    secondary_inductance: float
    mutual_inductance: float
    primary_capacitance: float
    secondary_capacitance: float
    primary_resistance: float = 0.0
    secondary_resistance: float = 0.0
    primary_scc: Optional[SCCParams] = None
    secondary_scc: Optional[SCCParams] = None

    _choices = {"type": ("ss",)}


@dataclass(frozen=True, kw_only=True)
class LCCLinkParams:
    """Symmetric-form LCC-LCC compensated two-winding inductive link in SI units.

    Each port has a series compensation inductor, a shunt compensation capacitor and a
    series coil capacitor.  The two coil inductances are coupled by ``mutual_inductance``.
    """

    type: str = "lcc"
    primary_inductance: float
    secondary_inductance: float
    mutual_inductance: float
    primary_series_capacitance: float
    secondary_series_capacitance: float
    primary_compensation_inductance: float
    secondary_compensation_inductance: float
    primary_parallel_capacitance: float
    secondary_parallel_capacitance: float
    primary_resistance: float = 0.0
    secondary_resistance: float = 0.0
    primary_compensation_resistance: float = 0.0
    secondary_compensation_resistance: float = 0.0
    primary_parallel_resistance: float = 0.0  # ESR in series with the shunt capacitor
    secondary_parallel_resistance: float = 0.0
    primary_scc: Optional[SCCParams] = None
    secondary_scc: Optional[SCCParams] = None

    _choices = {"type": ("lcc",)}


@dataclass(frozen=True, kw_only=True)
class LCCSLinkParams:
    """LCC-compensated primary and series-compensated secondary of a two-winding link (SI).

    The primary port has a series compensation inductor, a shunt capacitor (optional ESR) and a
    series coil capacitor; the secondary coil has a series capacitor only.
    """

    type: str = "lcc_s"
    primary_inductance: float
    secondary_inductance: float
    mutual_inductance: float
    primary_series_capacitance: float
    primary_compensation_inductance: float
    primary_parallel_capacitance: float
    secondary_capacitance: float
    primary_resistance: float = 0.0
    secondary_resistance: float = 0.0
    primary_compensation_resistance: float = 0.0
    primary_parallel_resistance: float = 0.0
    primary_scc: Optional[SCCParams] = None
    secondary_scc: Optional[SCCParams] = None

    _choices = {"type": ("lcc_s",)}


@dataclass(frozen=True, kw_only=True)
class MultiSSLinkParams:
    """N series-series modules with inter- and cross-coupling, in SI units.

    ``inductance`` is the symmetric 2N x 2N inductance matrix of the coils in the order
    Tx1..TxN, Rx1..RxN. Every Tx coil (with its series capacitor and resistance) connects in
    parallel to the primary port, every Rx coil to the secondary port.
    """

    type: str = "ss_multi"
    inductance: list
    primary_capacitance: list
    secondary_capacitance: list
    primary_resistance: Optional[list] = None  # default: zero per module
    secondary_resistance: Optional[list] = None

    _choices = {"type": ("ss_multi",)}

    def __post_init__(self) -> None:
        modules = len(self.primary_capacitance) if isinstance(
            self.primary_capacitance, (list, tuple)) else 0
        for name in ("primary_resistance", "secondary_resistance"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, [0.0] * modules)


@dataclass(frozen=True)
class _WPTUnitFields:
    """Fields unique to a composite wireless-power-transfer unit."""

    primary: Any = field(metadata={"unit": True})
    secondary: Any = field(metadata={"unit": True})
    link: Any = field(metadata={"types": {"ss": SSLinkParams, "lcc": LCCLinkParams,
                                          "lcc_s": LCCSLinkParams,
                                          "ss_multi": MultiSSLinkParams}})


# ------------------------------------------------------------------ the run

@dataclass(frozen=True)
class SolverParams:
    """Integrator settings.

    ``subsystems`` (fixed-step only) maps subsystem names to a step relative to ``dt``
    (integer N, or 1/N) or to ``{step, method}``, e.g. ``{dclink: 10, network: 0.1}``.
    """

    type: str = "fixed"  # "fixed" | "adaptive"
    method: str = "rk4"  # fixed: euler|heun|rk4 ; adaptive: RK45|DOP853|Radau|BDF|LSODA|RK23|DP45
    dt: float = 1e-6  # s; fixed: maximum system step
    rtol: float = 1e-6
    atol: float = 1e-9
    max_step: float = math.inf
    warm_start: bool = True  # adaptive: reuse the last step size across intervals
    subsystems: dict = field(default_factory=dict)  # fixed only: {name: step relative to dt | {step, method}}
    sweeps: int = 2  # coupling sweeps for sub-stepped subsystems (>= 1)
    linearisations: int = 3  # linearisation points for the split error bound (0: none)
    write_length: int = 1000  # streamed CSV rows per write batch (>= 1)
    phs_check_step: int = 1000  # snapshots between port-Hamiltonian energy audits (>= 1)
    _choices = {"type": ("fixed", "adaptive")}


@dataclass(frozen=True)
class InitialParams:
    """Start time ``t`` (s) and overrides of initial state values.

    ``states`` maps dotted state names to a number, ``[re, im]`` or ``.re``/``.im`` entries,
    or a keyword: ``source`` (bus source voltage at ``t``) or ``rated`` (unit DC reference).
    A run starts at ``t`` exactly; each unit resumes its own timer grid around that instant.
    """

    t: float = 0.0
    states: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OutputParams:
    """Histories streamed by ``Simulation.run`` into its output directory."""

    period: float = 5e-4  # plant snapshot interval, s
    record_every: int = 1  # keep every n-th controller sample
    states: bool = True  # states.csv
    signals: bool = False  # plant.csv and ctrl.<unit>.csv
    energy: bool = False  # energy.csv


@dataclass(frozen=True)
class ProgressParams:
    """Progress lines, controlled by ``enable``.

    ``period`` is simulated seconds between lines. ``watch`` names state-table or plant-table
    quantities to append to each line; a complex state may be named without ``.re``/``.im`` to
    report its magnitude.
    """

    enable: bool = False
    period: Optional[float] = None
    watch: list = field(default_factory=list)


@dataclass(frozen=True)
class SimulationParams:
    """How the model is simulated: time span, solver, initial state and output."""

    t_end: float
    solver: SolverParams = field(default_factory=SolverParams)
    initial: InitialParams = field(default_factory=InitialParams, metadata={"structural": True})
    output: OutputParams = field(default_factory=OutputParams)
    stop_on_trip: bool = True
    progress: ProgressParams = field(default_factory=ProgressParams)
    energy_check: str = "warn"  # "warn" | "strict" | "off": verify the energy declarations before the run

    _choices = {"energy_check": ("warn", "strict", "off")}


@dataclass(frozen=True)
class MetaParams:
    """Optional human-readable notes which do not affect the simulation."""

    title: Optional[str] = None
    description: Optional[str] = None

    _text = ("title", "description")
    _unknown_hint = "meta holds only a title and a description"


@dataclass(frozen=True)
class Params:
    """Complete configuration: system base, network elements, units and run settings.

    ``buses``, ``branches``, ``sources``, ``units`` and ``elements`` map names to entries.
    Names are scoped by section and prefix state names.
    """

    base: BaseValues  # system base, used for network pu inputs
    units: dict[str, UnitParams] = field(
        default_factory=dict, metadata={"entries": "unit"}, kw_only=True,
    )
    simulation: SimulationParams
    buses: dict[str, BusParams] = field(default_factory=dict)
    sources: dict[str, SourceParams] = field(default_factory=dict)
    branches: dict[str, BranchParams] = field(default_factory=dict)
    elements: dict[str, Any] = field(default_factory=dict, metadata={"entries": "element"})
    events: dict[str, Any] = field(default_factory=dict,
                                   metadata={"entries": "event", "structural": True})
    meta: MetaParams = field(default_factory=MetaParams)

    def unit(self, name: Optional[str] = None) -> UnitParams:
        """Return the unit ``name``, or the only unit if ``name`` is None."""
        if name is None:
            if len(self.units) != 1:
                raise ConfigError(f"this system has {len(self.units)} units {sorted(self.units)}: "
                                  f"name the one you mean")
            name = next(iter(self.units))
        if name in self.units:
            return self.units[name]
        unit = _unit_at(self.units, name)
        if unit is not None:
            return unit
        known = sorted(name for name, _unit in _unit_items(self.units, leaves_only=False))
        raise ConfigError(f"unknown unit {name!r}; known: {known}")

    def sections_of(self, name: str) -> tuple[str, ...]:
        """Return every section containing ``name``; nested unit paths belong to ``units``."""
        sections = [section for section in NAMED if name in getattr(self, section)]
        if _unit_at(self.units, name) is not None and "units" not in sections:
            sections.append("units")
        return tuple(sections)

    @functools.cached_property
    def changes(self) -> tuple["Change", ...]:
        """Checked variable writes for every event, in time/file order."""
        return event_changes(self)

    @staticmethod
    def _bases(data: dict, base: Any, where: str) -> tuple[Any, Optional[dict[str, float]]]:
        """The system base and its pu scales, for the network (``None`` without a base: reported as missing)."""
        if not isinstance(data.get("base"), dict):
            return None, None
        base = _build(BaseValues, data["base"], "base", None, None)
        return base, base.parameter_scales()

    def replace(self, **path_values: Any) -> "Params":
        """Return a copy with dotted paths replaced, for example a loop-class gain path."""
        d = to_dict(self)
        assigned = set()
        free = ("simulation.initial.states.", "simulation.solver.subsystems.")  # dotted keys themselves
        for path, value in path_values.items():
            head = next((h for h in free if path.startswith(h)), None)
            if head is None and path.startswith("events.") and ".set." in path:
                head = path[:path.index(".set.") + 5]
            if head is not None:  # remainder is one key
                node = d
                for k in head.rstrip(".").split("."):
                    node = node.setdefault(k, {})
                node[path[len(head):]] = value
                continue
            node = d
            obj = self
            keys = path.split(".")
            for k in keys[:-1]:
                node = node.setdefault(k, {})
                obj = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
            canonical = quantity_name(type(obj), keys[-1]) if obj is not None else keys[-1]
            identity = (*keys[:-1], canonical)
            if identity in assigned:
                raise ConfigError(f"{path}: both actual and pu values specify the same parameter")
            assigned.add(identity)
            if canonical != keys[-1]:
                node.pop(canonical, None)
            node[keys[-1]] = value
        return from_dict(d)


# ------------------------------------------------------------------ construction from mappings

def from_dict(data: dict) -> Params:
    """Build, validate and resolve parameters and every event's explicit variable bindings."""
    from .validate import validate

    p = _build(Params, _normalize(data), "", None, None)
    validate(p)
    p = replace(p, sources={name: source.resolved(p.base) for name, source in p.sources.items()},
                units={name: _resolve_unit(unit, p.base)
                       for name, unit in p.units.items()})
    changes = p.changes
    for name, unit in _unit_items(p.units, leaves_only=False):
        object.__setattr__(unit, "events", tuple(events_for(changes, f"units.{name}")))
    return p


def _resolve_unit(unit: UnitParams, system_base: BaseValues) -> UnitParams:
    """Resolve ratings and defaults once through the configured unit tree."""
    spec = unit_type(unit.type)
    if not spec.composite:
        return unit.resolved(system_base)
    resolved = unit.resolved(system_base)
    child_base = resolved.base
    selected = replace(resolved._unit_fields, **{
        child: _resolve_unit(
            getattr(unit, child), child_base,
        )
        for child in spec.children
    })
    object.__setattr__(resolved, "_unit_fields", selected)
    return resolved


def to_dict(p: Any, derived: bool = False) -> dict:
    """Convert a dataclass tree to mappings, omitting non-init fields.

    Values filled from other parameters are written as ``None`` unless ``derived`` is true,
    so rebuilding the mapping derives them from any replacements again.
    """
    if isinstance(p, UnitParams):
        common = p._common_fields(composite=unit_type(p.type).composite)
        d = {
            "type": p.type,
            **to_dict(p._unit_fields, derived),
            **{item.name: to_dict(getattr(p, item.name), derived) for item in fields(p)
               if item.name in common or (item.name == "ctrl" and p.ctrl is not None)},
        }
    elif is_dataclass(p) and not isinstance(p, type):
        d = {f.name: to_dict(getattr(p, f.name), derived) for f in fields(p) if f.init}
        if isinstance(p, ControlParams):
            for name, loop in p.loops.items():
                direct = loop_key_type(name)
                if direct is not None and direct.type == loop.type:
                    d["loops"][name].pop("type", None)
    elif isinstance(p, dict):
        return {k: to_dict(v, derived) for k, v in p.items()}
    else:
        return p
    if not derived:
        for path in getattr(p, "derived_defaults", ()):
            *head, last = path.split(".")
            node = d
            for key in head:
                node = node[key]
            node[last] = None
    return d


def _numeric_string(value: Any) -> Any:
    """Convert a numeric string to float; return other values unchanged."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def flat_paths(values: Any, where: str = "", prefix: str = "") -> dict:
    """Flatten nested mappings to dotted paths; lists and scalars remain values."""
    if not isinstance(values, dict):
        raise ConfigError(f"{where}{'.' + prefix if prefix else ''}: expected a mapping")
    out: dict = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flat_paths(value, where, name))
        else:
            out[name] = value
    return out


def _normalize(data: dict) -> dict:
    """A copy of the mapping with nested initial states flattened to dotted names and numeric strings read."""
    data = deepcopy(data)
    if not isinstance(data, dict):
        return data  # type error reported by the construction
    simulation = data.get("simulation")
    init = simulation.get("initial") if isinstance(simulation, dict) else None
    if isinstance(init, dict) and init.get("states") is not None:
        init["states"] = {key: _numeric_string(value)
                          for key, value in flat_paths(init["states"], "simulation.initial.states").items()}
    events = data.get("events")
    if isinstance(events, dict):
        for name, event in events.items():
            if isinstance(event, dict) and "set" in event:
                event["set"] = flat_paths(event["set"], f"events.{name}.set")
    solver = simulation.get("solver") if isinstance(simulation, dict) else None
    subsystems = solver.get("subsystems") if isinstance(solver, dict) else None
    if isinstance(subsystems, dict):
        for name, value in subsystems.items():
            if isinstance(value, dict):
                if "step" in value:
                    value["step"] = _numeric_string(value["step"])
            else:
                subsystems[name] = _numeric_string(value)
    return data


def _is_optional(tp) -> tuple[bool, Any]:
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(get_args(tp)) == 2:
            return True, args[0]
    return False, tp


_TYPED = {
    "loop": (LOOP_TYPES, ""),
    "element": (ELEMENT_TYPES, ". Register a custom element type with "
                               "peslite.register_element_type before loading the file"),
    "event": (EVENT_TYPES, ". Register a custom event type with peslite.register_event_type before loading the file"),
}


def _build_control_params(data: Any, path: str, base: BaseValues,
                          scales: dict[str, float], unit_type: str) -> ControlParams:
    """Build the single controller parameter type using its selected reference schema."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    kind = data.get("type")
    spec = control_type(kind, f"{path}.type")
    if unit_type not in spec.unit_types:
        raise ConfigError(
            f"{path}.type: unit type {unit_type!r} cannot use control type {kind!r}"
        )
    values = dict(data)
    values["references"] = _build(
        spec.references,
        values.get("references", {}),
        f"{path}.references",
        base,
        scales,
    )
    return _build(ControlParams, values, path, base, scales)


def _build_unit_params(data: Any, path: str, base: BaseValues) -> UnitParams:
    """Build common unit fields and one explicitly scoped set of unique fields."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    kind = data.get("type", UnitParams.type)
    spec = unit_type(kind, f"{path}.type")
    common_names = UnitParams._common_fields(composite=spec.composite)
    unit_field_names = {item.name for item in fields(spec.params) if item.init}
    known = {"type", *common_names, *unit_field_names}
    if not spec.composite:
        known.add("ctrl")
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}; known: {sorted(known)}")

    unit_base, unit_scales = UnitParams._bases(data, base, path, spec)
    common = _build_fields(
        UnitParams,
        {name: data[name] for name in common_names if name in data},
        path,
        unit_base,
        unit_scales,
        names=common_names,
    )
    # Device-only conduction has no controller or PWM peripheral.
    diode_rectifier = not spec.composite and not common["bridge"].controlled
    if diode_rectifier:
        if common["bridge"].type is None:
            raise ConfigError(
                f"{path}.bridge.type: required when no devices are controlled"
            )
        from ..components.converter import make_topology
        topology = make_topology(common["bridge"])
        if not callable(getattr(topology, "initialize_model", None)):
            raise ConfigError(f"{path}.bridge: the selected topology has no natural-conduction model")
        for name in ("ctrl", "pwm"):
            if data.get(name) is not None:
                raise ConfigError(f"{path}.{name}: a diode full-bridge rectifier has no {name}")
    elif "pwm" in common_names and common.get("pwm") is None:
        raise ConfigError(f"{path}.pwm: required value missing")
    unit_fields = _build(
        spec.params,
        {name: data[name] for name in unit_field_names if name in data},
        path,
        unit_base,
        unit_scales,
    )
    if not spec.composite and not diode_rectifier:
        control = _build_control_params(
            data.get("ctrl"), f"{path}.ctrl", unit_base, unit_scales, kind
        )
        bridge = common["bridge"]
        common["bridge"] = replace(bridge, type=bridge.type or spec.bridge_for(control))
        common["ctrl"] = control
    return UnitParams(
        type=kind,
        _unit_fields=unit_fields,
        **common,
    )


def _typed_class(what: str, data: Any, path: str) -> type:
    registry, hint = _TYPED[what]
    kind = data.get("type") if isinstance(data, dict) else None
    if what == "loop":
        return loop_type(kind, f"{path}.type").Params
    if not isinstance(kind, str) or kind not in registry:
        raise ConfigError(f"{path}.type: unknown {what} type {kind!r}; known: {sorted(registry)}{hint}")
    return registry[kind].Params


def _loop_entry(name: str, item: Any, path: str) -> Any:
    """Resolve a class-qualified loop entry without creating a loop object.

    YAML parses a bare ``ClassName:`` entry as ``None``. In this one scope it is the concise
    spelling of an empty parameter mapping.
    """
    if item is None:
        item = {}
    if is_dataclass(item):
        item = to_dict(item)
    if not isinstance(item, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(item).__name__}")
    values = dict(item)
    direct = loop_key_type(name, path)
    selected = values.get("type")
    if direct is None:
        if selected is not None:
            values["type"] = loop_type(selected, f"{path}.type").type
        return values
    if selected is not None and loop_type(selected, f"{path}.type") is not direct:
        raise ConfigError(
            f"{path}.type: {selected!r} conflicts with loop class {name!r}"
        )
    values["type"] = direct.type
    return values


def _target_scales(event: Any, parent: dict, base: Any) -> Optional[dict]:
    """Use the target Unit port's base, following a composite's default port."""
    target = event.get("target") if isinstance(event, dict) else None
    units = parent.get("units")
    if not isinstance(target, str) or not isinstance(units, dict):
        return None
    parts = target.split(".")
    if not parts or not isinstance(units.get(parts[0]), dict):
        return None

    path = parts[0]
    unit = units[parts[0]]
    current_base = base
    index = 1
    while True:
        spec = unit_type(unit.get("type", UnitParams.type), f"units.{path}.type")
        current_base, scales = UnitParams._bases(unit, current_base, f"units.{path}", spec)
        if index < len(parts) and parts[index] in spec.children:
            child = parts[index]
            value = unit.get(child)
            if not isinstance(value, dict):
                return None
            path += f".{child}"
            unit = value
            index += 1
            continue
        if spec.composite:
            port = spec.default_port
            if port is None or port not in spec.children:
                raise ConfigError(
                    f"units.{path}: composite type {spec.name!r} needs a default port"
                )
            value = unit.get(port)
            if not isinstance(value, dict):
                raise ConfigError(f"units.{path}.{port}: default port Unit is missing")
            path += f".{port}"
            unit = value
            continue
        return scales


def _build_named(item_cls, data: Any, path: str, base: Any, scales: Optional[dict], parent: dict) -> dict:
    """Build a name -> entry mapping; names must be non-empty and contain no dots.

    ``item_cls`` is a dataclass, a raw type, or a registered typed entry (``loop``/``event``).
    """
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping of names to entries, got {type(data).__name__}")
    out = {}
    for name, item in data.items():
        name = str(name)
        if not name or "." in name:
            raise ConfigError(f"{path}: {name!r} is not a usable name (non-empty, no '.')")
        where = f"{path}.{name}"
        if item_cls == "unit":
            out[name] = _build_unit_params(item, where, base)
        elif item_cls in _TYPED:
            if item_cls == "loop":
                item = _loop_entry(name, item, where)
            elif is_dataclass(item):
                item = to_dict(item)
            here = (_target_scales(item, parent, base) if item_cls == "event" else None) or scales
            out[name] = _build(_typed_class(item_cls, item, where), item, where, base, here)
        else:
            out[name] = _build(item_cls, item, where, base, scales) if is_dataclass(item_cls) else item
    return out


def _boolean(value: Any, path: str) -> bool:
    """Use the same switch syntax for fields and runtime connection commands."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ConfigError(f"{path}: expected 1 (on) or 0 (off), got {value!r}")


def _build(cls, data: Any, path: str, base: Any, scales: Optional[dict]):
    """Build the dataclass ``cls`` from a mapping at ``path``: convert its inputs with ``scales`` (SI
    values of one pu; ``None``: no conversion), then check its fields strictly and build its sections.

    A class with ``_bases(data, base, path)`` sets the base and scales of its sections.
    """
    if data is None:
        raise ConfigError(f"{path}: section is required")
    if is_dataclass(data):
        data = to_dict(data)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    if hasattr(cls, "_bases"):
        base, scales = cls._bases(data, base, path)
    if scales is not None:
        data = convert_quantities(data, cls, scales, path)
        if hasattr(cls, "_scaled_defaults"):
            data = {**cls._scaled_defaults(data, scales), **data}
    kwargs = _build_fields(cls, data, path, base, scales)
    try:
        obj = cls(**kwargs)
    except ConfigError as exc:
        raise ConfigError(f"{path}.{exc}" if path else str(exc)) from None
    for name, choices in getattr(cls, "_choices", {}).items():
        if getattr(obj, name) not in choices:
            raise ConfigError(f"{path or cls.__name__}.{name}: {getattr(obj, name)!r} not in {choices}")
    return obj


def _build_fields(cls, data: dict, path: str, base: Any, scales: Optional[dict],
                  *, names: Optional[frozenset[str]] = None) -> dict:
    """Parse selected dataclass fields through the same rules as an ordinary section."""
    hints = get_type_hints(cls)
    selected = tuple(f for f in fields(cls) if f.init and (names is None or f.name in names))
    known = {f.name for f in selected}
    unknown = set(data) - known
    if unknown:
        hint = getattr(cls, "_unknown_hint", "")
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}; known: {sorted(known)}"
                          + (f". {hint}" if hint else ""))
    kwargs = {}
    for f in selected:
        key = f"{path}.{f.name}" if path else f.name
        tp = hints[f.name]
        optional, inner = _is_optional(tp)
        if f.name in data:
            value = data[f.name]
            if value is None:
                if not optional:
                    raise ConfigError(f"{key}: null is not allowed")
                kwargs[f.name] = None
            elif f.metadata.get("unit"):
                kwargs[f.name] = _build_unit_params(value, key, base)
            elif "types" in f.metadata:
                choices = f.metadata["types"]
                kind = value.get("type") if isinstance(value, dict) else None
                if kind not in choices:
                    raise ConfigError(
                        f"{key}.type: unknown type {kind!r}; known: {sorted(choices)}"
                    )
                kwargs[f.name] = _build(choices[kind], value, key, base, scales)
            elif is_dataclass(inner) and isinstance(inner, type):
                kwargs[f.name] = _build(inner, value, key, base, scales)
            elif get_origin(inner) in (dict, _Mapping):
                kwargs[f.name] = _build_named(f.metadata.get("entries", get_args(inner)[1]), value, key,
                                              base, scales, data)
            elif inner is float:
                if isinstance(value, str):  # YAML 1.1 reads "2.0e6" as a string
                    try:
                        value = float(value)
                    except ValueError:
                        raise ConfigError(f"{key}: expected a number, got {value!r}") from None
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ConfigError(f"{key}: expected a number, got {value!r}")
                kwargs[f.name] = float(value)
            elif inner is int:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                    raise ConfigError(f"{key}: expected an integer, got {value!r}")
                kwargs[f.name] = int(value)
            elif inner is bool:
                kwargs[f.name] = _boolean(value, key)
            elif inner is str:
                if f.name in getattr(cls, "_text", ()) and not isinstance(value, str):
                    raise ConfigError(f"{key}: expected text, got {value!r}")
                kwargs[f.name] = str(value)
            elif inner is dict:
                kwargs[f.name] = dict(value)
            else:
                kwargs[f.name] = value
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            # an omitted section: built from nothing, so that its pu defaults are converted too
            kwargs[f.name] = (_build(inner, {}, key, base, scales) if is_dataclass(inner) and not optional
                              else f.default_factory())  # type: ignore[misc]
        else:
            raise ConfigError(f"{key}: required value missing")
    return kwargs


# ------------------------------------------------------------------ set events

RUNTIME = (
    *(f"{section}.*.connected" for section in SWITCHABLE),
    "buses.*.c", "buses.*.r_d",
    "branches.*.l", "branches.*.r",
    "sources.*.v", "sources.*.f", "sources.*.angle", "sources.*.l", "sources.*.r",
    "units.*.ctrl.references.*",
    "units.*.ctrl.loops.*.** (not period)",
    "units.*.protection.**",
    "units.*.dclink.source.* (not type)",
    "elements.*.* (not type, bus or declared structural fields)",
)


@dataclass(frozen=True)
class Change:
    """One checked event; command-only writes do not need a parameter snapshot."""

    t: float
    event: str
    params: Any
    paths: tuple[str, ...]
    values: dict = field(default_factory=dict)
    connections: tuple[str, ...] = ()
    ramp: float = 0.0

    def touches(self, prefix: str) -> list[str]:
        return [path[len(prefix):] for path in self.paths if path.startswith(prefix)]


def runtime_changeable(path: str, p: Params) -> bool:
    """Whether a stored parameter or connection command may be written during a run."""
    if _connection_variable(p, path) is not None:
        return True
    section, name, rest = _split_path(p, path)
    keys = rest.split(".")
    if section == "buses":
        return keys in (["c"], ["r_d"])
    if section == "branches":
        return keys in (["l"], ["r"])
    if section == "sources":
        return len(keys) == 1 and keys[0] in ("v", "f", "angle", "l", "r")
    if section == "units":
        if keys[:2] == ["ctrl", "references"]:
            return len(keys) == 3
        if keys[:2] == ["ctrl", "loops"]:
            return len(keys) >= 4 and keys[3] not in ("type", "period")
        if keys[0] == "protection":
            return True
        return keys[:2] == ["dclink", "source"] and len(keys) == 3 and keys[2] != "type"
    if section == "elements":
        element = p.elements.get(name)
        field = element.__dataclass_fields__.get(rest) if element is not None else None
        return (field is not None and len(keys) == 1 and keys[0] != "type"
                and field.metadata.get("runtime", True)
                and keys[0] not in element_buses(element))
    return False


def _event_assignments(p: Params, event: Any):
    """Resolve every event through the same parameter and connection-command variables."""
    values = EVENT_TYPES[event.type].variables(event, p)
    if not isinstance(values, _Mapping) or not values:
        raise ConfigError("event variable bindings must be a nonempty mapping; nothing to change")
    for path, value in values.items():
        parts = path.split(".") if isinstance(path, str) else ()
        if len(parts) < 3 or parts[0] not in NAMED or not all(parts):
            raise ConfigError(f"{path}: cannot change during a run; an event needs a scoped variable")
        connection = _connection_variable(p, path)
        if connection is not None:
            value = _boolean(value, path)
            section, name = connection
            targets = _unit_leaves(p.units, name) if section == "units" else (name,)
            for target in targets:
                yield f"{section}.{target}.connected", value
            continue
        *head, name = parts
        owner = p
        try:
            for key in head:
                owner = owner[key] if isinstance(owner, dict) else getattr(owner, key)
        except (AttributeError, KeyError):
            raise ConfigError(f"{path}: unknown writable variable") from None
        scales = p.base.parameter_scales()
        if head[0] == "units":
            unit = p.unit(_unit_path(p.units, ".".join(head[1:])))
            scales = unit.base.parameter_scales(
                unit.dclink.vdc_ref if unit.dclink is not None else 1.0)
        target = quantity_name(type(owner), name)
        converted = convert_quantities({name: value}, type(owner), scales, path)
        yield ".".join((*head, target)), converted[target]


def _connection_variable(p: Params, path: str):
    """Identify a writable connection command, not an arbitrary field named connected."""
    section, _, rest = path.partition(".")
    name, _, field = rest.rpartition(".")
    if section not in SWITCHABLE or field != "connected":
        return None
    if section == "units":
        exists = _unit_at(p.units, name) is not None
    else:
        exists = name in getattr(p, section)
    return (section, name) if exists else None


def event_changes(p: Params) -> tuple[Change, ...]:
    """Bind every event to checked variable writes, in time/file order, once at assembly."""
    events = [(event.t, index, name, event)
              for index, (name, event) in enumerate(p.events.items())]
    if not events:
        return ()
    current = replace(p, events={})
    flat = None
    changed_by: dict[tuple[float, str], str] = {}
    switched: dict[str, tuple[str, bool]] = {}
    out: list[Change] = []
    for event_t, _index, name, event in sorted(events, key=lambda item: item[:2]):
        where = f"events.{name}"
        try:
            values = {}
            for path, value in _event_assignments(current, event):
                if path in values:
                    raise ConfigError(f"{path}: more than one binding writes the same variable")
                values[path] = value
            connections = tuple(path for path in values if _connection_variable(current, path) is not None)
            ramp = getattr(event, "ramp", 0.0)
            if connections and (isinstance(ramp, bool) or not isinstance(ramp, (int, float))
                                or not math.isfinite(ramp) or ramp < 0.0):
                raise ConfigError(f"{where}.ramp must be finite and >= 0")
            if not connections and ramp != 0.0:
                raise ConfigError(f"{where}.ramp requires a connection command")
            parameters = {path: value for path, value in values.items() if path not in connections}
            if parameters and flat is None:
                flat = flat_paths(to_dict(current, derived=True))
            after = current.replace(**parameters) if parameters else current
        except (ConfigError, AttributeError, KeyError) as exc:
            raise ConfigError(f"{where}: after this event, {exc}") from None
        new = flat_paths(to_dict(after, derived=True)) if parameters else flat
        changed = (path for path in new if new[path] != flat.get(path)) if parameters else ()
        paths = tuple(dict.fromkeys([*changed, *values]))
        for path in paths:
            if path not in connections and not runtime_changeable(path, after):
                raise ConfigError(f"{where}: {path} cannot change during a run; an event changes "
                                  f"{', '.join(RUNTIME)}")
            key = (event_t, path)
            if key in changed_by:
                action = "switched" if path in connections else "set"
                raise ConfigError(f"{where}: {path} is also {action} at t = {event_t} "
                                  f"by events.{changed_by[key]}")
            changed_by[key] = name
            if path in connections:
                previous = switched.get(path)
                if previous is not None and previous[1] == values[path]:
                    state = "connected" if values[path] else "disconnected"
                    raise ConfigError(f"{where}: {path} is already {state} by events.{previous[0]}")
                switched[path] = name, values[path]
        out.append(Change(event_t, name, after if parameters else None, paths, values, connections, ramp))
        current, flat = after, new
    return tuple(out)


# ------------------------------------------------------------------ files

_LOAD_PREPARERS: list[Callable[[str | Path], Any]] = []


def _register_load_preparer(function: Callable[[str | Path], Any]) -> None:
    """Register construction-time preparation performed before a file is parsed."""
    if function not in _LOAD_PREPARERS:
        _LOAD_PREPARERS.append(function)

def load(path: str | Path, initial: str | Path | None = None,
         initial_time: Optional[float] = None, **overrides: Any) -> Params:
    """Load a YAML/JSON configuration, optionally replace its initial state, and apply overrides.

    ``initial``: states CSV (last row, or the row at ``initial_time`` in s) or YAML/JSON.
    ``overrides``: dotted paths, e.g. ``load("case.pes", **{"simulation.t_end": 5.0})``.
    """
    for prepare in _LOAD_PREPARERS:
        prepare(path)
    p = from_dict(read_tree(path))
    if initial is not None:
        init = read_initial(initial, initial_time)
        merged = to_dict(p)
        merged["simulation"]["initial"] = init
        if merged["simulation"]["t_end"] <= init.get("t", 0.0):
            # keep the configured duration after the new start time
            merged["simulation"]["t_end"] = init.get("t", 0.0) + p.simulation.t_end
        p = from_dict(merged)
    if overrides:
        p = p.replace(**overrides)
    return p


def _read(path: Path) -> Any:
    """Read YAML or JSON configuration text."""
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
        raise ConfigError(f"{path}: reading YAML needs PyYAML (pip install PyYAML), or write the "
                          f"configuration as .json") from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_initial(path: str | Path, t: Optional[float] = None) -> dict:
    """Read an ``initial`` mapping (``t``, ``states``) from YAML/JSON or a states CSV.

    CSV: last row, or the row at ``t`` (s); empty/NaN cells are omitted.
    YAML/JSON: a simulation file's ``simulation.initial`` block or the mapping itself;
    ``t`` overrides its time.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        if len(rows) < 2 or rows[0][0] != "t":
            raise ConfigError(f"{path}: not a states table (first column must be 't')")
        header, body = rows[0], rows[1:]
        if t is None:
            row = body[-1]
        else:
            times = [float(r[0]) for r in body]
            k = min(range(len(times)), key=lambda i: abs(times[i] - t))
            if abs(times[k] - t) > 1e-9 * max(1.0, abs(t)):
                raise ConfigError(f"{path}: no row at t = {t} (nearest: {times[k]})")
            row = body[k]
        states = {}
        for name, cell in zip(header[1:], row[1:]):
            if cell.strip() == "":
                continue
            value = float(cell)
            if not math.isnan(value):
                states[name] = value
        return {"t": float(row[0]), "states": states}
    data = _read(path)
    if isinstance(data, dict) and isinstance(data.get("simulation"), dict):
        if "initial" not in data["simulation"]:
            raise ConfigError(f"{path}: a simulation file without a simulation.initial section")
        data = data["simulation"]["initial"]
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected an initial mapping (t, states)")
    if t is not None:
        data = {**data, "t": t}
    return data


def read_tree(path: str | Path) -> dict:
    """Return a top-level mapping from YAML or JSON configuration text."""
    data = _read(Path(path))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


_FILE_ORDER = ("base", "buses", "branches", "sources", "units", "elements", "events", "simulation", "meta")


def _written_out(p: Params) -> dict:
    """Return ``p`` in file order, including defaults derived from other values."""
    d = to_dict(p, derived=True)
    for name, u in _unit_items(p.units, leaves_only=False):
        keys = name.split(".")
        unit = d["units"][keys[0]]
        for key in keys[1:]:
            unit = unit[key]
        if unit["s_base"] is None:
            unit["s_base"] = u.base.s_base
        if "meas" in unit and u.ctrl is not None:
            meas = unit["meas"]
            if meas["period"] is None:
                meas["period"] = u.ctrl.period
            if meas["window"] is None and meas["average"]:
                meas["window"] = u.ctrl.period
    return _switches({key: d[key] for key in _FILE_ORDER})


def _switches(value: Any) -> Any:
    """Write switches as 1 or 0, including switches nested in mappings and lists."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, dict):
        return {key: _switches(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_switches(item) for item in value]
    return value


def dumps(p: Params) -> str:
    """Return a complete resolved simulation file as YAML (or JSON without PyYAML)."""
    d = _written_out(p)
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        return json.dumps(d, indent=2) + "\n"
    from .. import __version__
    header = (f"# PESLite {__version__} simulation file with every value written out (plant values in SI).\n"
              f"# null: not used, or derived when the case is built (for example loop gains from bandwidth).\n"
              f"# Run it with: peslite <this file>\n")
    return header + yaml.safe_dump(d, sort_keys=False)


def dump(p: Params, path: str | Path) -> None:
    """Write a complete resolved simulation file (see :func:`dumps`)."""
    Path(path).write_text(dumps(p), encoding="utf-8")
