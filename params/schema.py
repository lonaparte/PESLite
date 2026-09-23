"""Frozen parameter dataclasses and strict construction from mappings.

``build`` checks fields, types and choices; ``to_dict`` converts back, omitting derived fields.
"""

from __future__ import annotations

import math
import dataclasses
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

from ..phs.protocols import ConfigError
from ..phs.solvers import ADAPTIVE_METHODS, FIXED_METHODS

from .base import BaseValues, DCBase, quantity_name

__all__ = [
    "ConfigError", "FIXED_METHODS", "ADAPTIVE_METHODS", "build", "to_dict",
    "BusParams",
    "BranchParams",
    "SourceParams",
    "ACFilterParams",
    "DCLinkParams",
    "DCCapacitorParams", "DCSourceParams",
    "PWMParams",
    "MeasurementParams",
    "PLLParams",
    "CurrentLoopParams",
    "DCVoltageLoopParams",
    "PSCParams",
    "DroopParams",
    "VSGParams",
    "DVOCParams",
    "MatchingParams",
    "GFMParams",
    "ControlParams", "ReferenceParams",
    "ProtectionParams",
    "StartupParams",
    "SourceEventParams",
    "SetpointStepParams",
    "PLLGainStepParams",
    "UnitEventParams",
    "UnitParams",
    "SolverParams",
    "LogParams",
    "SimulationParams",
    "InitialParams",
    "OutputParams",
    "Params",
]





@dataclass(frozen=True)
class BusParams:
    """AC network node with a shunt capacitor to ground.

    ``c``: capacitance, F (must be > 0); ``r_d``: series damping resistor, ohm.
    pu inputs use the system base.
    """

    c: float
    r_d: float = 0.0

    _quantities = {"c": "capacitance", "r_d": "resistance"}


@dataclass(frozen=True)
class BranchParams:
    """Series R-L branch between two buses; current positive from ``from_bus`` to ``to_bus``.

    ``l`` in H, ``r`` in ohm; pu inputs use the system base. ``breaker``: protection may open it.
    """

    from_bus: str
    to_bus: str
    l: float
    r: float = 0.0
    breaker: bool = False

    _quantities = {"l": "inductance", "r": "resistance"}
    _input_aliases = {"x_pu": "l_pu"}


@dataclass(frozen=True)
class SourceParams:
    """Three-phase emf behind a series R-L impedance, connected to ``bus``.

    ``l`` in H, ``r`` in ohm, ``v`` in phase-peak V (default 1 pu); pu inputs use the system base.
    """

    bus: str
    l: float
    r: float = 0.0
    v: float = 0.0  # emf magnitude, phase-peak V
    events: "SourceEventParams" = field(default_factory=lambda: SourceEventParams())

    _quantities = {"l": "inductance", "r": "resistance", "v": "voltage"}
    _input_aliases = {"x_pu": "l_pu"}
    _defaults_pu = {"v": 1.0}


@dataclass(frozen=True)
class ACFilterParams:
    """Series R-L branch of the converter's AC filter (``l_f`` H, ``r_f`` ohm).

    pu inputs use the unit's AC base. The filter capacitor is set on the bus (:class:`BusParams`).
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
    """DC supply; ``type`` selects which fields apply (pu inputs use the unit DC base)."""

    type: str = "none"  # "none" | "current" | "voltage"
    i: float = 0.0  # current source, A, positive into the DC link
    k: float = 0.0  # current-source voltage droop, A/V
    v: float = 0.0  # voltage-source emf, V
    r: float = 0.0  # voltage-source series resistance, ohm

    _choices = {"type": ("none", "current", "voltage")}

    _quantities = {"i": "dc_current", "k": "dc_current/dc_voltage",
                   "v": "dc_voltage", "r": "dc_resistance"}


@dataclass(frozen=True)
class DCLinkParams:
    """DC link of one unit: rated voltage ``vdc_ref`` (V), optional capacitor and supply."""

    vdc_ref: float
    capacitor: Optional[DCCapacitorParams] = None
    source: DCSourceParams = field(default_factory=DCSourceParams)

@dataclass(frozen=True)
class PWMParams:
    """PWM settings: carrier, modulation method and duty-cycle update period.

    ``sync``: ``"asynchronous"`` (carrier on absolute time) or ``"synchronous"`` (carrier
    locked to the controller angle; requires an integer ``f_sw / base.f0``).
    """

    f_sw: float  # carrier frequency, Hz
    modulation_limit: float = 1.0
    carrier_phase: float = 0.0  # carrier position at t = 0, in carrier periods
    method: str = "spwm"  # "spwm" | "svpwm"
    sync: str = "asynchronous"  # "asynchronous" | "synchronous"
    update_period: Optional[float] = None  # s; default: one carrier period

    _choices = {"method": ("spwm", "svpwm"), "sync": ("asynchronous", "synchronous")}

    @property
    def switching_period(self) -> float:
        return 1.0 / self.f_sw

    @property
    def effective_update_period(self) -> float:
        """Return ``update_period`` (s), or one carrier period if it is not set."""
        return self.update_period if self.update_period is not None else self.switching_period

@dataclass(frozen=True)
class MeasurementParams:
    """ADC measurement settings.

    ``average`` (AC) and ``u_dc`` (DC): ``"instantaneous"`` or ``"window"`` (mean over
    ``window_s``, s, at most the PWM update period).
    """

    average: str = "instantaneous"
    window_s: Optional[float] = None  # s; default: the PWM update period
    u_dc: str = "instantaneous"

    _choices = {"average": ("instantaneous", "window"), "u_dc": ("instantaneous", "window")}


@dataclass(frozen=True)
class PLLParams:
    """Synchronous-reference-frame PLL, ``w = w0 + kp*eps + ki*int(eps)``.

    ``normalisation``: ``"rated"`` (``eps`` is the q-axis voltage in pu of rated voltage)
    or ``"amplitude"`` (``eps`` is normalised by the estimated voltage amplitude).
    """

    kp_pu: float
    ki_pu: float
    normalisation: str = "rated"

    _choices = {"normalisation": ("rated", "amplitude")}

    _quantities = {"kp_pu": "1/voltage", "ki_pu": "1/voltage"}


@dataclass(frozen=True)
class CurrentLoopParams:
    """dq PI current controller; gains default from the bandwidth ``bw_hz`` (Hz)."""

    bw_hz: float
    kp_pu: Optional[float] = None  # pu voltage / pu current
    ki_pu: Optional[float] = None  # pu voltage / (pu current * s)
    decoupling: bool = True
    feedforward: bool = True
    antiwindup: str = "conditional"  # "conditional" | "backcalc" | "none"

    _choices = {"antiwindup": ("conditional", "backcalc", "none")}

    _quantities = {"kp_pu": "resistance", "ki_pu": "resistance"}


@dataclass(frozen=True)
class DCVoltageLoopParams:
    """DC-voltage PI controller producing the d-axis current reference (pu)."""

    kp_pu: float
    ki_pu: float
    id0_export_pu: float = 0.0  # d-axis current feed-forward, pu
    bidirectional: bool = False  # False: id_ref limited to >= 0

    _quantities = {"kp_pu": "current/dc_voltage", "ki_pu": "current/dc_voltage",
                   "id0_export_pu": "current"}


@dataclass(frozen=True)
class PSCParams:
    """Power-synchronization control, ``w = w0 + k_p (P* - P)``."""

    k_p_pu: float  # rad/s per pu
    k_v: float = 0.0  # AVR proportional gain, pu/pu (0: open-loop magnitude)
    k_vi: float = 0.0  # AVR integral gain, 1/s
    r_a_pu: float = 0.0  # active-resistance gain, pu
    alpha_d: float = 0.0  # high-pass filter bandwidth for r_a, rad/s

    _quantities = {"k_p_pu": "1/power", "r_a_pu": "resistance"}


@dataclass(frozen=True)
class DroopParams:
    m_p_pu: float  # rad/s per pu (P-f droop)
    n_q_pu: float  # pu/pu (Q-V droop)

    _quantities = {"m_p_pu": "1/power", "n_q_pu": "voltage/power"}


@dataclass(frozen=True)
class VSGParams:
    """Virtual synchronous generator: swing equation and Q-V droop."""

    h_s: float  # inertia constant, s
    d_p_pu: float  # damping, pu power per pu speed deviation
    k_q_pu: float  # Q-V droop, pu/pu
    t_q: float = 0.0  # voltage-magnitude lag time constant, s (0: none)

    _quantities = {"d_p_pu": "power/frequency", "k_q_pu": "voltage/power"}


@dataclass(frozen=True)
class DVOCParams:
    """Dispatchable virtual oscillator control (dVOC)."""

    eta_pu: float  # rad/s
    alpha_pu: float  # magnitude regulation gain
    kappa_rad: float  # rotation of the current error, rad

    _quantities = {"eta_pu": "resistance", "alpha_pu": "1/resistance"}


@dataclass(frozen=True)
class MatchingParams:
    """Matching control, ``w = k_theta * v_dc``; ``k_theta_pu`` defaults to ``w0 / vdc_ref_pu``."""

    k_theta_pu: Optional[float] = None  # rad/s per pu DC voltage (DC base)
    k_q_pu: float = 0.0  # Q-V droop, pu/pu

    _quantities = {"k_theta_pu": "1/dc_voltage", "k_q_pu": "voltage/power"}


@dataclass(frozen=True)
class GFMParams:
    law: str = "psc"  # "psc" | "droop" | "vsg" | "dvoc" | "matching"
    p_ref_pu: float = 0.0
    q_ref_pu: float = 0.0
    v_ref_pu: float = 1.0
    power_filter_bw_hz: float = 0.0  # first-order filter on P and Q; <= 0 disables
    inner: str = "direct"  # "direct" | "virtual_admittance"
    r_v_pu: float = 0.0  # virtual impedance, pu
    x_v_pu: float = 0.0
    current_limit_pu: float = 0.0  # <= 0 disables the current limiter
    psc: Optional[PSCParams] = None
    droop: Optional[DroopParams] = None
    vsg: Optional[VSGParams] = None
    dvoc: Optional[DVOCParams] = None
    matching: Optional[MatchingParams] = None

    _choices = {
        "law": ("psc", "droop", "vsg", "dvoc", "matching"),
        "inner": ("direct", "virtual_admittance"),
    }

    _quantities = {"p_ref_pu": "power", "q_ref_pu": "power", "v_ref_pu": "voltage",
                   "r_v_pu": "resistance", "x_v_pu": "resistance", "current_limit_pu": "current"}


@dataclass(frozen=True)
class ReferenceParams:
    """Controller references, all electrical quantities in this unit's pu bases."""

    id_ref_pu: float = 0.0
    iq_ref_pu: float = 0.0
    p_ref_pu: float = 0.0
    q_ref_pu: float = 0.0
    v_ref_pu: float = 1.0
    vdc_ref_pu: float = 1.0
    theta: float = 0.0
    omega: Optional[float] = None  # rad/s; defaults to the system frequency
    zero_v_pu: float = 0.0

    _quantities = {"id_ref_pu": "current", "iq_ref_pu": "current", "p_ref_pu": "power",
                   "q_ref_pu": "power", "v_ref_pu": "voltage", "vdc_ref_pu": "dc_voltage",
                   "zero_v_pu": "voltage"}


@dataclass(frozen=True)
class ControlParams:
    """Control configuration: named loops, their signal connections and references.

    ``type`` selects the default wiring; ``connections`` and ``outputs`` override it.
    Each ``loops`` entry is validated against the schema registered for its ``type``.
    """

    type: str  # "gfl" | "gfm" | "custom"
    loops: dict[str, Any]
    connections: dict = field(default_factory=dict)  # input port -> output port
    outputs: dict = field(default_factory=dict)  # u_dq, theta, omega -> output port
    references: ReferenceParams = field(default_factory=ReferenceParams)
    sampling_period: Optional[float] = None  # ADC sampling period, s; default: pwm update period
    samples_per_update: int = field(init=False, default=1)

    _choices = {"type": ("gfl", "gfm", "custom")}


@dataclass(frozen=True)
class DelayParams:
    steps: int = 0  # one step is one PWM output update

@dataclass(frozen=True)
class ProtectionParams:
    """Trip and alarm thresholds; ``<= 0`` disables a criterion; times in s."""

    i_alarm_pu: float = -1.0  # instantaneous phase current, trips immediately
    vac_uv_pu: float = -1.0
    vac_ov_pu: float = -1.0
    freq_band_hz: float = -1.0
    vdc_band_pu: float = -1.0
    hold_s: float = 0.02
    rocof_alarm_hz_s: float = -1.0  # alarm only
    rocof_window_s: float = 0.1

    _quantities = {"i_alarm_pu": "current", "vac_uv_pu": "voltage",
                   "vac_ov_pu": "voltage", "vdc_band_pu": "dc_voltage"}


@dataclass(frozen=True)
class StartupParams:
    start: float = 0.0
    duration: float = 0.0  # s; ramp of the DC source and of id0

@dataclass(frozen=True)
class SourceEventParams:
    """Events applied to one source; times in s, ``t < 0`` disables a step."""

    freq_step_t: float = -1.0
    freq_step_hz: float = 0.0
    phase_jump_t: float = -1.0
    phase_jump_rad: float = 0.0
    voltage_step_t: float = -1.0
    voltage_step: float = 0.0  # phase-peak emf magnitude after the step, V

    _quantities = {"voltage_step": "voltage"}
    _defaults_pu = {"voltage_step": 1.0}


@dataclass(frozen=True)
class SetpointStepParams:
    """Step of a controller setpoint (GFM ``p_ref`` / ``q_ref`` / ``v_ref``)."""

    t: float = -1.0
    p_ref_pu: Optional[float] = None
    q_ref_pu: Optional[float] = None
    v_ref_pu: Optional[float] = None

    _quantities = {"p_ref_pu": "power", "q_ref_pu": "power", "v_ref_pu": "voltage"}


@dataclass(frozen=True)
class PLLGainStepParams:
    t: float = -1.0
    kp_after_pu: float = 0.0

    _quantities = {"kp_after_pu": "1/voltage"}


@dataclass(frozen=True)
class UnitEventParams:
    """Events applied to one converter unit."""

    startup: StartupParams = field(default_factory=StartupParams)
    setpoint: SetpointStepParams = field(default_factory=SetpointStepParams)
    pll_gain: PLLGainStepParams = field(default_factory=PLLGainStepParams)

@dataclass(frozen=True)
class UnitParams:
    """One converter unit: rating, plant components, control, protection and events.

    ``bus``: terminal bus. Plant fields are stored in SI; controller parameters are
    pu on the unit's rating ``s_base`` (VA) and DC base (``dclink.vdc_ref``).
    :meth:`L`, :meth:`R`, :meth:`C` convert pu on this unit's base to H, ohm, F.
    """

    bus: str
    ac_filter: ACFilterParams
    dclink: DCLinkParams
    pwm: PWMParams
    control: ControlParams
    delay: DelayParams = field(default_factory=DelayParams)
    s_base: Optional[float] = None  # rating (VA); default: the system base
    measurement: MeasurementParams = field(default_factory=MeasurementParams)
    protection: ProtectionParams = field(default_factory=ProtectionParams)
    events: UnitEventParams = field(default_factory=UnitEventParams)
    base: BaseValues = field(init=False, default=None)  # set by resolved()

    def resolved(self, system_base: BaseValues) -> UnitParams:
        """Return a copy with ``base``, timing defaults and ``omega`` filled in (after validation)."""
        pwm = replace(self.pwm, update_period=self.pwm.effective_update_period)
        references = self.control.references
        if references.omega is None:
            references = replace(references, omega=system_base.w0)
        control = replace(self.control, references=references)
        sampling_period = control.sampling_period if control.sampling_period is not None else pwm.update_period
        object.__setattr__(control, "samples_per_update",
                           max(1, int(round(pwm.update_period / sampling_period))))
        unit = replace(self, pwm=pwm, control=control)
        base = system_base if self.s_base is None else BaseValues(
            self.s_base, system_base.v_ll_rms, system_base.f0)
        object.__setattr__(unit, "base", base)
        return unit

    @property
    def dc_base(self) -> DCBase:
        """This unit's DC base (:class:`~peslite.params.base.DCBase`)."""
        return self.base.dc(self.dclink.vdc_ref)

    # pu (this unit's base) -> SI
    def L(self, x_pu: float) -> float:
        return self.base.L(x_pu)

    def R(self, r_pu: float) -> float:
        return self.base.R(r_pu)

    def C(self, c_pu: float) -> float:
        return self.base.C(c_pu)

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

    _choices = {"type": ("fixed", "adaptive")}

@dataclass(frozen=True)
class LogParams:
    plant_period: float = 5e-4  # plant snapshot period, s
    control_every: int = 1  # keep every n-th controller sample

@dataclass(frozen=True)
class SimulationParams:
    t_end: float
    bridge: str = "switching"  # "switching" | "averaged" | "step_averaged"
    solver: SolverParams = field(default_factory=SolverParams)
    log: LogParams = field(default_factory=LogParams)
    stop_on_trip: bool = True
    progress_every: float = 0.0  # s of simulated time between progress lines; 0: silent
    energy_check: str = "warn"  # "warn" | "strict" | "off": verify the energy declarations before the run

    _choices = {"bridge": ("switching", "averaged", "step_averaged"), "energy_check": ("warn", "strict", "off")}


@dataclass(frozen=True)
class InitialParams:
    """Start time ``t`` (s) and overrides of initial state values.

    ``states`` maps dotted state names to a number, ``[re, im]`` or ``.re``/``.im`` entries,
    or a keyword: ``source`` (bus source voltage at ``t``) or ``rated`` (unit DC reference).
    ``t`` must lie on every unit's PWM update grid.
    """

    t: float = 0.0
    states: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OutputParams:
    """Files written by ``SimulationResult.save``."""

    states: bool = True  # states.csv
    signals: bool = False  # plant.csv and control.csv
    energy: bool = False  # energy.csv


@dataclass(frozen=True)
class Params:
    """Complete configuration: system base, network elements, units and run settings.

    ``buses``, ``branches``, ``sources`` and ``units`` map names to elements; names
    are unique across all four and prefix the state names.
    """

    base: BaseValues  # system base, used for network pu inputs
    buses: dict[str, BusParams]
    units: dict[str, UnitParams]
    simulation: SimulationParams
    sources: dict[str, SourceParams] = field(default_factory=dict)
    branches: dict[str, BranchParams] = field(default_factory=dict)
    initial: InitialParams = field(default_factory=InitialParams)
    output: OutputParams = field(default_factory=OutputParams)
    meta: dict = field(default_factory=dict)

    def unit(self, name: Optional[str] = None) -> UnitParams:
        """Return the unit ``name``, or the only unit if ``name`` is None."""
        if name is None:
            if len(self.units) != 1:
                raise ConfigError(f"this system has {len(self.units)} units {sorted(self.units)}: "
                                  f"name the one you mean")
            name = next(iter(self.units))
        if name not in self.units:
            raise ConfigError(f"unknown unit {name!r}; known: {sorted(self.units)}")
        return self.units[name]

    @property
    def elements(self) -> dict[str, Any]:
        """All named elements (buses, branches, sources, units) in assembly order."""
        return {**self.buses, **self.branches, **self.sources, **self.units}

    # pu (system base) -> SI, for the network
    def L(self, x_pu: float) -> float:
        return self.base.L(x_pu)

    def R(self, r_pu: float) -> float:
        return self.base.R(r_pu)

    def C(self, c_pu: float) -> float:
        return self.base.C(c_pu)

    def replace(self, **path_values: Any) -> "Params":
        """Return a copy with dotted paths replaced, e.g. ``replace(**{"units.vsc.control.loops.pll.kp_pu": 20})``."""
        from .io import from_dict, to_dict  # deferred: io imports this module
        d = to_dict(self)
        assigned = set()
        free = ("initial.states.", "simulation.solver.subsystems.")  # keys that contain dots themselves
        for path, value in path_values.items():
            head = next((h for h in free if path.startswith(h)), None)
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
            values = vars(obj) if hasattr(obj, "__dict__") else {}
            canonical = quantity_name(type(obj), keys[-1], values) if obj is not None else keys[-1]
            identity = (*keys[:-1], canonical)
            if identity in assigned:
                raise ConfigError(f"{path}: both actual and pu values specify the same parameter")
            assigned.add(identity)
            if canonical != keys[-1]:
                node.pop(canonical, None)
            node[keys[-1]] = value
        return from_dict(d)


def _is_optional(tp) -> tuple[bool, Any]:
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(get_args(tp)) == 2:
            return True, args[0]
    return False, tp

def _build_named(item_cls, data: Any, path: str) -> dict:
    """Build a name -> entry mapping; names must be non-empty and contain no dots."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping of names to entries, got {type(data).__name__}")
    out = {}
    for name, item in data.items():
        name = str(name)
        if not name or "." in name:
            raise ConfigError(f"{path}: {name!r} is not a usable name (non-empty, no '.')")
        out[name] = _build(item_cls, item, f"{path}.{name}") if is_dataclass(item_cls) else item
    return out


def _build(cls, data: Any, path: str):
    """Recursively build dataclass ``cls`` from a mapping, validating strictly."""
    if data is None:
        raise ConfigError(f"{path}: section is required")
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls) if f.init}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}; known: {sorted(known)}")
    kwargs = {}
    for f in fields(cls):
        if not f.init:
            continue
        key = f"{path}.{f.name}" if path else f.name
        tp = hints[f.name]
        optional, inner = _is_optional(tp)
        if f.name in data:
            value = data[f.name]
            if value is None:
                if not optional:
                    raise ConfigError(f"{key}: null is not allowed")
                kwargs[f.name] = None
            elif is_dataclass(inner) and isinstance(inner, type):
                kwargs[f.name] = _build(inner, value, key)
            elif get_origin(inner) in (dict, _Mapping):
                kwargs[f.name] = _build_named(get_args(inner)[1], value, key)
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
                if isinstance(value, bool):
                    kwargs[f.name] = value
                elif isinstance(value, (int, float)) and value in (0, 1):
                    kwargs[f.name] = bool(value)
                else:
                    raise ConfigError(f"{key}: expected a boolean, got {value!r}")
            elif inner is str:
                kwargs[f.name] = str(value)
            elif inner is dict:
                kwargs[f.name] = dict(value)
            else:
                kwargs[f.name] = value
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            raise ConfigError(f"{key}: required value missing")
    obj = cls(**kwargs)
    for name, choices in getattr(cls, "_choices", {}).items():
        if getattr(obj, name) not in choices:
            raise ConfigError(f"{path or cls.__name__}.{name}: {getattr(obj, name)!r} not in {choices}")
    return obj

def build(root: type, data: dict, validate: Any = None) -> Any:
    """Construct ``root`` from a mapping; optionally apply a cross-field validator."""
    tree = _build(root, data, "")
    return tree if validate is None else validate(tree)

def to_dict(p: Any) -> dict:
    """Convert a dataclass tree to mappings, omitting non-init (derived) fields."""
    if is_dataclass(p) and not isinstance(p, type):
        return {f.name: to_dict(getattr(p, f.name)) for f in fields(p) if f.init}
    if isinstance(p, dict):
        return {k: to_dict(v) for k, v in p.items()}
    return p
