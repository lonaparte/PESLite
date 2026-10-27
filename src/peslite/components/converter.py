"""The converter's power stage: bridge models and the dc link.

Quantities are SI. Every bridge keeps the same AC/DC and held-modulation connections; a concrete
bridge also owns the timing which turns sampled controller commands into its modulation value.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, ClassVar, Mapping, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from ..control.blocks import A120, phases, smoothstep
from ..solver.energy import PowerPort, StoragePort
from ..solver.model import Bag, ConfigError, Empty, LocatedZero, OutputMethod, _real
from .network import RCNode, Terminal
from .pwm import TIME_EPS, CarrierComparison, Modulator, PWM, ZOH, make_unit_modulator

__all__ = ["ThreePhaseBridge", "SwitchingModel", "PWMAveragingModel", "AveragingModel",
           "HalfBridge", "FullBridge", "DCLink", "DCCapacitor",
           "DCCurrentSource", "DCVoltageSource",
           "HalfBridgeDiodeConduction",
           "make_topology", "make_bridge", "make_dclink"]


class _BridgeInp(Bag):
    __slots__ = ("q", "u_dc", "i_c")


class _BridgeOut(Bag):
    __slots__ = ("u_c", "i_dc")


class _RectifierInp(Bag):
    __slots__ = ("u_dc", "i_c", "u_hold")


class _HalfBridgeInp(Bag):
    __slots__ = (
        "q", "upper_current_ratio", "lower_current_ratio",
        "u_dc", "u_lower", "i_c", "loss_factor",
    )


class _HalfBridgeOut(Bag):
    __slots__ = ("u_c", "i_dc", "i_lower")


def _set_three_phase_dc_current(bridge, _t: float) -> None:
    bridge.out.i_dc = 1.5 * (bridge.inp.q * bridge.inp.i_c.conjugate()).real


def _set_scalar_dc_current(bridge, _t: float) -> None:
    bridge.out.i_dc = (bridge.inp.q * bridge.inp.i_c).real


def _set_half_bridge_currents(bridge, _t: float) -> None:
    q, current = bridge.inp.q, bridge.inp.i_c
    bridge.out.i_dc = (q * current).real
    bridge.out.i_lower = ((1.0 - q) * current).real


def _set_half_bridge_transferred_currents(bridge, _t: float) -> None:
    current = bridge.inp.i_c
    bridge.out.i_dc = (bridge.inp.upper_current_ratio * current).real
    bridge.out.i_lower = (bridge.inp.lower_current_ratio * current).real


def _set_half_bridge_voltage(bridge, _t: float) -> None:
    q = bridge.inp.q
    bridge.out.u_c = q * bridge.inp.u_dc + (1.0 - q) * bridge.inp.u_lower


def _leg_conduction(upper_drop, lower_drop, upper_gate, lower_gate):
    """Return R_on times each device current; equal device resistance factors out of KCL."""
    upper_positive = 0.5 * (upper_drop + abs(upper_drop))
    lower_negative = 0.5 * (lower_drop - abs(lower_drop))
    return (upper_drop - upper_positive + upper_gate * upper_positive,
            lower_drop - lower_negative + lower_gate * lower_negative)


def _common_reference(model, gates):
    """Eliminate a floating pole reference using the exact piecewise-linear device law.

    Each of n legs contributes two breakpoints. Their current sums bracket zero; interpolation
    between the nearest bracketing points is exact. No iterative solve or tolerance is needed.
    """
    offsets = tuple((factor * model.state.u_C).real for factor in model._voltage_factors)
    lower = model.inp.u_lower if model.lower_terminal is not None else 0.0
    upper = model.inp.u_dc
    upper_on = model.enabled * model._upper_controlled
    lower_on = model.enabled * model._lower_controlled

    def balance(reference):
        total = 0.0
        for offset, gate in zip(offsets, gates):
            first, second = _leg_conduction(upper - reference - offset, lower - reference - offset,
                                            upper_on * gate, lower_on * (1.0 - gate))
            total += first + second
        return total

    points = tuple(rail - offset for rail in (lower, upper) for offset in offsets)
    lo = hi = points[0]
    for point in points[1:]:
        lo = np.minimum(lo, point).item()
        hi = np.maximum(hi, point).item()
    minimum, maximum = lo, hi
    for point in points:
        value = balance(point)
        lo = np.maximum(lo, np.where(value >= 0.0, point, minimum).item()).item()
        hi = np.minimum(hi, np.where(value <= 0.0, point, maximum).item()).item()
    first, second = balance(lo), balance(hi)
    denominator = first - second
    return lo + first * (hi - lo) / denominator if denominator > 0.0 else 0.5 * (lo + hi)


class ThreePhaseBridge:
    """Three-leg topology used by the existing AC converter model."""

    type = "three_phase_bridge"
    __slots__ = ()
    duty_names = ("d_a", "d_b", "d_c")
    zero = 0j
    power_scale = 1.5
    neutral_duty = 0.5
    reverse_port = False
    unilateral = False
    upper_controlled = lower_controlled = True
    config_fields = frozenset()
    set_dc_current = staticmethod(_set_three_phase_dc_current)
    voltage_factors = (1.0, A120.conjugate(), A120)
    series_devices = 1

    @staticmethod
    def pole_gates(q, switching):
        values = phases(q)
        lo = np.minimum(np.minimum(values[0], values[1]), values[2]).item()
        hi = np.maximum(np.maximum(values[0], values[1]), values[2]).item()
        offset = -lo if switching else 0.5 * (1.0 - hi - lo)
        return tuple(value + offset for value in values)

    @classmethod
    def from_config(cls, _cfg: Any) -> "ThreePhaseBridge":
        return cls()

    @staticmethod
    def switching_state(states):
        a, b, c = states
        return (2.0 / 3.0) * (a + complex(-0.5, math.sqrt(3.0) / 2.0) * b
                              + complex(-0.5, -math.sqrt(3.0) / 2.0) * c)

    @staticmethod
    def pwm_state(duties):
        values = tuple(min(1.0, max(0.0, duty)) for duty in duties)
        return ThreePhaseBridge.switching_state(values)

    @staticmethod
    def command_duties(output):
        return output.d_abc


TOPOLOGY_TYPES: dict[str, type] = {}


def _register_topology(cls: type) -> type:
    """Register one bridge topology under its configuration name."""
    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not name:
        raise TypeError("a bridge topology needs a non-empty type")
    if name in TOPOLOGY_TYPES:
        raise ValueError(f"bridge topology {name!r} is already registered")
    TOPOLOGY_TYPES[name] = cls
    return cls


_register_topology(ThreePhaseBridge)


class _BridgeModel:
    """One bridge topology composed with one simulation form.

    Topology selection binds its power equation once during construction.  The
    output hot path has no topology or simulation-form conditionals.
    """

    state_prefix = "bridge"

    state_names: ClassVar[tuple[str, ...]] = ()
    storage: ClassVar[tuple[StoragePort, ...]] = ()
    has_switching_events: ClassVar[bool] = False
    outputs_need_inputs: ClassVar[bool] = True
    dirac: ClassVar[bool] = True
    models_blocked_diodes: ClassVar[bool] = False
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod("set_dc_current", inputs=("q", "i_c"), outputs=("i_dc",)),
        OutputMethod("set_ac_voltage", inputs=("q", "u_dc"), outputs=("u_c",)),
    )

    def __init__(self, topology=None, *, parallel_capacitance: float = 0.0,
                 on_resistance: float = 0.0, switching: bool = False) -> None:
        self.topology = topology if topology is not None else ThreePhaseBridge()
        self.state = Empty()
        zero = self.topology.zero
        self.inp = getattr(self.topology, "input_type", _BridgeInp)(
            q=zero, u_dc=0.0, i_c=zero)
        self.out = getattr(self.topology, "output_type", _BridgeOut)(u_c=zero, i_dc=0.0)
        self.output_stages = getattr(self.topology, "output_stages", type(self).output_stages)
        voltage = getattr(self.topology, "set_ac_voltage", None)
        if voltage is not None:
            self.set_ac_voltage = voltage.__get__(self, type(self))
        self.ports = (
            PowerPort("inp.u_dc", "out.i_dc"),
            PowerPort("out.u_c", "inp.i_c", self.topology.power_scale, -1.0),
        )
        self.dc_terminal = Terminal(self, "u_dc", "i_dc", 1, electrical_domain="dc")
        self.power_terminal = Terminal(self, "u_c", "i_c", -1, drives_voltage=True)
        self.lower_terminal = None
        self.set_dc_current = self.topology.set_dc_current.__get__(self, type(self))
        self.modulator = None
        self.state_owner = self
        self.continuous = True
        self.enabled = False
        initialize = getattr(self.topology, "initialize_model", None)
        if initialize is not None:
            initialize(self, parallel_capacitance)
        if (not math.isfinite(parallel_capacitance) or parallel_capacitance < 0.0
                or not math.isfinite(on_resistance) or on_resistance < 0.0):
            raise ConfigError("bridge capacitance and on_resistance must be finite and >= 0")
        if parallel_capacitance > 0.0:
            if on_resistance <= 0.0:
                raise ConfigError("bridge.parallel_capacitance requires positive on_resistance")
            self._capacitive_devices(parallel_capacitance, on_resistance, switching)
        elif on_resistance > 0.0:
            self.resistance = self.topology.series_devices * _real(on_resistance)
            self._source_voltage = self.set_ac_voltage
            self.set_ac_voltage = self._resistive_voltage
            self.dissipated_power = self._resistive_loss
            self.supplied_power = self._zero_power
            self.dirac = False
            self.output_stages = tuple(
                replace(stage, inputs=tuple(dict.fromkeys((*stage.inputs, "i_c"))))
                if "u_c" in stage.outputs else stage for stage in self.output_stages)

    def diode_blocks_reverse_current(self) -> bool:
        """Whether the bridge also constrains its external branch current."""
        return self.topology.unilateral and not self.state_names

    def _resistive_voltage(self, t: float) -> None:
        self._source_voltage(t)
        self.out.u_c -= self.resistance * self.inp.i_c

    def _resistive_loss(self) -> float:
        return (self.topology.power_scale * self.resistance
                * getattr(self.inp, "loss_factor", 1.0) * abs(self.inp.i_c) ** 2)

    def _zero_power(self) -> float:
        return 0.0

    def _capacitive_devices(self, capacitance, resistance, switching) -> None:
        """Compose finite device conductances with the existing capacitor and algebraic solver."""
        self.on_resistance = _real(resistance)
        self._switching_poles = switching
        self._voltage_factors = self.topology.voltage_factors
        reference = getattr(self.topology, "pole_reference", _common_reference)
        self._pole_reference = reference.__get__(self, type(self)) if len(self._voltage_factors) > 1 else None
        self._upper_controlled = self.topology.upper_controlled
        self._lower_controlled = self.topology.lower_controlled
        self.capacitor = RCNode(capacitance, 0.0, phases=3 if isinstance(self.topology.zero, complex) else 1)
        self.state, self.state_names = self.capacitor.state, self.capacitor.state_names
        self.storage = tuple(replace(port, flows=()) for port in self.capacitor.storage)
        old = self.out
        names = (*type(old).__slots__, "i_device", "power_loss")
        self.out = type("_CapacitiveOut", (Bag,), {"__slots__": names})(
            **{name: getattr(old, name) for name in type(old).__slots__})
        self.out.i_device = self.topology.zero
        inputs = ("q", "u_dc") + (("u_lower",) if self.lower_terminal else ())
        currents = ("i_dc", "i_device", "power_loss")
        if self.lower_terminal is not None:
            currents += ("i_lower",)
        self.output_stages = (
            OutputMethod("set_ac_voltage", (), ("u_c",)),
            OutputMethod("set_dc_current", inputs, currents),
        )
        self.set_ac_voltage = self._capacitor_voltage
        self.set_dc_current = self._device_currents
        self.rhs = self._capacitor_rhs
        self.dissipated_power = self._device_loss
        self.supplied_power = self._zero_power
        self.dirac = False
        self.models_blocked_diodes = True

    def _capacitor_voltage(self, t: float) -> None:
        self.out.u_c = self.state.u_C

    def _device_currents(self, t: float) -> None:
        """One leg law for gated devices and their anti-parallel diodes in every topology."""
        voltage = self.state.u_C
        lower = self.inp.u_lower if self.lower_terminal is not None else 0.0
        gates = self.topology.pole_gates(self.inp.q, self._switching_poles)
        gates = tuple(np.minimum(1.0, np.maximum(0.0, gate)).item() for gate in gates)
        intervals = ((1.0, gates),) if self._switching_poles else CarrierComparison.state_fractions(gates)
        upper_on = self.enabled * self._upper_controlled
        lower_on = self.enabled * self._lower_controlled
        current, upper_current, lower_current, loss = self.topology.zero, 0.0, 0.0, 0.0
        for weight, states in intervals:
            neutral = self._pole_reference(states) if self._pole_reference is not None else 0.0
            for factor, gate in zip(self._voltage_factors, states):
                pole = neutral + (factor * voltage).real
                upper_drop, lower_drop = self.inp.u_dc - pole, lower - pole
                upper, lower_i = _leg_conduction(upper_drop, lower_drop,
                                                 upper_on * gate, lower_on * (1.0 - gate))
                upper /= self.on_resistance
                lower_i /= self.on_resistance
                current += weight * factor.conjugate() * (upper + lower_i)
                upper_current += weight * upper
                lower_current += weight * lower_i
                loss += weight * (upper_drop * upper + lower_drop * lower_i)
        self.out.i_device = current / self.topology.power_scale
        self.out.i_dc = upper_current
        if self.lower_terminal is not None:
            self.out.i_lower = lower_current
        self.out.power_loss = loss

    def _capacitor_rhs(self, t: float):
        self.capacitor.inp.i = self.out.i_device - self.inp.i_c
        return self.capacitor.rhs(t)

    def _device_loss(self) -> float:
        return self.out.power_loss

    def set_ac_voltage(self, t: float) -> None:
        self.out.u_c = self.inp.q * self.inp.u_dc

    def set_outputs(self, t: float) -> None:
        """Evaluate dc current and ac voltage from the current inputs."""
        self.set_dc_current(t)
        self.set_ac_voltage(t)

    def use_current_transfer_ratios(self) -> None:
        """Drive both half-bridge rail currents from explicit conduction ratios."""
        configure = getattr(self.topology, "use_current_transfer_ratios", None)
        if configure is None:
            raise TypeError("the selected topology has no independent rail-current transfer ratios")
        configure(self)

    def rhs(self, t: float):
        return ()

    def zoh_connections(self, label: str) -> dict:
        """An unclocked model has no held modulation input."""
        return {}

    @property
    def on(self) -> bool:
        return self.enabled

    @property
    def event_periods(self) -> list[float]:
        return []

    @property
    def next_control_time(self) -> float:
        return math.inf

    @property
    def control_index(self) -> int:
        return 0

    def previous_control_time(self) -> float:
        return -math.inf

    def control_count(self, t_a: float, t_b: float) -> int:
        return 0

    def next_time(self) -> float:
        return math.inf

    def control_due(self, t: float) -> bool:
        return False

    def fast_edge(self, t: float) -> bool:
        return False

    def reset(self, duties: NDArray[np.float64], on: bool = False) -> None:
        self.enabled = bool(on)

    def start(self, t: float, sync: tuple[float | None, float | None], continued: bool):
        return self.topology.zero

    def actuate(self, t: float, command: tuple[float, Any] | None,
                enabled: bool) -> list[Any]:
        self.enabled = bool(enabled)
        return []

    def describe(self) -> str:
        return "ideal averaging, continuous control"

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if values:
            raise KeyError(f"unclocked model has no states, got {sorted(values)}")


class _SampledModel(_BridgeModel, PWM):
    """Shared sampled timing for exact-switching and PWM-averaged forms."""

    state_prefix = "pwm"

    on = PWM.on
    reset = PWM.reset
    start = PWM.start
    get_state = PWM.get_state
    set_state = PWM.set_state
    describe = PWM.describe

    def zoh_connections(self, label: str) -> dict:
        return {(self, "q"): label}

    def __init__(self, period: float, load_period: float, offset: float, computation: float,
                 carrier_period: float, modulator: Modulator, topology=None, *,
                 pattern_sync: str = "interval",
                 parallel_capacitance: float = 0.0, on_resistance: float = 0.0) -> None:
        self.pulse_update = None
        _BridgeModel.__init__(self, topology, parallel_capacitance=parallel_capacitance,
                              on_resistance=on_resistance, switching=not isinstance(modulator, ZOH))
        registers = getattr(modulator, "registers", None)
        PWM.__init__(
            self,
            period,
            load_period,
            offset,
            computation,
            carrier_period,
            modulator,
            pattern_sync=pattern_sync,
            duty_names=(self.topology.duty_names + tuple(getattr(modulator, "carried", ()))
                        if registers is None else tuple(registers)),
            switching_state=self.topology.switching_state,
            pwm_state=self.topology.pwm_state,
        )
        self.continuous = False
        self.command_registers = (self.topology.command_duties if registers is None
                                else modulator.command_registers)
        if registers is None and getattr(modulator, "carried", ()):
            names = modulator.carried
            self.command_registers = lambda output: (*self.topology.command_duties(output),
                                                       *(getattr(output, name) for name in names))
        self.neutral_registers = (
            tuple(modulator.neutral_registers()) if registers is not None
            else (self.topology.neutral_duty,) * len(self.topology.duty_names)
                 + (0.0,) * len(getattr(modulator, "carried", ()))
        )
        self.reset(self.neutral_registers, on=False)
        self.state_owner = self
        self.has_switching_events = not isinstance(modulator, ZOH)

    @property
    def event_periods(self) -> list[float]:
        """Periods which can change this bridge's sampled-data map."""
        periods = [self.period, self.load_period]
        if self.has_switching_events:
            periods.append(self.carrier_period)
        return periods

    def next_time(self) -> float:
        """Next controller, register-load or internal switching event."""
        return min(self.t_interrupt, self.t_load, self.next_switch)

    @property
    def next_control_time(self) -> float:
        return self.t_interrupt

    @property
    def control_index(self) -> int:
        return self.k

    def previous_control_time(self) -> float:
        return self.interrupt(self.k - 1)

    def control_count(self, t_a: float, t_b: float) -> int:
        return self.count(t_a, t_b)

    def control_due(self, t: float) -> bool:
        return abs(self.t_interrupt - t) < TIME_EPS

    def fast_edge(self, t: float) -> bool:
        """Whether the old bridge output interval ends at ``t``."""
        return self.edge(t)

    def actuate(self, t: float, command: tuple[float, Any] | None,
                enabled: bool) -> list[Any]:
        """Accept one sampled command and return bridge inputs which become active at ``t``.

        The bridge owns the computation/load ordering. The converter unit deliberately does not
        know whether the implementation uses compare registers, a delayed averaged command or no
        carrier.
        """
        changes: list[Any] = []
        if command is not None and self.computation == 0.0:
            at, output = command
            self.write(at, self.command_registers(output), bool(output.gates) and enabled,
                       output.theta, output.omega)
        if abs(self.t_load - t) < TIME_EPS:
            changes.append(self.load(t))
        if command is not None and self.computation > 0.0:
            at, output = command
            self.write(at, self.command_registers(output), bool(output.gates) and enabled,
                       output.theta, output.omega)
        changes.extend(self.switches(t))
        return changes


class SwitchingModel(_SampledModel):
    """A bridge topology evaluated at every exact semiconductor switching edge."""


class PWMAveragingModel(_SampledModel):
    """A bridge topology driven by duty ratios held for one PWM interval."""


class AveragingModel(_BridgeModel):
    """Ideal averaged bridge driven directly by the continuous controller output.

    There is no timer, command history, held modulation input or delay state: the controller and
    power stage are evaluated in the same Model RHS plan. A unilateral half bridge may still use
    the configured physical switching period in its external branch's cycle-averaged DCM law;
    that parameter describes conduction physics and does not create simulation events.
    """


def make_topology(cfg: Any):
    """Build the bridge topology selected by ``cfg.type``."""
    return topology_type(cfg.type).from_config(cfg)


def topology_type(name: str, where: str = "bridge.type") -> type:
    """Return a registered bridge topology class by configuration name."""
    try:
        return TOPOLOGY_TYPES[name]
    except KeyError:
        raise ConfigError(
            f"{where}: unknown bridge topology {name!r}; known: {sorted(TOPOLOGY_TYPES)}"
        ) from None


def make_bridge(cfg: Any, modulator: Modulator | None = None, topology=None) -> _BridgeModel:
    """Compose a bridge topology with switching, PWM or averaging behavior."""
    topology = topology if topology is not None else make_topology(cfg.bridge)
    impedance = {name: getattr(cfg.bridge, name, 0.0)
                 for name in ("parallel_capacitance", "on_resistance")}
    if not getattr(topology, "controlled", True):
        if modulator is not None:
            raise ValueError("a diode full-bridge rectifier does not use a PWM modulator")
        return _BridgeModel(topology, switching=True, **impedance)
    pwm = cfg.pwm
    if cfg.bridge.model == "averaging":
        if modulator is not None:
            raise ValueError("an averaging bridge does not use a PWM modulator")
        return AveragingModel(topology, **impedance)
    model_type = SwitchingModel if cfg.bridge.model == "switching" else PWMAveragingModel
    return model_type(
        cfg.ctrl.period, pwm.load_period, pwm.grid_offset, cfg.ctrl.computation,
        pwm.switching_period,
        modulator if modulator is not None else make_unit_modulator(cfg),
        topology,
        pattern_sync=getattr(pwm, "pattern_sync", "interval"),
        **impedance,
    )


@_register_topology
class HalfBridge:
    """One scalar switching leg between upper and lower DC rails.

    Device placement is fixed once at construction.  Setting ``upper_controlled``
    or ``lower_controlled`` to false leaves only that device's diode available.
    The simulation hot path therefore uses the same two-bit gate word for
    diode-assisted and fully controlled legs.
    """

    input_type = _HalfBridgeInp
    output_type = _HalfBridgeOut
    set_ac_voltage = staticmethod(_set_half_bridge_voltage)
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod(
            "set_dc_current", inputs=("q", "i_c"), outputs=("i_dc", "i_lower")
        ),
        OutputMethod(
            "set_ac_voltage", inputs=("q", "u_dc", "u_lower"), outputs=("u_c",)
        ),
    )
    _transferred_current_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod(
            "set_dc_current",
            inputs=("upper_current_ratio", "lower_current_ratio", "i_c"),
            outputs=("i_dc", "i_lower"),
        ),
        OutputMethod(
            "set_ac_voltage", inputs=("q", "u_dc", "u_lower"), outputs=("u_c",)
        ),
    )


    def initialize_model(self, model, parallel_capacitance: float) -> None:
        model.inp.loss_factor = 1.0
        model.ports += (PowerPort("inp.u_lower", "out.i_lower"),)
        model.lower_terminal = Terminal(model, "u_lower", "i_lower", 1)

    def use_current_transfer_ratios(self, model) -> None:
        model.set_dc_current = _set_half_bridge_transferred_currents.__get__(model, type(model))
        model.output_stages = self._transferred_current_stages

    type = "half_bridge"
    duty_names = ("d",)
    zero = 0.0
    power_scale = 1.0
    neutral_duty = 0.0
    config_fields = frozenset((
        "upper_controlled", "lower_controlled", "reverse_port", "external_terminal",
    ))
    set_dc_current = staticmethod(_set_half_bridge_currents)
    voltage_factors = (1.0,)
    series_devices = 1

    @staticmethod
    def pole_gates(q, switching):
        return (q,)
    __slots__ = ("_gate_mask", "_reverse_port", "_external_terminal")

    def switching_state(self, states) -> float:
        state = states[0]
        return 1.0 - state if self._reverse_port else state

    def pwm_state(self, duties) -> float:
        duty = min(1.0, max(0.0, duties[0]))
        return 1.0 - duty if self._reverse_port else duty

    @staticmethod
    def command_duties(output):
        return (output.duty,)

    def __init__(self, *, upper_controlled: bool = True,
                 lower_controlled: bool = True,
                 reverse_port: bool = False,
                 external_terminal: str = "midpoint") -> None:
        if any(type(value) is not bool for value in (
                upper_controlled, lower_controlled, reverse_port)):
            raise TypeError("controlled-device and port-orientation flags must be bool")
        if not upper_controlled and not lower_controlled:
            raise ConfigError("a half bridge needs at least one controlled device")
        if reverse_port and not lower_controlled:
            raise ConfigError("a reverse half bridge needs a controlled lower device")
        if not reverse_port and not upper_controlled:
            raise ConfigError("a forward half bridge needs a controlled upper device")
        if external_terminal not in ("midpoint", "lower_rail"):
            raise ConfigError(
                "a half bridge external_terminal must be 'midpoint' or 'lower_rail'"
            )
        if external_terminal == "lower_rail" and reverse_port:
            raise ConfigError(
                "a lower-rail external terminal uses the forward half-bridge orientation"
            )
        self._gate_mask = int(upper_controlled) | (int(lower_controlled) << 1)
        self._reverse_port = reverse_port
        self._external_terminal = external_terminal

    @classmethod
    def from_config(cls, cfg: Any) -> "HalfBridge":
        return cls(
            upper_controlled=cfg.upper_controlled,
            lower_controlled=cfg.lower_controlled,
            reverse_port=cfg.reverse_port,
            external_terminal=cfg.external_terminal,
        )

    def state_from_duty(self, duty: float) -> float:
        """Return the bridge output ratio produced by its controlled device's duty."""
        return self.pwm_state((duty,))

    @property
    def reverse_port(self) -> bool:
        """Whether the external scalar port is on the bridge's input side."""
        return self._reverse_port

    @property
    def external_terminal(self) -> str:
        """The half-bridge terminal exposed as the converter's external bus."""
        return self._external_terminal

    @property
    def unilateral(self) -> bool:
        """Whether one device position is diode-only and therefore blocks reverse current."""
        return self.upper_controlled != self.lower_controlled

    def control_values(self, current, voltage, supply):
        """Return canonical control measurements and the topology's feed-forward duty."""
        current, voltage = np.real(current), np.real(voltage)
        if self._external_terminal == "lower_rail":
            voltage = -voltage
            denominator = supply + voltage
            feedforward = voltage / denominator if denominator > 0.0 else 0.0
        elif self._reverse_port:
            feedforward = 1.0 - supply / voltage if voltage > 0.0 else 0.0
        else:
            feedforward = voltage / supply if supply > 0.0 else 0.0
        return current, voltage, supply, feedforward

    @property
    def upper_controlled(self) -> bool:
        """Whether the upper position contains a controlled device."""
        return bool(self._gate_mask & 1)

    @property
    def lower_controlled(self) -> bool:
        """Whether the lower position contains a controlled device."""
        return bool(self._gate_mask & 2)

class _DiodeConductionResult(NamedTuple):
    controlled: Any
    diode: Any
    total: Any
    resistance_factor: Any
    derivative: Any
    in_dcm: Any
    upper_current_ratio: Any
    lower_current_ratio: Any


class _ConductionInp(Bag):
    __slots__ = ("current", "q", "upper", "lower", "reference")


class _ConductionOut(Bag):
    __slots__ = (
        "controlled", "diode", "total", "resistance_factor", "derivative",
        "in_dcm", "upper_current_ratio", "lower_current_ratio", "maximum_step",
    )


class HalfBridgeDiodeConduction:
    """Stateless cycle-averaged conduction relation for a unilateral half bridge.

    One evaluation supplies the branch derivative, bridge current-transfer ratio, copper-loss
    factor and local resolution request.  It is assembled as an ordinary algebraic subsystem so
    every consumer has an explicit model dependency and generated backends trace the same law.
    """

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = True
    observer: ClassVar[bool] = True
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod(
            "set_outputs",
            inputs=("current", "q", "upper", "lower", "reference"),
            outputs=(
                "controlled", "diode", "total", "resistance_factor", "derivative",
                "in_dcm", "upper_current_ratio", "lower_current_ratio", "maximum_step",
            ),
        ),
    )

    def __init__(self, switching_period: float, inductance: float, resistance: float,
                 topology: HalfBridge | None = None) -> None:
        self.period = switching_period
        self.inductance = inductance
        self.resistance = resistance
        self.topology = topology if topology is not None else HalfBridge(lower_controlled=False)
        if not isinstance(self.topology, HalfBridge) or not self.topology.unilateral:
            raise TypeError("diode conduction requires a unilateral half-bridge topology")
        self.state = Empty()
        self.inp = _ConductionInp(
            current=0.0, q=0.0, upper=0.0, lower=0.0, reference=0.0,
        )
        self.out = _ConductionOut(
            controlled=0.0,
            diode=0.0,
            total=1.0,
            resistance_factor=1.0,
            derivative=0.0,
            in_dcm=False,
            upper_current_ratio=0.0,
            lower_current_ratio=0.0,
            maximum_step=math.inf,
        )

    def equations(self, current: Any, q: Any, upper: Any, lower: Any,
                  reference: Any) -> _DiodeConductionResult:
        """Evaluate conduction between explicit upper, lower and midpoint-reference rails."""
        current = max(0.0, current)
        reverse = self.topology.reverse_port
        duty = 1.0 - q if reverse else q
        direction = -1.0 if reverse else 1.0
        controlled_rail = lower if reverse else upper
        diode_rail = upper if reverse else lower
        controlled_voltage = direction * (controlled_rail - reference)
        diode_voltage = direction * (diode_rail - reference)
        bias = controlled_voltage
        denominator = 2.0 * self.inductance + self.resistance * duty * self.period
        critical = duty * self.period * bias / denominator
        in_dcm = duty > 0.0 and duty < 1.0 and bias > 0.0 and current < critical
        if in_dcm:
            total = max(duty, current / critical)
            controlled = duty / total
            diode = max(0.0, total - duty) / total
            resistance_factor = 4.0 / (3.0 * total)
        else:
            total = 1.0
            controlled = duty
            diode = 1.0 - duty
            resistance_factor = 1.0
        dcm_derivative = total * (
            controlled * controlled_voltage
            + diode * diode_voltage
            - self.resistance * resistance_factor * current
        ) / self.inductance
        upper_current_ratio = diode if reverse else controlled
        lower_current_ratio = controlled if reverse else diode
        derivative = max(0.0, dcm_derivative) if current == 0.0 else dcm_derivative
        return _DiodeConductionResult(
            controlled, diode, total, resistance_factor, derivative, in_dcm,
            upper_current_ratio, lower_current_ratio,
        )

    def set_outputs(self, _t: float) -> None:
        result = self.equations(
            self.inp.current, self.inp.q, self.inp.upper, self.inp.lower,
            self.inp.reference,
        )
        out = self.out
        (out.controlled, out.diode, out.total, out.resistance_factor,
         out.derivative, out.in_dcm, out.upper_current_ratio,
         out.lower_current_ratio) = result
        out.maximum_step = self.period / 10.0 if result.in_dcm else math.inf

    def rhs(self, _t: float):
        return ()

    def diode_maximum_step(self) -> float:
        """Return the state-dependent resolution limit produced by the latest evaluation."""
        return self.out.maximum_step


@_register_topology
class FullBridge:
    """Two half bridges forming one scalar H-bridge.

    By default all four devices are controlled. With ``upper_controlled`` and ``lower_controlled``
    both false only the anti-parallel diodes remain: the bridge is a diode full-bridge rectifier
    without a controller or PWM, optionally with
    ``parallel_capacitance`` across its AC terminals.
    """

    zero_current_tolerance = 1e-6  # A, a blocked pair's located-current tolerance

    type = "full_bridge"
    duty_names = ("d_left", "d_right")
    zero = 0.0
    power_scale = 1.0
    neutral_duty = 0.5
    config_fields = frozenset(("upper_controlled", "lower_controlled"))
    set_dc_current = staticmethod(_set_scalar_dc_current)
    voltage_factors = (0.5, -0.5)
    series_devices = 2

    @staticmethod
    def pole_gates(q, switching):
        if switching:
            return np.maximum(q, 0.0).item(), np.maximum(-q, 0.0).item()
        return 0.5 * (1.0 + q), 0.5 * (1.0 - q)

    @staticmethod
    def pole_reference(model, gates):
        """Eliminate the common voltage of two identical legs from their symmetry."""
        reference = 0.5 * model.inp.u_dc
        if not model.topology.controlled:
            return reference
        # Equal gate states connect two devices to the same rail. At high differential
        # voltage one opposite diode also conducts, adding a third equal conductance.
        zero_vector = model.enabled * np.where(gates[0] == gates[1], 1.0, 0.0).item()
        excess = np.maximum(0.5 * abs(model.state.u_C) - model.inp.u_dc, 0.0).item() / 3.0
        zero_reference = gates[0] * model.inp.u_dc + (2.0 * gates[0] - 1.0) * excess
        return reference + zero_vector * (zero_reference - reference)
    __slots__ = ("controlled",)

    @property
    def upper_controlled(self) -> bool:
        return self.controlled

    lower_controlled = upper_controlled

    def __init__(self, *, upper_controlled: bool = True, lower_controlled: bool = True) -> None:
        if type(upper_controlled) is not bool or type(lower_controlled) is not bool:
            raise TypeError("controlled-device flags must be bool")
        if upper_controlled != lower_controlled:
            raise ConfigError("a full bridge has either four controlled devices or none "
                              "(upper_controlled and lower_controlled must be equal)")
        self.controlled = upper_controlled or lower_controlled

    @staticmethod
    def switching_state(states) -> float:
        return states[0] - states[1]

    @staticmethod
    def pwm_state(duties) -> float:
        left = min(1.0, max(0.0, duties[0]))
        right = min(1.0, max(0.0, duties[1]))
        return left - right

    @staticmethod
    def command_duties(output):
        duty = output.duty
        return duty, 1.0 - duty

    @classmethod
    def from_config(cls, cfg: Any) -> "FullBridge":
        upper = getattr(cfg, "upper_controlled", True)
        lower = getattr(cfg, "lower_controlled", True)
        return cls(upper_controlled=upper, lower_controlled=lower)

    def state_from_duty(self, duty: float) -> float:
        """Return the differential bridge ratio produced by a bipolar duty command."""
        return 2.0 * min(1.0, max(0.0, duty)) - 1.0

    reverse_port = False
    unilateral = False

    @staticmethod
    def control_values(current, voltage, supply):
        """Return canonical control measurements and the bipolar feed-forward duty."""
        current, voltage = np.real(current), np.real(voltage)
        feedforward = 0.5 * (1.0 + voltage / supply) if supply > 0.0 else 0.5
        return current, voltage, supply, feedforward

    def initialize_model(self, model, parallel_capacitance: float) -> None:
        """Bind the configured full bridge's device equations to its execution model."""
        if self.controlled or parallel_capacitance > 0.0:
            return
        model.enabled = True
        model.inp = _RectifierInp(u_dc=0.0, i_c=0.0, u_hold=0.0)
        # without Cp a blocked bridge needs its port's hold voltage (wired by the assembly)
        model.power_terminal = replace(model.power_terminal, hold_voltage_input="u_hold")
        model.output_stages = (
            OutputMethod("set_dc_current", inputs=("u_dc", "i_c"), outputs=("i_dc",)),
            OutputMethod("set_ac_voltage",
                         inputs=("u_dc", "u_hold"),
                         outputs=("u_c",)),
        )
        model.dirac = False
        model._current = LocatedZero()  # sign: the conducting direction, 0 while blocked
        model.commutations = 0
        model.blocked_time = 0.0
        model._blocked_since = math.nan
        model.discrete_state_names = ("direction",)
        model.linearization_state_names = ()
        model.dissipated_power = model._zero_power
        model.supplied_power = model._zero_power
        methods = (
            ("set_dc_current", "_diode_current"),
            ("set_ac_voltage", "_diode_voltage"),
            ("current_constraint", "_diode_current_constraint"),
            ("commutation_margin", "_commutation_margin"),
            ("commutate", "_commutate"),
            ("describe", "_diode_description"),
            ("get_state", "_conduction_state"),
            ("set_state", "_restore_conduction_state"),
            ("equation_state_bindings", "_conduction_bindings"),
        )
        for name, method in methods:
            setattr(model, name, getattr(type(self), method).__get__(model, type(model)))

    @staticmethod
    def _diode_current(model, t: float) -> None:
        model.out.i_dc = -model._current.sign * _real(np.real(model.inp.i_c))

    @staticmethod
    def _diode_voltage(model, t: float) -> None:
        if model._current.sign != 0.0:
            model.out.u_c = -model._current.sign * model.inp.u_dc
        else:
            model.out.u_c = model.inp.u_hold

    @staticmethod
    def _diode_current_constraint(model):
        return 0.0 if model._current.sign == 0.0 else None

    @staticmethod
    def _commutation_margin(model) -> float:
        """Nonnegative while the present conduction state is valid."""
        current = _real(np.real(model.inp.i_c))
        u_dc = _real(np.real(model.inp.u_dc))
        if model._current.sign != 0.0:
            return model._current.margin(current)
        # blocked: valid while |u_hold| < u_dc and the port current keeps its zero value
        return min(u_dc - abs(_real(np.real(model.inp.u_hold))),
                   model._current.margin(current, model.topology.zero_current_tolerance))

    @staticmethod
    def _commutate(model, t: float) -> None:
        """Switch the conducting pair at a located current zero, or end a blocking interval."""
        current = _real(np.real(model.inp.i_c))
        u_dc = _real(np.real(model.inp.u_dc))
        if model._current.sign == 0.0:
            if np.isfinite(model._blocked_since):
                model.blocked_time += t - model._blocked_since
                model._blocked_since = math.nan
            if abs(model._current.offset(current)) > model.topology.zero_current_tolerance:
                # current through blocked diodes (an initial state): conduct in its direction
                model._current.sign = _real(np.copysign(1.0, model._current.offset(current)))
            else:
                model._current.locate(current, -1.0 if _real(np.real(model.inp.u_hold)) > 0.0
                                     else 1.0)
        else:
            reverse = abs(_real(np.real(model.inp.u_hold))) > u_dc
            model._current.locate(current, -model._current.sign if reverse else 0.0)
            if not reverse:
                model._blocked_since = t
        model.commutations += 1

    @staticmethod
    def _diode_description(model) -> str:
        return "diode full-bridge rectifier, exact commutation with discontinuous conduction"

    @staticmethod
    def _conduction_state(model) -> dict[str, Any]:
        return {"direction": _real(model._current.sign), "i_ref": _real(model._current.residual)}

    @staticmethod
    def _conduction_bindings(model):
        """Mutable equation memory; the ODE and public restart schema stay unchanged."""
        return {"direction": (model._current, "sign"), "i_ref": (model._current, "residual"),
                **{name: (model, name) for name in ("commutations", "blocked_time", "_blocked_since")}}

    @staticmethod
    def _restore_conduction_state(model, values: Mapping[str, Any]) -> None:
        unknown = set(values) - {"direction", "i_ref"}
        if unknown:
            raise KeyError(f"diode bridge: no state(s) {sorted(unknown)}")
        if "direction" in values:
            direction = float(values["direction"])
            if direction not in (-1.0, 0.0, 1.0):
                raise ValueError(f"diode bridge direction must be -1, 0 or 1, got {direction}")
            model._current.sign = direction
        if "i_ref" in values:
            model._current.residual = float(values["i_ref"])

class _DCState(Bag):
    __slots__ = ("u_C",)


class _DCInp(Bag):
    __slots__ = ("i_dc",)


class _DCOut(Bag):
    __slots__ = ("u_dc", "u_C", "i_src")


class DCCapacitor:
    """Capacitor ``C`` (F) with series resistance ``R_esr`` (ohm); current positive into the capacitor."""

    def __init__(self, u0: float, C: float, R_esr: float = 0.0) -> None:
        self.C, self.R_esr = _real(C), _real(R_esr)
        self.state = _DCState(u_C=_real(u0))

    def terminal_voltage(self, i: float) -> float:
        return self.state.u_C + self.R_esr * i

    def derivative(self, i: float) -> float:
        return i / self.C

    def dissipated_power(self, i: float) -> float:
        return self.R_esr * i ** 2


def _zero(t: float) -> float:
    return 0.0


class DCCurrentSource:
    """Supply current ``ramp(t) i_nom + k_dc (u_ref - u_C)`` (A); may be negative.

    ``k_dc`` is in A/V and ``u_ref`` in V. :meth:`connect` controls its nominal-current ramp.
    """

    def __init__(self, i_nom: float, k_dc: float = 0.0, u_ref: float = 0.0) -> None:
        self.i_nom, self.k_dc, self.u_ref = _real(i_nom), _real(k_dc), _real(u_ref)
        self.ramp: Optional[Callable[[float], float]] = None

    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect at ``t``, optionally with a smooth ramp, or disconnect the nominal current."""
        if not on:
            self.ramp = _zero
        elif ramp <= 0.0:
            self.ramp = None
        else:
            end = t + ramp
            self.ramp = lambda at: 1.0 if at >= end else smoothstep((at - t) / ramp)

    def __call__(self, t: float, u_C: float) -> float:
        i = self.i_nom if self.ramp is None else self.ramp(t) * self.i_nom
        if self.k_dc != 0.0:
            i += self.k_dc * (self.u_ref - u_C)
        return i


class DCVoltageSource:
    """Constant dc voltage source ``u_dc`` (V) with series resistance ``R`` (ohm)."""

    def __init__(self, u_dc: float, R: float = 0.0) -> None:
        self.u_dc, self.R = _real(u_dc), _real(R)

    def terminal_voltage(self, i: float) -> float:
        return self.u_dc - self.R * i

    def dissipated_power(self, i: float) -> float:
        return self.R * i ** 2


class DCLink:
    """DC link made of an optional capacitor and a current source, voltage source or no source.

    Without a capacitor a voltage source is required; a voltage source with a
    capacitor requires ``R + R_esr > 0``. Named state: ``u_C`` (V) if a capacitor is present.
    """

    ports: ClassVar[tuple[PowerPort, ...]] = (PowerPort("out.u_dc", "inp.i_dc", 1.0, -1.0),)
    outputs_need_inputs: ClassVar[bool] = True  # i_src depends on i_dc
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod("set_stored_voltage", inputs=(), outputs=("u_C",)),
        OutputMethod("set_outputs", inputs=("i_dc",), outputs=("u_dc", "i_src")),
    )

    def __init__(self, capacitor: DCCapacitor | None = None,
                 source: DCCurrentSource | DCVoltageSource | None = None) -> None:
        if capacitor is not None and not isinstance(capacitor, DCCapacitor):
            raise TypeError("capacitor must be a DCCapacitor or None")
        if source is not None and not isinstance(source, (DCCurrentSource, DCVoltageSource)):
            raise TypeError("source must be a DCCurrentSource, DCVoltageSource or None")
        if capacitor is None and not isinstance(source, DCVoltageSource):
            raise ValueError("a DC link without a capacitor requires a voltage source")
        if capacitor is not None and isinstance(source, DCVoltageSource) and source.R + capacitor.R_esr <= 0:
            raise ValueError("a voltage source with a capacitor requires positive source resistance or ESR")
        self.capacitor, self.source = capacitor, source
        self.state = capacitor.state if capacitor is not None else Empty()
        self.state_names = ("u_C",) if capacitor is not None else ()
        self.storage = ((StoragePort("u_C", capacitor.C, 1.0, (("inp.i_dc", -1.0),)),)
                        if capacitor is not None else ())
        self.has_source = source is not None
        u0 = capacitor.state.u_C if capacitor is not None else source.u_dc
        self.inp = _DCInp(i_dc=0.0)
        self.out = _DCOut(u_dc=u0, u_C=u0, i_src=0.0)
        self.terminal = Terminal(
            self, "u_dc", "i_dc", -1, drives_voltage=True, electrical_domain="dc"
        )
        self.tripped = False
        # The circuit topology cannot change after assembly. Bind its equations once instead of
        # rediscovering the capacitor/source combination at every Runge--Kutta stage.
        if type(self) is DCLink:
            if capacitor is None:
                self.set_outputs = self._set_voltage_source_without_capacitor
                self.rhs = self._rhs_without_capacitor
            elif source is None:
                self.set_outputs = self._set_capacitor_without_source
                self.rhs = self._rhs_with_capacitor
            elif isinstance(source, DCVoltageSource):
                self.set_outputs = self._set_capacitor_with_voltage_source
                self.rhs = self._rhs_with_capacitor
            else:
                self.set_outputs = self._set_capacitor_with_current_source
                self.rhs = self._rhs_with_capacitor

    def open_breaker(self) -> None:
        """Disconnect the source from the capacitor; a link without capacitor is unchanged."""
        self.tripped = True

    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect or disconnect the unit's source; current sources honor the connection ramp."""
        if isinstance(self.source, DCCurrentSource):
            self.source.connect(on, t, ramp)

    def retune_source(self, i: float, k: float, v: float, r: float) -> None:
        """Apply new settings to the existing DC source without replacing its state."""
        source = self.source
        if isinstance(source, DCCurrentSource):
            source.i_nom, source.k_dc = float(i), float(k)
        elif isinstance(source, DCVoltageSource):
            source.u_dc, source.R = float(v), float(r)

    def set_stored_voltage(self, t: float) -> None:
        """Expose the capacitor voltage (or stiff source emf) without a dc-current dependency."""
        self.out.u_C = (self.capacitor.state.u_C if self.capacitor is not None
                        else self.source.u_dc)

    def set_outputs(self, t: float) -> None:
        """Generic topology dispatch retained for subclasses."""
        cap, source, i_dc = self.capacitor, self.source, self.inp.i_dc
        if cap is None:
            self.out.u_dc = self.out.u_C = source.terminal_voltage(i_dc)
            self.out.i_src = i_dc
            return
        if self.tripped or source is None:
            i_src = 0.0
        elif isinstance(source, DCVoltageSource):
            # from u_C + R_esr (i_src - i_dc) = u_dc - R i_src
            i_src = (source.u_dc - cap.state.u_C + cap.R_esr * i_dc) / (source.R + cap.R_esr)
        else:
            i_src = source(t, cap.state.u_C)
        self.out.i_src = i_src
        self.out.u_C = cap.state.u_C
        self.out.u_dc = cap.terminal_voltage(i_src - i_dc)

    def _set_voltage_source_without_capacitor(self, t: float) -> None:
        source, i_dc, out = self.source, self.inp.i_dc, self.out
        out.u_dc = out.u_C = source.u_dc - source.R * i_dc
        out.i_src = i_dc

    def _set_capacitor_without_source(self, t: float) -> None:
        cap, i_dc, out = self.capacitor, self.inp.i_dc, self.out
        out.i_src = 0.0
        out.u_C = cap.state.u_C
        out.u_dc = cap.state.u_C - cap.R_esr * i_dc

    def _set_capacitor_with_voltage_source(self, t: float) -> None:
        cap, source, i_dc, out = self.capacitor, self.source, self.inp.i_dc, self.out
        i_src = (0.0 if self.tripped else
                 (source.u_dc - cap.state.u_C + cap.R_esr * i_dc) /
                 (source.R + cap.R_esr))
        out.i_src = i_src
        out.u_C = cap.state.u_C
        out.u_dc = cap.state.u_C + cap.R_esr * (i_src - i_dc)

    def _set_capacitor_with_current_source(self, t: float) -> None:
        cap, source, i_dc, out = self.capacitor, self.source, self.inp.i_dc, self.out
        i_src = 0.0 if self.tripped else source(t, cap.state.u_C)
        out.i_src = i_src
        out.u_C = cap.state.u_C
        out.u_dc = cap.state.u_C + cap.R_esr * (i_src - i_dc)

    def rhs(self, t: float):
        if self.capacitor is None:
            return ()
        return (self.capacitor.derivative(self.out.i_src - self.inp.i_dc),)

    def _rhs_without_capacitor(self, t: float):
        return ()

    def _rhs_with_capacitor(self, t: float):
        return ((self.out.i_src - self.inp.i_dc) / self.capacitor.C,)

    def dissipated_power(self) -> float:
        loss = (self.capacitor.dissipated_power(self.out.i_src - self.inp.i_dc)
                if self.capacitor is not None else 0.0)
        if isinstance(self.source, DCVoltageSource):
            i_src = self.inp.i_dc if self.capacitor is None else self.out.i_src
            loss += self.source.dissipated_power(i_src)
        return loss

    def supplied_power(self) -> float:
        if isinstance(self.source, DCVoltageSource):
            i_src = self.inp.i_dc if self.capacitor is None else self.out.i_src
            return self.source.u_dc * i_src
        return self.out.u_dc * self.out.i_src


def make_dclink(cfg: Any) -> DCLink:
    """Build a DC link from its capacitor and optional DC source."""
    capacitor = None
    if cfg.capacitor is not None:
        capacitor = DCCapacitor(cfg.vdc_ref, cfg.capacitor.c, cfg.capacitor.r_esr)
    p = cfg.source
    if p.type == "current":
        source = DCCurrentSource(p.i, p.k, cfg.vdc_ref)
    elif p.type == "voltage":
        source = DCVoltageSource(p.v, p.r)
    elif p.type == "none":
        source = None
    else:
        raise ValueError(f"unknown DC source type {p.type!r}")
    return DCLink(capacitor, source)
