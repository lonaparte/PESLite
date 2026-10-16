"""The converter's power stage: bridge models and the dc link.

Quantities are SI. Every bridge keeps the same AC/DC and held-modulation connections; a concrete
bridge also owns the timing which turns sampled controller commands into its modulation value.
"""
from __future__ import annotations

import math
from typing import Any, Callable, ClassVar, Mapping, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from ..control.blocks import smoothstep
from ..solver.energy import PowerPort, StoragePort
from ..solver.model import Bag, ConfigError, Empty, OutputMethod
from .network import Terminal
from .pwm import TIME_EPS, Modulator, PWM, ZOH, make_modulator

__all__ = ["ThreePhaseBridge", "SwitchingBridge", "PWMBridge", "AveragingBridge",
           "HalfBridge", "FullBridge", "DCLink", "DCCapacitor",
           "DCCurrentSource", "DCVoltageSource", "HalfBridgeDiodeConduction",
           "make_topology", "make_bridge", "make_dclink"]


class _BridgeInp(Bag):
    __slots__ = ("q", "u_dc", "i_c")


class _BridgeOut(Bag):
    __slots__ = ("u_c", "i_dc")


class _HalfBridgeInp(Bag):
    __slots__ = (
        "q", "upper_current_ratio", "lower_current_ratio",
        "u_dc", "u_lower", "i_c",
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
    config_fields = frozenset()
    set_dc_current = staticmethod(_set_three_phase_dc_current)

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

    state_names: ClassVar[tuple[str, ...]] = ()
    has_switching_events: ClassVar[bool] = False
    outputs_need_inputs: ClassVar[bool] = True
    dirac: ClassVar[bool] = True
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod("set_dc_current", inputs=("q", "i_c"), outputs=("i_dc",)),
        OutputMethod("set_ac_voltage", inputs=("q", "u_dc"), outputs=("u_c",)),
    )
    _half_bridge_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod(
            "set_dc_current", inputs=("q", "i_c"), outputs=("i_dc", "i_lower")
        ),
        OutputMethod(
            "set_ac_voltage", inputs=("q", "u_dc", "u_lower"), outputs=("u_c",)
        ),
    )
    _half_bridge_transferred_current_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod(
            "set_dc_current",
            inputs=("upper_current_ratio", "lower_current_ratio", "i_c"),
            outputs=("i_dc", "i_lower"),
        ),
        OutputMethod(
            "set_ac_voltage", inputs=("q", "u_dc", "u_lower"), outputs=("u_c",)
        ),
    )

    def __init__(self, topology=None) -> None:
        self.topology = topology if topology is not None else ThreePhaseBridge()
        self.state = Empty()
        zero = self.topology.zero
        if isinstance(self.topology, HalfBridge):
            self.inp = _HalfBridgeInp(
                q=zero,
                upper_current_ratio=zero,
                lower_current_ratio=zero,
                u_dc=0.0,
                u_lower=0.0,
                i_c=zero,
            )
            self.out = _HalfBridgeOut(u_c=zero, i_dc=0.0, i_lower=0.0)
            self.output_stages = self._half_bridge_stages
            self.set_ac_voltage = _set_half_bridge_voltage.__get__(self, type(self))
        else:
            self.inp = _BridgeInp(q=zero, u_dc=0.0, i_c=zero)
            self.out = _BridgeOut(u_c=zero, i_dc=0.0)
        self.ports = (
            PowerPort("inp.u_dc", "out.i_dc"),
            PowerPort("out.u_c", "inp.i_c", self.topology.power_scale, -1.0),
        )
        if isinstance(self.topology, HalfBridge):
            self.ports += (PowerPort("inp.u_lower", "out.i_lower"),)
        self.dc_terminal = Terminal(self, "u_dc", "i_dc", 1)
        self.power_terminal = Terminal(self, "u_c", "i_c", -1, drives_voltage=True)
        self.lower_terminal = (
            Terminal(self, "u_lower", "i_lower", 1)
            if isinstance(self.topology, HalfBridge) else None
        )
        self.set_dc_current = self.topology.set_dc_current.__get__(self, type(self))

    def set_ac_voltage(self, t: float) -> None:
        self.out.u_c = self.inp.q * self.inp.u_dc

    def set_outputs(self, t: float) -> None:
        """Evaluate dc current and ac voltage from the current inputs."""
        self.set_dc_current(t)
        self.set_ac_voltage(t)

    def use_current_transfer_ratios(self) -> None:
        """Drive both half-bridge rail currents from explicit conduction ratios."""
        if not isinstance(self.topology, HalfBridge):
            raise TypeError("independent current-transfer ratios require a half bridge")
        self.set_dc_current = _set_half_bridge_transferred_currents.__get__(self, type(self))
        self.output_stages = self._half_bridge_transferred_current_stages

    def rhs(self, t: float):
        return ()

    def zoh_connections(self, label: str) -> dict:
        """Connect the held bridge conversion ratio to ``label``."""
        return {(self, "q"): label}


class _SampledBridge(_BridgeModel, PWM):
    """Shared sampled timing for exact-switching and PWM-averaged forms."""

    state_prefix = "pwm"

    def __init__(self, period: float, load_period: float, offset: float, computation: float,
                 carrier_period: float, modulator: Modulator, topology=None) -> None:
        _BridgeModel.__init__(self, topology)
        PWM.__init__(
            self,
            period,
            load_period,
            offset,
            computation,
            carrier_period,
            modulator,
            duty_names=self.topology.duty_names,
            switching_state=self.topology.switching_state,
            pwm_state=self.topology.pwm_state,
        )
        self._command_duties = self.topology.command_duties
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
            self.write(at, self._command_duties(output), bool(output.gates) and enabled,
                       output.theta, output.omega)
        if abs(self.t_load - t) < TIME_EPS:
            changes.append(self.load(t))
        if command is not None and self.computation > 0.0:
            at, output = command
            self.write(at, self._command_duties(output), bool(output.gates) and enabled,
                       output.theta, output.omega)
        changes.extend(self.switches(t))
        return changes


class SwitchingBridge(_SampledBridge):
    """A bridge topology evaluated at every exact semiconductor switching edge."""


class PWMBridge(_SampledBridge):
    """A bridge topology driven by duty ratios held for one PWM interval."""


class AveragingBridge(_BridgeModel):
    """Ideal averaged bridge driven directly by the continuous controller output.

    There is no timer, command history, held modulation input or delay state: the controller and
    power stage are evaluated in the same Model RHS plan. A unilateral half bridge may still use
    the configured physical switching period in its external branch's cycle-averaged DCM law;
    that parameter describes conduction physics and does not create simulation events.
    """

    state_prefix = "bridge"

    def __init__(self, topology=None) -> None:
        _BridgeModel.__init__(self, topology)
        self.modulator = None
        self.state_owner = self
        self.continuous = True
        self.enabled = False

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

    def zoh_connections(self, label: str) -> dict:
        return {}

    def describe(self) -> str:
        return "ideal averaging, continuous control"

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if values:
            raise KeyError(f"continuous averaging bridge has no states, got {sorted(values)}")


def make_topology(cfg: Any):
    """Build the bridge topology selected by ``cfg.type``."""
    try:
        topology_type = TOPOLOGY_TYPES[cfg.type]
    except KeyError:
        raise ConfigError(
            f"unknown bridge topology {cfg.type!r}; known: {sorted(TOPOLOGY_TYPES)}"
        ) from None
    return topology_type.from_config(cfg)


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
    pwm = cfg.pwm
    if cfg.bridge.model == "averaging":
        if modulator is not None:
            raise ValueError("an averaging bridge does not use a PWM modulator")
        return AveragingBridge(topology)
    bridge_type = SwitchingBridge if cfg.bridge.model == "switching" else PWMBridge
    return bridge_type(
        cfg.ctrl.period, pwm.load_period, pwm.grid_offset, cfg.ctrl.computation,
        pwm.switching_period,
        modulator if modulator is not None else make_modulator(pwm, cfg.base.f0, cfg.bridge),
        topology,
    )


@_register_topology
class HalfBridge:
    """One scalar switching leg between upper and lower DC rails.

    Device placement is fixed once at construction.  Setting ``upper_controlled``
    or ``lower_controlled`` to false leaves only that device's diode available.
    The simulation hot path therefore uses the same two-bit gate word for
    diode-assisted and fully controlled legs.
    """

    type = "half_bridge"
    duty_names = ("d",)
    zero = 0.0
    power_scale = 1.0
    neutral_duty = 0.0
    config_fields = frozenset((
        "upper_controlled", "lower_controlled", "reverse_port", "external_terminal",
    ))
    set_dc_current = staticmethod(_set_half_bridge_currents)
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
    """Two controlled half bridges forming one scalar H-bridge."""

    type = "full_bridge"
    duty_names = ("d_left", "d_right")
    zero = 0.0
    power_scale = 1.0
    neutral_duty = 0.5
    config_fields = frozenset()
    set_dc_current = staticmethod(_set_scalar_dc_current)
    __slots__ = ()

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
    def from_config(cls, _cfg: Any) -> "FullBridge":
        return cls()

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

class _DCState(Bag):
    __slots__ = ("u_C",)


class _DCInp(Bag):
    __slots__ = ("i_dc",)


class _DCOut(Bag):
    __slots__ = ("u_dc", "u_C", "i_src")


class DCCapacitor:
    """Capacitor ``C`` (F) with series resistance ``R_esr`` (ohm); current positive into the capacitor."""

    def __init__(self, u0: float, C: float, R_esr: float = 0.0) -> None:
        self.C, self.R_esr = float(C), float(R_esr)
        self.state = _DCState(u_C=float(u0))

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
        self.i_nom, self.k_dc, self.u_ref = float(i_nom), float(k_dc), float(u_ref)
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
        self.u_dc, self.R = float(u_dc), float(R)

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
        self.terminal = Terminal(self, "u_dc", "i_dc", -1, drives_voltage=True)
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
    """Build a :class:`DCLink` from SI parameters; source type ``"current"``, ``"voltage"`` or ``"none"``."""
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
