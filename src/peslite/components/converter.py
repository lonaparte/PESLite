"""The converter's power stage: bridge models and the dc link.

Quantities are SI. Every bridge keeps the same AC/DC and held-modulation connections; a concrete
bridge also owns the timing which turns sampled controller commands into its modulation value.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, ClassVar, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from ..control.blocks import abc2complex, smoothstep
from ..solver.energy import PowerPort, StoragePort
from ..solver.model import Bag, ConfigError, Empty, OutputStage
from .pwm import TIME_EPS, Modulator, PWM, ZOH, make_modulator

__all__ = ["ThreePhaseBridge", "SwitchingBridge", "PWMBridge", "AveragingBridge",
           "HalfBridge", "FullBridge", "DeviceParameters", "DCLink", "DCCapacitor",
           "DCCurrentSource", "DCVoltageSource", "make_bridge", "make_dclink"]


class _BridgeInp(Bag):
    __slots__ = ("q", "u_dc", "i_c")


class _BridgeOut(Bag):
    __slots__ = ("u_c", "i_dc")


def _set_three_phase_dc_current(bridge, _t: float) -> None:
    bridge.out.i_dc = 1.5 * (bridge.inp.q * bridge.inp.i_c.conjugate()).real


def _set_scalar_dc_current(bridge, _t: float) -> None:
    bridge.out.i_dc = (bridge.inp.q * bridge.inp.i_c).real


class ThreePhaseBridge:
    """Three-leg topology used by the existing AC converter model."""

    __slots__ = ()
    duty_names = ("d_a", "d_b", "d_c")
    zero = 0j
    switching_state = staticmethod(abc2complex)
    pwm_state = staticmethod(ZOH.value)
    set_dc_current = staticmethod(_set_three_phase_dc_current)

    @staticmethod
    def command_duties(output):
        return output.d_abc


class _BridgeModel:
    """One bridge topology composed with one simulation form.

    Topology selection binds its power equation once during construction.  The
    output hot path has no topology or simulation-form conditionals.
    """

    state_names: ClassVar[tuple[str, ...]] = ()
    has_switching_events: ClassVar[bool] = False
    outputs_need_inputs: ClassVar[bool] = True
    dirac: ClassVar[bool] = True
    output_stages: ClassVar[tuple[OutputStage, ...]] = (
        OutputStage("set_dc_current", inputs=("q", "i_c"), outputs=("i_dc",)),
        OutputStage("set_ac_voltage", inputs=("q", "u_dc"), outputs=("u_c",)),
    )

    def __init__(self, topology=None) -> None:
        self.topology = topology if topology is not None else ThreePhaseBridge()
        self.state = Empty()
        zero = self.topology.zero
        self.inp = _BridgeInp(q=zero, u_dc=0.0, i_c=zero)
        self.out = _BridgeOut(u_c=zero, i_dc=0.0)
        self.set_dc_current = self.topology.set_dc_current.__get__(self, type(self))

    def set_ac_voltage(self, t: float) -> None:
        self.out.u_c = self.inp.q * self.inp.u_dc

    def set_outputs(self, t: float) -> None:
        """Evaluate dc current and ac voltage from the current inputs."""
        self.set_dc_current(t)
        self.set_ac_voltage(t)

    def rhs(self, t: float):
        return ()

    def connections(self, dc_link, ac_branch) -> dict:
        """Return the connections of the bridge to ``dc_link`` and ``ac_branch``."""
        return {
            (ac_branch, "u1"): (self, "u_c"),
            (self, "i_c"): (ac_branch, "i"),
            (dc_link, "i_dc"): (self, "i_dc"),
            (self, "u_dc"): (dc_link, "u_dc"),
        }

    def zoh_connections(self, label: str) -> dict:
        """Return the connection of the held switching state ``label`` to ``q``."""
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

        The bridge owns the computation/load ordering.  Unit deliberately does not know whether
        the implementation uses compare registers, a delayed averaged command or no carrier.
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
    power stage are evaluated in the same Model RHS plan.
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


def make_bridge(cfg: Any, sim: Any = None,
                modulator: Modulator | None = None, topology=None) -> _BridgeModel:
    """Compose a bridge topology with switching, PWM or averaging behavior."""
    topology = topology if topology is not None else ThreePhaseBridge()
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


@dataclass(frozen=True, slots=True)
class DeviceParameters:
    """Static on-state parameters shared by scalar bridge devices."""

    on_resistance: float = 0.0
    diode_drop: float = 0.0
    diode_resistance: float = 0.0

    def __post_init__(self) -> None:
        values = (self.on_resistance, self.diode_drop, self.diode_resistance)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ConfigError("device resistance/drop must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class _LegResult:
    voltage: float
    upper_current: float
    lower_current: float
    loss: float
    path: str
    channel_current: float = 0.0
    diode_current: float = 0.0
    blocked: bool = False


class HalfBridge:
    """One scalar switching leg between upper and lower DC rails.

    Device placement is fixed once at construction.  Setting ``upper_controlled``
    or ``lower_controlled`` to false leaves only that device's diode available.
    The simulation hot path therefore uses the same two-bit gate word for
    asynchronous and synchronous legs.
    """

    duty_names = ("d",)
    zero = 0.0
    set_dc_current = staticmethod(_set_scalar_dc_current)
    __slots__ = ("devices", "_gate_mask")

    @staticmethod
    def switching_state(states) -> float:
        return float(states[0])

    @staticmethod
    def pwm_state(duties) -> float:
        return min(1.0, max(0.0, float(duties[0])))

    @staticmethod
    def command_duties(output):
        return (output.duty,)

    def __init__(self, devices: DeviceParameters | None = None, *,
                 upper_controlled: bool = True,
                 lower_controlled: bool = True) -> None:
        if type(upper_controlled) is not bool or type(lower_controlled) is not bool:
            raise TypeError("controlled-device flags must be bool")
        if not upper_controlled and not lower_controlled:
            raise ConfigError("a half bridge needs at least one controlled device")
        self.devices = devices or DeviceParameters()
        if not isinstance(self.devices, DeviceParameters):
            raise TypeError("devices must be DeviceParameters")
        self._gate_mask = int(upper_controlled) | (int(lower_controlled) << 1)

    def _gate_word(self, gates) -> int:
        if len(gates) != 2 or any(value not in (0, 1, False, True) for value in gates):
            raise ConfigError("a half bridge needs two binary gates")
        word = int(bool(gates[0])) | (int(bool(gates[1])) << 1)
        if word == 3:
            raise ConfigError("shoot-through: both half-bridge devices are on")
        if word & ~self._gate_mask:
            raise ConfigError("gate command targets a diode-only half-bridge branch")
        return word

    def bounds(self, gates, upper: float, lower: float = 0.0) -> tuple[float, float]:
        word = self._gate_word(gates)
        if upper < lower:
            raise ConfigError("upper DC rail must not be below lower rail")
        if word == 1:
            return upper, upper
        if word == 2:
            return lower, lower
        return lower - self.devices.diode_drop, upper + self.devices.diode_drop

    def resolve(self, gates, current: float, upper: float, lower: float = 0.0,
                floating: float = 0.0) -> _LegResult:
        word = self._gate_word(gates)
        if upper < lower:
            raise ConfigError("upper DC rail must not be below lower rail")
        p = self.devices
        if word:
            upper_on = word == 1
            rail = upper if upper_on else lower
            reverse = current < 0.0 if upper_on else current > 0.0
            channel, diode = current, 0.0
            drop = p.on_resistance * abs(current)
            if reverse and p.on_resistance > 0.0 and drop > p.diode_drop:
                diode = math.copysign(
                    (drop - p.diode_drop)
                    / (p.on_resistance + p.diode_resistance),
                    current,
                )
                channel = current - diode
                drop = p.on_resistance * abs(channel)
            voltage = rail - math.copysign(drop, current) if current else rail
            upper_current, lower_current = (
                (current, 0.0) if upper_on else (0.0, current)
            )
            side = "upper" if upper_on else "lower"
            suffix = "+diode" if diode else ""
            return _LegResult(
                voltage,
                upper_current,
                lower_current,
                max(0.0, (rail - voltage) * current),
                f"{side}.channel{suffix}",
                channel,
                diode,
            )
        if current > 0.0:
            voltage = lower - p.diode_drop - p.diode_resistance * current
            return _LegResult(
                voltage, 0.0, current, (lower - voltage) * current,
                "lower.diode", diode_current=current,
            )
        if current < 0.0:
            voltage = upper + p.diode_drop - p.diode_resistance * current
            return _LegResult(
                voltage, current, 0.0, (upper - voltage) * current,
                "upper.diode", diode_current=current,
            )
        low = lower - p.diode_drop
        high = upper + p.diode_drop
        tolerance = 1e-12 * max(1.0, abs(upper), abs(lower), abs(floating))
        if floating < low - tolerance:
            return _LegResult(low, 0.0, 0.0, 0.0, "lower.diode")
        if floating > high + tolerance:
            return _LegResult(high, 0.0, 0.0, 0.0, "upper.diode")
        return _LegResult(
            max(low, min(high, floating)), 0.0, 0.0, 0.0,
            "blocked", blocked=True,
        )


@dataclass(frozen=True, slots=True)
class _FullBridgeResult:
    voltage: float
    dc_current: float
    loss: float
    paths: tuple[str, str]
    blocked: bool
    channels: tuple[float, float, float, float]
    diodes: tuple[float, float, float, float]


class FullBridge:
    """Two controlled half bridges forming one scalar H-bridge."""

    duty_names = ("d_left", "d_right")
    zero = 0.0
    set_dc_current = staticmethod(_set_scalar_dc_current)
    __slots__ = ("left", "right", "devices")

    gates = {
        "positive": (1, 0, 0, 1),
        "negative": (0, 1, 1, 0),
        "zero_high": (1, 0, 1, 0),
        "zero_low": (0, 1, 0, 1),
        "blocked": (0, 0, 0, 0),
    }

    @staticmethod
    def switching_state(states) -> float:
        return float(states[0]) - float(states[1])

    @staticmethod
    def pwm_state(duties) -> float:
        left = min(1.0, max(0.0, float(duties[0])))
        right = min(1.0, max(0.0, float(duties[1])))
        return left - right

    @staticmethod
    def command_duties(output):
        duty = output.duty
        return duty, 1.0 - duty

    def __init__(self, devices: DeviceParameters | None = None) -> None:
        self.left = HalfBridge(devices)
        self.right = HalfBridge(self.left.devices)
        self.devices = self.left.devices

    def _gates(self, gates) -> tuple[int, int, int, int]:
        if isinstance(gates, str):
            try:
                return self.gates[gates]
            except KeyError as exc:
                raise ConfigError("unknown full-bridge gate state") from exc
        if len(gates) != 4:
            raise ConfigError("a full bridge needs four binary gates")
        result = tuple(gates)
        self.left._gate_word(result[:2])
        self.right._gate_word(result[2:])
        return result

    def voltage_bounds(self, gates, vdc: float) -> tuple[float, float]:
        values = self._gates(gates)
        left_low, left_high = self.left.bounds(values[:2], vdc, 0.0)
        right_low, right_high = self.right.bounds(values[2:], vdc, 0.0)
        return left_low - right_high, left_high - right_low

    def resolve(self, gates, current: float, vdc: float,
                floating: float = 0.0) -> _FullBridgeResult:
        if not math.isfinite(vdc) or vdc < 0.0:
            raise ConfigError("full-bridge DC voltage must be finite and nonnegative")
        values = self._gates(gates)
        left_gates, right_gates = values[:2], values[2:]
        if current == 0.0:
            left_low, left_high = self.left.bounds(left_gates, vdc, 0.0)
            right_low, right_high = self.right.bounds(right_gates, vdc, 0.0)
            target = max(left_low - right_high,
                         min(left_high - right_low, floating))
            left_voltage = max(left_low, min(left_high, right_low + target))
            right_voltage = left_voltage - target
            left = self.left.resolve(
                left_gates, 0.0, vdc, floating=left_voltage,
            )
            right = self.right.resolve(
                right_gates, 0.0, vdc, floating=right_voltage,
            )
            blocked = (
                not all((any(left_gates), any(right_gates)))
                and math.isclose(
                    target,
                    floating,
                    rel_tol=0.0,
                    abs_tol=1e-12 * max(1.0, vdc, abs(floating)),
                )
            )
        else:
            left = self.left.resolve(left_gates, current, vdc)
            right = self.right.resolve(right_gates, -current, vdc)
            blocked = False

        def currents(leg: _LegResult):
            upper = leg.path.startswith("upper")
            channels = (
                (leg.channel_current, 0.0)
                if upper else (0.0, -leg.channel_current)
            )
            diodes = (
                (-leg.diode_current, 0.0)
                if upper else (0.0, leg.diode_current)
            )
            return channels, diodes

        left_channels, left_diodes = currents(left)
        right_channels, right_diodes = currents(right)
        return _FullBridgeResult(
            left.voltage - right.voltage,
            left.upper_current + right.upper_current,
            left.loss + right.loss,
            (left.path, right.path),
            blocked,
            (*left_channels, *right_channels),
            (*left_diodes, *right_diodes),
        )


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
    output_stages: ClassVar[tuple[OutputStage, ...]] = (
        OutputStage("set_stored_voltage", inputs=(), outputs=("u_C",)),
        OutputStage("set_outputs", inputs=("i_dc",), outputs=("u_dc", "i_src")),
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
