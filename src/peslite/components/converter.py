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
from ..solver.energy import Port, StoragePort
from ..solver.model import Bag, ConfigError, Empty, LocatedZero, OutputMethod, _real
from ..solver.timing import TIME_EPS, control_clock_offset
from .network import RCNode, _extend_record
from .pwm import CarrierComparison, Modulator, PWM, ZOH, make_unit_modulator

__all__ = ["ThreePhaseBridge", "SwitchingModel", "PWMAveragingModel", "AveragingModel",
           "HalfBridge", "FullBridge", "DCLink", "DCCapacitor",
           "DCCurrentSource", "DCVoltageSource",
           "HalfBridgeDiodeConduction",
           "make_topology", "make_bridge", "make_dclink"]


class _BridgeInput(Bag):
    __slots__ = ("q", "u_dc", "i_c")


class _BridgeOutput(Bag):
    __slots__ = ("u_c", "i_dc")


class _HalfBridgeInput(Bag):
    __slots__ = (
        "q", "upper_current_ratio", "lower_current_ratio",
        "u_dc", "u_lower", "i_c", "loss_factor",
    )


class _HalfBridgeOutput(Bag):
    __slots__ = ("u_c", "i_dc", "i_lower")


class _LegGates:
    """Read complementary leg gates from the held switching state."""

    __slots__ = ("model", "q", "gates")

    def __init__(self, model) -> None:
        self.model, self.q, self.gates = model, None, ()

    def __call__(self, leg: int) -> bool:
        selected = getattr(self.model, "_netlist_gate_pattern", None)
        if selected is not None:
            return bool(selected[leg])
        q = self.model.input.q
        if not isinstance(q, (int, float, complex, np.number)):
            return tuple(
                gate > 0.5 for gate in self.model.topology.pole_gates(q, True)
            )[leg]
        if self.q is None or q != self.q:
            self.q = q
            self.gates = tuple(
                gate > 0.5 for gate in self.model.topology.pole_gates(q, True)
            )
        return self.gates[leg]


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
    def pole_currents(current):
        """Currents leaving the three pole midpoints, in modulation-leg order."""
        return phases(current)

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
        self.input = getattr(self.topology, "input_type", _BridgeInput)(
            q=zero, u_dc=0.0, i_c=zero)
        self.output = getattr(self.topology, "output_type", _BridgeOutput)(u_c=zero, i_dc=0.0)
        self.output_stages = getattr(self.topology, "output_stages", type(self).output_stages)
        self.dc_port = Port(
            self, "dc", voltage=0.0, current=0.0,
            electrical_domain="dc", standalone="voltage"
        )
        self.power_port = Port(
            self, "power", voltage=zero, current=zero,
            scale=self.topology.power_scale,
            standalone="current"
        )
        self.ports = (self.dc_port, self.power_port)
        self.lower_port = None
        self.modulator = None
        self.state_owner = self
        self.continuous = True
        self.enabled = False
        self._current_transfer = False
        initialize = getattr(self.topology, "initialize_model", None)
        if initialize is not None:
            initialize(self, parallel_capacitance)
        self.set_dc_current = self._device_outputs
        self.set_ac_voltage = self._device_outputs
        if (not math.isfinite(parallel_capacitance) or parallel_capacitance < 0.0
                or not math.isfinite(on_resistance) or on_resistance < 0.0):
            raise ConfigError("bridge capacitance and on_resistance must be finite and >= 0")
        self.device_on_resistance = _real(on_resistance)
        if parallel_capacitance > 0.0:
            if on_resistance <= 0.0:
                raise ConfigError("bridge.parallel_capacitance requires positive on_resistance")
            self._capacitive_devices(parallel_capacitance, on_resistance, switching)
        elif on_resistance > 0.0:
            self.resistance = self.topology.series_devices * _real(on_resistance)
            self.dirac = False
            self.output_stages = tuple(
                replace(stage, inputs=tuple(dict.fromkeys((*stage.inputs, "i_c"))))
                if "u_c" in stage.outputs else stage for stage in self.output_stages)

    def diode_blocks_reverse_current(self) -> bool:
        """Whether an external averaged branch must enforce reverse-current blocking."""
        return (
            self.topology.unilateral
            and not self.state_names
            and not self.has_switching_events
        )

    def netlist(self, nb):
        """Declare the bridge as explicit legs or a lossless conversion ratio."""
        from ..solver import netlist as nl

        topology = self.topology
        nb.port_output(self, "u_c", self.power_port, "voltage")
        nb.port_output(self, "i_dc", self.dc_port, "current")
        if self.lower_port is not None:
            nb.port_output(self, "i_lower", self.lower_port, "current")
        if self._current_transfer:
            nb.local_model()
            ac = nb.node(self.power_port)
            dc = nb.node(self.dc_port)
            lower = None if self.lower_port is None else nb.node(self.lower_port)

            def voltage(_t):
                q = topology.switching_state((self.input.q,))
                return q * self.input.u_dc + (1.0 - q) * self.input.u_lower

            nb.add(
                nl.VoltageSource(
                    ac, None,
                    nb.function(voltage, ("q", "u_dc", "u_lower")),
                ),
                nl.CurrentSource(
                    dc, None,
                    nb.function(
                        lambda _t: self.input.upper_current_ratio * self.input.i_c,
                        ("upper_current_ratio", "i_c"),
                    ),
                ),
            )
            if lower is not None:
                nb.add(nl.CurrentSource(
                    lower, None,
                    nb.function(
                        lambda _t: self.input.lower_current_ratio * self.input.i_c,
                        ("lower_current_ratio", "i_c"),
                    ),
                ))
            resistance = getattr(self, "resistance", 0.0)
            if resistance:
                loss = nb.internal(1, topology.power_scale)
                loss_current = nb.function(
                    lambda _t: max(0.0, self.input.loss_factor) ** 0.5
                    * abs(self.input.i_c),
                    ("loss_factor", "i_c"),
                )
                nb.add(
                    nl.CurrentSource(None, loss, loss_current),
                    nl.Resistor(loss, None, resistance),
                )
            nb.losses(self)
            return None
        if self.state_names:
            capacitor = nb.add(nl.Capacitor(
                nb.node(self.power_port), None, self.capacitor.C,
                nb.state(self, "u_C"),
            ))
            if not self._switching_poles:
                nb.mixture(self._gate_patterns, self._select_gate_pattern)
            self._netlist_legs(nb)
            nb.node_current_output(
                self, "i_device", nb.node(self.power_port), exclude=(capacitor,)
            )
            return None
        if (not getattr(topology, "controlled", True)
                or not (topology.upper_controlled and topology.lower_controlled)):
            return self._netlist_legs(nb)

        three_phase = isinstance(topology.zero, complex)
        ac = nb.node(self.power_port)
        resistance = getattr(self, "resistance", 0.0)
        pole = nb.internal(*((2, 1.5) if three_phase else (1, 1.0))) if resistance else ac
        if resistance:
            nb.add(nl.Resistor(pole, ac, resistance))
        lower = nb.node(self.lower_port) if self.lower_port is not None else None
        nb.add(nl.Transformer(
            pole, lower, nb.node(self.dc_port), lower, nb.value(self.input, "q")
        ))
        nb.losses(self)

    def _netlist_legs(self, nb) -> None:
        """Declare one pair of controlled-device/diode positions per bridge leg."""
        from ..solver import netlist as nl

        topology = self.topology
        on = self.device_on_resistance
        gate = _LegGates(self)
        upper = topology.upper_controlled and getattr(topology, "controlled", True)
        lower = topology.lower_controlled and getattr(topology, "controlled", True)

        def leg(midpoint, high, low, index):
            pair = (
                nl.Diode(
                    midpoint,
                    high,
                    on,
                    gate=(lambda: self.enabled and gate(index)) if upper else None,
                ),
                nl.Diode(
                    low,
                    midpoint,
                    on,
                    gate=(lambda: self.enabled and not gate(index)) if lower else None,
                ),
            )
            nb.add(*pair)
            return pair

        ac, dc = nb.node(self.power_port), nb.node(self.dc_port)
        if (self.lower_port is None and not isinstance(topology.zero, complex)
                and len(topology.voltage_factors) > 1):
            high, low = nb.internal(), nb.internal()
            supply = nb.add(nl.Transformer(high, low, dc, None, ((1.0,),)))
            devices = (*leg(ac, high, low, 0), *leg(None, high, low, 1))
            if getattr(self, "_current", None) is not None:
                nb.bind(self._bind_netlist_devices, devices)
                nb.commutation(self, after=self._mirror_netlist_conduction, diodes=devices)
        else:
            high = nb.internal()
            upper_feed = nb.add(nl.Resistor(dc, high, 0.0))
            if self.lower_port is not None:
                low = nb.internal()
                lower_node = nb.node(self.lower_port)
                lower_feed = nb.add(nl.Resistor(lower_node, low, 0.0))
                leg(ac, high, low, 0)
            else:
                count = len(topology.voltage_factors)
                chain = [ac, *(nb.internal(2, 1.5) for _ in range(count - 1)), None]
                basis = tuple(tuple(float(row == column) for column in range(count))
                              for row in range(count))
                for index, state in enumerate(basis):
                    factor = topology.switching_state(state)
                    column = (
                        (factor.real, factor.imag)
                        if isinstance(topology.zero, complex) else (float(factor),)
                    )
                    midpoint = nb.internal()
                    leg(midpoint, high, None, index)
                    nb.add(nl.Transformer(
                        chain[index], chain[index + 1], midpoint, None,
                        tuple((value,) for value in column),
                    ))
        nb.losses(self)

    def _capacitive_devices(self, capacitance, resistance, switching) -> None:
        """Compose finite device conductances with the existing capacitor and algebraic solver."""
        self._switching_poles = switching
        self._netlist_gate_pattern = None
        self.capacitor = RCNode(capacitance, 0.0, phases=3 if isinstance(self.topology.zero, complex) else 1)
        self.state, self.state_names = self.capacitor.state, self.capacitor.state_names
        self.storage = tuple(replace(port, power_terms=()) for port in self.capacitor.storage)
        self.output = _extend_record(self.output, "i_device", "power_loss")
        self.output.i_device = self.topology.zero
        inputs = ("q", "u_dc") + (("u_lower",) if self.lower_port else ())
        currents = ("i_dc", "i_device")
        if self.lower_port is not None:
            currents += ("i_lower",)
        self.output_stages = (
            OutputMethod("set_ac_voltage", (), ("u_c",)),
            OutputMethod("set_dc_current", inputs, currents),
            OutputMethod("set_power_loss", (), ("power_loss",), rhs=False),
        )
        self.dirac = False
        self.models_blocked_diodes = True

    def _device_outputs(self, t: float) -> None:
        from ..solver.netlist import device_outputs
        device_outputs(self, t)

    def set_ac_voltage(self, t: float) -> None:
        self._device_outputs(t)

    def set_outputs(self, t: float) -> None:
        self._device_outputs(t)

    def set_power_loss(self, _t: float) -> None:
        self.output.power_loss = self.dissipated_power()

    def _gate_patterns(self):
        duties = self.topology.pole_gates(self.input.q, False)
        return CarrierComparison.state_fractions(duties)

    def _select_gate_pattern(self, pattern) -> None:
        self._netlist_gate_pattern = pattern

    def use_current_transfer_ratios(self) -> None:
        """Drive both half-bridge rail currents from explicit conduction ratios."""
        configure = getattr(self.topology, "use_current_transfer_ratios", None)
        if configure is None:
            raise TypeError("the selected topology has no independent rail-current transfer ratios")
        configure(self)

    def _bind_netlist_devices(self, netlist, devices) -> None:
        self._netlist_devices = netlist, tuple(devices)

    def _device_commutation_margin(self) -> float:
        from ..solver.netlist import device_commutation_margin
        return device_commutation_margin(self)

    def _device_commutate(self, t: float) -> None:
        from ..solver.netlist import device_commutate
        device_commutate(self, t)

    def rhs(self, t: float):
        from ..solver.netlist import device_rhs
        return device_rhs(self, t)

    def dissipated_power(self) -> float:
        from ..solver.netlist import device_dissipated_power
        return device_dissipated_power(self)

    def supplied_power(self) -> float:
        return 0.0

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

    def actuate(self, t: float, command: tuple[float, Any, complex | float] | None,
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
    control_count = PWM.control_count
    fast_edge = PWM.fast_edge
    get_state = PWM.get_state
    set_state = PWM.set_state
    describe = PWM.describe

    def zoh_connections(self, label: str) -> dict:
        return {(self, "q"): label}

    def __init__(self, period: float, load_period: float, control_offset: float,
                 load_offset: float, computation: float,
                 carrier_period: float, modulator: Modulator, topology=None, *,
                 pattern_sync: str = "interval",
                 dead_time: float = 0.0,
                 dead_time_compensation: bool = False,
                 parallel_capacitance: float = 0.0, on_resistance: float = 0.0) -> None:
        self.pulse_update = None
        self.has_switching_events = not isinstance(modulator, ZOH)
        _BridgeModel.__init__(self, topology, parallel_capacitance=parallel_capacitance,
                              on_resistance=on_resistance, switching=self.has_switching_events)
        registers = getattr(modulator, "registers", None)
        carried = tuple(modulator.carried)
        PWM.__init__(
            self,
            period,
            load_period,
            control_offset,
            load_offset,
            computation,
            carrier_period,
            modulator,
            pattern_sync=pattern_sync,
            dead_time=0.0 if self.topology.unilateral else dead_time,
            dead_time_compensation=dead_time_compensation,
            pole_currents=self.topology.pole_currents,
            duty_names=(self.topology.duty_names + carried if registers is None else tuple(registers)),
            switching_state=self.topology.switching_state,
        )
        self.continuous = False
        self.command_registers = (self.topology.command_duties if registers is None
                                else modulator.command_registers)
        if registers is None and carried:
            self.command_registers = lambda output: (*self.topology.command_duties(output),
                                                       *(getattr(output, name) for name in carried))
        self.neutral_registers = (
            tuple(modulator.neutral_registers()) if registers is not None
            else (self.topology.neutral_duty,) * len(self.topology.duty_names) + (0.0,) * len(carried)
        )
        self.reset(self.neutral_registers, on=False)
        self.state_owner = self

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

    def control_due(self, t: float) -> bool:
        return abs(self.t_interrupt - t) < TIME_EPS

    def actuate(self, t: float, command: tuple[float, Any, Any] | None,
                enabled: bool) -> list[Any]:
        """Accept one sampled command and return bridge inputs which become active at ``t``.

        The bridge owns the computation/load ordering. The converter unit deliberately does not
        know whether the implementation uses compare registers, a delayed averaged command or no
        carrier.
        """
        changes: list[Any] = []
        if command is not None and self.computation == 0.0:
            at, registers, output = command
            self.write(at, registers, bool(output.gates) and enabled,
                       output.theta, output.omega)
        if abs(self.t_load - t) < TIME_EPS:
            changes.append(self.load(t))
        if command is not None and self.computation > 0.0:
            at, registers, output = command
            self.write(at, registers, bool(output.gates) and enabled,
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
    pwm = cfg.pwm
    impedance = {"parallel_capacitance": cfg.bridge.parallel_capacitance,
                 "on_resistance": cfg.bridge.on_resistance}
    if not getattr(topology, "controlled", True):
        if modulator is not None:
            raise ValueError("a diode full-bridge rectifier does not use a PWM modulator")
        if pwm is not None and (pwm.dead_time > 0.0 or pwm.dead_time_compensation):
            raise ValueError("PWM dead time requires a controlled bridge")
        return _BridgeModel(topology, switching=True, **impedance)
    if cfg.bridge.model == "averaging":
        if modulator is not None:
            raise ValueError("an averaging bridge does not use a PWM modulator")
        return AveragingModel(topology, **impedance)
    model_type = SwitchingModel if cfg.bridge.model == "switching" else PWMAveragingModel
    control_offset = control_clock_offset(cfg.ctrl, pwm)
    return model_type(
        cfg.ctrl.period, pwm.load_period, control_offset, pwm.grid_offset, cfg.ctrl.computation,
        pwm.switching_period,
        modulator if modulator is not None else make_unit_modulator(cfg),
        topology,
        pattern_sync=pwm.pattern_sync,
        dead_time=pwm.dead_time,
        dead_time_compensation=pwm.dead_time_compensation,
        **impedance,
    )


@_register_topology
class HalfBridge:
    """One scalar switching leg between upper and lower DC rails.

    Device placement is fixed once at construction.  Setting ``upper_controlled``
    or ``lower_controlled`` to false leaves only that device's diode available.
    The simulation hot path therefore uses the same two gate flags for
    diode-assisted and fully controlled legs.
    """

    input_type = _HalfBridgeInput
    output_type = _HalfBridgeOutput
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod(
            "set_dc_current", inputs=("q", "i_c"), outputs=("i_dc", "i_lower")
        ),
        OutputMethod(
            "set_ac_voltage", inputs=("q", "u_dc", "u_lower"), outputs=("u_c",)
        ),
    )
    def initialize_model(self, model, parallel_capacitance: float) -> None:
        model.input.loss_factor = 1.0
        external = self._external_port == "lower_rail"
        model.lower_port = Port(
            model,
            "lower",
            standalone="voltage" if external else None,
            grounded=not external,
        )
        model.ports += (model.lower_port,)

    def use_current_transfer_ratios(self, model) -> None:
        model._current_transfer = True

    type = "half_bridge"
    duty_names = ("d",)
    zero = 0.0
    power_scale = 1.0
    neutral_duty = 0.0
    config_fields = frozenset((
        "upper_controlled", "lower_controlled", "reverse_port", "external_port",
    ))
    voltage_factors = (1.0,)
    series_devices = 1

    @staticmethod
    def pole_gates(q, switching):
        return (q,)
    __slots__ = ("upper_controlled", "lower_controlled", "_reverse_port", "_external_port")

    def switching_state(self, states) -> float:
        state = states[0]
        return 1.0 - state if self._reverse_port else state

    def pole_currents(self, current):
        """Pole current expressed in the commanded leg's orientation."""
        current = np.real(current)
        return (-current if self._reverse_port else current,)

    @staticmethod
    def command_duties(output):
        return (output.duty,)

    def __init__(self, *, upper_controlled: bool = True,
                 lower_controlled: bool = True,
                 reverse_port: bool = False,
                 external_port: str = "midpoint") -> None:
        if any(type(value) is not bool for value in (
                upper_controlled, lower_controlled, reverse_port)):
            raise TypeError("controlled-device and port-orientation flags must be bool")
        if not upper_controlled and not lower_controlled:
            raise ConfigError("a half bridge needs at least one controlled device")
        if reverse_port and not lower_controlled:
            raise ConfigError("a reverse half bridge needs a controlled lower device")
        if not reverse_port and not upper_controlled:
            raise ConfigError("a forward half bridge needs a controlled upper device")
        if external_port not in ("midpoint", "lower_rail"):
            raise ConfigError(
                "a half bridge external_port must be 'midpoint' or 'lower_rail'"
            )
        if external_port == "lower_rail" and reverse_port:
            raise ConfigError(
                "a lower-rail external port uses the forward half-bridge orientation"
            )
        self.upper_controlled, self.lower_controlled = upper_controlled, lower_controlled
        self._reverse_port = reverse_port
        self._external_port = external_port

    @classmethod
    def from_config(cls, cfg: Any) -> "HalfBridge":
        return cls(
            upper_controlled=cfg.upper_controlled,
            lower_controlled=cfg.lower_controlled,
            reverse_port=cfg.reverse_port,
            external_port=cfg.external_port,
        )

    def state_from_duty(self, duty: float) -> float:
        """Return the bridge output ratio produced by its controlled device's duty."""
        return self.switching_state((min(1.0, max(0.0, duty)),))

    @property
    def reverse_port(self) -> bool:
        """Whether the external scalar port is on the bridge's input side."""
        return self._reverse_port

    @property
    def external_port(self) -> str:
        """The half-bridge port exposed as the converter's external bus."""
        return self._external_port

    @property
    def unilateral(self) -> bool:
        """Whether one device position is diode-only and therefore blocks reverse current."""
        return self.upper_controlled != self.lower_controlled

    def control_values(self, current, voltage, supply):
        """Return canonical control measurements and the topology's feed-forward duty."""
        current, voltage = np.real(current), np.real(voltage)
        if self._external_port == "lower_rail":
            voltage = -voltage
            denominator = supply + voltage
            feedforward = voltage / denominator if denominator > 0.0 else 0.0
        elif self._reverse_port:
            feedforward = 1.0 - supply / voltage if voltage > 0.0 else 0.0
        else:
            feedforward = voltage / supply if supply > 0.0 else 0.0
        return current, voltage, supply, feedforward

class _DiodeConductionResult(NamedTuple):
    controlled: Any
    diode: Any
    total: Any
    resistance_factor: Any
    derivative: Any
    in_dcm: Any
    upper_current_ratio: Any
    lower_current_ratio: Any


class _ConductionInput(Bag):
    __slots__ = ("current", "q", "upper", "lower", "reference")


class _ConductionOutput(Bag):
    __slots__ = (*_DiodeConductionResult._fields, "maximum_step")


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
        OutputMethod("set_outputs", inputs=_ConductionInput.__slots__,
                     outputs=_ConductionOutput.__slots__),
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
        self.input = _ConductionInput()
        self.output = _ConductionOutput(total=1.0, resistance_factor=1.0, in_dcm=False,
                                  maximum_step=math.inf)

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
            self.input.current, self.input.q, self.input.upper, self.input.lower,
            self.input.reference,
        )
        out = self.output
        (out.controlled, out.diode, out.total, out.resistance_factor,
         out.derivative, out.in_dcm, out.upper_current_ratio,
         out.lower_current_ratio) = result  # unpacked directly: this runs at every RHS evaluation
        out.maximum_step = self.period / 10.0 if result.in_dcm else math.inf

    def rhs(self, _t: float):
        return ()

    def diode_maximum_step(self) -> float:
        """Return the state-dependent resolution limit produced by the latest evaluation."""
        return self.output.maximum_step


@_register_topology
class FullBridge:
    """Two half bridges forming one scalar H-bridge.

    By default all four devices are controlled. With ``upper_controlled`` and ``lower_controlled``
    both false only the anti-parallel diodes remain: the bridge is a diode full-bridge rectifier
    without a controller or PWM, optionally with
    ``parallel_capacitance`` across its AC ports.
    """

    type = "full_bridge"
    duty_names = ("d_left", "d_right")
    zero = 0.0
    power_scale = 1.0
    neutral_duty = 0.5
    config_fields = frozenset(("upper_controlled", "lower_controlled"))
    voltage_factors = (0.5, -0.5)
    series_devices = 2

    @staticmethod
    def pole_gates(q, switching):
        if switching:
            return np.maximum(q, 0.0).item(), np.maximum(-q, 0.0).item()
        return 0.5 * (1.0 + q), 0.5 * (1.0 - q)

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
    def pole_currents(current):
        """Currents leaving the left and right pole midpoints."""
        current = np.real(current)
        return current, -current

    @staticmethod
    def command_duties(output):
        duty = output.duty
        return duty, 1.0 - duty

    @classmethod
    def from_config(cls, cfg: Any) -> "FullBridge":
        return cls(upper_controlled=cfg.upper_controlled, lower_controlled=cfg.lower_controlled)

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
        model.dirac = False
        model._current = LocatedZero()  # sign: the conducting direction, 0 while blocked
        model.commutations = 0
        model.discrete_state_names = ("direction",)
        model.linearization_state_names = ()
        methods = (
            ("describe", "_diode_description"),
            ("get_state", "_conduction_state"),
            ("set_state", "_restore_conduction_state"),
            ("equation_state_bindings", "_conduction_bindings"),
            ("_mirror_netlist_conduction", "_mirror_netlist_conduction"),
        )
        model._netlist_devices = None
        for name, method in methods:
            setattr(model, name, getattr(type(self), method).__get__(model, type(model)))
        model.commutation_margin = model._device_commutation_margin
        model.commutate = model._device_commutate

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
                "commutations": (model, "commutations")}

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
        if model._netlist_devices is not None:
            netlist, devices = model._netlist_devices
            sign, residual = model._current.sign, model._current.residual
            states = (sign < 0.0, sign > 0.0, sign > 0.0, sign < 0.0)
            currents = (-residual, residual, residual, -residual)
            for device, conducting, current in zip(devices, states, currents):
                netlist.set_conducting(device, conducting, current if conducting else 0.0)

    @staticmethod
    def _mirror_netlist_conduction(model, t: float, before: Mapping[str, Any]) -> None:
        """Mirror the netlist diode pair into the rectifier's public discrete state."""
        netlist, devices = model._netlist_devices
        (c1, r1), (c2, r2), (c3, _), (c4, _) = (
            netlist.conduction(device) for device in devices
        )
        positive, negative = c2 and c3, c1 and c4
        sign = 1.0 if positive and not negative else -1.0 if negative and not positive else 0.0
        if sign != model._current.sign:
            model.commutations += 1
        model._current.locate(r2 if sign > 0.0 else -r1 if sign < 0.0 else 0.0, sign)

class _DCState(Bag):
    __slots__ = ("u_C",)


class _DCOutput(Bag):
    __slots__ = ("u_dc", "u_C", "i_src")


class DCCapacitor:
    """Capacitor ``C`` (F) with series resistance ``R_esr`` (ohm); current positive into the capacitor."""

    def __init__(self, u0: float, C: float, R_esr: float = 0.0) -> None:
        self.C, self.R_esr = _real(C), _real(R_esr)
        self.state = _DCState(u_C=_real(u0))

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

class DCLink:
    """DC link made of an optional capacitor and a current source, voltage source or no source.

    Without a capacitor a voltage source is required; a voltage source with a
    capacitor requires ``R + R_esr > 0``. Named state: ``u_C`` (V) if a capacitor is present.
    """

    outputs_need_inputs: ClassVar[bool] = False
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod("set_stored_voltage", inputs=(), outputs=("u_C",)),
        OutputMethod("set_outputs", inputs=(), outputs=("u_dc", "i_src")),
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
        self.has_source = source is not None
        u0 = capacitor.state.u_C if capacitor is not None else source.u_dc
        self.input = Empty()
        self.output = _DCOutput(u_dc=u0, u_C=u0, i_src=0.0)
        self.port = Port(
            self, "dc", voltage=u0, current=0.0,
            electrical_domain="dc", standalone="current",
        )
        self.ports = (self.port,)
        self.storage = ((StoragePort(
            "u_C", capacitor.C, 1.0, ((self.port, "current", 1.0),)
        ),) if capacitor is not None else ())
        self.tripped = False

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
        from ..solver.netlist import device_outputs
        device_outputs(self, t)

    def set_outputs(self, t: float) -> None:
        from ..solver.netlist import device_outputs
        device_outputs(self, t)

    def rhs(self, t: float):
        from ..solver.netlist import device_rhs
        return device_rhs(self, t)

    def netlist(self, nb):
        """Declare the DC-link capacitor, source and source breaker."""
        from ..solver import netlist as nl

        node, capacitor, source = nb.node(self.port), self.capacitor, self.source
        nb.port_output(self, "u_dc", self.port, "voltage")
        if capacitor is not None:
            inner = nb.internal() if capacitor.R_esr else node
            if capacitor.R_esr:
                nb.add(nl.Resistor(node, inner, capacitor.R_esr))
            capacitor_element = nb.add(nl.Capacitor(
                inner, None, capacitor.C, nb.state(self, "u_C")
            ))
            nb.output(self, "u_C", ("branch_voltage", capacitor_element))
        if isinstance(source, DCCurrentSource):
            supply = nb.add(nl.CurrentSource(
                None,
                node,
                nb.function(
                    lambda t: 0.0 if self.tripped
                    else self.source(t, self.capacitor.state.u_C)
                ),
            ))
            nb.output(self, "i_src", ("branch_current", supply))
        elif isinstance(source, DCVoltageSource):
            emf = nb.internal() if source.R or capacitor is not None else node
            supply = nb.add(nl.VoltageSource(
                emf, None, nb.function(lambda _t: self.source.u_dc)
            ))
            nb.linear_output(self, "i_src", ((-1.0, ("branch_current", supply)),))
            if capacitor is None:
                nb.output(self, "u_C", ("branch_voltage", supply))
            if capacitor is not None:
                connected = nb.internal() if source.R else node
                nb.add(nl.Switch(emf, connected, nb.value(self, "tripped", invert=True)))
                emf = connected
            if source.R:
                nb.add(nl.Resistor(emf, node, source.R))
        else:
            nb.linear_output(self, "i_src", ())
        nb.losses(self)
        nb.supplies(self)

    def dissipated_power(self) -> float:
        from ..solver.netlist import device_dissipated_power
        return device_dissipated_power(self)

    def supplied_power(self) -> float:
        from ..solver.netlist import device_supplied_power
        return device_supplied_power(self)


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
