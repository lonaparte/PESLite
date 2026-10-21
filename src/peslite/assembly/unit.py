"""A converter unit: power stage, sensing, controller and protection.

Switching and PWM-period averaging use an ADC and a closed sampled controller. Ideal averaging
wires instantaneous measurements and a continuous controller into the plant ODE. All protection
decisions, their single trip latch, breakers and gates belong to the unit.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Callable, Iterator, Mapping, Optional

import numpy as np

from ..components.adc import ADC, ADCSample, MeasurementPorts
from ..components.converter import (HalfBridgeDiodeConduction, make_bridge, make_dclink,
                                    make_topology)
from ..components.network import RLBranch, Terminal
from ..components.pwm import TIME_EPS, Modulator
from ..control.blocks import phases
from ..control.controller import (Controller, ConverterMeasurement, Measurement,
                                  make_controller)
from .events import Scenario
from .params import SimulationParams, UnitParams, _parameterized
from .protection import Protection, ProtectionSample
from .types import unit_type

__all__ = ["Plant", "Unit", "VSC", "DCConverter"]


class Plant:
    """Shared state and output synchronization for execution and export initialization.

    Units only call ``outputs``, ``hold`` and ``change``. The host owns integration, supplies
    its current state vector and invalidates outputs whenever that state changes.
    """

    def __init__(self, model: Any, state: Any, *,
                 project: Optional[Callable] = None, settle: Optional[Callable] = None) -> None:
        self.model, self.state = model, state
        self.project, self.settle = project, settle
        self.inputs_changed = self.zoh_changed = False
        self._synced_at: Optional[float] = None

    def invalidate(self) -> None:
        self._synced_at = None

    def read_state(self) -> None:
        """Repack states after a lifecycle or file event, applying the model's bounds."""
        self.state = self.model.get_initial_values()
        if self.project is not None:
            self.state = self.project(self.state)
        self.invalidate()

    def outputs(self, t: float) -> None:
        if self._synced_at != t:
            self.model.sync(t, self.state)
            self._synced_at = t
        elif self.zoh_changed:
            self.model.sync_zoh(t)
        self.zoh_changed = False

    def hold(self, label: str, value: Any) -> None:
        self.model.set_zoh_input(label, value)
        self.zoh_changed = True

    @contextmanager
    def change(self) -> Iterator[None]:
        if self.settle is not None:
            self.settle()
        yield
        self.read_state()
        self.inputs_changed = True


class Unit(ABC):
    """Common runtime interface and lifecycle mechanics for one converter unit."""

    @property
    def dc_bus(self) -> str:
        """The scoped internal bus between this unit's DC link and bridge."""
        return f"{self.name}.bus.dc"

    @property
    def power_bus(self) -> str:
        """The scoped internal bus on this unit's bridge power side."""
        return f"{self.name}.bus.power"

    def _configure_runtime(
        self,
        sim: SimulationParams,
        ports: MeasurementPorts,
        channel_zeros: Mapping[str, complex | float],
        current_base: Callable[[UnitParams], float],
    ) -> None:
        """Build the sensing, protection and lifecycle state shared by converter units."""
        self.ports = ports
        self.protection = _parameterized(lambda values: Protection(
            values[0].protection,
            values[1].output.period if self.continuous else values[0].ctrl.period,
            current_base(values[0]),
            rms_available=bool(values[0].meas.rms)), (self.cfg, sim))
        self._trip_due = False
        self._track_due = False
        self._write_due: Optional[tuple[float, Any]] = None
        self._initially_connected = False
        self.breaker = True
        self.gates = False

        if self.continuous and not self.cfg.meas.rms:
            self.adc = None
            self.windowed = self._oversamples = False
            return
        meas = self.cfg.meas
        channels = {} if self.continuous else {name: channel_zeros[name] for name in meas.average}
        self.adc = _parameterized(lambda values: ADC(
            ports,
            values[1].output.period if self.continuous else values[0].ctrl.period,
            values[0].meas.period if values[0].meas.period is not None else (
                values[1].output.period if self.continuous else values[0].ctrl.period),
            values[0].meas.window if values[0].meas.window is not None else (
                values[1].output.period if self.continuous else values[0].ctrl.period),
            channels,
            rms=values[0].meas.rms,
        ), (self.cfg, sim))
        self.windowed = self.adc.accumulating
        self.accumulate, self.restart_windows = self.adc.accumulate, self.adc.seed
        self._oversamples = self.adc.samples > 1

    @abstractmethod
    def subsystems(self) -> dict[str, Any]: ...

    @abstractmethod
    def connections(self) -> dict: ...

    @abstractmethod
    def zoh_connections(self) -> dict: ...

    @abstractmethod
    def terminals(self) -> tuple: ...

    def _power_terminals(
        self,
        network: tuple[str, Terminal, Terminal] | None = None,
    ) -> tuple[tuple[str, Terminal], ...]:
        """Expose the DC rail and power terminal through the common named-bus contract."""
        terminals = [
            (self.dc_bus, self.dclink.terminal),
            (self.dc_bus, self.bridge.dc_terminal),
            (self.power_bus, self.bridge.power_terminal),
        ]
        if network is not None:
            external_bus, bridge_side, network_side = network
            terminals.extend((
                (self.power_bus, bridge_side),
                (external_bus, network_side),
            ))
        return tuple(terminals)

    @abstractmethod
    def aliases(self) -> dict[str, str]: ...

    def state_parts(self) -> list[tuple[str, Any]]:
        """Return this unit's fixed state owners and their public prefixes."""
        parts = [(self.name, self), (f"{self.name}.ctrl", self.ctrl),
                 (f"{self.name}.{self.bridge.state_prefix}", self.bridge.state_owner)]
        adc = getattr(self, "adc", None)
        if adc is not None and adc.accumulating:
            parts.append((f"{self.name}.meas", adc))
        return parts

    @property
    def periods(self) -> list[float]:
        periods = ([] if self.continuous else [
            *self.bridge.event_periods,
            *(period for period in getattr(self.ctrl, "periods", {}).values() if period),
        ])
        if self.adc is not None:
            periods.extend(self.adc.periods)
        return periods

    @property
    def tripped(self) -> bool:
        return self.protection.tripped

    @property
    def blocked(self) -> bool:
        return self.breaker and not self.gates and not self.tripped

    def next_time(self) -> float:
        if self.continuous:
            return math.inf
        bridge = self.bridge
        next_ = bridge.next_time()
        if self.windowed:
            next_ = min(next_, self.adc.t_window(bridge.next_control_time))
        if self._oversamples:
            next_ = min(next_, self.adc.t_sample(bridge.previous_control_time()))
        return next_

    def get_state(self) -> dict[str, Any]:
        return {
            "tripped": self.tripped,
            "fault": self.protection.fault,
            **{f"prot.{name}": value for name, value in self.protection.get_state().items()},
        }

    @property
    def discrete_state_names(self) -> tuple[str, ...]:
        """Unit-owned state is protection/lifecycle state; electrical state belongs to the model."""
        return tuple(self.get_state())

    def set_state(self, values: Mapping[str, Any]) -> None:
        self.protection.set_state({
            name[5:]: value for name, value in values.items() if name.startswith("prot.")
        })
        tripped = values.get("tripped")
        if tripped is None and values.get("fault"):
            tripped = True
        self.protection.restore_latches(tripped, values.get("fault"))
        if self.tripped:
            self.trip(math.nan, "restored")

    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect or disconnect this converter through the common unit boundary."""
        was_active = self.breaker and self.gates and not self.tripped
        effective = bool(on) and not self.tripped
        self.breaker = effective
        self._update_terminal()
        self._connection_command(effective, t, ramp)
        if self.continuous:
            self._track_due = effective and not was_active
            self.bridge.enabled = effective
            self._set_gates(effective)
        elif self.branch_f is None:
            self._set_gates(effective and self.bridge.on)

    def retune(self, cfg: Any, paths: list[str], t: float) -> None:
        if any(path.startswith("protection.") for path in paths):
            self.protection.retune(cfg.protection)
        self._retune_controller(cfg, paths, t, ("protection.",))

    def trip(self, t: float, cause: str, plant: Optional[Plant] = None,
             detail: str = "") -> None:
        if plant is not None:
            with plant.change():
                self.trip(t, cause, detail=detail)
            return
        self.protection.latch(t, cause, detail)
        self._track_due = False
        self._trip_due = False
        self._shutdown(t)

    def start(self, t: float) -> None:
        """Initialize the source and controller command before loading named control state."""
        initialize = getattr(self.ctrl, "initialize", None)
        if initialize is not None:
            initialize()
        self._initially_connected, since, ramp = self.scenario.at(t)
        start = since if self._initially_connected and math.isfinite(since) else t
        self._connection_command(self._initially_connected, start,
                                 ramp if self._initially_connected else 0.0,
                                 reset=False, at=t)
        if not self._initially_connected or self.continuous:
            return
        states = getattr(self.ctrl, "get_state", None)
        if math.isfinite(since) and states is not None and "startup.steps" in states():
            self.ctrl.set_state({"startup.steps": self.bridge.control_count(since, t)})

    def states_loaded(self) -> None:
        """Apply restored lifecycle and bridge state to the physical terminal."""
        self.breaker = self._initially_connected and not self.tripped
        current = self.branch_f.state.i if self.branch_f is not None else None
        self._restore_bridge_state(current)

    def begin(self, t: float, plant: Plant, continued: bool, window_given: bool) -> None:
        """Track the initial plant state and restore bridge and ADC timing."""
        if self._initially_connected and not continued:
            self.track(t, plant)
        self.begin_timing(t, plant, continued, window_given)

    def track(self, t: float, plant: Plant) -> None:
        """Track control state from instantaneous physical inputs, without advancing time."""
        track = getattr(self.ctrl, "track", None)
        if track is not None:
            plant.outputs(t)
            with plant.change():
                track(self._instantaneous_measurement(t))

    def begin_timing(self, t: float, plant: Plant, continued: bool, window_given: bool) -> None:
        """Restore bridge and ADC timing after initial state has been loaded."""
        self._track_due = False
        if self.continuous:
            self._start_continuous_bridge(t)
            if self.adc is not None:
                plant.outputs(t)
                self.adc.seed()
            return
        initial_sync = getattr(self.ctrl, "initial_sync", None)
        sync = initial_sync() if initial_sync is not None else (None, None)
        self._begin_sampled(t, plant, sync, continued, window_given)

    def protect(self, t: float, plant: Plant) -> bool:
        """Run the common fast over-current path at a physical bridge edge."""
        if self.tripped or (not self.continuous and not self.bridge.fast_edge(t)):
            return False
        if not self.protection.check_fast(t, self._protection_currents()):
            return False
        event = self.protection.trip
        assert event is not None
        self.trip(event.t, event.cause, plant, event.detail)
        return True

    def sense(self, t: float, plant: Plant) -> Optional[tuple[int, Any]]:
        """Sample, control and protect one converter through the common runtime path."""
        if self.continuous:
            plant.outputs(t)
            if self._track_due:
                self.track(t, plant)
                plant.outputs(t)
                self._track_due = False
            self.ctrl.startup.at(t)
            rms_fresh = self.adc is not None and self.adc.sample_rms(t)
            self._check_sampled_protection(t, self._instantaneous_measurement(t),
                                           rms_fresh=rms_fresh)
            return None
        measurement = self._sample_control(t, plant)
        if measurement is None:
            return None
        self._before_sampled_control(t, measurement)
        output = self.ctrl(t, measurement)
        self._check_sampled_protection(t, measurement, output)
        self._write_due = (t, output)
        return self.bridge.control_index, output

    def actuate(self, t: float, plant: Plant) -> bool:
        """Apply pending protection, bridge, gate and ADC-window transitions."""
        tripped_now = self._actuate_sampled(t, plant, self.branch_f is not None or self.breaker)
        if not self.continuous and self.branch_f is None and not self.gates:
            plant.hold(self.zoh, self.bridge.topology.zero)
        return tripped_now

    def summary(self) -> dict[str, Any]:
        seen = dict(self.ctrl.summary()) if hasattr(self.ctrl, "summary") else {}
        protection = self.protection.summary()
        protection["alarms"] = list(dict.fromkeys(
            [*seen.pop("alarms", []), *protection["alarms"]]
        ))
        seen.update(protection)
        return seen

    @abstractmethod
    def signals(self) -> dict[str, float | complex]: ...

    @abstractmethod
    def control_inputs(self) -> dict[str, float | complex]:
        """Return instantaneous values at the controller graph's public input ports."""

    @abstractmethod
    def control_inputs_from(
        self, values: Mapping[str, complex | float],
    ) -> dict[str, Any]:
        """Adapt named physical measurements to the controller's public input ports."""

    def enabled_subsystems(self) -> tuple[Any, ...]:
        """Return subsystems whose equations are enabled and disabled with the unit gates."""
        return ()

    @property
    def trip_cause(self) -> Optional[str]:
        trip = self.protection.trip
        return trip.cause if trip is not None else None

    def source_preset(self) -> float | None:
        """Return an internal source voltage for initial-state presets, when one exists."""
        return None

    def _controller_command(self, run: bool, ramp: float = 0.0,
                            t: float | None = None, *, reset: bool = True) -> None:
        """Dispatch one run command through the controller's common lifecycle interface."""
        if self.continuous:
            command = getattr(self.ctrl, "command_at", None)
            if command is not None:
                command(run, ramp, 0.0 if t is None else t, reset=reset)
                return
        command = getattr(self.ctrl, "command", None)
        if command is not None:
            command(run, ramp)

    def _connection_command(self, run: bool, t: float, ramp: float, *, reset: bool = True,
                            at: float | None = None) -> None:
        """Apply one source/controller command; completed initial ramps need no live callback."""
        source_ramp = ramp if at is None or at < t + ramp else 0.0
        self.dclink.connect(run, t, source_ramp)
        self._controller_command(run, ramp, t, reset=reset)

    def _retune_controller(self, cfg: UnitParams, paths: list[str], t: float,
                           excluded: tuple[str, ...] = ()) -> None:
        """Apply the common DC-source update and pass remaining paths to the controller."""
        self.cfg = cfg
        if any(path.startswith("dclink.source.") for path in paths):
            source = cfg.dclink.source
            self.dclink.retune_source(source.i, source.k, source.v, source.r)
        rest = [path for path in paths
                if not path.startswith(("dclink.", *excluded))]
        if rest:
            self.ctrl.retune(cfg, rest, t)

    def _actuate_bridge(self, t: float, plant: Plant, enabled: bool) -> None:
        """Apply one pending command and every bridge edge at the current instant."""
        bridge = self.bridge
        changes = bridge.actuate(t, self._write_due, enabled)
        active = bridge.on and enabled
        if active != self.gates:
            with plant.change():
                self._set_gates(active)
        for value in changes:
            plant.hold(self.zoh, value)
        self._write_due = None

    def _update_terminal(self) -> None:
        """Apply breaker and gate state to the optional series branch."""
        if self.branch_f is None:
            return
        method = (
            self.branch_f.close_breaker
            if self.breaker and self.gates and not self.tripped
            else self.branch_f.open_breaker
        )
        method()

    def _set_gates(self, on: bool) -> None:
        """Apply one gate state to a filtered or unfiltered named power bus."""
        self.gates = bool(on) and not self.tripped
        if self.branch_f is None:
            self.gates = self.gates and self.breaker
            if not self.continuous:
                self.bridge.inp.q = self._held_bridge_command(self.bridge.inp.q)
        else:
            self._update_terminal()

    def _held_bridge_command(self, value: Any) -> Any:
        """Block a sampled command on a disconnected unfiltered power bus."""
        if self.branch_f is None and not self.gates:
            return self.bridge.topology.zero
        return value

    def _shutdown(self, t: float) -> None:
        """Stop controller and power stage through the common unit boundary."""
        self._controller_command(False, 0.0, t)
        if hasattr(self.bridge, "enabled"):
            self.bridge.enabled = False
        self.breaker = False
        self._set_gates(False)
        self.dclink.open_breaker()

    def _restore_bridge_state(self, given_current: complex | float | None) -> None:
        """Reconcile restored trip, controller and bridge state through one lifecycle path."""
        if self.tripped:
            self._controller_command(False, 0.0, 0.0, reset=False)
        if self.continuous:
            self.bridge.enabled = (
                self.ctrl.startup.active and self.breaker and not self.tripped
            )
        self._set_gates(self.bridge.on)
        if given_current is not None and self.blocked and given_current != 0:
            warnings.warn(
                f"simulation.initial.states: unit {self.name!r} starts with its PWM blocked, "
                f"so {self.name}.branch_f.i starts at zero, not at the value given",
                stacklevel=5,
            )

    def _start_continuous_bridge(self, t: float) -> None:
        """Apply continuous startup state to the common bridge/gate boundary."""
        self.ctrl.startup.at(t)
        self.bridge.enabled = self.ctrl.startup.active and self.breaker and not self.tripped
        self._set_gates(self.bridge.enabled)

    @abstractmethod
    def _protection_currents(self) -> tuple[float, ...]:
        """Return instantaneous physical currents used by fast protection."""

    @abstractmethod
    def _control_measurement(self, sample: ADCSample) -> Any:
        """Adapt a generic ADC sample to this controller family's input object."""

    @abstractmethod
    def _protection_sample(self, measurement: Any, output: Any | None) -> ProtectionSample:
        """Adapt one controller-rate observation to common protection quantities."""

    def _instantaneous_measurement(self, t: float) -> Any:
        return self._control_measurement(ADCSample(t=t, values=self.ports.read()))

    def _check_sampled_protection(self, t: float, measurement: Any,
                                  output: Any | None = None, *, rms_fresh: bool = True) -> None:
        if self.tripped:
            return
        sample = self._protection_sample(measurement, output)
        if self.protection.rms_enabled:
            window = self.adc.rms[unit_type(self.cfg.type).current_channel]
            value = window.value if rms_fresh else None
            sample = replace(sample, rms_current_pu=(
                None if value is None else value / self.protection.i_base
            ))
        if (not self.tripped
                and self.protection.check_sampled(t, sample)):
            self._trip_due = True

    def measurement_signals(self) -> dict[str, float]:
        if self.adc is None:
            return {}
        signals = {}
        for channel, window in self.adc.rms.items():
            value = window.value
            signals[f"{self.name}.meas.{channel}_rms"] = math.nan if value is None else value
        return signals

    def _held_start_command(self, value: Any) -> Any:
        """Adapt the reconstructed bridge command to the unit's physical boundary."""
        return self._held_bridge_command(value)

    def _before_sampled_control(self, t: float, measurement: Any) -> None:
        """Run an optional converter-family check immediately before sampled control."""

    def _begin_sampled(
        self,
        t: float,
        plant: Plant,
        sync: tuple[float | None, float | None],
        continued: bool,
        window_given: bool,
    ) -> None:
        plant.hold(
            self.zoh,
            self._held_start_command(self.bridge.start(t, sync, continued)),
        )
        if self.windowed:
            plant.outputs(t)
            self.adc.seed()
        bridge = self.bridge
        self.adc.start(
            t,
            bridge.previous_control_time(),
            bridge.next_control_time,
            held=not window_given,
        )

    def _sample_control(self, t: float, plant: Plant) -> Any | None:
        """Run common oversampling and return one controller sample when its interrupt is due."""
        bridge = self.bridge
        if not bridge.control_due(t):
            if (self._oversamples
                    and abs(self.adc.t_sample(bridge.previous_control_time()) - t) < TIME_EPS):
                plant.outputs(t)
                self.adc.peek(t)
            return None
        plant.outputs(t)
        return self._control_measurement(self.adc.sample(t))

    def _actuate_sampled(self, t: float, plant: Plant, enabled: bool) -> bool:
        """Apply a pending trip, bridge events and the next ADC window."""
        tripped_now = self._trip_due
        if tripped_now:
            self._trip_due = False
            event = self.protection.trip
            assert event is not None
            self.trip(event.t, event.cause, plant, event.detail)
        if self.continuous:
            return tripped_now
        self._actuate_bridge(t, plant, enabled and not self.tripped)
        bridge = self.bridge
        if self.windowed and abs(self.adc.t_window(bridge.next_control_time) - t) < TIME_EPS:
            self.adc.open()
        return tripped_now


class VSC(Unit):
    """One converter unit built from its parameter section, connected to ``bus``.

    ``ctrl`` and ``modulator`` replace the parts built from the section. The unit samples with
    ``adc`` and controls a bridge which owns its model-specific actuation timing.
    Events connect or disconnect its filter and DC source and may retune runtime parameters.
    """

    def __init__(self, name: str, cfg: UnitParams, sim: SimulationParams, bus: Any,
                 ctrl: Optional[Controller] = None, modulator: Optional[Modulator] = None) -> None:
        self.name = name
        self.cfg = cfg
        self.bus = bus
        self.scenario = Scenario(cfg.events, f"units.{name}", initial=cfg.connected)

        # ------------------------------------------------------------ power
        self.dclink = _parameterized(make_dclink, cfg.dclink)
        try:
            self.bridge = _parameterized(lambda c: make_bridge(c, modulator), cfg)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None
        self.branch_f = _parameterized(lambda c: RLBranch(c.l_f, c.r_f), cfg.ac_filter)
        self.continuous = bool(getattr(self.bridge, "continuous", False))
        self._diodes_warned = False

        # ------------------------------------------------------------ ADC
        self.ports = MeasurementPorts(
            {
                "u_g": lambda: bus.out.u,
                "i_c": lambda: self.branch_f.out.i,
                "u_dc": lambda: self.dclink.out.u_dc,
            }
        )
        self._configure_runtime(
            sim,
            self.ports,
            {"u_g": 0j, "i_c": 0j, "u_dc": 0.0},
            lambda c: c.base.i_phase_peak,
        )

        # ------------------------------------------------------------ bridge actuation and control
        self.bridge.reset(np.full(3, 0.5), on=False)
        self.zoh = f"{name}.q"                   # model label of the bridge's held switching state
        self.ctrl = ctrl if ctrl is not None else _parameterized(make_controller, cfg)
        if bool(getattr(self.ctrl, "continuous", False)) != self.continuous:
            raise TypeError(
                f"{name}: the controller and bridge must both be sampled or both be continuous"
            )
        parts = [(self.ctrl, Controller)]
        modulator = getattr(self.bridge, "modulator", None)
        if modulator is not None:
            parts.append((modulator, Modulator))
        for obj, proto in parts:
            if not isinstance(obj, proto):
                raise TypeError(f"{name}: {type(obj).__name__} does not satisfy the {proto.__name__} protocol")

    # ---------------------------------------------------------------- assembly
    def subsystems(self) -> dict[str, Any]:
        """Return this unit's blocks keyed by namespaced name."""
        parts = {f"{self.name}.branch_f": self.branch_f,
                 f"{self.name}.bridge": self.bridge,
                 f"{self.name}.dclink": self.dclink}
        if self.continuous:
            parts[f"{self.name}.ctrl"] = self.ctrl
        return parts

    def connections(self) -> dict:
        """Return this unit's internal and measurement wiring."""
        connections = {}
        if self.continuous:
            connections.update({
                (self.ctrl, "u_g"): (self.bus, "u"),
                (self.ctrl, "i_c"): (self.branch_f, "i"),
                (self.ctrl, "u_dc"): (self.dclink, "u_dc"),
                (self.bridge, "q"): (self.ctrl, "q"),
            })
        return connections

    def zoh_connections(self) -> dict:
        return {} if self.continuous else self.bridge.zoh_connections(self.zoh)

    def terminals(self) -> tuple[tuple[str, Terminal], ...]:
        """Bind the DC link, bridge, filter and public terminal through named buses."""
        return self._power_terminals((
            self.cfg.bus,
            self.branch_f.terminal1,
            self.branch_f.terminal2,
        ))

    def aliases(self) -> dict[str, str]:
        """Return the aliases ``<unit>.i_c`` (filter current) and ``<unit>.u_dc`` (dc-link voltage)."""
        out = {f"{self.name}.i_c": f"{self.name}.branch_f.i"}
        if self.cfg.dclink.capacitor is not None:
            out[f"{self.name}.u_dc"] = f"{self.name}.dclink.u_C"
        return out

    # ---------------------------------------------------------------- run instants
    def _before_sampled_control(self, t: float, measurement: Measurement) -> None:
        if self.blocked and not self._diodes_warned:
            self._warn_diodes(t, measurement)

    def _warn_diodes(self, t: float, measurement: Measurement) -> None:
        latest = measurement.samples[-1] if measurement.samples else measurement
        u_g, u_dc = latest.u_g, latest.u_dc
        v_ll_peak = math.sqrt(3.0) * abs(u_g)
        if v_ll_peak > u_dc:
            self._diodes_warned = True
            warnings.warn(
                f"t = {t:.6g} s: unit {self.name!r} has blocked gates with peak line voltage "
                f"({v_ll_peak:.4g} V) above dc voltage ({u_dc:.4g} V); its diodes would conduct, "
                "which the open-circuit model leaves out",
                stacklevel=4,
            )

    def signals(self) -> dict[str, float | complex]:
        """Return this unit's recorded signals (SI); outputs must be synced first."""
        return {
            f"{self.name}.i_c": self.branch_f.out.i,
            f"{self.name}.u_g": self.bus.out.u,
            f"{self.name}.u_dc": self.dclink.out.u_dc,
            f"{self.name}.i_dc": self.dclink.inp.i_dc,
        }

    def control_inputs(self) -> dict[str, float | complex]:
        """Return the instantaneous per-unit inputs exposed by the control interface."""
        return self.control_inputs_from(self.ports.read())

    def control_inputs_from(
        self, values: Mapping[str, complex | float],
    ) -> dict[str, Any]:
        return {
            "meas.u_g": values["u_g"] / self.cfg.base.v_phase_peak,
            "meas.i_c": values["i_c"] / self.cfg.base.i_phase_peak,
            "meas.u_dc": values["u_dc"] / self.cfg.dclink.vdc_ref,
        }

    def _protection_currents(self) -> tuple[float, ...]:
        return tuple(phases(self.branch_f.state.i))

    def _control_measurement(self, sample: ADCSample) -> Measurement:
        values = sample.values
        latest = sample.samples[-1] if sample.samples else sample
        raw_current = complex(latest.values["i_c"])
        return Measurement(
            t=sample.t,
            u_g=complex(values["u_g"]),
            i_c=complex(values["i_c"]),
            u_dc=float(values["u_dc"]),
            i_abc=np.asarray(phases(raw_current)),
            samples=tuple(self._control_measurement(item) for item in sample.samples),
        )

    def _protection_sample(self, measurement: Measurement,
                           output: Any | None) -> ProtectionSample:
        omega = self.ctrl.omega if output is None else output.omega
        if omega is None and self.protection.needs_frequency:
            raise ValueError(
                f"unit {self.name!r}: frequency or ROCOF protection needs "
                "the controller's omega output"
            )
        return ProtectionSample(
            terminal_voltage_pu=abs(measurement.u_g) / self.cfg.base.v_phase_peak,
            frequency_deviation=(
                None if omega is None
                else (omega - self.cfg.base.w0) / (2.0 * math.pi)
            ),
            dc_voltage_error_pu=(
                measurement.u_dc / self.cfg.dc_base.v
                - self.cfg.ctrl.references.vdc_ref_pu
            ),
            startup_complete=(
                self.ctrl.startup.complete if output is None
                else getattr(output, "startup_complete", True)
            ),
        )


class DCConverter(Unit):
    """Open DC converter unit with the same network boundary as :class:`VSC`.

    The unit owns its DC link, selected half/full bridge, controller and optional series filter.
    Every power connection, including a composite unit's internal connection, is assembled through
    the same named-bus contract.
    """

    def __init__(self, name: str, cfg: UnitParams, sim: SimulationParams,
                 bus: Any = None, *,
                 ctrl: Optional[Any] = None, modulator: Optional[Modulator] = None) -> None:
        self.name = name
        self.cfg = cfg
        self.scenario = Scenario(cfg.events, f"units.{name}", initial=cfg.connected)
        self.bus = bus
        topology = make_topology(cfg.bridge)
        self.dclink = _parameterized(make_dclink, cfg.dclink)
        self.branch_f = None if cfg.filter is None else _parameterized(lambda c: RLBranch(
            c.l_f,
            c.r_f,
            phases=1,
            unilateral=topology.unilateral,
        ), cfg.filter)
        self.bridge = _parameterized(lambda c: make_bridge(c, modulator, topology=topology), cfg)
        self._external_lower_rail = getattr(topology, "external_terminal", None) == "lower_rail"
        power_voltage_port = (bus, "u") if bus is not None else (self.bridge, "u_c")
        self._control_voltage_port = (
            power_voltage_port if self._external_lower_rail or not topology.reverse_port
            else (self.dclink, "u_dc")
        )
        self._control_supply_port = (
            power_voltage_port if topology.reverse_port else (self.dclink, "u_dc")
        )
        self.conduction = None
        if (self.branch_f is not None and self.branch_f.unilateral
                and cfg.bridge.model != "switching"):
            self.conduction = _parameterized(lambda c: HalfBridgeDiodeConduction(
                c.pwm.switching_period,
                c.filter.l_f,
                c.filter.r_f,
                topology,
            ), cfg)
            self.conduction.local_step_state = self.branch_f
            self.branch_f.use_external_conduction()
            self.bridge.use_current_transfer_ratios()
        self.continuous = bool(getattr(self.bridge, "continuous", False))
        self.ctrl = ctrl if ctrl is not None else _parameterized(
            lambda c: make_controller(c, topology=topology), cfg)
        if bool(getattr(self.ctrl, "continuous", False)) != self.continuous:
            raise TypeError(
                f"{name}: the controller and bridge must both be sampled or both be continuous"
            )
        if not isinstance(self.ctrl, Controller):
            raise TypeError(
                f"{name}: {type(self.ctrl).__name__} does not satisfy the Controller protocol"
            )
        self.bridge.reset(np.full(len(topology.duty_names), topology.neutral_duty), on=False)
        self.zoh = f"{name}.q"
        self.ports = MeasurementPorts({
            "current": self._current,
            "voltage": lambda: self._port_value(self._control_voltage_port),
            "vin": lambda: self._port_value(self._control_supply_port),
        })
        channel_zeros = {"current": 0.0, "voltage": 0.0, "vin": 0.0}
        interface = getattr(self.ctrl, "interface", None)
        needs_power = interface is not None and "meas.power" in interface.inputs
        if (needs_power or "power" in cfg.meas.average or "power" in cfg.meas.rms
                or "power" in cfg.meas.sources):
            external_bus = cfg.bus if cfg.bus is not None else self.power_bus
            self._power_terminal, = (
                terminal for bus_name, terminal in self.terminals() if bus_name == external_bus
            )
            self.ports.readers["power"] = lambda: -self._power_terminal.power()
            channel_zeros["power"] = 0.0
            if needs_power:
                self.ctrl.bind_power_port(self._power_terminal.power_port)
        self._configure_runtime(
            sim,
            self.ports,
            channel_zeros,
            lambda c: c.dc_base.i,
        )

    def subsystems(self) -> dict[str, Any]:
        parts = {
            f"{self.name}.bridge": self.bridge,
            f"{self.name}.dclink": self.dclink,
        }
        if self.branch_f is not None:
            parts[f"{self.name}.branch_f"] = self.branch_f
        if self.conduction is not None:
            parts[f"{self.name}.conduction"] = self.conduction
        if self.continuous:
            parts[f"{self.name}.ctrl"] = self.ctrl
        return parts

    def connections(self) -> dict:
        connections = {}
        if self.conduction is not None:
            assert self.branch_f is not None
            connections.update({
                (self.conduction, "current"): (self.branch_f, "i"),
                (self.conduction, "upper"): (self.dclink, "u_dc"),
                (self.branch_f, "di_dt"): (self.conduction, "derivative"),
                (self.branch_f, "loss_factor"): (
                    self.conduction, "resistance_factor",
                ),
                (self.bridge, "upper_current_ratio"): (
                    self.conduction, "upper_current_ratio",
                ),
                (self.bridge, "lower_current_ratio"): (
                    self.conduction, "lower_current_ratio",
                ),
            })
            if self._external_lower_rail:
                connections[(self.conduction, "lower")] = self._control_voltage_port
            else:
                connections[(self.conduction, "reference")] = (
                    self._control_supply_port
                    if self.bridge.topology.reverse_port
                    else self._control_voltage_port
                )
        if self.continuous:
            connections.update({
                (self.ctrl, "current"): self._current_connection(),
                (self.ctrl, "voltage"): self._control_voltage_port,
                (self.ctrl, "vin"): self._control_supply_port,
                (self.bridge, "q"): (self.ctrl, "q"),
            })
            if self.conduction is not None:
                connections[(self.conduction, "q")] = (self.ctrl, "q")
            if "meas.power" in self.ctrl.interface.inputs and "power" not in self.cfg.meas.sources:
                voltage, current = self._power_terminal.signal_sources()
                connections[(self.ctrl, "power_voltage")] = voltage
                connections[(self.ctrl, "power_current")] = current
        return connections

    def zoh_connections(self) -> dict:
        if self.continuous:
            return {}
        connections = self.bridge.zoh_connections(self.zoh)
        if self.conduction is not None:
            connections[(self.conduction, "q")] = self.zoh
        return connections

    def terminals(self) -> tuple:
        if self.branch_f is None:
            return self._power_terminals()
        if self._external_lower_rail:
            assert self.bridge.lower_terminal is not None
            return (
                *self._power_terminals(),
                (self.power_bus, self.branch_f.terminal1),
                (self.cfg.bus, self.bridge.lower_terminal),
            )
        if self.bridge.topology.reverse_port:
            bridge_side, network_side = self.branch_f.terminal2, self.branch_f.terminal1
        else:
            bridge_side, network_side = self.branch_f.terminal1, self.branch_f.terminal2
        return self._power_terminals((self.cfg.bus, bridge_side, network_side))

    def aliases(self) -> dict[str, str]:
        out = ({f"{self.name}.i_c": f"{self.name}.branch_f.i"}
               if self.branch_f is not None else {})
        if self.cfg.dclink.capacitor is not None:
            out[f"{self.name}.u_dc"] = f"{self.name}.dclink.u_C"
        return out

    def signals(self) -> dict[str, float]:
        return {
            f"{self.name}.i_c": self._current(),
            f"{self.name}.u_g": (
                self.bus.out.u if self.bus is not None else self.bridge.out.u_c
            ),
            f"{self.name}.u_dc": self.dclink.out.u_dc,
            f"{self.name}.i_dc": self.bridge.out.i_dc,
        }

    def control_inputs(self) -> dict[str, float]:
        """Return the SI values exposed by the converter control interface."""
        return self.control_inputs_from(self.ports.read())

    def control_inputs_from(
        self, values: Mapping[str, complex | float],
    ) -> dict[str, Any]:
        return self.ctrl.control_inputs(
            values["current"], values["voltage"], values["vin"],
            values.get("power"),
        )

    def source_preset(self) -> float:
        source = self.cfg.dclink.source
        return source.v if source.type == "voltage" else self.cfg.dclink.vdc_ref

    def _current(self) -> float:
        if self.branch_f is not None:
            return self.branch_f.out.i
        return self.bridge.inp.i_c

    def _current_connection(self) -> Any:
        if self.branch_f is not None:
            return self.branch_f, "i"
        sources = self.bridge.power_terminal.current_sources
        if sources is None:
            raise RuntimeError(f"{self.name}: the bridge power bus has not been assembled")
        return sources

    def _protection_currents(self) -> tuple[float, ...]:
        return (self._current(),)

    def _control_measurement(self, sample: ADCSample) -> ConverterMeasurement:
        values = sample.values
        return ConverterMeasurement(
            t=sample.t,
            current=float(values["current"]),
            voltage=float(values["voltage"]),
            vin=float(values["vin"]),
            power=values.get("power"),
            samples=tuple(self._control_measurement(item) for item in sample.samples),
        )

    def _protection_sample(self, measurement: ConverterMeasurement,
                           output: Any | None) -> ProtectionSample:
        return ProtectionSample(
            terminal_voltage_pu=abs(measurement.voltage) / self.cfg.dc_base.v,
            dc_voltage_error_pu=(
                float(self.dclink.out.u_dc) / self.cfg.dc_base.v - 1.0
            ),
            startup_complete=(
                self.ctrl.startup.complete if output is None
                else getattr(output, "startup_complete", True)
            ),
        )

    @staticmethod
    def _port_value(port: tuple[Any, str]) -> float:
        subsystem, field = port
        return getattr(subsystem.out, field)

