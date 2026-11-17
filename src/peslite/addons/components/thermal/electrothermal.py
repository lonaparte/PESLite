"""Electrothermal add-ons composed with PESLite's existing electrical Model.

Thermal state is an ordinary restartable subsystem; no second electrical circuit,
RHS integrator or global thermal configuration is introduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import numpy as np

from ....components.network import Element, register_component_type
from ....solver import Bag, ConfigError, Empty
from ...functions.thermal import SemiconductorLossModel, SwitchingEvent
from ...functions.thermal.losses import _lookup
from .network import FosterImpedance, LumpedThermalNetwork, _positive, _record

__all__ = ["ElectroThermalModel", "ElectroThermal"]


class _ThermalInput(Bag):
    __slots__ = ("heat_W", "current_A", "voltage_V", "conduction_fraction")


class _ThermalOutput(Bag):
    __slots__ = ("temperature_K", "case_K", "sink_K", "heat_W", "accounted_heat_W",
                 "estimated_heat_W", "switching_energy_J", "cooling_W")


class ElectroThermalModel:
    """A two-node Cauer junction/case or fixed-ambient Foster temperature model.

    The external heat power comes from a REAL circuit output via Model.connections.
    An optional semiconductor loss model supplies ADDITIONAL, estimated conduction
    heat using the current and the **current junction temperature**. No estimated
    heat is silently subtracted from electrical supply.
    """

    outputs_need_inputs: ClassVar[bool] = True
    observer: ClassVar[bool] = True  # Exclude separate thermal energy from electrical P-H closure.

    def __init__(
        self, network: LumpedThermalNetwork | FosterImpedance, *,
        loss_model: SemiconductorLossModel | None = None,
        initial_temperature_K: float | None = None,
        switching_devices: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(network, (LumpedThermalNetwork, FosterImpedance)):
            raise TypeError("network must be a Cauer/lumped network or Foster impedance")
        if loss_model is not None and not isinstance(loss_model, SemiconductorLossModel):
            raise TypeError("loss_model must be a SemiconductorLossModel")
        self.network = network
        self.loss_model = loss_model
        self.switching_devices = frozenset(switching_devices)
        self._foster = isinstance(network, FosterImpedance)
        self._temperature_names = (
            tuple(f"rise_{k}_K" for k in range(len(network.capacitances_J_K)))
            if self._foster else
            tuple(f"T_{name}_K" for name in network.names)
        )
        self.state_names = self._temperature_names
        self.discrete_state_names = ("last_transition_id", "switching_energy_J")
        reference = (network.ambient_K if initial_temperature_K is None
                     else _positive(initial_temperature_K, "initial_temperature_K"))
        values = ([reference - network.ambient_K] * len(self._temperature_names)
                  if self._foster else [reference] * len(self._temperature_names))
        self.state = _record("_ThermalState", self.state_names)(
            **dict(zip(self._temperature_names, values))
        )
        self.last_transition_id = -1
        self.switching_energy_J = 0.0
        self.input = _ThermalInput(heat_W=0.0, current_A=0.0,
                                   voltage_V=0.0, conduction_fraction=1.0)
        self.output = _ThermalOutput()
        self.set_outputs(0.0)

    def _states(self) -> np.ndarray:
        return np.asarray([getattr(self.state, name) for name in self._temperature_names],
                          dtype=float)

    def temperature_K(self) -> float:
        values = self._states()
        return (self.network.junction_K(values) if self._foster else float(values[0]))

    def _heat(self) -> tuple[float, float]:
        realized = float(self.input.heat_W)
        current = float(self.input.current_A)
        voltage = float(self.input.voltage_V)
        fraction = float(self.input.conduction_fraction)
        if (not all(math.isfinite(v) for v in (realized, current, voltage, fraction))
                or realized < 0.0 or not 0.0 <= fraction <= 1.0):
            raise ValueError("thermal heat/current/voltage inputs must be finite and physical")
        curve = None if self.loss_model is None else self.loss_model.conduction_voltage_V
        additional = float(fraction * abs(current) * _lookup(
            curve, current, voltage, self.temperature_K()
        ))
        return realized, additional

    def set_outputs(self, t: float) -> None:
        values = self._states()
        realized, additional = self._heat()
        self.output.temperature_K = self.temperature_K()
        self.output.case_K = (math.nan if self._foster or len(values) < 2
                              else float(values[1]))
        self.output.sink_K = (math.nan if self._foster or len(values) < 3
                              else float(values[-1]))
        self.output.accounted_heat_W = realized
        self.output.estimated_heat_W = additional
        self.output.heat_W = realized + additional
        self.output.switching_energy_J = self.switching_energy_J
        self.output.cooling_W = (math.nan if self._foster
                                 else self.network.cooling_power_W(values))

    def rhs(self, t: float) -> tuple[float, ...]:
        realized, additional = self._heat()
        p = realized + additional
        state = self._states()
        derivatives = (self.network.rates(state, p) if self._foster
                       else self.network.rates(state, {self.network.names[0]: p}))
        return tuple(float(value) for value in derivatives)

    def get_state(self) -> dict[str, int | float]:
        return {
            "last_transition_id": self.last_transition_id,
            "switching_energy_J": self.switching_energy_J,
        }

    def set_state(self, values: Mapping[str, Any]) -> None:
        unknown = set(values) - {"last_transition_id", "switching_energy_J"}
        if unknown:
            raise KeyError(f"unknown electrothermal state {sorted(unknown)}")
        if "last_transition_id" in values:
            value = values["last_transition_id"]
            if isinstance(value, bool) or not isinstance(value, int) or value < -1:
                raise ValueError("last_transition_id must be an integer >= -1")
            self.last_transition_id = value
        if "switching_energy_J" in values:
            value = float(values["switching_energy_J"])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("switching_energy_J must be finite and nonnegative")
            self.switching_energy_J = value

    def _transition_energy(self, event: SwitchingEvent, *, strict: bool) -> float | None:
        if self.loss_model is None:
            if strict:
                raise ValueError("switching events require a semiconductor loss model")
            return None
        if (event.kind == "reverse_recovery"
                and self.loss_model.turn_on_includes_reverse_recovery):
            if strict:
                raise ValueError("reverse recovery is already included in turn-on energy")
            return None
        field = {"turn_on": "turn_on_energy_J", "turn_off": "turn_off_energy_J",
                 "reverse_recovery": "reverse_recovery_energy_J"}[event.kind]
        curve = getattr(self.loss_model, field)
        if curve is None:
            if strict:
                raise ValueError(f"no {event.kind} energy curve was provided")
            return None
        return float(_lookup(curve, event.current_A, event.voltage_V,
                             self.temperature_K()))

    def _apply_transition_energy(self, energy: float, time_s: float) -> None:
        states = self._states()
        if self._foster:
            updated = self.network.committed_impulse(states, energy)
        else:
            updated = self.network.committed_impulse(
                states, {self.network.names[0]: energy}
            )
        for name, value in zip(self._temperature_names, updated):
            setattr(self.state, name, float(value))
        self.switching_energy_J += energy
        self.set_outputs(time_s)

    def apply_committed_transition(self, event: SwitchingEvent, *, event_id: int) -> float:
        """Apply one accepted event in J, once per monotonic device event sequence.

        Invoke only at an actual model boundary, then repack the existing model
        state with `model.get_initial_values()`. The monotonic event ID is kept in
        a restartable state so continuation cannot reapply an already counted
        event. Event IDs must increase monotonically per attachment.
        """
        if not isinstance(event, SwitchingEvent):
            raise TypeError("event must be a confirmed SwitchingEvent")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
            raise ValueError("event_id must be a nonnegative integer")
        if event_id <= self.last_transition_id:
            raise ValueError("event_id has already been committed or is out of order")
        energy = self._transition_energy(event, strict=True)
        assert energy is not None
        self._apply_transition_energy(energy, event.time_s)
        self.last_transition_id = event_id
        return energy

    def apply_committed_transitions(self, events: Any) -> float:
        """Apply selected, available device energies from one committed circuit boundary."""
        accepted = []
        for raw in events:
            event = raw if isinstance(raw, SwitchingEvent) else SwitchingEvent(
                raw.time_s, raw.kind, raw.current_A, raw.voltage_V, raw.device
            )
            if self.switching_devices and event.device not in self.switching_devices:
                continue
            energy = self._transition_energy(event, strict=False)
            if energy is not None:
                accepted.append((event, energy))
        if not accepted:
            return 0.0
        energy = sum(value for _event, value in accepted)
        self._apply_transition_energy(energy, accepted[-1][0].time_s)
        self.last_transition_id += len(accepted)
        return energy


@register_component_type
class ElectroThermal(Element):
    """Configure a passive, two-node electrothermal attachment in ``elements``.

    ``heat_source`` names a pre-existing scalar physical output (in W).
    This is the circuit's realized dissipation, never an extra electrical load.
    Optional ``current_source`` selects the actual device current for an ideal
    electrical model with a separate estimated conduction voltage. Those modes
    cannot be combined without proven disjoint heat mechanisms.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        c_j_J_K: float
        r_jc_K_W: float
        c_c_J_K: float
        r_ca_K_W: float
        ambient_K: float = 298.15
        initial_K: float = 298.15
        heat_source: str | None = field(default=None, metadata={"runtime": False})
        current_source: str | None = field(default=None, metadata={"runtime": False})
        switching_source: str | None = field(default=None, metadata={"runtime": False})
        switching_devices: tuple[str, ...] = field(default=(), metadata={"runtime": False})
        conduction_voltage_V: float | None = None
        turn_on_energy_J: float | None = None
        turn_off_energy_J: float | None = None
        reverse_recovery_energy_J: float | None = None
        turn_on_includes_reverse_recovery: bool = False
        type: str = "electrothermal"

        def __post_init__(self):
            try:
                for name in ("c_j_J_K", "r_jc_K_W", "c_c_J_K", "r_ca_K_W",
                             "ambient_K", "initial_K"):
                    _positive(getattr(self, name), name)
            except ValueError as exc:
                raise ConfigError(str(exc)) from None
            if self.conduction_voltage_V is not None:
                if (not math.isfinite(self.conduction_voltage_V)
                        or self.conduction_voltage_V < 0):
                    raise ConfigError("conduction_voltage_V must be nonnegative")
            if self.heat_source is not None and self.current_source is not None:
                raise ConfigError("heat_source and current_source are mutually exclusive")
            if (self.heat_source is None and self.current_source is None
                    and self.switching_source is None):
                raise ConfigError(
                    "electrothermal requires a heat, current or switching source"
                )
            if (self.heat_source is not None and
                    self.conduction_voltage_V is not None):
                raise ConfigError(
                    "electrically accounted and estimated conduction must not overlap"
                )
            if (self.current_source is not None and self.conduction_voltage_V is None):
                raise ConfigError("current_source requires a conduction voltage estimate")
            energies = (
                self.turn_on_energy_J,
                self.turn_off_energy_J,
                self.reverse_recovery_energy_J,
            )
            if self.switching_source is None and (
                    any(value is not None for value in energies)
                    or self.switching_devices or self.turn_on_includes_reverse_recovery):
                raise ConfigError("switching-loss fields require switching_source")
            if self.switching_source is not None and not any(
                    value is not None for value in energies):
                raise ConfigError("switching_source requires at least one transition energy")
            for name, value in zip((
                    "turn_on_energy_J", "turn_off_energy_J", "reverse_recovery_energy_J"
            ), energies):
                if value is not None and (not math.isfinite(value) or value < 0.0):
                    raise ConfigError(f"{name} must be finite and nonnegative")
            if type(self.turn_on_includes_reverse_recovery) is not bool:
                raise ConfigError("turn_on_includes_reverse_recovery must be bool")
            if (len(set(self.switching_devices)) != len(self.switching_devices)
                    or any(not name or any(
                        not part.isidentifier() for part in name.split(".")
                    ) for name in self.switching_devices)):
                raise ConfigError("switching_devices must contain unique device paths")

    type = "electrothermal"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        self.model = self._new_model(cfg)

    @staticmethod
    def _new_model(cfg: Any) -> ElectroThermalModel:
        loss_values = (
            cfg.conduction_voltage_V,
            cfg.turn_on_energy_J,
            cfg.turn_off_energy_J,
            cfg.reverse_recovery_energy_J,
        )
        model = (SemiconductorLossModel(
            conduction_voltage_V=cfg.conduction_voltage_V,
            turn_on_energy_J=cfg.turn_on_energy_J,
            turn_off_energy_J=cfg.turn_off_energy_J,
            reverse_recovery_energy_J=cfg.reverse_recovery_energy_J,
            turn_on_includes_reverse_recovery=cfg.turn_on_includes_reverse_recovery,
        ) if any(value is not None for value in loss_values) else None)
        network = LumpedThermalNetwork.cauer(
            [cfg.c_j_J_K, cfg.c_c_J_K], [cfg.r_jc_K_W, cfg.r_ca_K_W],
            ambient_K=cfg.ambient_K,
        )
        return ElectroThermalModel(network, loss_model=model,
                                  initial_temperature_K=cfg.initial_K,
                                  switching_devices=cfg.switching_devices)

    def subsystems(self) -> dict[str, ElectroThermalModel]:
        return {self.name: self.model}

    def connections(self) -> dict:
        source = (self.cfg.heat_source if self.cfg.heat_source is not None
                  else self.cfg.current_source)
        if source is None:
            return {}
        field = "heat_W" if self.cfg.heat_source is not None else "current_A"
        return {(self.model, field): source}

    def transition_bindings(self) -> tuple[tuple[str, ElectroThermalModel], ...]:
        return (() if self.cfg.switching_source is None else
                ((self.cfg.switching_source, self.model),))

    @staticmethod
    def port_bindings() -> tuple:
        return ()

    def signals(self) -> dict[str, float]:
        return {f"{self.name}.temperature_K": self.model.output.temperature_K,
                f"{self.name}.case_K": self.model.output.case_K,
                f"{self.name}.heat_W": self.model.output.heat_W,
                f"{self.name}.switching_energy_J": self.model.output.switching_energy_J,
                f"{self.name}.cooling_W": self.model.output.cooling_W}

    def retune(self, cfg: Any) -> bool:
        if (cfg.heat_source != self.cfg.heat_source
                or cfg.current_source != self.cfg.current_source
                or cfg.switching_source != self.cfg.switching_source
                or cfg.switching_devices != self.cfg.switching_devices):
            raise ConfigError("cannot retune electrothermal source connections during a run")
        if cfg == self.cfg:
            return False
        state = self.model._states()
        replacement = self._new_model(cfg)
        if len(replacement._temperature_names) != len(self.model._temperature_names):
            raise ConfigError("electrothermal retune cannot change the state layout")
        self.model.network = replacement.network
        self.model.loss_model = replacement.loss_model
        for name, value in zip(self.model._temperature_names, state):
            setattr(self.model.state, name, float(value))
        self.cfg = cfg
        return False
