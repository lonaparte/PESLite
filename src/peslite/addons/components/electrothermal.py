"""Electrothermal add-ons composed with PESLite's existing electrical Model.

Thermal state is an ordinary restartable subsystem; no second electrical circuit,
RHS integrator or global thermal configuration is introduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar

import numpy as np

from ...components.network import Element, _register_assembled_type
from ...solver import Bag, ConfigError, Empty
from ..functions.semiconductor_losses import SemiconductorLossModel, SwitchingEvent, _lookup
from .thermal_network import FosterImpedance, LumpedThermalNetwork

__all__ = ["ElectroThermalModel", "ElectroThermal"]


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or (
        not math.isfinite(value) or value <= 0.0
    ):
        raise ConfigError(f"{name} must be finite and > 0")
    return float(value)


@lru_cache(maxsize=None)
def _record(names: tuple[str, ...]) -> type[Bag]:
    return type("_ThermalState_" + "_".join(names), (Bag,), {"__slots__": names})


class _ThermalInput(Bag):
    __slots__ = ("accounted_heat_W", "current_A", "voltage_V", "conduction_fraction")


class _ThermalOutput(Bag):
    __slots__ = ("junction_K", "case_K", "sink_K", "heat_W", "accounted_heat_W",
                 "estimated_heat_W", "cooling_W")


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
    ) -> None:
        if not isinstance(network, (LumpedThermalNetwork, FosterImpedance)):
            raise TypeError("network must be a Cauer/lumped network or Foster impedance")
        if loss_model is not None and not isinstance(loss_model, SemiconductorLossModel):
            raise TypeError("loss_model must be a SemiconductorLossModel")
        self.network = network
        self.loss_model = loss_model
        self._foster = isinstance(network, FosterImpedance)
        self._temperature_names = (
            tuple(f"rise_{k}_K" for k in range(len(network.capacitances_J_K)))
            if self._foster else
            tuple(f"T_{name}_K" for name in network.names)
        )
        self.state_names = (*self._temperature_names, "last_transition_id")
        reference = (network.ambient_K if initial_temperature_K is None
                     else _positive("initial_temperature_K", initial_temperature_K))
        values = ([reference - network.ambient_K] * len(self._temperature_names)
                  if self._foster else [reference] * len(self._temperature_names))
        self.state = _record(self.state_names)(
            **dict(zip(self._temperature_names, values)), last_transition_id=-1.0
        )
        self.input = _ThermalInput(accounted_heat_W=0.0, current_A=0.0,
                                   voltage_V=0.0, conduction_fraction=1.0)
        self.output = _ThermalOutput()
        self.set_outputs(0.0)

    def _states(self) -> np.ndarray:
        return np.asarray([getattr(self.state, name) for name in self._temperature_names],
                          dtype=float)

    def junction_K(self) -> float:
        values = self._states()
        return (self.network.junction_K(values) if self._foster else float(values[0]))

    def _heat(self) -> tuple[float, float]:
        realized = float(self.input.accounted_heat_W)
        current = float(self.input.current_A)
        voltage = float(self.input.voltage_V)
        fraction = float(self.input.conduction_fraction)
        if (not all(math.isfinite(v) for v in (realized, current, voltage, fraction))
                or realized < 0.0 or not 0.0 <= fraction <= 1.0):
            raise ValueError("thermal heat/current/voltage inputs must be finite and physical")
        curve = None if self.loss_model is None else self.loss_model.conduction_voltage_V
        additional = float(fraction * abs(current) * _lookup(
            curve, current, voltage, self.junction_K()
        ))
        return realized, additional

    def set_outputs(self, t: float) -> None:
        values = self._states()
        realized, additional = self._heat()
        self.output.junction_K = self.junction_K()
        self.output.case_K = (math.nan if self._foster or len(values) < 2
                              else float(values[1]))
        self.output.sink_K = (math.nan if self._foster or len(values) < 3
                              else float(values[-1]))
        self.output.accounted_heat_W = realized
        self.output.estimated_heat_W = additional
        self.output.heat_W = realized + additional
        self.output.cooling_W = (math.nan if self._foster
                                 else self.network.cooling_power_W(values))

    def rhs(self, t: float) -> tuple[float, ...]:
        realized, additional = self._heat()
        p = realized + additional
        state = self._states()
        derivatives = (self.network.rates(state, p) if self._foster
                       else self.network.rates(state, {self.network.names[0]: p}))
        return (*tuple(float(value) for value in derivatives), 0.0)

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
        if event_id <= self.state.last_transition_id:
            raise ValueError("event_id has already been committed or is out of order")
        if self.loss_model is None:
            raise ValueError("switching events require a semiconductor loss model")
        if (event.kind == "reverse_recovery"
                and self.loss_model.turn_on_includes_reverse_recovery):
            raise ValueError("reverse recovery is already included in turn-on energy")
        field = {"turn_on": "turn_on_energy_J", "turn_off": "turn_off_energy_J",
                 "reverse_recovery": "reverse_recovery_energy_J"}[event.kind]
        curve = getattr(self.loss_model, field)
        if curve is None:
            raise ValueError(f"no {event.kind} energy curve was provided")
        energy = float(_lookup(curve, event.current_A, event.voltage_V,
                               self.junction_K()))
        states = self._states()
        if self._foster:
            updated = self.network.committed_impulse(states, energy)
        else:
            updated = self.network.committed_impulse(
                states, {self.network.names[0]: energy}
            )
        for name, value in zip(self._temperature_names, updated):
            setattr(self.state, name, float(value))
        self.state.last_transition_id = float(event_id)
        self.set_outputs(event.time_s)
        return energy


@_register_assembled_type
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
        conduction_voltage_V: float | None = None
        type: str = "electrothermal"

        def __post_init__(self):
            for name in ("c_j_J_K", "r_jc_K_W", "c_c_J_K", "r_ca_K_W",
                         "ambient_K", "initial_K"):
                _positive(name, getattr(self, name))
            if self.conduction_voltage_V is not None:
                if (not math.isfinite(self.conduction_voltage_V)
                        or self.conduction_voltage_V < 0):
                    raise ConfigError("conduction_voltage_V must be nonnegative")
            if bool(self.heat_source) == bool(self.current_source):
                raise ConfigError("select exactly one of heat_source or current_source")
            if (self.heat_source is not None and
                    self.conduction_voltage_V is not None):
                raise ConfigError("electrically accounted and estimated conduction must not overlap")
            if (self.current_source is not None and self.conduction_voltage_V is None):
                raise ConfigError("current_source requires a conduction voltage estimate")

    type = "electrothermal"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        self.model = self._new_model(cfg)

    @staticmethod
    def _new_model(cfg: Any) -> ElectroThermalModel:
        model = (SemiconductorLossModel(conduction_voltage_V=cfg.conduction_voltage_V)
                 if cfg.conduction_voltage_V is not None else None)
        network = LumpedThermalNetwork.cauer(
            [cfg.c_j_J_K, cfg.c_c_J_K], [cfg.r_jc_K_W, cfg.r_ca_K_W],
            ambient_K=cfg.ambient_K,
        )
        return ElectroThermalModel(network, loss_model=model,
                                  initial_temperature_K=cfg.initial_K)

    def subsystems(self) -> dict[str, ElectroThermalModel]:
        return {self.name: self.model}

    def connections(self) -> dict:
        source = self.cfg.heat_source or self.cfg.current_source
        field = ("accounted_heat_W" if self.cfg.heat_source is not None
                 else "current_A")
        return {(self.model, field): source}

    @staticmethod
    def port_bindings() -> tuple:
        return ()

    def signals(self) -> dict[str, float]:
        return {f"{self.name}.junction_K": self.model.output.junction_K,
                f"{self.name}.case_K": self.model.output.case_K,
                f"{self.name}.heat_W": self.model.output.heat_W,
                f"{self.name}.cooling_W": self.model.output.cooling_W}

    def retune(self, cfg: Any) -> bool:
        if (cfg.heat_source != self.cfg.heat_source
                or cfg.current_source != self.cfg.current_source):
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
