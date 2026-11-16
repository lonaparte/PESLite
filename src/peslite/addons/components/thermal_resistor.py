"""A temperature-dependent *electrical* R-L element.

Unlike estimated thermal loss, this element places R(T)*i in the actual
electrical voltage equation and declares the matching dissipated R(T)*i².
Temperature is connected through the existing Model input/output graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ...components.network import Element, NetworkComponent, _register_assembled_type
from ...solver import Bag, ConfigError, Empty, Port, StoragePort

__all__ = ["ThermalResistor", "ThermalResistorModel"]


class _State(Bag):
    __slots__ = ("i",)


class _Input(Bag):
    __slots__ = ("temperature_K",)


class _Output(Bag):
    __slots__ = ("i", "heat_W", "resistance_Ohm")


class ThermalResistorModel(NetworkComponent):
    """Scalar R-L branch to ground with a live thermal resistance law."""

    state_names: ClassVar[tuple[str, ...]] = ("i",)
    outputs_need_inputs: ClassVar[bool] = True

    def __init__(self, r_reference_Ohm: float, alpha_per_K: float, l_H: float,
                 reference_K: float) -> None:
        self.r_reference_Ohm = r_reference_Ohm
        self.alpha_per_K = alpha_per_K
        self.l_H = l_H
        self.reference_K = reference_K
        self.state = _State(i=0.0)
        self.input = _Input(temperature_K=reference_K)
        # Calculated heat is a read-only observation of the same electrical
        # current state, not an extra netlist power branch or output callback.
        record = type("_ThermalResistorOutput", (Bag,), {
            "__slots__": ("i",),
            "heat_W": property(lambda out: self._loss(0.0)),
            "resistance_Ohm": property(lambda out: self.resistance()),
        })
        self.output = record(i=0.0)
        self.port = Port(self, "power", standalone="voltage", electrical_domain="dc")
        self.ports = (self.port,)
        self.storage = (StoragePort(
            "i", l_H, 1.0, ((self.port, "voltage", 1.0),), "inductor"
        ),)

    def resistance(self) -> float:
        temperature = float(self.input.temperature_K)
        value = self.r_reference_Ohm * (
            1.0 + self.alpha_per_K * (temperature - self.reference_K)
        )
        if not math.isfinite(temperature) or temperature <= 0.0 or (
            not math.isfinite(value) or value <= 0.0
        ):
            raise ValueError("thermal resistance must remain positive and finite")
        return value

    def _drop(self, _t: float) -> float:
        return self.resistance() * float(self.state.i)

    def _loss(self, _t: float) -> float:
        return self.resistance() * float(self.state.i) ** 2

    def network(self, nb: Any) -> None:
        from ...solver import netlist as nl

        end = nb.internal()
        # This constitutive voltage uses the already registered R-L current state.
        # It is *not* an extra ideal supply or a duplicate network branch.
        nb.add(nl.VoltageSource(
            nb.node(self.port), end,
            nb.function(self._drop, ("temperature_K",), states=("i",)),
        ))
        nb.add(nl.Inductor(end, None, self.l_H, nb.state(self, "i")))
        nb.add(nl.Dissipation(
            nb.function(self._loss, ("temperature_K",), states=("i",))
        ))
        nb.port_output(self, "i", self.port, "current")
        nb.losses(self)


@_register_assembled_type
class ThermalResistor(Element):
    """Bind an electrical resistor to an electrothermal output.

    Install ``peslite`` and use ``elements`` of type ``thermal_resistor``.
    The temperature path is a scalar physical output, e.g. ``chip.junction_K``.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bus: str
        temperature_source: str = field(metadata={"runtime": False})
        r_reference_Ohm: float
        l_H: float
        alpha_per_K: float = 0.0
        reference_K: float = 298.15
        type: str = "thermal_resistor"

        def __post_init__(self):
            for label in ("r_reference_Ohm", "l_H", "reference_K"):
                value = getattr(self, label)
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                    raise ConfigError(f"{label} must be finite and positive")
            if not math.isfinite(self.alpha_per_K):
                raise ConfigError("alpha_per_K must be finite")
            if not isinstance(self.temperature_source, str) or not self.temperature_source:
                raise ConfigError("temperature_source must be a physical output path")

    type = "thermal_resistor"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        if buses[cfg.bus].phases != 1:
            raise ConfigError("thermal_resistor supports scalar/DC buses only")
        self.name, self.cfg = name, cfg
        self.model = ThermalResistorModel(
            cfg.r_reference_Ohm, cfg.alpha_per_K, cfg.l_H, cfg.reference_K
        )

    def subsystems(self) -> dict[str, ThermalResistorModel]:
        return {self.name: self.model}

    def connections(self) -> dict:
        return {(self.model, "temperature_K"): self.cfg.temperature_source}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return ((self.cfg.bus, self.model.port),)

    def signals(self) -> dict[str, float]:
        return {f"{self.name}.i": self.model.output.i,
                f"{self.name}.heat_W": self.model.output.heat_W,
                f"{self.name}.r_Ohm": self.model.output.resistance_Ohm}

    def retune(self, cfg: Any) -> bool:
        if cfg.bus != self.cfg.bus or cfg.temperature_source != self.cfg.temperature_source:
            raise ConfigError("thermal resistor wiring cannot change during a run")
        self.cfg = cfg
        self.model.r_reference_Ohm = cfg.r_reference_Ohm
        self.model.alpha_per_K = cfg.alpha_per_K
        self.model.l_H = cfg.l_H
        self.model.reference_K = cfg.reference_K
        return True
