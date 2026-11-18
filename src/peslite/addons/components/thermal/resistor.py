"""A temperature-dependent electrical resistor.

Unlike estimated thermal loss, this element places R(T)*i in the actual
electrical current equation and declares the matching dissipated R(T)*i².
Temperature is connected through the existing Model input/output graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from ....components.network import Element, register_component_type
from ....solver import Bag, ConfigError, Empty, OutputMethod, Port
from .component import ThermalComponent, ThermalParams, ThermalPropertyLaw

__all__ = ["ThermalResistor", "ThermalResistorModel"]


class _Input(Bag):
    __slots__ = ("voltage", "temperature_K")


class _Output(Bag):
    __slots__ = ("current", "loss_W", "resistance_Ohm")


@dataclass(frozen=True)
class _LinearResistance:
    property_names = ("resistance_Ohm",)

    reference_Ohm: float
    coefficient_per_K: float
    reference_K: float

    def properties(self, temperature_K: float) -> tuple[float]:
        return (self.reference_Ohm * (
            1.0 + self.coefficient_per_K * (temperature_K - self.reference_K)
        ),)


class ThermalResistorModel(ThermalComponent):
    """Memoryless scalar resistor with temperature supplied as a separate signal."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = True
    storage: ClassVar[tuple] = ()
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod(
            "set_outputs",
            inputs=("voltage", "temperature_K"),
            outputs=("current", "loss_W", "resistance_Ohm"),
        ),
    )

    def __init__(self, r_reference_Ohm: float, alpha_per_K: float,
                 reference_K: float, phases: int = 1, *,
                 thermal_law: ThermalPropertyLaw | None = None) -> None:
        if phases not in (1, 3):
            raise ValueError("phases must be 1 or 3")
        super().__init__(thermal_law or _LinearResistance(
            r_reference_Ohm, alpha_per_K, reference_K
        ))
        if self.thermal_property_names != ("resistance_Ohm",):
            raise ValueError("a thermal resistor law must provide resistance_Ohm")
        self.power_scale = 1.5 if phases == 3 else 1.0
        zero = 0j if phases == 3 else 0.0
        self.state = Empty()
        self.input = _Input(voltage=zero, temperature_K=reference_K)
        self.output = _Output(
            current=zero,
            loss_W=0.0,
            resistance_Ohm=r_reference_Ohm,
        )
        self.port = Port(
            self,
            "power",
            scale=self.power_scale,
            effort="input.voltage",
            flow="output.current",
        )
        self.ports = (self.port,)

    def resistance(self) -> float:
        value = self._thermal_property_values()[0]
        if value <= 0.0:
            raise ValueError("thermal resistance must remain positive and finite")
        return value

    def set_outputs(self, _t: float) -> None:
        resistance = self.resistance()
        current = self.input.voltage / resistance
        self.output.resistance_Ohm = resistance
        self.output.current = current
        self.output.loss_W = self.power_scale * resistance * abs(current) ** 2

    @staticmethod
    def rhs(_t: float) -> tuple:
        return ()

@register_component_type
class ThermalResistor(Element):
    """Bind an electrical resistor to an electrothermal output.

    Use an ``elements`` entry of type ``thermal_resistor``.
    The temperature path is a scalar physical output, e.g. ``chip.temperature_K``.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params(ThermalParams):
        bus: str
        r_reference_Ohm: float
        alpha_per_K: float = 0.0
        reference_K: float = 298.15
        type: str = "thermal_resistor"

        def __post_init__(self):
            super().__post_init__()
            for label in ("r_reference_Ohm", "reference_K"):
                value = getattr(self, label)
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                    raise ConfigError(f"{label} must be finite and positive")
            if not math.isfinite(self.alpha_per_K):
                raise ConfigError("alpha_per_K must be finite")

    type = "thermal_resistor"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        self.model = ThermalResistorModel(
            cfg.r_reference_Ohm,
            cfg.alpha_per_K,
            cfg.reference_K,
            buses[cfg.bus].phases,
        )

    def subsystems(self) -> dict[str, ThermalResistorModel]:
        return {self.name: self.model}

    def connections(self) -> dict:
        return {(self.model, "temperature_K"): self.cfg.temperature_source}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return ((self.cfg.bus, self.model.port),)

    def signals(self) -> dict[str, complex | float]:
        return {f"{self.name}.i": self.model.output.current,
                f"{self.name}.loss_W": self.model.output.loss_W,
                f"{self.name}.r_Ohm": self.model.output.resistance_Ohm}

    def retune(self, cfg: Any) -> bool:
        if cfg.bus != self.cfg.bus or cfg.temperature_source != self.cfg.temperature_source:
            raise ConfigError("thermal resistor wiring cannot change during a run")
        self.cfg = cfg
        self.model.thermal_law = _LinearResistance(
            cfg.r_reference_Ohm, cfg.alpha_per_K, cfg.reference_K
        )
        return False
