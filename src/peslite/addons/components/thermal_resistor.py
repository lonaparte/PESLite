"""A temperature-dependent electrical resistor.

Unlike estimated thermal loss, this element places R(T)*i in the actual
electrical current equation and declares the matching dissipated R(T)*i².
Temperature is connected through the existing Model input/output graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ...components.network import Element, register_component_type
from ...solver import Bag, ConfigError, Empty, OutputMethod, Port

__all__ = ["ThermalResistor", "ThermalResistorModel"]


class _Input(Bag):
    __slots__ = ("voltage", "temperature_K")


class _Output(Bag):
    __slots__ = ("current", "heat_W", "resistance_Ohm")


class ThermalResistorModel:
    """Memoryless scalar resistor with temperature supplied as a separate signal."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = True
    storage: ClassVar[tuple] = ()
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod(
            "set_outputs",
            inputs=("voltage", "temperature_K"),
            outputs=("current", "heat_W", "resistance_Ohm"),
        ),
    )

    def __init__(self, r_reference_Ohm: float, alpha_per_K: float,
                 reference_K: float, phases: int = 1) -> None:
        if phases not in (1, 3):
            raise ValueError("phases must be 1 or 3")
        self.r_reference_Ohm = r_reference_Ohm
        self.alpha_per_K = alpha_per_K
        self.reference_K = reference_K
        self.power_scale = 1.5 if phases == 3 else 1.0
        zero = 0j if phases == 3 else 0.0
        self.state = Empty()
        self.input = _Input(voltage=zero, temperature_K=reference_K)
        self.output = _Output(
            current=zero,
            heat_W=0.0,
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
        temperature = float(self.input.temperature_K)
        value = self.r_reference_Ohm * (
            1.0 + self.alpha_per_K * (temperature - self.reference_K)
        )
        if not math.isfinite(temperature) or temperature <= 0.0 or (
            not math.isfinite(value) or value <= 0.0
        ):
            raise ValueError("thermal resistance must remain positive and finite")
        return value

    def set_outputs(self, _t: float) -> None:
        resistance = self.resistance()
        current = self.input.voltage / resistance
        self.output.resistance_Ohm = resistance
        self.output.current = current
        self.output.heat_W = self.power_scale * resistance * abs(current) ** 2

    @staticmethod
    def rhs(_t: float) -> tuple:
        return ()

    def dissipated_power(self) -> float:
        return float(self.output.heat_W)

    @staticmethod
    def supplied_power() -> float:
        return 0.0


@register_component_type
class ThermalResistor(Element):
    """Bind an electrical resistor to an electrothermal output.

    Install ``peslite`` and use ``elements`` of type ``thermal_resistor``.
    The temperature path is a scalar physical output, e.g. ``chip.temperature_K``.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bus: str
        temperature_source: str = field(metadata={"runtime": False})
        r_reference_Ohm: float
        alpha_per_K: float = 0.0
        reference_K: float = 298.15
        type: str = "thermal_resistor"

        def __post_init__(self):
            for label in ("r_reference_Ohm", "reference_K"):
                value = getattr(self, label)
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                    raise ConfigError(f"{label} must be finite and positive")
            if not math.isfinite(self.alpha_per_K):
                raise ConfigError("alpha_per_K must be finite")
            if not isinstance(self.temperature_source, str) or not self.temperature_source:
                raise ConfigError("temperature_source must be a physical output path")

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
                f"{self.name}.heat_W": self.model.output.heat_W,
                f"{self.name}.r_Ohm": self.model.output.resistance_Ohm}

    def retune(self, cfg: Any) -> bool:
        if cfg.bus != self.cfg.bus or cfg.temperature_source != self.cfg.temperature_source:
            raise ConfigError("thermal resistor wiring cannot change during a run")
        self.cfg = cfg
        self.model.r_reference_Ohm = cfg.r_reference_Ohm
        self.model.alpha_per_K = cfg.alpha_per_K
        self.model.reference_K = cfg.reference_K
        return False
