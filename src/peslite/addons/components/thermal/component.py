"""Common contract for temperature-dependent physical components."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ....components.network import NetworkComponent
from ....solver import ConfigError

__all__ = ["ThermalComponent", "ThermalParams", "ThermalPropertyLaw"]


class ThermalPropertyLaw(Protocol):
    """Return named physical properties at one component temperature."""

    property_names: tuple[str, ...]

    def properties(self, temperature_K: float) -> Sequence[float]: ...


@dataclass(frozen=True, kw_only=True)
class ThermalParams:
    """Common temperature connection for thermal component adapters."""

    temperature_source: str = field(metadata={"runtime": False})

    def __post_init__(self) -> None:
        if not isinstance(self.temperature_source, str) or not self.temperature_source:
            raise ConfigError("temperature_source must name a physical output")


class ThermalComponent(NetworkComponent):
    """Physical component whose constitutive properties depend on temperature.

    Concrete components retain the ordinary network-component interface. The thermal addon only
    adds a named property law, a temperature input and a realized-loss output.
    """

    def __init__(self, thermal_law: ThermalPropertyLaw) -> None:
        names = tuple(getattr(thermal_law, "property_names", ()))
        if (
            not names
            or any(not isinstance(name, str) or not name.isidentifier() for name in names)
            or len(set(names)) != len(names)
            or not callable(getattr(thermal_law, "properties", None))
        ):
            raise TypeError(
                "a thermal property law needs unique property_names and properties(temperature_K)"
            )
        self.thermal_law = thermal_law
        self.thermal_property_names = names

    def temperature_K(self) -> float:
        """Return the connected physical temperature."""
        try:
            temperature = float(self.input.temperature_K)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("a thermal component needs a numeric temperature_K input") from None
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("thermal component temperature_K must be finite and positive")
        return temperature

    def _thermal_property_values(self) -> tuple[float, ...]:
        """Evaluate and validate the current constitutive-property vector."""
        try:
            values = tuple(self.thermal_law.properties(self.temperature_K()))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"cannot evaluate thermal component properties: {exc}") from None
        if len(values) != len(self.thermal_property_names):
            raise ValueError("thermal property values must match property_names")
        try:
            values = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("thermal component properties must be numeric") from None
        if not all(math.isfinite(value) for value in values):
            raise ValueError("thermal component properties must be finite")
        return values

    def thermal_properties(self) -> Mapping[str, float]:
        """Return the component's current constitutive properties by name."""
        return dict(zip(self.thermal_property_names, self._thermal_property_values()))

    def dissipated_power(self) -> float:
        """Return the realized loss published by the concrete component."""
        try:
            loss = float(self.output.loss_W)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("a thermal component must publish numeric loss_W") from None
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError("thermal component loss_W must be finite and nonnegative")
        return loss

    @staticmethod
    def supplied_power() -> float:
        return 0.0
