"""Shared cooling networks for the thermal addon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import numpy as np

from ....components.network import Element, register_component_type
from ....solver import ConfigError
from .network import LumpedThermalNetwork, ThermalLink, _finite, _record

__all__ = ["SharedThermalModel", "SharedThermalNetwork"]


class SharedThermalModel:
    """Couple independent heat inputs to one shared RC graph."""

    outputs_need_inputs: ClassVar[bool] = True
    observer: ClassVar[bool] = True

    def __init__(
        self,
        network: LumpedThermalNetwork,
        sources: Mapping[str, str],
        *,
        initial_K: float | None = None,
    ) -> None:
        if not isinstance(network, LumpedThermalNetwork):
            raise TypeError("shared thermal backend requires a physical lumped network")
        if not sources:
            raise ValueError("shared heat model needs at least one source")
        if (
            any(not isinstance(name, str) or not name.isidentifier() for name in sources)
            or any(node not in network.names for node in sources.values())
        ):
            raise ValueError("heat sources need identifier names and existing thermal nodes")
        self.network = network
        self.source_nodes = dict(sources)
        self.state_names = tuple(f"T_{node}_K" for node in network.names)
        self.input_names = tuple(f"heat_{name}_W" for name in sources)
        self.temperature_names = tuple(f"temperature_{node}_K" for node in network.names)
        self.output_names = (*self.temperature_names, "heat_total_W", "cooling_W")
        initial = network.ambient_K if initial_K is None else _finite(initial_K, "initial_K")
        if initial <= 0:
            raise ValueError("initial_K must be positive")
        self.state = _record("_SharedThermalState", self.state_names)(
            **dict.fromkeys(self.state_names, initial)
        )
        self.input = _record("_SharedThermalInput", self.input_names)(
            **dict.fromkeys(self.input_names, 0.0)
        )
        self.output = _record("_SharedThermalOutput", self.output_names)()
        self.set_outputs(0.0)

    def _temperatures(self) -> np.ndarray:
        return np.fromiter(
            (getattr(self.state, name) for name in self.state_names),
            dtype=float,
            count=len(self.state_names),
        )

    def _heat(self) -> dict[str, float]:
        heat = {name: 0.0 for name in self.network.names}
        for source, node in self.source_nodes.items():
            value = _finite(getattr(self.input, f"heat_{source}_W"), f"{source}.heat_W")
            if value < 0:
                raise ValueError("passive thermal sources require nonnegative losses")
            heat[node] += value
        return heat

    def set_outputs(self, _t: float) -> None:
        temperatures = self._temperatures()
        for name, temperature in zip(self.temperature_names, temperatures):
            setattr(self.output, name, float(temperature))
        self.output.heat_total_W = sum(self._heat().values())
        self.output.cooling_W = self.network.cooling_power_W(temperatures)

    def rhs(self, _t: float) -> tuple[float, ...]:
        return tuple(self.network.rates(self._temperatures(), self._heat()))


@register_component_type
class SharedThermalNetwork(Element):
    """Connect several heat signals to one configured thermal network."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        capacities_J_K: dict[str, float] = field(metadata={"structural": True})
        links: list[dict] = field(metadata={"structural": True})
        sources: dict[str, dict[str, str]] = field(metadata={"structural": True})
        ambient_K: float = 298.15
        initial_K: float = 298.15
        type: str = "shared_thermal_network"

        def __post_init__(self) -> None:
            if not isinstance(self.sources, Mapping) or not self.sources:
                raise ConfigError("shared thermal network requires heat sources")
            for name, source in self.sources.items():
                if (
                    not isinstance(name, str)
                    or not name.isidentifier()
                    or not isinstance(source, Mapping)
                    or set(source) != {"path", "node"}
                    or not all(isinstance(value, str) and value for value in source.values())
                ):
                    raise ConfigError("each shared heat source needs a named path and node")
            try:
                links = [ThermalLink(**item) for item in self.links]
                LumpedThermalNetwork(
                    self.capacities_J_K,
                    links,
                    ambient_K=self.ambient_K,
                )
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"invalid shared thermal network: {exc}") from None

    type = "shared_thermal_network"

    def __init__(self, name: str, cfg: Any, _buses: dict, _params: Any) -> None:
        self.name, self.cfg = name, cfg
        network = LumpedThermalNetwork(
            cfg.capacities_J_K,
            [ThermalLink(**item) for item in cfg.links],
            ambient_K=cfg.ambient_K,
        )
        self.model = SharedThermalModel(
            network,
            {source: item["node"] for source, item in cfg.sources.items()},
            initial_K=cfg.initial_K,
        )

    def subsystems(self) -> dict[str, SharedThermalModel]:
        return {self.name: self.model}

    def connections(self) -> dict:
        return {
            (self.model, f"heat_{name}_W"): source["path"]
            for name, source in self.cfg.sources.items()
        }

    @staticmethod
    def port_bindings() -> tuple:
        return ()

    def signals(self) -> dict[str, float]:
        return {
            f"{self.name}.{field}": getattr(self.model.output, field)
            for field in self.model.output_names
        }
