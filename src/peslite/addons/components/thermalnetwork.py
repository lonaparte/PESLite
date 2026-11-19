"""Compact, passive, SI thermal RC networks for the thermal addon.

One thermal state belongs to each capacitive node. Positive heat (W) enters a
node; conduction through R (K/W) and convection to fixed ambient remove heat.
These equations are evaluated by PESLite's existing Model and solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from ...components.network import Element, register_component_type
from ...solver import Bag, ConfigError

__all__ = [
    "ThermalLink", "LumpedThermalNetwork", "FosterImpedance",
    "SharedThermalModel", "SharedThermalNetwork",
]


def _positive(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a finite positive number") from None
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return number


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite") from None
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@lru_cache(maxsize=None)
def _record(prefix: str, names: tuple[str, ...]) -> type[Bag]:
    """Return one cached state/input/output record type for a field layout."""
    return type(prefix, (Bag,), {"__slots__": names})


@dataclass(frozen=True)
class ThermalLink:
    """Thermal R in K/W between two named nodes, or a node and ambient (other=None)."""

    node: str
    other: str | None
    resistance_K_W: float

    def __post_init__(self):
        object.__setattr__(
            self, "resistance_K_W", _positive(self.resistance_K_W, "resistance_K_W")
        )
        if self.other is not None and self.node == self.other:
            raise ValueError("thermal link endpoints must differ")


class LumpedThermalNetwork:
    """Temperature dynamics of named capacitances and conductive/convective links.

    An ambient link is required for physical steady state. A node can represent
    a junction, case, shared heat sink or cooling plate. Multiple sources sharing
    the same sink are represented by ONE network, not duplicated electrical parts.
    """

    def __init__(self, capacities_J_K: Mapping[str, float], links: Sequence[ThermalLink],
                 *, ambient_K: float = 298.15) -> None:
        if not capacities_J_K:
            raise ValueError("a thermal network needs at least one capacitive node")
        if any(not isinstance(name, str) or not name.isidentifier()
               for name in capacities_J_K):
            raise ValueError("thermal node names must be identifiers")
        self.names = tuple(capacities_J_K)
        self.capacities_J_K = np.array([
            _positive(capacity, f"{name}.capacity_J_K")
            for name, capacity in capacities_J_K.items()
        ], dtype=float)
        self.ambient_K = _positive(ambient_K, "ambient_K")
        self.links = tuple(links)
        positions = {name: i for i, name in enumerate(self.names)}
        self._connections: list[tuple[int, int, float]] = []
        self._ambient: list[tuple[int, float]] = []
        for link in self.links:
            if not isinstance(link, ThermalLink):
                raise TypeError("links must contain ThermalLink entries")
            if link.node not in positions:
                raise ValueError(f"unknown thermal node {link.node!r}")
            conductance = 1.0 / link.resistance_K_W
            a = positions[link.node]
            if link.other is None:
                self._ambient.append((a, conductance))
            else:
                if link.other not in positions:
                    raise ValueError(f"unknown thermal node {link.other!r}")
                self._connections.append((a, positions[link.other], conductance))
        if not self._ambient:
            raise ValueError("at least one link to ambient is required")
        # Every node must conduct heat to ambient through the resistance graph.
        reachable = {idx for idx, _ in self._ambient}
        for _ in self.names:
            previous = len(reachable)
            for a, b, _g in self._connections:
                if a in reachable or b in reachable:
                    reachable.update((a, b))
            if len(reachable) == previous:
                break
        if len(reachable) != len(self.names):
            raise ValueError("each thermal node must have a path to ambient")
        self._positions = positions
        self.capacities_J_K.setflags(write=False)

    @classmethod
    def cauer(cls, capacities_J_K: Sequence[float], resistances_K_W: Sequence[float],
              *, ambient_K: float = 298.15) -> "LumpedThermalNetwork":
        """Physical junction-to-ambient ladder with N capacities and N resistances.

        The first N-1 resistances link neighboring nodes; the last attaches the
        final case/heatsink node to the fixed ambient temperature.
        """
        if not capacities_J_K or len(capacities_J_K) != len(resistances_K_W):
            raise ValueError("Cauer ladder needs equally sized, nonempty C and R lists")
        names = tuple(f"node_{index}" for index in range(len(capacities_J_K)))
        edges = [
            ThermalLink(names[index], names[index + 1], resistance)
            for index, resistance in enumerate(resistances_K_W[:-1])
        ]
        edges.append(ThermalLink(names[-1], None, resistances_K_W[-1]))
        return cls(dict(zip(names, capacities_J_K)), edges, ambient_K=ambient_K)

    def _array(self, temperatures_K: Sequence[float]) -> np.ndarray:
        x = np.asarray(temperatures_K, dtype=float)
        # Numerical/energy-structure probes may evaluate trial states outside the
        # physical temperature range. Only committed impulses enforce positive K;
        # otherwise the solver's speculative RHS would reject audit probes.
        if x.shape != (len(self.names),) or not np.isfinite(x).all():
            raise ValueError("temperatures_K need one finite K per node")
        return x

    def rates(self, temperatures_K: Sequence[float],
              heat_W: Mapping[str, float]) -> np.ndarray:
        """Return Kelvin/s with positive heat entering named nodes."""
        x = self._array(temperatures_K)
        unknown = set(heat_W) - self._positions.keys()
        if unknown:
            raise ValueError(f"heat sources name unknown thermal nodes: {sorted(unknown)}")
        q = np.zeros_like(x)
        for name, value in heat_W.items():
            q[self._positions[name]] += _finite(value, f"{name}.heat_W")
        for a, b, conductance in self._connections:
            flow = conductance * (x[b] - x[a])
            q[a] += flow
            q[b] -= flow
        for a, conductance in self._ambient:
            q[a] += conductance * (self.ambient_K - x[a])
        return q / self.capacities_J_K

    def cooling_power_W(self, temperatures_K: Sequence[float]) -> float:
        """Net heat rejected to the ambient, positive for hot components."""
        x = self._array(temperatures_K)
        return float(sum(g * (x[a] - self.ambient_K) for a, g in self._ambient))

    def stored_excess_heat_J(self, temperatures_K: Sequence[float]) -> float:
        """Thermal energy relative to constant ambient, in J."""
        x = self._array(temperatures_K)
        return float(np.dot(self.capacities_J_K, x - self.ambient_K))

    def committed_impulse(self, temperatures_K: Sequence[float],
                          energy_J: Mapping[str, float]) -> np.ndarray:
        """Apply a confirmed energy impulse to the owning capacitive node exactly."""
        x = self._array(temperatures_K).copy()
        for name, energy in energy_J.items():
            if name not in self._positions:
                raise ValueError(f"unknown thermal node {name!r}")
            x[self._positions[name]] += (
                _finite(energy, f"{name}.energy_J") / self.capacities_J_K[self._positions[name]]
            )
        if np.any(x <= 0):
            raise ValueError("heat impulse produced nonphysical temperature")
        return x


class FosterImpedance:
    """Parallel Foster thermal impedance referenced to a *fixed* ambient.

    Each branch has a temperature rise x_i and obeys
    d(x_i)/dt = (P - x_i/R_i)/C_i; T_j = T_ambient + sum(x_i).
    These are behavioral impedance states, not physical case or sink nodes:
    Foster branches cannot be directly joined into a shared heat-sink network.
    """

    def __init__(self, resistances_K_W: Sequence[float], capacitances_J_K: Sequence[float],
                 *, ambient_K: float = 298.15):
        if not resistances_K_W or len(resistances_K_W) != len(capacitances_J_K):
            raise ValueError("Foster impedance needs equally sized nonempty R and C lists")
        self.resistances_K_W = np.array([
            _positive(v, "resistance_K_W") for v in resistances_K_W], dtype=float)
        self.capacitances_J_K = np.array([
            _positive(v, "capacitance_J_K") for v in capacitances_J_K], dtype=float)
        self.ambient_K = _positive(ambient_K, "ambient_K")
        self.resistances_K_W.setflags(write=False)
        self.capacitances_J_K.setflags(write=False)

    def rates(self, rises_K: Sequence[float], heat_W: float) -> np.ndarray:
        x = np.asarray(rises_K, dtype=float)
        if x.shape != self.resistances_K_W.shape or not np.isfinite(x).all():
            raise ValueError("Foster rises must match the branch count")
        q = _finite(heat_W, "heat_W")
        return (q - x / self.resistances_K_W) / self.capacitances_J_K

    def junction_K(self, rises_K: Sequence[float]) -> float:
        x = np.asarray(rises_K, dtype=float)
        if x.shape != self.resistances_K_W.shape or not np.isfinite(x).all():
            raise ValueError("Foster rises must match the branch count")
        return float(self.ambient_K + np.sum(x))

    def committed_impulse(self, rises_K: Sequence[float], energy_J: float) -> np.ndarray:
        """Advance every branch's rise by E/C at an accepted impulse."""
        x = np.asarray(rises_K, dtype=float)
        if x.shape != self.resistances_K_W.shape or not np.isfinite(x).all():
            raise ValueError("Foster rises must match the branch count")
        return x + _finite(energy_J, "energy_J") / self.capacitances_J_K


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
