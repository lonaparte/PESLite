"""Advanced electro-magneto-thermal exchange, shared cooling and imported ROMs.

These optional providers extend the standard thermal/loss contracts under
`peslite-full`; they use the existing Model states and integration methods.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence

import numpy as np

from ...components.network import Element, register_component_type
from ...solver import Bag, ConfigError
from .thermal_network import LumpedThermalNetwork, ThermalLink

__all__ = [
    "ThermalLaw", "SharedThermalModel", "SharedThermalNetwork",
    "ReducedThermalModel", "ReducedThermalElement", "load_thermal_rom",
    "foster_step_response",
]


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        answer = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite") from None
    if not math.isfinite(answer):
        raise ValueError(f"{label} must be finite")
    return answer


@lru_cache(maxsize=None)
def _fields(prefix: str, names: tuple[str, ...]) -> type[Bag]:
    return type(prefix, (Bag,), {"__slots__": names})


class ThermalLaw(Protocol):
    """Custom FEM/co-simulation law called from the existing Model RHS.

    Implementations must be pure for fixed (t, state, heat) during speculative
    integrator stages and may not send side effects to an external solver.
    External exchange belongs at committed scheduling boundaries.
    """

    def rhs(self, t: float, state: np.ndarray, heat_W: float) -> np.ndarray: ...
    def temperature_K(self, t: float, state: np.ndarray, heat_W: float) -> float: ...


class SharedThermalModel:
    """Couple independent electrical/magnetic heat inputs to ONE shared RC graph."""

    outputs_need_inputs: ClassVar[bool] = True
    observer: ClassVar[bool] = True

    def __init__(
        self, network: LumpedThermalNetwork,
        sources: Mapping[str, str], *,
        initial_K: float | None = None,
    ) -> None:
        if not isinstance(network, LumpedThermalNetwork):
            raise TypeError("shared thermal backend requires a physical lumped network")
        if not sources:
            raise ValueError("shared heat model needs at least one source")
        if (any(not isinstance(name, str) or not name.isidentifier()
                for name in sources)
                or any(node not in network.names for node in sources.values())):
            raise ValueError("heat sources need identifier names and existing thermal nodes")
        self.network = network
        self.source_nodes = dict(sources)
        self.state_names = tuple(f"T_{node}_K" for node in network.names)
        self.input_names = tuple(f"heat_{name}_W" for name in sources)
        self.output_names = (*self.state_names, "heat_total_W", "cooling_W")
        initial = (network.ambient_K if initial_K is None else _finite(initial_K, "initial_K"))
        if initial <= 0:
            raise ValueError("initial_K must be positive")
        self.state = _fields("_SharedThermalState", self.state_names)(
            **dict.fromkeys(self.state_names, initial)
        )
        self.input = _fields("_SharedThermalInput", self.input_names)(
            **dict.fromkeys(self.input_names, 0.0)
        )
        self.output = _fields("_SharedThermalOutput", self.output_names)()
        self.set_outputs(0.0)

    def _temperatures(self) -> np.ndarray:
        return np.fromiter(
            (getattr(self.state, name) for name in self.state_names),
            dtype=float, count=len(self.state_names),
        )

    def _heat(self) -> dict[str, float]:
        heat = {name: 0.0 for name in self.network.names}
        for source, node in self.source_nodes.items():
            value = _finite(getattr(self.input, f"heat_{source}_W"), f"{source}.heat_W")
            if value < 0:
                raise ValueError("passive thermal sources require nonnegative losses")
            heat[node] += value
        return heat

    def set_outputs(self, t: float) -> None:
        temperatures = self._temperatures()
        for name, temperature in zip(self.state_names, temperatures):
            setattr(self.output, name, float(temperature))
        self.output.heat_total_W = sum(self._heat().values())
        self.output.cooling_W = self.network.cooling_power_W(temperatures)

    def rhs(self, t: float) -> tuple[float, ...]:
        return tuple(self.network.rates(self._temperatures(), self._heat()))


@register_component_type
class SharedThermalNetwork(Element):
    """Configured shared heatsink for electrical and magnetic output signals.

    ``sources`` maps each unique heat-source label to {path, node}.
    ``links`` contains {node, other, resistance_K_W}; other=null is ambient.
    No electrical ports or duplicate magnetic/circuit model are constructed.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        capacities_J_K: dict[str, float] = field(metadata={"structural": True})
        links: list[dict] = field(metadata={"structural": True})
        sources: dict[str, dict[str, str]] = field(metadata={"structural": True})
        ambient_K: float = 298.15
        initial_K: float = 298.15
        type: str = "shared_thermal_network"

        def __post_init__(self):
            if not isinstance(self.sources, Mapping) or not self.sources:
                raise ConfigError("shared thermal network requires heat sources")
            for name, source in self.sources.items():
                if (not isinstance(name, str) or not name.isidentifier()
                        or not isinstance(source, Mapping)
                        or set(source) != {"path", "node"}
                        or not all(isinstance(v, str) and v for v in source.values())):
                    raise ConfigError("each shared heat source needs a named path and node")
            try:
                links = [ThermalLink(**item) for item in self.links]
                LumpedThermalNetwork(self.capacities_J_K, links,
                                     ambient_K=self.ambient_K)
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"invalid shared thermal network: {exc}") from None

    type = "shared_thermal_network"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        network = LumpedThermalNetwork(
            cfg.capacities_J_K,
            [ThermalLink(**item) for item in cfg.links],
            ambient_K=cfg.ambient_K,
        )
        self.model = SharedThermalModel(
            network, {name: item["node"] for name, item in cfg.sources.items()},
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


class ReducedThermalModel:
    """Stable reduced-order thermal state, imported from an external FEM/ROM.

    A: 1/s, B: K/J, C: dimensionless, D: K/W.
    xdot = A*x + B*P; T_j = T_ambient + C*x + D*P.
    B/C/D are user-supplied reduced coefficients, never inferred geometry.
    """

    outputs_need_inputs: ClassVar[bool] = True
    observer: ClassVar[bool] = True

    def __init__(self, A_per_s: Any, B_K_per_J: Any, C: Any, D_K_W: float,
                 *, ambient_K: float = 298.15, initial: Sequence[float] | None = None):
        try:
            A = np.asarray(A_per_s, dtype=float)
            B = np.asarray(B_K_per_J, dtype=float)
            C = np.asarray(C, dtype=float)
        except (ValueError, TypeError):
            raise ValueError("thermal ROM matrices must be numeric") from None
        if (A.ndim != 2 or A.shape[0] < 1 or A.shape[0] != A.shape[1]
                or B.shape != (A.shape[0],) or C.shape != (A.shape[0],)
                or not all(np.isfinite(arr).all() for arr in (A, B, C))):
            raise ValueError("thermal ROM must have finite A(N,N), B(N), C(N)")
        if np.max(np.real(np.linalg.eigvals(A))) >= 0.0:
            raise ValueError("thermal ROM A must be strictly stable (negative real eigenvalues)")
        if np.any(B < 0) or np.any(C < 0):
            raise ValueError("thermal ROM thermal gain coefficients must be nonnegative")
        D = _finite(D_K_W, "D_K_W")
        if D < 0:
            raise ValueError("D_K_W must be nonnegative")
        ambient = _finite(ambient_K, "ambient_K")
        if ambient <= 0:
            raise ValueError("ambient_K must be positive")
        initial_values = np.zeros(len(B)) if initial is None else np.asarray(initial, dtype=float)
        if initial_values.shape != B.shape or not np.isfinite(initial_values).all():
            raise ValueError("ROM initial state must be a finite vector of length N")
        self.A_per_s, self.B_K_per_J, self.C, self.D_K_W = A, B, C, D
        self.ambient_K = ambient
        self.state_names = tuple(f"rom_{k}_K" for k in range(len(B)))
        self.state = _fields("_ReducedThermalState", self.state_names)(
            **dict(zip(self.state_names, initial_values))
        )
        self.input = _fields("_ReducedThermalInput", ("heat_W",))(heat_W=0.0)
        self.output = _fields("_ReducedThermalOutput", ("junction_K", "heat_W"))()
        self.set_outputs(0.0)

    def _vector(self) -> np.ndarray:
        return np.fromiter((getattr(self.state, k) for k in self.state_names), dtype=float)

    def set_outputs(self, t: float) -> None:
        heat = _finite(self.input.heat_W, "heat_W")
        if heat < 0:
            raise ValueError("thermal ROM heat must be nonnegative")
        self.output.heat_W = heat
        self.output.junction_K = (
            self.ambient_K + float(self.C @ self._vector()) + self.D_K_W * heat
        )

    def rhs(self, t: float) -> tuple[float, ...]:
        p = _finite(self.input.heat_W, "heat_W")
        if p < 0:
            raise ValueError("thermal ROM heat must be nonnegative")
        return tuple(float(x) for x in self.A_per_s @ self._vector() + self.B_K_per_J * p)


@register_component_type
class ReducedThermalElement(Element):
    """Use one imported reduced-order thermal backend in the original circuit."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        heat_source: str = field(metadata={"runtime": False})
        A_per_s: list[list[float]] = field(metadata={"structural": True})
        B_K_per_J: list[float] = field(metadata={"structural": True})
        C: list[float] = field(metadata={"structural": True})
        D_K_W: float = 0.0
        ambient_K: float = 298.15
        type: str = "thermal_rom"

        def __post_init__(self):
            if not isinstance(self.heat_source, str) or not self.heat_source:
                raise ConfigError("heat_source must be an existing physical output path")
            try:
                ReducedThermalModel(self.A_per_s, self.B_K_per_J, self.C,
                                    self.D_K_W, ambient_K=self.ambient_K)
            except ValueError as exc:
                raise ConfigError(str(exc)) from None

    type = "thermal_rom"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any):
        self.name, self.cfg = name, cfg
        self.model = ReducedThermalModel(
            cfg.A_per_s, cfg.B_K_per_J, cfg.C, cfg.D_K_W, ambient_K=cfg.ambient_K
        )

    def subsystems(self) -> dict:
        return {self.name: self.model}

    def connections(self) -> dict:
        return {(self.model, "heat_W"): self.cfg.heat_source}

    @staticmethod
    def port_bindings() -> tuple:
        return ()

    def signals(self) -> dict[str, float]:
        return {f"{self.name}.junction_K": self.model.output.junction_K,
                f"{self.name}.heat_W": self.model.output.heat_W}


def load_thermal_rom(file: str | Path) -> ReducedThermalModel:
    """Read a declared-SI JSON ROM exported by a FEM or identification backend.

    This does not pretend that raw field data is a coupled PDE solver: the file
    must already contain a validated reduced-order state-space model.
    """
    with Path(file).open(encoding="utf-8") as stream:
        data = json.load(stream)
    required = {"format", "units", "A_per_s", "B_K_per_J", "C", "D_K_W", "ambient_K"}
    if set(data) != required or data["format"] != "peslite-thermal-rom-v1" or (
        data["units"] != "SI"
    ):
        raise ValueError("thermal ROM JSON must declare format, SI units and exact matrices")
    return ReducedThermalModel(
        data["A_per_s"], data["B_K_per_J"], data["C"], data["D_K_W"],
        ambient_K=data["ambient_K"],
    )


def foster_step_response(
    time_s: Sequence[float], heat_W: float,
    resistances_K_W: Sequence[float], capacitances_J_K: Sequence[float],
    *, ambient_K: float = 298.15,
) -> np.ndarray:
    """Exact analytical zero-initial-state step response of a Foster impedance."""
    from .thermal_network import FosterImpedance

    network = FosterImpedance(resistances_K_W, capacitances_J_K,
                               ambient_K=ambient_K)
    times = np.asarray(time_s, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("time_s must be finite nonnegative observations")
    heat = _finite(heat_W, "heat_W")
    if heat < 0:
        raise ValueError("step heat must be nonnegative")
    z = network.resistances_K_W[None, :] * heat * (
        1 - np.exp(-times[:, None] / (
            network.resistances_K_W * network.capacitances_J_K
        ))
    )
    return ambient_K + np.sum(z, axis=1)
