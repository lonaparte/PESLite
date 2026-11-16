"""Compact, passive, SI thermal RC networks for the standard PESLite add-on.

One thermal state belongs to each capacitive node. Positive heat (W) enters a
node; conduction through R (K/W) and convection to fixed ambient remove heat.
These equations are evaluated by PESLite's existing Model and solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = ["ThermalLink", "LumpedThermalNetwork", "FosterImpedance"]


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
