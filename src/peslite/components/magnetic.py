"""Magnetic windings represented by the ordinary PESLite component and netlist contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..solver import Bag, ConfigError, Empty, Port, StoragePort
from .network import Element, NetworkComponent, _register_assembled_type

__all__ = ["MagComponent", "WindingParams"]


GROUND = "ground"


@dataclass(frozen=True, kw_only=True)
class WindingParams:
    """The electrical terminals and copper resistance of one named winding."""

    positive: str
    negative: str
    r: float = 0.0

    def __post_init__(self) -> None:
        if not self.positive or not self.negative or self.positive == self.negative:
            raise ConfigError("positive and negative must be distinct, nonempty terminals")
        if not math.isfinite(self.r) or self.r < 0.0:
            raise ConfigError(f"r must be finite and >= 0, got {self.r}")


def _validate_windings(windings: dict[str, WindingParams]) -> tuple[str, ...]:
    names = tuple(windings)
    if not names:
        raise ConfigError("windings must define at least one winding")
    if any(not name.isidentifier() for name in names):
        raise ConfigError("winding names must be Python identifiers")
    return names


def _connection_buses(windings: dict[str, WindingParams]) -> dict[str, str]:
    return {
        f"windings.{name}.{side}": bus
        for name, winding in windings.items()
        for side in ("positive", "negative")
        if (bus := getattr(winding, side)) != GROUND
    }


def _inductance_matrix(inductance: Any, count: int) -> list[list[float]]:
    try:
        matrix = np.asarray(inductance, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError("inductance must be a numeric square matrix") from None
    if matrix.shape != (count, count) or not np.isfinite(matrix).all():
        raise ConfigError(f"inductance must be a finite {count} x {count} matrix")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0):
        raise ConfigError("inductance must be symmetric")
    if np.linalg.eigvalsh(matrix).min() <= 0.0:
        raise ConfigError("inductance must be positive definite")
    return matrix.tolist()


@dataclass(frozen=True, kw_only=True)
class _MatrixParams:
    inductance: list[list[float]] = field(metadata={"structural": True})
    type: str = "matrix"

    def __post_init__(self) -> None:
        if self.type != "matrix":
            raise ConfigError(f"unknown magnetic model type {self.type!r}; known: ['matrix']")


class MagComponent(NetworkComponent):
    """Named electrical windings coupled by one constant reciprocal inductance matrix."""

    outputs_need_inputs: ClassVar[bool] = False

    def __init__(self, names: tuple[str, ...], inductance: Any,
                 resistance: tuple[float, ...],
                 terminals: tuple[tuple[bool, bool], ...]) -> None:
        matrix = np.asarray(_inductance_matrix(inductance, len(names)), dtype=float)
        modes, shapes = np.linalg.eigh(matrix)
        self.names = names
        self.resistance = resistance
        self.inductance = matrix
        self._shapes = tuple(tuple(float(value) for value in row) for row in shapes)
        self.state_names = tuple(f"i_mode_{index + 1}" for index in range(len(names)))
        self.state = type("_MagneticState", (Bag,), {"__slots__": self.state_names})(
            **dict.fromkeys(self.state_names, 0.0)
        )
        output_names = tuple(label for name in names for label in (f"i_{name}", f"u_{name}"))
        self.output = type("_MagneticOutput", (Bag,), {"__slots__": output_names})(
            **dict.fromkeys(output_names, 0.0)
        )
        self.input = Empty()
        self.storage = tuple(StoragePort(name, float(mode), kind="inductor")
                             for name, mode in zip(self.state_names, modes))

        ground = Port(self, GROUND, grounded=True) if any(any(pair) for pair in terminals) else None
        ports = []
        for name, (positive_ground, negative_ground) in zip(names, terminals):
            ports.append((
                ground if positive_ground else Port(self, f"{name}.positive", standalone="voltage"),
                ground if negative_ground else Port(self, f"{name}.negative", standalone="voltage"),
            ))
        self.winding_ports = tuple(ports)
        self.ports = tuple(dict.fromkeys(port for pair in ports for port in pair))

    def network(self, nb: Any) -> None:
        from ..solver import netlist as nl

        branches = []
        for (positive, negative), resistance in zip(self.winding_ports, self.resistance):
            high, low = nb.node(positive), nb.node(negative)
            if resistance:
                inner = nb.internal()
                nb.add(nl.Resistor(high, inner, resistance))
                high = inner
            branches.append((high, low))
        coupled = nb.add(nl.CoupledInductors(
            branches,
            self.inductance,
            nb.state(self, self.state_names, transform=self._shapes),
        ))
        for index, (name, (positive, negative)) in enumerate(zip(self.names,
                                                                  self.winding_ports)):
            nb.output(self, f"i_{name}", ("branch_current", coupled, index))
            terminal_voltage = []
            if not positive.grounded:
                terminal_voltage.append((1.0, ("node_voltage", nb.node(positive))))
            if not negative.grounded:
                terminal_voltage.append((-1.0, ("node_voltage", nb.node(negative))))
            nb.linear_output(self, f"u_{name}", terminal_voltage)
        nb.losses(self)


@_register_assembled_type
class _MagneticElement(Element):
    """Full configuration entry for the ordinary magnetic component."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        windings: dict[str, WindingParams] = field(metadata={"structural": True})
        model: _MatrixParams = field(metadata={"structural": True})
        type: str = "magnetic_component"

        def __post_init__(self) -> None:
            names = _validate_windings(self.windings)
            normalized = _inductance_matrix(self.model.inductance, len(names))
            object.__setattr__(self, "model", _MatrixParams(inductance=normalized))

        def connection_buses(self) -> dict[str, str]:
            return _connection_buses(self.windings)

    type = "magnetic_component"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self._assemble(name, cfg, cfg.windings, cfg.model.inductance, buses)

    def _assemble(self, name: str, cfg: Any, windings: dict[str, WindingParams],
                  inductance: Any, buses: dict) -> None:
        self.name, self.cfg = name, cfg
        names = _validate_windings(windings)
        for label, winding in windings.items():
            for side in ("positive", "negative"):
                bus = getattr(winding, side)
                if bus == GROUND:
                    continue
                if bus not in buses:
                    raise ConfigError(f"elements.{name}.windings.{label}.{side}: unknown bus {bus!r}")
                if buses[bus].phases != 1:
                    raise ConfigError(
                        f"elements.{name}.windings.{label}.{side}: magnetic windings require scalar buses"
                    )
        self.model = MagComponent(
            names,
            inductance,
            tuple(winding.r for winding in windings.values()),
            tuple((winding.positive == GROUND, winding.negative == GROUND)
                  for winding in windings.values()),
        )

    def subsystems(self) -> dict[str, MagComponent]:
        return {self.name: self.model}

    @staticmethod
    def connections() -> dict:
        return {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return tuple(
            (bus, port)
            for winding, pair in zip(self.cfg.windings.values(), self.model.winding_ports)
            for bus, port in zip((winding.positive, winding.negative), pair)
            if bus != GROUND
        )

    def signals(self) -> dict[str, float]:
        return {f"{self.name}.{field}": getattr(self.model.output, field)
                for field in type(self.model.output).__slots__}
