"""Coupled windings represented by the ordinary PESLite netlist contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..solver import Bag, ConfigError, Empty, Port, StoragePort
from .network import Element, NetworkComponent

__all__ = ["MagComponent", "MagneticParams", "WindingParams"]


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


def _validate_winding_names(winding_names: Any) -> tuple[str, ...]:
    names = tuple(winding_names)
    if not names:
        raise ConfigError("winding_names must contain at least one name")
    if any(not isinstance(name, str) or not name.isidentifier() for name in names):
        raise ConfigError("winding_names must contain identifiers")
    if len(set(names)) != len(names):
        raise ConfigError("winding_names must contain unique names")
    return names


def _validate_windings(windings: dict[str, WindingParams]) -> tuple[str, ...]:
    return _validate_winding_names(windings)


def _winding_buses(windings: dict[str, WindingParams]) -> dict[str, str]:
    return {
        f"windings.{name}.{side}": bus
        for name, winding in windings.items()
        for side in ("positive", "negative")
        if (bus := getattr(winding, side)) != GROUND
    }


@dataclass(frozen=True, kw_only=True)
class MagneticParams:
    """Common winding configuration for magnetic element adapters."""

    windings: dict[str, WindingParams] = field(metadata={"structural": True})

    def __post_init__(self) -> None:
        _validate_windings(self.windings)

    def connection_buses(self) -> dict[str, str]:
        return _winding_buses(self.windings)


def _inductance_decomposition(
    inductance: Any, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        matrix = np.asarray(inductance, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError("inductance must be a numeric square matrix") from None
    if matrix.shape != (count, count) or not np.isfinite(matrix).all():
        raise ConfigError(f"inductance must be a finite {count} x {count} matrix")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0):
        raise ConfigError("inductance must be symmetric")
    modes, shapes = np.linalg.eigh(matrix)
    if modes.min() <= 0.0:
        raise ConfigError("inductance must be positive definite")
    matrix.setflags(write=False)
    return matrix, modes, shapes


class MagComponent(NetworkComponent):
    """PESLite component for an electrical self- and mutual-inductance matrix."""

    outputs_need_inputs: ClassVar[bool] = False

    def __init__(self, winding_names: tuple[str, ...], inductance: Any,
                 resistance: tuple[float, ...],
                 terminals: tuple[tuple[bool, bool], ...]) -> None:
        names = _validate_winding_names(winding_names)
        if len(resistance) != len(names) or len(terminals) != len(names):
            raise ConfigError("every winding must define resistance and terminals")
        matrix, modes, shapes = _inductance_decomposition(inductance, len(names))

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

    def magnetic_signals(self) -> dict[str, float]:
        """Return the public winding currents and terminal voltages."""
        return {
            field: getattr(self.output, field)
            for field in type(self.output).__slots__
        }


class _MagneticElement(Element):
    """Shared winding validation and assembly for magnetic component implementations."""

    @staticmethod
    def _winding_data(
        name: str,
        windings: dict[str, WindingParams],
        winding_names: tuple[str, ...],
        buses: dict,
    ) -> tuple[tuple[str, ...], tuple[float, ...], tuple[tuple[bool, bool], ...]]:
        names = _validate_windings(windings)
        if names != tuple(winding_names):
            raise ConfigError(
                f"elements.{name}: winding order {names!r} does not match "
                f"model order {tuple(winding_names)!r}"
            )
        for label, winding in windings.items():
            for side in ("positive", "negative"):
                bus = getattr(winding, side)
                if bus == GROUND:
                    continue
                if bus not in buses:
                    raise ConfigError(
                        f"elements.{name}.windings.{label}.{side}: unknown bus {bus!r}"
                    )
                if buses[bus].phases != 1:
                    raise ConfigError(
                        f"elements.{name}.windings.{label}.{side}: "
                        "magnetic windings require scalar buses"
                    )
        return (
            names,
            tuple(winding.r for winding in windings.values()),
            tuple((winding.positive == GROUND, winding.negative == GROUND)
                  for winding in windings.values()),
        )

    def _attach(self, name: str, cfg: Any, model: Any) -> None:
        self.name, self.cfg, self.model = name, cfg, model

    def _assemble(self, name: str, cfg: Any, windings: dict[str, WindingParams],
                  winding_names: tuple[str, ...], inductance: Any, buses: dict) -> None:
        names, resistance, terminals = self._winding_data(
            name, windings, winding_names, buses
        )
        self._attach(
            name, cfg, MagComponent(names, inductance, resistance, terminals)
        )

    def subsystems(self) -> dict[str, Any]:
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
        return {
            f"{self.name}.{field}": value
            for field, value in self.model.magnetic_signals().items()
        }
