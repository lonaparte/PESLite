"""Independent winding terminals coupled by a constant inductance matrix.

The circuit and energy equations reuse the netlist's coupled-inductor primitive
and the modal storage convention already used by compensated WPT links. No
compensation network is part of this component.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ...components.network import Element, NetworkComponent, _register_assembled_type
from ...solver import Bag, ConfigError, Empty, Port, StoragePort

__all__ = ["CoupledCoils", "CoupledCoilsModel", "WindingParams"]

GROUND = "ground"


@dataclass(frozen=True, kw_only=True)
class WindingParams:
    positive: str
    negative: str
    r: float = 0.0

    def __post_init__(self) -> None:
        if not self.positive or not self.negative or self.positive == self.negative:
            raise ConfigError("positive and negative must be distinct, nonempty terminals")
        if not math.isfinite(self.r) or self.r < 0.0:
            raise ConfigError(f"r must be finite and >= 0, got {self.r}")


class CoupledCoilsModel(NetworkComponent):
    """One scalar electrical branch per winding, with symmetric magnetic coupling."""

    outputs_need_inputs: ClassVar[bool] = False

    def __init__(self, names: tuple[str, ...], inductance: np.ndarray,
                 resistance: tuple[float, ...], terminals: tuple[tuple[bool, bool], ...]) -> None:
        n = len(names)
        modes, shapes = np.linalg.eigh(inductance)
        self.names = names
        self.resistance = resistance
        self.inductance = inductance
        self._shapes = tuple(tuple(float(x) for x in row) for row in shapes)
        self.state_names = tuple(f"i_mode_{k + 1}" for k in range(n))
        self.state = type("_CoilState", (Bag,), {"__slots__": self.state_names})(
            **dict.fromkeys(self.state_names, 0.0)
        )
        output_names = tuple(label for name in names for label in (f"i_{name}", f"u_{name}"))
        self.output = type("_CoilOutput", (Bag,), {"__slots__": output_names})(
            **dict.fromkeys(output_names, 0.0)
        )
        self.input = Empty()
        self.storage = tuple(StoragePort(state, float(mode), kind="inductor")
                             for state, mode in zip(self.state_names, modes))
        # The connected-netlist contract permits one grounded port per subsystem.
        # All grounded winding ends refer to the same electrical reference node.
        ground = Port(self, GROUND, grounded=True) if any(any(pair) for pair in terminals) else None
        ports = []
        for name, (positive_ground, negative_ground) in zip(names, terminals):
            ports.append((
                ground if positive_ground else Port(self, f"{name}.positive", standalone="voltage"),
                ground if negative_ground else Port(self, f"{name}.negative", standalone="voltage"),
            ))
        self.winding_ports = tuple(ports)
        self.ports = tuple(dict.fromkeys(port for pair in ports for port in pair))

    def network(self, nb) -> None:
        from ...solver import netlist as nl

        branches = []
        for (positive, negative), resistance in zip(self.winding_ports, self.resistance):
            high, low = nb.node(positive), nb.node(negative)
            if resistance:
                inner = nb.internal()
                nb.add(nl.Resistor(high, inner, resistance))
                high = inner
            branches.append((high, low))
        coupled = nb.add(nl.CoupledInductors(
            branches, self.inductance,
            nb.state(self, self.state_names, transform=self._shapes),
        ))
        for index, (name, (positive, negative)) in enumerate(zip(self.names, self.winding_ports)):
            nb.output(self, f"i_{name}", ("branch_current", coupled, index))
            terminal_voltage = []
            if not positive.grounded:
                terminal_voltage.append((1.0, ("node_voltage", nb.node(positive))))
            if not negative.grounded:
                terminal_voltage.append((-1.0, ("node_voltage", nb.node(negative))))
            nb.linear_output(self, f"u_{name}", terminal_voltage)
        nb.losses(self)


@_register_assembled_type
class CoupledCoils(Element):
    """Named two-terminal windings with an explicit SI inductance matrix."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        windings: dict[str, WindingParams] = field(metadata={"structural": True})
        inductance: list[list[float]] = field(metadata={"structural": True})
        type: str = "coupled_coils"

        def __post_init__(self) -> None:
            n = len(self.windings)
            if not n:
                raise ConfigError("windings must define at least one winding")
            if any(not name.isidentifier() for name in self.windings):
                raise ConfigError("winding names must be Python identifiers")
            try:
                matrix = np.asarray(self.inductance, dtype=float)
            except (TypeError, ValueError):
                raise ConfigError("inductance must be a numeric square matrix") from None
            if matrix.shape != (n, n) or not np.isfinite(matrix).all():
                raise ConfigError(f"inductance must be a finite {n} x {n} matrix")
            if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0):
                raise ConfigError("inductance must be symmetric")
            if np.linalg.eigvalsh(matrix).min() <= 0.0:
                raise ConfigError("inductance must be positive definite")
            object.__setattr__(self, "inductance", matrix.tolist())

        def connection_buses(self) -> dict[str, str]:
            return {
                f"windings.{name}.{side}": bus
                for name, winding in self.windings.items()
                for side in ("positive", "negative")
                if (bus := getattr(winding, side)) != GROUND
            }

    type = "coupled_coils"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self.name, self.cfg = name, cfg
        for label, winding in cfg.windings.items():
            for side in ("positive", "negative"):
                bus = getattr(winding, side)
                if bus == GROUND:
                    continue
                if bus not in buses:
                    raise ConfigError(f"elements.{name}.windings.{label}.{side}: unknown bus {bus!r}")
                if buses[bus].phases != 1:
                    raise ConfigError(
                        f"elements.{name}.windings.{label}.{side}: coupled coils require scalar buses"
                    )
        self.model = CoupledCoilsModel(
            tuple(cfg.windings), np.asarray(cfg.inductance, dtype=float),
            tuple(winding.r for winding in cfg.windings.values()),
            tuple((winding.positive == GROUND, winding.negative == GROUND)
                  for winding in cfg.windings.values()),
        )

    def subsystems(self) -> dict[str, CoupledCoilsModel]:
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
