"""A reusable two-terminal compensation capacitor with optional series resistance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from ...components.network import register_component_type
from ...solver import ConfigError

__all__ = ["Capacitor"]


@register_component_type
@dataclass(frozen=True, kw_only=True)
class Capacitor:
    bus1: str
    bus2: str
    c: float
    r: float = 0.0
    type: str = "capacitor"
    domain_transparent: ClassVar[bool] = True

    _quantities = {"c": "capacitance", "r": "resistance"}

    def __post_init__(self) -> None:
        if self.bus1 == self.bus2:
            raise ConfigError("bus1 and bus2 must be distinct")
        if not math.isfinite(self.c) or self.c <= 0.0:
            raise ConfigError(f"c must be finite and > 0, got {self.c}")
        if not math.isfinite(self.r) or self.r < 0.0:
            raise ConfigError(f"r must be finite and >= 0, got {self.r}")

    def network(self, n) -> None:
        from ...solver import netlist as nl

        positive = n.bus1
        if self.r:
            middle = n.internal()
            n.add(nl.Resistor(positive, middle, self.r))
            positive = middle
        capacitor = n.add(nl.Capacitor(positive, n.bus2, self.c, n.state("u")))
        n.output("u", ("branch_voltage", capacitor))
        n.port_output("i", "bus1", "current")
        n.losses()
