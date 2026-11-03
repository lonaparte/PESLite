"""A distributed-parameter cable assembled as one custom continuous subsystem.

The cable is a symmetric ladder of ``sections`` cell-centred shunt G-C elements and
``sections + 1`` series R-L elements.  The two end R-L elements have half a cell's
length, so the configured per-kilometre values integrate to exactly the requested
physical length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar

from . import (Bag, ConfigError, Element, Empty, NetworkComponent, Port, StoragePort,
               register_component_type)

__all__ = ["Cable", "CableModel"]


class _CableOutput(Bag):
    __slots__ = ("i1", "i2")


@lru_cache(maxsize=None)
def _state_record(sections: int) -> type[Bag]:
    names = tuple(f"iL{k}" for k in range(1, sections + 2)) + tuple(
        f"uC{k}" for k in range(1, sections + 1)
    )
    return type(f"_CableState{sections}", (Bag,), {"__slots__": names})


class CableModel(NetworkComponent):
    """Two-port R-L/G-C cable; both port currents are positive into the cable."""

    outputs_need_inputs: ClassVar[bool] = False

    def __init__(self, *, length_km: float, r_per_km: float, l_per_km: float,
                 g_per_km: float, c_per_km: float, sections: int,
                 voltage0: complex) -> None:
        cell = length_km / sections
        end = 0.5 * cell
        lengths = (end, *(cell for _ in range(sections - 1)), end)
        self.sections = sections
        self.series_r = tuple(r_per_km * span for span in lengths)
        self.series_l = tuple(l_per_km * span for span in lengths)
        self.shunt_g = g_per_km * cell
        self.shunt_c = c_per_km * cell
        self.state_names = _state_record(sections).__slots__
        self.state_dimensions = dict.fromkeys(self.state_names, 2)
        initial = {name: 0j for name in self.state_names}
        initial.update({f"uC{k}": complex(voltage0) for k in range(1, sections + 1)})
        self.state = _state_record(sections)(**initial)
        self.input = Empty()
        self.output = _CableOutput(i1=0j, i2=0j)
        self.port1 = Port(
            self, "1", voltage=complex(voltage0), current=0j, scale=1.5,
            electrical_domain="ac", standalone="voltage"
        )
        self.port2 = Port(
            self, "2", voltage=complex(voltage0), current=0j, scale=1.5,
            electrical_domain="ac", standalone="voltage"
        )
        self.ports = (self.port1, self.port2)
        self.storage = tuple(
            StoragePort(
                f"iL{k + 1}", inductance, 1.5,
                ((self.port1, "voltage", 1.0),) if k == 0 else
                ((self.port2, "voltage", -1.0),) if k == sections else (),
                "inductor",
            )
            for k, inductance in enumerate(self.series_l)
        ) + tuple(
            StoragePort(f"uC{k + 1}", self.shunt_c, 1.5, kind="capacitor")
            for k in range(sections)
        )

    def network(self, nb):
        """Declare the cable's distributed R-L and G-C ladder."""
        from ...solver import netlist as nl

        nodes = [
            nb.node(self.port1),
            *(nb.internal(2, 1.5) for _ in range(self.sections)),
            nb.node(self.port2),
        ]
        nb.port_output(self, "i1", self.port1, "current")
        nb.port_output(self, "i2", self.port2, "current")
        for index, (resistance, inductance) in enumerate(
                zip(self.series_r, self.series_l)):
            start = nodes[index]
            if resistance:
                inner = nb.internal(2, 1.5)
                nb.add(nl.Resistor(start, inner, resistance))
                start = inner
            nb.add(nl.Inductor(
                start,
                nodes[index + 1],
                inductance,
                nb.state(self, f"iL{index + 1}"),
            ))
        for index in range(1, self.sections + 1):
            nb.add(nl.Capacitor(
                nodes[index], None, self.shunt_c, nb.state(self, f"uC{index}")
            ))
            if self.shunt_g:
                nb.add(nl.Resistor(nodes[index], None, 1.0 / self.shunt_g))
        nb.losses(self)


@register_component_type
class Cable(Element):
    """Registered long-cable element using SI per-kilometre parameters."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bus1: str
        bus2: str
        length_km: float
        r_per_km: float
        l_per_km: float
        c_per_km: float
        g_per_km: float = 0.0
        sections: int = field(default=4, metadata={"structural": True})
        type: str = "cable"

        def __post_init__(self) -> None:
            positive = {
                "length_km": self.length_km,
                "l_per_km": self.l_per_km,
                "c_per_km": self.c_per_km,
            }
            nonnegative = {"r_per_km": self.r_per_km, "g_per_km": self.g_per_km}
            for name, value in positive.items():
                if not math.isfinite(value) or value <= 0.0:
                    raise ConfigError(f"{name} must be finite and > 0, got {value}")
            for name, value in nonnegative.items():
                if not math.isfinite(value) or value < 0.0:
                    raise ConfigError(f"{name} must be finite and >= 0, got {value}")
            if self.sections < 1:
                raise ConfigError(f"sections must be >= 1, got {self.sections}")
            if self.bus1 == self.bus2:
                raise ConfigError("bus1 and bus2 must name different buses")

    type = "cable"
    domain_transparent = True

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        self.model = CableModel(
            length_km=cfg.length_km,
            r_per_km=cfg.r_per_km,
            l_per_km=cfg.l_per_km,
            g_per_km=cfg.g_per_km,
            c_per_km=cfg.c_per_km,
            sections=cfg.sections,
            voltage0=complex(p.base.v_phase_peak),
        )

    def subsystems(self) -> dict[str, CableModel]:
        return {self.name: self.model}

    @staticmethod
    def connections() -> dict:
        return {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return ((self.cfg.bus1, self.model.port1),
                (self.cfg.bus2, self.model.port2))

    def signals(self) -> dict[str, complex]:
        return {f"{self.name}.i1": self.model.output.i1,
                f"{self.name}.i2": self.model.output.i2}
