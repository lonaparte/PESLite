"""A distributed-parameter cable assembled as one custom continuous subsystem.

The cable is a symmetric ladder of ``sections`` cell-centred shunt G-C elements and
``sections + 1`` series R-L elements.  The two end R-L elements have half a cell's
length, so the configured per-kilometre values integrate to exactly the requested
physical length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from types import FunctionType
from typing import Any, ClassVar

from . import (Bag, ConfigError, Element, PowerPort, StoragePort, Terminal,
               register_element_type)

__all__ = ["Cable", "CableModel"]


class _CableInput(Bag):
    __slots__ = ("u1", "u2")


class _CableOutput(Bag):
    __slots__ = ("i1", "i2")


@lru_cache(maxsize=None)
def _state_record(sections: int) -> type[Bag]:
    names = tuple(f"iL{k}" for k in range(1, sections + 2)) + tuple(
        f"uC{k}" for k in range(1, sections + 1)
    )
    return type(f"_CableState{sections}", (Bag,), {"__slots__": names})


@lru_cache(maxsize=None)
def _rhs_function(sections: int) -> FunctionType:
    """Build a straight-line RHS once; no section loop or mode branch remains at run time."""
    lines = ["def rhs(self, t):", "    s = self.state", "    inp = self.inp", "    return ("]
    for k in range(sections + 1):
        left = "inp.u1" if k == 0 else f"s.uC{k}"
        right = "inp.u2" if k == sections else f"s.uC{k + 1}"
        lines.append(
            f"        ({left} - {right} - self.series_r[{k}] * s.iL{k + 1}) "
            f"/ self.series_l[{k}],"
        )
    for k in range(sections):
        lines.append(
            f"        (s.iL{k + 1} - s.iL{k + 2} - self.shunt_g * s.uC{k + 1}) "
            "/ self.shunt_c,"
        )
    lines.append("    )")
    namespace: dict[str, Any] = {}
    exec(compile("\n".join(lines) + "\n", "<peslite cable rhs>", "exec"), namespace)
    return namespace["rhs"]


class CableModel:
    """Two-terminal R-L/G-C cable; both terminal currents are positive into the cable."""

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
        initial = {name: 0j for name in self.state_names}
        initial.update({f"uC{k}": complex(voltage0) for k in range(1, sections + 1)})
        self.state = _state_record(sections)(**initial)
        self.inp = _CableInput(u1=complex(voltage0), u2=complex(voltage0))
        self.out = _CableOutput(i1=0j, i2=0j)
        self.storage = tuple(
            StoragePort(
                f"iL{k + 1}", inductance, 1.5,
                (("inp.u1", 1.0),) if k == 0 else
                (("inp.u2", -1.0),) if k == sections else (),
                "inductor",
            )
            for k, inductance in enumerate(self.series_l)
        ) + tuple(
            StoragePort(f"uC{k + 1}", self.shunt_c, 1.5, (), "capacitor")
            for k in range(sections)
        )
        self.ports = (
            PowerPort("inp.u1", "out.i1", 1.5, 1.0),
            PowerPort("inp.u2", "out.i2", 1.5, 1.0),
        )
        self.terminal1 = Terminal(self, "u1", "i1", 1)
        self.terminal2 = Terminal(self, "u2", "i2", 1)
        # Specialise the short ladder equation once, rather than looping over dynamic records on
        # every solver stage.  The generated function is shared by every cable with this size.
        self.rhs = _rhs_function(sections).__get__(self, type(self))

    def set_outputs(self, t: float) -> None:
        self.out.i1 = self.state.iL1
        self.out.i2 = -getattr(self.state, f"iL{self.sections + 1}")

    def rhs(self, t: float):  # replaced by the size-specialised method during construction
        raise RuntimeError("cable RHS was not initialised")

    def dissipated_power(self) -> float:
        state = self.state
        series = sum(r * abs(getattr(state, f"iL{k + 1}")) ** 2
                     for k, r in enumerate(self.series_r))
        shunt = self.shunt_g * sum(abs(getattr(state, f"uC{k + 1}")) ** 2
                                  for k in range(self.sections))
        return 1.5 * (series + shunt)

    def supplied_power(self) -> float:
        return 0.0


@register_element_type
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
        sections: int = 4
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

    def terminals(self) -> tuple[tuple[str, Terminal], ...]:
        return ((self.cfg.bus1, self.model.terminal1),
                (self.cfg.bus2, self.model.terminal2))

    def signals(self) -> dict[str, complex]:
        return {f"{self.name}.i1": self.model.out.i1,
                f"{self.name}.i2": self.model.out.i2}
