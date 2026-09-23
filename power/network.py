"""AC network blocks: series R-L branches and nodes with a damped shunt capacitor (SI units)."""

from __future__ import annotations

from typing import ClassVar

from ..phs.containers import Bag
from ..phs.protocols import PowerPort, StoragePort

__all__ = ["RLBranch", "RCNode"]


class _BranchState(Bag):
    __slots__ = ("i",)

class _BranchInp(Bag):
    __slots__ = ("u_from", "u_to")

class _BranchOut(Bag):
    __slots__ = ("i",)

class RLBranch:
    """Series R-L branch (H, ohm); current positive from ``u_from`` to ``u_to``; ``open_breaker`` zeroes it."""

    state_names: ClassVar[tuple[str, ...]] = ("i",)
    outputs_need_inputs: ClassVar[bool] = False
    ports: ClassVar[tuple[PowerPort, ...]] = (PowerPort("inp.u_from", "out.i", 1.5, 1.0),
                                              PowerPort("inp.u_to", "out.i", 1.5, -1.0))

    def __init__(self, L: float, R: float, i0: complex = 0j) -> None:
        self.L, self.R = L, R
        self.storage = (StoragePort("i", L, 1.5, (("inp.u_from", 1.0), ("inp.u_to", -1.0)), "inductor"),)
        self.state = _BranchState(i=complex(i0))
        self.inp = _BranchInp(u_from=0j, u_to=0j)
        self.out = _BranchOut(i=complex(i0))
        self.breaker_open = False

    def open_breaker(self) -> None:
        self.breaker_open = True
        self.state.i = 0j

    def set_outputs(self, t: float) -> None:
        self.out.i = self.state.i

    def rhs(self, t: float):
        if self.breaker_open:
            return (0j,)
        return ((self.inp.u_from - self.inp.u_to - self.R * self.state.i) / self.L,)

    def dissipated_power(self) -> float:
        return 1.5 * self.R * abs(self.state.i) ** 2

    def supplied_power(self) -> float:
        return 0.0

class _NodeState(Bag):
    __slots__ = ("u_C",)

class _NodeInp(Bag):
    __slots__ = ("i_in",)

class _NodeOut(Bag):
    __slots__ = ("u",)

class RCNode:
    """Node with a shunt capacitor ``C`` (F) in series with ``R_d`` (ohm) to ground.

    Input ``i_in``: sum of currents into the node (fan-in connection). Output ``u = u_C + R_d i_in``.
    """

    state_names: ClassVar[tuple[str, ...]] = ("u_C",)
    outputs_need_inputs: ClassVar[bool] = True
    ports: ClassVar[tuple[PowerPort, ...]] = (PowerPort("out.u", "inp.i_in", 1.5),)

    def __init__(self, C: float, R_d: float, u0: complex = 0j) -> None:
        self.C, self.R_d = C, R_d
        self.storage = (StoragePort("u_C", C, 1.5, (("inp.i_in", 1.0),)),)
        self.state = _NodeState(u_C=complex(u0))
        self.inp = _NodeInp(i_in=0j)
        self.out = _NodeOut(u=complex(u0))

    def set_outputs(self, t: float) -> None:
        self.out.u = self.state.u_C + self.R_d * self.inp.i_in

    def rhs(self, t: float):
        return (self.inp.i_in / self.C,)

    def dissipated_power(self) -> float:
        return 1.5 * self.R_d * abs(self.inp.i_in) ** 2

    def supplied_power(self) -> float:
        return 0.0
