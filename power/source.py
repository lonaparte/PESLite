"""Ideal three-phase grid voltage source.

Angle deviation and magnitude are given as functions of time, e.g. from
:class:`peslite.assembly.events.SourceScenario`.
"""

from __future__ import annotations

import math
from typing import Callable, ClassVar, Optional

from ..phs.containers import Bag, Empty
from ..phs.protocols import PowerPort, StoragePort

__all__ = ["ThreePhaseSource"]


class _SourceOut(Bag):
    __slots__ = ("e_g", "phi", "theta")


class _SourceInp(Bag):
    __slots__ = ("i",)


class ThreePhaseSource:
    """Ideal source ``e_g(t) = E(t) exp(j (w0 t + phi(t)))``.

    ``w0`` in rad/s; ``phi(t)`` angle deviation (rad) and ``magnitude(t)`` peak voltage (V),
    each ``None`` for zero / constant ``e_peak``. Input ``i``: delivered current (A), for energy accounting.
    """

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    storage: ClassVar[tuple[StoragePort, ...]] = ()
    ports: ClassVar[tuple[PowerPort, ...]] = (PowerPort("out.e_g", "inp.i", 1.5, -1.0),)
    has_source: ClassVar[bool] = True

    def __init__(self, w0: float, e_peak: float, phi: Optional[Callable[[float], float]] = None,
                 magnitude: Optional[Callable[[float], float]] = None) -> None:
        self.w0, self.e_peak = w0, e_peak
        self.phi = phi
        self.magnitude = magnitude
        self.state = Empty()
        self.inp = _SourceInp(i=0j)
        self.out = _SourceOut()

    def set_outputs(self, t: float) -> None:
        phi = self.phi(t) if self.phi is not None else 0.0
        theta = self.w0 * t + phi
        out = self.out
        out.phi = phi
        out.theta = theta
        e = self.magnitude(t) if self.magnitude is not None else self.e_peak
        out.e_g = complex(e * math.cos(theta), e * math.sin(theta))

    def rhs(self, t: float):
        return ()

    def dissipated_power(self) -> float:
        return 0.0

    def supplied_power(self) -> float:
        return 1.5 * (self.out.e_g * self.inp.i.conjugate()).real
