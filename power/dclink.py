"""DC-side blocks: capacitor, current and voltage sources, and the composed dc link (SI units)."""
from __future__ import annotations

import math
from typing import Callable, ClassVar, Optional

from ..phs.containers import Bag, Empty
from ..phs.protocols import PowerPort, StoragePort
from ..params import DCLinkParams

__all__ = ["DCLink", "DCCapacitor", "DCCurrentSource", "DCVoltageSource", "make_dclink"]


def _finite(value: float, name: str, minimum: float | None = None) -> float:
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return value


class _DCState(Bag):
    __slots__ = ("u_C",)


class _DCInp(Bag):
    __slots__ = ("i_dc",)


class _DCOut(Bag):
    __slots__ = ("u_dc", "u_C", "i_src")


class DCCapacitor:
    """Capacitor ``C`` (F) with series resistance ``R_esr`` (ohm); current positive into the capacitor."""

    def __init__(self, u0: float, C: float, R_esr: float = 0.0) -> None:
        self.C = _finite(C, "C", 0.0)
        if self.C == 0:
            raise ValueError("C must be positive")
        self.R_esr = _finite(R_esr, "R_esr", 0.0)
        self.state = _DCState(u_C=_finite(u0, "u0"))

    def terminal_voltage(self, i: float) -> float:
        return self.state.u_C + self.R_esr * i

    def derivative(self, i: float) -> float:
        return i / self.C

    def dissipated_power(self, i: float) -> float:
        return self.R_esr * i ** 2


class DCCurrentSource:
    """Supply current ``ramp(t) i_nom + k_dc (u_ref - u_C)`` (A); may be negative.

    ``k_dc`` in A/V, ``u_ref`` in V; ``ramp`` is a function of time, ``None`` for 1.
    """

    def __init__(self, i_nom: float, k_dc: float = 0.0, u_ref: float = 0.0,
                 ramp: Optional[Callable[[float], float]] = None) -> None:
        self.i_nom = _finite(i_nom, "i_nom")
        self.k_dc = _finite(k_dc, "k_dc")
        self.u_ref = _finite(u_ref, "u_ref")
        self.ramp = ramp

    def __call__(self, t: float, u_C: float) -> float:
        i = self.i_nom if self.ramp is None else self.ramp(t) * self.i_nom
        if self.k_dc != 0.0:
            i += self.k_dc * (self.u_ref - u_C)
        return i


class DCVoltageSource:
    """Constant dc voltage source ``u_dc`` (V) with series resistance ``R`` (ohm)."""

    def __init__(self, u_dc: float, R: float = 0.0) -> None:
        self.u_dc = _finite(u_dc, "u_dc", 0.0)
        self.R = _finite(R, "R", 0.0)

    def terminal_voltage(self, i: float) -> float:
        return self.u_dc - self.R * i

    def dissipated_power(self, i: float) -> float:
        return self.R * i ** 2


class DCLink:
    """DC link made of an optional capacitor and a current source, voltage source or no source.

    Without a capacitor a voltage source is required; a voltage source with a
    capacitor requires ``R + R_esr > 0``. Named state: ``u_C`` (V) if a capacitor is present.
    """

    ports: ClassVar[tuple[PowerPort, ...]] = (PowerPort("out.u_dc", "inp.i_dc", 1.0, -1.0),)
    outputs_need_inputs: ClassVar[bool] = True  # i_src depends on i_dc

    def __init__(self, capacitor: DCCapacitor | None = None,
                 source: DCCurrentSource | DCVoltageSource | None = None) -> None:
        if capacitor is not None and not isinstance(capacitor, DCCapacitor):
            raise TypeError("capacitor must be a DCCapacitor or None")
        if source is not None and not isinstance(source, (DCCurrentSource, DCVoltageSource)):
            raise TypeError("source must be a DCCurrentSource, DCVoltageSource or None")
        if capacitor is None and not isinstance(source, DCVoltageSource):
            raise ValueError("a DC link without a capacitor requires a voltage source")
        if capacitor is not None and isinstance(source, DCVoltageSource) and source.R + capacitor.R_esr <= 0:
            raise ValueError("a voltage source with a capacitor requires positive source resistance or ESR")
        self.capacitor, self.source = capacitor, source
        self.state = capacitor.state if capacitor is not None else Empty()
        self.state_names = ("u_C",) if capacitor is not None else ()
        self.storage = ((StoragePort("u_C", capacitor.C, 1.0, (("inp.i_dc", -1.0),)),)
                        if capacitor is not None else ())
        self.has_source = source is not None
        u0 = capacitor.state.u_C if capacitor is not None else source.u_dc
        self.inp = _DCInp(i_dc=0.0)
        self.out = _DCOut(u_dc=u0, u_C=u0, i_src=0.0)
        self.tripped = False

    def open_breaker(self) -> None:
        """Disconnect the source from the capacitor; a link without capacitor is unchanged."""
        self.tripped = True

    def set_outputs(self, t: float) -> None:
        cap, source, i_dc = self.capacitor, self.source, self.inp.i_dc
        if cap is None:
            self.out.u_dc = self.out.u_C = source.terminal_voltage(i_dc)
            self.out.i_src = i_dc
            return
        if self.tripped or source is None:
            i_src = 0.0
        elif isinstance(source, DCVoltageSource):
            # from u_C + R_esr (i_src - i_dc) = u_dc - R i_src
            i_src = (source.u_dc - cap.state.u_C + cap.R_esr * i_dc) / (source.R + cap.R_esr)
        else:
            i_src = source(t, cap.state.u_C)
        self.out.i_src = i_src
        self.out.u_C = cap.state.u_C
        self.out.u_dc = cap.terminal_voltage(i_src - i_dc)

    def rhs(self, t: float):
        if self.capacitor is None:
            return ()
        return (self.capacitor.derivative(self.out.i_src - self.inp.i_dc),)

    def dissipated_power(self) -> float:
        loss = (self.capacitor.dissipated_power(self.out.i_src - self.inp.i_dc)
                if self.capacitor is not None else 0.0)
        if isinstance(self.source, DCVoltageSource):
            i_src = self.inp.i_dc if self.capacitor is None else self.out.i_src
            loss += self.source.dissipated_power(i_src)
        return loss

    def supplied_power(self) -> float:
        if isinstance(self.source, DCVoltageSource):
            i_src = self.inp.i_dc if self.capacitor is None else self.out.i_src
            return self.source.u_dc * i_src
        return self.out.u_dc * self.out.i_src


def make_dclink(cfg: DCLinkParams,
                ramp: Optional[Callable[[float], float]] = None) -> DCLink:
    """Build a :class:`DCLink` from SI parameters; source type ``"current"``, ``"voltage"`` or ``"none"``."""
    capacitor = None
    if cfg.capacitor is not None:
        capacitor = DCCapacitor(cfg.vdc_ref, cfg.capacitor.c, cfg.capacitor.r_esr)
    p = cfg.source
    if p.type == "current":
        source = DCCurrentSource(p.i, p.k, cfg.vdc_ref, ramp)
    elif p.type == "voltage":
        source = DCVoltageSource(p.v, p.r)
    elif p.type == "none":
        source = None
    else:
        raise ValueError(f"unknown DC source type {p.type!r}")
    return DCLink(capacitor, source)
