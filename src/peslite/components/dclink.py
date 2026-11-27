"""DC-link storage and source elements."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional

from ..control.blocks import smoothstep
from ..solver.energy import Port, StoragePort
from ..solver.model import Bag, Empty, OutputMethod, _real
from .network import Element

__all__ = [
    "DCCapacitor", "DCCurrentSource", "DCVoltageSource", "DCLink", "make_dclink",
]


class _DCState(Bag):
    __slots__ = ("u_C",)


class _DCOutput(Bag):
    __slots__ = ("u_dc", "u_C", "i_src")


class DCCapacitor:
    """Capacitor ``C`` (F) with series resistance ``R_esr`` (ohm); current positive into the capacitor."""

    def __init__(self, u0: float, C: float, R_esr: float = 0.0) -> None:
        self.C, self.R_esr = _real(C), _real(R_esr)
        self.state = _DCState(u_C=_real(u0))


def _zero(t: float) -> float:
    return 0.0


class DCCurrentSource:
    """Supply current ``ramp(t) i_nom + k_dc (u_ref - u_C)`` (A); may be negative.

    ``k_dc`` is in A/V and ``u_ref`` in V. :meth:`connect` controls its nominal-current ramp.
    """

    def __init__(self, i_nom: float, k_dc: float = 0.0, u_ref: float = 0.0) -> None:
        self.i_nom, self.k_dc, self.u_ref = _real(i_nom), _real(k_dc), _real(u_ref)
        self.ramp: Optional[Callable[[float], float]] = None

    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect at ``t``, optionally with a smooth ramp, or disconnect the nominal current."""
        if not on:
            self.ramp = _zero
        elif ramp <= 0.0:
            self.ramp = None
        else:
            end = t + ramp
            self.ramp = lambda at: 1.0 if at >= end else smoothstep((at - t) / ramp)

    def __call__(self, t: float, u_C: float) -> float:
        i = self.i_nom if self.ramp is None else self.ramp(t) * self.i_nom
        if self.k_dc != 0.0:
            i += self.k_dc * (self.u_ref - u_C)
        return i


class DCVoltageSource:
    """Constant dc voltage source ``u_dc`` (V) with series resistance ``R`` (ohm)."""

    def __init__(self, u_dc: float, R: float = 0.0) -> None:
        self.u_dc, self.R = _real(u_dc), _real(R)


class DCLink(Element):
    """DC link made of an optional capacitor and a current source, voltage source or no source.

    Without a capacitor a voltage source is required; a voltage source with a
    capacitor requires ``R + R_esr > 0``. Named state: ``u_C`` (V) if a capacitor is present.
    """

    outputs_need_inputs: ClassVar[bool] = False
    equation_fields = {"tripped": "tripped"}
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod("set_stored_voltage", inputs=(), outputs=("u_C",)),
        OutputMethod("set_outputs", inputs=(), outputs=("u_dc", "i_src")),
    )
    set_stored_voltage = Element.set_outputs

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
        self.has_source = source is not None
        u0 = capacitor.state.u_C if capacitor is not None else source.u_dc
        self.input = Empty()
        self.output = _DCOutput(u_dc=u0, u_C=u0, i_src=0.0)
        self.port = Port(
            self, "dc", voltage=u0, current=0.0,
            electrical_domain="dc", standalone="current",
        )
        self.ports = (self.port,)
        self.storage = (() if capacitor is None else (StoragePort(
            "u_C", capacitor.C, 1.0, ((self.port, "current", 1.0),)
        ),))
        self.tripped = False

    def open_breaker(self) -> None:
        """Disconnect the source from the capacitor; a link without capacitor is unchanged."""
        self.tripped = True

    def equation_objects(self) -> dict[str, Any]:
        """Objects referenced by equations and therefore retained by generated backends."""
        return {"source": self.source}

    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect or disconnect the source; current sources honor the connection ramp."""
        if isinstance(self.source, DCCurrentSource):
            self.source.connect(on, t, ramp)

    def retune_source(self, i: float, k: float, v: float, r: float) -> bool:
        """Retune the source and report whether its network declaration changed."""
        source = self.source
        if isinstance(source, DCCurrentSource):
            source.i_nom, source.k_dc = float(i), float(k)
            return False
        elif isinstance(source, DCVoltageSource):
            changed = source.R != r
            source.u_dc, source.R = float(v), float(r)
            return changed
        return False

    def network(self, nb):
        """Declare the DC-link capacitor, source and source breaker."""
        from ..solver import netlist as nl

        node, capacitor, source = nb.node(self.port), self.capacitor, self.source
        nb.port_output(self, "u_dc", self.port, "voltage")
        if capacitor is not None:
            inner = nb.internal() if capacitor.R_esr else node
            if capacitor.R_esr:
                nb.add(nl.Resistor(node, inner, capacitor.R_esr))
            capacitor_element = nb.add(nl.Capacitor(
                inner, None, capacitor.C, nb.state(self, "u_C")
            ))
            nb.output(self, "u_C", ("branch_voltage", capacitor_element))
        if isinstance(source, DCCurrentSource):
            supply = nb.add(nl.CurrentSource(
                None,
                node,
                nb.function(
                    lambda t: 0.0 if self.tripped
                    else self.source(t, self.capacitor.state.u_C),
                    states=("u_C",),
                ),
            ))
            nb.output(self, "i_src", ("branch_current", supply))
        elif isinstance(source, DCVoltageSource):
            emf = nb.internal() if source.R or capacitor is not None else node
            supply = nb.add(nl.VoltageSource(
                emf, None, nb.function(lambda _t: self.source.u_dc)
            ))
            nb.linear_output(self, "i_src", ((-1.0, ("branch_current", supply)),))
            if capacitor is None:
                nb.output(self, "u_C", ("branch_voltage", supply))
            if capacitor is not None:
                connected = nb.internal() if source.R else node
                nb.add(nl.Switch(emf, connected, nb.value(self, "tripped", invert=True)))
                emf = connected
            if source.R:
                nb.add(nl.Resistor(emf, node, source.R))
        else:
            nb.linear_output(self, "i_src", ())
        nb.losses(self)
        nb.supplies(self)


def make_dclink(cfg: Any) -> DCLink:
    """Build a DC link from its capacitor and optional DC source."""
    capacitor = None
    if cfg.capacitor is not None:
        capacitor = DCCapacitor(cfg.vdc_ref, cfg.capacitor.c, cfg.capacitor.r_esr)
    p = cfg.source
    if p.type == "current":
        source = DCCurrentSource(p.i, p.k, cfg.vdc_ref)
    elif p.type == "voltage":
        source = DCVoltageSource(p.v, p.r)
    elif p.type == "none":
        source = None
    else:
        raise ValueError(f"unknown DC source type {p.type!r}")
    return DCLink(capacitor, source)
