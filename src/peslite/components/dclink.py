"""DC-link storage and source elements."""

from __future__ import annotations

from typing import Any, ClassVar

from ..control.blocks import smoothstep
from ..solver.energy import Port, StoragePort
from ..solver.model import Bag, Empty, OutputMethod, _real
from .network import Element
from .source import DCSource

__all__ = [
    "DCCapacitor", "DCLink", "make_dclink",
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
                 source: DCSource | None = None,
                 current: tuple[float, float, float] | None = None) -> None:
        if capacitor is not None and not isinstance(capacitor, DCCapacitor):
            raise TypeError("capacitor must be a DCCapacitor or None")
        if source is not None and not isinstance(source, DCSource):
            raise TypeError("source must be a DCSource or None")
        if source is not None and current is not None:
            raise ValueError("a DC link cannot have both voltage and current supplies")
        if capacitor is None and not isinstance(source, DCSource):
            raise ValueError("a DC link without a capacitor requires a voltage source")
        if (capacitor is not None and isinstance(source, DCSource)
                and (source.branch.R or 0.0) + capacitor.R_esr <= 0):
            raise ValueError("a voltage source with a capacitor requires positive source resistance or ESR")
        self.capacitor = capacitor
        self.current_mode = current is not None
        if current is not None:
            self.i_nom, self.k_dc, self.u_ref = map(_real, current)
            self.ramp = None
        self.source = self if self.current_mode else source
        self.state = capacitor.state if capacitor is not None else Empty()
        self.state_names = ("u_C",) if capacitor is not None else ()
        self.has_source = source is not None or self.current_mode
        u0 = capacitor.state.u_C if capacitor is not None else source.voltage
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

    def __call__(self, t: float, voltage: float) -> float:
        current = self.i_nom if self.ramp is None else self.ramp(t) * self.i_nom
        if self.k_dc != 0.0:
            current += self.k_dc * (self.u_ref - voltage)
        return current

    def open_breaker(self) -> None:
        """Disconnect the source from the capacitor; a link without capacitor is unchanged."""
        self.tripped = True

    def equation_objects(self) -> dict[str, Any]:
        """Objects referenced by equations and therefore retained by generated backends."""
        return {"source": self.source}

    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect or disconnect the source; current sources honor the connection ramp."""
        if self.current_mode:
            if not on:
                self.ramp = lambda _at: 0.0
            elif ramp <= 0.0:
                self.ramp = None
            else:
                end = t + ramp
                self.ramp = lambda at: 1.0 if at >= end else smoothstep((at - t) / ramp)

    def set_source_parameters(self, i: float, k: float, v: float, r: float) -> bool:
        """Apply source parameters and report whether its network declaration changed."""
        source = self.source
        if source is self:
            self.i_nom, self.k_dc = float(i), float(k)
            return False
        elif isinstance(source, DCSource):
            return source.set_parameters(voltage=v, r=r, l=None)
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
        if source is self:
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
        elif isinstance(source, DCSource):
            resistance = source.branch.R
            emf = nb.internal() if resistance or capacitor is not None else node
            supply = source.declare(nb, emf)
            nb.linear_output(self, "i_src", ((-1.0, ("branch_current", supply)),))
            if capacitor is None:
                nb.output(self, "u_C", ("branch_voltage", supply))
            if capacitor is not None:
                connected = nb.internal() if resistance else node
                nb.add(nl.Switch(emf, connected, nb.value(self, "tripped", invert=True)))
                emf = connected
            if resistance:
                source.branch.declare_network(nb, emf, node)
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
        source, current = None, (p.i, p.k, cfg.vdc_ref)
    elif p.type == "voltage":
        source, current = DCSource(p.v, R=p.r), None
    elif p.type == "none":
        source = current = None
    else:
        raise ValueError(f"unknown DC source type {p.type!r}")
    return DCLink(capacitor, source, current)
