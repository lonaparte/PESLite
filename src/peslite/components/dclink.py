"""A converter DC link assembled from the common capacitor and source models."""

from __future__ import annotations

from typing import Any

from ..control.blocks import smoothstep
from ..solver.model import record_type
from .passive import Capacitor
from .source import DCCurrentSource, DCVoltageSource

__all__ = ["DCLink", "make_dclink"]


class DCLink(Capacitor):
    """Optional shunt capacitor supplied by a DC voltage or current source."""

    Output = record_type("DCLinkOutput", ("i", "u"))
    equation_fields = {"tripped": "tripped"}
    parameter_fields = {"r": "R", "c": "C"}
    structural_parameters = frozenset(("c",))
    output_stages = ()
    signal_fields = {"i": "i", "voltage": "u"}

    def __init__(self, voltage: float, capacitance: float | None = None,
                 resistance: float = 0.0,
                 source: DCVoltageSource | DCCurrentSource | None = None,
                 droop: float = 0.0) -> None:
        if source is not None and not isinstance(source, (DCVoltageSource, DCCurrentSource)):
            raise TypeError("source must be a DCVoltageSource, DCCurrentSource or None")
        voltage_source = isinstance(source, DCVoltageSource)
        if capacitance is None and not voltage_source:
            raise ValueError("a DC link without a capacitor requires a voltage source")
        if (capacitance is not None and voltage_source
                and (source.branch.R or 0.0) + resistance <= 0.0):
            raise ValueError(
                "a voltage source with a capacitor requires positive source resistance or ESR"
            )
        super().__init__(
            R=resistance, C=capacitance, u0=voltage,
            phases=1, breakable=False,
        )
        self.port2.grounded = True
        self.port = self.port1
        self.source = source
        self.k_dc = float(droop)
        self.u_ref = float(voltage)
        self.ramp = None
        self.current_profile = None if source is None or voltage_source else self
        self.has_source = source is not None
        self.tripped = False

    def __call__(self, t: float, voltage: float) -> float:
        """Evaluate the configured current supply at ``t`` and sensed DC voltage."""
        source = self.source
        gain = 1.0 if self.ramp is None else self.ramp(t)
        return gain * source.current + self.k_dc * (self.u_ref - voltage)

    def open_breaker(self) -> None:
        """Disconnect the supply while retaining the passive DC link."""
        self.tripped = True

    def equation_objects(self) -> dict[str, Any]:
        """Retain the optional current profile in generated backends."""
        return ({"current_profile": self.current_profile}
                if self.current_profile is not None else {})

    def connect(self, on: bool, t: float, ramp: float = 0.0) -> None:
        """Connect or disconnect the current supply, optionally through a smooth ramp."""
        if self.current_profile is None:
            return
        if not on:
            self.ramp = lambda _at: 0.0
        elif ramp <= 0.0:
            self.ramp = None
        else:
            end = t + ramp
            self.ramp = lambda at: 1.0 if at >= end else smoothstep((at - t) / ramp)

    def set_source_parameters(self, i: float, k: float, v: float, r: float) -> bool:
        """Apply source parameters and report whether its netlist coefficients changed."""
        source = self.source
        if isinstance(source, DCVoltageSource):
            return source.set_parameters(voltage=v, r=r, l=None)
        if source is not None:
            source.set_parameters(current=i)
            self.k_dc = float(k)
        return False

    def network(self, nb) -> None:
        """Declare the common capacitor and source primitives at one DC node."""
        from ..solver import netlist as nl

        super().network(nb)
        node, source = nb.node(self.port), self.source
        nb.port_output(self, "u", self.port, "voltage")
        if isinstance(source, DCVoltageSource):
            resistance = source.branch.R
            emf = nb.internal() if resistance or self.C is not None else node
            source.declare(nb, emf)
            if self.C is not None:
                connected = nb.internal() if resistance else node
                nb.add(nl.Switch(emf, connected, nb.value(self, "tripped", invert=True)))
                emf = connected
            if resistance:
                source.branch.declare_network(nb, emf, node)
        elif source is not None:
            source.declare(
                nb, node,
                nb.function(
                    lambda t: 0.0 if self.tripped
                    else self.current_profile(t, self.state.u_C),
                    states=("u_C",),
                ),
            )
        nb.supplies(self)


def make_dclink(cfg: Any) -> DCLink:
    """Build one DC link from the shared capacitor and DC source implementations."""
    capacitor = cfg.capacitor
    capacitance = None if capacitor is None else capacitor.c
    resistance = 0.0 if capacitor is None else capacitor.r_esr
    source = cfg.source
    if source.type == "current":
        supply = DCCurrentSource(source.i)
    elif source.type == "voltage":
        supply = DCVoltageSource(source.v, R=source.r)
    elif source.type == "none":
        supply = None
    else:
        raise ValueError(f"unknown DC source type {source.type!r}")
    return DCLink(
        cfg.vdc_ref, capacitance, resistance,
        source=supply, droop=source.k,
    )
