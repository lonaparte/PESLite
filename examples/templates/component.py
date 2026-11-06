# PESLite 0.5.1
"""Series R-L component template; resistance is in ohm and inductance is in H."""

from dataclasses import dataclass
from typing import ClassVar

from peslite.addons.components import register_component_type
from peslite.solver import netlist as nl


@register_component_type
@dataclass(frozen=True, kw_only=True)
class TemplateComponent:
    bus: str
    r: float = 1.0
    l: float = 2e-3
    type: str = "template_component"
    domain_transparent: ClassVar[bool] = True

    def network(self, n):
        middle = n.internal()
        n.add(
            nl.Resistor(n.bus, middle, self.r),
            nl.Inductor(middle, n.ground, self.l, n.state("i")),
        )
        n.port_output("i", "bus", "current")
        n.losses()
