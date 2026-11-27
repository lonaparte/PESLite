"""AC and DC voltage sources built on one physical source contract."""

from __future__ import annotations

import math
from typing import Any, Callable, ClassVar, Optional

from ..solver.energy import Port, StoragePort
from ..solver.model import Empty, record_type
from .network import Element, _phase_convention
from .passive import RLBranch

__all__ = ["VoltageSource", "ACSource", "DCSource"]


class VoltageSource(Element):
    """Common ideal voltage source and its existing series R-L branch."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    storage: ClassVar[tuple[StoragePort, ...]] = ()
    has_source: ClassVar[bool] = True
    source_signals: ClassVar[dict[str, str]] = {}

    def __init__(self, output: Any, *, R: float = 0.0, L: float | None = None,
                 phases: int, electrical_domain: str, breakable: bool) -> None:
        self.power_scale, zero, _ = _phase_convention(phases)
        self.phases = phases
        self.state = Empty()
        self.input = Empty()
        self.output = output
        self.emf_port = Port(
            self, "power", voltage=zero, current=zero,
            scale=self.power_scale, electrical_domain=electrical_domain,
        )
        self.ports = (self.emf_port,)
        self.branch = RLBranch(R=R, L=L, phases=phases, breakable=breakable)
        self.branch.port2.electrical_domain = electrical_domain
        self.port = self.branch.port2

    def set_parameters(self, *, voltage: float,
                       r: float, l: float | None) -> bool:
        """Apply source voltage and impedance through the shared parameter path."""
        self.voltage = float(voltage)
        return self.branch.set_parameters(r=r, l=l)

    def declare(self, nb, node):
        """Declare the ideal source at ``node`` and return its netlist primitive."""
        from ..solver import netlist as nl

        return nb.add(nl.VoltageSource(
            node, None, nb.value(self.output, "voltage")
        ))

    def network(self, nb) -> None:
        self.declare(nb, nb.node(self.emf_port))
        nb.supplies(self)

    def subsystems(self) -> dict[str, Any]:
        return {self.name: self, f"{self.name}.branch": self.branch}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        internal = f"{self.name}.source"
        return ((internal, self.emf_port),
                (internal, self.branch.port1),
                (self.bus, self.port))

    def output_bindings(self) -> dict[str, tuple[Any, str]]:
        return {f"{self.name}.i": (self.branch, "i")}

    def signals(self) -> dict[str, float | complex]:
        return {
            f"{self.name}.i": self.branch.output.i,
            **{
                f"{self.name}.{name}": getattr(self.output, field)
                for name, field in self.source_signals.items()
            },
        }


class ACSource(VoltageSource):
    """Single- or three-phase sinusoidal voltage source behind a series R-L branch."""

    Output = record_type("ACSourceOutput", ("voltage", "phi", "theta"))
    source_signals = {"angle": "phi"}

    def __init__(self, w0: float, e_peak: float,
                 phi: Optional[Callable[[float], float]] = None, *,
                 R: float = 0.0, L: float | None = None, phases: int = 3,
                 breakable: bool = True) -> None:
        _scale, zero, _ = _phase_convention(phases)
        self.w0, self.voltage = w0, 0.0
        self.phase = None
        self.waveform = ((lambda angle: complex(math.cos(angle), math.sin(angle)))
                         if phases == 3 else math.cos)
        super().__init__(
            self.Output(voltage=zero, phi=0.0, theta=0.0),
            R=R, L=L, phases=phases, electrical_domain="ac", breakable=breakable,
        )
        self.set_parameters(voltage=e_peak, r=R, l=L, phase=phi)

    def set_parameters(self, *, voltage: float, r: float,
                       l: float | None, phase: Optional[Callable[[float], float]]) -> bool:
        """Apply the AC magnitude, phase law and series impedance."""
        self.phase = phase
        return super().set_parameters(voltage=voltage, r=r, l=l)

    def set_outputs(self, t: float) -> None:
        phi = self.phase(t) if self.phase is not None else 0.0
        theta = self.w0 * t + phi
        self.output.phi = phi
        self.output.theta = theta
        self.output.voltage = self.voltage * self.waveform(theta)

    @classmethod
    def uses_standard_outputs(cls, method: Any) -> bool:
        return getattr(method, "__func__", None) is cls.set_outputs

    def cpp_output_stage(self, generator: Any, method: Any):
        return (generator.source_output_lines(self)
                if self.uses_standard_outputs(method) else None)


class DCSource(VoltageSource):
    """Constant DC voltage source behind a series resistance."""

    Output = record_type("DCSourceOutput", ("voltage",))

    def __init__(self, voltage: float, *, R: float = 0.0,
                 breakable: bool = False) -> None:
        super().__init__(
            self.Output(voltage=0.0), R=R, phases=1,
            electrical_domain="dc", breakable=breakable,
        )
        self.set_parameters(voltage=voltage, r=R, l=None)

    @property
    def voltage(self) -> float:
        return self.output.voltage

    @voltage.setter
    def voltage(self, value: float) -> None:
        self.output.voltage = float(value)
