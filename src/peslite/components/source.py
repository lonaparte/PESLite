"""Ideal AC and scalar DC sources connected through the component network."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional

from ..solver.energy import Port, StoragePort
from ..solver.model import Bag, ConfigError, Empty
from .network import Element, NetworkComponent, _phase_convention, register_component_type

__all__ = ["ACVoltageSource", "VoltageSource"]


class _SourceOutput(Bag):
    __slots__ = ("e_g", "phi", "theta")


class ACVoltageSource(NetworkComponent):
    """Ideal single- or three-phase AC voltage source.

    Three phases use ``E exp(j (w0 t + phi(t)))``; one phase uses its real cosine.
    ``w0`` is in rad/s, ``e_peak`` in V and ``phi(t)`` is the relative angle in rad.
    """

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    storage: ClassVar[tuple[StoragePort, ...]] = ()
    has_source: ClassVar[bool] = True

    def __init__(self, w0: float, e_peak: float,
                 phi: Optional[Callable[[float], float]] = None, *, phases: int = 3) -> None:
        self.power_scale, zero, _ = _phase_convention(phases)
        self.w0, self.e_peak, self.phases = w0, e_peak, phases
        self.phi = phi
        self.state = Empty()
        self.input = Empty()
        self.output = _SourceOutput()
        self.output.e_g = zero
        self.port = Port(
            self, "power", voltage=zero, current=zero,
            scale=self.power_scale, electrical_domain="ac",
        )
        self.ports = (self.port,)
        self._output_equation = (self._set_three_phase_outputs if phases == 3
                                 else self._set_single_phase_outputs)
        if type(self).set_outputs is ACVoltageSource.set_outputs:
            self.set_outputs = self._output_equation

    def set_outputs(self, t: float) -> None:
        """Evaluate the phase representation selected when the source was constructed."""
        self._output_equation(t)

    def _set_single_phase_outputs(self, t: float) -> None:
        phi = self.phi(t) if self.phi is not None else 0.0
        theta = self.w0 * t + phi
        out = self.output
        out.phi = phi
        out.theta = theta
        out.e_g = self.e_peak * math.cos(theta)

    def _set_three_phase_outputs(self, t: float) -> None:
        phi = self.phi(t) if self.phi is not None else 0.0
        theta = self.w0 * t + phi
        out = self.output
        out.phi = phi
        out.theta = theta
        out.e_g = complex(self.e_peak * math.cos(theta), self.e_peak * math.sin(theta))

    @classmethod
    def uses_standard_outputs(cls, method: Any) -> bool:
        """Whether ``method`` is one of the built-in phase-specialized source equations."""
        return getattr(method, "__func__", None) in (
            cls.set_outputs, cls._set_single_phase_outputs, cls._set_three_phase_outputs,
        )

    def cpp_output_stage(self, generator: Any, method: Any):
        """Use the configured native source scenario for the built-in waveform."""
        return (generator.source_output_lines(self)
                if self.uses_standard_outputs(method) else None)

    def rhs(self, t: float):
        return ()

    def voltage(self, t: float) -> complex | float:
        """Evaluate and return the source's selected output equation."""
        self._output_equation(t)
        return self.output.e_g

    def network(self, nb):
        """Declare an ideal voltage source from the port to ground."""
        from ..solver import netlist as nl

        nb.add(nl.VoltageSource(
            nb.node(self.port), None, nb.value(self.output, "e_g")
        ))
        nb.supplies(self)



class _VoltageSourceOutput(Bag):
    __slots__ = ("i",)


class _VoltageSourceModel(NetworkComponent):
    """Scalar voltage source behind a series resistance."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    storage: ClassVar[tuple[StoragePort, ...]] = ()

    def __init__(self, voltage: float, resistance: float) -> None:
        self.voltage, self.resistance = voltage, resistance
        self.state = Empty()
        self.input = Empty()
        self.output = _VoltageSourceOutput(i=0.0)
        self.port = Port(
            self, "power", voltage=0.0, current=0.0,
            electrical_domain="dc"
        )
        self.ports = (self.port,)

    def network(self, nb):
        """Declare the source and its series resistance."""
        from ..solver import netlist as nl

        inner = nb.internal()
        nb.add(nl.Resistor(nb.node(self.port), inner, self.resistance))
        nb.add(nl.VoltageSource(
            inner, None, nb.function(lambda _t: self.voltage)
        ))
        nb.port_output(self, "i", self.port, "current")
        nb.losses(self)
        nb.supplies(self)


@register_component_type
class VoltageSource(Element):
    """Scalar named-bus voltage source behind a positive series resistance."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "voltage_source"
        bus: str
        v: float
        r: float

        _quantities = {"v": "voltage", "r": "resistance"}

        def __post_init__(self) -> None:
            if not math.isfinite(self.v):
                raise ConfigError(f"v must be finite, got {self.v}")
            if not math.isfinite(self.r) or self.r <= 0.0:
                raise ConfigError(f"r must be finite and positive, got {self.r}")

    type = "voltage_source"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        if buses[cfg.bus].phases != 1:
            raise ConfigError("a scalar voltage source needs a single-phase bus")
        self.name, self.cfg = name, cfg
        self.source = _VoltageSourceModel(cfg.v, cfg.r)

    def subsystems(self) -> dict[str, Any]:
        return {self.name: self.source}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return ((self.cfg.bus, self.source.port),)

    def signals(self) -> dict[str, float]:
        return {f"{self.name}.i": self.source.output.i}

    def retune(self, cfg: Any) -> bool:
        changed = self.cfg.r != cfg.r
        self.cfg = cfg
        self.source.voltage, self.source.resistance = cfg.v, cfg.r
        return changed

