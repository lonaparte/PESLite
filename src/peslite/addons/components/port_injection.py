"""Physical series-voltage and shunt-current injection on the native circuit netlist.

Positive series voltage raises the downstream bus relative to its upstream
inlet. Positive shunt current flows *into* the measured bus from ground; its
physical component port current is therefore negative. These source signs are
encoded in their primitive orientations, not corrected after the solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

from ...components.network import Element, NetworkComponent, _register_assembled_type
from ...solver import ConfigError, Empty, Port
from ...solver import netlist as nl

__all__ = ["PortInjection", "PortInjectionModel"]


def _numbers(values: Any, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ConfigError(f"{label} must be a sequence of finite numbers")
    try:
        result = tuple(values)
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in result):
            raise ValueError
        return tuple(float(v) for v in result)
    except (TypeError, ValueError):
        raise ConfigError(f"{label} must be a sequence of finite numbers") from None


class PortInjectionModel(NetworkComponent):
    """A stateless, restartable electrical source; only the wiring differs by mode."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    storage: ClassVar[tuple] = ()
    has_source: ClassVar[bool] = True

    def __init__(self, cfg: Any, *, phases: int, domain: str | None) -> None:
        self.cfg = cfg
        self.phases = phases
        zero = 0j if phases == 3 else 0.0
        self.state = Empty()
        self.input = Empty()
        outputs = {"injection": zero, "voltage": zero, "current": zero}
        if cfg.kind == "voltage":
            outputs.update(source_current=zero, upstream_voltage=zero)
        self.output = SimpleNamespace(**outputs)
        scale = 1.5 if phases == 3 else 1.0
        self.port = Port(self, "power", voltage=zero, current=zero,
                         scale=scale, standalone="voltage", electrical_domain=domain)
        self.ports = (self.port,)
        if cfg.kind == "voltage":
            self.inlet = Port(self, "inlet", voltage=zero, current=zero,
                              scale=scale, standalone="voltage", electrical_domain=domain)
            self.ports += (self.inlet,)

    def injection_at(self, t: float) -> complex | float:
        """Return the oriented source quantity in SI; never mutate simulation state."""
        cfg = self.cfg
        envelope = cfg.bias + cfg.amplitude * math.cos(math.tau * cfg.frequency * t + cfg.phase)
        if cfg.tones:
            envelope += sum(amplitude * math.cos(math.tau * f * t + phase)
                            for f, amplitude, phase in zip(cfg.tones, cfg.tone_amplitudes,
                                                           cfg.tone_phases))
        if self.phases == 1:
            return envelope
        frame = math.tau * cfg.frame_frequency * t
        if cfg.axis in ("d", "q"):
            frame += math.pi / 2 if cfg.axis == "q" else 0.0
        elif cfg.axis == "beta":
            frame = math.pi / 2
        elif cfg.axis == "alpha":
            frame = 0.0
        # Positive/negative sequence components are represented directly in
        # the space vector, not by three independent phase sources.
        elif cfg.axis in ("positive", "negative"):
            frame = (math.tau * (cfg.frame_frequency + (
                cfg.frequency if cfg.axis == "positive" else -cfg.frequency)) * t
                     + cfg.phase)
            envelope = cfg.amplitude
        return envelope * complex(math.cos(frame), math.sin(frame))

    def network(self, builder) -> None:
        """Declare exactly one source across the native series/shunt port topology."""
        bus = builder.node(self.port)
        value = builder.function(self.injection_at)
        if self.cfg.kind == "voltage":
            inlet = builder.node(self.inlet)
            # V_bus - V_inlet = +injection; the current is positive bus -> inlet.
            source = nl.VoltageSource(bus, inlet, value)
            builder.add(source)
            builder.output(self, "injection", ("branch_voltage", source))
            builder.output(self, "source_current", ("branch_current", source))
            builder.port_output(self, "upstream_voltage", self.inlet, "voltage")
        else:
            # Source current from ground -> bus is positive INTO the external circuit.
            source = nl.CurrentSource(None, bus, value)
            builder.add(source)
            builder.output(self, "injection", ("branch_current", source))
        builder.port_output(self, "voltage", self.port, "voltage")
        builder.port_output(self, "current", self.port, "current")
        builder.supplies(self)


@_register_assembled_type
class PortInjection(Element):
    """An add-on component supporting series-voltage and shunt-current injection."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bus: str
        kind: str = field(default="current", metadata={"runtime": False})
        inlet: str = field(default="", metadata={"runtime": False})
        amplitude: float = 0.0
        frequency: float = 0.0
        phase: float = 0.0
        bias: float = 0.0
        axis: str = field(default="alpha", metadata={"runtime": False})
        frame_frequency: float = 0.0
        tones: tuple[float, ...] = ()
        tone_amplitudes: tuple[float, ...] = ()
        tone_phases: tuple[float, ...] = ()
        type: str = "port_injection"

        def __post_init__(self) -> None:
            if not isinstance(self.bus, str) or not self.bus:
                raise ConfigError("bus must name an electrical port")
            if self.kind not in ("voltage", "current"):
                raise ConfigError("kind must be 'voltage' or 'current'")
            if self.inlet and (self.kind != "voltage" or self.inlet == self.bus):
                raise ConfigError("inlet must name a distinct bus for voltage injection only")
            if self.axis not in ("alpha", "beta", "d", "q", "positive", "negative"):
                raise ConfigError("axis must be alpha, beta, d, q, positive, or negative")
            for key in ("amplitude", "frequency", "phase", "bias", "frame_frequency"):
                value = getattr(self, key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ConfigError(f"{key} must be a finite number")
            if self.frequency < 0 or self.frame_frequency < 0:
                raise ConfigError("frequency and frame_frequency must be nonnegative")
            for key in ("tones", "tone_amplitudes", "tone_phases"):
                object.__setattr__(self, key, _numbers(getattr(self, key), key))
            if len(self.tones) != len(self.tone_amplitudes) or len(self.tones) != len(self.tone_phases):
                raise ConfigError("tones, tone_amplitudes and tone_phases must have the same length")
            if any(f <= 0 for f in self.tones):
                raise ConfigError("tone frequencies must be positive")
            if self.axis in ("positive", "negative") and (self.tones or self.bias):
                raise ConfigError("sequence injection requires a single tone with zero bias")

    type = "port_injection"
    domain_transparent = True

    @staticmethod
    def _phases(bus: Any) -> int:
        """A named bus publishes phases; an assembled Unit endpoint is a Port."""
        if isinstance(bus, Port):
            return 3 if isinstance(bus.voltage, complex) else 1
        return bus.phases

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self.name, self.cfg = name, cfg
        bus = buses[cfg.bus]
        phases = self._phases(bus)
        if cfg.inlet:
            if cfg.inlet not in buses:
                raise ConfigError(f"unknown inlet bus {cfg.inlet!r}")
            other = buses[cfg.inlet]
            if self._phases(other) != phases:
                raise ConfigError("voltage injection inlet and bus must have matching phases")
            if (getattr(other, "electrical_domain", None) is not None
                    and getattr(bus, "electrical_domain", None) is not None
                    and other.electrical_domain != bus.electrical_domain):
                raise ConfigError("voltage injection inlet and bus must have matching domains")
        self.model = PortInjectionModel(
            cfg, phases=phases, domain=getattr(bus, "electrical_domain", None)
        )
        if phases == 1 and cfg.axis not in ("alpha", "d"):
            raise ConfigError("a single-phase injection supports only the alpha/d axis")

    def subsystems(self) -> dict:
        return {self.name: self.model}

    @staticmethod
    def connections() -> dict:
        return {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        if self.cfg.kind == "voltage":
            return ((self.cfg.inlet or f"{self.name}.bus.in", self.model.inlet),
                    (self.cfg.bus, self.model.port))
        return ((self.cfg.bus, self.model.port),)

    def series_connections(self) -> dict[str, str]:
        return ({self.cfg.bus: f"{self.name}.bus.in"}
                if self.cfg.kind == "voltage" and not self.cfg.inlet else {})

    def signals(self) -> dict[str, complex | float]:
        signals = {f"{self.name}.{field}": value for field, value in vars(self.model.output).items()}
        return signals

    def retune(self, cfg: Any) -> bool:
        """Retune only source values; topology fields are structural by construction."""
        changed = (cfg.kind, cfg.bus, cfg.inlet, cfg.axis) != (
            self.cfg.kind, self.cfg.bus, self.cfg.inlet, self.cfg.axis)
        self.cfg, self.model.cfg = cfg, cfg
        return changed
