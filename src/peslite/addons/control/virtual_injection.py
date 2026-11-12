"""Typed, restart-free virtual sinusoidal injection at existing control-graph edges.

Each block is a pure feed-through with a deterministic function of simulation
time. Its two ports use the ordinary ControlGraph type checking and scheduling;
it never reaches into a controller's private state or changes the solver path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from ...control.loops import (ANGLE, DUTY_RATIO, FREQUENCY, GENERIC_CURRENT,
                              GENERIC_VOLTAGE, I_DQ, POWER, V_DQ,
                              Loop, register_loop_type)
from ...solver import ConfigError

__all__ = ["VirtualVoltageInjection", "VirtualCurrentInjection",
           "VirtualDqVoltageInjection", "VirtualDqCurrentInjection",
           "VirtualDutyInjection", "VirtualPowerInjection",
           "VirtualFrequencyInjection", "VirtualAngleInjection"]


@dataclass(frozen=True, kw_only=True)
class _SineParams:
    amplitude: float
    frequency: float
    phase: float = 0.0
    bias: float = 0.0
    axis: str = "d"
    period: float | None = None
    type: str = "virtual_injection"

    def __post_init__(self):
        for name in ("amplitude", "frequency", "phase", "bias"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ConfigError(f"{name} must be finite")
        if self.frequency < 0:
            raise ConfigError("frequency must be nonnegative")
        if self.period is not None and (
                isinstance(self.period, bool) or not isinstance(self.period, (int, float))
                or not math.isfinite(self.period) or self.period <= 0):
            raise ConfigError("period must be finite and positive")
        if self.axis not in ("d", "q"):
            raise ConfigError("axis must be 'd' or 'q'")


class _VirtualSine(Loop):
    """Implement one numeric injection law for all typed graph-edge adapters."""

    complex_signal: ClassVar[bool] = False
    cpp_export: ClassVar[bool] = False
    outputs_from_state: ClassVar[bool] = False
    observations: ClassVar[tuple[str, ...]] = ("injection_d", "injection_q",
                                               "before_d", "before_q",
                                               "after_d", "after_q")
    state_names: tuple[str, ...] = ()

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        if not self.complex_signal and cfg.axis != "d":
            raise ConfigError("scalar control injection supports only the d axis")
        self.last = 0j if self.complex_signal else 0.0
        self.before = self.after = 0j if self.complex_signal else 0.0

    def initial_outputs(self):
        return {"injected": 0j if self.complex_signal else 0.0}

    def injected(self, t, original):
        """Inject in the input's existing units, with a stable phase relative to t=0."""
        cfg = self.cfg
        perturbation = cfg.bias + cfg.amplitude * math.cos(math.tau * cfg.frequency * t + cfg.phase)
        if self.complex_signal:
            perturbation *= 1j if cfg.axis == "q" else 1.0
        self.last = perturbation
        self.before = original
        self.after = original + perturbation
        return self.after

    def sample(self, t, inputs):
        return {"injected": self.injected(t, inputs["signal"])}

    def flow(self, t, inputs):
        return {"injected": self.injected(t, inputs["signal"])}, {}

    def observe(self):
        def parts(value):
            return (float(value.real), float(value.imag)) if isinstance(value, complex) else (float(value), 0.0)
        d, q = parts(self.last)
        before_d, before_q = parts(self.before)
        after_d, after_q = parts(self.after)
        return {"injection_d": d, "injection_q": q,
                "before_d": before_d, "before_q": before_q,
                "after_d": after_d, "after_q": after_q}


@register_loop_type
class VirtualVoltageInjection(_VirtualSine):
    @dataclass(frozen=True, kw_only=True)
    class Params(_SineParams):
        type: str = "virtual_voltage_injection"

    type = "virtual_voltage_injection"
    inputs = {"signal": GENERIC_VOLTAGE}
    outputs = {"injected": GENERIC_VOLTAGE}


@register_loop_type
class VirtualCurrentInjection(_VirtualSine):
    @dataclass(frozen=True, kw_only=True)
    class Params(_SineParams):
        type: str = "virtual_current_injection"

    type = "virtual_current_injection"
    inputs = {"signal": GENERIC_CURRENT}
    outputs = {"injected": GENERIC_CURRENT}


@register_loop_type
class VirtualDqVoltageInjection(_VirtualSine):
    @dataclass(frozen=True, kw_only=True)
    class Params(_SineParams):
        type: str = "virtual_dq_voltage_injection"

    type = "virtual_dq_voltage_injection"
    complex_signal = True
    inputs = {"signal": V_DQ}
    outputs = {"injected": V_DQ}


@register_loop_type
class VirtualDqCurrentInjection(_VirtualSine):
    @dataclass(frozen=True, kw_only=True)
    class Params(_SineParams):
        type: str = "virtual_dq_current_injection"

    type = "virtual_dq_current_injection"
    complex_signal = True
    inputs = {"signal": I_DQ}
    outputs = {"injected": I_DQ}


@register_loop_type
class VirtualDutyInjection(_VirtualSine):
    @dataclass(frozen=True, kw_only=True)
    class Params(_SineParams):
        type: str = "virtual_duty_injection"

    type = "virtual_duty_injection"
    inputs = {"signal": DUTY_RATIO}
    outputs = {"injected": DUTY_RATIO}


@register_loop_type
class VirtualPowerInjection(_VirtualSine):
    @dataclass(frozen=True, kw_only=True)
    class Params(_SineParams):
        type: str = "virtual_power_injection"

    type = "virtual_power_injection"
    inputs = {"signal": POWER}
    outputs = {"injected": POWER}


@register_loop_type
class VirtualFrequencyInjection(_VirtualSine):
    @dataclass(frozen=True, kw_only=True)
    class Params(_SineParams):
        type: str = "virtual_frequency_injection"

    type = "virtual_frequency_injection"
    inputs = {"signal": FREQUENCY}
    outputs = {"injected": FREQUENCY}


@register_loop_type
class VirtualAngleInjection(_VirtualSine):
    @dataclass(frozen=True, kw_only=True)
    class Params(_SineParams):
        type: str = "virtual_angle_injection"

    type = "virtual_angle_injection"
    inputs = {"signal": ANGLE}
    outputs = {"injected": ANGLE}
