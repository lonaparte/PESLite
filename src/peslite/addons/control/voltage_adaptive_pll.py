"""Voltage-adaptive SRF-PLL add-on with frequency tracking."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Optional

from . import (ANGLE, FREQUENCY, V_AB, ConfigError, Filter, Integrator, Loop,
               register_loop_type)

__all__ = ["VoltageAdaptivePLL"]


@register_loop_type
class VoltageAdaptivePLL(Loop):
    """PLL with filtered measured-voltage magnitude and grid-frequency tracking.

    With ``alpha = 2*pi*bandwidth``, the equations in pu are::

        eps = imag(v * exp(-j*theta)) / u_g
        omega = omega_g + 2*alpha*eps
        d(theta)/dt = omega
        d(omega_g)/dt = alpha**2*eps
        d(u_g)/dt = 2*alpha*(real(v * exp(-j*theta)) - u_g)

    This differs from PESLite's default rated-voltage SRF-PLL because the measured voltage updates
    the normalising magnitude. ``theta`` remains unwrapped to match PESLite's state convention.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bandwidth: float
        period: Optional[float] = None
        type: str = "voltage_adaptive_pll"

        def __post_init__(self):
            if not math.isfinite(self.bandwidth) or self.bandwidth <= 0.0:
                raise ConfigError(f"bandwidth must be finite and positive, got {self.bandwidth}")

    type = "voltage_adaptive_pll"
    role = "pll"
    outputs_from_state = True
    inputs = {"v": V_AB}
    outputs = {"theta": ANGLE, "frame": ANGLE, "omega": FREQUENCY}
    flow_inputs = ("v",)

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        alpha = 2.0 * math.pi * cfg.bandwidth
        self.kp, self.ki = 2.0 * alpha, alpha * alpha
        self.w0 = unit.base.w0
        self.angle = self.state_block("theta", Integrator(self.block_period, initial=0.0))
        self.frequency = self.state_block(
            "omega_g", Integrator(self.block_period, initial=self.w0)
        )
        self.magnitude = self.state_block(
            "u_g_pu", Filter((self.kp,), (1.0, self.kp),
                              self.block_period, initial=1.0)
        )
        self.omega = self.w0

    @property
    def theta(self):
        return self.angle.value

    @property
    def omega_g(self):
        return self.frequency.value

    @property
    def u_g(self):
        return self.magnitude.x0

    def initial_outputs(self):
        return {"theta": self.theta, "frame": self.theta, "omega": self.omega}

    def equation(self, voltage):
        frame = self.theta
        measured = voltage * cmath.exp(-1j * frame)
        u_g = self.u_g
        error = measured.imag / u_g if u_g > 0.0 else 0.0
        omega_g = self.frequency(self.ki * error)
        self.omega = omega_g + self.kp * error
        theta = self.angle(self.omega)
        self.magnitude(measured.real)
        return theta, frame, self.omega

    def reset_integrator(self):
        self.frequency.reset()
