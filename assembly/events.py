"""Prescribed time functions: source disturbances and unit start-up, setpoint and gain steps.

An event time below zero disables that event.
"""

from __future__ import annotations

import math
from typing import Optional

from ..firmware.blocks import smoothstep
from ..params import SourceParams, UnitParams

__all__ = ["SourceScenario", "UnitScenario"]


class SourceScenario:
    """Frequency step, phase jump and voltage step of one source."""

    def __init__(self, cfg: SourceParams) -> None:
        e = cfg.events
        self.freq_step_t, self.freq_step_rad_s = e.freq_step_t, 2.0 * math.pi * e.freq_step_hz
        self.phase_jump_t, self.phase_jump_rad = e.phase_jump_t, e.phase_jump_rad
        self.voltage_step_t = e.voltage_step_t
        self.voltage, self.voltage_step = cfg.v, e.voltage_step
        self._angle_steps = self.freq_step_t >= 0.0 or self.phase_jump_t >= 0.0

    @property
    def has_angle(self) -> bool:
        return self._angle_steps

    @property
    def has_voltage_step(self) -> bool:
        return self.voltage_step_t >= 0.0

    def angle(self, t: float) -> float:
        """Return the source angle deviation (rad) from the frequency step and phase jump."""
        phi = 0.0
        if self.freq_step_t >= 0.0 and t >= self.freq_step_t:
            phi += self.freq_step_rad_s * (t - self.freq_step_t)
        if self.phase_jump_t >= 0.0 and t >= self.phase_jump_t:
            phi += self.phase_jump_rad
        return phi

    def magnitude(self, t: float) -> float:
        """Return the source phase-peak voltage (V) at ``t``, including the voltage step."""
        if self.voltage_step_t >= 0.0 and t >= self.voltage_step_t:
            return self.voltage_step
        return self.voltage


class UnitScenario:
    """Start-up ramp, setpoint step and PLL gain step of one converter unit."""

    def __init__(self, cfg: UnitParams) -> None:
        e = cfg.events
        st = e.startup
        self.startup_start, self.startup_duration = st.start, st.duration
        self._startup_end = st.start + st.duration
        self.setpoint = e.setpoint
        self.pll_gain = e.pll_gain

    @property
    def arm_time(self) -> float:
        """Time (s) at which protections arm: the end of the start-up ramp."""
        return self._startup_end

    def startup(self, t: float) -> float:
        """Return the start-up ramp in [0, 1] at ``t`` (smoothstep over ``startup.duration``)."""
        if self.startup_duration > 0.0:
            if t >= self._startup_end:
                return 1.0
            if t <= self.startup_start:
                return 0.0
            return smoothstep((t - self.startup_start) / self.startup_duration)
        return 1.0 if t >= self.startup_start else 0.0

    def setpoints(self, t: float, p_ref: float, q_ref: float, v_ref: float) -> tuple[float, float, float]:
        """Return the grid-forming setpoints ``(p, q, v)`` in pu at ``t``, with ``p`` scaled by the start-up ramp."""
        sp = self.setpoint
        if sp.t >= 0.0 and t >= sp.t:
            p_ref = sp.p_ref_pu if sp.p_ref_pu is not None else p_ref
            q_ref = sp.q_ref_pu if sp.q_ref_pu is not None else q_ref
            v_ref = sp.v_ref_pu if sp.v_ref_pu is not None else v_ref
        return self.startup(t) * p_ref, q_ref, v_ref

    def pll_kp(self, t: float) -> Optional[float]:
        """Return the PLL proportional gain (pu) after the gain step, or ``None`` before it."""
        g = self.pll_gain
        if g.t >= 0.0 and t >= g.t:
            return g.kp_after_pu
        return None
