"""Modulators and triangular-carrier helpers.

A modulator is called as ``mod(t, T_s, d_abc, theta=None, omega=None) -> SwitchingSequence``
with times in s, duty ratios in [0, 1], ``theta`` in rad and ``omega`` in rad/s.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..assembly.protocols import Modulator, SwitchingSequence
from ..params import PWMParams, SimulationParams

__all__ = ["CarrierComparison", "SynchronousCarrier", "ZOH", "StepAveragedCarrier",
           "SampledCarrier", "carrier", "carrier_position", "duty_fraction", "make_modulator"]


_SNAP = 1e-9  # vertex snapping tolerance (carrier periods)

def carrier_position(t: float, f_sw: float, phase: float) -> float:
    """Return the carrier position in [0, 1) periods; ``f_sw`` in Hz, ``phase`` in periods."""
    u = math.fmod(t * f_sw + phase, 1.0)
    if u < 0.0:
        u += 1.0
    if u > 1.0 - _SNAP or u < _SNAP:
        return 0.0
    if abs(u - 0.5) < _SNAP:
        return 0.5
    return u

def carrier(t: float, f_sw: float, phase: float = 0.0) -> float:
    """Return the triangular carrier value in [-1, 1]; ``phase`` in carrier periods.

    With ``phase = 0`` the carrier is at -1 and rising at ``t = 0``.
    """
    u = carrier_position(t, f_sw, phase)
    return -1.0 + 4.0 * u if u < 0.5 else 3.0 - 4.0 * u

def duty_fraction(m: float, t0: float, dt: float, f_sw: float, phase: float = 0.0) -> float:
    """Return the fraction of ``[t0, t0 + dt]`` during which ``m`` is at or above the carrier."""
    covered = 0.0
    duty_time = 0.0
    guard = 0
    while guard < 8 and covered < dt - 1e-18:
        guard += 1
        u = carrier_position(t0 + covered, f_sw, phase)
        rising = u < 0.5
        seg = ((0.5 if rising else 1.0) - u) / f_sw
        seg = min(seg, dt - covered)
        c0 = -1.0 + 4.0 * u if rising else 3.0 - 4.0 * u
        slope = 4.0 * f_sw if rising else -4.0 * f_sw
        crossing = min(seg, max(0.0, (m - c0) / slope))
        duty_time += crossing if rising else seg - crossing
        covered += seg
    return duty_time / dt


_EPS = 1e-13

class CarrierComparison:
    """Triangular-carrier comparison with exact switching instants.

    ``f_sw`` in Hz, ``phase`` in carrier periods, ``min_interval`` in s.
    """

    def __init__(self, f_sw: float, phase: float = 0.0, min_interval: float = 1e-12) -> None:
        self.f_sw = float(f_sw)
        self.phase = float(phase)
        self.min_interval = min_interval

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        f = self.f_sw
        m = np.clip(2.0 * np.asarray(d_abc, dtype=float) - 1.0, -1.0, 1.0).tolist()
        # initial switching state at t
        u_t = carrier_position(t, f, self.phase)
        c_t = -1.0 + 4.0 * u_t if u_t < 0.5 else 3.0 - 4.0 * u_t
        if u_t < 0.5:
            q = [1.0 if m[k] > c_t else 0.0 for k in range(3)]
        else:
            q = [1.0 if m[k] >= c_t else 0.0 for k in range(3)]
        events: list[tuple[float, int, float]] = []
        covered = 0.0
        guard = 0
        while covered < T_s - _EPS and guard < 64:
            guard += 1
            u = carrier_position(t + covered, f, self.phase)
            rising = u < 0.5
            seg = ((0.5 if rising else 1.0) - u) / f
            seg = min(seg, T_s - covered)
            c0 = -1.0 + 4.0 * u if rising else 3.0 - 4.0 * u
            slope = 4.0 * f if rising else -4.0 * f
            for k in range(3):
                x = (m[k] - c0) / slope
                if 0.0 < x < seg:
                    # rising carrier crosses m from below -> switch off; falling -> on
                    events.append((covered + x, k, 0.0 if rising else 1.0))
            covered += seg
        events.sort(key=lambda e: e[0])
        dts: list[float] = []
        states: list[list[float]] = []
        t_prev = 0.0
        for time, k, new_state in events:
            if time - t_prev > self.min_interval:
                dts.append(time - t_prev)
                states.append(list(q))
                t_prev = time
            q[k] = new_state
        if T_s - t_prev > self.min_interval or not dts:
            dts.append(T_s - t_prev)
            states.append(list(q))
        else:  # absorb a vanishing last interval into the previous one
            dts[-1] += T_s - t_prev
        return SwitchingSequence(np.asarray(dts), np.asarray(states))

class ZOH:
    """Averaged bridge: hold the duty ratios as the switching state for the period."""

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        d = np.clip(np.asarray(d_abc, dtype=float), 0.0, 1.0)
        return SwitchingSequence(np.array([T_s]), d.reshape(1, 3))

class StepAveragedCarrier:
    """Step-averaged bridge: per ``dt`` step, the on-fraction of the carrier comparison."""

    def __init__(self, f_sw: float, dt: float, phase: float = 0.0) -> None:
        self.f_sw, self.dt, self.phase = float(f_sw), float(dt), float(phase)

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        n = max(1, int(round(T_s / self.dt)))
        h = T_s / n
        m = np.clip(2.0 * np.asarray(d_abc, dtype=float) - 1.0, -1.0, 1.0)
        states = np.empty((n, 3))
        for i in range(n):
            t0 = t + i * h
            for k in range(3):
                states[i, k] = duty_fraction(m[k], t0, h, self.f_sw, self.phase)
        return SwitchingSequence(np.full(n, h), states)

class SampledCarrier:
    """Carrier comparison evaluated once per ``dt`` step."""

    def __init__(self, f_sw: float, dt: float, phase: float = 0.0) -> None:
        self.f_sw, self.dt, self.phase = float(f_sw), float(dt), float(phase)

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        n = max(1, int(round(T_s / self.dt)))
        h = T_s / n
        m = np.clip(2.0 * np.asarray(d_abc, dtype=float) - 1.0, -1.0, 1.0)
        states = np.empty((n, 3))
        for i in range(n):
            c = carrier(t + (i + 1) * h, self.f_sw, self.phase)
            states[i] = [1.0 if m[k] >= c else 0.0 for k in range(3)]
        return SwitchingSequence(np.full(n, h), states)


class SynchronousCarrier:
    """Carrier comparison with the carrier locked to the controller angle.

    Carrier position is ``pulse_ratio * theta / (2 pi) + phase`` (periods);
    requires ``theta`` (rad) and ``omega`` (rad/s).
    """

    def __init__(self, pulse_ratio: int, phase: float = 0.0, min_interval: float = 1e-12) -> None:
        self.pulse_ratio = int(pulse_ratio)
        self.phase = float(phase)
        self.min_interval = min_interval

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        if theta is None or omega is None:
            raise ValueError("synchronous modulation needs the controller's angle: this controller "
                             "reports no theta/omega in its ControlOutput")
        f = self.pulse_ratio * omega / (2.0 * np.pi)  # carrier frequency (Hz)
        if f <= 0.0:
            raise ValueError(f"synchronous modulation needs omega > 0, got {omega}")
        position = self.pulse_ratio * theta / (2.0 * np.pi) + self.phase  # carrier position at t (periods)
        return CarrierComparison(f, position - t * f, self.min_interval)(t, T_s, d_abc)


def make_modulator(pwm: PWMParams, f0: float, sim: SimulationParams) -> Modulator:
    """Return the modulator for ``simulation.bridge`` and ``pwm.sync``.

    ``bridge``: ``"switching"``, ``"averaged"`` or ``"step_averaged"``.
    """
    bridge = sim.bridge
    if bridge == "switching":
        if pwm.sync == "synchronous":
            return SynchronousCarrier(round(pwm.f_sw / f0), pwm.carrier_phase)
        return CarrierComparison(pwm.f_sw, pwm.carrier_phase)
    if bridge == "averaged":
        return ZOH()
    return StepAveragedCarrier(pwm.f_sw, sim.solver.dt, pwm.carrier_phase)
