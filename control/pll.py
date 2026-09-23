"""Synchronous-reference-frame PLL, with a choice of phase-detector normalisation."""

from __future__ import annotations

import cmath
from typing import Any, Mapping

from ..phs.states import assign

__all__ = ["SRFPLL"]


class SRFPLL:
    """SRF-PLL: ``omega = w0 + kp*eps + ki*integral``, ``theta += T*omega``.

    Input: dq voltage (pu) in the frame of the previous angle. ``normalisation``:
    ``"rated"`` (eps = v_q) or ``"amplitude"`` (eps = v_q / u_g, u_g a filtered v_d estimate).
    Named states: ``theta`` (rad, unwrapped), ``integral`` (pu*s), ``u_g`` (pu, amplitude only).
    """

    def __init__(self, kp: float, ki: float, w0: float, T: float, theta0: float = 0.0,
                 normalisation: str = "rated", u_g0: float = 1.0) -> None:
        if normalisation not in ("rated", "amplitude"):
            raise ValueError(f"PLL normalisation must be 'rated' or 'amplitude', got {normalisation!r}")
        self.kp, self.ki, self.w0, self.T = kp, ki, w0, T
        self.normalisation = normalisation
        self.theta = theta0
        self.omega = w0
        self.integral = 0.0
        self.u_g = u_g0 if normalisation == "amplitude" else None

    def rotate(self, u_ab: complex) -> complex:
        return u_ab * cmath.exp(-1j * self.theta)

    def update(self, v_pu: complex) -> tuple[float, float]:
        """Advance one step from the dq voltage (pu) and return ``(theta, omega)``."""
        u_g = self.u_g
        if u_g is None:                                    # rated
            eps = v_pu.imag
        else:                                              # amplitude
            eps = v_pu.imag / u_g if u_g > 0.0 else 0.0
        self.integral += self.T * eps
        self.omega = self.w0 + self.kp * eps + self.ki * self.integral
        self.theta += self.T * self.omega
        if u_g is not None:
            self.u_g = u_g + self.T * self.kp * (v_pu.real - u_g)
        return self.theta, self.omega

    @property
    def freq_dev_hz(self) -> float:
        return (self.omega - self.w0) / (2.0 * cmath.pi)

    def get_state(self) -> dict[str, Any]:
        s: dict[str, Any] = {"theta": self.theta, "integral": self.integral}
        if self.u_g is not None:
            s["u_g"] = self.u_g
        return s

    def set_state(self, values: Mapping[str, Any]) -> None:
        assign(self, values, ("theta", "integral") if self.u_g is None
               else ("theta", "integral", "u_g"))
