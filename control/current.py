"""dq-frame PI current controller with decoupling and voltage feed-forward."""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["CurrentController"]


class CurrentController:
    """dq PI current controller: ``u = u_ff + (r_pu + j*omega/w0*x_pu)*i + kp*e + ki*integral``.

    Signals are pu; T in s, w0 and omega in rad/s. Anti-windup hooks:
    ``rollback`` (conditional) and ``backcalculate``; the caller decides when to use them.
    Named state: ``integral_pu`` (complex dq, pu*s).
    """

    def __init__(self, x_pu: float, r_pu: float, w0: float, T: float, kp: float, ki: float,
                 decoupling: bool = True, feedforward: bool = True,
                 antiwindup: str = "conditional") -> None:
        self.x_pu, self.r_pu, self.w0, self.T = x_pu, r_pu, w0, T
        self.kp, self.ki = kp, ki
        self.decoupling, self.feedforward = decoupling, feedforward
        self.antiwindup = antiwindup
        self.integral = 0j
        self._prev_integral = 0j
        self._last: tuple[complex, complex, complex, float] | None = None

    @classmethod
    def from_bandwidth(cls, x_pu: float, r_pu: float, w0: float, T: float, bw_hz: float, **kw) -> "CurrentController":
        import math

        wc = 2.0 * math.pi * bw_hz
        return cls(x_pu, r_pu, w0, T, kp=x_pu / w0 * wc, ki=r_pu * wc, **kw)

    def command(self, e: complex, i: complex, u_ff: complex, omega: float) -> complex:
        """Voltage command from the current integrator state (no integration)."""
        u = self.kp * e + self.ki * self.integral
        if self.feedforward:
            u += u_ff
        if self.decoupling:
            u += (self.r_pu + 1j * omega / self.w0 * self.x_pu) * i
        return u

    def update(self, i_ref: complex, i: complex, u_ff: complex, omega: float) -> complex:
        """Integrate the error and return the voltage command (dq)."""
        e = i_ref - i
        self._prev_integral = self.integral
        self.integral = self.integral + self.T * e
        self._last = (e, i, u_ff, omega)
        return self.command(e, i, u_ff, omega)

    def rollback(self) -> complex:
        """Undo the last integration and return the recomputed command."""
        self.integral = self._prev_integral
        assert self._last is not None
        return self.command(*self._last)

    def backcalculate(self, u_cmd: complex, u_limited: complex) -> None:
        if self.kp != 0.0:
            self.integral = self.integral + self.T * (u_limited - u_cmd) / self.kp

    def get_state(self) -> dict[str, Any]:
        return {"integral_pu": self.integral}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if "integral_pu" in values:
            self.integral = self._prev_integral = complex(values["integral_pu"])
        unknown = set(values) - {"integral_pu"}
        if unknown:
            raise KeyError(f"CurrentController has no state(s) {sorted(unknown)}")
