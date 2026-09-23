"""dc-voltage outer loop -> d-axis current reference."""

from __future__ import annotations

from typing import Any, Mapping

from ..phs.states import assign
from ..firmware.blocks import clamp

__all__ = ["DCVoltageController"]


class DCVoltageController:
    """PI dc-voltage loop returning ``id_ref = ff + kp*e + ki*int(e)`` clamped to ``[floor, limit]``.

    ``e = u_dc - vdc_ref`` in dc pu; ``id_ref`` in ac current pu (positive = export).
    ``frozen`` zeroes the output and holds the integrator. ``antiwindup``: ``"none"`` or ``"conditional"``.
    Named states: ``integral`` (pu*s) and, with conditional anti-windup, ``clamped``.
    """

    def __init__(self, kp: float, ki: float, T: float, limit_pu: float, bidirectional: bool = False, antiwindup: str = "none") -> None:
        self.kp, self.ki, self.T = kp, ki, T
        self.limit = limit_pu
        self.floor = -limit_pu if bidirectional else 0.0
        self.antiwindup = antiwindup
        self.integral = 0.0
        self.error = 0.0
        self.raw = 0.0
        self.clamped = False
        self.n_clamped = 0
        self.n_reverse = 0  # steps asking for reverse (import) current
        self.first_clamp_t = -1.0

    def update(self, t: float, u_dc: float, vdc_ref: float, ff_pu: float, frozen: bool = False) -> float:
        self.error = u_dc - vdc_ref
        if frozen:
            self.raw = 0.0
        else:
            if not (self.antiwindup == "conditional" and self.clamped):
                self.integral += self.T * self.error
            self.raw = ff_pu + self.kp * self.error + self.ki * self.integral
        id_ref = clamp(self.raw, self.floor, self.limit)
        self.clamped = id_ref != self.raw
        if self.clamped:
            self.n_clamped += 1
            if self.first_clamp_t < 0.0:
                self.first_clamp_t = t
        if self.raw < 0.0:
            self.n_reverse += 1
        return id_ref

    def get_state(self) -> dict[str, Any]:
        s: dict[str, Any] = {"integral": self.integral}
        if self.antiwindup == "conditional":
            s["clamped"] = self.clamped
        return s

    def set_state(self, values: Mapping[str, Any]) -> None:
        assign(self, values, ("integral", "clamped") if self.antiwindup == "conditional" else ("integral",))
