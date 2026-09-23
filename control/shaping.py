"""Grid-forming voltage-reference shaping: virtual impedance, virtual admittance, active damping.

Signals are pu (ac base); time s, omega and w0 rad/s.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..firmware.blocks import HighPass1
from ..phs.states import gather, scatter

__all__ = ["VirtualImpedance", "VirtualAdmittance", "ActiveDamping"]


class VirtualImpedance:
    def __init__(self, r_pu: float, x_pu: float, w0: float) -> None:
        self.r_pu, self.x_pu, self.w0 = r_pu, x_pu, w0

    def drop(self, i_dq: complex, omega: float) -> complex:
        return (self.r_pu + 1j * omega / self.w0 * self.x_pu) * i_dq


class VirtualAdmittance:
    """Current reference from ``x_pu/w0 di/dt = v_ref - v - (r_pu + j omega/w0 x_pu) i``, limited to ``|i| <= i_limit``."""

    def __init__(self, r_pu: float, x_pu: float, w0: float, T: float, i_limit: float = math.inf) -> None:
        if x_pu <= 0.0:
            raise ValueError("a virtual admittance needs L_v > 0 (virtual_admittance.x_v_pu > 0)")
        self.r_pu, self.x_pu, self.w0, self.T, self.i_limit = r_pu, x_pu, w0, T, i_limit
        self.i_ref = 0j

    def update(self, v_ref_dq: complex, v_dq: complex, omega: float) -> complex:
        i_ref = self.i_ref + self.T * self.w0 / self.x_pu * (
            v_ref_dq - v_dq - (self.r_pu + 1j * omega / self.w0 * self.x_pu) * self.i_ref)
        mag = abs(i_ref)
        if mag > self.i_limit:
            i_ref *= self.i_limit / mag
        self.i_ref = i_ref
        return i_ref

    def get_state(self) -> dict[str, Any]:
        return {"i_ref_pu": self.i_ref}  # dq, pu

    def set_state(self, values: Mapping[str, Any]) -> None:
        if set(values) - {"i_ref_pu"}:
            raise KeyError("VirtualAdmittance state is i_ref_pu (current pu)")
        if "i_ref_pu" in values:
            self.i_ref = complex(values["i_ref_pu"])


class ActiveDamping:
    """Damping voltage ``r_a`` times the current high-pass filtered at ``alpha_d`` (rad/s).

    Named state: ``hpf_pu`` (low-passed current, pu; starts at zero).
    """

    def __init__(self, r_a: float, alpha_d: float, T: float) -> None:
        self.r_a = r_a
        self.hpf = HighPass1(alpha_d / (2.0 * math.pi), T, init_on_first=False, y0=0j)

    def __call__(self, i_dq: complex) -> complex:
        return self.r_a * self.hpf.update(i_dq)

    def get_state(self) -> dict[str, Any]:
        return gather({"hpf_pu": self.hpf})  # low-passed current, pu

    def set_state(self, values: Mapping[str, Any]) -> None:
        scatter({"hpf_pu": self.hpf}, values)
