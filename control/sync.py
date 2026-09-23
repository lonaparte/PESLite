"""Grid-forming synchronization laws: PSC, droop, VSG, dVOC and matching control.

Each law returns angle (rad), frequency (rad/s) and voltage magnitude (pu); time is s.
"""

from __future__ import annotations

import cmath
from typing import Any, Mapping

from ..assembly.protocols import SyncOutput
from ..phs.states import assign
from ..params import DroopParams, DVOCParams, GFMParams, MatchingParams, PSCParams, VSGParams

__all__ = ["PSC", "Droop", "VSG", "DVOC", "Matching", "make_law"]


class PSC:
    """Power-synchronization control with a PI voltage-magnitude loop.

    Named states: ``theta`` (rad), ``v_int`` (pu*s).
    """

    def __init__(self, cfg: PSCParams, w0: float, T: float, v0_pu: float = 1.0) -> None:
        self.k_p, self.k_v, self.k_vi = cfg.k_p_pu, cfg.k_v, cfg.k_vi
        self.w0, self.T = w0, T
        self.theta = 0.0
        self.omega = w0
        self.v_int = 0.0
        self.v_mag = v0_pu

    def update(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq=0j) -> SyncOutput:
        self.omega = self.w0 + self.k_p * (p_ref_pu - p_pu)
        self.theta += T * self.omega
        e_v = v_ref_pu - v_mag_pu
        self.v_int += T * e_v
        self.v_mag = v_ref_pu + self.k_v * e_v + self.k_vi * self.v_int
        return SyncOutput(self.theta, self.omega, self.v_mag)

    def _state_names(self) -> tuple[str, ...]:
        return ("theta", "v_int")

    def get_state(self) -> dict[str, Any]:
        return {n: getattr(self, n) for n in self._state_names()}

    def set_state(self, values: Mapping[str, Any]) -> None:
        assign(self, values, self._state_names())


class Droop:
    """P-f / Q-V droop control. Named state: ``theta`` (rad)."""

    def __init__(self, cfg: DroopParams, w0: float, T: float, v0_pu: float = 1.0) -> None:
        self.m_p, self.n_q = cfg.m_p_pu, cfg.n_q_pu
        self.w0, self.T = w0, T
        self.theta = 0.0
        self.omega = w0
        self.v_mag = v0_pu

    def update(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq=0j) -> SyncOutput:
        self.omega = self.w0 + self.m_p * (p_ref_pu - p_pu)
        self.theta += T * self.omega
        self.v_mag = v_ref_pu + self.n_q * (q_ref_pu - q_pu)
        return SyncOutput(self.theta, self.omega, self.v_mag)

    def _state_names(self) -> tuple[str, ...]:
        return ("theta",)

    def get_state(self) -> dict[str, Any]:
        return {n: getattr(self, n) for n in self._state_names()}

    def set_state(self, values: Mapping[str, Any]) -> None:
        assign(self, values, self._state_names())


class VSG:
    """Virtual synchronous generator: ``2H dw/dt = p* - p - d_p (w - w0)/w0``.

    Named states: ``theta`` (rad), ``dw_pu`` and, if ``t_q > 0``, ``v_mag`` (pu).
    """

    def __init__(self, cfg: VSGParams, w0: float, T: float, v0_pu: float = 1.0) -> None:
        self.h_s, self.d_p, self.k_q, self.t_q = cfg.h_s, cfg.d_p_pu, cfg.k_q_pu, cfg.t_q
        self.w0, self.T = w0, T
        self.theta = 0.0
        self.omega = w0
        self.dw_pu = 0.0
        self.v_mag = v0_pu

    def update(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq=0j) -> SyncOutput:
        self.dw_pu += T / (2.0 * self.h_s) * (p_ref_pu - p_pu - self.d_p * self.dw_pu)
        self.omega = self.w0 * (1.0 + self.dw_pu)
        self.theta += T * self.omega
        v_cmd = v_ref_pu + self.k_q * (q_ref_pu - q_pu)
        if self.t_q > 0.0:
            self.v_mag += T / self.t_q * (v_cmd - self.v_mag)
        else:
            self.v_mag = v_cmd
        return SyncOutput(self.theta, self.omega, self.v_mag)

    def _state_names(self) -> tuple[str, ...]:
        return ("theta", "dw_pu", "v_mag") if self.t_q > 0.0 else ("theta", "dw_pu")

    def get_state(self) -> dict[str, Any]:
        return {n: getattr(self, n) for n in self._state_names()}

    def set_state(self, values: Mapping[str, Any]) -> None:
        assign(self, values, self._state_names())


class DVOC:
    """Dispatchable virtual oscillator control, integrated in polar form.

    ``dv/dt = j w0 v + eta e^{j kappa} ((p* - j q*) v / V*^2 - i) + eta alpha (1 - |v|^2 / V*^2) v``
    with oscillator voltage ``v`` and current ``i`` in pu. Named states: ``theta`` (rad), ``v_mag`` (pu).
    """

    def __init__(self, cfg: DVOCParams, w0: float, T: float, v0_pu: float = 1.0) -> None:
        self.eta, self.alpha = cfg.eta_pu, cfg.alpha_pu
        self.rot = cmath.exp(1j * cfg.kappa_rad)
        self.w0, self.T = w0, T
        self.theta = 0.0
        self.omega = w0
        self.v_mag = v0_pu

    def update(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq=0j) -> SyncOutput:
        V = self.v_mag
        i_star = complex(p_ref_pu, -q_ref_pu) * V / (v_ref_pu * v_ref_pu)
        w = self.eta * self.rot * (i_star - i_dq) + self.eta * self.alpha * (1.0 - V * V / (v_ref_pu * v_ref_pu)) * V
        self.omega = self.w0 + w.imag / V
        self.v_mag = V + T * w.real
        self.theta += T * self.omega
        return SyncOutput(self.theta, self.omega, self.v_mag)

    def _state_names(self) -> tuple[str, ...]:
        return ("theta", "v_mag")

    def get_state(self) -> dict[str, Any]:
        return {n: getattr(self, n) for n in self._state_names()}

    def set_state(self, values: Mapping[str, Any]) -> None:
        assign(self, values, self._state_names())


class Matching:
    """Matching control: frequency proportional to the dc-link voltage. Named state: ``theta`` (rad)."""

    def __init__(self, cfg: MatchingParams, w0: float, T: float, v0_pu: float = 1.0,
                 vdc_ref_pu: float = 1.0) -> None:
        self.k_theta = cfg.k_theta_pu if cfg.k_theta_pu is not None else w0 / vdc_ref_pu
        self.k_q = cfg.k_q_pu
        self.w0, self.T = w0, T
        self.theta = 0.0
        self.omega = w0
        self.v_mag = v0_pu

    def update(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq=0j) -> SyncOutput:
        self.omega = self.k_theta * v_dc_pu
        self.theta += T * self.omega
        self.v_mag = v_ref_pu + self.k_q * (q_ref_pu - q_pu)
        return SyncOutput(self.theta, self.omega, self.v_mag)

    def _state_names(self) -> tuple[str, ...]:
        return ("theta",)

    def get_state(self) -> dict[str, Any]:
        return {n: getattr(self, n) for n in self._state_names()}

    def set_state(self, values: Mapping[str, Any]) -> None:
        assign(self, values, self._state_names())


def make_law(cfg: GFMParams, w0: float, T: float, vdc_ref_pu: float):
    if hasattr(cfg, "type"):
        classes = {"psc": PSC, "droop": Droop, "vsg": VSG, "dvoc": DVOC, "matching": Matching}
        cls = classes[cfg.type]
        if cfg.type == "matching":
            return cls(cfg, w0, T, 1.0, vdc_ref_pu)
        return cls(cfg, w0, T)
    if cfg.law == "psc":
        return PSC(cfg.psc, w0, T, cfg.v_ref_pu)
    if cfg.law == "droop":
        return Droop(cfg.droop, w0, T, cfg.v_ref_pu)
    if cfg.law == "vsg":
        return VSG(cfg.vsg, w0, T, cfg.v_ref_pu)
    if cfg.law == "dvoc":
        return DVOC(cfg.dvoc, w0, T, cfg.v_ref_pu)
    if cfg.law == "matching":
        return Matching(cfg.matching, w0, T, cfg.v_ref_pu, vdc_ref_pu)
    raise ValueError(f"unknown GFM law {cfg.law!r}")
