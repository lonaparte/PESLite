"""Compensated two-port magnetic links for wireless power transfer.

The coupled-inductor matrix is diagonalized once during construction.  The ODE therefore stores
two modal currents and keeps the existing diagonal :class:`StoragePort` energy representation;
there is no per-step matrix inversion and no special solver path.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..solver.energy import PowerPort, StoragePort
from ..solver.model import Bag, OutputMethod, _real
from .network import Terminal

__all__ = ["SSLink", "LCCLink", "make_wpt_link"]


class _LinkInp(Bag):
    __slots__ = ("u_primary", "u_secondary")


class _LinkOut(Bag):
    __slots__ = ("i_primary", "i_secondary", "i_primary_coil", "i_secondary_coil")


class _SSState(Bag):
    __slots__ = ("i_mode_1", "i_mode_2", "u_primary_cap", "u_secondary_cap")


class _LCCState(Bag):
    __slots__ = (
        "i_primary_comp", "i_secondary_comp",
        "u_primary_parallel", "u_secondary_parallel",
        "i_mode_1", "i_mode_2",
        "u_primary_series", "u_secondary_series",
    )


class _CoupledLink:
    """Shared modal-current transformation for a passive reciprocal magnetic link."""

    outputs_need_inputs: ClassVar[bool] = False
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod(
            "set_outputs",
            inputs=(),
            outputs=("i_primary", "i_secondary", "i_primary_coil", "i_secondary_coil"),
        ),
    )
    ports: ClassVar[tuple[PowerPort, ...]] = (
        PowerPort("inp.u_primary", "out.i_primary"),
        PowerPort("inp.u_secondary", "out.i_secondary"),
    )

    def __init__(self, primary_inductance: float, secondary_inductance: float,
                 mutual_inductance: float) -> None:
        # Closed-form orthogonal diagonalization of this two-winding matrix. Scalar arithmetic
        # keeps the same initialization available to parameterized standalone backends.
        angle = 0.5 * np.arctan2(2.0 * mutual_inductance,
                               primary_inductance - secondary_inductance)
        cosine, sine = np.cos(angle), np.sin(angle)
        mean = 0.5 * (primary_inductance + secondary_inductance)
        radius = np.sqrt((0.5 * (primary_inductance - secondary_inductance)) ** 2
                         + mutual_inductance ** 2)
        first_sign = np.where(abs(sine) >= abs(cosine),
                              np.where(sine > 0.0, -1.0, 1.0), 1.0)
        second_sign = np.where(abs(sine) > abs(cosine),
                               np.where(sine < 0.0, -1.0, 1.0), 1.0)
        self._inductance = (_real(mean - radius), _real(mean + radius))
        self._mode_primary = (_real(-sine * first_sign), _real(cosine * second_sign))
        self._mode_secondary = (_real(cosine * first_sign), _real(sine * second_sign))
        self.inp = _LinkInp()
        self.out = _LinkOut()
        self.primary_terminal = Terminal(self, "u_primary", "i_primary", 1)
        self.secondary_terminal = Terminal(self, "u_secondary", "i_secondary", 1)

    def _coil_currents(self) -> tuple[float, float]:
        i1, i2 = self.state.i_mode_1, self.state.i_mode_2
        p1, p2 = self._mode_primary
        s1, s2 = self._mode_secondary
        return p1 * i1 + p2 * i2, s1 * i1 + s2 * i2

    def _modal_derivatives(self, primary_voltage: float,
                           secondary_voltage: float) -> tuple[float, float]:
        p1, p2 = self._mode_primary
        s1, s2 = self._mode_secondary
        l1, l2 = self._inductance
        return ((p1 * primary_voltage + s1 * secondary_voltage) / l1,
                (p2 * primary_voltage + s2 * secondary_voltage) / l2)

    def supplied_power(self) -> float:
        return 0.0


class SSLink(_CoupledLink):
    """Series-series compensated coupled-inductor link."""

    state_names = ("i_mode_1", "i_mode_2", "u_primary_cap", "u_secondary_cap")

    def __init__(self, cfg: Any) -> None:
        super().__init__(
            cfg.primary_inductance,
            cfg.secondary_inductance,
            cfg.mutual_inductance,
        )
        self.primary_capacitance = cfg.primary_capacitance
        self.secondary_capacitance = cfg.secondary_capacitance
        self.primary_resistance = cfg.primary_resistance
        self.secondary_resistance = cfg.secondary_resistance
        self.state = _SSState()
        self.storage = (
            StoragePort("i_mode_1", self._inductance[0], kind="inductor"),
            StoragePort("i_mode_2", self._inductance[1], kind="inductor"),
            StoragePort("u_primary_cap", self.primary_capacitance, kind="capacitor"),
            StoragePort("u_secondary_cap", self.secondary_capacitance, kind="capacitor"),
        )

    def set_outputs(self, t: float) -> None:
        primary, secondary = self._coil_currents()
        self.out.i_primary = self.out.i_primary_coil = primary
        self.out.i_secondary = self.out.i_secondary_coil = secondary

    def rhs(self, t: float):
        primary, secondary = self._coil_currents()
        d_mode = self._modal_derivatives(
            self.inp.u_primary - self.primary_resistance * primary
            - self.state.u_primary_cap,
            self.inp.u_secondary - self.secondary_resistance * secondary
            - self.state.u_secondary_cap,
        )
        return (
            *d_mode,
            primary / self.primary_capacitance,
            secondary / self.secondary_capacitance,
        )

    def dissipated_power(self) -> float:
        primary, secondary = self._coil_currents()
        return (self.primary_resistance * primary * primary
                + self.secondary_resistance * secondary * secondary)


class LCCLink(_CoupledLink):
    """LCC-LCC compensated coupled-inductor link."""

    state_names = (
        "i_primary_comp", "i_secondary_comp",
        "u_primary_parallel", "u_secondary_parallel",
        "i_mode_1", "i_mode_2",
        "u_primary_series", "u_secondary_series",
    )

    def __init__(self, cfg: Any) -> None:
        super().__init__(
            cfg.primary_inductance,
            cfg.secondary_inductance,
            cfg.mutual_inductance,
        )
        self.primary_series_capacitance = cfg.primary_series_capacitance
        self.secondary_series_capacitance = cfg.secondary_series_capacitance
        self.primary_compensation_inductance = cfg.primary_compensation_inductance
        self.secondary_compensation_inductance = cfg.secondary_compensation_inductance
        self.primary_parallel_capacitance = cfg.primary_parallel_capacitance
        self.secondary_parallel_capacitance = cfg.secondary_parallel_capacitance
        self.primary_resistance = cfg.primary_resistance
        self.secondary_resistance = cfg.secondary_resistance
        self.primary_compensation_resistance = cfg.primary_compensation_resistance
        self.secondary_compensation_resistance = cfg.secondary_compensation_resistance
        self.state = _LCCState()
        self.storage = (
            StoragePort("i_primary_comp", self.primary_compensation_inductance, kind="inductor"),
            StoragePort("i_secondary_comp", self.secondary_compensation_inductance, kind="inductor"),
            StoragePort("u_primary_parallel", self.primary_parallel_capacitance, kind="capacitor"),
            StoragePort("u_secondary_parallel", self.secondary_parallel_capacitance, kind="capacitor"),
            StoragePort("i_mode_1", self._inductance[0], kind="inductor"),
            StoragePort("i_mode_2", self._inductance[1], kind="inductor"),
            StoragePort("u_primary_series", self.primary_series_capacitance, kind="capacitor"),
            StoragePort("u_secondary_series", self.secondary_series_capacitance, kind="capacitor"),
        )

    def set_outputs(self, t: float) -> None:
        primary_coil, secondary_coil = self._coil_currents()
        self.out.i_primary = self.state.i_primary_comp
        self.out.i_secondary = self.state.i_secondary_comp
        self.out.i_primary_coil = primary_coil
        self.out.i_secondary_coil = secondary_coil

    def rhs(self, t: float):
        state = self.state
        primary_coil, secondary_coil = self._coil_currents()
        d_primary_comp = (
            self.inp.u_primary - state.u_primary_parallel
            - self.primary_compensation_resistance * state.i_primary_comp
        ) / self.primary_compensation_inductance
        d_secondary_comp = (
            self.inp.u_secondary - state.u_secondary_parallel
            - self.secondary_compensation_resistance * state.i_secondary_comp
        ) / self.secondary_compensation_inductance
        d_mode = self._modal_derivatives(
            state.u_primary_parallel - state.u_primary_series
            - self.primary_resistance * primary_coil,
            state.u_secondary_parallel - state.u_secondary_series
            - self.secondary_resistance * secondary_coil,
        )
        return (
            d_primary_comp,
            d_secondary_comp,
            (state.i_primary_comp - primary_coil) / self.primary_parallel_capacitance,
            (state.i_secondary_comp - secondary_coil) / self.secondary_parallel_capacitance,
            *d_mode,
            primary_coil / self.primary_series_capacitance,
            secondary_coil / self.secondary_series_capacitance,
        )

    def dissipated_power(self) -> float:
        primary_coil, secondary_coil = self._coil_currents()
        return (
            self.primary_compensation_resistance * self.state.i_primary_comp ** 2
            + self.secondary_compensation_resistance * self.state.i_secondary_comp ** 2
            + self.primary_resistance * primary_coil ** 2
            + self.secondary_resistance * secondary_coil ** 2
        )


def make_wpt_link(cfg: Any) -> SSLink | LCCLink:
    """Build the configured compensated magnetic link."""
    return SSLink(cfg) if cfg.type == "ss" else LCCLink(cfg)
