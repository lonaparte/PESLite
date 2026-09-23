"""Protocols and data classes shared by sensing, control, firmware, modulation and assembly.

Signal chain: Controller -> Delay -> Modulator -> solver. Model, energy and state
protocols are in :mod:`peslite.phs.protocols`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["SystemLike", "UnitLike", "Controller", "ControlOutput", "Delay", "Modulator", "SwitchingSequence",
           "SynchronizationLaw", "SyncOutput", "Measurement", "ControlMeasurement", "PWMMethod", "SignalLimiter"]

@dataclass
class Measurement:
    """SI sample at ``t``: peak-scaled alpha-beta AC vectors (V, A) and DC voltage (V).

    Windowed channels hold means over ``(t - T_avg, t]``; ``*_raw`` and ``i_abc`` are instantaneous.
    ``samples`` holds an oversampled update's samples, oldest first, this one last; otherwise empty.
    """

    t: float
    u_g: complex  # PCC voltage (V)
    i_c: complex  # converter current (A), positive out of the bridge
    u_dc: float  # dc-link voltage (V)
    i_abc: NDArray[np.float64]  # instantaneous phase currents (A)
    u_g_raw: Optional[complex] = None  # before window averaging
    i_c_raw: Optional[complex] = None
    u_dc_raw: Optional[float] = None
    extras: dict[str, float] = field(default_factory=dict)
    samples: tuple["Measurement", ...] = ()


@dataclass
class ControlMeasurement:
    """Controller-side sample in pu: alpha-beta AC vectors on peak phase bases, DC voltage on ``vdc_ref``.

    ``t`` is in s.
    """

    t: float
    u_g: complex
    i_c: complex
    u_dc: float
    i_abc: NDArray[np.float64]

@runtime_checkable
class UnitLike(Protocol):
    """Converter unit as used by the simulation loop."""

    name: str
    zoh: str  # model label of the held switching state

    def measure(self, t: float) -> "Measurement": ...

    def phase_currents(self) -> NDArray[np.float64]: ...

    def trip(self) -> None: ...


@runtime_checkable
class SystemLike(Protocol):
    """System as used by the simulation loop: one model and its converter units."""

    model: Any  # a peslite.phs.model.Model
    units: Any  # a mapping of name to UnitLike

    def signals(self, t: float) -> dict[str, Any]: ...

    def trip(self) -> None: ...


# ------------------------------------------------------------------------ control

class PWMMethod(Protocol):
    """Map an alpha-beta voltage command (V) and a positive dc voltage (V) to modulating signals ``m_abc``.

    Must be deterministic and return three finite real values, shape ``(3,)``, possibly outside [-1, 1].
    """

    def __call__(self, u_ab: complex, u_dc: float) -> NDArray[np.float64]: ...


class SignalLimiter(Protocol):
    """Map three modulating signals to three finite limited signals, shape ``(3,)``.

    Must be deterministic; it may be called on tentative commands and keeps no saturation state.
    """

    def __call__(self, m_abc: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass
class ControlOutput:
    """Result of one controller call.

    ``theta`` (rad) and ``omega`` (rad/s): the controller's synchronization angle and frequency,
    required by synchronous modulators; ``None`` allows only asynchronous modulation.
    """

    d_abc: NDArray[np.float64]  # duty ratios of phases a, b, c in [0, 1]
    tripped: bool = False
    log: dict[str, float] | None = None
    theta: float | None = None  # synchronization angle (rad)
    omega: float | None = None  # synchronization frequency (rad/s)

@runtime_checkable
class Controller(Protocol):
    """Sampled controller ``(t, Measurement) -> ControlOutput``, called once per PWM publication.

    Optional: ``initial_sync() -> (theta, omega)`` seeds a synchronous carrier for the first period;
    ``next_event``, ``periods``, ``reset_clocks(t)`` and ``update(t, meas)`` for independently clocked loops.
    """

    T_s: float  # PWM publication interval (s), pwm.update_period

    def __call__(self, t: float, meas: Measurement) -> ControlOutput: ...

    def initial_duty(self) -> NDArray[np.float64]:
        """Duty ratios applied before the first control update."""

@runtime_checkable
class Delay(Protocol):
    """An N-sample delay line for the duty ratios (computation delay)."""

    n_samples: int

    def __call__(self, d_abc: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def reset(self, d_abc: NDArray[np.float64]) -> None: ...


# --------------------------------------------------------------------- modulation

@dataclass
class SwitchingSequence:
    """Piecewise-constant switching pattern over one PWM publication interval.

    ``dt``: interval lengths (s) summing to the publication interval.
    ``q_abc[i]``: phase states in interval ``i`` (0/1 when switching, fractional when averaged).
    """

    dt: NDArray[np.float64]
    q_abc: NDArray[np.float64]

@runtime_checkable
class Modulator(Protocol):
    """Convert duty ratios into a SwitchingSequence spanning ``T_c`` (s), the PWM publication interval.

    Optional ``theta`` (rad) and ``omega`` (rad/s) set synchronous carrier timing; asynchronous modulators ignore them.
    """

    def __call__(self, t: float, T_c: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence: ...


# ---------------------------------------------------------------- synchronization

@dataclass
class SyncOutput:
    """Angle (rad), angular frequency (rad/s) and voltage-magnitude reference (pu) of a GFM law."""

    theta: float
    omega: float
    v_mag: float
    log: dict[str, float] | None = None

@runtime_checkable
class SynchronizationLaw(Protocol):
    """Grid-forming synchronization law; ``update`` advances it by one sampling period ``T`` (s).

    Inputs in pu: filtered ``p``/``q`` (export positive), PCC voltage magnitude (ac base),
    dc voltage (dc base), setpoints and converter current ``i_dq`` in the law's frame.
    """

    def update(
        self,
        T: float,
        p_pu: float,
        q_pu: float,
        v_mag_pu: float,
        v_dc_pu: float,
        p_ref_pu: float,
        q_ref_pu: float,
        v_ref_pu: float,
        i_dq: complex = 0j,
    ) -> SyncOutput: ...
