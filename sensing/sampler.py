"""ADC sampling of a converter's measurement ports, instantaneous or window-averaged (SI values)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..assembly.protocols import Measurement
from ..firmware.transforms import complex2abc
from .window import SamplingWindow

__all__ = ["MeasurementPorts", "Sampler"]


@dataclass(frozen=True)
class MeasurementPorts:
    """Zero-argument callables that read one converter's quantities from the model (SI).

    ``u_g`` terminal voltage, ``i_c`` converter current, ``u_dc`` dc voltage;
    ``i_c_state`` reads the current from the state vector; ``i_dc`` is recorded only.
    """

    u_g: Callable[[], complex]
    i_c: Callable[[], complex]
    i_c_state: Callable[[], complex]
    u_dc: Callable[[], float]
    i_dc: Optional[Callable[[], float]] = None


class Sampler:
    """Stateless sampler of the measurement ports.

    ``window``: a :class:`~peslite.sensing.SamplingWindow` whose channels (``"v"``,
    ``"i"``, ``"dc"``) are averaged, or ``None`` for instantaneous sampling.
    """

    def __init__(self, ports: MeasurementPorts, window: Optional[SamplingWindow] = None) -> None:
        self.ports = ports
        self.window = window

    @property
    def averaging(self) -> bool:
        return self.window is not None

    def open_window(self) -> None:
        """Mark this instant as the start of the next window (no-op without a window)."""
        if self.window is not None:
            self.window.open()

    def measure(self, t: float) -> Measurement:
        """Return the :class:`Measurement` at ``t``; model outputs must be up to date at ``t``.

        ``u_g``, ``i_c``, ``u_dc`` are window means for averaged channels;
        ``u_g_raw``, ``i_c_raw``, ``u_dc_raw`` and ``i_abc`` are instantaneous.
        """
        ports = self.ports
        u_raw, i_raw, dc_raw = ports.u_g(), ports.i_c(), ports.u_dc()
        mean = self.window.mean() if self.window is not None else {}
        return Measurement(t=t,
                           u_g=mean.get("v", u_raw), i_c=mean.get("i", i_raw),
                           u_dc=mean.get("dc", dc_raw), i_abc=complex2abc(i_raw),
                           u_g_raw=u_raw, i_c_raw=i_raw, u_dc_raw=dc_raw)

    def phase_currents(self) -> np.ndarray:
        """Return the instantaneous phase currents read from the state vector."""
        return complex2abc(self.ports.i_c_state())
