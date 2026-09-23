"""Active and reactive power feedback of a grid-forming law."""

from __future__ import annotations

from typing import Any, Mapping

from ..firmware.blocks import LowPass1
from ..phs.states import gather, scatter

__all__ = ["PowerCalculator"]


class PowerCalculator:
    """Compute ``p + j q = v conj(i)`` (pu inputs) and low-pass filter it.

    ``bw_hz``: filter bandwidth (Hz); ``<= 0`` disables the filter.
    Named states: filtered ``p`` and ``q`` (pu).
    """

    def __init__(self, bw_hz: float, T: float) -> None:
        self.lpf_p = LowPass1(bw_hz, T, 0.0, init_on_first=False)
        self.lpf_q = LowPass1(bw_hz, T, 0.0, init_on_first=False)
        self.p = 0.0
        self.q = 0.0

    def update(self, v_dq: complex, i_dq: complex) -> tuple[float, float]:
        s = v_dq * i_dq.conjugate()
        self.p = self.lpf_p.update(s.real)
        self.q = self.lpf_q.update(s.imag)
        return self.p, self.q

    def get_state(self) -> dict[str, Any]:
        return gather({"p": self.lpf_p, "q": self.lpf_q})

    def set_state(self, values: Mapping[str, Any]) -> None:
        scatter({"p": self.lpf_p, "q": self.lpf_q}, values)
