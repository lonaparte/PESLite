"""Uniform limiting of three-phase modulating signals.

Custom limiters use the same ``limiter(m_abc) -> m_abc`` interface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .blocks import peak_abs

__all__ = ["ModulationLimiter"]


class ModulationLimiter:
    """Return a copy of ``m_abc`` scaled uniformly so that ``max(abs(m_abc)) <= limit``."""

    def __init__(self, limit: float) -> None:
        self.limit = limit

    def __call__(self, m_abc: NDArray[np.float64]) -> NDArray[np.float64]:
        m = np.array(m_abc, dtype=float, copy=True)
        max_abs = peak_abs(m)
        if max_abs > self.limit:
            m *= self.limit / max_abs
        return m
