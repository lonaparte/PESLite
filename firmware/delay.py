"""N-sample computation delay between the controller and the modulator."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

__all__ = ["ComputationDelay"]


class ComputationDelay:
    """Delay duty ratios by ``n_samples`` periods: ``d_applied[k] = d_ref[k - n_samples]``.

    :meth:`reset` fills the pipeline with initial duty ratios; ``n_samples = 0`` passes through.
    Named states: ``"<j>.d_a"``, ``"<j>.d_b"``, ``"<j>.d_c"``; ``j = 0`` is applied next.
    """

    def __init__(self, n_samples: int = 0) -> None:
        if n_samples < 0:
            raise ValueError("n_samples must be >= 0")
        self.n_samples = int(n_samples)
        self._buf: deque[NDArray[np.float64]] = deque()

    def reset(self, d_abc: NDArray[np.float64]) -> None:
        self._buf = deque(np.array(d_abc, dtype=float, copy=True) for _ in range(self.n_samples))

    def __call__(self, d_abc: NDArray[np.float64]) -> NDArray[np.float64]:
        d = np.asarray(d_abc, dtype=float)
        if self.n_samples == 0:
            return d
        if len(self._buf) != self.n_samples:
            self.reset(d)
        self._buf.append(d)
        return self._buf.popleft()

    def get_state(self) -> dict[str, Any]:
        return {f"{j}.d_{ph}": float(d[k]) for j, d in enumerate(self._buf) for k, ph in enumerate("abc")}

    def set_state(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            j_str, _, name = key.partition(".")
            if not j_str.isdigit() or name not in ("d_a", "d_b", "d_c") or int(j_str) >= len(self._buf):
                raise KeyError(f"computation delay: no state {key!r} (pipeline length {len(self._buf)})")
            self._buf[int(j_str)]["abc".index(name[-1])] = float(value)
