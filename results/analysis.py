"""Post-processing helpers for recorded runs: reading, alignment and errors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

__all__ = ["read_csv", "align", "window_ptp", "pointwise_errors"]


def read_csv(path: str | Path, skip_header: int = 0) -> np.ndarray:
    """Read a CSV file with a header row into a structured array, keeping column names verbatim (dots included)."""
    return np.genfromtxt(path, delimiter=",", names=True, skip_header=skip_header, deletechars="")


def align(a: np.ndarray, b: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``a`` and ``b`` whose times fall on the same multiple of ``period``."""
    ia = np.round(a["t"] / period).astype(int)
    ib = np.round(b["t"] / period).astype(int)
    _common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
    return a[ka], b[kb]


def window_ptp(d: np.ndarray, t0: float, t1: float, key: str) -> float:
    """Peak-to-peak of ``d[key]`` on ``[t0, t1)`` (NaN with fewer than 11 samples)."""
    m = (d["t"] >= t0) & (d["t"] < t1)
    return float(np.ptp(d[key][m])) if m.sum() > 10 else float("nan")


def pointwise_errors(a: np.ndarray, b: np.ndarray, keys: Iterable[str],
                     scale: Optional[Mapping[str, float]] = None) -> dict[str, dict[str, float]]:
    """Max and RMS of ``|a[k] - b[k]| / scale[k]`` for aligned arrays, per key present in both."""
    scale = scale or {}
    out = {}
    for key in keys:
        if key not in a.dtype.names or key not in b.dtype.names:
            continue
        s = scale.get(key, 1.0)
        x, y = a[key] / s, b[key] / s
        e = np.abs(x - y)
        out[key] = {"max": float(e.max()), "rms": float(np.sqrt(np.mean(e * e))),
                    "t_max": float(a["t"][int(e.argmax())]), "rms_ref": float(np.sqrt(np.mean(y * y)))}
    return out
