"""Post-run estimate of the multirate split error from a linearised one-period loop map.

Provides the :class:`LoopMap` protocol and :func:`split_error_bound`; the result is a
local linear estimate, not a guaranteed bound.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = ["LoopMap", "split_error_bound"]

_EPS = 1e-9
_FD = 1e-4  # relative size of the finite-difference step
_DECAY = 1e-5  # stop following the response below this fraction of its peak
_MIN_FOLLOW = 200  # minimum number of periods followed
_MAX_FOLLOW_S = 2.0  # maximum lag followed (s)


@runtime_checkable
class LoopMap(Protocol):
    """Application interface for linearising its sampled-data loop over one period.

    ``period``: sampling period (s).
    ``coordinates(names)``: list of ``(kind, names, scale)``; ``kind`` is ``"vector"`` (column pair
    rotated with the frame), ``"fixed"`` (pair already in the frame), ``"triple"``, ``"scalar"`` or
    ``"frozen"`` (not perturbed); ``scale`` is the finite-difference step floor.
    ``to_frame`` / ``from_frame``: convert a state row to and from the frame vector.
    ``advance(row, t, view)``: run one period; ``view = (state label, offset, "b" | "c")`` applies
    the offset to the integration (``"b"``) or the sampler (``"c"``).
    """

    period: float

    def coordinates(self, names: list[str]) -> list[tuple[str, list[str], float]]: ...

    def to_frame(self, row: dict[str, float], t: float, cols: list) -> np.ndarray: ...

    def from_frame(self, x: np.ndarray, t: float, cols: list, row: dict[str, float]) -> dict[str, float]: ...

    def advance(self, row: dict[str, float], t: float,
                view: tuple[str, float, str] | None = None) -> dict[str, float]: ...


def _width(kind: str) -> int:
    return 2 if kind in ("vector", "fixed") else 3 if kind == "triple" else 1


def _frame_index(cols: list, prefix: str) -> dict[str, list[int]]:
    """Rows of the frame vector per state label under ``prefix`` (a vector: its two rows)."""
    idx: dict[str, list[int]] = {}
    i = 0
    for kind, names, _s in cols:
        if names[0].startswith(prefix):
            lab = names[0][len(prefix):]
            if kind in ("vector", "fixed"):
                idx[lab[:-3]] = [i, i + 1]
            elif kind == "scalar":
                idx[lab] = [i]
        i += _width(kind)
    return idx


def _period_map(loop: LoopMap, row: dict[str, float], t0: float, cols: list,
                held: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``Phi``, ``B_b``, ``B_c`` at ``row`` by finite differences, plus the evaluation count."""
    t1 = t0 + loop.period
    base = loop.to_frame(loop.advance(row, t0), t1, cols)
    x0 = loop.to_frame(row, t0, cols)
    n = len(x0)
    phi = np.eye(n)
    evaluations = 1
    j = 0
    for kind, _names, floor in cols:
        if kind != "frozen":
            for w in range(_width(kind)):
                h = _FD * max(abs(float(x0[j + w])), floor)
                outs = []
                for sign in (1.0, -1.0):
                    xp = x0.copy()
                    xp[j + w] += sign * h
                    outs.append(loop.to_frame(loop.advance(loop.from_frame(xp, t0, cols, row), t0), t1, cols))
                    evaluations += 1
                phi[:, j + w] = (outs[0] - outs[1]) / (2.0 * h)
        j += _width(kind)
    b_b = np.zeros((n, len(held)))
    b_c = np.zeros((n, len(held)))
    for k, (label, scale) in enumerate(held):
        h = _FD * (scale or 1.0)
        b_b[:, k] = (loop.to_frame(loop.advance(row, t0, (label, h, "b")), t1, cols) - base) / h
        b_c[:, k] = (loop.to_frame(loop.advance(row, t0, (label, h, "c")), t1, cols) - base) / h
        evaluations += 2
    return phi, b_b, b_c, evaluations


def _block_norms(g: np.ndarray, outputs: list[tuple[str, list[int]]], inputs: list[list[int]]) -> np.ndarray:
    """Induced 2-norm of every (output, input) block: ``[lag, n, n_in] -> [lag, output, input]``."""
    out = np.zeros((g.shape[0],) + (len(outputs), len(inputs)))
    for a, (_lab, rows) in enumerate(outputs):
        for b, cols_ in enumerate(inputs):
            blk = g[:, rows][:, :, cols_]  # [lag, r, c]
            if blk.shape[1] == 2 and blk.shape[2] == 2:
                fro2 = np.sum(blk * blk, axis=(1, 2))
                det = blk[:, 0, 0] * blk[:, 1, 1] - blk[:, 0, 1] * blk[:, 1, 0]
                out[:, a, b] = np.sqrt(0.5 * (fro2 + np.sqrt(np.maximum(fro2 * fro2 - 4.0 * det * det, 0.0))))
            else:
                out[:, a, b] = np.sqrt(np.sum(blk * blk, axis=(1, 2)))
    return out


def _response(phi: np.ndarray, b_b: np.ndarray, b_c: np.ndarray, outputs: list[tuple[str, list[int]]],
              inputs: list[list[int]], n_max: int, T_s: float) -> tuple[np.ndarray, np.ndarray, bool]:
    """Block norms of ``Phi^m B_b`` and ``Phi^m B_c`` per lag, and whether the lag limit cut them off."""
    chunk = 256
    n_cap = int(_MAX_FOLLOW_S / T_s)
    gb, gc = b_b.copy(), b_c.copy()
    resp_b: list[np.ndarray] = []
    resp_c: list[np.ndarray] = []
    peak = 1e-300
    m = 0
    while True:
        raw_b, raw_c = [gb], [gc]
        while len(raw_b) < chunk and m + len(raw_b) <= n_max and m + len(raw_b) <= n_cap:
            gb, gc = phi @ gb, phi @ gc
            raw_b.append(gb)
            raw_c.append(gc)
        rb, rc = _block_norms(np.stack(raw_b), outputs, inputs), _block_norms(np.stack(raw_c), outputs, inputs)
        resp_b.append(rb)
        resp_c.append(rc)
        m += len(raw_b)
        peak = max(peak, float(rb.max()), float(rc.max()))
        last = max(float(rb[-1].max()), float(rc[-1].max()))
        if m > n_max or m > n_cap:
            return np.concatenate(resp_b), np.concatenate(resp_c), m > n_cap and n_cap < n_max
        if m >= _MIN_FOLLOW and last < _DECAY * peak:
            return np.concatenate(resp_b), np.concatenate(resp_c), False
        gb, gc = phi @ gb, phi @ gc


def _convolve(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    if len(a) * len(b) > 5e7:  # long inputs: FFT
        try:
            from scipy.signal import fftconvolve
            return fftconvolve(a, b)[:n]
        except ImportError:  # pragma: no cover
            pass
    return np.convolve(a, b)[:n]


# ------------------------------------------------------------------ the bound


def split_error_bound(solver: Any, loop: LoopMap, t: np.ndarray, states: dict[str, np.ndarray],
                      rows: int = 3, labels: Sequence[str] | None = None,
                      prefix: str = "") -> dict[str, Any]:
    """Estimate the split error of a finished multirate run.

    ``solver``: the :class:`~peslite.phs.solvers.MultirateSolver` that ran; ``loop``: the
    application's :class:`LoopMap`; ``t``: logged instants (s); ``states``: recorded state table
    including ``"t"``; ``rows``: number of rows to linearise at; ``labels``: the model's state
    labels; ``prefix``: prefix of those labels in the state table.
    Returns ``split_error_bound_rated`` (relative to rated effort) and ``split_bound`` (a summary);
    only ``split_bound`` when no estimate can be formed; ``{}`` without ratings.
    """
    log = solver.window_log
    rated = solver.rated_effort
    t_rows = np.asarray(states.get("t", ()), dtype=float)
    if not log or rated is None or rows < 1 or len(t) == 0 or len(t_rows) == 0:
        return {"split_bound": "no windows closed: nothing to bound"} if rated is not None else {}
    labels = list(labels if labels is not None else [])
    T_s, W = loop.period, solver.window
    n = len(labels)
    t = np.asarray(t, dtype=float)
    t_start, t_stop = float(t[0]), float(t[-1])
    n_periods = int(round((t_stop - t_start) / T_s)) + 1
    cuts = solver.held_storages
    if not cuts:
        return {"split_bound": "the windowed group holds no declared storage, so its cut carries no "
                               "accountable defect: no bound (see the structure report's 'unaccounted')"}
    # -- per-period defects of each cut: integration (b) and sampler (c)
    d_b = np.zeros((n_periods, len(cuts)))
    d_c = np.zeros((n_periods, len(cuts)))
    for t_close, change, err, ripple in log:
        t_open = t_close - W
        k0 = max(0, int(math.floor((t_open - t_start) / T_s + 1e-9)))
        k1 = min(n_periods - 1, int(math.ceil((t_close - t_start) / T_s - 1e-9)) - 1)
        if k1 < k0:
            continue
        for c, cut in enumerate(cuts):
            cols_ = list(cut.columns)
            amp_b = float(np.linalg.norm(err[cols_] + ripple[cols_]))
            d_b[k0:k1 + 1, c] = np.maximum(d_b[k0:k1 + 1, c], amp_b)
            for k in range(k0, k1 + 1):
                t_s = t_start + (k + 1) * T_s
                if t_open <= t_s < t_close - _EPS:
                    lag = float(np.linalg.norm((t_s - t_open) / W * change[cols_] + ripple[cols_]))
                    d_c[k, c] = max(d_c[k, c], lag)
    # -- recording lag of each held state at the logged instants
    direct = np.zeros(n)
    closes = np.array([tc for tc, *_ in log])
    for t_log in t:
        w = int(np.searchsorted(closes, t_log + _EPS))
        if w >= len(log):
            continue
        t_close, change, _err, ripple = log[w]
        t_open = t_close - W
        if t_open - _EPS <= t_log < t_close - _EPS:
            for cut in cuts:
                for j, entry in zip(cut.columns, range(cut.states.start, cut.states.stop)):
                    direct[entry] = max(direct[entry], (t_log - t_open) / W * change[j] + ripple[j])
    # -- rows to linearise at: on the sampling grid with every flag clear
    names = [k for k in states if k != "t"]
    cols = loop.coordinates(names)
    ok = np.abs(t_rows / T_s - np.round(t_rows / T_s)) < 1e-6
    for kind, flag_names, _s in cols:
        if kind == "frozen":
            ok &= np.abs(np.asarray(states[flag_names[0]])) < 0.5
    candidates = np.flatnonzero(ok)
    if len(candidates) == 0:
        return {"split_bound": "no recorded row on the sampling grid has every flag clear: "
                               "nothing to linearise at"}
    picks = sorted({int(candidates[min(len(candidates) - 1, int(round(f * (len(candidates) - 1))))])
                    for f in (k / rows for k in range(1, rows + 1))})
    # -- outputs (rated storages) and inputs (cut entries)
    findex = _frame_index(cols, prefix)
    outputs: list[tuple[str, list[int]]] = []
    for i in range(n):
        if rated[i] > 0 and not labels[i].endswith(".im"):
            lab = labels[i][:-3] if labels[i].endswith(".re") else labels[i]
            if lab in findex:
                outputs.append((lab, findex[lab]))
    if not outputs:
        return {"split_bound": "no rated storage of the model is a state of the loop map: no bound"}
    held = [(labels[j], float(rated[j])) for cut in cuts for j in range(cut.states.start, cut.states.stop)]
    inputs: list[list[int]] = []
    col = 0
    for cut in cuts:
        width = cut.states.stop - cut.states.start
        inputs.append(list(range(col, col + width)))
        col += width
    # -- linearised maps and the largest response per lag
    resp_b = resp_c = None
    evaluations = 0
    truncated = False
    for i_row in picks:
        row = {k: float(states[k][i_row]) for k in names}
        phi, b_b, b_c, n_eval = _period_map(loop, row, float(t_rows[i_row]), cols, held)
        evaluations += n_eval
        rb, rc, cut_off = _response(phi, b_b, b_c, outputs, inputs, n_periods, T_s)
        truncated |= cut_off
        if resp_b is None:
            resp_b, resp_c = rb, rc
        else:
            length = max(len(resp_b), len(rb))
            pad = lambda a: np.pad(a, ((0, length - len(a)), (0, 0), (0, 0)), mode="edge")  # noqa: E731
            resp_b, resp_c = np.maximum(pad(resp_b), pad(rb)), np.maximum(pad(resp_c), pad(rc))
    assert resp_b is not None and resp_c is not None
    # -- worst-case superposition per rated storage
    worst, worst_label = 0.0, ""
    for a, (lab, _r) in enumerate(outputs):
        i = labels.index(lab + ".re") if lab + ".re" in labels else labels.index(lab)
        total = np.zeros(n_periods)
        for c in range(len(cuts)):
            total += (_convolve(d_b[:, c], resp_b[:, a, c], n_periods)
                      + _convolve(d_c[:, c], resp_c[:, a, c], n_periods))
        amp = float(total.max()) + float(direct[i])
        if amp / rated[i] > worst:
            worst, worst_label = amp / rated[i], lab
    defect_b = max((d_b[:, c].max() / cut.rated for c, cut in enumerate(cuts) if cut.rated > 0.0), default=0.0)
    defect_c = max((d_c[:, c].max() / cut.rated for c, cut in enumerate(cuts) if cut.rated > 0.0), default=0.0)
    text = (f"linearised loop at t = {', '.join(f'{t_rows[i]:.4g}' for i in picks)} s "
            f"({evaluations} one-period evaluations): {len(log)} windows, defect up to {defect_b:.3g} of "
            f"rating for the integration and {defect_c:.3g} for the sampler; response followed "
            f"{(len(resp_b) - 1) * T_s:.3g} s" + (" (not died away there: cut off)" if truncated else "")
            + f"; worst state {worst_label}")
    return {"split_error_bound_rated": float(worst), "split_bound": text}
