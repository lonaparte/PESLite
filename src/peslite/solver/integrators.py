"""The solver interface ``solver(f, t0, t1, y0) -> SolverStep`` and the single-rate integrators.

Fixed-step explicit Runge-Kutta (Euler, Heun, RK4) landing exactly on the interval end, SciPy's
adaptive methods and a built-in Dormand-Prince RK5(4). RK4 runs on Python floats when ``f``'s
object offers ``rhs_list(t, v)``, with the same result as on arrays; on a non-finite value the
interval is recomputed on arrays.
"""

from __future__ import annotations

import functools
import linecache
import math
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["RHS", "SolverStep", "Solver", "FIXED_METHODS", "ADAPTIVE_METHODS", "TABLEAUS", "combine", "integrator",
           "FixedStepSolver", "AdaptiveSolver", "DormandPrince45"]

RHS = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class SolverStep:
    """Result of integrating one interval; ``y`` is the state at ``t``."""

    t: float
    y: NDArray[np.float64]
    n_rhs: int = 0


@runtime_checkable
class Solver(Protocol):
    """Integrate ``y' = f(t, y)`` from ``t0`` to ``t1`` and return ``SolverStep``.

    A solver which defers part of the integration may additionally provide
    ``settle(t, y) -> y``. The simulation calls it immediately before the model
    changes at an event. A solver which caches model parameters may provide
    ``parameters_changed()``; it is called after ``set`` events. Both hooks are
    optional and are discovered with :func:`getattr`.
    """

    def __call__(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64]) -> SolverStep: ...


FIXED_METHODS = ("euler", "heun", "rk4")
ADAPTIVE_METHODS = ("RK45", "DOP853", "Radau", "BDF", "LSODA", "RK23", "DP45")

# explicit Runge-Kutta tableaus: (c, a_rows, b); a_rows[j - 1] holds the coefficients of stage j
TABLEAUS: dict[str, tuple[tuple[float, ...], tuple[tuple[float, ...], ...], tuple[float, ...]]] = {
    "euler": ((0.0,), (), (1.0,)),
    "heun": ((0.0, 1.0), ((1.0,),), (0.5, 0.5)),
    "rk4": ((0.0, 0.5, 0.5, 1.0), ((0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)), (1 / 6, 1 / 3, 1 / 3, 1 / 6)),
}


def combine(method: str, y: NDArray[np.float64], h: float, ks: list) -> NDArray[np.float64]:
    """The final combination of an explicit RK step of ``method`` from its stage derivatives ``ks``."""
    if method == "rk4":
        return y + (h / 6.0) * (ks[0] + 2.0 * (ks[1] + ks[2]) + ks[3])
    if method == "heun":
        return y + 0.5 * h * (ks[0] + ks[1])
    return y + h * ks[0]


def stage_state(y: NDArray[np.float64], h: float, row: tuple[float, ...], ks: list) -> NDArray[np.float64]:
    """The state at which a stage is evaluated: ``y + h * sum(a_j k_j)`` over the nonzero coefficients."""
    acc = None
    for coef, k in zip(row, ks):
        if coef == 0.0:
            continue
        acc = coef * h * k if acc is None else acc + coef * h * k
    return y + acc


def integrator(method: str, dt: float, rtol: float = 1e-6, atol: float = 1e-9,
               max_step: float = math.inf, warm_start: bool = True) -> Solver:
    """Return the single-rate integrator of ``method``: fixed-step with maximum step ``dt`` (s),
    the built-in ``"DP45"`` or a SciPy method."""
    if method in FIXED_METHODS:
        return FixedStepSolver(dt, method)
    if method == "DP45":
        return DormandPrince45(rtol, atol, max_step)
    return AdaptiveSolver(method, rtol, atol, max_step, warm_start)


class FixedStepSolver:
    """Explicit RK with a maximum step ``dt`` (s); sub-steps are equal within an interval.

    ``method``: ``"euler"``, ``"heun"`` or ``"rk4"``.
    """

    def __init__(self, dt: float, method: str = "rk4") -> None:
        if method not in TABLEAUS:
            raise ValueError(f"unknown fixed-step method {method!r}")
        self.dt = float(dt)
        self.method = method
        self.n_rhs = 0

    def __call__(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64]) -> SolverStep:
        span = t1 - t0
        if span <= 0.0:
            return SolverStep(t1, y0, 0)
        n = max(1, int(math.ceil(span / self.dt - 1e-9)))
        h = span / n
        c, a, _b = TABLEAUS[self.method]
        self.n_rhs += len(c) * n
        if self.method == "rk4":
            lists = getattr(getattr(f, "__self__", None), "rhs_list", None)
            if lists is not None:
                v = _rk4_on_floats(len(y0))(lists, t0, y0.tolist(), n, h)
                if v is not None:
                    return SolverStep(t1, np.array(v), self.n_rhs)
        y, t = y0, t0
        for _ in range(n):
            ks = []
            for j in range(len(c)):
                ks.append(f(t + c[j] * h, y if j == 0 else stage_state(y, h, a[j - 1], ks)))
            y = combine(self.method, y, h, ks)
            t += h
        return SolverStep(t1, y, self.n_rhs)


@functools.lru_cache(maxsize=None)
def _rk4_on_floats(size: int):
    """Build an RK4 stepper on ``size`` floats that returns ``None`` on a non-finite value."""
    ys = [f"y{i}" for i in range(size)]
    y = "[" + ", ".join(ys) + "]"

    def stage(k: str) -> str:  # stage derivative names: [a0, a1, ...]
        return "[" + ", ".join(f"{k}{i}" for i in range(size)) + "]"

    def args(k: str, h: str) -> str:  # stage state: [y0 + h * a0, ...]
        return "[" + ", ".join(f"y{i} + {h} * {k}{i}" for i in range(size)) + "]"

    lines = ["def rk4(g, t, y, steps, h):",
             "    h2 = 0.5 * h",
             "    h6 = h / 6.0",
             f"    {y} = y",
             "    for _ in range(steps):",
             f"        {stage('a')} = g(t, {y})",
             f"        x = {args('a', 'h2')}",
             "        if not isfinite(sum(x)):",
             "            return None",
             f"        {stage('b')} = g(t + h2, x)",
             f"        x = {args('b', 'h2')}",
             "        if not isfinite(sum(x)):",
             "            return None",
             f"        {stage('c')} = g(t + h2, x)",
             f"        x = {args('c', 'h')}",
             "        if not isfinite(sum(x)):",
             "            return None",
             f"        {stage('d')} = g(t + h, x)"]
    lines += [f"        y{i} = y{i} + h6 * ((a{i} + 2.0 * (b{i} + c{i})) + d{i})" for i in range(size)]
    lines += [f"        if not isfinite({' + '.join(['0.0'] + ys)}):",
              "            return None",
              "        t += h",
              f"    return {y}"]
    source, filename = "\n".join(lines) + "\n", f"<peslite rk4, {size} floats>"
    env = {"isfinite": math.isfinite}
    exec(compile(source, filename, "exec"), env)
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    return env["rk4"]


class AdaptiveSolver:
    """SciPy ``solve_ivp`` behind the common signature, warm-started across intervals."""

    def __init__(
        self,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_step: float = math.inf,
        warm_start: bool = True,
    ) -> None:
        from scipy.integrate import solve_ivp  # local import keeps SciPy optional

        self._solve_ivp = solve_ivp
        self.method, self.rtol, self.atol, self.max_step = method, rtol, atol, max_step
        self.warm_start = warm_start
        self._h_last: float | None = None
        self.n_rhs = 0

    def __call__(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64]) -> SolverStep:
        span = t1 - t0
        if span <= 0.0:
            return SolverStep(t1, y0, 0)
        kwargs = dict(method=self.method, rtol=self.rtol, atol=self.atol)
        if math.isfinite(self.max_step):
            kwargs["max_step"] = self.max_step
        if self.warm_start and self._h_last is not None:
            kwargs["first_step"] = min(self._h_last, span)
        sol = self._solve_ivp(f, (t0, t1), y0, **kwargs)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed on [{t0}, {t1}]: {sol.message}")
        if sol.t.size >= 2:
            self._h_last = float(sol.t[-1] - sol.t[-2])
        self.n_rhs += int(sol.nfev)
        return SolverStep(t1, sol.y[:, -1].copy(), self.n_rhs)


class DormandPrince45:
    """Embedded Dormand-Prince RK5(4) with PI step control; the step size carries over between calls.

    A step is accepted when ``max_i |e_i| / (atol + rtol * max(|y0_i|, |y1_i|)) <= 1``.
    """

    _c = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
    _a = (
        (),
        (1 / 5,),
        (3 / 40, 9 / 40),
        (44 / 45, -56 / 15, 32 / 9),
        (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
        (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
        (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
    )
    _b = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
    _e = (71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40)

    def __init__(self, rtol: float = 1e-6, atol: float = 1e-9, max_step: float = math.inf,
                 h_init: float | None = None, safety: float = 0.9) -> None:
        self.rtol, self.atol, self.max_step, self.safety = rtol, atol, max_step, safety
        self._h = h_init
        self._err_prev = 1.0
        self.n_rhs = 0
        self.n_rejected = 0

    def __call__(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64]) -> SolverStep:
        span = t1 - t0
        if span <= 0.0:
            return SolverStep(t1, y0, 0)
        lists = getattr(getattr(f, "__self__", None), "rhs_list", None)
        if lists is not None:
            return self._call_lists(lists, t0, t1, y0)
        a, b, c, e = self._a, self._b, self._c, self._e
        t, y = t0, np.asarray(y0, dtype=float)
        h = self._h if self._h is not None else span
        h = min(h, span, self.max_step)
        k1 = f(t, y)
        self.n_rhs += 1
        while t < t1 - 1e-15 * max(1.0, abs(t1)):
            if t + h > t1:
                h = t1 - t
            k2 = f(t + c[1] * h, y + h * (a[1][0] * k1))
            k3 = f(t + c[2] * h, y + h * (a[2][0] * k1 + a[2][1] * k2))
            k4 = f(t + c[3] * h, y + h * (a[3][0] * k1 + a[3][1] * k2 + a[3][2] * k3))
            k5 = f(t + c[4] * h, y + h * (a[4][0] * k1 + a[4][1] * k2 + a[4][2] * k3 + a[4][3] * k4))
            k6 = f(t + h, y + h * (a[5][0] * k1 + a[5][1] * k2 + a[5][2] * k3 + a[5][3] * k4 + a[5][4] * k5))
            y1 = y + h * (b[0] * k1 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6)
            k7 = f(t + h, y1)
            self.n_rhs += 6
            err_vec = h * (e[0] * k1 + e[2] * k3 + e[3] * k4 + e[4] * k5 + e[5] * k6 + e[6] * k7)
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y1))
            err = float(np.max(np.abs(err_vec) / scale))
            if err <= 1.0 or h <= 1e-15:
                t += h
                y, k1 = y1, k7
                # PI step-size controller
                if err == 0.0:
                    factor = 5.0
                else:
                    factor = self.safety * err ** -0.14 * self._err_prev ** 0.08
                    factor = min(5.0, max(0.2, factor))
                self._err_prev = max(err, 1e-4)
                h = min(h * factor, self.max_step)
            else:
                self.n_rejected += 1
                h = h * max(0.1, self.safety * err ** -0.25)
        self._h = h
        return SolverStep(t1, y, self.n_rhs)

    def _call_lists(self, f, t0: float, t1: float,
                    y0: NDArray[np.float64]) -> SolverStep:
        """The same DP5(4) pair on Python floats when the model provides ``rhs_list``."""
        span = t1 - t0
        a, b, c, e = self._a, self._b, self._c, self._e
        c2, c3, c4, c5 = c[1], c[2], c[3], c[4]
        a21 = a[1][0]
        a31, a32 = a[2]
        a41, a42, a43 = a[3]
        a51, a52, a53, a54 = a[4]
        a61, a62, a63, a64, a65 = a[5]
        b1, b3, b4, b5, b6 = b[0], b[2], b[3], b[4], b[5]
        e1, e3, e4, e5, e6, e7 = e[0], e[2], e[3], e[4], e[5], e[6]
        atol, rtol = self.atol, self.rtol
        t, y = t0, y0.tolist()
        h = self._h if self._h is not None else span
        h = min(h, span, self.max_step)
        k1 = f(t, y)
        self.n_rhs += 1
        end_eps = 1e-15 * max(1.0, abs(t1))
        while t < t1 - end_eps:
            if t + h > t1:
                h = t1 - t
            k2 = f(t + c2 * h,
                   [v + h * a21 * q1 for v, q1 in zip(y, k1)])
            k3 = f(t + c3 * h,
                   [v + h * (a31 * q1 + a32 * q2)
                    for v, q1, q2 in zip(y, k1, k2)])
            k4 = f(t + c4 * h,
                   [v + h * (a41 * q1 + a42 * q2 + a43 * q3)
                    for v, q1, q2, q3 in zip(y, k1, k2, k3)])
            k5 = f(t + c5 * h,
                   [v + h * (a51 * q1 + a52 * q2 + a53 * q3 + a54 * q4)
                    for v, q1, q2, q3, q4 in zip(y, k1, k2, k3, k4)])
            k6 = f(t + h,
                   [v + h * (a61 * q1 + a62 * q2 + a63 * q3 + a64 * q4 + a65 * q5)
                    for v, q1, q2, q3, q4, q5 in zip(y, k1, k2, k3, k4, k5)])
            y1 = [v + h * (b1 * q1 + b3 * q3 + b4 * q4 + b5 * q5 + b6 * q6)
                  for v, q1, q3, q4, q5, q6 in zip(y, k1, k3, k4, k5, k6)]
            k7 = f(t + h, y1)
            self.n_rhs += 6
            err = 0.0
            for v, v1, q1, q3, q4, q5, q6, q7 in zip(y, y1, k1, k3, k4, k5, k6, k7):
                estimate = abs(h * (e1 * q1 + e3 * q3 + e4 * q4
                                    + e5 * q5 + e6 * q6 + e7 * q7))
                ratio = estimate / (atol + rtol * max(abs(v), abs(v1)))
                if ratio > err:
                    err = ratio
            if err <= 1.0 or h <= 1e-15:
                t += h
                y, k1 = y1, k7
                if err == 0.0:
                    factor = 5.0
                else:
                    factor = self.safety * err ** -0.14 * self._err_prev ** 0.08
                    factor = min(5.0, max(0.2, factor))
                self._err_prev = max(err, 1e-4)
                h = min(h * factor, self.max_step)
            else:
                self.n_rejected += 1
                h = h * max(0.1, self.safety * err ** -0.25)
        self._h = h
        return SolverStep(t1, np.asarray(y), self.n_rhs)


