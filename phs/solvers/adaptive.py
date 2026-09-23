"""Adaptive solvers: a SciPy ``solve_ivp`` wrapper and a built-in Dormand-Prince RK5(4)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..protocols import RHS, SolverStep

__all__ = ["AdaptiveSolver", "DormandPrince45"]


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
