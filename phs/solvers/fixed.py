"""Explicit fixed-step Runge-Kutta solver (Euler, Heun, RK4) that lands exactly on the interval end.

RK4 runs on Python floats when ``f``'s object offers ``rhs_list(t, v)``, with the same result as on
arrays; on a non-finite value the interval is recomputed on arrays.
"""

from __future__ import annotations

import functools
import linecache
import math

import numpy as np
from numpy.typing import NDArray

from ..protocols import RHS, SolverStep

__all__ = ["FixedStepSolver"]


class FixedStepSolver:
    """Explicit RK with a maximum step ``dt`` (s); sub-steps are equal within an interval.

    ``method``: ``"euler"``, ``"heun"`` or ``"rk4"``.
    """

    def __init__(self, dt: float, method: str = "rk4") -> None:
        if method not in ("euler", "heun", "rk4"):
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
        y = y0
        t = t0
        method = self.method
        if method == "rk4":
            lists = getattr(getattr(f, "__self__", None), "rhs_list", None)
            if lists is not None:
                v = _rk4_on_floats(len(y0))(lists, t0, y0.tolist(), n, h)
                if v is not None:
                    self.n_rhs += 4 * n
                    return SolverStep(t1, np.array(v), self.n_rhs)
            h2 = 0.5 * h
            h6 = h / 6.0
            for _ in range(n):
                k1 = f(t, y)
                k2 = f(t + h2, y + h2 * k1)
                k3 = f(t + h2, y + h2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h6 * (k1 + 2.0 * (k2 + k3) + k4)
                t += h
            self.n_rhs += 4 * n
        elif method == "heun":
            for _ in range(n):
                k1 = f(t, y)
                k2 = f(t + h, y + h * k1)
                y = y + 0.5 * h * (k1 + k2)
                t += h
            self.n_rhs += 2 * n
        else:
            for _ in range(n):
                y = y + h * f(t, y)
                t += h
            self.n_rhs += n
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
