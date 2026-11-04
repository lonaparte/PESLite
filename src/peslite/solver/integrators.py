"""The solver interface ``solver(f, t0, t1, y0) -> SolverStep`` and the single-rate integrators.

Fixed-step explicit Runge-Kutta (Euler, Heun, RK4) landing exactly on the interval end, SciPy's
adaptive methods and a built-in Dormand-Prince RK5(4); :mod:`.implicit` adds fixed-step implicit
methods. RK4 runs on Python floats when ``f``'s
object offers ``rhs_list(t, v)``, with the same result as on arrays; on a non-finite value the
interval is recomputed on arrays.
"""

from __future__ import annotations

import functools
import linecache
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .model import commutation_tolerance
from .netlist import FORMS as CIRCUIT_MODELS

__all__ = ["RHS", "SolverStep", "Solver", "FIXED_METHODS", "ADAPTIVE_METHODS", "EXPLICIT_FIXED_METHODS",
           "IMPLICIT_FIXED_METHODS", "IMPLICIT_ADAPTIVE_METHODS", "CIRCUIT_MODELS", "TABLEAUS", "combine",
           "canonical_method", "integrator", "FixedStepSolver", "AdaptiveSolver", "DormandPrince45", "AutoSolver",
           "subintervals"]

RHS = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class SolverStep:
    """Result of integrating one interval; ``y`` is the state at ``t``."""

    t: float
    y: NDArray[np.float64]
    n_rhs: int = 0


@dataclass
class _GuardedStep:
    """First internal solver step that makes a scalar guard negative.

    ``(previous_t, previous_y, previous_margin)`` is the last accepted nonnegative point and
    ``(t, y, margin)`` the first negative point. If no crossing occurs, both points are the final
    state and ``crossed`` is false. A missing ``previous_margin`` means the first step crossed.
    """

    t: float
    y: NDArray[np.float64]
    margin: float
    previous_t: float
    previous_y: NDArray[np.float64]
    previous_margin: float | None
    crossed: bool
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


EXPLICIT_FIXED_METHODS = ("euler", "heun", "RK4")
IMPLICIT_FIXED_METHODS = ("trapezoidal", "radau")
FIXED_METHODS = EXPLICIT_FIXED_METHODS + IMPLICIT_FIXED_METHODS
ADAPTIVE_METHODS = ("RK23", "RK45", "DOP853", "radau", "BDF", "LSODA", "DP45", "auto")
IMPLICIT_ADAPTIVE_METHODS = ("radau", "BDF", "LSODA")


def canonical_method(method: str, solver_type: str | None = None) -> str:
    """Return the canonical spelling of a solver method, leaving an unknown value unchanged."""
    if not isinstance(method, str):
        return method
    if isinstance(solver_type, str) and solver_type.casefold() == "fixed":
        methods = FIXED_METHODS
    elif isinstance(solver_type, str) and solver_type.casefold() == "adaptive":
        methods = ADAPTIVE_METHODS
    else:
        methods = FIXED_METHODS + ADAPTIVE_METHODS
    folded = method.casefold()
    return next((name for name in methods if name.casefold() == folded), method)


# explicit Runge-Kutta tableaus: (c, a_rows, b); a_rows[j - 1] holds the coefficients of stage j
TABLEAUS: dict[str, tuple[tuple[float, ...], tuple[tuple[float, ...], ...], tuple[float, ...]]] = {
    "euler": ((0.0,), (), (1.0,)),
    "heun": ((0.0, 1.0), ((1.0,),), (0.5, 0.5)),
    "RK4": ((0.0, 0.5, 0.5, 1.0), ((0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)), (1 / 6, 1 / 3, 1 / 3, 1 / 6)),
}


def combine(method: str, y: NDArray[np.float64], h: float, ks: list) -> NDArray[np.float64]:
    """The final combination of an explicit RK step of ``method`` from its stage derivatives ``ks``."""
    if method == "RK4":
        return y + (h / 6.0) * (ks[0] + 2.0 * (ks[1] + ks[2]) + ks[3])
    if method == "heun":
        return y + 0.5 * h * (ks[0] + ks[1])
    if method == "euler":
        return y + h * ks[0]
    total = None
    for weight, derivative in zip(TABLEAUS[method][2], ks):
        if weight:
            total = (weight * derivative if total is None
                     else total + weight * derivative)
    return y if total is None else y + h * total


def stage_state(y: NDArray[np.float64], h: float, row: tuple[float, ...], ks: list) -> NDArray[np.float64]:
    """The state at which a stage is evaluated: ``y + h * sum(a_j k_j)`` over the nonzero coefficients."""
    acc = None
    for coef, k in zip(row, ks):
        if coef == 0.0:
            continue
        acc = coef * h * k if acc is None else acc + coef * h * k
    return y + acc


def integrator(method: str, dt: float, rtol: float = 1e-6, atol: float = 1e-9,
               max_step: float = math.inf, warm_start: bool = True, *,
               model=None, solver_type: str | None = None) -> Solver:
    """Return the single-rate integrator of ``method``: fixed-step with maximum step ``dt`` (s),
    the built-in ``"DP45"`` or a SciPy method.

    Implicit methods use the state partition supplied by the assembled model.
    """
    adaptive = isinstance(solver_type, str) and solver_type.casefold() == "adaptive"
    if method in EXPLICIT_FIXED_METHODS and not adaptive:
        return FixedStepSolver(dt, method, rtol=rtol, atol=atol, model=model)
    if method == "DP45":
        return DormandPrince45(rtol, atol, max_step)
    if method == "auto":
        return AutoSolver(rtol, atol, max_step, warm_start, model=model)
    if method in IMPLICIT_FIXED_METHODS and not adaptive:
        from .implicit import ImplicitSolver, linear_system

        return ImplicitSolver(dt, method, rtol, atol, linear_system(model))
    return AdaptiveSolver(method, rtol, atol, max_step, warm_start, model=model)


def subintervals(span: float, step: float) -> int:
    """Return equal pieces that cover ``span`` without exceeding ``step``."""
    return max(1, int(math.ceil(span / step - 1e-9)))


class FixedStepSolver:
    """Explicit RK with a maximum step ``dt`` (s); sub-steps are equal within an interval.

    ``method``: ``"euler"``, ``"heun"`` or ``"RK4"``.
    """

    def __init__(self, dt: float, method: str = "RK4", *, rtol: float = 1e-6,
                 atol: float = 1e-9, model=None) -> None:
        if method not in TABLEAUS:
            raise ValueError(f"unknown fixed-step method {method!r}")
        self.dt = float(dt)
        self.method = method
        self.rtol, self.atol = float(rtol), float(atol)
        self.model = model
        self._implicit = None
        self._stability_revision = -1
        self._stability_mode = None
        self._stable_step = 0.0
        self._unstable_step = math.inf
        self._stability_modes: OrderedDict[object, tuple[float, float]] = OrderedDict()
        self._mode_eigenvalues: OrderedDict[object, NDArray[np.complex128]] = OrderedDict()
        self._dynamic_stability = bool(
            getattr(model, "explicit_stability_dynamic", False)
        )
        self.n_rhs = 0

    def parameters_changed(self) -> None:
        """Discard the lazy stiff-mode factors after a parameter event."""
        self._stability_revision = -1
        self._stability_mode = None
        self._stable_step = 0.0
        self._unstable_step = math.inf
        self._stability_modes.clear()
        self._mode_eigenvalues.clear()
        if self._implicit is not None:
            self._implicit.parameters_changed()

    def _explicit_stable(self, step: float) -> bool:
        """Check assembled linear modes against this Runge-Kutta stability region."""
        modes = getattr(self.model, "linear_modes", None)
        if not callable(modes):
            return True
        _c, rows, weights = TABLEAUS[self.method]
        for key, matrix in modes():
            eigenvalues = self._mode_eigenvalues.get(key)
            if eigenvalues is None:
                eigenvalues = np.linalg.eigvals(matrix)
                self._mode_eigenvalues[key] = eigenvalues
                self._mode_eigenvalues.move_to_end(key)
                while len(self._mode_eigenvalues) > 64:
                    self._mode_eigenvalues.popitem(last=False)
            else:
                self._mode_eigenvalues.move_to_end(key)
            for eigenvalue in eigenvalues:
                scale = max(1.0, abs(eigenvalue))
                if eigenvalue.real > 1e-12 * scale:
                    continue
                z = step * eigenvalue
                stages = [1.0 + 0.0j]
                for row in rows:
                    stages.append(1.0 + z * sum(
                        coefficient * value for coefficient, value in zip(row, stages)
                    ))
                amplification = 1.0 + z * sum(
                    coefficient * value for coefficient, value in zip(weights, stages)
                )
                if abs(amplification) > 1.0 + 1e-10:
                    return False
        return True

    def _stiff(self, step: float) -> bool:
        if not callable(getattr(self.model, "linear_modes", None)):
            return False
        revision = getattr(self.model, "equation_revision", 0)
        if self._dynamic_stability:
            return not self._explicit_stable(step)
        if revision != self._stability_revision:
            self._stability_revision = revision
            mode_key = getattr(self.model, "linear_mode_key", lambda: None)()
            self._stability_mode = mode_key
            bounds = self._stability_modes.get(mode_key) if mode_key is not None else None
            self._stable_step, self._unstable_step = bounds or (0.0, math.inf)
            if bounds is not None:
                self._stability_modes.move_to_end(mode_key)
        if step <= self._stable_step:
            return False
        if step >= self._unstable_step:
            return True
        result = not self._explicit_stable(step)
        if result:
            self._unstable_step = min(self._unstable_step, step)
        else:
            self._stable_step = max(self._stable_step, step)
        if self._stability_mode is not None:
            self._stability_modes[self._stability_mode] = (
                self._stable_step, self._unstable_step
            )
            self._stability_modes.move_to_end(self._stability_mode)
            while len(self._stability_modes) > 32:
                self._stability_modes.popitem(last=False)
        return result

    def _implicit_advance(self, f: RHS, t0: float, t1: float,
                          y0: NDArray[np.float64], step: float) -> SolverStep:
        """Use the shared implicit linear system for a mode outside the explicit stability region."""
        if self._implicit is None:
            from .implicit import ImplicitSolver, linear_system

            self._implicit = ImplicitSolver(
                self.dt, "radau", self.rtol, self.atol, linear_system(self.model)
            )
        before = self._implicit.n_rhs
        result = self._implicit.advance(f, t0, t1, y0, max_step=step)
        self.n_rhs += self._implicit.n_rhs - before
        return SolverStep(result.t, result.y, self.n_rhs)

    def __call__(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64]) -> SolverStep:
        return self.advance(f, t0, t1, y0)

    def advance(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64], *,
                max_step: float = math.inf) -> SolverStep:
        """Integrate with an optional per-call maximum step without changing configuration."""
        span = t1 - t0
        if span <= 0.0:
            return SolverStep(t1, y0, 0)
        step = min(self.dt, max_step)
        if step <= 0.0:
            raise ValueError("max_step must be positive")
        n = subintervals(span, step)
        h = span / n
        if self._stiff(h):
            return self._implicit_advance(f, t0, t1, y0, h)
        c, a, _b = TABLEAUS[self.method]
        self.n_rhs += len(c) * n
        if self.method == "RK4":
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

    def advance_guarded(
        self,
        f: RHS,
        t0: float,
        t1: float,
        y0: NDArray[np.float64],
        guard: Callable[[float, NDArray[np.float64] | list[float]], float],
        *,
        max_step: float = math.inf,
        synchronized_guard: Callable[[float, NDArray[np.float64] | list[float]], float] | None = None,
    ) -> _GuardedStep:
        """Advance through internal steps and stop at the first negative ``guard(t, y)``.

        This is the fixed-step solver's generic boundary-observation interface. It retains the
        last valid state, so a caller can locate a boundary only inside the crossing step instead
        of repeatedly invoking the solver for every ordinary step. ``synchronized_guard`` may be
        supplied when an RHS evaluation also refreshes everything read by the guard; interior
        observations then reuse the first RHS stage of the following step.
        """
        span = t1 - t0
        if span <= 0.0:
            margin = float(guard(t1, y0))
            return _GuardedStep(t1, y0, margin, t1, y0, margin, margin < 0.0, self.n_rhs)
        step = min(self.dt, max_step)
        if step <= 0.0:
            raise ValueError("max_step must be positive")
        n = subintervals(span, step)
        h = span / n
        if self._stiff(h):
            y, t = np.asarray(y0, dtype=float), t0
            previous_margin = None
            for _ in range(n):
                previous_t, previous_y = t, y
                result = self._implicit_advance(f, t, t + h, y, h)
                t, y = result.t, result.y
                observe = synchronized_guard if synchronized_guard is not None else guard
                margin = float(observe(t, y))
                if margin < 0.0:
                    return _GuardedStep(
                        t, y, margin, previous_t, previous_y,
                        previous_margin, True, self.n_rhs,
                    )
                previous_margin = margin
            return _GuardedStep(t1, y, margin, t1, y, margin, False, self.n_rhs)
        lists = getattr(getattr(f, "__self__", None), "rhs_list", None)
        if self.method == "RK4" and lists is not None:
            result = _rk4_on_floats_guarded(len(y0), synchronized_guard is not None)(
                lists, guard, synchronized_guard, t0, y0.tolist(), n, h
            )
            if result is not None:
                previous_t, previous_y, previous_margin, t, y, margin, calls = result
                self.n_rhs += calls
                crossed = margin < 0.0
                if not crossed:
                    previous_t, previous_y, previous_margin, t = t1, y, margin, t1
                return _GuardedStep(
                    t, np.asarray(y), margin, previous_t, np.asarray(previous_y),
                    previous_margin, crossed, self.n_rhs,
                )

        c, a, _b = TABLEAUS[self.method]
        y, t = np.asarray(y0, dtype=float), t0
        previous_margin = None
        calls = 0
        for completed in range(1, n + 1):
            previous_t, previous_y = t, y
            ks = []
            ks.append(f(t, y))
            calls += 1
            if synchronized_guard is not None and completed > 1:
                margin = float(synchronized_guard(t, y))
                if margin < 0.0:
                    self.n_rhs += calls
                    return _GuardedStep(
                        t, y, margin, last_t, last_y, previous_margin, True, self.n_rhs,
                    )
                previous_margin = margin
            last_t, last_y = previous_t, previous_y
            for j in range(1, len(c)):
                ks.append(f(t + c[j] * h, stage_state(y, h, a[j - 1], ks)))
                calls += 1
            y = combine(self.method, y, h, ks)
            t += h
            if synchronized_guard is None:
                margin = float(guard(t, y))
                if margin < 0.0:
                    self.n_rhs += calls
                    return _GuardedStep(
                        t, y, margin, previous_t, previous_y, previous_margin, True, self.n_rhs,
                    )
                previous_margin = margin
        if synchronized_guard is not None:
            margin = float(guard(t, y))
            if margin < 0.0:
                self.n_rhs += calls
                return _GuardedStep(
                    t, y, margin, previous_t, previous_y, previous_margin, True, self.n_rhs,
                )
        self.n_rhs += calls
        return _GuardedStep(t1, y, margin, t1, y, margin, False, self.n_rhs)


def _rk4_on_floats(size: int):
    """Build an RK4 stepper on ``size`` floats that returns ``None`` on a non-finite value."""
    return _rk4_float_program(size, "plain")


def _rk4_on_floats_guarded(size: int, synchronized: bool):
    """Build an RK4 loop which observes a scalar guard after every completed step."""
    return _rk4_float_program(size, "synchronized" if synchronized else "guarded")


@functools.lru_cache(maxsize=None)
def _rk4_float_program(size: int, observation: str):
    """Compile the shared scalar RK4 equations with an optional boundary observer."""
    guarded = observation != "plain"
    synchronized = observation == "synchronized"
    ys = [f"y{i}" for i in range(size)]
    ps = [f"p{i}" for i in range(size)]
    y = "[" + ", ".join(ys) + "]"
    p = "[" + ", ".join(ps) + "]"

    def stage(k: str) -> str:
        return "[" + ", ".join(f"{k}{i}" for i in range(size)) + "]"

    def args(k: str, h: str) -> str:
        return "[" + ", ".join(f"y{i} + {h} * {k}{i}" for i in range(size)) + "]"

    signature = ("g, guard, synchronized_guard, t, y, steps, h" if guarded
                 else "g, t, y, steps, h")
    lines = [f"def rk4({signature}):",
             "    h2 = 0.5 * h",
             "    h6 = h / 6.0",
             f"    {y} = y"]
    if guarded:
        lines.insert(3, "    previous_margin = None")
    if synchronized:
        lines += [f"    {p} = {y}", "    previous_t = t"]
    loop = "completed in range(1, steps + 1)" if guarded else "_ in range(steps)"
    lines += [f"    for {loop}:"]
    if guarded and not synchronized:
        lines += [f"        {p} = {y}", "        previous_t = t"]
    lines += [f"        {stage('a')} = g(t, {y})"]
    if synchronized:
        lines += ["        if completed > 1:",
                  f"            margin = float(synchronized_guard(t, {y}))",
                  "            if margin < 0.0:",
                  f"                return previous_t, {p}, previous_margin, t, {y}, margin, 4 * completed - 3",
                  "            previous_margin = margin",
                  f"            {p} = {y}",
                  "            previous_t = t"]
    lines += [
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
    lines += [f"        y{i} = y{i} + h6 * ((a{i} + 2.0 * (b{i} + c{i})) + d{i})"
              for i in range(size)]
    lines += [f"        if not isfinite({' + '.join(['0.0'] + ys)}):",
              "            return None",
              "        t += h",
    ]
    if synchronized:
        lines += [f"    margin = float(guard(t, {y}))",
                  "    if margin < 0.0:",
                  f"        return previous_t, {p}, previous_margin, t, {y}, margin, 4 * steps",
                  f"    return t, {y}, margin, t, {y}, margin, 4 * steps"]
    elif guarded:
        lines += [f"        margin = float(guard(t, {y}))",
                  "        if margin < 0.0:",
                  f"            return previous_t, {p}, previous_margin, t, {y}, margin, 4 * completed",
                  "        previous_margin = margin",
                  f"    return t, {y}, previous_margin, t, {y}, previous_margin, 4 * steps"]
    else:
        lines += [f"    return {y}"]
    label = "RK4" if not guarded else f"{observation} RK4"
    source, filename = "\n".join(lines) + "\n", f"<peslite {label}, {size} floats>"
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
        jac: Callable | None = None,
        *, model=None,
    ) -> None:
        from scipy.integrate import solve_ivp  # Load SciPy only when this solver is selected.

        self._solve_ivp = solve_ivp
        self.method = canonical_method(method, "adaptive")
        self.rtol, self.atol, self.max_step = rtol, atol, max_step
        self.warm_start = warm_start
        if self.method in IMPLICIT_ADAPTIVE_METHODS and jac is None:
            from .implicit import ModeJacobian, linear_system

            jac = ModeJacobian(getattr(model, "rhs", None), linear_system(model))
        self.jac = jac if self.method in IMPLICIT_ADAPTIVE_METHODS else None
        self._h_last: float | None = None
        self.n_rhs = 0

    def __call__(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64]) -> SolverStep:
        return self.advance(f, t0, t1, y0)

    def parameters_changed(self) -> None:
        """Discard a circuit model's Jacobians after an event changes parameters."""
        linear = getattr(self.jac, "linear", None)
        if linear is not None:
            linear.invalidate()

    def advance(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64], *,
                max_step: float = math.inf) -> SolverStep:
        """Integrate with an optional per-call maximum step without changing configuration."""
        span = t1 - t0
        if span <= 0.0:
            return SolverStep(t1, y0, 0)
        scipy_method = "Radau" if self.method == "radau" else self.method
        kwargs = dict(method=scipy_method, rtol=self.rtol, atol=self.atol)
        step = min(self.max_step, max_step)
        if step <= 0.0:
            raise ValueError("max_step must be positive")
        if math.isfinite(step):
            kwargs["max_step"] = step
        if self.warm_start and self._h_last is not None:
            kwargs["first_step"] = min(self._h_last, span, step)
        jac_rhs = getattr(self.jac, "n_rhs", 0)
        if self.jac is not None:
            bind = getattr(self.jac, "bind", None)
            if bind is not None:
                bind(f)
            kwargs["jac"] = self.jac
        sol = self._solve_ivp(f, (t0, t1), y0, **kwargs)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed on [{t0}, {t1}]: {sol.message}")
        steps = np.diff(sol.t)
        if steps.size > 1:
            self._h_last = float(steps[-2])
        elif steps.size == 1:
            self._h_last = max(self._h_last or 0.0, float(steps[0]))
        self.n_rhs += int(sol.nfev) + getattr(self.jac, "n_rhs", 0) - jac_rhs
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
        return self.advance(f, t0, t1, y0)

    def advance(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64], *,
                max_step: float = math.inf, step_callback: Callable | None = None) -> SolverStep:
        """Integrate, optionally stopping when an accepted-step callback returns true.

        The callback receives ``(h, y6, y1, k6, k7)`` from the existing final two stages.
        """
        span = t1 - t0
        if span <= 0.0:
            return SolverStep(t1, y0, 0)
        step = min(self.max_step, max_step)
        if step <= 0.0:
            raise ValueError("max_step must be positive")
        lists = getattr(getattr(f, "__self__", None), "rhs_list", None)
        if lists is not None:
            return self._call_lists(lists, t0, t1, y0, step, step_callback)
        a, b, c, e = self._a, self._b, self._c, self._e
        t, y = t0, np.asarray(y0, dtype=float)
        h = self._h if self._h is not None else span
        h = min(h, span, step)
        k1 = f(t, y)
        self.n_rhs += 1
        while t < t1 - commutation_tolerance(t1):
            if t + h > t1:
                h = t1 - t
            k2 = f(t + c[1] * h, y + h * (a[1][0] * k1))
            k3 = f(t + c[2] * h, y + h * (a[2][0] * k1 + a[2][1] * k2))
            k4 = f(t + c[3] * h, y + h * (a[3][0] * k1 + a[3][1] * k2 + a[3][2] * k3))
            k5 = f(t + c[4] * h, y + h * (a[4][0] * k1 + a[4][1] * k2 + a[4][2] * k3 + a[4][3] * k4))
            y6 = y + h * (a[5][0] * k1 + a[5][1] * k2 + a[5][2] * k3 + a[5][3] * k4 + a[5][4] * k5)
            k6 = f(t + h, y6)
            y1 = y + h * (b[0] * k1 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6)
            k7 = f(t + h, y1)
            self.n_rhs += 6
            err_vec = h * (e[0] * k1 + e[2] * k3 + e[3] * k4 + e[4] * k5 + e[5] * k6 + e[6] * k7)
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y1))
            err = float(np.max(np.abs(err_vec) / scale))
            accepted, h_next = self._control(h, err, step)
            if accepted:
                t, y, k1 = t + h, y1, k7
                if step_callback is not None and step_callback(h, y6, y1, k6, k7):
                    self._h = h_next
                    return SolverStep(t, y, self.n_rhs)
            h = h_next
        self._h = h
        return SolverStep(t1, y, self.n_rhs)

    def _control(self, h: float, err: float, max_step: float) -> tuple[bool, float]:
        """Whether a step ``h`` with scaled error ``err`` is accepted, and the next step (PI control)."""
        if err <= 1.0 or h <= 1e-15:
            if err == 0.0:
                factor = 5.0
            else:
                factor = self.safety * err ** -0.14 * self._err_prev ** 0.08
                factor = min(5.0, max(0.2, factor))
            self._err_prev = max(err, 1e-4)
            return True, min(h * factor, max_step)
        self.n_rejected += 1
        return False, h * max(0.1, self.safety * err ** -0.25)

    def _call_lists(self, f, t0: float, t1: float,
                    y0: NDArray[np.float64], max_step: float,
                    step_callback: Callable | None = None) -> SolverStep:
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
        h = min(h, span, max_step)
        k1 = f(t, y)
        self.n_rhs += 1
        while t < t1 - commutation_tolerance(t1):
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
            y6 = [v + h * (a61 * q1 + a62 * q2 + a63 * q3 + a64 * q4 + a65 * q5)
                  for v, q1, q2, q3, q4, q5 in zip(y, k1, k2, k3, k4, k5)]
            k6 = f(t + h, y6)
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
            accepted, h_next = self._control(h, err, max_step)
            if accepted:
                t, y, k1 = t + h, y1, k7
                if step_callback is not None and step_callback(h, y6, y1, k6, k7):
                    self._h = h_next
                    return SolverStep(t, np.asarray(y), self.n_rhs)
            h = h_next
        self._h = h
        return SolverStep(t1, np.asarray(y), self.n_rhs)


class AutoSolver:
    """Use DP45 until its stage estimates detect stiffness, then use adaptive Radau.

    The existing DP45 stages supply the stiffness estimate. Returning to DP45 uses the Jacobian
    already computed by Radau; neither decision runs a second model or probes extra RHS values.
    """

    def __init__(self, rtol: float = 1e-6, atol: float = 1e-9, max_step: float = math.inf,
                 warm_start: bool = True, *, model=None) -> None:
        self._dp = DormandPrince45(rtol, atol, max_step)
        self._warm_start, self._model = warm_start, model
        self.stiff = False
        self.switches = 0
        self._backoff = 1
        self._stiff_steps = self._nonstiff_steps = self._nonstiff_intervals = 0

    @functools.cached_property
    def _implicit(self):
        return AdaptiveSolver("radau", self._dp.rtol, self._dp.atol, self._dp.max_step,
                              self._warm_start, model=self._model)

    @property
    def max_step(self) -> float:
        return self._dp.max_step

    @property
    def n_rhs(self) -> int:
        implicit = self.__dict__.get("_implicit")
        return self._dp.n_rhs + (implicit.n_rhs if implicit is not None else 0)

    def parameters_changed(self) -> None:
        implicit = self.__dict__.get("_implicit")
        if implicit is not None:
            implicit.parameters_changed()
        self._stiff_steps = self._nonstiff_steps = self._nonstiff_intervals = 0

    def _accepted_step(self, h, y6, y1, k6, k7) -> bool:
        denominator = sum((a - b)**2 for a, b in zip(y1, y6))
        if denominator > 0.0:
            stiffness = h * h * sum((a - b)**2 for a, b in zip(k7, k6)) / denominator
            if stiffness > 3.25**2:
                self._nonstiff_steps = 0
                self._stiff_steps += 1
                if self._stiff_steps >= 15 * self._backoff:
                    self.stiff = True
                    self.switches += 1
                    self._implicit._h_last = h
                    return True
            else:
                self._nonstiff_steps += 1
                if self._nonstiff_steps >= 6:
                    self._stiff_steps = 0
        return False

    def _spectral_radius(self) -> float:
        """Return the smaller induced-norm upper bound of the Jacobian radius."""
        matrix = self._implicit.jac.linear.J
        if matrix is None or not matrix.size:
            return 0.0
        magnitude = np.abs(matrix)
        return float(min(magnitude.sum(axis=1).max(), magnitude.sum(axis=0).max()))

    def __call__(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64]) -> SolverStep:
        return self.advance(f, t0, t1, y0)

    def advance(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64], *,
                max_step: float = math.inf) -> SolverStep:
        if t1 <= t0:
            return SolverStep(t1, y0, self.n_rhs)
        if not self.stiff:
            result = self._dp.advance(f, t0, t1, y0, max_step=max_step,
                                      step_callback=self._accepted_step)
            if not self.stiff or result.t >= t1 - commutation_tolerance(t1):
                return SolverStep(t1, result.y, self.n_rhs)
            t0, y0 = result.t, result.y
        result = self._implicit.advance(f, t0, t1, y0, max_step=max_step)
        h = self._implicit._h_last
        if h is not None and h * self._spectral_radius() <= 3.25:
            self._nonstiff_intervals += 1
            if self._nonstiff_intervals >= 6 * self._backoff:
                self.stiff = False
                self.switches += 1
                self._backoff = min(2 * self._backoff, 64)
                self._stiff_steps = self._nonstiff_steps = self._nonstiff_intervals = 0
                self._dp._h = h
                self._dp._err_prev = 1.0
        else:
            self._nonstiff_intervals = 0
        return SolverStep(t1, result.y, self.n_rhs)


