"""MPC algorithms and their native export implementation.

Models use the caller's coordinates and units. At stage ``k``, the input ``u[k]``
produces ``x[k + 1]`` and ``y[k + 1] = C[k] @ x[k + 1]``. Tracking costs and
state/output bounds apply to these predicted endpoints, not the initial state.
The final control move is held when the control horizon is shorter than the
prediction horizon. Work limits count solver iterations or complete FCS plans;
they are deterministic and do not impose a wall-clock deadline.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import islice, product
from numbers import Integral, Real
from time import perf_counter
from typing import Any, Callable, ClassVar

import numpy as np

from ...control.blocks import abc2complex
from ...control.loops import (ANGLE, DUTY_RATIO, ENABLE, FREQUENCY, I, I_AB, SI_CURRENT,
                              SI_VOLTAGE, V, V_AB, Loop, register_loop_type)
from ...solver.model import ConfigError
from ...solver.timing import TIME_EPS, PeriodicClock

__all__ = ["LinearModel", "QuadraticProblem", "SolverResult", "MPCResult",
           "CCSMPC", "FCSMPC", "solve_active_set", "solve_slsqp",
           "PredictiveCurrentControl", "ThreePhasePredictiveCurrentControl"]


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _array(value, name, *, ndim=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values")
    array = np.array(value, dtype=float, copy=True)
    if (ndim is not None and array.ndim != ndim) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite" + (f" with {ndim} dimensions" if ndim else ""))
    return array


def _vector(value, size, name):
    array = _array(value, name).reshape(-1) if np.ndim(value) <= 1 else _array(value, name)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return array


def _forecast(value, horizon, width, name, *, finite=True):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values")
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full((horizon, width), float(array))
    elif array.shape == (width,):
        array = np.broadcast_to(array, (horizon, width)).copy()
    elif array.shape == (horizon, width):
        array = array.copy()
    else:
        raise ValueError(f"{name} must be scalar or have shape ({width},) or ({horizon}, {width})")
    if np.any(np.isnan(array)) or (finite and not np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain {'finite values' if finite else 'no NaN values'}")
    return array


def _bounds(value, horizon, width, name):
    if value is None:
        return np.full((horizon, width), -np.inf), np.full((horizon, width), np.inf)
    if len(value) != 2:
        raise ValueError(f"{name} must be a (lower, upper) pair")
    lower = _forecast(value[0], horizon, width, f"{name} lower", finite=False)
    upper = _forecast(value[1], horizon, width, f"{name} upper", finite=False)
    if np.any(lower > upper) or np.any(np.isposinf(lower)) or np.any(np.isneginf(upper)):
        raise ValueError(f"{name} requires lower <= upper and nonempty intervals")
    return lower, upper


def _weight_entries(value, width, name):
    """Expand one scalar, diagonal vector or full matrix without changing its values."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return tuple(value if row == column else 0.0
                     for row in range(width) for column in range(width))
    values = tuple(value)
    nested = tuple(isinstance(item, Sequence) and not isinstance(item, (str, bytes))
                   for item in values)
    if len(values) != width or (any(nested) and not all(nested)):
        raise ValueError(
            f"{name} must be a symmetric {width} x {width} matrix, diagonal vector or scalar"
        )
    if not any(nested):
        return tuple(values[row] if row == column else 0.0
                     for row in range(width) for column in range(width))
    if any(len(row) != width for row in values):
        raise ValueError(
            f"{name} must be a symmetric {width} x {width} matrix, diagonal vector or scalar"
        )
    return tuple(item for row in values for item in row)


def _weight(value, width, name):
    weight = _array(_weight_entries(value, width, name), name).reshape(width, width)
    if not np.array_equal(weight, weight.T):
        raise ValueError(f"{name} must be a symmetric {width} x {width} matrix, diagonal vector or scalar")
    # Eigenvalue roundoff is relative to the matrix scale; no regularization is applied.
    scale = max(1.0, float(np.linalg.norm(weight, ord=2)))
    if np.linalg.eigvalsh(weight)[0] < -1e-12 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    weight.setflags(write=False)
    return weight


class _QuadraticStageCost:
    """Validated quadratic tracking, input and input-increment cost."""

    __slots__ = ("Q", "R", "S")

    def __init__(self, output_width, input_width, tracking_weight, input_weight, rate_weight):
        self.Q = _weight(tracking_weight, output_width, "tracking_weight")
        self.R = _weight(input_weight, input_width, "input_weight")
        self.S = _weight(rate_weight, input_width, "rate_weight")

    def evaluate(self, output, action, reference, previous):
        error = output - reference
        delta = action - previous
        value = (
            np.einsum("...i,ij,...j->...", error, self.Q, error)
            + np.einsum("...i,ij,...j->...", action, self.R, action)
            + np.einsum("...i,ij,...j->...", delta, self.S, delta)
        )
        return float(value) if np.ndim(value) == 0 else value


@dataclass(frozen=True, init=False)
class LinearModel:
    """Discrete affine model ``x_next = A x + B u + offset``, ``y_next = C x_next``.

    Matrices are copied and read-only. ``C`` defaults to the identity and
    ``offset`` to zero. Discretization and physical-unit selection belong to the
    model provider; the algorithm never reads or advances the running plant.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    offset: np.ndarray

    def __init__(self, A, B, C=None, offset=None):
        a, b = _array(A, "A", ndim=2), _array(B, "B", ndim=2)
        if a.shape[0] == 0 or a.shape[0] != a.shape[1]:
            raise ValueError("A must be a nonempty square matrix")
        nx = a.shape[0]
        if b.shape[0] != nx or b.shape[1] == 0:
            raise ValueError("B must have one row per state and at least one input")
        c = np.eye(nx) if C is None else _array(C, "C", ndim=2)
        if c.shape[1] != nx or c.shape[0] == 0:
            raise ValueError("C must have one column per state and at least one output")
        d = np.zeros(nx) if offset is None else _vector(offset, nx, "offset")
        for name, array in (("A", a), ("B", b), ("C", c), ("offset", d)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)


@dataclass(frozen=True)
class QuadraticProblem:
    """Condensed QP passed to a custom backend.

    Minimize ``0.5 z.T H z + linear.T z + constant`` with
    ``lower <= z <= upper`` and ``constraint_lower <= constraint_matrix @ z
    <= constraint_upper``. ``z`` is the row-major control plan of length
    ``control_horizon * number_of_inputs``. Return a :class:`SolverResult`;
    the MPC independently checks dimensions, finiteness and feasibility.
    """

    hessian: np.ndarray
    linear: np.ndarray
    constant: float
    lower: np.ndarray
    upper: np.ndarray
    constraint_matrix: np.ndarray
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    initial: np.ndarray
    max_iterations: int
    tolerance: float
    feasibility_tolerance: float

    def __post_init__(self) -> None:
        # Backends receive their own immutable copy. The controller retains separate
        # coefficients for independent post-solve validation.
        for name in ("hessian", "linear", "lower", "upper", "constraint_matrix",
                     "constraint_lower", "constraint_upper", "initial"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SolverResult:
    solution: np.ndarray | None
    success: bool
    status: str
    iterations: int = 0
    evaluations: int = 0
    message: str = ""


@dataclass(frozen=True)
class MPCResult:
    """Optimization outcome; failed solves never expose an applicable action.

    Successful ``inputs`` include all prediction stages, and ``states`` include
    the initial state followed by each predicted endpoint. Objective values are
    recomputed by the algorithm, not trusted from the backend.
    """

    success: bool
    action: np.ndarray | None
    inputs: np.ndarray | None
    states: np.ndarray | None
    objective: float | None
    status: str
    iterations: int = 0
    evaluations: int = 0
    message: str = ""


def solve_slsqp(problem: QuadraticProblem) -> SolverResult:
    """Solve the convex quadratic problem using the existing SciPy dependency.

    SLSQP receives the objective and its analytic gradient divided by the same
    positive coefficient scale. This preserves the minimizer while preventing
    large physical current scales from causing false convergence at input
    bounds. ``tolerance`` is SLSQP's ``ftol`` for this normalized objective;
    feasibility is checked separately in the original model coordinates.
    The constant term is omitted from line-search evaluations because it does
    not affect the minimizer. Reported MPC objectives retain the original cost.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize

    scale = max(1.0, float(np.max(np.abs(problem.hessian))),
                float(np.max(np.abs(problem.linear))))
    hessian, linear = problem.hessian / scale, problem.linear / scale
    constraints = ()
    if len(problem.constraint_lower):
        constraints = (LinearConstraint(problem.constraint_matrix,
                                        problem.constraint_lower, problem.constraint_upper),)
    result = minimize(
        lambda z: 0.5 * z @ hessian @ z + linear @ z,
        problem.initial,
        jac=lambda z: hessian @ z + linear,
        method="SLSQP", bounds=Bounds(problem.lower, problem.upper), constraints=constraints,
        options={"maxiter": problem.max_iterations, "ftol": problem.tolerance},
    )
    iterations = int(result.get("nit", 0))
    status = ("optimal" if result.success else
              "iteration_limit" if iterations >= problem.max_iterations else "solver_failure")
    return SolverResult(result.x, bool(result.success), status,
                        iterations, int(result.get("nfev", 0)), str(result.message))


def _qp_rows(problem: QuadraticProblem):
    """Return normalized equality and upper-bound rows for one QP."""
    size = len(problem.linear)
    equalities, equality_limits, equality_errors = [], [], []
    inequalities, limits, errors = [], [], []
    impossible = False

    def interval(row, low, high):
        nonlocal impossible
        scale = float(np.max(np.abs(row), initial=0.0))
        if scale == 0.0:
            impossible |= ((np.isfinite(low) and low > problem.feasibility_tolerance)
                           or (np.isfinite(high) and high < -problem.feasibility_tolerance))
            return
        normalized = row / scale
        error = problem.feasibility_tolerance / scale
        if np.isfinite(low) and np.isfinite(high) and low == high:
            equalities.append(normalized)
            equality_limits.append(low / scale)
            equality_errors.append(error)
            return
        if np.isfinite(high):
            inequalities.append(normalized)
            limits.append(high / scale)
            errors.append(error)
        if np.isfinite(low):
            inequalities.append(-normalized)
            limits.append(-low / scale)
            errors.append(error)

    for row, low, high in zip(np.eye(size), problem.lower, problem.upper):
        interval(row, low, high)
    for row, low, high in zip(problem.constraint_matrix,
                              problem.constraint_lower, problem.constraint_upper):
        interval(row, low, high)

    def array(rows):
        return np.asarray(rows, dtype=float).reshape(-1, size)

    return (array(equalities), np.asarray(equality_limits), np.asarray(equality_errors),
            array(inequalities), np.asarray(limits), np.asarray(errors), impossible)


def solve_active_set(problem: QuadraticProblem) -> SolverResult:
    """Solve a convex QP with a deterministic primal active-set method.

    The initial plan is used directly when feasible; otherwise a linear phase-I solve finds a
    feasible point or proves infeasibility. Equality constraints remain in the working set, and
    dependent active inequalities are handled through rank-revealing reduced-space solves. The
    original positive-semidefinite Hessian is not regularized, so the reported optimum is for the
    configured objective rather than a perturbed problem.
    """
    from scipy.optimize import linprog, lsq_linear

    size = len(problem.linear)
    equality, equality_limit, equality_error, matrix, limit, error, impossible = _qp_rows(problem)
    if impossible:
        return SolverResult(None, False, "infeasible", message="The constraints are inconsistent")

    coefficient_scale = max(
        1.0,
        float(np.max(np.abs(problem.hessian), initial=0.0)),
        float(np.max(np.abs(problem.linear), initial=0.0)),
    )
    hessian = 0.5 * (problem.hessian + problem.hessian.T) / coefficient_scale
    linear = problem.linear / coefficient_scale
    eigenvalues = np.linalg.eigvalsh(hessian)
    curvature_tolerance = 256.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(eigenvalues), initial=0.0)), size,
    )
    if eigenvalues[0] < -curvature_tolerance:
        return SolverResult(None, False, "numerical_failure",
                            message="The QP Hessian is not positive semidefinite")

    solution = np.asarray(problem.initial, dtype=float).copy()

    def feasible(value):
        return ((not len(equality) or np.all(np.abs(equality @ value - equality_limit)
                                             <= equality_error))
                and (not len(matrix) or np.all(matrix @ value - limit <= error)))

    if not feasible(solution):
        phase = linprog(
            np.zeros(size),
            A_ub=matrix if len(matrix) else None,
            b_ub=limit if len(matrix) else None,
            A_eq=equality if len(equality) else None,
            b_eq=equality_limit if len(equality) else None,
            bounds=[(None, None)] * size,
            method="highs",
        )
        if not phase.success:
            status = "infeasible" if phase.status == 2 else "solver_failure"
            return SolverResult(None, False, status, message=str(phase.message))
        solution = np.asarray(phase.x, dtype=float)
        if not feasible(solution):
            return SolverResult(None, False, "numerical_failure",
                                message="The phase-I solution violates the QP constraints")

    stopping = max(problem.tolerance, 128.0 * np.finfo(float).eps)
    active = [index for index, slack in enumerate(limit - matrix @ solution)
              if slack <= stopping]
    iterations = 0
    while iterations < problem.max_iterations:
        iterations += 1
        working = (np.vstack((equality, matrix[active])) if active else equality)
        if len(working):
            _left, singular, right = np.linalg.svd(working, full_matrices=True)
            rank_tolerance = np.finfo(float).eps * max(working.shape) * max(
                1.0, float(singular[0]) if len(singular) else 1.0,
            )
            rank = int(np.count_nonzero(singular > rank_tolerance))
            nullspace = right[rank:].T
        else:
            nullspace = np.eye(size)

        gradient = hessian @ solution + linear
        if nullspace.shape[1]:
            reduced_hessian = nullspace.T @ hessian @ nullspace
            reduced_gradient = nullspace.T @ gradient
            if np.linalg.norm(reduced_gradient, ord=np.inf) <= stopping:
                step = np.zeros(size)
                has_flat_descent = False
            else:
                values, vectors = np.linalg.eigh(reduced_hessian)
                components = vectors.T @ reduced_gradient
                reduced_step = np.zeros_like(components)
                curved = values > curvature_tolerance
                reduced_step[curved] = -components[curved] / values[curved]
                flat = ~curved
                has_flat_descent = bool(
                    np.any(np.abs(components[flat]) > stopping)
                    and not np.any(np.abs(reduced_step[curved]) > stopping)
                )
                if has_flat_descent:
                    reduced_step[flat] = -components[flat]
                step = nullspace @ (vectors @ reduced_step)
        else:
            step = np.zeros(size)
            has_flat_descent = False

        if np.linalg.norm(step, ord=np.inf) <= stopping:
            if not len(active):
                return SolverResult(solution, True, "optimal", iterations, iterations, "Optimal")
            unconstrained = np.linalg.lstsq(working.T, -gradient, rcond=None)[0]
            inequality_multipliers = unconstrained[len(equality):]
            residual = np.linalg.norm(working.T @ unconstrained + gradient, ord=np.inf)
            if residual <= stopping and np.all(inequality_multipliers >= -stopping):
                return SolverResult(solution, True, "optimal", iterations, iterations, "Optimal")
            lower = np.r_[np.full(len(equality), -np.inf), np.zeros(len(active))]
            multipliers = lsq_linear(
                working.T, -gradient, bounds=(lower, np.full(len(working), np.inf)),
                tol=max(min(stopping, 1e-12), 1e-14),
                lsmr_tol=max(min(stopping, 1e-12), 1e-14),
            )
            residual = np.linalg.norm(working.T @ multipliers.x + gradient, ord=np.inf)
            if residual <= stopping:
                return SolverResult(solution, True, "optimal", iterations, iterations, "Optimal")
            leaving = int(np.argmin(inequality_multipliers))
            if inequality_multipliers[leaving] >= -stopping:
                return SolverResult(solution, False, "numerical_failure", iterations, iterations,
                                    "The active constraints have inconsistent multipliers")
            del active[leaving]
            continue

        alpha, blocking = (np.inf if has_flat_descent else 1.0), None
        active_set = set(active)
        for index, row in enumerate(matrix):
            if index in active_set:
                continue
            direction = float(row @ step)
            if direction <= stopping:
                continue
            distance = float((limit[index] - row @ solution) / direction)
            if distance <= alpha:
                alpha, blocking = max(0.0, distance), index
        if blocking is None and has_flat_descent:
            return SolverResult(None, False, "unbounded", iterations, iterations,
                                "The QP objective is unbounded")
        solution += alpha * step
        if blocking is not None:
            active.append(blocking)

    return SolverResult(solution, False, "iteration_limit", iterations, iterations,
                        "Active-set iteration limit reached")


def _linear_feasibility(lower, upper, matrix, constraint_lower, constraint_upper):
    """Return whether a bounded linear region is feasible, or ``None`` if undetermined."""
    from scipy.optimize import linprog

    rows, limits = [], []
    for row, low, high in zip(matrix, constraint_lower, constraint_upper):
        if np.isfinite(high):
            rows.append(row)
            limits.append(high)
        if np.isfinite(low):
            rows.append(-row)
            limits.append(-low)
    if not rows:
        return True
    bounds = [
        (None if np.isneginf(low) else low, None if np.isposinf(high) else high)
        for low, high in zip(lower, upper)
    ]
    result = linprog(np.zeros(len(lower)), A_ub=np.asarray(rows), b_ub=np.asarray(limits),
                     bounds=bounds, method="highs")
    if result.success:
        return True
    return False if result.status == 2 else None


def _optimality_residual(hessian, linear, lower, upper, matrix,
                         constraint_lower, constraint_upper, solution, active_tolerance):
    """Return a scale-independent KKT stationarity residual for one feasible QP point."""
    coefficient_scale = max(1.0, float(np.max(np.abs(hessian))),
                            float(np.max(np.abs(linear))))
    gradient = (hessian @ solution + linear) / coefficient_scale
    normals = []
    identity = np.eye(len(solution))
    for index, value in enumerate(solution):
        if np.isfinite(lower[index]) and value - lower[index] <= active_tolerance:
            normals.append(-identity[index])
        if np.isfinite(upper[index]) and upper[index] - value <= active_tolerance:
            normals.append(identity[index])
    values = matrix @ solution
    for row, value, low, high in zip(matrix, values, constraint_lower, constraint_upper):
        if np.isfinite(low) and value - low <= active_tolerance:
            normals.append(-row)
        if np.isfinite(high) and high - value <= active_tolerance:
            normals.append(row)
    if not normals:
        return float(np.linalg.norm(gradient, ord=np.inf))
    from scipy.optimize import nnls

    multipliers, _residual = nnls(np.asarray(normals).T / coefficient_scale, -gradient)
    stationarity = gradient + np.asarray(normals).T @ multipliers / coefficient_scale
    return float(np.linalg.norm(stationarity, ord=np.inf))


class CCSMPC:
    """Linear quadratic continuous-control-set MPC with hard endpoint bounds.

    The stage cost is ``(y_next-r).T Q (y_next-r) + u.T R u +
    (u-u_previous).T S (u-u_previous)``. Weights may be scalars, diagonal
    vectors or full symmetric positive-semidefinite matrices. Bounds are
    ``(lower, upper)`` pairs, broadcast from scalars/vectors or specified per
    stage. Feasibility tolerance is an absolute tolerance in each model
    coordinate; callers should scale coordinates when their magnitudes differ.
    """

    def __init__(self, model: LinearModel, prediction_horizon, control_horizon=None, *,
                 tracking_weight=1.0, input_weight=0.0, rate_weight=0.0,
                 input_bounds=None, state_bounds=None, output_bounds=None,
                 max_iterations=100, tolerance=1e-9, feasibility_tolerance=1e-7,
                 backend: Callable[[QuadraticProblem], SolverResult] | None = None):
        if not isinstance(model, LinearModel):
            raise TypeError("model must be a LinearModel")
        self.model = model
        self.prediction_horizon = _positive_integer(prediction_horizon, "prediction_horizon")
        self.control_horizon = (self.prediction_horizon if control_horizon is None
                                else _positive_integer(control_horizon, "control_horizon"))
        if self.control_horizon > self.prediction_horizon:
            raise ValueError("control_horizon must not exceed prediction_horizon")
        self.nx, self.nu, self.ny = model.A.shape[0], model.B.shape[1], model.C.shape[0]
        self._cost = _QuadraticStageCost(
            self.ny, self.nu, tracking_weight, input_weight, rate_weight,
        )
        self.input_bounds = _bounds(input_bounds, self.prediction_horizon, self.nu, "input_bounds")
        self.state_bounds = _bounds(state_bounds, self.prediction_horizon, self.nx, "state_bounds")
        self.output_bounds = _bounds(output_bounds, self.prediction_horizon, self.ny, "output_bounds")
        self.max_iterations = _positive_integer(max_iterations, "max_iterations")
        self.tolerance = _positive(tolerance, "tolerance")
        self.feasibility_tolerance = _positive(feasibility_tolerance, "feasibility_tolerance")
        if backend is not None and not callable(backend):
            raise TypeError("backend must be callable")
        self.backend = solve_active_set if backend is None else backend

    def solve(self, state, reference, previous_input=None, *, models=None, initial_plan=None):
        """Optimize from a copied state, optionally with one affine model per stage.

        A model sequence supports known disturbances, time-varying coefficients
        and delay-augmented models without callbacks into the running plant.
        ``initial_plan`` is only an initial guess, with shape ``(M, nu)``;
        input-bound projection here does not modify the returned solution.
        """
        x0 = _vector(state, self.nx, "state")
        previous = np.zeros(self.nu) if previous_input is None else _vector(previous_input, self.nu, "previous_input")
        horizon, moves = self.prediction_horizon, self.control_horizon
        refs = _forecast(reference, horizon, self.ny, "reference")
        sequence = (self.model,) * horizon if models is None else tuple(models)
        if len(sequence) != horizon or any(
            not isinstance(model, LinearModel) or model.A.shape != self.model.A.shape
            or model.B.shape != self.model.B.shape or model.C.shape != self.model.C.shape
            for model in sequence
        ):
            raise ValueError("models must contain one dimension-compatible LinearModel per prediction stage")
        lower, upper = (bound[:moves].copy() for bound in self.input_bounds)
        lower[-1] = np.max(self.input_bounds[0][moves - 1:], axis=0)
        upper[-1] = np.min(self.input_bounds[1][moves - 1:], axis=0)
        if np.any(lower > upper):
            return MPCResult(False, None, None, None, None, "infeasible",
                             message="Held final move has incompatible input bounds")
        guess = np.tile(previous, (moves, 1)) if initial_plan is None else _array(initial_plan, "initial_plan")
        if guess.shape != (moves, self.nu):
            raise ValueError(f"initial_plan must have shape ({moves}, {self.nu})")
        guess = np.clip(guess, lower, upper).ravel()
        size = moves * self.nu
        hessian, linear, constant = np.zeros((size, size)), np.zeros(size), 0.0
        free, response = x0.copy(), np.zeros((self.nx, size))
        prior_response = np.zeros((self.nu, size))
        prior_free = previous
        matrices, lows, highs = [], [], []
        for k, model in enumerate(sequence):
            move = min(k, moves - 1)
            selection = np.zeros((self.nu, size))
            selection[:, move * self.nu:(move + 1) * self.nu] = np.eye(self.nu)
            response = model.A @ response + model.B @ selection
            free = model.A @ free + model.offset
            output_response, output_free = model.C @ response, model.C @ free
            for mapping, offset, weight in (
                (output_response, output_free - refs[k], self._cost.Q),
                (selection, np.zeros(self.nu), self._cost.R),
                (selection - prior_response, -prior_free, self._cost.S),
            ):
                hessian += 2.0 * mapping.T @ weight @ mapping
                linear += 2.0 * mapping.T @ weight @ offset
                constant += float(offset @ weight @ offset)
            for mapping, offset, bounds in (
                (response, free, self.state_bounds),
                (output_response, output_free, self.output_bounds),
            ):
                mask = np.isfinite(bounds[0][k]) | np.isfinite(bounds[1][k])
                matrices.append(mapping[mask])
                lows.append(bounds[0][k][mask] - offset[mask])
                highs.append(bounds[1][k][mask] - offset[mask])
            prior_response, prior_free = selection, np.zeros(self.nu)
        constraint_matrix = np.vstack(matrices)
        constraint_lower, constraint_upper = np.concatenate(lows), np.concatenate(highs)
        if not (np.all(np.isfinite(hessian)) and np.all(np.isfinite(linear))
                and np.isfinite(constant) and np.all(np.isfinite(constraint_matrix))):
            return MPCResult(False, None, None, None, None, "numerical_failure",
                             message="Prediction produced non-finite condensed coefficients")
        lower_flat, upper_flat = lower.ravel(), upper.ravel()
        problem = QuadraticProblem(
            hessian, linear, constant, lower_flat, upper_flat,
            constraint_matrix, constraint_lower, constraint_upper, guess,
            self.max_iterations, self.tolerance, self.feasibility_tolerance,
        )
        try:
            result = self.backend(problem)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            return MPCResult(False, None, None, None, None, "numerical_failure", message=str(exc))
        if not isinstance(result, SolverResult):
            raise TypeError("MPC backend must return SolverResult")
        if not result.success:
            feasibility = _linear_feasibility(
                lower_flat, upper_flat, constraint_matrix,
                constraint_lower, constraint_upper,
            )
            status = "infeasible" if feasibility is False else result.status
            return MPCResult(False, None, None, None, None, status,
                             result.iterations, result.evaluations, result.message)
        z = np.asarray(result.solution)
        valid = (z.dtype.kind in "fiu" and z.shape == (size,) and np.all(np.isfinite(z)))
        if valid:
            tolerance = self.feasibility_tolerance
            constraint = constraint_matrix @ z
            valid = (np.all(z >= lower_flat - tolerance) and np.all(z <= upper_flat + tolerance)
                     and np.all(constraint >= constraint_lower - tolerance)
                     and np.all(constraint <= constraint_upper + tolerance))
        if not valid:
            return MPCResult(False, None, None, None, None, "invalid_solution",
                             result.iterations, result.evaluations,
                             "Backend solution is non-finite, has the wrong shape or violates constraints")
        residual = _optimality_residual(
            hessian, linear, lower_flat, upper_flat, constraint_matrix,
            constraint_lower, constraint_upper, z, self.feasibility_tolerance,
        )
        stationarity_tolerance = max(
            10.0 * self.tolerance,
            10.0 * self.feasibility_tolerance,
            100.0 * np.finfo(float).eps,
        )
        if residual > stationarity_tolerance:
            return MPCResult(
                False, None, None, None, None, "inaccurate",
                result.iterations, result.evaluations,
                f"Backend solution failed the optimality check (residual {residual:.3g})",
            )
        plan = z.reshape(moves, self.nu)
        inputs = np.asarray([plan[min(k, moves - 1)] for k in range(horizon)])
        states = [x0]
        for model, action in zip(sequence, inputs):
            states.append(model.A @ states[-1] + model.B @ action + model.offset)
        objective = float(0.5 * z @ hessian @ z + linear @ z + constant)
        if not np.isfinite(objective) or not np.all(np.isfinite(states)):
            return MPCResult(False, None, None, None, None, "numerical_failure",
                             result.iterations, result.evaluations,
                             "Backend solution produced a non-finite prediction or objective")
        return MPCResult(True, inputs[0].copy(), inputs, np.asarray(states), objective,
                         "optimal", result.iterations, result.evaluations, result.message)


class _FCSCallbacks:
    """Present scalar or vectorized user callbacks through one batched contract."""

    def __init__(self, prediction, objective, constraint, vectorized):
        self.prediction, self.objective, self.constraint = prediction, objective, constraint
        self.vectorized = bool(vectorized)

    def predict(self, states, actions, stage):
        if self.vectorized:
            return self.prediction(states.copy(), actions.copy(), stage)
        return [self.prediction(x.copy(), u.copy(), stage) for x, u in zip(states, actions)]

    def cost(self, states, actions, references, previous, stage):
        if self.vectorized:
            return self.objective(
                states.copy(), actions.copy(), references.copy(), previous.copy(), stage,
            )
        return [
            self.objective(x.copy(), u.copy(), r.copy(), last.copy(), stage)
            for x, u, r, last in zip(states, actions, references, previous)
        ]

    def residuals(self, states, actions, stage):
        if self.constraint is None:
            return None
        if self.vectorized:
            return self.constraint(states.copy(), actions.copy(), stage)
        return [self.constraint(x.copy(), u.copy(), stage) for x, u in zip(states, actions)]


class FCSMPC:
    """Exhaustive finite-control-set MPC with deterministic minimum-switching ties.

    ``prediction(x, u, k)`` returns the next state. ``objective(x_next, u,
    reference[k], u_previous, k)`` returns a scalar stage cost. An optional
    ``constraint(x_next, u, k)`` returns residuals that must be nonnegative
    within ``feasibility_tolerance``. With ``vectorized=True``, every array has a
    leading candidate dimension and callbacks return one result per candidate.
    Scalar callbacks are adapted to that same contract, so validation and search
    have one implementation. Callbacks receive copies, must be deterministic and
    must not capture mutable plant state.
    """

    def __init__(self, actions, prediction_horizon, prediction, objective, *,
                 control_horizon=None, constraint=None, max_evaluations=100_000,
                 feasibility_tolerance=1e-7, vectorized=False):
        self.actions = _array(actions, "actions")
        if self.actions.ndim == 1:
            self.actions = self.actions[:, None]
        if self.actions.ndim != 2 or not all(self.actions.shape):
            raise ValueError("actions must be a nonempty matrix of candidate input vectors")
        self.actions.setflags(write=False)
        action_scale = np.ptp(self.actions, axis=0)
        self._action_scale = np.where(action_scale > 0.0, action_scale, 1.0)
        self.prediction_horizon = _positive_integer(prediction_horizon, "prediction_horizon")
        self.control_horizon = (self.prediction_horizon if control_horizon is None
                                else _positive_integer(control_horizon, "control_horizon"))
        if self.control_horizon > self.prediction_horizon:
            raise ValueError("control_horizon must not exceed prediction_horizon")
        if not callable(prediction) or not callable(objective) or (constraint is not None and not callable(constraint)):
            raise TypeError("prediction, objective and optional constraint must be callable")
        if not isinstance(vectorized, (bool, np.bool_)):
            raise TypeError("vectorized must be boolean")
        self.prediction, self.objective, self.constraint = prediction, objective, constraint
        self._callbacks = _FCSCallbacks(prediction, objective, constraint, vectorized)
        self.max_evaluations = _positive_integer(max_evaluations, "max_evaluations")
        self.feasibility_tolerance = _positive(feasibility_tolerance, "feasibility_tolerance")

    def solve(self, state, reference, previous_input=None):
        x0 = _array(state, "state").reshape(-1)
        if not len(x0) or np.ndim(state) > 1:
            raise ValueError("state must be a nonempty vector")
        reference = _array(reference, "reference")
        if reference.ndim <= 1:
            reference = np.broadcast_to(reference.reshape(1, -1),
                                        (self.prediction_horizon, reference.size)).copy()
        elif reference.ndim != 2 or reference.shape[0] != self.prediction_horizon:
            raise ValueError("reference must be a vector or one row per prediction stage")
        if reference.shape[1] == 0:
            raise ValueError("reference must have at least one component")
        previous = (np.zeros(self.actions.shape[1]) if previous_input is None else
                    _vector(previous_input, self.actions.shape[1], "previous_input"))
        total = len(self.actions) ** self.control_horizon
        if total > self.max_evaluations:
            return MPCResult(
                False, None, None, None, None, "evaluation_limit",
                evaluations=self.max_evaluations,
                message="Finite-set search budget is smaller than the exhaustive search",
            )
        best_cost, best_distance = np.inf, np.inf
        best_inputs, best_states, evaluations = None, None, 0
        plans = product(range(len(self.actions)), repeat=self.control_horizon)
        while chunk := tuple(islice(plans, 256)):
            choices = np.asarray(chunk, dtype=np.intp)
            count = len(choices)
            held = choices[:, np.minimum(
                np.arange(self.prediction_horizon), self.control_horizon - 1,
            )]
            inputs = self.actions[held]
            states = np.empty((count, self.prediction_horizon + 1, x0.size))
            states[:, 0] = x0
            costs = np.zeros(count)
            feasible = np.ones(count, dtype=bool)
            previous_actions = np.broadcast_to(previous, (count, previous.size)).copy()
            try:
                for stage in range(self.prediction_horizon):
                    active = np.flatnonzero(feasible)
                    if not active.size:
                        break
                    actions = inputs[active, stage]
                    next_states = np.asarray(
                        self._callbacks.predict(states[active, stage], actions, stage)
                    )
                    if (np.iscomplexobj(next_states)
                            or next_states.shape != (active.size, x0.size)):
                        raise ValueError(
                            "prediction must return one real state vector per candidate"
                        )
                    if not np.all(np.isfinite(next_states)):
                        raise FloatingPointError("prediction returned a non-finite state")
                    states[active, stage + 1] = next_states
                    residuals = self._callbacks.residuals(next_states, actions, stage)
                    if residuals is not None:
                        residuals = np.asarray(residuals)
                        if (np.iscomplexobj(residuals) or residuals.dtype.kind == "b"
                                or residuals.ndim == 0 or residuals.shape[0] != active.size):
                            raise ValueError(
                                "constraint must return real residuals with one row per candidate, not booleans"
                            )
                        residuals = residuals.reshape(active.size, -1)
                        if not np.all(np.isfinite(residuals)):
                            raise FloatingPointError("constraint returned non-finite residuals")
                        accepted = np.all(
                            residuals >= -self.feasibility_tolerance, axis=1,
                        )
                        feasible[active[~accepted]] = False
                        active = active[accepted]
                        if not active.size:
                            continue
                        next_states, actions = states[active, stage + 1], inputs[active, stage]
                    stage_cost = np.asarray(self._callbacks.cost(
                        next_states, actions,
                        np.broadcast_to(reference[stage], (active.size, reference.shape[1])),
                        previous_actions[active], stage,
                    ), dtype=float)
                    if stage_cost.shape != (active.size,):
                        raise ValueError("objective must return one scalar cost per candidate")
                    costs[active] += stage_cost
                    if not np.all(np.isfinite(costs[active])):
                        raise FloatingPointError("objective returned a non-finite cost")
                    previous_actions[active] = actions
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                return MPCResult(
                    False, None, None, None, None, "numerical_failure",
                    evaluations=evaluations + count, message=str(exc),
                )
            evaluations += count
            for index in np.flatnonzero(feasible):
                cost = float(costs[index])
                distance = float(np.sum(
                    np.abs(inputs[index, 0] - previous) / self._action_scale
                ))
                tolerance = (32.0 * np.finfo(float).eps
                             * max(1.0, abs(cost), abs(best_cost)))
                better = best_inputs is None or cost < best_cost - tolerance
                tied = best_inputs is not None and abs(cost - best_cost) <= tolerance
                if better or (tied and distance < best_distance):
                    best_cost, best_distance = cost, distance
                    best_inputs, best_states = inputs[index].copy(), states[index].copy()
        if best_inputs is None:
            return MPCResult(False, None, None, None, None, "infeasible", evaluations=evaluations,
                             message="No admissible finite-action plan satisfies the constraints")
        return MPCResult(True, best_inputs[0].copy(), best_inputs, best_states, float(best_cost),
                         "optimal", evaluations=evaluations)


_NATIVE_KERNEL = r'''
template <std::size_t NX, std::size_t NU, std::size_t NY> struct PredictiveModel {
    std::array<double, NX * NX> a{};
    std::array<double, NX * NU> b{};
    std::array<double, NY * NX> c{};
    std::array<double, NX> offset{};
};

template <std::size_t NU> struct PredictiveResult {
    std::array<double, NU> action{};
    double objective = 0.0;
    std::size_t iterations = 0;
    std::size_t evaluations = 0;
    bool success = false;
};

inline double predictive_integral(double inductance, double resistance,
                                  double duration) noexcept {
    return resistance == 0.0 ? duration / inductance
                             : -std::expm1(-resistance * duration / inductance) / resistance;
}

template <std::size_t NC, std::size_t NU>
PredictiveModel<NC + NU, NU, NC> make_rl_prediction(
        double inductance, double resistance, double period, double delay, double dc_voltage,
        const std::array<double, NC>& current,
        const std::array<double, NC>& disturbance_before,
        const std::array<double, NC>& disturbance_after,
        const std::array<double, NC * NU>& voltage_map,
        bool previous_applied, bool has_blocked_min,
        const std::array<double, NC>& blocked_min) noexcept {
    constexpr std::size_t NX = NC + NU;
    PredictiveModel<NX, NU, NC> model;
    const double before_decay = std::exp(-resistance * delay / inductance);
    const double after = period - delay;
    const double after_decay = std::exp(-resistance * after / inductance);
    const double previous_gain = after_decay * predictive_integral(
        inductance, resistance, delay) * dc_voltage;
    const double new_gain = predictive_integral(inductance, resistance, after) * dc_voltage;
    for (std::size_t row = 0; row < NC; ++row) {
        double prior_state = before_decay;
        double prior_offset = -disturbance_before[row];
        if (!previous_applied && delay > 0.0 && has_blocked_min
                && prior_state * current[row] + prior_offset <= blocked_min[row]) {
            prior_state = 0.0;
            prior_offset = blocked_min[row];
        }
        model.a[row * NX + row] = after_decay * prior_state;
        model.offset[row] = after_decay * prior_offset - disturbance_after[row];
        model.c[row * NX + row] = 1.0;
        for (std::size_t column = 0; column < NU; ++column) {
            const double gain = voltage_map[row * NU + column];
            if (previous_applied)
                model.a[row * NX + NC + column] = previous_gain * gain;
            model.b[row * NU + column] = new_gain * gain;
        }
    }
    for (std::size_t column = 0; column < NU; ++column)
        model.b[(NC + column) * NU + column] = 1.0;
    return model;
}

template <std::size_t MAXIMUM>
bool predictive_solve(std::array<double, MAXIMUM * MAXIMUM> matrix,
                      std::array<double, MAXIMUM>& value,
                      std::size_t size) noexcept {
    double scale = 0.0;
    for (std::size_t row = 0; row < size; ++row)
        for (std::size_t column = 0; column < size; ++column)
            scale = std::max(scale, std::abs(matrix[row * MAXIMUM + column]));
    const double threshold = std::numeric_limits<double>::epsilon()
        * std::max(1.0, scale) * size;
    for (std::size_t column = 0; column < size; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < size; ++row)
            if (std::abs(matrix[row * MAXIMUM + column])
                    > std::abs(matrix[pivot * MAXIMUM + column])) pivot = row;
        if (!(std::abs(matrix[pivot * MAXIMUM + column]) > threshold)) return false;
        if (pivot != column) {
            for (std::size_t entry = 0; entry < size; ++entry)
                std::swap(matrix[column * MAXIMUM + entry],
                          matrix[pivot * MAXIMUM + entry]);
            std::swap(value[column], value[pivot]);
        }
        const double diagonal = matrix[column * MAXIMUM + column];
        for (std::size_t row = column + 1; row < size; ++row) {
            const double multiplier = matrix[row * MAXIMUM + column] / diagonal;
            matrix[row * MAXIMUM + column] = 0.0;
            for (std::size_t entry = column + 1; entry < size; ++entry)
                matrix[row * MAXIMUM + entry]
                    -= multiplier * matrix[column * MAXIMUM + entry];
            value[row] -= multiplier * value[column];
        }
    }
    for (std::size_t row = size; row-- > 0;) {
        for (std::size_t column = row + 1; column < size; ++column)
            value[row] -= matrix[row * MAXIMUM + column] * value[column];
        value[row] /= matrix[row * MAXIMUM + row];
    }
    for (std::size_t index = 0; index < size; ++index)
        if (!std::isfinite(value[index])) return false;
    return true;
}

template <std::size_t N, std::size_t MAX_LINEAR>
bool predictive_qp(
        const std::array<double, N * N>& hessian,
        const std::array<double, N>& linear,
        const std::array<double, N>& lower,
        const std::array<double, N>& upper,
        const std::array<double, MAX_LINEAR * N>& linear_matrix,
        const std::array<double, MAX_LINEAR>& linear_lower,
        const std::array<double, MAX_LINEAR>& linear_upper,
        std::size_t linear_rows, std::array<double, N>& solution,
        std::size_t max_iterations, double tolerance, double feasibility_tolerance,
        std::size_t& iterations) noexcept {
    constexpr std::size_t MAX_INEQUALITIES = 2 * (N + MAX_LINEAR);
    std::array<double, MAX_INEQUALITIES * N> inequalities{};
    std::array<double, MAX_INEQUALITIES> limits{}, allowed_error{};
    std::size_t count = 0;
    auto add_interval = [&](const double* row, double low, double high,
                            double error) {
        if (std::isfinite(high)) {
            for (std::size_t column = 0; column < N; ++column)
                inequalities[count * N + column] = row[column];
            limits[count] = high;
            allowed_error[count++] = error;
        }
        if (std::isfinite(low)) {
            for (std::size_t column = 0; column < N; ++column)
                inequalities[count * N + column] = -row[column];
            limits[count] = -low;
            allowed_error[count++] = error;
        }
    };
    std::array<double, N> identity_row{};
    for (std::size_t row = 0; row < N; ++row) {
        identity_row.fill(0.0);
        identity_row[row] = 1.0;
        add_interval(identity_row.data(), lower[row], upper[row], feasibility_tolerance);
    }
    for (std::size_t row = 0; row < linear_rows; ++row) {
        double scale = 0.0;
        for (std::size_t column = 0; column < N; ++column)
            scale = std::max(scale, std::abs(linear_matrix[row * N + column]));
        if (!(scale > 0.0)) {
            if (0.0 < linear_lower[row] - feasibility_tolerance
                    || 0.0 > linear_upper[row] + feasibility_tolerance) return false;
            continue;
        }
        std::array<double, N> normalized{};
        for (std::size_t column = 0; column < N; ++column)
            normalized[column] = linear_matrix[row * N + column] / scale;
        add_interval(normalized.data(), linear_lower[row] / scale,
                     linear_upper[row] / scale, feasibility_tolerance / scale);
    }
    double coefficient_scale = 1.0;
    for (double value : hessian)
        coefficient_scale = std::max(coefficient_scale, std::abs(value));
    for (double value : linear)
        coefficient_scale = std::max(coefficient_scale, std::abs(value));
    for (std::size_t column = 0; column < N; ++column)
        solution[column] = std::clamp(solution[column], lower[column], upper[column]);
    const std::size_t projection_limit = std::max<std::size_t>(32, 8 * count);
    for (std::size_t sweep = 0; sweep < projection_limit; ++sweep) {
        double violation_max = 0.0;
        for (std::size_t constraint = 0; constraint < count; ++constraint) {
            double value = -limits[constraint], norm = 0.0;
            for (std::size_t column = 0; column < N; ++column) {
                const double entry = inequalities[constraint * N + column];
                value += entry * solution[column];
                norm += entry * entry;
            }
            violation_max = std::max(
                violation_max, value - allowed_error[constraint]);
            if (value > allowed_error[constraint] && norm > 0.0)
                for (std::size_t column = 0; column < N; ++column)
                    solution[column] -= value / norm
                        * inequalities[constraint * N + column];
        }
        if (violation_max <= 0.0) break;
    }
    auto feasible = [&]() {
        for (std::size_t constraint = 0; constraint < count; ++constraint) {
            double value = -limits[constraint];
            for (std::size_t column = 0; column < N; ++column)
                value += inequalities[constraint * N + column] * solution[column];
            if (value > allowed_error[constraint]) return false;
        }
        return true;
    };
    if (!feasible()) return false;
    constexpr std::size_t KKT = 2 * N;
    std::array<bool, MAX_INEQUALITIES> active{};
    std::array<std::size_t, N> working{};
    std::size_t working_size = 0;
    const double stopping = std::max(tolerance,
        128.0 * std::numeric_limits<double>::epsilon());
    for (iterations = 1; iterations <= max_iterations; ++iterations) {
        std::array<double, N> gradient{};
        for (std::size_t row = 0; row < N; ++row) {
            gradient[row] = linear[row] / coefficient_scale;
            for (std::size_t column = 0; column < N; ++column)
                gradient[row] += hessian[row * N + column]
                    / coefficient_scale * solution[column];
        }
        std::array<double, KKT * KKT> system{};
        std::array<double, KKT> right{};
        const std::size_t size = N + working_size;
        for (std::size_t row = 0; row < N; ++row) {
            right[row] = -gradient[row];
            for (std::size_t column = 0; column < N; ++column)
                system[row * KKT + column]
                    = hessian[row * N + column] / coefficient_scale;
            system[row * KKT + row] += 1.0e-12;
        }
        for (std::size_t index = 0; index < working_size; ++index) {
            const std::size_t constraint = working[index];
            for (std::size_t column = 0; column < N; ++column) {
                const double value = inequalities[constraint * N + column];
                system[column * KKT + N + index] = value;
                system[(N + index) * KKT + column] = value;
            }
        }
        if (!predictive_solve<KKT>(system, right, size)) return false;
        double step_norm = 0.0;
        for (std::size_t column = 0; column < N; ++column)
            step_norm = std::max(step_norm, std::abs(right[column]));
        if (step_norm <= stopping) {
            std::size_t remove = working_size;
            double multiplier = -stopping;
            for (std::size_t index = 0; index < working_size; ++index)
                if (right[N + index] < multiplier) {
                    multiplier = right[N + index];
                    remove = index;
                }
            if (remove == working_size) return feasible();
            active[working[remove]] = false;
            for (std::size_t index = remove + 1; index < working_size; ++index)
                working[index - 1] = working[index];
            --working_size;
            continue;
        }
        double alpha = 1.0;
        std::size_t blocking = count;
        for (std::size_t constraint = 0; constraint < count; ++constraint) {
            if (active[constraint]) continue;
            double direction = 0.0, gap = limits[constraint];
            for (std::size_t column = 0; column < N; ++column) {
                const double entry = inequalities[constraint * N + column];
                direction += entry * right[column];
                gap -= entry * solution[column];
            }
            if (direction > stopping && gap / direction < alpha) {
                alpha = std::max(0.0, gap / direction);
                blocking = constraint;
            }
        }
        for (std::size_t column = 0; column < N; ++column)
            solution[column] += alpha * right[column];
        if (blocking != count) {
            if (working_size == N) return false;
            active[blocking] = true;
            working[working_size++] = blocking;
        }
    }
    iterations = max_iterations;
    return false;
}

template <std::size_t NX, std::size_t NU, std::size_t NY,
          std::size_t HORIZON, std::size_t MOVES>
PredictiveResult<NU> solve_ccs_mpc(
        const std::array<PredictiveModel<NX, NU, NY>, HORIZON>& models,
        const std::array<double, NX>& initial_state,
        const std::array<std::array<double, NY>, HORIZON>& references,
        const std::array<double, NU>& previous,
        const std::array<double, NY * NY>& tracking_weight,
        const std::array<double, NU * NU>& input_weight,
        const std::array<double, NU * NU>& rate_weight,
        double input_min, double input_max,
        double output_min, double output_max,
        std::size_t max_iterations, double tolerance,
        double feasibility_tolerance) noexcept {
    constexpr std::size_t NZ = MOVES * NU;
    constexpr std::size_t MAX_LINEAR = HORIZON * NY;
    PredictiveResult<NU> result;
    std::array<double, NZ * NZ> hessian{};
    std::array<double, NZ> linear{};
    double constant = 0.0;
    std::array<double, NX> free = initial_state;
    std::array<double, NX * NZ> response{};
    std::array<double, NU * NZ> prior_response{};
    std::array<double, NU> prior_free = previous;
    std::array<double, MAX_LINEAR * NZ> constraints{};
    std::array<double, MAX_LINEAR> constraint_lower{}, constraint_upper{};
    std::size_t constraint_rows = 0;
    for (std::size_t stage = 0; stage < HORIZON; ++stage) {
        const auto& model = models[stage];
        const std::size_t move = std::min(stage, MOVES - 1);
        std::array<double, NX> next_free{};
        std::array<double, NX * NZ> next_response{};
        for (std::size_t row = 0; row < NX; ++row) {
            next_free[row] = model.offset[row];
            for (std::size_t inner = 0; inner < NX; ++inner) {
                next_free[row] += model.a[row * NX + inner] * free[inner];
                for (std::size_t column = 0; column < NZ; ++column)
                    next_response[row * NZ + column]
                        += model.a[row * NX + inner] * response[inner * NZ + column];
            }
            for (std::size_t input = 0; input < NU; ++input)
                next_response[row * NZ + move * NU + input]
                    += model.b[row * NU + input];
        }
        free = next_free;
        response = next_response;
        std::array<double, NY> output_free{};
        std::array<double, NY * NZ> output_response{};
        for (std::size_t row = 0; row < NY; ++row)
            for (std::size_t inner = 0; inner < NX; ++inner) {
                output_free[row] += model.c[row * NX + inner] * free[inner];
                for (std::size_t column = 0; column < NZ; ++column)
                    output_response[row * NZ + column]
                        += model.c[row * NX + inner] * response[inner * NZ + column];
            }
        auto add_cost = [&](const auto& mapping, const auto& offset,
                            const auto& weight, std::size_t width) {
            for (std::size_t left = 0; left < NZ; ++left) {
                for (std::size_t right = 0; right < NZ; ++right)
                    for (std::size_t row = 0; row < width; ++row)
                        for (std::size_t column = 0; column < width; ++column)
                            hessian[left * NZ + right] += 2.0
                                * mapping[row * NZ + left]
                                * weight[row * width + column]
                                * mapping[column * NZ + right];
                for (std::size_t row = 0; row < width; ++row)
                    for (std::size_t column = 0; column < width; ++column)
                        linear[left] += 2.0 * mapping[row * NZ + left]
                            * weight[row * width + column] * offset[column];
            }
            for (std::size_t row = 0; row < width; ++row)
                for (std::size_t column = 0; column < width; ++column)
                    constant += offset[row] * weight[row * width + column]
                        * offset[column];
        };
        std::array<double, NY> tracking_offset{};
        for (std::size_t row = 0; row < NY; ++row)
            tracking_offset[row] = output_free[row] - references[stage][row];
        add_cost(output_response, tracking_offset, tracking_weight, NY);
        std::array<double, NU * NZ> selection{};
        for (std::size_t row = 0; row < NU; ++row)
            selection[row * NZ + move * NU + row] = 1.0;
        std::array<double, NU> zero{};
        add_cost(selection, zero, input_weight, NU);
        std::array<double, NU * NZ> delta{};
        std::array<double, NU> delta_offset{};
        for (std::size_t row = 0; row < NU; ++row) {
            delta_offset[row] = -prior_free[row];
            for (std::size_t column = 0; column < NZ; ++column)
                delta[row * NZ + column] = selection[row * NZ + column]
                    - prior_response[row * NZ + column];
        }
        add_cost(delta, delta_offset, rate_weight, NU);
        prior_response = selection;
        prior_free.fill(0.0);
        if (std::isfinite(output_min) || std::isfinite(output_max))
            for (std::size_t row = 0; row < NY; ++row) {
                for (std::size_t column = 0; column < NZ; ++column)
                    constraints[constraint_rows * NZ + column]
                        = output_response[row * NZ + column];
                constraint_lower[constraint_rows] = output_min - output_free[row];
                constraint_upper[constraint_rows] = output_max - output_free[row];
                ++constraint_rows;
            }
    }
    std::array<double, NZ> lower{}, upper{}, solution{};
    lower.fill(input_min);
    upper.fill(input_max);
    for (std::size_t move = 0; move < MOVES; ++move)
        for (std::size_t input = 0; input < NU; ++input)
            solution[move * NU + input]
                = std::clamp(previous[input], input_min, input_max);
    std::size_t iterations = 0;
    result.success = predictive_qp<NZ, MAX_LINEAR>(
        hessian, linear, lower, upper, constraints, constraint_lower, constraint_upper,
        constraint_rows, solution, max_iterations, tolerance,
        feasibility_tolerance, iterations);
    result.iterations = iterations;
    result.evaluations = iterations;
    if (!result.success) return result;
    for (std::size_t input = 0; input < NU; ++input)
        result.action[input] = solution[input];
    double objective = constant;
    for (std::size_t row = 0; row < NZ; ++row) {
        objective += linear[row] * solution[row];
        for (std::size_t column = 0; column < NZ; ++column)
            objective += 0.5 * solution[row] * hessian[row * NZ + column]
                * solution[column];
    }
    result.objective = objective;
    return result;
}

template <std::size_t NX, std::size_t NU, std::size_t NY,
          std::size_t HORIZON, std::size_t MOVES, std::size_t ACTIONS>
PredictiveResult<NU> solve_fcs_mpc(
        const std::array<PredictiveModel<NX, NU, NY>, HORIZON>& models,
        const std::array<double, NX>& initial_state,
        const std::array<std::array<double, NY>, HORIZON>& references,
        const std::array<double, NU>& previous,
        const std::array<std::array<double, NU>, ACTIONS>& actions,
        const std::array<double, NY * NY>& tracking_weight,
        const std::array<double, NU * NU>& input_weight,
        const std::array<double, NU * NU>& rate_weight,
        double input_min, double input_max,
        double output_min, double output_max,
        std::size_t max_evaluations, double feasibility_tolerance) noexcept {
    PredictiveResult<NU> result;
    std::array<std::size_t, ACTIONS> allowed{};
    std::size_t allowed_count = 0;
    for (std::size_t index = 0; index < ACTIONS; ++index) {
        bool admissible = true;
        for (double value : actions[index])
            admissible = admissible && value >= input_min && value <= input_max;
        if (admissible) allowed[allowed_count++] = index;
    }
    if (allowed_count == 0) return result;
    std::size_t plans = 1;
    for (std::size_t move = 0; move < MOVES; ++move) {
        if (plans > max_evaluations / allowed_count) {
            result.evaluations = max_evaluations;
            return result;
        }
        plans *= allowed_count;
    }
    result.evaluations = plans;
    double best_cost = std::numeric_limits<double>::infinity();
    double best_distance = std::numeric_limits<double>::infinity();
    std::array<double, NU> action_scale{};
    for (std::size_t input = 0; input < NU; ++input) {
        double low = actions[allowed[0]][input], high = actions[allowed[0]][input];
        for (std::size_t index = 0; index < allowed_count; ++index) {
            const auto& action = actions[allowed[index]];
            low = std::min(low, action[input]);
            high = std::max(high, action[input]);
        }
        action_scale[input] = high > low ? high - low : 1.0;
    }
    for (std::size_t plan = 0; plan < plans; ++plan) {
        std::array<std::size_t, MOVES> choices{};
        std::size_t divisor = plans;
        for (std::size_t move = 0; move < MOVES; ++move) {
            divisor /= allowed_count;
            choices[move] = allowed[(plan / divisor) % allowed_count];
        }
        std::array<double, NX> state = initial_state;
        std::array<double, NU> prior = previous;
        double cost = 0.0;
        bool feasible = true;
        for (std::size_t stage = 0; stage < HORIZON && feasible; ++stage) {
            const auto& action = actions[choices[std::min(stage, MOVES - 1)]];
            const auto& model = models[stage];
            std::array<double, NX> next = model.offset;
            for (std::size_t row = 0; row < NX; ++row) {
                for (std::size_t column = 0; column < NX; ++column)
                    next[row] += model.a[row * NX + column] * state[column];
                for (std::size_t input = 0; input < NU; ++input)
                    next[row] += model.b[row * NU + input] * action[input];
            }
            std::array<double, NY> output{};
            for (std::size_t row = 0; row < NY; ++row) {
                for (std::size_t column = 0; column < NX; ++column)
                    output[row] += model.c[row * NX + column] * next[column];
                feasible = feasible
                    && output[row] >= output_min - feasibility_tolerance
                    && output[row] <= output_max + feasibility_tolerance;
            }
            if (!feasible) break;
            for (std::size_t row = 0; row < NY; ++row)
                for (std::size_t column = 0; column < NY; ++column)
                    cost += (output[row] - references[stage][row])
                        * tracking_weight[row * NY + column]
                        * (output[column] - references[stage][column]);
            for (std::size_t row = 0; row < NU; ++row)
                for (std::size_t column = 0; column < NU; ++column) {
                    cost += action[row] * input_weight[row * NU + column]
                        * action[column];
                    cost += (action[row] - prior[row])
                        * rate_weight[row * NU + column]
                        * (action[column] - prior[column]);
                }
            state = next;
            prior = action;
        }
        if (!feasible || !std::isfinite(cost)) continue;
        double distance = 0.0;
        const auto& first = actions[choices[0]];
        for (std::size_t input = 0; input < NU; ++input)
            distance += std::abs(first[input] - previous[input]) / action_scale[input];
        const double tie = 32.0 * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(cost), std::abs(best_cost)});
        const bool better = !result.success || cost < best_cost - tie;
        const bool tied = result.success && std::abs(cost - best_cost) <= tie;
        if (better || (tied && distance < best_distance)) {
            result.success = true;
            result.action = first;
            result.objective = cost;
            best_cost = cost;
            best_distance = distance;
        }
    }
    return result;
}
'''


def _native_array(generator: Any, values: Iterable[Any]) -> str:
    return "{{" + ", ".join(
        value if isinstance(value, str) else generator.number(value) for value in values
    ) + "}}"


class _PredictiveLoopGenerator:
    """Lower predictive loops to the target-independent fixed-size MPC kernel."""

    shared_code = _NATIVE_KERNEL

    def members(self, generator: Any, loop_name: str, node: Any) -> list[str]:
        lines = []
        for local, value in node.get_state().items():
            lines.append(
                f"{generator.cpp_type(value)} "
                f"{generator.node(loop_name, local)} = {generator.literal(value)};"
            )
        for field, value in node.observe().items():
            lines.append(f"double {generator.g(loop_name + '.' + field)} = {generator.number(value)};")
        return lines

    @staticmethod
    def reset(_generator: Any, _loop_name: str, _node: Any) -> list[str]:
        return []

    def _prediction(self, generator: Any, loop_name: str, node: Any,
                    names: Mapping[str, str]) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def _bound(value: Any, lower: bool) -> Any:
        if value is not None:
            return value
        return -math.inf if lower else math.inf

    def _common(self, generator: Any, loop_name: str, node: Any) -> tuple[list[str], dict[str, str]]:
        horizon = int(node.cfg.prediction_horizon)
        nc, nu, nx = node.n_current, node.n_input, node.n_current + node.n_input
        prefix = generator.m("mpc_" + loop_name)
        previous = [generator._state(loop_name, f"previous_input.{index}")
                    for index in range(nu)]
        names = {
            "prefix": prefix,
            "models": prefix + "_models",
            "current": prefix + "_current",
            "initial": prefix + "_initial",
            "references": prefix + "_references",
            "previous": prefix + "_previous",
            "result": prefix + "_result",
            "action": prefix + "_action",
            "valid": prefix + "_valid",
            "enabled": prefix + "_enabled",
        }
        q = _weight_entries(node.cfg.tracking_weight, nc, "tracking_weight")
        r = _weight_entries(node.cfg.input_weight, nu, "input_weight")
        s = _weight_entries(node.cfg.rate_weight, nu, "rate_weight")
        lines = [
            f"std::array<PredictiveModel<{nx}, {nu}, {nc}>, {horizon}> {names['models']}{{}};",
            f"const std::array<double, {nu}> {names['previous']}{_native_array(generator, previous)};",
            f"std::array<double, {nx}> {names['initial']}{{}};",
            *[f"{names['initial']}[{index}] = {value};"
              for index, value in enumerate((*(["0.0"] * nc), *previous))],
            f"std::array<std::array<double, {nc}>, {horizon}> {names['references']}{{}};",
            f"const std::array<double, {nc * nc}> {prefix}_q{_native_array(generator, q)};",
            f"const std::array<double, {nu * nu}> {prefix}_r{_native_array(generator, r)};",
            f"const std::array<double, {nu * nu}> {prefix}_s{_native_array(generator, s)};",
        ]
        return lines, names

    def _dc_prediction(self, generator: Any, loop_name: str, node: Any,
                       names: Mapping[str, str]) -> list[str]:
        current = generator.input(loop_name, "current")
        voltage = generator.input(loop_name, "voltage")
        vdc = generator.input(loop_name, "vin")
        reference = generator.input(loop_name, "reference")
        prefix = names["prefix"]
        plant = node.plant
        horizon = int(node.cfg.prediction_horizon)
        previous_enabled = generator._state(loop_name, "previous_enabled")
        return [
            f"const std::array<double, 1> {names['current']}{{{current}}};",
            f"{names['initial']}[0] = {current};",
            f"const double {prefix}_before = {voltage} * predictive_integral("
            f"{generator.number(plant.L)}, {generator.number(plant.R)}, {generator.number(plant.delay)});",
            f"const double {prefix}_after = {voltage} * predictive_integral("
            f"{generator.number(plant.L)}, {generator.number(plant.R)}, {generator.number(plant.period - plant.delay)});",
            *[line for stage in range(horizon) for line in (
                f"{names['references']}[{stage}][0] = {reference} * {generator.m('startup_value')};",
                f"{names['models']}[{stage}] = make_rl_prediction<1, 1>("
                f"{generator.number(plant.L)}, {generator.number(plant.R)}, {generator.number(plant.period)}, "
                f"{generator.number(plant.delay)}, {vdc}, {names['current']}, "
                f"std::array<double, 1>{{{prefix}_before}}, "
                f"std::array<double, 1>{{{prefix}_after}}, "
                f"std::array<double, 1>{{1.0}}, "
                f"{'true' if stage else previous_enabled}, true, std::array<double, 1>{{0.0}});",
            )],
        ]

    def _three_phase_prediction(self, generator: Any, loop_name: str, node: Any,
                                names: Mapping[str, str]) -> list[str]:
        prefix = names["prefix"]
        plant = node.plant
        horizon = int(node.cfg.prediction_horizon)
        current = generator.input(loop_name, "i")
        voltage = generator.input(loop_name, "v")
        frame = generator.input(loop_name, "frame")
        omega = generator.input(loop_name, "omega")
        id_ref = generator.input(loop_name, "id_ref")
        iq_ref = generator.input(loop_name, "iq_ref")
        vdc = generator.input(loop_name, "u_dc")
        previous_enabled = generator._state(loop_name, "previous_enabled")
        voltage_map = tuple(float(value) for value in plant.voltage_map.ravel())
        lines = [
            f"const Complex {prefix}_i = {current} * {generator.number(node.i_base)};",
            f"const std::array<double, 2> {names['current']}{{std::real({prefix}_i), std::imag({prefix}_i)}};",
            f"{names['initial']}[0] = std::real({prefix}_i);",
            f"{names['initial']}[1] = std::imag({prefix}_i);",
            f"const Complex {prefix}_target = Complex{{{id_ref}, {iq_ref}}} "
            f"* {generator.number(node.i_base)} * {generator.m('startup_value')};",
            f"const double {prefix}_vdc = {vdc} * {generator.number(node.vdc_base)};",
            f"const Complex {prefix}_denominator{{{generator.number(plant.R)}, {omega} * {generator.number(plant.L)}}};",
            f"auto {prefix}_disturbance = [&](double start, double duration) {{",
            f"    const Complex factor = std::abs({prefix}_denominator) == 0.0",
            f"        ? Complex{{duration / {generator.number(plant.L)}, 0.0}}",
            f"        : (std::exp(Complex{{0.0, {omega} * duration}}) "
            f"- std::exp(-{generator.number(plant.R)} * duration / {generator.number(plant.L)})) / {prefix}_denominator;",
            f"    const Complex value = {voltage} * {generator.number(node.v_base)} "
            f"* std::exp(Complex{{0.0, {omega} * start}}) * factor;",
            "    return std::array<double, 2>{std::real(value), std::imag(value)};",
            "};",
        ]
        for stage in range(horizon):
            before = f"{prefix}_before_{stage}"
            after = f"{prefix}_after_{stage}"
            lines += [
                f"const Complex {prefix}_reference_{stage} = {prefix}_target * std::exp("
                f"Complex{{0.0, {frame} + {omega} * {generator.number((stage + 1) * plant.period)}}});",
                f"{names['references']}[{stage}] = {{std::real({prefix}_reference_{stage}), "
                f"std::imag({prefix}_reference_{stage})}};",
                f"const auto {before} = {prefix}_disturbance("
                f"{generator.number(stage * plant.period)}, {generator.number(plant.delay)});",
                f"const auto {after} = {prefix}_disturbance("
                f"{generator.number(stage * plant.period + plant.delay)}, "
                f"{generator.number(plant.period - plant.delay)});",
                f"{names['models']}[{stage}] = make_rl_prediction<2, 3>("
                f"{generator.number(plant.L)}, {generator.number(plant.R)}, {generator.number(plant.period)}, "
                f"{generator.number(plant.delay)}, {prefix}_vdc, {names['current']}, {before}, {after}, "
                f"std::array<double, 6>{_native_array(generator, voltage_map)}, "
                f"{'true' if stage else previous_enabled}, false, std::array<double, 2>{{}});",
            ]
        return lines

    def _solver(self, generator: Any, node: Any, names: Mapping[str, str]) -> list[str]:
        horizon = int(node.cfg.prediction_horizon)
        moves = (int(node.cfg.control_horizon)
                 if node.cfg.control_horizon is not None else horizon)
        nc, nu, nx = node.n_current, node.n_input, node.n_current + node.n_input
        prefix = names["prefix"]
        output_min = self._bound(node.cfg.current_min, True)
        output_max = self._bound(node.cfg.current_max, False)
        common = (
            f"{names['models']}, {names['initial']}, {names['references']}, "
            f"{names['previous']}, {prefix}_q, {prefix}_r, {prefix}_s, "
            f"{generator.number(node.cfg.input_min)}, {generator.number(node.cfg.input_max)}, "
            f"{generator.number(output_min)}, {generator.number(output_max)}, "
        )
        if node.cfg.algorithm == "ccs":
            call = (
                f"solve_ccs_mpc<{nx}, {nu}, {nc}, {horizon}, {moves}>("
                f"{common}static_cast<std::size_t>({generator.number(node.cfg.max_iterations)}), "
                f"{generator.number(node.cfg.tolerance)}, {generator.number(node.cfg.feasibility_tolerance)})"
            )
            return [f"{names['result']} = {call};"]
        actions = tuple(product((0.0, 1.0), repeat=nu))
        action_values = ", ".join(_native_array(generator, action) for action in actions)
        call = (
            f"solve_fcs_mpc<{nx}, {nu}, {nc}, {horizon}, {moves}, {len(actions)}>("
            f"{names['models']}, {names['initial']}, {names['references']}, "
            f"{names['previous']}, {prefix}_actions, {prefix}_q, {prefix}_r, {prefix}_s, "
            f"{generator.number(node.cfg.input_min)}, {generator.number(node.cfg.input_max)}, "
            f"{generator.number(output_min)}, {generator.number(output_max)}, "
            f"static_cast<std::size_t>({generator.number(node.cfg.max_evaluations)}), "
            f"{generator.number(node.cfg.feasibility_tolerance)})"
        )
        return [
            f"const std::array<std::array<double, {nu}>, {len(actions)}> "
            f"{prefix}_actions{{{{{action_values}}}}};",
            f"{names['result']} = {call};",
        ]

    def _fixed_action(self, generator: Any, node: Any, prefix: str) -> tuple[list[str], str]:
        name = prefix + "_fixed"
        if node.cfg.fallback_input is not None:
            return ([f"const std::array<double, {node.n_input}> {name}"
                     f"{_native_array(generator, node.cfg.fallback_input)};"], name)
        if node.cfg.algorithm == "ccs":
            return ([f"std::array<double, {node.n_input}> {name}{{}};",
                     f"{name}.fill({generator.number(node.cfg.input_min)});"], name)
        default = (f"({generator.number(node.cfg.input_min)} <= 0.0 && "
                   f"{generator.number(node.cfg.input_max)} >= 0.0 ? 0.0 : 1.0)")
        return ([f"std::array<double, {node.n_input}> {name}{{}};",
                 f"{name}.fill({default});"], name)

    def lines(self, generator: Any, loop_name: str, node: Any) -> list[str]:
        lines, names = self._common(generator, loop_name, node)
        fixed_lines, fixed = self._fixed_action(generator, node, names["prefix"])
        lines += [*fixed_lines, f"PredictiveResult<{node.n_input}> {names['result']}{{}};"]
        prediction = self._prediction(generator, loop_name, node, names)
        solve = [*prediction, *self._solver(generator, node, names)]
        lines += [f"if ({generator.m('startup_run')}) {{",
                  *("    " + line for line in solve), "}"]
        previous_enabled = generator._state(loop_name, "previous_enabled")
        solves = generator._state(loop_name, "solve_count")
        fallbacks = generator._state(loop_name, "fallback_count")
        lines += [
            f"std::array<double, {node.n_input}> {names['action']} = {fixed};",
            f"bool {names['enabled']} = {generator.m('startup_run')};",
            f"bool {names['valid']} = false;",
            f"if ({generator.m('startup_run')}) {{",
            f"    ++{solves};",
            f"    {names['valid']} = {names['result']}.success;",
            f"    for (double value : {names['result']}.action)",
            f"        {names['valid']} = {names['valid']} && std::isfinite(value) "
            f"&& value >= {generator.number(node.cfg.input_min)} && value <= {generator.number(node.cfg.input_max)};",
            f"    if ({names['valid']}) {names['action']} = {names['result']}.action;",
            "    else {",
            f"        ++{fallbacks};",
        ]
        if node.cfg.fallback == "hold":
            hold_valid = (f"std::all_of({names['previous']}.begin(), {names['previous']}.end(), "
                          f"[&](double value) {{ return std::isfinite(value) "
                          f"&& value >= {generator.number(node.cfg.input_min)} "
                          f"&& value <= {generator.number(node.cfg.input_max)}; }})")
            lines += [f"        if ({hold_valid}) {names['action']} = {names['previous']};"]
        if node.cfg.fallback == "block":
            lines.append(f"        {names['enabled']} = false;")
        lines += [
            "    }",
            "}",
            *[f"{generator._state(loop_name, f'previous_input.{index}')} = "
              f"{names['action']}[{index}];" for index in range(node.n_input)],
            f"{previous_enabled} = {names['enabled']};",
        ]
        output_ports = tuple(port for port in node.outputs if port != "enable")
        lines += [f"{generator.g(loop_name + '.' + port)} = {names['action']}[{index}];"
                  for index, port in enumerate(output_ports)]
        lines += [
            f"{generator.g(loop_name + '.enable')} = {names['enabled']} ? 1.0 : 0.0;",
            f"{generator.g(loop_name + '.success')} = {names['valid']} ? 1.0 : 0.0;",
            f"{generator.g(loop_name + '.objective')} = "
            f"{names['result']}.success ? {names['result']}.objective : 0.0;",
            f"{generator.g(loop_name + '.iterations')} = "
            f"static_cast<double>({names['result']}.iterations);",
            f"{generator.g(loop_name + '.evaluations')} = "
            f"static_cast<double>({names['result']}.evaluations);",
            f"{generator.g(loop_name + '.solves')} = {solves};",
            f"{generator.g(loop_name + '.fallbacks')} = {fallbacks};",
            f"{generator.g(loop_name + '.delay')} = {generator.number(node.delay)};",
        ]
        return lines


class _DCPredictiveLoopGenerator(_PredictiveLoopGenerator):
    """Lower the scalar converter prediction owned by PredictiveCurrentControl."""

    def _prediction(self, generator: Any, loop_name: str, node: Any,
                    names: Mapping[str, str]) -> list[str]:
        return self._dc_prediction(generator, loop_name, node, names)


class _ThreePhasePredictiveLoopGenerator(_PredictiveLoopGenerator):
    """Lower the vector prediction owned by ThreePhasePredictiveCurrentControl."""

    def _prediction(self, generator: Any, loop_name: str, node: Any,
                    names: Mapping[str, str]) -> list[str]:
        return self._three_phase_prediction(generator, loop_name, node, names)


_DC_NATIVE_GENERATOR = _DCPredictiveLoopGenerator()
_THREE_PHASE_NATIVE_GENERATOR = _ThreePhasePredictiveLoopGenerator()


def _numerical_configuration(value):
    """Read the default number from a generated-parameter expression for numerical setup."""
    if isinstance(value, np.ndarray):
        items = [_numerical_configuration(item) for item in value.flat]
        return np.asarray(items).reshape(value.shape)
    if isinstance(value, (list, tuple)):
        return type(value)(_numerical_configuration(item) for item in value)
    return getattr(value, "_peslite_numeric_value", value)


def _positive_configuration(value, name, *, integer=False):
    """Validate one numerical parameter while retaining a generated range condition."""
    kind = "a positive integer" if integer else "finite and positive"
    generated = hasattr(value, "_peslite_numeric_value")
    numerical = _numerical_configuration(value)
    if (integer and not generated
            and (isinstance(numerical, bool) or not isinstance(numerical, Integral))):
        raise ConfigError(f"{name} must be {kind}")
    if not isinstance(numerical, Real) or not math.isfinite(numerical) or numerical <= 0:
        raise ConfigError(f"{name} must be {kind}")
    if generated and (not np.isfinite(value) or not value > 0):
        raise ConfigError(f"{name} must be {kind}")


def _scalar_weight_configuration(value, name):
    """Validate tunable scalar weights; vector and matrix forms remain numerical constants."""
    numerical = _numerical_configuration(value)
    if np.ndim(numerical) != 0:
        return
    if (not isinstance(numerical, Real) or not math.isfinite(numerical)
            or numerical < 0):
        raise ConfigError(f"{name} must be finite and nonnegative")
    if (hasattr(value, "_peslite_numeric_value")
            and (not np.isfinite(value) or not value >= 0)):
        raise ConfigError(f"{name} must be finite and nonnegative")


class _RLCurrentPrediction:
    """Exact discrete affine prediction for one RL current path per coordinate."""

    def __init__(self, inductance, resistance, period, delay, voltage_map, *, blocked_min=None):
        self.L, self.R = inductance, resistance
        self.period, self.delay = period, delay
        self.voltage_map = np.asarray(voltage_map, dtype=float)
        self.voltage_map.setflags(write=False)
        self.n_current, self.n_input = self.voltage_map.shape
        numerical = _numerical_configuration
        self.decay = math.exp(
            -numerical(self.R) * numerical(self.period) / numerical(self.L)
        )
        self.blocked_min = (None if blocked_min is None else
                            np.broadcast_to(blocked_min, (self.n_current,)).copy())
        if self.blocked_min is not None:
            self.blocked_min.setflags(write=False)

    def integral(self, duration):
        """Exact RL input integral, including the physical R = 0 limit."""
        return (duration / self.L if self.R == 0
                else -math.expm1(-self.R * duration / self.L) / self.R)

    def model(self, vdc, current, disturbance_before, disturbance_after, previous_applied):
        n, m = self.n_current, self.n_input
        A = np.zeros((n + m, n + m))
        B = np.zeros((n + m, m))
        before_decay = math.exp(-self.R * self.delay / self.L)
        after = self.period - self.delay
        after_decay = math.exp(-self.R * after / self.L)
        prior_state = np.eye(n) * before_decay
        prior_offset = -np.asarray(disturbance_before)
        if (not previous_applied and self.delay > 0.0
                and self.blocked_min is not None):
            blocked = prior_state @ current + prior_offset <= self.blocked_min
            prior_state[blocked] = 0.0
            prior_offset[blocked] = self.blocked_min[blocked]
        A[:n, :n] = after_decay * prior_state
        if previous_applied:
            A[:n, n:] = after_decay * self.integral(self.delay) * vdc * self.voltage_map
        B[:n] = self.integral(after) * vdc * self.voltage_map
        B[n:] = np.eye(m)
        C = np.zeros((n, n + m))
        C[:, :n] = np.eye(n)
        offset = after_decay * prior_offset - np.asarray(disturbance_after)
        return LinearModel(A, B, C, np.r_[offset, np.zeros(m)])


@dataclass(frozen=True, kw_only=True)
class _PredictiveCurrentParams:
    """Parameters shared by scalar and three-phase predictive current loops."""

    period: float | None = None
    algorithm: str = field(default="ccs", metadata={"runtime": False})
    prediction_horizon: int = field(default=3, metadata={"runtime": False})
    control_horizon: int | None = field(default=None, metadata={"runtime": False})
    tracking_weight: Any = 1.0
    input_weight: Any = 0.0
    rate_weight: Any = 0.01
    input_min: float = 0.0
    input_max: float = 1.0
    current_min: float | None = None
    current_max: float | None = None
    max_iterations: int = 100
    max_evaluations: int = 10000
    tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-7
    fallback: str = "hold"
    fallback_input: list[float] | None = field(default=None, metadata={"runtime": False})

    def __post_init__(self) -> None:
        if self.algorithm not in ("ccs", "fcs"):
            raise ConfigError("algorithm must be 'ccs' or 'fcs'")
        if self.fallback not in ("hold", "fixed", "block"):
            raise ConfigError("fallback must be 'hold', 'fixed' or 'block'")
        if not (math.isfinite(self.input_min) and math.isfinite(self.input_max)
                and 0.0 <= self.input_min < self.input_max <= 1.0):
            raise ConfigError("input bounds must satisfy 0 <= input_min < input_max <= 1")
        for name in ("current_min", "current_max"):
            value = getattr(self, name)
            if (value is not None
                    and (not math.isfinite(value)
                         or hasattr(value, "_peslite_numeric_value")
                         and not np.isfinite(value))):
                raise ConfigError(f"{name} must be finite")
        if (self.current_min is not None and self.current_max is not None
                and self.current_min > self.current_max):
            raise ConfigError("current_min must not exceed current_max")
        _positive_configuration(self.prediction_horizon, "prediction_horizon", integer=True)
        if self.control_horizon is not None:
            _positive_configuration(self.control_horizon, "control_horizon", integer=True)
            if self.control_horizon > self.prediction_horizon:
                raise ConfigError("control_horizon must not exceed prediction_horizon")
        _positive_configuration(self.max_iterations, "max_iterations", integer=True)
        _positive_configuration(self.max_evaluations, "max_evaluations", integer=True)
        _positive_configuration(self.tolerance, "tolerance")
        _positive_configuration(self.feasibility_tolerance, "feasibility_tolerance")
        for name in ("tracking_weight", "input_weight", "rate_weight"):
            _scalar_weight_configuration(getattr(self, name), name)
        if (self.algorithm == "fcs"
                and self.input_min * (1.0 - self.input_max) != 0.0):
            raise ConfigError("input bounds exclude every finite control action")
        if self.fallback_input is not None:
            for value in self.fallback_input:
                if (not isinstance(value, Real) or not math.isfinite(value)
                        or not 0.0 <= value <= 1.0
                        or not self.input_min <= value <= self.input_max
                        or self.algorithm == "fcs" and value not in (0.0, 1.0)):
                    raise ConfigError(
                        "fallback_input must contain finite admissible commands"
                    )


class _PredictiveCurrentControl(Loop):
    """Shared prediction, optimization, fallback and named-state implementation."""

    n_current: ClassVar[int]
    n_input: ClassVar[int]
    outputs_from_state = True
    cpp_export = False
    carried = ("last_result", "solve_seconds")
    observations = ("success", "objective", "iterations", "evaluations", "solves",
                    "fallbacks", "delay")

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        pwm_timed = unit.bridge.pwm_timed
        if pwm_timed and (
                unit.pwm.sync != "asynchronous" or unit.pwm.pattern_sync != "interval"):
            raise ConfigError(
                f"{type(self).__name__} requires asynchronous interval PWM"
            )
        if pwm_timed and unit.pwm.dead_time_compensation:
            raise ConfigError(
                f"{type(self).__name__} commands cannot use duty-ratio dead-time compensation"
            )
        self.T = cfg.period
        actuation = (PeriodicClock(unit.pwm.load_period, unit.pwm.grid_offset)
                     if pwm_timed else PeriodicClock(
                         unit.ctrl.period,
                         (unit.ctrl.offset + unit.ctrl.computation) % unit.ctrl.period,
                     ))
        ratio = self.T / actuation.period
        if not math.isclose(ratio, round(ratio), rel_tol=1e-9) or ratio < 1:
            raise ConfigError(
                f"{type(self).__name__}.period must be a whole multiple of the actuation period"
            )
        t0 = unit.ctrl.offset
        index = actuation.index_at_or_after(t0 + unit.ctrl.computation)
        if unit.ctrl.computation > 0:
            index = max(index, actuation.index_after(t0))
        delay = actuation.at(index) - t0
        if delay < -TIME_EPS or delay > self.T + TIME_EPS:
            raise ConfigError("predictive-control execution delay must be within one loop period")
        self.delay = min(self.T, max(0.0, delay))
        self.plant = self._configure_plant(unit)
        if cfg.algorithm == "fcs":
            input_min = _numerical_configuration(cfg.input_min)
            input_max = _numerical_configuration(cfg.input_max)
            actions = np.array(list(product((0.0, 1.0), repeat=self.n_input)))
            self.actions = np.array([
                action for action in actions
                if all(input_min <= float(value) <= input_max for value in action)
            ])
            if len(self.actions) == 0:
                raise ConfigError("input bounds exclude every finite control action")
            default = self.actions[0]
        else:
            default = [cfg.input_min] * self.n_input
        fixed = default if cfg.fallback_input is None else cfg.fallback_input
        self.fixed_input = np.asarray(_numerical_configuration(fixed), dtype=float).copy()
        if self.fixed_input.shape != (self.n_input,):
            raise ConfigError(
                "fallback_input must contain one finite admissible command per controlled leg"
            )
        self.previous_input = self.fixed_input.copy()
        self.previous_enabled = False
        self.solve_count = 0
        self.fallback_count = 0
        self.solve_seconds = 0.0
        self.last_result: MPCResult | None = None
        self._last_diagnostics = self._diagnostics(None, False)
        if cfg.algorithm == "ccs":
            numerical_plant = _RLCurrentPrediction(
                _numerical_configuration(self.plant.L),
                _numerical_configuration(self.plant.R),
                _numerical_configuration(self.plant.period),
                _numerical_configuration(self.plant.delay),
                self.plant.voltage_map,
                blocked_min=self.plant.blocked_min,
            )
            template = numerical_plant.model(
                _numerical_configuration(unit.dclink.vdc_ref), np.zeros(self.n_current),
                np.zeros(self.n_current), np.zeros(self.n_current),
                previous_applied=True,
            )
            self.optimizer = CCSMPC(
                template, _numerical_configuration(cfg.prediction_horizon),
                (_numerical_configuration(cfg.control_horizon)
                 if cfg.control_horizon is not None else None),
                tracking_weight=_numerical_configuration(cfg.tracking_weight),
                input_weight=_numerical_configuration(cfg.input_weight),
                rate_weight=_numerical_configuration(cfg.rate_weight),
                input_bounds=(_numerical_configuration(cfg.input_min),
                              _numerical_configuration(cfg.input_max)),
                output_bounds=(
                    -np.inf if cfg.current_min is None
                    else _numerical_configuration(cfg.current_min),
                    np.inf if cfg.current_max is None
                    else _numerical_configuration(cfg.current_max),
                ), max_iterations=_numerical_configuration(cfg.max_iterations),
                tolerance=_numerical_configuration(cfg.tolerance),
                feasibility_tolerance=_numerical_configuration(cfg.feasibility_tolerance),
            )
        else:
            self._cost = _QuadraticStageCost(
                self.n_current, self.n_input,
                _numerical_configuration(cfg.tracking_weight),
                _numerical_configuration(cfg.input_weight),
                _numerical_configuration(cfg.rate_weight),
            )
            self.optimizer = FCSMPC(
                self.actions, _numerical_configuration(cfg.prediction_horizon),
                self._predict, self._stage_cost,
                control_horizon=(_numerical_configuration(cfg.control_horizon)
                                 if cfg.control_horizon is not None else None),
                constraint=self._constraint,
                max_evaluations=_numerical_configuration(cfg.max_evaluations),
                feasibility_tolerance=_numerical_configuration(cfg.feasibility_tolerance),
                vectorized=True,
            )

    def _configure_plant(self, unit) -> _RLCurrentPrediction:
        raise NotImplementedError

    def _rl_plant(self, unit, inductance, resistance, voltage_map, *, blocked_min=None):
        """Build the shared RL predictor after checking its physical model boundary."""
        if unit.bridge.parallel_capacitance > 0.0:
            raise ConfigError(
                f"{type(self).__name__} requires a bridge without internal energy storage"
            )
        return _RLCurrentPrediction(
            inductance, resistance, self.T, self.delay, voltage_map,
            blocked_min=blocked_min,
        )

    def _forecast(self, inputs: Mapping[str, Any]):
        raise NotImplementedError

    def _disturbance(self, inputs: Mapping[str, Any], start: float,
                     duration: float) -> np.ndarray:
        raise NotImplementedError

    def _write_action(self, outputs: dict[str, Any], action: np.ndarray) -> None:
        raise NotImplementedError

    def prediction(self, t: float, inputs: Mapping[str, Any]):
        """Build affine prediction data without reading or mutating the running plant."""
        current, vdc, references = self._forecast(inputs)
        models = []
        for k in range(self.cfg.prediction_horizon):
            start = k * self.T
            models.append(self.plant.model(
                vdc,
                current,
                self._disturbance(inputs, start, self.delay),
                self._disturbance(inputs, start + self.delay, self.T - self.delay),
                previous_applied=k > 0 or self.previous_enabled,
            ))
        return np.r_[current, self.previous_input], references, models

    def _predict(self, state, action, k):
        model = self._models[k]
        return state @ model.A.T + action @ model.B.T + model.offset

    def _stage_cost(self, state, action, reference, previous, k):
        return self._cost.evaluate(state[..., :self.n_current], action, reference, previous)

    def _constraint(self, state, action, k):
        current = state[..., :self.n_current]
        residuals = []
        if self.cfg.current_min is not None:
            residuals.append(current - self.cfg.current_min)
        if self.cfg.current_max is not None:
            residuals.append(self.cfg.current_max - current)
        return np.concatenate(residuals, axis=-1) if residuals else np.empty((*state.shape[:-1], 0))

    def solve(self, t: float, inputs: Mapping[str, Any]) -> MPCResult:
        """Optimize once; a custom registered loop may override this method."""
        state, reference, self._models = self.prediction(t, inputs)
        if self.cfg.algorithm == "ccs":
            return self.optimizer.solve(
                state, reference, self.previous_input, models=self._models,
            )
        return self.optimizer.solve(state, reference, self.previous_input)

    def _admissible(self, action) -> bool:
        if not self._valid_history(action):
            return False
        action = np.asarray(action)
        input_min = _numerical_configuration(self.cfg.input_min)
        input_max = _numerical_configuration(self.cfg.input_max)
        if any(not input_min <= float(value) <= input_max for value in action):
            return False
        return (self.cfg.algorithm != "fcs"
                or bool(np.any(np.all(self.actions == action, axis=1))))

    def _valid_history(self, action) -> bool:
        """Whether an action can represent a previously applied bridge command."""
        action = np.asarray(action)
        if (action.shape != (self.n_input,) or action.dtype.kind not in "fiu"
                or not np.all(np.isfinite(action))
                or np.any(action < 0.0) or np.any(action > 1.0)):
            return False
        return self.cfg.algorithm != "fcs" or bool(np.all((action == 0.0) | (action == 1.0)))

    def _diagnostics(self, result, valid):
        return {
            "success": float(valid),
            "objective": (float(result.objective)
                          if result is not None and result.objective is not None else 0.0),
            "iterations": float(result.iterations) if result is not None else 0.0,
            "evaluations": float(result.evaluations) if result is not None else 0.0,
            "solves": float(self.solve_count),
            "fallbacks": float(self.fallback_count),
            "delay": self.delay,
        }

    def initial_outputs(self):
        outputs = {"enable": float(self.previous_enabled)}
        self._write_action(outputs, self.previous_input)
        return outputs

    def sample_into(self, t, inputs, outputs):
        result = None
        enable = self.startup.active
        if enable:
            start = perf_counter()
            result = self.solve(t, inputs)
            elapsed = perf_counter() - start
            self.solve_seconds += elapsed
            self.solve_count += 1
            valid = result.success and result.action is not None and self._admissible(result.action)
            if result.success and not valid:
                result = replace(
                    result, success=False, action=None, status="invalid_solution",
                    message="Optimizer returned an inadmissible actuator command",
                )
            if valid:
                action = np.asarray(result.action, dtype=float).copy()
            else:
                self.fallback_count += 1
                action = (self.previous_input.copy()
                          if self.cfg.fallback == "hold" and self._admissible(self.previous_input)
                          else self.fixed_input.copy())
                enable = self.cfg.fallback != "block"
        else:
            action, valid = self.fixed_input.copy(), False
        self.previous_input = action
        self.previous_enabled = enable
        self.last_result = result
        self._last_diagnostics = self._diagnostics(result, valid)
        self._write_action(outputs, action)
        outputs["enable"] = float(enable)

    def sample(self, t, inputs):
        outputs = self.initial_outputs()
        self.sample_into(t, inputs, outputs)
        return outputs

    def get_state(self):
        return {
            **{f"previous_input.{i}": float(value)
               for i, value in enumerate(self.previous_input)},
            "previous_enabled": self.previous_enabled,
            "solve_count": self.solve_count,
            "fallback_count": self.fallback_count,
        }

    @property
    def discrete_state_names(self):
        names = ("previous_enabled", "solve_count", "fallback_count")
        if self.cfg.algorithm == "fcs":
            return tuple(f"previous_input.{i}" for i in range(self.n_input)) + names
        return names

    @property
    def linearization_state_names(self):
        return tuple(f"previous_input.{i}" for i in range(self.n_input)) + ("previous_enabled",)

    def periodic_state(self, values):
        return {name: value for name, value in values.items()
                if name not in ("solve_count", "fallback_count")}

    def set_state(self, values):
        unknown = set(values) - set(self.get_state())
        if unknown:
            raise KeyError(f"{type(self).__name__} has no states {sorted(unknown)}")
        action = self.previous_input.tolist()
        enabled = self.previous_enabled
        counters = {
            "solve_count": self.solve_count,
            "fallback_count": self.fallback_count,
        }
        for key, value in values.items():
            if key.startswith("previous_input."):
                action[int(key.rsplit(".", 1)[1])] = value
            elif key == "previous_enabled":
                if value not in (0, 1):
                    raise ValueError("previous_enabled must be 0 or 1")
                enabled = bool(value)
            else:
                if not math.isfinite(value) or value < 0 or value != int(value):
                    raise ValueError(f"{key} must be a nonnegative integer")
                counters[key] = int(value)
        action = np.asarray(action)
        if not self._valid_history(action):
            raise ValueError("previous input must be a valid finite bridge command")
        self.previous_input = np.asarray(action, dtype=float)
        self.previous_enabled = enabled
        self.solve_count = counters["solve_count"]
        self.fallback_count = counters["fallback_count"]

    def summary(self):
        return {
            "algorithm": self.cfg.algorithm,
            "solves": self.solve_count,
            "fallbacks": self.fallback_count,
            "solve_seconds": self.solve_seconds,
            "delay": self.delay,
            "status": None if self.last_result is None else self.last_result.status,
        }

    def observe(self):
        return dict(self._last_diagnostics)


@register_loop_type
class PredictiveCurrentControl(_PredictiveCurrentControl):
    """Scalar inductor-current MPC which directly supplies a converter duty ratio."""

    @dataclass(frozen=True, kw_only=True)
    class Params(_PredictiveCurrentParams):
        type: str = "predictive_current_control"

    type = "predictive_current_control"
    role = "current"
    native_generator = _DC_NATIVE_GENERATOR
    n_current, n_input = 1, 1
    inputs = {
        "reference": SI_CURRENT,
        "current": SI_CURRENT,
        "voltage": SI_VOLTAGE,
        "vin": SI_VOLTAGE,
    }
    outputs = {
        "duty": DUTY_RATIO,
        "enable": ENABLE,
    }

    def _configure_plant(self, unit):
        bridge = unit.bridge
        if (bridge.type != "half_bridge" or not bridge.upper_controlled
                or not bridge.lower_controlled or bridge.reverse_port
                or bridge.external_port != "midpoint"):
            raise ConfigError(
                "PredictiveCurrentControl requires a synchronous Buck half bridge"
            )
        if unit.filter is None:
            raise ConfigError("PredictiveCurrentControl requires an inductive filter")
        return self._rl_plant(
            unit,
            unit.filter.l_f,
            unit.filter.r_f + bridge.switch_resistance,
            np.ones((1, 1)),
            blocked_min=0.0,
        )

    def _forecast(self, inputs):
        references = np.full(
            (self.cfg.prediction_horizon, 1),
            inputs["reference"] * self.startup.value,
        )
        return np.array([inputs["current"]]), inputs["vin"], references

    def _disturbance(self, inputs, start, duration):
        return np.array([inputs["voltage"] * self.plant.integral(duration)])

    def _write_action(self, outputs, action):
        outputs["duty"] = float(action[0])


@register_loop_type
class ThreePhasePredictiveCurrentControl(_PredictiveCurrentControl):
    """Three-phase current MPC which directly supplies all bridge duty ratios."""

    @dataclass(frozen=True, kw_only=True)
    class Params(_PredictiveCurrentParams):
        type: str = "three_phase_predictive_current_control"

    type = "three_phase_predictive_current_control"
    role = "modulation"
    native_generator = _THREE_PHASE_NATIVE_GENERATOR
    n_current, n_input = 2, 3
    inputs = {
        "v": V_AB,
        "i": I_AB,
        "frame": ANGLE,
        "omega": FREQUENCY,
        "id_ref": I,
        "iq_ref": I,
        "u_dc": V,
    }
    outputs = {
        "d_a": DUTY_RATIO,
        "d_b": DUTY_RATIO,
        "d_c": DUTY_RATIO,
        "enable": ENABLE,
    }

    def _configure_plant(self, unit):
        if unit.bridge.type != "three_phase_bridge":
            raise ConfigError(
                "ThreePhasePredictiveCurrentControl requires a two-level three-phase bridge"
            )
        vectors = [abc2complex(row) for row in np.eye(3)]
        plant = self._rl_plant(
            unit,
            unit.ac_filter.l_f,
            unit.ac_filter.r_f + unit.bridge.switch_resistance,
            np.array([
                [value.real for value in vectors],
                [value.imag for value in vectors],
            ]),
        )
        self.i_base = unit.base.i_phase_peak
        self.v_base = unit.base.v_phase_peak
        self.vdc_base = unit.dc_base.v
        return plant

    def _forecast(self, inputs):
        current = inputs["i"] * self.i_base
        target = complex(inputs["id_ref"], inputs["iq_ref"]) * self.i_base * self.startup.value
        frame, omega = inputs["frame"], inputs["omega"]
        references = np.array([
            [value.real, value.imag]
            for value in (
                target * cmath.exp(1j * (frame + omega * (k + 1) * self.T))
                for k in range(self.cfg.prediction_horizon)
            )
        ])
        return np.array([current.real, current.imag]), inputs["u_dc"] * self.vdc_base, references

    def _disturbance(self, inputs, start, duration):
        omega = inputs["omega"]
        denominator = self.plant.R + 1j * omega * self.plant.L
        factor = (duration / self.plant.L if denominator == 0 else
                  (cmath.exp(1j * omega * duration)
                   - math.exp(-self.plant.R * duration / self.plant.L)) / denominator)
        integral = inputs["v"] * self.v_base * cmath.exp(1j * omega * start) * factor
        return np.array([integral.real, integral.imag])

    def _write_action(self, outputs, action):
        outputs["d_a"], outputs["d_b"], outputs["d_c"] = map(float, action)
