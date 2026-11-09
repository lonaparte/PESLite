"""Pure numerical MPC algorithms, independent of the simulation and its state.

Models use the caller's coordinates and units. At stage ``k``, the input ``u[k]``
produces ``x[k + 1]`` and ``y[k + 1] = C[k] @ x[k + 1]``. Tracking costs and
state/output bounds apply to these predicted endpoints, not the initial state.
The final control move is held when the control horizon is shorter than the
prediction horizon. Work limits count solver iterations or complete FCS plans;
they are deterministic and do not impose a wall-clock deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, product
from numbers import Integral
from typing import Callable

import numpy as np

__all__ = ["LinearModel", "QuadraticProblem", "SolverResult", "MPCResult",
           "CCSMPC", "FCSMPC", "solve_slsqp"]


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


def _weight(value, width, name):
    weight = _array(value, name)
    if weight.ndim == 0:
        weight = np.eye(width) * float(weight)
    elif weight.shape == (width,):
        weight = np.diag(weight)
    if weight.shape != (width, width) or not np.array_equal(weight, weight.T):
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
        self.backend = solve_slsqp if backend is None else backend

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
            10.0 * np.sqrt(self.tolerance),
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
