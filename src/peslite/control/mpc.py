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
from itertools import product
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
    status = "optimal" if result.success else {
        4: "infeasible", 9: "iteration_limit",
    }.get(result.get("status"), "solver_failure")
    return SolverResult(result.x, bool(result.success), status,
                        int(result.get("nit", 0)), int(result.get("nfev", 0)), str(result.message))


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
        self.Q = _weight(tracking_weight, self.ny, "tracking_weight")
        self.R = _weight(input_weight, self.nu, "input_weight")
        self.S = _weight(rate_weight, self.nu, "rate_weight")
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
                (output_response, output_free - refs[k], self.Q),
                (selection, np.zeros(self.nu), self.R),
                (selection - prior_response, -prior_free, self.S),
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
        problem = QuadraticProblem(
            hessian, linear, constant, lower.ravel(), upper.ravel(),
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
            return MPCResult(False, None, None, None, None, result.status,
                             result.iterations, result.evaluations, result.message)
        z = np.asarray(result.solution)
        valid = (z.dtype.kind in "fiu" and z.shape == (size,) and np.all(np.isfinite(z)))
        if valid:
            tolerance = self.feasibility_tolerance
            constraint = constraint_matrix @ z
            valid = (np.all(z >= problem.lower - tolerance) and np.all(z <= problem.upper + tolerance)
                     and np.all(constraint >= constraint_lower - tolerance)
                     and np.all(constraint <= constraint_upper + tolerance))
        if not valid:
            return MPCResult(False, None, None, None, None, "invalid_solution",
                             result.iterations, result.evaluations,
                             "Backend solution is non-finite, has the wrong shape or violates constraints")
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


class FCSMPC:
    """Exhaustive finite-control-set MPC with deterministic configured-order ties.

    ``prediction(x, u, k)`` returns the next state. ``objective(x_next, u,
    reference[k], u_previous, k)`` returns a scalar stage cost. An optional
    ``constraint(x_next, u, k)`` returns residuals that must be nonnegative
    within ``feasibility_tolerance``. Callbacks receive copies, so they cannot
    mutate the input state, action set, reference, or another candidate plan.
    Callbacks must be deterministic and must not capture mutable plant state.
    """

    def __init__(self, actions, prediction_horizon, prediction, objective, *,
                 control_horizon=None, constraint=None, max_evaluations=100_000,
                 feasibility_tolerance=1e-7):
        self.actions = _array(actions, "actions")
        if self.actions.ndim == 1:
            self.actions = self.actions[:, None]
        if self.actions.ndim != 2 or not all(self.actions.shape):
            raise ValueError("actions must be a nonempty matrix of candidate input vectors")
        self.actions.setflags(write=False)
        self.prediction_horizon = _positive_integer(prediction_horizon, "prediction_horizon")
        self.control_horizon = (self.prediction_horizon if control_horizon is None
                                else _positive_integer(control_horizon, "control_horizon"))
        if self.control_horizon > self.prediction_horizon:
            raise ValueError("control_horizon must not exceed prediction_horizon")
        if not callable(prediction) or not callable(objective) or (constraint is not None and not callable(constraint)):
            raise TypeError("prediction, objective and optional constraint must be callable")
        self.prediction, self.objective, self.constraint = prediction, objective, constraint
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
        best_cost, best_inputs, best_states, evaluations = np.inf, None, None, 0
        for indices in product(range(len(self.actions)), repeat=self.control_horizon):
            if evaluations == self.max_evaluations:
                return MPCResult(False, None, None, None, None, "evaluation_limit",
                                 evaluations=evaluations,
                                 message="Finite-set search budget exhausted before all plans were checked")
            evaluations += 1
            states, inputs, cost, last, feasible = [x0.copy()], [], 0.0, previous.copy(), True
            for k in range(self.prediction_horizon):
                action = self.actions[indices[min(k, self.control_horizon - 1)]]
                try:
                    next_state = np.asarray(self.prediction(states[-1].copy(), action.copy(), k))
                    if np.iscomplexobj(next_state) or next_state.shape != x0.shape:
                        raise ValueError("prediction must return a real vector with the initial state's shape")
                    if not np.all(np.isfinite(next_state)):
                        raise FloatingPointError("prediction returned a non-finite state")
                    next_state = next_state.astype(float, copy=True)
                    if self.constraint is not None:
                        residuals = np.asarray(self.constraint(next_state.copy(), action.copy(), k))
                        if np.iscomplexobj(residuals) or residuals.dtype.kind == "b" or residuals.ndim > 1:
                            raise ValueError("constraint must return real nonnegative residuals, not booleans")
                        if not np.all(np.isfinite(residuals)):
                            raise FloatingPointError("constraint returned non-finite residuals")
                        if np.any(residuals < -self.feasibility_tolerance):
                            feasible = False
                            break
                    stage_cost = float(self.objective(next_state.copy(), action.copy(), reference[k].copy(), last.copy(), k))
                    cost += stage_cost
                    if not np.isfinite(cost):
                        raise FloatingPointError("objective returned a non-finite cost")
                except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                    return MPCResult(False, None, None, None, None, "numerical_failure",
                                     evaluations=evaluations, message=str(exc))
                states.append(next_state)
                inputs.append(action.copy())
                last = action
            if feasible and cost < best_cost:
                best_cost, best_inputs, best_states = cost, np.asarray(inputs), np.asarray(states)
        if best_inputs is None:
            return MPCResult(False, None, None, None, None, "infeasible", evaluations=evaluations,
                             message="No admissible finite-action plan satisfies the constraints")
        return MPCResult(True, best_inputs[0].copy(), best_inputs, best_states, float(best_cost),
                         "optimal", evaluations=evaluations)
