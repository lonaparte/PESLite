"""Independent numerical references for continuous and finite-control-set MPC."""

from itertools import product

import numpy as np
import pytest

from peslite.control.mpc import CCSMPC, FCSMPC, LinearModel, SolverResult


def test_two_step_integrator_matches_analytic_unconstrained_and_active_bound_solutions():
    model = LinearModel([[1.0]], [[1.0]])
    # Objective stopping accuracy is not an action-error bound; request the
    # tighter solve needed for the independently specified 1e-8 action check.
    unconstrained = CCSMPC(model, 2, input_weight=1.0, tolerance=1e-14).solve([1.0], [0.0])
    assert unconstrained.success
    np.testing.assert_allclose(unconstrained.inputs[:, 0], [-0.6, -0.2], atol=1e-8)
    np.testing.assert_allclose(unconstrained.states[:, 0], [1.0, 0.4, 0.2], atol=1e-8)
    assert unconstrained.objective == pytest.approx(0.6, abs=1e-10)

    bounded = CCSMPC(model, 2, input_weight=1.0, input_bounds=(-0.4, 1.0),
                     tolerance=1e-14).solve([1.0], [0.0])
    assert bounded.success
    np.testing.assert_allclose(bounded.inputs[:, 0], [-0.4, -0.3], atol=1e-8)
    assert bounded.objective == pytest.approx(0.7, abs=1e-10)


@pytest.mark.parametrize("scale", [1.0, 1e6])
@pytest.mark.parametrize("bounded", [False, True])
def test_positive_objective_scaling_preserves_the_analytic_minimizer(scale, bounded):
    result = CCSMPC(LinearModel([[1.0]], [[1.0]]), 2,
                    tracking_weight=scale, input_weight=scale,
                    input_bounds=(-0.4, 1.0) if bounded else None).solve([1.0], [0.0])
    assert result.success
    expected = [-0.4, -0.3] if bounded else [-0.6, -0.2]
    np.testing.assert_allclose(result.inputs[:, 0], expected, atol=2e-6)
    assert result.objective / scale == pytest.approx(0.7 if bounded else 0.6, abs=1e-10)


@pytest.mark.parametrize("gain,target", [(100.0, 100.0), (100.0, 1000.0),
                                         (1000.0, 100.0), (1000.0, 1000.0)])
def test_physical_current_scales_match_independent_box_constrained_quadratic_solution(gain, target):
    # Two current endpoints driven by bounded duty ratios, with an input-effort penalty.
    response = gain * np.array([[1.0, 0.0], [1.0, 1.0]])
    hessian = response.T @ response + np.eye(2)
    linear = -response.T @ np.full(2, target)
    expected = -np.linalg.solve(hessian, linear)
    if gain == 100.0 and target == 1000.0:
        # Both upper bounds are active: the gradient is negative in each coordinate.
        expected = np.ones(2)
        assert np.all(hessian @ expected + linear < 0.0)
    else:
        assert np.all((expected > 0.0) & (expected < 1.0))
    result = CCSMPC(LinearModel([[1.0]], [[gain]]), 2,
                    input_weight=1.0, input_bounds=(0.0, 1.0), tolerance=1e-12).solve([0.0], [target])
    assert result.success
    np.testing.assert_allclose(result.inputs[:, 0], expected, atol=2e-6)
    direct_cost = np.sum((response @ result.inputs[:, 0] - target) ** 2) + np.sum(result.inputs ** 2)
    assert result.objective == pytest.approx(direct_cost, rel=1e-9, abs=1e-8)


def test_full_matrix_tracking_and_increment_weights_match_independent_normal_equations():
    a = np.array([[0.8, 0.1], [0.0, 0.9]])
    b = np.array([[1.0, 0.2], [-0.1, 0.5]])
    c = np.array([[1.0, 0.4], [0.2, -1.0]])
    q = np.array([[2.0, 0.3], [0.3, 1.0]])
    r = np.array([[0.2, 0.05], [0.05, 0.4]])
    s = np.array([[0.7, 0.1], [0.1, 0.5]])
    initial, previous = np.array([1.0, -0.2]), np.array([0.4, -0.3])
    reference = np.array([[0.2, 0.1], [0.3, -0.1]])
    # Independently stack the two endpoint equations and both input differences.
    response = np.block([[c @ b, np.zeros((2, 2))], [c @ a @ b, c @ b]])
    free = np.r_[c @ a @ initial, c @ a @ a @ initial] - reference.ravel()
    difference = np.block([[np.eye(2), np.zeros((2, 2))], [-np.eye(2), np.eye(2)]])
    block_q, block_r, block_s = (np.kron(np.eye(2), weight) for weight in (q, r, s))
    hessian = response.T @ block_q @ response + block_r + difference.T @ block_s @ difference
    linear = response.T @ block_q @ free - difference.T @ block_s @ np.r_[previous, [0.0, 0.0]]
    expected = -np.linalg.solve(hessian, linear)
    solver = CCSMPC(LinearModel(a, b, c), 2, tracking_weight=q, input_weight=r, rate_weight=s,
                    tolerance=1e-14)
    result = solver.solve(initial, reference, previous)
    assert result.success
    np.testing.assert_allclose(result.inputs.ravel(), expected, atol=2e-7)


def test_state_and_nonidentity_output_constraints_have_endpoint_semantics():
    model = LinearModel(np.eye(2), np.eye(2), [[1.0, 1.0]])
    result = CCSMPC(model, 2, input_weight=0.1,
                    input_bounds=([-0.5, -0.5], [0.5, 0.5]),
                    state_bounds=([-0.1, -0.1], [0.2, 0.6]),
                    output_bounds=(-np.inf, 0.5)).solve([0.0, 0.0], [2.0])
    assert result.success
    assert np.max(result.states[1:, 0]) <= 0.2 + 1e-7
    assert np.max(result.states[1:, 1]) <= 0.6 + 1e-7
    np.testing.assert_allclose(result.states[1:].sum(axis=1), [0.5, 0.5], atol=1e-7)
    assert np.all(np.abs(result.inputs) <= 0.5 + 1e-7)


def test_short_control_horizon_holds_the_last_move_and_accounts_for_rate_cost_once():
    # With one held input: J=(1+u)^2+(1+2u)^2+(1+3u)^2+3u^2+(u-0.5)^2.
    result = CCSMPC(LinearModel([[1.0]], [[1.0]]), 3, 1,
                    input_weight=1.0, rate_weight=1.0).solve([1.0], [0.0], [0.5])
    assert result.success
    np.testing.assert_allclose(result.inputs, np.full((3, 1), -5.5 / 18.0), atol=1e-8)


def test_stage_dependent_models_and_affine_offsets_match_direct_quadratic_reference():
    models = (LinearModel([[0.8]], [[0.5]], offset=[0.3]),
              LinearModel([[0.9]], [[0.7]], offset=[-0.2]))
    # x1=.8+.5u0+.3; x2=.9x1+.7u1-.2.
    response = np.array([[0.5, 0.0], [0.45, 0.7]])
    free = np.array([1.1, 0.79])
    expected = -np.linalg.solve(response.T @ response + np.eye(2), response.T @ free)
    result = CCSMPC(models[0], 2, input_weight=1.0, tolerance=1e-14).solve([1.0], [0.0], models=models)
    assert result.success
    np.testing.assert_allclose(result.inputs[:, 0], expected, atol=1e-7)


def test_held_move_intersects_all_stage_input_bounds():
    controller = CCSMPC(LinearModel([[1.0]], [[1.0]]), 3, 1,
                        input_bounds=([[-1.0], [-0.3], [-0.2]], [[1.0], [0.5], [0.4]]))
    result = controller.solve([1.0], [0.0])
    assert result.success
    np.testing.assert_allclose(result.inputs, -0.2, atol=1e-8)
    impossible = CCSMPC(LinearModel([[1.0]], [[1.0]]), 2, 1,
                         input_bounds=([[0.0], [0.5]], [[0.2], [1.0]])).solve([0.0], [0.0])
    assert not impossible.success and impossible.status == "infeasible"
    assert impossible.action is None


def test_backend_receives_condensed_problem_and_its_solution_is_independently_checked():
    calls = []

    def exact_backend(problem):
        calls.append(problem)
        return SolverResult(-np.linalg.solve(problem.hessian, problem.linear), True, "optimal", 1, 1)

    controller = CCSMPC(LinearModel([[1.0]], [[1.0]]), 2, input_weight=1.0, backend=exact_backend)
    result = controller.solve([1.0], [0.0])
    assert len(calls) == 1 and result.success
    assert result.objective == pytest.approx(0.6)
    np.testing.assert_allclose(calls[0].hessian, [[6.0, 2.0], [2.0, 4.0]])
    np.testing.assert_allclose(calls[0].linear, [4.0, 2.0])


@pytest.mark.parametrize("solution", [[np.nan, 0.0], [0.0], [2.0, 0.0], [0.0 + 1j, 0.0], ["bad", "data"]])
def test_invalid_backend_solution_never_becomes_an_action(solution):
    controller = CCSMPC(LinearModel([[1.0]], [[1.0]]), 2, input_bounds=(-1.0, 1.0),
                        backend=lambda _problem: SolverResult(np.asarray(solution), True, "optimal"))
    result = controller.solve([0.0], [0.0])
    assert not result.success and result.status == "invalid_solution"
    assert result.action is None


def test_solver_limits_and_numerical_failures_are_explicit_and_programming_errors_propagate():
    controller = CCSMPC(LinearModel([[1.0]], [[1.0]]), 3, input_weight=1.0, max_iterations=1)
    result = controller.solve([3.0], [0.0])
    assert not result.success and result.status == "iteration_limit"
    assert result.action is None

    def failed_backend(_problem):
        raise np.linalg.LinAlgError("factorization failed")

    failed = CCSMPC(controller.model, 2, backend=failed_backend).solve([1.0], [0.0])
    assert not failed.success and failed.status == "numerical_failure"

    def broken_backend(_problem):
        raise TypeError("programming error")

    with pytest.raises(TypeError, match="programming error"):
        CCSMPC(controller.model, 2, backend=broken_backend).solve([1.0], [0.0])


def test_infeasible_state_constraints_fail_without_applying_invalid_control():
    controller = CCSMPC(LinearModel([[1.0]], [[0.0]]), 2, input_weight=1.0,
                        input_bounds=(-1.0, 1.0), state_bounds=(-0.1, 0.1))
    result = controller.solve([1.0], [0.0])
    assert not result.success and result.action is None
    assert result.status in ("infeasible", "solver_failure")


def test_equal_input_bounds_produce_the_only_admissible_plan():
    result = CCSMPC(LinearModel([[1.0]], [[1.0]]), 2,
                    input_bounds=(0.2, 0.2)).solve([0.0], [1.0])
    assert result.success
    np.testing.assert_array_equal(result.inputs, [[0.2], [0.2]])


def test_fixed_inputs_and_infeasible_state_constraints_return_failure():
    result = CCSMPC(LinearModel([[1.0]], [[1.0]]), 2,
                    input_bounds=(0.0, 0.0), state_bounds=(-0.1, 0.1)).solve([1.0], [0.0])
    assert not result.success and result.action is None
    assert result.status in ("infeasible", "solver_failure")


def test_model_and_solve_inputs_are_not_modified():
    a, b, offset = np.eye(2), np.eye(2), np.array([0.1, -0.1])
    model = LinearModel(a, b, offset=offset)
    a[0, 0], b[0, 0], offset[0] = 5.0, 5.0, 5.0
    np.testing.assert_array_equal(model.A, np.eye(2))
    np.testing.assert_array_equal(model.B, np.eye(2))
    np.testing.assert_array_equal(model.offset, [0.1, -0.1])
    state, reference, previous, guess = np.ones(2), np.zeros(2), np.ones(2), np.ones((2, 2))
    before = [value.copy() for value in (state, reference, previous, guess)]
    CCSMPC(model, 2, input_weight=1.0).solve(state, reference, previous, initial_plan=guess)
    for actual, expected in zip((state, reference, previous, guess), before):
        np.testing.assert_array_equal(actual, expected)
    assert not model.A.flags.writeable


@pytest.mark.parametrize("kwargs,match", [
    ({"prediction_horizon": 0}, "prediction_horizon"),
    ({"prediction_horizon": True}, "prediction_horizon"),
    ({"prediction_horizon": 1.5}, "prediction_horizon"),
    ({"control_horizon": 3}, "control_horizon"),
    ({"tracking_weight": -1.0}, "positive semidefinite"),
    ({"input_weight": np.nan}, "finite"),
    ({"input_bounds": (1.0, -1.0)}, "lower <= upper"),
    ({"state_bounds": (np.nan, 1.0)}, "NaN"),
    ({"output_bounds": (-1.0, -np.inf)}, "lower <= upper"),
    ({"max_iterations": 0}, "max_iterations"),
    ({"tolerance": 0.0}, "tolerance"),
    ({"feasibility_tolerance": np.inf}, "feasibility_tolerance"),
])
def test_invalid_ccs_configuration_is_rejected(kwargs, match):
    arguments = {"prediction_horizon": 2, **kwargs}
    with pytest.raises(ValueError, match=match):
        CCSMPC(LinearModel([[1.0]], [[1.0]]), **arguments)


@pytest.mark.parametrize("weight", [[[1.0, 1.0], [0.0, 1.0]], [[1.0, 2.0], [2.0, 1.0]], [1.0, 2.0, 3.0]])
def test_matrix_weights_must_be_symmetric_positive_semidefinite_and_dimensioned(weight):
    with pytest.raises(ValueError, match="tracking_weight"):
        CCSMPC(LinearModel(np.eye(2), np.eye(2)), 2, tracking_weight=weight)


@pytest.mark.parametrize("args", [([], [[1.0]]), ([[1.0, 0.0]], [[1.0]]),
                                   ([[1.0]], [[1.0], [2.0]]), ([[np.nan]], [[1.0]]),
                                   ([[1.0]], [[1.0]], [[1.0, 2.0]])])
def test_invalid_linear_model_dimensions_and_values_are_rejected(args):
    with pytest.raises(ValueError):
        LinearModel(*args)


def test_invalid_runtime_dimensions_are_rejected():
    controller = CCSMPC(LinearModel(np.eye(2), np.eye(2)), 2)
    for kwargs in ({"state": [1.0]}, {"reference": [1.0]}, {"previous_input": [1.0]},
                   {"initial_plan": np.ones((3, 2))}, {"models": [controller.model]}):
        with pytest.raises(ValueError):
            controller.solve(**{"state": [1.0, 0.0], "reference": [0.0, 0.0], **kwargs})


def test_fcs_matches_independent_nonlinear_enumeration_with_constraints_and_rate_cost():
    actions, horizon, previous = np.array([[-0.6], [0.2], [0.7]]), 3, np.array([0.2])
    reference = np.array([[0.4], [0.1], [-0.2]])

    def prediction(x, u, _k):
        return 0.8 * x + u + 0.05 * x ** 2

    def objective(x, u, ref, prior, _k):
        return float((x - ref) @ (x - ref) + 0.1 * u @ u + 0.2 * (u - prior) @ (u - prior))

    controller = FCSMPC(actions, horizon, prediction, objective,
                        constraint=lambda x, _u, _k: np.r_[x + 0.5, 0.8 - x])
    actual = controller.solve([0.5], reference, previous)
    candidates = []
    for choices in product([-0.6, 0.2, 0.7], repeat=horizon):
        x, prior, cost, valid = 0.5, 0.2, 0.0, True
        for k, u in enumerate(choices):
            x = 0.8 * x + u + 0.05 * x * x
            valid = valid and -0.5 <= x <= 0.8
            cost += (x - reference[k, 0]) ** 2 + 0.1 * u * u + 0.2 * (u - prior) ** 2
            prior = u
        if valid:
            candidates.append((cost, choices))
    expected = min(candidates)
    assert actual.success and actual.evaluations == len(actions) ** horizon
    assert actual.objective == pytest.approx(expected[0])
    np.testing.assert_array_equal(actual.inputs[:, 0], expected[1])


def test_fcs_short_horizon_holds_last_move_and_ties_keep_configured_order():
    controller = FCSMPC([[1.0], [-1.0]], 3, lambda x, u, _k: x + u,
                        lambda _x, _u, _ref, _previous, _k: 0.0, control_horizon=1)
    result = controller.solve([0.0], [0.0])
    assert result.success and result.evaluations == 2
    np.testing.assert_array_equal(result.inputs, [[1.0], [1.0], [1.0]])
    np.testing.assert_array_equal(result.states, [[0.0], [1.0], [2.0], [3.0]])


def test_fcs_infeasibility_budget_exhaustion_and_nonfinite_cost_have_distinct_statuses():
    arguments = dict(actions=[-1.0, 1.0], prediction_horizon=2,
                     prediction=lambda x, u, _k: x + u,
                     objective=lambda x, _u, _ref, _previous, _k: float(x @ x))
    infeasible = FCSMPC(**arguments, constraint=lambda _x, _u, _k: [-1.0]).solve([0.0], [0.0])
    assert infeasible.status == "infeasible" and infeasible.evaluations == 4
    limited = FCSMPC(**arguments, max_evaluations=3).solve([0.0], [0.0])
    assert limited.status == "evaluation_limit" and limited.evaluations == 3
    numerical = FCSMPC(**{**arguments, "objective": lambda *_args: np.nan}).solve([0.0], [0.0])
    assert numerical.status == "numerical_failure"
    for result in (infeasible, limited, numerical):
        assert not result.success and result.action is None


def test_fcs_callbacks_cannot_modify_caller_arrays_or_other_candidates():
    state, reference, previous, actions = np.array([0.2]), np.array([0.1]), np.array([0.0]), np.array([[-1.0], [1.0]])
    originals = [value.copy() for value in (state, reference, previous, actions)]

    def prediction(x, u, _k):
        x += u
        u[:] = 42.0
        return x

    def objective(x, u, ref, prior, _k):
        cost = float((x - ref) @ (x - ref))
        x[:], u[:], ref[:], prior[:] = 42.0, 42.0, 42.0, 42.0
        return cost

    controller = FCSMPC(actions, 2, prediction, objective)
    result = controller.solve(state, reference, previous)
    assert result.success
    for actual, expected in zip((state, reference, previous, actions), originals):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(controller.actions, originals[-1])
    np.testing.assert_allclose(result.inputs[:, 0], [-1.0, 1.0])


@pytest.mark.parametrize("kwargs,match", [
    ({"actions": []}, "actions"), ({"actions": [[np.nan]]}, "finite"),
    ({"prediction_horizon": 0}, "prediction_horizon"),
    ({"control_horizon": 4}, "control_horizon"),
    ({"max_evaluations": 0}, "max_evaluations"),
    ({"feasibility_tolerance": 0.0}, "feasibility_tolerance"),
])
def test_invalid_fcs_configuration_is_rejected(kwargs, match):
    arguments = dict(actions=[-1.0, 1.0], prediction_horizon=2,
                     prediction=lambda x, u, _k: x + u, objective=lambda *_args: 0.0)
    with pytest.raises(ValueError, match=match):
        FCSMPC(**{**arguments, **kwargs})


def test_fcs_custom_constraint_uses_residuals_instead_of_ambiguous_booleans():
    controller = FCSMPC([-1.0, 1.0], 2, lambda x, u, _k: x + u, lambda *_args: 0.0,
                        constraint=lambda *_args: False)
    with pytest.raises(ValueError, match="not booleans"):
        controller.solve([0.0], [0.0])
