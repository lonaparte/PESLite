"""Fixed-step implicit integration and circuit-model linear systems."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

import peslite
from conftest import EXAMPLES
from peslite.solver import AdaptiveSolver, AutoSolver, ConfigError, DormandPrince45, FixedStepSolver, make_solver
from peslite.solver.integrators import integrator
from peslite.solver.implicit import (ImplicitSolver, LinearSystem, ModeJacobian, Structure,
                                     _Factor, _ShiftFactor, _TABLEAUX, _increment, linear_system)


A_OSCILLATOR = np.array([[-1.0, 10.0], [-10.0, -1.0]])
QUIET = {"simulation.progress.enable": 0, "simulation.solver.linearisations": 0}


@pytest.mark.parametrize(("solver_type", "method", "canonical", "expected"), [
    ("FiXeD", "rK4", "RK4", FixedStepSolver),
    ("FIXED", "RaDaU", "radau", ImplicitSolver),
    ("AdApTiVe", "rAdAu", "radau", AdaptiveSolver),
    ("AdApTiVe", "AuTo", "auto", AutoSolver),
])
def test_solver_method_values_are_normalized(solver_type, method, canonical, expected):
    params = peslite.load(
        EXAMPLES / "gfl-example.pes",
        **{"simulation.solver.type": solver_type, "simulation.solver.method": method},
    )
    solver = make_solver(params.simulation.solver, peslite.System(params).model)
    assert params.simulation.solver.method == canonical
    assert type(solver) is expected


@pytest.mark.parametrize("method, order", [("trapezoidal", 2), ("radau", 5)])
def test_implicit_methods_have_their_design_order(method, order):
    y0 = np.array([1.0, 0.0])
    exact = expm(A_OSCILLATOR) @ y0
    errors = [
        np.max(np.abs(
            ImplicitSolver(dt, method, 1e-12, 1e-14)(
                lambda _t, y: A_OSCILLATOR @ y, 0.0, 1.0, y0
            ).y - exact
        ))
        for dt in (0.02, 0.01)
    ]
    assert math.log2(errors[0] / errors[1]) == pytest.approx(order, abs=0.15)


@pytest.mark.parametrize("lists", [False, True])
def test_auto_reuses_dp45_without_constructing_an_unused_implicit_solver(lists):
    class Oscillator:
        def rhs_list(self, t, y):
            return [y[1], -4.0 * y[0]]

        def rhs(self, t, y):
            return np.asarray(self.rhs_list(t, y))

    model = Oscillator()
    rhs = model.rhs if lists else lambda t, y: model.rhs(t, y)
    y0 = np.array([1.0, 0.0])
    solver = AutoSolver(rtol=1e-8, atol=1e-11)
    actual = solver(rhs, 0.0, 0.5, y0)
    expected = DormandPrince45(rtol=1e-8, atol=1e-11)(rhs, 0.0, 0.5, y0)
    np.testing.assert_array_equal(actual.y, expected.y)
    assert actual.n_rhs == expected.n_rhs
    assert solver.switches == 0 and "_implicit" not in vars(solver)


def test_auto_switches_for_stiffness_and_returns_when_the_dynamics_change():
    rate = 1e5
    calls = 0

    def rhs(t, y):
        nonlocal calls
        calls += 1
        return -rate * (y - math.cos(t)) - math.sin(t)

    solver = AutoSolver(rtol=1e-8, atol=1e-11)
    result = solver(rhs, 0.0, 0.05, np.array([2.0]))
    assert solver.stiff and solver.switches == 1
    assert result.n_rhs == calls < 3000
    np.testing.assert_allclose(result.y, [math.cos(0.05)], rtol=1e-8, atol=1e-10)
    rate = 1.0
    solver.parameters_changed()
    for k in range(6):
        result = solver(rhs, 0.05 + k * 0.02, 0.05 + (k + 1) * 0.02, result.y)
    assert not solver.stiff and solver.switches == 2
    assert result.n_rhs == calls
    np.testing.assert_allclose(result.y, [math.cos(0.17)], rtol=1e-8, atol=1e-10)


def test_auto_keeps_the_implicit_solver_on_a_stiff_cable(tmp_path):
    params = peslite.load(EXAMPLES / "cable-gfl-example.pes", **QUIET, **{
        "simulation.t_end": 0.002,
        "simulation.energy_check": "off",
        "elements.cable1.sections": 8,
        "elements.cable1.length_km": 5.0,
        "simulation.solver.type": "adaptive",
        "simulation.solver.method": "auto",
        "simulation.solver.rtol": 1e-4,
        "simulation.solver.atol": 1e-6,
    })
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)
    assert simulation.solver.stiff and simulation.solver.switches == 1
    assert result.n_rhs < 2400


def test_radau_damps_a_stiff_mode_while_trapezoidal_keeps_it_bounded():
    def f(t, y):
        return np.array([-1e7 * (y[0] - math.cos(t)) - math.sin(t)])

    radau = ImplicitSolver(1e-3, "radau")(f, 0.0, 1.0, np.array([2.0])).y[0]
    trapezoidal = ImplicitSolver(1e-3, "trapezoidal")(f, 0.0, 1.0, np.array([2.0])).y[0]
    assert radau == pytest.approx(math.cos(1.0), rel=1e-9)
    assert abs(trapezoidal - math.cos(1.0)) > 0.1
    assert abs(trapezoidal) < 3.0


def test_radau_solves_each_complex_conjugate_pair_once(monkeypatch):
    original_eig = np.linalg.eig

    def reordered_eig(matrix):
        values, vectors = original_eig(matrix)
        order = np.roll(np.arange(values.size), 1)
        values, vectors = values[order], vectors[:, order]
        phases = (np.exp(0.37j), np.exp(-0.81j))
        for index, phase in zip(np.flatnonzero(np.abs(values.imag) > 1e-12), phases):
            vectors[:, index] *= phase
        return values, vectors

    monkeypatch.setattr(np.linalg, "eig", reordered_eig)
    solves = []
    original_solve = _ShiftFactor.solve

    def counted_solve(self, residual):
        solves.append(self)
        return original_solve(self, residual)

    monkeypatch.setattr(_ShiftFactor, "solve", counted_solve)
    matrix = np.array([[-4.0, 2.0], [-3.0, -5.0]])
    tableau = _TABLEAUX["radau"][0]
    owner = LinearSystem(Structure.single(matrix.shape[0]))
    factor = _Factor(matrix, owner.structure, tableau, 2e-3, owner, "radau")
    residual = np.random.default_rng(7).normal(size=(tableau.shape[0], matrix.shape[0]))

    actual = factor.solve(residual)
    expanded = np.eye(residual.size) - 2e-3 * np.kron(tableau, matrix)
    expected = np.linalg.solve(expanded, residual.ravel()).reshape(residual.shape)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert len(solves) == 2


def test_linear_system_reuses_returning_jacobians():
    modes = [np.array([[-1.0, 2.0], [-2.0, -1.0]]),
             np.array([[-2e4, 0.0], [1e3, -3e4]])]
    selected = {"mode": 0}

    def f(_t, y):
        return modes[selected["mode"]] @ y

    cached = LinearSystem(cache=True)
    solver = ImplicitSolver(1e-3, "trapezoidal", linear=cached)
    y = np.array([1.0, 1.0])
    for k in range(6):
        selected["mode"] = k % 2
        y = solver(f, 0.01 * k, 0.01 * (k + 1), y).y
    assert cached.n_jacobians == 4

    uncached = LinearSystem()
    solver = ImplicitSolver(1e-3, "trapezoidal", linear=uncached)
    y = np.array([1.0, 1.0])
    for k in range(6):
        selected["mode"] = k % 2
        y = solver(f, 0.01 * k, 0.01 * (k + 1), y).y
    assert uncached.n_jacobians == 12


def test_returning_mode_refreshes_its_exact_linear_block():
    selected = {"mode": 0, "matrix": A_OSCILLATOR.copy()}
    linear = LinearSystem(
        mode_key=lambda: selected["mode"],
        exact_blocks=lambda: ((np.arange(2), selected["matrix"]),),
    )
    y = np.array([1.0, 2.0])

    for mode, scale in ((0, 1.0), (1, 2.0), (0, 3.0)):
        selected["mode"] = mode
        selected["matrix"] = scale * A_OSCILLATOR
        rhs = lambda _t, state: selected["matrix"] @ state
        linear.ensure(rhs, 0.0, y, rhs(0.0, y))

    np.testing.assert_array_equal(linear.J, selected["matrix"])
    assert linear.n_jacobians == 0


def test_exact_submatrix_preserves_feedback_outside_the_netlist_block():
    matrix = np.array([[-1.0, 0.0], [2.0, -3.0]])
    linear = LinearSystem(
        exact_blocks=lambda: ((
            np.array([0]), np.array([[-1.0]]), np.array([0]), np.array([], dtype=int)
        ),),
    )
    y = np.array([1.0, 2.0])
    rhs = lambda _t, state: matrix @ state

    linear.ensure(rhs, 0.0, y, rhs(0.0, y))

    np.testing.assert_allclose(linear.J, matrix, rtol=1e-8, atol=1e-10)
    assert linear.J[0, 0] == -1.0


@pytest.mark.parametrize("form", ["device", "statewise", "nodal"])
def test_exact_circuit_block_retains_continuous_control_feedback(form):
    params = peslite.load(EXAMPLES / "gfl-example.pes", **{
        "units.vsc.bridge.model": "averaging",
        "simulation.solver.circuit_model": form,
        "events.connect_vsc.t": 0.0,
    })
    model = peslite.System(params).model
    y = model.get_initial_values()
    f0 = model.rhs(0.0, y)
    linear = linear_system(model)
    linear.ensure(model.rhs, 0.0, y, f0)

    circuit_states = np.asarray(model.linear_blocks()[0][0], dtype=int)
    other_states = np.setdiff1d(np.arange(model.n_states), circuit_states)
    increments = _increment(y)
    numerical = np.column_stack([
        (model.rhs(0.0, y + np.eye(1, model.n_states, index)[0] * increments[index]) - f0)
        / increments[index]
        for index in circuit_states
    ])
    feedback = numerical[other_states]

    assert np.any(feedback != 0.0)
    np.testing.assert_allclose(
        linear.J[np.ix_(other_states, circuit_states)], feedback, rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("method", ["trapezoidal", "radau"])
def test_returning_jacobian_reuses_its_newton_factorization(method):
    matrices = (A_OSCILLATOR, 100.0 * A_OSCILLATOR)
    selected = 0

    def rhs(_t, y):
        return matrices[selected] @ y

    linear = LinearSystem(cache=True)
    y = np.array([1.0, 2.0])
    h = 2.0**-10
    for selected in (0, 1, 0, 1, 0, 1):
        f0 = rhs(0.0, y)
        if linear.J is None:
            linear.ensure(rhs, 0.0, y, f0)
        else:
            linear.lookup(rhs, 0.0, y, f0)
        factor = linear.factor(h, method)
        residual = np.ones((factor.s, y.size))
        actual = factor.solve(residual)
        fresh = LinearSystem()
        fresh.J = linear.J.copy()
        fresh._setup(y.size)
        np.testing.assert_allclose(
            actual, fresh.factor(h, method).solve(residual), rtol=1e-12, atol=1e-12
        )
    assert linear.n_jacobians == 4
    factors = 1 if method == "trapezoidal" else 2
    assert linear.n_factorizations == 2 * factors

    matrices = tuple(2.0 * matrix for matrix in matrices)
    linear.invalidate()
    linear.ensure(rhs, 0.0, y, rhs(0.0, y))
    assert linear.factor(h, method) is not factor
    assert linear.n_factorizations == 3 * factors


def test_new_state_layout_discards_old_newton_factors():
    linear = LinearSystem(cache=True)
    for n in (1, 2):
        y = np.ones(n)
        rhs = lambda _t, state: -state
        linear.ensure(rhs, 0.0, y, rhs(0.0, y))
        actual = linear.factor(0.25, "trapezoidal").solve(np.ones((1, n)))
        np.testing.assert_allclose(actual, np.full((1, n), 1.0 / 1.125))


def test_a_single_interior_block_uses_one_dense_factorization():
    partition = Structure([np.array([0])], np.array([1]))
    linear = LinearSystem(partition, cache=True)
    y = np.ones(2)
    rhs = lambda _t, state: A_OSCILLATOR @ state
    linear.ensure(rhs, 0.0, y, rhs(0.0, y))
    actual = linear.factor(0.01, "trapezoidal").solve(np.ones((1, 2)))
    expected = np.linalg.solve(np.eye(2) - 0.005 * A_OSCILLATOR, np.ones(2))
    np.testing.assert_allclose(actual[0], expected, rtol=1e-8)
    assert linear.n_factorizations == 1


@pytest.mark.parametrize("method", ["RK45", "radau", "BDF", "LSODA", "RaDaU", "bdf"])
def test_adaptive_rhs_count_includes_numerical_jacobians(method):
    calls = 0

    def rhs(_t, y):
        nonlocal calls
        calls += 1
        return A_OSCILLATOR @ y

    result = AdaptiveSolver(method, rtol=1e-8, atol=1e-10)(
        rhs, 0.0, 0.01, np.array([1.0, 0.0])
    )
    assert result.n_rhs == calls


def test_adaptive_jacobian_uses_the_actual_integrated_rhs():
    def unused_rhs(_t, _y):
        raise AssertionError("the Jacobian evaluated a different RHS")

    calls = 0

    def rhs(_t, y):
        nonlocal calls
        calls += 1
        return A_OSCILLATOR @ y

    solver = integrator("BDF", 1e-3, 1e-10, 1e-12,
                        model=SimpleNamespace(rhs=unused_rhs), solver_type="adaptive")
    y0 = np.array([1.0, 0.0])
    result = solver(rhs, 0.0, 0.01, y0)
    np.testing.assert_allclose(result.y, expm(0.01 * A_OSCILLATOR) @ y0,
                               rtol=1e-8, atol=1e-10)
    assert result.n_rhs == calls


@pytest.mark.parametrize("form", ["device", "statewise", "nodal"])
def test_jacobian_includes_state_feedback_inside_a_source(form):
    params = peslite.load(
        EXAMPLES / "gfl-example.pes",
        **{"simulation.solver.circuit_model": form, "units.vsc.dclink.source.k": 10.0},
    )
    system = peslite.System(params)
    model = system.model
    dclink = system.units["vsc"].dclink
    state_owner = next(sub for sub in model.subsystems if sub.state is dclink.state)
    index = model.state_slice(state_owner, "u_C").start
    y = model.get_initial_values()
    linear = linear_system(model)
    linear.ensure(model.rhs, 0.0, y, model.rhs(0.0, y))
    assert linear.J[index, index] == pytest.approx(
        -dclink.source.k_dc / dclink.capacitor.C, rel=2e-6
    )


def test_nodal_group_solve_matches_the_dense_newton_system():
    model = peslite.System(peslite.load(EXAMPLES / "two-converters-example.pes")).model
    structure = model.linear_structure()
    assert structure.groups and structure.interface.size
    y = model.get_initial_values() + 1.0
    grouped = LinearSystem(structure, cache=True)
    grouped.ensure(model.rhs, 0.0, y, model.rhs(0.0, y))

    for method in ("trapezoidal", "radau"):
        stages = 3 if method == "radau" else 1
        residual = np.random.default_rng(1).normal(size=(stages, y.size))
        actual = grouped.factor(2e-5, method).solve(residual)
        dense = LinearSystem(Structure.single(y.size))
        dense.J = grouped.J
        dense._setup(y.size)
        expected = dense.factor(2e-5, method).solve(residual)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_assembly_and_solvers_share_one_circuit_structure():
    system = peslite.System(peslite.load(EXAMPLES / "two-converters-example.pes"))
    model = system.model
    linear = linear_system(model)

    assert not hasattr(model, "structural_analysis")
    assert linear.structure is model.linear_structure()
    assert linear.mode_key is model.linear_mode_key
    assert linear.exact_blocks is model.linear_blocks
    assert system.circuits and all(
        circuit._structure_plan is not None for circuit in system.circuits
    )


def test_make_solver_uses_each_configured_circuit_model():
    params = peslite.load(EXAMPLES / "two-converters-example.pes")
    model = peslite.System(params).model
    settings = params.simulation.solver

    device = make_solver(settings.__class__(
        **{**vars(settings), "method": "trapezoidal"}
    ), model)
    statewise = make_solver(settings.__class__(
        **{**vars(settings), "method": "radau", "circuit_model": "statewise"}
    ), model)
    nodal = make_solver(settings.__class__(
        **{**vars(settings), "method": "radau", "circuit_model": "nodal"}
    ), model)
    adaptive = make_solver(settings.__class__(
        **{**vars(settings), "type": "adaptive", "method": "Radau",
           "circuit_model": "statewise"}
    ), model)

    assert type(device) is ImplicitSolver and device.linear.cache
    assert type(statewise) is ImplicitSolver and statewise.linear.cache
    assert type(nodal) is ImplicitSolver and nodal.linear.structure.interface.size
    assert type(adaptive) is AdaptiveSolver and type(adaptive.jac) is ModeJacobian
    step = adaptive(model.rhs, 0.0, 1e-6, model.get_initial_values())
    assert np.all(np.isfinite(step.y))
    assert adaptive.jac.linear.n_jacobians < model.n_states


@pytest.mark.full
@pytest.mark.parametrize("method", ["trapezoidal", "radau", "auto"])
def test_circuit_models_preserve_the_same_simulation_equations(method, tmp_path):
    runs = 0

    def run(**changes):
        nonlocal runs
        params = peslite.load(
            EXAMPLES / "gfl-example.pes",
            **QUIET,
            **{"simulation.t_end": 0.003, "events.connect_vsc.t": 0.0,
               "events.connect_vsc.ramp": 0.005},
            **changes,
        )
        result = peslite.Simulation(params).run(out_dir=tmp_path / str(runs))
        runs += 1
        return np.array(list(result.final_states().values()), dtype=float)

    reference = run(**{"simulation.solver.dt": 1e-6})
    scale = np.maximum(np.abs(reference), 1e-3 * np.max(np.abs(reference)))
    results = [
        run(**{"simulation.solver.method": method,
               "simulation.solver.type": "adaptive" if method == "auto" else "fixed",
               "simulation.solver.circuit_model": circuit_model,
               "simulation.solver.rtol": 1e-10,
               "simulation.solver.atol": 1e-11})
        for circuit_model in ("device", "statewise", "nodal")
    ]
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], rtol=1e-9, atol=1e-9)
    assert np.max(np.abs(results[0] - reference) / scale) < (
        1e-2 if method == "trapezoidal" else 1e-6
    )


@pytest.mark.parametrize("changes, message", [
    ({"simulation.solver.circuit_model": "unknown"}, "not in"),
    ({"simulation.solver.method": "radau",
      "simulation.solver.subsystems": {"vsc.dclink": 4}}, "explicit system method"),
    ({"simulation.solver.method": "trapezoidal", "simulation.solver.rtol": 0.0},
     "rtol must be positive"),
])
def test_implicit_solver_settings_reject_ineffective_combinations(changes, message):
    with pytest.raises(ConfigError, match=message):
        peslite.load(EXAMPLES / "gfl-example.pes", **changes)
