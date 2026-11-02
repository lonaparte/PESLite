"""Fixed-step implicit integration and circuit-model linear systems."""

import math

import numpy as np
import pytest
from scipy.linalg import expm

import peslite
from conftest import EXAMPLES
from peslite.solver import AdaptiveSolver, ConfigError, FixedStepSolver, make_solver
from peslite.solver.implicit import (ImplicitSolver, LinearSystem, ModeJacobian, Structure,
                                     circuit_structure)


A_OSCILLATOR = np.array([[-1.0, 10.0], [-10.0, -1.0]])
QUIET = {"simulation.progress.enable": 0, "simulation.solver.linearisations": 0}


@pytest.mark.parametrize(("solver_type", "method", "canonical", "expected"), [
    ("FiXeD", "rK4", "RK4", FixedStepSolver),
    ("FIXED", "RaDaU", "radau", ImplicitSolver),
    ("AdApTiVe", "rAdAu", "radau", AdaptiveSolver),
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


def test_radau_damps_a_stiff_mode_while_trapezoidal_keeps_it_bounded():
    def f(t, y):
        return np.array([-1e7 * (y[0] - math.cos(t)) - math.sin(t)])

    radau = ImplicitSolver(1e-3, "radau")(f, 0.0, 1.0, np.array([2.0])).y[0]
    trapezoidal = ImplicitSolver(1e-3, "trapezoidal")(f, 0.0, 1.0, np.array([2.0])).y[0]
    assert radau == pytest.approx(math.cos(1.0), rel=1e-9)
    assert abs(trapezoidal - math.cos(1.0)) > 0.1
    assert abs(trapezoidal) < 3.0


def test_statewise_linear_system_reuses_returning_modes():
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

    device = LinearSystem()
    solver = ImplicitSolver(1e-3, "trapezoidal", linear=device)
    y = np.array([1.0, 1.0])
    for k in range(6):
        selected["mode"] = k % 2
        y = solver(f, 0.01 * k, 0.01 * (k + 1), y).y
    assert device.n_jacobians == 12


def test_mna_group_solve_matches_the_dense_newton_system():
    model = peslite.System(peslite.load(EXAMPLES / "two-converters-example.pes")).model
    structure = circuit_structure(model)
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
    mna = make_solver(settings.__class__(
        **{**vars(settings), "method": "radau", "circuit_model": "mna"}
    ), model)
    adaptive = make_solver(settings.__class__(
        **{**vars(settings), "type": "adaptive", "method": "Radau",
           "circuit_model": "statewise"}
    ), model)

    assert type(device) is ImplicitSolver and not device.linear.cache
    assert type(statewise) is ImplicitSolver and statewise.linear.cache
    assert type(mna) is ImplicitSolver and mna.linear.structure.interface.size
    assert type(adaptive) is AdaptiveSolver and type(adaptive.jac) is ModeJacobian
    step = adaptive(model.rhs, 0.0, 1e-6, model.get_initial_values())
    assert np.all(np.isfinite(step.y)) and adaptive.jac.linear.n_jacobians > 0


@pytest.mark.full
@pytest.mark.parametrize("method", ["trapezoidal", "radau"])
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
               "simulation.solver.circuit_model": circuit_model})
        for circuit_model in ("device", "statewise", "mna")
    ]
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], rtol=1e-9, atol=1e-9)
    assert np.max(np.abs(results[0] - reference) / scale) < (
        1e-2 if method == "trapezoidal" else 1e-6
    )


@pytest.mark.parametrize("changes, message", [
    ({"simulation.solver.circuit_model": "nodal"}, "not in"),
    ({"simulation.solver.circuit_model": "statewise"}, "needs an implicit method"),
    ({"simulation.solver.type": "adaptive", "simulation.solver.method": "RK45",
      "simulation.solver.circuit_model": "mna"}, "needs an implicit method"),
    ({"simulation.solver.method": "radau",
      "simulation.solver.subsystems": {"vsc.dclink": 4}}, "explicit system method"),
    ({"simulation.solver.method": "trapezoidal", "simulation.solver.rtol": 0.0},
     "rtol must be positive"),
])
def test_implicit_solver_settings_reject_ineffective_combinations(changes, message):
    with pytest.raises(ConfigError, match=message):
        peslite.load(EXAMPLES / "gfl-example.pes", **changes)

