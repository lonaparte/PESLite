"""MPC loop construction, custom-result and named-state boundaries."""

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.control import ConverterMeasurement, MPCResult, PredictiveCurrentControl
from peslite.solver.model import ConfigError, StateRegistry


_MPC = "PredictiveCurrentControl"


@pytest.fixture
def mpc_params():
    return peslite.load(EXAMPLES / "mpc-buck-example.pes")


def _parts(params):
    simulation = peslite.Simulation(params)
    host = simulation.unit("converter").ctrl
    return simulation, host, host.graph.nodes[_MPC]


def test_predictive_current_control_is_an_ordinary_resolved_loop(mpc_params):
    _simulation, host, controller = _parts(mpc_params)

    assert isinstance(controller, PredictiveCurrentControl)
    assert controller.n_current == controller.n_input == 1
    assert controller.T == mpc_params.unit("converter").ctrl.period
    assert controller.delay == pytest.approx(controller.T)
    assert host.control_inputs(2.0, 24.0, 48.0) == {
        "meas.current": 2.0,
        "meas.voltage": 24.0,
        "meas.vin": 48.0,
    }


@pytest.mark.parametrize("algorithm", ["ccs", "fcs"])
def test_state_registry_keeps_control_memory_and_excludes_diagnostic_recurrence(
    mpc_params, algorithm,
):
    params = mpc_params.replace(**{
        f"units.converter.ctrl.loops.{_MPC}.algorithm": algorithm,
    })
    _simulation, host, controller = _parts(params)
    registry = StateRegistry({"converter.ctrl": host})
    duty = f"converter.ctrl.{_MPC}.previous_input.0"
    counters = {
        f"converter.ctrl.{_MPC}.solve_count",
        f"converter.ctrl.{_MPC}.fallback_count",
    }
    initial = registry.read_flat()

    assert (duty in registry.discrete_names) == (algorithm == "fcs")
    assert duty in registry.linearization_columns
    assert counters.isdisjoint(registry.linearization_columns)
    controller.solve_count, controller.fallback_count = 7, 2
    updated = registry.read_flat()
    assert registry.periodic_state(updated) == registry.periodic_state(initial)
    assert counters <= updated.keys()
    assert updated[f"converter.ctrl.{_MPC}.solve_count"] == 7
    assert initial[f"converter.ctrl.{_MPC}.solve_count"] == 0

    controller.previous_input[:] = 1.0
    assert registry.periodic_state(registry.read_flat())[duty] == 1.0
    assert registry.periodic_state(registry.read_flat()) != registry.periodic_state(initial)


@pytest.mark.parametrize("action", [
    np.array([0.5 + 0.25j]),
    np.array(["0.5"]),
    np.array([True]),
])
def test_invalid_custom_action_uses_fallback_without_coercing_values(
    mpc_params, action, monkeypatch,
):
    params = mpc_params.replace(**{
        f"units.converter.ctrl.loops.{_MPC}.fallback": "fixed",
        f"units.converter.ctrl.loops.{_MPC}.fallback_input": [0.25],
    })
    _simulation, host, controller = _parts(params)
    monkeypatch.setattr(
        controller, "solve",
        lambda _t, _inputs: MPCResult(True, action, None, None, 0.0, "optimal"),
    )
    host.command(True)

    output = host(0.0, ConverterMeasurement(current=0.0, voltage=24.0, vin=48.0))

    assert output.duty == 0.25
    assert output.gates
    assert controller.last_result.status == "invalid_solution"
    assert not controller.last_result.success
    assert controller.last_result.action is None
    assert controller._last_diagnostics["success"] == 0.0
    assert controller.fallback_count == controller.solve_count == 1


@pytest.mark.parametrize("lower, upper, expected", [(0.2, 1.0, 1.0), (0.0, 0.8, 0.0)])
def test_fcs_default_fallback_is_selected_from_admissible_actions(
    mpc_params, lower, upper, expected,
):
    params = mpc_params.replace(**{
        f"units.converter.ctrl.loops.{_MPC}.algorithm": "fcs",
        f"units.converter.ctrl.loops.{_MPC}.input_min": lower,
        f"units.converter.ctrl.loops.{_MPC}.input_max": upper,
    })
    _simulation, _host, controller = _parts(params)

    np.testing.assert_array_equal(controller.actions, [[expected]])
    np.testing.assert_array_equal(controller.fixed_input, [expected])
    np.testing.assert_array_equal(controller.previous_input, [expected])


@pytest.mark.parametrize(("field", "value", "message"), [
    ("max_iterations", 0, "max_iterations"),
    ("max_evaluations", 0, "max_evaluations"),
    ("tolerance", 0.0, "tolerance"),
    ("feasibility_tolerance", -1.0, "feasibility_tolerance"),
    ("tracking_weight", -1.0, "tracking_weight"),
    ("input_weight", -1.0, "input_weight"),
    ("rate_weight", -1.0, "rate_weight"),
    ("current_min", np.inf, "current_min"),
    ("current_max", np.nan, "current_max"),
])
def test_predictive_loop_rejects_invalid_numerical_parameters(
        mpc_params, field, value, message):
    with pytest.raises(ConfigError, match=message):
        params = mpc_params.replace(**{
            f"units.converter.ctrl.loops.{_MPC}.{field}": value,
        })
        peslite.Simulation(params)


def test_fcs_rejects_bounds_without_a_binary_action(mpc_params):
    with pytest.raises(ConfigError, match="exclude every finite control action"):
        params = mpc_params.replace(**{
            f"units.converter.ctrl.loops.{_MPC}.algorithm": "fcs",
            f"units.converter.ctrl.loops.{_MPC}.input_min": 0.2,
            f"units.converter.ctrl.loops.{_MPC}.input_max": 0.8,
        })
        peslite.Simulation(params)


def test_predictive_loop_rejects_fallback_outside_input_bounds(mpc_params):
    with pytest.raises(ConfigError, match="finite admissible commands"):
        params = mpc_params.replace(**{
            f"units.converter.ctrl.loops.{_MPC}.input_min": 0.2,
            f"units.converter.ctrl.loops.{_MPC}.fallback_input": [0.1],
        })
        peslite.Simulation(params)


def test_blocked_prediction_keeps_the_measured_current_as_the_initial_state(mpc_params):
    _simulation, _host, controller = _parts(mpc_params)
    controller.previous_enabled = False
    controller.previous_input[:] = 0.75
    current = 5.0

    state, _references, models = controller.prediction(0.0, {
        "reference": 0.0,
        "current": current,
        "voltage": 0.0,
        "vin": 48.0,
    })
    predicted = models[0].A @ state + models[0].B @ np.zeros(1) + models[0].offset

    assert models[0].A[0, 0] == pytest.approx(controller.plant.decay)
    assert models[0].A[0, 1] == 0.0
    assert predicted[0] == pytest.approx(current * controller.plant.decay)

    state, _references, models = controller.prediction(0.0, {
        "reference": 0.0,
        "current": 0.01,
        "voltage": 24.0,
        "vin": 48.0,
    })
    predicted = models[0].A @ state + models[0].B @ np.zeros(1) + models[0].offset
    assert predicted[0] == 0.0


def test_fcs_state_restore_validates_the_complete_physical_action(mpc_params):
    params = mpc_params.replace(**{
        f"units.converter.ctrl.loops.{_MPC}.algorithm": "fcs",
        f"units.converter.ctrl.loops.{_MPC}.input_max": 0.8,
    })
    _simulation, _host, controller = _parts(params)
    initial = controller.previous_input.copy()

    with pytest.raises(ValueError, match="valid finite bridge command"):
        controller.set_state({"previous_input.0": 0.5})
    np.testing.assert_array_equal(controller.previous_input, initial)

    controller.set_state({"previous_input.0": 1.0})
    np.testing.assert_array_equal(controller.previous_input, [1.0])
    assert not controller._admissible(controller.previous_input)


def test_bridge_energy_storage_is_not_silently_omitted_from_the_predictor(mpc_params):
    params = mpc_params.replace(**{
        "units.converter.bridge.on_resistance": 0.01,
        "units.converter.bridge.parallel_capacitance": 1e-6,
    })

    with pytest.raises(ConfigError, match="without internal energy storage"):
        peslite.Simulation(params)
