"""MPC loop construction, custom-result and named-state boundaries."""

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.control import ConverterMeasurement, MPCResult, PredictiveCurrentControl
from peslite.solver.model import StateRegistry


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


def test_recovery_prediction_discards_current_reset_by_coincident_block_load(
    mpc_params, tmp_path, monkeypatch,
):
    params = mpc_params.replace(**{
        "events": {},
        "simulation.t_end": 250.0e-6,
        "simulation.output.period": 50.0e-6,
        f"units.converter.ctrl.loops.{_MPC}.fallback": "block",
    })
    simulation, _host, controller = _parts(params)
    period = controller.T
    samples = {}

    def solve(t, inputs):
        state, references, models = controller.prediction(t, inputs)
        action = np.array([0.5])
        predicted = models[0].A @ state + models[0].B @ action + models[0].offset
        samples[round(t / period)] = (
            inputs["current"], controller.previous_enabled, predicted[0],
        )
        if round(t / period) == 2:
            return MPCResult(False, None, None, None, None, "solver_failure")
        return MPCResult(True, action, None, None, 0.0, "optimal")

    monkeypatch.setattr(controller, "solve", solve)
    result = simulation.run(out_dir=tmp_path)

    current_before_block, prior_enabled, predicted_after_block = samples[3]
    assert current_before_block > 0.0
    assert not prior_enabled
    assert predicted_after_block == pytest.approx(0.0, abs=1e-14)
    assert samples[4][0] == pytest.approx(predicted_after_block, abs=1e-14)
    blocked_row = np.argmin(abs(result.t - 3 * period))
    assert result.states["converter.pwm.on"][blocked_row] == 0.0
    assert controller.fallback_count == 1
