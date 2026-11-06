"""MPC public construction, custom-result and named-state boundaries."""

from dataclasses import replace

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.control import ConverterMeasurement, MPCCurrentController, MPCResult
from peslite.solver.model import StateRegistry


@pytest.fixture
def mpc_config():
    return peslite.load(EXAMPLES / "mpc-buck-example.pes").unit("converter")


def test_current_controller_constructs_from_resolved_config_without_topology(mpc_config):
    controller = MPCCurrentController(mpc_config)

    assert controller.n_current == controller.n_input == 1
    assert controller.T == mpc_config.ctrl.period
    assert controller.delay == pytest.approx(controller.T)
    assert controller.control_inputs(2.0, 24.0, 48.0) == {
        "meas.current": 2.0, "meas.voltage": 24.0, "meas.vin": 48.0,
    }


@pytest.mark.parametrize("algorithm", ["ccs", "fcs"])
def test_state_registry_keeps_control_memory_and_excludes_diagnostic_recurrence(
    mpc_config, algorithm,
):
    options = replace(mpc_config.ctrl.options, algorithm=algorithm)
    configured = replace(mpc_config, ctrl=replace(mpc_config.ctrl, options=options))
    controller = MPCCurrentController(configured)
    registry = StateRegistry({"converter.ctrl": controller})
    duty = "converter.ctrl.previous_input.0"
    counters = {"converter.ctrl.solve_count", "converter.ctrl.fallback_count"}
    initial = registry.read_flat()

    assert (duty in registry.discrete_names) == (algorithm == "fcs")
    assert duty in registry.linearization_columns
    assert counters.isdisjoint(registry.linearization_columns)
    controller.solve_count, controller.fallback_count = 7, 2
    updated = registry.read_flat()
    assert registry.periodic_state(updated) == registry.periodic_state(initial)
    assert counters <= updated.keys()
    assert updated["converter.ctrl.solve_count"] == 7
    assert initial["converter.ctrl.solve_count"] == 0

    controller.previous_input[:] = 1.0
    assert registry.periodic_state(registry.read_flat())[duty] == 1.0
    assert registry.periodic_state(registry.read_flat()) != registry.periodic_state(initial)


@pytest.mark.parametrize("action", [
    np.array([0.5 + 0.25j]),
    np.array(["0.5"]),
    np.array([True]),
])
def test_invalid_custom_action_uses_fallback_without_coercing_values(mpc_config, action):
    class InvalidOptimizer(MPCCurrentController):
        def solve(self, t, measurement):
            return MPCResult(True, action, None, None, 0.0, "optimal")

    options = replace(mpc_config.ctrl.options, fallback="fixed", fallback_input=[0.25])
    configured = replace(mpc_config, ctrl=replace(mpc_config.ctrl, options=options))
    controller = InvalidOptimizer(configured)
    controller.command(True)

    output = controller(0.0, ConverterMeasurement(current=0.0, voltage=24.0, vin=48.0))

    assert output.duty == 0.25
    assert output.gates
    assert controller.last_result.status == "invalid_solution"
    assert not controller.last_result.success
    assert controller.last_result.action is None
    assert controller.last_log["mpc.success"] == 0.0
    assert controller.fallback_count == controller.solve_count == 1


@pytest.mark.parametrize("lower, upper, expected", [(0.2, 1.0, 1.0), (0.0, 0.8, 0.0)])
def test_fcs_default_fallback_is_selected_from_admissible_actions(
    mpc_config, lower, upper, expected,
):
    options = replace(mpc_config.ctrl.options, algorithm="fcs", input_min=lower, input_max=upper)
    configured = replace(mpc_config, ctrl=replace(mpc_config.ctrl, options=options))

    controller = MPCCurrentController(configured)

    np.testing.assert_array_equal(controller.actions, [[expected]])
    np.testing.assert_array_equal(controller.fixed_input, [expected])
    np.testing.assert_array_equal(controller.previous_input, [expected])


def test_recovery_prediction_discards_current_reset_by_coincident_block_load(tmp_path, monkeypatch):
    params = peslite.load(EXAMPLES / "mpc-buck-example.pes").replace(**{
        "events": {},
        "simulation.t_end": 250.0e-6,
        "simulation.output.period": 50.0e-6,
        "units.converter.ctrl.options.fallback": "block",
    })
    simulation = peslite.Simulation(params)
    controller = simulation.unit("converter").ctrl
    period = controller.T
    samples = {}

    def solve(t, measurement):
        state, references, models = controller.prediction(t, measurement)
        action = np.array([0.5])
        predicted = models[0].A @ state + models[0].B @ action + models[0].offset
        samples[round(t / period)] = (measurement.current, controller.previous_enabled, predicted[0])
        if round(t / period) == 2:
            return MPCResult(False, None, None, None, None, "solver_failure")
        return MPCResult(True, action, None, None, 0.0, "optimal")

    monkeypatch.setattr(controller, "solve", solve)
    result = simulation.run(out_dir=tmp_path)

    # The interrupt samples nonzero current just before loading the preceding block command.
    # Opening the real filter branch then resets current, which stays zero until recovery loads.
    current_before_block, prior_enabled, predicted_after_block = samples[3]
    assert current_before_block > 0.0
    assert not prior_enabled
    assert predicted_after_block == pytest.approx(0.0, abs=1e-14)
    assert samples[4][0] == pytest.approx(predicted_after_block, abs=1e-14)
    blocked_row = np.argmin(abs(result.t - 3 * period))
    assert result.states["converter.pwm.on"][blocked_row] == 0.0
    assert controller.fallback_count == 1
