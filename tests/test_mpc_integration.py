"""Predictive control physics, timing, lifecycle and extension integration."""

import copy
import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import quad

import peslite
from peslite.addons.control.voltage_adaptive_pll import VoltageAdaptivePLL
from peslite.addons.functions import energy_closure, plot_result
from peslite.addons.functions.small_signal import small_signal
from peslite.assembly.params import from_dict, read_initial
from peslite.control import (ConverterMeasurement, Measurement, PredictiveCurrentControl,
                             register_loop_type)
from peslite.control.mpc import MPCResult
from test_dc_converters import _case as dc_case


_DC_MPC = "PredictiveCurrentControl"
_AC_MPC = "ThreePhasePredictiveCurrentControl"


def _controller(simulation, unit=None):
    host = simulation.unit(unit).ctrl
    return host, host.graph.nodes[_DC_MPC if unit in (None, "converter") else _AC_MPC]


def _dc(mode="pwm_averaging", **overrides):
    params = dc_case(model=mode, lower_controlled=True, t_end=0.003, ac_base=False)
    return params.replace(**{
        "units.converter.ctrl": {
            "type": "current", "references": {"current": 4.0},
            "loops": {_DC_MPC: {
                "algorithm": "ccs", "prediction_horizon": 4, "control_horizon": 3,
            }},
        },
        "units.converter.connected": True,
        "simulation.progress.enable": False,
        "simulation.output.period": 50e-6,
        "simulation.output.precision": 17,
        **overrides,
    })


def _ac(case, mode="pwm_averaging", **overrides):
    tree = case()
    unit = tree["units"]["vsc"]
    unit["ctrl"] = {
        "type": "gfl", "references": {"id_ref_pu": 0.3},
        "loops": {
            "SRFPLL": {"kp_pu": 20.0, "ki_pu": 1200.0},
            _AC_MPC: {"algorithm": "fcs", "prediction_horizon": 2, "control_horizon": 2},
        },
    }
    unit["bridge"]["model"] = mode
    unit["dclink"] = {"vdc_ref": 1500.0, "source": {"type": "voltage"}}
    unit["connected"] = True
    tree["simulation"].update(t_end=0.008, progress={"enable": False},
                              output={"period": 50e-6, "precision": 17})
    return from_dict(tree).replace(**overrides)


def test_current_profile_is_not_specific_to_mpc():
    params = _dc(**{
        "units.converter.ctrl.loops": {
            "InductorCurrentControl": {
                "kp": 0.02, "ki": 20.0, "minimum": 0.0, "maximum": 1.0,
            },
        },
    })
    graph = peslite.Simulation(params).unit().ctrl.graph
    assert graph.connections["InductorCurrentControl.reference"] == "references.current"
    assert graph.outputs == {"duty": "InductorCurrentControl.duty"}


def test_optimizer_wall_time_is_summary_only(tmp_path):
    simulation = peslite.Simulation(_dc(**{"simulation.t_end": 0.0001}))
    host, controller = _controller(simulation)

    assert "seconds" not in host.graph.observation_names
    result = simulation.run(out_dir=tmp_path)
    summary = result.summary[f"converter.PredictiveCurrentControl"]
    assert summary["solve_seconds"] >= 0.0
    assert "converter.PredictiveCurrentControl.seconds" not in result.ctrl


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_buck_mpc_tracks_current_and_load_change_with_bounded_duty(mode, tmp_path):
    params = _dc(mode, **{
        "simulation.t_end": 0.008,
        "events": {
            "reference": {"type": "set", "t": 0.003,
                          "set": {"units.converter.ctrl.references.current": 5.0}},
            "load": {"type": "set", "t": 0.005, "set": {"elements.load.r": 4.0}},
        },
    })
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)
    current = result.states["converter.branch_f.i"]
    # Current balance predicts Vout -> Iref * Rload independently of the MPC implementation.
    before = (result.t >= 0.002) & (result.t < 0.003)
    after = result.t >= 0.007
    assert np.mean(current[before]) == pytest.approx(4.0, abs=0.2)
    assert np.mean(current[after]) == pytest.approx(5.0, abs=0.2)
    assert result.states["output.u_C"][-1] == pytest.approx(20.0, abs=1.0)
    duties = result.states[f"converter.ctrl.{_DC_MPC}.previous_input.0"]
    assert np.all((duties >= 0) & (duties <= 1))
    assert _controller(simulation)[1].fallback_count == 0


@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
def test_vsc_mpc_example_tracks_its_mean_current_with_each_actuation_timing(
        examples, mode, tmp_path):
    params = peslite.load(examples / "mpc-vsc-example.pes", **{
        "simulation.progress.enable": False,
        "simulation.t_end": 0.01,
        "units.vsc.bridge.model": mode,
    })
    result = peslite.Simulation(params).run(out_dir=tmp_path)
    settled = np.asarray(result.t) >= 0.005
    assert np.mean(np.asarray(result.plant["vsc.id_pu"])[settled]) == pytest.approx(
        0.4, abs=0.01,
    )


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_vsc_mpc_full_matrix_example_runs_through_each_bridge_model(examples, mode, tmp_path):
    params = peslite.load(examples / "mpc-vsc-matrices-example.pes", **{
        "simulation.progress.enable": False,
        "simulation.t_end": 0.006,
        "units.vsc.bridge.model": mode,
    })
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)
    controller = _controller(simulation, "vsc")[1]
    weights = tuple(np.asarray(value) for value in (
        controller.cfg.tracking_weight,
        controller.cfg.input_weight,
        controller.cfg.rate_weight,
    ))
    assert tuple(weight.shape for weight in weights) == ((2, 2), (3, 3), (3, 3))
    assert all(np.count_nonzero(weight - np.diag(np.diag(weight))) for weight in weights)
    settled = np.asarray(result.t) >= 0.003
    assert np.mean(np.asarray(result.plant["vsc.id_pu"])[settled]) == pytest.approx(
        0.4, abs=0.02,
    )
    assert controller.fallback_count == 0


def test_vsc_mpc_full_matrix_example_tracks_its_reference_event(examples, tmp_path):
    params = peslite.load(examples / "mpc-vsc-matrices-example.pes", **{
        "simulation.progress.enable": False,
        "simulation.t_end": 0.016,
    })
    result = peslite.Simulation(params).run(out_dir=tmp_path)
    settled = np.asarray(result.t) >= 0.014
    assert np.mean(np.asarray(result.plant["vsc.id_pu"])[settled]) == pytest.approx(
        0.65, abs=0.02,
    )
    assert np.mean(np.asarray(result.plant["vsc.iq_pu"])[settled]) == pytest.approx(
        -0.10, abs=0.02,
    )


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
@pytest.mark.parametrize("algorithm", ["ccs", "fcs"])
def test_three_phase_mpc_tracks_dq_current_at_rated_physical_scale(case, mode, algorithm, tmp_path):
    params = _ac(case, mode, **{f"units.vsc.ctrl.loops.{_AC_MPC}.algorithm": algorithm})
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)
    current = result.states["vsc.branch_f.i.re"] + 1j * result.states["vsc.branch_f.i.im"]
    # Compare against the measured PCC frame, as specified by the current reference contract.
    voltage = result.states["pcc.u_C.re"] + 1j * result.states["pcc.u_C.im"]
    dq = current * np.exp(-1j * np.angle(voltage)) / params.base.i_phase_peak
    settled = result.t >= 0.006
    assert np.mean(dq[settled].real) == pytest.approx(0.3, abs=0.06)
    assert np.mean(dq[settled].imag) == pytest.approx(0.0, abs=0.06)
    for leg in range(3):
        duty = result.states[f"vsc.ctrl.{_AC_MPC}.previous_input.{leg}"]
        assert np.all((duty >= 0.0) & (duty <= 1.0))
        if algorithm == "fcs":
            assert set(duty) <= {0.0, 1.0}
    assert _controller(simulation, "vsc")[1].fallback_count == 0


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_finite_set_buck_tracks_current_using_binary_commands(mode, tmp_path):
    params = _dc(mode, **{f"units.converter.ctrl.loops.{_DC_MPC}.algorithm": "fcs"})
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)
    settled = result.t >= 0.002
    # The finite voltage set creates a bounded current ripple around the mean reference.
    assert np.mean(result.states["converter.branch_f.i"][settled]) == pytest.approx(4.0, abs=0.4)
    assert set(result.states[f"converter.ctrl.{_DC_MPC}.previous_input.0"]) <= {0.0, 1.0}
    assert _controller(simulation)[1].fallback_count == 0


@pytest.mark.parametrize("resistance", [0.0, 0.05])
@pytest.mark.parametrize("offset,computation,delay", [
    (0.0, 0.0, 0.0),
    (0.0, 1e-16, 50e-6),
    (0.0, 1e-6, 50e-6),
    (10e-6, 5e-6, 40e-6),
])
def test_dc_predictor_matches_independent_split_interval_rl_solution(
        resistance, offset, computation, delay):
    params = _dc(**{
        "units.converter.filter.r_f": resistance,
        "units.converter.ctrl.offset": offset,
        "units.converter.ctrl.computation": computation,
    })
    simulation = peslite.Simulation(params)
    host, controller = _controller(simulation)
    host.command(True)
    host.startup.advance()
    controller.previous_enabled = True
    controller.previous_input[:] = 0.25
    inputs = {"reference": 4.0, "current": 2.0, "voltage": 18.0, "vin": 48.0}
    state, _, models = controller.prediction(offset, inputs)
    action = np.array([0.7])
    predicted = models[0].A @ state + models[0].B @ action + models[0].offset

    def advance(current, duty, duration):
        forcing = 48.0 * duty - 18.0
        if resistance == 0:
            return current + duration * forcing / 1e-3
        decay = math.exp(-resistance * duration / 1e-3)
        return decay * current + (1 - decay) * forcing / resistance

    expected = advance(advance(2.0, 0.25, delay), 0.7, 50e-6 - delay)
    assert controller.delay == pytest.approx(delay, abs=1e-15)
    assert predicted[0] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert predicted[1] == 0.7


def test_ac_predictor_matches_independent_sinusoidal_forcing_integral(case):
    params = _ac(case, **{"units.vsc.ctrl.offset": 10e-6, "units.vsc.ctrl.computation": 2e-6})
    simulation = peslite.Simulation(params)
    host, controller = _controller(simulation, "vsc")
    host.command(True)
    host.startup.advance()
    controller.previous_enabled = True
    controller.previous_input[:] = [1, 0, 0]
    measurement = Measurement(10e-6, 420 + 150j, 20 - 5j, 1500.0, np.zeros(3))
    inputs = {
        "v": measurement.u_g / params.base.v_phase_peak,
        "i": measurement.i_c / params.base.i_phase_peak,
        "u_dc": measurement.u_dc / params.unit("vsc").dc_base.v,
        "frame": np.angle(measurement.u_g),
        "omega": params.base.w0,
        "id_ref": 0.3,
        "iq_ref": 0.0,
    }
    state, _, models = controller.prediction(measurement.t, inputs)
    action = np.array([0, 1, 0])
    predicted = models[0].A @ state + models[0].B @ action + models[0].offset
    T, L, R, delay = (controller.T, controller.plant.L,
                      controller.plant.R, controller.delay)
    old_voltage = 1000 + 0j
    new_voltage = -500 + 500j * math.sqrt(3)

    def derivative_forcing(t):
        converter = old_voltage if t < delay else new_voltage
        terminal = measurement.u_g * np.exp(1j * params.base.w0 * t)
        return np.exp(-R * (T - t) / L) * (converter - terminal) / L

    forcing = sum(quad(lambda t: derivative_forcing(t).real, left, right, epsabs=1e-11)[0]
                  + 1j * quad(lambda t: derivative_forcing(t).imag, left, right, epsabs=1e-11)[0]
                  for left, right in ((0, delay), (delay, T)))
    expected = np.exp(-R * T / L) * measurement.i_c + forcing
    np.testing.assert_allclose(predicted[:2], [expected.real, expected.imag], rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging"])
@pytest.mark.parametrize("offset,computation,delay", [
    (10e-6, 3e-6, 40e-6), (0.0, 0.0, 0.0), (0.0, 1e-16, 50e-6),
])
def test_mpc_uses_existing_adc_interrupt_and_pwm_load_clocks(
        mode, offset, computation, delay, monkeypatch, tmp_path):
    adc_offset = 5e-6
    params = _dc(mode, **{
        "simulation.t_end": 210e-6,
        "units.converter.ctrl.offset": offset,
        "units.converter.ctrl.computation": computation,
        "units.converter.meas.period": 25e-6,
        "units.converter.meas.offset": adc_offset,
    })
    simulation = peslite.Simulation(params)
    unit = simulation.unit()
    controller = unit.ctrl.graph.nodes[_DC_MPC]
    calls, conversions, writes, loads = [], [], [], []
    solve, convert, write, load = controller.solve, unit.adc.convert, unit.bridge.write, unit.bridge.load

    def solve_recorded(t, inputs):
        calls.append((t, inputs["current"]))
        return solve(t, inputs)

    def convert_recorded(t):
        value = convert(t)
        conversions.append((t, unit.adc.values["current"]))
        return value

    def write_recorded(t, registers, *args):
        writes.append((t, np.array(registers)))
        return write(t, registers, *args)

    def load_recorded(t):
        result = load(t)
        loads.append((t, unit.bridge.get_state()))
        return result

    monkeypatch.setattr(controller, "solve", solve_recorded)
    monkeypatch.setattr(unit.adc, "convert", convert_recorded)
    monkeypatch.setattr(unit.bridge, "write", write_recorded)
    monkeypatch.setattr(unit.bridge, "load", load_recorded)
    simulation.run(out_dir=tmp_path)
    np.testing.assert_allclose([t for t, _ in calls], np.arange(offset, 211e-6, 50e-6), atol=1e-15)
    np.testing.assert_allclose([t for t, _ in conversions], np.arange(adc_offset, 211e-6, 25e-6), atol=1e-15)
    np.testing.assert_allclose([t for t, _ in writes], [t for t, _ in calls], atol=1e-15)
    for interrupt, value in calls:
        # Before the first independent ADC tick the sample is seeded from the zero-current plant.
        latest = [0.0, *(sample for t, sample in conversions if t <= interrupt)][-1]
        assert value == latest
    assert controller.solve_count == len(calls)
    assert controller.delay == pytest.approx(delay)
    for at, state in loads:
        # Positive computation always writes after a coincident load, even below TIME_EPS.
        eligible = [value for when, value in writes
                    if when + computation <= at + 1e-14
                    and (computation == 0 or when < at - 1e-14)]
        if eligible:
            assert state[unit.bridge.register_names[0]] == pytest.approx(eligible[-1][0])


@pytest.mark.parametrize("method", ["DP45", "radau"])
def test_sampled_averaging_mpc_respects_control_boundaries_with_adaptive_steps(
    method, monkeypatch, tmp_path,
):
    period = 50e-6
    common = {
        "simulation.t_end": 0.001,
        "simulation.output.period": 10e-6,
    }
    reference = peslite.Simulation(_dc("averaging", **{
        **common,
        "simulation.solver.type": "fixed",
        "simulation.solver.method": "RK4",
        "simulation.solver.dt": 1e-6,
    })).run(out_dir=tmp_path / "reference")
    params = _dc("averaging", **{
        **common,
        "simulation.solver.type": "adaptive",
        "simulation.solver.method": method,
    })
    simulation = peslite.Simulation(params)
    host, controller = _controller(simulation)
    solve = controller.solve
    calls = []

    def recorded(t, inputs):
        calls.append(t)
        return solve(t, inputs)

    monkeypatch.setattr(controller, "solve", recorded)
    result = simulation.run(out_dir=tmp_path)

    np.testing.assert_allclose(calls, np.arange(0.0, 0.001 + period, period), atol=1e-14)
    for state in ("converter.branch_f.i", "output.u_C"):
        np.testing.assert_allclose(result.states[state], reference.states[state], atol=2e-10)
    assert host.graph.nodes[_DC_MPC].fallback_count == 0


def test_prediction_does_not_mutate_plant_or_adc_measurement(case):
    simulation = peslite.Simulation(_ac(case))
    host, controller = _controller(simulation, "vsc")
    host.command(True)
    host.startup.advance()
    before = simulation.system.get_state()
    measurement = Measurement(0.0, 500 + 20j, 10 + 2j, 1500.0, np.array([10.0, -3.0, -7.0]))
    original = copy.deepcopy(measurement)
    controller.solve(0.0, {
        "v": measurement.u_g / simulation.p.base.v_phase_peak,
        "i": measurement.i_c / simulation.p.base.i_phase_peak,
        "u_dc": measurement.u_dc / simulation.p.unit("vsc").dc_base.v,
        "frame": np.angle(measurement.u_g),
        "omega": simulation.p.base.w0,
        "id_ref": 0.3,
        "iq_ref": 0.0,
    })
    assert simulation.system.get_state() == before
    assert measurement.u_g == original.u_g
    assert measurement.i_c == original.i_c
    np.testing.assert_array_equal(measurement.i_abc, original.i_abc)


def test_current_limit_is_enforced_in_predictions_and_representative_averaged_transient(monkeypatch, tmp_path):
    params = _dc(**{
        "simulation.t_end": 0.003,
        "units.converter.ctrl.references.current": 8.0,
        f"units.converter.ctrl.loops.{_DC_MPC}.current_max": 5.0,
        f"units.converter.ctrl.loops.{_DC_MPC}.input_max": 0.8,
    })
    simulation = peslite.Simulation(params)
    controller = _controller(simulation)[1]
    solve = controller.solve
    predictions = []

    def recorded(t, inputs):
        result = solve(t, inputs)
        if result.success:
            predictions.append(result.states[1:, 0])
        return result

    monkeypatch.setattr(controller, "solve", recorded)
    result = simulation.run(out_dir=tmp_path)
    assert predictions
    assert max(np.max(values) for values in predictions) <= 5.0 + controller.cfg.feasibility_tolerance
    assert np.max(result.states[f"converter.ctrl.{_DC_MPC}.previous_input.0"]) <= 0.8
    assert np.max(result.states["converter.branch_f.i"]) <= 5.05
    assert result.states["converter.branch_f.i"][-1] == pytest.approx(5.0, abs=0.05)


def test_mpc_results_are_consistent_under_step_refinement_and_multirate_integration(tmp_path):
    params = _dc(**{"simulation.t_end": 0.001})
    baseline = peslite.Simulation(params).run(out_dir=tmp_path / "baseline")
    for label, changes in (
        ("refined", {"simulation.solver.dt": 1.25e-6}),
        ("multirate", {"simulation.solver.subsystems": {"output": 0.5}}),
    ):
        actual = peslite.Simulation(params.replace(**changes)).run(out_dir=tmp_path / label)
        for key in ("converter.branch_f.i", "output.u_C",
                    f"converter.ctrl.{_DC_MPC}.previous_input.0"):
            np.testing.assert_allclose(actual.states[key], baseline.states[key],
                                       rtol=2e-5, atol=2e-5, err_msg=f"{label}: {key}")
        np.testing.assert_array_equal(actual.states[f"converter.ctrl.{_DC_MPC}.solve_count"],
                                      baseline.states[f"converter.ctrl.{_DC_MPC}.solve_count"])


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_repeated_and_resumed_mpc_run_preserves_decisions_and_plant_state(mode, tmp_path):
    params = _dc(mode, **{"simulation.t_end": 0.0008, "simulation.output.period": 5e-6})
    whole = peslite.Simulation(params).run(out_dir=tmp_path / "whole")
    repeated = peslite.Simulation(params).run(out_dir=tmp_path / "repeated")
    for key in whole.states:
        np.testing.assert_array_equal(whole.states[key], repeated.states[key], err_msg=key)
    peslite.Simulation(params.replace(**{"simulation.t_end": 0.000375})).run(out_dir=tmp_path / "first")
    initial = read_initial(tmp_path / "first" / "states.csv")
    resumed = peslite.Simulation(params.replace(**{"simulation.initial": initial})).run(out_dir=tmp_path / "resumed")
    start = np.argmin(abs(whole.t - 0.000375))
    np.testing.assert_allclose(resumed.t, whole.t[start:], atol=1e-15)
    for key in whole.states:
        np.testing.assert_allclose(resumed.states[key], whole.states[key][start:],
                                   rtol=1e-10, atol=1e-10, err_msg=key)


def test_runtime_options_and_reference_events_apply_without_changing_state_schema(tmp_path):
    params = _dc(**{"simulation.t_end": 0.0004, "events": {
        "retune": {"type": "set", "t": 0.0002, "set": {
            f"units.converter.ctrl.loops.{_DC_MPC}.rate_weight": 0.2,
            "units.converter.ctrl.references.current": 2.0,
        }},
    }})
    simulation = peslite.Simulation(params)
    names = set(simulation.unit().ctrl.get_state())
    simulation.run(out_dir=tmp_path)
    host, controller = _controller(simulation)
    assert controller.cfg.rate_weight == 0.2
    assert host.p.ctrl.references.current == 2.0
    assert set(host.get_state()) == names
    assert controller.solve_count == 9


def test_retune_preserves_applied_input_history_but_fallback_uses_new_bounds(
    monkeypatch,
):
    params = _dc()
    simulation = peslite.Simulation(params)
    host, controller = _controller(simulation)
    controller.previous_input[:] = 0.8
    controller.previous_enabled = True
    changed = params.replace(**{
        f"units.converter.ctrl.loops.{_DC_MPC}.input_max": 0.5,
    })

    host.retune(
        changed.unit("converter"),
        [f"ctrl.loops.{_DC_MPC}.input_max"],
        0.0,
    )
    host.initialize()
    controller = host.graph.nodes[_DC_MPC]
    np.testing.assert_array_equal(controller.previous_input, [0.8])
    assert not controller._admissible(controller.previous_input)

    monkeypatch.setattr(
        controller,
        "solve",
        lambda _t, _inputs: MPCResult(False, None, None, None, None, "solver_failure"),
    )
    host.command(True)
    output = host(0.0, ConverterMeasurement(current=1.0, voltage=24.0, vin=48.0))

    assert output.duty == 0.0
    assert output.gates


@pytest.mark.parametrize("fallback", ["hold", "fixed", "block"])
def test_infeasible_current_constraint_uses_admissible_fallback(fallback):
    params = _dc(**{
        f"units.converter.ctrl.loops.{_DC_MPC}.current_max": 1.0,
        f"units.converter.ctrl.loops.{_DC_MPC}.fallback": fallback,
        f"units.converter.ctrl.loops.{_DC_MPC}.fallback_input": [0.4],
        f"units.converter.ctrl.loops.{_DC_MPC}.input_min": 0.1,
        f"units.converter.ctrl.loops.{_DC_MPC}.input_max": 0.9,
    })
    simulation = peslite.Simulation(params)
    host, controller = _controller(simulation)
    host.command(True)
    controller.previous_input[:] = 0.3
    controller.previous_enabled = True
    output = host(0.0, ConverterMeasurement(current=100.0, voltage=24.0, vin=48.0))
    assert not controller.last_result.success
    assert controller.fallback_count == 1
    assert output.duty == pytest.approx(0.3 if fallback == "hold" else 0.4)
    assert output.gates == (fallback != "block")


def test_start_stop_and_reconnect_suspend_optimization_and_resume_current_control(tmp_path):
    params = _dc(**{"simulation.t_end": 0.0008, "events": {
        "stop": {"type": "disconnect", "target": "converter", "t": 0.0002},
        "restart": {"type": "connect", "target": "converter", "t": 0.0005, "ramp": 0.0001},
    }})
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)
    stopped = (result.t >= 0.00025) & (result.t < 0.0005)
    assert np.ptp(result.states[f"converter.ctrl.{_DC_MPC}.solve_count"][stopped]) == 0
    assert np.all(result.states[f"converter.ctrl.{_DC_MPC}.previous_enabled"][stopped] == 0)
    assert simulation.unit().ctrl.startup.complete
    assert simulation.unit().gates


def test_reconnect_between_interrupts_preserves_last_submitted_enable_in_prediction(monkeypatch, tmp_path):
    params = _dc(**{"simulation.t_end": 0.0003, "events": {
        "stop": {"type": "disconnect", "target": "converter", "t": 0.000212},
        "restart": {"type": "connect", "target": "converter", "t": 0.000225},
    }})
    simulation = peslite.Simulation(params)
    unit = simulation.unit()
    controller = unit.ctrl.graph.nodes[_DC_MPC]
    solve = controller.solve
    restarted = []

    def recorded(t, inputs):
        if math.isclose(t, 0.00025, abs_tol=1e-15):
            restarted.append((controller.previous_enabled, unit.bridge.on, unit.breaker))
        return solve(t, inputs)

    monkeypatch.setattr(controller, "solve", recorded)
    simulation.run(out_dir=tmp_path)
    # No interrupt submitted a disabled PWM word while the terminal was disconnected.
    assert restarted == [(True, True, True)]


def test_converter_protection_remains_effective_during_optimizer_fallback(monkeypatch, tmp_path):
    params = _dc(**{
        "simulation.t_end": 0.0004,
        "simulation.stop_on_trip": False,
        "units.converter.protection.overcurrent.enable": True,
        "units.converter.protection.overcurrent.limit_pu": 0.05,
        f"units.converter.ctrl.loops.{_DC_MPC}.fallback": "fixed",
        f"units.converter.ctrl.loops.{_DC_MPC}.fallback_input": [1.0],
    })
    simulation = peslite.Simulation(params)
    controller = _controller(simulation)[1]
    invalid = MPCResult(True, np.array([np.nan]), None, None, None, "invalid")
    monkeypatch.setattr(controller, "solve", lambda t, measurement: invalid)
    result = simulation.run(out_dir=tmp_path)
    assert controller.fallback_count > 0
    assert result.summary["converter.trip_cause"] == "overcurrent"
    assert not simulation.unit().gates


@register_loop_type
class _DeadbeatMPC(PredictiveCurrentControl):
    @dataclass(frozen=True, kw_only=True)
    class Params(PredictiveCurrentControl.Params):
        type: str = "test_deadbeat_mpc"

    type = "test_deadbeat_mpc"

    def solve(self, t, inputs):
        decay = math.exp(-self.plant.R * self.T / self.plant.L)
        gain = self.plant.integral(self.T)
        duty = ((inputs["reference"] - decay * inputs["current"]) / gain
                + inputs["voltage"]) / inputs["vin"]
        action = np.array([np.clip(duty, self.cfg.input_min, self.cfg.input_max)])
        return MPCResult(True, action, None, None, None, "deadbeat")


def test_custom_mpc_loop_is_registered_without_runner_changes(tmp_path):
    params = _dc(**{
        "units.converter.ctrl.loops": {
            "deadbeat": {"type": "test_deadbeat_mpc", "prediction_horizon": 4,
                         "control_horizon": 3},
        },
        "units.converter.ctrl.computation": 0.0,
        "simulation.t_end": 0.001,
    })
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)
    controller = simulation.unit().ctrl.graph.nodes["deadbeat"]
    assert isinstance(controller, _DeadbeatMPC)
    assert controller.last_result.status == "deadbeat"
    assert result.states["converter.branch_f.i"][-1] == pytest.approx(4.0, abs=0.1)


def test_mpc_composes_with_control_and_component_addons(examples, tmp_path):
    params = peslite.load(examples / "cable-gfl-example.pes", **{
        "simulation.t_end": 0.001,
        "simulation.progress.enable": 0,
        "simulation.solver.linearisations": 0,
        "units.vsc.connected": True,
        "units.vsc.dclink": {
            "vdc_ref": 20000.0,
            "source": {"type": "voltage"},
        },
        "units.vsc.ctrl": {
            "type": "gfl",
            "references": {"id_ref_pu": 0.2},
            "loops": {
                "VoltageAdaptivePLL": {"bandwidth": 20.0},
                _AC_MPC: {
                    "algorithm": "fcs",
                    "prediction_horizon": 2,
                    "control_horizon": 2,
                },
            },
        },
    })
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)
    nodes = simulation.unit("vsc").ctrl.graph.nodes

    assert isinstance(nodes["VoltageAdaptivePLL"], VoltageAdaptivePLL)
    assert nodes[_AC_MPC].fallback_count == 0
    assert np.isfinite(result.states["cable1.iL1.re"]).all()
    assert not result.tripped


@pytest.mark.full
def test_mpc_results_work_with_function_addons(tmp_path):
    params = _dc("averaging", **{
        "simulation.t_end": 0.02,
        "simulation.output.energy": True,
    })
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)

    closure = energy_closure(result)
    figure = plot_result(result, "converter.branch_f.i", latex=False)
    analysis = small_signal(simulation, period=50e-6)

    assert np.isfinite(tuple(closure.values())).all()
    assert figure.is_file()
    assert analysis.phi.shape[0] == analysis.phi.shape[1] > 0
    assert analysis.periodic


@pytest.mark.static
@pytest.mark.parametrize("format", ["cpp", "imperix"])
def test_mpc_export_is_explicitly_rejected_without_generating_files(format, tmp_path):
    simulation = peslite.Simulation(_dc())
    options = {}
    if format == "imperix":
        options = {"build": False, "target_config": {"units": {"converter": {
            "adc": {"current": 0, "voltage": 1, "vin": 2}, "pwm": [0],
            "startup_ramp": 0.0, "hardware_overcurrent_configured": True,
        }}}}
    out = tmp_path / "exported"
    with pytest.raises(TypeError, match="loop|generator"):
        simulation.export(format, out, **options)
    assert not out.exists()
