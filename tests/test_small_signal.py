"""Shared loop-map differences and the opt-in stability addon on ordinary simulation paths."""

import dataclasses
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import peslite
import pytest
from scipy.linalg import expm

from conftest import EXAMPLES
from peslite.addons.functions.small_signal import small_signal, small_signal_report
from peslite.components.adc import ADC, MeasurementPorts
from peslite.solver.linearization import (LoopMap, active_coordinates, coordinate_difference,
                                         linearize_loop)
from peslite.solver.model import Empty, Model, StateRegistry
from peslite.solver.simulation import SystemLoop, macro_period
from peslite.solver.splitbound import LoopMap as SplitLoopMap, _period_map


class _AffineMap:
    period = 0.01
    matrix = np.array([[0.9, 0.2], [-0.1, 0.8]])
    cols = [("scalar", ["x"], 1.0), ("scalar", ["y"], 1.0), ("mode", ["gate"], 1.0)]

    def __init__(self):
        self.calls = 0

    def coordinates(self, names):
        return self.cols

    def to_frame(self, row, t, cols):
        return np.array([row[name] for _kind, names, _scale in cols for name in names])

    def from_frame(self, x, t, cols, row):
        return {**row, **dict(zip((name for _kind, names, _s in cols for name in names), x))}

    def advance(self, row, t, view=None):
        self.calls += 1
        x = np.array([row["x"], row["y"]])
        if view is not None and view[2] == "b":
            x[("x", "y").index(view[0])] += view[1]
        y = self.matrix @ x
        if view is not None and view[2] == "c":
            y[("x", "y").index(view[0])] += view[1]
        return {"x": y[0], "y": y[1], "gate": row["gate"]}


def test_shared_map_matches_known_discrete_matrix_and_preserves_split_inputs():
    loop, row = _AffineMap(), {"x": 1.5, "y": -0.5, "gate": 1.0}
    assert LoopMap is SplitLoopMap
    assert isinstance(loop, LoopMap)
    linear = linearize_loop(loop, row, 0.0, loop.cols)
    assert linear.active == (0, 1)
    assert linear.evaluations == loop.calls == 5
    assert not linear.changed_modes
    expected = np.eye(3)
    expected[:2, :2] = loop.matrix
    np.testing.assert_allclose(linear.phi, expected, atol=2e-12)

    phi, b_b, b_c, count = _period_map(loop, row, 0.0, loop.cols, [("x", 1.0)])
    np.testing.assert_allclose(phi, expected, atol=2e-12)
    np.testing.assert_allclose(b_b[:, 0], [*loop.matrix[:, 0], 0.0], atol=2e-12)
    np.testing.assert_allclose(b_c[:, 0], [1.0, 0.0, 0.0], atol=2e-12)
    assert count == 7
    assert row == {"x": 1.5, "y": -0.5, "gate": 1.0}


def test_prechecked_baseline_is_not_evaluated_again():
    loop, row = _AffineMap(), {"x": 0.0, "y": 0.0, "gate": 0.0}
    baseline = loop.advance(row, 0.0)
    result = linearize_loop(loop, row, 0.0, loop.cols, baseline=baseline)
    assert result.evaluations == 4
    assert loop.calls == 5


def test_coordinate_difference_handles_wrapped_angles_without_losing_their_mode():
    class Wrapped(_AffineMap):
        cols = [("angle", ["theta"], 1.0), ("mode", ["gate"], 1.0)]

        def advance(self, row, t, view=None):
            self.calls += 1
            return {"theta": math.remainder(row["theta"] + math.tau, math.tau),
                    "gate": row["gate"]}

    loop = Wrapped()
    row = {"theta": math.pi, "gate": 0.0}
    linear = linearize_loop(loop, row, 0.0, loop.cols)
    assert linear.active == (0,)
    assert linear.phi[0, 0] == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(coordinate_difference(linear.final, linear.initial, loop.cols), 0.0)


def test_split_input_differences_use_the_same_wrapped_angle_coordinates():
    class Wrapped(_AffineMap):
        cols = [("angle", ["theta"], 1.0)]

        def advance(self, row, t, view=None):
            offset = view[1] if view is not None else 0.0
            return {"theta": math.remainder(row["theta"] + offset, math.tau)}

    loop = Wrapped()
    phi, b_b, b_c, count = _period_map(
        loop, {"theta": math.pi - 5e-5}, 0.0, loop.cols, [("theta", 1.0)])
    for matrix in (phi, b_b, b_c):
        np.testing.assert_allclose(matrix, [[1.0]], atol=1e-10)
    assert count == 5


def test_mode_changes_are_reported_instead_of_becoming_a_smooth_mode():
    class Guarded(_AffineMap):
        def advance(self, row, t, view=None):
            result = super().advance(row, t, view)
            result["gate"] = float(row["x"] > 0.0)
            return result

    loop = Guarded()
    linear = linearize_loop(loop, {"x": 0.0, "y": 0.0, "gate": 0.0}, 0.0, loop.cols)
    assert linear.changed_modes == ("gate",)


def test_discrete_state_view_uses_owner_declarations_and_existing_state_types():
    class State:
        discrete_state_names = ("elapsed",)

        def get_state(self):
            return {"elapsed": 0.0, "count": 0, "enabled": True, "integral": 0.0}

    registry = StateRegistry({"custom": State()})
    assert registry.discrete_names == {"custom.elapsed", "custom.count", "custom.enabled"}
    assert "custom.integral" in registry.names


def test_feedback_state_view_does_not_duplicate_or_remove_restart_columns():
    class State:
        linearization_state_names = ("feedback",)

        def get_state(self):
            return {"feedback": 1.0j, "cached_sum": 2.0}

    registry = StateRegistry({"custom": State()})
    assert registry.linearization_columns == ("custom.feedback.re", "custom.feedback.im")
    assert "custom.cached_sum" in registry.columns


class _LinearPlant:
    state_names = ("x", "y")
    outputs_need_inputs = False

    def __init__(self, matrix):
        self.matrix = matrix
        self.state = SimpleNamespace(x=0.0, y=0.0)
        self.inp = SimpleNamespace(feedback=0.0)
        self.out = SimpleNamespace(x=0.0)

    def set_outputs(self, t):
        self.out.x = self.state.x

    def rhs(self, t):
        result = self.matrix @ np.array([self.state.x, self.state.y])
        return result[0] + self.inp.feedback, result[1]


class _Feedback:
    state_names = ()
    outputs_need_inputs = True

    def __init__(self):
        self.state = Empty()
        self.inp = SimpleNamespace(u=0.0, feedback=0.0)
        self.out = SimpleNamespace(y=0.0)

    def set_outputs(self, t):
        self.out.y = 0.2 * self.inp.u + 0.5 * self.inp.feedback

    def rhs(self, t):
        return ()


def _params(**overrides):
    return peslite.load(EXAMPLES / "buck-example.pes", **{
        "simulation.t_end": 0.01, "simulation.energy_check": "off",
        "simulation.output.states": False, "simulation.output.signals": False,
        "simulation.output.energy": False, "simulation.progress.enable": False,
        **overrides,
    })


def _linear_sim(matrix, *, feedback=False, **overrides):
    def factory(params):
        plant = _LinearPlant(matrix)
        subsystems, connections = {"plant": plant}, {}
        if feedback:
            gain = _Feedback()
            subsystems["feedback"] = gain
            connections = {(gain, "u"): (plant, "x"),
                           (gain, "feedback"): (gain, "y"),
                           (plant, "feedback"): (gain, "y")}
        model = Model(subsystems, connections)
        return SimpleNamespace(units={}, model=model, _port_bindings={},
                               state_parts=lambda: [("", model)])

    sim = peslite.Simulation(_params(**{"simulation.solver.dt": 1e-4, **overrides}),
                             parts={"system": factory})
    sim.run(out_dir=None)
    return sim


@pytest.mark.parametrize("growth,frequency", [(-4.0, 0.0), (2.0, 3.0), (-0.05, 5.0), (0.0, 0.0)])
@pytest.mark.parametrize("feedback", [False, True])
def test_public_addon_matches_independent_continuous_matrix(growth, frequency, feedback):
    w = math.tau * frequency
    matrix = np.array([[growth, -w], [w, growth]])
    sim = _linear_sim(matrix, feedback=feedback)
    before = sim.result.final_states()
    states = sim._states.read()
    # The reference is exactly linear: a larger perturbation keeps the existing algebraic
    # solver's absolute residual floor well below the finite-difference signal.
    analysis = small_signal(sim, period=0.01, relative_step=0.01)
    effective = matrix.copy()
    if feedback:
        effective[0, 0] += 0.2 / (1.0 - 0.5)
        assert sim.system.model.algebraic_loops
    np.testing.assert_allclose(analysis.phi, expm(effective * analysis.period), rtol=1e-8, atol=1e-10)
    assert analysis.coordinates == ("plant.x", "plant.y")
    assert analysis.evaluations == 5
    assert analysis.periodic_error == pytest.approx(0.0, abs=1e-13)
    assert sim.result.final_states() == before
    assert sim._states.read() == states
    assert analysis.wall_time > 0.0
    assert "multiplier" in analysis.mode_table
    assert not hasattr(sim, "small_signal")
    expected = np.linalg.eigvals(effective)
    np.testing.assert_allclose(np.sort_complex(analysis.exponents), np.sort_complex(expected), atol=1e-6)


@pytest.mark.parametrize("solver,settings", [
    ("fixed", (2e-4, 1e-4, 5e-5)), ("adaptive", (1e-3, 1e-5, 1e-7)),
])
def test_integration_refinement_converges_to_the_existing_exact_linear_reference(solver, settings):
    w = math.tau * 400.0
    matrix = np.array([[-30.0, -w], [w, -30.0]])
    period = 0.001
    reference = expm(matrix * period)
    errors = []
    for value in settings:
        options = ({"simulation.solver.dt": value} if solver == "fixed" else {
            "simulation.solver.type": "adaptive", "simulation.solver.method": "DP45",
            "simulation.solver.rtol": value, "simulation.solver.atol": value * 1e-5})
        sim = _linear_sim(matrix, **{"simulation.output.period": period, **options})
        analysis = small_signal(sim, period=period)
        errors.append(np.linalg.norm(analysis.phi - reference))
    assert errors[1] < errors[0] / 10.0
    assert errors[2] < errors[1] / 10.0
    assert errors[-1] < (1e-5 if solver == "fixed" else 1e-7)


def test_external_profiler_accounts_for_actual_rebuilds_advances_and_matrix_work():
    from benchmark_small_signal import profile_analysis

    sim = _linear_sim(-np.eye(2))
    before = sim.result.final_states()
    analysis, sample = profile_analysis(sim, period=0.01)
    assert sample["reconstructions"] == sample["period_advances"] == analysis.evaluations
    stages = sample["stage_seconds"]
    assert all(value > 0.0 for value in stages.values())
    assert sum(stages.values()) == pytest.approx(sample["profiled_wall_seconds"])
    assert stages["eigenvalues"] > 0.0
    assert sim.result.final_states() == before


def test_neutral_and_zero_multipliers_are_preserved():
    result = small_signal(_linear_sim(np.zeros((2, 2))), period=0.01)
    assert result.stability == "near-unit-circle"
    result = dataclasses.replace(result, multipliers=np.array([0.0j, 0.5 + 0.0j]))
    assert result.stability == "stable"
    assert result.exponents[0].real == -math.inf
    assert result.exponents[0].imag == 0.0


def test_a_failed_run_is_not_treated_as_a_completed_operating_point():
    sim = _linear_sim(-np.eye(2))
    sim.result.summary["stop_reason"] = "diverged before the requested end"
    with pytest.raises(ValueError, match="stopped early"):
        small_signal(sim, period=0.01)


@pytest.mark.parametrize("mode", ["pwm_averaging", "switching", "averaging"])
def test_buck_modes_reuse_complete_state_without_artificial_unit_multipliers(mode):
    sim = peslite.Simulation(_params(**{
        "simulation.t_end": 0.05, "units.converter.bridge.model": mode,
    }))
    sim.run(out_dir=None)
    period = 1e-4 if mode == "averaging" else None
    analysis = small_signal(sim, period=period)
    assert analysis.stability == "stable"
    assert analysis.phi.shape == (len(analysis.coordinates),) * 2
    assert analysis.evaluations <= 1 + 2 * len(analysis.coordinates)
    assert not any(".prot." in name or ".startup." in name or name.endswith(".on")
                   for name in analysis.coordinates)
    assert any(".integral" in name for name in analysis.coordinates)
    if mode != "averaging":
        assert "converter.pwm.d" in analysis.coordinates
        assert "converter.pwm.shadow.d" in analysis.coordinates
    tighter = small_signal(sim, period=period, relative_step=5e-5)
    np.testing.assert_allclose(tighter.phi, analysis.phi, rtol=2e-4, atol=2e-6)


def test_continuous_buck_matches_independent_circuit_and_PI_matrix():
    params = _params(**{"simulation.t_end": 0.05, "units.converter.bridge.model": "averaging"})
    sim = peslite.Simulation(params)
    sim.run(out_dir=None)
    result = small_signal(sim, period=1e-4, relative_step=1e-3)
    cfg = params.unit("converter")
    c, r = params.buses["output"].c, params.elements["load"].r
    l, loss, vin = cfg.filter.l_f, cfg.filter.r_f, cfg.dclink.vdc_ref
    voltage, current = cfg.ctrl.loops["DCVoltageControl"], cfg.ctrl.loops["InductorCurrentControl"]
    # Coordinates: output voltage, inductor current, outer PI integral, inner PI integral.
    # The topology's v/vin duty feedforward cancels the -v term in the inductor equation.
    expected = np.array([
        [-1.0 / (r * c), 1.0 / c, 0.0, 0.0],
        [-vin * current.kp * voltage.kp / l, -(vin * current.kp + loss) / l,
         vin * current.kp * voltage.ki / l, vin * current.ki / l],
        [-1.0, 0.0, 0.0, 0.0],
        [-voltage.kp, -1.0, voltage.ki, 0.0],
    ])
    assert result.phi.shape == expected.shape
    np.testing.assert_allclose(result.phi, expm(expected * result.period), rtol=1e-6, atol=1e-8)


def test_adc_memory_and_midperiod_restart_use_the_same_map():
    sim = peslite.Simulation(_params(**{
        "simulation.t_end": 0.050025,
        "units.converter.meas.average": ["current", "voltage"],
        "units.converter.meas.window": 2.5e-5,
    }))
    sim.run(out_dir=None)
    result = small_signal(sim)
    assert {"converter.meas.x_current", "converter.meas.x_voltage"} <= set(result.coordinates)
    assert result.stability == "stable"


@pytest.mark.parametrize("protection", [False, True])
def test_rms_history_is_checked_without_creating_feedback_modes(protection):
    sim = peslite.Simulation(_params(**{
        "simulation.t_end": 0.05, "units.converter.meas.rms.current": 0.001,
        "units.converter.protection.rms_current.enable": protection,
        "units.converter.protection.rms_current.limit_pu": 1.0,
    }))
    sim.run(out_dir=None)
    result = small_signal(sim)
    assert result.stability == "stable"
    assert any(".rms_current." in name for name in sim.result.final_states())
    assert not any(".rms_current." in name for name in result.coordinates)


def test_rms_periodicity_uses_logical_history_without_mutating_restart_storage():
    adc = ADC(MeasurementPorts({"current": lambda: 1.0}), 0.1, 0.1, rms={"current": 0.3})
    prefix = "rms_current."
    first = {**adc.get_state(), prefix + "count": 3,
             **{f"{prefix}sample_{i}": value for i, value in enumerate((1.0, 4.0, 9.0))}}
    rotated = {**first, prefix + "index": 1,
               **{f"{prefix}sample_{i}": value for i, value in enumerate((9.0, 1.0, 4.0))}}
    before = dict(rotated)
    assert adc.periodic_state(first) == adc.periodic_state(rotated)
    changed = {**rotated, prefix + "sample_1": 2.0}
    assert adc.periodic_state(first) != adc.periodic_state(changed)
    assert rotated == before
    assert prefix + "index" not in adc.periodic_state(rotated)


@pytest.mark.parametrize("limit,trip", [(0.1, True), (1.0, False)])
def test_rms_warmup_defers_analysis_until_the_complete_operating_state_repeats(monkeypatch,
                                                                           limit, trip):
    p = _params(**{"simulation.t_end": 0.05, "units.converter.bridge.model": "averaging",
                   "units.converter.meas.rms.current": 0.001})
    settled = peslite.Simulation(p)
    settled.run(out_dir=None)
    row = settled.result.final_states()
    start = row.pop("t")
    row = {key: value for key, value in row.items()
           if ".rms_current." not in key and not key.startswith("solver.")}
    p = p.replace(**{"units.converter.protection.rms_current.enable": True,
                     "units.converter.protection.rms_current.limit_pu": limit,
                     "units.converter.protection.hold": 0.0})
    p = dataclasses.replace(p, simulation=dataclasses.replace(
        p.simulation, initial=dataclasses.replace(p.simulation.initial, t=start, states=row),
        t_end=start + 0.002))
    warmup = peslite.Simulation(p)
    warmup.run(t_end=start + 0.0001, out_dir=None)
    module = __import__(small_signal.__module__, fromlist=["linearize_loop"])
    linearize = module.linearize_loop
    calls = []

    def record(loop, row, t0, *args, **kwargs):
        calls.append(t0)
        return linearize(loop, row, t0, *args, **kwargs)

    monkeypatch.setattr(module, "linearize_loop", record)
    with pytest.raises(ValueError, match="discrete state.*rms_current.count"):
        small_signal(warmup, period=0.0001)
    assert calls == []
    sim = peslite.Simulation(p)
    report = small_signal_report(sim, period=0.0001, interval=0.0004, out_dir=None)
    labels = [entry["changes"]["analysis.stability"] for entry in report.changes
              if "analysis.stability" in entry["changes"]]
    assert labels == (["transient", "unavailable"] if trip else ["transient", "stable"])
    assert sim.result.tripped == trip
    if trip:
        assert calls == []
        assert sim.result.summary["converter.trip_cause"] == "rms_current"
    else:
        assert calls and min(calls) >= start + 0.001


def test_normal_run_does_not_construct_a_map_or_compute_modes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary simulations must not invoke stability analysis")

    monkeypatch.setattr("peslite.solver.simulation.SystemLoop", forbidden)
    monkeypatch.setattr(np.linalg, "eigvals", forbidden)
    sim = peslite.Simulation(_params())
    sim.run(out_dir=None)
    assert sim.result.summary["t_stop"] == sim.p.simulation.t_end


def test_loop_scales_plant_coordinates_by_rating_with_a_single_rate_solver(gfl):
    sim = peslite.Simulation(gfl())
    cols = SystemLoop(sim, period=0.02, frame_frequency=0.0).coordinates(list(sim._states.read_flat()))
    assert {group[0]: scale for _kind, group, scale in cols}["pcc.u_C.re"] == pytest.approx(
        sim.p.base.v_phase_peak)


def test_loop_preserves_output_grid_when_disabling_histories(gfl):
    sim = peslite.Simulation(gfl())
    loop = SystemLoop(sim, period=0.02, frame_frequency=0.0)
    assert loop._p.simulation.output.period == sim.p.simulation.output.period
    assert not loop._p.simulation.output.states
    assert not loop._p.simulation.output.signals


@pytest.mark.parametrize("model", ["switching", "averaging"])
def test_loop_reads_controller_angle_coordinates_from_state_owner_metadata(gfl, model):
    sim = peslite.Simulation(gfl(**{"units.vsc.bridge.model": model}))
    angles = sim._states.angle_names
    assert "vsc.ctrl.SRFPLL.theta" in angles
    if model == "switching":
        assert {"vsc.ctrl.held.SRFPLL.theta", "vsc.ctrl.command.theta"} <= angles
    loop = SystemLoop(sim, period=0.02, frame_frequency=0.0)
    assert loop._angles == set(angles)


def test_analysis_requires_settled_state_and_does_not_silently_trim():
    sim = peslite.Simulation(_params(**{"simulation.t_end": 0.001}))
    sim.run(out_dir=None)
    with pytest.raises(ValueError, match="not periodic"):
        small_signal(sim)


def test_analysis_rejects_events_inside_the_period():
    sim = peslite.Simulation(_params(**{
        "simulation.t_end": 0.05,
        "events.change": {"type": "set", "t": 0.050025,
                          "set": {"units.converter.ctrl.references.voltage": 18.0}},
    }))
    sim.run(out_dir=None)
    with pytest.raises(ValueError, match="simulation-file event"):
        small_signal(sim)


def test_analysis_rejects_instance_only_parts_and_unfinished_runs():
    sim = peslite.Simulation(_params())
    with pytest.raises(ValueError, match="completed"):
        small_signal(sim)
    sim = peslite.Simulation(_params(), instances={"solver": sim.solver})
    sim.run(out_dir=None)
    with pytest.raises(ValueError, match="factories"):
        small_signal(sim)


@pytest.mark.parametrize("name", ["relative_step", "periodic_tolerance", "multiplier_tolerance", "period"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_analysis_rejects_invalid_options(name, value):
    sim = _linear_sim(-np.eye(2))
    with pytest.raises(ValueError, match=name):
        small_signal(sim, **{name: value})


def test_declared_schedules_and_waveforms_determine_the_period(gfl):
    sim = peslite.Simulation(gfl())
    loop = SystemLoop(sim, frame_frequency=0.0, repeat_forcing=True)
    assert loop.period == pytest.approx(1.0 / sim.p.sources["grid"].f)
    with pytest.raises(ValueError, match="repeat"):
        SystemLoop(sim, period=sim.unit().bridge.period, frame_frequency=0.0, repeat_forcing=True)
    assert macro_period([1.0, math.sqrt(2.0)]) is None
    assert active_coordinates(loop.coordinates(sim.state_names()))


def test_period_must_be_supplied_for_an_autonomous_continuous_model():
    sim = _linear_sim(-np.eye(2))
    with pytest.raises(ValueError, match="supply.*period"):
        small_signal(sim)


def test_split_error_map_does_not_expand_to_a_slow_waveform_beat_period(gfl):
    sim = peslite.Simulation(gfl(**{"sources.grid.f": 49.8}))
    assert SystemLoop(sim).period == pytest.approx(sim.unit().bridge.period)


def test_resonant_forcing_is_declared_by_the_loop_not_by_device_name():
    params = peslite.load(EXAMPLES / "wpt-example.pes")
    sim = peslite.Simulation(params)
    loop = SystemLoop(sim, frame_frequency=0.0, repeat_forcing=True)
    forcing = 1.0 / params.unit("charger.primary").ctrl.references.frequency
    assert forcing in loop.periods
    assert loop.period is not None
    for period in loop.periods:
        assert loop.period / period == pytest.approx(round(loop.period / period))


def test_other_addons_and_normal_runs_do_not_import_stability_analysis():
    root = Path(peslite.__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-c", """
import sys
import peslite
from peslite.addons import plot_result
assert 'peslite.addons.functions.small_signal' not in sys.modules
assert not hasattr(peslite.Simulation, 'small_signal')
"""], env=env, check=True, capture_output=True, text=True)
