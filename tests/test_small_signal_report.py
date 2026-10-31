"""Sparse reports reuse observations without adding a second simulator or signal namespace."""

import importlib
import json
import math
import weakref
import warnings
from types import SimpleNamespace

import numpy as np
import peslite
import pytest

from conftest import EXAMPLES
from peslite.addons.functions.small_signal import small_signal_report
from peslite.solver.linearization import linearize_loop
from peslite.solver.model import Empty, Model
from peslite.solver.simulation import SystemLoop

pytestmark = pytest.mark.slow


def _params(**overrides):
    return peslite.load(EXAMPLES / "buck-example.pes", **{
        "simulation.t_end": 0.002, "simulation.energy_check": "off",
        "simulation.output.states": False, "simulation.output.signals": False,
        "simulation.output.energy": False, "simulation.output.period": 0.0001,
        "simulation.progress.enable": False, **overrides,
    })


def _linear_sim(*, initial=0.0, growth=-4.0, decay_from_load=False, fail_at=None, **overrides):
    class Plant:
        state_names = ("x",)
        outputs_need_inputs = False

        def __init__(self):
            self.state = SimpleNamespace(x=initial)
            self.inp, self.out = Empty(), SimpleNamespace(x=initial)
            self.growth = growth

        def set_outputs(self, t):
            self.out.x = self.state.x

        def rhs(self, t):
            if fail_at is not None and fail_at + 1e-12 < t <= self.end:
                raise FloatingPointError("deliberate integration failure")
            return (self.growth * self.state.x,)

    def factory(params):
        plant = Plant()
        plant.end = params.simulation.t_end
        def retune(cfg):
            if decay_from_load:
                plant.growth = -1.0 / (cfg.elements["load"].r * cfg.buses["output"].c)

        retune(params)
        model = Model({"plant": plant}, {})
        return SimpleNamespace(units={}, model=model, _port_bindings={},
                               state_parts=lambda: [("", model)],
                               apply=lambda change: retune(change.params),
                               signals=lambda t: {"plant.signal": plant.out.x})

    return peslite.Simulation(_params(**overrides), parts={"system": factory})


def _values(report, name):
    return [(entry["t"], entry["changes"][name]) for entry in report.changes if name in entry["changes"]]


def test_constant_report_has_one_row_and_reuses_main_run_endpoints(monkeypatch):
    calls = []
    advance = SystemLoop.advance

    def count(self, row, t, view=None):
        calls.append(t)
        return advance(self, row, t, view)

    monkeypatch.setattr(SystemLoop, "advance", count)
    sim = _linear_sim()
    report = small_signal_report(sim, interval=0.0005, period=0.0001,
                                 watch=["plant.signal", "plant.x"], out_dir=None)
    assert report.checks == 5
    assert len(report.changes) == 1
    assert _values(report, "analysis.stability") == [(0.0, "stable")]
    assert report.reused_baselines == 4
    assert report.evaluations == len(calls) == 2 * report.checks + 1
    assert report.files == ()
    assert report.changes[0]["changes"]["analysis.growth_rate"] == pytest.approx(-4.0, rel=1e-7)


def test_transient_skips_perturbations_until_a_periodic_state_is_established(monkeypatch):
    module = importlib.import_module("peslite.addons.functions.small_signal")

    def forbidden(*args, **kwargs):
        raise AssertionError("non-periodic operating states must not be linearized")

    monkeypatch.setattr(module, "linearize_loop", forbidden)
    sim = _linear_sim(initial=1.0)
    report = small_signal_report(sim, period=0.0001, periodic_tolerance=1e-6,
                                 watch=["plant.x"], out_dir=None)
    assert _values(report, "analysis.stability") == [(0.0, "transient")]
    first = report.changes[0]["changes"]
    assert "not periodic" in first["analysis.reason"]
    assert first["analysis.amplification"] is None
    assert first["analysis.growth_rate"] is None
    assert first["analysis.frequency"] is None
    assert report.evaluations == 1


def test_report_starts_analysis_at_the_first_selected_periodic_operating_state(monkeypatch):
    module = importlib.import_module("peslite.addons.functions.small_signal")
    linearize = module.linearize_loop
    calls = []

    def record(loop, row, t0, *args, **kwargs):
        calls.append(t0)
        return linearize(loop, row, t0, *args, **kwargs)

    monkeypatch.setattr(module, "linearize_loop", record)
    sim = _linear_sim(initial=1.0, growth=-4000.0, **{"simulation.t_end": 0.004})
    report = small_signal_report(sim, period=0.0001, interval=0.0005, out_dir=None)
    assert _values(report, "analysis.stability") == [(0.0, "transient"), (0.0015, "stable")]
    assert calls == [0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]


def test_exact_extra_times_do_not_add_output_rows_or_control_interrupts(tmp_path):
    params = _params(**{"simulation.output.states": True})
    ordinary = peslite.Simulation(params)
    ordinary.run(out_dir=tmp_path / "plain")
    observed = peslite.Simulation(params)
    rows = []
    observed.run(out_dir=tmp_path / "observed", times=[0.000037],
                 observe=lambda t, states, values, **_: rows.append((t, dict(states), dict(values()))))
    assert any(t == 0.000037 for t, _states, _values in rows)
    assert observed.result.t.tolist() == ordinary.result.t.tolist()
    assert observed.unit().bridge.control_index == ordinary.unit().bridge.control_index
    np.testing.assert_allclose(list(observed.result.final_states().values()),
                               list(ordinary.result.final_states().values()), rtol=1e-8, atol=1e-8)


def test_observer_reuses_existing_state_and_signal_reads(monkeypatch):
    sim = _linear_sim()
    counts = {"read": 0, "signals": 0}
    read, signals = sim._states.read, sim.system.signals

    def state():
        counts["read"] += 1
        return read()

    def signal(t):
        counts["signals"] += 1
        return signals(t)

    monkeypatch.setattr(sim._states, "read", state)
    monkeypatch.setattr(sim.system, "signals", signal)
    rows = []
    def observe(t, states, read_values, **_):
        values = read_values()
        assert read_values() is values
        rows.append((t, states, values))

    sim.run(out_dir=None, observe=observe, observe_values=["plant.signal"])
    # One state read during initialization, then one per observed snapshot. No second read
    # for flattening, aliases, watched values, or a baseline recorder.
    assert counts["read"] == len(rows) + 1
    assert counts["signals"] == len(rows)


def test_event_parameters_and_watches_keep_canonical_names_and_same_time_merge(tmp_path):
    params = _params(**{
        "simulation.t_end": 0.004,
        "events.reference": {"type": "set", "t": 0.001237,
                             "set": {"units.converter.ctrl.references.voltage": 18.0}},
        "events.load": {"type": "set", "t": 0.001237, "set": {"elements.load.r": 8.0}},
    })
    sim = peslite.Simulation(params)
    report = small_signal_report(sim, times=[0.002037], watch=["output.u", "elements.load.r"],
                                 out_dir=tmp_path, name="buck-example")
    path = "units.converter.ctrl.references.voltage"
    assert _values(report, path) == [(0.0, 24.0), (0.001237, 18.0)]
    assert _values(report, "elements.load.r") == [(0.0, 6.0), (0.001237, 8.0)]
    assert any(entry["t"] == 0.002037 for entry in report.changes)
    assert sum(entry["t"] == 0.001237 for entry in report.changes) == 1
    assert "output.u" in report.changes[0]["changes"]
    assert report.files == (tmp_path / "buck-example_report.json",)
    saved = json.loads(report.files[0].read_text())
    assert set(saved) == {"small_signal_stability"}
    assert saved["small_signal_stability"]["changes"] == list(report.changes)
    assert not list(tmp_path.glob("*.md"))
    assert all(path in sim.result.files for path in report.files)


def test_discrete_nominal_evolution_is_not_mistaken_for_a_perturbation_mode_change():
    class Map:
        cols = [("scalar", ["x"], 1.0), ("mode", ["step"], 1.0)]
        period = 1.0

        def to_frame(self, row, t, cols):
            return np.array([row["x"], row["step"]])

        def from_frame(self, x, t, cols, row):
            return {"x": x[0], "step": row["step"]}

        def advance(self, row, t):
            return {"x": 0.5 * row["x"], "step": row["step"] + 1}

    loop = Map()
    linear = linearize_loop(loop, {"x": 1.0, "step": 0}, 0.0, loop.cols)
    assert linear.changed_modes == ()
    assert linear.phi[0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize("options", [{"interval": 0}, {"interval": math.nan},
                                    {"times": [-1]}, {"times": [1]},
                                    {"change_tolerance": -1}, {"period": 0}])
def test_report_options_fail_before_running(options):
    sim = _linear_sim()
    with pytest.raises(ValueError):
        small_signal_report(sim, out_dir=None, **options)
    assert sim.result is None


def test_report_requires_a_fresh_simulation_and_does_not_run_it_twice():
    sim = _linear_sim()
    sim.run(out_dir=None)
    with pytest.raises(ValueError, match="fresh"):
        small_signal_report(sim, period=0.0001, out_dir=None)


def test_no_declared_period_becomes_an_explicit_unavailable_report():
    report = small_signal_report(_linear_sim(), out_dir=None)
    assert _values(report, "analysis.stability") == [(0.0, "unavailable")]
    assert "supply period" in report.changes[0]["changes"]["analysis.reason"]
    assert report.evaluations == 0


def test_unknown_watch_names_are_not_silently_ignored():
    with pytest.raises(ValueError, match="unknown watch"):
        small_signal_report(_linear_sim(), period=0.0001, watch=["not_a_signal"], out_dir=None)


def test_observer_errors_are_not_swallowed_as_physical_divergence():
    def fail(*_args, **_metadata):
        raise FloatingPointError("observer failure")

    with pytest.raises(FloatingPointError, match="observer failure"):
        _linear_sim().run(out_dir=None, observe=fail)


@pytest.mark.parametrize("fail_at", [None, 0.001])
def test_normal_and_early_termination_use_one_final_observation(fail_at):
    sim = _linear_sim(initial=1.0, fail_at=fail_at)
    seen = []

    def observe(t, states, values, *, final, stop_reason):
        seen.append((t, dict(states), values(), final, stop_reason))

    with warnings.catch_warnings(record=True):
        result = sim.run(out_dir=None, times=[], observe=observe,
                         observe_values=["plant.x", "plant.signal"])
    assert len(seen) == 1
    t, states, values, final, reason = seen[0]
    assert final
    assert t == result.summary["t_stop"]
    assert reason == result.summary.get("stop_reason", "")
    assert bool(reason) == (fail_at is not None)
    assert states["plant.x"] == result.final_states()["plant.x"]
    assert values == {"plant.x": states["plant.x"], "plant.signal": states["plant.x"]}


@pytest.mark.parametrize("times", [[], [0.001]])
def test_early_stop_retains_final_watches_and_its_reason_even_with_pending_windows(times):
    sim = _linear_sim(initial=1.0, fail_at=0.001)
    with pytest.warns(UserWarning, match="simulation stopped"):
        report = small_signal_report(sim, times=times, period=0.0004,
                                     watch=["plant.x", "plant.signal"], out_dir=None)
    final = sim.result.final_states()
    assert _values(report, "plant.x")[-1] == (final["t"], final["plant.x"])
    assert _values(report, "plant.signal")[-1] == (final["t"], final["plant.x"])
    assert report.changes[-1]["t"] == final["t"]
    assert report.changes[-1]["changes"]["analysis.reason"] == sim.result.summary["stop_reason"]
    assert report.changes[-1]["changes"]["analysis.stability"] == "unavailable"


def test_a_final_observation_cannot_request_more_runtime_work():
    def observe(t, states, values, *, final, stop_reason):
        return t + 0.0001 if final else None

    with pytest.raises(ValueError, match="final observation"):
        _linear_sim().run(out_dir=None, times=[], observe=observe)


def test_terminal_metadata_is_delivered_even_when_its_physical_snapshot_is_unavailable(monkeypatch):
    sim = _linear_sim(initial=1.0)
    read = sim._states.read
    calls = 0

    def state():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise FloatingPointError("unavailable terminal state")
        return read()

    monkeypatch.setattr(sim._states, "read", state)
    seen = []

    def observe(t, states, values, *, final, stop_reason):
        seen.append((t, states, values(), final, stop_reason))

    with pytest.warns(UserWarning, match="final physical snapshot unavailable"):
        result = sim.run(out_dir=None, times=[], observe=observe, observe_values=["plant.x"])
    assert len(seen) == 1
    t, states, values, final, reason = seen[0]
    assert t == sim.p.simulation.t_end and final
    assert states is None and values == {}
    assert "unavailable terminal state" in reason
    assert reason == result.summary["stop_reason"]


@pytest.mark.parametrize("error", [ValueError, TypeError, OSError])
def test_observation_failures_do_not_hide_programming_or_io_errors(error):
    from peslite.solver.simulation import _optional_output

    def read():
        raise error("not a numerical observation failure")

    with pytest.raises(error, match="not a numerical"):
        _optional_output(0.0, "custom output", read)


def test_observer_can_request_an_endpoint_without_polling_every_pwm_edge():
    sim = _linear_sim()
    seen = []

    def observe(t, states, values, **_):
        seen.append(t)
        return 0.000037 if t == 0.0 else None

    sim.run(out_dir=None, observe=observe)
    assert 0.000037 in seen
    assert len(seen) == 22  # 21 native output snapshots and one requested observation.


def test_explicit_observation_schedule_does_not_read_unused_output_snapshots(monkeypatch):
    sim = _linear_sim()
    seen, signals = [], sim.system.signals
    calls = []

    def read_signals(t):
        calls.append(t)
        return signals(t)

    monkeypatch.setattr(sim.system, "signals", read_signals)
    def observe(t, states, read_values, **_):
        read_values()
        seen.append(t)

    sim.run(out_dir=None, times=[0.000037], observe=observe, observe_values=["plant.signal"])
    assert seen == calls == [0.000037, 0.002]


def test_parameter_event_updates_the_actual_modes_without_device_specific_analysis():
    sim = _linear_sim(decay_from_load=True, **{
        "events.load": {"type": "set", "t": 0.001237, "set": {"elements.load.r": 8.0}},
    })
    report = small_signal_report(sim, period=0.0001, out_dir=None)
    growth = _values(report, "analysis.growth_rate")
    assert growth[0][0] == 0.0
    assert growth[0][1] == pytest.approx(-1.0 / (6.0 * sim.p.buses["output"].c), rel=1e-7)
    assert growth[1][0] == 0.001237
    assert growth[1][1] == pytest.approx(-1.0 / (8.0 * sim.p.buses["output"].c), rel=1e-7)
    assert _values(report, "analysis.stability") == [(0.0, "stable")]
    assert report.reused_baselines == report.checks - 1


def test_a_window_crossing_a_file_event_is_not_mislabeled_as_periodic_stability():
    sim = _linear_sim(decay_from_load=True, **{
        "events.load": {"type": "set", "t": 0.001237, "set": {"elements.load.r": 8.0}},
    })
    report = small_signal_report(sim, times=[0.0012], period=0.0001,
                                 out_dir=None)
    states = _values(report, "analysis.stability")
    assert [t for t, _value in states] == pytest.approx([0.0, 0.0012, 0.001237])
    assert [value for _t, value in states] == ["stable", "transient", "stable"]


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_real_converter_reuses_baselines_and_does_not_mutate_the_main_simulation(mode):
    params = _params(**{"units.converter.bridge.model": mode})
    plain = peslite.Simulation(params)
    plain.run(out_dir=None)
    sim = peslite.Simulation(params)
    report = small_signal_report(sim, times=[0.001], period=0.0001, out_dir=None)
    assert report.reused_baselines == 2
    assert sim.result.n_rhs == plain.result.n_rhs
    assert sim.result.final_states() == plain.result.final_states()


def test_unreused_baselines_give_the_same_report_but_cost_one_advance_per_check(monkeypatch):
    module = importlib.import_module("peslite.addons.functions.small_signal")
    sim = _linear_sim()
    reused = small_signal_report(sim, period=0.0001, interval=0.0005, out_dir=None)
    analyze = module._analyze

    def recompute(*args, **kwargs):
        kwargs["baseline"] = None
        return analyze(*args, **kwargs)

    monkeypatch.setattr(module, "_analyze", recompute)
    independent = small_signal_report(_linear_sim(), period=0.0001, interval=0.0005, out_dir=None)
    assert reused.changes == independent.changes
    assert reused.evaluations == independent.evaluations - reused.reused_baselines


@pytest.mark.parametrize("stop", [True, False])
def test_trip_is_reported_at_the_runtime_transition_even_without_a_selected_check(stop):
    sim = peslite.Simulation(_params(**{
        "simulation.stop_on_trip": stop,
        "units.converter.protection.overcurrent.enable": True,
        "units.converter.protection.overcurrent.limit_pu": 1e-4,
    }))
    report = small_signal_report(sim, period=0.0001, watch=["converter.tripped"], out_dir=None)
    at = sim.result.summary["converter.trip_time"]
    assert at is not None
    assert (at, 1.0) in _values(report, "converter.tripped")
    assert (at, "tripped operating state") in _values(report, "analysis.reason")


def test_window_mean_memory_and_exact_midcycle_observations_are_retained():
    sim = peslite.Simulation(_params(**{
        "units.converter.meas.average": ["current", "voltage"],
        "units.converter.meas.window": 2.5e-5,
    }))
    report = small_signal_report(sim, times=[0.001037], period=0.0001, out_dir=None)
    assert any(entry["t"] == 0.001037 for entry in report.changes)
    assert report.checks == 3
    assert report.reused_baselines == 2
    assert "not periodic" in _values(report, "analysis.reason")[-1][1]


@pytest.mark.parametrize("next_time", [-1.0, 0.0, math.nan, math.inf, 1.0])
def test_observer_cannot_schedule_an_invalid_or_nonadvancing_time(next_time):
    with pytest.raises(ValueError, match="next time"):
        _linear_sim().run(out_dir=None, observe=lambda *_, **__: next_time)


def test_nonzero_start_and_offset_interval_need_no_history_interpolation():
    sim = _linear_sim(**{"simulation.initial.t": 0.000037})
    report = small_signal_report(sim, period=0.0001, interval=0.0005, out_dir=None)
    assert report.changes[0]["t"] == 0.000037
    assert report.checks == 5
    assert report.reused_baselines == 4
    assert len(report.changes) == 1


def test_split_run_endpoints_are_not_reused_for_the_unsplit_reference():
    sim = peslite.Simulation(_params(**{
        "simulation.solver.subsystems": {"converter.branch_f": 0.5},
    }))
    report = small_signal_report(sim, period=0.0001, out_dir=None)
    assert report.reused_baselines == 0
    assert report.evaluations is not None and report.evaluations > 0
    assert _values(report, "analysis.integration") == [(0.0, "unsplit reference")]


def test_a_single_rate_subsystem_setting_does_not_unnecessarily_disable_baseline_reuse():
    sim = peslite.Simulation(_params(**{
        "simulation.solver.subsystems": {"converter.branch_f": 1},
    }))
    report = small_signal_report(sim, period=0.0001, out_dir=None)
    assert report.reused_baselines == 1
    assert _values(report, "analysis.integration") == [(0.0, "configured solver")]


def test_unrequested_signals_are_not_evaluated_by_the_report(monkeypatch):
    sim = _linear_sim()

    def unavailable(t):
        raise AssertionError("a state-only report must not evaluate public signals")

    monkeypatch.setattr(sim.system, "signals", unavailable)
    report = small_signal_report(sim, times=[0.001], period=0.0001, out_dir=None)
    assert _values(report, "analysis.stability") == [(0.0, "stable")]
    assert sim.result.summary["t_stop"] == 0.002


@pytest.mark.parametrize("record_signals", [False, True])
def test_optional_signal_overflow_preserves_final_time_and_physical_state(tmp_path, monkeypatch,
                                                                         record_signals):
    plain = _linear_sim(initial=1.0)
    expected = plain.run(out_dir=None)
    sim = _linear_sim(initial=1.0, **{"simulation.output.signals": record_signals})
    signals = sim.system.signals

    def overflow(t):
        if abs(t - sim.p.simulation.t_end) < 1e-10:
            raise FloatingPointError("optional end-of-run signal overflow")
        return signals(t)

    monkeypatch.setattr(sim.system, "signals", overflow)
    seen = []
    with pytest.warns(RuntimeWarning, match="optional output unavailable"):
        result = sim.run(out_dir=tmp_path, times=[], observe_values=["plant.signal", "plant.x"],
                         observe=lambda t, states, values, **_: seen.append((t, dict(states), values())))
    assert result.summary["t_stop"] == expected.summary["t_stop"]
    assert result.final_states() == expected.final_states()
    assert seen[-1][0] == sim.p.simulation.t_end
    assert seen[-1][1]["plant.x"] == result.final_states()["plant.x"]
    assert math.isnan(seen[-1][2]["plant.signal"])
    assert seen[-1][2]["plant.x"] == seen[-1][1]["plant.x"]


def test_failed_watch_values_do_not_discard_a_valid_stability_check(monkeypatch):
    sim = _linear_sim()
    signals = sim.system.signals

    def overflow(t):
        if abs(t - 0.001) < 1e-10:
            raise FloatingPointError("optional watched signal overflow")
        return signals(t)

    monkeypatch.setattr(sim.system, "signals", overflow)
    with pytest.warns(RuntimeWarning, match="optional output unavailable"):
        report = small_signal_report(sim, times=[0.001], period=0.0001,
                                     watch=["plant.signal", "elements.load.r"], out_dir=None)
    assert _values(report, "analysis.stability") == [(0.0, "stable")]
    assert _values(report, "plant.signal") == [(0.0, 0.0), (0.001, None), (0.002, 0.0)]
    assert _values(report, "elements.load.r") == [(0.0, 6.0)]


def test_optional_continuous_log_overflow_does_not_discard_the_final_state(monkeypatch):
    params = _params(**{"units.converter.bridge.model": "averaging"})
    plain = peslite.Simulation(params)
    expected = plain.run(out_dir=None)
    sim = peslite.Simulation(params)
    log = sim.unit().ctrl.continuous_log
    name = f"converter.{sim.unit().ctrl.log_names()[0]}"

    def overflow(t):
        if abs(t - params.simulation.t_end) < 1e-10:
            raise FloatingPointError("optional controller log overflow")
        return log(t)

    monkeypatch.setattr(sim.unit().ctrl, "continuous_log", overflow)
    with pytest.warns(RuntimeWarning, match="optional output unavailable"):
        report = small_signal_report(sim, watch=[name], period=0.0001, out_dir=None)
    assert sim.result.final_states() == expected.final_states()
    assert sim.result.summary["t_stop"] == params.simulation.t_end
    assert _values(report, name)[-1] == (params.simulation.t_end, None)


@pytest.mark.parametrize("failed", ["logs", "plant", "energy", "limits"])
@pytest.mark.parametrize("failure_time", [0.0, 0.001, 0.002])
def test_independent_snapshot_outputs_survive_a_different_provider_failure(tmp_path, monkeypatch,
                                                                          failed, failure_time):
    params = _params(**{"units.converter.bridge.model": "averaging",
                        "simulation.output.states": True,
                        "simulation.output.signals": True,
                        "simulation.output.energy": True,
                        "simulation.energy_check": "warn", "simulation.solver.phs_check_step": 1})
    plain = peslite.Simulation(params)
    expected = plain.run(out_dir=tmp_path / "plain")
    sim = peslite.Simulation(params)
    ctrl = sim.unit().ctrl
    key = ctrl.log_names()[0]
    name = f"converter.{key}"
    owner, attribute = {"logs": (ctrl, "continuous_log"), "plant": (sim.system, "signals"),
                        "energy": (sim.system.model, "energy_balance"),
                        "limits": (ctrl, "observe_limits")}[failed]
    original = getattr(owner, attribute, lambda t: None)
    failures = []

    def read(t, *args):
        if abs(t - failure_time) < 1e-12:
            failures.append(t)
            raise FloatingPointError("deliberate optional output failure")
        return original(t, *args)

    monkeypatch.setattr(owner, attribute, read, raising=False)
    observed = []

    def observe(t, states, values, *, final, stop_reason):
        assert final and not stop_reason
        observed.append(values())
        assert values() is observed[-1]

    with pytest.warns(RuntimeWarning, match="optional output unavailable"):
        result = sim.run(out_dir=tmp_path / "failed", times=[], observe=observe,
                         observe_values=[name, "output.u"])
    assert failures == [failure_time]
    assert result.final_states() == expected.final_states()
    assert result.summary["t_stop"] == params.simulation.t_end
    assert result.states["t"][-1] == params.simulation.t_end
    if failed != "plant":
        np.testing.assert_array_equal(result.t, expected.t)
        np.testing.assert_array_equal(result.plant["output.u"], expected.plant["output.u"])
        assert observed[-1]["output.u"] == expected.plant["output.u"][-1]
    elif failure_time == params.simulation.t_end:
        assert math.isnan(observed[-1]["output.u"])
    else:
        assert observed[-1]["output.u"] == expected.plant["output.u"][-1]
    if failed != "energy":
        np.testing.assert_array_equal(result.energy["t"], expected.energy["t"])
    if failed == "logs":
        index = np.flatnonzero(np.isclose(result.ctrl["converter.t"], failure_time,
                                          atol=1e-12, rtol=0.0))[0]
        assert math.isnan(result.ctrl[name][index])
        if failure_time == params.simulation.t_end:
            assert math.isnan(observed[-1][name])
        else:
            assert observed[-1][name] == expected.ctrl[name][-1]
    else:
        np.testing.assert_array_equal(result.ctrl[name], expected.ctrl[name])


@pytest.mark.parametrize("explicit_names", [False, True])
def test_controller_only_watch_never_reads_unrelated_plant_signals(monkeypatch, explicit_names):
    sim = peslite.Simulation(_params())
    key = sim.unit().ctrl.log_names()[0]
    name = f"converter.{key}"

    def forbidden(t):
        raise AssertionError("a controller-only watch must not read plant signals")

    monkeypatch.setattr(sim.system, "signals", forbidden)
    declared = {"converter": iter((key,))} if explicit_names else None
    assert sim.watch_values(0.0, {"converter": {key: 0.4}}, names=[name],
                            ctrl_names=declared) == {name: 0.4}


def test_plant_only_watch_never_reads_unrelated_controller_logs():
    sim = _linear_sim()
    sim.run(out_dir=None)

    def forbidden():
        raise AssertionError("a plant-only watch must not read controller logs")

    assert sim.watch_values(0.0, forbidden, names=["plant.signal"]) == {"plant.signal": 0.0}


@pytest.mark.parametrize("failed", ["plant", "logs"])
def test_watch_provider_failure_does_not_erase_valid_values_from_another_provider(failed):
    sim = peslite.Simulation(_params())
    key = sim.unit().ctrl.log_names()[0]
    log_name = f"converter.{key}"

    def plant():
        if failed == "plant":
            raise FloatingPointError("unrelated plant output")
        return {"output.u": 24.0}

    def logs():
        if failed == "logs":
            raise FloatingPointError("unrelated controller log")
        return {"converter": {key: 0.4}}

    with pytest.warns(RuntimeWarning, match="optional output unavailable"):
        values = sim.watch_values(0.0, logs, plant=plant, names=[log_name, "output.u"])
    assert math.isnan(values["output.u" if failed == "plant" else log_name])
    assert values[log_name if failed == "plant" else "output.u"] == (
        0.4 if failed == "plant" else 24.0)


@pytest.mark.parametrize("on", [False, True])
def test_report_connection_commands_include_the_inferred_initial_value(on):
    path = "units.converter.connected"
    sim = peslite.Simulation(_params(**{
        "events.switch": {"type": "set", "t": 0.001, "set": {path: on}},
    }))
    report = small_signal_report(sim, period=0.0001, watch=[path], out_dir=None)
    assert _values(report, path) == [(0.0, not on), (0.001, on)]


def test_boolean_changes_do_not_use_the_numerical_change_tolerance():
    path = "units.converter.connected"
    sim = peslite.Simulation(_params(**{
        "events.switch": {"type": "set", "t": 0.001, "set": {path: False}},
    }))
    report = small_signal_report(sim, period=0.0001, change_tolerance=2.0, out_dir=None)
    assert _values(report, path) == [(0.0, True), (0.001, False)]


def test_observer_reads_source_commands_and_effective_parameters_at_runtime(gfl):
    connected, resistance = "sources.grid.connected", "sources.grid.r"
    sim = peslite.Simulation(gfl(**{
        "simulation.output.states": False, "simulation.output.signals": False,
        "simulation.t_end": 0.002, "simulation.energy_check": "off",
        "events.source": {"type": "set", "t": 0.001,
                          "set": {connected: False, resistance: 0.1}},
    }))
    rows = []
    sim.run(out_dir=None, times=[0.0], observe_values=[connected, resistance],
            observe=lambda t, states, values, **_: rows.append((t, values())))
    assert rows[0] == (0.0, {connected: True, resistance: sim.p.sources["grid"].r})
    assert rows[1] == (0.001, {connected: False, resistance: 0.1})
    assert sim.watch_values(0.002, {}, names=[connected, resistance]) == rows[1][1]


def test_connection_reads_share_the_same_path_for_all_switchable_sections(gfl):
    paths = [f"{target}.connected" for target in
             ("units.vsc", "sources.grid", "branches.line", "elements.load")]
    sim = peslite.Simulation(gfl(**{
        "simulation.output.states": False, "simulation.output.signals": False,
        "simulation.t_end": 0.002, "simulation.energy_check": "off",
        "buses.remote": {"c": 1e-5},
        "branches.line": {"bus1": "pcc", "bus2": "remote", "l": 0.001, "r": 0.1},
        "elements.load": {"type": "load", "bus": "remote", "l": 0.001, "r": 10.0},
        "events.switch": {"type": "set", "t": 0.001,
                          "set": dict.fromkeys(paths, False)},
    }))
    rows = []
    sim.run(out_dir=None, times=[0.0], observe_values=paths,
            observe=lambda t, states, values, **_: rows.append((t, values())))
    assert rows[0] == (0.0, dict.fromkeys(paths, True))
    assert rows[1] == (0.001, dict.fromkeys(paths, False))


def test_initial_event_values_come_from_the_applied_parameter_snapshot():
    path = "units.converter.ctrl.references.voltage"
    sim = peslite.Simulation(_params(**{
        "events.reference": {"type": "set", "t": 0.0, "set": {path: 18.0}},
    }))
    report = small_signal_report(sim, period=0.0001, out_dir=None)
    assert _values(report, path) == [(0.0, 18.0)]


def test_report_reads_values_only_at_checks_not_at_reused_endpoints(monkeypatch):
    sim = _linear_sim()
    signals, calls = sim.system.signals, []

    def count(t):
        calls.append(t)
        return signals(t)

    monkeypatch.setattr(sim.system, "signals", count)
    small_signal_report(sim, period=0.0001, interval=0.0005,
                        watch=["plant.signal"], out_dir=None)
    assert calls == [0.0, 0.0005, 0.001, 0.0015, 0.002]


@pytest.mark.parametrize("watched", [(), ("output.u",)])
def test_report_does_not_enable_unrequested_control_logs(watched):
    sim = peslite.Simulation(_params())
    small_signal_report(sim, watch=watched, period=0.0001, out_dir=None)
    assert not sim.unit().ctrl.logging


def test_report_enables_control_logs_when_explicitly_watched():
    sim = peslite.Simulation(_params())
    name = f"converter.{sim.unit().ctrl.log_names()[0]}"
    report = small_signal_report(sim, watch=[name], period=0.0001, out_dir=None)
    assert sim.unit().ctrl.logging
    assert name in report.changes[0]["changes"]


def test_completed_analysis_windows_release_their_restart_rows_during_the_run(monkeypatch):
    module = importlib.import_module("peslite.addons.functions.small_signal")
    original_point, original_analyze = module._ReportPoint, module._analyze
    live = peak = 0
    during_run = []
    sim = _linear_sim(**{"simulation.t_end": 0.01})

    def released():
        nonlocal live
        live -= 1

    def point(*args):
        nonlocal live, peak
        value = original_point(*args)
        live += 1
        peak = max(peak, live)
        weakref.finalize(value, released)
        return value

    def analyze(*args, **kwargs):
        during_run.append(sim.result is None)
        return original_analyze(*args, **kwargs)

    monkeypatch.setattr(module, "_ReportPoint", point)
    monkeypatch.setattr(module, "_analyze", analyze)
    report = small_signal_report(sim, period=0.0001, interval=0.0005, out_dir=None)
    assert report.checks == len(during_run) == 21
    assert all(during_run)
    assert live == 0
    assert peak <= 2


@pytest.mark.parametrize("times", [[-1], [math.inf], [math.nan]])
def test_generic_observation_time_validation(times):
    with pytest.raises(ValueError, match="observation times"):
        _linear_sim().run(out_dir=None, observe=lambda *_, **__: None, times=times)
