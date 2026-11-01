"""Control timing, sampling clocks, interrupts and PWM registers."""

import math

import numpy as np
import pytest

import peslite
from peslite.assembly.params import ConfigError
from peslite.assembly.unit import Plant
from peslite.components.network import RCNode
from peslite.components.converter import PWMAveragingModel, ThreePhaseBridge
from peslite.components.pwm import ZOH, carrier_position
from peslite.solver.model import Model


def test_shared_plant_refreshes_same_time_held_changes_and_discontinuous_states(monkeypatch):
    nodes = {name: RCNode(1.0, 1.0, phases=1) for name in ("first", "second")}
    model = Model(nodes, {}, {(node, "i"): name for name, node in nodes.items()})
    calls = []
    for method in ("sync", "sync_zoh"):
        original = getattr(model, method)

        def traced(*args, method=method, original=original):
            calls.append(method)
            return original(*args)

        monkeypatch.setattr(model, method, traced)
    plant = Plant(model, model.get_initial_values())
    plant.hold("first", 1.0)
    plant.outputs(0.0)
    plant.hold("second", 2.0)
    plant.outputs(0.0)
    plant.outputs(0.0)
    assert nodes["first"].out.u == 1.0
    assert nodes["second"].out.u == 2.0
    assert calls == ["sync", "sync_zoh"]
    with plant.change():
        nodes["first"].state.u_C = 3.0
    plant.outputs(0.0)
    assert nodes["first"].out.u == 4.0
    assert plant.inputs_changed
    assert calls == ["sync", "sync_zoh", "sync"]


T = 5e-5  # carrier period of examples/gfl-example.pes (20 kHz)


class _Recorder:
    """Delegate to a controller or modulator and record its calls."""

    def __init__(self, inner, calls, kind):
        self._inner, self._calls, self._kind = inner, calls, kind

    def __call__(self, t, *args):
        out = self._inner(t, *args)
        if self._kind == "ctrl":
            self._calls.append((t, np.array(out.d_abc, dtype=float)))
        else:
            self._calls.append((t, args[0], np.array(args[1], dtype=float)))
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _record(p, out):
    sim = peslite.Simulation(p)
    unit = sim.unit("vsc")
    interrupts, loads = [], []
    unit.ctrl = _Recorder(unit.ctrl, interrupts, "ctrl")
    unit.bridge.modulator = _Recorder(unit.bridge.modulator, loads, "mod")
    result = sim.run(out_dir=out)
    return result, interrupts, loads[1:]  # first call resumes the registers present at the start


def _lags(interrupts, loads):
    out = []
    for t_k, duty in interrupts:
        taken = [t for t, _period, loaded in loads
                 if t >= t_k - 1e-12 and np.array_equal(loaded, duty)]
        if taken:
            out.append(taken[0] - t_k)
    return np.asarray(out)


def test_timing_defaults_follow_the_carrier(gfl):
    unit = gfl().unit("vsc")
    assert (unit.ctrl.period, unit.ctrl.computation, unit.pwm.update) == (T, 1e-6, "single")
    assert all(loop.period == pytest.approx(T) for loop in unit.ctrl.loops.values())
    assert unit.pwm.load_period == T
    assert gfl(**{"units.vsc.pwm.update": "double"}).unit("vsc").pwm.load_period == T / 2


@pytest.mark.parametrize("over, message", [
    ({"units.vsc.ctrl.period": 3e-5}, "not a multiple of half the carrier period"),
    ({"units.vsc.ctrl.computation": 5e-5}, "shorter than the control period"),
    ({"units.vsc.ctrl.computation": -1e-6}, "must be at least 0"),
    ({"units.vsc.ctrl.computation": math.nan}, "computation"),
    ({"units.vsc.ctrl.offset": -1e-6}, "offset"),
    ({"units.vsc.ctrl.offset": T}, "offset"),
    ({"units.vsc.ctrl.loops.DCVoltageControl.period": 7.5e-5},
     "not a multiple of ctrl.period"),
    ({"units.vsc.meas.period": 2e-5}, "does not divide ctrl.period"),
    ({"units.vsc.meas.period": 1e-4}, "longer than the sampling period"),
    ({"units.vsc.meas.offset": -1e-6}, "offset"),
    ({"units.vsc.meas.period": T / 2, "units.vsc.meas.offset": T / 2}, "offset"),
    ({"units.vsc.meas.average": "window"}, "expected a list of channels"),
    ({"units.vsc.meas.average": ["u_g", "unknown"]}, "unknown channel"),
    ({"units.vsc.meas.average": ["u_g", "u_g"]}, "must not be repeated"),
    ({"units.vsc.meas.average": ["u_g", "i_c"], "units.vsc.meas.window": 1e-4},
     "longer than the sampling period"),
    ({"units.vsc.pwm.update": "triple"}, "update"),
    ({"units.vsc.delay.steps": 1}, "unknown key"),
    ({"units.vsc.pwm.computation_delay.steps": 1}, "unknown key"),
    ({"units.vsc.pwm.update_period": 5e-5}, "unknown key"),
    ({"units.vsc.ctrl.sampling_period": 2.5e-5}, "unknown key"),
])
def test_timing_configuration_is_checked(gfl, over, message):
    with pytest.raises(ConfigError, match=message):
        gfl(**over)


def test_single_update_applies_a_duty_one_carrier_period_after_its_sample(gfl, tmp_path):
    _result, interrupts, loads = _record(gfl(), tmp_path)
    assert all(t / T == pytest.approx(round(t / T)) and period == pytest.approx(T)
               for t, period, _duty in loads)
    lags = _lags(interrupts, loads)
    assert len(lags) > 0.9 * len(interrupts)
    assert np.allclose(lags, T)


def test_zero_computation_is_an_ideal_controller(gfl, tmp_path):
    _result, interrupts, loads = _record(gfl(**{"units.vsc.ctrl.computation": 0.0}), tmp_path)
    assert np.allclose(_lags(interrupts, loads), 0.0)


def test_twice_per_carrier_control_loads_the_previous_peak_duty_at_a_valley(gfl, tmp_path):
    """The valley load consumes the prior peak interrupt's shadow before the valley overwrites it."""
    _result, interrupts, loads = _record(gfl(**{"units.vsc.ctrl.period": T / 2}), tmp_path)
    lags = _lags(interrupts, loads)
    assert len(interrupts) == pytest.approx(2 * len(loads), abs=2)
    assert np.allclose(lags, T / 2)


@pytest.mark.parametrize("over, lag", [
    ({"units.vsc.ctrl.computation": 1e-6}, T / 2),
    ({"units.vsc.ctrl.computation": 2.5e-5}, T / 2),
    ({"units.vsc.ctrl.computation": 3e-5}, T),
    # Every load falls within an interrupt's computation and takes the previous completed word.
    ({"units.vsc.ctrl.period": T / 2, "units.vsc.ctrl.offset": T / 2 - 5e-7}, T / 2 + 5e-7),
])
def test_double_update_uses_the_first_load_after_computation(gfl, over, lag, tmp_path):
    _result, interrupts, loads = _record(gfl(**{"units.vsc.pwm.update": "double", **over}), tmp_path)
    assert all(period == pytest.approx(T / 2) for _t, period, _duty in loads)
    lags = _lags(interrupts, loads)
    assert len(lags) > 0.9 * len(interrupts)
    assert np.allclose(lags, lag)


def test_timing_follows_an_asynchronous_carrier_phase(gfl, tmp_path):
    phase = 0.3
    _result, interrupts, loads = _record(gfl(**{
        "units.vsc.pwm.carrier_phase": phase,
        "units.vsc.bridge.model": "switching",
    }), tmp_path)
    assert interrupts[0][0] == pytest.approx((1.0 - phase) * T)
    assert all(carrier_position(t, 1 / T, phase) == 0.0 for t, _period, _duty in loads)
    assert np.allclose(_lags(interrupts, loads), T)


def test_control_adc_and_pwm_use_independent_offsets_on_the_shared_clock_grid(gfl, tmp_path):
    params = gfl(**{
        "units.vsc.ctrl.offset": 10e-6,
        "units.vsc.meas.period": T / 2,
        "units.vsc.meas.offset": 5e-6,
        "simulation.t_end": 3 * T,
        "events.connect_vsc.t": 1.0,  # an initially disconnected unit has no tracking sync
    })
    simulation = peslite.Simulation(params)
    unit = simulation.unit("vsc")
    conversions = []
    convert = unit.adc.convert

    def record_conversion(t):
        conversions.append(t)
        convert(t)

    unit.adc.convert = record_conversion
    controller = []
    unit.ctrl = _Recorder(unit.ctrl, controller, "ctrl")
    result = simulation.run(out_dir=tmp_path)

    # The held conversion before the first one is the start value, not a construction leftover.
    assert result.states["vsc.meas.held_u_dc"][0] == result.states["vsc.dclink.u_C"][0]
    assert unit.bridge.control_clock.offset == pytest.approx(10e-6)
    assert unit.bridge.load_clock.offset == pytest.approx(0.0)
    assert unit.adc.independent
    np.testing.assert_allclose([t for t, _output in controller], [10e-6, 60e-6, 110e-6])
    np.testing.assert_allclose(conversions, np.arange(5e-6, 3 * T, T / 2))


def test_a_slow_loop_runs_at_every_nth_interrupt(gfl):
    sim = peslite.Simulation(gfl(**{
        "units.vsc.ctrl.loops.DCVoltageControl.period": 4 * T,
    }))
    graph = sim.unit().ctrl.graph
    due = [graph.due(k * T) for k in range(8)]
    assert [("DCVoltageControl" in names) for names in due] == [True, False, False, False] * 2
    assert all({"SRFPLL", "CurrentLoop"} <= names for names in due)


def _pwm(computation):
    return PWMAveragingModel(T, T, 0.0, 0.0, computation, T, ZOH(), ThreePhaseBridge())


def test_fresh_timer_processes_the_starting_grid_point_once():
    pwm = _pwm(0.0)

    pwm.start(0.0, (None, None), continued=False)
    assert (pwm.t_interrupt, pwm.t_load) == pytest.approx((0.0, 0.0))

    pwm.start(0.0, (None, None), continued=True)
    assert (pwm.t_interrupt, pwm.t_load) == pytest.approx((T, T))


def test_initial_state_overrides_do_not_turn_a_fresh_run_into_a_continuation(
        gfl, tmp_path):
    _result, interrupts, _loads = _record(gfl(**{
        "simulation.initial.states.vsc.dclink.u_C": 1300.0,
        "simulation.t_end": T,
    }), tmp_path)

    assert interrupts[0][0] == pytest.approx(0.0)


def test_shadow_duty_loads_only_after_computation():
    pwm = _pwm(2e-5)
    pwm.reset(np.full(3, 0.5))
    pwm.write(0.0, np.full(3, 0.1))
    pwm.load(0.0)
    assert pwm.active[0] == 0.5
    pwm.load(1e-5)
    assert pwm.active[0] == 0.5
    pwm.load(2e-5)
    assert pwm.active[0] == 0.1

    pwm.write(T, np.full(3, 0.2))
    assert (pwm.active[0], pwm.shadow[0]) == (0.1, 0.2)

    # Interrupts within the computation before every load: each load takes the replaced word.
    offset = PWMAveragingModel(T, T, T - 1e-5, 0.0, 2e-5, T, ZOH(), ThreePhaseBridge())
    offset.reset(np.full(3, 0.5))
    loaded = []
    for k, duty in enumerate((0.1, 0.2, 0.3)):
        offset.write(offset.interrupt(k), np.full(3, duty))
        offset.load(offset.load_time(k + 1))
        loaded.append(offset.active[0])
    assert loaded == [0.5, 0.1, 0.2]

    other = _pwm(2e-5)
    other.set_state(pwm.get_state())
    other.start(6e-5, (None, None), True)
    other.load(7e-5)
    assert other.active[0] == 0.2
    assert (other.t_interrupt, other.t_load) == (
        other.interrupt(other.k), other.load_time(other.j))


def test_timer_starts_at_the_first_carrier_valley(gfl):
    pwm = peslite.Simulation(gfl(**{
        "units.vsc.pwm.carrier_phase": 0.25,
        "units.vsc.pwm.update": "double",
    })).unit().bridge
    assert (pwm.control_clock.offset, pwm.load_clock.offset, pwm.period, pwm.load_period) == (
        pytest.approx(0.75 * T), pytest.approx(0.75 * T), T, T / 2,
    )
    assert pwm.after(0.0) == 0
    assert pwm.after(pwm.interrupt(3)) == 4
    assert pwm.after(pwm.interrupt(3) + 1e-7) == 4
    for k in (3, 10**7, 2 * 10**7 + 1, 10**9):
        assert pwm.after(pwm.interrupt(k)) == k + 1
        assert pwm.after(pwm.interrupt(k) - 1e-9) == k

    synchronous = peslite.Simulation(gfl(**{
        "units.vsc.pwm.carrier_phase": 0.25,
        "units.vsc.pwm.sync": "synchronous",
    })).unit().bridge
    assert synchronous.control_clock.offset == synchronous.load_clock.offset == 0.0


def test_duty_registers_are_states_and_loop_clocks_are_not(gfl):
    names = set(peslite.Simulation(gfl()).state_names())
    for part in ("", "shadow."):
        assert {f"vsc.pwm.{part}d_{phase}" for phase in "abc"} <= names
    assert not [name for name in names if ".clock." in name or ".pending." in name or "delay" in name]


@pytest.mark.parametrize("over, t0", [
    ({"units.vsc.pwm.update": "double", "units.vsc.ctrl.computation": 3e-5},
     0.003005),  # computation is in progress
    ({"units.vsc.pwm.carrier_phase": 0.3, "units.vsc.bridge.model": "switching"},
     0.003010),  # between timer points
    ({"units.vsc.meas.average": ["u_g", "i_c", "u_dc"],
      "units.vsc.meas.window": 2.5e-5, "units.vsc.bridge.model": "switching"},
     0.003025),  # an ADC window is open
    ({"units.vsc.ctrl.offset": 4.5e-5, "units.vsc.ctrl.computation": 1e-5},
     0.003000),  # a load took the previous word during the computation
])
def test_a_run_continues_exactly_from_any_output_row(gfl, tmp_path, over, t0):
    params = gfl(**{"simulation.t_end": 0.004, "simulation.output.period": 5e-6, **over})
    whole_dir = tmp_path / "whole"
    whole = peslite.Simulation(params).run(out_dir=whole_dir)

    restarted = peslite.load(
        whole_dir / "simulation.pes",
        initial=whole_dir / "states.csv",
        initial_time=t0,
    )
    continued = peslite.Simulation(restarted).run(out_dir=tmp_path / "continued")

    k0 = int(np.argmin(abs(whole.t - t0)))
    assert whole.t[k0] == pytest.approx(t0)
    assert np.array_equal(continued.t, whole.t[k0:])
    for name, values in continued.states.items():
        assert np.array_equal(values, whole.states[name][k0:]), name


def test_initial_and_end_times_are_not_moved_to_a_pwm_grid(gfl, tmp_path):
    t0, t1 = 3.0e-6, 123.0e-6
    result = peslite.Simulation(gfl(**{
        "simulation.initial.t": t0,
        "simulation.t_end": t1,
        "simulation.output.period": 10e-6,
    })).run(out_dir=tmp_path)
    assert result.summary["t_start"] == t0
    assert result.summary["t_stop"] == t1
    assert result.states["t"][-1] == t1


def test_an_oversampled_continuation_starts_at_an_interrupt(gfl):
    over = {"units.vsc.meas.period": 2.5e-5, "units.vsc.pwm.carrier_phase": 0.3}
    state = {"simulation.initial.states": {"vsc.ctrl.SRFPLL.integral_pu": 0.0}}
    gfl(**over, **state, **{"simulation.initial.t": 3.5e-5 + 1e-4})
    with pytest.raises(ConfigError, match="between two control interrupts"):
        gfl(**over, **state, **{"simulation.initial.t": 1.25e-4})
    # The next load would take the word the interrupt replaced, which no load has applied.
    late = {"units.vsc.ctrl.offset": 4.5e-5, "units.vsc.ctrl.computation": 1e-5}
    gfl(**late, **state, **{"simulation.initial.t": 1e-4})
    with pytest.raises(ConfigError, match="outside a controller computation window"):
        gfl(**late, **state, **{"simulation.initial.t": 9.5e-5})


def test_a_window_open_before_a_new_run_uses_the_start_values(gfl, tmp_path):
    over = {"units.vsc.meas.average": ["u_g", "i_c", "u_dc"],
            "simulation.output.signals": 1}
    shifted = peslite.Simulation(gfl(**over, **{"units.vsc.pwm.carrier_phase": 0.3})).run(
        out_dir=tmp_path / "shifted")
    plain = peslite.Simulation(gfl(**over)).run(out_dir=tmp_path / "plain")
    for result in (shifted, plain):
        assert result.ctrl["vsc.vdc_pu"][0] == pytest.approx(1.0, abs=0.01)
        assert result.ctrl["vsc.vac_pu"][0] == pytest.approx(1.0, abs=0.02)
