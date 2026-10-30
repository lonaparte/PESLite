"""Complete-pulse updates share one sequence policy across modulation waveforms."""

import math

import numpy as np
import pytest
import yaml

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import from_dict, read_initial
from peslite.components.pwm import CarrierComparison, PulseSequence, SynchronousCarrier
from test_cpp_parameters import _case, _compare_run, _compile


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("synchronous", [False, True])
def test_carrier_updates_preserve_each_complete_gate_pulse(width, synchronous):
    frequency = 100e3
    period = 1 / frequency
    modulator = SynchronousCarrier(1) if synchronous else CarrierComparison(frequency)
    update = PulseSequence(modulator, width)
    starts, words = [], []
    choices = np.array([.2, .5, .8])
    for k in range(24):
        at, span = k * period / 2, period / 2
        duties = np.roll(choices, k // 3)[:width]
        sequence = update(at, span, duties, 2 * math.pi * frequency * at,
                          2 * math.pi * frequency)
        cursor = at
        for dt, word in zip(sequence.dt, sequence.q_abc):
            starts.append(cursor)
            words.append(word)
            cursor += dt
        assert cursor == pytest.approx(at + span, abs=1e-18)
    words, starts = np.asarray(words), np.asarray(starts)
    for column in range(width):
        edges = np.r_[True, np.diff(words[:, column]) != 0]
        times, levels = starts[edges], words[edges, column]
        durations = np.diff(times)[levels[:-1] != 0]
        assert len(durations) > 3
        assert np.all(np.min(np.abs(durations[:, None] / period - choices), axis=1) < 1e-9)


def test_constant_gate_commands_do_not_wait_for_nonexistent_edges():
    update = PulseSequence(CarrierComparison(100e3), 1)
    for at, duty in [(0.0, 0.0), (1e-6, 1.0), (2e-6, 0.0)]:
        sequence = update(at, 1e-6, np.array([duty]))
        assert sequence.q_abc.tolist() == [[duty]]
    sequence = update(3e-6, 1e-6, np.array([1.0]), enabled=False)
    assert sequence.q_abc.tolist() == [[0.0]]
    assert update.next == (0.0,)


def test_bridge_phase_clock_describes_the_applied_reference():
    from peslite.components.converter import SwitchingModel, ThreePhaseBridge

    omega = 2 * math.pi * 100e3
    pwm = SwitchingModel(1e-5, 1e-5, 0.0, 0.0, 0.0, 1e-5, SynchronousCarrier(1), ThreePhaseBridge())
    pwm.reset(np.full(3, .5), on=True)
    pwm.start(0.0, (0.0, omega), False)
    pwm.tick(1e-5, .4, omega)
    assert pwm.phase_clock().at(1e-5) == pytest.approx(2 * math.pi)
    pwm.load(1e-5)
    assert pwm.phase_clock().at(1e-5) == pytest.approx(.4)


@pytest.mark.parametrize("registers,modulator,match", [
    (("d", "d"), CarrierComparison(100e3), "unique"),
    (("on",), CarrierComparison(100e3), "unique"),
    (("shadow.d",), CarrierComparison(100e3), "path segments"),
    (("angle",), SynchronousCarrier(1), "conflict with modulator states"),
])
def test_pwm_register_layout_rejects_state_collisions(registers, modulator, match):
    from peslite.components.converter import HalfBridge
    from peslite.components.pwm import PWM

    topology = HalfBridge()
    with pytest.raises(ValueError, match=match):
        PWM(1e-5, 1e-5, 0.0, 0.0, 0.0, 1e-5, modulator, duty_names=registers,
            switching_state=topology.switching_state, pwm_state=topology.pwm_state)


@pytest.mark.parametrize("example", ["gfl", "buck", "wpt"])
@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_explicit_default_does_not_add_policy_or_state(example, mode):
    tree = yaml.safe_load((EXAMPLES / f"{example}-example.pes").read_text())
    from peslite.assembly.params import _unit_items
    initial = from_dict(tree)
    changes = {f"units.{name}.bridge.model": mode for name, _ in _unit_items(initial.units)}
    omitted = initial.replace(**changes)
    explicit = omitted.replace(**{f"units.{name}.pwm.pattern_sync": "interval"
                                   for name, _ in _unit_items(omitted.units)})
    assert omitted == explicit
    left, right = peslite.Simulation(omitted), peslite.Simulation(explicit)
    assert left._states.columns == right._states.columns
    assert left.system.get_state() == right.system.get_state()
    for simulation in (left, right):
        for unit in simulation.system.units.values():
            if not unit.continuous:
                assert unit.bridge.pulse_update is None
                assert unit.bridge._sequence is unit.bridge.modulator


@pytest.mark.parametrize("example", ["gfl", "buck", "wpt"])
def test_default_configuration_generates_identical_cpp(example, tmp_path):
    params = _case(example, "switching")
    from peslite.assembly.params import _unit_items
    explicit = params.replace(**{f"units.{name}.pwm.pattern_sync": "interval"
                                 for name, _ in _unit_items(params.units)})
    for name, config in (("omitted", params), ("explicit", explicit)):
        peslite.Simulation(config).export("cpp", tmp_path / name, name="equivalent", variables=("common",))
    assert (tmp_path / "omitted/peslite.cpp").read_bytes() == (tmp_path / "explicit/peslite.cpp").read_bytes()


@pytest.mark.parametrize("update", ["interval", "pulse"])
def test_synchronous_continuation_restores_the_applied_waveform(update, tmp_path):
    params = _case("gfl", "switching").replace(**{
        "units.vsc.pwm.sync": "synchronous", "units.vsc.pwm.pattern_sync": update,
        "simulation.t_end": .001, "simulation.output.period": 1e-6,
    })
    whole = peslite.Simulation(params)
    whole.run(out_dir=tmp_path / "whole")
    peslite.Simulation(params.replace(**{"simulation.t_end": .000437})).run(out_dir=tmp_path / "first")
    initial = read_initial(tmp_path / "first/states.csv")
    peslite.Simulation(params.replace(**{"simulation.initial": initial})).run(out_dir=tmp_path / "second")
    from test_cpp_parameters import _table

    names, reference = _table(tmp_path / "whole/states.csv")
    actual_names, actual = _table(tmp_path / "second/states.csv")
    assert names == actual_names
    rows = {round(row[0], 12): row for row in reference}
    expected = np.asarray([rows[round(row[0], 12)] for row in actual])
    error = actual - expected
    for index, name in enumerate(names):
        if name in whole._states.angle_names:
            error[:, index] = (error[:, index] + math.pi) % (2 * math.pi) - math.pi
    scale = np.maximum(1.0, np.max(np.abs(expected), axis=0))
    assert np.max(np.abs(error) / scale) < 1e-9


@pytest.mark.parametrize("example, synchronous, update, control_offset", [
    ("gfl", False, "pulse", None), ("gfl", True, "pulse", None),
    ("buck", False, "pulse", None), ("wpt", False, "pulse", None),
    ("gfl", True, "interval", None), ("gfl", True, "interval", 10e-6),
])
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("variables", [("common",), ("all",)])
def test_native_carrier_pulse_updates_match_python(example, synchronous, update, control_offset, restart,
                                                   variables, tmp_path):
    params = _case(example, "switching")
    from peslite.assembly.params import _unit_items
    changes = {f"units.{name}.pwm.pattern_sync": update for name, _ in _unit_items(params.units)}
    changes["simulation.t_end"] = 80e-6 if example == "wpt" else .001
    if synchronous:
        changes["units.vsc.pwm.sync"] = "synchronous"
    if control_offset is not None:  # the control clock differs from the PWM load clock
        changes["units.vsc.ctrl.offset"] = control_offset
    params = params.replace(**changes)
    if restart:
        first = params.replace(**{"simulation.t_end": params.simulation.t_end * .43})
        peslite.Simulation(first).run(out_dir=tmp_path / "first")
        params = params.replace(**{"simulation.initial": read_initial(tmp_path / "first/states.csv")})
    binary = _compile(params, tmp_path / "model", variables)
    _compare_run(params, binary, {}, tmp_path / "comparison")
