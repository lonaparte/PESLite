"""Typed control-edge injection and feedback return-ratio regression tests."""

import math
from types import SimpleNamespace

import numpy as np
import peslite
import pytest

from peslite.addons.control.virtual_injection import (
    VirtualDqVoltageInjection, VirtualDutyInjection, VirtualVoltageInjection,
)
from peslite.addons.functions.loop_gain import measure_return_ratio, loop_gain_sweep
from peslite.addons.functions.frequency import FrequencyMeasurement
from peslite.assembly.params import from_dict, read_tree
from peslite.control import LOOP_TYPES


def _loop(klass, **options):
    settings = klass.Params(amplitude=0.2, frequency=10.0, **options)
    unit = SimpleNamespace(bridge=SimpleNamespace(model="pwm_averaging"))
    startup = SimpleNamespace(continuous=False)
    return klass(settings, unit, startup)


def test_virtual_scalar_dq_and_continuous_laws_share_deterministic_clock():
    scalar = _loop(VirtualVoltageInjection)
    assert scalar.sample(0.0, {"signal": 2.0})["injected"] == pytest.approx(2.2)
    assert scalar.flow(0.025, {"signal": 2.0}) == ({"injected": 2.0}, {})
    assert scalar.get_state() == {}
    assert scalar.observe()["before_d"] == 2.0
    assert scalar.observe()["after_d"] == pytest.approx(2)
    dq = _loop(VirtualDqVoltageInjection, axis="q")
    assert dq.sample(0.0, {"signal": 2 + 3j})["injected"] == pytest.approx(2 + 3.2j)
    assert dq.observe()["injection_q"] == pytest.approx(0.2)
    assert dq.observe()["before_q"] == pytest.approx(3)
    assert "virtual_dq_voltage_injection" in LOOP_TYPES
    with pytest.raises(peslite.solver.ConfigError, match="only the d axis"):
        _loop(VirtualVoltageInjection, axis="q")


def test_virtual_duty_injection_wires_into_normal_sampled_controller(examples, tmp_path):
    tree = read_tree(examples / "buck-open-loop-example.pes")
    tree["events"] = {}
    ctrl = tree["units"]["converter"]["ctrl"]
    ctrl["loops"]["VirtualDutyInjection"] = {"amplitude": 0.03,
                                              "frequency": 100.0}
    ctrl["connections"] = {
        "VirtualDutyInjection.signal": "references.duty",
        "DutyModulation.duty": "VirtualDutyInjection.injected",
    }
    tree["simulation"].update(t_end=0.015, energy_check="off")
    tree["simulation"]["output"].update(period=1e-5, states=False,
                                        energy=False, signals=True)
    simulation = peslite.Simulation(from_dict(tree))
    assert "VirtualDutyInjection" in simulation.unit().ctrl.graph.nodes
    result = simulation.run(out_dir=tmp_path)
    data = result.ctrl["converter.VirtualDutyInjection.injection_d"]
    assert len(data) > 30
    assert np.max(data) == pytest.approx(0.03, rel=0.02)
    before = result.ctrl["converter.VirtualDutyInjection.before_d"]
    after = result.ctrl["converter.VirtualDutyInjection.after_d"]
    np.testing.assert_allclose(after - before, data, rtol=0.0, atol=1e-5)
    assert not result.tripped


def test_return_ratio_uses_both_sides_of_feedback_cut_not_disturbance():
    t = np.arange(0, 1, 0.001)
    input_signal = np.cos(2 * math.pi * 20 * t)
    # The return is -3 times the actual signal AFTER the injection point.
    before = -3 * input_signal
    output = {"ctrl.t": t, "ctrl.probe.before_d": before,
              "ctrl.probe.after_d": input_signal}
    result = SimpleNamespace(plant={}, ctrl=output, states={}, t=t)
    measurement = measure_return_ratio(result, injection="ctrl.probe", frequency=20)
    assert measurement.valid, measurement.flags
    assert measurement.gain == pytest.approx(3, abs=1e-10)
    assert measurement.conditions["definition"] == "feedback_sign * before / after"
    with pytest.raises(ValueError, match="feedback_sign"):
        measure_return_ratio(result, injection="ctrl.probe", frequency=20,
                             feedback_sign=0)


def test_gain_and_phase_margins_respect_invalid_frequency_gaps(tmp_path):
    frequency = [10, 20, 40, 80]
    magnitudes = [10, 2, 0.5, 0.1]
    phases = [-45, -90, -135, -200]
    points = {}
    for f, mag, phase in zip(frequency, magnitudes, phases):
        value = mag * np.exp(1j * np.deg2rad(phase))
        points[f] = FrequencyMeasurement(f, value, 1 + 0j, value,
                                         1.0, 0.0, 60.0, 60.0,
                                         {"source": "virtual_feedback"}, ())
    margins = loop_gain_sweep(points)
    assert margins.gain_margin_db is not None
    assert margins.gain_margin_db > 0
    assert margins.phase_margin_deg is not None
    assert 40 < margins.phase_margin_deg < 90
    assert 20 < margins.gain_crossover_hz < 40
    assert 40 < margins.phase_crossover_hz < 80
    assert margins.to_dict()["margins"]["gain_margin"]["unit"] == "dB"
    margins.write_json(tmp_path / "margins.json")
    points[40] = FrequencyMeasurement(40, None, 1 + 0j, 0j,
                                      1.0, None, None, None, {}, ("noise",))
    invalid = loop_gain_sweep(points)
    assert invalid.gain_margin_db is None
    assert invalid.phase_margin_deg is None
    assert not invalid.valid[2]
