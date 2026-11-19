"""Feedback return-ratio and crossover-margin regression tests."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from peslite.addons.functions import loop_gain_sweep, measure_return_ratio
from peslite.addons.functions.frequency import FrequencyMeasurement


def test_return_ratio_uses_both_sides_of_feedback_cut_not_disturbance():
    t = np.arange(0, 1, 0.001)
    after = np.cos(2 * math.pi * 20 * t)
    before = -3 * after
    output = {"ctrl.t": t, "ctrl.cut.before": before, "ctrl.cut.after": after}
    result = SimpleNamespace(plant={}, ctrl=output, states={}, t=t)
    measurement = measure_return_ratio(
        result, before="ctrl.cut.before", after="ctrl.cut.after", frequency=20,
    )

    assert measurement.valid, measurement.flags
    assert measurement.gain == pytest.approx(3, abs=1e-10)
    assert measurement.conditions["definition"] == "feedback_sign * before / after"
    with pytest.raises(ValueError, match="feedback_sign"):
        measure_return_ratio(
            result, before="ctrl.cut.before", after="ctrl.cut.after",
            frequency=20, feedback_sign=0,
        )


def test_gain_and_phase_margins_respect_invalid_frequency_gaps(tmp_path):
    frequency = [10, 20, 40, 80]
    magnitudes = [10, 2, 0.5, 0.1]
    phases = [-45, -90, -135, -200]
    points = {}
    for f, magnitude, phase in zip(frequency, magnitudes, phases):
        value = magnitude * np.exp(1j * np.deg2rad(phase))
        points[f] = FrequencyMeasurement(
            f, value, 1 + 0j, value, 1.0, 0.0, 60.0, 60.0,
            {"source": "feedback_cut"}, (),
        )
    margins = loop_gain_sweep(points)

    assert margins.gain_margin_db is not None
    assert margins.gain_margin_db > 0
    assert margins.phase_margin_deg is not None
    assert 40 < margins.phase_margin_deg < 90
    assert 20 < margins.gain_crossover_hz < 40
    assert 40 < margins.phase_crossover_hz < 80
    assert margins.to_dict()["margins"]["gain_margin"]["unit"] == "dB"
    margins.write_json(tmp_path / "margins.json")

    points[40] = FrequencyMeasurement(
        40, None, 1 + 0j, 0j, 1.0, None, None, None, {}, ("noise",),
    )
    invalid = loop_gain_sweep(points)
    assert invalid.gain_margin_db is None
    assert invalid.phase_margin_deg is None
    assert not invalid.valid[2]


def test_phase_crossings_are_independent_of_the_unwrapped_phase_branch():
    frequencies = [10, 20, 40, 80]
    phases = [150, 170, -170, -150]
    points = {
        frequency: FrequencyMeasurement(
            frequency, 0.5 * np.exp(1j * np.deg2rad(phase)), 1 + 0j,
            0.5 * np.exp(1j * np.deg2rad(phase)), 1.0, 0.0, 60.0, 60.0, {}, (),
        )
        for frequency, phase in zip(frequencies, phases)
    }
    result = loop_gain_sweep(points)

    assert result.gain_margin_db == pytest.approx(20 * math.log10(2), rel=1e-12)
    assert 20 < result.phase_crossover_hz < 40


def test_zero_gain_rejects_only_its_own_sample():
    points = {
        frequency: FrequencyMeasurement(
            frequency, complex(gain), 1 + 0j, complex(gain), 1.0,
            0.0, 60.0, 60.0, {}, (),
        )
        for frequency, gain in zip((10, 20, 40, 80), (2.0, 0.0, 0.5, 0.25))
    }
    result = loop_gain_sweep(points)

    assert result.valid == (True, False, True, True)
