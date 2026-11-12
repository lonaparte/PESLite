"""Transient response metrics and explicit quality flags."""

import json

import numpy as np
import pytest

from peslite.addons.functions import dynamic_performance


def test_first_order_step_rise_settling_tracking_and_units():
    t = np.linspace(0, 1, 1001)
    y = 1 - np.exp(-t / 0.05)
    result = dynamic_performance(t, y, reference=1, initial=0, unit="V")
    assert result.valid
    assert result.rise_time == pytest.approx(0.05 * np.log(9), abs=0.001)
    assert result.settling_time == pytest.approx(-0.05 * np.log(0.02), abs=0.001)
    assert result.overshoot_percent == pytest.approx(0)
    assert abs(result.steady_state_error) < 1e-7
    assert result.tracking_rmse == pytest.approx(np.sqrt(np.sum(0.5 * np.diff(t) * (((y - 1)**2)[1:] + ((y - 1)**2)[:-1]))))
    assert result.to_dict()["metrics"]["rise_time"]["unit"] == "s"
    assert result.to_dict()["metrics"]["ripple_peak_to_peak"]["unit"] == "V"
    json.dumps(result.to_dict(), allow_nan=False)


def test_negative_step_overshoot_and_tail_ripple():
    t = np.arange(9, dtype=float)
    y = [5, 4, 2.5, 1.7, 2.0, 2.0, 2.1, 1.9, 2.0]
    result = dynamic_performance(t, y, reference=2, initial=5,
                                 tail_fraction=0.5, settling_tolerance=0.05)
    assert result.overshoot_percent == pytest.approx(10)
    assert result.rise_time is not None
    assert result.settling_time == 4.0
    assert result.ripple_peak_to_peak == pytest.approx(0.2)


def test_missing_step_or_incomplete_response_is_flagged_not_fabricated():
    t = np.arange(10, dtype=float)
    flat = dynamic_performance(t, np.ones(10), reference=1)
    assert not flat.valid
    assert "zero_step_amplitude" in flat.flags
    assert flat.rise_time is None
    incomplete = dynamic_performance(t, 1 - np.exp(-t / 100), reference=1, initial=0)
    assert "not_settled" in incomplete.flags
    assert incomplete.settling_time is None


def test_time_varying_reference_keeps_tracking_metrics_but_rejects_step_metrics():
    t = np.linspace(0, 1, 101)
    ref = 2 + t
    measured = ref - 0.1
    result = dynamic_performance(t, measured, reference=ref, initial=1)
    assert "reference_changes_after_step" in result.flags
    assert result.rise_time is None
    assert result.tracking_rmse == pytest.approx(0.1)
    assert result.tracking_max_abs == pytest.approx(0.1)


@pytest.mark.parametrize("t,y", [
    ([0, 0, 1], [1, 2, 3]),
    ([0, 1, 2], [1, float("nan"), 3]),
    ([0, 1], [1, 2]),
    ([0, 1, 2], [1, 2]),
])
def test_invalid_measurements_are_rejected(t, y):
    with pytest.raises(ValueError):
        dynamic_performance(t, y, reference=1)
