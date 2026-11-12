"""Deterministic identification waveforms through the native reference-table contract."""

import math

import pytest

from peslite.addons.functions import reference_chirp, reference_multitone, reference_prbs
from peslite.control.signals import TableSignal


def test_prbs_seed_is_repeatable_and_maximal_length():
    a = reference_prbs(start=0.0, stop=127e-4, period=1e-4, order=7, seed=1)
    b = reference_prbs(start=0.0, stop=127e-4, period=1e-4, order=7, seed=1)
    c = reference_prbs(start=0.0, stop=127e-4, period=1e-4, order=7, seed=3)
    assert a == b
    assert a != c
    assert a["type"] == "table"
    assert a["interpolation"] == "zero_order_hold"
    assert len(a["value"]) == 128
    assert set(a["value"]) == {-1.0, 1.0}
    assert a["value"][:-1].count(1.0) == 64
    assert a["value"][:-1].count(-1.0) == 63
    table = TableSignal(**a)
    assert table(0.5e-4) == a["value"][0]
    assert table(1.5e-4) == a["value"][1]


def test_prbs_table_repeat_and_scaling():
    data = reference_prbs(
        start=0, stop=7, period=1, order=3, seed=2,
        repeat=True, low=0, high=1, value_scale=2, value_offset=4,
        time_scale=0.1, time_offset=0.5,
    )
    table = TableSignal(**data)
    assert table(0.5) == table(1.2) == data["value"][0]
    assert data["time"][0] == pytest.approx(0.5)
    assert set(data["value"]) == {4.0, 6.0}


@pytest.mark.parametrize("options", [
    {"order": 1}, {"order": 17}, {"order": 3, "seed": 0},
    {"order": 3, "seed": 8}, {"order": 3, "seed": True},
    {"low": 1, "high": 1}, {"period": 0}, {"stop": 0},
])
def test_prbs_rejects_invalid_parameters(options):
    with pytest.raises(ValueError):
        reference_prbs(**{"start": 0, "stop": 1, "period": 0.01, **options})


def test_multitone_matches_the_requested_phases_and_coherent_period():
    table = reference_multitone(
        start=0, stop=1.0, period=0.001,
        frequencies=[10, 20], amplitudes=[2, 0.5], phases=[0, math.pi / 2],
        offset=3, repeat=True,
    )
    assert table["interpolation"] == "linear"
    assert table["repeat"]
    assert table["value"][0] == pytest.approx(3.5)
    assert table["value"][250] == pytest.approx(3.5, abs=1e-12)
    assert TableSignal(**table)(1.0) == pytest.approx(3.5)


@pytest.mark.parametrize("options", [
    {"frequencies": []}, {"frequencies": [1, 1]},
    {"frequencies": [510]}, {"frequencies": [-1]},
    {"frequencies": [10], "amplitudes": [1, 2]},
    {"frequencies": [10], "repeat": True, "stop": 0.95},
])
def test_multitone_rejects_ambiguous_or_aliased_excitation(options):
    with pytest.raises(ValueError):
        reference_multitone(**{"start": 0, "stop": 1, "period": 0.001, **options})


def test_linear_chirp_uses_the_integrated_instantaneous_frequency():
    table = reference_chirp(
        start=1, stop=2, period=0.001, start_frequency=10, stop_frequency=30,
        amplitude=2, phase=0, offset=1,
    )
    assert table["value"][0] == pytest.approx(1)
    t = 1.25
    expected = 1 + 2 * math.sin(math.tau * (10 * 0.25 + 0.5 * 20 * 0.25**2))
    assert TableSignal(**table)(t) == pytest.approx(expected, abs=1e-10)
    with pytest.raises(ValueError, match="Nyquist"):
        reference_chirp(start=0, stop=1, period=0.01,
                        start_frequency=10, stop_frequency=70)
