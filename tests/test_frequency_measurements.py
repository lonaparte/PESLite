"""Frequency-response validity and complex impedance serialization."""

import json
import math

import numpy as np
import pytest

from peslite.addons.functions import frequency_response, frequency_sweep, impedance_matrix


def _sinusoid(f=40.0, duration=1.0, step=0.0005, gain=0.2 + 0.3j):
    t = np.arange(0, duration, step)
    x = 3 * np.sin(math.tau * f * t + 0.3) + 2
    y = 3 * abs(gain) * np.sin(math.tau * f * t + 0.3 + np.angle(gain)) + 5
    return t, x, y


def test_complex_gain_is_phase_accurate_and_serializable(tmp_path):
    t, x, y = _sinusoid()
    result = frequency_response(t, x, y, frequency=40, input_unit="A", output_unit="V")
    assert result.valid, result.flags
    assert result.gain == pytest.approx(0.2 + 0.3j, abs=1e-10)
    assert result.excitation_amplitude == pytest.approx(3)
    assert result.distortion < 1e-10
    assert result.conditions["frequency_resolution_hz"] == pytest.approx(1 / (t[-1] - t[0]))
    payload = result.to_dict()
    assert payload["gain"]["unit"] == "V/A"
    assert payload["gain"]["value"]["imag"] == pytest.approx(0.3)
    assert json.loads(result.write_json(tmp_path / "measurement.json").read_text()) == payload


def test_frequency_measurement_flags_nonconvergence_distortion_low_amplitude_and_aliasing():
    t, x, y = _sinusoid()
    transient = y * (1 - np.exp(-t / 0.3))
    result = frequency_response(t, x, transient, frequency=40)
    assert not result.valid
    assert "steady_state_not_converged" in result.flags
    assert result.gain is None

    distorted = y + 0.3 * np.sin(2 * math.tau * 40 * t)
    result = frequency_response(t, x, distorted, frequency=40, max_distortion=0.05)
    assert "nonlinear_distortion" in result.flags
    assert result.gain is None

    weak = frequency_response(t, x * 1e-10, y, frequency=40)
    assert "insufficient_excitation_amplitude" in weak.flags
    assert weak.gain is None

    sparse = frequency_response(t[::5], x[::5], y[::5], frequency=250)
    assert "sampling_aliasing" in sparse.flags
    assert sparse.gain is None


def test_frequency_validation_flags_short_windows_noise_and_nonuniform_sampling():
    t, x, y = _sinusoid(duration=0.025)
    short = frequency_response(t, x, y, frequency=40)
    assert "insufficient_frequency_resolution" in short.flags
    assert short.conditions["cycles"] < 6

    t, x, y = _sinusoid()
    rng = np.random.default_rng(4)
    noisy = frequency_response(t, x, y + rng.normal(0, 3, len(y)), frequency=40)
    assert "low_signal_to_noise_ratio" in noisy.flags
    uneven = t.copy()
    uneven[1::2] += 0.0001
    result = frequency_response(uneven, x, y, frequency=40)
    assert "nonuniform_sampling" in result.flags


def test_sweep_preserves_complex_units_and_failed_measurements(tmp_path):
    records = []
    for frequency in (20, 40, 60):
        t, x, y = _sinusoid(f=frequency)
        if frequency == 40:
            x *= 0
        records.append(frequency_response(t, x, y, frequency=frequency,
                                          input_unit="A", output_unit="V"))
    sweep = frequency_sweep(records)
    assert sweep.valid == (True, False, True)
    assert sweep.to_dict()["transfer"]["value"][1] is None
    assert sweep.to_dict()["transfer"]["value"][0]["imag"] == pytest.approx(0.3)
    json.loads(sweep.write_json(tmp_path / "sweep.json").read_text())
    with pytest.raises(ValueError, match="increasing"):
        frequency_sweep(records[::-1])


def test_two_port_impedance_recovers_complex_couplings(tmp_path):
    matrices = np.array([
        [[2 + 1j, 0.2j], [-0.4j, 5 - 2j]],
        [[3 + 2j, 0.5j], [-0.3j, 4 - 1j]],
    ])
    current = np.array([[1, 0], [0, 1], [1, 1]], dtype=complex)
    injections = np.repeat(current[None], 2, axis=0)
    observed = np.stack([current @ z.T for z in matrices])
    result = impedance_matrix([50, 100], injections, observed,
                              input_names=["grid", "load"],
                              output_names=["pcc", "dc"])
    assert result.valid == (True, True)
    np.testing.assert_allclose(result.values, matrices, rtol=0, atol=1e-12)
    payload = result.to_dict()
    assert payload["impedance"]["unit"] == "ohm"
    assert payload["impedance"]["value"][0][0][0]["real"] == pytest.approx(2)
    assert json.loads(result.write_json(tmp_path / "impedance.json").read_text()) == payload


def test_impedance_rejects_dependent_or_untrusted_injection_sets():
    current = np.array([[1, 1], [2, 2], [3, 3]], dtype=complex)
    observed = np.ones((3, 2), dtype=complex)
    result = impedance_matrix([50], current, observed)
    assert not result.valid[0]
    assert "rank_deficient_excitation" in result.flags[0]
    assert result.to_dict()["impedance"]["value"][0][0][0] is None
    marked = impedance_matrix([50], np.eye(2), np.eye(2),
                              measurement_valid=[[True, False]])
    assert not marked.valid[0]
    assert "unreliable_input_measurements" in marked.flags[0]


@pytest.mark.parametrize("f, current, voltage", [
    ([50, 50], np.ones((2, 2, 1)), np.ones((2, 2, 1))),
    ([50], [[1, float("nan")]], [[1, 1]]),
    ([50], np.ones((2, 2, 1)), np.ones((3, 2, 1))),
])
def test_impedance_rejects_bad_array_contracts(f, current, voltage):
    with pytest.raises(ValueError):
        impedance_matrix(f, current, voltage)
