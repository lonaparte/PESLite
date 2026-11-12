"""Recorded electrical-port transfers, synchronized frames and matrix validation."""

import json
import math
from types import SimpleNamespace

import numpy as np
import peslite
import pytest

from peslite.addons.functions.port_measurement import (
    measure_multitone, measure_port_matrix, measure_recorded_transfer,
    measure_recorded_sweep, recorded_signal, run_port_matrix, run_port_sweep,
)
from peslite.assembly.params import from_dict


def _fake_result(t, *, scalar=None, vectors=None, ctrl=None):
    plant = dict(scalar or {})
    for name, phasor in (vectors or {}).items():
        plant.update({f"{name}_a": phasor.real,
                      f"{name}_b": (phasor * np.exp(-2j * np.pi / 3)).real,
                      f"{name}_c": (phasor * np.exp(+2j * np.pi / 3)).real})
    return SimpleNamespace(t=t, plant=plant, ctrl=ctrl or {}, states={})


def test_recorded_transfer_uses_native_fit_and_enforces_matching_sample_clocks():
    t = np.arange(0, 1, .001)
    stim = np.cos(2 * math.pi * 20 * t)
    result = _fake_result(t, scalar={"stim": stim, "out": 3 * stim})
    measurement = measure_recorded_transfer(result, "stim", "out", frequency=20,
                                            input_unit="A", output_unit="V")
    assert measurement.valid, measurement.flags
    assert measurement.gain == pytest.approx(3 + 0j, abs=1e-10)
    assert measurement.to_dict()["gain"]["unit"] == "V/A"
    sweep = measure_recorded_sweep({20: result}, "stim", "out")
    assert sweep.valid == (True,)
    assert sweep.gains[0] == pytest.approx(3)
    with pytest.raises(ValueError, match="three-phase"):
        recorded_signal(result, "stim", component="d")


def test_multitone_joint_fit_separates_unrelated_tones_and_rejects_overlap():
    t = np.arange(0, 2, .001)
    stim = np.cos(math.tau * 17 * t) + 2 * np.cos(math.tau * 29 * t)
    output = 2 * np.cos(math.tau * 17 * t) + 6 * np.cos(math.tau * 29 * t)
    result = _fake_result(t, scalar={"stim": stim, "out": output})
    spectrum = measure_multitone(result, "stim", "out", [17, 29])
    assert spectrum.valid == (True, True), spectrum.flags
    np.testing.assert_allclose(spectrum.gains, [2, 3], atol=1e-10)
    with pytest.raises(ValueError, match="overlapping"):
        measure_multitone(result, "stim", "out", [17, 34])


def test_dq_and_sequence_fit_keep_cross_coupling_and_signed_port_current(tmp_path):
    t = np.arange(0, 1, .001)
    f0, frequency = 50, 12
    excitation = np.cos(math.tau * frequency * t)
    z = np.array([[2 + 0.2j, 0.5 - 0.1j], [0.25 + 0.4j, 3 - 0.3j]])
    rotations = np.exp(1j * math.tau * f0 * t)
    runs = []
    for axis in range(2):
        i_d = excitation if axis == 0 else 0 * excitation
        i_q = excitation if axis == 1 else 0 * excitation
        # Complex impedance entries require quadrature-time response terms.
        vd = z[0, axis].real * excitation + z[0, axis].imag * -np.sin(math.tau * frequency * t)
        vq = z[1, axis].real * excitation + z[1, axis].imag * -np.sin(math.tau * frequency * t)
        currents = (i_d + 1j * i_q) * rotations
        voltages = (vd + 1j * vq) * rotations
        runs.append(_fake_result(t, vectors={"stim": currents, "voltage": voltages,
                                             "current": -currents}))
    experiments = {frequency: runs}
    matrix = measure_port_matrix(experiments, voltage="voltage", current="current",
                                 injection="stim", frame_frequency=f0,
                                 frame="dq", current_sign=-1)
    assert matrix.valid == (True,), matrix.flags
    np.testing.assert_allclose(matrix.values[0], z, atol=1e-8)
    response = measure_port_matrix(experiments, voltage="voltage", current="current",
                                   injection="stim", frame_frequency=f0,
                                   frame="sequence", current_sign=-1)
    T = np.array([[1, 1j], [1, -1j]]) / 2
    np.testing.assert_allclose(response.values[0], T @ z @ np.linalg.inv(T), atol=1e-8)
    payload = json.loads(response.write_json(tmp_path / "seq.json").read_text())
    assert payload["impedance"]["unit"] == "ohm"
    assert payload["impedance"]["inputs"] == ["positive", "negative"]
    admittance = measure_port_matrix(experiments, voltage="voltage", current="current",
                                     injection="stim", frame_frequency=f0,
                                     frame="dq", kind="admittance", current_sign=-1)
    assert admittance.valid == (True,), admittance.flags
    np.testing.assert_allclose(admittance.values[0], np.linalg.inv(z), atol=1e-8)


def test_rank_deficient_matrix_is_explicitly_invalid():
    t = np.arange(0, 1, .001)
    stim = np.cos(math.tau * 20 * t)
    result = _fake_result(t, scalar={"stim": stim, "u": stim * 10, "i": stim})
    matrix = measure_port_matrix({20: [result]}, voltage="u", current="i", injection="stim",
                                 frame="scalar")
    assert matrix.valid == (True,)
    dq = _fake_result(t, vectors={"stim": stim.astype(complex),
                                  "u": (10 * stim).astype(complex),
                                  "i": stim.astype(complex)})
    invalid = measure_port_matrix({20: [dq, dq]}, voltage="u", current="i",
                                  injection="stim", frame="dq")
    assert invalid.valid == (False,)
    assert "rank_deficient_excitation" in invalid.flags[0]
    assert invalid.to_dict()["impedance"]["value"][0][0][0] is None


def test_real_circuit_shunt_injection_recovers_resistor_impedance(tmp_path):
    tree = {
        "base": {"s_base": 1000, "v_ll_rms": 230, "f0": 50},
        "buses": {"pcc": {"phases": 1}},
        "elements": {
            "resistor": {"type": "resistor", "bus": "pcc", "r": 10.0},
            "stimulus": {"type": "port_injection", "kind": "current", "bus": "pcc",
                         "amplitude": 1.0, "frequency": 20.0},
        },
        "simulation": {"t_end": 0.5,
                       "solver": {"dt": 0.0001, "linearisations": 0},
                       "output": {"period": 0.0005, "states": False,
                                  "signals": True, "energy": False}},
    }
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    measured = measure_recorded_transfer(result, "stimulus.injection", "stimulus.voltage",
                                         frequency=20, window=(0.1, 0.5),
                                         input_unit="A", output_unit="V")
    assert measured.valid, measured.flags
    assert measured.gain == pytest.approx(10, rel=1e-4)
    matrix = measure_port_matrix({20: [result]}, voltage="stimulus.voltage",
                                 current="stimulus.current", injection="stimulus.injection",
                                 frame="scalar", current_sign=-1, window=(0.1, 0.5))
    assert matrix.valid == (True,), matrix.flags
    assert matrix.values[0, 0, 0] == pytest.approx(10, rel=1e-4)


def test_recorded_nested_unit_controller_signals_keep_their_own_sample_clock():
    t = np.arange(0, 0.1, 0.001)
    name = "charger.primary.VirtualDutyInjection.before_d"
    samples = np.cos(math.tau * 20 * t)
    result = _fake_result(t, ctrl={"charger.primary.t": t, name: samples})
    observed_t, values = recorded_signal(result, name)
    np.testing.assert_array_equal(observed_t, t)
    np.testing.assert_array_equal(values, samples)


def test_sweep_and_impedance_helpers_run_fresh_physical_experiments(tmp_path):
    """The high-level APIs must rerun the native simulator, not just fit mock arrays."""
    params = from_dict({
        "base": {"s_base": 1000.0, "v_ll_rms": 230.0, "f0": 50.0},
        "buses": {"pcc": {"phases": 1}},
        "elements": {
            "load": {"type": "resistor", "bus": "pcc", "r": 10.0},
            "stimulus": {"type": "port_injection", "bus": "pcc",
                         "amplitude": 1.0, "frequency": 40.0},
        },
        "simulation": {
            "t_end": 0.4,
            "solver": {"dt": 0.0002, "linearisations": 0},
            "output": {"period": 0.001, "states": False, "signals": True},
        },
    })
    transfer = run_port_sweep(
        params, injection="stimulus", frequencies=[40.0],
        response="stimulus.voltage", directory=tmp_path / "sweep",
        window=(0.1, 0.4), input_unit="A", output_unit="V",
    )
    assert transfer.valid == (True,), transfer.flags
    assert transfer.gains[0] == pytest.approx(10.0, rel=1e-3)
    assert (tmp_path / "sweep/experiment-000/plant.csv").exists()
    impedance = run_port_matrix(
        params, injection="stimulus", frequencies=[40.0],
        voltage="stimulus.voltage", current="stimulus.current",
        directory=tmp_path / "impedance", frame="scalar",
        current_sign=-1, window=(0.1, 0.4),
    )
    assert impedance.valid == (True,), impedance.flags
    assert impedance.values[0, 0, 0] == pytest.approx(10.0, rel=1e-3)
    assert (tmp_path / "impedance/experiment-000/plant.csv").exists()
