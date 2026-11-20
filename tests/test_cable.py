"""Core cable models, history, events and bridge-mode integration."""

import numpy as np
import peslite
import pytest

from peslite.assembly.params import from_dict, runtime_changeable
from peslite.components.cable import BergeronCableModel, Cable, LumpedCableModel
from peslite.solver import ConfigError
from peslite.solver.history import DelayHistory


def test_cable_model_has_two_inward_ports_and_closes_its_power_balance():
    cable = LumpedCableModel(
        length_km=2.0,
        r_per_km=0.08,
        l_per_km=0.6e-3,
        g_per_km=2.0e-7,
        c_per_km=0.3e-6,
        sections=2,
        voltage0=0j,
    )
    cable.port1.voltage = 8000.0 + 200.0j
    cable.port2.voltage = 7600.0 - 100.0j
    cable.state.iL1 = 20.0 + 3.0j
    cable.state.iL2 = 18.0 + 2.0j
    cable.state.iL3 = 16.0 + 1.0j
    cable.state.uC1 = 7900.0 + 100.0j
    cable.state.uC2 = 7750.0
    cable.set_outputs(0.0)
    derivatives = cable.rhs(0.0)

    stored_rate = 0.0
    for storage, derivative in zip(cable.storage, derivatives):
        value = getattr(cable.state, storage.state)
        stored_rate += storage.scale * storage.value * (value * derivative.conjugate()).real
    power_in = cable.port1.power() + cable.port2.power()

    assert cable.port1.component is cable and cable.port2.component is cable
    assert cable.output.i1 == cable.state.iL1
    assert cable.output.i2 == -cable.state.iL3
    assert sum(cable.series_r) == pytest.approx(0.16)
    assert sum(cable.series_l) == pytest.approx(1.2e-3)
    assert power_in == pytest.approx(stored_rate + cable.dissipated_power())


def test_delay_history_interpolates_committed_values_and_restores_checkpoints():
    history = DelayHistory(1.0, 0.0)
    history.accept(0.0, 0.0)
    history.accept(0.5, 1.0)
    checkpoint = history.checkpoint()
    history.accept(1.0, 1.0)

    assert history(0.25) == pytest.approx(0.5)
    assert history.derivative(0.5) == pytest.approx(2.0)
    assert history.integral_squared(1.0) == pytest.approx(2.0 / 3.0)
    history.accept(1.5, 2.0)
    assert history.derivative(0.5) == pytest.approx(2.0)

    history.restore(checkpoint)
    assert history.times == [0.0, 0.5]
    assert history.values == [0.0, 1.0]


def test_cable_defaults_to_bergeron_and_rejects_unsupported_shunt_loss():
    values = {
        "bus1": "left", "bus2": "right", "length_km": 5.0,
        "r_per_km": 0.05, "l_per_km": 0.4e-3, "c_per_km": 0.2e-6,
    }
    assert Cable.Params(**values).model == "bergeron"
    with pytest.raises(ConfigError, match="requires g_per_km=0"):
        Cable.Params(**values, g_per_km=1.0e-8)


@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_custom_cable_gfl_case_runs_in_every_bridge_mode(examples, tmp_path, model):
    solver = ("adaptive", "DP45") if model == "averaging" else ("fixed", "rk4")
    params = peslite.load(
        examples / "cable-gfl-example.pes",
        **{
            "simulation.t_end": 0.004,
            "simulation.solver.type": solver[0],
            "simulation.solver.method": solver[1],
            "simulation.solver.linearisations": 0,
            "simulation.energy_check": "strict",
            "simulation.output.signals": 1,
            "units.vsc.bridge.model": model,
        },
    )
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path / model)

    assert not runtime_changeable("elements.cable1.sections", params)
    assert simulation.ph_report.verdict == "port-hamiltonian"
    assert not result.tripped
    assert np.isfinite(result.states["cable1.iL1.re"]).all()
    assert np.isfinite(result.states["cable1.uC2.im"]).all()
    assert {"cable1.i1_a", "cable1.i2_a"} <= result.plant.keys()


def test_custom_cable_parameters_can_change_at_an_event(examples, tmp_path):
    params = peslite.load(
        examples / "cable-gfl-example.pes",
        **{
            "simulation.t_end": 0.001,
            "simulation.solver.type": "adaptive",
            "simulation.solver.method": "DP45",
            "simulation.solver.linearisations": 0,
            "simulation.output.signals": 0,
            "units.vsc.bridge.model": "averaging",
        },
    )
    initial = params.replace(**{"elements.cable1.r_per_km": 0.2})
    event = params.replace(**{
        "events.cable_r": {
            "type": "set",
            "t": 0.0,
            "set": {"elements.cable1.r_per_km": 0.2},
        }
    })
    finals = [
        peslite.Simulation(case).run(out_dir=tmp_path / str(index)).final_states()
        for index, case in enumerate((initial, event))
    ]
    assert finals[1] == pytest.approx(finals[0], rel=1e-9)


def test_bergeron_cable_runs_with_adaptive_steps_and_strict_energy(examples, tmp_path):
    params = peslite.load(
        examples / "cable-gfl-example.pes",
        **{
            "elements.cable1.model": "bergeron",
            "elements.cable1.g_per_km": 0.0,
            "elements.cable1.r_per_km": 0.0,
            "units.vsc.bridge.model": "averaging",
            "simulation.t_end": 2.0e-4,
            "simulation.solver.type": "adaptive",
            "simulation.solver.method": "DP45",
            "simulation.solver.max_step": 50.0e-6,
            "simulation.solver.linearisations": 0,
            "simulation.energy_check": "strict",
            "simulation.output.period": 25.0e-6,
            "simulation.output.signals": 1,
        },
    )
    simulation = peslite.Simulation(params)
    cable = simulation.system.named_elements["cable1"].model
    result = simulation.run(out_dir=tmp_path / "bergeron")

    assert isinstance(cable, BergeronCableModel)
    assert cable.segment_delay < params.simulation.solver.max_step
    assert not any(name.startswith("cable1.") for name in result.states)
    assert np.isfinite(result.plant["cable1.i1_a"]).all()
    assert np.isfinite(result.plant["cable1.i2_a"]).all()
    assert simulation.ph_report.verdict == "port-hamiltonian"


@pytest.mark.long
def test_lumped_cable_converges_to_bergeron_terminal_behavior(tmp_path):
    inductance_per_km = 0.4e-3
    capacitance_per_km = 0.2e-6
    surge_impedance = np.sqrt(inductance_per_km / capacitance_per_km)

    def run(model, sections=4):
        line = {
            "type": "cable", "model": model,
            "bus1": "send", "bus2": "receive", "length_km": 5.0,
            "r_per_km": 0.0, "l_per_km": inductance_per_km,
            "c_per_km": capacitance_per_km,
        }
        if model == "lumped":
            line["sections"] = sections
        params = from_dict({
            "base": {"s_base": 1.0e6, "v_ll_rms": 10.0e3, "f0": 50.0},
            "buses": {"send": {"phases": 3}, "receive": {"phases": 3}},
            "sources": {
                "grid": {"bus": "send", "l": 1.0e-6, "r": surge_impedance},
            },
            "elements": {"line": line},
            "simulation": {
                "t_end": 3.0e-4,
                "solver": {
                    "type": "fixed", "method": "rk4", "dt": 1.0e-7,
                    "linearisations": 0,
                },
                "energy_check": "off",
                "progress": {"enable": False},
                "output": {
                    "period": 2.0e-6, "precision": 17,
                    "states": False, "signals": True, "energy": False,
                },
            },
        })
        simulation = peslite.Simulation(params)
        result = simulation.run(out_dir=tmp_path / f"{model}-{sections}")
        trajectory = np.column_stack([
            result.plant[name]
            for name in ("receive.u_a", "line.i1_a")
        ])
        return simulation, np.asarray(result.t), trajectory

    reference_simulation, times, reference = run("bergeron")
    travel_time = reference_simulation.system.named_elements["line"].model.travel_time
    before_wavefront = times < 0.9 * travel_time
    assert np.ptp(reference[before_wavefront, 0]) < 1.0e-10

    comparison = times >= 2.0 * travel_time
    scale = np.maximum(1.0, np.max(np.abs(reference[comparison]), axis=0))
    rms_errors, maximum_errors = [], []
    for sections in (2, 8, 32):
        _simulation, actual_times, actual = run("lumped", sections)
        np.testing.assert_array_equal(actual_times, times)
        normalized = (actual[comparison] - reference[comparison]) / scale
        rms_errors.append(float(np.sqrt(np.mean(normalized ** 2))))
        maximum_errors.append(float(np.max(np.abs(normalized))))

    assert rms_errors[0] > rms_errors[1] > rms_errors[2]
    assert rms_errors[2] < 2.0e-6
    assert maximum_errors[2] < 2.0e-5
