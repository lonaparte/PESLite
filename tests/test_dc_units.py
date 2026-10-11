"""Unit-level Buck, Boost and Buck-Boost simulations."""

import numpy as np
import pytest

import peslite
from peslite.assembly.params import ConfigError, from_dict
from peslite.control import CapacitorVoltageControl, InductorCurrentControl


CASES = {
    "buck": (48.0, 24.0, 6.0, 24.0),
    "boost": (24.0, 48.0, 12.0, 48.0),
    "buck_boost": (24.0, 24.0, 6.0, -24.0),
}


def _case(topology, model="pwm_averaging", t_end=0.03):
    source, reference, load, _expected = CASES[topology]
    solver = {"dt": 2.5e-6, "linearisations": 0}
    if model == "averaging":
        solver.update(type="adaptive", method="DP45", max_step=1e-4)
    return from_dict({
        "base": {"s_base": 500.0, "v_ll_rms": 230.0, "f0": 50.0},
        "buses": {},
        "units": {"converter": {
            "topology": topology,
            "source_voltage": source,
            "inductance": 1e-3,
            "capacitance": 470e-6,
            "load_resistance": load,
            "series_resistance": 0.05,
            "pwm": {"f_sw": 20_000.0},
            "bridge": {"model": model},
            "ctrl": {
                "loops": {
                    "voltage": {
                        "type": "capacitor_voltage_control",
                        "kp": 0.3,
                        "ki": 80.0,
                        "minimum": 0.0,
                        "maximum": 15.0,
                    },
                    "current": {
                        "type": "inductor_current_control",
                        "topology": topology,
                        "kp": 0.03,
                        "ki": 30.0,
                        "minimum": 0.02,
                        "maximum": 0.95,
                    },
                },
                "references": {"voltage": reference},
            },
        }},
        "simulation": {
            "t_end": t_end,
            "solver": solver,
            "output": {"states": False, "signals": False, "energy": False},
        },
    })


@pytest.mark.parametrize("topology", CASES)
def test_each_scalar_converter_is_a_closed_loop_simulation(topology):
    simulation = peslite.Simulation(_case(topology))
    unit = simulation.unit()

    assert isinstance(unit, peslite.DCUnit)
    assert isinstance(unit.bridge.topology, peslite.HalfBridge)
    assert isinstance(unit.ctrl.graph.nodes["voltage"], CapacitorVoltageControl)
    assert isinstance(unit.ctrl.graph.nodes["current"], InductorCurrentControl)

    result = simulation.run(out_dir=None)
    final = result.final_states()
    expected = CASES[topology][3]
    assert simulation.ph_report.verdict == "port-hamiltonian"
    assert final["converter.plant.u_o"] == pytest.approx(expected, rel=0.02)
    assert np.isfinite(list(final.values())).all()


@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_buck_uses_each_shared_bridge_simulation_form(model):
    result = peslite.Simulation(_case("buck", model, t_end=0.004)).run(out_dir=None)
    assert np.isfinite(result.final_states()["converter.plant.u_o"])


def test_scalar_topology_must_match_the_current_controller():
    params = _case("buck")
    with pytest.raises(ConfigError, match="must match unit topology"):
        params.replace(**{"units.converter.ctrl.loops.current.topology": "boost"})
