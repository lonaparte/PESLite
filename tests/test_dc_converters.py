"""Buck, Boost and Buck-Boost DC-converter simulations."""

from dataclasses import fields

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
            "type": "dc_converter",
            "topology": topology,
            "source_voltage": source,
            "inductance": 1e-3,
            "capacitance": 470e-6,
            "load_resistance": load,
            "series_resistance": 0.05,
            "pwm": {"f_sw": 20_000.0},
            "bridge": {"model": model},
            "ctrl": {
                "type": "dc",
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

    assert type(simulation.p.unit()) is peslite.assembly.UnitParams
    assert type(simulation.p.unit().ctrl) is peslite.assembly.ControlParams
    assert simulation.p.unit().type == "dc_converter"
    assert isinstance(unit, peslite.Unit)
    assert isinstance(unit, peslite.DCConverter)
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


@pytest.mark.parametrize("topology,upper_controlled,lower_controlled", (
    ("buck", True, False),
    ("boost", False, True),
    ("buck_boost", False, True),
))
def test_dc_topology_configures_the_half_bridge_devices(
        topology, upper_controlled, lower_controlled):
    unit = peslite.Simulation(_case(topology)).unit()
    bridge = unit.bridge.topology

    assert bridge.upper_controlled is upper_controlled
    assert bridge.lower_controlled is lower_controlled


@pytest.mark.parametrize("topology", CASES)
def test_diode_path_enforces_the_zero_current_boundary(topology):
    simulation = peslite.Simulation(_case(topology))
    plant = simulation.unit().plant
    plant.set_enabled(True)
    plant.state.i_L = -1.0
    plant.state.u_o = CASES[topology][3]
    plant.inp.q = 0.0
    plant.set_outputs(0.0)

    assert plant.out.i_L == 0.0
    assert plant.rhs(0.0)[0] == 0.0

    state = simulation.system.model.get_initial_values()
    current = simulation.system.model.state_slice(plant, "i_L").start
    state[current] = -1.0
    assert simulation.system.model.project_states(state)[current] == 0.0

    plant.state.i_L = 0.0
    plant.inp.q = 1.0
    assert plant.rhs(0.0)[0] > 0.0


@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_light_load_buck_enters_dcm_without_negative_current(model, tmp_path):
    params = _case("buck", model, t_end=0.005).replace(**{
        "units.converter.load_resistance": 600.0,
        "simulation.output.period": 5e-6,
        "simulation.output.states": True,
        "simulation.output.energy": False,
    })
    result = peslite.Simulation(params).run(out_dir=tmp_path / model)
    current = result.states["converter.plant.i_L"]
    tail = current[result.states["t"] >= 0.0025]

    assert np.all(current >= 0.0)
    assert np.any(tail == 0.0)


def test_buck_boost_control_boundary_owns_voltage_magnitude():
    unit = peslite.Simulation(_case("buck_boost")).unit()
    unit.plant.out.i_L = 2.0
    unit.plant.out.u_o = -24.0
    unit.plant.out.v_in = 48.0

    assert unit.control_inputs() == unit.ctrl.control_inputs(2.0, -24.0, 48.0)
    assert unit.control_inputs()["meas.voltage"] == 24.0


@pytest.mark.parametrize("topology,voltage,expected", (
    ("buck", 12.0, 0.25),
    ("boost", 12.0, 0.0),
    ("buck_boost", -12.0, 0.2),
))
def test_scalar_control_boundary_owns_topology_feedforward(topology, voltage, expected):
    unit = peslite.Simulation(_case(topology)).unit()
    unit.plant.out.i_L = 0.0
    unit.plant.out.u_o = voltage
    unit.plant.out.v_in = 48.0

    assert unit.control_inputs()["meas.duty_feedforward"] == pytest.approx(expected)


def test_scalar_log_schema_comes_from_its_public_sources():
    ctrl = peslite.Simulation(_case("buck")).unit().ctrl
    ctrl.set_logging(True)
    ctrl.command(True)

    output = ctrl.sample(0.0, 0.0, 12.0, 48.0)

    assert tuple(output.log) == ctrl.log_names() == tuple(ctrl.log_sources())


def test_unit_and_control_type_boundaries_are_explicit():
    params = _case("buck")
    common = {item.name for item in fields(peslite.assembly.UnitParams)}

    assert {"type", "pwm", "ctrl", "bridge", "s_base"} <= common
    assert {"topology", "source_voltage", "bus", "dclink"}.isdisjoint(common)
    assert not hasattr(params.unit().ctrl.loops["current"], "topology")
    assert not hasattr(params.unit().ctrl.loops["current"], "blocked_bridge")
    for field, value in (("topology", "buck"), ("blocked_bridge", False)):
        with pytest.raises(ConfigError, match="unknown key"):
            params.replace(**{f"units.converter.ctrl.loops.current.{field}": value})
    with pytest.raises(ConfigError, match="unknown key"):
        params.replace(**{"units.converter.bus": "pcc"})
    with pytest.raises(ConfigError, match="cannot use control type"):
        params.replace(**{"units.converter.ctrl.type": "gfl"})
    with pytest.raises(ConfigError, match=r"ctrl\.references: unknown key"):
        params.replace(**{"units.converter.ctrl.references.p_ref_pu": 0.5})
