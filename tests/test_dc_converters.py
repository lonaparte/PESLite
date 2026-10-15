"""Open DCConverter assembly, scalar network wiring and control behavior."""

from dataclasses import fields

import numpy as np
import pytest

import peslite
from peslite.assembly.params import ConfigError, from_dict
from peslite.components import HalfBridgeDiodeConduction, RLBranch
from peslite.control import DCVoltageControl, InductorCurrentControl


def _case(*, model="pwm_averaging", bridge_type="half_bridge",
          lower_controlled=False, load=6.0, t_end=0.01):
    return from_dict({
        "base": {"s_base": 500.0, "v_ll_rms": 230.0, "f0": 50.0},
        "buses": {"output": {"phases": 1, "c": 470.0e-6}},
        "elements": {
            "load": {"type": "resistor", "bus": "output", "r": load},
        },
        "units": {
            "converter": {
                "type": "dc_converter",
                "bus": "output",
                "filter": {"l_f": 1.0e-3, "r_f": 0.05},
                "dclink": {
                    "vdc_ref": 48.0,
                    "source": {"type": "voltage"},
                },
                "bridge": {
                    "type": bridge_type,
                    "model": model,
                    "lower_controlled": lower_controlled,
                },
                "pwm": {"f_sw": 20_000.0},
                "ctrl": {
                    "type": "dc",
                    "loops": {
                        "voltage": {
                            "type": "dc_voltage_control",
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
                    "references": {"voltage": 24.0},
                },
            },
        },
        "simulation": {
            "t_end": t_end,
            "solver": {"dt": 2.5e-6, "linearisations": 0},
            "output": {"period": 100.0e-6, "states": 1, "signals": 1, "energy": 0},
        },
    })


def test_dc_converter_is_an_open_unit_connected_through_the_common_terminal_api():
    simulation = peslite.Simulation(_case())
    unit = simulation.unit()
    names = set(simulation.system.model.names)

    assert isinstance(unit.bridge.topology, peslite.HalfBridge)
    assert unit.bus is simulation.system.buses["output"]
    assert unit.terminals() == (("output", unit.branch_f.terminal2),)
    assert {"output", "load", "converter.branch_f", "converter.bridge",
            "converter.dclink"} <= names
    assert not hasattr(unit, "plant")
    assert isinstance(unit.ctrl.graph.nodes["voltage"], DCVoltageControl)
    assert isinstance(unit.ctrl.graph.nodes["current"], InductorCurrentControl)


def test_dc_converter_can_use_the_same_three_phase_point_network_interface(tmp_path):
    params = _case(t_end=0.0002).replace(**{"buses.output.phases": 3})
    simulation = peslite.Simulation(params)
    unit = simulation.unit()

    simulation.run(out_dir=tmp_path)

    assert isinstance(unit.branch_f.state.i, complex)
    assert unit.bridge.topology.power_scale == 1.5
    assert unit.terminals() == (("output", unit.branch_f.terminal2),)


@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_dc_converter_uses_each_shared_bridge_form(model, tmp_path):
    simulation = peslite.Simulation(_case(model=model, t_end=0.002))
    result = simulation.run(out_dir=tmp_path / model)

    assert result.summary["t_stop"] == pytest.approx(0.002)
    assert np.isfinite(simulation.system.model.get_initial_values()).all()
    assert 0.0 <= float(simulation.unit().bridge.inp.q) <= 1.0


@pytest.mark.parametrize("bridge_type,topology", [
    ("half_bridge", peslite.HalfBridge),
    ("full_bridge", peslite.FullBridge),
])
def test_dc_converter_selects_its_bridge_structure(bridge_type, topology):
    params = _case(bridge_type=bridge_type, lower_controlled=True)
    unit = peslite.Simulation(params).unit()

    assert isinstance(unit.bridge.topology, topology)


def test_half_bridge_device_placement_controls_unilateral_current():
    diode_unit = peslite.Simulation(_case(lower_controlled=False)).unit()
    controlled_unit = peslite.Simulation(_case(lower_controlled=True)).unit()

    assert diode_unit.branch_f.unilateral
    assert not controlled_unit.branch_f.unilateral
    assert not diode_unit.bridge.topology.lower_controlled
    assert controlled_unit.bridge.topology.lower_controlled


def test_light_load_enters_dcm_without_a_mode_setting(tmp_path):
    params = _case(model="switching", load=120.0, t_end=0.004).replace(
        **{"simulation.output.period": 2.5e-6},
    )
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)
    current = result.states["converter.branch_f.i"]

    assert current.min() >= -1e-12
    assert np.count_nonzero(np.isclose(current, 0.0, atol=1e-10)) > 2


@pytest.mark.parametrize("load,expected_voltage,expected_total", [
    (6.0, 14.4, 1.0),
    (600.0, 32.0993023218081, None),
])
def test_averaged_half_bridge_has_the_ccm_or_dcm_operating_point_naturally(
        load, expected_voltage, expected_total):
    branch = RLBranch(1.0e-3, 0.0, phases=1, unilateral=True)
    branch.conduction = HalfBridgeDiodeConduction(50.0e-6)
    branch.inp.q = 0.3
    branch.inp.u_dc = 48.0
    branch.inp.u2 = expected_voltage
    branch.state.i = expected_voltage / load

    _controlled, _diode, total, _rms = branch.conduction.fractions(branch)

    assert branch.conduction.derivative(branch) == pytest.approx(0.0, abs=2e-10)
    if expected_total is None:
        assert total < 1.0
    else:
        assert total == expected_total


def test_control_feedforward_depends_on_bridge_structure_not_an_application_name():
    half = peslite.Simulation(_case()).unit().ctrl
    full = peslite.Simulation(_case(bridge_type="full_bridge", lower_controlled=True)).unit().ctrl

    assert half.control_inputs(1.0, 12.0, 48.0)["meas.duty_feedforward"] == pytest.approx(0.25)
    assert full.control_inputs(1.0, 12.0, 48.0)["meas.duty_feedforward"] == pytest.approx(0.625)


def test_unit_and_control_scopes_have_no_closed_circuit_fields():
    params = _case()
    unit = params.unit()
    common = {item.name for item in fields(peslite.assembly.UnitParams)}

    assert {"type", "pwm", "ctrl", "bridge", "s_base"} <= common
    assert {"bus", "filter", "dclink"}.isdisjoint(common)
    assert unit.bus == "output"
    for old in ("topology", "source_voltage", "inductance", "capacitance",
                "load_resistance", "series_resistance"):
        assert not hasattr(unit, old)
        with pytest.raises(ConfigError, match="unknown key"):
            params.replace(**{f"units.converter.{old}": 1.0})


def test_dc_converter_rejects_wrong_network_and_bridge_scopes():
    params = _case()
    with pytest.raises(ConfigError, match="requires 'half_bridge' or 'full_bridge'"):
        params.replace(**{"units.converter.bridge.type": "three_phase_bridge"})
    with pytest.raises(ConfigError, match="specified together|unknown bus None"):
        params.replace(**{"units.converter.bus": None})
