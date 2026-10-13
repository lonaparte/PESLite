"""Buck, Boost and Buck-Boost DC-converter simulations."""

import copy
import math
from dataclasses import fields

import numpy as np
import pytest

import peslite
from peslite.assembly.conduction import DiodeConduction
from peslite.assembly.params import ConfigError, from_dict, to_dict
from peslite.control import CapacitorVoltageControl, InductorCurrentControl
from peslite.solver.integrators import DormandPrince45, FixedStepSolver
from peslite.solver import integrators
from peslite.solver.model import Model


CASES = {
    "buck": (48.0, 24.0, 6.0, 24.0),
    "boost": (24.0, 48.0, 12.0, 48.0),
    "buck_boost": (24.0, 24.0, 6.0, -24.0),
}


def test_diode_conduction_is_not_a_generic_integrator_feature():
    assert "DiodeConduction" not in integrators.__all__


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


@pytest.mark.parametrize("topology", CASES)
def test_averaged_dcm_uses_cycle_conduction_fractions(topology):
    source, inductance, load, duty = 24.0, 200e-6, 400.0, 0.4
    period = 1.0 / 20_000.0
    conduction = 2.0 * inductance / (load * period)
    if topology == "buck":
        ratio = (-duty**2 + math.sqrt(duty**4 + 4.0 * conduction * duty**2)) / (
            2.0 * conduction
        )
        diode = duty * (1.0 - ratio) / ratio
        voltage = ratio * source
        current = voltage / load
    elif topology == "boost":
        ratio = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * duty**2 / conduction))
        diode = duty / (ratio - 1.0)
        voltage = ratio * source
        current = voltage * (duty + diode) / (load * diode)
    else:
        ratio = duty / math.sqrt(conduction)
        diode = duty / ratio
        voltage = -ratio * source
        current = -voltage * (duty + diode) / (load * diode)

    params = _case(topology, "averaging").replace(**{
        "units.converter.source_voltage": source,
        "units.converter.inductance": inductance,
        "units.converter.capacitance": 100e-6,
        "units.converter.load_resistance": load,
        "units.converter.series_resistance": 0.0,
    })
    plant = peslite.Simulation(params).unit().plant
    plant.set_enabled(True)
    plant.inp.q = duty
    plant.state.i_L = current
    plant.state.u_o = voltage

    assert diode < 1.0 - duty
    assert plant.rhs(0.0) == pytest.approx((0.0, 0.0), abs=1e-8)


def test_buck_averaged_dcm_reaches_the_analytic_operating_point():
    params = _case("buck", "averaging").replace(**{
        "units.converter.source_voltage": 24.0,
        "units.converter.inductance": 200e-6,
        "units.converter.capacitance": 100e-6,
        "units.converter.load_resistance": 400.0,
        "units.converter.series_resistance": 0.0,
    })
    plant = peslite.Simulation(params).unit().plant
    plant.set_enabled(True)
    plant.inp.q = 0.4
    model = Model({"plant": plant}, {})
    state = DormandPrince45(1e-8, 1e-10, 2e-6)(
        model.rhs, 0.0, 0.06, model.get_initial_values()
    ).y

    assert state[0] == pytest.approx(0.05393877, rel=2e-5)
    assert state[1] == pytest.approx(21.57550765, rel=2e-5)


def test_buck_averaged_dcm_limits_an_explicit_step_near_its_fast_boundary():
    params = _case("buck", "pwm_averaging").replace(**{
        "units.converter.source_voltage": 24.0,
        "units.converter.inductance": 200e-6,
        "units.converter.capacitance": 470e-6,
        "units.converter.load_resistance": 400.0,
        "units.converter.series_resistance": 0.0,
    })
    plant = peslite.Simulation(params).unit().plant
    plant.set_enabled(True)
    plant.inp.q = 0.6
    conduction = 2.0 * plant.L / (plant.R_load * plant.period)
    ratio = (-0.6**2 + math.sqrt(0.6**4 + 4.0 * conduction * 0.6**2)) / (
        2.0 * conduction
    )
    expected_voltage = ratio * plant.v_in
    expected_current = expected_voltage / plant.R_load
    plant.state.u_o = expected_voltage
    plant.state.i_L = expected_current
    model = Model({"plant": plant}, {})
    conduction = DiodeConduction(model, FixedStepSolver(5e-6, "rk4"))
    assert conduction.resolution_subsystems == (plant,)
    state = conduction.advance(0.0, 0.03, model.get_initial_values())

    assert state[0] == pytest.approx(expected_current, rel=1e-8)
    assert state[1] == pytest.approx(expected_voltage, rel=1e-8)


def test_buck_switching_dcm_lands_on_the_natural_current_zero():
    params = _case("buck", "switching").replace(**{
        "units.converter.source_voltage": 24.0,
        "units.converter.inductance": 200e-6,
        "units.converter.capacitance": 100e-6,
        "units.converter.load_resistance": 400.0,
        "units.converter.series_resistance": 0.0,
    })
    plant = peslite.Simulation(params).unit().plant
    plant.set_enabled(True)
    model = Model({"plant": plant}, {})
    solver = FixedStepSolver(2.5e-6, "rk4")
    state = model.get_initial_values()
    current_index = model.state_slice(plant, "i_L").start
    conduction = DiodeConduction(model, solver)
    assert conduction.current_states == ((plant, current_index),)
    time, period, duty = 0.0, 50e-6, 0.4
    for _ in range(1200):
        plant.inp.q = 1.0
        end = time + duty * period
        state = conduction.advance(time, end, state)
        time = end
        plant.inp.q = 0.0
        end = time + (1.0 - duty) * period
        state = conduction.advance(time, end, state)
        time = end

    assert state[current_index] == 0.0
    assert state[current_index + 1] == pytest.approx(21.57550765, rel=1e-4)


def test_diode_boundary_does_not_block_current_reversal_through_an_on_channel():
    plant = peslite.Simulation(_case("buck", "switching")).unit().plant
    plant.set_enabled(True)
    plant.inp.q = 1.0
    plant.state.i_L = 0.0
    plant.state.u_o = 60.0

    assert not plant.diode_blocks_reverse_current()
    assert plant.rhs(0.0)[0] < 0.0


def _light_load_buck(model):
    return _case("buck", model, t_end=0.005).replace(**{
        "units.converter.load_resistance": 600.0,
        "simulation.output.period": 5e-6,
        "simulation.output.states": True,
        "simulation.output.energy": False,
    })


@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_light_load_buck_enters_dcm_without_negative_current(model, tmp_path):
    params = _light_load_buck(model)
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path / model)
    current = result.states["converter.plant.i_L"]
    tail = current[result.states["t"] >= 0.0025]

    assert np.all(current >= 0.0)
    if model == "switching":
        assert np.any(tail == 0.0)
    else:
        assert np.all(tail > 0.0)
        plant = simulation.unit().plant
        alpha, beta, _total, _resistance, _rms = plant._averaged_conduction(plant.inp.q)
        assert plant.inp.q * beta / alpha < 1.0 - plant.inp.q


@pytest.mark.parametrize("model", ["switching", "pwm_averaging"])
def test_adaptive_dcm_matches_the_fixed_step_reference(model, tmp_path):
    fixed_params = _light_load_buck(model)
    adaptive_params = fixed_params.replace(**{
        "simulation.solver.type": "adaptive",
        "simulation.solver.method": "DP45",
        "simulation.solver.max_step": 1e-4,
    })
    fixed = peslite.Simulation(fixed_params).run(out_dir=tmp_path / f"{model}-fixed")
    adaptive = peslite.Simulation(adaptive_params).run(
        out_dir=tmp_path / f"{model}-adaptive"
    )

    assert np.all(adaptive.states["converter.plant.i_L"] >= 0.0)
    for state in ("converter.plant.i_L", "converter.plant.u_o"):
        assert adaptive.final_states()[state] == pytest.approx(
            fixed.final_states()[state], rel=1e-7, abs=1e-8
        )


def test_multiple_switching_dc_converters_enforce_their_own_diode_boundaries(tmp_path):
    tree = to_dict(_light_load_buck("switching"))
    tree["units"]["converter_b"] = copy.deepcopy(tree["units"]["converter"])
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path / "multiple")

    for name in ("converter", "converter_b"):
        assert np.all(result.states[f"{name}.plant.i_L"] >= 0.0)
    assert result.final_states()["converter.plant.u_o"] == pytest.approx(
        result.final_states()["converter_b.plant.u_o"], rel=0.0, abs=1e-12
    )


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
    unit = params.unit()
    common = {item.name for item in fields(peslite.assembly.UnitParams)}

    assert {"type", "pwm", "ctrl", "bridge", "s_base"} <= common
    assert {"topology", "source_voltage", "bus", "dclink"}.isdisjoint(common)
    assert unit.topology == "buck"
    assert {"topology", "source_voltage", "inductance", "capacitance"}.isdisjoint(vars(unit))
    assert not hasattr(unit.ctrl.loops["current"], "topology")
    assert not hasattr(unit.ctrl.loops["current"], "blocked_bridge")
    for field, value in (("topology", "buck"), ("blocked_bridge", False)):
        with pytest.raises(ConfigError, match="unknown key"):
            params.replace(**{f"units.converter.ctrl.loops.current.{field}": value})
    with pytest.raises(ConfigError, match="unknown key"):
        params.replace(**{"units.converter.bus": "pcc"})
    with pytest.raises(ConfigError, match="cannot use control type"):
        params.replace(**{"units.converter.ctrl.type": "gfl"})
    with pytest.raises(ConfigError, match=r"ctrl\.references: unknown key"):
        params.replace(**{"units.converter.ctrl.references.p_ref_pu": 0.5})
