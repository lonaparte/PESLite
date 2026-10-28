"""Open DCConverter assembly, scalar network wiring and control behavior."""

from dataclasses import fields

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import ConfigError, from_dict
from peslite.components import HalfBridge, HalfBridgeDiodeConduction, PWMAveragingModel
from peslite.assembly.conduction import DiodeConduction
from peslite.control import (BridgeControlOutput, DCVoltageControl, GraphController,
                             InductorCurrentControl, Startup)


def _case(*, model="pwm_averaging", bridge_type="half_bridge",
          lower_controlled=False, load=6.0, t_end=0.01, ac_base=True):
    base = {"s_base": 500.0}
    if ac_base:
        base.update(v_ll_rms=230.0, f0=50.0)
    return from_dict({
        "base": base,
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
                        "DCVoltageControl": {
                            "kp": 0.3,
                            "ki": 80.0,
                            "minimum": 0.0,
                            "maximum": 15.0,
                        },
                        "InductorCurrentControl": {
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


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
@pytest.mark.parametrize("case,voltage,current,expected", [
    ("buck", 24.0, 5.0, 120.0),
    ("boost", 24.0, 5.0, -120.0),
    ("buck-boost", -48.0, 2.5, 120.0),
])
def test_scalar_power_uses_the_external_terminal_not_control_feedback(
        mode, case, voltage, current, expected):
    params = peslite.load(EXAMPLES / f"{case}-example.pes").replace(**{
        "units.converter.bridge.model": mode,
        "units.converter.meas.average": ["power"],
    })
    unit = peslite.System(params).units["converter"]
    terminal, = (terminal for bus, terminal in unit.terminals() if bus == unit.cfg.bus)
    assert unit._power_terminal is terminal
    setattr(terminal.subsystem.inp, terminal.voltage, voltage)
    setattr(terminal.subsystem.out, terminal.current, current)
    # Keep the regulation feedback deliberately different from the measured power port.
    unit.branch_f.out.i = 5.0
    unit.bus.out.u = voltage
    unit.dclink.out.u_dc = 48.0 if case == "boost" else 24.0
    assert unit.ports.value("power") == pytest.approx(expected)
    if case != "buck":
        control_voltage = unit._port_value(unit._control_voltage_port)
        assert unit.ports.value("power") != control_voltage * unit._current()


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_scalar_feedback_binds_the_same_terminal_power_for_sampled_and_continuous_modes(mode):
    params = peslite.load(EXAMPLES / "boost-example.pes").replace(**{
        "units.converter.bridge.model": mode,
        "units.converter.ctrl": {
            "type": "resonant",
            "references": {"modulation": 0.5, "frequency": 100.0, "power": -120.0},
            "loops": {
                "ResonantModulation": {},
                "PowerControl": {"kp": 0.001, "ki": 0.1, "minimum": 0.0, "maximum": 0.9},
            },
            "connections": {
                "PowerControl.reference": "references.power",
                "PowerControl.power": "meas.power",
                "PowerControl.feedforward": "references.modulation",
                "ResonantModulation.modulation": "PowerControl.modulation",
            },
        },
    })
    unit = peslite.System(params).units["converter"]
    terminal = unit._power_terminal
    terminal.subsystem.inp.u1 = 24.0
    terminal.subsystem.out.i = 5.0
    assert unit.control_inputs()["meas.power"] == -120.0
    if mode == "averaging":
        connections = unit.connections()
        voltage_source, current_source = terminal.signal_sources()
        assert connections[unit.ctrl, "power_voltage"] == voltage_source
        assert connections[unit.ctrl, "power_current"] == current_source
        unit.ctrl.inp.power_voltage = 24.0
        unit.ctrl.inp.power_current = 5.0
        assert unit.ctrl._flow_measurement().power == -120.0


class _FixedDutyController:
    """Test-only source used to validate the power circuit independently of its control loops."""

    def __init__(self, period, duty):
        self.startup = Startup(period, run=False)
        self.duty = duty
        self.continuous = False
        self.periods = {}

    def command(self, run, ramp=0.0):
        self.startup.command(run, ramp)

    def __call__(self, _t, _measurement):
        if self.startup.in_progress:
            self.startup.advance()
        return BridgeControlOutput(
            duty=self.duty,
            gates=self.startup.active,
            startup_complete=self.startup.complete,
        )


def _reference_buck_boost(vin, duty, inductance, capacitance, resistance,
                          switching_period, duration, step):
    """Independent ideal switched-circuit integration in positive output-magnitude coordinates."""
    current = voltage = 0.0
    samples = []

    def derivative(t, i_l, v_out):
        on = t % switching_period < duty * switching_period
        if on:
            return vin / inductance, -v_out / (resistance * capacitance)
        if i_l > 0.0:
            return -v_out / inductance, (i_l - v_out / resistance) / capacitance
        return 0.0, -v_out / (resistance * capacitance)

    steps = int(round(duration / step))
    for index in range(steps):
        t = index * step
        k1 = derivative(t, current, voltage)
        k2 = derivative(t + step / 2.0,
                        current + step * k1[0] / 2.0,
                        voltage + step * k1[1] / 2.0)
        k3 = derivative(t + step / 2.0,
                        current + step * k2[0] / 2.0,
                        voltage + step * k2[1] / 2.0)
        k4 = derivative(t + step,
                        current + step * k3[0],
                        voltage + step * k3[1])
        current += step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        voltage += step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        current = max(0.0, current)
        if t >= duration - 10.0 * switching_period:
            samples.append((current, voltage))
    return np.asarray(samples)


def test_dc_converter_is_an_open_unit_connected_through_the_common_terminal_api():
    simulation = peslite.Simulation(_case())
    unit = simulation.unit()
    names = set(simulation.system.model.names)

    assert isinstance(unit.bridge.topology, peslite.HalfBridge)
    assert unit.bus is simulation.system.buses["output"]
    assert ("converter.bus.dc", unit.dclink.terminal) in unit.terminals()
    assert ("converter.bus.dc", unit.bridge.dc_terminal) in unit.terminals()
    assert ("converter.bus.power", unit.bridge.power_terminal) in unit.terminals()
    assert ("output", unit.branch_f.terminal2) in unit.terminals()
    assert {"output", "load", "converter.branch_f", "converter.bridge",
            "converter.dclink"} <= names
    assert not hasattr(unit, "plant")
    assert isinstance(unit.ctrl.graph.nodes["DCVoltageControl"], DCVoltageControl)
    assert isinstance(unit.ctrl.graph.nodes["InductorCurrentControl"], InductorCurrentControl)


def test_dc_converter_si_model_is_independent_of_the_optional_ac_base(tmp_path):
    configured = _case(t_end=0.0002)
    dc_only = _case(t_end=0.0002, ac_base=False)
    assert dc_only.base.v_ll_rms is None and dc_only.base.f0 is None

    expected = peslite.Simulation(configured).run(out_dir=tmp_path / "configured")
    actual = peslite.Simulation(dc_only).run(out_dir=tmp_path / "dc-only")
    assert actual.states.keys() == expected.states.keys()
    for name, values in expected.states.items():
        np.testing.assert_array_equal(actual.states[name], values)


def test_dc_converter_rejects_a_three_phase_point_network():
    with pytest.raises(ConfigError, match="needs a single-phase bus"):
        _case(t_end=0.0002).replace(**{"buses.output.phases": 3})


def test_scalar_control_limits_need_a_nonzero_range():
    with pytest.raises(ConfigError, match="minimum must be below maximum"):
        _case().replace(**{
            "units.converter.ctrl.loops.InductorCurrentControl.minimum": 0.95,
            "units.converter.ctrl.loops.InductorCurrentControl.maximum": 0.95,
        })


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


def test_public_factories_follow_the_registered_unit_configuration():
    cfg = _case().unit()
    bridge = peslite.make_bridge(cfg)
    controller = peslite.make_controller(cfg, topology=bridge.topology)

    assert isinstance(bridge, PWMAveragingModel)
    assert isinstance(bridge.topology, HalfBridge)
    assert isinstance(controller, GraphController)


def test_half_bridge_device_placement_controls_unilateral_current():
    diode_unit = peslite.Simulation(_case(lower_controlled=False)).unit()
    controlled_unit = peslite.Simulation(_case(lower_controlled=True)).unit()

    assert diode_unit.branch_f.unilateral
    assert not controlled_unit.branch_f.unilateral
    assert not diode_unit.bridge.topology.lower_controlled
    assert controlled_unit.bridge.topology.lower_controlled


def test_buck_boost_uses_the_three_physical_half_bridge_terminals():
    params = peslite.load(EXAMPLES / "buck-boost-example.pes").replace(**{
        "units.converter.bridge.model": "switching",
    })
    simulation = peslite.Simulation(params)
    unit = simulation.unit()
    terminals = unit.terminals()

    assert (unit.power_bus, unit.bridge.power_terminal) in terminals
    assert (unit.power_bus, unit.branch_f.terminal1) in terminals
    assert (unit.cfg.bus, unit.bridge.lower_terminal) in terminals
    assert all(terminal is not unit.branch_f.terminal2 for _bus, terminal in terminals)

    bridge = unit.bridge
    bridge.inp.u_dc = 24.0
    bridge.inp.u_lower = -36.0
    bridge.inp.i_c = 2.0
    bridge.inp.q = 1.0
    bridge.set_outputs(0.0)
    assert (bridge.out.u_c, bridge.out.i_dc, bridge.out.i_lower) == (24.0, 2.0, 0.0)
    bridge.inp.q = 0.0
    bridge.set_outputs(0.0)
    assert (bridge.out.u_c, bridge.out.i_dc, bridge.out.i_lower) == (-36.0, 0.0, 2.0)


@pytest.mark.parametrize("resistance,mode", [(10.0, "ccm"), (40.0, "dcm")])
def test_buck_boost_switching_matches_an_independent_circuit_reference(
        resistance, mode, tmp_path):
    vin, duty = 24.0, 0.4
    inductance, capacitance = 200.0e-6, 100.0e-6
    switching_period = 50.0e-6
    duration, solver_step = 0.03, 0.5e-6
    params = peslite.load(EXAMPLES / "buck-boost-example.pes").replace(**{
        "units.converter.filter.l_f": inductance,
        "units.converter.filter.r_f": 0.0,
        "units.converter.bridge.model": "switching",
        "units.converter.pwm.f_sw": 1.0 / switching_period,
        "elements.load.r": resistance,
        "buses.output.c": capacitance,
        "simulation.t_end": duration,
        "simulation.solver.dt": solver_step,
        "simulation.output.period": 2.5e-6,
        "simulation.output.record_every": 1,
        "simulation.output.states": True,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    })
    controller = _FixedDutyController(params.unit().ctrl.period, duty)
    result = peslite.Simulation(
        params, instances={"converter.ctrl": controller},
    ).run(out_dir=tmp_path)
    settled = result.t >= duration - 10.0 * switching_period
    actual_current = result.states["converter.branch_f.i"][settled]
    actual_voltage = -result.states["output.u_C"][settled]

    reference = _reference_buck_boost(
        vin, duty, inductance, capacitance, resistance,
        switching_period, duration, solver_step / 2.0,
    )
    reference_current = reference[:, 0]
    reference_voltage = reference[:, 1]

    assert actual_voltage.mean() == pytest.approx(reference_voltage.mean(), rel=0.012)
    assert actual_current.mean() == pytest.approx(reference_current.mean(), rel=0.02)
    if mode == "ccm":
        assert actual_current.min() > 0.05
        assert actual_voltage.mean() == pytest.approx(duty / (1.0 - duty) * vin, rel=0.02)
    else:
        expected = vin * duty * np.sqrt(
            resistance * switching_period / (2.0 * inductance)
        )
        assert np.count_nonzero(np.isclose(actual_current, 0.0, atol=1e-10)) > 2
        assert actual_voltage.mean() == pytest.approx(expected, rel=0.025)


def test_averaged_buck_boost_has_the_same_ccm_and_dcm_balance_relations():
    vin, duty = 24.0, 0.4
    period, inductance = 50.0e-6, 200.0e-6
    topology = HalfBridge(lower_controlled=False, external_terminal="lower_rail")
    conduction = HalfBridgeDiodeConduction(period, inductance, 0.0, topology)

    ccm_voltage = duty / (1.0 - duty) * vin
    ccm_current = ccm_voltage / (10.0 * (1.0 - duty))
    ccm = conduction.equations(ccm_current, duty, vin, -ccm_voltage, 0.0)
    assert ccm.total == 1.0
    assert ccm.derivative == pytest.approx(0.0, abs=1e-10)

    resistance = 40.0
    dcm_voltage = vin * duty * np.sqrt(resistance * period / (2.0 * inductance))
    dcm_total = duty * (1.0 + vin / dcm_voltage)
    peak_current = duty * period * vin / inductance
    dcm_current = 0.5 * peak_current * dcm_total
    dcm = conduction.equations(dcm_current, duty, vin, -dcm_voltage, 0.0)
    assert dcm.in_dcm
    assert dcm.total == pytest.approx(dcm_total)
    assert dcm.derivative == pytest.approx(0.0, abs=1e-10)
    assert dcm.lower_current_ratio * dcm_current == pytest.approx(dcm_voltage / resistance)


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
    current = expected_voltage / load
    conduction = HalfBridgeDiodeConduction(50.0e-6, 1.0e-3, 0.0)
    result = conduction.equations(current, 0.3, 48.0, 0.0, expected_voltage)

    assert result.derivative == pytest.approx(0.0, abs=2e-10)
    if expected_total is None:
        assert result.total < 1.0
        assert result.upper_current_ratio == result.controlled
    else:
        assert result.total == expected_total


def test_control_feedforward_depends_on_bridge_structure_not_an_application_name():
    half = peslite.Simulation(_case()).unit().ctrl
    full = peslite.Simulation(_case(bridge_type="full_bridge", lower_controlled=True)).unit().ctrl

    assert half.control_inputs(1.0, 12.0, 48.0)["meas.duty_feedforward"] == pytest.approx(0.25)
    assert full.control_inputs(1.0, 12.0, 48.0)["meas.duty_feedforward"] == pytest.approx(0.625)


def test_explicit_half_bridge_orientation_sets_the_named_bus_and_boost_feedforward():
    params = peslite.load(EXAMPLES / "boost-example.pes")
    unit = peslite.Simulation(params).unit()

    assert unit.bridge.topology.reverse_port
    assert ("converter.bus.power", unit.branch_f.terminal2) in unit.terminals()
    assert ("input", unit.branch_f.terminal1) in unit.terminals()
    assert unit.ctrl.control_inputs(2.0, 48.0, 24.0)["meas.duty_feedforward"] == pytest.approx(0.5)


def test_averaged_boost_enters_dcm_from_its_operating_point():
    conduction = HalfBridgeDiodeConduction(
        50.0e-6,
        1.0e-3,
        0.0,
        HalfBridge(upper_controlled=False, reverse_port=True),
    )
    result = conduction.equations(0.108, 0.7, 48.0, 0.0, 24.0)

    assert result.total == pytest.approx(0.6)
    assert result.upper_current_ratio == result.diode
    assert result.derivative == pytest.approx(0.0, abs=2e-10)


def test_averaged_dcm_relation_drives_branch_current_loss_and_bridge_dc_current():
    simulation = peslite.Simulation(_case(model="averaging", load=600.0))
    unit = simulation.unit()
    law = unit.conduction
    connections = simulation.system.model.connections

    assert law is not None
    assert connections[(unit.branch_f, "di_dt")] == (law, "derivative")
    assert connections[(unit.branch_f, "loss_factor")] == (
        law, "resistance_factor",
    )
    assert connections[(unit.bridge, "upper_current_ratio")] == (
        law, "upper_current_ratio",
    )
    assert connections[(unit.bridge, "lower_current_ratio")] == (
        law, "lower_current_ratio",
    )

    current = 32.0993023218081 / 600.0
    result = law.equations(current, 0.3, 48.0, 0.0, 32.0993023218081)
    assert result.in_dcm
    assert result.upper_current_ratio > 0.3


def test_ordinary_bridge_path_does_not_copy_unused_conduction_ratios():
    simulation = peslite.Simulation(_case(model="pwm_averaging", lower_controlled=True))
    unit = simulation.unit()

    assert (unit.bridge, "q") in unit.zoh_connections()
    for field in ("upper_current_ratio", "lower_current_ratio"):
        assert (unit.bridge, field) not in unit.zoh_connections()
        assert (unit.bridge, field) not in simulation.system.model.connections


def test_strict_energy_audit_rejects_cycle_averaged_dcm_storage():
    params = _case(model="averaging").replace(**{
        "simulation.energy_check": "strict",
    })

    with pytest.raises(ConfigError, match="cycle-averaged DCM current"):
        peslite.Simulation(params)


def test_ratio_one_solver_schedule_still_honours_the_dcm_local_step():
    params = _case(model="averaging", load=600.0, t_end=50.0e-6).replace(**{
        "simulation.solver.dt": 50.0e-6,
        "simulation.solver.subsystems": {"converter.branch_f": 1},
    })
    simulation = peslite.Simulation(params)
    assert type(simulation.solver) is peslite.FixedStepSolver
    simulation.unit().conduction.diode_maximum_step = lambda: 5.0e-6

    stage = DiodeConduction(simulation.system.model, simulation.solver)
    initial = simulation.system.model.get_initial_values()
    before = simulation.solver.n_rhs
    stage.advance(0.0, 50.0e-6, initial)

    assert simulation.solver.n_rhs - before >= 40


def test_dcm_local_step_subsystem_cannot_be_window_averaged():
    params = _case(model="averaging", load=600.0, t_end=50.0e-6).replace(**{
        "simulation.solver.subsystems": {"converter.branch_f": 4},
        "simulation.output.states": 0,
        "simulation.output.signals": 0,
        "simulation.output.energy": 0,
        "simulation.energy_check": "off",
    })
    simulation = peslite.Simulation(params)

    with pytest.raises(ConfigError, match="state-dependent local step.*windowed"):
        simulation.run(out_dir=None)


def test_scalar_bus_rating_is_inherited_from_its_connected_unit_port():
    buck = peslite.Simulation(_case())
    storage = buck.system.buses["output"].storage[0]
    assert buck._ratings()("output", storage) == pytest.approx((48.0, 500.0 / 48.0))

    buck_boost = peslite.Simulation(peslite.load(EXAMPLES / "buck-boost-example.pes"))
    storage = buck_boost.system.buses["output"].storage[0]
    assert buck_boost._ratings()("output", storage) == pytest.approx((24.0, 500.0 / 24.0))


def test_scalar_bridge_feedforward_is_safe_before_the_named_buses_are_energised():
    forward = HalfBridge()
    reverse = HalfBridge(reverse_port=True)
    full = peslite.FullBridge()

    assert forward.control_values(0.0, 0.0, 0.0)[3] == 0.0
    assert reverse.control_values(0.0, 0.0, 0.0)[3] == 0.0
    assert full.control_values(0.0, 0.0, 0.0)[3] == 0.5


def test_unit_and_control_scopes_have_only_shared_and_type_specific_fields():
    params = _case()
    unit = params.unit()
    common = {item.name for item in fields(peslite.assembly.UnitParams)}

    assert {
        "type", "pwm", "ctrl", "bridge", "meas", "protection", "s_base", "bus", "dclink",
    } <= common
    assert "filter" not in common
    assert unit.bus == "output"
    for old in ("topology", "source_voltage", "inductance", "capacitance",
                "load_resistance", "series_resistance"):
        assert not hasattr(unit, old)
        with pytest.raises(ConfigError, match="unknown key"):
            params.replace(**{f"units.converter.{old}": 1.0})


def test_converter_families_share_one_unit_lifecycle_implementation():
    lifecycle = {
        "connect", "start", "states_loaded", "begin", "sense", "actuate", "_set_gates",
    }

    assert lifecycle <= set(peslite.Unit.__dict__)
    assert lifecycle.isdisjoint(peslite.VSC.__dict__)
    assert lifecycle.isdisjoint(peslite.DCConverter.__dict__)


def test_dc_converter_rejects_wrong_network_and_bridge_scopes():
    params = _case()
    with pytest.raises(ConfigError, match="unit type 'dc_converter' requires one of"):
        params.replace(**{"units.converter.bridge.type": "three_phase_bridge"})
    with pytest.raises(ConfigError, match="specified together|unknown bus None"):
        params.replace(**{"units.converter.bus": None})


def test_restored_dc_converter_trip_stops_the_common_controller_lifecycle():
    unit = peslite.Simulation(_case(t_end=0.0002)).unit()
    unit.start(0.0)
    assert unit.ctrl.startup.run

    unit.set_state({"tripped": True})
    unit.states_loaded()

    assert not unit.ctrl.startup.run
    assert not unit.breaker
    assert not unit.gates
    assert unit.trip_cause == "restored"


def test_dc_converter_uses_the_common_adc_and_protection_runtime(tmp_path):
    params = _case(t_end=0.001).replace(**{
        "units.converter.meas.average": ["current", "voltage", "vin"],
        "units.converter.meas.window": 50.0e-6,
        "units.converter.protection.overcurrent.enable": 1,
        "units.converter.protection.overcurrent.limit_pu": 10.0,
    })
    simulation = peslite.Simulation(params)
    unit = simulation.unit()
    result = simulation.run(out_dir=tmp_path)

    assert isinstance(unit.adc, peslite.ADC)
    assert unit.protection is not None
    assert {"converter.meas.x_current", "converter.meas.x_voltage",
            "converter.meas.x_vin"} <= set(result.states)
    assert result.summary["converter.max_current_pu"] > 0.0
    assert result.summary["converter.trip_cause"] is None


def test_dc_converter_rejects_protection_without_a_physical_signal():
    with pytest.raises(ConfigError, match="does not provide the required signal"):
        _case().replace(**{
            "units.converter.protection.frequency.enable": 1,
            "units.converter.protection.frequency.limit": 1.0,
        })


def test_dc_converter_common_overcurrent_protection_trips_its_unit(tmp_path):
    params = _case(t_end=0.002).replace(**{
        "units.converter.protection.overcurrent.enable": 1,
        "units.converter.protection.overcurrent.limit_pu": 0.01,
        "simulation.stop_on_trip": 0,
    })
    result = peslite.Simulation(params).run(out_dir=tmp_path)

    assert result.summary["converter.trip_cause"] == "overcurrent"
    assert result.summary["converter.tripped"] == 1
    assert {"OVERCURRENT", "TRIP_OVERCURRENT"} <= set(
        result.summary["converter.alarms"]
    )


def test_dc_converter_restart_uses_the_common_named_state_contract(tmp_path):
    params = _case(
        bridge_type="full_bridge", lower_controlled=True, t_end=0.002,
    ).replace(**{
        "units.converter.meas.average": ["current", "voltage", "vin"],
        "units.converter.meas.window": 50.0e-6,
        "simulation.output.period": 5.0e-6,
    })
    whole_dir = tmp_path / "whole"
    whole = peslite.Simulation(params).run(out_dir=whole_dir)
    start_time = 1.005e-3
    restarted = peslite.load(
        whole_dir / "simulation.pes",
        initial=whole_dir / "states.csv",
        initial_time=start_time,
    )
    continued = peslite.Simulation(restarted).run(out_dir=tmp_path / "continued")

    start = int(np.argmin(abs(whole.t - start_time)))
    assert np.array_equal(continued.t, whole.t[start:])
    for name, values in continued.states.items():
        np.testing.assert_allclose(
            values, whole.states[name][start:], rtol=1e-12, atol=1e-12, err_msg=name,
        )
