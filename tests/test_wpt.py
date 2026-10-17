"""Composite WPT configuration, assembly and compensated-link behavior."""

import copy
from dataclasses import dataclass

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.assembly import Event, register_event_type
from peslite.assembly.params import ConfigError, from_dict, read_tree, to_dict
from peslite.components import LCCLink, SSLink
from peslite.control import GraphController, ResonantModulation


@register_event_type
class _PortScaledEvent(Event):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "test_wpt_port_scale"
        target: str
        t: float
        level_pu: float

        _quantities = {"level_pu": "dc_current"}

    type = "test_wpt_port_scale"

    @staticmethod
    def apply(event, system, t):
        pass


def _tree():
    return read_tree(EXAMPLES / "wpt-example.pes")


def _short(tree=None):
    tree = copy.deepcopy(_tree() if tree is None else tree)
    tree["simulation"]["t_end"] = 20.0e-6
    tree["simulation"]["solver"]["dt"] = 50.0e-9
    tree["simulation"]["output"].update({"states": 0, "signals": 0, "energy": 0})
    return from_dict(tree)


def test_wpt_configuration_has_one_authoritative_nested_unit_tree():
    params = from_dict(_tree())

    assert tuple(params.units) == ("charger",)
    assert params.unit("charger").type == "wpt"
    assert not hasattr(params, "runtime_units")
    assert not hasattr(params, "unit_groups")
    assert params.unit("charger.primary").type == "dc_converter"
    assert params.unit("charger.primary").ctrl.period == pytest.approx(1.0e-6)
    assert params.unit("charger.secondary").ctrl.period == pytest.approx(1.25e-6)


def test_wpt_default_port_controls_composite_scaling_and_link_rating():
    tree = _tree()
    tree["events"] = {
        "default": {
            "type": "test_wpt_port_scale",
            "target": "charger",
            "t": 10.0e-6,
            "level": 12.5,
        },
        "secondary": {
            "type": "test_wpt_port_scale",
            "target": "charger.secondary",
            "t": 20.0e-6,
            "level": 5000.0 / 300.0,
        },
    }
    params = from_dict(tree)
    assert params.events["default"].level_pu == pytest.approx(1.0)
    assert params.events["secondary"].level_pu == pytest.approx(1.0)

    simulation = peslite.Simulation(_short(tree))
    subsystems = dict(zip(simulation.system.model.names, simulation.system.model.subsystems))
    link = subsystems["charger.link"]
    rating = simulation._ratings()
    assert simulation.system._port_bindings["charger.link"][0].owner == "charger.primary"
    assert all(
        rating("charger.link", storage) == pytest.approx((400.0, 12.5))
        for storage in link.storage
    )


def test_wpt_assembly_flattens_only_once_and_shares_one_link():
    simulation = peslite.Simulation(_short())
    subsystems = dict(zip(simulation.system.model.names, simulation.system.model.subsystems))

    assert tuple(simulation.units) == ("charger.primary", "charger.secondary")
    assert isinstance(subsystems["charger.link"], SSLink)
    assert "charger" not in simulation.units
    link = subsystems["charger.link"]
    for side, unit in (("primary", simulation.units["charger.primary"]),
                       ("secondary", simulation.units["charger.secondary"])):
        assert not hasattr(unit, "link")
        assert not hasattr(unit, "side")
        assert unit.power_bus == f"charger.{side}.bus.power"
        assert (unit.power_bus, unit.bridge.power_terminal) in unit.terminals()
        assert simulation.system.model.connections[(link, f"u_{side}")] == (
            unit.bridge, "u_c"
        )
        assert simulation.system.model.connections[(unit.bridge, "i_c")] == [
            (link, f"i_{side}", 1.0)
        ]
        assert type(unit.ctrl) is GraphController
        assert isinstance(unit.ctrl.graph.nodes["ResonantModulation"], ResonantModulation)
    assert not hasattr(simulation.system, "unit_groups")
    assert not simulation.energy_problems


def test_nested_wpt_exports_its_declared_default_port():
    tree = _tree()
    tree["units"]["charger"]["primary"] = copy.deepcopy(tree["units"]["charger"])
    simulation = peslite.Simulation(_short(tree))
    subsystems = dict(zip(simulation.system.model.names, simulation.system.model.subsystems))
    outer_link = subsystems["charger.link"]
    primary = simulation.units["charger.primary.primary"]

    assert tuple(simulation.units) == (
        "charger.primary.primary",
        "charger.primary.secondary",
        "charger.secondary",
    )
    assert simulation.system.model.connections[(outer_link, "u_primary")] == (
        primary.dclink,
        "u_dc",
    )
    result = simulation.run(out_dir=None)
    assert result.summary["ph_verdict"] == "port-hamiltonian"


@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_wpt_runs_with_each_bridge_form(model):
    tree = _tree()
    for side in ("primary", "secondary"):
        tree["units"]["charger"][side]["bridge"] = {"model": model}
    params = _short(tree)
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=None)

    assert result.summary["t_stop"] == pytest.approx(params.simulation.t_end)
    assert result.summary["ph_verdict"] == "port-hamiltonian"
    assert np.isfinite(simulation.system.model.get_initial_values()).all()


def test_nested_reverse_half_bridge_uses_the_scoped_power_bus_voltage():
    tree = _tree()
    tree["units"]["charger"]["primary"]["bridge"] = {
        "type": "half_bridge",
        "model": "averaging",
        "reverse_port": True,
    }
    simulation = peslite.Simulation(_short(tree))
    unit = simulation.units["charger.primary"]

    assert unit._control_supply_port == (unit.bridge, "u_c")
    assert simulation.system.model.connections[(unit.ctrl, "vin")] == (
        unit.bridge, "u_c",
    )
    result = simulation.run(out_dir=None)
    assert result.summary["t_stop"] == pytest.approx(20.0e-6)


@pytest.mark.parametrize("model", ["switching", "pwm_averaging"])
def test_named_power_bus_blocks_held_command_while_disconnected(model):
    tree = _tree()
    for side in ("primary", "secondary"):
        tree["units"]["charger"][side]["bridge"] = {"model": model}
    tree["events"] = {
        "stop_primary": {
            "type": "disconnect",
            "target": "charger.primary",
            "t": 8.0e-6,
        },
    }
    simulation = peslite.Simulation(_short(tree))
    simulation.run(out_dir=None)
    primary = simulation.units["charger.primary"]

    assert not primary.breaker
    assert not primary.gates
    assert primary.bridge.inp.q == primary.bridge.topology.zero


def test_composite_switching_event_is_inherited_by_both_children():
    tree = _tree()
    tree["events"] = {
        "start": {"type": "connect", "target": "charger", "t": 5.0e-6},
    }
    params = from_dict(tree)

    for unit in (params.unit("charger.primary"), params.unit("charger.secondary")):
        assert [(event.type, event.t) for event in unit.events] == [("connect", 5.0e-6)]

    system = peslite.System(params)
    system.switch("charger", False, 0.0)
    assert all(not unit.breaker for unit in system.units.values())
    system.switch("charger", True, 1.0e-6)
    assert all(unit.breaker for unit in system.units.values())


def test_nested_child_reference_can_change_during_a_run():
    tree = _tree()
    tree["events"] = {
        "retune": {
            "type": "set",
            "t": 10.0e-6,
            "set": {"units.charger.primary.ctrl.references.modulation": 0.6},
        },
    }
    params = from_dict(tree)

    assert params.changes[0].paths == (
        "units.charger.primary.ctrl.references.modulation",
    )
    system = peslite.System(params)
    system.apply(params.changes[0])
    assert system.units["charger.primary"].ctrl.p.ctrl.references.modulation == 0.6


def test_lcc_link_uses_the_same_composite_assembly_path():
    tree = _tree()
    tree["units"]["charger"]["link"] = {
        "type": "lcc",
        "primary_inductance": 100.0e-6,
        "secondary_inductance": 100.0e-6,
        "mutual_inductance": 20.0e-6,
        "primary_series_capacitance": 35.07e-9,
        "secondary_series_capacitance": 35.07e-9,
        "primary_compensation_inductance": 30.0e-6,
        "secondary_compensation_inductance": 30.0e-6,
        "primary_parallel_capacitance": 100.0e-9,
        "secondary_parallel_capacitance": 100.0e-9,
        "primary_resistance": 0.5,
        "secondary_resistance": 0.5,
        "primary_compensation_resistance": 0.1,
        "secondary_compensation_resistance": 0.1,
    }
    simulation = peslite.Simulation(_short(tree))
    subsystems = dict(zip(simulation.system.model.names, simulation.system.model.subsystems))

    assert isinstance(subsystems["charger.link"], LCCLink)
    assert not simulation.energy_problems
    result = simulation.run(out_dir=None)
    assert result.summary["t_stop"] == pytest.approx(20.0e-6)


def test_wpt_rejects_parameters_in_the_wrong_scope():
    tree = _tree()
    tree["units"]["charger"]["pwm"] = {"f_sw": 100.0e3}

    with pytest.raises(ConfigError, match=r"units\.charger: unknown key.*pwm"):
        from_dict(tree)


def test_directly_connected_dc_converter_requires_a_composite_parent():
    tree = _tree()
    tree["units"] = {"port": tree["units"]["charger"]["primary"]}

    with pytest.raises(ConfigError, match="unknown bus None|standalone DC converter"):
        from_dict(tree)


def test_wpt_tree_round_trips_without_losing_child_unit_types():
    params = from_dict(_tree())
    rebuilt = from_dict(to_dict(params))

    assert rebuilt.unit("charger").type == "wpt"
    assert {rebuilt.unit(f"charger.{side}").type for side in ("primary", "secondary")} == {
        "dc_converter"
    }


def test_wpt_children_have_independent_common_adc_clocks():
    tree = _tree()
    tree["units"]["charger"]["primary"]["meas"] = {"period": 0.5e-6}
    tree["units"]["charger"]["secondary"]["meas"] = {"period": 0.25e-6}
    simulation = peslite.Simulation(_short(tree))
    primary = simulation.units["charger.primary"]
    secondary = simulation.units["charger.secondary"]

    assert primary.adc.sample_period == pytest.approx(0.5e-6)
    assert secondary.adc.sample_period == pytest.approx(0.25e-6)
    assert primary.adc.samples == 2
    assert secondary.adc.samples == 5
    result = simulation.run(out_dir=None)
    assert result.summary["t_stop"] == pytest.approx(20.0e-6)


def test_wpt_restart_derives_both_child_clock_positions(tmp_path):
    tree = _tree()
    tree["simulation"]["t_end"] = 20.0e-6
    tree["simulation"]["output"].update({
        "period": 0.5e-6,
        "states": 1,
        "signals": 0,
        "energy": 0,
    })
    params = from_dict(tree)
    whole_dir = tmp_path / "whole"
    whole = peslite.Simulation(params).run(out_dir=whole_dir)
    start_time = 10.5e-6
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


@pytest.mark.parametrize("case,side,loop", [
    ("ss", "primary", "PowerControl"),
    ("lcc", "secondary", "PhasePowerControl"),
])
@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_wpt_power_feedback_uses_only_local_ports_and_the_shared_graph(case, side, loop, mode):
    tree = read_tree(EXAMPLES / f"wpt-{case}-example.pes")
    for child in ("primary", "secondary"):
        tree["units"]["charger"][child]["bridge"] = {"model": mode}
    simulation = peslite.Simulation(_short(tree))
    unit = simulation.units[f"charger.{side}"]
    graph = unit.ctrl.graph

    assert type(unit.ctrl) is GraphController
    assert set(graph.nodes) == {loop, "ResonantModulation"}
    assert graph.connections[f"{loop}.power"] == "meas.power"
    assert unit._control_voltage_port == (unit.bridge, "u_c")
    assert unit.windowed is (mode != "averaging")
    result = simulation.run(out_dir=None)
    assert result.summary["ph_verdict"] == "port-hamiltonian"
    assert np.isfinite(simulation.system.model.get_initial_values()).all()


def test_wpt_power_reference_rejects_voltage_signal_connections():
    tree = read_tree(EXAMPLES / "wpt-ss-example.pes")
    tree["units"]["charger"]["primary"]["ctrl"]["connections"][
        "PowerControl.reference"
    ] = "meas.voltage"
    with pytest.raises(ConfigError, match="incompatible signal"):
        peslite.System(from_dict(tree))


@pytest.mark.parametrize("case,side,loop,command,targets", [
    ("ss", "primary", "PowerControl", "modulation", (1200.0, 1800.0)),
    ("lcc", "secondary", "PhasePowerControl", "phase", (-500.0, -800.0)),
])
@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_wpt_tracks_both_power_plateaus_without_remote_measurements(
        case, side, loop, command, targets, mode, tmp_path):
    params = peslite.load(EXAMPLES / f"wpt-{case}-example.pes").replace(**{
        "units.charger.primary.bridge.model": mode,
        "units.charger.secondary.bridge.model": mode,
        "simulation.solver.dt": 250.0e-9,
        "simulation.output.states": False,
        "simulation.output.energy": False,
        "simulation.output.record_every": 1,
        "simulation.energy_check": "off",
    })
    result = peslite.Simulation(params).run(out_dir=tmp_path)
    prefix = f"charger.{side}"
    for begin, end, target in ((0.0024, 0.0028, targets[0]),
                               (0.0054, 0.0058, targets[1])):
        selected = (result.t >= begin - 1e-12) & (result.t <= end + 1e-12)
        for name in ("power", f"{loop}.power"):
            values = result.plant[f"{prefix}.{name}"][selected]
            mean = np.sum(0.5 * (values[1:] + values[:-1])
                          * np.diff(result.t[selected])) / (end - begin)
            assert mean == pytest.approx(target, rel=0.005), (mode, name, target, mean)
    limits = params.unit(prefix).ctrl.loops[loop]
    commands = result.plant[f"{prefix}.{command}"]
    finite = commands[np.isfinite(commands)]
    assert np.min(finite) >= limits.minimum - 1e-12
    assert np.max(finite) <= limits.maximum + 1e-12


@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
def test_wpt_power_feedback_restart_preserves_filter_pi_and_adc_window(mode, tmp_path):
    params = peslite.load(EXAMPLES / "wpt-ss-example.pes").replace(**{
        "units.charger.primary.bridge.model": mode,
        "units.charger.secondary.bridge.model": mode,
        "simulation.t_end": 80.0e-6,
        "simulation.output.period": 0.5e-6,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    })
    whole_dir = tmp_path / "whole"
    whole = peslite.Simulation(params).run(out_dir=whole_dir)
    continued_params = peslite.load(
        whole_dir / "simulation.pes", initial=whole_dir / "states.csv", initial_time=40.5e-6,
    )
    continued = peslite.Simulation(continued_params).run(out_dir=tmp_path / "continued")
    start = int(np.argmin(abs(whole.t - continued.t[0])))
    for name, values in continued.states.items():
        np.testing.assert_allclose(
            values, whole.states[name][start:], rtol=1e-10, atol=1e-10, err_msg=name,
        )


def test_power_measurement_is_optional_and_precedes_adc_channel_averaging():
    from peslite.components.adc import ADC, MeasurementPorts

    plain = peslite.System(from_dict(_tree())).units["charger.primary"]
    assert set(plain.ports.readers) == {"current", "voltage", "vin"}
    assert "meas.power" not in plain.ctrl.graph.interface.inputs

    time = 0.0
    ports = MeasurementPorts({
        "voltage": lambda: 10.0 * np.cos(2.0 * np.pi * time),
        "current": lambda: 2.0 * np.cos(2.0 * np.pi * time),
        "power": lambda: 20.0 * np.cos(2.0 * np.pi * time) ** 2,
    })
    adc = ADC(ports, 1.0, 1.0, 1.0, {"voltage": 0.0, "current": 0.0, "power": 0.0})
    adc.seed()
    for time in np.linspace(0.001, 1.0, 1000):
        adc.accumulate(0.001)
    sample = adc.measure(1.0)
    assert sample.values["voltage"] == pytest.approx(0.0, abs=1e-12)
    assert sample.values["current"] == pytest.approx(0.0, abs=1e-12)
    assert sample.values["power"] == pytest.approx(10.0)


