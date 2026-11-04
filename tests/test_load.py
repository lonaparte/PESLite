"""Built-in R-L load and switching events in a running network."""

import numpy as np
import pytest

import peslite
from dataclasses import replace
from conftest import EXAMPLES
from peslite.assembly.params import ConfigError, from_dict


def test_unit_dc_base_is_reused_but_replaced_parameters_get_a_fresh_base():
    params = peslite.load(EXAMPLES / "buck-example.pes")
    unit = params.unit()
    original = unit.dc_base
    assert unit.dc_base is original
    changed = replace(unit, dclink=replace(unit.dclink, vdc_ref=2.0 * original.v))
    changed = changed.resolved(unit.base)
    assert changed.dc_base is changed.dc_base
    assert changed.dc_base is not original
    assert changed.dc_base.v == 2.0 * original.v
    assert changed.dc_base.i == original.i / 2.0
    assert unit.dc_base is original
    from peslite.assembly.params import to_dict
    assert "dc_base" not in to_dict(unit, derived=True)


def _current(result, name):
    return np.abs(result.states[f"{name}.re"] + 1j * result.states[f"{name}.im"])


def _two_buses(case, events):
    tree = case()
    tree["buses"]["b2"] = {"c_pu": 0.02, "r_d_pu": 0.5}
    tree["branches"] = {
        "line": {"bus1": "pcc", "bus2": "b2", "x_pu": 0.05}
    }
    tree["elements"] = {
        "load": {"type": "load", "bus": "b2", "r_pu": 1.0, "x_pu": 0.5}
    }
    tree["events"] = events
    tree["simulation"]["output"] = {"period": 2.5e-5}
    return from_dict(tree)


def test_branch_source_and_load_switch_during_a_run(case, tmp_path):
    params = _two_buses(case, {
        "line_off": {"type": "disconnect", "target": "line", "t": 0.001},
        "line_on": {"type": "connect", "target": "line", "t": 0.002},
        "load_on": {"type": "connect", "target": "load", "t": 0.0015},
        "grid_off": {"type": "disconnect", "target": "grid", "t": 0.003},
    })
    result = peslite.Simulation(params).run(out_dir=tmp_path)
    t = result.states["t"]
    line = _current(result, "line.i")
    load = _current(result, "load.branch.i")
    grid = _current(result, "grid.branch.i")
    assert np.all(line[(t > 0.001 + 1e-9) & (t < 0.002 - 1e-9)] == 0.0)
    assert np.max(line[t > 0.0022]) > 0.0
    assert np.all(load[t < 0.0015 - 1e-9] == 0.0)
    assert np.max(load[t > 0.0022]) > 0.0
    assert np.all(grid[t > 0.003 + 1e-9] == 0.0)
    assert np.max(grid[t < 0.003]) > 0.0


def test_every_network_component_uses_the_same_port_assembly(case):
    params = _two_buses(case, {})
    system = peslite.System(params)
    line = system.branches["line"]

    source = system.sources["grid"]
    assert source.port_bindings() == (
        ("grid.source", source.emf.port),
        ("grid.source", source.branch.port1),
        ("pcc", source.branch.port2),
    )
    unit = system.units["vsc"]
    assert ("vsc.bus.dc", unit.dclink.port) in unit.port_bindings()
    assert ("vsc.bus.dc", unit.bridge.dc_port) in unit.port_bindings()
    assert ("vsc.bus.power", unit.bridge.power_port) in unit.port_bindings()
    assert ("pcc", unit.branch_f.port2) in unit.port_bindings()
    assert system.named_elements["load"].port_bindings() == (
        ("b2", system.named_elements["load"].branch.port1),
    )


def test_public_bus_may_have_no_attached_component(case):
    tree = case()
    tree["buses"]["unused"] = dict(tree["buses"]["pcc"])
    system = peslite.System(from_dict(tree))
    node = system.buses["unused"]

    assert node.port.current == 0j


@pytest.mark.parametrize("circuit_model", ("device", "statewise", "nodal"))
@pytest.mark.parametrize("bridge_model", ("switching", "pwm_averaging", "averaging"))
def test_bus_without_capacitance_joins_devices_without_adding_a_state(
        case, tmp_path, circuit_model, bridge_model):
    tree = case()
    tree["buses"]["pcc"] = {"phases": 3}
    tree["elements"] = {"load": {"type": "resistor", "bus": "pcc", "r": 10.0}}
    tree["units"]["vsc"]["bridge"] = {"model": bridge_model}
    tree["simulation"].update(t_end=5e-5, energy_check="off")
    tree["simulation"]["solver"]["circuit_model"] = circuit_model
    tree["simulation"]["progress"] = {"enable": 0}
    simulation = peslite.Simulation(from_dict(tree))
    simulation.run(out_dir=tmp_path)
    bus = simulation.system.buses["pcc"]

    assert bus.state_names == ()
    assert bus.port is not None
    assert simulation.system.units["vsc"].bus is bus
    assert abs(simulation.system.signals(tree["simulation"]["t_end"])["pcc.u"]) > 0.0


def test_empty_bus_without_capacitance_has_no_model_or_signal(case):
    tree = case()
    tree["buses"]["unused"] = {}
    system = peslite.System(from_dict(tree))

    assert system.buses["unused"].port is None
    assert "unused" not in system.model.names
    assert "unused.u" not in system.signals(0.0)


def test_bus_damping_resistance_requires_capacitance(case):
    tree = case()
    tree["buses"]["pcc"] = {"r_d": 1.0}

    with pytest.raises(ConfigError, match=r"buses\.pcc\.r_d requires c"):
        from_dict(tree)


def test_bus_capacitance_cannot_be_added_during_a_run(case):
    tree = case()
    tree["buses"]["pcc"] = {"phases": 3}
    tree["events"] = {"add_capacitor": {
        "type": "set", "t": 1e-3, "set": {"buses.pcc.c": 1e-3},
    }}

    with pytest.raises(ConfigError, match=r"buses\.pcc\.c cannot change during a run"):
        from_dict(tree)


def test_assembly_rejects_a_subsystem_name_collision(case):
    params = from_dict(case())
    unit = peslite.System(params).units["vsc"]
    original = unit.subsystems()
    unit.subsystems = lambda: {**original, "pcc": unit.bridge}

    with pytest.raises(ConfigError, match="subsystem name collision.*pcc"):
        peslite.System(params, parts={"vsc": unit})


def test_assembly_rejects_a_continuous_and_held_target_collision(case):
    params = from_dict(case())
    unit = peslite.System(params).units["vsc"]
    unit.connections = lambda: {(unit.bridge, "q"): (unit.branch_f, "i")}
    unit.zoh_connections = lambda: {(unit.bridge, "q"): "duplicate"}

    with pytest.raises(ConfigError, match="input target is already driven"):
        peslite.System(params, parts={"vsc": unit})


def test_state_alias_reuse_requires_the_same_target():
    params = peslite.load(EXAMPLES / "two-converters-example.pes")
    units = peslite.System(params).units
    for unit in units.values():
        unit.aliases = lambda: {"shared": "same.state"}

    system = peslite.System(params, parts=units)
    assert system.state_aliases["shared"] == "same.state"

    first, second = units.values()
    first.aliases = lambda: {"shared": "first.state"}
    second.aliases = lambda: {"shared": "second.state"}

    with pytest.raises(ConfigError, match="state alias collision.*shared"):
        peslite.System(params, parts=units)


def test_signal_collection_rejects_a_name_collision(case):
    system = peslite.System(from_dict(case()))
    system.sources["grid"].signals = lambda: {"shared": 1.0}
    system.units["vsc"].signals = lambda: {"shared": 2.0}

    with pytest.raises(ConfigError, match="signal name collision.*shared"):
        system.signals(0.0)


def test_signal_collection_allows_the_same_producer_to_be_reused(case):
    system = peslite.System(from_dict(case()))
    source = system.sources["grid"]
    system.composite_signal_sources += (source,)

    signals = system.signals(0.0)
    assert signals["grid.i"] == source.branch.output.i


def test_small_load_can_model_a_switched_fault(case, tmp_path):
    tree = case()
    tree["elements"] = {
        "fault": {"type": "load", "bus": "pcc", "r_pu": 0.01, "x_pu": 0.01}
    }
    tree["events"] = {
        "fault_on": {"type": "connect", "target": "fault", "t": 0.002}
    }
    tree["simulation"].update(t_end=0.004, output={"period": 2.5e-5})
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    t = result.states["t"]
    voltage = _current(result, "pcc.u_C")
    assert np.max(voltage[t > 0.0035]) < 0.2 * np.min(voltage[t < 0.002])
