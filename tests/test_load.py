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


def test_every_network_component_uses_the_same_terminal_assembly(case):
    params = _two_buses(case, {})
    system = peslite.System(params)
    line = system.branches["line"]

    assert system.model.connections[(line, "u1")] == (system.buses["pcc"], "u")
    assert system.model.connections[(line, "u2")] == (system.buses["b2"], "u")
    source = system.sources["grid"]
    assert source.terminals() == (("pcc", source.branch.terminal2),)
    assert system.model.connections[(source.branch, "u1")] == (source.emf, "e_g")
    assert system.model.connections[(source.emf, "i")] == [(source.branch, "i", 1.0)]
    unit = system.units["vsc"]
    assert ("vsc.bus.dc", unit.dclink.terminal) in unit.terminals()
    assert ("vsc.bus.dc", unit.bridge.dc_terminal) in unit.terminals()
    assert ("vsc.bus.power", unit.bridge.power_terminal) in unit.terminals()
    assert ("pcc", unit.branch_f.terminal2) in unit.terminals()
    assert system.named_elements["load"].terminals() == (
        ("b2", system.named_elements["load"].branch.terminal1),
    )


def test_public_bus_may_have_no_attached_component(case):
    tree = case()
    tree["buses"]["unused"] = dict(tree["buses"]["pcc"])
    system = peslite.System(from_dict(tree))
    node = system.buses["unused"]

    assert node.terminal.current_sources == []
    assert (node, "i") not in system.model.connections


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
    unit.zoh_connections = lambda: {(unit.branch_f, "u1"): "duplicate"}

    with pytest.raises(ConfigError, match="driven both continuously"):
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
    assert signals["grid.i"] == source.branch.out.i


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
