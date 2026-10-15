"""Composite WPT configuration, assembly and compensated-link behavior."""

import copy

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import ConfigError, from_dict, read_tree, to_dict
from peslite.components import LCCLink, SSLink


def _tree():
    return read_tree(EXAMPLES / "wpt-example.pes")


def _short(tree=None):
    tree = copy.deepcopy(_tree() if tree is None else tree)
    tree["simulation"]["t_end"] = 20.0e-6
    tree["simulation"]["solver"]["dt"] = 50.0e-9
    tree["simulation"]["output"].update({"states": 0, "signals": 0, "energy": 0})
    return from_dict(tree)


def test_wpt_configuration_stays_composite_and_runtime_units_are_flat():
    params = from_dict(_tree())

    assert params.unit("charger").type == "wpt"
    assert tuple(params.runtime_units) == ("charger.primary", "charger.secondary")
    assert params.unit_groups == {
        "charger": ("charger.primary", "charger.secondary"),
    }
    assert params.unit("charger.primary").type == "dc_converter"
    assert params.unit("charger.primary").ctrl.period == pytest.approx(1.0e-6)
    assert params.unit("charger.secondary").ctrl.period == pytest.approx(1.25e-6)


def test_wpt_assembly_flattens_only_once_and_shares_one_link():
    simulation = peslite.Simulation(_short())
    subsystems = dict(zip(simulation.system.model.names, simulation.system.model.subsystems))

    assert tuple(simulation.units) == ("charger.primary", "charger.secondary")
    assert isinstance(subsystems["charger.link"], SSLink)
    assert "charger" not in simulation.units
    assert simulation.units["charger.primary"].link is simulation.units["charger.secondary"].link
    assert simulation.system.unit_groups["charger"] == tuple(simulation.units)
    assert not simulation.energy_problems


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


def test_composite_switching_event_is_inherited_by_both_children():
    tree = _tree()
    tree["events"] = {
        "start": {"type": "connect", "target": "charger", "t": 5.0e-6},
    }
    params = from_dict(tree)

    for unit in params.runtime_units.values():
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
    assert {unit.type for unit in rebuilt.runtime_units.values()} == {"dc_converter"}


def test_cpp_export_rejects_wpt_before_writing_backend_files(tmp_path):
    with pytest.raises(TypeError, match="does not yet lower composite WPT units"):
        peslite.export(_short(), "cpp", tmp_path)
    assert not list(tmp_path.iterdir())
