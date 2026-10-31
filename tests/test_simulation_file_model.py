"""Parameter layout, switches, derived defaults and resolved files."""

import math
import re

import pytest
import yaml

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import ConfigError, dumps, from_dict, read_tree, to_dict


@pytest.mark.parametrize("filename", [
    "wpt-example.pes", "wpt-ss-example.pes", "wpt-lcc-example.pes", "wpt-load-example.pes",
])
def test_buses_section_may_be_omitted_when_no_public_bus_is_used(filename):
    tree = read_tree(EXAMPLES / filename)
    assert "buses" not in tree
    params = from_dict(tree)
    explicit = from_dict({**tree, "buses": {}})
    assert params.buses == {}
    assert params.buses is not explicit.buses
    assert to_dict(params) == to_dict(explicit)
    sim = peslite.Simulation(params.replace(**{
        "simulation.t_end": 2e-6,
        "simulation.solver.linearisations": 0,
        "simulation.progress.enable": False,
        "simulation.output.states": False,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
    }))
    assert not sim.system.buses
    result = sim.run(out_dir=None)
    assert result.summary["t_stop"] == pytest.approx(2e-6)


def test_scalar_si_network_needs_neither_converters_nor_ac_base_values(tmp_path):
    params = from_dict({
        "base": {"s_base": 500.0},
        "buses": {"node": {"c": 1.0e-3, "phases": 1}},
        "elements": {"load": {"type": "resistor", "bus": "node", "r": 10.0}},
        "simulation": {
            "t_end": 0.01,
            "solver": {"dt": 1.0e-5},
            "initial": {"states": {"node.u_C": 10.0}},
            "output": {"period": 1.0e-3},
            "energy_check": "strict",
        },
    })
    assert params.units == {}
    assert params.base.v_ll_rms is None and params.base.f0 is None
    assert from_dict(to_dict(params)) == params

    result = peslite.Simulation(params).run(out_dir=tmp_path)
    assert result.states["node.u_C"][-1] == pytest.approx(10.0 / math.e, rel=2e-8)


def test_missing_optional_base_values_are_reported_only_when_used():
    tree = {
        "base": {"s_base": 500.0},
        "buses": {"node": {"c": 1.0e-3, "phases": 1}},
        "simulation": {"t_end": 0.01},
    }
    with pytest.raises(ConfigError, match=r"per-unit capacitance.*corresponding base"):
        from_dict({**tree, "buses": {"node": {"c_pu": 0.1, "phases": 1}}})
    with pytest.raises(ConfigError, match=r"three-phase bus requires base\.v_ll_rms"):
        from_dict({**tree, "buses": {"node": {"c": 1.0e-3, "phases": 3}}})

    vsc = read_tree(EXAMPLES / "gfl-example.pes")
    vsc["base"] = {"s_base": vsc["base"]["s_base"]}
    with pytest.raises(ConfigError, match=r"per-unit .*capacitance.*corresponding base|a VSC requires"):
        from_dict(vsc)


@pytest.mark.parametrize("filename", ["gfl-example.pes", "buck-example.pes", "wpt-load-example.pes"])
def test_shared_unit_schema_round_trips_without_persisting_cached_derived_fields(filename):
    params = peslite.load(EXAMPLES / filename)
    from peslite.assembly.params import _unit_items
    for _name, unit in _unit_items(params.units):
        assert unit.dc_base is unit.dc_base
    written = to_dict(params)
    assert to_dict(from_dict(written)) == written
    if filename.startswith("wpt"):
        assert set(written["units"]["charger"]) == {"type", "s_base", "primary", "secondary", "link"}


@pytest.mark.parametrize("name", ["pwm", "dclink"])
def test_required_common_unit_fields_stay_required_under_the_single_schema(name):
    tree = read_tree(EXAMPLES / "buck-example.pes")
    tree["units"]["converter"].pop(name)
    with pytest.raises(ConfigError, match=name):
        from_dict(tree)


def _tree():
    return read_tree(EXAMPLES / "gfl-example.pes")


def test_initial_and_output_belong_to_simulation():
    p = from_dict(_tree())
    assert p.simulation.initial.t == 0.0
    assert p.simulation.output.period == 5e-4
    assert not hasattr(p, "initial") and not hasattr(p, "output")

    tree = _tree()
    tree["initial"] = {"t": 0.0, "states": {}}
    with pytest.raises(ConfigError, match=r"unknown key.*initial"):
        from_dict(tree)


def test_reserved_controller_loop_name_is_rejected():
    tree = _tree()
    loops = tree["units"]["vsc"]["ctrl"]["loops"]
    loops["startup"] = {**loops.pop("DCVoltageControl"), "type": "dc_voltage_control"}
    with pytest.raises(ConfigError, match=r"loops\.startup: reserved loop name"):
        from_dict(tree)

    tree = _tree()
    tree["units"]["vsc"]["measurement"] = {"average": "instantaneous"}
    with pytest.raises(ConfigError, match=r"units\.vsc: unknown key.*measurement"):
        from_dict(tree)

    tree = _tree()
    tree["units"]["vsc"]["delay"] = {"steps": 1}
    with pytest.raises(ConfigError, match=r"units\.vsc: unknown key.*delay"):
        from_dict(tree)

    tree = _tree()
    tree["units"]["vsc"]["pwm"]["delay"] = {"steps": 1}
    with pytest.raises(ConfigError, match=r"units\.vsc\.pwm: unknown key.*delay"):
        from_dict(tree)


def test_function_first_state_names_are_rejected(tmp_path):
    params = peslite.load(
        EXAMPLES / "gfl-example.pes",
        **{"simulation.initial.states.plant.pcc.u_C": [500.0, 0.0]},
    )
    with pytest.raises(ConfigError, match=r"unknown state 'plant\.pcc\.u_C'"):
        peslite.Simulation(params).run(out_dir=tmp_path)


def test_nested_electrical_values_follow_si_and_pu_spelling():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    amperes = 1.15 * p.unit("vsc").base.i_phase_peak
    q = p.replace(**{"units.vsc.protection.overcurrent.limit": amperes})
    assert q.unit("vsc").protection.overcurrent.limit_pu == pytest.approx(1.15)
    with pytest.raises(ConfigError, match="both actual and pu values"):
        p.replace(**{"units.vsc.protection.overcurrent.limit": amperes,
                     "units.vsc.protection.overcurrent.limit_pu": 1.15})

    # A scalar unit's protection compares against its DC base, in the file and in set events,
    # and needs no AC base for SI limits.
    buck = peslite.load(EXAMPLES / "buck-example.pes")
    limits = {"units.converter.protection.overcurrent.limit": 20.0,
              "units.converter.protection.overvoltage.limit": 30.0,
              "events.tighten.type": "set", "events.tighten.t": 1e-3,
              "events.tighten.set.units.converter.protection.undervoltage.limit": 12.0}
    for q in (buck.replace(**limits),
              buck.replace(**limits, **{"base.v_ll_rms": None, "base.f0": None})):
        converter = q.unit("converter")
        dc = converter.dc_base
        assert converter.protection.overcurrent.limit_pu == pytest.approx(20.0 / dc.i)
        assert converter.protection.overvoltage.limit_pu == pytest.approx(30.0 / dc.v)
        written = q.changes[0].values["units.converter.protection.undervoltage.limit_pu"]
        assert written == pytest.approx(12.0 / dc.v)


def test_switched_sections_require_settings_and_write_zero_or_one():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    written = yaml.safe_load(dumps(p))
    protection = written["units"]["vsc"]["protection"]
    assert protection["overcurrent"] == {"enable": 1, "limit_pu": 1.15}
    assert protection["frequency"] == {"enable": 0, "limit": None}
    assert written["simulation"]["stop_on_trip"] == 1
    assert written["simulation"]["output"]["signals"] == 0
    assert not re.search(r":\s*(true|false)\b", dumps(p))

    with pytest.raises(ConfigError, match=r"frequency\.limit: required when enable is 1"):
        p.replace(**{"units.vsc.protection.frequency.enable": 1})
    with pytest.raises(ConfigError, match=r"expected 1 \(on\) or 0 \(off\)"):
        p.replace(**{"simulation.output.signals": 2})


def test_dependent_defaults_follow_replace_and_stay_fixed_when_explicit():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    u = p.replace(**{"units.vsc.bridge.model": "switching",
                     "units.vsc.pwm.f_sw": 10_000.0,
                     "base.f0": 60.0}).unit("vsc")
    assert u.ctrl.period == pytest.approx(1e-4)
    assert all(loop.period == pytest.approx(1e-4) for loop in u.ctrl.loops.values())
    assert u.ctrl.references.omega == pytest.approx(120 * math.pi)
    assert u.meas.period is None and u.meas.window is None and u.meas.average == []

    fixed = p.replace(**{"units.vsc.bridge.model": "switching",
                         "units.vsc.ctrl.period": 5e-5})
    fixed = fixed.replace(**{"units.vsc.pwm.f_sw": 10_000.0})
    assert fixed.unit("vsc").ctrl.period == 5e-5

    rebased = p.replace(**{"base.s_base": 1e6})
    assert rebased.unit("vsc").base.s_base == 1e6
    revolted = p.replace(**{"base.v_ll_rms": 400.0})
    assert revolted.sources["grid"].v == pytest.approx(revolted.base.v_phase_peak)
    explicit = p.replace(**{"sources.grid.v": 500.0}).replace(**{"base.v_ll_rms": 400.0})
    assert explicit.sources["grid"].v == 500.0
    buck = peslite.load(EXAMPLES / "buck-example.pes")
    assert buck.replace(**{"units.converter.dclink.vdc_ref": 60.0}).unit(
        "converter").dclink.source.v == 60.0
    assert buck.replace(**{"units.converter.dclink.source.v": 50.0}).replace(
        **{"units.converter.dclink.vdc_ref": 60.0}).unit("converter").dclink.source.v == 50.0
    averaged = p.replace(**{"units.vsc.bridge.model": "pwm_averaging",
                            "units.vsc.meas.average": ["u_g", "i_c"],
                            "units.vsc.pwm.f_sw": 10_000.0})
    written = yaml.safe_load(dumps(averaged))
    assert written["units"]["vsc"]["meas"]["period"] == pytest.approx(1e-4)
    assert written["units"]["vsc"]["meas"]["window"] == pytest.approx(1e-4)
    assert written["units"]["vsc"]["meas"]["average"] == ["u_g", "i_c"]
    assert "u_dc" not in written["units"]["vsc"]["meas"]


def test_to_dict_preserves_following_defaults_but_dumps_resolves_them():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    compact = to_dict(p)
    complete = yaml.safe_load(dumps(p))
    assert compact["units"]["vsc"]["ctrl"]["period"] is None
    assert compact["units"]["vsc"]["ctrl"]["references"]["omega"] is None
    assert complete["units"]["vsc"]["ctrl"]["period"] == pytest.approx(5e-5)
    assert complete["units"]["vsc"]["ctrl"]["computation"] == pytest.approx(1e-6)
    assert complete["units"]["vsc"]["ctrl"]["references"]["omega"] == pytest.approx(100 * math.pi)
    assert complete["units"]["vsc"]["meas"]["period"] == pytest.approx(5e-5)
    assert complete["units"]["vsc"]["s_base"] == 2e6
    assert dumps(from_dict(complete)) == dumps(p)


def test_class_named_control_loops_round_trip_without_a_type_field():
    p = peslite.load(EXAMPLES / "gfl-example.pes")
    loops = to_dict(p)["units"]["vsc"]["ctrl"]["loops"]

    assert tuple(loops) == ("SRFPLL", "CurrentLoop", "DCVoltageControl")
    assert all("type" not in parameters for parameters in loops.values())
    assert from_dict(to_dict(p)) == p


def test_a_null_class_named_control_loop_is_an_empty_parameter_mapping():
    tree = read_tree(EXAMPLES / "gfm-psc-example.pes")
    tree["units"]["vsc"]["ctrl"]["loops"]["VirtualImpedance"] = yaml.safe_load(
        "VirtualImpedance:\n"
    )["VirtualImpedance"]

    p = from_dict(tree)

    assert p.unit().ctrl.loops["VirtualImpedance"].type == "virtual_impedance"


def test_a_class_named_control_loop_rejects_a_conflicting_type():
    tree = _tree()
    tree["units"]["vsc"]["ctrl"]["loops"]["DCVoltageControl"]["type"] = "srf_pll"

    with pytest.raises(ConfigError, match="conflicts with loop class"):
        from_dict(tree)


def test_a_loop_class_uses_a_class_qualified_name_without_type():
    tree = _tree()
    loops = tree["units"]["vsc"]["ctrl"]["loops"]
    loops["SRFPLL[grid]"] = loops.pop("SRFPLL")

    p = from_dict(tree)
    written = to_dict(p)["units"]["vsc"]["ctrl"]["loops"]
    graph = peslite.Simulation(p).unit().ctrl.graph

    assert "SRFPLL[grid]" in p.unit().ctrl.loops
    assert "type" not in written["SRFPLL[grid]"]
    assert "SRFPLL[grid]" in graph.nodes
    assert graph.outputs["theta"] == "SRFPLL[grid].theta"


def test_resolved_cli_and_meta_contract(capsys):
    assert peslite.main(["gfl-example", "--resolved", "--set", "units.vsc.pwm.f_sw=10000"]) == 0
    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["units"]["vsc"]["ctrl"]["period"] == pytest.approx(1e-4)
    from_dict(resolved)

    assert peslite.main(["gfl-example", "--switching", "--resolved",
                         "--set", "units.vsc.pwm.f_sw=10000"]) == 0
    sampled = yaml.safe_load(capsys.readouterr().out)
    assert sampled["units"]["vsc"]["ctrl"]["period"] == pytest.approx(1e-4)

    tree = _tree()
    tree["meta"] = {"title": "A case", "description": "Text only."}
    assert from_dict(tree).meta.title == "A case"
    tree["meta"] = {"custom": 3}
    with pytest.raises(ConfigError, match="meta holds only a title and a description"):
        from_dict(tree)


def test_dump_and_initial_file_use_the_nested_simulation_section(tmp_path):
    p = peslite.load(EXAMPLES / "gfl-example.pes").replace(
        **{"simulation.initial.t": 0.001,
           "simulation.initial.states.pcc.u_C": [500.0, 0.0]})
    path = tmp_path / "restart.pes"
    peslite.dump(p, path)

    restarted = peslite.load(EXAMPLES / "gfl-example.pes", initial=path)
    assert restarted.simulation.initial.t == 0.001
    assert restarted.simulation.initial.states["pcc.u_C"] == [500.0, 0.0]
