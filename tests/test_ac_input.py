"""Single-phase AC sources use the ordinary network and converter power port."""

import math

import pytest

import peslite
from peslite.assembly.params import ConfigError, from_dict
from peslite.components.network import ACVoltageSource


def _rectifier_tree(source_bus="rectifier.power"):
    return {
        "base": {"s_base": 1000.0, "v_ll_rms": 230.0, "f0": 50.0},
        "sources": {"line": {
            "bus": source_bus, "l": 1.0e-3, "r": 0.2, "v": 325.0,
        }},
        "units": {"rectifier": {
            "type": "dc_converter",
            "dclink": {
                "vdc_ref": 400.0,
                "capacitor": {"c": 1.0e-3},
                "source": {"type": "none"},
            },
            "bridge": {
                "type": "full_bridge", "model": "switching",
                "upper_controlled": False, "lower_controlled": False,
            },
        }},
        "simulation": {
            "t_end": 10.0e-6,
            "solver": {"dt": 0.1e-6, "linearisations": 0},
        },
    }


def test_ac_source_representation_follows_the_connected_phase_count():
    angle = lambda _t: 0.3
    single = ACVoltageSource(2.0 * math.pi * 50.0, 325.0, angle, phases=1)
    three = ACVoltageSource(2.0 * math.pi * 50.0, 325.0, angle, phases=3)

    for source in (single, three):
        source.set_outputs(1.0e-3)
        assert source.terminal.electrical_domain == "ac"
    assert isinstance(single.out.e_g, float)
    assert isinstance(three.out.e_g, complex)
    assert single.out.e_g == pytest.approx(three.out.e_g.real)


def test_single_phase_source_connects_directly_to_converter_power_port(tmp_path):
    simulation = peslite.Simulation(from_dict(_rectifier_tree()))
    source = simulation.system.sources["line"]

    assert source.emf.phases == 1
    assert source.branch.phases == 1
    assert simulation.system._port_bindings["rectifier.dclink"][0].electrical_domain == "dc"
    result = simulation.run(out_dir=tmp_path)
    assert result.t[-1] == pytest.approx(10.0e-6)


@pytest.mark.full
def test_single_phase_source_cpp_export_matches_python(tmp_path):
    from test_cpp_export import _compiler, _final_csv, _run_generated

    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = from_dict(_rectifier_tree())
    _source, cpp_out, python_out = _run_generated(params, tmp_path / "rectifier", compiler)
    assert _final_csv(cpp_out / "states.csv") == pytest.approx(
        _final_csv(python_out / "states.csv"), rel=1.0e-9, abs=1.0e-10
    )


def test_ac_source_cannot_connect_to_the_converter_dc_port():
    with pytest.raises(ConfigError, match="electrical-domain conflict"):
        peslite.Simulation(from_dict(_rectifier_tree("rectifier.dc")))


def test_dc_converter_power_port_accepts_a_dc_network():
    tree = _rectifier_tree()
    tree.pop("sources")
    tree["units"]["rectifier"]["bridge"].update(
        upper_controlled=True, lower_controlled=True
    )
    tree["units"]["rectifier"]["pwm"] = {"f_sw": 20.0e3}
    tree["units"]["rectifier"]["ctrl"] = {
        "type": "open_loop",
        "loops": {"DutyModulation": {}},
        "references": {"duty": 0.5},
    }
    tree["elements"] = {
        "battery": {
            "type": "voltage_source", "bus": "rectifier.power", "v": 200.0, "r": 1.0,
        },
    }
    simulation = peslite.Simulation(from_dict(tree))
    assert simulation.system._port_bindings["rectifier.bus.power"][0].electrical_domain is None


def test_domain_constraints_propagate_through_passive_branches():
    tree = _rectifier_tree()
    tree["sources"]["line"]["bus"] = "ac"
    tree["buses"] = {
        "ac": {"phases": 1, "c": 1.0e-3},
        "dc": {"phases": 1, "c": 1.0e-3},
    }
    tree["branches"] = {
        "tie": {"bus1": "ac", "bus2": "dc", "l": 1.0e-3},
    }
    tree["elements"] = {
        "battery": {"type": "voltage_source", "bus": "dc", "v": 200.0, "r": 1.0},
    }
    with pytest.raises(ConfigError, match="electrical-domain conflict"):
        peslite.Simulation(from_dict(tree))


def test_dc_link_rejects_ac_source_fields():
    tree = _rectifier_tree()
    source = tree["units"]["rectifier"]["dclink"]["source"]
    source.update(type="ac_voltage", v=325.0, f=50.0)
    with pytest.raises(ConfigError, match="ac_voltage|unknown key"):
        from_dict(tree)
