"""Reference issue #2 simulation-file acceptance under the current example naming rules."""

import copy
import dataclasses
import io
import math
import re
import shutil
import warnings
from contextlib import redirect_stdout
from dataclasses import fields, is_dataclass
from typing import get_args, get_type_hints

import numpy as np
import pytest
import yaml

import peslite
from conftest import EXAMPLES
from peslite.assembly.events import EVENT_TYPES
from peslite.assembly.params import ConfigError, Params, dumps, from_dict, read_tree, to_dict
from peslite.components import ELEMENT_TYPES
from peslite.control import LOOP_TYPES


BUNDLED = sorted(path.name for path in EXAMPLES.glob("*-example.pes"))


@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_examples_declare_the_current_interface_version(name):
    first_line = (EXAMPLES / name).read_text(encoding="utf-8").splitlines()[0]
    assert first_line == f"# PESLite {peslite.__version__}"


def _leaves(tree, path=()):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _leaves(value, path + (key,))
    else:
        yield path


def _without(tree, path):
    result = copy.deepcopy(tree)
    node = result
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return result


@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_examples_restate_no_defaults(name):
    tree = read_tree(EXAMPLES / name)
    reference = to_dict(from_dict(tree))
    restated = []
    for path in _leaves(tree):
        try:
            reduced = from_dict(_without(tree, path))
        except ConfigError:
            continue
        if to_dict(reduced) == reference:
            if len(path) == 3 and path[0] == "units" and path[2] == "type":
                continue  # keep explicit unit discriminators in published examples
            restated.append(".".join(map(str, path)))
    assert restated == []


def _options(cls, found=None):
    found = set() if found is None else found
    hints = get_type_hints(cls)
    for item in fields(cls):
        if not item.init or item.name.startswith("_"):
            continue
        found.add((cls, item.name))
        types = (hints[item.name], *get_args(hints[item.name]),
                 *(arg for value in get_args(hints[item.name]) for arg in get_args(value)))
        for value in types:
            if is_dataclass(value) and isinstance(value, type) and not any(old is value for old, _ in found):
                _options(value, found)
    return found


def test_annotated_gfl_example_lists_every_option():
    text = (EXAMPLES / "gfl-example.pes").read_text(encoding="utf-8")
    vsc_params = type(from_dict(read_tree(EXAMPLES / "gfl-example.pes")).unit()._unit_fields)
    options = _options(Params) | _options(vsc_params)
    for kind in ("srf_pll", "dq_current_pi", "dc_voltage_pi"):
        options |= _options(LOOP_TYPES[kind].Params)
    for kind in ("connect", "disconnect", "set"):
        options |= _options(EVENT_TYPES[kind].Params)
    options |= _options(ELEMENT_TYPES["load"].Params)

    missing = []
    for cls, name in options:
        spellings = {name, f"{name}_pu"} | {
            alias for alias, target in getattr(cls, "_input_aliases", {}).items()
            if target in (name, f"{name}_pu")
        }
        if not any(re.search(rf"(?<![\w.]){re.escape(spelling)}\s*:", text)
                   for spelling in spellings):
            missing.append(f"{cls.__qualname__}.{name}")
    assert sorted(missing) == []


def test_legacy_vsc_without_unit_type_keeps_the_vsc_default():
    tree = read_tree(EXAMPLES / "gfl-example.pes")
    tree["units"]["vsc"].pop("type")

    unit = from_dict(tree).unit("vsc")

    assert type(unit) is peslite.assembly.UnitParams
    assert type(unit.ctrl) is peslite.assembly.ControlParams
    assert unit.type == "vsc"


def _bus_start(params, bus, out):
    result = peslite.Simulation(params).run(out_dir=out)
    voltage = complex(result.states[f"{bus}.u_C.re"][0],
                      result.states[f"{bus}.u_C.im"][0])
    return voltage / params.base.v_phase_peak


def test_initial_states_default_to_source_and_rated_dc(gfl, tmp_path):
    params = gfl()
    assert params.simulation.initial.states == {}
    assert _bus_start(params, "pcc", tmp_path / "default-bus") == pytest.approx(1.0)
    result = peslite.Simulation(params).run(out_dir=tmp_path / "default-unit")
    assert result.states["vsc.dclink.u_C"][0] == 1500.0
    assert _bus_start(gfl(**{"sources.grid.v_pu": 1.05}), "pcc",
                      tmp_path / "raised-source") == pytest.approx(1.05)


def test_bus_without_own_source_keeps_default_and_explicit_values_win(case, tmp_path):
    tree = case()
    tree["buses"].update(
        b2={"c_pu": 0.02, "r_d_pu": 0.5},
        b3={"c_pu": 0.02, "r_d_pu": 0.5},
    )
    tree["branches"] = {
        "l2": {"bus1": "pcc", "bus2": "b2", "x_pu": 0.1},
        "l3": {"bus1": "b2", "bus2": "b3", "x_pu": 0.1},
    }
    tree["sources"]["grid"]["v_pu"] = 1.05
    tree["sources"]["g2"] = {"bus": "b2", "x_pu": 0.4, "v_pu": 0.95}
    params = from_dict(tree)
    assert [abs(_bus_start(params, bus, tmp_path / bus)) for bus in ("pcc", "b2", "b3")] == pytest.approx(
        [1.05, 0.95, 1.0]
    )
    explicit = params.replace(**{"simulation.initial.states.pcc.u_C": [500.0, 0.0]})
    assert (_bus_start(explicit, "pcc", tmp_path / "explicit") * params.base.v_phase_peak
            == pytest.approx(500.0))


def test_dataclass_rebuild_keeps_defaults_following_the_system_base(gfl):
    params = gfl()
    unit = dataclasses.replace(params.unit("vsc"), ctrl=dataclasses.replace(
        params.unit("vsc").ctrl, computation=2e-6))
    rebuilt = dataclasses.replace(params, units={"vsc": unit}).replace(**{"base.s_base": 1.0e6})
    assert rebuilt.unit("vsc").base.s_base == 1.0e6


def test_unused_measurement_window_is_warned_about(gfl):
    with pytest.warns(UserWarning, match="window has no effect"):
        gfl(**{"units.vsc.meas.window": 25e-6})


def test_resolved_cli_prints_a_loadable_complete_file():
    stream = io.StringIO()
    with redirect_stdout(stream):
        assert peslite.main([
            "gfl-example", "--resolved",
            "--set", "simulation.t_end=0.5",
            "--set", "units.vsc.pwm.f_sw=10000",
        ]) == 0
    tree = yaml.safe_load(stream.getvalue())
    assert tree["simulation"]["t_end"] == 0.5
    assert tree["units"]["vsc"]["ctrl"]["period"] == pytest.approx(1e-4)
    assert tree["units"]["vsc"]["ctrl"]["computation"] == pytest.approx(1e-6)
    from_dict(tree)


def test_saved_simulation_file_repeats_the_run(gfl, tmp_path):
    params = gfl(**{"simulation.t_end": 0.002})
    first_dir = tmp_path / "first"
    first = peslite.Simulation(params).run(out_dir=first_dir)
    text = (first_dir / "simulation.pes").read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = peslite.load(first_dir / "simulation.pes")
    assert dumps(loaded) == text
    second = peslite.Simulation(loaded).run(out_dir=tmp_path / "second")
    for name in first.states:
        assert np.array_equal(first.states[name], second.states[name], equal_nan=True), name


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_yaml_suffixes_remain_supported(tmp_path, suffix):
    copied = tmp_path / f"gfl{suffix}"
    shutil.copy(EXAMPLES / "gfl-example.pes", copied)
    assert to_dict(peslite.load(copied)) == to_dict(peslite.load(EXAMPLES / "gfl-example.pes"))
