"""``.cir`` simulation files: a netlist of R, L, C and X device cards."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from peslite.assembly.cir import convert_text, read_cir_text, write_cir_text
from peslite.assembly.params import dump, load, to_dict
from peslite.assembly.syntax import SimulationFileError, read_simulation_file
from peslite.cli import main
from peslite.solver.model import ConfigError

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
FIXTURE = Path(__file__).parent / "fixtures" / "codesim_passive.cir"


def _cir(tmp_path: Path, example: str) -> Path:
    source = EXAMPLES / f"{example}.pes"
    target = tmp_path / f"{example}.cir"
    target.write_text(convert_text(source.read_text(encoding="utf-8"), source), encoding="utf-8")
    return target


@pytest.mark.parametrize("source", sorted(EXAMPLES.glob("*-example.pes")), ids=lambda path: path.stem)
def test_every_example_converts_to_an_equivalent_netlist(tmp_path, source):
    target = _cir(tmp_path, source.stem)
    text = target.read_text(encoding="utf-8")
    assert not any(line.startswith(("@", "base:", "units:")) for line in text.splitlines())
    assert to_dict(load(target)) == to_dict(load(source))


def test_netlist_example_is_the_grid_following_case():
    assert to_dict(load(EXAMPLES / "gfl-example2.cir")) == to_dict(load(EXAMPLES / "gfl-example.pes"))


def test_netlist_cards_describe_topology_and_devices(tmp_path):
    lines = _cir(tmp_path, "gfl-example").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "gfl-example"
    assert ".options sbase=2.0e6 vbase=690.0 freq=50.0 units=si" in lines
    assert "Cpcc pcc 0 c_pu=0.02 r_d_pu=0.5" in lines
    assert "Xgrid pcc 0 source x_pu=0.4 r_pu=0.04" in lines
    assert any(line.startswith("Xvsc pcc 0 vsc ac_filter.l_f_pu=0.2") for line in lines)
    assert any("ctrl.loops.SRFPLL.kp_pu=20.0" in line for line in lines)
    assert ".event connect_vsc type=connect target=vsc t=0.2 ramp=1.0" in lines
    assert lines[-2:] == [".tran 2.5e-05 3.0", ".end"]
    _, interface = read_simulation_file(tmp_path / "gfl-example.cir")
    assert interface == "@circuit\ngrid -- pcc\nvsc -- pcc\n"


def test_written_netlist_reads_back_to_the_same_mapping():
    tree = {
        "base": {"s_base": 1e6, "v_ll_rms": 400.0, "f0": 50.0},
        "buses": {"a": {"c": 1e-4}, "b": {"phases": 1}},
        "branches": {"line": {"bus1": "a", "bus2": "ground", "l": 1e-3, "r": 0.1}},
        "sources": {"grid": {"bus": "a", "l": 2e-3}},
        "elements": {"cap": {"type": "capacitor", "bus1": "a", "bus2": "b", "c": 1e-6}},
        "events": {"step": {"type": "set", "t": 0.1, "set": {"sources.grid.v_pu": 0.9}}},
        "simulation": {"t_end": 0.5, "solver": {"method": "RK4", "dt": 1e-5},
                       "progress": {"watch": ["a.u"]}},
        "meta": {"description": "two buses; one line = test"},
    }
    tree_back, _ = read_cir_text(write_cir_text(tree, "case"))
    assert tree_back == tree


def test_dump_writes_a_netlist_that_loads_back(tmp_path):
    p = load(EXAMPLES / "gfl-example.pes")
    dump(p, tmp_path / "case.cir")
    dump(p, tmp_path / "case.pes")
    assert to_dict(load(tmp_path / "case.cir")) == to_dict(load(tmp_path / "case.pes"))


def test_errors_report_the_netlist_line(tmp_path):
    target = tmp_path / "bad.cir"
    target.write_text("bad\n.options sbase=1e6\nXgrid a 0 source l=1e-3\nQ1 a b c\n.end\n",
                      encoding="utf-8")
    with pytest.raises(SimulationFileError) as caught:
        read_simulation_file(target)
    assert caught.value.line == 4


def test_convert_command_writes_both_formats(tmp_path):
    source = tmp_path / "case.pes"
    source.write_text((EXAMPLES / "buck-example.pes").read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["convert", str(source)]) == 0
    assert main(["convert", str(tmp_path / "case.cir"), str(tmp_path / "back.pes")]) == 0
    assert to_dict(load(tmp_path / "back.pes")) == to_dict(load(source))


def test_plain_codesim_netlist_is_imported():
    tree, interface = read_simulation_file(FIXTURE)
    assert tree["base"] == {"s_base": 1e6, "v_ll_rms": 400.0, "f0": 50.0}
    grid = tree["sources"]["grid"]
    assert grid["bus"] == "pcc" and grid["l"] == pytest.approx(0.5e-3)
    assert grid["v"] == pytest.approx(400.0 * math.sqrt(2.0 / 3.0))
    assert tree["branches"]["line"] == {"bus1": "pcc", "bus2": "load",
                                        "l": pytest.approx(2e-3), "r": pytest.approx(0.05)}
    assert tree["buses"]["pcc"]["c"] == pytest.approx(100e-6)
    assert tree["elements"]["rl"] == {"type": "load", "bus": "load", "l": pytest.approx(5e-3),
                                      "r": pytest.approx(1.5)}
    assert tree["simulation"] == {"t_end": 0.05, "output": {"period": pytest.approx(1e-4)}}
    assert interface.startswith("@circuit\n")
    assert set(load(FIXTURE).sources) == {"grid"}


def test_per_unit_netlist_values_are_converted():
    tree, _ = read_cir_text("pu case\n.options sbase=1meg vbase=1k freq=50 units=pu\n"
                            "Xg a 0 Infbus\nLg a b 0.1 R=0.01\nC1 b 0 0.02\n.end\n")
    z_base, w = 1e3 ** 2 / 1e6, 2 * math.pi * 50
    assert tree["sources"]["g"]["l"] == pytest.approx(0.1 * z_base / w)
    assert tree["sources"]["g"]["r"] == pytest.approx(0.01 * z_base)
    assert tree["buses"]["b"]["c"] == pytest.approx(0.02 / (z_base * w))


@pytest.mark.parametrize("card, message", [
    ("Xvsc pcc 0 GFL_C_F_L L=0.07", "GFL_C_F_L"),
    (".subckt foo a", "not supported"),
    ("D1 a 0 dmod", "unsupported card"),
])
def test_unsupported_cards_are_rejected(tmp_path, card, message):
    target = tmp_path / "bad.cir"
    target.write_text(f"bad\nXg a 0 Infbus\nLg a pcc 1m\n{card}\n.end\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load(target)


def test_spice_number_suffixes():
    tree, _ = read_cir_text("t\nXg a 0 Infbus\nLg a b 1m R=6k34\nC1 b 0 10uF\n.end\n")
    assert tree["sources"]["g"]["l"] == pytest.approx(1e-3)
    assert tree["sources"]["g"]["r"] == pytest.approx(6340.0)
    assert tree["buses"]["b"]["c"] == pytest.approx(10e-6)
