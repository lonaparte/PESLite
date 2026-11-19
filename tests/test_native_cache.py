"""Exported C++ runs reuse content-identical compiled simulators."""

from __future__ import annotations

import shutil

import pytest

import peslite
from peslite.assembly.params import from_dict
from peslite.solver import simulation as simulation_module
from peslite.solver.simulation import _export_main


pytestmark = [pytest.mark.long, pytest.mark.serial]


def _case():
    return from_dict({
        "base": {"s_base": 1.0},
        "buses": {"node": {"phases": 1, "c": 1e-3}},
        "elements": {"load": {"type": "resistor", "bus": "node", "r": 2.0}},
        "simulation": {
            "t_end": 1e-4,
            "initial": {"states": {"node.u_C": 1.0}},
            "energy_check": "off", "progress": {"enable": False},
            "solver": {"type": "fixed", "method": "rk4", "dt": 1e-6},
            "output": {"period": 1e-5, "states": True, "energy": False},
        },
    })


def test_export_run_reuses_the_generated_program(tmp_path, monkeypatch):
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is required")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(simulation_module, "_RESULTS", tmp_path / "output")
    case = tmp_path / "case.pes"
    peslite.dump(_case(), case)

    first = _export_main([str(case), "--run", "--out", "first", "--json"])
    second = _export_main([str(case), "--run", "--out", "second", "--json"])
    changed = _export_main([
        str(case), "--run", "--out", "changed", "--json",
        "--set", "elements.load.r=3.0",
    ])

    assert first.code == second.code == changed.code == 0
    assert not first.payload["export"]["run"]["build_cache_hit"]
    assert second.payload["export"]["run"]["build_cache_hit"]
    assert not changed.payload["export"]["run"]["build_cache_hit"]
    assert first.payload["simulation"]["status"] == "completed"
    assert first.payload["simulation"]["t_stop"] == pytest.approx(1e-4)
    assert (tmp_path / "first" / "peslite.cpp").is_file()
    assert (tmp_path / "output" / "case" / "summary.json").is_file()
