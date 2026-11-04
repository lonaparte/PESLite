"""Deterministic native lowering of independent run-time switch clusters."""

import csv
import subprocess

import numpy as np
import pytest

import peslite
from peslite.assembly.params import from_dict
from test_cpp_export import _compiler

pytestmark = pytest.mark.full


def _interleaved(units: int, *, t_end: float = 0.004) -> dict:
    """Open-loop interleaved Buck converters with independently switched filter branches."""
    tree = {
        "base": {"s_base": 500.0, "v_ll_rms": 230.0, "f0": 50.0},
        "buses": {"output": {"phases": 1, "c": 100e-6 * units}},
        "elements": {"load": {"type": "resistor", "bus": "output", "r": 2.0 / units}},
        "units": {},
        "simulation": {
            "t_end": t_end,
            "solver": {"type": "fixed", "method": "RK4", "dt": 2.5e-6,
                       "linearisations": 0},
            "output": {"period": 5e-5, "signals": True, "precision": 17},
            "progress": {"enable": False},
            "energy_check": "off",
        },
    }
    for index in range(units):
        tree["units"][f"conv{index}"] = {
            "type": "dc_converter",
            "bus": "output",
            "filter": {"l_f": 0.5e-3, "r_f": 0.05},
            "dclink": {"vdc_ref": 48.0, "source": {"type": "voltage"}},
            "pwm": {"f_sw": 20000.0, "carrier_phase": index / units},
            "bridge": {"model": "switching"},
            "ctrl": {"type": "open_loop", "loops": {"DutyModulation": {}},
                     "references": {"duty": 0.4}},
        }
    return tree


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_independent_switches_grow_locally_and_export_deterministically(tmp_path):
    sizes = {}
    for units in (2, 4, 8):
        first = tmp_path / f"units-{units}-first"
        second = tmp_path / f"units-{units}-second"
        params = from_dict(_interleaved(units))
        peslite.export(params, "cpp", first)
        peslite.export(params, "cpp", second)
        source = (first / "peslite.cpp").read_bytes()
        assert source == (second / "peslite.cpp").read_bytes()
        sizes[units] = len(source)
    assert sizes[8] < 4 * sizes[4] < 16 * sizes[2]


def test_independent_switch_clusters_match_the_python_run(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = from_dict(_interleaved(4, t_end=0.0004))
    project = tmp_path / "native"
    peslite.export(params, "cpp", project)
    executable = project / "peslite"
    subprocess.run(
        [compiler, "-O2", "-DNDEBUG", "-std=c++17", str(project / "peslite.cpp"),
         "-o", str(executable)],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable), str(tmp_path / "cpp-output")], check=True, timeout=30)
    peslite.Simulation(params).run(out_dir=tmp_path / "python-output")
    for name in ("states.csv", "plant.csv"):
        native = _rows(tmp_path / "cpp-output" / name)
        python = _rows(tmp_path / "python-output" / name)
        assert [row["t"] for row in native] == [row["t"] for row in python]
        columns = [key for key in python[0] if key != "t"]
        actual = np.array([[float(row[key]) for key in columns] for row in native])
        expected = np.array([[float(row[key]) for key in columns] for row in python])
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9, err_msg=name)
