"""Generated-backend boundaries for the third simulation-capability group."""

import pytest

import peslite
from peslite.addons.functions import write_magport
from peslite.assembly.params import from_dict
from conftest import EXAMPLES
from test_cpp_export import _compiler, _final_csv, _run_generated

pytestmark = pytest.mark.long


def test_generated_cpp_matches_the_open_loop_buck(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / "buck-open-loop-example.pes").replace(**{
        "simulation.t_end": 2.0e-3,
        "simulation.output.states": True,
        "simulation.output.signals": False,
        "events.duty_step.t": 1.0e-3,
    })
    _source, cpp_out, python_out = _run_generated(params, tmp_path / "buck", compiler)
    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    scale = max(1.0, *(abs(value) for value in expected.values()))

    assert actual.keys() == expected.keys()
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 1.0e-8 * scale


def test_cpp_traces_mode_sequence_through_the_common_loop_path(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / "wpt-phase-shift-example.pes").replace(**{
        "simulation.t_end": 50.0e-6,
        "simulation.output.states": True,
        "simulation.output.signals": False,
        "simulation.energy_check": "off",
    })
    source, cpp_out, python_out = _run_generated(params, tmp_path, compiler)
    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    scale = max(1.0, *(abs(value) for value in expected.values()))

    assert "ModeSequence_position_0" in source.read_text(encoding="utf-8")
    assert actual.keys() == expected.keys()
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 1.0e-8 * scale


@pytest.mark.parametrize("kind", ["magport", "magcircuit"])
def test_cpp_reuses_the_constant_inductance_netlist_model(kind, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    element = {
        "type": kind,
        "windings": {"coil": {"positive": "node", "negative": "ground", "r": 0.1}},
    }
    if kind == "magport":
        element["file"] = str(write_magport(
            tmp_path / "coil.npz", winding_names=("coil",), inductance=[[0.01]]
        ))
    else:
        element["magnetic"] = {
            "branches": {"core": {"reluctance": 1.0e4}},
            "coupling": {"core": {"coil": 10.0}},
        }
    params = from_dict({
        "base": {"s_base": 1.0},
        "buses": {"node": {"phases": 1}},
        "elements": {
            "coils": element,
            "load": {"type": "resistor", "bus": "node", "r": 2.0},
        },
        "simulation": {
            "t_end": 1.0e-4,
            "energy_check": "off",
            "initial": {"states": {"coils.i_mode_1": 0.5}},
            "solver": {"dt": 1.0e-6},
            "output": {"period": 1.0e-5, "precision": 17, "states": True},
        },
    })
    _source, cpp_out, python_out = _run_generated(params, tmp_path / kind, compiler)
    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    assert actual == pytest.approx(expected, rel=1.0e-9, abs=1.0e-12)
