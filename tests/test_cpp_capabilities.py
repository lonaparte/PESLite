"""Generated-backend boundaries for the third simulation-capability group."""

import pytest

import peslite
from conftest import EXAMPLES
from test_cpp_export import _compiler, _final_csv, _run_generated

pytestmark = pytest.mark.slow


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
