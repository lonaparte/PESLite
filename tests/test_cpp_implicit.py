"""Native implicit integration over the same AC, DC and coupled-link netlists."""

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from test_cpp_export import _compiler, _final_csv, _run_generated

pytestmark = pytest.mark.full


def _compare_native(params, tmp_path, *, variables=()):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    _source, native, python = _run_generated(params, tmp_path / "native", compiler, variables=variables)
    actual, expected = _final_csv(native / "states.csv"), _final_csv(python / "states.csv")
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=1e-8, atol=1e-9)
    return expected


@pytest.mark.parametrize("case,duration", [("gfl-example", 2e-4), ("buck-example", 2e-4),
                                          ("wpt-example", 2e-5)])
@pytest.mark.parametrize("form", ["device", "statewise", "nodal"])
@pytest.mark.parametrize("method", ["RK4", "trapezoidal", "radau"])
def test_native_solver_netlist_matches_python(case, duration, form, method, tmp_path):
    changes = {
        "simulation.solver.type": "fixed",
        "simulation.solver.method": method,
        "simulation.solver.circuit_model": form,
        "simulation.solver.rtol": 1e-8,
        "simulation.solver.atol": 1e-9,
        "simulation.solver.linearisations": 0,
        "simulation.t_end": duration,
        "simulation.progress.enable": 0,
        "simulation.energy_check": "off",
        "simulation.output.states": True,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
    }
    if case == "gfl-example":
        changes.update({"events.connect_vsc.t": 0.0, "events.connect_vsc.ramp": 0.005})
    params = peslite.load(EXAMPLES / (case + ".pes"), **changes)
    _compare_native(params, tmp_path)


@pytest.mark.parametrize("case", ["wpt-phase-shift-example", "wpt-scc-example"])
@pytest.mark.parametrize("form", ["device", "statewise", "nodal"])
@pytest.mark.parametrize("method", ["trapezoidal", "radau"])
def test_native_implicit_switching_commutations(case, form, method, tmp_path):
    params = peslite.load(EXAMPLES / (case + ".pes"), **{
        "simulation.solver.type": "fixed", "simulation.solver.method": method,
        "simulation.solver.circuit_model": form,
        "simulation.solver.rtol": 1e-8, "simulation.solver.atol": 1e-9,
        "simulation.solver.linearisations": 0, "simulation.t_end": 6e-5,
        "simulation.progress.enable": 0, "simulation.energy_check": "off",
        "simulation.output.states": True, "simulation.output.signals": False,
        "simulation.output.energy": False,
    })
    final = _compare_native(params, tmp_path)
    assert max(abs(value) for name, value in final.items()
               if name.startswith("charger.link.")) > 0.1


def test_native_export_keeps_python_adaptive_methods_explicit(tmp_path):
    params = peslite.load(EXAMPLES / "gfl-example.pes", **{
        "simulation.solver.type": "adaptive", "simulation.solver.method": "auto",
    })
    with pytest.raises(TypeError, match="supported solver"):
        peslite.export(params, "cpp", tmp_path / "unsupported")


def test_native_adaptive_commutation_requires_a_finite_observation_step(tmp_path):
    params = peslite.load(EXAMPLES / "wpt-phase-shift-example.pes", **{
        "simulation.solver.type": "adaptive",
        "simulation.solver.method": "DP45",
        "simulation.solver.max_step": float("inf"),
    })
    with pytest.raises(ValueError, match="max_step must be finite"):
        peslite.export(params, "cpp", tmp_path / "unbounded")


def test_native_fixed_explicit_reuses_the_stiff_mode_fallback(tmp_path):
    params = peslite.load(EXAMPLES / "boost-example.pes").replace(**{
        "units.converter.bridge.model": "switching",
        "units.converter.bridge.on_resistance": 1e-6,
        "units.converter.ctrl": {
            "type": "open_loop",
            "period": 50e-6,
            "computation": 0.0,
            "loops": {"DutyModulation": {}},
            "references": {"duty": 0.5},
        },
        "simulation.initial.states.converter.dclink.u_C": 0.0,
        "simulation.solver.type": "fixed",
        "simulation.solver.method": "RK4",
        "simulation.solver.dt": 0.125e-6,
        "simulation.t_end": 1e-4,
        "simulation.progress.enable": False,
        "simulation.energy_check": "off",
        "simulation.output.states": True,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
    })
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    source, native, python = _run_generated(params, tmp_path / "native", compiler)
    assert "integrate_stiff" in source.read_text(encoding="utf-8")
    actual = _final_csv(native / "states.csv")
    expected = _final_csv(python / "states.csv")
    np.testing.assert_allclose(
        list(actual.values()), list(expected.values()), rtol=2e-8, atol=2e-9
    )
    assert actual["converter.dclink.u_C"] > 0.0


@pytest.mark.parametrize("method", ["radau", "RK4"])
def test_native_stiff_cable_matches_python(method, tmp_path):
    params = peslite.load(EXAMPLES / "cable-gfl-example.pes", **{
        "elements.cable1.sections": 10,
        "elements.cable1.length_km": 5.0,
        "simulation.solver.type": "fixed",
        "simulation.solver.method": method,
        "simulation.solver.circuit_model": "statewise",
        "simulation.solver.rtol": 1e-8,
        "simulation.solver.atol": 1e-9,
        "simulation.t_end": 5e-4,
        "simulation.progress.enable": 0,
        "simulation.energy_check": "off",
        "simulation.output.states": True,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
    })
    _compare_native(params, tmp_path)


@pytest.mark.parametrize("method", ["trapezoidal", "radau"])
@pytest.mark.parametrize("variables", [(), ("common",), ("all",)])
def test_native_implicit_parameters_and_events(method, variables, tmp_path):
    params = peslite.load(EXAMPLES / "buck-example.pes", **{
        "simulation.solver.method": method,
        "simulation.t_end": 2e-4, "simulation.progress.enable": 0,
        "simulation.energy_check": "off", "simulation.solver.linearisations": 0,
        "simulation.output.states": True, "simulation.output.signals": False,
        "simulation.output.energy": False,
        "events.load_step": {"type": "set", "t": 1e-4, "set": {"elements.load.r": 12.0}},
    })
    _compare_native(params, tmp_path, variables=variables)
