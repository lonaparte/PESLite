"""Standalone C++ export: API, CLI layout and numerical smoke tests."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

import peslite
from conftest import EXAMPLES
from peslite.components import Bag, Element, Terminal, register_element_type
from peslite.solver.simulation import convert_main


class _ConditionalState(Bag):
    __slots__ = ("i",)


class _ConditionalInput(Bag):
    __slots__ = ("u",)


class _ConditionalOutput(Bag):
    __slots__ = ("i",)


class _ConditionalModel:
    state_names = ("i",)
    outputs_need_inputs = False

    def __init__(self, inductance, resistance_low, resistance_high, threshold):
        self.inductance = inductance
        self.resistance_low = resistance_low
        self.resistance_high = resistance_high
        self.threshold = threshold
        self.state = _ConditionalState(i=0j)
        self.inp = _ConditionalInput(u=0j)
        self.out = _ConditionalOutput(i=0j)
        self.terminal = Terminal(self, "u", "i", 1)

    def set_outputs(self, t):
        self.out.i = self.state.i

    def rhs(self, t):
        resistance = (self.resistance_high
                      if abs(self.state.i) > self.threshold else self.resistance_low)
        return ((self.inp.u - resistance * self.state.i) / self.inductance,)


@register_element_type
class _ConditionalElement(Element):
    type = "test_conditional_element"

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bus: str
        inductance: float
        resistance_low: float
        resistance_high: float
        threshold: float
        type: str = "test_conditional_element"

    def __init__(self, name, cfg, buses, p):
        self.name, self.cfg = name, cfg
        self.model = _ConditionalModel(
            cfg.inductance, cfg.resistance_low, cfg.resistance_high, cfg.threshold
        )

    def subsystems(self):
        return {self.name: self.model}

    def connections(self):
        return {}

    def terminals(self):
        return ((self.cfg.bus, self.model.terminal),)

    def signals(self):
        return {f"{self.name}.i": self.model.out.i,
                f"{self.name}.i_mag": abs(self.model.out.i)}


def _compiler() -> str | None:
    return next((path for name in ("c++", "g++", "clang++")
                 if (path := shutil.which(name)) is not None), None)


def _summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _quick(mode: str):
    solver = ("adaptive", "DP45") if mode == "averaging" else ("fixed", "rk4")
    return peslite.load(EXAMPLES / "gfl-example.pes").replace(**{
        "units.vsc.bridge.model": mode,
        "simulation.solver.type": solver[0],
        "simulation.solver.method": solver[1],
        "simulation.solver.subsystems": {},
        "simulation.t_end": 0.202,
        "simulation.output.period": 1e-3,
        "simulation.energy_check": "off",
    })


def test_python_export_api_uses_export_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = peslite.export(_quick("pwm_averaging"), "cpp")
    assert result.out_dir == Path("export/run")
    assert tuple(path.name for path in result.files) == ("peslite.cpp",)
    assert tuple(path.name for path in result.out_dir.iterdir()) == ("peslite.cpp",)
    assert "Standalone C++17" in (result.out_dir / "peslite.cpp").read_text()


def test_cli_export_preserves_mode_name_and_emits_one_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert peslite.main([str(EXAMPLES / "gfl-example.pes"), "--averaging",
                         "--export", "cpp"]) == 0
    project = tmp_path / "export/gfl-example-averaging"
    source = project / "peslite.cpp"
    assert tuple(project.iterdir()) == (source,)
    text = source.read_text()
    assert "void integrate(double& t, double target)" in text
    assert "integrate_fixed" not in text
    assert "integrate_adaptive" not in text


def test_convert_alias_and_all_variables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert convert_main([str(EXAMPLES / "gfl-example.pes"), "all"]) == 0
    source = (tmp_path / "export/gfl-example/peslite.cpp").read_text()
    assert "param_simulation_t_end" in source
    assert "param_units_vsc_ctrl_references_p_ref_pu" in source
    assert '"__PESLITE_RUNTIME_PARAMETER_1__"' in source
    assert '"__PESLITE_RUNTIME_PARAMETER_10__"' in source


def test_export_rejects_a_non_variable_parameter(tmp_path):
    with pytest.raises(ValueError, match="is not variable"):
        peslite.export(_quick("pwm_averaging"), "cpp", tmp_path,
                       variables=["units.vsc.bridge.model"])


@pytest.mark.parametrize("mode", ("switching", "pwm_averaging", "averaging"))
def test_generated_cpp_runs_all_bridge_modes(mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = _quick(mode)
    project = tmp_path / mode
    peslite.export(params, "cpp", project, name=mode)
    executable = project / "peslite"
    subprocess.run([compiler, "-O3", "-DNDEBUG", "-std=c++17",
                    str(project / "peslite.cpp"), "-o", str(executable)], check=True)
    cpp_out, python_out = project / "cpp-output", project / "python-output"
    subprocess.run([str(executable), str(cpp_out)], check=True)
    peslite.Simulation(params).run(out_dir=python_out)

    def final(path: Path) -> dict[str, float]:
        with path.open(newline="", encoding="utf-8") as stream:
            return {key: float(value) for key, value in list(csv.DictReader(stream))[-1].items()}

    expected, actual = final(python_out / "states.csv"), final(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    assert _summary(cpp_out).keys() == _summary(python_out).keys()
    scale = max(1.0, max(abs(value) for value in expected.values()))
    tolerance = 2e-5 if mode == "averaging" else 2e-8
    assert max(abs(actual[key] - expected[key]) for key in expected) <= tolerance * scale


@pytest.mark.parametrize("topology,source,reference,load", (
    ("buck", 48.0, 24.0, 6.0),
    ("buck", 48.0, 24.0, 600.0),
    ("boost", 24.0, 48.0, 12.0),
    ("buck_boost", 24.0, 24.0, 6.0),
))
@pytest.mark.parametrize("mode", ("switching", "pwm_averaging", "averaging"))
def test_generated_cpp_uses_public_ports_for_scalar_converters(
        topology, source, reference, load, mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    changes = {
        "units.converter.topology": topology,
        "units.converter.source_voltage": source,
        "units.converter.load_resistance": load,
        "units.converter.bridge.model": mode,
        "units.converter.ctrl.references.voltage": reference,
        "simulation.t_end": 0.004,
        "simulation.output.states": True,
        "simulation.output.signals": True,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    }
    if mode == "averaging":
        changes.update({
            "simulation.solver.type": "adaptive",
            "simulation.solver.method": "DP45",
            "simulation.solver.max_step": 1e-4,
        })
    params = peslite.load(EXAMPLES / "buck-example.pes").replace(**changes)
    project = tmp_path / f"{topology}-{mode}"
    peslite.export(params, "cpp", project, name=project.name)
    executable = project / "peslite"
    subprocess.run([compiler, "-O3", "-DNDEBUG", "-std=c++17",
                    str(project / "peslite.cpp"), "-o", str(executable)], check=True)
    cpp_out, python_out = project / "cpp-output", project / "python-output"
    subprocess.run([str(executable), str(cpp_out)], check=True)
    peslite.Simulation(params).run(out_dir=python_out)

    def final(path: Path) -> dict[str, float]:
        with path.open(newline="", encoding="utf-8") as stream:
            return {key: float(value) for key, value in list(csv.DictReader(stream))[-1].items()}

    expected, actual = final(python_out / "states.csv"), final(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    assert _summary(cpp_out).keys() == _summary(python_out).keys()
    scale = max(1.0, max(abs(value) for value in expected.values()))
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 2e-8 * scale


def test_generated_cpp_runs_custom_cable_equations(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / "cable-gfl-example.pes", **{
        "simulation.t_end": 0.01,
        "simulation.solver.linearisations": 0,
        "simulation.energy_check": "strict",
        "simulation.output.signals": 1,
    })
    project = tmp_path / "cable"
    peslite.export(params, "cpp", project, name="cable")
    executable = project / "peslite"
    subprocess.run([compiler, "-O3", "-DNDEBUG", "-std=c++17",
                    str(project / "peslite.cpp"), "-o", str(executable)], check=True)
    cpp_out, python_out = project / "cpp-output", project / "python-output"
    subprocess.run([str(executable), str(cpp_out)], check=True)
    peslite.Simulation(params).run(out_dir=python_out)

    def final(path: Path) -> dict[str, float]:
        with path.open(newline="", encoding="utf-8") as stream:
            return {key: float(value) for key, value in list(csv.DictReader(stream))[-1].items()}

    expected, actual = final(python_out / "states.csv"), final(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 2e-8 * max(
        1.0, max(abs(value) for value in expected.values())
    )
    expected_plant = final(python_out / "plant.csv")
    actual_plant = final(cpp_out / "plant.csv")
    assert actual_plant.keys() == expected_plant.keys()
    assert {"cable1.i1_a", "cable1.i2_a"} <= actual_plant.keys()


def test_generic_cpp_fallback_lowers_custom_equations_and_state_branches(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    tree = yaml.safe_load((EXAMPLES / "gfl-example.pes").read_text(encoding="utf-8"))
    tree["elements"] = {"conditional": {
        "type": "test_conditional_element",
        "bus": "pcc",
        "inductance": 0.01,
        "resistance_low": 10.0,
        "resistance_high": 20.0,
        "threshold": 50.0,
    }}
    tree["simulation"]["t_end"] = 0.002
    tree["simulation"]["solver"]["linearisations"] = 0
    tree["simulation"]["energy_check"] = "off"
    tree["simulation"].setdefault("output", {})["signals"] = 1
    case = tmp_path / "conditional.pes"
    case.write_text(yaml.safe_dump(tree, sort_keys=False), encoding="utf-8")
    params = peslite.load(case)
    project = tmp_path / "conditional"
    peslite.export(params, "cpp", project, name="conditional")
    source = (project / "peslite.cpp").read_text(encoding="utf-8")
    assert "std::abs" in source and "?" in source
    assert "20.0" in source and "10.0" in source

    executable = project / "peslite"
    subprocess.run([compiler, "-O3", "-DNDEBUG", "-std=c++17",
                    str(project / "peslite.cpp"), "-o", str(executable)], check=True)
    cpp_out, python_out = project / "cpp-output", project / "python-output"
    subprocess.run([str(executable), str(cpp_out)], check=True)
    peslite.Simulation(params).run(out_dir=python_out)

    def final(path: Path) -> dict[str, float]:
        with path.open(newline="", encoding="utf-8") as stream:
            return {key: float(value) for key, value in list(csv.DictReader(stream))[-1].items()}

    expected, actual = final(python_out / "states.csv"), final(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 2e-8 * max(
        1.0, max(abs(value) for value in expected.values())
    )
    expected_plant = final(python_out / "plant.csv")
    actual_plant = final(cpp_out / "plant.csv")
    assert actual_plant.keys() == expected_plant.keys()
    assert "conditional.i_mag" in actual_plant


def test_cpp_export_rejects_an_already_run_simulation(tmp_path):
    simulation = peslite.Simulation(_quick("pwm_averaging"))
    simulation.run(out_dir=tmp_path / "output")
    with pytest.raises(RuntimeError, match="already-run"):
        simulation.export("cpp", tmp_path / "export")


def test_generated_cpp_runtime_parameters(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = _quick("pwm_averaging")
    project = tmp_path / "runtime-parameters"
    peslite.export(params, "cpp", project, variables=[
        "simulation.t_end", "units.vsc.ctrl.references.p_ref_pu",
    ])
    executable = project / "peslite"
    subprocess.run([compiler, "-O3", "-DNDEBUG", "-std=c++17",
                    str(project / "peslite.cpp"), "-o", str(executable)], check=True)

    listed = subprocess.run([executable, "--list-params"], check=True,
                            capture_output=True, text=True).stdout
    assert "simulation.t_end" in listed
    assert "units.vsc.ctrl.references.p_ref_pu" in listed

    output = project / "changed-output"
    subprocess.run([executable,
                    "--set", "simulation.t_end=0.204",
                    "--set", "units.vsc.ctrl.references.p_ref_pu=0.25",
                    "--out", output], check=True)
    resolved = peslite.load(output / "simulation.pes")
    assert resolved.simulation.t_end == pytest.approx(0.204)
    assert resolved.unit("vsc").ctrl.references.p_ref_pu == pytest.approx(0.25)

    config = project / "runtime.pes"
    peslite.dump(params.replace(**{
        "simulation.t_end": 0.203,
        "units.vsc.ctrl.references.p_ref_pu": 0.4,
    }), config)
    configured_output = project / "configured-output"
    configured = subprocess.run([
        executable, "--config", config,
        "--set", "simulation.t_end=0.205",
        "--out", configured_output,
    ], capture_output=True, text=True)
    assert configured.returncode == 0
    assert "warning:" in configured.stderr
    configured_params = peslite.load(configured_output / "simulation.pes")
    assert configured_params.simulation.t_end == pytest.approx(0.205)
    assert configured_params.unit("vsc").ctrl.references.p_ref_pu == pytest.approx(0.4)

    rejected = subprocess.run(
        [executable, "--set", "units.vsc.ctrl.references.q_ref_pu=0.2"],
        capture_output=True, text=True,
    )
    assert rejected.returncode == 1
    assert "parameter 'units.vsc.ctrl.references.q_ref_pu' is not variable" in rejected.stderr
