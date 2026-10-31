"""Parameter-sensitive feedback detection must reuse lowering, not parameter sweeps."""

from dataclasses import dataclass
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import peslite
from peslite.assembly import cpp
from peslite.assembly.cpp import _ModelGenerator, _RuntimeParameters
from peslite.solver import Bag, Model
from peslite.solver.simulation import convert_main
from test_cpp_equations import _Feedback
from test_cpp_parameters import _case, _compare_run, _compile
from conftest import EXAMPLES

pytestmark = pytest.mark.full


@dataclass(frozen=True)
class _Parameters:
    gain: float
    offset: float = 1.0
    changes: tuple = ()


def _generator(factory=_Feedback, *, gain=0.0, offset=1.0, variable=("all",), feedback=True):
    p = _Parameters(gain, offset)
    runtime = _RuntimeParameters(p, variable)
    component = factory(p.gain, p.offset)
    runtime.pair(component, factory(runtime.p.gain, runtime.p.offset))
    model = Model({"custom": component},
                  {(component, "x"): (component, "value")} if feedback else {})
    simulation = SimpleNamespace(p=p, system=SimpleNamespace(
        model=model, units={}, sources={}, named_elements={}))
    return _ModelGenerator(simulation, runtime)


@pytest.mark.parametrize("gain", [0.0, 0.5])
def test_zero_and_nonzero_defaults_identify_the_same_parameter_and_edge(gain):
    generator = _generator(gain=gain)
    diagnostic, = generator.diagnostics
    assert diagnostic["parameters"] == ("gain",)
    assert diagnostic["condition"] == "((parameters.param_gain) == 0.0)"
    assert diagnostic["dependencies"] == (("custom.inp.x", "custom.out.value"),)
    assert "offset" not in diagnostic["parameters"]
    expected = list(generator.diagnostics)
    for _ in range(3):
        generator.body()
        generator.members()
    assert generator.diagnostics == expected


def test_arithmetic_coefficients_resolve_all_participating_parameters():
    class Difference(_Feedback):
        def set_outputs(self, _t):
            self.out.value = (self.gain - self.offset) * self.inp.x + 1.0

    diagnostic, = _generator(Difference, gain=1.5).diagnostics
    assert diagnostic["parameters"] == ("gain", "offset")
    assert "parameters.param_gain" in diagnostic["condition"]
    assert "parameters.param_offset" in diagnostic["condition"]


def test_derived_coefficient_uses_existing_initializer_dependencies():
    class Derived(_Feedback):
        def __init__(self, gain, offset):
            super().__init__(gain, offset)
            self.coefficient = np.sin(gain) ** 2

        def set_outputs(self, _t):
            self.out.value = self.coefficient * self.inp.x + self.offset

    diagnostic, = _generator(Derived, gain=0.4).diagnostics
    assert diagnostic["parameters"] == ("gain",)
    assert "initial_" in diagnostic["condition"]


@pytest.mark.parametrize("gain", [0.0, 0.5])
def test_parameter_conditions_are_not_limited_to_zero_coefficients(gain):
    class Switched(_Feedback):
        def set_outputs(self, _t):
            self.out.value = self.inp.x if self.gain > 0.25 else self.offset

    diagnostic, = _generator(Switched, gain=gain).diagnostics
    assert diagnostic["parameters"] == ("gain",)
    assert "0.25" in diagnostic["condition"]
    assert diagnostic["dependencies"] == (("custom.inp.x", "custom.out.value"),)


def test_unconditional_read_prevents_a_false_conditional_edge():
    class AlwaysDependent(_Feedback):
        def set_outputs(self, _t):
            self.out.value = self.gain * self.inp.x + self.inp.x + self.offset

    assert not _generator(AlwaysDependent).diagnostics


def test_shared_branch_reads_are_not_parameter_sensitive():
    class AlwaysDependent(_Feedback):
        def set_outputs(self, _t):
            self.out.value = (0.5 * self.inp.x + self.offset if self.gain > 0.25
                              else 0.25 * self.inp.x + self.offset)

    assert not _generator(AlwaysDependent).diagnostics


def test_conditional_downstream_output_is_not_a_feedback_parameter():
    class Output(Bag):
        __slots__ = ("value", "downstream")

    class Observation(_Feedback):
        def __init__(self, gain, offset):
            super().__init__(gain, offset)
            self.out = Output()

        def set_outputs(self, _t):
            self.out.value = 0.5 * self.inp.x + self.offset
            self.out.downstream = self.gain * self.inp.x

    assert not _generator(Observation).diagnostics


def test_dag_and_additive_parameters_do_not_request_scc_analysis(monkeypatch):
    def unexpected_scc(*_args):
        pytest.fail("SCC analysis is unnecessary without conditional feedback")

    monkeypatch.setattr(cpp, "_strong_components", unexpected_scc)
    assert not _generator(feedback=False).diagnostics
    assert not _generator(gain=0.5, variable=("offset",)).diagnostics
    assert not _generator(gain=0.5, variable=()).diagnostics


@pytest.mark.parametrize("resistance", [0.0, 0.1])
def test_export_finds_the_original_resistances_without_a_name_list(resistance, tmp_path):
    p = _case("buck", "pwm_averaging").replace(**{
        "buses.output.r_d": resistance,
        "units.converter.dclink.source.r": resistance,
    })
    result = peslite.export(p, "cpp", tmp_path, variables=("all",))
    assert {path for item in result.diagnostics for path in item["parameters"]} == {
        "buses.output.r_d", "units.converter.dclink.source.r"}
    assert all(item["dependencies"] for item in result.diagnostics)
    assert tuple(path.name for path in result.files) == ("peslite.cpp",)


def test_cli_prints_the_same_export_diagnostics(tmp_path, capsys):
    assert convert_main([str(EXAMPLES / "buck-example.pes"), "all", "--out", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "buses.output.r_d" in captured.err
    assert "units.converter.dclink.source.r" in captured.err
    assert "not a timing prediction" in captured.err


@pytest.mark.parametrize("name", ["buck", "gfl", "wpt"])
@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_common_is_exactly_all_minus_the_existing_diagnostics(name, mode, tmp_path):
    p = _case(name, mode)
    all_result = peslite.export(p, "cpp", tmp_path / "all", variables=("all",))
    sensitive = {path for item in all_result.diagnostics for path in item["parameters"]}
    remaining = tuple(item.path for item in _RuntimeParameters(p, ("all",)).selected
                      if item.path not in sensitive)
    common = peslite.export(p, "cpp", tmp_path / "common", variables=("common",))
    explicit = peslite.export(p, "cpp", tmp_path / "explicit", variables=remaining)
    assert common.files[0].read_bytes() == explicit.files[0].read_bytes()
    assert {path for item in common.diagnostics if item["code"] == "common-constants"
            for path in item["parameters"]} == sensitive
    if not sensitive:
        assert common.files[0].read_bytes() == all_result.files[0].read_bytes()


@pytest.mark.parametrize("name, expected", [("buck", 2), ("gfl", 1)])
def test_common_does_not_repeat_analysis_without_exclusions(name, expected, tmp_path, monkeypatch):
    original = cpp._ModelGenerator
    calls = []

    class Counted(original):
        def __init__(self, *args):
            calls.append(self)
            super().__init__(*args)

    monkeypatch.setattr(cpp, "_ModelGenerator", Counted)
    peslite.export(_case(name, "pwm_averaging"), "cpp", tmp_path, variables=("common",))
    assert len(calls) == expected


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_common_native_keeps_ordinary_set_and_rejects_only_detected_parameters(mode, tmp_path):
    p = _case("buck", mode)
    detected = peslite.export(p, "cpp", tmp_path / "all", variables=("all",))
    sensitive = {path for item in detected.diagnostics for path in item["parameters"]}
    expected = {item.path for item in _RuntimeParameters(p, ("all",)).selected} - sensitive
    binary = _compile(p, tmp_path / "model", variables=("common",))
    listing = subprocess.run([str(binary), "--list-params"], check=True, capture_output=True, text=True)
    assert {line.split("  value=", 1)[0] for line in listing.stdout.splitlines()} == expected
    _compare_run(p, binary, {"elements.load.r": 8.0, "units.converter.filter.l_f": 1.4e-3,
                             "units.converter.dclink.source.v": 52.0}, tmp_path / "allowed")
    for index, path in enumerate(sorted(sensitive)):
        directory = tmp_path / f"rejected-{index}"
        result = subprocess.run([str(binary), "--set", f"{path}=0.05", "--out", str(directory)],
                                capture_output=True, text=True)
        assert result.returncode == 1
        assert f"parameter '{path}' is not variable" in result.stderr
        assert not directory.exists()


def test_common_preserves_nonzero_sensitive_values(tmp_path):
    p = _case("buck", "pwm_averaging").replace(**{
        "buses.output.r_d": 0.1, "units.converter.dclink.source.r": 0.1})
    binary = _compile(p, tmp_path / "model", variables=("common",))
    _compare_run(p, binary, {"elements.load.r": 8.0}, tmp_path / "run")


def test_common_cli_reports_the_constants_and_preserves_path_validation(tmp_path, capsys):
    assert convert_main([str(EXAMPLES / "buck-example.pes"), "common", "--out", str(tmp_path)]) == 0
    assert "common keeps detected structure-sensitive parameters constant" in capsys.readouterr().err
    with pytest.raises(ValueError, match="not variable"):
        peslite.export(_case("buck", "pwm_averaging"), "cpp", tmp_path / "invalid",
                       variables=("common", "missing.parameter"))
