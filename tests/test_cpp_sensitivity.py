"""Parameter-sensitive feedback detection must reuse lowering, not parameter sweeps."""

from dataclasses import dataclass
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import peslite
from peslite.assembly import cpp
from peslite.solver import model as kernel
from peslite.assembly.cpp import _ModelGenerator, _RuntimeParameters
from peslite.components.network import Element, _register_assembled_type
from peslite.solver import Bag, Model
from test_cpp_equations import _Feedback
from test_cpp_parameters import _case, _compare_run, _compile, _table

pytestmark = pytest.mark.full


@dataclass(frozen=True)
class _Parameters:
    gain: float
    offset: float = 1.0
    changes: tuple = ()


@_register_assembled_type
class _FeedbackElement(Element):
    type = "test_sensitive_feedback"

    @dataclass(frozen=True, kw_only=True)
    class Params:
        gain: float = 0.0
        type: str = "test_sensitive_feedback"

    def __init__(self, name, cfg, buses, p):
        self.name = name
        self.model = _Feedback(cfg.gain, 1.0)

    def subsystems(self):
        return {self.name: self.model}

    def connections(self):
        return {(self.model, "x"): (self.model, "value")}

    def port_bindings(self):
        return ()

    def signals(self):
        return {f"{self.name}.value": self.model.output.value}


def _feedback_case(mode="pwm_averaging", gain=0.0):
    return _case("buck", mode).replace(**{
        "elements.feedback": {"type": _FeedbackElement.type, "gain": gain},
        "simulation.output.signals": True,
        "simulation.output.precision": 17,
    })


def _generator(factory=_Feedback, *, gain=0.0, offset=1.0, variable=("all",), feedback=True):
    p = _Parameters(gain, offset)
    runtime = _RuntimeParameters(p, variable)
    component = factory(p.gain, p.offset)
    runtime.pair(component, factory(runtime.p.gain, runtime.p.offset))
    model = Model({"custom": component},
                  {(component, "x"): (component, "value")} if feedback else {})
    simulation = SimpleNamespace(p=p, system=SimpleNamespace(
        model=model, units={}, sources={}, named_elements={}, event_subsystems=(),
        circuits=()))
    return _ModelGenerator(simulation, runtime)


@pytest.mark.parametrize("gain", [0.0, 0.5])
def test_zero_and_nonzero_defaults_identify_the_same_parameter_and_edge(gain):
    generator = _generator(gain=gain)
    diagnostic, = generator.diagnostics
    assert diagnostic["parameters"] == ("gain",)
    assert diagnostic["condition"] == "((parameters.param_gain) == 0.0)"
    assert diagnostic["dependencies"] == (("custom.input.x", "custom.output.value"),)
    assert "offset" not in diagnostic["parameters"]
    expected = list(generator.diagnostics)
    for _ in range(3):
        generator.body()
        generator.members()
    assert generator.diagnostics == expected


def test_arithmetic_coefficients_resolve_all_participating_parameters():
    class Difference(_Feedback):
        def set_outputs(self, _t):
            self.output.value = (self.gain - self.offset) * self.input.x + 1.0

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
            self.output.value = self.coefficient * self.input.x + self.offset

    diagnostic, = _generator(Derived, gain=0.4).diagnostics
    assert diagnostic["parameters"] == ("gain",)
    assert "initial_" in diagnostic["condition"]


@pytest.mark.parametrize("gain", [0.0, 0.5])
def test_parameter_conditions_are_not_limited_to_zero_coefficients(gain):
    class Switched(_Feedback):
        def set_outputs(self, _t):
            self.output.value = self.input.x if self.gain > 0.25 else self.offset

    diagnostic, = _generator(Switched, gain=gain).diagnostics
    assert diagnostic["parameters"] == ("gain",)
    assert "0.25" in diagnostic["condition"]
    assert diagnostic["dependencies"] == (("custom.input.x", "custom.output.value"),)


def test_unconditional_read_prevents_a_false_conditional_edge():
    class AlwaysDependent(_Feedback):
        def set_outputs(self, _t):
            self.output.value = self.gain * self.input.x + self.input.x + self.offset

    assert not _generator(AlwaysDependent).diagnostics


def test_shared_branch_reads_are_not_parameter_sensitive():
    class AlwaysDependent(_Feedback):
        def set_outputs(self, _t):
            self.output.value = (0.5 * self.input.x + self.offset if self.gain > 0.25
                              else 0.25 * self.input.x + self.offset)

    assert not _generator(AlwaysDependent).diagnostics


def test_conditional_downstream_output_is_not_a_feedback_parameter():
    class Output(Bag):
        __slots__ = ("value", "downstream")

    class Observation(_Feedback):
        def __init__(self, gain, offset):
            super().__init__(gain, offset)
            self.output = Output()

        def set_outputs(self, _t):
            self.output.value = 0.5 * self.input.x + self.offset
            self.output.downstream = self.gain * self.input.x

    assert not _generator(Observation).diagnostics


def test_dag_and_additive_parameters_do_not_request_scc_analysis(monkeypatch):
    generators = (
        _generator(feedback=False),
        _generator(gain=0.5, variable=("offset",)),
        _generator(gain=0.5, variable=()),
    )

    def unexpected_scc(*_args):
        pytest.fail("SCC analysis is unnecessary without conditional feedback")

    monkeypatch.setattr(kernel, "_strong_components", unexpected_scc)
    assert all(not generator.diagnostics for generator in generators)


@pytest.mark.parametrize("gain", [0.0, 0.1])
def test_export_finds_feedback_parameters_without_a_name_list(gain, tmp_path):
    p = _feedback_case(gain=gain)
    result = peslite.export(p, "cpp", tmp_path, variables=("all",))
    feedback = [item for item in result.diagnostics
                if item["code"] == "conditional-algebraic-dependency"]
    assert {path for item in feedback for path in item["parameters"]} == {
        "elements.feedback.gain"}
    assert all(item["dependencies"] for item in feedback)
    structure = [item for item in result.diagnostics
                 if item["code"] == "conditional-network-structure"]
    assert {path for item in structure for path in item["parameters"]} == {
        "buses.output.r_d", "units.converter.dclink.source.r",
        "units.converter.filter.r_f",
    }
    assert tuple(path.name for path in result.files) == ("peslite.cpp",)


def test_cli_prints_the_same_export_diagnostics(tmp_path, capsys):
    path = tmp_path / "feedback.pes"
    path.write_text(cpp.dumps(_feedback_case()), encoding="utf-8")
    assert peslite.main(["export", str(path), "all", "--out", str(tmp_path / "model")]) == 0
    captured = capsys.readouterr()
    assert "elements.feedback.gain" in captured.err
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


@pytest.mark.parametrize("feedback", [False, True])
def test_common_rebuilds_once_after_detecting_exclusions(feedback, tmp_path, monkeypatch):
    original = cpp._ModelGenerator
    calls = []

    class Counted(original):
        def __init__(self, *args, **kwargs):
            calls.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cpp, "_ModelGenerator", Counted)
    p = _feedback_case() if feedback else _case("buck", "pwm_averaging")
    peslite.export(p, "cpp", tmp_path, variables=("common",))
    assert len(calls) == 2


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_common_native_keeps_ordinary_set_and_rejects_only_detected_parameters(mode, tmp_path):
    p = _feedback_case(mode)
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
    p = _feedback_case(gain=0.1)
    binary = _compile(p, tmp_path / "model", variables=("common",))
    _compare_run(p, binary, {"elements.load.r": 8.0}, tmp_path / "run")
    keys, values = _table(tmp_path / "run/native/plant.csv")
    assert values[-1, keys.index("feedback.value")] == pytest.approx(1.0 / 0.9)


def test_common_cli_reports_the_constants_and_preserves_path_validation(tmp_path, capsys):
    path = tmp_path / "feedback.pes"
    path.write_text(cpp.dumps(_feedback_case()), encoding="utf-8")
    assert peslite.main(["export", str(path), "common", "--out", str(tmp_path / "model")]) == 0
    assert "common keeps detected structure-sensitive parameters constant" in capsys.readouterr().err
    with pytest.raises(ValueError, match="not variable"):
        peslite.export(_case("buck", "pwm_averaging"), "cpp", tmp_path / "invalid",
                       variables=("common", "missing.parameter"))


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_zero_coefficient_fast_path_keeps_real_feedback_when_parameters_change(mode, tmp_path):
    p = _feedback_case(mode)
    binary = _compile(p, tmp_path / "model")
    source = (tmp_path / "model/peslite.cpp").read_text(encoding="utf-8")
    direct = source.split("inline void evaluate(", 1)[1].split(
        "PESLITE_NOINLINE void evaluate_feedback(", 1)[0]
    assert "solve_algebraic(" not in direct
    assert direct.index("evaluate_feedback(t, y, dy)") < direct.index("= y[")
    for index, gain in enumerate((0.0, 0.25, -0.5)):
        directory = tmp_path / str(index)
        _compare_run(p, binary, {"elements.feedback.gain": gain}, directory)
        keys, values = _table(directory / "native/plant.csv")
        assert values[-1, keys.index("feedback.value")] == pytest.approx(1.0 / (1.0 - gain))
