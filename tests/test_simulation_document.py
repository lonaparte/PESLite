"""Direct native-text parsing and legacy-format delegation."""

import json
from pathlib import Path

import pytest

from peslite.assembly.simulation_file import (
    SimulationFileError, parse_native_text, parse_simulation_text,
)
from peslite.assembly.params import read_initial

pytestmark = pytest.mark.static


SYNTAX_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "pes_syntax_cases.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", SYNTAX_CASES, ids=lambda case: case["name"])
def test_shared_native_syntax_cases(case):
    if case["valid"]:
        parse_native_text(case["text"], "case.pes")
    else:
        with pytest.raises(SimulationFileError):
            parse_native_text(case["text"], "case.pes")


def test_native_text_separates_interface_source_from_parameter_domains():
    source = """# PESLite
base: {s_base: 1.0e+6}

@circuit
Grid -- PCC -- VSC

@VSC
Current -> PWM
VSC: vsc, gfl
  Current: CurrentLoop
  PWM: ThreePhaseModulation

simulation:
  t_end: 0.1
"""
    parameters, interface = parse_native_text(source, "case.pes")
    assert parameters == {
        "base": {"s_base": 1.0e6},
        "simulation": {"t_end": 0.1},
    }
    assert interface == """@circuit
Grid -- PCC -- VSC

@VSC
Current -> PWM
VSC: vsc, gfl
  Current: CurrentLoop
  PWM: ThreePhaseModulation
"""


def test_native_text_requires_one_unambiguous_circuit_view():
    with pytest.raises(SimulationFileError, match="require an @circuit"):
        parse_native_text("base: {s_base: 1}\n", "case.pes")
    with pytest.raises(SimulationFileError, match="duplicate @circuit"):
        parse_native_text("@circuit\n@circuit\nbase: {s_base: 1}\n", "case.pes")
    with pytest.raises(SimulationFileError, match="invalid interface view header"):
        parse_native_text("@circuit wrong\nbase: {s_base: 1}\n", "case.pes")


def test_parameter_only_yaml_is_loaded_by_the_registered_format_addon():
    parameters, interface = parse_simulation_text(
        "base: {s_base: 1.0e6}\nsimulation: {t_end: 0.1}\n", Path("legacy.pes")
    )
    assert parameters["simulation"]["t_end"] == 0.1
    assert interface == ""


def test_native_text_reports_structural_and_parameter_errors():
    with pytest.raises(SimulationFileError, match="requires a displayed type at line 2"):
        parse_native_text("@circuit\nConverter:\nbase: {s_base: 1}\n")
    with pytest.raises(SimulationFileError, match="duplicate key: t_end"):
        parse_native_text(
            "@circuit\nbase: {s_base: 1}\nsimulation:\n  t_end: 1\n  t_end: 2\n"
        )
    with pytest.raises(SimulationFileError, match="unknown parameter domain: extra"):
        parse_native_text("@circuit\nbase: {s_base: 1}\nextra: {value: 2}\n")


def test_native_simulation_file_can_supply_continuation_state(tmp_path):
    source = tmp_path / "saved.pes"
    source.write_text("""@circuit

base: {s_base: 1.0e+6}
simulation:
  t_end: 0.2
  initial: {t: 0.1, states: {plant.current: 2.5}}
""", encoding="utf-8")
    assert read_initial(source) == {"t": 0.1, "states": {"plant.current": 2.5}}


@pytest.mark.parametrize(("suffix", "text"), [
    (".yaml", "t: 0.1\nstates: {plant.current: 2.5}\n"),
    (".json", '{"t": 0.1, "states": {"plant.current": 2.5}}'),
])
def test_plain_initial_data_remains_independent_of_simulation_syntax(tmp_path, suffix, text):
    source = tmp_path / f"saved{suffix}"
    source.write_text(text, encoding="utf-8")
    assert read_initial(source) == {"t": 0.1, "states": {"plant.current": 2.5}}
