"""Native simulation-document parsing and legacy-format delegation."""

from pathlib import Path

import pytest

from peslite.assembly.simulation_file import (
    SimulationFileError, parse_native_document, parse_simulation_document,
)
from peslite.assembly.params import read_initial

pytestmark = pytest.mark.static


def test_native_document_separates_views_from_parameter_domains():
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
    document = parse_native_document(source, "case.pes")
    assert document.parameters == {
        "base": {"s_base": 1.0e6},
        "simulation": {"t_end": 0.1},
    }
    assert [(view.name, view.line) for view in document.views] == [
        ("circuit", 4), ("VSC", 7),
    ]
    assert "Grid -- PCC -- VSC" in document.views[0].source
    assert "Current -> PWM" in document.views[1].source


def test_native_document_requires_one_unambiguous_circuit_view():
    with pytest.raises(SimulationFileError, match="require an @circuit"):
        parse_native_document("base: {s_base: 1}\n", "case.pes")
    with pytest.raises(SimulationFileError, match="duplicate @circuit"):
        parse_native_document("@circuit\n@circuit\nbase: {s_base: 1}\n", "case.pes")
    with pytest.raises(SimulationFileError, match="invalid interface view header"):
        parse_native_document("@circuit wrong\nbase: {s_base: 1}\n", "case.pes")


def test_parameter_only_yaml_is_loaded_by_the_registered_format_addon():
    document = parse_simulation_document(
        "base: {s_base: 1.0e6}\nsimulation: {t_end: 0.1}\n", Path("legacy.pes")
    )
    assert document.parameters["simulation"]["t_end"] == 0.1
    assert document.views == ()
    assert document.source_format.endswith("addons.formats.legacy_yaml")


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
