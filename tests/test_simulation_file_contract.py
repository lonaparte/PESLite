"""Bundled simulation-file discovery and packaging."""

import pytest
from conftest import EXAMPLES
from peslite.assembly.simulation_file import parse_native_document
from peslite.solver import simulation
from peslite.solver.simulation import _config_path

pytestmark = pytest.mark.static


def test_bundled_examples_are_native_pes_files():
    paths = {path.name: path for path in EXAMPLES.iterdir() if path.is_file()}
    assert paths and all(name.endswith("-example.pes") for name in paths)
    for path in paths.values():
        assert path.is_file()
        document = parse_native_document(path.read_text(encoding="utf-8"), path)
        assert document.parameters
        assert document.views[0].name == "circuit"


def test_command_line_finds_bundled_pes_files():
    found = _config_path("gfl-example")
    assert found == (EXAMPLES / "gfl-example.pes").resolve()
    assert _config_path("gfl-example.pes") == found
    assert _config_path(None) == found


def test_suffix_inference_discovers_only_native_pes_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local.yaml").write_text("base: {s_base: 1}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        _config_path("local")
    native = tmp_path / "local.pes"
    native.write_text("@circuit\n\nbase: {s_base: 1}\n", encoding="utf-8")
    assert _config_path("local") == native.resolve()


def test_installed_package_finds_examples_in_its_data_directory(tmp_path, monkeypatch):
    installed = tmp_path / "installed"
    examples = installed / "share" / "peslite" / "examples"
    examples.mkdir(parents=True)

    class Distribution:
        @staticmethod
        def locate_file(_entry):
            return installed

    fake_module = tmp_path / "package" / "peslite" / "solver" / "simulation.py"
    monkeypatch.setattr(simulation, "__file__", str(fake_module))
    monkeypatch.setattr(simulation, "distribution", lambda _name: Distribution())
    assert simulation._example_configs_dir() == examples
