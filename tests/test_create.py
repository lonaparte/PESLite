"""Create named files from the bundled templates without overwriting project files."""

import ast
import json
import os
import subprocess
import sys
import textwrap
import warnings

import pytest

import peslite
from conftest import EXAMPLES, ROOT
from peslite.solver import simulation as simulation_module


@pytest.mark.parametrize("name,relative,class_name", [
    ("study.pes", "study.pes", None),
    ("component.py", "PESaddons/components/component.py", "Component"),
    ("controller.py", "PESaddons/control/controller.py", "Controller"),
    ("my_control.py", "PESaddons/control/my_control.py", "MyControl"),
])
def test_create_uses_the_requested_name_and_existing_addon_paths(
        name, relative, class_name, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert peslite.main(["create", "--json", name]) == 0
    payload = json.loads(capsys.readouterr().out)
    target = tmp_path / relative
    assert payload["command"] == "create" and payload["status"] == "ok"
    assert payload["output"]["files"] == [str(target)]
    assert payload["output"]["directory"] == str(target.parent)
    text = target.read_text(encoding="utf-8")
    if class_name is None:
        assert text == (EXAMPLES / "templates/template.pes").read_text(encoding="utf-8")
        assert peslite.load(target).simulation.t_end > 0.0
        assert not (tmp_path / "PESaddons").exists()
    else:
        assert [node.name for node in ast.parse(text).body
                if isinstance(node, ast.ClassDef)] == [class_name]


@pytest.mark.parametrize("name,relative", [
    ("case.pes", "case.pes"),
    ("component.py", "PESaddons/components/component.py"),
    ("a.py", "PESaddons/control/a.py"),
])
def test_create_preserves_an_existing_file(name, relative, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Existing project content.\n", encoding="utf-8")
    assert peslite.main(["create", name, "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error" and payload["error"] is not None
    assert target.read_text(encoding="utf-8") == "Existing project content.\n"


def test_create_opens_only_when_requested(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opened = []
    monkeypatch.setattr(simulation_module, "_open_file", opened.append)
    assert peslite.main(["create", "closed.pes"]) == 0
    assert opened == []
    assert peslite.main(["create", "opened.pes", "--open"]) == 0
    assert opened == [tmp_path / "opened.pes"]


def test_create_keeps_the_file_when_opening_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def fail(_path):
        raise OSError("no associated application")

    monkeypatch.setattr(simulation_module, "_open_file", fail)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert peslite.main(["create", "case.pes", "--open", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert (tmp_path / "case.pes").is_file()
    assert payload["status"] == "ok"
    assert len(caught) == 0
    assert payload["diagnostics"][0]["severity"] == "warning"
    assert "could not open" in payload["diagnostics"][0]["message"]


def test_open_file_prefers_visual_then_editor(tmp_path, monkeypatch):
    target = tmp_path / "case.pes"
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("VISUAL", "code --wait")
    monkeypatch.setenv("EDITOR", "nano")
    simulation_module._open_file(target)
    monkeypatch.delenv("VISUAL")
    simulation_module._open_file(target)
    assert calls == [
        (["code", "--wait", str(target)], {}),
        (["nano", str(target)], {}),
    ]


def test_open_file_uses_xdg_open_without_an_editor(tmp_path, monkeypatch):
    target = tmp_path / "case.pes"
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, stderr="")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    simulation_module._open_file(target)
    assert calls[0][0] == ["xdg-open", str(target)]
    assert calls[0][1]["timeout"] == 10


def test_create_requires_a_file_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        peslite.main(["create"])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["case.txt", "../case.pes", "bad-name.py"])
def test_create_rejects_invalid_file_names(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        peslite.main(["create", name])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_created_templates_load_together_and_use_the_normal_component_and_loop_apis(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("case.pes", "component.py", "controller.py", "a.py"):
        assert peslite.main(["create", name]) == 0
    script = textwrap.dedent('''
        from types import SimpleNamespace
        import peslite
        from peslite.control.loops import loop_type

        params = peslite.load("case.pes", **{
            "elements.load": {"type": "component", "bus": "pcc"},
            "simulation.t_end": 1e-4,
            "simulation.progress.enable": 0,
            "simulation.solver.linearisations": 0,
        })
        result = peslite.Simulation(params).run(out_dir="result")
        assert result.summary["status"] == "completed"
        assert {"load.i.re", "load.i.im"} <= result.final_states().keys()
        for name in ("Controller", "A"):
            cls = loop_type(name)
            for mode in ("pwm_averaging", "averaging"):
                loop = cls(cls.Params(period=1e-4),
                           SimpleNamespace(bridge=SimpleNamespace(model=mode)), None)
                inputs = {"reference": 1.0, "voltage": 0.25}
                values = (loop.sample(0.0, inputs) if mode == "pwm_averaging"
                          else loop.flow(0.0, inputs)[0])
                assert values["current"] > 0.0
    ''')
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
