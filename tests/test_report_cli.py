"""The optional CLI report uses the addon once and produces one named JSON document."""

import json
import shutil
from types import SimpleNamespace

import peslite
import pytest

from conftest import EXAMPLES
from peslite.solver import simulation


def _args(out, *options):
    return ["buck-example", "--out", str(out),
            "--set", "simulation.t_end=0.001",
            "--set", "simulation.energy_check=off",
            "--set", "simulation.progress.enable=0", *options]


def test_report_runs_the_main_simulation_once_and_preserves_config_metadata(tmp_path, monkeypatch):
    run, primary = peslite.Simulation.run, []

    def record(self, *args, **kwargs):
        if kwargs.get("info") is not None:
            primary.append(kwargs["info"])
        return run(self, *args, **kwargs)

    monkeypatch.setattr(peslite.Simulation, "run", record)
    assert peslite.main(_args(tmp_path, "--report")) == 0
    path = tmp_path / "buck-example_report.json"
    payload = json.loads(path.read_text())
    assert set(payload) == {"small_signal_stability"}
    assert payload["small_signal_stability"]["checks"] == 2
    assert payload["small_signal_stability"]["changes"]
    assert len(primary) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["config"] == str((EXAMPLES / "buck-example.pes").resolve())
    assert not list(tmp_path.glob("*.md"))
    assert list(tmp_path.glob("*_report.json")) == [path]


def test_report_name_comes_from_the_input_file_not_the_output_directory(tmp_path):
    source = tmp_path / "my-converter.v2.pes"
    shutil.copyfile(EXAMPLES / "buck-example.pes", source)
    out = tmp_path / "different-output-name"
    args = _args(out, "--report")
    args[0] = str(source)
    assert peslite.main(args) == 0
    assert (out / "my-converter.v2_report.json").is_file()
    assert not (out / "different-output-name_report.json").exists()


@pytest.mark.parametrize("preset", ["--switching", "--pwm-averaging", "--averaging"])
def test_report_filename_does_not_change_with_the_bridge_preset(tmp_path, preset):
    assert peslite.main(_args(tmp_path, preset, "--report")) == 0
    assert (tmp_path / "buck-example_report.json").is_file()


@pytest.mark.parametrize("missing", ["peslite.addons.functions.small_signal", "peslite.addons.functions"])
def test_missing_addon_warns_and_continues_the_ordinary_simulation(tmp_path, monkeypatch, missing):
    def absent(_name):
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(simulation, "import_module", absent)
    with pytest.warns(RuntimeWarning, match="addon is unavailable.*without a report"):
        assert peslite.main(_args(tmp_path, "--report")) == 0
    assert (tmp_path / "summary.json").is_file()
    assert not list(tmp_path.glob("*_report.json"))


@pytest.mark.parametrize("addon", [SimpleNamespace(), SimpleNamespace(small_signal_report=None)])
def test_addon_without_a_report_entry_point_warns_and_continues(tmp_path, monkeypatch, addon):
    monkeypatch.setattr(simulation, "import_module", lambda _name: addon)
    with pytest.warns(RuntimeWarning, match="addon is unavailable"):
        assert peslite.main(_args(tmp_path, "--report")) == 0
    assert (tmp_path / "summary.json").is_file()
    assert not list(tmp_path.glob("*_report.json"))


def test_an_installed_addons_missing_dependency_is_not_misreported_as_an_absent_addon(tmp_path, monkeypatch):
    def broken(_name):
        raise ModuleNotFoundError("missing analysis dependency", name="analysis_dependency")

    monkeypatch.setattr(simulation, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="missing analysis dependency"):
        peslite.main(_args(tmp_path, "--report"))
    assert not (tmp_path / "summary.json").exists()


def test_report_runtime_errors_do_not_fall_back_to_running_the_simulation_again(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("report implementation failed")

    monkeypatch.setattr(simulation, "import_module",
                        lambda _name: SimpleNamespace(small_signal_report=broken))
    with pytest.raises(ValueError, match="report implementation failed"):
        peslite.main(_args(tmp_path, "--report"))
    assert not (tmp_path / "summary.json").exists()


def test_without_the_flag_the_addon_is_not_imported(tmp_path, monkeypatch, capsys):
    def forbidden(_name):
        raise AssertionError("normal CLI runs must not import the report addon")

    monkeypatch.setattr(simulation, "import_module", forbidden)
    assert peslite.main(_args(tmp_path)) == 0
    assert not list(tmp_path.glob("*_report.json"))
    assert "ph_verdict" not in json.loads((tmp_path / "summary.json").read_text())

    capsys.readouterr()
    assert peslite.main(_args(tmp_path, "--ph-report")) == 0
    assert "port-hamiltonian" in capsys.readouterr().out


@pytest.mark.parametrize("options", [["--list-states"], ["--ph-report"], ["--resolved"],
                                     ["--export", "cpp"], ["--imperix"]])
def test_report_requires_a_simulation_not_an_inspection_or_export(tmp_path, options, capsys):
    with pytest.raises(SystemExit) as exc:
        peslite.main(_args(tmp_path, "--report", *options))
    assert exc.value.code == 2
    assert "--report requires a simulation run" in capsys.readouterr().err
    assert not (tmp_path / "summary.json").exists()


def test_report_can_be_combined_with_the_existing_plot_workflow(tmp_path, monkeypatch):
    plotted = []
    monkeypatch.setattr(peslite.SimulationResult, "plot", lambda self, *args, **kwargs: plotted.append(args))
    assert peslite.main(_args(tmp_path, "--report", "--plot", "output.u_C")) == 0
    assert plotted == [(["output.u_C"],)]
    assert (tmp_path / "buck-example_report.json").is_file()


@pytest.mark.parametrize("missing_addon", [False, True])
def test_a_new_run_clears_its_previous_report_but_preserves_unrelated_files(tmp_path, monkeypatch,
                                                                          missing_addon):
    assert peslite.main(_args(tmp_path, "--report")) == 0
    report = tmp_path / "buck-example_report.json"
    assert report.is_file()
    unrelated = tmp_path / "other-case_report.json"
    unrelated.write_text('{"other": true}\n', encoding="utf-8")
    if missing_addon:
        monkeypatch.setattr(simulation, "import_module", lambda _name: SimpleNamespace())
        with pytest.warns(RuntimeWarning, match="addon is unavailable"):
            assert peslite.main(_args(tmp_path, "--report")) == 0
    else:
        assert peslite.main(_args(tmp_path)) == 0
    assert not report.exists()
    assert unrelated.read_text(encoding="utf-8") == '{"other": true}\n'


def test_python_runs_use_the_same_report_file_lifecycle(tmp_path):
    from peslite.addons.functions.small_signal import small_signal_report

    params = peslite.load(EXAMPLES / "buck-example.pes", **{
        "simulation.t_end": 0.001, "simulation.energy_check": "off",
    })
    small_signal_report(peslite.Simulation(params), name="my-case", out_dir=tmp_path)
    path = tmp_path / "my-case_report.json"
    assert path.is_file()
    peslite.Simulation(params).run(name="my-case", out_dir=tmp_path)
    assert not path.exists()
