"""Structured simulation commands and lifecycle status."""

import json
from pathlib import Path

import pytest

import peslite


def _command(capsys, *args):
    code = peslite.main([*args, "--json"])
    captured = capsys.readouterr()
    assert captured.err == ""
    return code, json.loads(captured.out)


def test_validate_assembles_without_running_or_writing(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    code, payload = _command(capsys, "validate", "gfl-example")

    assert code == payload["exit_code"] == 0
    assert payload["schema"] == "peslite.cli/v1"
    assert payload["command"] == "validate" and payload["status"] == "ok"
    assert payload["output"] is None and payload["simulation"] is None
    assert payload["validation"]["status"] == "valid"
    assert payload["validation"]["units"] == ["vsc"]
    assert payload["validation"]["state_count"] > 0
    assert not (tmp_path / "output").exists()


def test_inspect_states_returns_the_registered_names(capsys):
    code, payload = _command(capsys, "inspect", "gfl-example", "--states")

    expected = peslite.Simulation(peslite.load("examples/gfl-example.pes")).state_names()
    assert code == 0
    assert payload["state_count"] == len(payload["states"])
    assert payload["states"] == expected
    assert "vsc.dclink.u_C" in payload["states"]


def test_inspect_defaults_to_every_view_and_accepts_combinations(capsys):
    code, payload = _command(capsys, "inspect", "gfl-example")
    assert code == 0
    assert {"resolved", "states", "ph_report", "groups"} <= payload.keys()

    code, payload = _command(capsys, "inspect", "gfl-example", "--resolved", "--states")
    assert code == 0
    assert "resolved" in payload and "states" in payload and "ph_report" not in payload


def test_inspect_human_output_defaults_to_every_view(capsys):
    assert peslite.main(["inspect", "gfl-example"]) == 0
    output = capsys.readouterr().out
    assert "Resolved configuration:" in output
    assert "States:" in output
    assert "Port-Hamiltonian report:" in output
    assert "Electrical groups:" in output


def test_inspect_exports_the_existing_electrical_groups_without_running(capsys, monkeypatch):
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("inspection started a simulation")

    monkeypatch.setattr(peslite.Simulation, "run", unexpected_run)
    code, payload = _command(capsys, "inspect", "two-converters-example", "--groups", "--states",
                             "--set", "simulation.solver.circuit_model=nodal")
    assert code == 0 and payload["groups"]
    report = payload["groups"][0]
    assert report["circuit_model"] == "nodal"
    assert report["groups"] and report["interface"]["nodes"]
    assert "states" in payload and "resolved" not in payload and "ph_report" not in payload


def test_run_json_is_the_only_stdout_and_uses_absolute_result_paths(capsys, tmp_path):
    out = tmp_path / "result"
    code, payload = _command(
        capsys, "run", "gfl-example", "--pwm-averaging",
        "--set", "simulation.t_end=0.001",
        "--set", "simulation.solver.linearisations=0",
        "--out", str(out),
    )

    assert code == payload["exit_code"] == 0
    assert payload["command"] == "run" and payload["status"] == "ok"
    assert payload["simulation"]["status"] == "completed"
    assert payload["simulation"]["t_stop"] == pytest.approx(0.001)
    assert payload["simulation"]["rhs_evaluations"] > 0
    assert payload["output"]["directory"] == str(out.resolve())
    assert payload["output"]["summary"] == str((out / "summary.json").resolve())
    assert all(Path(path).is_absolute() for path in payload["output"]["files"])
    assert payload["physical_validation"] == "not_performed"
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_expected_trip_is_a_successful_run_with_diagnostics(capsys, tmp_path):
    code, payload = _command(
        capsys, "run", "gfl-example",
        "--set", "units.vsc.protection.overcurrent.limit_pu=0.001",
        "--set", "events.connect_vsc.t=0",
        "--set", "events.connect_vsc.ramp=0",
        "--set", "simulation.t_end=0.002",
        "--out", str(tmp_path / "trip"),
    )

    assert code == 0 and payload["status"] == "ok" and payload["error"] is None
    assert payload["simulation"]["status"] == "stopped"
    assert payload["simulation"]["tripped"] is True
    assert any(item["code"] == "unit_warning" for item in payload["diagnostics"])


def test_json_configuration_error_has_a_stable_envelope(capsys):
    code, payload = _command(
        capsys, "validate", "gfl-example", "--set", "simulation.t_end=invalid",
    )

    assert code == payload["exit_code"] == 2
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "configuration_error"
    assert payload["config"].endswith("gfl-example.pes")
    assert payload["output"] is None and payload["simulation"] is None


def test_timeout_closes_output_and_reports_the_partial_run(capsys, tmp_path):
    out = tmp_path / "limited"
    code, payload = _command(
        capsys, "run", "gfl-example", "--timeout", "1e-12", "--out", str(out),
    )

    assert code == payload["exit_code"] == 5
    assert payload["status"] == "error"
    assert payload["simulation"]["status"] == "timed_out"
    assert payload["error"]["code"] == "timeout"
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "timed_out"
    assert "wall-time limit" in saved["stop_reason"]


def test_rhs_limit_uses_the_same_lifecycle_path(capsys, tmp_path):
    code, payload = _command(
        capsys, "run", "gfl-example", "--max-rhs", "1",
        "--set", "simulation.solver.linearisations=0", "--out", str(tmp_path / "limited"),
    )

    assert code == 5
    assert payload["simulation"]["status"] == "stopped"
    assert payload["simulation"]["rhs_evaluations"] >= 1
    assert payload["error"]["code"] == "resource_limit"


def test_export_uses_the_same_structured_command_interface(capsys, tmp_path):
    out = tmp_path / "export"
    code, payload = _command(capsys, "export", "buck-open-loop-example", "--out", str(out))

    assert code == 0 and payload["command"] == "export"
    assert payload["export"]["format"] == "cpp"
    assert payload["output"]["directory"] == str(out.resolve())
    assert (out / "peslite.cpp").is_file()
