"""Static checks for the manual C++ performance benchmark."""

from __future__ import annotations

import json

import pytest

from benchmark_cpp_export import (DEFAULTS, _check_baseline_settings, _comparison,
                                  _largest_changes, _load_settings, _mean_changes,
                                  _mode_summary, _validate_summary)


pytestmark = pytest.mark.static


def _rows(scale=1.0):
    return [
        {
            "case": case,
            "mode": mode,
            "export_s": scale * (case_index + 1),
            "compile_s": scale * 2.0,
            "cpp_s": scale * (mode_index + 1),
            "cpp_process_s": scale * (mode_index + 2),
            "source_bytes": scale * 1000.0,
            "dynamic_executable_bytes": scale * 2000.0,
            "estimated_static_executable_bytes": scale * 3000.0,
        }
        for case_index, case in enumerate(("first", "second"))
        for mode_index, mode in enumerate(("switching", "pwm_averaging", "averaging"))
    ]


def test_cpp_benchmark_summarises_and_compares_the_shared_matrix():
    before, after = _rows(), _rows(1.1)
    summary = _mode_summary(after)
    changes = _comparison(after, {"rows": before})

    assert summary["switching"]["mean_cpp_s"] == pytest.approx(1.1)
    assert _mean_changes(changes)["averaging"]["cpp_s"] == pytest.approx(10.0)
    assert _largest_changes(changes)["compile_s"][2] == pytest.approx(10.0)


def test_cpp_benchmark_config_resolves_its_own_case_paths(tmp_path):
    case = tmp_path / "case.pes"
    case.write_text("base: {}\n", encoding="utf-8")
    config = tmp_path / "benchmark.json"
    config.write_text(json.dumps({
        "cases": {"local": "case.pes"},
        "t_end": 0.25,
        "cpp_repeats": 1,
    }), encoding="utf-8")

    settings = _load_settings(config, {})

    assert settings["cases"] == {"local": case.resolve()}
    assert settings["t_end"] == 0.25
    assert settings["cpp_repeats"] == 1


def test_cpp_benchmark_defaults_use_the_two_machine_case():
    assert DEFAULTS["cases"]["two_machine"] == "examples/two-converters-example.pes"
    assert all("gfm-psc" not in path for path in DEFAULTS["cases"].values())


def test_cpp_benchmark_rejects_unsafe_case_names(tmp_path):
    case = tmp_path / "case.pes"
    case.write_text("base: {}\n", encoding="utf-8")
    config = tmp_path / "benchmark.json"
    config.write_text(json.dumps({"cases": {"../outside": "case.pes"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="one path component"):
        _load_settings(config, {})


def test_cpp_benchmark_compares_only_equivalent_workloads():
    current = {
        "t_end": 0.5,
        "output_period": 1.0e-3,
        "optimization": "-O3",
        "variables": [],
    }
    _check_baseline_settings(current, {"settings": dict(current)})

    previous = dict(current)
    previous["t_end"] = 1.0
    with pytest.raises(ValueError, match="different t_end"):
        _check_baseline_settings(current, {"settings": previous})


def test_cpp_benchmark_rejects_incomplete_or_tripped_runs():
    _validate_summary({"status": "completed", "tripped": False}, "case", "switching")
    with pytest.raises(RuntimeError, match="did not complete"):
        _validate_summary({"status": "stopped", "tripped": False}, "case", "switching")
    with pytest.raises(RuntimeError, match="did not complete"):
        _validate_summary({"status": "completed", "tripped": True}, "case", "switching")
