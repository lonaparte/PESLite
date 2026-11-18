"""Benchmark the standard C++ export matrix and compare it with a saved result.

The built-in matrix covers a two-converter system, WPT, the CIGRE 14-node feeder and a
cable-connected GFL in switching, PWM-averaging and averaging modes. A JSON file passed
through ``--config`` may override the defaults without changing this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import peslite
from peslite.assembly.params import _unit_items


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("switching", "pwm_averaging", "averaging")
METRICS = (
    "export_s", "compile_s", "cpp_s", "cpp_process_s", "source_bytes",
    "dynamic_executable_bytes", "estimated_static_executable_bytes",
)
DEFAULTS: dict[str, Any] = {
    "cases": {
        "two_machine": "examples/two-converters-example.pes",
        "wpt": "examples/wpt-example.pes",
        "cigre-14-example": "examples/cigre-14-example.pes",
        "cable_gfl": "examples/cable-gfl-example.pes",
    },
    "t_end": 0.5,
    "output_period": 1.0e-3,
    "optimization": "-O3",
    "export_repeats": 1,
    "compile_repeats": 1,
    "cpp_repeats": 3,
    "variables": [],
    "process_timeout": 300.0,
    "static_estimate": True,
}
CONFIG_FIELDS = frozenset(DEFAULTS)


def _compiler(command: str | Sequence[str] | None = None) -> list[str]:
    if command is None:
        command = os.environ.get("CXX")
    if command:
        parts = shlex.split(command) if isinstance(command, str) else list(command)
        if parts and shutil.which(parts[0]):
            return parts
        raise RuntimeError(f"C++ compiler not found: {parts[0] if parts else command}")
    path = next((path for name in ("c++", "g++", "clang++")
                 if (path := shutil.which(name)) is not None), None)
    if path is None:
        raise RuntimeError("a C++17 compiler is required (or set CXX, for example 'zig c++')")
    return [path]


def _load_settings(path: Path | None, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a benchmark JSON and explicit CLI values to the built-in settings."""
    settings = {**DEFAULTS, "cases": dict(DEFAULTS["cases"]),
                "variables": list(DEFAULTS["variables"])}
    base = PROJECT_ROOT
    if path is not None:
        configured = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(configured, dict):
            raise ValueError("benchmark config must contain one JSON object")
        unknown = sorted(set(configured) - CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"unknown benchmark config fields: {', '.join(unknown)}")
        settings.update(configured)
        if "cases" in configured:
            base = path.resolve().parent
    settings.update({key: value for key, value in overrides.items() if value is not None})

    cases = settings["cases"]
    if not isinstance(cases, dict) or not cases:
        raise ValueError("benchmark cases must be a non-empty name-to-file mapping")
    resolved = {}
    for name, value in cases.items():
        if not isinstance(name, str) or not name or not isinstance(value, str):
            raise ValueError("benchmark cases must map non-empty names to file paths")
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"benchmark case name must be one path component: {name}")
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else base / candidate
        if not candidate.is_file():
            raise ValueError(f"benchmark case does not exist: {candidate}")
        resolved[name] = candidate.resolve()
    settings["cases"] = resolved

    if not isinstance(settings["t_end"], (int, float)) or settings["t_end"] <= 0.0:
        raise ValueError("t_end must be positive")
    if (not isinstance(settings["output_period"], (int, float))
            or settings["output_period"] <= 0.0):
        raise ValueError("output_period must be positive")
    if settings["optimization"] not in ("-O0", "-O1", "-O2", "-O3", "-Os"):
        raise ValueError("optimization must be one of -O0, -O1, -O2, -O3 or -Os")
    for field in ("export_repeats", "compile_repeats", "cpp_repeats"):
        if not isinstance(settings[field], int) or settings[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if (not isinstance(settings["process_timeout"], (int, float))
            or settings["process_timeout"] <= 0.0):
        raise ValueError("process_timeout must be positive")
    if not isinstance(settings["variables"], list) or not all(
            isinstance(value, str) for value in settings["variables"]):
        raise ValueError("variables must be a list of parameter paths")
    if not isinstance(settings["static_estimate"], bool):
        raise ValueError("static_estimate must be true or false")
    return settings


def _case_params(config: Path, mode: str, settings: Mapping[str, Any]):
    solver = ("adaptive", "DP45") if mode == "averaging" else ("fixed", "RK4")
    loaded = peslite.load(config)
    changes = {
        **{f"units.{name}.bridge.model": mode for name, _unit in _unit_items(loaded.units)},
        "simulation.t_end": settings["t_end"],
        "simulation.solver.type": solver[0],
        "simulation.solver.method": solver[1],
        "simulation.solver.subsystems": {},
        "simulation.progress.enable": False,
        "simulation.output.period": settings["output_period"],
        "simulation.output.record_every": 1,
        "simulation.output.states": True,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    }
    return loaded.replace(**changes)


def _compile(source: Path, executable: Path, compiler: Sequence[str],
             settings: Mapping[str, Any], *, static: bool = False) -> float:
    command = [*compiler, settings["optimization"], "-DNDEBUG", "-std=c++17"]
    if static:
        command.append("-static")
    command.extend((str(source), "-o", str(executable)))
    start = time.perf_counter()
    subprocess.run(command, check=True, timeout=settings["process_timeout"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return time.perf_counter() - start


def _benchmark_case(name: str, config: Path, work: Path, compiler: Sequence[str],
                    settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode in MODES:
        params = _case_params(config, mode, settings)
        project = work / name / mode
        if project.exists():
            shutil.rmtree(project)
        project.mkdir(parents=True)

        export_times = []
        source = None
        for repeat in range(settings["export_repeats"]):
            destination = project / f"export-{repeat}"
            start = time.perf_counter()
            peslite.export(params, "cpp", destination, name=mode,
                           variables=settings["variables"])
            export_times.append(time.perf_counter() - start)
            source = destination / "peslite.cpp"
        assert source is not None
        for repeat in range(settings["export_repeats"] - 1):
            shutil.rmtree(project / f"export-{repeat}")

        executable = project / "peslite"
        compile_times = [
            _compile(source, executable, compiler, settings)
            for _repeat in range(settings["compile_repeats"])
        ]
        process_times, runtime_times = [], []
        for repeat in range(settings["cpp_repeats"]):
            output = project / f"run-{repeat}"
            start = time.perf_counter()
            subprocess.run([str(executable), "--out", str(output)], check=True,
                           timeout=settings["process_timeout"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            process_times.append(time.perf_counter() - start)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            runtime_times.append(float(summary["wall_time"]))
            if repeat + 1 < settings["cpp_repeats"]:
                shutil.rmtree(output)

        rows.append({
            "case": name,
            "config": str(config),
            "mode": mode,
            "export_s": statistics.median(export_times),
            "compile_s": statistics.median(compile_times),
            "cpp_s": statistics.median(runtime_times),
            "cpp_process_s": statistics.median(process_times),
            "source_bytes": source.stat().st_size,
            "dynamic_executable_bytes": executable.stat().st_size,
            "estimated_static_executable_bytes": None,
        })
    return rows


def _estimate_static_sizes(rows: list[dict[str, Any]], work: Path,
                           compiler: Sequence[str], settings: Mapping[str, Any]
                           ) -> dict[str, Any] | None:
    """Calibrate static-link overhead once and apply it to the shared generated backend."""
    if not settings["static_estimate"] or not rows:
        return None
    basis = rows[0]
    project = work / basis["case"] / basis["mode"]
    source = project / f"export-{settings['export_repeats'] - 1}" / "peslite.cpp"
    executable = project / "peslite-static-size-probe"
    try:
        _compile(source, executable, compiler, settings, static=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    static_bytes = executable.stat().st_size
    executable.unlink()
    overhead = max(0, static_bytes - basis["dynamic_executable_bytes"])
    for row in rows:
        row["estimated_static_executable_bytes"] = (
            row["dynamic_executable_bytes"] + overhead
        )
    return {
        "case": basis["case"],
        "mode": basis["mode"],
        "measured_static_bytes": static_bytes,
        "dynamic_bytes": basis["dynamic_executable_bytes"],
        "applied_static_link_overhead_bytes": overhead,
    }


def _mode_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    summary = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            f"mean_{metric}": statistics.fmean(
                float(row[metric]) for row in selected if row[metric] is not None
            )
            for metric in METRICS
            if any(row[metric] is not None for row in selected)
        }
    return summary


def _comparison(rows: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any]
                ) -> dict[tuple[str, str], dict[str, float]]:
    previous = {(row["case"], row["mode"]): row for row in baseline["rows"]}
    changes = {}
    for row in rows:
        key = row["case"], row["mode"]
        if key not in previous:
            raise ValueError(f"baseline has no result for {key[0]} / {key[1]}")
        values = {}
        for metric in METRICS:
            before, after = previous[key].get(metric), row.get(metric)
            if before not in (None, 0) and after is not None:
                values[metric] = 100.0 * (float(after) / float(before) - 1.0)
        changes[key] = values
    return changes


def _check_baseline_settings(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    """Reject comparisons made with simulation or compiler settings that change the work."""
    previous = baseline.get("settings")
    if not isinstance(previous, dict):
        raise ValueError("baseline does not contain benchmark settings")
    for field in ("t_end", "output_period", "optimization", "variables"):
        if previous.get(field) != current[field]:
            raise ValueError(f"baseline uses a different {field}")


def _mean_changes(changes: Mapping[tuple[str, str], Mapping[str, float]]
                  ) -> dict[str, dict[str, float]]:
    return {
        mode: {
            metric: statistics.fmean(values)
            for metric in METRICS
            if (values := [entry[metric] for (_case, selected_mode), entry in changes.items()
                           if selected_mode == mode and metric in entry])
        }
        for mode in MODES
    }


def _largest_changes(changes: Mapping[tuple[str, str], Mapping[str, float]]
                     ) -> dict[str, tuple[str, str, float]]:
    result = {}
    for metric in ("export_s", "compile_s", "cpp_s", "dynamic_executable_bytes"):
        candidates = [(case, mode, values[metric])
                      for (case, mode), values in changes.items() if metric in values]
        if candidates:
            result[metric] = max(candidates, key=lambda value: abs(value[2]))
    return result


def _format_bytes(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value) / 1024.0:.1f} KiB"


def _print_report(report: Mapping[str, Any],
                  changes: Mapping[tuple[str, str], Mapping[str, float]] | None) -> None:
    print("| case | mode | export | compile | C++ run | process | source | dynamic | static estimate |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["rows"]:
        print(
            f"| {row['case']} | {row['mode']} | {row['export_s']:.3f} s | "
            f"{row['compile_s']:.3f} s | {row['cpp_s']:.3f} s | "
            f"{row['cpp_process_s']:.3f} s | {_format_bytes(row['source_bytes'])} | "
            f"{_format_bytes(row['dynamic_executable_bytes'])} | "
            f"{_format_bytes(row['estimated_static_executable_bytes'])} |"
        )

    print("\n| mode | mean export | mean compile | mean C++ run | mean dynamic | mean static estimate |")
    print("|---|---:|---:|---:|---:|---:|")
    for mode, values in report["mode_summary"].items():
        print(
            f"| {mode} | {values['mean_export_s']:.3f} s | "
            f"{values['mean_compile_s']:.3f} s | {values['mean_cpp_s']:.3f} s | "
            f"{_format_bytes(values['mean_dynamic_executable_bytes'])} | "
            f"{_format_bytes(values.get('mean_estimated_static_executable_bytes'))} |"
        )

    if changes is None:
        return
    averages = _mean_changes(changes)
    print("\n| mode | mean run change | mean export change | mean compile change |")
    print("|---|---:|---:|---:|")
    for mode, values in averages.items():
        print(f"| {mode} | {values['cpp_s']:+.1f}% | {values['export_s']:+.1f}% | "
              f"{values['compile_s']:+.1f}% |")
    print("\nLargest case-level changes:")
    for metric, (case, mode, change) in _largest_changes(changes).items():
        print(f"- {metric}: {case} / {mode}: {change:+.1f}%")


def _relative_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(settings)
    values["cases"] = {name: str(path) for name, path in settings["cases"].items()}
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON file overriding the benchmark defaults")
    parser.add_argument("--work", type=Path,
                        default=Path(tempfile.gettempdir()) / "peslite-cpp-benchmark")
    parser.add_argument("--baseline", type=Path, help="saved result to compare against")
    parser.add_argument("--save", type=Path, help="write the complete result as JSON")
    parser.add_argument("--t-end", type=float)
    parser.add_argument("--output-period", type=float)
    parser.add_argument("--optimization", choices=("-O0", "-O1", "-O2", "-O3", "-Os"))
    parser.add_argument("--export-repeats", type=int)
    parser.add_argument("--compile-repeats", type=int)
    parser.add_argument("--cpp-repeats", type=int)
    parser.add_argument("--process-timeout", type=float)
    parser.add_argument("--variables", nargs="+")
    parser.add_argument("--static-estimate", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    overrides = {
        "t_end": args.t_end,
        "output_period": args.output_period,
        "optimization": args.optimization,
        "export_repeats": args.export_repeats,
        "compile_repeats": args.compile_repeats,
        "cpp_repeats": args.cpp_repeats,
        "process_timeout": args.process_timeout,
        "variables": args.variables,
        "static_estimate": args.static_estimate,
    }
    try:
        settings = _load_settings(args.config, overrides)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    compiler = _compiler()
    rows = []
    for name, config in settings["cases"].items():
        rows.extend(_benchmark_case(name, config, args.work, compiler, settings))
    calibration = _estimate_static_sizes(rows, args.work, compiler, settings)
    version = subprocess.run([*compiler, "--version"], check=True, capture_output=True,
                             text=True, timeout=settings["process_timeout"]).stdout.splitlines()[0]
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "peslite_version": getattr(peslite, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "compiler": {"command": compiler, "version": version},
        "settings": _relative_settings(settings),
        "static_estimate_calibration": calibration,
        "rows": rows,
        "mode_summary": _mode_summary(rows),
    }
    changes = None
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline.get("schema_version") != report["schema_version"]:
            parser.error("baseline uses a different benchmark schema")
        try:
            _check_baseline_settings(report["settings"], baseline)
        except ValueError as exc:
            parser.error(str(exc))
        changes = _comparison(rows, baseline)
        report["mean_changes_percent"] = _mean_changes(changes)
        report["largest_changes_percent"] = {
            metric: {"case": value[0], "mode": value[1], "change": value[2]}
            for metric, value in _largest_changes(changes).items()
        }
    _print_report(report, changes)
    if calibration is None and settings["static_estimate"]:
        print("\nStatic linking is unavailable; static size estimates were not produced.",
              file=sys.stderr)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
