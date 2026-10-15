"""Manual Python/C++ timing comparison; prints measurements and has no performance assertion."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

import peslite
from peslite.assembly.params import _unit_items


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


def benchmark_cpp_export(config: str | Path, work: str | Path, *,
                         compiler: str | Sequence[str] | None = None,
                         t_end: float | None = None,
                         optimization: str = "-O3", cpp_repeats: int = 5,
                         ) -> list[dict[str, float | int | str]]:
    """Build and time all bridge presets for one simulation file."""
    compiler_command = _compiler(compiler)
    root, rows = Path(work), []
    for mode in ("switching", "pwm_averaging", "averaging"):
        solver = ("adaptive", "DP45") if mode == "averaging" else ("fixed", "rk4")
        loaded = peslite.load(config)
        changes = {
            **{f"units.{name}.bridge.model": mode for name, _unit in _unit_items(loaded.units)},
            "simulation.solver.type": solver[0], "simulation.solver.method": solver[1],
            "simulation.solver.subsystems": {}, "simulation.progress.enable": 0,
        }
        if t_end is not None:
            changes["simulation.t_end"] = t_end
        params = loaded.replace(**changes)
        project = root / mode
        start = time.perf_counter()
        peslite.export(params, "cpp", project, name=mode)
        export_seconds = time.perf_counter() - start
        executable = project / "peslite"
        start = time.perf_counter()
        subprocess.run([*compiler_command, optimization, "-DNDEBUG", "-std=c++17",
                        str(project / "peslite.cpp"), "-o", str(executable)], check=True)
        compile_seconds = time.perf_counter() - start
        start = time.perf_counter()
        peslite.Simulation(params).run(t_end=t_end, out_dir=project / "python-output")
        python_seconds = time.perf_counter() - start
        cpp_process_times, cpp_times = [], []
        for _ in range(cpp_repeats):
            start = time.perf_counter()
            subprocess.run([str(executable), str(project / "cpp-output")], check=True)
            cpp_process_times.append(time.perf_counter() - start)
            cpp_times.append(json.loads(
                (project / "cpp-output/summary.json").read_text(encoding="utf-8")
            )["wall_time"])

        def final(path: Path) -> dict[str, float]:
            with path.open(newline="", encoding="utf-8") as stream:
                return {key: float(value)
                        for key, value in list(csv.DictReader(stream))[-1].items()}

        expected = final(project / "python-output/states.csv")
        actual = final(project / "cpp-output/states.csv")
        scale = max(1.0, *(abs(value) for value in expected.values()))
        relative_error = max(abs(actual[key] - expected[key])
                             for key in expected) / scale
        cpp_seconds = statistics.median(cpp_times)
        rows.append({"mode": mode, "export_s": export_seconds,
                     "compile_s": compile_seconds,
                     "python_s": python_seconds, "cpp_s": cpp_seconds,
                     "cpp_process_s": statistics.median(cpp_process_times),
                     "speedup": python_seconds / cpp_seconds,
                     "relative_error": relative_error,
                     "source_bytes": (project / "peslite.cpp").stat().st_size,
                     "executable_bytes": executable.stat().st_size})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="examples/gfl-example.pes")
    parser.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "peslite-cpp-benchmark"))
    parser.add_argument("--t-end", type=float)
    parser.add_argument("--optimization", choices=("-O0", "-O1", "-O2", "-O3", "-Os"),
                        default="-O3")
    parser.add_argument("--cpp-repeats", type=int, default=5)
    args = parser.parse_args()
    rows = benchmark_cpp_export(args.config, args.work, t_end=args.t_end,
                                optimization=args.optimization,
                                cpp_repeats=args.cpp_repeats)
    print("| mode | export | compile | Python | C++ | process | speedup | error | source | executable |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['mode']} | {row['export_s']:.3f} s | {row['compile_s']:.3f} s | "
            f"{row['python_s']:.3f} s | {row['cpp_s']:.3f} s | "
            f"{row['cpp_process_s']:.3f} s | {row['speedup']:.2f}x | "
            f"{row['relative_error']:.2e} | "
            f"{row['source_bytes']:,} B | {row['executable_bytes']:,} B |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
