"""Profile the existing stability addon without changing its execution or runtime interfaces.

Run from a checkout, for example::

    python -B tests/benchmark_small_signal.py examples/buck-example.pes \
        --set simulation.t_end=0.05

The input must reach a periodic operating state. Histories, progress and energy audits are
disabled for this experiment. Timings are profiler-instrumented, not uninstrumented speed
claims. Matrix assembly excludes its period-advance calls; reconstruction, simulation, matrix,
eigenvalue and remaining orchestration costs are disjoint.
"""

from __future__ import annotations

import argparse
import cProfile
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peslite import Simulation, load  # noqa: E402
from peslite.addons.functions.small_signal import small_signal  # noqa: E402
from peslite.solver.linearization import linearize_loop  # noqa: E402
from peslite.solver.simulation import SystemLoop, parse_override  # noqa: E402


def profile_analysis(sim, **options):
    """Return the ordinary analysis and disjoint timings from its actual call graph."""
    profile = cProfile.Profile()
    result = profile.runcall(small_signal, sim, **options)
    entries = {entry.code: entry for entry in profile.getstats()}

    def entry(function):
        return entries[inspect.unwrap(function).__code__]

    total = entry(small_signal).totaltime
    rebuild = entry(Simulation.__init__)
    advance = entry(Simulation.run)
    linear = entry(linearize_loop)
    matrix = linear.totaltime - sum(call.totaltime for call in linear.calls
                                   if call.code is SystemLoop.advance.__code__)
    eigenvalues = entry(np.linalg.eigvals).totaltime
    remaining = total - rebuild.totaltime - advance.totaltime - matrix - eigenvalues
    sample = {
        "profiled_wall_seconds": total,
        "stage_seconds": {
            "model_reconstruction": rebuild.totaltime,
            "period_simulation": advance.totaltime,
            "transition_matrix_assembly": matrix,
            "eigenvalues": eigenvalues,
            "checks_and_orchestration": remaining,
        },
        "reconstructions": rebuild.callcount,
        "period_advances": advance.callcount,
        "evaluations": result.evaluations,
        "coordinates": len(result.coordinates),
        "period": result.period,
        "periodic_error": result.periodic_error,
        "stability": result.stability,
    }
    return result, sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", action="append", default=[], type=parse_override, metavar="PATH=VALUE")
    parser.add_argument("--period", type=float)
    parser.add_argument("--relative-step", type=float, default=1e-4)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    overrides = {
        **dict(args.set), "simulation.energy_check": "off", "simulation.progress.enable": False,
        "simulation.output.states": False, "simulation.output.signals": False,
        "simulation.output.energy": False, "simulation.solver.linearisations": 0,
    }
    start = time.perf_counter()
    sim = Simulation(load(args.config, **overrides))
    sim.run(out_dir=None)
    print(json.dumps({"settling_wall_seconds": time.perf_counter() - start,
                      "t_stop": sim.result.summary["t_stop"], "n_rhs": sim.result.n_rhs}), flush=True)
    samples = []
    for repeat in range(args.repeats):
        _analysis, sample = profile_analysis(sim, period=args.period, relative_step=args.relative_step)
        samples.append(sample)
        print(json.dumps({"repeat": repeat + 1, **sample}), flush=True)
    medians = {key: statistics.median(sample["stage_seconds"][key] for sample in samples)
               for key in samples[0]["stage_seconds"]}
    print(json.dumps({"median_stage_seconds": medians, "profiled": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
