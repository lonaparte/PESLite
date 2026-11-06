"""Reproducible pure-solver MPC cost measurements, with no timing pass/fail threshold.

Run from the source checkout with ``python tests/benchmark_mpc.py --repeats 100``.
Both algorithms solve the same scalar affine tracking problem. FCS restricts
inputs to three actions, so its optimum need not match the continuous optimum.
Model construction is outside the measured loop; each solve starts from the
same state and previous input. This is not an end-to-end or real-time benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peslite.control.mpc import CCSMPC, FCSMPC, LinearModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100,
                        help="measured solves per algorithm after one warmup solve")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    model = LinearModel([[0.9]], [[0.2]])
    horizon, moves = 4, 3
    state, reference, previous = np.array([0.0]), np.array([0.3]), np.array([0.0])
    actions = np.array([[-1.0], [0.0], [1.0]])
    ccs = CCSMPC(model, horizon, moves, input_weight=0.02, rate_weight=0.1,
                 input_bounds=(-1.0, 1.0), state_bounds=(-1.0, 1.0),
                 max_iterations=100, tolerance=1e-10, feasibility_tolerance=1e-8)

    def prediction(x, u, stage):
        return model.A @ x + model.B @ u + model.offset

    def objective(x, u, target, old, stage):
        error, change = x - target, u - old
        return float(error @ error + 0.02 * (u @ u) + 0.1 * (change @ change))

    def constraint(x, u, stage):
        return np.concatenate((x + 1.0, 1.0 - x))

    fcs = FCSMPC(actions, horizon, prediction, objective, control_horizon=moves,
                 constraint=constraint, max_evaluations=len(actions) ** moves,
                 feasibility_tolerance=1e-8)
    report = {
        "environment": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
        },
        "problem": {
            "states": 1, "inputs": 1, "prediction_horizon": horizon,
            "control_horizon": moves, "fcs_actions": len(actions),
            "fcs_plans": len(actions) ** moves, "repeats": args.repeats,
            "initial_state": state.tolist(), "reference": reference.tolist(),
            "previous_input": previous.tolist(),
        },
        "results": {},
    }
    print("| Algorithm | Median wall (ms) | Mean CPU (ms) | Iterations | Evaluations | Objective |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, solver in (("CCS", ccs), ("FCS", fcs)):
        warmup = solver.solve(state, reference, previous)
        if not warmup.success:
            raise RuntimeError(f"{name} warmup failed: {warmup.status}: {warmup.message}")
        wall, iterations, evaluations = [], [], []
        cpu_start = time.process_time()
        for repeat in range(args.repeats):
            wall_start = time.perf_counter()
            result = solver.solve(state, reference, previous)
            wall.append(time.perf_counter() - wall_start)
            if not result.success:
                raise RuntimeError(f"{name} solve {repeat + 1} failed: {result.status}: {result.message}")
            iterations.append(result.iterations)
            evaluations.append(result.evaluations)
        row = {
            "wall_seconds": statistics.median(wall),
            "cpu_seconds": (time.process_time() - cpu_start) / args.repeats,
            "iterations": statistics.median(iterations),
            "evaluations": statistics.median(evaluations),
            "status": result.status,
            "objective": result.objective,
            "action": result.action.tolist(),
        }
        report["results"][name] = row
        print(f"| {name} | {row['wall_seconds'] * 1000:.3f} | "
              f"{row['cpu_seconds'] * 1000:.3f} | {row['iterations']:g} | "
              f"{row['evaluations']:g} | {row['objective']:.9g} |")
    print("BENCHMARK_JSON " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
