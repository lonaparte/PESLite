"""Compare adaptive solvers for the ideal averaging bridge against fixed-step RK4.

This is a benchmark, not a pytest module: timings are machine dependent and there are no
performance assertions. Run a short sweep with::

    python -B tests/benchmark_averaging_solvers.py --t-end 0.3

Omit ``--t-end`` for the complete three-second example. Temporary CSV files are removed when the
benchmark finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peslite import Simulation, load  # noqa: E402
from peslite.solver.simulation import read_csv  # noqa: E402


CASES = {
    "fixed-rk4": {"type": "fixed", "method": "rk4"},
    "fixed-rk4-25us": {"type": "fixed", "method": "rk4", "dt": 25e-6},
    "fixed-rk4-50us": {"type": "fixed", "method": "rk4", "dt": 50e-6},
    "dp45-1e-1": {"type": "adaptive", "method": "DP45", "rtol": 1e-1},
    "dp45-1e-2": {"type": "adaptive", "method": "DP45", "rtol": 1e-2},
    "dp45-1e-3": {"type": "adaptive", "method": "DP45", "rtol": 1e-3},
    "dp45-1e-4": {"type": "adaptive", "method": "DP45", "rtol": 1e-4},
    "dp45-1e-5": {"type": "adaptive", "method": "DP45", "rtol": 1e-5},
    "dp45-1e-6": {"type": "adaptive", "method": "DP45", "rtol": 1e-6},
    "rk45-1e-4": {"type": "adaptive", "method": "RK45", "rtol": 1e-4},
    "rk45-1e-6": {"type": "adaptive", "method": "RK45", "rtol": 1e-6},
    "dop853-1e-4": {"type": "adaptive", "method": "DOP853", "rtol": 1e-4},
    "dop853-1e-6": {"type": "adaptive", "method": "DOP853", "rtol": 1e-6},
}


def _scale(name: str, reference: np.ndarray) -> float:
    """A reporting scale: SI plant magnitude, one pu/radian, or nominal angular speed."""
    if ".ctrl." in name or ".bridge." in name or name.endswith(".tripped"):
        if name.endswith(".omega"):
            return max(1.0, float(np.max(np.abs(reference))))
        return 1.0
    return max(1.0, float(np.max(np.abs(reference))))


def _errors(reference: np.ndarray, candidate: np.ndarray,
            scale: Callable[[str, np.ndarray], float]) -> dict[str, float | str]:
    """Scaled error on the state columns shared with the reference."""
    if reference.shape != candidate.shape or not np.allclose(reference["t"], candidate["t"],
                                                              rtol=0.0, atol=1e-12):
        return {"max_scaled": float("inf"), "rms_scaled": float("inf"),
                "worst_state": "incompatible output grid"}
    shared = tuple(name for name in reference.dtype.names or ()
                   if name != "t" and name in (candidate.dtype.names or ()))
    worst_name, worst = "", 0.0
    squared: list[np.ndarray] = []
    for name in shared:
        scaled = (candidate[name] - reference[name]) / scale(name, reference[name])
        value = float(np.max(np.abs(scaled)))
        if value > worst:
            worst_name, worst = name, value
        squared.append(scaled * scaled)
    rms = float(np.sqrt(np.mean(np.concatenate(squared)))) if squared else 0.0
    return {"max_scaled": worst, "rms_scaled": rms, "worst_state": worst_name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-end", type=float, default=None,
                        help="optional shorter stop time; default: the complete example")
    parser.add_argument("--case", action="append", choices=tuple(CASES),
                        help="run only this case; repeat to select several cases")
    args = parser.parse_args()
    selected = tuple(dict.fromkeys(args.case)) if args.case else tuple(CASES)
    if "fixed-rk4" not in selected:
        selected = ("fixed-rk4", *selected)

    base = load(ROOT / "examples" / "gfl-example.pes").replace(
        **{"units.vsc.bridge.model": "averaging"}
    )
    rows: list[dict[str, float | int | str]] = []
    reference = None
    with tempfile.TemporaryDirectory(prefix=".benchmark-adaptive-", dir=ROOT) as temporary:
        output_root = Path(temporary)
        for name in selected:
            settings = CASES[name]
            changes = {f"simulation.solver.{key}": value for key, value in settings.items()}
            if settings.get("type") == "adaptive":
                changes["simulation.solver.atol"] = 1e-9
            params = base.replace(**changes)
            simulation = Simulation(params)
            cpu0 = time.process_time()
            result = simulation.run(t_end=args.t_end, out_dir=output_root / name)
            states = read_csv(output_root / name / "states.csv")
            if reference is None:
                reference = states
                error = {"max_scaled": 0.0, "rms_scaled": 0.0, "worst_state": ""}
            else:
                error = _errors(reference, states, _scale)
            row = {
                "case": name,
                "wall_time": result.wall_time,
                "cpu_time": time.process_time() - cpu0,
                "n_rhs": result.n_rhs,
                "rejected": int(getattr(simulation.solver, "n_rejected", 0)),
                **error,
            }
            rows.append(row)
            print(
                f"{name:13s} wall {row['wall_time']:8.3f} s  RHS {row['n_rhs']:9,d}  "
                f"reject {row['rejected']:5,d}  max {row['max_scaled']:.3e}  "
                f"rms {row['rms_scaled']:.3e}  {row['worst_state']}",
                flush=True,
            )

    baseline = float(rows[0]["wall_time"])
    print("\n| solver | wall | CPU | RHS | speedup | max scaled error | RMS scaled error | worst state |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['case']} | {row['wall_time']:.3f} s | {row['cpu_time']:.3f} s | "
            f"{row['n_rhs']:,} | {baseline / float(row['wall_time']):.3f}x | "
            f"{row['max_scaled']:.3e} | {row['rms_scaled']:.3e} | {row['worst_state']} |"
        )
    print("BENCHMARK_JSON " + json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
