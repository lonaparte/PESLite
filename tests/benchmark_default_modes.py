"""Benchmark every bundled example with the principal averaged configurations.

This is deliberately not a pytest module: timings are machine dependent and the complete default
simulation lengths take several minutes.  It preserves every example setting except the bridge
model and the explicitly named solver choices below.  Temporary result files are removed.

Run from the repository root with::

    python -B tests/benchmark_default_modes.py
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peslite import Simulation, load  # noqa: E402
from peslite.assembly.params import _unit_items  # noqa: E402
from peslite.solver.simulation import read_csv  # noqa: E402
from benchmark_averaging_solvers import _errors  # noqa: E402


CASES = {
    "averaging-fixed": {
        "bridge": "averaging", "type": "fixed", "method": "rk4",
    },
    "averaging-adaptive-1e-6": {
        "bridge": "averaging", "type": "adaptive", "method": "DP45", "rtol": 1e-6,
    },
    "averaging-adaptive-1e-3": {
        "bridge": "averaging", "type": "adaptive", "method": "DP45", "rtol": 1e-3,
    },
    "pwm-averaging-fixed": {
        "bridge": "pwm_averaging", "type": "fixed", "method": "rk4",
    },
}


def _scale(name: str, reference: np.ndarray) -> float:
    """Representative magnitude used only to make cross-state errors readable."""
    return max(1.0, float(np.max(np.abs(reference))))


def main() -> int:
    examples = sorted((ROOT / "examples").glob("*-example.pes"))
    rows: list[dict[str, float | int | str]] = []
    with tempfile.TemporaryDirectory(prefix=".benchmark-default-modes-", dir=ROOT) as temporary:
        output_root = Path(temporary)
        for path in examples:
            base = load(path)
            reference = None
            for case, settings in CASES.items():
                changes = {
                    **{f"units.{name}.bridge.model": settings["bridge"]
                       for name, _unit in _unit_items(base.units)},
                    "simulation.solver.type": settings["type"],
                    "simulation.solver.method": settings["method"],
                    "simulation.output.states": True,
                }
                if "rtol" in settings:
                    changes["simulation.solver.rtol"] = settings["rtol"]
                    changes["simulation.solver.atol"] = 1e-9
                params = base.replace(**changes)
                simulation = Simulation(params)
                gc.collect()
                cpu0 = time.process_time()
                result = simulation.run(out_dir=output_root / path.stem / case)
                states = read_csv(output_root / path.stem / case / "states.csv")
                if reference is None:
                    reference = states
                    error = {"max_scaled": 0.0, "rms_scaled": 0.0, "worst_state": ""}
                else:
                    error = _errors(reference, states, _scale)
                row = {
                    "example": path.stem,
                    "case": case,
                    "t_end": params.simulation.t_end,
                    "wall_time": result.wall_time,
                    "cpu_time": time.process_time() - cpu0,
                    "n_rhs": result.n_rhs,
                    "rejected": int(getattr(simulation.solver, "n_rejected", 0)),
                    "t_stop": float(result.summary.get("t_stop", 0.0)),
                    "tripped": int(result.summary.get("tripped", 0)),
                    **error,
                }
                rows.append(row)
                print(
                    f"{path.stem:28s} {case:27s} wall {row['wall_time']:8.3f} s  "
                    f"RHS {row['n_rhs']:9,d}  reject {row['rejected']:4,d}  "
                    f"max {row['max_scaled']:.3e}  trip {row['tripped']}",
                    flush=True,
                )

    print("\n| example | case | wall | CPU | RHS | rejected | max scaled | RMS scaled | worst state |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['example']} | {row['case']} | {row['wall_time']:.3f} s | "
            f"{row['cpu_time']:.3f} s | {row['n_rhs']:,} | {row['rejected']:,} | "
            f"{row['max_scaled']:.3e} | {row['rms_scaled']:.3e} | {row['worst_state']} |"
        )

    print("\n| case | total wall | total CPU | total RHS | speedup vs fixed averaging |")
    print("|---|---:|---:|---:|---:|")
    totals = {}
    for case in CASES:
        selected = [row for row in rows if row["case"] == case]
        totals[case] = {
            "wall_time": sum(float(row["wall_time"]) for row in selected),
            "cpu_time": sum(float(row["cpu_time"]) for row in selected),
            "n_rhs": sum(int(row["n_rhs"]) for row in selected),
        }
    fixed = totals["averaging-fixed"]["wall_time"]
    for case, total in totals.items():
        print(
            f"| {case} | {total['wall_time']:.3f} s | {total['cpu_time']:.3f} s | "
            f"{total['n_rhs']:,} | {fixed / total['wall_time']:.3f}x |"
        )
    print("BENCHMARK_JSON " + json.dumps({"rows": rows, "totals": totals}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
