"""Run one case with each solver variant in ``VARIANTS`` and several computation delays, and print a comparison.

    python examples/compare_solvers.py configs/gfl.yaml --t-end 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import peslite  # noqa: E402

VARIANTS = {
    "fixed rk4, dt_max 1 us (reference)": {"simulation.solver.type": "fixed", "simulation.solver.method": "rk4",
                                          "simulation.solver.dt": 1e-6},
    "fixed rk4, dt_max 12.5 us": {"simulation.solver.type": "fixed", "simulation.solver.method": "rk4",
                                 "simulation.solver.dt": 12.5e-6},
    "fixed heun, dt_max 5 us": {"simulation.solver.type": "fixed", "simulation.solver.method": "heun",
                               "simulation.solver.dt": 5e-6},
    "adaptive DP45 (in-house)": {"simulation.solver.type": "adaptive", "simulation.solver.method": "DP45",
                                 "simulation.solver.rtol": 1e-6, "simulation.solver.atol": 1e-3},
    "adaptive SciPy RK45": {"simulation.solver.type": "adaptive", "simulation.solver.method": "RK45",
                            "simulation.solver.rtol": 1e-6, "simulation.solver.atol": 1e-3},
    "adaptive SciPy BDF": {"simulation.solver.type": "adaptive", "simulation.solver.method": "BDF",
                           "simulation.solver.rtol": 1e-6, "simulation.solver.atol": 1e-3},
    # dc link on its own step of 10 system steps
    "rk4 12.5 us, dc link on a 10-step window": {
        "simulation.solver.type": "fixed", "simulation.solver.method": "rk4", "simulation.solver.dt": 12.5e-6,
        "simulation.solver.subsystems": {"vsc.dclink": 10}},
    # ac network on 1/10 of the system step
    "rk4 12.5 us, ac network on dt/10": {
        "simulation.solver.type": "fixed", "simulation.solver.method": "rk4", "simulation.solver.dt": 12.5e-6,
        "simulation.solver.subsystems": {"pcc": 0.1, "grid.branch": 0.1,
                                        "vsc.branch_f": 0.1}},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--delays", default="0,1,2")
    args = ap.parse_args(argv)
    over = {"simulation.progress_every": 0.0, "simulation.log.control_every": 1}
    if args.t_end is not None:
        over["simulation.t_end"] = args.t_end

    results = {}
    print(f"{'variant':38s} {'wall s':>8s} {'rhs evals':>10s} {'max|di_d| pu':>13s} {'max|dv_dc| pu':>14s}")
    ref = None
    for name, variant in VARIANTS.items():
        p = peslite.load(args.config, **over, **variant)
        r = peslite.Simulation(p).run()
        results[name] = r
        if ref is None:
            ref = r
            print(f"{name:38s} {r.wall_time:8.1f} {r.n_rhs:10d} {'-':>13s} {'-':>14s}")
            continue
        n = min(len(r.control["vsc.id_pu"]), len(ref.control["vsc.id_pu"]))
        d_id = np.max(np.abs(r.control["vsc.id_pu"][:n] - ref.control["vsc.id_pu"][:n]))
        d_v = np.max(np.abs(r.control["vsc.vdc_pu"][:n] - ref.control["vsc.vdc_pu"][:n]))
        print(f"{name:38s} {r.wall_time:8.1f} {r.n_rhs:10d} {d_id:13.3e} {d_v:14.3e}")

    print("\ncomputation delay (fixed rk4, dt_max 12.5 us):")
    print(f"{'delay':>8s} {'wall s':>8s} {'final i_d pu':>13s} {'max |i| pu':>11s} {'tripped':>8s}")
    for n in (int(x) for x in args.delays.split(",")):
        p = peslite.load(args.config, **over, **VARIANTS["fixed rk4, dt_max 12.5 us"],
                        **{"units.vsc.delay.steps": n})
        r = peslite.Simulation(p).run()
        print(f"{n:8d} {r.wall_time:8.1f} {r.control['vsc.id_pu'][-1]:13.5f} "
              f"{r.summary['vsc.max_current_pu']:11.4f} {str(r.tripped):>8s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
