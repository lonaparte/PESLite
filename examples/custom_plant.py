"""Run a case with an extra bus, a user synchronization law and a user solver.

    python examples/custom_plant.py configs/gfm-psc.yaml

The law's time constant is read from ``meta.custom.psc_lag_tau_s`` in the configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import peslite  # noqa: E402
from peslite.phs.protocols import SolverStep  # noqa: E402
from peslite.assembly import UniteType  # noqa: E402
from peslite.assembly.protocols import SyncOutput  # noqa: E402


def two_section_system(p: peslite.Params) -> peslite.Params:
    """Return ``p`` with the source impedance split by a new bus ``mid``.

    Topology: ``grid --[Z_s/2]-- mid --[Z_s/2]-- pcc --[Z_f]-- vsc``.
    """
    src, bus = p.sources["grid"], p.buses["pcc"]
    return p.replace(**{
        "buses": {"pcc": {"c": bus.c, "r_d": bus.r_d},
                  "mid": {"c": bus.c, "r_d": bus.r_d}},
        "sources.grid.bus": "mid",
        "sources.grid.l": 0.5 * src.l,
        "sources.grid.r": 0.5 * src.r,
        "branches": {"line": {"from_bus": "mid", "to_bus": "pcc",
                              "l": 0.5 * src.l, "r": 0.5 * src.r}},
    })


class LaggedPSC:
    """User synchronization law: PSC with a first-order lag ``tau`` (s) on the power feedback."""

    def __init__(self, k_p: float, tau: float, w0: float) -> None:
        self.k_p, self.tau, self.w0 = k_p, tau, w0
        self.theta, self.omega, self.p_f = 0.0, w0, 0.0

    def update(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu, q_ref_pu, v_ref_pu, i_dq=0j) -> SyncOutput:
        self.p_f += T / self.tau * (p_pu - self.p_f)
        self.omega = self.w0 + self.k_p * (p_ref_pu - self.p_f)
        self.theta += T * self.omega
        return SyncOutput(self.theta, self.omega, v_ref_pu, {"p_filtered_pu": self.p_f})

    # named states
    def get_state(self) -> dict[str, Any]:
        return {"theta": self.theta, "p_f": self.p_f}

    def set_state(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            setattr(self, key, float(value))


class Midpoint:
    """User solver: explicit midpoint rule with maximum step ``dt`` (s)."""

    def __init__(self, dt: float) -> None:
        self.dt, self.n_rhs = dt, 0

    def __call__(self, f, t0, t1, y0) -> SolverStep:
        n = max(1, int(np.ceil((t1 - t0) / self.dt - 1e-9)))
        h, y, t = (t1 - t0) / n, y0, t0
        for _ in range(n):
            k1 = f(t, y)
            y = y + h * f(t + 0.5 * h, y + 0.5 * h * k1)
            t += h
        self.n_rhs += 2 * n
        return SolverStep(t1, y, self.n_rhs)


def main(argv=None) -> int:
    config = argv[0] if argv else (sys.argv[1] if len(sys.argv) > 1 else "configs/gfm-psc.yaml")
    p = two_section_system(peslite.load(config, **{"simulation.t_end": 1.0, "simulation.progress_every": 0.0}))
    p = p.replace(**{"initial.states.plant.mid.u_C": "source"})  # the extra bus, pre-charged
    u = p.unit("vsc")
    tau = p.meta["custom"]["psc_lag_tau_s"]
    sim = peslite.Simulation(
        p,
        parts={"vsc.ctrl": lambda cfg: UniteType(cfg, loop_overrides={
            "sync": LaggedPSC(cfg.control.loops["sync"].k_p_pu, tau, cfg.base.w0)})},
        solver=Midpoint(p.simulation.solver.dt))
    print("states:", ", ".join(sim.state_names()))
    r = sim.run()
    print(f"two-section system, lagged PSC, midpoint solver: wall {r.wall_time:.1f} s, "
          f"rhs {r.n_rhs}, final p = {r.control['vsc.p_pu'][-1]:.4f} pu, "
          f"|v_pcc| = {abs(r.plant['vsc.u_g'][-1]) / u.base.v_phase_peak:.4f} pu, tripped = {r.tripped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
