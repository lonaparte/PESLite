# Solvers

PESLite integrates every interval up to the next exact boundary: an event, output point, sampled
hardware instant or requested end time. Solver stages never cross a model-changing boundary.

## Fixed-step solvers

```yaml
simulation:
  solver:
    type: fixed
    method: rk4
    dt: 25e-6
```

Available fixed methods are `euler`, `heun` and `rk4`. `dt` (default `1e-6` s) is the maximum step;
PESLite divides intervals as needed to end exactly at scheduled boundaries.

RK4 is the normal starting point for `switching` and `pwm_averaging`. Reduce `dt` until important
outputs and event responses are insensitive to further reduction.

## Adaptive solvers

```yaml
simulation:
  solver:
    type: adaptive
    method: DP45
    rtol: 1e-3
    atol: 1e-9
    max_step: .inf
```

`DP45` is PESLite's built-in Dormand-Prince 4(5) implementation. SciPy methods `RK23`, `RK45`,
`DOP853`, `Radau`, `BDF` and `LSODA` are also available in Python simulations.

`rtol` controls error relative to the current state scale, while `atol` supplies an absolute floor
near zero. Smaller tolerances generally increase accepted steps and RHS evaluations. `max_step`
limits how far the solver may advance even when the error estimate would permit more. Defaults are
`rtol: 1e-6`, `atol: 1e-9`, `max_step: .inf` and `warm_start: 1` (reuse the last step size).

Adaptive DP45 is the normal starting point for ideal `averaging`, where no PWM time grid forces
small intervals. Sampled modes are supported with adaptive integration, but the solver still stops
at every controller, register and switching boundary, which can remove much of the adaptive-speed
benefit.

## Command presets

The bridge/solver presets are listed in
[Bridge and PWM](Bridge-and-PWM.md#command-presets). Explicit `--set`
values for solver type or method take priority over them.

## Event-aware solver contract

A solver is called only for the interval presented by `Simulation`; shared `settle()` and
`parameters_changed()` hooks are described under [Custom solvers](Extending-PESLite.md#custom-solvers).
Built-in solvers also provide `advance(f, t0, t1, y0, max_step=...)`, which lets a physical
subsystem request a local resolution limit without changing the configured solver step. When a
user solver provides only the callable interface, the diode-conduction stage enforces such a limit
by dividing the requested interval before calling it.

## Energy checks and diagnostics

Before a run, `simulation.energy_check` controls validation of component energy declarations:

- `warn`: report structural or balance problems and continue;
- `strict`: reject a problematic model;
- `off`: skip the check and its structure report (no `ph_*` entries in `summary.json`).

During a run, `simulation.solver.phs_check_step` sets the number of output snapshots between energy
audits. It defaults to 1000 and is intended as an internal performance/diagnostic control rather
than a primary model setting.

`--ph-report` prints the assembled port-Hamiltonian structure without running, including when the
configuration sets `energy_check: off`. `summary.json` records `n_rhs` and any multirate interface
indicators.

## Selection guidance

| Study | Starting choice |
|---|---|
| Switching edges and ripple | fixed RK4 with a converged `dt` |
| Sampled control without carrier ripple | fixed RK4, PWM-period averaging |
| Continuous averaged dynamics | adaptive DP45 |
| Stiff continuous dynamics | a suitable SciPy implicit method, verified against a reference |
| Strongly separated subsystem time scales | [fixed multirate](Multirate-Simulation.md), after a single-rate baseline |

Always establish convergence for the quantities used in the conclusion. Output density is not an
integration-accuracy setting.
