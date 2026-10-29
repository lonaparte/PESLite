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

Available fixed methods are `euler`, `heun` and `rk4`. `dt` is the maximum step, not a demand that
every interval have exactly that length. When an exact boundary falls inside a step, PESLite divides
the interval into equal shorter steps ending precisely at the boundary.

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
limits how far the solver may advance even when the error estimate would permit more.

Adaptive DP45 is the normal starting point for ideal `averaging`, where no PWM time grid forces
small intervals. Sampled modes are supported with adaptive integration, but the solver still stops
at every controller, register and switching boundary, which can remove much of the adaptive-speed
benefit.

## Command presets

| Command | Bridge | Solver |
|---|---|---|
| `--switching` | exact switching | fixed RK4 |
| `--pwm-averaging` | PWM-period averaging | fixed RK4 |
| `--averaging` | ideal averaging | adaptive DP45 |

These flags are configuration presets. Explicit `--set` values for solver type or method take
priority, so one file can be reused while deliberately testing another combination.

## Multirate integration

Named subsystems can use a step relative to the fixed system step. This supports fast substeps and
slow windows while keeping event boundaries exact. Multirate schedules require
`simulation.solver.type: fixed`; see [Multirate Simulation](Multirate-Simulation.md).

## Event-aware solver contract

A solver is called only for the interval presented by `Simulation`. Before a model-changing event,
an optional `settle(t, y)` hook can finish deferred work such as a multirate window. After a `set`
event, an optional `parameters_changed()` hook can refresh cached parameter-dependent data.
Built-in solvers also provide `advance(f, t0, t1, y0, max_step=...)`, which lets a physical
subsystem request a local resolution limit without changing the configured solver step. When a
user solver provides only the callable interface, the diode-conduction stage enforces such a limit
by dividing the requested interval before calling it.

This contract also applies to user solvers. A simple user solver only needs the normal callable
interface; the hooks are required only when it defers state or caches mutable parameters.

## Energy checks and diagnostics

Before a run, `simulation.energy_check` controls validation of component energy declarations:

- `warn`: report structural or balance problems and continue;
- `strict`: reject a problematic model;
- `off`: skip the check and its structure report (no `ph_*` entries in `summary.json`).

During a run, `simulation.solver.phs_check_step` sets the number of output snapshots between energy
audits. It defaults to 1000 and is intended as an internal performance/diagnostic control rather
than a primary model setting.

`--ph-report` prints the assembled port-Hamiltonian structure without running, including when the
configuration sets `energy_check: off`. Solver counts, rejected adaptive steps and multirate
interface indicators are written to `summary.json`.

## Selection guidance

| Study | Starting choice |
|---|---|
| Switching edges and ripple | fixed RK4 with a converged `dt` |
| Sampled control without carrier ripple | fixed RK4, PWM-period averaging |
| Continuous averaged dynamics | adaptive DP45 |
| Stiff continuous dynamics | a suitable SciPy implicit method, verified against a reference |
| Strongly separated subsystem time scales | fixed multirate, after a single-rate baseline |

Always establish convergence for the quantities used in the conclusion. Output density is not an
integration-accuracy setting.
