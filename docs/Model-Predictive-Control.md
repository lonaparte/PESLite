# Model Predictive Control

PESLite provides continuous-control-set MPC (CCS-MPC) and finite-control-set MPC (FCS-MPC)
as ordinary sampled control loops. They use the same typed connections, ADC, actuation,
protection, start-up and saved-state mechanisms as other loops. The converter adapters support
`switching`, `pwm_averaging` and `averaging`; C++ and Imperix export are unsupported.

## Run the examples

```bash
peslite mpc-buck-example --pwm-averaging
peslite mpc-buck-example --averaging
peslite mpc-buck-example --switching
peslite mpc-vsc-example --pwm-averaging
peslite mpc-vsc-example --averaging
peslite mpc-vsc-example --switching
```

`mpc-buck-example` regulates the inductor current of a synchronous Buck using CCS-MPC. It
includes current-reference and load changes. Its output voltage follows the load and current;
this example does not add an outer voltage regulator.

`mpc-vsc-example` uses FCS-MPC to track a three-phase converter's current reference. The
controller chooses one of the eight binary three-leg commands, including both zero vectors.
Each chosen command follows the bridge model's ordinary actuation timing. Binary duties produce
held bridge states, so the ideal FCS command has the same bridge voltage in every model.

## Configuration and units

For a DC converter with a synchronous half bridge and an external output bus:

```yaml
ctrl:
  type: current
  period: 50.0e-6
  computation: 1.0e-6
  references: {current: 4.0}
  loops:
    PredictiveCurrentControl:
      algorithm: ccs
      prediction_horizon: 4
      control_horizon: 3
      tracking_weight: 1.0
      input_weight: 0.0
      rate_weight: 0.01
      input_min: 0.0
      input_max: 1.0
      current_min: -8.0
      current_max: 8.0
      max_iterations: 100
      max_evaluations: 10000
      tolerance: 1.0e-9
      feasibility_tolerance: 1.0e-7
      fallback: hold
```

For a three-phase VSC, use the ordinary `gfl` profile with a synchronization loop and replace
the current-loop/modulation chain with `ThreePhasePredictiveCurrentControl`:

```yaml
ctrl:
  type: gfl
  references: {id_ref_pu: 0.4, iq_ref_pu: 0.0}
  loops:
    SRFPLL: {kp_pu: 20.0, ki_pu: 1200.0}
    ThreePhasePredictiveCurrentControl:
      algorithm: fcs
      prediction_horizon: 3
```

Either predictive loop accepts `algorithm: ccs` or `algorithm: fcs`.

| Setting | Meaning |
|---|---|
| `prediction_horizon` | Number of predicted controller intervals; positive integer. |
| `control_horizon` | Number of independent input decisions, no larger than the prediction horizon; omitted means the prediction horizon. The last decision is held thereafter. |
| `tracking_weight` (Q) | Nonnegative squared current-error weight; current is in amperes. |
| `input_weight` (R) | Nonnegative squared duty-ratio weight. |
| `rate_weight` (S) | Nonnegative squared duty-increment weight relative to the preceding command. |
| `input_min`, `input_max` | Duty bounds, within `[0, 1]`, applied to every controlled leg. |
| `current_min`, `current_max` | Optional lower/upper predicted current bounds in amperes, applied separately to each controlled coordinate. |
| `max_iterations` | CCS solver iteration limit. |
| `max_evaluations` | FCS search-work limit. |
| `tolerance` | SLSQP stopping tolerance for the coefficient-normalized objective. |
| `feasibility_tolerance` | Absolute tolerance used when checking constraint satisfaction. |
| `fallback` | Action on failure: `hold`, `fixed` or `block`. |
| `fallback_input` | Explicit duty vector for `fixed` fallback. |

The DC reference and current prediction use SI amperes. VSC `id_ref_pu`/`iq_ref_pu` use the
unit's peak phase-current base, then convert to SI alpha-beta currents before optimization.
The three-phase current limits bound alpha and beta independently; they are not phase-current
peak or current-magnitude limits. Input and rate penalties use dimensionless duty ratios.
Choose weights with these scales in mind; they do not normalize quantities automatically.

The three-phase loop obtains frame and frequency through its typed graph inputs; the example
connects them to `SRFPLL`. Existing frequency, ROCOF, overcurrent and voltage protection remains
owned by the unit.

### Weight matrices

The three configuration fields directly define the stage-cost matrices Q, R and S:

`(current_error).T @ Q @ current_error + duty.T @ R @ duty + duty_change.T @ S @ duty_change`.

Each field accepts a scalar for a uniform diagonal, a vector for an explicit diagonal or a full
symmetric positive-semidefinite matrix:

```yaml
# Uniform diagonals: Q = 1.0 I, R = 0 and S = 0.02 I.
tracking_weight: 1.0
input_weight: 0.0
rate_weight: 0.02

# Equivalent coordinate-specific diagonals for a three-phase loop.
tracking_weight: [1.0, 0.8]
input_weight: [0.001, 0.001, 0.001]
rate_weight: [0.02, 0.02, 0.02]
```

A full three-phase tracking matrix can include coupling between the alpha and beta errors:

```yaml
tracking_weight:
  - [1.0, 0.1]
  - [0.1, 0.8]
```

For `PredictiveCurrentControl`, Q, R and S are all 1-by-1. For
`ThreePhasePredictiveCurrentControl`, Q is 2-by-2 for the alpha-beta current error, while R and S
are 3-by-3 for the three bridge-leg commands. PESLite validates the dimensions and matrix
properties when the loop is constructed.

Start with `tracking_weight: 1.0`, `input_weight: 0.0` and `rate_weight: 0.01`. Increase Q to
prioritize current tracking, R to penalize command magnitude, or S to penalize command changes.
Because current errors are in amperes and duties are dimensionless, these tradeoffs remain an
engineering design choice. The built-in loops derive A, B, C and the affine offset from the
converter, filter and timing configuration; users do not enter those plant matrices.

## Prediction and optimization

`CCSMPC` accepts a `LinearModel` describing `x_next = A @ x + B @ u + offset` and
`y_next = C @ x_next`. Its stage cost is
`(y_next - reference).T @ Q @ (y_next - reference) + u.T @ R @ u + du.T @ S @ du`, where
`du = u - previous_input`. State/output bounds and tracking apply to predicted endpoints;
they do not constrain the supplied initial state. `C` defaults to the identity and `offset`
to zero. Model matrices are copied and read-only.

```python
from peslite.control.mpc import CCSMPC, LinearModel

model = LinearModel(A=[[1.0]], B=[[1.0]])
controller = CCSMPC(
    model, prediction_horizon=3, control_horizon=2,
    tracking_weight=1.0, input_weight=0.1, rate_weight=0.05,
    input_bounds=(-0.5, 0.5), output_bounds=(-2.0, 2.0),
)
result = controller.solve(state=[0.0], reference=[1.0], previous_input=[0.0])
if result.success:
    command = result.action
```

Each `input_bounds`, `state_bounds` or `output_bounds` is a `(lower, upper)` pair; each side can
be scalar, one coordinate vector or a horizon-by-coordinate array. References similarly
accept a coordinate vector or one row per prediction stage. `solve(..., models=...)` accepts
one dimension-compatible affine model per stage for known disturbances or varying coefficients.
`initial_plan` supplies an explicit control-horizon-by-input initial guess. Model coordinates
determine the units of every weight and constraint; the feasibility tolerance is absolute in
each coordinate.

CCS-MPC solves the constrained quadratic objective with SciPy SLSQP. SciPy is already a core
dependency; no additional optimizer package is required. Solver success alone is insufficient:
the returned action must also be finite and satisfy the configured feasibility tolerance.
The SLSQP adapter divides the quadratic and linear objective coefficients by one common
positive scale and omits the constant during optimization. This leaves the minimizer and
constraints unchanged and avoids false convergence at large physical-current scales.
Reported objective values retain their original units. `tolerance` is the stopping tolerance
for this normalized objective, not a bound on control-action error; feasibility is checked
separately in the configured coordinates.

`FCSMPC(actions, prediction_horizon, prediction, objective, ...)` takes a matrix of finite
input vectors and independent callbacks: `prediction(x, u, k)` returns the next state,
`objective(x_next, u, reference, previous_u, k)` returns a scalar cost, and optional
`constraint(x_next, u, k)` returns residuals that must be nonnegative within tolerance.
Callbacks receive copies and must not capture or modify the running plant. This supports
custom finite-set models and objectives without changing the simulator.

FCS-MPC enumerates admissible action sequences in deterministic order. Its computational cost
grows exponentially with the control horizon and action count. Keep converter horizons short;
the explicit evaluation limit counts attempted complete plans, including ones rejected by an
infeasible prefix. Exhausting the budget returns a failure rather than applying a partial-search
incumbent. Equal-cost choices follow the configured action ordering, making ties reproducible.

Both algorithms return `MPCResult(success, action, inputs, states, objective, status, ...)`.
Successful `inputs` cover the full prediction horizon and `states` include the initial state
followed by every predicted endpoint. Failure results have no applicable `action`. CCS also
accepts a callable `backend(QuadraticProblem) -> SolverResult`; the backend receives read-only
coefficient copies, and the MPC checks its returned dimensions, finiteness and feasibility
against private originals. Import these records from
`peslite.control.mpc`. Additional optimization libraries are needed only by backends that
choose to use them.

The built-in predictive loops reuse the configured filter inductance, resistance and bridge
on-resistance. They hold the measured DC-link voltage across the prediction. The scalar loop
also holds its measured output voltage; the three-phase loop extrapolates the measured terminal
voltage at its connected `omega` input and integrates the resulting RL forcing analytically. They
do not construct a second plant or mutate the running electrical state. The scalar loop supports
the synchronous half bridge with its ordinary midpoint output. Reverse-port, lower-rail,
unilateral diode and other scalar bridge configurations require an appropriate custom model
and are rejected by this loop. The built-in RL predictors also require a bridge without internal
energy storage; configure a custom predictive loop when bridge capacitance belongs in the model.

CCS prediction describes bridge-average voltage, including when the plant runs in switching
mode. FCS prediction uses the chosen held binary level. Both ideal predictors omit physical
dead-time and other switching nonidealities. Constraints apply to predicted sample values,
not to every point in the electrical waveform. In particular, satisfying a predicted current
bound is not proof that switching ripple or a transient phase-current peak stays below it.

## Sampling and known delay

Optimization occurs at existing controller interrupts. ADC averaging and offsets, computation
time, command loads, physical switching and multirate integration retain their normal
[timing semantics](Bridge-and-PWM.md). Solver wall-clock execution does not advance simulation
time; `ctrl.computation` is the configured simulated computation delay. With `averaging`, the
command is zero-order held at its actuation boundary and the electrical plant remains continuous;
fixed- or variable-step integration never crosses a controller or actuation boundary.

The built-in loops compensate a known, constant control-interrupt-to-actuation delay. With
`switching` or `pwm_averaging`, their timing requires:

- `ctrl.period` is an integer multiple of `pwm.load_period`.
- The first eligible PWM load after a computation is no later than the following control interrupt.
- PWM is asynchronous and uses the `interval` update pattern.

With `averaging`, `ctrl.period` is the command-hold period and `ctrl.computation` places the
actuation boundary; carrier synchronization and PWM load periods do not constrain the control
clock.

The delay is derived from the shared control/load clocks, including their offsets. Zero
computation allows a coincident load to take the new command. Positive computation takes the
first eligible later load. Unsupported timing combinations raise configuration errors rather
than treating an overwritten or variably delayed command as applied.

Within a controller interval, the predictor applies an enabled previous command until its known
load delay and the new command afterward. A blocked previous command contributes no commanded
bridge voltage, but the measured current remains the initial state and follows the ordinary RL
dynamics; the scalar Buck predictor applies its diode-imposed zero-current boundary only if that
trajectory reaches it. The previous duty is an augmented prediction state. If the delay equals one
full control period, a new decision first affects the second predicted sample; use a horizon greater
than one. This is compensation for the configured register delay, not a general delay estimator or
prediction of individual carrier pulses.

ADC timing is independent of this execution-delay model. A held ADC conversion may precede
the control interrupt, and a window-averaged current is not an instantaneous state. The
predictor uses those supplied values without reconstructing their acquisition or averaging
delay; it does not add an observer. Account for that approximation when selecting measurement
settings and evaluating tracking or constraint accuracy.

Duty-ratio dead-time compensation is unsupported by these loops in PWM-timed models. In
particular, it would turn an FCS binary action into a fractional duty and change the finite
action set. An ideal averaged bridge has no physical dead time.

## Failure handling, diagnostics and continuation

The numerical status is `optimal`, `infeasible`, `solver_failure`, `iteration_limit`,
`evaluation_limit`, `numerical_failure` or `invalid_solution`. A custom backend may supply a
more specific failure status. Iteration/evaluation limits are deterministic work limits;
there is no wall-clock deadline.

An exhaustive FCS search reporting `infeasible` found no admissible sequence. A generic SLSQP
`solver_failure` does not prove infeasibility; it reports that the numerical solve failed.

Infeasibility, solver failure, nonfinite output and solver-work limits produce an unsuccessful
result and a configured deterministic fallback:

- `hold` uses the preceding admissible command.
- `fixed` uses `fallback_input`, validated against the input bounds and the FCS action set.
- `block` requests blocked gates through the usual command interface.

Without an explicit `fallback_input`, CCS uses the configured lower duty bounds and FCS uses
the first admissible binary action. If a solve fails after an input-bound update makes the old
held command inadmissible, `hold` uses that validated fallback command.

A fallback supplies an admissible actuator command or blocks actuation. It cannot promise
state/output feasibility when the prediction problem itself is infeasible. Unit protection
continues to override optimized and fallback commands through its normal trip mechanism.

The predictive loop's `success`, `objective`, `iterations`, `evaluations`, `solves`, `fallbacks`
and `delay` diagnostics are recorded under its configured loop name. Cumulative optimizer time is
reported as `solve_seconds` in the simulation summary.
The controller summary includes the last textual solve status. Computation cost is an observation;
wall-clock timing is not a reproducible electrical state. Previous decisions and necessary
startup/controller state use the normal named-state mechanism, so continuation restores them
together with the sampled bridge commands and ADC.
The existing restrictions on continuation during unrecorded ADC/computation windows still apply.

Reference changes and supported option changes use ordinary `set` events. Structural changes
such as controller type, sampling period, model dimensions or horizon layout require a new
configuration. See [Events Timeline](Events-Timeline.md).

## Custom algorithms

Subclass `PredictiveCurrentControl` or `ThreePhasePredictiveCurrentControl` when a custom
optimizer uses the same current model and converter boundary. `prediction(t, inputs)` returns
the augmented state, reference forecast and affine model sequence; overriding `solve(t, inputs)`
retains the ordinary graph, fallback and named-state behavior. Register the subclass with
`register_loop_type` and place it under `peslite.addons.control`, or in `PESaddons/control`
beside a simulation file. See [Extending PESLite](Extending-PESLite.md).

The generic numerical classes can also be used independently of a converter adapter. Custom
prediction models, objectives, constraints and algorithms belong behind this control boundary;
application-specific equations should not be added to the simulation scheduler.

## Verification and performance

The MPC tests use analytically solvable quadratic problems and independently enumerated finite
action sequences, including horizons above one. Converter tests exercise the existing sensing,
actuation, state-continuation and protection paths. Numerical feasibility tolerances concern the
predicted problem; converter tracking criteria must also account for model approximations and
switching ripple.

From a source checkout, run the normal fast suite and the separate cost benchmark:

```bash
python -m pytest
python tests/benchmark_mpc.py --repeats 100
```

The benchmark reports problem size, solver work and elapsed time for fixed numerical problems.
It has no machine-dependent speed threshold and is not a real-time deployment guarantee.

Initial scope excludes NMPC, specialized optional optimization backends, automatic arbitrary
plant-model extraction, observers, general variable-delay compensation, soft constraints,
robust/stochastic MPC, and formal stability or recursive-feasibility guarantees. A custom
algorithm can extend the Python control boundary without adding a separate simulation timeline.
