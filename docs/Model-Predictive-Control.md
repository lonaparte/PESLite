# Model Predictive Control

PESLite provides continuous-control-set MPC (CCS-MPC) and finite-control-set MPC (FCS-MPC)
through the existing sampled controller, ADC, PWM, protection and saved-state mechanisms.
The built-in current controllers require no `ctrl.loops` graph. They support `switching` and
`pwm_averaging`; continuous `averaging`, C++ export and Imperix export are unsupported.

## Run the examples

```bash
peslite mpc-buck-example --pwm-averaging
peslite mpc-buck-example --switching
peslite mpc-vsc-example --pwm-averaging
peslite mpc-vsc-example --switching
```

`mpc-buck-example` regulates the inductor current of a synchronous Buck using CCS-MPC. It
includes current-reference and load changes. Its output voltage follows the load and current;
this example does not add an outer voltage regulator.

`mpc-vsc-example` uses FCS-MPC to track a three-phase converter's current reference. The
controller chooses one of the eight binary three-leg commands, including both zero vectors.
Each chosen command follows the ordinary PWM register timing. Binary duties produce held
switching states, so the ideal FCS command has the same bridge voltage in both supported modes.

## Configuration and units

For a DC converter with a synchronous half bridge and an external output bus:

```yaml
ctrl:
  type: mpc_dc
  period: 50.0e-6
  computation: 1.0e-6
  references: {current: 4.0}
  options:
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

For a three-phase VSC, select `type: mpc_vsc` and, for example,
`references: {id_ref_pu: 0.4, iq_ref_pu: 0.0}`. Either adapter accepts `algorithm: ccs` or
`algorithm: fcs`.

| Setting | Meaning |
|---|---|
| `prediction_horizon` | Number of predicted controller intervals; positive integer. |
| `control_horizon` | Number of independent input decisions, no larger than the prediction horizon; omitted means the prediction horizon. The last decision is held thereafter. |
| `tracking_weight` | Nonnegative squared current-error weight; current is in amperes. |
| `input_weight` | Nonnegative squared duty-ratio weight. |
| `rate_weight` | Nonnegative squared duty-increment weight relative to the preceding command. |
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
The current limits of `mpc_vsc` bound alpha and beta independently; they are not phase-current
peak or current-magnitude limits. Input and rate penalties use dimensionless duty ratios.
Choose weights with these scales in mind; they do not normalize quantities automatically.

The VSC adapter obtains phase from the measured terminal-voltage vector and extrapolates a
sinusoidal reference at `references.omega` (rad/s, defaulting to the nominal frequency).
It does not implement a PLL or a
frequency estimator. Frequency/ROCOF protection cannot use this adapter as a frequency
measurement. Existing physical overcurrent and voltage protection remains owned by the unit.

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

Weights accept a scalar, diagonal vector or symmetric positive-semidefinite matrix. Each
`input_bounds`, `state_bounds` or `output_bounds` is a `(lower, upper)` pair; each side can
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
accepts a callable `backend(QuadraticProblem) -> SolverResult`; the MPC checks its returned
dimensions, finiteness and feasibility independently. Import these records from
`peslite.control.mpc`. Additional optimization libraries are needed only by backends that
choose to use them.

The built-in adapters reuse the configured filter inductance, resistance and bridge on-resistance.
They hold the measured DC-link voltage across the prediction. The DC adapter also holds its
measured output voltage; the AC adapter extrapolates the measured terminal-voltage vector
sinusoidally at `references.omega` and integrates the resulting RL forcing analytically. They
do not construct a second plant or mutate the running electrical state. The DC adapter supports
the synchronous half bridge with its ordinary midpoint output. Reverse-port, lower-rail,
unilateral diode and other scalar bridge configurations require an appropriate custom model
and are rejected by this adapter.

CCS prediction describes bridge-average voltage, including when the plant runs in switching
mode. FCS prediction uses the chosen held binary level. Both ideal predictors omit physical
dead-time and other switching nonidealities. Constraints apply to predicted sample values,
not to every point in the electrical waveform. In particular, satisfying a predicted current
bound is not proof that switching ripple or a transient phase-current peak stays below it.

## Sampling and known delay

Optimization occurs at existing controller interrupts. ADC averaging and offsets, computation
time, PWM register loads, physical switching and multirate integration retain their normal
[timing semantics](Bridge-and-PWM.md). Solver wall-clock execution does not advance simulation
time; `ctrl.computation` is the configured simulated computation delay.

The built-in adapters compensate a known, constant control-interrupt-to-register-load delay. Their
supported timing requires:

- `ctrl.period` is an integer multiple of `pwm.load_period`.
- The first eligible load after a computation is no later than the following control interrupt.
- PWM is asynchronous and uses the `interval` update pattern.

The delay is derived from the shared control/load clocks, including their offsets. Zero
computation allows a coincident load to take the new command. Positive computation takes the
first eligible later load. Unsupported timing combinations raise configuration errors rather
than treating an overwritten or variably delayed command as applied.

Within a controller interval, the predictor applies the previous command until its known load
delay and the new command afterward. The previous duty is an augmented prediction state. If
the delay equals one full control period, a new decision first affects the second predicted
sample; use a horizon greater than one. This is compensation for the configured register
delay, not a general delay estimator or prediction of individual carrier pulses.

ADC timing is independent of this execution-delay model. A held ADC conversion may precede
the control interrupt, and a window-averaged current is not an instantaneous state. The
predictor uses those supplied values without reconstructing their acquisition or averaging
delay; it does not add an observer. Account for that approximation when selecting measurement
settings and evaluating tracking or constraint accuracy.

Duty-ratio dead-time compensation is unsupported by these adapters. In particular, it would
turn an FCS binary action into a fractional duty and change the finite action set.

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
the first admissible binary action. After an input-bound update makes the old held command
inadmissible, `hold` uses that validated fallback command.

A fallback supplies an admissible actuator command or blocks actuation. It cannot promise
state/output feasibility when the prediction problem itself is infeasible. Unit protection
continues to override optimized and fallback commands through its normal trip mechanism.

The controller CSV fields `mpc.success`, `mpc.objective`, `mpc.iterations`, `mpc.evaluations`,
`mpc.seconds`, `mpc.solves`, `mpc.fallbacks` and `mpc.delay` report each sampled observation.
The summary includes the last textual solve status. Computation cost is an observation;
wall-clock timing is not a reproducible
electrical state. Previous decisions and necessary startup/controller state use the normal
named-state mechanism, so continuation restores them together with the PWM registers and ADC.
The existing restrictions on continuation during unrecorded ADC/computation windows still apply.

Reference changes and supported option changes use ordinary `set` events. Structural changes
such as controller type, sampling period, model dimensions or horizon layout require a new
configuration. See [Events Timeline](Events-Timeline.md).

## Custom algorithms

Use `MPCCurrentController` when a custom optimizer can use the same current-model and converter
boundary. Its `prediction(t, measurement)` method returns the augmented state, reference
forecast and affine model sequence. Its `solve(t, measurement)` method returns an `MPCResult`;
overriding that method retains the shared command, startup, fallback and state interface.
`previous_input` supplies the preceding duty vector for rate penalties. Keep any additional
state that affects later decisions in the controller's named-state methods. A custom CCS QP
backend can instead be supplied as `MPCCurrentController(cfg, topology, backend=backend)`.

Register a custom control profile through `register_control_type`, using a controller factory,
its typed options/reference schema, validation and supported modes. Place the implementation
under `peslite.addons.control`, or in `PESaddons/control` beside a simulation file. The existing
add-on discovery imports it before parsing the file; no simulation-runner changes are needed.
See [Extending PESLite](Extending-PESLite.md).

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
