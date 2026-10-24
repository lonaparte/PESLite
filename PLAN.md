# Small-Signal Stability Analysis Addon Plan

Status: implementation and planned validation completed, pending user acceptance. The addon,
shared calculation, regression cases, external profiling tool, and maintained documentation are
present. Keep this temporary plan until acceptance; maintained usage is in
`docs/Small-Signal-Analysis.md`.

The dynamic addon also provides an opt-in simulation timeline. It consumes existing observations,
retains only unfinished windows' restart rows, shares the same finite-difference map,
and reports only changed items. It waits for a complete periodic operating state before running
perturbations; earlier checks are marked transient without modal analysis.
Static, no-time-integration analysis remains outside this implementation.
Shared snapshot readers cache values and failures, while independent output providers retain
their own failure boundaries. Observer lifecycle metadata supplies the same final notification
for normal completion and early termination; the report does not patch in a second terminal row.

Source: [Issue #23: Idea on small signal stability analysis](https://github.com/lonaparte/PESLite/issues/23),
proposed by [@NoClassDefError](https://github.com/NoClassDefError).
The period-map approach and reuse suggestions below originate from that contribution.

This is a temporary repository-root plan. Remove it after the work is implemented or superseded
by maintained documentation; retain the proposal attribution in the implementation history.
Adding this plan does not complete or close issue #23.

## Objective

Identify unstable or weakly damped modes in GFL, GFM, and multi-converter systems by linearizing
the complete closed-loop simulation, including sampled controllers, PWM updates, computation
delays, switching events, and algebraic loops. A Jacobian of the continuous plant's `Model.rhs()`
alone does not represent all of these dynamics.

## Placement and boundaries

- Implement the feature under `src/peslite/addons/functions/small_signal.py`, following the
  existing optional-functions boundary. It is an explicitly requested post-run analysis, not
  part of the simulation kernel, a control loop, or a new unit type.
- Keep operating-point selection, analysis-specific validity checks, modal calculations, and
  result presentation in the addon. Use existing NumPy functionality without new dependencies.
- Keep only the reusable loop-map contract and finite-difference transition-matrix calculation
  in the solver layer, shared with split-error analysis. Do not move stability-specific policy
  into the solver or duplicate its existing simulation capabilities inside the addon.
- Do not add `Simulation.small_signal()`, monkey-patch `Simulation`, or introduce a second
  registry, discovery mechanism, or dispatch layer. The addon calls the existing simulation
  interfaces; the normal simulation path does not import or invoke the analysis.

## Proposed method

1. Select a settled operating state and a phase-consistent common control/PWM period `T`.
2. Advance the complete simulation for one period to obtain its state-transition map:

   ```text
   x_next = F_T(x)
   Phi = dF_T/dx evaluated at the operating state x_star
   ```

3. Perturb each independent continuous-valued state coordinate, run the same one-period map,
   and estimate the corresponding column of `Phi` by finite differences. Reuse the existing
   central-difference calculation rather than adding a second implementation.
4. Calculate the eigenvalues `mu` of `Phi`. These are discrete-time multipliers; an optional
   continuous-time representation is `lambda = log(mu) / T`, with its logarithm-branch ambiguity
   documented. Report multiplier magnitude, growth/decay, and oscillation frequency so unstable
   and weakly damped modes can be identified.

The source issue contains the original equation illustrations. The expressions above restate
its one-period linearization proposal in searchable text.

## Reuse existing infrastructure

| Existing implementation | Intended reuse |
| --- | --- |
| `SystemLoop` in `src/peslite/solver/simulation.py` | Advance the assembled closed loop and restore its named states between perturbation runs. |
| `SystemLoop.to_frame()` and `SystemLoop.from_frame()` | Reuse coordinate grouping and conversion; the addon uses stationary coordinates and a complete repeating period, while split-error analysis retains its rotating-frame interface. |
| `_period_map()` in `src/peslite/solver/splitbound.py` | Extract only its reusable `Phi` calculation into a shared helper; retain `B_b`, `B_c`, and error-response calculations in split-error analysis. |
| `StateRegistry` and the existing named-state restart contract | Select and restore existing state coordinates without maintaining an addon-specific state registry. |
| Existing simulation and algebraic-loop evaluation | Re-solve algebraic dependencies in every perturbed run using the ordinary execution path. |

### Shared calculation

Extract the existing central-difference calculation into a small solver-side module, such as
`src/peslite/solver/linearization.py`. Move the existing `LoopMap` contract with it if needed to
keep imports one-way; do not add another equivalent protocol. Both split-error analysis and the
addon call the same implementation. Preserve the existing perturbation scales initially and
return enough information to reuse the unperturbed evaluation and count actual map evaluations.
Do not repeat the baseline run merely to serve the second caller.

Keep held-input perturbations (`B_b` and `B_c`), response propagation, and error summaries in
`splitbound.py`; the addon does not request or compute those quantities.

### State and execution contract

Reuse `SystemLoop.advance()` with its existing reconstruction from parameters, factories, and
named states. Initially accept that reconstruction cost rather than introducing a separate
checkpoint, reset, or lightweight simulation implementation. Every perturbation starts from the
same unperturbed state and time; analysis must not mutate the completed simulation or its result.
Respect the existing factory/rebuildability requirement and report unsupported instance-only
customizations explicitly instead of silently replacing them with built-ins.

Distinguish the complete restart row from the differentiable coordinates used for modal analysis:

- Retain electrical states, controller integrals, and numerical held/delayed values that carry
  independent memory across the selected boundary, including relevant PWM registers.
- Restore flags and discrete modes, but do not perturb them or include them as modal coordinates.
  The current `_period_map()` leaves identity columns for `frozen` and `mode` entries; those
  bookkeeping columns must not create artificial unit multipliers in the addon result.
- Use the existing state layout and coordinate grouping, with the necessary active-coordinate
  indices, rather than introducing another persistent state table. Do not remove genuine neutral
  modes merely because their multipliers are close to one.
- Do not add copies of derived values as new dynamic states or discard genuine sample/hold
  memory just because it resembles another recorded value. Check the existing restoration path
  before changing which coordinates are independent.

The implementation obtains these views from the existing `StateRegistry`: optional owner
declarations `discrete_state_names` identify lifecycle state, and `linearization_state_names`
select smooth modal coordinates without dropping other operating-state history. ADC mean
integrals remain coordinates; RMS history remains restored and checked for periodicity without
introducing observation/cache modes. The existing registry applies optional owner-defined
`periodic_state(values)` projections to compare logical history rather than storage indices.
No additional state store is created and normal runs are unchanged.

Keep the original output time grid when disabling analysis histories, because sampling or
protection timing can depend on it. Explicit-time loops declare their forcing periods through
`forcing_periods(inputs)` rather than device-name checks in the addon.

Verify that the selected period actually repeats the relevant control and PWM schedules; the
existing common-control-period calculation is a starting point, not proof for every model.
Review the existing frame grouping as well, including rotating angles and sampled quantities.
Any required corrections belong in the shared map/state contract, not in addon branches for VSC,
DCConverter, or WPT. Do not assume that every complex controller state is already a dq quantity.
Preserve timing phase through the existing restart contract rather than adding clock counters
as differentiable states. If no valid period can be established, report that limitation instead
of silently substituting a control period or switching to an averaged model.

## Implementation sequence

1. Extract and regression-test the shared transition-matrix calculation. Keep the split-error
   algorithm and diagnostics intact; verify restart, active-coordinate selection, period, and
   frame handling before interpreting its matrix as a stability result.
2. Add the post-run function in `addons/functions/small_signal.py`, using the shared matrix
   calculation and existing numerical dependencies. Start with the completed run's final named
   state, and return the matrix, coordinate names, period, modes, validity diagnostics, and
   analysis cost. Do not modify ordinary simulation execution or add a method to `Simulation`.
3. Verify the addon and existing split-error analysis, add maintained user documentation, and
   remove this temporary plan after implementation is accepted.

Proposed addon API, adapting the post-run workflow from issue #23 to the existing addon boundary:

```python
import peslite
from peslite.addons.functions.small_signal import small_signal

sim = peslite.Simulation(params)
result = sim.run()

analysis = small_signal(sim)
print(analysis.mode_table)
```

The initial API analyzes the final state. A later `at=` option may select an actual recorded
complete state row; it must not interpolate controller/PWM state, silently choose a nearby time,
or claim that an unrecorded historical state is available. Do not add extra recording or retain
the entire trajectory merely to support this addon.

## Validation and limits

- Verify the transition matrix and modes against independently known linear continuous and
  sampled-data systems, including stable, unstable, and weakly damped cases.
- Check finite-difference and integration-tolerance convergence, repeatability, and consistent
  results at equivalent phases of a settled trajectory.
- Verify state restoration and sampled/PWM timing in GFL, GFM, and multi-converter cases, with
  algebraic dependencies handled through the existing solver.
- Verify that fixed flags and modes introduce no artificial unit multipliers, while numerical
  delay/hold states and genuine neutral modes remain represented correctly.
- Check that the operating state is approximately periodic in the chosen coordinates. Do not
  silently report a transient-state linearization as a settled stability result.
- Flag unsupported schedules and analysis intervals containing non-repeating command changes,
  protection transitions, or nonsmooth operating-mode changes. Interpret neutral modes and
  multipliers near the unit circle explicitly rather than forcing a binary classification.
- Preserve existing split-error diagnostics and ordinary simulation behavior. Confirm that an
  ordinary run performs no extra recording, map evaluations, or modal calculations, and that
  using another addon does not eagerly load this analysis implementation.
- Record reconstruction, one-period simulation, and matrix/eigenvalue work separately in
  performance experiments. The central difference requires two period evaluations per active
  coordinate that is not proven overwritten, plus the shared baseline and explicit validation
  runs. Optimize measured shared costs only; do not add an alternative simulator speculatively.

Completion requires the shared implementation, verification, and user documentation, not merely
the presence of this plan or the proposed API example.

## Completed verification

- `tests/test_small_signal_ac.py` verifies the existing GFL, PSC and VSG converters and a
  shared-bus GFL/PSC pair at documented constant-command operating points. It checks complete
  period-map state, exact continuation, independently directed time-domain perturbations,
  cross-converter coupling, the online report, and matched spectra at other cycle phases.
  The native sampled controllers, computation delay and 20 kHz PWM registers remain active.
- The sampled AC map is checked under fixed-step and finite-difference refinement without
  changing its completed operating state. `tests/test_small_signal.py` additionally verifies
  fixed-step and adaptive-tolerance convergence against its existing independent linear
  reference, as well as the shared discrete map and existing bridge-mode regressions.
- `tests/benchmark_small_signal.py` measures the actual addon call graph externally. Model
  reconstruction, period simulation, matrix assembly, eigensystem calculation and other
  orchestration are disjoint categories. Regression checks verify their accounting and the
  agreement between actual reconstruction/advance counts and reported map evaluations.
  Profiling experiments covered all four AC cases; no timing logic was added to the runtime.
- Verification required no new production model, solver, state store or device-specific
  analysis path. Maintained instructions and operating-point limits are recorded in
  `docs/Small-Signal-Analysis.md`. Static analysis and historical `at=` selection remain outside
  this plan's scope. Removing this temporary plan remains an acceptance-time cleanup action.
- Shared linearization skips only structurally proven zero columns, retaining the complete
  matrix and its zero modes. Owner declarations use the existing state registry and graph
  schedules; exhaustive comparisons cover clock ratios, phase, bridge models, light loads and
  custom loops in `tests/test_linearization_pruning.py` and the AC verification module.
