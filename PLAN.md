# Small-Signal Stability Analysis Plan

Status: planning only; no small-signal analysis API is implemented by this document.

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
| `SystemLoop.to_frame()` and `SystemLoop.from_frame()` | Convert rotating alpha-beta states to and from synchronous-frame coordinates. |
| `_period_map()` in `src/peslite/solver/splitbound.py` | Extract the existing finite-difference transition-matrix calculation into a shared helper, preserving its use by split-error analysis. |
| Existing simulation and algebraic-loop evaluation | Re-solve algebraic dependencies in every perturbed run using the ordinary execution path. |

Verify that the selected period actually repeats the relevant control and PWM schedules; the
existing common-control-period calculation is a starting point, not proof for every model.
Include controller integrals and other independent sampled-control states where required for
a complete closed-loop state. Preserve held measurements, pending commands, and timing phase
through the existing restart contract. Keep discrete flags and clock counters fixed rather than
treating them as differentiable state variables.

## Implementation sequence

1. Extract the reusable period-map linearization from split-error analysis without changing
   its existing results or introducing a new simulator, integrator, or algebraic-loop solver.
2. Add modal analysis using the shared transition matrix and existing numerical dependencies.
   Keep the feature opt-in so ordinary simulations do not perform extra perturbation runs.
3. Add a small post-run API, focused verification cases, and maintained documentation; then
   remove this temporary file.

Proposed API from issue #23, subject to implementation review:

```python
sim = peslite.Simulation(params)
result = sim.run()

analysis = sim.small_signal(at=0.8)
print(analysis.mode_table)
```

## Validation and limits

- Verify the transition matrix and modes against independently known linear continuous and
  sampled-data systems, including stable, unstable, and weakly damped cases.
- Check finite-difference and integration-tolerance convergence, repeatability, and consistent
  results at equivalent phases of a settled trajectory.
- Verify state restoration and sampled/PWM timing in GFL, GFM, and multi-converter cases, with
  algebraic dependencies handled through the existing solver.
- Check that the operating state is approximately periodic in the chosen coordinates. Do not
  silently report a transient-state linearization as a settled stability result.
- Flag unsupported schedules and analysis intervals containing non-repeating command changes,
  protection transitions, or nonsmooth operating-mode changes. Interpret neutral modes and
  multipliers near the unit circle explicitly rather than forcing a binary classification.
- Preserve existing split-error diagnostics and ordinary simulation behavior. Record analysis
  cost separately from the original simulation runtime.

Completion requires the shared implementation, verification, and user documentation, not merely
the presence of this plan or the proposed API example.
