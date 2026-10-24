# Small-Signal Stability Analysis

Small-signal analysis is an optional Python addon under `peslite.addons.functions`.
It uses the existing complete simulation loop, including control state, sample/hold memory,
computation delay, PWM registers, electrical integration, and algebraic-loop solves. It does not
add a method to `Simulation`, run automatically, or replace a switching model with an averaged one.

The period-map approach was proposed by [@NoClassDefError in issue #23](https://github.com/lonaparte/PESLite/issues/23).

## Use

From a source checkout:

```python
import peslite
from peslite.addons.functions.small_signal import small_signal

params = peslite.load("examples/buck-example.pes", **{"simulation.t_end": 0.05})
sim = peslite.Simulation(params)
result = sim.run(out_dir="output/buck")

analysis = small_signal(sim)
print(analysis.mode_table)
print(analysis.stability)
```

Only the final complete named state is required. The addon also works when state-history output
is disabled; it does not add snapshots or retain an in-memory trajectory. Historical `at=`
selection and automatic operating-point trimming are not part of this interface.

## Time-based simulation report

From the command line:

```bash
peslite buck-example.pes --report
```

The report is `buck-example_report.json` in the run's output directory. Its name always comes
from the input file's stem, independently of `--out` and bridge presets. The addon is imported
only with `--report`; if it is absent, PESLite warns and continues the ordinary simulation.
Inspection/export options cannot be combined with `--report`; plotting can.

Use a fresh simulation to generate a timeline alongside its ordinary output:

```python
from peslite.addons.functions.small_signal import small_signal_report

sim = peslite.Simulation(params)
report = small_signal_report(
    sim,
    times=[0.001, 0.01, 0.02],
    watch=["output.u", "units.converter.ctrl.references.voltage"],
    out_dir="output/buck",
    name="buck-example",
)
result = sim.result
```

This function runs the main simulation once. Start, end, and simulation-file event times are
always checked; `times` adds exact observation points. Alternatively, or additionally, use
`interval=0.005` to check every 5 ms from the simulation start. Detection is not performed at
every PWM edge. A protection trip is reported at the runtime's detected transition, including
when the main simulation is configured to continue after a trip.
Normal and early termination use the same final observation. The report retains any valid watched
values at the actual stop time, even when that time was not a requested check; a runtime failure
marks the analysis unavailable with the same stop reason as the simulation summary.

The JSON report, `<name>_report.json`, contains a `small_signal_stability` section with the
initial row and then only changed items, grouped by time. Event-written parameter paths and command values
are included automatically. `watch` uses existing progress-watch signal/state names or full
scalar parameter paths, without signal renaming or a device whitelist. For example, `output.u`
is a public plant signal; `units.converter.ctrl.references.voltage` is a parameter path. Use
`change_tolerance` to set the absolute/relative tolerance for reporting numerical changes.
Continuously varying watched quantities can therefore produce a row at every selected check.
Parameter values come from the snapshot retained when the runtime applies a checked event.
Connection commands use the existing `Scenario` interpretation, including inferred initial
values; the report does not replay events or maintain a separate parameter/command state.

Periodic operating states receive the same stability classification as the final-state addon.
Non-periodic states, changing nominal discrete modes, and windows crossing a one-off file event
instead receive `transient`, with a reason and no matrix, modal values, or amplification.
Only the unperturbed period is checked at those points; perturbation runs begin when a selected
check establishes a periodic operating state. The report does not extend the main simulation
or silently settle its operating point. For a periodic state, `analysis.amplification` is the
largest singular value of the map after coordinate scaling. The scale for each coordinate is
the greater of its existing perturbation floor and its initial/final magnitude, used identically
at both boundaries; angles use their angular scale. This is not an energy norm. Periodic states
can also have transient amplification greater than one.
Unavailable checks, including perturbation-dependent discrete modes, carry an explicit reason
and clear preceding modal values. Dominant modal growth and unsigned oscillation frequency are
reported only for periodic states. `analysis.*` is reserved for report metrics.

Report times are detection times. Changes between checks are not located or ruled out, and no
controller or PWM state is interpolated to invent an unrecorded historical point. The existing
final-state `small_signal()` remains a separate, strict periodic-state entry point; neither
function implements a no-time-integration static analysis.

Observation times within the runtime's existing time tolerance of a native event use that event
boundary, rather than shifting it by floating-point roundoff. Truly additional observation times
split an integration interval, so normal numerical step/tolerance effects can still occur.

### Reuse and cost

The report consumes the state already read at main-run snapshots. Watched signals are read only
when requested, and only explicitly watched control-log columns enable controller logging when
signal histories and progress watches are disabled. A controller-only watch does not read plant
signals, and a plant-only watch does not read controller logs. Baseline-only endpoints do not read watched
values. Read results, including failed calculations, are cached once per snapshot and shared by
CSV output and watches. Numerical observation failures produce a warning and unavailable values
only for that provider. Controller logs, plant signals, limit statistics and energy observations
are isolated at their shared output boundary: a failed log does not discard a valid plant row,
physical snapshot, or energy observation. Missing declared controller-log values are NaN in CSV
and null in JSON rather than reusing stale values.

Only unfinished analysis windows retain complete restart rows. As each endpoint arrives, its
matrix is calculated through the shared map and its restart row is released; endpoints are not
stored as a second history. The ordinary waveform history is not loaded.
The optional observer can request an exact endpoint
on the existing event grid, without creating CSV rows or extra control/protection interrupts.
When it coincides with an existing runtime event, no extra integration boundary is needed.
Python observation callbacks use
`observe(t, states, read_values, *, final, stop_reason)`. The keyword metadata distinguishes a
terminal snapshot from a scheduled check or a baseline-only endpoint. A final callback must not
request another time. If the terminal physical snapshot itself is unavailable, `states` is
`None`, watched values are empty, and the final reason is still delivered. Callback errors still
propagate; only numerical output calculations are treated as optional failures.

An endpoint at `t + period` supplies the unperturbed result directly. With `n` active coordinates,
that check needs `2*n` extra period advances rather than `1 + 2*n`. If the endpoint is beyond the
main run, the baseline is advanced once through the existing map. Before periodicity is established,
only that baseline is needed, with no perturbation advances. Multirate runs use the shared
unsplit reference map and cannot reuse the split run's endpoint; the report identifies the
reference integration. Perturbed runs are still required, and unchanged parameters alone do
not justify reusing a matrix computed at a different state.

`report.checks`, `report.reused_baselines`, `report.evaluations`, and `report.wall_time` expose
the analysis cost. `evaluations` is unavailable if a failed check may have performed additional
advances; `wall_time` measures the matrix-analysis portion, while `sim.result.wall_time` measures
the main run including observation collection and analyses completed during the run. These times
overlap and must not be added. No matrices or restart rows remain in the returned
report. With all output histories disabled, `out_dir=None` avoids files; otherwise reports are
written next to the ordinary simulation output. `report.write(directory, name="case")` can also
save the report later. The Python API defaults to `name="simulation"` when no input-file name is
provided; the CLI passes the actual input stem automatically. No Markdown report is generated.
A new run clears its previous `<name>_report.json` along with its other managed outputs, including
when no new report is requested or the addon is unavailable. Python callers reusing a directory
should pass the same `name` to `Simulation.run()` and `small_signal_report()`; unrelated report
files are preserved.

## Period and state coordinates

The addon selects a common period of the declared control, ADC, PWM-load, carrier, source-waveform,
and explicit loop-forcing schedules. It uses stationary coordinates, keeping the existing real
state names and complex `.re`/`.im` columns. In particular, an AC source's electrical period must
repeat; a short control period is not automatically a valid full-system period.

Specify `period` for an autonomous continuous model with no declared period, or for additional
custom periodic forcing. An explicit period must repeat every declared schedule:

```python
analysis = small_signal(sim, period=0.02)
```

Automatic common-period selection uses the existing search limit of 64 times the longest declared
period. An unsupported schedule is reported rather than approximated as another clock. Custom
explicit-time loop laws may implement `forcing_periods(inputs)`, returning their forcing periods
in seconds from the existing wired loop inputs; `ResonantModulation` uses this interface. A caller
must account for any undeclared time dependence when supplying a period.

The initial and one-period final logical states must agree within `periodic_tolerance`, including
protection feedback history even when it is excluded from the smooth modal coordinates. Angular differences
are taken modulo a full turn, using the control graph's angle-port declarations; unwrapped angle
magnitude does not relax the periodicity tolerance. Numerical sample/hold and PWM memory remains
in the map. Boolean/integer modes and owner-declared lifecycle state, such as protection timers
and startup bookkeeping, are restored but excluded from modal coordinates.

The existing `StateRegistry` supplies that discrete-state view. A custom named-state owner may
declare `discrete_state_names` for float-valued lifecycle state; it does not need another registry
or a separate snapshot representation. The addon does not remove genuine unit multipliers merely
because they are near the unit circle.

An owner may also declare `linearization_state_names` to select independent smooth coordinates
without dropping other restart or protection history. For example, ADC window-mean integrals remain
coordinates; RMS history does not introduce cached-sum or ring-buffer modes into the smooth
closed-loop spectrum. Those histories are still restored and checked for recurrence, and protection
still runs. The registry's optional owner method `periodic_state(values)` defines logical equivalence
using existing named columns: ADC compares a full RMS ring in chronological order, not by its
rotating storage index. A filling or changing history therefore prevents a periodic verdict, while
an equivalent storage rotation does not. Neither view creates another state table or changes restart data.

## Options and results

```python
analysis = small_signal(
    sim,
    relative_step=1e-4,
    periodic_tolerance=1e-3,
    multiplier_tolerance=1e-5,
)
```

| Option | Meaning |
| --- | --- |
| `period` | Map period in seconds; inferred from declared schedules when not given. |
| `relative_step` | Central-difference perturbation relative to each coordinate's value/scale floor. |
| `periodic_tolerance` | Maximum scaled state residual after the unperturbed one-period advance. |
| `multiplier_tolerance` | Margin around the unit circle used by the summary classification. |

`SmallSignalResult` provides:

- `phi` and `coordinates`: the differentiable map matrix and its named row/column order;
- `multipliers`: eigenvalues, ordered by decreasing magnitude;
- `exponents`: principal-logarithm equivalents in inverse seconds;
- `mode_table`: a printable table of multipliers, growth rates, and frequencies in Hz;
- `stability`: `stable`, `unstable`, or `near-unit-circle` under the requested margin;
- `at`, `period`, `periodic_error`, `evaluations`, `wall_time`, and `diagnostics`.

Equivalent frequencies are ambiguous modulo `1 / period`; the principal-logarithm output is not
an unaliased reconstruction of high-frequency modes. Zero multipliers are retained, with growth
rate `-inf`. Near-unit-circle results require interpretation, not automatic rejection or removal.

## Validation and cost

The baseline is checked before perturbation runs and reused by the shared finite-difference
implementation. With `n` active coordinates, analysis takes `1 + 2*n` one-period evaluations.
Each evaluation rebuilds through the existing parameters/factories and named-state restart path;
it does not mutate the completed run. Ordinary simulations do no extra recording, linearization,
or eigensystem work, and unrelated optional functions do not import this analysis module.

Check convergence with respect to perturbation size and integration settings before interpreting
weakly damped modes. A smaller perturbation is not always more accurate: integration and existing
algebraic-solver residual tolerances can dominate the differenced signal. The addon does not
silently tighten solver settings or introduce another algebraic-loop method.

The addon rejects unfinished runs, tripped operating states, active startup ramps, simulation-file
events inside the analysis interval, non-periodic states, incomplete period advances, and changes
to saved discrete modes under perturbation. These checks do not certify differentiability of
arbitrary custom code or operation exactly at a nonsmooth boundary. Switching within a recurring
cycle is handled by the ordinary event loop, not rejected merely because switching occurs.

Ready-made custom instances cannot be reconstructed; provide the existing `parts` factories
instead. For multirate configurations, the existing `SystemLoop` uses an unsplit reference solve
with the configured base solver and tolerances. It does not identify numerical modes of the
multirate splitting approximation; the result diagnostics state this distinction.

### Reproducible verification

`tests/test_small_signal_ac.py` exercises the shipped GFL, PSC and VSG converters and a shared-bus
GFL/PSC pair through the ordinary simulation and online report. The AC cases retain 20 kHz
`pwm_averaging`, the computation delay, controller sample/hold state and PWM shadow registers.
They use a 50 ms connection ramp followed by constant commands, with no intervening reference
or grid-frequency steps. The test operating points are 0.3 pu DC input for GFL, 0.5 pu active
power for PSC, 0.2 pu for VSG, and 0.3 pu per converter in the pair. Four seconds of time-domain
settling precede analysis. These are verification conditions, not new defaults or a stability
guarantee for other operating points. Existing Buck checks separately cover all three bridge modes.

The checks cover exact named-state continuation, independent perturbation directions, cross-unit
coupling, online/final-state agreement, and matching multiplier sets a quarter-cycle and a full
cycle later. Matrices at different phases need not be elementwise equal; their spectra are
matched without assuming an eigenvalue ordering. Fixed-step refinement differentiates the same
completed physical state rather than conflating integration error with a changed operating point.
The existing independent linear reference also checks fixed-step and adaptive-tolerance
convergence; finite-difference refinement is checked on the actual sampled AC map.

Run these checks from a checkout with the test dependencies installed:

```bash
python -B -m pytest tests/test_small_signal.py tests/test_small_signal_ac.py
```

`tests/benchmark_small_signal.py` profiles any supplied simulation file through the same addon.
For the GFL verification operating point:

```bash
OPENBLAS_NUM_THREADS=1 python -B tests/benchmark_small_signal.py examples/gfl-example.pes \
  --set simulation.t_end=4.0 \
  --set events.connect_vsc.t=0.0 \
  --set events.connect_vsc.ramp=0.05 \
  --set units.vsc.dclink.source.i_pu=0.3 \
  --set units.vsc.ctrl.loops.DCVoltageControl.feedforward=0.3
```

The tool reports model reconstruction, one-period simulation, transition-matrix assembly,
eigenvalue calculation, and remaining validity/orchestration work as disjoint costs. Period
advances are subtracted from the matrix helper's inclusive profiler time, so they are not
counted twice. The measured reconstruction/advance counts must agree with the addon's evaluation
count. `--repeats` repeats analysis of the same completed state; `--period` and `--relative-step`
use the ordinary analysis options. These are profiler-instrumented measurements, not portable
wall-time guarantees. No profiler, extra timers or performance assertions enter normal simulation.

The same solver-side transition-matrix implementation serves split-error analysis. Its held-input
perturbations and error propagation remain in `splitbound.py`, outside the addon. Split-error
estimates retain their existing local common-clock period; waveform recurrence is requested only
for the stability analysis, so an off-nominal source does not expand ordinary split diagnostics
into long beat-period simulations.
