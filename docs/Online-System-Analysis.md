# Online System Analysis

The online system analysis addon evaluates the operating state or recorded response of an ordinary
PESLite simulation. It does not replace the plant, insert a test source or use a separate
reduced-order model. Sampled controllers, ADC and PWM state, computation delays, switching events
and algebraic loops therefore remain part of the analyzed system.

For transfer identification from deliberately excited experiments, use
[Offline System Analysis](Offline-System-Analysis.md).

## Choose an analysis

| Goal | Interface | Input |
| --- | --- | --- |
| Record stability changes during a run | `small_signal_report` or `peslite ... --report` | A fresh `Simulation` |
| Analyze the final recurring operating state | `small_signal` | A completed restartable `Simulation` |
| Check whether the final state is recurring | `detect_operating_point` | A completed restartable `Simulation` |
| Measure step and tracking performance | `dynamic_performance` | Recorded time, response and reference arrays |

The first three interfaces use the complete named restart state and the same `SystemLoop` period
map. `dynamic_performance` reads an existing waveform and does not advance the simulation. None of
these functions runs during an ordinary simulation unless explicitly requested.

## Online simulation report

Generate a report beside the ordinary simulation output:

```bash
peslite buck-example --report
```

The command writes `buck-example_report.json`; the name comes from the input file rather than the
output directory or bridge preset. If the addon is unavailable, PESLite warns and completes the
ordinary simulation. Plotting and reporting may be requested together.

The Python interface exposes the same workflow:

```python
from peslite.addons.functions import small_signal_report

sim = peslite.Simulation(params)
report = small_signal_report(
    sim,
    times=[0.001, 0.01, 0.02],
    interval=0.005,
    watch=["output.u", "units.converter.ctrl.references.voltage"],
    out_dir="output/buck",
    name="buck-example",
)
result = sim.result
```

`small_signal_report` runs the main simulation once. The start, end and simulation-file event
times are always checked. `times` adds exact analysis points, while `interval` adds a regular
analysis schedule; neither implies analysis at every PWM edge. A requested point within the
runtime tolerance of an existing event uses that boundary. A genuinely new point splits an
integration interval and can therefore produce the usual numerical step-size effects.

### Watched values and report changes

The report contains a `small_signal_stability` section. Its first entry is complete; later entries
contain only values that changed at a checked time. Event-written parameters and command values
are included automatically. `watch` accepts existing progress-watch signal or state names and
full scalar parameter paths. `change_tolerance` controls the absolute and relative comparison
tolerance.

Watched values are read from the snapshot at which an event is applied. The report does not replay
events, interpolate controller state or maintain a second command timeline. A protection trip is
recorded at its detected transition. Early termination retains readable values at the actual stop
time; a runtime failure marks the analysis unavailable with the simulation stop reason.

### Stability states

A checked point has one of three analysis outcomes:

- a recurring state produces modal values and a stability classification;
- a nonrecurring state or a window crossing a one-off event is `transient`;
- a state that cannot be reconstructed or differentiated is `unavailable` with a reason.

The report never silently settles or extends the main simulation. It first advances the
unperturbed state by one period. Perturbation runs begin only after that endpoint passes the
recurrence checks. Perturbation-dependent discrete modes invalidate the point rather than being
treated as smooth coordinates.

For recurring states, `analysis.amplification` is the largest singular value of the scaled
one-period map. It measures transient amplification in the selected coordinate norm, not stored
energy. A stable periodic state may still have amplification greater than one. Dominant modal
growth and unsigned frequency are reported only when a valid map exists.

Report times are sample times, not transition-location guarantees: a change between two checks is
neither located nor ruled out.

### Reuse, output and cost

The report reuses snapshots and output values already produced by the main run. Watched controller
columns enable only the required controller logging; plant-only and controller-only watches do not
force each other's output. Read results, including failed optional calculations, are shared with
CSV output at the same snapshot.

Only unfinished analysis windows retain restart rows. Once an endpoint arrives, its matrix is
calculated and the row is released; the waveform history is not copied into the report. An
unperturbed endpoint at `t + period` is reused as the finite-difference baseline. With `n` active
coordinates, a checked recurring point therefore needs at most `2*n` additional period advances,
and proven overwritten coordinates reduce that count further.

The returned report exposes `checks`, `reused_baselines`, `evaluations` and `wall_time`.
`wall_time` measures matrix analysis, while `sim.result.wall_time` includes the main run and any
analysis performed during it; the two overlap and must not be added. A failed check can make the
exact evaluation count unavailable.

With output histories disabled, `out_dir=None` avoids files. Otherwise the report is written next
to the simulation output. `report.write(directory, name="case")` writes it later. A new run clears
its own previous `<name>_report.json`; unrelated report files remain untouched.

## Operating-point validation

Before interpreting a modal result, check that the final state actually repeats:

```python
import peslite
from peslite.addons.functions import detect_operating_point

sim = peslite.Simulation(peslite.load("buck-example"))
sim.run(out_dir="output/buck")
point = detect_operating_point(sim)
print(point.kind, point.periodic_error, point.valid)
print(point.to_dict())
```

`detect_operating_point` compares the completed state with a reconstructed one-period endpoint and
samples two interior positions. A constant equilibrium is `dc`; a recurring orbit is `periodic`.
Failed recurrence, changing discrete state, an unfinished or tripped run, an active startup ramp,
a one-off event inside the period and reconstruction errors produce `unconverged` with a reason.

The default scaled recurrence tolerance is `1e-3`. An explicit `period` may be supplied when it
repeats every declared schedule. Without a declared schedule, the configured output period is used
as the comparison interval. This check performs at most three reconstructed advances, does not read
waveform histories and does not modify the completed simulation. A valid operating point confirms
recurrence; it does not by itself establish asymptotic stability.

## Final-state modal analysis

Analyze the final complete state after the operating point has settled:

```python
from peslite.addons.functions import small_signal

analysis = small_signal(
    sim,
    relative_step=1e-4,
    periodic_tolerance=1e-3,
    multiplier_tolerance=1e-5,
)
print(analysis.mode_table)
print(analysis.stability)
```

`small_signal` works without state-history output and does not retain an in-memory trajectory.
Historical point selection and automatic trimming are intentionally absent; use the online report
when several simulation times are required.

### Period and state coordinates

The map period repeats every declared control, ADC, PWM-load, carrier, source-waveform and explicit
loop-forcing schedule. An AC source period therefore remains part of the system period even when a
controller runs faster. Automatic selection searches up to 64 times the longest declared period.
Unsupported or undeclared time dependence is reported rather than approximated. Autonomous
continuous models and custom periodic forcing can supply `period` explicitly.

The map uses stationary named coordinates. Complex states retain `.re` and `.im` coordinates;
angular differences use their declared angular topology. Boolean, integer and lifecycle states are
restored and checked for recurrence but excluded from the smooth modal coordinates. Owners may
declare:

- `discrete_state_names` for float-valued lifecycle state;
- `linearization_state_names` for independent smooth coordinates;
- `periodic_state(values)` for logical equivalence of alternate storage layouts;
- `overwritten_state_names(period)` for coordinates proven unread before being overwritten.

These declarations refine one existing state registry; they do not create a second snapshot or
restart format. Histories that affect future behavior remain part of recurrence checking even when
their redundant storage indices are excluded from the smooth map.

### Result fields

`SmallSignalResult` provides:

- `phi` and `coordinates`, the one-period transition matrix and its named order;
- `multipliers`, ordered by decreasing magnitude;
- `exponents`, principal-logarithm growth rates and frequencies;
- `mode_table` and `stability`;
- `at`, `period`, `periodic_error`, `evaluations`, `wall_time` and `diagnostics`.

`multiplier_tolerance` defines the margin around the unit circle used for `stable`, `unstable` and
`near-unit-circle`. Equivalent frequencies are ambiguous modulo `1 / period`; principal-logarithm
frequencies are not an unaliased reconstruction of faster modes. Zero multipliers are retained.

## Time-domain performance

Use an ordinary recorded response for step and tracking metrics:

```python
from peslite.addons.functions import dynamic_performance

metrics = dynamic_performance(
    sim.result.t,
    sim.result.plant["output.u"],
    target=24.0,
    step_time=float(sim.result.t[0]),
    unit="V",
)
print(metrics.to_dict())
```

`time` must be strictly increasing and match the response length. `reference` may be a constant or
sampled array; an explicit `target` is preferred for comparisons. Without one, the final reference
or response tail supplies the target and the result is flagged as inferred. `initial` overrides the
pre-step value, and `step_time` must be a recorded sample.

| Metric | Definition |
| --- | --- |
| `rise_time` | Interpolated first 10% to first 90% crossings. |
| `overshoot` | Peak excursion beyond the target as a percentage of the signed step. |
| `settling_time` | First sample after the final tolerance violation. |
| `steady_state_error` | Time-weighted tail mean minus the target. |
| `ripple_peak_to_peak` | Maximum minus minimum over the final tail. |
| `tracking_rmse`, `tracking_max_abs` | Tracking error after the step. |

`settling_tolerance` defaults to 2% of the step; `absolute_tolerance` can impose a larger physical
band. `tail_fraction` defaults to the final 10% of the post-step interval. Missing or invalid
metrics are `None` with explicit flags rather than fabricated values.

## Accuracy and limitations

Finite-difference accuracy depends on the perturbation size, integration settings and algebraic
solver tolerance. A smaller perturbation can be less accurate when numerical residuals dominate.
The analysis does not tighten solver tolerances or replace the configured algebraic-loop method.

The modal interfaces reject unfinished or tripped runs, active ramps, nonrecurring states,
incomplete advances, one-off events inside the period and discrete-mode changes under perturbation.
They do not certify differentiability at a nonsmooth boundary. Recurring switching is handled by
the ordinary event loop and is not rejected merely because switching occurs.

Each perturbation reconstructs the model through the existing factories and named-state restart
path, leaving the completed run unchanged. Multirate configurations use the unsplit reference map
with the configured base solver and tolerances; the result is a physical-system map, not a spectrum
of the multirate splitting approximation.

## Verification

Run the focused online-analysis tests with the test dependencies installed:

```bash
python -m pytest tests/test_operating_point.py tests/test_dynamic_metrics.py
python -m pytest --run-full tests/test_small_signal.py tests/test_small_signal_ac.py
```

`tests/benchmark_small_signal.py` separates reconstruction, period advances, transition-matrix
assembly, eigensystem work and orchestration when profiling a supplied simulation file. Its timing
is diagnostic and does not enter normal simulation execution.
