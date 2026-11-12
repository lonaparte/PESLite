# Analysis Functions

PESLite provides opt-in, nonintrusive measurements in `peslite.addons.functions`. The simulator,
control graph, and event scheduler do not import the analysis modules during ordinary execution.
The functions consume a completed restartable simulation or recorded numeric signal arrays. They
do not change the solver, override the physical plant, or invent unrecorded measurements.

## Steady-state operating point

```python
import peslite
from peslite.addons.functions import detect_operating_point

sim = peslite.Simulation(peslite.load("examples/buck-example.pes"))
sim.run(out_dir="output/buck")
point = detect_operating_point(sim)
print(point.kind, point.periodic_error, point.valid)
print(point.to_dict())
```

`detect_operating_point` reuses the same complete `SystemLoop` period map, named restart state,
logical periodic-state projection, angular differences and state scaling as the existing
[online stability analysis](Online-Stability-Analysis.md). It compares the final state with a
reconstructed one-period endpoint and samples two interior positions to distinguish a static
DC equilibrium from a recurring orbit. A recurring orbit is labeled `periodic`; a static
equilibrium is labeled `dc`. Failed recurrence, changing discrete states, unfinished or tripped runs, active
startup ramps, crossing one-off events, and reconstruction errors are `unconverged` with a reason.

An explicit `period` can override the common schedule period if it repeats all declared
schedules. When the system has no declared periodic schedule, the configured output period is
used as a comparison interval. `tolerance` (default `1e-3`) is a dimensionless, scaled state
residual. This is an **operating-point recurrence check**, not an eigenvalue-based stability
classification, and `valid` does not establish asymptotic stability.

The function runs at most three reconstructed advances after the completed run. It does not
read waveform histories or modify the original final state. For local modal stability of a
recurring state, use the existing `small_signal` addon instead.

## Time-domain performance

```python
from peslite.addons.functions import dynamic_performance

result = sim.result
metrics = dynamic_performance(
    result.t,
    result.plant["output.u"],
    target=24.0,
    step_time=float(result.t[0]),
    unit="V",
)
print(metrics.to_dict())
```

Supply strictly increasing `time` in seconds and an equally sized `response`. An optional
constant or sampled `reference` enables tracking-error RMS and maximum error; `target` provides
an explicit setpoint; otherwise the last reference or response tail determines the target.
Use a fixed target for a step-response comparison. `initial` optionally specifies the pre-step
value; by default the response at `step_time` is used. `step_time` must be a recorded sample.

The result contains:

| Metric | Definition |
| --- | --- |
| `rise_time` | Linearly interpolated first 10% to first 90% response crossings, seconds. |
| `overshoot` | Maximum excursion beyond the target, divided by signed step magnitude, percent. |
| `settling_time` | Earliest post-step sampled instant after the last tolerance violation, seconds. |
| `steady_state_error` | Tail-average response minus target, in the declared signal unit. |
| `ripple_peak_to_peak` | Maximum minus minimum response in the final tail, signal unit. |
| `tracking_rmse`, `tracking_max_abs` | Reference-following error after the step, signal unit. |

`settling_tolerance` defaults to 2% of the step, with an optional `absolute_tolerance`
(error band uses the larger). `tail_fraction` defaults to 10% of the post-step interval;
`final_value` is a time-weighted trapezoidal average over this tail. The response and
reference must have been recorded at compatible timestamps; different PESLite CSV files can
have different clocks. Missing metrics are `None`, marked invalid per metric in `to_dict()`.
A failed settling check, changing command after the step, missing rise threshold, or unknown
step amplitude is explicitly flagged. A target inferred from the response is flagged rather
than presented as an independently known command.

## Frequency-domain measurement validation

```python
from peslite.addons.functions import frequency_response, frequency_sweep

# Example: sample arrays with the same timestamps, captured after transients decay.
point = frequency_response(
    time, excitation, response,
    frequency=50.0,
    input_unit="A", output_unit="V",
)
print(point.valid, point.flags, point.gain)
point.write_json("output/measurements/50hz.json")

# Each input array must be obtained from a separate steady-state single-tone experiment.
spectrum = frequency_sweep([point_20hz, point_50hz, point_100hz])
spectrum.write_json("output/measurements/response.json")
```

`frequency_response` fits the fundamental, DC offset, and supported harmonics by least squares
on the original timestamps. Phasors use `cos` as the real axis and `sin` as the negative
imaginary axis, referenced to the first sample. The complex transfer ratio preserves phase.
Both halves of the time window are fitted separately to check phasor and DC-offset convergence;
other validity checks include excitation amplitude, harmonic distortion (THD), residual-noise
SNR in dB, observed sampling jitter, conservative Nyquist frequency based on the **largest**
sampling gap, and available cycles (`frequency * window_duration`).

Validity limits can be tuned (`min_cycles`, `min_excitation_amplitude`, `max_steady_drift`,
`max_distortion`, `min_snr_db`, `max_sample_jitter`); they are measurement acceptance criteria,
not assertions of linearity or model stability. Nonuniform sampling, insufficient harmonic
bandwidth, aliasing risk, nonconvergence, distortion, noise and short windows are flagged.
An invalid point retains its diagnostic phasors and conditions but sets `gain=None`; it is
never silently accepted. One sinusoidal excitation does not identify nonlinear cross-frequency
coupling. Measure each requested frequency independently after it has settled, then call
`frequency_sweep` with strictly increasing frequencies and consistent physical units.

## Frequency-dependent impedance matrices

```python
from peslite.addons.functions import impedance_matrix

# For each frequency, collect independent sinusoidal current injections and
# corresponding complex voltage phasors. Shapes: [frequency, experiment, port].
impedance = impedance_matrix(
    frequencies_hz, current_phasors, voltage_phasors,
    input_names=["id", "iq"], output_names=["vd", "vq"],
    measurement_valid=validated_experiments,
)
impedance.write_json("output/measurements/impedance.json")
```

The transfer matrix is fitted to `V = Z I` by complex least squares. The axes of
`values` are `[frequency, voltage_output_port, current_input_port]`, and `unit` is `ohm`.
The caller supplies the measured phasors, their common coordinates, and the validity of
individual experiments; this function **does not** derive impedance or perform injection
runs automatically. Independent excitation directions are necessary to resolve the ports.
Rank deficiency, ill-conditioning, missing or invalid experiments, and large fit residuals
are flagged per frequency. Rejected matrices contain `NaN` in memory and `null` values in
JSON. The output includes input/output port order, complex real/imaginary pairs, frequency
units and numerical fit conditions. Frequency-sweep and impedance JSON outputs use stable,
non-NaN serialization (`allow_nan=False`).

## Reproducible identification excitations

The helpers `reference_prbs`, `reference_multitone` and `reference_chirp` generate the exact
same native table schema as `reference_table`, `reference_csv` and `reference_expression`.
See [Reference Signals](Reference-Signals.md#identification-excitations) for examples. They
can be embedded in `signals` and referenced from the existing controller graph with no
additional runtime state. Existing native C++ and Imperix restrictions on dynamic tables
continue to apply.

## Verification

The focused cases exercise real/complex phasor accuracy, noise/distortion and sampling rejection,
periodic-state and discrete-state detection, PRBS reproducibility, matrix rank and fit validation,
step and tracking metrics, and JSON schema consistency. Run:

```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_identification.py tests/test_operating_point.py \
    tests/test_dynamic_metrics.py tests/test_frequency_measurements.py
python -m pytest
```
