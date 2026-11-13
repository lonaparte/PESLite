# Offline System Analysis

The offline system analysis addon identifies a response from deliberately excited simulations or
recorded waveforms. Each experiment uses the ordinary PESLite plant, controller, solver, event
schedule and output schema; the analysis runs after the required data has been recorded. Nothing is
inserted into an ordinary simulation unless the case explicitly configures an excitation addon.

For operating-point, modal and time-domain analysis of an ordinary run, use
[Online System Analysis](Online-System-Analysis.md).

## Identification workflow

Use the same sequence for scalar transfer functions, impedance matrices and loop gain:

1. choose the physical port or control edge and define its sign and coordinates;
2. run a fresh experiment for each independent excitation;
3. select a stationary recorded window after transients have decayed;
4. validate sampling, excitation strength, convergence, distortion and noise;
5. combine only valid points into a sweep, matrix or margin report.

Successful simulation does not imply a valid measurement. Every measurement result retains
diagnostic conditions and flags; an invalid point has no numerical gain.

## Excitations

### Reference signals

`reference_prbs`, `reference_multitone` and `reference_chirp` generate the same serializable table
schema as the ordinary reference helpers. Bind the returned table through `signals` and the existing
controller graph; no additional runtime state is introduced. See
[Reference Signals](Reference-Signals.md#identification-excitations).

Use separate settled single-tone experiments for ordinary frequency sweeps. PRBS and multitone
signals are useful when their bandwidth and coherence assumptions are understood, but they do not
make a nonlinear or time-varying system behave as independent scalar frequency points.

### Physical port injection

The `port_injection` component can add a scalar or three-phase electrical stimulus. The native
circuit view represents series and shunt placement directly:

```text
@circuit
upstream -- series_stimulus -- downstream
shunt_stimulus -- pcc
```

For the corresponding component parameters, a voltage injection names `inlet` and `bus`; a current
injection names only `bus`. Omitting `inlet` from a voltage injection requests automatic insertion
at a bus that already joins exactly two ports. The bundled `series-port-injection-example` and
`port-injection-example` are complete simulation files.

A positive series source enforces `V_bus - V_inlet = +injection`. A positive shunt stimulus flows
into the network. Since component port current is positive into the component,
`stimulus.current = -stimulus.injection` for the shunt source.

The injection component is only a source. For a conductance perturbation around voltage `V0`, keep
the nominal resistor and add its small-signal Norton current `-V0 * delta_G(t)`:

```yaml
elements:
  load: {type: resistor, bus: output, r: 10.0}
  load_probe: {type: port_injection, kind: current, bus: output,
               amplitude: -0.048, frequency: 20.0}
```

This approximation is local to the chosen operating voltage; use an actual component change for a
large-signal load event.

Three-phase sources use the complex peak-scaled stationary space vector. `axis: alpha` or `beta`
selects a stationary axis. `axis: d` or `q` rotates with `frame_frequency`, and `axis: positive` or
`negative` selects the corresponding sideband. Sequence excitation permits one zero-bias tone.
`tones`, `tone_amplitudes` and `tone_phases` define a planned multitone stimulus without changing
the source topology. All physical source quantities are SI values.

### Control-edge injection

Virtual injection nodes add a typed signal at an existing `ControlGraph` edge. They are stateless
and use the edge's existing unit and compatibility checks. Insert one between the original producer
and consumer rather than reaching into a controller implementation:

```yaml
units:
  converter:
    ctrl:
      type: open_loop
      loops:
        DutyModulation: {}
        VirtualDutyInjection: {amplitude: 0.02, frequency: 100.0}
      references: {duty: 0.4}
      connections:
        VirtualDutyInjection.signal: references.duty
        DutyModulation.duty: VirtualDutyInjection.injected
```

The addon supplies injection nodes for scalar, dq, duty, power, frequency and angle signals. Their
recorded `before`, `after` and `injection` channels are measured at the graph edge. Unsupported unit
or frame conversions are rejected rather than inferred.

## Recording and signal alignment

Enable `simulation.output.signals: 1` for plant and control measurements. `recorded_signal` reads
named plant, control and state scalars and can project three-phase signals with
`component="alpha"`, `"beta"`, `"d"` or `"q"`. A dq projection requires an explicit
`frame_frequency`.

Plant, control and state files may use different clocks. Recorded transfer functions require equal
timestamps by default. `align="linear"` explicitly permits interpolation for well-bandlimited
signals, but it does not increase the original temporal resolution.

Choose a tail window after the excitation and response have settled. A nonzero source amplitude or
a completed run is not evidence of steady state.

## Frequency response from recorded data

For a completed PESLite result, measure a scalar transfer directly from named signals:

```python
from peslite.addons.functions import measure_recorded_transfer

point = measure_recorded_transfer(
    result,
    "stimulus.injection",
    "stimulus.voltage",
    frequency=20.0,
    window=(0.1, 0.5),
    input_unit="A",
    output_unit="V",
)
print(point.gain, point.valid, point.flags)
point.write_json("output/20hz.json")
```

The returned `FrequencyMeasurement` contains the complex phasors and gain, amplitudes, THD, SNR,
window duration, Nyquist estimate, sample jitter, convergence checks and failure flags.

The lower-level interface accepts arrays that already share timestamps:

```python
from peslite.addons.functions import frequency_response, frequency_sweep

point = frequency_response(
    time,
    excitation,
    response,
    frequency=50.0,
    input_unit="A",
    output_unit="V",
)
sweep = frequency_sweep([point_20hz, point_50hz, point_100hz])
```

The fit includes the fundamental, DC offset and supported harmonics on the original timestamps.
Phasors use cosine as the real axis and sine as the negative imaginary axis, referenced to the first
sample. Both halves of the window are fitted independently to detect drift. Other checks include
excitation amplitude, distortion, residual-noise SNR, jitter, available cycles and a conservative
Nyquist limit based on the largest sample gap.

Tune acceptance criteria with `min_cycles`, `min_excitation_amplitude`, `max_steady_drift`,
`max_distortion`, `min_snr_db` and `max_sample_jitter`. These are measurement-validity limits, not
claims of plant linearity or stability. An invalid point retains diagnostics but sets `gain=None`.

`measure_recorded_sweep({frequency: result, ...}, excitation, response, ...)` analyzes independently
settled runs. `run_port_sweep(...)` creates those fresh simulations, writes each resolved case and
CSV, and never reuses an unrelated final state. `measure_multitone(...)` performs a joint
least-squares fit and removes the other planned tones before validating each one. Tones closer than
the recording resolution or overlapping through harmonics are rejected.

## Multiport impedance and admittance

A full matrix requires independent excitation directions at every frequency. For dq impedance,
run separate d-axis and q-axis current-injection experiments and solve `V = Z I`; for admittance,
solve `I = Y V`.

```python
from peslite.addons.functions import measure_port_matrix

matrix = measure_port_matrix(
    {20.0: [result_d, result_q]},
    voltage="stimulus.voltage",
    current="stimulus.current",
    injection="stimulus.injection",
    frame="dq",
    frame_frequency=50.0,
    current_sign=-1,
    window=(0.2, 1.0),
)
print(matrix.values[0], matrix.valid, matrix.flags)
matrix.write_json("output/dq_impedance.json")
```

`current_sign=-1` converts the shunt component's inward port current to network-injected current.
Never invert a measured transfer without checking rank, conditioning and validity.

`run_port_matrix(...)` creates the independent runs automatically. Use `frame="scalar"` for a
scalar source, `frame="dq"` for rotating axes, or `frame="sequence"` for the modified-sequence
basis. The latter applies `x+ = (xd + j*xq)/2`, `x- = (xd - j*xq)/2` and
`Zseq = T Zdq T^-1`; it retains off-diagonal coupling but does not identify arbitrary unbalanced
three-phase dynamics without additional measurements.

When phasors are already available, use the lower-level matrix fit:

```python
from peslite.addons.functions import impedance_matrix

matrix = impedance_matrix(
    frequencies_hz,
    current_phasors,
    voltage_phasors,
    input_names=["id", "iq"],
    output_names=["vd", "vq"],
    measurement_valid=validated_experiments,
)
```

The phasor arrays use axes `[frequency, experiment, port]`; the fitted values use
`[frequency, output, input]`. Rank deficiency, ill-conditioning, invalid experiments and excessive
fit residuals are flagged per frequency. Rejected values are `NaN` in memory and `null` in strict
JSON output.

## Feedback return ratio and margins

A control-edge injection exposes the values immediately before and after the additive cut. For
ordinary negative feedback, the return ratio is `L = -before / after`, with a configurable sign for
other summing conventions. Dividing the return signal by the added perturbation instead measures a
closed-loop sensitivity, not the return ratio.

```python
from peslite.addons.functions import measure_return_ratio, loop_gain_sweep

point = measure_return_ratio(
    result,
    injection="converter.VirtualDutyInjection",
    frequency=100.0,
    component="d",
    window=(0.05, 0.1),
)
margin_report = loop_gain_sweep({20.0: point20, 40.0: point40, 80.0: point80})
margin_report.write_json("output/loop_gain.json")
```

The sweep reports validated return ratios, gain and phase crossover frequencies, gain margin and
phase margin. Crossings are interpolated only between neighboring valid points. Missing, multiple
or disconnected crossings remain explicit diagnostics. Scalar margins alone do not certify a
nonlinear or multivariable closed loop.

## Validity and limitations

Offline analysis does not create an analytical small-signal circuit, infer injection polarity or
assume that an unforced response is zero. Select the terminal, sign, frame and response projection
explicitly. Use an excitation small enough for local behavior but large enough for acceptable SNR.
Switching sidebands, a time-varying plant and coupled ports can require additional frequencies and
independent directions.

The native C++ and Imperix exporters do not promise support for Python analysis injection types;
the workflows on this page use the Python backend and its ordinary result files.

## Verification

```bash
python -m pytest tests/test_identification.py tests/test_frequency_measurements.py
python -m pytest tests/test_port_injection.py tests/test_port_measurement.py \
    tests/test_virtual_injection.py
```
