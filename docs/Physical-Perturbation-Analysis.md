# Physical Perturbation and Port Identification

Electrical perturbations are optional circuit components under `peslite.addons.components`.
Measurements are optional functions under `peslite.addons.functions`. The existing electrical
netlist, normal simulation solver, port-current orientation, event scheduling, and recorded CSV
schema remain authoritative. Identification is **not** part of the normal simulation hot path.

## Physical connection and sign conventions

Use the `port_injection` element in `elements`. **Voltage injection is a series source**:
it inserts a potential difference at a two-port cut. **Current injection is a shunt source**:
it connects between ground and the selected node, without severing that node's original
connections. Both support scalar and three-phase electrical ports.

```text
Series voltage injection:            Shunt current injection:

upstream ---- [ + V(t) - ] ---- bus   ground ---- [ I(t) ] ---- bus
                 (series cut)                  (in parallel)
```

A positive series stimulus enforces `V_bus - V_inlet = +injection`. Positive shunt stimulus
flows **from ground into the bus**. PESLite's physical port current is always positive *into
the component*: for the shunt stimulus, `stimulus.current = -stimulus.injection`.
The source quantities and port currents obey the existing peak-space-vector convention
(three-phase physical power scale `1.5`) and require compatible electrical domains.

### Automatic insertion into an existing connection

```yaml
elements:
  stimulus:
    type: port_injection
    kind: voltage
    bus: pcc
    amplitude: 1.0  # series volts
    frequency: 20.0 # Hz
```

The generic series insertion mechanism moves one existing driver into
`stimulus.bus.in` and connects the two-port source between it and `pcc`. The target must
be a node that **already joins exactly two ports**, as required by the existing
`series_connections()` contract. If that topology is unsuitable, use an explicit
`inlet` on a new bus; unlike automatic insertion, that is a genuine second electrical
connection and does not modify other links:

```yaml
elements:
  stimulus:
    type: port_injection
    kind: voltage
    inlet: upstream
    bus: downstream
    amplitude: 2.0
    frequency: 40.0
```

### Parallel current injection

```yaml
elements:
  stimulus: {type: port_injection, kind: current, bus: pcc,
             amplitude: 0.5, frequency: 20.0}
```

`stimulus.injection` is positive current injected into the surrounding network.
`stimulus.current` is current entering the source itself, the opposite sign. A current
source does not have a series inlet.

### Load perturbation

A port injection is only a source. Represent the nominal load with the existing passive
component and its small-signal conductance disturbance with the Norton current
`-V0 * delta_G(t)`:

```yaml
elements:
  load: {type: resistor, bus: output, r: 10.0}
  load_probe: {type: port_injection, kind: current, bus: output,
               amplitude: -0.048, frequency: 20.0}
```

Here `V0 = 24 V` and the conductance amplitude is `0.002 S`. This approximation presumes
the bus voltage stays near `V0`; model a large-signal load change with the passive component.

### Three-phase orientation and excitation

A three-phase probe uses PESLite's complex peak-scaled stationary space vector.
`axis: d` or `q` rotates a sinusoidal modulation with `frame_frequency` (Hz), so
`amplitude` measures an excitation in the synchronous d or q channel. `axis: alpha`
or `beta` applies a stationary-axis modulation. `axis: positive` and `negative`
select respectively the upper and lower sideband of one sinusoidal rotating vector.
Sequence options permit only one tone, with zero bias. Keep `frame_frequency`
consistent with the controller frame used to interpret a dq result.

`tones`, `tone_amplitudes`, and `tone_phases` may describe multiple added cosines;
`frequency`, `amplitude`, and `phase` describe the primary tone. All quantities are
physical SI quantities. The voltage-source and shunt-current circuit topologies
remain unchanged for multi-tone signals. Each source is deterministic in absolute time
and retains no additional simulation state.

## Recorded transfer functions

Enable `simulation.output.signals: 1`. After a completed run, identify a transfer
using the existing frequency-response validation and its phase convention:

```python
from peslite.addons.functions import measure_recorded_transfer

measurement = measure_recorded_transfer(
    result, "stimulus.injection", "stimulus.voltage",
    frequency=20.0, window=(0.1, 0.5),
    input_unit="A", output_unit="V",
)
print(measurement.gain, measurement.valid, measurement.flags)
measurement.write_json("output/20hz.json")
```

The result is the same `FrequencyMeasurement` as the existing frequency analysis:
complex gain, amplitudes, THD, SNR, time-window length, Nyquist estimate, sampling
jitter, convergence checks and explicit failure flags. An invalid frequency **does
not** produce a numerical gain. A nonzero source at the physical node is not itself
proof that the measured response has settled. Select a tail window *after* transients.

`measure_recorded_sweep({frequency: result, ...}, excitation, response, ...)` analyzes
independently settled experiments. Alternatively, `run_port_sweep(params,
injection="stimulus", frequencies=[20, 40, 80], response="pcc.u", directory="output/sweep",
window=(0.1, 0.5))` builds fresh simulations using the normal runner and retains each
resolved configuration and CSV. The code never reuses an unrelated final state.

`measure_multitone(result, excitation, response, frequencies, ...)` separates planned
tones by a **joint least-squares fit**, then applies the common validity checks to
each tone with the others removed. Tones separated by less than the recording's
frequency resolution, and overlapping harmonic tones, are rejected rather than
silently misidentified. For ordinary independent-frequency validation, separate
single-tone runs are preferable.

The recorder does **not** guarantee that control, plant and state tables use the same
clock. `measure_recorded_transfer` requires matching timestamps by default. For
explicitly well-bandlimited measurements, `align="linear"` opts into interpolation;
this does not improve the underlying temporal resolution. `recorded_signal` reads
all named plant/control/state scalars and three-phase signals (via `component="alpha"`,
`"beta"`, `"d"`, or `"q"`). For dq projections, set `frame_frequency` explicitly.

## Impedance, admittance and cross-coupling

To identify a full two-axis dq matrix, run **independent d-axis and q-axis physical
injection experiments** at each modulation frequency. The least-squares equations are
`V = Z I` for impedance (ohms), or `I = Y V` for admittance (siemens).
For the shunt source's *component* port current, use `current_sign=-1` to express
network-injected current. Never invert a measured transfer without checking its
rank, conditioning and validity.

```python
from peslite.addons.functions import measure_port_matrix

matrix = measure_port_matrix(
    {20.0: [result_d, result_q]},
    voltage="stimulus.voltage", current="stimulus.current",
    injection="stimulus.injection", frame="dq", frame_frequency=50.0,
    current_sign=-1, window=(0.2, 1.0),
)
print(matrix.values[0], matrix.valid, matrix.flags)
matrix.write_json("output/dq_impedance.json")
```

`run_port_matrix(params, injection="stimulus", frequencies=[20, 40],
voltage="stimulus.voltage", current="stimulus.current", directory="output/matrix",
frame="dq", frame_frequency=50, current_sign=-1, window=(0.2, 1.0))` constructs
these independent simulations automatically. For a scalar source, use `frame="scalar"`.
To output the **modified-sequence** basis with all off-diagonal coupling terms,
use `frame="sequence"`. The transformation is
`x+ = (xd + j*xq)/2`, `x- = (xd - j*xq)/2`, yielding
`Zseq = T Zdq T^-1`; it does **not** claim to resolve arbitrary unbalanced
three-phase dynamics without additional measurements. Modulation frequencies are
offsets from the stated frame frequency. The JSON includes axes, units, complex
real/imaginary entries and per-frequency fit validity.

## Virtual control-signal injection

The `peslite.addons.control.virtual_injection` add-on registers typed, stateless
controller-graph nodes. Inserting one between existing graph edges allows a sinusoidal
**additive** perturbation without reaching into controller internals. The available
classes are `VirtualVoltageInjection`, `VirtualCurrentInjection`,
`VirtualDqVoltageInjection`, `VirtualDqCurrentInjection`,
`VirtualDutyInjection`, `VirtualPowerInjection`, `VirtualFrequencyInjection`,
and `VirtualAngleInjection`. The input and output types use the existing
`ControlGraph` compatibility checks, and the amplitude is in that graph signal's
**existing unit** (possibly per-unit, not necessarily SI). Unsupported output-frame
conversions are not guessed.

For an open-loop scalar duty command:

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

Copy the full unit hardware fields from a normal PESLite case. The add-on exposes
recorded `injection_d`, `injection_q`, `before_d`, `before_q`, `after_d`, and
`after_q` controller diagnostic channels. The values are measured directly at
that graph edge. You can measure a control-to-control transfer using the same
`measure_recorded_transfer`, provided the two recorded signals are on compatible
sampling clocks and measurement validity checks pass.

## Loop gain from a feedback cut

An *additive feedback cut* provides the signals immediately before and after
injection. **The return ratio is not the return signal divided by the added
perturbation**: that would be a closed-loop sensitivity function. For ordinary
negative feedback, define `L = -before/after`, with the sign adjustable when the
controller's summing convention differs. Only a correctly placed cut that traverses
exactly the return path of interest supports this interpretation.

```python
from peslite.addons.functions import measure_return_ratio, loop_gain_sweep

point = measure_return_ratio(
    result, injection="converter.VirtualDutyInjection",
    frequency=100.0, component="d", window=(0.05, 0.1),
)
# Run an independently settled experiment at each frequency.
margin_report = loop_gain_sweep({20: point20, 40: point40, 80: point80})
margin_report.write_json("output/loop_gain.json")
```

The report returns validated complex return ratios, gain/phase crossover
frequencies (Hz), gain margin (dB), and phase margin (degrees). These use
log-frequency interpolation **between validated neighbouring points only**.
Unobserved crossovers return `null` with diagnostics; disjoint invalid windows are
not connected, and several crossings are flagged. Margins do not by themselves
certify nonlinear or multivariable closed-loop stability.

## Numerical and modelling limitations

The above functions operate on recorded data. They do not create an analytical
small-signal circuit, assume a zero disturbance response, or automatically infer
an injection-port polarity. Select the circuit terminal, response projection,
rotating frame and excitation amplitude deliberately. Use a small enough
perturbation to remain locally linear while retaining sufficient amplitude for
acceptable SNR. A time-varying plant, switching sidebands, or multiple simultaneous
physical ports can require more independent experiments and frequency-resolved
modelling than a single scalar transfer reveals.

The existing native C++ and Imperix exporters do not promise support for these
optional Python injection/control types; the documented workflows use the
Python simulation backend and its ordinary state continuation and output files.

## Tests

```bash
python -m pytest tests/test_port_injection.py tests/test_port_measurement.py \
    tests/test_virtual_injection.py
python -m pytest
python -m pytest --run-full tests/test_small_signal.py tests/test_small_signal_ac.py
```
