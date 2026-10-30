# Examples

PESLite ships eighteen simulation files with the Python wheel. They can be run by name from any
directory, so cloning the repository is not required.

```bash
peslite gfl-example
```

Each example is an ordinary YAML simulation file. Use it directly, print its fully resolved
configuration, or copy that resolved form as the starting point for another case.

```bash
peslite gfl-example --resolved > case.pes
peslite case.pes
```

## Bundled cases

| Name | System and control purpose |
|---|---|
| `buck-example` | A closed-loop 48 V to 24 V Buck unit using the half bridge's controlled-device/diode placement, an L-C output network, native DCM and the shared scalar control graph. |
| `buck-open-loop-example` | An open-loop synchronous Buck whose duty-ratio reference changes through an ordinary event. |
| `boost-example` | A closed-loop 24 V to 48 V Boost application using the same DCConverter with the lower device controlled, the upper diode active and the external scalar bus as its input port. |
| `buck-boost-example` | A closed-loop single-inductor Buck-Boost built from one DCConverter, with the half bridge's lower rail connected to the negative output bus. |
| `cable-gfl-example` | A grid-following converter connected to the grid through the distributed R-L/G-C model registered under `addons/components`. |
| `custom-pll-example` | Grid-following control assembled from built-in current/DC-voltage loops and an automatically discovered voltage-adaptive PLL under `addons/control`. |
| `gfl-example` | One grid-following converter with PLL, dc-voltage control, current control and unit protection. This is the most extensively annotated configuration. |
| `gfm-psc-example` | Grid-forming power-synchronization control with virtual impedance and active damping. |
| `gfm-vsg-example` | Grid-forming virtual synchronous generator control. |
| `gfm-dvoc-example` | Grid-forming dispatchable virtual oscillator control. |
| `gfm-matching-example` | Matching control coupled to a dynamic dc link and current source. |
| `two-converters-example` | A GFM and a GFL unit on separate buses, with different ratings and switching frequencies. |
| `wpt-example` | A composite series-series compensated WPT unit whose independently clocked full-bridge primary and secondary are assembled from its single nested unit tree. |
| `wpt-ss-example` | SS WPT with local primary-side power feedback controlling modulation amplitude, including a 1200 to 1800 W reference step. |
| `wpt-lcc-example` | LCC-LCC WPT with local secondary-side power feedback controlling phase, including a -500 to -800 W reference step. |
| `wpt-load-example` | Independent series load with explicitly routed load-power feedback, a 5 to 8 ohm load step and per-side RMS overcurrent protection. |
| `wpt-phase-shift-example` | Phase-shift WPT with power-controlled pulse width, a repeating bridge-mode sequence and a diode full-bridge secondary. |
| `wpt-scc-example` | Square-wave WPT with primary power feedback commanding an SCC conduction angle. |

The files are also available in the repository's
[`examples/`](https://github.com/lonaparte/PESLite/tree/main/examples) directory. The root
directory contains only simulation files, not Python example programs.

## Compare bridge models

The same case can be run under each bridge/solver preset without editing its configuration:

```bash
peslite gfl-example --switching
peslite gfl-example --pwm-averaging
peslite gfl-example --averaging
```

The default output directories receive a mode suffix, so these runs do not overwrite one another.
They represent different model fidelity and timing semantics; compare quantities in their common
bandwidth rather than expecting identical switching-scale waveforms. See
[Converter and Bridge Models](Converter-and-Bridge-Models.md).

## WPT power feedback

Keep `wpt-example` for open-loop operation. The two feedback cases use the same `DCConverter`,
`GraphController`, magnetic-link components and `ResonantModulation`; neither creates a WPT-specific
controller, solver or measurement exchange between sides.

```bash
peslite wpt-ss-example --switching
peslite wpt-ss-example --pwm-averaging
peslite wpt-lcc-example --averaging
```

`PowerControl` produces a dimensionless modulation amplitude; `PhasePowerControl` produces radians.
They share one signed-power feedback law built from the existing `Filter`, `PI` and `OutputLimit`.
Both use `reference - measured_power`. The feedforward port has the same units as the command, and
the gains have units command/W and command/(W*s). Set phase limits and gain polarity for a monotonic
operating branch of the configured circuit; the controller does not infer them from SS or LCC.
Phase stays in signed radians. Before saturation, `PhasePowerControl` translates the command by
its configured minimum, so the unchanged shared limiter operates on `[0, maximum - minimum]`.
It then restores the phase coordinate. This preserves the configured operating branch and PI
anti-windup; it neither wraps the angle nor removes the constraint.

The power channel uses the paired voltage and current of the converter's external `power` terminal,
including its direction and power convention, independently of control feedback coordinates. In
these unfiltered full bridges that is the local magnetic-link terminal. Positive power leaves the
converter, so negative
secondary power means power is delivered to its ideal DC voltage source. These examples demonstrate
power-transfer regulation between stiff DC rails, not battery dynamics or output-voltage regulation.

Use `meas.average: [power]` for sampled power feedback. The existing ADC window then integrates
instantaneous `voltage * current` over the control interval. Multiplying separately averaged voltage
and current is not equivalent; sampling only at a carrier boundary may also return zero voltage.
The feedback loop's optional power filter removes the remaining excitation-frequency ripple.
Ideal `averaging` uses the instantaneous product in the continuous control graph and the same filter,
without an ADC or artificial sampling delay.

The controller CSV files record each example's command, filtered power and power reference. The
reference changes at 3 ms; compare settled, time-averaged power, not individual ripple samples or
decimated switching waveforms. Reduce `simulation.output.period` when inspecting switching-scale
voltage/current.

## Independent loads and measurement selection

`wpt-load-example` uses the existing `Resistor`, power loop and `set` event. `series: true`
inserts the resistor after the selected bus's voltage driver, before all of that bus's followers.
Omit it for a shunt resistor. No artificial capacitor/inductor is inserted, and load dissipation
is recorded separately from winding losses. Multiple series elements on the same bus form a chain
in declaration order, from the original voltage driver to the followers.

An element's `bus` may name a configured bus, a Unit's default port (`charger` or
`charger.secondary`), or a named port (`charger.secondary.dc` or `charger.secondary.power`).
Nested Units resolve through the same declared default-port contract.
Element per-unit inputs retain the network/system base; selecting a Unit port does not silently
change their base. The example specifies load resistance directly in ohms.

```yaml
elements:
  load: {type: resistor, bus: charger.secondary.power, r: 5.0, series: true}
# In the controller-owning Unit:
meas:
  sources: {power: load.power}
  average: [power]
  rms: {current: 200.0e-6}
protection:
  rms_current: {enable: true, limit_pu: 2.0}
  hold: 50.0e-6
```

`meas.sources` overrides existing channel readers with named physical outputs. For a configured
bus or Unit port, append `.voltage`, `.current` or `.power`, such as `charger.secondary.dc.power`.
Port power is positive outward from the bus's voltage-driving terminal; a resistor's `load.power`
is its nonnegative dissipation. Port observations are non-loading and use the existing energy-port
power convention. All bindings are resolved at assembly, including dependency ordering for
continuous control and C++ export. Selecting another Unit's measurement is explicit signal routing,
not an implicit communication model or a claim that hardware makes that measurement locally.

RMS is `sqrt(mean(abs(signal)**2))`, integrated before sampling over a sliding window of whole
control periods. Ideal averaging uses whole periods of its protection clock instead (`meas.period`,
default `simulation.output.period`); it does not discretize the continuous controller. This is
total waveform RMS, not a fundamental/envelope estimator; for a complex space vector it is the RMS
of its magnitude, not a per-phase RMS conversion.
An averaged model's RMS does not reconstruct the switching harmonics omitted by that model.
The statistic is unavailable until a complete window has elapsed. Signals are named
`<unit>.meas.<channel>_rms`. Only configured windows are allocated.

RMS protection uses the same current base and trip latch as existing protection. Either side may
enable it independently. It is a held overcurrent trip, not soft current regulation: it does not
force phase to zero or change the common command limiter. Changing `elements.load.r` uses the
ordinary `set` event and preserves electrical and controller state in Python and C++.

## Short exploratory runs

Use `--set` to shorten a bundled case or change a reference without maintaining another file:

```bash
peslite gfm-psc-example \
  --set simulation.t_end=1.0 \
  --set units.vsc.ctrl.references.p_ref_pu=0.3 \
  --out output/psc-short
```

For progress lines with selected live values:

```bash
peslite two-converters-example \
  --progress 0.05 \
  --watch vsc_1.vdc_pu,vsc_2.vdc_pu
```

Device names keep otherwise similar values unambiguous. `vsc_1.vdc_pu` and `vsc_2.vdc_pu`
therefore refer to different converters.

## Inspect before running

```bash
peslite gfl-example --list-states
peslite gfl-example --ph-report
peslite gfl-example --resolved
```

These commands use the same configuration resolution and accept the same bridge presets and
`--set` overrides as a simulation run.

## Export an example to C++

```bash
peslite-convert gfl-example all
c++ -O3 -DNDEBUG -std=c++17 export/gfl-example/peslite.cpp \
  -o export/gfl-example/peslite
export/gfl-example/peslite --list-params
```

See [C++ Export](Cpp-Export.md) for variable selection, runtime configuration and backend limits.

## Build a new case

Start with `gfl-example` when learning the full configuration surface; its comments describe the
available network, converter, timing, protection, event, solver and output options. Start with the
closest compact GFM case when changing only the synchronization law. For a network with more than
one independently named unit, use `two-converters-example`.

Continue with [Getting Started](Getting-Started.md) and
[Simulation Files](Simulation-Files.md) when turning an example into a maintained case.
