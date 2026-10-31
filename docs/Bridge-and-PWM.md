# Bridge and PWM

Converter Units share the same bridge-model, control-period, PWM-register and ADC configuration.

## Bridge configuration

```yaml
bridge:
  type: three_phase_bridge
  model: pwm_averaging
  on_resistance: 0.0
  parallel_capacitance: 0.0
```

| Field | Values | Default | Purpose |
|---|---|---|---|
| `type` | `half_bridge`, `full_bridge`, `three_phase_bridge` | selected by Unit type | electrical topology |
| `model` | `switching`, `pwm_averaging`, `averaging` | `pwm_averaging` | simulation form |
| `upper_controlled`, `lower_controlled` | switches | 1 | controlled rows of a half bridge or full bridge |
| `reverse_port` | switch | 0 | half-bridge power-port direction |
| `external_terminal` | `midpoint`, `lower_rail` | `midpoint` | exposed half-bridge terminal |
| `on_resistance` | ohm per conducting device | 0 | device voltage drop and loss |
| `parallel_capacitance` | F | 0 | capacitance across the bridge power port |

### Model comparison

| `bridge.model` | Electrical representation | Controller execution | Use for |
|---|---|---|---|
| `switching` | exact carrier-comparison switch states | sampled ADC, computation and PWM loads | ripple, harmonics and commutation |
| `pwm_averaging` | held active duty between register loads | same sampled periods and updates as switching | sampled control without carrier ripple |
| `averaging` | continuous controlled voltage | continuous controller equations | control-envelope and long-duration studies |

Each converter Unit selects its model independently.

### Device resistance and port capacitance

| Configuration | Result |
|---|---|
| both values zero | ideal bridge; no additional state |
| nonzero `on_resistance` | finite conducting-path voltage drop and loss |
| nonzero `parallel_capacitance` | adds continuous state `bridge.u_C` |
| nonzero `parallel_capacitance` with zero resistance | rejected |

A half-bridge or three-phase conducting leg uses one device resistance; a full-bridge current path
uses two. Use `switching` to resolve individual capacitor charging and commutation pulses.

## Control and PWM period

```yaml
ctrl:
  period: 50e-6
  offset: 0.0
  computation: 1e-6
pwm:
  f_sw: 20000.0
  update: single
  sync: asynchronous
  pattern_sync: interval
  dead_time: 0.0
  dead_time_compensation: false
meas:
  period: 25e-6
  offset: 0.0
  window: 50e-6
```

| Setting | Default | Constraint | Effect |
|---|---|---|---|
| `ctrl.period` | `1 / pwm.f_sw` | whole multiple of half a carrier period | controller interrupt period |
| `ctrl.offset` | PWM load-clock position | within one control period | controller interrupt position |
| `ctrl.computation` | 1 us in sampled models | `0 <= computation < period` | command eligibility delay |
| `pwm.f_sw` | required for sampled control | positive Hz | carrier and default control frequency |
| `pwm.update` | `single` | `single` or `double` | active-register load rate |
| `pwm.carrier_phase` | 0 | carrier periods | asynchronous carrier position at `t = 0` |
| `pwm.sync` | `asynchronous` | `asynchronous` or `synchronous` | carrier clock source |
| `pwm.pattern_sync` | `interval` | `interval` or `pulse` | waveform replacement rule |
| `pwm.dead_time` | 0 | seconds, less than half a switching period | complementary-device turn-on delay |
| `pwm.dead_time_compensation` | `false` | Boolean | cancel the ideal current-polarity dead-time error |
| `meas.period` | `ctrl.period`; otherwise `simulation.output.period` | divides `ctrl.period` when sampled | ADC conversion period |
| `meas.offset` | aligned with controller interrupts | within one ADC period | ADC conversion position |
| `meas.window` | `ctrl.period` | no longer than `ctrl.period` | averaging window |

At one timestamp, all due ADC conversions run before any Unit applies a newly computed command.

## Controller and loop periods

```yaml
ctrl:
  period: 50e-6
  loops:
    CurrentLoop: {period: 50e-6}
    DCVoltageControl: {period: 500e-6}
```

- A loop period defaults to `ctrl.period`.
- A loop period must be an integer multiple of `ctrl.period`.
- Loop keys are registered class names and graph-node names.
- Repeated classes use qualified names such as `Observer[voltage]`.
- `{}`, a bare key and YAML `null` are equivalent empty parameter mappings.

### Scalar feedback loops

Scalar feedback loops calculate
`command = feedforward + PI(startup * reference - measured)`.

| Measurement | Modulation command | Phase command | Frequency command |
|---|---|---|---|
| power | `PowerControl` | `PhasePowerControl` | `FrequencyPowerControl` |
| voltage | `VoltageControl` | `PhaseVoltageControl` | `FrequencyVoltageControl` |
| current | `CurrentControl` | `PhaseCurrentControl` | `FrequencyCurrentControl` |

The control graph selects the measurement and command. Optional `<measurement>_filter` settings
filter the selected feedback.

## ADC sampling

```yaml
meas:
  average: [voltage, current, power]
  window: 50e-6
  period: 12.5e-6
  offset: 0.0
```

Channels in `meas.average` return the window mean; other channels are sampled instantaneously. If
ADC conversion and control coincide, conversion runs first. An offset conversion is held until the
next controller interrupt. For an oscillating port, average `power` directly rather than
multiplying separately averaged voltage and current.

### Measurement sources and RMS

```yaml
meas:
  sources: {power: load.power}
  rms: {current: 200e-6}
```

`meas.sources` accepts public outputs ending in `.voltage`, `.current` or `.power`. `meas.rms` maps
a channel to a window in seconds. Results use `<unit>.meas.<channel>_rms`; configured means use
`<unit>.meas.<channel>_mean`.

## Computation time

| PWM load instant | Applied value |
|---|---|
| after `ctrl.computation` completes | newly computed shadow value |
| at the interrupt or during computation | previous completed shadow value |
| consistently less than `computation` after the interrupt | one additional control-period delay |

Only the newest completed value is retained in the shadow register.

## PWM register updates

| `pwm.update` | Active-register loads |
|---|---|
| `single` | carrier valleys |
| `double` | carrier valleys and peaks |

Three-phase PWM uses `pwm.method: spwm` or `svpwm`; `pwm.modulation_limit` defaults to 1.
Synchronous PWM requires an integer ratio `pwm.f_sw / base.f0`.

| `pwm.pattern_sync` | Update rule |
|---|---|
| `interval` | replace the waveform at the register load |
| `pulse` | finish an active pulse before starting the new pattern |

`pulse` may insert a zero gap of up to one pattern period. Averaged models do not construct
pulse-preservation state.

Dead time is applied to the leg sequence after modulation, so carrier and phase-shift modulation
share the same current-directed diode-clamping rule. `pwm_averaging` applies its cycle average;
legs held at duty 0 or 1 do not commutate and have no dead-time error. Ideal `averaging` has no PWM
hardware, so it warns and ignores dead time and compensation. Compensation in the sampled models
applies an ideal polarity-based correction and removes the corresponding voltage error.

## Protection periods

| Criterion | Evaluation boundary |
|---|---|
| instantaneous overcurrent | switching or held-bridge edges |
| voltage and DC-link voltage | controller or measurement period |
| RMS current | configured RMS window |
| frequency and ROCOF | controller or measurement period |

A Unit without sampled control uses `meas.period` for measurement, RMS and held protection
criteria. A trip blocks the bridge and disconnects the DC source.

## Ideal averaging

With `bridge.model: averaging`, controller and plant states are integrated continuously.

| Setting or feature | Behavior |
|---|---|
| ADC interrupt and `meas.offset` | not used |
| `ctrl.period`, `ctrl.offset`, `ctrl.computation` | not used for controller execution |
| carrier and PWM register loads | not used |
| `pwm.dead_time`, `pwm.dead_time_compensation` | ignored |
| `meas.period` | protection and RMS period |
| loop implementation | must provide a continuous-time equation |

## Command presets

| Command | Bridge model | Solver preset |
|---|---|---|
| `--switching` | `switching` for every Unit | fixed-step RK4 |
| `--pwm-averaging` | `pwm_averaging` for every Unit | fixed-step RK4 |
| `--averaging` | `averaging` for every Unit | adaptive DP45 |

Explicit `--set` options take priority over a preset.
