# DC Converters and Wireless Power Transfer

`dc_converter` is a scalar converter Unit. `wpt` combines two scalar child Units with one coupled
link.

## DC converter

```yaml
units:
  converter:
    type: dc_converter
    bus: output
    filter: {l_f: 1e-3, r_f: 0.05}
    dclink:
      vdc_ref: 48.0
      source: {type: voltage}
    bridge: {type: half_bridge, lower_controlled: 0}
    pwm: {f_sw: 20000.0}
    ctrl:
      type: dc
      loops:
        DCVoltageControl: {kp: 0.3, ki: 80.0, minimum: 0.0, maximum: 15.0}
        InductorCurrentControl: {kp: 0.03, ki: 30.0, minimum: 0.02, maximum: 0.95}
      references: {voltage: 24.0}
```

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `type` | yes | — | `dc_converter` |
| `bus`, `filter` | together for an external connection | omitted | scalar terminal and series R-L filter |
| `s_base` | no | `base.s_base` | Unit rating in VA |
| `dclink` | yes | — | rated DC voltage and source |
| `bridge.type` | no | selected by `ctrl.type` | `half_bridge` or `full_bridge` |
| `pwm`, `ctrl`, `meas`, `protection` | by application | — | common converter configuration |

An external bus must use `phases: 1`. The bridge, PWM periods, ADC and protection settings are
documented under [Bridge and PWM](Bridge-and-PWM.md). DC-link sources are
documented under [Electrical Sources](Electrical-Sources.md#converter-dc-source).

### Circuit connections

| Connection | Bridge settings | External connection |
|---|---|---|
| Buck | `lower_controlled: 0` | midpoint through `filter` to output `bus` |
| Boost | `upper_controlled: 0`, `reverse_port: 1` | input `bus` through `filter` to midpoint |
| Buck-Boost | `lower_controlled: 0`, `external_terminal: lower_rail` | lower rail to output `bus`; midpoint through `filter` to common point |
| bidirectional half bridge | both device rows controlled | midpoint through `filter` to `bus` |
| full bridge | `type: full_bridge` | power port through `filter` to `bus` |

`external_terminal: lower_rail` requires `bus`, `filter` and `reverse_port: 0`.

### CCM and DCM

For a half bridge with an uncontrolled device row, CCM and DCM follow the electrical state; there
is no mode setting.

| Bridge model | DCM handling | Step requirement |
|---|---|---|
| `switching` | locates diode-current zero and blocks reverse diode current | switching-event boundaries |
| `pwm_averaging` | cycle-mean controlled, diode and idle fractions | at most one tenth of the switching period while in DCM |
| `averaging` | same cycle-mean conduction relation | same local DCM limit |

The DCM step limit is inactive in CCM.

### Controllers and measurements

| `ctrl.type` | Use |
|---|---|
| `dc` | cascaded voltage and inductor-current control |
| `open_loop` | direct duty reference |
| `resonant` | amplitude, frequency and phase modulation graph |
| `phase_shift` | square-wave full-bridge modulation |

Built-in measurement channels are `current`, `voltage`, `vin` and requested port `power`.
Available protection criteria are overcurrent, RMS current, terminal voltage and DC-link voltage.

## Diode full-bridge rectifier

```yaml
bridge: {type: full_bridge, upper_controlled: 0, lower_controlled: 0}
```

Omit `ctrl`, `pwm` and protection. Set `meas.period` explicitly when the default output period is
unsuitable. All bridge-model selections retain the natural diode commutation law. Nonzero
`parallel_capacitance` requires positive `on_resistance`.

## Phase-shift modulation

```yaml
bridge: {type: full_bridge, model: switching}
pwm: {f_sw: 85000.0, pattern_sync: pulse}
ctrl:
  type: phase_shift
  loops: {PhaseShiftModulation: {}}
  references: {width: 0.8, frequency: 85000.0, phase: 0.0, mode: 0}
```

| Reference | Value |
|---|---|
| `width` | pulse width in half periods, clipped to [0, 1] |
| `frequency` | waveform frequency in Hz |
| `phase` | positive-pulse centre in rad |
| `mode` | 0 bipolar, 1 positive only, 3 zero, 4 negative only |

Phase-shift modulation requires a controlled full bridge with `bridge.model: switching`. Omit
`pwm.sync` and `pwm.carrier_phase`. `pwm.f_sw` sets the register-load period;
`references.frequency` sets the square-wave frequency.

## WPT Unit

```yaml
units:
  charger:
    type: wpt
    primary:
      type: dc_converter
      dclink: {vdc_ref: 400.0, source: {type: voltage}}
      pwm: {f_sw: 1e6}
      ctrl:
        type: resonant
        loops: {ResonantModulation: {}}
        references: {modulation: 0.8, frequency: 85000.0}
    secondary:
      type: dc_converter
      dclink: {vdc_ref: 300.0, source: {type: voltage}}
      pwm: {f_sw: 800e3}
      ctrl:
        type: resonant
        loops: {ResonantModulation: {}}
        references: {modulation: 0.7, frequency: 85000.0, phase: 1.5708}
    link:
      type: ss
      primary_inductance: 100e-6
      secondary_inductance: 100e-6
      mutual_inductance: 20e-6
      primary_capacitance: 35.06e-9
      secondary_capacitance: 35.06e-9
```

| Field | Required | Meaning |
|---|---:|---|
| `primary` | yes | scalar non-composite child Unit |
| `secondary` | yes | scalar non-composite child Unit |
| `link` | yes | coupled electrical link |
| `s_base` | no | shared rating; defaults to `base.s_base` |

The children use the ordinary `dc_converter` fields and may have independent PWM, controller and
measurement periods.

### Coupled links

| `link.type` | Required configuration |
|---|---|
| `ss` | primary/secondary inductance and capacitance, mutual inductance |
| `lcc` | coil and compensation inductances, series and parallel capacitances, mutual inductance |
| `lcc_s` | LCC primary, series-compensated secondary, mutual inductance |
| `ss_multi` | inductance matrix and primary/secondary capacitance lists |

Corresponding resistance fields are optional. `mutual_inductance` may be negative according to
winding polarity.

### Switch-controlled capacitors

`ss`, `lcc` and `lcc_s` accept `primary_scc` and `secondary_scc`:

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `capacitance` | yes | — | switched capacitance in F |
| `angle` | no | 0 | conduction angle in [0, pi] |
| `drive` | no | `common` | `common` or `alternating` gate pattern |
| `sync` | no | `current` | `current` or `bridge` phase reference |
| `sync_angle` | no | 0 | bridge-synchronised phase offset in rad |
| `command` | no | 0 | consume controller output `scc_angle` |

## Exact commutations

Natural diode and switch-controlled capacitor transitions are located inside a solver step.

| Solver configuration | Requirement |
|---|---|
| fixed step | `simulation.solver.dt` is the largest probe interval |
| adaptive | set a finite `simulation.solver.max_step` |
| multirate | follow [commutating-group restrictions](Multirate-Simulation.md#limitations) |

Discrete conduction and pulse states are restored during continuation.
