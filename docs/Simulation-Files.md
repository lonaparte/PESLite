# Simulation Files

A `.pes` file is a YAML description of one simulation. PESLite validates the complete file before
assembly and reports unknown fields, invalid values and unresolved connections.

## Structure

```yaml
base: {s_base: 2.0e6, v_ll_rms: 690.0, f0: 50.0}

buses: {}
branches: {}
sources: {}
elements: {}
units: {}
events: {}

simulation:
  t_end: 1.0
  solver: {type: fixed, method: rk4, dt: 1e-6}
  output: {}

meta: {}
```

| Section | Contents | Reference |
|---|---|---|
| `base` | system power, voltage and frequency bases | [SI and per-unit values](#si-and-per-unit-values) |
| `buses`, `branches`, `elements` | passive electrical network | [Passive Network Components](Passive-Network-Components.md) |
| `sources` | independent network sources | [Electrical Sources](Electrical-Sources.md) |
| `units` with `type: vsc` | voltage-source converters | [Voltage-Source Converters](Voltage-Source-Converters.md) |
| converter `bridge`, `pwm`, `ctrl`, `meas`, `protection` | bridge model and sampled control | [Bridge and PWM](Bridge-and-PWM.md) |
| `units` with `type: dc_converter` or `wpt` | DC converters and coupled links | [DC converters and WPT](DC-Converters-and-WPT.md) |
| `events` and `simulation.initial` | event timeline and continuation | [Events Timeline](Events-Timeline.md) |
| `simulation.output` and `simulation.progress` | files, signals and progress | [Output and Results](Output-and-Results.md) |
| `simulation.solver` | integration settings | [Solvers](Solvers.md) |
| `simulation.solver.subsystems` | subsystem step ratios | [Multirate Simulation](Multirate-Simulation.md) |
| `meta` | optional title and description | — |

All top-level sections except `base` and `simulation` are optional. A model without a component
section simply omits it.

## Names and paths

Names are unique within each of `buses`, `branches`, `sources`, `elements` and `units`. They must
be non-empty and cannot contain a dot. Dots separate public configuration, state and signal paths:

```text
units.vsc.ctrl.references.p_ref_pu
units.converter.bridge.model
simulation.solver.dt
```

A network connection uses a bus name or a named Unit port. Event targets must identify one
switchable object.

## SI and per-unit values

Power-circuit values are stored internally in SI. A field ending in `_pu` is converted using the
applicable system or Unit base; do not provide both SI and per-unit forms of one field.

| `base` field | Required | Meaning |
|---|---:|---|
| `s_base` | yes | system power rating in VA |
| `v_ll_rms` | when an AC voltage base is needed | three-phase line-to-line RMS voltage in V |
| `f0` | when a frequency base is needed | nominal frequency in Hz |

Each Unit may override `s_base`. Time is in seconds, angles are in radians and ordinary frequency
fields are in hertz. Switch values accept `0`/`1` or YAML `false`/`true`.
