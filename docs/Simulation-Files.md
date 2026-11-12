# Simulation Files

A `.pes` file combines graphical interface views with YAML parameter domains. PESLite validates the
parameter mapping before assembly and reports unknown fields, invalid values and unresolved
connections.

## Structure

```text
@circuit
Grid -- PCC -- Converter

base: {s_base: 2.0e6, v_ll_rms: 690.0, f0: 50.0}

buses: {}
branches: {}
sources: {}
elements: {}
units: {}
signals: {}
events: {}

simulation:
  t_end: 1.0
  solver: {type: fixed, method: RK4, dt: 1e-6}
  output: {}

meta: {}
```

| Section | Contents | Reference |
|---|---|---|
| `base` | system power, voltage and frequency bases | [SI and per-unit values](#si-and-per-unit-values) |
| `buses`, `branches`, `elements` | passive electrical network | [Passive Components](Passive-Components.md) |
| `sources` | independent network sources | [Electrical Sources](Electrical-Sources.md) |
| `units` with `type: vsc` | voltage-source converters | [Voltage-Source Converters](Voltage-Source-Converters.md) |
| converter `bridge`, `pwm`, `ctrl`, `meas`, `protection` | bridge model and sampled control | [Bridge and PWM](Bridge-and-PWM.md) |
| `units` with `type: dc_converter` or `wpt` | DC converters and coupled links | [DC converters and WPT](DC-Converters-and-WPT.md) |
| `signals` | time-varying controller reference tables | [Reference Signals](Reference-Signals.md) |
| `events` and `simulation.initial` | event timeline and continuation | [Events Timeline](Events-Timeline.md) |
| `simulation.output` and `simulation.progress` | files, signals and progress | [Output and Results](Output-and-Results.md) |
| `simulation.solver` | integration settings | [Solvers](Solvers.md) |
| `simulation.solver.subsystems` | subsystem step ratios | [Multirate Simulation](Multirate-Simulation.md) |
| `meta` | optional title and description | — |

All top-level sections except `base` and `simulation` are optional. A model without a component
section simply omits it.

## Interface views

Every simulation file defines `@circuit`; additional named views are optional. Only view headers use
`@`; parameter domains such as `base:`, `simulation:` and `events:` remain ordinary YAML mappings.

```text
@circuit
Grid -- PCC -- VSC
PCC -- Load
&Loads: Load

@VSC
PLL.theta -> Current.theta
Current -> PWM

VSC: vsc, gfl
  PLL: SRFPLL
  Current: CurrentLoop
  PWM: ThreePhaseModulation

base:
  s_base: 2.0e6

simulation:
  t_end: 1.0
```

`@circuit` is the required system view; `@Name` defines a detailed view for a named Unit or
subsystem. Within a view:

- `A -- B` is an undirected electrical connection and `A -> B` is a directed signal connection;
- connections may be chained and may address named ports, such as `A.output -> B.input`;
- `&Group: A, B` creates a collapsible group, and groups may contain other groups;
- `Name: type` declares the displayed type, with optional indented block declarations;
- repeated names identify the same displayed object within that view.

A view ends at the next `@` header or parameter domain. A declaration always includes its displayed
type (`VSC: vsc`, not `VSC:`); quoted parameter keys and the standard domains (`base`, `buses`,
`branches`, `sources`, `units`, `elements`, `signals`, `events`, `simulation` and `meta`) are
unambiguous boundaries.

View blocks describe presentation. The parameter mappings remain the source of simulation values,
and their schemas are documented in the component and simulation pages linked above. See
[Graphical Interface](Graphical-Interface.md) for rendering and layout behavior.

Resolved files written by `inspect --resolved`, a simulation run or exported C++ retain the source
views. A simulation constructed directly from parameters writes an empty `@circuit` view.

## Names and paths

Names are unique within each of `buses`, `branches`, `sources`, `elements`, `units` and `signals`.
They must be non-empty and cannot contain a dot. Dots separate public configuration, state and
signal paths:

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
