# Passive Components

Passive components are configured under `buses`, `branches` and `elements`.

## Buses

A bus is a named electrical connection point with an optional shunt capacitor:

```yaml
buses:
  pcc: {c_pu: 0.02, r_d_pu: 0.5, phases: 3}
  output: {phases: 1}
```

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `c` or `c_pu` | no | none | shunt capacitance |
| `r_d` or `r_d_pu` | no | 0 | series damping resistance |
| `phases` | no | 3 | `1` for a real single-phase node or `3` for a three-phase space-vector node |

Without `c`, the bus only joins the ports of branches, sources, Units or other elements and adds no
state or physical component. Declaring an empty bus is allowed but not recommended because the
connection point then has no electrical effect. `r_d` is valid only with `c`.

## Branches

A branch is a switchable series R-L connection:

```yaml
branches:
  line: {bus1: grid_bus, bus2: pcc, x_pu: 0.2, r_pu: 0.02}
  shunt: {bus1: pcc, bus2: ground, l: 1e-3, r: 0.1}
```

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `bus1`, `bus2` | yes | — | bus or Unit-port names |
| `l` or `x_pu` | yes | — | series inductance |
| `r` or `r_pu` | no | 0 | series resistance |

Branch current is positive from `bus1` to `bus2`. Both ports must use the same phase count;
`ground` takes the representation of the other port.
Use [Events Timeline](Events-Timeline.md) to connect or disconnect a branch.

## RLC elements

```yaml
elements:
  load: {type: series_rlc, bus: pcc, x_pu: 0.4, r_pu: 0.1}
  damping: {type: resistor, bus: pcc, r: 20.0}
  compensation: {type: capacitor, bus1: input, bus2: output, c: 35.0e-9, r: 0.08}
```

| `type` | Values | Connection |
|---|---|---|
| `resistor` | positive `r` | `bus` or `bus1` and `bus2`; `series: true` is supported with `bus` |
| `capacitor` | positive `c` with optional series `r` | `bus1` to `bus2` |
| `series_rlc` | `r`, `l` and `c` in series; at least one must be present and nonzero | `bus` or `bus1` and `bus2`; `series: true` is supported with `bus` |
| `parallel_rlc` | any nonempty parallel combination of `r`, `l` and `c` | `bus` to ground or `bus1` to `bus2` |

Use `r_pu`, `l_pu` or `x_pu`, and `c_pu` for system-base per-unit values. Resistances are
positive for `resistor` and parallel RLC branches; a series resistance may also be zero.
Inductances and capacitances are positive when present. Every RLC element exposes
`<element>.i`, `<element>.voltage` and `<element>.power`; energy accounting includes its storage
and resistive loss.

With `bus`, an element is normally connected between that point and `ground`. For `resistor` and
`series_rlc`, `series: true` instead places it in the connection path. Use `bus1` and `bus2` for
an explicit two-terminal connection.

## Cables

The bundled cable component is a two-port element whose currents are positive into the cable:

```yaml
elements:
  line:
    type: cable
    bus1: sending
    bus2: receiving
    length_km: 5.0
    r_per_km: 0.05
    l_per_km: 0.4e-3
    c_per_km: 0.2e-6
```

`model: bergeron` is the default. It represents distributed L-C propagation with an exact travel
delay and splits the total series resistance into quarter-half-quarter sections. Use
`model: lumped` for the segmented R-L/G-C model; `sections` selects its number of shunt G-C cells
and `g_per_km` its shunt conductance. All per-kilometre fields use SI units, and the Bergeron model
requires `g_per_km: 0`.

## Connections

Component ports accept a public bus name, Unit-port path or the reserved `ground` reference.
`ground` is always available and is not declared under `buses`:

```yaml
branches:
  dc_line: {bus1: converter.power, bus2: output, l: 1e-3, r: 0.05}
elements:
  grounded_capacitor: {type: capacitor, bus1: output, bus2: ground, c: 1e-3}
```

An element using `bus` without `series: true` does not need a separate `bus2: ground`. A network
with no connection to `ground` remains floating. PESLite may select a numerical voltage reference
to solve it, but that reference does not add a physical ground branch. `ground` denotes ideal zero
voltage, not a neutral conductor or protective earth. The three-phase space-vector representation
contains no zero-sequence quantity, so neutral impedance, ground current and common-mode networks
require a multi-conductor model.

The phase count is resolved during assembly. A connection between incompatible explicit AC and DC
domains is rejected before simulation.

Project component types registered under `PESaddons/components` use the same `elements` mapping;
see [Extending PESLite](Extending-PESLite.md#custom-circuit-elements).
