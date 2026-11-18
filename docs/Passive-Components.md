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
```

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `bus1`, `bus2` | yes | — | bus or Unit-port names |
| `l` or `x_pu` | yes | — | series inductance |
| `r` or `r_pu` | no | 0 | series resistance |

Branch current is positive from `bus1` to `bus2`. Both ports must use the same phase count.
Use [Events Timeline](Events-Timeline.md) to connect or disconnect a branch.

## Passive elements

```yaml
elements:
  load: {type: load, bus: pcc, x_pu: 0.4, r_pu: 0.1}
  damping: {type: resistor, bus: pcc, r: 20.0}
  compensation: {type: capacitor, bus1: input, bus2: output, c: 35.0e-9, r: 0.08}
```

| `type` | Required fields | Optional fields | Connection |
|---|---|---|---|
| `load` | `bus`, `l` or `x_pu` | `r` or `r_pu` | series R-L branch from the bus to ground |
| `resistor` | `bus`, `r` | `series` (default 0) | shunt resistor, or series resistor after the bus voltage driver |
| `capacitor` | `bus1`, `bus2`, `c` | `r` (default 0) | capacitance with optional series resistance between two scalar terminals |

`load.l` must be positive, `load.r` non-negative and `resistor.r` positive. Per-unit values use the
system base. Capacitor terminals must be distinct, `c` must be positive and `r` non-negative. The
capacitor exposes `<element>.u`; energy accounting includes its stored energy and series loss.

## Connections

Component ports accept a public bus name or Unit-port path. Examples:

```yaml
branches:
  dc_line: {bus1: converter.power, bus2: output, l: 1e-3, r: 0.05}
```

The phase count is resolved during assembly. A connection between incompatible explicit AC and DC
domains is rejected before simulation.

Project component types registered under `PESaddons/components` use the same `elements` mapping;
see [Extending PESLite](Extending-PESLite.md#custom-circuit-elements).
