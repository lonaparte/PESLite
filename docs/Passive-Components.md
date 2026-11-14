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

## Loads and resistors

```yaml
elements:
  load: {type: load, bus: pcc, x_pu: 0.4, r_pu: 0.1}
  damping: {type: resistor, bus: pcc, r: 20.0}
```

| `type` | Required fields | Optional fields | Connection |
|---|---|---|---|
| `load` | `bus`, `l` or `x_pu` | `r` or `r_pu` | series R-L branch from the bus to ground |
| `resistor` | `bus`, `r` | `series` (default 0) | shunt resistor, or series resistor after the bus voltage driver |

`load.l` must be positive, `load.r` non-negative and `resistor.r` positive. Per-unit values use the
system base.

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

## Magnetic components and compensation

`magnetic_component` represents named windings through the ordinary component and netlist
interfaces. Its `matrix` model accepts a symmetric positive-definite inductance matrix.
`coupled_coils` is a concise configuration of the same component, not a second physical
implementation.

Each winding has positive and negative terminals connected to a named single-phase bus or scalar
Unit port. The literal `ground` denotes the reference node. Winding names must be identifiers; the
matrix row and column order follows `windings`. Negative mutual inductance reverses relative winding
polarity. Optional winding `r` is the series copper resistance in ohms.

Both forms publish `<element>.i_<winding>` (A), `<element>.u_<winding>` (V, including winding
resistance), and restartable `i_mode_1`, `i_mode_2`, ... states. Stored energy includes every
mutual-inductance term.

```yaml
buses:
  tx_drive: {phases: 1}
  tx_coil: {phases: 1}
  rx_coil: {phases: 1}
  rx_load: {phases: 1}

elements:
  drive: {type: voltage_source, bus: tx_drive, v: 24.0, r: 0.5}
  tx_comp: {type: capacitor, bus1: tx_drive, bus2: tx_coil, c: 35.0e-9}
  rx_comp: {type: capacitor, bus1: rx_coil, bus2: rx_load, c: 35.0e-9}
  load: {type: resistor, bus: rx_load, r: 5.0}
  coils:
    type: coupled_coils
    windings:
      tx: {positive: tx_coil, negative: ground, r: 0.08}
      rx: {positive: rx_coil, negative: ground, r: 0.10}
    inductance:
      - [100.0e-6, -20.0e-6]
      - [-20.0e-6, 100.0e-6]
```

The equivalent general configuration replaces `type` and `inductance` with:

```yaml
    type: magnetic_component
    model:
      type: matrix
      inductance:
        - [100.0e-6, -20.0e-6]
        - [-20.0e-6, 100.0e-6]
```

A separate `capacitor` add-on provides a two-terminal capacitance with optional series `r`
(ESR). Combine it with existing branches, resistors and switches to construct SS, LCC or
other compensation networks without changing the coupled magnetic component. Both elements
reuse the ordinary circuit netlist and the same integration and energy accounting paths.

The matrix model is linear and stationary; its inductance, winding count and terminals are
assembly-time settings. All connected winding terminals use scalar electrical nodes. FEM execution
and field reconstruction remain offline work.
