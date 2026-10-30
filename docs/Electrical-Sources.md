# Electrical Sources

Independent network sources are configured under `sources` or as a `voltage_source` element.
Converter DC supplies are configured inside the Unit `dclink`; see
[Voltage-Source Converters](Voltage-Source-Converters.md) and
[DC Converters and Wireless Power Transfer](DC-Converters-and-WPT.md).

## AC voltage source

The top-level `sources` section defines a sinusoidal emf behind a series R-L impedance:

```yaml
sources:
  grid:
    bus: pcc
    x_pu: 0.4
    r_pu: 0.04
    v_pu: 1.0
    f: 50.0
    angle: 0.0
```

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `bus` | yes | — | terminal bus or Unit port |
| `l` or `x_pu` | yes | — | series inductance |
| `r` or `r_pu` | no | 0 | series resistance |
| `v` or `v_pu` | no | system voltage base | phase-peak emf magnitude |
| `f` | no | `base.f0` | frequency in Hz |
| `angle` | no | 0 | initial angle in rad |

A three-phase target uses the complex space-vector representation. A scalar target uses a real
single-phase sinusoid. The source and target phase counts are resolved from the connection.

## Constant scalar voltage source

Use `elements.type: voltage_source` for a constant scalar voltage behind resistance:

```yaml
elements:
  supply: {type: voltage_source, bus: input, v: 48.0, r: 0.1}
```

| Field | Required | Constraint |
|---|---:|---|
| `bus` | yes | scalar bus or Unit port |
| `v` | yes | finite V |
| `r` | yes | positive ohm |

## Converter DC source

Converter Units configure their DC supply under `dclink`:

```yaml
dclink:
  vdc_ref: 1500.0
  capacitor: {c: 0.02, r_esr: 0.05}
  source: {type: current, i: 100.0, k: 0.0}
```

| Field | Required | Meaning |
|---|---:|---|
| `vdc_ref` | yes | rated DC voltage and DC voltage base |
| `capacitor.c` | when a capacitor is present | capacitance in F |
| `capacitor.r_esr` | no | capacitor ESR in ohm |
| `source.type` | no | `voltage`, `current` or `none` |
| `source.v`, `source.r` | voltage source | emf and series resistance |
| `source.i`, `source.k` | current source | current and voltage-droop gain |

Without a DC-link capacitor, the source must be a voltage source. A voltage source with a
capacitor requires positive source resistance or capacitor ESR.

## Events

Top-level AC sources support `connect` and `disconnect` events:

```yaml
events:
  connect_grid: {type: connect, target: grid, t: 0.1, ramp: 0.05}
```

See [Events Timeline](Events-Timeline.md) for ordering and ramp behavior.
