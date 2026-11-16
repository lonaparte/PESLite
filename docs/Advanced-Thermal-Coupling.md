# Advanced Thermal Coupling

PESLite keeps electrical, magnetic and thermal model boundaries independent. A magnetic model does
not implement a thermal port, and a thermal model does not depend on `MagPort`. Multiphysics
coupling is formed by matching their public signals in the normal model connection graph.

## Coupling contract

The two directions of an electro-magneto-thermal connection are:

| Source | Target | Quantity |
|---|---|---|
| electrical or magnetic component | thermal model | dissipated heat in W |
| thermal model | electrical or magnetic component | temperature in K |

Magnetic components publish `copper_loss_W`, `core_loss_W` and their sum `loss_W`. A configured
`temperature_source` feeds temperature back into winding resistance. These signals are part of the
`MagComponent` contract and do not change the magnetic constitutive boundary.

A scalar thermal port consumes `heat_W` and publishes `temperature_K`; multi-node thermal models
expose the same quantities under their configured source and node names. Their state equations and
initial conditions remain ordinary PESLite states, so restart and solver selection use the existing
simulation path.

## Reduced-order thermal ports

The `thermal_rom` element is one implementation of the independent thermal-port contract. It uses
a stable continuous-time model

$$
\dot{\boldsymbol{x}} = \boldsymbol{A}\boldsymbol{x} + \boldsymbol{B}P,
\qquad
T = T_{\mathrm{ambient}} + \boldsymbol{C}\boldsymbol{x} + DP.
$$

`A_per_s`, `B_K_per_J`, `C` and `D_K_W` use SI units. Connect it to any component that publishes a
nonnegative loss signal:

```yaml
elements:
  thermal:
    type: thermal_rom
    heat_source: magnet.loss_W
    A_per_s: [[-0.125]]
    B_K_per_J: [0.5]
    C: [1.0]
    D_K_W: 0.0
    ambient_K: 300.0
```

The corresponding magnetic element can set:

```yaml
temperature_source: thermal.temperature_K
temperature_coefficient_per_K: 0.0039
temperature_reference_K: 300.0
```

Neither direction owns the other; omitting either connection leaves that direction uncoupled.

Use `load_thermal_rom()` when the same model is stored in a JSON file:

```json
{
  "format": "peslite-thermal-rom-v1",
  "units": "SI",
  "A_per_s": [[-0.125]],
  "B_K_per_J": [0.5],
  "C": [1.0],
  "D_K_W": 0.0,
  "ambient_K": 300.0
}
```

## Shared thermal networks

`shared_thermal_network` places several independent heat inputs on one lumped thermal network. Each
source maps a public loss path to a named thermal node:

```yaml
elements:
  cooling:
    type: shared_thermal_network
    capacities_J_K:
      inverter: 0.2
      winding: 0.5
      sink: 10.0
    links:
      - {node: inverter, other: sink, resistance_K_W: 0.5}
      - {node: winding, other: sink, resistance_K_W: 1.2}
      - {node: sink, other: null, resistance_K_W: 4.0}
    sources:
      electric: {path: load.power, node: inverter}
      magnetic: {path: magnet.loss_W, node: winding}
    ambient_K: 300.0
    initial_K: 300.0
```

The network publishes each `temperature_<node>_K`, total incoming heat and heat rejected to ambient.
Inter-node conduction is internal to this thermal component.

## Custom thermal laws

`ThermalPortModel` accepts a thermal law with named `initial_state`, `rhs()` and `temperature_K()`.
The law is evaluated during speculative solver stages, so it must be deterministic and free of
external side effects. Communication with an external solver belongs at accepted simulation
boundaries.
