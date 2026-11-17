# Multiphysics Analysis

Electrical, magnetic and thermal components are independent PESLite subsystems. Coupling does not
make one physical domain an implementation of another: assembly only connects compatible public
signals through the existing model connection graph.

## Signal contract

Electromagnetic and thermal coupling uses two directed quantities without assigning ownership or
priority to either subsystem:

| Published quantity | Consumed quantity | Meaning |
|---|---|---|
| component `loss_W` | thermal `heat_W` | dissipated power entering a thermal model |
| thermal `temperature_K` | component `temperature_K` | temperature affecting a constitutive law |

Multi-node and multi-source models expose the same quantities under configured node or source names.
Omitting one connection leaves only that direction uncoupled.

## Magnetic and thermal coupling

Every `MagComponent` publishes `copper_loss_W`, `core_loss_W` and their sum `loss_W`. A magnetic
constitutive law reports material loss independently of winding loss. Temperature-dependent winding
resistance uses `temperature_source`, `temperature_coefficient_per_K` and
`temperature_reference_K`.

For example, add the following connection fields to an existing magnetic element and thermal
element; their remaining definitions do not change:

```yaml
# Magnetic element fields
temperature_source: thermal.temperature_K
temperature_coefficient_per_K: 0.0039
temperature_reference_K: 300.0

# Thermal element field
heat_source: magnet.loss_W
```

The magnetic model neither contains nor selects the thermal implementation. The thermal model only
sees heat and does not depend on the magnetic constitutive law. Either side can be replaced while the
signal contract remains unchanged.

## Electrical and thermal coupling

`thermal_resistor` is an electrical resistance whose constitutive equation uses an external
temperature signal:

```yaml
elements:
  load:
    type: thermal_resistor
    bus: output
    temperature_source: thermal_load.temperature_K
    r_reference_Ohm: 2.0
    alpha_per_K: 0.004
    reference_K: 300.0

  thermal_load:
    type: electrothermal
    heat_source: load.heat_W
    c_j_J_K: 0.2
    r_jc_K_W: 1.5
    c_c_J_K: 1.0
    r_ca_K_W: 3.0
    ambient_K: 300.0
    initial_K: 300.0
```

Its current is `u / R(T)` and its published heat is `R(T) i²`; it introduces no artificial
electrical state. A nonpositive resistance is rejected.

## Custom boundaries

`ThermalPort` accepts an external thermal law with named heat inputs, named temperature outputs,
`initial_state`, `rhs()` and `temperatures_K()`. `initial_state` may be empty for an algebraic
thermal law. A law may declare additional `signal_names` and provide `signals()`. Magnetic
constitutive laws use the independent `MagComponent` and `MagPort` interfaces described in
[Magnetic Components](Magnetic-Components.md).

Custom laws are evaluated during speculative solver stages and must be deterministic and free of
external side effects. Communication with an external program belongs at accepted simulation
boundaries. Thermal models and loss-analysis interfaces are documented in
[Thermal Analysis](Thermal-Analysis.md).
