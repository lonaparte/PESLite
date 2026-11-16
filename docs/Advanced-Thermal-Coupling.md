# Advanced Electro-Magneto-Thermal Coupling (Issue #40, Level 3)

This `peslite-full` feature connects electrical and magnetic **realized**
loss sources to a standard thermal network using the original PESLite
`Model` inputs, state registry and integrators. It extends
[Electrothermal Coupling](Electrothermal-Coupling.md). The `peslite-core`
distribution remains lightweight.

## Advanced assembly and ownership

- The **standard** `peslite` layer supplies compact `LumpedThermalNetwork`,
  `ElectroThermalModel`, semiconductor loss data, device comparisons and
  actual temperature-dependent electrical resistance where explicitly modeled.
- The `peslite-full` **full-addons** group owns `MagPort`, `MagCircuit`,
  `SharedThermalNetwork`, `ReducedThermalElement` and optional magnetic
  analysis/field helpers. It does not introduce another physics kernel.
- `MagPortModel` publishes **`copper_loss_W`** for its actual
  `I²R(T)` winding law and **`core_loss_W`** only when the constitutive
  magnetic law *explicitly* provides a `dissipated_power` model.
  `loss_W` sums those disjoint mechanisms. Constant-inductance
  `CoupledCoilsModel` also exposes realized copper loss and explicitly
  claims **zero modeled core loss**, not a fabricated magnetic estimate.
- `temperature_source` on a `magport` or `magcircuit` element connects
  junction/sink temperature to the winding-resistance equation; 
  `temperature_coefficient_per_K` and `temperature_reference_K` control the
  law. The connection is through normal Model signals; the source, magnetic
  and thermal states remain in one simulation vector.

## Shared heatsinks and multiphysics losses

Use a configured shared sink with two separate physical loss producers:

```yaml
elements:
  common_cooling:
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
      electric:
        {path: load.power, node: inverter}
      magnetic:
        {path: coil.loss_W, node: winding}
    ambient_K: 300.0
    initial_K: 300.0
```

The electrical and magnetic sources must exist in the same circuit and
publish **continuous, nonnegative watts**; this component does not
invent a network or a magnetic loss model. `common_cooling.T_sink_K`
can feed a separate supported electrical or magnetic temperature
input, forming genuine bidirectional coupling.

Thermal balance uses the same physical network:
`d(sum C_i T_i)/dt = sum(source_heat_W) - ambient_cooling_W`.
A shared sink can receive heat from multiple branches, conserving
inter-node heat exactly. Initial conditions and states are restartable.

## Imported FEM and reduced-order thermal response

`ReducedThermalModel` supports an externally constructed **stable,
continuous-time linear ROM**:
`dx/dt = A x + B P`,
`T_j = T_ambient + C x + D P`.
All variables use SI units (A: 1/s, B: K/J, C: dimensionless,
D: K/W), and nonnegative thermal gains plus strictly negative-real-part
A eigenvalues are enforced. This is a validated adapter for supplied
thermal models—not an internal full FEM field solver or a PDE mesher.

Import a previously identified model (not arbitrary field meshes):

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

```python
from peslite.addons.components.advanced_thermal import load_thermal_rom
reduced = load_thermal_rom("thermal.json")
```

Or register a `thermal_rom` element, providing these matrices and an
`heat_source` name, to participate in the original PESLite simulation.
`foster_step_response` computes exact analytical step responses for
validating reduced systems against physical RC reference models.

The `ThermalLaw` protocol is an extension boundary for custom external
models; a provider must evaluate RHS deterministically and without
network side effects during speculative integrator evaluations. External
co-simulation communication must be committed at accepted boundaries,
not inside RHS calls. Unavailable FEM/field reconstruction is reported
as unsupported rather than silently substituted.

## Accuracy, compatibility and solver limitations

1. Distinguish realized electrical/magnetic power from **additional
   estimated semiconductor heat**. The electrical energy closure remains
   unchanged; a reported heat budget is not automatically a new electrical sink.
2. Do not apply arbitrary switching energies at tentative transitions.
   Use committed event IDs and the standard restartable impulse interface.
3. Cauer/lumped shared cooling is a physical network; Foster impedance
   is referenced to fixed ambient and cannot be grafted directly as a
   series case/heatsink network.
4. Use the existing Python single-rate reference for nonlinear feedback
   and commutating converters. PESLite's current multirate window
   restrictions still apply; this add-on does not claim an independently
   validated external multirate co-solver. Temperature-dependent electrical
   callbacks and nonlinear magnetic constitutive models require Python;
   unsupported standalone C++ export configurations should be rejected.
5. Imported reduced models provide only declared temperatures/heat
   dynamics, **not** reconstructed spatial FEM fields or universal
   hysteresis/eddy-current physics.

For underlying magnetic modeling read
[Magnetic Components](Magnetic-Components.md) and
[Extending PESLite](Extending-PESLite.md).
