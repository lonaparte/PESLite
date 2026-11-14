# Magnetic Components

`MagComponent` is a component defined by PESLite for magnetic devices with one or more coupled
windings. Like a resistor, capacitor or inductor, it connects directly to the electrical circuit.
It does not require an external file.

## Component behavior

A `MagComponent` defines named windings, their terminal connections and their electrical coupling.
Each winding may include copper resistance. An inductance matrix contains every self- and
mutual-inductance term; its rows and columns follow the winding order.

PESLite checks the component while assembling the circuit and then uses the same netlist and solver
path as other components. No separate magnetic solver runs during the simulation.

For every winding, the component exposes:

| Signal | Meaning |
|---|---|
| `<element>.i_<winding>` | winding current in A |
| `<element>.u_<winding>` | terminal voltage, including winding resistance, in V |

PESLite includes all self- and mutual-inductance terms in stored energy, accounts for copper loss
and preserves the electrical states across restart and continuation.

## Example: Coupled coils

`coupled_coils` is one concrete `MagComponent`. Its inductance matrix is written directly in the
simulation file, so it requires neither an external model file nor the optional full installation:

```yaml
buses:
  primary: {phases: 1}
  secondary: {phases: 1}

elements:
  coils:
    type: coupled_coils
    windings:
      primary: {positive: primary, negative: ground, r: 0.08}
      secondary: {positive: secondary, negative: ground, r: 0.10}
    inductance:
      - [100.0e-6, -20.0e-6]
      - [-20.0e-6, 100.0e-6]
```

`ground` is the reference node; other terminals name a single-phase bus or scalar Unit port. Matrix
rows and columns follow the winding declaration order. Negative mutual inductance reverses relative
winding polarity. Compensation capacitors and other electrical elements remain separate components;
see [Passive Components](Passive-Components.md).

## External model files

The external-file interface is optional. Use it when component data comes from FEM, an analytical
calculation or another external tool, or when the same data must be reused by several cases. The
resulting component has the same circuit behavior as `coupled_coils`:

```yaml
elements:
  transformer:
    type: magnetic_component
    windings:
      primary: {positive: primary, negative: ground, r: 0.08}
      secondary: {positive: secondary, negative: ground, r: 0.10}
    model: {file: models/transformer.npz}
```

Relative paths start beside the simulation file. Winding names and their order must match the file
exactly.

External tools calculate geometry, materials, inductance and optional field data. PESLite reads the
result and performs the electrical simulation; it does not run the external tool during simulation.

The file contains data rather than executable objects:

| Data | Required | Meaning |
|---|---:|---|
| `winding_names` | yes | ordered winding names |
| `inductance` | yes | self- and mutual-inductance matrix in H |
| `physical` | no | JSON-compatible source metadata |
| `field_points` and `field_basis` | no | spatial coordinates and a linear current-to-field map |

PESLite checks the file format, winding order, inductance matrix and optional field data while
assembling the circuit.

Convert arrays from an external calculation into this format with `import_magnetic_model`:

```python
from peslite.addons.functions import import_magnetic_model

import_magnetic_model(
    "models/transformer.npz",
    winding_names=("primary", "secondary"),
    inductance=[[100e-6, -20e-6], [-20e-6, 100e-6]],
    physical=source_data,
    field_points=field_points,
    field_basis=field_basis,
)
```

`physical` is optional metadata. `field_points` has shape `(points, 3)`, and `field_basis` has shape
`(points, 3, windings)` and maps winding currents to magnetic flux density.

### Reluctance-network results

`build_reluctance_model` reduces a linear reluctance network and writes the same external model
format:

```python
from peslite.addons.functions import build_reluctance_model

build_reluctance_model(
    "models/transformer.npz",
    winding_names=("primary", "secondary"),
    reluctance=reluctance,
    coupling=turns,
    incidence=incidence,
)
```

`reluctance` is the branch reluctance vector or matrix, `coupling` maps winding currents to branch
magnetomotive force, and optional `incidence` enforces flux continuity. Supplying `field_points` and
`field_per_flux` also stores a field basis.

### Field reconstruction

When the external model contains field data, reconstruct the field after simulation:

```python
from peslite.addons.functions import magnetic_field

field = magnetic_field(
    result,
    "models/transformer.npz",
    element="transformer",
)
```

`field.time`, `field.points` and `field.flux_density` contain the recorded times, coordinates and an
array shaped `(times, points, 3)`. Pass `points=` to interpolate inside the stored field map. This
post-processing does not add work to electrical simulation steps.

### Nonlinear and stateful external models

The external-file interface supplies linear inductance data. A nonlinear, hysteretic or otherwise
stateful external model uses the existing
[black-box model](Black-Box-Models.md) interface. Flux, hysteresis memory, filters and learned hidden
state must be registered as PESLite state, while ordinary physical ports connect the model to the
circuit. ONNX models use the same boundary when ONNX Runtime is installed.
