# Magnetic Components

`MagComponent` is the magnetic component addon installed with `peslite`. It connects magnetic
devices to the electrical circuit through ordinary PESLite ports; `peslite-core` contains no
magnetic implementation. Its electrical behavior may be a netlist, a constitutive law or an
imported model, and its physical magnetic description is optional.

## Component behavior

A winding component defines named windings, their terminal connections and their electrical
coupling. Each winding may include copper resistance.

The ordinary component ports and `network()` declaration are the required electrical domain. An
optional `magnetic` section records the corresponding physical branches, flux-continuity nodes and
signed branch-to-winding coupling. The electrical model remains authoritative for simulation;
external processing may use the magnetic domain without changing the circuit connections.

PESLite checks the component while assembling the circuit and then uses the same solver path as
other components. `coupled_coils` uses a constant inductance matrix as an optimized electrical
representation and runs no separate magnetic solver.

For every winding, the component exposes:

| Signal | Meaning |
|---|---|
| `<element>.i_<winding>` | winding current in A |
| `<element>.u_<winding>` | terminal voltage, including winding resistance, in V |

PESLite includes all self- and mutual-inductance terms in stored energy, accounts for copper loss
and preserves the electrical states across restart and continuation.

## Example: Coupled coils

`coupled_coils` is the constant-inductance implementation included in `MagComponent`. Its inductance matrix is written directly in the simulation file, so it requires
neither an external model file nor the optional full installation:

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

## MagPort addon

Install `peslite-full` to use `magport` and `magcircuit`. `MagComponent` and its `coupled_coils`
implementation remain available with the default installation.

`magport` connects an explicit magnetic constitutive law to ordinary electrical winding terminals.
Each winding flux linkage is a PESLite state. The law maps the winding flux-linkage vector and any
additional magnetic states to winding currents, while terminal voltage advances flux linkage:

\[
\dot{\boldsymbol\lambda}
=\boldsymbol u-\boldsymbol R\boldsymbol i,
\qquad
\boldsymbol i=g(t,\boldsymbol\lambda,\boldsymbol z).
\]

A MagPort file can contain either a constant inductance matrix or a sampled nonlinear
`flux_linkage`-to-`current` map. Use it for results from FEM, an analytical calculation or another
external tool, or when the same constitutive data is reused by several cases:

```yaml
elements:
  transformer:
    type: magport
    file: models/transformer.npz
    windings:
      primary: {positive: primary, negative: ground, r: 0.08}
      secondary: {positive: secondary, negative: ground, r: 0.10}
```

Relative paths start beside the simulation file. Winding names and their order must match the file
exactly.

External tools may calculate geometry, materials and optional field data. PESLite reads the
constitutive result and owns all simulation states; it does not run the external tool during a
simulation.

The file contains data rather than executable objects:

| Data | Required | Meaning |
|---|---:|---|
| `winding_names` | yes | ordered winding names |
| `inductance` | one law | self- and mutual-inductance matrix in H |
| `flux_linkage` and `current` | one law | sampled nonlinear map, shaped `(samples, windings)` |
| `physical` | no | JSON-compatible source metadata |
| `magnetic` | no | validated physical branch, node and winding-coupling description |
| `field_points` and `field_basis` | no | spatial coordinates and a linear current-to-field map |

Exactly one constitutive law is stored. PESLite checks the file format, winding order, constitutive
data and optional field data while assembling the circuit.

Write arrays from an external calculation in this format with `write_magport`:

```python
from peslite.addons.functions import write_magport

write_magport(
    "models/transformer.npz",
    winding_names=("primary", "secondary"),
    inductance=[[100e-6, -20e-6], [-20e-6, 100e-6]],
    physical=source_data,
    magnetic=magnetic_description,
    field_points=field_points,
    field_basis=field_basis,
)
```

`physical` is optional metadata. `field_points` has shape `(points, 3)`, and `field_basis` has shape
`(points, 3, windings)` and maps winding currents to magnetic flux density.

The optional `magnetic` mapping uses named branches, nodes and couplings. A branch provides one
constitutive description: `reluctance`; `area`, `path_length` and either `mu_r` or `bh_curve`; or
matching `flux` and `mmf` samples. Optional `hysteresis`, `core_loss` and `eddy_current` mappings
preserve provider parameters without implying that the selected electrical law evaluates them.

Write sampled nonlinear data by replacing `inductance=` with matching arrays:

```python
write_magport(
    "models/saturating-core.npz",
    winding_names=("primary", "secondary"),
    flux_linkage=flux_linkage_samples,
    current=current_samples,
)
```

For an analytical or externally coupled law, implement `currents()` and add `derivatives()` when
the law declares internal state. `MagPortElement` turns that law into a configurable addon without
reimplementing winding ports:

```python
from dataclasses import dataclass

from peslite.addons.components import register_component_type
from peslite.addons.components.magport import MagPortElement
from peslite.addons.components.magnetic import MagneticParams

class CoreLaw:
    winding_names = ("coil",)
    initial_state = {"memory": 0.0}

    def currents(self, t, flux_linkage, state):
        return constitutive_current(t, flux_linkage, state)

    def derivatives(self, t, flux_linkage, currents, state):
        return {"memory": memory_rate(t, flux_linkage, currents, state)}

@register_component_type
class CoreElement(MagPortElement):
    type = "core_model"

    @dataclass(frozen=True, kw_only=True)
    class Params(MagneticParams):
        type: str = "core_model"

    def __init__(self, name, cfg, buses, _params):
        super().__init__(name, cfg, buses, CoreLaw())
```

Every entry in `initial_state` becomes an ordinary restartable PESLite state. A conservative law
may also implement `stored_energy()` and `stored_energy_rate()` for strict energy accounting;
otherwise its constitutive calculation remains available with the default energy declaration.

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

## MagCircuit addon

`magcircuit` evaluates the `magnetic` branch network during simulation. `coupling` maps named
branches to named windings, while each entry in `nodes` states one flux-continuity equation. A
branch may use reluctance, material geometry or a monotone sampled flux-to-MMF law:

```yaml
elements:
  transformer:
    type: magcircuit
    windings:
      primary: {positive: primary, negative: ground, r: 0.08}
      secondary: {positive: secondary, negative: ground, r: 0.10}
    magnetic:
      branches:
        core:
          area: 1.0e-4
          path_length: 0.08
          bh_curve: [[-1.0, -800.0], [0.0, 0.0], [1.0, 800.0]]
        gap: {reluctance: 2.0e6}
      nodes:
        continuity: {core: 1.0, gap: -1.0}
      coupling:
        core: {primary: 20.0}
        gap: {secondary: 20.0}
```

At each constitutive evaluation, the addon solves branch MMF relations, flux continuity and winding
flux linkages as one magnetic system. A fully linear circuit is reduced analytically once; nonlinear
branches use the same winding state and electrical port path through `MagPortModel`.

Programmatic models use `MagCircuitLaw`. Its `mmf(t, branch_flux, state)` callback may contain
coupled nonlinear material behavior, and optional internal states use the same `initial_state` and
`state_derivatives` contract as MagPort. Supplying a branch-MMF Jacobian through `differential`
accelerates the nonlinear solve but is not required.

For offline linear reduction, write the exact equivalent through MagPort:

```python
from peslite.addons.functions import export_magnetic_circuit

export_magnetic_circuit(
    "models/transformer.npz",
    winding_names=("primary", "secondary"),
    reluctance=reluctance,
    coupling=turns,
    incidence=incidence,
)
```

`reluctance` is the branch reluctance vector or matrix. Supplying `field_points` and
`field_per_flux` also stores a linear field-reconstruction basis. This offline helper is an
optimization for a linear circuit, not a limitation on the online `magcircuit` component.
