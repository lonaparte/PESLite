# Magnetic Components

PESLite connects magnetic devices to the electrical circuit through named windings. The magnetic
component addon defines `MagComponent` as the common component contract; it is not a separate type
written in a simulation file. `coupled_coils` is its constant-inductance implementation.

The default `peslite` installation includes this addon, while `peslite-core` contains no magnetic
implementation. Install `peslite-full` for `magcircuit` and `magport`.

Use `coupled_coils` when an inductance matrix describes the device. Use the `magcircuit` addon to
evaluate a branch-based magnetic network during simulation. Use `magport` when a constitutive model
is supplied as reusable external data. All three use the same winding terminals and published
electrical signals.

## Winding interface

A magnetic component declares each winding by name, its positive and negative electrical terminals
and its copper resistance. A terminal may be `ground`, a scalar bus or a scalar Unit port.

Every winding publishes:

| Signal | Meaning |
|---|---|
| `<element>.i_<winding>` | winding current in A |
| `<element>.u_<winding>` | terminal voltage, including winding resistance, in V |

PESLite includes self- and mutual-inductance terms in stored energy, accounts for copper loss and
preserves the electrical states across restart and continuation. Compensation capacitors and other
electrical elements remain separate components; see
[Passive Components](Passive-Components.md).

| Component type | Constitutive input | Runtime representation | Installation |
|---|---|---|---|
| `coupled_coils` | inductance matrix in the simulation file | constant-inductance netlist model | `peslite` |
| `magcircuit` | magnetic branches, continuity nodes and winding coupling | linear reduction or nonlinear magnetic solve | `peslite-full` |
| `magport` | reusable constitutive data in a MagPort file | constant-inductance netlist model or nonlinear flux-linkage states | `peslite-full` |

## Constant inductance

`coupled_coils` is the standard constant-inductance implementation. It requires neither an external
model file nor `peslite-full`:

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

Matrix rows and columns follow the winding declaration order. Diagonal entries are self
inductances; off-diagonal entries are mutual inductances. A negative mutual inductance reverses the
relative winding polarity. The matrix must be symmetric and positive definite.

PESLite reduces the matrix to modal current states during assembly and then uses the ordinary
netlist and solver path. There is no separate magnetic iteration in this representation.

## MagCircuit addon

The `magcircuit` addon describes a magnetic network directly in the simulation file. Each named
branch has one constitutive description, `nodes` define flux-continuity equations and `coupling`
gives the signed winding turns on each branch:

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
        leakage: {reluctance: 2.0e7}
      nodes:
        split: {core: 1.0, gap: -1.0, leakage: -1.0}
      coupling:
        core: {primary: 20.0}
        gap: {secondary: 20.0}
```

A branch provides exactly one of:

- `reluctance`;
- `area`, `path_length` and `mu_r`;
- `area`, `path_length` and a monotone `bh_curve`;
- matching monotone `flux` and `mmf` samples.

If every branch is linear, the addon reduces the complete circuit once to its exact winding
inductance and passes the same netlist model as `coupled_coils` to the PESLite solver. If any branch
is nonlinear, the winding flux linkages are PESLite states and a reduced damped Newton iteration
solves only the remaining independent magnetic paths.

Optional `hysteresis`, `core_loss` and `eddy_current` mappings retain parameters for a provider.
The built-in branch laws do not evaluate those mappings, so `magcircuit` emits a construction-time
warning when they are present.

Programmatic magnetic circuits use `MagCircuitLaw`. Its `mmf(t, branch_flux, state)` callback
returns branch MMF, and its required `differential(t, branch_flux, state)` callback returns the
branch-MMF Jacobian. Optional internal states use `initial_state` and `state_derivatives`.

## Custom constitutive laws

`peslite-full` also provides `MagPortElement` for a device that is easier to express as a direct
winding constitutive law than as a branch network. For a nonlinear law, PESLite advances winding
flux linkage according to

\[
\dot{\boldsymbol\lambda}
=\boldsymbol u-\boldsymbol R\boldsymbol i,
\qquad
\boldsymbol i=g(t,\boldsymbol\lambda,\boldsymbol z).
\]

The law supplies currents and, when it declares internal state, its state derivatives. The shared
element adapter supplies winding ports, electrical records and state registration:

```python
from dataclasses import dataclass

from peslite.addons.components import register_component_type
from peslite.addons.components.magnetic import MagneticParams
from peslite.addons.components.magport import MagPortElement


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

Every `initial_state` entry becomes an ordinary restartable PESLite state. A conservative law may
also implement `stored_energy()` and `stored_energy_rate()` for strict energy accounting.

## External magnetic data

External files are optional. PESLite does not require FEM or another magnetic tool to use magnetic
components. A MagPort file is useful when an external calculation or measurement supplies a
constitutive model, or when several simulation cases reuse the same data.

### MagPort files

The `magport` component connects a MagPort file to ordinary winding terminals:

```yaml
elements:
  transformer:
    type: magport
    file: models/transformer.npz
    windings:
      primary: {positive: primary, negative: ground, r: 0.08}
      secondary: {positive: secondary, negative: ground, r: 0.10}
```

Relative paths start beside the simulation file. Winding names and their order must match the file.
PESLite loads the constitutive result and owns all simulation states; it does not run the external
tool during simulation.

| Data | Required | Meaning |
|---|---:|---|
| `winding_names` | yes | ordered winding names |
| `inductance` | one law | self- and mutual-inductance matrix in H |
| `flux_linkage` and `current` | one law | sampled nonlinear map shaped `(samples, windings)` |
| `physical` | no | JSON-compatible source metadata |
| `magnetic` | no | physical branches, continuity nodes and winding coupling |
| `field_points` and `field_basis` | no | spatial coordinates and a linear current-to-field map |

Exactly one constitutive law is stored. A constant inductance matrix uses the same netlist model as
`coupled_coils`. A sampled nonlinear map uses winding flux linkages as states. For several
windings, interpolation is defined only inside the convex hull of the sampled flux-linkage vectors;
the samples must cover the complete operating range.

Create a MagPort file with `write_magport`:

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

Replace `inductance=` with matching `flux_linkage=` and `current=` arrays for a sampled nonlinear
law. `physical` is optional metadata. The optional `magnetic` value records the same named branch,
node and winding-coupling structure used by `magcircuit`, but does not replace the stored
constitutive law.

`field_points` has shape `(points, 3)`. `field_basis` has shape `(points, 3, windings)` and maps
winding currents to magnetic flux density.

### Field reconstruction

When a MagPort file contains field data, reconstruct the field after simulation:

```python
from peslite.addons.functions import magnetic_field

field = magnetic_field(
    result,
    "models/transformer.npz",
    element="transformer",
)
```

`field.time`, `field.points` and `field.flux_density` contain the recorded times, coordinates and an
array shaped `(times, points, 3)`. Pass `points=` to interpolate inside the stored field map. Field
reconstruction is post-processing and adds no work to electrical simulation steps.

### Exporting a linear magnetic circuit

Reduce an externally prepared linear magnetic circuit to a reusable MagPort file with
`export_magnetic_circuit`:

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
`field_per_flux` also stores a linear field-reconstruction basis. This helper performs the linear
reduction offline; `magcircuit` performs the corresponding reduction during assembly.
