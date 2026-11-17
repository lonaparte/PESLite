# Magnetic Components

PESLite connects magnetic devices to the electrical circuit through ordinary component ports.
`MagComponent` is the abstract component boundary for project-defined magnetic devices: it adds an
optional physical magnetic description to the normal electrical component contract without
prescribing a device type. Its electrical behavior may be supplied by a netlist, a constitutive law
or imported data.

The standard PESLite installation includes the `magcomponent` addon. `peslite-full` additionally
includes the `magport` and `magcircuit` addons.

## MagComponent interface

A magnetic component defines named windings, their terminal connections and their electrical
coupling. Each winding may include copper resistance. The ordinary component ports and `network()`
declaration form the required electrical model.

An optional `magnetic` section records physical branches, flux-continuity nodes and signed
branch-to-winding coupling. This description does not replace the electrical model. PESLite uses
the electrical declaration for simulation, while addons and external tools may use the magnetic
description without changing the circuit connections.

PESLite validates the component during assembly and then runs it through the same netlist and solver
path as other components. Every winding publishes:

| Signal | Meaning |
|---|---|
| `<element>.i_<winding>` | winding current in A |
| `<element>.u_<winding>` | terminal voltage, including winding resistance, in V |
| `<element>.copper_loss_W` | winding loss in W |
| `<element>.core_loss_W` | loss reported by the magnetic constitutive law in W |
| `<element>.loss_W` | total magnetic component loss in W |

The component owns its electrical states, stored energy and losses, so restart, continuation and
energy accounting use the normal PESLite mechanisms.

For temperature-dependent winding resistance, set `temperature_source`,
`temperature_coefficient_per_K` and `temperature_reference_K`. The source is an ordinary public
temperature signal; the magnetic model remains independent of the thermal model that provides it.

## Example

The bundled `magcomponent-wpt-example` connects a two-winding `MagComponent` between ordinary
source and receiver buses. The magnetic component supplies the winding coupling, while excitation
and loading remain ordinary electrical components. Run it in the same way as the other bundled
cases:

```bash
peslite magcomponent-wpt-example
```

The complete component definition is kept in the example file rather than duplicated here.

## `magport` addon

`MagPort` connects a magnetic constitutive law to ordinary electrical winding terminals. A constant
inductance matrix uses the ordinary coupled-inductor netlist model. A nonlinear law uses winding
flux linkages as PESLite states and maps them, together with any additional magnetic states, to
winding currents:

$$
\dot{\boldsymbol\lambda}
=\boldsymbol u-\boldsymbol R\boldsymbol i,
\qquad
\boldsymbol i=g(t,\boldsymbol\lambda,\boldsymbol z).
$$

For an analytical or externally coupled law, implement `currents()` and add `derivatives()` when
the law has internal state. Project-defined elements may inherit `MagPort` and pass the law to its
constructor; winding ports, electrical records and state registration remain shared. Every entry
in `initial_state` becomes an ordinary restartable PESLite state. A conservative law may also
provide `stored_energy()` and `stored_energy_rate()` for energy accounting.

When the surrounding circuit needs a winding-current rate, PESLite differentiates the
constitutive law along its registered state derivatives and solves the resulting algebraic group
with the circuit. Series inductors and other rate-dependent network structures therefore use the
same law without a separate magnetic topology.

## `magcircuit` addon

`MagCircuit` evaluates a branch-based magnetic network during simulation. Each branch supplies one
constitutive description, `nodes` define flux-continuity equations and `coupling` gives the signed
winding turns on each branch:

```yaml
elements:
  magnetic_network:
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

A branch provides exactly one constitutive relation:

- `reluctance`;
- `area`, `path_length` and `mu_r`;
- `area`, `path_length` and a monotone `bh_curve`; or
- matching monotone `flux` and `mmf` samples.

If every branch is linear, the addon reduces the complete circuit once to its exact winding
inductance before simulation. If any branch is nonlinear, winding flux linkages remain PESLite
states and a reduced damped Newton iteration solves the independent magnetic paths.

Programmatic magnetic circuits use `MagCircuitLaw`. Its `mmf(t, branch_flux, state)` callback
returns branch MMF, and `differential(t, branch_flux, state)` returns the branch-MMF Jacobian.
Optional internal states use `initial_state` and `state_derivatives`.

Optional `hysteresis`, `core_loss` and `eddy_current` mappings retain provider data. The built-in
branch laws do not evaluate those mappings, so PESLite emits a construction-time warning when they
are present.

## External magnetic data

External files are optional. Use them when an external calculation or measurement supplies a
constitutive model, or when several simulation cases reuse the same data. PESLite loads the
constitutive result and owns all simulation states; it does not run the external tool during a
simulation.

### `magport` files

A `magport` file may contain either a constant inductance matrix or a sampled nonlinear
flux-linkage-to-current map:

```yaml
elements:
  magnetic_device:
    type: magport
    file: models/magnetic-device.npz
    windings:
      primary: {positive: primary, negative: ground, r: 0.08}
      secondary: {positive: secondary, negative: ground, r: 0.10}
```

Relative paths start beside the simulation file. Winding names and their order must match the file.

| Data | Required | Meaning |
|---|---:|---|
| `winding_names` | yes | ordered winding names |
| `inductance` | one law | self- and mutual-inductance matrix in H |
| `flux_linkage` and `current` | one law | sampled nonlinear map shaped `(samples, windings)` |
| `physical` | no | JSON-compatible source metadata |
| `magnetic` | no | physical branches, continuity nodes and winding coupling |
| `field_points` and `field_basis` | no | spatial coordinates and a linear current-to-field map |

Exactly one constitutive law is stored. For several windings, nonlinear interpolation is defined
only inside the convex hull of the sampled flux-linkage vectors, so the samples must cover the
complete operating range.

Create a `magport` file with `write_magport`:

```python
from peslite.addons.functions import write_magport

write_magport(
    "models/magnetic-device.npz",
    winding_names=("primary", "secondary"),
    inductance=[[100e-6, -20e-6], [-20e-6, 100e-6]],
    physical=source_data,
    magnetic=magnetic_description,
    field_points=field_points,
    field_basis=field_basis,
)
```

Replace `inductance=` with matching `flux_linkage=` and `current=` arrays for a sampled nonlinear
law. The optional `magnetic` value records the same branch, node and winding-coupling structure used
by `magcircuit`, but does not replace the stored constitutive law.

### Field reconstruction

When a `magport` file contains field data, reconstruct the field after simulation:

```python
from peslite.addons.functions import magnetic_field

field = magnetic_field(
    result,
    "models/magnetic-device.npz",
    element="magnetic_device",
)
```

`field.time`, `field.points` and `field.flux_density` contain the recorded times, coordinates and an
array shaped `(times, points, 3)`. Pass `points=` to interpolate inside the stored field map. Field
reconstruction is post-processing and adds no work to electrical simulation steps.

### Exporting a linear magnetic circuit

Reduce an externally prepared linear magnetic circuit to a reusable `magport` file with
`export_magnetic_circuit`:

```python
from peslite.addons.functions import export_magnetic_circuit

export_magnetic_circuit(
    "models/magnetic-device.npz",
    winding_names=("primary", "secondary"),
    reluctance=reluctance,
    coupling=turns,
    incidence=incidence,
)
```

`reluctance` is the branch reluctance vector or matrix. Supplying `field_points` and
`field_per_flux` also stores a linear field-reconstruction basis. This helper performs the linear
reduction offline; `magcircuit` performs the corresponding reduction during assembly.
