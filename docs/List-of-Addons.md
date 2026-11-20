# List of Addons

PESLite addons extend the core registries or provide optional functions without adding dispatch to
the simulation loop. `peslite` installs the standard addons; `peslite-full` adds the full addons.

Function addons use explicit module imports:

```python
from peslite.addons.functions.plotting import plot_csv
from peslite.addons.functions.online_analysis import small_signal
```

Component and control addons register their implementations when parameters are constructed, so
simulation files use their component types and loop classes directly.

| Addon | Version | Categories | Installation | Dependencies | Purpose |
| --- | --- | --- | --- | --- | --- |
| `injection` | 1.0.0 | component | standard | — | Series-voltage and shunt-current perturbation sources. |
| `magcomponent` | 1.0.0 | component | standard | — | Magnetic components and compensated links. |
| `magport` | 1.0.0 | component, function | full | `magcomponent` | External magnetic data and field reconstruction. |
| `magcircuit` | 1.0.0 | component, function | full | `magcomponent`, `magport` | Nonlinear magnetic circuits and linear reduction. |
| `thermal` | 1.0.0 | component, function | standard | — | Thermal networks, components, losses and reports. |
| `thermalport` | 1.0.0 | component | full | `thermal` | External thermal-law adapter. |
| `mpc` | 1.0.0 | control | standard | — | Continuous- and finite-control-set model predictive control. |
| `debug` | 1.0.0 | function | standard | `offline_analysis`, `online_analysis`, `reference_signals` | Structured simulation diagnostics and `.pesdbg` execution. |
| `offline_analysis` | 1.0.0 | function | standard | `injection` | Frequency response, recorded identification, port matrices and loop gain. |
| `legacy_yaml` | 1.0.0 | function | standard | — | Legacy YAML simulation-file compatibility. |
| `online_analysis` | 1.0.0 | function | standard | — | Online stability, operating-point, response and energy analysis. |
| `plotting` | 1.0.0 | function | standard | — | Waveform plotting from completed results. |
| `reference_signals` | 1.0.0 | function | standard | — | Table, expression, PRBS, multitone and chirp references. |

Project-local addons use the same category modules under a sibling `PESaddons/` directory; see
[Project Workspace](Project-Workspace.md).
