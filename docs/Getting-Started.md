# Getting Started

This guide installs PESLite, runs a bundled case, explains the output directory and turns a
resolved example into a new simulation file.

## Install

```bash
python -m pip install peslite
```

Addon modules are included in the package. To install PESLite with all dependencies, including the
optional dependencies required by addons such as [PDF plotting](Plotting.md):

```bash
python -m pip install "peslite[all]"
```

Check the command-line interface:

```bash
peslite --help
```

PESLite requires Python 3.10 or newer.

## Run the first case

### Bundled example

```bash
peslite gfl-example
```

`gfl-example` is included with PESLite, so it is available without cloning the repository.

### Your own case

If `case.pes` is in the current directory, run:

```bash
peslite case.pes
```

To run a file in another directory, provide its path:

```bash
peslite cases/case.pes
```

## Results

Results are written to `output/<case-name>/` by default. For `peslite case.pes`, the directory is:

```text
output/case/
├── states.csv
├── summary.json
└── simulation.pes
```

Signal and energy files appear only when enabled ([Output and Results](Output-and-Results.md#files)).

`simulation.pes` records the complete configuration used for the run after defaults, derived
values, presets and command-line overrides.

## Inspect and override a case

Print the resolved configuration without running:

```bash
peslite gfl-example --resolved
```

Override a value by dotted path and choose an output directory:

```bash
peslite gfl-example \
  --set simulation.t_end=1.0 \
  --set units.vsc.ctrl.references.p_ref_pu=0.4 \
  --out output/short-run
```

`--set` values take priority over presets and file values
([CLI Reference](CLI-Reference.md#override-parameters)).

Use `--progress SECONDS` to print progress and `--watch NAME` to include a state or output value:

```bash
peslite gfl-example --progress 0.1 --watch vsc.vdc_pu
```

## Create a simulation file

The quickest starting point is a resolved bundled example:

```bash
peslite gfl-example --resolved > my-case.pes
```

Edit `my-case.pes`, then run it with `peslite my-case.pes`; its sections are listed in
[Simulation Files](Simulation-Files.md#structure).

A `.pes` file may optionally use a sibling `PESaddons/` directory for project-local extensions.
The simulation file, extension paths and local output layout are described together in
[Project Workspace](Project-Workspace.md).

A compact complete case looks like this:

```yaml
base: {s_base: 2.0e6, v_ll_rms: 690.0, f0: 50.0}

buses:
  pcc: {c_pu: 0.02, r_d_pu: 0.5}

sources:
  grid: {bus: pcc, x_pu: 0.4, r_pu: 0.04}

units:
  vsc:
    bus: pcc
    ac_filter: {l_f_pu: 0.2, r_f_pu: 0.01}
    dclink:
      vdc_ref: 1500.0
      source: {type: voltage}
    pwm: {f_sw: 20000.0, modulation_limit: 0.95}
    ctrl:
      type: gfm
      loops:
        PSC: {k_p_pu: 6.2832, k_vi: 20.0}
        VirtualImpedance: {}
        ActiveDamping: {r_a_pu: 0.2, alpha_d: 40.0}
      references: {p_ref_pu: 0.5}

events:
  connect_vsc: {type: connect, target: vsc, t: 0.2, ramp: 0.2}

simulation:
  t_end: 1.0
  solver: {dt: 25e-6}
  output: {record_every: 10}
```

Names without a suffix are SI quantities and names ending in `_pu` are per-unit quantities; see
[SI and per-unit values](Simulation-Files.md#si-and-per-unit-values).

## Choose a bridge model

The file default is `pwm_averaging`. For a quick comparison, use one of the run presets without
rewriting the case:

```bash
peslite my-case.pes --switching
peslite my-case.pes --pwm-averaging
peslite my-case.pes --averaging
```

The presets also select an appropriate default solver. See
[Bridge and PWM](Bridge-and-PWM.md#model-comparison) before comparing
their results.

## Use the Python API

```python
import peslite

params = peslite.load("my-case.pes")
result = peslite.Simulation(params).run()

print(result.summary)
print(result.final_states())
u_dc = result.states["vsc.dclink.u_C"]
```

See [Output and Results](Output-and-Results.md#python-result-api) for the result object.

## Next steps

- Configure bridge fidelity and periods in [Bridge and PWM](Bridge-and-PWM.md).
- Split fast and slow states with [Multirate Simulation](Multirate-Simulation.md).
- Generate a standalone simulator with [C++ Export](Cpp-Export.md).
- Choose and adapt a bundled case from [Examples](Examples.md).
