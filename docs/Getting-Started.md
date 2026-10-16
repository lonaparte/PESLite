# Getting Started

This guide installs PESLite, runs a bundled case, explains the output directory and turns a
resolved example into a new simulation file.

## Install

```bash
python -m pip install peslite
```

To include every optional runtime add-on, including PDF plotting:

```bash
python -m pip install "peslite[all]"
```

Check the command-line interface:

```bash
peslite --help
```

PESLite requires Python 3.10 or newer. Simulation files are YAML text; `.pes` is the conventional
extension, but the parser does not require it.

## Run the first case

```bash
peslite gfl-example
```

`gfl-example` is bundled in the wheel, so it is available without cloning the repository. The
default result directory is:

```text
output/gfl-example/
├── states.csv
├── summary.json
└── simulation.pes
```

Additional signal or energy CSV files appear only when enabled by the configuration. PESLite
streams enabled histories directly to these files instead of retaining the complete run in RAM.

`simulation.pes` is the complete configuration actually used after defaults, derived values,
presets and command-line overrides. It can be run again directly:

```bash
peslite output/gfl-example/simulation.pes
```

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

The priority is:

1. explicit `--set` values;
2. a command preset such as `--averaging`;
3. values in the simulation file;
4. PESLite defaults.

Use `--progress SECONDS` to print progress and `--watch NAME` to include a state or output value:

```bash
peslite gfl-example --progress 0.1 --watch vsc.vdc_pu
```

## Create a simulation file

The quickest starting point is a resolved bundled example:

```bash
peslite gfl-example --resolved > my-case.pes
```

Edit `my-case.pes`, then run it with `peslite my-case.pes`. A file is organized into these
top-level sections:

| Section | Purpose |
|---|---|
| `base` | System power, voltage and frequency base. |
| `buses`, `branches`, `sources` | Network and grid source. |
| `units` | Converter hardware, controller and protection. |
| `elements` | Optional registered loads or custom circuit elements. |
| `events` | Timed connect, disconnect and parameter-change operations. |
| `simulation` | End time, solver, initial values, output and progress. |
| `meta` | Optional title and description; does not affect the run. |

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

Names without a suffix are SI quantities. Names ending in `_pu` are per-unit quantities, times are
seconds, angles are radians and switches are written as `0` or `1`.

## Choose a bridge model

The file default is `pwm_averaging`. For a quick comparison, use one of the run presets without
rewriting the case:

```bash
peslite my-case.pes --switching
peslite my-case.pes --pwm-averaging
peslite my-case.pes --averaging
```

The presets also select an appropriate default solver. See
[Converter and Bridge Models](Converter-and-Bridge-Models.md) before comparing their results.

## Use the Python API

```python
import peslite

params = peslite.load("my-case.pes")
result = peslite.Simulation(params).run()

print(result.summary)
print(result.final_states())
u_dc = result.states["vsc.dclink.u_C"]
```

`Simulation.run()` uses `output/run` by default. `result.states`, `result.plant` and `result.ctrl`
load their final CSV data when accessed.

## Next steps

- Compare model fidelity and timing in [Converter and Bridge Models](Converter-and-Bridge-Models.md).
- Split fast and slow states with [Multirate Simulation](Multirate-Simulation.md).
- Generate a standalone simulator with [C++ Export](Cpp-Export.md).
- Choose and adapt a bundled case from [Examples](Examples.md).
