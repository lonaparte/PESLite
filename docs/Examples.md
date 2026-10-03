# Examples

PESLite ships eight simulation files with the Python wheel. They can be run by name from any
directory, so cloning the repository is not required.

```bash
peslite gfl-example
```

Each example is an ordinary YAML simulation file. Use it directly, print its fully resolved
configuration, or copy that resolved form as the starting point for another case.

```bash
peslite gfl-example --resolved > case.pes
peslite case.pes
```

## Bundled cases

| Name | System and control purpose |
|---|---|
| `cable-gfl-example` | A grid-following converter connected to the grid through the distributed R-L/G-C model registered under `addons/components`. |
| `custom-pll-example` | Grid-following control assembled from built-in current/DC-voltage loops and an automatically discovered voltage-adaptive PLL under `addons/control`. |
| `gfl-example` | One grid-following converter with PLL, dc-voltage control, current control and unit protection. This is the most extensively annotated configuration. |
| `gfm-psc-example` | Grid-forming power-synchronization control with virtual impedance and active damping. |
| `gfm-vsg-example` | Grid-forming virtual synchronous generator control. |
| `gfm-dvoc-example` | Grid-forming dispatchable virtual oscillator control. |
| `gfm-matching-example` | Matching control coupled to a dynamic dc link and current source. |
| `two-converters-example` | A GFM and a GFL unit on separate buses, with different ratings and switching frequencies. |

The files are also available in the repository's
[`examples/`](https://github.com/lonaparte/PESLite/tree/main/examples) directory. The root
directory contains only simulation files, not Python example programs.

## Compare bridge models

The same case can be run under each bridge/solver preset without editing its configuration:

```bash
peslite gfl-example --switching
peslite gfl-example --pwm-averaging
peslite gfl-example --averaging
```

The default output directories receive a mode suffix, so these runs do not overwrite one another.
They represent different model fidelity and timing semantics; compare quantities in their common
bandwidth rather than expecting identical switching-scale waveforms. See
[Converter and Bridge Models](Converter-and-Bridge-Models.md).

## Short exploratory runs

Use `--set` to shorten a bundled case or change a reference without maintaining another file:

```bash
peslite gfm-psc-example \
  --set simulation.t_end=1.0 \
  --set units.vsc.ctrl.references.p_ref_pu=0.3 \
  --out output/psc-short
```

For progress lines with selected live values:

```bash
peslite two-converters-example \
  --progress 0.05 \
  --watch vsc_1.vdc_pu,vsc_2.vdc_pu
```

Device names keep otherwise similar values unambiguous. `vsc_1.vdc_pu` and `vsc_2.vdc_pu`
therefore refer to different converters.

## Inspect before running

```bash
peslite gfl-example --list-states
peslite gfl-example --ph-report
peslite gfl-example --resolved
```

These commands use the same configuration resolution and accept the same bridge presets and
`--set` overrides as a simulation run.

## Export an example to C++

```bash
peslite-convert gfl-example all
c++ -O3 -DNDEBUG -std=c++17 export/gfl-example/peslite.cpp \
  -o export/gfl-example/peslite
export/gfl-example/peslite --list-params
```

See [C++ Export](Cpp-Export.md) for variable selection, runtime configuration and backend limits.

## Build a new case

Start with `gfl-example` when learning the full configuration surface; its comments describe the
available network, converter, timing, protection, event, solver and output options. Start with the
closest compact GFM case when changing only the synchronization law. For a network with more than
one independently named unit, use `two-converters-example`.

Continue with [Getting Started](Getting-Started.md) and
[Simulation Files](Simulation-Files.md) when turning an example into a maintained case.
