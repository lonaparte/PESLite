# CLI Reference

PESLite installs two commands:

- `peslite`: run, inspect or export a simulation;
- `peslite-convert`: export a standalone C++ simulator or an experimental Imperix ELF executable.

Use `peslite --help` and `peslite-convert --help` for the interface of the installed version.

## Select a simulation

With no argument, `peslite` runs the bundled `gfl-example`:

```bash
peslite
```

Pass a bundled case name or a file path:

```bash
peslite gfl-example
peslite case.pes
peslite ./cases/my-converter.pes
```

The suffix may be omitted. PESLite tries the path, then `.pes`, `.yaml`, `.yml` and `.json`
variants, then bundled examples, and uses the first match:

```bash
peslite case
```

The default result directory is `output/<name>/`.

## Override parameters

`--set PATH=VALUE` is repeatable and accepts the same dotted paths as `Params.replace()`:

```bash
peslite case.pes \
  --set simulation.t_end=1.0 \
  --set units.vsc.ctrl.computation=2e-6
```

Select another solver explicitly:

```bash
peslite case.pes \
  --set simulation.solver.type=adaptive \
  --set simulation.solver.method=DP45 \
  --set simulation.solver.rtol=1e-3
```

The priority, from highest, is: `--set`, a command preset, the simulation file, package defaults.

## Select a bridge/solver preset

```bash
peslite case.pes --switching
peslite case.pes --pwm-averaging
peslite case.pes --averaging
```

What each preset selects is listed in
[Command presets](Converter-and-Bridge-Models.md#command-presets).

## Choose the output directory

```bash
peslite case.pes --out output/my-run
```

Without `--out`, a bridge preset adds its mode name to the default directory. Files and output
settings are documented in [Output and Results](Output-and-Results.md).

## Progress and watched values

```bash
peslite case.pes --progress 0.1 --watch vsc.vdc_pu
peslite case.pes --progress 0.1 --watch vsc.i_c,vsc.u_dc
```

`--progress SECONDS` uses simulated time. `--watch` may be repeated or contain comma-separated
names; see [Progress and watch](Output-and-Results.md#progress-and-watch).

## Optional simulation report

```bash
peslite case.pes --report
```

With the small-signal analysis addon available, this also writes `case_report.json`; see
[Small-Signal Stability Analysis](Small-Signal-Analysis.md#time-based-simulation-report) for its
contents and compatible options.

## Plot recorded waveforms

Install the optional plotting dependency, then name the recorded columns after `--plot`:

```bash
pip install "peslite[plot]"
peslite case.pes --plot vsc.dclink.u_C vsc.ctrl.SRFPLL.theta \
  --plot-ylabel '$x$ (pu)'
```

`pip install "peslite[all]"` installs every optional runtime add-on in one step.

PESLite always runs the simulation first. It finds the actual generated CSV that contains all the
requested columns and writes `fig_<table>.pdf` beside it. The columns must belong to one CSV; no
implicit resampling is performed between output grids. Plant and controller waveforms require
`simulation.output.signals: 1`, as they do for CSV output generally.

The horizontal column defaults to `t`. Its labels automatically use seconds, milliseconds or
microseconds instead of a scientific multiplier. Use `--plot-time-unit s|ms|us` to override that
choice. Common presentation options include:

```bash
peslite case.pes --plot vsc.dclink.u_C \
  --plot-title 'DC-link voltage' \
  --plot-label '$u_{dc}$' \
  --plot-ylabel '$u_{dc}$ (V)' \
  --plot-xlim 0.1 0.2 \
  --plot-width double \
  --plot-output dc-link.pdf
```

Further options are `--plot-x` (another horizontal column) and `--plot-xlabel`, per-waveform
`--plot-color` and `--plot-style` in `--plot` order, `--plot-height` (inches), `--plot-linewidth`
(points), `--plot-ylim`, `--plot-no-grid` and `--plot-no-legend`.

`--plot-xlim` remains in the CSV time unit (seconds); only its presentation is scaled. The default
style uses IEEE dimensions, serif/Computer Modern text, inward ticks, restrained colors and vector
PDF output. External LaTeX is used when available, with a bundled Computer Modern-style fallback.

## Continue from saved state

`--initial FILE` starts from the last row of a saved `states.csv`; `--initial-time T` selects the
row saved at another time. See [Events and Restart](Events-and-Restart.md#continue-from-a-saved-csv)
for the commands and what is restored.

## Inspect without running

```bash
peslite case.pes --resolved
peslite case.pes --list-states
peslite case.pes --ph-report
```

- `--resolved` prints the complete configuration after defaults, presets and overrides;
- `--list-states` prints the generated `states.csv` state columns;
- `--ph-report` prints the assembled port-Hamiltonian structure and coverage.

The inspection command respects `--set` and the bridge/solver presets supplied with it.

## Export through `peslite`

The general command can invoke a registered exporter:

```bash
peslite case.pes --export cpp
peslite case.pes --export cpp simulation.t_end units.vsc.ctrl.references.p_ref_pu
peslite case.pes --export cpp all --out export/my-simulator
peslite case.pes --export imperix --imperix-source-only
```

The first item after `--export` is the format; later items are variable parameter paths for that
backend.

## Export through `peslite-convert`

`peslite-convert` defaults to standalone C++ simulator export:

```bash
peslite-convert case.pes
peslite-convert case.pes simulation.t_end units.vsc.ctrl.references.p_ref_pu
peslite-convert case.pes all --out export/my-simulator
```

It emits `peslite.cpp` in `export/<name>/` or the explicit `--out` directory. Compilation, runtime
options and backend limits are documented in [C++ Export](Cpp-Export.md).

Use `--imperix` for [Imperix ELF export (Experimental)](Imperix-Export.md):

```bash
peslite-convert case.pes --imperix
```

This builds an ELF using the configured official SDK. Add `--imperix-source-only` to generate
sources without building the ELF.

## Common command patterns

Print a short run with live values:

```bash
peslite case.pes --set simulation.t_end=0.5 \
  --progress 0.05 --watch vsc.vdc_pu,vsc.i_c
```

Compare bridge presets without editing the file:

```bash
peslite case.pes --switching
peslite case.pes --pwm-averaging
peslite case.pes --averaging
```

Create an editable, fully resolved starting file:

```bash
peslite gfl-example --resolved > case.pes
```
