# CLI Reference

PESLite installs two commands:

- `peslite`: run, inspect or export a simulation;
- `peslite-convert`: export a standalone C++ simulator or an experimental Imperix ELF executable.

Use `peslite --help` and `peslite-convert --help` for the interface of the installed version.

## Select a simulation

With no argument, `peslite` runs the first bundled case by filename:

```bash
peslite
```

Pass a bundled case name or a file path:

```bash
peslite gfl-example
peslite case.pes
peslite ./cases/my-converter.pes
```

For a local file, the suffix may be omitted when the path resolves unambiguously:

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

`--set` has priority over a command preset, file value and default.

## Select a bridge/solver preset

```bash
peslite case.pes --switching
peslite case.pes --pwm-averaging
peslite case.pes --averaging
```

The flags select, respectively, switching + fixed RK4, PWM-period averaging + fixed RK4, and ideal
averaging + adaptive DP45 for the run. They change every unit unless a unit path is explicitly
overridden with `--set`.

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
names. It does not enable result signal files and does not change simulation results.

## Optional simulation report

```bash
peslite case.pes --report
peslite case.pes --report --out output/my-run
```

With the small-signal analysis addon available, this runs the simulation once and writes
`case_report.json` in the output directory, with a `small_signal_stability` section. Only JSON
is generated; the report filename uses the input stem even with a bridge preset or custom output
directory. Without the addon, a warning is emitted and the ordinary simulation continues.

The default checks cover the initial state, simulation-file events and the final state, with
only changed items retained. Stability analysis runs only after a check establishes a periodic
operating state; earlier checks report `transient` without running perturbations.
Use the [Python addon API](Small-Signal-Analysis.md) for custom
analysis times, intervals and watched values. `--report` can accompany `--plot`, but not
inspection or export options.

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

`--plot-xlim` remains in the CSV time unit (seconds); only its presentation is scaled. The default
style uses IEEE dimensions, serif/Computer Modern text, inward ticks, restrained colors and vector
PDF output. External LaTeX is used when available, with a bundled Computer Modern-style fallback.

## Continue from saved state

```bash
peslite case.pes --out output/part-a
peslite output/part-a/simulation.pes \
  --initial output/part-a/states.csv \
  --out output/part-b
```

By default, `--initial` selects the last CSV row. Select another saved time with:

```bash
peslite output/part-a/simulation.pes \
  --initial output/part-a/states.csv \
  --initial-time 0.75 \
  --out output/part-b
```

See [Events and Restart](Events-and-Restart.md) for what is restored.

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

It emits `peslite.cpp` in `export/<name>/` or the explicit `--out` directory. Compilation and
backend limitations are documented in [C++ Export](Cpp-Export.md).

Use `--imperix` for [Imperix ELF export (Experimental)](Imperix-Export.md):

```bash
peslite-convert case.pes --imperix
```

This builds an ELF using the configured official SDK. Add `--imperix-source-only` to generate
sources without building the ELF.

## Generated C++ executable

After compilation, inspect and run it with familiar override/output options:

```bash
export/case/peslite --list-params
export/case/peslite --config another.pes \
  --set simulation.t_end=5.0 \
  --out output/cpp-run
```

Only paths shown by `--list-params` are mutable. The generated executable applies compiled defaults,
then `--config`, then repeatable `--set` values.

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
