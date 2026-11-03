# CLI Reference

PESLite installs one `peslite` command with five operations:

```bash
peslite create case.pes
peslite run case.pes
peslite validate case.pes
peslite inspect case.pes
peslite export case.pes
```

`peslite case.pes` is the short form of `peslite run case.pes`. Use `peslite --help` or
`peslite <operation> --help` for the installed interface.

## Create a file

Specify the new file name:

```bash
peslite create case.pes
peslite create component.py
peslite create controller.py
peslite create my_control.py
```

Simulation files are created in the current directory. `component.py` is created in
`PESaddons/components/`; other `.py` files use the controller template in `PESaddons/control/`.
Python class and registration names follow the supplied file name. Templates are bundled under
`examples/templates/`, and existing files are never overwritten.

Creation does not open the file unless `--open` is supplied:

```bash
peslite create case.pes --open
```

On Linux and macOS, the option uses `VISUAL`, then `EDITOR`, before falling back to the operating
system's default application for the file type.

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

The suffix may be omitted. PESLite checks the supplied path and its `.pes`, `.yaml`, `.yml` and
`.json` variants before checking bundled examples. The default result directory is
`output/<name>/`.

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

The priority, from highest, is `--set`, a command preset, the simulation file and package defaults.

## Select a bridge and solver preset

```bash
peslite case.pes --switching
peslite case.pes --pwm-averaging
peslite case.pes --averaging
```

See [Command presets](Bridge-and-PWM.md#command-presets) for the model and solver selected by each
preset.

## Choose the output directory

```bash
peslite case.pes --out output/my-run
```

Without `--out`, a bridge preset adds its mode name to the default directory. See
[Output and Results](Output-and-Results.md) for the generated files.

## Progress and watched values

Interactive simulations show one updating progress bar by default:

```bash
peslite case.pes --progress --watch vsc.vdc_pu
peslite case.pes --progress --watch vsc.i_c,vsc.u_dc
peslite case.pes --no-progress
```

`--watch` may be repeated or contain comma-separated names. Redirected runs remain quiet unless
progress is explicitly enabled; see
[Progress and watch](Output-and-Results.md#progress-and-watch).

## Optional simulation report

```bash
peslite case.pes --report
```

With the online stability analysis addon available, this also writes `case_report.json`; see
[Online Stability Analysis](Online-Stability-Analysis.md#online-simulation-report).

## Plot recorded waveforms

`--plot NAME [NAME ...]` invokes the optional plotting addon after the simulation. See
[Plotting](Plotting.md) for installation, output requirements and presentation options.

## Continue from saved state

`--initial FILE` starts from the last row of a saved `states.csv`; `--initial-time T` selects
another row. See [Events Timeline](Events-Timeline.md#continue-from-a-saved-csv).

## Validate without running

```bash
peslite validate case.pes
```

`validate` loads and assembles the configuration without running or creating output files.

## Inspect without running

```bash
peslite inspect case.pes
```

With no view option, `inspect` prints the resolved configuration, generated state names, electrical
groups and port-Hamiltonian structure report. Select any subset by combining these options:

```bash
peslite inspect case.pes --resolved
peslite inspect case.pes --states
peslite inspect case.pes --ph-report
peslite inspect case.pes --groups
peslite inspect case.pes --resolved --states
```

Inspection accepts the same configuration presets and `--set` overrides as a run.

## Machine-readable results

All operations accept `--json`:

```bash
peslite create case.pes --json
peslite validate case.pes --json
peslite inspect case.pes --json
peslite run case.pes --out output/case --json
peslite export case.pes common --json
```

Standard output then contains one strict JSON object with schema `peslite.cli/v1`. Its envelope
includes `command`, `status`, `exit_code`, absolute configuration and result paths, `diagnostics`
and `error`. A run also reports simulated start, requested end, actual stop time, wall time, RHS
evaluations and trip status. Terminal progress options cannot be combined with `--json`.

Execution success does not assert that the configured physical system is correct; model warnings,
protection results and numerical diagnostics remain separate fields.

Exit codes are `0` for a completed command, `2` for command or configuration errors, `3` for
assembly errors, `4` for simulation failures, `5` for a run limit and `70` for an internal error.

## Run limits

```bash
peslite run case.pes --timeout 60
peslite run case.pes --max-rhs 1000000
```

Limits are checked at existing event boundaries and therefore do not add simulation events.
PESLite closes the histories and writes the partial `summary.json` before returning exit code `5`.

## Export

Standalone C++ is the default export target:

```bash
peslite export case.pes
peslite export case.pes simulation.t_end units.vsc.ctrl.references.p_ref_pu
peslite export case.pes common --out export/my-simulator
peslite export case.pes all
```

Items after the simulation file are variable parameter paths, `common` or `all`. See
[C++ Export](Cpp-Export.md) for compilation, runtime options and backend limits.

Use `--imperix` for [Imperix ELF export (Experimental)](Imperix-Export.md):

```bash
peslite export case.pes all --imperix
peslite export case.pes all --imperix --imperix-source-only
```

## Common command patterns

Print a short run with live values:

```bash
peslite case.pes --set simulation.t_end=0.5 \
  --progress --watch vsc.vdc_pu,vsc.i_c
```

Compare bridge presets without editing the file:

```bash
peslite case.pes --switching
peslite case.pes --pwm-averaging
peslite case.pes --averaging
```

Create an editable, fully resolved starting file:

```bash
peslite inspect gfl-example --resolved > case.pes
```
