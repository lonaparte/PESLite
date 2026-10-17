# Output and Results

PESLite streams enabled histories directly to a run directory and returns a lightweight
`SimulationResult`. Python and exported C++ simulators use the same file names and column schema.

## Output directory

The default Python CLI destination is `output/<configuration-name>/`. A bridge preset adds its mode
to the name so comparison runs do not overwrite one another. Set an explicit destination with:

```bash
peslite case.pes --out output/my-run
```

Files are written directly to their final paths. PESLite does not build a complete in-memory
history and then export it afterward.

## Files

| File | When written | Content |
|---|---|---|
| `states.csv` | `simulation.output.states: 1` | Every named state at each output snapshot. |
| `plant.csv` | `simulation.output.signals: 1` | Network and converter plant outputs. |
| `ctrl.<unit>.csv` | `simulation.output.signals: 1` | Controller signals for one unit. |
| `energy.csv` | `simulation.output.energy: 1` | Stored energy and power-balance history. |
| `summary.json` | Always | Stop reason, solver counts, protection, peaks and diagnostics. |
| `simulation.pes` | Always | Complete resolved configuration used for the run. |

`simulation.pes` is not a temporary file. It records defaults and command-line changes and can be
used to repeat or continue the run.

## Output settings

```yaml
simulation:
  output:
    period: 0.5e-3
    record_every: 10
    states: 1
    signals: 0
    energy: 0
  solver:
    write_length: 1000
```

- `period` is the plant/state snapshot interval. It does not set the integration step.
- `record_every` keeps every N-th controller log sample.
- `states`, `signals` and `energy` enable their corresponding histories.
- `write_length` is the number of rows buffered per streamed CSV write batch.

If `states` is disabled, the terminal state is still retained so `final_states()` works. If
`energy` is disabled, summary energy checks still run unless `energy_check` disables them.

## State and signal names

Paths are entity-first:

```text
pcc.u_C
grid.branch.i
vsc.dclink.u_C
vsc.ctrl.SRFPLL.theta
vsc.pwm.shadow.d_a
vsc.meas.x_i_c
solver.window_end
```

Complex states are split into `.re` and `.im` CSV columns. Boolean state is written as `0` or `1`.
Unit prefixes distinguish the same signal on multiple converters, such as `vsc_1.vdc_pu` and
`vsc_2.vdc_pu`.

Aliases such as `<unit>.i_c` and `<unit>.u_dc` may resolve to the underlying physical state. Use
`--list-states` to print the exact state columns for one configuration.

## Summary conventions

`summary.json` uses stable machine-readable values:

- switches are `0` or `1`;
- a time or result that did not occur is `null`;
- alarms, energy problems and port-Hamiltonian defaults are lists;
- solver counts and timing values are numeric;
- per-unit quantities retain `_pu` in their names.

Protection entries are prefixed by unit name in a multi-converter system. Multirate runs add
interface/window diagnostics; adaptive runs include rejected-step counts.

## Python result API

```python
import peslite

result = peslite.Simulation(peslite.load("case.pes")).run()

states = result.states
plant = result.plant
controller = result.ctrl
summary = result.summary
last = result.final_states()
files = result.files
```

CSV-backed properties are loaded lazily from their final files when first requested. They are not
part of the write hot path.

The optional plotting add-on operates only after the CSV streams have closed:

```python
figure = result.plot(
    ["vsc.dclink.u_C", "vsc.ctrl.SRFPLL.theta"],
    labels=[r"$u_{dc}$", r"$\theta$"],
    ylabel=r"$x$ (pu)",
)

# The same function can process Python or standalone C++ output directly.
from peslite.addons import plot_csv

figure = plot_csv(
    "output/run/states.csv",
    "vsc.dclink.u_C",
)
```

It is implemented under `peslite.addons`, imports Matplotlib only when called and is not included in
the generated C++ simulator. Install it with `pip install "peslite[plot]"`.

Controller logs are grouped by unit under `result.ctrl`. The final-state mapping can be passed back
as initial state through a saved CSV or used programmatically.

## Progress and watch

Progress is console output, not a result history:

```bash
peslite case.pes --progress 0.1 \
  --watch vsc.vdc_pu \
  --watch vsc.i_c,vsc.u_dc
```

`--progress` is measured in simulated seconds. `--watch` accepts state columns, public
plant/controller columns and aliases. Naming a complex state without `.re` or `.im` displays its
magnitude. Unknown names are reported with the available names when progress first prints.

Watching a value does not alter solver state or simulation results.

## Restart from output

Any `states.csv` row can initialize another run. Use the matching resolved `simulation.pes` to
preserve every parameter and derived default. See [Events and Restart](Events-and-Restart.md) for the
continuation commands and discrete-state limitations.
