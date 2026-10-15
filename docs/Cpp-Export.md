# C++ Export

PESLite can turn a resolved simulation into a standalone, parameter-specialized C++17 program. The
generated simulator performs the numerical run itself; it is not a recording or wrapper around
Python.

## Export

Use `peslite-convert` with a simulation file or bundled example:

```bash
peslite-convert case.pes simulation.t_end units.vsc.ctrl.references.p_ref_pu
```

The paths after the file are the parameters that remain variable in the compiled program. Use
`all` to retain every variable parameter supported for that configuration:

```bash
peslite-convert case.pes all
```

By default, the command writes one file:

```text
export/case/peslite.cpp
```

No generated `README.md`, CMake project or temporary configuration is required. Model structure,
fixed parameters, schedules, initial state and the resolved configuration template are embedded in
the translation unit.

An explicit destination uses the familiar `--out` option:

```bash
peslite-convert case.pes all --out export/my-simulator
```

The Python API provides the same operation:

```python
import peslite

params = peslite.load("case.pes")
project = peslite.export(
    params,
    "cpp",
    variables=["simulation.t_end"],
)
```

## Compile

Any conforming C++17 compiler is sufficient. No third-party C++ library is used:

```bash
c++ -O3 -DNDEBUG -std=c++17 export/case/peslite.cpp -o export/case/peslite
```

The compiler may create its normal build cache or object files when invoked through a build system,
but `peslite-convert` itself emits only `peslite.cpp`.

## Run

The generated executable follows the PESLite conventions for parameter overrides and output:

```bash
export/case/peslite --list-params
export/case/peslite \
  --config another.pes \
  --set simulation.t_end=5.0 \
  --out output/cpp-run
```

`--list-params` shows the paths compiled as variables and their current defaults. Only these paths
can be changed. Setting any other path fails with:

```text
parameter 'path' is not variable
```

The precedence is:

1. `--set PATH=VALUE`;
2. values loaded with `--config PESFILE`;
3. defaults embedded during export.

`--config` reads supported scalar values from the simulation file. Other leaf values are ignored
with a warning because changing model structure after compilation would invalidate the specialized
program. This makes it possible to reuse a normal resolved `simulation.pes` without maintaining a
separate C++-only configuration.

## Exported variables

The exact list depends on the selected solver and the entities in the case. The current backend can
retain:

- `simulation.t_end` and `simulation.output.period`;
- `simulation.solver.dt` for a fixed solver;
- `simulation.solver.rtol`, `atol` and `max_step` for adaptive DP45;
- source voltage, frequency and angle;
- controller reference fields for each unit.

`all` means all parameters supported by this backend, not every field in the original file. Use the
generated program's `--list-params` as the authoritative list.

Runtime values are stored as direct typed fields. Configuration parsing and path lookup happen once
before simulation and do not introduce dictionary lookup in the integration hot path. Exporting
`all` mainly increases source and executable size; it does not change the simulated state layout.

## Results

The compiled simulator uses the same result structure as the Python runner:

```text
output/cpp-run/
├── states.csv
├── summary.json
└── simulation.pes
```

Optional plant, controller and energy files follow the same configuration. The output
`simulation.pes` records the actual values after `--config` and `--set`, so the C++ run remains
auditable.

## What is specialized

These choices are fixed when `peslite.cpp` is generated:

- network topology and component types;
- bridge model and controller graph;
- events and state layout;
- output schema;
- every parameter not explicitly retained as a variable.

Specialization lets the compiler inline the configured RHS and remove generic dispatch. To change a
fixed choice, update the source `.pes` file and run `peslite-convert` again.

Built-in plant and converter modules have dedicated lowering for their complete switching,
protection and scheduling semantics. A custom continuous subsystem also has a fallback: PESLite
traces its fixed `state`, `inp`, `out`, `set_outputs()` and `rhs()` scalar equations while exporting,
specialises configuration-dependent branches and loops, and emits ordinary static C++. Common
arithmetic, complex operations, state-dependent branches and NumPy scalar ufuncs are lowered. The
generated binary does not contain an expression interpreter or Python runtime, so using the
fallback does not add a generic dispatch layer to each solver stage.

The dedicated modules remain preferable when they fit: their event and timing semantics are fully
covered, their generated expressions are deliberately tuned, and unsupported behavior is found
before a long build. The fallback is intended for pure numerical component equations with a fixed
state and port layout.

## Current limitations

- Supported solvers are fixed `euler`, `heun` and `rk4`, plus adaptive `DP45`.
- Multirate subsystem schedules are not currently lowered to C++.
- Composite WPT units and their resonant-link schedule are not currently lowered to C++; the
  exporter rejects them before writing backend files.
- Arbitrary Python behavior is not embedded in the standalone executable. File or network I/O,
  reflection, run-time type/layout changes and unsupported library calls in a custom equation are
  rejected during export rather than silently approximated.
- Symbolic branch exploration is bounded; deeply state-dependent iterative control flow needs
  dedicated backend support instead of an exponentially large generated expression.
- Custom controller loops, events and modulators still need backend support because they alter the
  discrete schedule rather than only contributing continuous scalar equations.
- Only parameters reported by `--list-params` can change in the compiled simulator.

The exporter supports the built-in switching, PWM-period-averaged and ideal averaged bridge models.
For their semantics, see [Converter and Bridge Models](Converter-and-Bridge-Models.md).
