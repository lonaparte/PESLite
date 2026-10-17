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

Discrete switching, protection and scheduling have native execution adapters. Numerical component
outputs, derivatives and energy equations reuse the assembled Python equations, including those of
R-L branches, capacitor nodes and DC links. PESLite traces each component's fixed `state`, `inp`,
`out`, `set_outputs()` and `rhs()` scalar equations while exporting,
specialises configuration-dependent branches and loops, and emits ordinary static C++. Common
arithmetic, complex operations, state-dependent branches and NumPy scalar ufuncs are lowered. The
generated binary does not contain an expression interpreter or Python runtime, so using the
fallback does not add a generic dispatch layer to each solver stage.

Subclass overrides and explicit component export hooks are respected. Runtime event state is bound
during tracing rather than by maintaining another copy of the physical formula. RK coefficients
come from the shared fixed-step tables or the actual DP45 solver object's coefficients, including
factory-supplied instance and subclass values. Algebraic loops use matching tolerances and
iteration limits, and nonconvergence is an error rather than an accepted approximate result.
Dependencies eliminated by fixed parameter values are reduced to straight-line assignments by a
conservative assignment analysis and standard topological sorting. Unrecognized code keeps the
iterative solver. Genuine feedback retains its convergence checks and the same record fields
between calls as the Python model; it does not keep a second value cache.

Exporting an unrun `Simulation` reconstructs it with its `parts` factories without mutating the
caller. A supplied ready-made `system`, `solver` or other `instances` entry is rejected because it
cannot be safely reconstructed from parameters. Solver replacements must match a supported solver
and its configured numerical settings; they are never silently replaced by a default solver.
Native execution adapters check the execution callbacks they replace and the implementation
methods referenced by those callbacks. Unrelated presentation methods do not prevent export.
Overrides of controller-host, solver or modulator execution need backend support and are rejected
rather than silently executed as the parent implementation. Registered numerical loops inside the
shared controller graph continue to use ordinary tracing. Loop resets use that loop's own reset
method, include declared `carried` values, and evaluate all new values before writing any state.
Sampled and continuous controllers use the same owned-value storage and update rules; continuous
ODE states are read and written directly in the solver vector rather than duplicated in members.

Composite WPT configurations are expanded before export exactly as they are for Python execution.
Their primary and secondary remain ordinary `DCConverter` units with independent clocks, while the
shared compensated link is lowered through its fixed scalar state equations.

Named measurement bindings, independent series/shunt resistors and configured RMS windows use
the same assembly as Python. `set` events may change an element scalar declared by its
`parameter_bindings()` contract; the built-in resistor declares `r`. The generated parameter is
shared by output and energy equations, and a load step does not recreate electrical/controller
state. This event support does not make the field a command-line `--set` variable: the retained
variable list above remains authoritative. RMS protection uses the common trip path, not a WPT
phase override.

## Current limitations

- Supported solvers are fixed `euler`, `heun` and `rk4`, plus adaptive `DP45`.
- Multirate subsystem schedules are not currently lowered to C++.
- Arbitrary Python behavior is not embedded in the standalone executable. File or network I/O,
  reflection, run-time type/layout changes and unsupported library calls in a custom equation are
  rejected during export rather than silently approximated.
- Symbolic branch exploration is bounded; deeply state-dependent iterative control flow needs
  dedicated backend support instead of an exponentially large generated expression.
- Custom controller loops with non-traceable Python behavior, events and modulators still need
  backend support because they alter the discrete schedule rather than only contributing
  continuous scalar equations.
- Traced loop state and `carried` values must remain numeric scalars (including Boolean and
  complex values). `None` and transitions to nonnumeric values are rejected; the exporter does not
  infer an optional-value protocol from a field's initial value. Built-in first-occurrence times
  use numeric sentinels internally and report absent occurrences as JSON `null`.
- Only parameters reported by `--list-params` can change in the compiled simulator.

The exporter supports the built-in switching, PWM-period-averaged and ideal averaged bridge models.
For their semantics, see [Converter and Bridge Models](Converter-and-Bridge-Models.md).
