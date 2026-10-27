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

Numerical inputs are discovered from the ordinary parameter schemas, not from a separate C++
device whitelist. This includes network R/L/C, converter filters and DC supplies, WPT link
inductances/capacitances/mutual inductance, controller gains and limits, references, protection
thresholds and numerical timing/solver settings. The same paths apply to AC, DC and nested Units.
You can select one field, a section, `all`, or `common`:

```bash
peslite-convert buck-example elements.load units.converter.filter
peslite-convert wpt-example all
peslite-convert buck-example common
```

Leaf converters also expose `units.<name>.connected`, the initial connection command. Select it
individually or through a unit section, `all`, or `common`, and set it to `0` or `1` with `--set`.
Scheduled connection events retain their normal precedence and timing. The command uses the same
startup, PWM and protection lifecycle as a local simulation; it is not a writable gate-status flag.

`common` means `all` minus the parameters identified by the existing structural sensitivity
check. It keeps those detected parameters constant at their configured values; every other
available numerical parameter remains adjustable. No separate parameter list is maintained.
For the default Buck model this freezes `buses.output.r_d` and
`units.converter.dclink.source.r`. The export diagnostic lists the actual exclusions, and
`--list-params` lists the parameters still adjustable with `--set`. Use
`variables=["common"]` for the same selection through the Python API.

Selection uses one `all` analysis, then the ordinary constant specialization for the filtered
set. It does not repeatedly search for more exclusions, and does not guarantee constant-build
performance or detect every possible performance-sensitive parameter. If the check finds no
sensitive parameters, `common` is equivalent to `all` and reuses that analysis.

After compilation, use the same simulation-file paths:

```bash
./peslite --set elements.load.r=8 --set units.converter.filter.l_f=0.002
```

`all` retains the existing numerical leaves, including resolved numerical defaults. An omitted
control period still follows the PWM period; omitted loop periods still follow the control period.
Retaining a field does not make its default explicit. An actual `--set` or `--config` value does,
just as an explicit value in a Python simulation file does. Saved configuration includes the
effective values after these dependencies have been recomputed. `--list-params` is the authoritative
list for a generated program. Unset optional fields with no numerical default are not invented.

Types, connections, feature switches, initial-state specifications and event schedules remain structural.
So do represented history dimensions: numerical periods/windows can change together, but a different
ADC oversample count or RMS/ROCOF history size requires a new export. Parameter validation and
checked initialization branches come from the ordinary schemas and constructors. Invalid values
and changes outside the compiled specialization fail before simulation output is created.
Clock changes that would alter an already initialized switching-edge sequence also require a new export.

Runtime values are stored as direct typed fields. Component construction is recorded during export
and replayed with parameter expressions, reusing the normal numerical initialization rather than
another set of physical formulas. Derived coefficients, including PI/filter coefficients and the
two-winding modal transform, are initialized once in the executable. Both equations and energy
declarations use those values. Configuration parsing and path lookup do not enter the integration
hot path. Unretained inputs remain constants and the simulated state layout is unchanged.
Zero-coefficient dependency elimination also applies to retained parameters: a checked numerical
condition selects the existing straight-line plan when feedback vanishes, and the ordinary
algebraic solver remains in use otherwise. Retaining all parameters can still cost more than a
fully constant build; select only the fields you need when maximum specialization matters.

Export also reports retained parameters that control conditional dependencies inside algebraic
loops. The command prints these diagnostics; the Python API exposes the same records through
`project.diagnostics`, including `parameters`, `condition`, `dependencies` and `loop`. For example,
the Buck model reports `buses.output.r_d` and `units.converter.dclink.source.r`, with their public
input-to-output dependency paths. These names are discovered from the traced equations, not a
device or parameter list. They remain adjustable with `--set`.

The check reuses the existing equation trace and topological sort. Only cyclic residual graphs
need SCC analysis. Products with parameter-only coefficients and parameter-controlled scalar
branches identify conditional edges; reads that remain unconditional are not reported. Derived
coefficients resolve through the existing initializer dependencies. No parameter-value sweep,
simulation, Jacobian or runtime instrumentation is performed. Repeated lowering uses the cached
result. An empty diagnostic list does not prove parameter insensitivity: opaque export hooks and
code outside the supported scalar analysis remain unclassified. A diagnostic indicates a potential
structural performance change, not a guaranteed slowdown or a numerical timing prediction. The
`condition` describes suppression of one traced contribution, not necessarily the entire edge or
loop; parallel contributions may need to vanish together. Reachability under all parameter
validation constraints is not proved by this basic check.

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
With retained parameters, numerical customization must also be reproducible from the construction
inputs. A later mutation of a recorded component that cannot be reproduced is rejected rather than
silently overwritten by parameter initialization.
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
state. A retained initial resistance and a scheduled resistance step share that same model
parameter. Starting a new run with different parameters is distinct from changing them during a
run: event changes retain their existing state-continuity restrictions. RMS protection uses the
common trip path, not a WPT phase override.

## Current limitations

- Supported solvers are fixed `euler`, `heun` and `rk4`, plus adaptive `DP45`.
- Multirate subsystem schedules are not currently lowered to C++.
- An additional ADC integration grid is currently Python-only; generated backends require
  `simulation.solver.measurement_interval` to be omitted.
- `ModeSequence` is currently Python-only. Open-loop duty control is supported by standalone C++.
- Imperix hardware adapters do not implement plant commutations. Standalone C++ uses the
  component commutation interfaces described below.
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

The exporter supports the built-in switching, PWM-period-averaged and ideal averaged bridge models,
including square-wave phase-shift modulation in the switching model.
For their semantics, see [Converter and Bridge Models](Converter-and-Bridge-Models.md).

Controllerless Units retain their electrical equations and independently clocked ADCs. Ideal
diode full bridges and switch-controlled capacitors participate through the ordinary
`commutation_margin()`, `commutate(t)` and optional `next_commutation_time()` component methods.
The native dispatcher shares scheduled boundaries, margin location, simultaneous transitions,
state resets and the no-progress guard. Adaptive commutations require a finite `max_step`.
Component-owned state projections, discharge-energy columns and diagnostics are exported too.

Custom components declare mutable scalar equation memory with `equation_state_bindings()`, a
mapping from local names to `(owner, attribute)` storage references. These bindings describe the
existing memory; they do not create another state store or restart schema. `get_state()` and
`set_state()` retain public restart names. Annotated scalar and void helper methods are traced
at their existing boundaries to avoid multiplying unrelated branch paths. Component equations and
state-changing callbacks are traced from their Python implementation; no converter-specific
commutation equations are maintained in the backend.
