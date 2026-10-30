# FAQ

## Must a simulation file use the `.pes` extension?

No. `.pes` is the project convention; see [Simulation Files](Simulation-Files.md).

## Which bridge model is used by default?

When neither the file nor the command line selects another model, each unit uses
`pwm_averaging`. This retains sampled control periods and PWM-register updates while omitting carrier
ripple. See [Bridge and PWM](Bridge-and-PWM.md#model-comparison).

## What do the three bridge flags select?

They are complete run presets that select the bridge model of every unit and the solver; see
[Command presets](Bridge-and-PWM.md#command-presets).

## Can one system mix bridge models?

Yes. Set `units.<name>.bridge.model` independently for each unit and run without a preset, or
override a unit with `--set` ([CLI Reference](CLI-Reference.md#override-parameters)).

## Does ideal averaging preserve PWM and computation delay?

No. Ideal `averaging` uses a controlled voltage-source bridge and continuous controller states.
ADC windows, controller interrupts, computation completion, carrier comparison and PWM
shadow/active registers belong to the sampled `switching` and `pwm_averaging` models. See
[Ideal averaging](Bridge-and-PWM.md#ideal-averaging) for which sampled settings it ignores.

## How should solver accuracy be checked?

For fixed-step runs, reduce `simulation.solver.dt`. For adaptive runs, reduce `rtol` and `atol`,
and constrain `max_step` if the dynamics require it. A result is numerically credible when the
conclusion no longer changes materially under further refinement. See [Solvers](Solvers.md).

## Can state initial values be specified?

Yes. Use `simulation.initial.states` for explicit state values, or continue from a saved
`states.csv` with `--initial`. See
[Events Timeline](Events-Timeline.md#initial-state).

## Can a run continue from an intermediate output row?

Usually. Supply the saved `states.csv` with `--initial` and select the row using `--initial-time`.
[Events Timeline](Events-Timeline.md#what-is-restored)
lists what is restored and the rows
that are rejected.

## Why is a watched name prefixed by the unit name?

The entity prefix distinguishes the same quantity on different devices. For example,
`vsc_1.vdc_pu` and `vsc_2.vdc_pu` are different values. Run `--list-states` to inspect state names,
and use `--progress` together with `--watch` because watched values are printed on progress lines.

## Does output remain in memory until the run ends?

No. Enabled histories are streamed to CSV in batches controlled by
`simulation.solver.write_length`, and a `SimulationResult` loads a requested output file when it
is read. See [Output and Results](Output-and-Results.md).

## Why are some output files absent?

Signal and energy files are written only when enabled; see
[Files](Output-and-Results.md#files).

## Are parameter changes validated during a run?

Yes. `set` event paths and values are validated when loading the configuration. Derived defaults
are resolved again when affected parameters change. Event targets and component names must be
unambiguous within the assembled system.

## Can custom components and controllers be used from a simulation file?

Yes. Modules in `peslite.addons.control` and `peslite.addons.components` are discovered into the
same registries as built-in types. Types elsewhere may use the public registration functions for
control loops, circuit elements and events. A control-loop entry uses the registered class
name; the other extension entries use their registered type names. All entries go through normal
parameter validation. See [Extending PESLite](Extending-PESLite.md).

## Is the exported C++ simulator a wrapper around Python?

No. It is standalone C++17 with no third-party C++ runtime dependency; see
[C++ Export](Cpp-Export.md).
