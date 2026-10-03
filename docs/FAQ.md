# FAQ

## Must a simulation file use the `.pes` extension?

No. A simulation file is YAML text, and the parser does not require a particular extension.
`.pes` is the project convention and makes the file's purpose clear.

## Which bridge model is used by default?

When neither the file nor the command line selects another model, each unit uses
`pwm_averaging`. This retains sampled control and PWM-register timing while omitting carrier
ripple. See [Converter and Bridge Models](Converter-and-Bridge-Models.md).

## What do the three bridge flags select?

They are complete run presets:

- `--switching` selects switching bridges and fixed-step RK4;
- `--pwm-averaging` selects PWM-period-averaged bridges and fixed-step RK4;
- `--averaging` selects ideal averaged bridges with continuous control and adaptive DP45.

A repeatable `--set` override has higher priority than the preset. The preset has higher priority
than the simulation file, which has higher priority than defaults.

## Can one system mix bridge models?

Yes. Set `units.<name>.bridge.model` independently for each unit. The three command presets change
all units unless an explicit `--set units.<name>.bridge.model=...` override takes priority.

## Does ideal averaging preserve PWM and computation delay?

No. Ideal `averaging` uses a controlled voltage-source bridge and continuous controller states.
ADC windows, controller interrupts, computation completion, carrier comparison and PWM
shadow/active registers belong to the sampled `switching` and `pwm_averaging` models. Sampled-only
settings may remain in a shared configuration and are silently ignored by ideal averaging.

## How should solver accuracy be checked?

For fixed-step runs, reduce `simulation.solver.dt`. For adaptive runs, reduce `rtol` and `atol`,
and constrain `max_step` if the dynamics require it. A result is numerically credible when the
conclusion no longer changes materially under further refinement. The CSV output interval is not
the integration step. See [Solvers](Solvers.md).

## Can state initial values be specified?

Yes. Use `simulation.initial.states` for explicit state values, or continue from a saved
`states.csv` with `--initial`. The initial time is exact and is not rounded onto a controller or
PWM grid. See [Events and Restart](Events-and-Restart.md).

## Can a run continue from an intermediate output row?

Yes. Supply the saved `states.csv` with `--initial` and select the row using `--initial-time`.
PESLite restores the saved physical and discrete continuation state needed by the selected model.
The accompanying `simulation.pes` records the resolved configuration used for the original run.

## Why is a watched name prefixed by the unit name?

The entity prefix distinguishes the same quantity on different devices. For example,
`vsc_1.vdc_pu` and `vsc_2.vdc_pu` are different values. Run `--list-states` to inspect state names,
and use `--progress` together with `--watch` because watched values are printed on progress lines.

## Does output remain in memory until the run ends?

No. Enabled histories are streamed to CSV in batches controlled by
`simulation.solver.write_length`. `simulation.output.record_every` or `period` controls which
snapshots are retained. Access through the Python `Result` object loads a requested output file
when that result is read. See [Output and Results](Output-and-Results.md).

## Why are some output files absent?

`states.csv`, `summary.json` and `simulation.pes` form the basic result. Signal and energy files
appear only when their corresponding output options are enabled. Fields that have no result use
`None` in the Python API or `null` in JSON rather than a fabricated numeric value.

## Are parameter changes validated during a run?

Yes. `set` events validate their dotted paths and values at the event boundary. Derived defaults
are resolved again when affected parameters change. Event targets and component names must be
unambiguous within the assembled system.

## Can custom components and controllers be used from a simulation file?

Yes. Modules in `peslite.addons.control` and `peslite.addons.components` are discovered into the
same registries as built-in types. Types elsewhere may use the public registration functions for
control loops, circuit elements, events and solvers. The YAML entry then uses the registered type
name and goes through normal parameter validation. See [Extending PESLite](Extending-PESLite.md).

## Is the exported C++ simulator a wrapper around Python?

No. It is a standalone C++17 source file with no third-party C++ runtime dependency. Only
parameters selected during export are mutable afterward; the generated executable reports them
with `--list-params`. Unsupported configuration values supplied through `--config` are warned
about and ignored. See [C++ Export](Cpp-Export.md).
