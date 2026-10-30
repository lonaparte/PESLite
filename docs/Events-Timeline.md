# Events Timeline

Events describe model changes at absolute simulation times. Restart loads named state from a saved
row and reconstructs the discrete timing needed to continue from that exact time.

## Built-in events

All events live in one named top-level mapping:

```yaml
events:
  connect_vsc: {type: connect, target: vsc, t: 0.2, ramp: 0.5}
  p_step: {type: set, t: 1.0, set: {units.vsc.ctrl.references.p_ref_pu: 0.8}}
  line_trip: {type: disconnect, target: line, t: 2.0}
```

Event names are user-defined and unique. `target` refers to a named unit, source, branch or element
for connect/disconnect operations.

Every event is bound to explicit writable variable paths during assembly. The scheduler writes
those variables at the event time; it does not execute an arbitrary event callback. Parameter
paths retain their ordinary SI/pu conversion, validation and retuning behavior.

Connection commands use `<section>.<object>.connected`, with boolean values or the usual `1`/`0`
switch syntax. `connect` and
`disconnect` are shorthand for writing that same command variable. For example:

```yaml
events:
  start: {type: set, t: 0.2, ramp: 0.5, set: {units.vsc.connected: true}}
  stop: {type: set, t: 2.0, set: {units.vsc.connected: false}}
```

These are commands, not writable protection or breaker-status flags: the normal unit connection,
startup and trip-latch logic still determines the physical state. Composite-unit commands bind
to their existing leaf-unit paths once during assembly. Fully scoped paths distinguish objects
with the same name in different sections. Multiple events may write the same variable at different
times; no per-event trigger variable is created. Conflicting writes at one time are rejected.

An individual converter's optional `connected` field supplies its initial command before event
writes. If unspecified, the existing rule applies: the unit starts in the opposite state to its
first connection event, or connected when it has no connection events. Events at or before the
simulation start take precedence; later events continue to write the same command normally.

Its C++ export selection is described in [C++ Export](Cpp-Export.md#exported-variables).
The command does not reset a protection latch or force a PWM register load. In sampled models,
startup, control computation and the existing PWM load timing determine when the gates activate.

### `connect`

Connects the target at `t`. A converter connect also issues its controller startup command and may
ramp its reference over `ramp` seconds. A unit with a latched protection trip remains disconnected.

### `disconnect`

Disconnects the target at `t`. For a converter, the common unit path stops the controller command,
opens its series branch when present, and blocks an unfiltered bridge on its named power bus.

### `set`

Changes runtime-safe parameter paths and validates the complete parameter tree before use. Supported
families include source values, line/bus electrical parameters, controller references and loop
gains, protection settings, DC-source values and retunable element parameters. Structural fields
such as component type, bus assignment, loop type and loop period cannot change at run time, nor can
loop switches that add or remove states, such as `SRFPLL.normalisation` or a filter `enable`.

Changed control loops retain their named state. Source frequency/angle changes preserve phase
continuity, and solver caches are notified after the parameter update.

## Exact event semantics

Every event time is an exact integration boundary:

1. integrate to the event with the old model;
2. settle deferred solver work at that time;
3. apply all actions scheduled there in file order;
4. refresh the assembled outputs and affected parameters;
5. integrate the next interval with the new model.

This rule applies to fixed, adaptive, multirate and user solvers. If an event shortens a multirate
window, the shortened work uses its actual duration and later windows keep their original grid.

## Initial state

`simulation.initial` sets an exact start time and optional named state overrides:

```yaml
simulation:
  initial:
    t: 0.4
    states:
      pcc.u_C: source
      vsc.dclink.u_C: rated
      vsc.ctrl.PSC.theta: 0.1
```

The state names are the `states.csv` columns. Complex state can use a two-element value or `.re`
and `.im` paths. The keywords `source` and `rated` initialize supported electrical states from the
connected source or the converter rating. The start time is not aligned or rounded to a PWM or
solver grid.

Parameter events at or before the start time, including `t: 0`, take effect before controller
startup. Explicit controller state overrides are loaded afterward and retain precedence. Python
simulation and standalone C++ export use the same initialization order.

## Continue from a saved CSV

Run the first segment:

```bash
peslite case.pes --out output/part-a
```

Continue from its final saved row with the resolved configuration from that run:

```bash
peslite output/part-a/simulation.pes \
  --initial output/part-a/states.csv \
  --out output/part-b
```

If the configured `simulation.t_end` is not after the restart time, the continuation runs for that
duration beyond it. `--initial` also accepts a YAML or JSON file holding a `t`/`states` mapping or
a simulation file's `simulation.initial` block; `--initial-time` then replaces its `t`. In Python,
`peslite.load(path, initial=..., initial_time=...)` does the same.

Select another saved output time with `--initial-time T`:

```bash
peslite output/part-a/simulation.pes \
  --initial output/part-a/states.csv \
  --initial-time 0.75 \
  --out output/restarted-at-075
```

The requested time must identify a saved row. Reduce `simulation.output.period` in the original
run if a denser set of restart points is required.

## What is restored

Depending on the selected model, named state includes:

- physical network, filter and DC-link state;
- controller integrators, filters, startup progress and protection timers;
- PWM active and shadow registers and enable bits;
- ADC averaging-window accumulators;
- multirate window anchors, slopes and input integrals, so a row inside a window also continues
  exactly;
- unit trip/fault latches.

Timer indices and the next ADC/PWM instants are reconstructed from the saved time; a continuous
unit's clock resumes at its first tick after that time. Intermediate ADC oversamples and a shadow
word replaced during a computation are not states, so a continuation from saved unit state is
rejected:

- between two control interrupts of a unit whose ADC oversamples (`meas.period` shorter than
  `ctrl.period`);
- inside a controller computation window when the next PWM register load would take a word that
  no load has applied yet (see
  [Computation time](Bridge-and-PWM.md#computation-time)).

Ideal averaging has no PWM register state or ADC samples; its continuous controller state, and the
history of any `meas.rms` window, are restored with the plant state instead.

## Restart consistency

For a reproducible split run:

- use the saved `simulation.pes` rather than reconstructing defaults by hand;
- restart from a row written by `states.csv`;
- keep future event definitions in the configuration;
- do not overwrite restored state through conflicting `simulation.initial.states` entries.

The continuation output starts a new result directory; PESLite does not append to the previous CSV.

## Custom events

A registered event type declares the variables it writes and follows the built-in event path; see
[Custom events](Extending-PESLite.md#custom-events).
