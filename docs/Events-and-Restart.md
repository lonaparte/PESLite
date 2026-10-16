# Events and Restart

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

### `connect`

Connects the target at `t`. A converter connect also issues its controller startup command and may
ramp its reference over `ramp` seconds. A unit with a latched protection trip remains disconnected.

### `disconnect`

Disconnects the target at `t`. For a converter, the normal hardware path blocks PWM, opens the AC
terminal and stops the controller command.

### `set`

Changes runtime-safe parameter paths and validates the complete parameter tree before use. Supported
families include source values, line/bus electrical parameters, controller references and loop
gains, protection settings, DC-source values and retunable element parameters. Structural fields
such as component type, bus assignment, loop type and loop period cannot change at run time.

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
and `.im` paths. The start time is not aligned or rounded to a PWM or solver grid.

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

Select another saved output time with `--initial-time T`:

```bash
peslite output/part-a/simulation.pes \
  --initial output/part-a/states.csv \
  --initial-time 0.75 \
  --out output/restarted-at-075
```

The requested time must identify a saved row. Increase output frequency before the original run if
a denser set of restart points is required.

## What is restored

Depending on the selected model, named state includes:

- physical network, filter and DC-link state;
- controller integrators, filters, startup progress and protection timers;
- PWM active and shadow registers and enable bits;
- ADC averaging-window accumulators;
- multirate window anchors and slopes;
- unit trip/fault latches.

Timer indices and the next ADC/PWM instants are reconstructed from the saved time. Intermediate ADC
oversamples are deliberately not stored, so a case using oversampling should restart from a control
interrupt for exact continuation.

Ideal averaging has no ADC or PWM register state; its continuous controller state is restored with
the plant state instead.

## Restart consistency

For a reproducible split run:

- use the saved `simulation.pes` rather than reconstructing defaults by hand;
- restart from a row written by `states.csv`;
- keep future event definitions in the configuration;
- do not overwrite restored state through conflicting `simulation.initial.states` entries.

The continuation output starts a new result directory; PESLite does not append to the previous CSV.

## Custom events

Register a custom event class with `register_event_type`. Its frozen `Params` dataclass must include
`type` and `t`, and `apply(event, system, t)` performs the action. Custom events receive the same
exact-boundary scheduling as built-in events. See [Extending PESLite](Extending-PESLite.md).
