# Reference Signals

A controller reference accepts either a constant or a named time table. Tables use absolute
simulation time and are evaluated on the controller's existing sampled or continuous schedule.

## Inline tables

Define tables in the top-level `signals` section and bind them through `ctrl.references`:

```yaml
signals:
  power_command:
    type: table
    time: [0.0, 0.1, 0.2]
    value: [0.0, 0.5, 0.2]
    repeat: true
    interpolation: linear

units:
  vsc:
    ctrl:
      references:
        p_ref_pu: signals.power_command
```

`time` must be strictly increasing and have the same length as `value`. `repeat: true` repeats the
interval from the first to the last time. The available interpolation methods are
`zero_order_hold`, `linear`, `cubic` and `pchip`; extrapolation outside a nonrepeating table can be
`hold`, `zero`, `linear` or `error` and defaults to `hold`.

The table uses the units implied by the reference spelling. For example, `p_ref_pu` reads per-unit
values while `p_ref` reads watts and follows the ordinary SI-to-per-unit conversion. A `set` event
on the same reference replaces its table binding with the event's constant value.

## Generated tables

The built-in function addon creates the same serializable table mapping from arrays, CSV columns or
a mathematical expression:

```python
from peslite.addons.functions import reference_csv, reference_expression

from_csv = reference_csv(
    "command.csv", time_column="time", value_column="power",
    interpolation="pchip", time_scale=0.5, value_offset=100.0,
)
chirp = reference_expression(
    "sin(2*pi*(20*t + 50*t**2))",
    start=0.0, stop=1.0, period=1e-4,
)
```

Expressions may combine `t`, numeric constants, `pi`, `e`, `tau` and the common mathematical
functions. Sums of sine terms produce multiharmonic waveforms. Both helpers support repetition,
interpolation, extrapolation, time scaling and time/value offsets.

Table references are deterministic and require no additional restart state because playback uses
absolute simulation time. Native C++ and Imperix export reject dynamic reference tables; replace
them with constant references before export.

See `reference-signal-example` for a complete simulation file.
