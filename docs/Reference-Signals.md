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
interpolation, extrapolation, time scaling and value/time offsets.

### Identification excitations

Use `reference_prbs` for a deterministic maximal-length pseudorandom binary sequence, or
`reference_multitone` for multiple coherent sinusoidal tones. The linear-frequency
`reference_chirp` is a convenient swept-frequency excitation. These generate ordinary native
`TableSignal` data; no new simulation source type or random runtime state is introduced.

```python
from peslite.addons.functions import reference_prbs, reference_multitone, reference_chirp

prbs = reference_prbs(
    start=0.0, stop=0.127, period=0.001, order=7, seed=29,
    low=-0.01, high=0.01,
)
multitone = reference_multitone(
    start=0.0, stop=1.0, period=0.0005,
    frequencies=(10.0, 20.0, 30.0), amplitudes=(0.01, 0.005, 0.005),
    repeat=True,
)
sweep = reference_chirp(
    start=0.0, stop=1.0, period=0.0005,
    start_frequency=10.0, stop_frequency=200.0, amplitude=0.01,
)
```

For PRBS, `period` is one held bit interval and `order` is between 2 and 16; the full
sequence length is `2**order - 1` intervals. A nonzero seed fixes its phase and repeated
invocations reproduce exactly the same samples. With `repeat=True`, playback repeats the
provided table window; choose a complete LFSR period for a continuous sequence.

Multitone frequencies are distinct positive Hz values below the sample-grid Nyquist limit;
phases are in radians. Repeated multitone playback requires every tone to contain an integer
number of cycles in the requested time window. Chirp start and stop frequencies are in Hz,
limited by the same sampling constraint. Amplitude and offset are in the reference's own
physical or per-unit coordinates; `time_scale`, `time_offset`, `value_scale` and `value_offset`
apply the same transform as the existing reference helpers.

Use the returned mapping under a simulation's `signals` entry and bind the target controller
reference to `signals.<name>`. When measuring a frequency response, respect the sample clock
and select a stationary single-tone window after the run has settled; see
[Analysis Functions](Analysis-Functions.md).

Table references are deterministic and require no additional restart state because playback uses
absolute simulation time. Native C++ and Imperix export reject dynamic reference tables; replace
them with constant references before export.

See `reference-signal-example` for a complete simulation file.
