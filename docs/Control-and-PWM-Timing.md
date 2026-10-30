# Control and PWM Timing

Switching and PWM-period averaging share one sampled-data timing model. Ideal averaging replaces
that schedule with continuous controller equations; see
[Converter and Bridge Models](Converter-and-Bridge-Models.md) for the model-level distinction.

## Sampled-control sequence

At one controller interrupt, PESLite performs these operations in order:

1. finish the integration interval exactly at the interrupt;
2. read instantaneous ADC channels, or consume the latest held conversion from an independent
   ADC clock, and close any configured averaging windows;
3. update the control loops due at this interrupt;
4. evaluate sampled protection from the same measurement;
5. make the controller command eligible after `ctrl.computation`;
6. load an eligible shadow word into the active PWM register at a permitted load instant;
7. apply the resulting held duty value or switching sequence to the bridge.

All units sample before any unit actuates at the same time, so unit ordering does not give one
converter access to another converter's newly updated output.

## Controller period and loop periods

For sampled bridges, `units.<u>.ctrl.period` defaults to one switching period. It must be a whole
multiple of half a carrier period because interrupts are tied to carrier valleys or peaks.
`ctrl.offset` is the first clock position in seconds within one control period. It defaults to the
PWM load-clock position, so existing files retain the carrier-aligned schedule. Setting it
explicitly moves controller interrupts without moving PWM register loads; both schedules use the
same period-and-offset clock representation.

Individual loops may run more slowly. This fragment shows only the periods; the loops' other
fields are omitted:

```yaml
ctrl:
  period: 50e-6
  loops:
    CurrentLoop: {period: 50e-6}        # other fields omitted
    DCVoltageControl: {period: 500e-6}  # other fields omitted
```

Loop keys and graph-node names follow the naming rules in
[Simulation Files](Simulation-Files.md#units-and-controllers).

A loop period must be an integer multiple of `ctrl.period`. If omitted, it inherits the controller
period. Loop state is preserved when run-time parameter changes rebuild a loop.

`DCVoltageControl` is the common voltage-to-current loop for VSC and DCConverter control graphs.
Its error is always `reference - voltage`, and its output is `current_reference`. Parameters use
the coordinates of the containing controller graph: per-unit values in a VSC graph and SI values
in a scalar DC graph. `output_polarity` describes the plant direction. It is `+1` when positive
commanded current charges the controlled capacitor and `-1` when positive commanded current draws
energy from it, as with the VSC convention that positive d-axis current exports power. A DCConverter
normally sends the result to `InductorCurrentControl`; a VSC sends it to the d-axis input of
`CurrentLoop`.

### Scalar feedback loops

Scalar voltage, current and power loops use
`command = feedforward + PI(startup * reference - measured)`. Each loop pairs one measurement with
one command:

| Measured quantity | Modulation command | Phase command | Frequency command |
|---|---|---|---|
| power | `PowerControl` | `PhasePowerControl` | `FrequencyPowerControl` |
| voltage | `VoltageControl` | `PhaseVoltageControl` | `FrequencyVoltageControl` |
| current | `CurrentControl` | `PhaseCurrentControl` | `FrequencyCurrentControl` |

The measurement is selected through the control graph, and an optional measurement filter is
configured on the loop. Command limits and gains use the units of the selected command and
measurement. A positive phase or frequency command follows the polarity of the connected plant;
the controller does not infer it.

## ADC sampling

Every sampled converter uses the same ADC implementation. A VSC exposes `u_g`, `i_c` and `u_dc`;
a scalar DC converter exposes `current`, `voltage` and `vin`. An active control path using
`meas.power`, or an explicit `meas.average: [power]` request, also enables instantaneous outward
power at the unit's external `power` terminal. This uses that terminal's paired voltage/current,
direction and power coefficient, not a product of potentially different control feedback ports.
An unused loop does not enable this channel. Direct controller calls must supply port power when
the active graph requires it; missing power is not interpreted as a measured zero.
The optional channel is not evaluated in configurations that do not request it. Channels not listed under
`meas.average` are instantaneous at the controller interrupt. Listed channels use the mean over a
window ending at that interrupt:

```yaml
meas:
  average: [u_g, i_c]
  window: 50e-6
  period: 12.5e-6
```

`meas.period` is the ADC conversion period. For a sampled unit it defaults to `ctrl.period` and must
divide it. A continuous unit (ideal averaging, or a bridge without a controller) uses `meas.period`
as its own measurement and protection clock, which defaults to `simulation.output.period` (see
[Protection clocks](#protection-clocks)). `meas.window` defaults to the control period and cannot
exceed it. Window integrals and their latest quadrature anchors are
named states so a saved run can restore an open window; the individual intermediate conversions
are not persisted.

`meas.offset` is the ADC conversion-clock position in seconds within one `meas.period`. When it is
omitted, the ADC grid contains every controller interrupt and preserves the existing direct-sample
path. An explicit offset may place conversions between controller interrupts. In that case the ADC
runs on its own periodic grid and the controller consumes the most recent held conversion. If a
conversion and controller interrupt coincide, conversion occurs first. This held value is a named
restart state. Ideal averaging has no sampled controller/ADC schedule, so these offsets do not
affect it. For a unit without a controller, `meas.offset` positions its own measurement clock
(default 0).

Sampled-converter ADC windows integrate on physical event boundaries, including configured ADC
conversions, control interrupts, PWM edges and file events, and immediately before a trip changes
the plant. Output snapshots and the end time do not add ADC integration points, and neither does a
continuous unit's protection/measurement clock, which is an integration point only for that unit's
own windows. A window open at the end of a run continues from its saved quadrature anchor. Use
`meas.period` when a window needs a finer conversion grid than the controller period. Python and
generated C++ use the same ADC schedule.

For oscillating scalar ports, average `power` itself rather than multiplying separately averaged
voltage and current. A carrier-boundary voltage sample may be zero even during nonzero power
transfer. The WPT feedback cases use the ordinary ADC window for that product and an ordinary
power-loop low-pass filter for the remaining excitation ripple; see [Examples](Examples.md#wpt-power-feedback).

### Measurement sources and RMS

`meas.sources` routes an existing physical output to a channel; bus and Unit ports expose
`.voltage`, `.current` and `.power`, with port power positive outward from the voltage-driving
terminal; named element outputs such as `load.power` are also valid. Bindings are non-loading,
resolve during assembly and do not imply hardware-local measurement or communication.

RMS is `sqrt(mean(abs(signal)**2))` over whole control periods, or whole `meas.period` intervals for
a continuous unit. It is total waveform RMS, not a fundamental or per-phase estimate, and an
averaged model cannot restore omitted switching harmonics; for a complex space vector it is the
RMS of its magnitude. Values become available after one full window as
`<unit>.meas.<channel>_rms`; only configured windows allocate state.

## Computation time

`ctrl.computation` is the time from the ADC sample until that controller result may be used by a
PWM register load. It defaults to 1 microsecond in sampled modes and must be nonnegative and shorter
than the controller period.

There is one shadow register, not a queue of delayed duty words. As on a DSP, an interrupt's result
replaces the shadow word when its computation completes, and a register load copies whatever the
shadow holds: a load during an interrupt's computation, including one at the interrupt instant,
takes the previous completed result. An explicit `ctrl.offset` that places interrupts less than
`ctrl.computation` before loads therefore adds one control period of delay. The replaced word is
not a saved state: a continuation (`simulation.initial`) restores it from the active word where
possible ([What is restored](Events-and-Restart.md#what-is-restored)).

## Shadow and active registers

Both register banks contain the selected topology's duty fields and the PWM enable bit. A
three-phase bridge uses `d_a`, `d_b` and `d_c`; scalar half/full bridges expose their registered
duty names through the same bridge contract:

- `shadow`: most recently computed duty word waiting for an eligible load;
- `active`: word currently applied to the modulator and bridge.

The enable bit follows the same path as the duty ratios. This prevents startup, stop and trip logic
from bypassing PWM timing. On a new start, controller integrators are reset by the startup command
before the first enabled duty word is released.

Public saved-state names are entity-first:

```text
<unit>.pwm.d_a
<unit>.pwm.shadow.d_a
<unit>.pwm.on
<unit>.pwm.shadow.on
```

## Single and double update

```yaml
pwm:
  f_sw: 20000.0
  update: single
```

- `single` loads the active register at carrier valleys: one load per switching period;
- `double` loads at valleys and peaks: two loads per switching period.

`pwm.method` selects `spwm` or `svpwm` for a three-phase VSC, whose output stage scales the
modulating signals uniformly so that their peak does not exceed `pwm.modulation_limit` (default 1).
`pwm.sync: asynchronous` places the carrier on absolute time and allows `carrier_phase` (the
carrier position at `t = 0`, in carrier periods); `synchronous` locks three-phase carrier timing
to the controller angle and requires an integer pulse ratio `f_sw / base.f0`. Scalar converters use
the same PWM scheduler with asynchronous timing.

## Startup and synchronization

The controller's `Startup` state handles run/stop commands and reference ramping. A connect event
requests a start; a tripped unit rejects that request. Before an initially connected unit starts,
its synchronization law may track the usable terminal voltage so enabling the bridge does not
introduce an artificial phase jump.

Grid-following synchronization normally comes from the PLL. Grid-forming controllers use their
configured synchronization law and startup tracking rather than assuming the grid angle is their
free-running angle.

## Protection clocks

All protection and the trip latch belong to the converter `Unit`, independently of its AC or DC
signal adapter:

- instantaneous overcurrent protection runs at switching or held-bridge edges;
- voltage, frequency, DC-voltage and ROCOF criteria run from controller-rate measurements;
- a continuous unit (ideal averaging, or a controllerless natural-conduction bridge) runs both on
  its own clock with period `meas.period` ([ADC sampling](#adc-sampling)); other units' events,
  file events and output snapshots do not advance its hold timers, ROCOF window or RMS periods;
- sampled criteria are armed after startup reports completion;
- a trip blocks the gates, opens an external terminal when present and disconnects the DC source.

Frequency and ROCOF require the synchronization frequency supplied by a VSC controller. A scalar
DC converter therefore accepts overcurrent, terminal under/overvoltage and DC-link voltage
protection, but rejects frequency criteria during validation.

The controller reports measurements and startup status but does not own the physical trip action.

## Ideal averaging

With `bridge.model: averaging`, controller states and plant states are integrated together. There
is no ADC sampler, interrupt grid, computation wait, carrier, shadow register or active register.
The sampled timing settings may remain in a shared file but are silently ignored for that unit,
except `meas.period`, which sets the unit's protection and RMS clock (see
[Protection clocks](#protection-clocks)). Every loop in the active graph needs a continuous-time
implementation (`flow` or `equation`, or `_flow` for a synchronization law); a loop without one,
such as the `ModeSequence` add-on, is rejected.
