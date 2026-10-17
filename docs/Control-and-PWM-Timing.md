# Control and PWM Timing

Switching and PWM-period averaging share one sampled-data timing model. Ideal averaging replaces
that schedule with continuous controller equations; see
[Converter and Bridge Models](Converter-and-Bridge-Models.md) for the model-level distinction.

## Sampled-control sequence

At one controller interrupt, PESLite performs these operations in order:

1. finish the integration interval exactly at the interrupt;
2. read instantaneous ADC channels and close any configured averaging windows;
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

Individual loops may run more slowly:

```yaml
ctrl:
  period: 50e-6
  loops:
    CurrentLoop: {period: 50e-6}
    DCVoltageControl: {period: 500e-6}
```

The class names are also the graph-node names used by `connections`, outputs and state paths.
Configuration loading resolves them to parameter schemas without constructing control objects.
During assembly, PESLite traces dependencies backward from controller outputs and constructs only
the nodes on that active graph; an unreferenced declaration consumes no runtime loop object.
If a class occurs more than once, an optional label remains part of the node name, for example
`Observer[voltage]` and `Observer[current]`. No loop entry needs a `type` field.

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

`meas.period` is the oversampling period and defaults to `ctrl.period`. `meas.window` defaults to
the control period and cannot exceed it. Window integrals are named states so a saved run can
restore an open window; the individual intermediate oversamples are not persisted.

For oscillating scalar ports, average `power` itself rather than multiplying separately averaged
voltage and current. A carrier-boundary voltage sample may be zero even during nonzero power
transfer. The WPT feedback cases use the ordinary ADC window for that product and an ordinary
power-loop low-pass filter for the remaining excitation ripple; see [Examples](Examples.md#wpt-power-feedback).

## Computation time

`ctrl.computation` is the time from the ADC sample until that controller result may be used by a
PWM register load. It defaults to 1 microsecond in sampled modes and must be nonnegative and shorter
than the controller period.

There is one shadow register, not a queue of delayed duty words. A new controller result replaces
the shadow word. At a load instant, the active register copies the shadow word only when the
corresponding computation has completed; otherwise the previous active word remains in force.

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

`pwm.method` selects `spwm` or `svpwm` for a three-phase VSC. `pwm.sync: asynchronous` places the
carrier on absolute time and allows `carrier_phase`; `synchronous` locks three-phase carrier timing
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
- sampled criteria are armed after startup reports completion;
- a trip blocks the gates, opens an external terminal when present and disconnects the DC source.

Frequency and ROCOF require the synchronization frequency supplied by a VSC controller. A scalar
DC converter therefore accepts overcurrent, terminal under/overvoltage and DC-link voltage
protection, but rejects frequency criteria during validation.

The controller reports measurements and startup status but does not own the physical trip action.

## Ideal averaging

With `bridge.model: averaging`, controller states and plant states are integrated together. There
is no ADC sampler, interrupt grid, computation wait, carrier, shadow register or active register.
The sampled timing settings may remain in a shared file but are silently ignored for that unit.
