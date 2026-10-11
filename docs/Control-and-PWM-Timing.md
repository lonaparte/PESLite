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
    current: {type: dq_current_pi, period: 50e-6}
    dc_voltage: {type: dc_voltage_pi, period: 500e-6}
```

A loop period must be an integer multiple of `ctrl.period`. If omitted, it inherits the controller
period. Loop state is preserved when run-time parameter changes rebuild a loop.

## ADC sampling

The ADC exposes terminal voltage `u_g`, converter current `i_c` and DC voltage `u_dc`. Channels not
listed under `meas.average` are instantaneous at the controller interrupt. Listed channels use the
mean over a window ending at that interrupt:

```yaml
meas:
  average: [u_g, i_c]
  window: 50e-6
  period: 12.5e-6
```

`meas.period` is the oversampling period and defaults to `ctrl.period`. `meas.window` defaults to
the control period and cannot exceed it. Window integrals are named states so a saved run can
restore an open window; the individual intermediate oversamples are not persisted.

## Computation time

`ctrl.computation` is the time from the ADC sample until that controller result may be used by a
PWM register load. It defaults to 1 microsecond in sampled modes and must be nonnegative and shorter
than the controller period.

There is one shadow register, not a queue of delayed duty words. A new controller result replaces
the shadow word. At a load instant, the active register copies the shadow word only when the
corresponding computation has completed; otherwise the previous active word remains in force.

## Shadow and active registers

Both register banks contain `d_a`, `d_b`, `d_c` and the PWM enable bit:

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

`pwm.method` selects `spwm` or `svpwm`. `pwm.sync: asynchronous` places the carrier on absolute
time and allows `carrier_phase`; `synchronous` locks carrier timing to the controller angle and
requires an integer pulse ratio `f_sw / base.f0`.

## Startup and synchronization

The controller's `Startup` state handles run/stop commands and reference ramping. A connect event
requests a start; a tripped unit rejects that request. Before an initially connected unit starts,
its synchronization law may track the usable terminal voltage so enabling the bridge does not
introduce an artificial phase jump.

Grid-following synchronization normally comes from the PLL. Grid-forming controllers use their
configured synchronization law and startup tracking rather than assuming the grid angle is their
free-running angle.

## Protection clocks

All protection and the trip latch belong to the `VSC`:

- instantaneous overcurrent protection runs at switching or held-bridge edges;
- voltage, frequency, DC-voltage and ROCOF criteria run from controller-rate measurements;
- sampled criteria are armed after startup reports completion;
- a trip blocks the gates, opens the AC terminal and disconnects the DC source.

The controller reports measurements and startup status but does not own the physical trip action.

## Ideal averaging

With `bridge.model: averaging`, controller states and plant states are integrated together. There
is no ADC sampler, interrupt grid, computation wait, carrier, shadow register or active register.
The sampled timing settings may remain in a shared file but are silently ignored for that unit.
