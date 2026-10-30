# Voltage-Source Converters

Use `type: vsc` for a three-phase voltage-source converter.

```yaml
units:
  vsc:
    type: vsc
    bus: pcc
    ac_filter: {l_f_pu: 0.2, r_f_pu: 0.01}
    dclink:
      vdc_ref: 1500.0
      source: {type: voltage}
    bridge: {model: pwm_averaging}
    pwm: {f_sw: 20000.0, update: single, method: spwm}
    ctrl:
      type: gfl
      loops:
        SRFPLL: {k_p_pu: 20.0, k_i_pu: 1200.0}
        CurrentLoop: {bandwidth: 200.0}
        DCVoltageControl:
          kp: 0.8
          ki: 90.0
          minimum: 0.0
          maximum: 1.15
          feedforward: 0.9
          output_polarity: -1.0
```

## Unit fields

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `type` | no | `vsc` | Unit type |
| `bus` | yes | — | three-phase terminal bus |
| `s_base` | no | `base.s_base` | converter rating in VA |
| `connected` | no | event-derived | initial connection command |
| `ac_filter` | yes | — | series converter-side R-L filter |
| `dclink` | yes | — | rated DC voltage, source and optional capacitor |
| `bridge` | no | three-phase bridge with `pwm_averaging` | bridge model and device parameters |
| `pwm` | sampled controlled bridge | — | carrier and register settings |
| `ctrl` | controlled bridge | — | controller graph |
| `meas` | no | `{}` | ADC channels, means, sources and RMS windows |
| `protection` | no | disabled | protection criteria |

The connected bus must use `phases: 3`. `base.v_ll_rms` and `base.f0` are required. Filter fields
are `l_f` or `l_f_pu` and optional `r_f` or `r_f_pu`.

## Controllers

| `ctrl.type` | Use |
|---|---|
| `gfl` | grid-following control graph |
| `gfm` | grid-forming control graph |
| `custom` | explicit loop graph and wiring |

`ctrl.loops` keys are registered loop classes. `ctrl.connections` and `ctrl.outputs` override
profile wiring. Periods, ADC configuration and bridge models are documented under
[Bridge and PWM](Bridge-and-PWM.md). DC-link source fields are documented under
[Electrical Sources](Electrical-Sources.md#converter-dc-source).

## Measurements and protection

Built-in VSC measurement channels are `u_g`, `i_c` and `u_dc`. Optional `meas.sources` may bind
public bus, port or element outputs.

Protection criteria are `overcurrent`, `rms_current`, `undervoltage`, `overvoltage`, `dc_voltage`,
`frequency` and `rocof`. Each criterion is enabled independently; see
[Bridge and PWM](Bridge-and-PWM.md#protection-periods) for evaluation periods.

Complete GFL and GFM configurations are listed in [Examples](Examples.md).
