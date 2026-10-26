# Simulation Files

PESLite simulations are YAML mappings. `.pes` is the conventional extension, but any text-file
extension is accepted. Files are parsed strictly: unknown keys, invalid types and inconsistent
cross-references are reported before the run starts.

## Top-level structure

| Section | Purpose |
|---|---|
| `base` | System power, line-line RMS voltage and nominal frequency. |
| `buses` | Electrical nodes, including optional shunt capacitance and damping. |
| `branches` | R-L connections between buses. |
| `sources` | Grid voltage sources and their connection impedance. |
| `units` | Converter power stage, sensing, controller, bridge and protection. |
| `elements` | Built-in or registered circuit elements such as R-L loads. |
| `events` | Timed connect, disconnect and parameter-change operations. |
| `simulation` | End time, solver, initial state, output and progress settings. |
| `meta` | Optional title and description; never affects the model. |

Names are unique within each of `buses`, `branches`, `sources`, `units` and `elements`. The same
spelling may be used in different sections when their assembled subsystem and signal names remain
distinct. An event target must still identify exactly one switchable object. Names must be
non-empty and cannot contain a dot, because dots separate public parameter and state paths.

## Units and per-unit values

Power-circuit values are stored internally in SI. A parameter without a unit suffix is an SI
quantity; a parameter ending in `_pu` is converted using the applicable system or converter base.
Do not provide both forms of the same parameter.

```yaml
base: {s_base: 2.0e6, v_ll_rms: 690.0, f0: 50.0}

branches:
  line: {bus1: grid_bus, bus2: pcc, l_pu: 0.2, r_pu: 0.02}
```

Each converter may define its own `s_base`; otherwise it uses the system base. Controller
references and loop parameters use that converter's base. Time is in seconds, angles are radians,
frequencies are hertz unless named `omega`, and switches are written as `0` or `1`.

`bus`, `dclink`, `pwm`, `ctrl`, `bridge`, `meas`, `protection` and `s_base` are common to every
non-composite converter unit. The selected `unit.type` contributes its own electrical fields:
`vsc` adds `ac_filter`, while `dc_converter` adds `filter`. The common ADC and protection lifecycle
are shared by both converter families.

`meas.sources` optionally maps existing channel names to physical subsystem outputs or named
port quantities. `meas.rms` maps channels to sliding RMS windows in seconds. Both use the same
measurement path for VSC and DCConverter; no WPT-specific channel schema is required. Configured
RMS overcurrent criteria use `protection.rms_current` and the existing `hold` and trip latch.
For wiring, window semantics and an independent load example, see
[Independent loads and measurement selection](Examples.md#independent-loads-and-measurement-selection).

## Converter section

A unit names its terminal bus and groups its physical and control configuration:

```yaml
units:
  vsc:
    bus: pcc
    ac_filter: {l_f_pu: 0.2, r_f_pu: 0.01}
    dclink:
      vdc_ref: 1500.0
      source: {type: voltage}
    pwm: {f_sw: 20000.0, update: single, method: spwm}
    bridge: {model: pwm_averaging}
    ctrl:
      type: gfm
      loops:
        PSC: {k_p_pu: 6.2832, k_vi: 20.0}
        VirtualImpedance: {}
        ActiveDamping: {r_a_pu: 0.2, alpha_d: 40.0}
      references: {p_ref_pu: 0.5}
```

`ctrl.type` selects the controller architecture: its reference schema, public ports and default
wiring. Built-in architectures include `gfl`, `gfm`, scalar `dc` and `resonant`; a custom graph can
override loop connections and controller outputs explicitly. Normally, each key under `ctrl.loops`
is the registered loop class name and also its graph-node name; no duplicate `type` field is
needed. PESLite builds only nodes that can reach a controller output through the connection graph.
Examples use `{}` for an empty parameter mapping. A bare key or explicit YAML `null` is accepted
as an equivalent input in `ctrl.loops`.
Repeated uses of one class use a class-qualified name, such as `Observer[voltage]` and
`Observer[current]`; connections address those complete names. See
[Control and PWM Timing](Control-and-PWM-Timing.md) and
[Converter and Bridge Models](Converter-and-Bridge-Models.md).

Shared operations such as `complex_power`, coordinate transforms and `unit_delay` are functions,
not configurable graph nodes. A synchronization law calls `complex_power` directly and owns its
optional power-filter states. A loop that needs a one-sample delay owns the previous value and
passes it explicitly to `unit_delay(value, previous)`. Consequently, unused helper functions create
no objects or runtime state. `VirtualImpedance`, `ActiveDamping` and `ResonantModulation` remain
ordinary registered loops because they have configurable behavior and graph ports.

A WPT unit is a configuration-time composite. It owns no bridge or controller itself; each child
is an ordinary independently clocked `DCConverter`. Each bridge and its matching link terminal
attach through the child's own assembly boundary. The WPT exposes `primary` and `secondary` Unit
ports and explicitly selects `primary` as its default port. Composite event parameters and shared
subsystem ratings therefore inherit the primary port unless another port is selected. The bundled
open-loop example selects a full bridge and `ResonantModulation` for each child; neither choice
creates a separate converter class. `wpt-ss-example` and `wpt-lcc-example` connect ordinary
power-feedback loops to the same modulation inputs. The `resonant` profile's `references.power`
is signed watts; `modulation`, `frequency` and `phase` remain amplitude, hertz and radians. The
complete runnable schemas are shown in [Examples](Examples.md#wpt-power-feedback).

```yaml
units:
  charger:
    type: wpt
    primary:
      type: dc_converter
      dclink: {vdc_ref: 400.0, source: {type: voltage}}
      pwm: {f_sw: 1.0e6}
      ctrl:
        type: resonant
        computation: 0.0
        loops:
          ResonantModulation: {}
        references: {modulation: 0.8, frequency: 85000.0}
    secondary:
      # The same fields, with its own DC link, PWM clock and phase command.
      type: dc_converter
      dclink: {vdc_ref: 300.0, source: {type: voltage}}
      pwm: {f_sw: 800.0e3}
      ctrl:
        type: resonant
        computation: 0.0
        loops:
          ResonantModulation: {}
        references: {modulation: 0.7, frequency: 85000.0, phase: -1.5708}
    link:
      type: ss
      primary_inductance: 100.0e-6
      secondary_inductance: 100.0e-6
      mutual_inductance: 20.0e-6
      primary_capacitance: 35.07e-9
      secondary_capacitance: 35.07e-9
```

### Compensated links

All link types are configurations of one compensated-link model: coupled coils, each with a series
capacitor and resistance, and per side either *series* compensation (the coils connect to the port)
or *LCC* compensation (a series compensation inductor from the port to a shunt-capacitor node).

| `link.type` | Primary | Secondary | Fields |
|---|---|---|---|
| `ss` | series | series | `primary_/secondary_inductance`, `mutual_inductance`, `primary_/secondary_capacitance`, optional `primary_/secondary_resistance` |
| `lcc` | LCC | LCC | coil inductances and `mutual_inductance`; per side `*_series_capacitance`, `*_compensation_inductance`, `*_parallel_capacitance`, optional `*_resistance`, `*_compensation_resistance`, `*_parallel_resistance` (ESR of the shunt capacitor) |
| `lcc_s` | LCC | series | the `lcc` primary fields and `secondary_capacitance`, `secondary_resistance` |
| `ss_multi` | N series modules | N series modules | `inductance` (symmetric 2N x 2N matrix in the order Tx1..TxN, Rx1..RxN), lists `primary_/secondary_capacitance` and optional `primary_/secondary_resistance` with one value per module |

`mutual_inductance` may be negative; its sign is that of the coupled flux with both currents
positive into their dotted terminals. The link records its modal currents (`i_mode_k`) and every
capacitor voltage as states; the capacitor voltages are also link outputs, so `meas.sources` can
select them (for example `charger.link.u_secondary_cap`). The link signals are the two port
currents `i_primary`/`i_secondary`, the summed coil currents `i_primary_coil`/`i_secondary_coil`
and, for `ss_multi`, each coil current `i_primary_coil_k`/`i_secondary_coil_k`. All declared
compensation-capacitor voltages are also published with their ordinary link output names.

### Diode full-bridge rectifiers

A `full_bridge` with neither device row gated is a diode rectifier. It has no `ctrl` or `pwm`;
`bridge.model` must be `switching`. For example, a WPT secondary may be configured as:

```yaml
    secondary:
      type: dc_converter
      dclink: {vdc_ref: 300.0, source: {type: voltage}}
      bridge: {type: full_bridge, upper_controlled: 0, lower_controlled: 0, model: switching}
```

The diode commutations are located exactly (see
[Converter and Bridge Models](Converter-and-Bridge-Models.md#diode-full-bridge-rectifier)). Blocking
uses the connected terminal's zero-current voltage law, available on ordinary R-L branches and
compensated links. No unit-type or restriction to one rectifying side is imposed.
`meas` may configure independent sampling, means, RMS and existing physical sources; its
`period` defaults to the output period when there is no controller.
Optional `bridge.parallel_capacitance` (F) models the equivalent port capacitance and requires
positive `bridge.on_resistance` (ohm per device). These fields apply to all bridge structures;
the device law, capacitor and ordinary DC-network connections determine the charging current.
A diode full-bridge rectifier accepts no protection criterion (it has no gates to block).
Without port capacitance, an adaptive solver requires finite `simulation.solver.max_step`:
it is the interval at which ideal diode commutations are checked.

## Simulation settings

```yaml
simulation:
  t_end: 3.0
  solver: {type: fixed, method: rk4, dt: 25e-6}
  initial:
    t: 0.0
    states: {}
  output:
    period: 0.5e-3
    record_every: 10
    states: 1
    signals: 0
    energy: 0
  energy_check: warn
```

The start and end times are exact and are not rounded to an ADC, PWM or solver grid. Initial state
keys use the same names as `states.csv`; complex values may be written as `[real, imag]` or through
`.re` and `.im` entries. The keywords `source` and `rated` initialize supported electrical states
from the connected source or converter rating.

## Defaults and resolved files

Some defaults are derived from other values. For example, sampled control defaults to one carrier
period and ADC timing defaults to the control period. Derived defaults are recomputed after
`--set` or `Params.replace()` changes their source parameter.

Print the complete resolved tree with:

```bash
peslite case.pes --resolved
```

Every completed run writes the same complete tree to `simulation.pes`. Loading that file reproduces
the configuration without relying on implicit defaults.

## Command-line overrides

Use repeatable dotted paths:

```bash
peslite case.pes \
  --set simulation.t_end=1.0 \
  --set units.vsc.ctrl.references.p_ref_pu=0.4
```

The priority is `--set` over a bridge/solver command preset, over the simulation file, over package
defaults. Invalid combinations are checked after all overrides are applied.

## Python loading and replacement

```python
import peslite

params = peslite.load("case.pes")
changed = params.replace(**{
    "simulation.t_end": 1.0,
    "units.vsc.ctrl.references.p_ref_pu": 0.4,
})
peslite.dump(changed, "changed.pes")
```

`Params` objects are immutable; `replace()` returns a new validated tree. For event-time changes,
see [Events and Restart](Events-and-Restart.md).
