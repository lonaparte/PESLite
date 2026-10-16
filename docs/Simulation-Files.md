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

Names in `buses`, `branches`, `sources`, `units` and `elements` are unique together. They must be
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
is an ordinary independently clocked `DCConverter` whose scalar power port is connected to the
shared link during assembly. The bundled example selects a full bridge and the temporary
open-loop resonant-modulation loop for each child; neither choice creates a separate converter
class. `link.type` selects series-series (`ss`) or LCC-LCC (`lcc`) compensation. The complete
runnable schema is shown in `examples/wpt-example.pes`.

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
