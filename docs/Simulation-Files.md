# Simulation Files

PESLite simulations are YAML mappings. `.pes` is the conventional extension, but any text-file
extension is accepted. Files are parsed strictly: unknown keys, invalid types and inconsistent
cross-references are reported before the run starts.

## Top-level structure

| Section | Purpose |
|---|---|
| `base` | System power rating and, when needed, AC voltage and nominal frequency bases. |
| `buses` | Electrical nodes with a shunt capacitance and optional damping. |
| `branches` | R-L connections between named buses or Unit ports. |
| `sources` | AC voltage sources behind R-L impedance, connected to a bus or Unit port. |
| `units` | Converter power stage, sensing, controller, bridge and protection, when present. |
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
frequencies are hertz unless named `omega`, and switches are `0` or `1`. YAML `true` and `false` are
also accepted; resolved files write `0` and `1`.

`s_base` is the only unconditional base field. `v_ll_rms` is required by three-phase buses, VSCs
and AC per-unit voltage or impedance values; `f0` is required by VSCs, AC sources, synchronous PWM
and frequency-based per-unit values. A scalar circuit written in SI therefore uses, for example,
`base: {s_base: 500.0}` without placeholder AC values. A model with no converters simply has no
`units` section; buses, branches, sources and elements use the same assembly and solver path.

| Field | Required | Unit | Default | Meaning |
|---|---:|---|---|---|
| `s_base` | yes | VA | — | system power rating |
| `v_ll_rms` | when an AC voltage base is needed | V RMS | — | three-phase line-to-line voltage base |
| `f0` | when a frequency base is needed | Hz | — | nominal frequency |

## Network and elements

The network sections are optional mappings of user-defined names to electrical components. A bus
is a named capacitor node; a circuit that does not need such a node does not need a `buses` entry.

### Buses

```yaml
buses:
  pcc: {c_pu: 0.02, r_d_pu: 0.5, phases: 3}
```

| Field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `c` or `c_pu` | yes | F or pu | — | shunt capacitance |
| `r_d` or `r_d_pu` | no | ohm or pu | 0 | series damping resistance |
| `phases` | no | `1` or `3` | `3` | scalar or three-phase space-vector node |

### Branches

A branch is an R-L connection between two buses or Unit ports:

```yaml
branches:
  line: {bus1: grid_bus, bus2: pcc, l_pu: 0.2, r_pu: 0.02}
```

| Field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `bus1`, `bus2` | yes | bus or Unit-port name | — | branch terminals |
| `l` or `x_pu` | yes | H or pu | — | series inductance |
| `r` or `r_pu` | no | ohm or pu | 0 | series resistance |

### Sources

```yaml
sources:
  grid: {bus: pcc, x_pu: 0.4, r_pu: 0.04}
```

| Field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `bus` | yes | bus or Unit-port name | — | source terminal |
| `l` or `x_pu` | yes | H or pu | — | series inductance |
| `r` or `r_pu` | no | ohm or pu | 0 | series resistance |
| `v` or `v_pu` | no | phase-peak V or pu | system base | voltage magnitude |
| `f` | no | Hz | `base.f0` | frequency |
| `angle` | no | rad | 0 | initial angle |

Source voltage is phase-peak volts, frequency is hertz and angle is radians. Omitted `v` and `f`
use the applicable system-base values.

### Elements

```yaml
elements:
  load: {type: resistor, bus: pcc, r: 10.0}
```

| `type` | Required fields | Optional fields | Meaning |
|---|---|---|---|
| `load` | `bus`, `l` or `x_pu` | `r` or `r_pu` | series R-L load |
| `resistor` | `bus`, `r` | `series` (default 0) | shunt resistor or series resistor after the bus voltage driver |
| `voltage_source` | scalar `bus`, `v`, `r` | — | constant voltage behind series resistance |

Per-unit element inputs use the system base. Registered element types define their own fields
([Extending PESLite](Extending-PESLite.md#custom-circuit-elements)).

## Units and controllers

A Unit groups one converter or a composite of converters.

### Common Unit fields

| Field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `type` | no | `vsc`, `dc_converter` or `wpt` | `vsc` | Unit type |
| `bus` | by Unit type | bus name | — | external network connection |
| `s_base` | no | VA | `base.s_base` | Unit power rating |
| `connected` | no | switch | event-derived | initial connection command |
| `dclink` | converter Units | mapping | — | DC rating, source and optional capacitor |
| `bridge` | converter Units | mapping | defaults by Unit type | bridge topology and simulation model |
| `pwm` | sampled controlled bridges | mapping | — | switching and register-update settings |
| `ctrl` | controlled bridges | mapping | — | controller definition |
| `meas` | no | mapping | `{}` | measurements and sampling |
| `protection` | no | mapping | disabled | converter protection |

### Unit types

A VSC adds `ac_filter` and connects to a three-phase bus:

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

| `unit.type` | Required fields | Optional fields | Meaning |
|---|---|---|---|
| `vsc` | `bus`, `ac_filter`, `dclink` | common Unit fields | three-phase voltage-source converter |
| `dc_converter` | `dclink` | paired `bus` and scalar `filter`, common Unit fields | open scalar converter |
| `wpt` | `primary`, `secondary`, `link` | `s_base` | composite of two scalar converter Units and a compensated link |

`dclink.vdc_ref` is the converter's rated DC voltage. `dclink.capacitor` accepts `c` and `r_esr`;
`dclink.source.type` is `voltage`, `current` or `none`. See
[Converter and Bridge Models](Converter-and-Bridge-Models.md) for bridge topology, DCM and device
behavior.

### Controllers

`ctrl.type` selects the controller interface and default wiring: `gfl`, `gfm`, `dc`, `open_loop`,
`resonant`, `phase_shift` or `custom`.

| Field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `type` | yes | registered controller profile | — | controller interface and default wiring |
| `period` | sampled control only | s | PWM period | controller interrupt period |
| `offset` | no | s | PWM load-clock offset | controller clock position |
| `computation` | no | s | 1 us | command-computation delay |
| `loops` | yes | mapping | — | control-loop nodes and parameters |
| `connections` | no | mapping | profile wiring | explicit graph connections |
| `outputs` | no | mapping | profile outputs | outputs supplied to the Unit |
| `references` | by profile | mapping | profile defaults | controller reference values |

Each `ctrl.loops` key is both the registered class and the graph-node name; no nested `type` field
is needed. `{}`, a bare key and YAML `null` all mean an empty parameter mapping. Repeated uses of
one class use names such as `Observer[voltage]` and `Observer[current]`. See
[Control and PWM Timing](Control-and-PWM-Timing.md) for loop periods, ADC timing and register
updates.

### Bridge and PWM

| `bridge` field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `type` | no | `half_bridge`, `full_bridge` or `three_phase_bridge` | Unit-specific | topology |
| `model` | no | `switching`, `pwm_averaging` or `averaging` | `pwm_averaging` | simulation form |
| `upper_controlled`, `lower_controlled` | no | switches | 1 | controlled device rows |
| `reverse_port` | no | switch | 0 | reverse the scalar power-port direction |
| `external_terminal` | no | `midpoint` or `lower_rail` | `midpoint` | exposed half-bridge terminal |
| `on_resistance` | no | ohm per conducting device | 0 | device conduction resistance |
| `parallel_capacitance` | no | F | 0 | capacitance across the bridge power port |

| `pwm` field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `f_sw` | sampled controlled bridges | Hz | — | carrier and default control frequency |
| `modulation_limit` | no | ratio | 1 | modulation magnitude limit |
| `update` | no | `single` or `double` | `single` | PWM register-load rate |
| `carrier_phase` | no | carrier periods | 0 | carrier position |
| `method` | no | `spwm` or `svpwm` | `spwm` | modulation method |
| `sync` | no | `asynchronous` or `synchronous` | `asynchronous` | carrier synchronization |
| `pattern_sync` | no | `interval` or `pulse` | `interval` | command-update boundary |

See [Converter and Bridge Models](Converter-and-Bridge-Models.md) for model selection and
[Control and PWM Timing](Control-and-PWM-Timing.md) for register timing.

### DC converters and WPT

A `dc_converter` uses the same common Unit fields and may add a scalar series `filter`. Buck,
Boost and Buck-Boost are circuit connections assembled from that Unit and external network
components; see the bundled cases in [Examples](Examples.md).

A `wpt` contains `primary` and `secondary` child Units and one compensated `link`. The children
are configured like ordinary `dc_converter` Units and may use independent PWM and controller
settings. The complete SS and LCC examples are listed under
[WPT power feedback](Examples.md#wpt-power-feedback).

### Phase-shift bridges

`ctrl.type: phase_shift` uses a controlled full bridge with the references `width`, `frequency`,
`phase` and `mode`. It requires `bridge.model: switching`. See
[Phase-shift modulation](Converter-and-Bridge-Models.md#phase-shift-square-wave-modulation) and the
bundled `wpt-phase-shift-example` for the waveform and a complete configuration.

### Compensated links

| `link.type` | Required fields | Optional fields | Meaning |
|---|---|---|---|
| `ss` | primary/secondary inductance and capacitance, mutual inductance | primary/secondary resistance and SCC | series-series link |
| `lcc` | coil and compensation inductances, series and parallel capacitances, mutual inductance | corresponding resistances and SCC | LCC-LCC link |
| `lcc_s` | LCC primary fields, secondary inductance and capacitance, mutual inductance | corresponding resistances and SCC | LCC-series link |
| `ss_multi` | inductance matrix and primary/secondary capacitance lists | primary/secondary resistance lists | multi-module series-series link |

`mutual_inductance` may be negative according to the selected winding polarity.

### Switch-controlled capacitors

`ss`, `lcc` and `lcc_s` links accept `primary_scc` and `secondary_scc`:

| Field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `capacitance` | yes | F | — | switched capacitance |
| `angle` | no | rad in [0, pi] | 0 | conduction angle |
| `drive` | no | `common` or `alternating` | `common` | gate pattern |
| `sync` | no | `current` or `bridge` | `current` | phase reference |
| `sync_angle` | no | rad | 0 | bridge-synchronised phase offset |
| `command` | no | switch | 0 | use controller output `scc_angle` |

See [Exact commutations](Converter-and-Bridge-Models.md#exact-commutations) and the bundled
`wpt-scc-example` for behavior and a complete configuration.

### Diode full-bridge rectifiers

A `full_bridge` with neither device row controlled is a diode rectifier and has no `ctrl` or `pwm`:

```yaml
bridge: {type: full_bridge, upper_controlled: 0, lower_controlled: 0}
```

See [Diode full-bridge rectifier](Converter-and-Bridge-Models.md#diode-full-bridge-rectifier) for
its electrical behavior.

### Measurement

| Field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `average` | no | list of channels | `[]` | channels sampled as window means |
| `window` | no | s | controller period | measurement window |
| `period` | no | s | controller period | measurement period |
| `offset` | no | s | controller offset | measurement-clock position |
| `sources` | no | mapping | `{}` | channels mapped to physical outputs or Unit-port quantities |
| `rms` | no | channel-to-seconds mapping | `{}` | sliding RMS windows |

See [ADC sampling](Control-and-PWM-Timing.md#adc-sampling) for timing and
[Examples](Examples.md#independent-loads-and-measurement-selection) for source selection.

### Protection

Each `protection` criterion is switched by `enable`; an enabled limit must be positive. Current and
voltage limits are per unit of the converter's base (the DC base for a scalar unit, as above).

| Criterion | Required fields | Default | Effect |
|---|---|---|---|
| `overcurrent` | `enable`, `limit_pu` | disabled | trip on instantaneous current |
| `rms_current` | `enable`, `limit_pu` and a current `meas.rms` window | disabled | trip after the RMS limit persists for `hold` |
| `undervoltage`, `overvoltage` | `enable`, `limit_pu` | disabled | trip after the terminal-voltage limit persists for `hold` |
| `dc_voltage` | `enable`, `limit_pu` | disabled | trip after the DC-link deviation persists for `hold` |
| `frequency` | `enable`, `limit` in Hz | disabled | trip after the frequency deviation persists for `hold`; VSC only |
| `rocof` | `enable`, `limit` in Hz/s; optional `window` | disabled; window 0.1 s | raise the `ROCOF` alarm; VSC only |

`hold` (s, default 0.02) is the time a held criterion must persist.
[Protection clocks](Control-and-PWM-Timing.md#protection-clocks) describes when each criterion is
evaluated and armed, and what a trip does.

## Events

```yaml
events:
  connect_vsc: {type: connect, target: vsc, t: 0.2, ramp: 0.2}
  power_step: {type: set, t: 0.5, set: {units.vsc.ctrl.references.p_ref_pu: 0.8}}
```

| `type` | Required fields | Optional fields | Meaning |
|---|---|---|---|
| `connect` | `target`, `t` | `ramp` | connect a Unit, source, branch or element |
| `disconnect` | `target`, `t` | — | disconnect a Unit, source, branch or element |
| `set` | `t`, `set` | `ramp` for connection commands | change one or more writable variables |

Event times are exact simulation boundaries. See [Events and Restart](Events-and-Restart.md) for
ordering, writable paths, initial states and continuation.

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
  stop_on_trip: 1
  energy_check: warn
```

| Field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `t_end` | yes | s | — | simulation end time |
| `solver` | no | mapping | fixed-step RK4 | numerical solver settings |
| `initial` | no | mapping | `t: 0`, no explicit states | start time and state values |
| `output` | no | mapping | standard state output | output periods and files |
| `stop_on_trip` | no | switch | 1 | stop at the first converter trip |
| `progress` | no | mapping | disabled | console progress and watched values |
| `energy_check` | no | `warn`, `strict` or `off` | `warn` | construction-time energy audit |

### Solver

| Field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `type` | no | `fixed` or `adaptive` | `fixed` | solver family |
| `method` | no | method name | `rk4` | integration method |
| `dt` | fixed solver | s | 1 us | maximum fixed system step |
| `rtol` | adaptive solver | ratio | 1e-6 | relative error tolerance |
| `atol` | adaptive solver | state units | 1e-9 | absolute error tolerance |
| `max_step` | adaptive solver | s | unlimited | maximum adaptive step |
| `warm_start` | no | switch | 1 | reuse the previous adaptive step size |
| `subsystems` | no | mapping | `{}` | multirate subsystem schedule |
| `sweeps` | no | positive integer | 2 | coupling sweeps for sub-stepped subsystems |
| `linearisations` | no | non-negative integer | 3 | split-error linearisation points |
| `write_length` | no | positive integer | 1000 | rows per streamed CSV write |
| `phs_check_step` | no | positive integer | 1000 | snapshots between energy audits |

See [Solvers](Solvers.md) and [Multirate Simulation](Multirate-Simulation.md) for the available
methods and multirate fields.

### Initial state

| Field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `t` | no | s | 0 | simulation start time |
| `states` | no | mapping | `{}` | initial values keyed by public state path |

The start and end times are exact and are not rounded to an ADC, PWM or solver grid. State names,
values and keywords are described in [Events and Restart](Events-and-Restart.md#initial-state).

### Output and progress

| `output` field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `period` | no | s | 0.5 ms | plant snapshot period |
| `record_every` | no | positive integer | 1 | keep every nth controller sample |
| `states` | no | switch | 1 | write `states.csv` |
| `signals` | no | switch | 0 | write plant and controller signal files |
| `energy` | no | switch | 0 | write `energy.csv` |

| `progress` field | Required | Unit/type | Default | Meaning |
|---|---:|---|---|---|
| `enable` | no | switch | 0 | print simulation progress |
| `period` | when enabled | s | — | simulated time between progress lines |
| `watch` | no | list of public output paths | `[]` | values appended to each progress line |

See [Output and Results](Output-and-Results.md) for result files and public output names.

## Metadata

```yaml
meta:
  title: Example study
  description: Optional descriptive text
```

`meta` never changes the model or simulation.

## Defaults and resolved files

Some defaults are derived from other values. For example, sampled control defaults to one carrier
period, ADC timing defaults to the control period, an AC source's `v` defaults to 1 pu of the
system base and a DC voltage source's `v` to `dclink.vdc_ref`. Derived defaults are recomputed
after `--set` or `Params.replace()` changes their source parameter.

Print the complete resolved tree with:

```bash
peslite case.pes --resolved
```

Every completed run writes the same complete tree to `simulation.pes`. Loading that file reproduces
the configuration; fields written as `null` are derived again in the same way.

## Command-line overrides

Use repeatable dotted paths:

```bash
peslite case.pes \
  --set simulation.t_end=1.0 \
  --set units.vsc.ctrl.references.p_ref_pu=0.4
```

Their priority is given in [CLI Reference](CLI-Reference.md#override-parameters). Invalid
combinations are checked after all overrides are applied.

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
