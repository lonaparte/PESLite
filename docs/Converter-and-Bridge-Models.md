# Converter and Bridge Models

Every converter chooses its own bridge model. The network connection and external bridge ports stay
the same, so a case can change fidelity without rewiring the system, and different units in one
network may use different models.

```yaml
units:
  vsc_1:
    bridge: {model: switching}
  vsc_2:
    bridge: {model: pwm_averaging}
```

When `bridge.model` is omitted, the file default is `pwm_averaging`.

## Model comparison

| Model | Electrical bridge | Digital timing | Typical cost |
|---|---|---|---|
| `switching` | Ideal switch states at exact carrier-comparison instants. | ADC, controller interrupts, computation completion and PWM register loads. | Highest. |
| `pwm_averaging` | Active duty ratios applied continuously between register loads; no carrier ripple. | Same sampled controller and PWM-register schedule as switching. | Lower than switching. |
| `averaging` | Ideal controlled voltage source driven by continuous controller equations. | No ADC, controller-interrupt, computation or PWM-register schedule. | Usually lowest for control-envelope studies. |

The models answer different questions. Similar low-frequency trajectories do not make them
numerically identical.

## Scalar converter ports, device placement and DCM

`DCConverter` is an open Unit, not a packaged Buck/Boost circuit. It owns a DC link, a selected
`half_bridge` or `full_bridge`, its controller and an optional series filter. The common terminal
contract connects that filter to a named bus. A scalar DC network uses
`buses.<name>.phases: 1`; a DC converter cannot connect its scalar bridge to a three-phase bus.
The bus capacitor, load and any further branches remain independent network components. Buck,
Boost, resonant conversion and similar names describe assembled applications rather than
additional bridge structures.

A half bridge has an upper rail, a midpoint and a lower rail. Its upper rail connects to the unit's
DC link. `upper_controlled` and `lower_controlled` declare the device at each position; a false
value leaves that position's diode available. A fully controlled half bridge and a full bridge
remain bidirectional.

The same half bridge and filter support these application connections:

| Application | Half-bridge settings | Inductor connection | External `bus` |
|---|---|---|---|
| Buck | `lower_controlled: 0` | Midpoint to output | Output bus; lower rail at the common point |
| Boost | `upper_controlled: 0`, `reverse_port: 1` | Input to midpoint | Input bus; upper rail and DC link form the output |
| Buck-Boost | `lower_controlled: 0`, `external_terminal: lower_rail` | Midpoint to common point | Negative output bus on the lower rail |

`bridge.external_terminal` defaults to `midpoint`, which exposes the midpoint through the series
filter. With `lower_rail`, the named external bus connects to the lower rail instead, and the
filter connects the midpoint to the common point. This setting requires both `bus` and `filter`
and uses `reverse_port: 0`. The DC link supplies the input in the Buck and Buck-Boost examples;
it stores the output voltage in the Boost example.

`buck-boost-example.pes` contains one DCConverter and one inductor. Its electrical output voltage,
including `output.u_C`, is negative relative to the common point. The control adapter supplies its
positive magnitude to the voltage loop, so `ctrl.references.voltage` remains positive. The three
application examples use the same bridge simulation forms and solver path.

`InductorCurrentControl` is structure-independent. Its feedforward comes from the selected bridge:
`u_out/u_in` for the Buck connection, `1 - u_in/u_out` for the Boost connection,
`u_out/(u_in + u_out)` for the negative-output Buck-Boost connection (using the output magnitude),
and `(1 + u/Vdc)/2` for a full bridge. The loop itself is responsible only for current regulation
and limits.

A diode current is unilateral. The dedicated `DiodeConduction` assembly stage owns that device
behavior; it is not part of RK4, DP45 or another generic integrator. In `switching`, the stage uses
the selected solver to locate the natural current zero before the bridge is re-evaluated, so the
discharge interval ends at the physical boundary rather than at the end of a numerical step.
Projection remains only a numerical safety net. A fully controlled `HalfBridge` remains capable of
reverse current and does not receive this boundary handling.

The averaged forms store cycle-mean filter current, which remains positive in DCM even though the
instantaneous current has a zero-current interval. They derive the controlled, diode and idle
fractions from duty, carrier period, mean current and the upper, lower and filter-reference
voltages. The diode law belongs to the half bridge while the R-L state remains in the external
filter branch. One explicit algebraic conduction subsystem supplies the branch derivative, both
rail-current transfer ratios, the copper-loss factor and the local step request. Consequently the
averaged DCM conversion ratio and DC-terminal current are not forced to their CCM duty-ratio
relation, Python and exported
C++ use the same loss equation, and no output capacitor or load is hidden inside the converter
Unit.

The cycle-mean DCM state does not have the ordinary instantaneous `L i^2 / 2` storage function.
Energy reporting therefore keeps the physical port power and copper loss but flags strict energy
auditing when this averaged conduction relation is present instead of reporting false closure.

For either averaged form, the diode-conduction stage requests a local step of at most one tenth of
the physical switching period while the operating point is in DCM. In CCM it imposes no extra
limit, so the configured solver step remains in force. This resolution limit creates no PWM or
carrier events and applies equally to fixed, adaptive and multirate solvers. A subsystem with a
state-dependent local limit cannot be placed in a window-averaged group because that group cannot
resolve its conduction transition.

## Switching

`switching` evaluates the carrier comparison and changes the ideal bridge state at the exact edge
time. Every edge becomes an integration boundary, so solver stages before and after it use the
correct bridge state.

Use switching when the study needs:

- switching ripple or harmonics;
- carrier phase and synchronization effects;
- exact device-state transitions;
- fast protection behavior tied to switching or register-load instants.

The fixed step is a maximum step: PESLite shortens an interval when an event, output point,
controller interrupt or PWM edge occurs first. A smaller `simulation.solver.dt` improves the
continuous-state integration between those boundaries, at additional cost.

## PWM-period averaging

`pwm_averaging` removes the carrier waveform from the electrical bridge, but it does not remove the
digital control system. The controller still samples through the ADC, computes on its interrupt
grid and writes a shadow duty register. The active register loads at the configured carrier
instants, and the bridge holds that active duty until the next load.

This model therefore retains:

- controller sampling and computation delay;
- active and shadow PWM-register semantics;
- single or double PWM updates;
- asynchronous or synchronous timer placement;
- startup, protection and event timing.

Use it when digital timing matters but switching ripple does not. It is generally the best default
for controller and system-level studies with sampled control.

## Ideal averaging

`averaging` represents the bridge as an ideal controlled voltage source. Measurements and
controller outputs are evaluated at solver stages, and controller states join the continuous plant
state vector.

There is no ADC window, controller-interrupt grid, computation wait, carrier, or active/shadow PWM
register in this mode. Sampled-only settings may remain in a shared file and are silently ignored,
which allows the same case to be run under any of the three presets.

Use ideal averaging for:

- control-envelope and electromechanical dynamics;
- long simulations where switching detail is outside the study bandwidth;
- adaptive integration with error tolerances;
- fast parameter sweeps where continuous control is the intended model.

Do not use it as a substitute for switching when conclusions depend on ripple, carrier timing or
sampled-data delay. Continuous feedback can also create algebraic loops; PESLite detects connected
feedback variables and solves them internally at each derivative evaluation.

## Command presets

The three command-line flags are configuration presets rather than simple display options:

| Preset | Bridge model | Solver preset |
|---|---|---|
| `--switching` | `switching` for every unit | fixed-step RK4 |
| `--pwm-averaging` | `pwm_averaging` for every unit | fixed-step RK4 |
| `--averaging` | `averaging` for every unit | adaptive DP45 |

An explicit `--set` has higher priority. For example, a run can select the averaging preset while
overriding its solver choice:

```bash
peslite case.pes --averaging \
  --set simulation.solver.type=fixed \
  --set simulation.solver.method=rk4
```

Without a preset, each unit keeps the model written in the file, so mixed-model systems are
supported.

## Accuracy and comparison practice

Compare models on quantities within their common bandwidth and use identical events, references,
initial values and output times. A useful workflow is:

1. establish the control behavior with ideal averaging;
2. repeat with PWM-period averaging to include sampled control and register timing;
3. use switching for the intervals where ripple and edge behavior must be verified;
4. reduce solver step or tolerances until the conclusion is insensitive to further refinement.

The output interval controls saved snapshots, not the integrator step. A visually smooth CSV does
not by itself demonstrate numerical convergence.

## Selection summary

| Question | Recommended starting model |
|---|---|
| Switching harmonics, ripple or carrier interaction | `switching` |
| Sampled controller and PWM delay without ripple | `pwm_averaging` |
| Continuous control and long-duration dynamics | `averaging` |
| Mixed-detail network | Select the model independently for each unit |

For initial setup and command priority, see [Getting Started](Getting-Started.md).
