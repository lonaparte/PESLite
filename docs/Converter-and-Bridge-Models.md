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
`buses.<name>.phases: 1`; the same interface also permits a three-phase point-network connection,
where the scalar bridge acts on the real stationary axis and has no synchronisation law. Such an
unsynchronised case is intentionally possible, though it is not a normal grid-control strategy.
The bus capacitor, load and any further branches remain independent network components. Buck,
Boost, resonant conversion and similar names describe assembled applications rather than
additional bridge structures.

A half bridge independently declares whether its upper and lower positions are controlled. For a
diode-bounded Buck application, the upper position is controlled and the lower position is a
diode. A fully controlled half bridge and a full bridge remain bidirectional.

`InductorCurrentControl` is structure-independent. Its feedforward comes from the selected bridge:
`u/Vdc` for a half bridge and `(1 + u/Vdc)/2` for a full bridge. The loop itself is responsible only
for current regulation and limits.

A diode current is unilateral. The dedicated `DiodeConduction` assembly stage owns that device
behavior; it is not part of RK4, DP45 or another generic integrator. In `switching`, the stage uses
the selected solver to locate the natural current zero before the bridge is re-evaluated, so the
discharge interval ends at the physical boundary rather than at the end of a numerical step.
Projection remains only a numerical safety net. A fully controlled `HalfBridge` remains capable of
reverse current and does not receive this boundary handling.

The averaged forms store cycle-mean filter current, which remains positive in DCM even though the
instantaneous current has a zero-current interval. They derive the controlled, diode and idle
fractions from duty, carrier period, mean current and the external terminal voltage. The diode law
belongs to the half bridge while the R-L state remains in the external filter branch. Consequently
the averaged DCM conversion ratio is not forced to the CCM duty-ratio relation, and no output
capacitor or load is hidden inside the converter Unit.

For a sampled averaged diode-bounded half bridge, the same diode-conduction stage requests finer
local resolution only near the DCM boundary. CCM retains the configured solver step; continuous
averaging relies on the configured adaptive solver in the same way as other continuous models.

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
