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

## Scalar DC converter device placement and DCM

The scalar Buck, Boost and Buck-Boost circuits configure the device positions through the existing
`HalfBridge` topology. Buck uses a controlled upper device and a lower diode; Boost and Buck-Boost
use a controlled lower device and an upper diode. The assembly does not replace that device
placement with a default fully controlled leg.

`InductorCurrentControl` is topology-independent. The scalar controller boundary derives the
steady-state duty feedforward from the configured converter topology and supplies it through a
typed control port, leaving the current loop responsible only for current regulation and limits.

A diode current is unilateral. The DC plant therefore constrains the inductor current to be
nonnegative: at zero current, a negative unconstrained derivative is replaced by zero, and any
numerical crossing is projected back onto the zero-current boundary. This produces the idle
interval of discontinuous-conduction mode (DCM). A fully controlled `HalfBridge` component remains
capable of reverse current and does not receive this constraint.

All three simulation forms enforce the same electrical current boundary. `switching` resolves the
individual charge, discharge and zero-current intervals; the averaged forms preserve the
low-frequency constrained trajectory but do not reproduce sub-cycle ripple.

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
