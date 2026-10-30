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

`HalfBridge`, `FullBridge` and `ThreePhaseBridge` describe circuit structure. For controlled
bridges, the following Python model classes implement the simulation form selected by `bridge.model`.

| Model | Python class | Electrical bridge | Digital timing | Typical cost |
|---|---|---|---|---|
| `switching` | `SwitchingModel` | Ideal switch states at exact carrier-comparison instants. | ADC, controller interrupts, computation completion and PWM register loads. | Highest. |
| `pwm_averaging` | `PWMAveragingModel` | Active duty ratios applied continuously between register loads; no carrier ripple. | Same sampled controller and PWM-register schedule as switching. | Lower than switching. |
| `averaging` | `AveragingModel` | Ideal controlled voltage source driven by continuous controller equations. | No ADC, controller-interrupt, computation or PWM-register schedule. | Usually lowest for control-envelope studies. |

The models answer different questions. Similar low-frequency trajectories do not make them
numerically identical.

| Feature | `switching` | `pwm_averaging` | `averaging` | Standalone C++ | Imperix |
|---|---:|---:|---:|---:|---:|
| Controlled half, full and three-phase bridges | yes | yes | yes | yes | sampled only |
| Square-wave phase-shift modulation | yes | no | no | yes | no |
| Diode full-bridge commutation | yes | yes, same natural-conduction law | yes, same natural-conduction law | yes | no |
| Single-phase AC network source | yes | yes | yes | yes | plant not exported |
| `open_loop` duty control | yes | yes | yes | yes | yes |
| `ModeSequence` add-on | yes | no | no | yes | no |

Backend entries describe export support, not whether the hardware executes the simulated plant.
Imperix exports sampled control and peripheral bindings; continuous `averaging` is not a hardware
control schedule. [Imperix ELF Export](Imperix-Export.md#timing-and-run-commands) lists what it
rejects.

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
carrier events and applies equally to fixed, adaptive and multirate solvers; see the windowed-group
restriction in [Multirate Simulation](Multirate-Simulation.md#limitations).

## Device resistance and port capacitance

Every bridge uses `bridge.on_resistance` (ohm per conducting device) and
`bridge.parallel_capacitance` (F). Both default to zero. Nonzero capacitance requires an explicit,
positive resistance; no artificial minimum resistance is inserted. A conducting half-bridge or
three-phase leg contains one device; a full-bridge current path contains two.

The capacitance connects the half-bridge midpoint to the electrical reference, the full bridge's
two AC terminals, or each three-phase terminal to a floating star point. It uses the existing
`RCNode` capacitor model and stores `bridge.u_C`. Its voltage stays continuous when gates or DC
voltages change. Switches and anti-parallel diodes share a finite-resistance leg law. The topology
eliminates common voltages using symmetry or the device law's piecewise-linear breakpoints;
only the ordinary external network feedback remains for the algebraic solver.
DC sources, capacitors and series impedances retain their ordinary connection interfaces.

Switching evaluates the active switching states. PWM-period and ideal averaging weight the same
device equations by the carrier-state durations, including charging currents and losses. They
require small intra-period capacitor-voltage ripple; use switching to resolve
individual charging pulses, commutation ringing and their losses. Choose solver tolerances and
step size to resolve the additional resistance-capacitance time constants. Zero capacitance
creates no capacitor state or neutral-point equation. Resistance alone adds the conducting-path
voltage drop and loss to the existing bridge model.
Three-phase quantities retain the zero-sequence-free space-vector convention; this is equivalent
port capacitance, not a model of individual device capacitances or common-mode transients.

```yaml
bridge:
  type: full_bridge
  model: switching
  on_resistance: 0.1
  parallel_capacitance: 4.7e-9
```

## Diode full-bridge rectifier

A `full_bridge` with `upper_controlled: 0` and `lower_controlled: 0` keeps only its four
anti-parallel diodes. The unit then has no controller or PWM timer. `FullBridge` binds its diode
equations to the common execution model. Its discrete state is a pair conducting the port current
`i_c` (positive out of the bridge) in either direction, or all diodes blocked. A conducting
pair opposes the current, `u_c = -sign(i_c) u_dc`, and passes `|i_c|` to the DC side.

Because this bridge has no commanded PWM waveform, `bridge.model` does not replace its conduction
law with a duty-ratio model. The `switching`, `pwm_averaging` and `averaging` run selections all
retain the same exactly located natural commutations; the selection still controls the solver
preset and any controlled bridges elsewhere in the system.

A blocked bridge holds its port current at the located zero by applying the port's *hold
voltage*: the voltage at which the port current has no rate of change. Its connected
current-owning terminal provides that law. Ordinary scalar R-L branches and compensated links
support it; no WPT-specific unit placement is required. On a series-compensated link it depends on
the other port's terminal voltage through the mutual inductance, so a series element between the
other converter and the link is accounted for. Blocking ends when the hold voltage reaches
`+-u_dc`; a conducting pair stops at its current zero, after which the opposite pair conducts
(continuous conduction) or all diodes block (discontinuous conduction).

With `parallel_capacitance`, the same finite-resistance device law used by controlled bridges
also handles rectification. The capacitor charges while the diodes are blocked and discharges
through their finite resistance when forward biased. DC voltage steps produce finite charging
currents. This representation uses continuous capacitor state instead of the ideal clamp's
discrete conduction state and does not require a stiff DC source.
Standalone C++ supports controllerless rectifiers, including their independent ADCs.

Only an explicit `type: full_bridge` can have no controlled device. A diode full-bridge rectifier has no
gates to block, so it accepts no protection criterion.

Measurements use the existing ADC independently of a controller, on the unit's own measurement
clock ([ADC sampling](Control-and-PWM-Timing.md#adc-sampling)); window means, RMS windows and
`meas.sources` use the same channel and state interfaces. Held means are published as
`<unit>.meas.<channel>_mean`, alongside the existing `<unit>.meas.<channel>_rms` signals.
Instantaneous conversions held by an independent ADC use `<unit>.meas.<channel>`.

A controlled bridge without port capacitance keeps its zero state when gates are blocked. On a port without a series
filter the unit reports once when such a blocked bridge carries current, because its
anti-parallel diodes would then conduct, which the zero state does not represent.

## Switching waveform updates

The common `pwm.pattern_sync` setting decides how a new switching waveform replaces the old one:

- `interval` (default): at the load. The modulator runs directly, with no pulse-preservation state.
- `pulse`: a pulse in progress at the load runs to its end under the old registers; the output
  then stays at zero until the next pulse of the new registers starts. No pulse is cut or
  shortened; the price is a zero gap of up to one period. The output before the first load is zero,
  so the first pulse after the gates are enabled is also whole. Clock-rate changes always act at the load.
  Carrier PWM preserves each pole's gate pulse independently. Phase-shift modulation preserves
  both poles as one bridge-output pulse. The common sequence processor keeps each accepted
  pulse's value and remaining phase in cycles; it does not copy controller commands or waveform equations.
  These are named states under `pwm.pulse.*`, so continuation reproduces an unfinished pulse.

This policy applies to asynchronous and synchronous carrier comparison as well as phase-shift
modulation, in Python and C++. Constant gate commands require no pulse boundary. Averaged bridge
models do not resolve switching pulses and do not construct this state. Omitting `pattern_sync`
and explicitly setting `pattern_sync: interval` select the same execution path.

Synchronous carriers retain the angle and frequency applied at the current load as
`pwm.angle` and `pwm.frequency`. These restore the waveform during a computation delay;
the next load takes its reference from the controller through the ordinary PWM timing path.


## Phase-shift (square-wave) modulation

`ctrl.type: phase_shift` commands a controlled `full_bridge` with square-wave pulses instead of
carrier-compared duty ratios. The controller's outputs are four compare registers, which pass the
same shadow/active path, computation delay and gate enable as duty ratios:

| Register | Meaning |
|---|---|
| `width` | pulse width in half periods, clipped to [0, 1] (1 is a full square wave) |
| `phase` | angle (rad) of the centre of the positive pulse on the bridge's angle axis |
| `frequency` | angle-axis frequency (Hz) |
| `mode` | 0 bipolar pulses (`+u_dc` and `-u_dc`), 1 positive pulse only, 3 zero level, 4 negative pulse only (centred half a period after `phase`) |

Extra scalar outputs wired in `ctrl.outputs` are carried as further registers on the same path.
This applies to every modulation profile: the signal type comes from the wired source, without
a device-specific output list. These registers do not shape the waveform; a connected component
reads their active values. For example, an SCC can read a carried `scc_angle` command. Register
names must be unique path segments and must not collide with modulator state names. Every
register must be finite; a non-finite controller output stops the run.

With half-width `h = width * pi / 2` the positive pulse covers `theta - phase` in `[-h, h)` and the
negative pulse `[pi - h, pi + h)`. The angle axis `theta` is continuous: each register load
continues it with the previously active frequency and then runs the newly loaded one, so the
frequency may differ from the PWM timer (`pwm.f_sw` sets the load and control instants) and may
change at every load without a phase jump. A fresh run starts the axis at `2 pi f t`; converters on
one time base therefore share it, and their `phase` registers set their relative phase directly.
The axis angle at the current load is a named state (`pwm.angle`) and is restored on continuation.


Edges within `1e-9` of a period after a load belong to that load; edges within the same distance
before the end of an interval belong to the next one, so a load that coincides with an edge neither
loses nor repeats it. The modulator requires resolved switching edges; the current
`pwm_averaging` and `averaging` forms accept duty ratios and cannot represent this pattern. Their
period mean can be zero while the switching-frequency fundamental carries the resonant power, so
use the switching model unless a separately validated harmonic approximation is available.
`pwm.sync` and `pwm.carrier_phase` are also rejected (the phase is a register). Standalone C++
supports both pattern modes, including register events, parameter overrides and restart state.

## Exact commutations

The `DiodeConduction` assembly stage also owns every *commutating* subsystem: one with a discrete
state that is valid while `commutation_margin()` is nonnegative. After each probe interval (the
fixed step `dt`, or `max_step` for an adaptive solver) the stage checks every margin
against the new state; a negative margin is located by a bracketed search on the solver's step map and
`commutate(t)` switches the state at that instant. Because validity rather than a sign change is
checked, a margin that an outer event makes negative at the start of an interval (for example the
open-circuit voltage of a series-compensated port at a bridge edge) is handled like one crossing
zero inside it. A subsystem may also name an exact scheduled instant through
`next_commutation_time()`; steps then end at that instant. `commutate` may reinitialize the
subsystem's own continuous states, which are written back into the state vector. The discrete
state is a named state and is restored on continuation. Bounded diode currents, margin crossings
and scheduled instants use one locator and boundary dispatcher. Due initial events and coincident
events are settled before measurement, control and output reads; each transition sees the effects
of preceding transitions at the same boundary.

The locator uses secant proposals with bisection safeguards and stops on the time bracket's width.
A small guard residual alone does not end the search, since a valid mode may have a zero-margin
plateau. The same search handles bounded currents and general commutation margins.

Switch-controlled capacitors on a compensated link use the same protocol: the link's margin covers
their body-diode turn-on (capacitor-voltage zero), current reversals and gate edges. Gate state
and scheduled edges use the same absolute timestamps and time tolerance as the dispatcher,
including a phase or frequency change at a register load. See
[Simulation Files](Simulation-Files.md#link-fields).

Commutating subsystems run with fixed-step and adaptive solvers. An adaptive solver needs a finite
`max_step`, so that how often the margins are checked does not depend on how far apart the outer
events (output, control, PWM) are; windowed multirate restrictions are listed in
[Multirate Simulation](Multirate-Simulation.md#limitations). Both are reported when the run starts.
A configuration that commutates repeatedly without time advancing raises `CommutationError`
instead of looping. Standalone C++ uses the same margin checks, safeguarded secant/bisection
locator, scheduled boundaries and state resets.

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
for controller and system-level studies with sampled control. It cannot represent square-wave
phase-shift modulation ([Phase-shift](#phase-shift-square-wave-modulation)).

## Ideal averaging

`averaging` represents the bridge as an ideal controlled voltage source. Measurements and
controller outputs are evaluated at solver stages, and controller states join the continuous plant
state vector.

There is no ADC window, controller-interrupt grid, computation wait, carrier, or active/shadow PWM
register in this mode. Which sampled settings it ignores is described in
[Control and PWM Timing](Control-and-PWM-Timing.md#ideal-averaging); the same case can therefore be
run under any of the three presets.

Use ideal averaging for:

- control-envelope and electromechanical dynamics;
- long simulations where switching detail is outside the study bandwidth;
- adaptive integration with error tolerances;
- fast parameter sweeps where continuous control is the intended model.

Do not use it as a substitute for switching when conclusions depend on ripple, carrier timing or
sampled-data delay. Continuous feedback can also create algebraic loops; PESLite detects connected
feedback variables and solves them internally at each derivative evaluation. Like PWM-period
averaging, it cannot represent square-wave phase-shift modulation.

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

Judge convergence by solver refinement, not CSV smoothness: the output interval does not set the
integration step.

## Selection summary

| Question | Recommended starting model |
|---|---|
| Switching harmonics, ripple or carrier interaction | `switching` |
| Sampled controller and PWM delay without ripple | `pwm_averaging` |
| Continuous control and long-duration dynamics | `averaging` |
| Mixed-detail network | Select the model independently for each unit |

For initial setup and command priority, see [Getting Started](Getting-Started.md).
