# Architecture

PESLite separates physical components, controller logic, system assembly, numerical execution and
optional add-ons. The five core directories keep ownership explicit while allowing the same
configuration to run with different bridge and solver implementations.

## Interface versioning

PESLite versions its Python, command-line and simulation-file interfaces with the software release.
Every bundled simulation example declares its target release in the first comment line. Before
version 1.0, any `0.x.x` release may change an interface; a project that depends on an exact
interface must use the matching PESLite version.

## Package layers

```text
src/peslite/
├── components/   network, converter, WPT link, ADC and PWM hardware
├── control/      control loops, graphs, startup and modulation
├── assembly/     parameters, events, protection, units, systems and exporters
├── solver/       connected model kernel, energy model, integrators and simulation run
└── addons/       custom extension entry points and optional functions
    ├── control/      automatically discovered custom control-loop modules
    ├── components/   automatically discovered custom circuit-element modules
    └── functions/    optional user-facing functions such as IEEE PDF plotting
```

Dependencies point inward through small interfaces: components use the model kernel, controller
logic uses typed signals, assembly wires both into a system, and `solver.simulation` schedules the
assembled system.

Controller and component modules in their respective add-on packages are imported once at package
initialisation. Their decorators populate the same registries used by built-in types, so there is
no add-on lookup or dispatch in the simulation hot path. Optional dependencies stay behind the
`addons/functions` boundary; [plotting](Plotting.md) is loaded only when requested.
The optional online stability analysis functions also live in `addons/functions`. They consume
selected observations during a run and share the solver-side loop-map linearization. The runner
exposes an optional observation callback on its existing event grid; core time stepping never
imports the analysis implementation, and the CLI loads it only with `--report`. Split-error
diagnostics use the same finite-difference calculation without depending on the addon. See
[Online Stability Analysis](Online-Stability-Analysis.md).

Loading a simulation file also merges a sibling `PESaddons/` tree into these three package paths
without adding run-loop work; see [Project Workspace](Project-Workspace.md).

## Components

`components/network.py` contains grid sources, R-L branches, bus R-C nodes and registered element
types. `components/converter.py` provides the common lossless bridge boundary, bridge topologies,
simulation models and DC-link models. `components/wpt.py` provides compensated WPT links and
switch-controlled capacitors. `components/adc.py` and `components/pwm.py` implement sampled
converter peripherals.

Physical component state and parameters are SI quantities. Components declare ports, state,
derivatives and energy/storage relationships without knowing the global system order.

Every physical network attachment uses the construction-time `Terminal` contract. A terminal names
its voltage and current, declares whether positive current enters or leaves the component and
identifies whether it drives or follows voltage; assembly lowers those declarations to direct
model connections. See
[Extending PESLite](Extending-PESLite.md#custom-circuit-elements).

## Control

Each loop owns its parameter schema, typed ports, state and sampled/continuous update. `ControlGraph`
checks wiring and creates an execution plan. Sampled graphs require an explicit delay to break a
cycle; continuous graphs identify strongly connected loop groups and solve only those cyclic
groups iteratively.

The graph has one typed, directed boundary. Its input ports are measured feedback signals
(`meas.*`) and controller references (`references.*`). Its output schema belongs to that boundary:
AC control exposes `u_dq`, `theta` and `omega`, while scalar bridge control exposes its modulator
registers, such as `duty` (see [Bridge and PWM](Bridge-and-PWM.md)). A
connection always binds a loop input to a boundary or loop output. Default wiring and explicit
custom wiring are both lowered to this same graph before the run.

Control ports use one canonical set of signal representations: complex vectors `I_AB`, `V_AB`,
`I_DQ` and `V_DQ`; scalar `I`, `V` and `POWER`; and scalar `ANGLE` and `FREQUENCY`.

`Startup` owns run/stop and reference-ramp progress. The modulation output stage converts controller
voltage commands into duty ratios. Protection does not belong to the controller.

## Assembly

The assembly layer:

1. parses strict parameter dataclasses and converts per-unit inputs to SI;
2. validates topology, timing and cross-section constraints;
3. creates buses, branches, sources, elements and converter units;
4. registers globally unique entity-first state and signal names;
5. compiles model connection and output-evaluation plans;
6. builds event scenarios and validated parameter changes.

`Unit` is the common converter lifecycle and runtime interface. Both `VSC` and `DCConverter` expose
their electrical side through the same named-bus `Terminal` contract; `VSC` supplies a three-phase
bridge and `DCConverter` a selectable scalar half or full bridge. The same
`DCConverter` also serves as each full-bridge port of a resonant link. Each assembled specialization
owns its power stage, controller, sensing, startup interface, breakers and gates.

Construction assembles these objects without issuing connection commands. Before loading named
control state, the connection owner initializes its source and controller command once from the
event timeline. After state loading, cold starts track the physical inputs and restore bridge and
ADC clock state; restarts preserve the supplied control state. Cold-start and reconnect tracking
use the same loop `track()` callbacks in Python and generated C++, including custom loops.
Parameter selection does not change this initialization sequence.

The contract also assembles two internal power boundaries. `<unit>.bus.dc` connects a DCLink
voltage driver to the bridge DC terminal. `<unit>.bus.power` always denotes the bridge power-side
bus: it connects the bridge to a series filter or, for an unfiltered nested converter, directly to
a WPT link. Normally, a filter's other terminal connects to the public bus selected by the unit's
`bus` parameter. For a half bridge with `external_terminal: lower_rail`, that public bus instead
connects to the lower rail, while the filter connects the midpoint power bus to the common point.
This is the single-inductor Buck-Boost connection. The scoped internal names disappear into direct
connections; there is no runtime bus lookup or WPT-specific solver path. Every Unit exposes named
ports and one default port. A single-port Unit selects it automatically; a multi-port Unit type
must select one explicitly.
Composite ports forward existing child ports without copying their electrical base. WPT exposes
`primary` and `secondary` and selects `primary` by default, so common assembly, event scaling and
solver ratings never interpret those WPT-specific names.

A source needs no internal bus. Its `<source>.emf` is wired directly to `<source>.branch`, whose
second terminal attaches to the configured public bus.

Every entry in `Params.units` is a `UnitParams`. Its `type` selects `vsc`, `dc_converter` or the
composite `wpt`; type-specific schema selection and runtime construction happen once while the
system is assembled. A WPT configuration contains `primary` and `secondary` UnitParams plus one
shared compensated link. `Params.units` is the only stored configuration tree. Assembly walks it
and places its leaf converters in `System.units` beside the shared link subsystem; it does not keep
a second flattened parameter mapping or a group table. The solver therefore sees ordinary
converter units without WPT-specific scheduling.

For compatibility with version 0.3.0 files, an omitted type means `vsc`; DC converters always use
the explicit `dc_converter` type.

Parameter ownership is explicit: all converter units share common fields; `unit.type` adds
electrical fields, `bridge.type` topology fields, `ctrl.type` reference/default-wiring fields, and
each `ctrl.loops` key its class parameters. See
[Simulation Files](Simulation-Files.md#si-and-per-unit-values).

Every `ctrl` section is represented by the same `ControlParams` type. Its discriminator narrows the
valid fields inside that object; it never switches to another public parameter container. The unit
discriminator does not select or interpret controller-specific fields. For example, a DC converter
declares both `type: dc_converter` in its unit scope and `type: dc` in its `ctrl` scope.

## Model kernel

`solver/model.py` packs subsystem state into one flat real vector, using two entries for a complex
state. The model compiles direct connection and output plans during construction; ordinary RHS
evaluation does not repeatedly merge arbitrary state dictionaries.

Port dependency analysis identifies algebraic groups. Acyclic outputs use one pass, while only
connected feedback variables are solved iteratively. The same mechanism supports ideal averaging
without manual case-specific loop breaks.

## Port-Hamiltonian information

Components can declare storage and power ports. The energy layer uses these declarations to:

- account for stored, supplied and dissipated energy;
- check Tellegen/power-balance residuals;
- produce the `--ph-report` structure view;
- evaluate multirate interface and hold diagnostics.

The port-Hamiltonian kernel is a structural and diagnostic layer; it does not force every solver to
use one integration method.

## Simulation execution flow

`Simulation.run()` repeatedly selects the next exact boundary among:

- requested end and output times;
- file events;
- converter ADC, control, PWM load and switching instants;
- continuous-unit measurement/protection clocks;
- observation times requested by analyses such as the online stability report.

Multirate windows advance inside the solver and settle before model-changing events.

The solver advances only to that boundary. The simulation then settles deferred work, applies file
events, evaluates fast protection, samples every due unit, actuates every unit and records output.
This ordering gives all units the same pre-actuation view at a shared instant.

## State and continuation

State ownership is fixed during construction. The public registry combines physical, controller,
ADC/PWM, protection and solver state into unique entity-first paths. Output rows and restart use the
same schema, while reconstructible timer indices and combinational outputs are derived from saved
time/state rather than stored redundantly.

## C++ generation

The C++ backend consumes the already assembled `Simulation`, lowers its fixed model/output plans to
straight-line C++ and specializes fixed configuration values. The result is one translation unit;
runtime-variable parameters become typed fields, not a generic dictionary in the RHS path.

See [C++ Export](Cpp-Export.md) for supported solvers and customization limits.
