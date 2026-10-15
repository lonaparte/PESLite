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
├── components/   network, converter, ADC and PWM hardware
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
`addons/functions` boundary. IEEE-style plotting reads closed result CSV files after a run and is
absent from the solver path and generated C++ source.

Loading a simulation file also merges a sibling `PESaddons/` tree into these three package paths.
Its controller and component modules register before parameter-schema resolution; its function
modules remain lazy. The merge is a construction-time project extension and adds no work to the
simulation loop. `PESaddons` is anchored to the simulation file, while the CLI's default `output/`
directory is anchored to the current working directory.

## Components

`components/network.py` contains grid sources, R-L branches, bus R-C nodes and registered element
types. `components/converter.py` provides the common lossless bridge boundary, three bridge
implementations and DC-link models. `components/adc.py` and `components/pwm.py` implement sampled
converter peripherals.

Physical component state and parameters are SI quantities. Components declare ports, state,
derivatives and energy/storage relationships without knowing the global system order.

Every physical network attachment uses the same construction-time `Terminal` contract. A terminal
names its voltage and current once, with positive direction defined into the component; a one-port
component binds `bus`, while a branch or other two-port component binds `bus1` and `bus2`. System
assembly lowers this metadata to direct model connections before integration begins.

## Control

Each loop owns its parameter schema, typed ports, state and sampled/continuous update. `ControlGraph`
checks wiring and creates an execution plan. Sampled graphs require an explicit delay to break a
cycle; continuous graphs identify strongly connected loop groups and solve only those cyclic
groups iteratively.

The graph has one typed, directed boundary. Its input ports are the measured feedback signals
(`meas.*`) and controller references (`references.*`); its output ports are `u_dq`, `theta` and
`omega`. A connection always binds a loop input to a boundary or loop output. Default GFL/GFM
wiring and explicit custom wiring are both lowered to this same graph before the run.

Control ports use one canonical set of signal representations: complex vectors `I_AB`, `V_AB`,
`I_DQ`, `V_DQ` and `PQ`; scalar `I`, `V` and `POWER`; and scalar `ANGLE` and `FREQUENCY`.

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
`DCConverter` also serves as each full-bridge port of a resonant link. Each runtime specialization
owns its power stage, controller, sensing, startup interface, breakers and gates.

Every entry in `Params.units` is a `UnitParams`. Its `type` selects `vsc`, `dc_converter` or the
composite `wpt`; type-specific schema selection and runtime construction happen once while the
system is assembled. A WPT configuration contains `primary` and `secondary` UnitParams plus one
shared compensated link. Assembly expands that tree once into two ordinary runtime units and a
link subsystem. The solver sees only the flat result, so its event loop has no composite recursion
or WPT-specific scheduling. `Params.runtime_units` exposes the same flat leaf-unit mapping.

For compatibility with version 0.3.0 files, an omitted type means `vsc`; DC converters always use
the explicit `dc_converter` type.

Parameter ownership is explicit:

| Scope | Selector | Fields interpreted in that scope |
|---|---|---|
| Common unit | none | `pwm`, `ctrl`, `bridge`, `s_base`, resolved `base`, and unit events |
| VSC only | `unit.type: vsc` | `bus`, `ac_filter`, `dclink`, `meas`, and `protection` |
| DC converter only | `unit.type: dc_converter` | `bus`, `filter`, and `dclink`; a composite parent may bind the power port internally instead of naming a bus |
| WPT composite only | `unit.type: wpt` | `primary`, `secondary`, and `link`; it has no bridge, PWM or controller of its own |
| Bridge | `bridge.type` | `three_phase_bridge`, `half_bridge`, or `full_bridge`; `model` independently selects switching, PWM averaging, or ideal averaging |
| Common control | none | `type`, `period`, `computation`, `loops`, `connections`, and `outputs` |
| Control architecture | `ctrl.type` | reference fields and default wiring unique to the selected architecture |
| Control loop | `ctrl.loops.<name>.type` | gains, limits, and other fields unique to the selected loop |

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
- multirate window boundaries.

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
