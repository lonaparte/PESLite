# Extending PESLite

PESLite uses registries and small typed protocols for custom control loops, circuit elements and
events. Installed modules under `peslite/addons/` and project-local modules under a `PESaddons/`
directory beside a simulation file share the same package search paths and registries. Types kept
elsewhere can still be imported and registered explicitly before loading a simulation file that
names them.

The root `examples/` directory contains simulation files only. Bundled Python implementations live
under `peslite.addons`, while additional test-only implementations exercise the same interfaces in
the test suite.

Project-local controllers, components and functions can live beside a simulation rather than in
the installed package. Their exact search rules and their relationship to the output directory are
defined once in [Project Workspace](Project-Workspace.md).

The bundled `custom-pll-example` demonstrates this layout: its `.pes` file remains in `examples/`,
while `peslite.addons.control.voltage_adaptive_pll` contains the executable custom loop. It
uses a filtered measured-voltage magnitude and grid-frequency state, then joins the same GFL graph
as the built-in current and DC-voltage loops.

The bundled `cable-gfl-example` similarly keeps its case in `examples/`, while
`peslite.addons.components.cable` implements its own distributed R-L/G-C state equations. It is a
two-terminal custom element rather than a renamed built-in branch: both terminal currents are
positive into the cable, and its internal inductor currents and capacitor voltages are ordinary
restartable PESLite states.

## Custom control loops

Register a control-loop class with `register_loop_type`:

```python
from dataclasses import dataclass
from peslite.addons.control import SyncLaw, register_loop_type

@register_loop_type
class LaggedSync(SyncLaw):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float
        k_p_pu: float
        tau: float
        type: str = "lagged_sync"

    type = "lagged_sync"
    state_names = {"theta": "theta", "p_f_pu": "p_f"}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.p_f = 0.0

    def _sample(self, period, p_pu, q_pu, v_mag_pu, v_dc_pu,
                p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.p_f += period / self.cfg.tau * (p_pu - self.p_f)
        self.omega = self.w0 + self.cfg.k_p_pu * (p_ref_pu - self.p_f)
        self.theta += period * self.omega
        self.v_mag = v_ref_pu

    def _flow(self, p_pu, q_pu, v_mag_pu, v_dc_pu,
              p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0 + self.cfg.k_p_pu * (p_ref_pu - self.p_f)
        self.v_mag = v_ref_pu
        return self.omega, (p_pu - self.p_f) / self.cfg.tau
```

The nested frozen `Params` dataclass defines the file schema and defaults. A module stored under
`peslite/addons/control/` is discovered automatically; the class name is then available directly
under `units.<u>.ctrl.loops` exactly like a built-in loop:

```yaml
loops:
  LaggedPSC: {k_p_pu: 6.2832, tau: 0.02}
```

When one graph needs the same class more than once, append a graph-local label without adding a
second type selector, for example `Observer[voltage]` and `Observer[current]`.

A specialized loop may expose separate sampled and continuous-flow laws. When its dynamics can be
built from the standard blocks, define one `equation()` instead: the graph supplies its inputs,
collects the registered states and derivatives, and selects sampled or continuous block
implementations during construction. Typed ports still validate all connections before running.

The essential pattern is:

```python
from peslite.addons.control import Filter, Integrator, PI

def __init__(self, cfg, unit, startup):
    super().__init__(cfg, unit, startup)
    self.angle = self.state_block(
        "theta", Integrator(self.block_period, initial=0.0)
    )
    self.regulator = self.state_block(
        "integral_pu", PI(kp, ki, self.block_period)
    )
    self.feedback = self.state_block(
        "feedback_pu",
        Filter(num=(1.0,), den=(tau, 1.0), period=self.block_period),
    )

def equation(self, signal, omega, reference, feedforward):
    measured = self.feedback(signal)
    command = self.regulator(reference, measured, feedforward)
    theta = self.angle(omega)
    return command, theta
```

`Loop.block_period` is the configured loop period for sampled bridge models and `None` for ideal
averaging. Construction therefore selects a sampled or continuous implementation once; the call
itself has no mode branch. `Filter` follows the Transfer Fcn coefficient convention: numerator and
denominator coefficients are in descending powers of `s`, and the transfer function must be
proper. Its sampled implementation, and the sampled integral branch of `Integrator` and `PI`, use
the trapezoidal (Tustin) method. The complete bundled example is
`peslite.addons.control.voltage_adaptive_pll`.

## Custom circuit elements

An element class registered by `register_element_type` owns:

- a frozen `Params` dataclass including its `type` and `bus` or `busN` fields;
- `subsystems()` for its physical model objects;
- `connections()` for its internal and auxiliary signal wiring;
- `terminals()` binding each configured bus to an electrical `Terminal`;
- optional `signals()` returning namespaced scalar or complex values for `plant.csv`;
- optional `connect()`, `disconnect()` or `retune()` behavior.

```python
from peslite.addons.components import Element, Terminal, register_element_type

@register_element_type
class MyElement(Element):
    type = "my_element"
    # Params and assembly methods...
```

The file can then use:

```yaml
elements:
  device: {type: my_element, bus: pcc}
```

Modules under `peslite/addons/components/` are discovered automatically. Built-in and add-on
elements follow the same registry mechanism. Public names are namespaced by the element instance,
so two instances do not share state paths.

An electrical `Terminal` names a subsystem voltage and current once. Its direction is `+1` when
positive current enters the subsystem and `-1` when it leaves. The default terminal receives
voltage through `inp` and publishes current through `out`; `drives_voltage=True` reverses those
roles. A one-terminal element uses `bus` and returns one binding; a multi-terminal element uses
`bus1`, `bus2`, and so on. Each bus needs exactly one voltage driver. System assembly converts the
voltage fan-out and signed current sum to direct model connections, so terminal metadata adds no
work to the integration loop.

Terminals can also declare circuit constraints. A voltage-driving terminal with
`hold_voltage_input="u_hold"` receives the voltage that keeps its connected current constant
while it is blocked. Its current-owning peer implements
`hold_voltage_source(terminal, *, current_constraint=None)` and
returns the ordinary model output `(subsystem, output_name)` carrying that voltage. Its owner
may also supply `current_constraint()`, returning its constrained current or `None` when free.
The current-owning peer receives this callback and installs `project_state()` to enforce active
constraints on its own states; coupled constraints must be projected together.
Assembly checks these capabilities after resolving default and named Unit ports; they apply to
ordinary branches, composite links and add-on elements alike.

Configured element buses may also name an assembled Unit default or named port. An optional
`series_connections()` mapping `{downstream_bus: inlet_bus}` inserts an element after the original
voltage driver; its ordinary `terminals()` still declares both electrical boundaries. This is a
construction-time operation, not a new network solver. The built-in `Resistor(series=True)` is a
stateless reference implementation.

An optional `parameter_bindings()` mapping `{field: (subsystem, attribute)}` exposes retunable
scalar parameters to C++ event lowering. Implement `retune()` using the same bindings so Python
and native output/energy equations share one parameter definition. Structural changes, such as
changing the element's series connection, remain assembly-time configuration. Mark structural
parameter fields with dataclass metadata `{"runtime": False}` to reject their `set` events.

Every entry returned by `signals()` follows the same output path. Complex values become three
phase columns with `_a`, `_b` and `_c` suffixes, while scalar values retain their declared name;
custom signal names do not require recorder-specific code.

The cable add-on is a complete multi-terminal reference implementation:

```yaml
elements:
  cable1:
    type: cable
    bus1: grid_bus
    bus2: pcc
    length_km: 5.0
    r_per_km: 0.05
    l_per_km: 0.4e-3
    c_per_km: 0.2e-6
    g_per_km: 1.0e-8
    sections: 2
```

Its per-kilometre parameters are SI values. `sections` changes the spatial discretisation and
therefore the state layout, so it is fixed when the simulation is assembled.

## Custom events

Register an `Event` subclass with `register_event_type`. Its `Params` dataclass contains at least
`type` and `t`. Its pure `variables(event, params)` function declares the existing variables to
write; custom events use the same validation and runtime update path as built-in events:

```python
from dataclasses import dataclass
from peslite.assembly import Event, register_event_type

@register_event_type
class LoadStep(Event):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        t: float
        target: str
        factor: float
        type: str = "load_step"

    type = "load_step"

    @staticmethod
    def variables(event, params):
        return {f"elements.{event.target}.r": params.elements[event.target].r * event.factor}
```

Bindings are evaluated against the preceding parameter state during assembly. The named element
must support the normal `retune(parameters)` interface. A custom connection event can instead
bind a boolean to `<section>.<object>.connected`; optional `ramp` has the same meaning as for
`connect`. Empty bindings and action-only `apply()` callbacks are rejected. Event callbacks must
not perform I/O or mutate the model while bindings are being resolved.

## Custom solvers

A user solver is a callable receiving `(rhs, t0, t1, y0)` and returning `SolverStep`. It owns its
RHS count and must finish exactly at `t1`. Optional hooks are:

- `settle(t, y)`: finish deferred work before a model-changing event;
- `parameters_changed()`: refresh cached data after a `set` event;
- named state methods when the solver has continuation state.

Pass the object to `Simulation(params, solver=solver)`. A simple single-rate solver does not need
the optional hooks.

## Custom controllers and modulators

`VSCController` builds a three-phase controller from a parameter tree, PWM method and output stage. A replacement
controller satisfies the `Controller` protocol; a replacement modulator satisfies `Modulator` and
returns a `SwitchingSequence`. The unit checks these protocols at assembly time.

An optional controller `initialize()` hook applies queued initial parameter updates before the
startup command and named-state restoration. The built-in graph controllers use their existing
retuning path here; later updates still take effect at the next control update.

Keep protection decisions in the unit hardware layer. Controller output contains commands and
startup status; it should not directly open breakers or mutate the system.

## Validation and state rules

- Validate custom scalar relationships in `Params.__post_init__` by raising `ConfigError`.
- Declare stable public state names so restart and CSV output remain deterministic.
- Preserve named state when rebuilding a retuned object.
- Keep structural changes out of `set` events; expose only parameters that can be applied safely.
- Use SI inside physical components and explicit per-unit conversion at controller boundaries.

## C++ export

Pure numerical custom subsystems with a fixed `state`/`inp`/`out` layout have an automatic C++
fallback. During export, PESLite symbolically traces their scalar `set_outputs()` and `rhs()`
equations, specialises configuration branches and fixed loops, and emits static C++. The cable
add-on is exported through this fallback; it does not carry a second handwritten C++ equation.
Built-in modules still use dedicated lowering, which covers their complete event and timing
semantics and gives the generator more opportunities for deliberate optimisation.

The fallback does not embed Python in the binary. Dynamic Python behavior outside a pure numerical
equation—such as I/O, reflection, changing the state layout or an unsupported library call—is
rejected. Custom loops, events and modulators that alter discrete scheduling still need explicit
backend support rather than being approximated.
