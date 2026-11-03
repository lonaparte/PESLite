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
two-port custom element rather than a renamed built-in branch: both port currents are
positive into the cable, and its internal inductor currents and capacitor voltages are ordinary
restartable PESLite states.

## Custom control loops

Register a control-loop class with `register_loop_type`:

```python
from dataclasses import dataclass
from typing import Optional

from peslite.addons.control import SyncLaw, register_loop_type

@register_loop_type
class LaggedSync(SyncLaw):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: Optional[float] = None
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

The nested frozen `Params` dataclass defines the file schema and defaults. Loop `type` and `period`
are structural; `set` events may write other fields except those declared with dataclass metadata
`{"runtime": False}`. Use that metadata for additional fields that add or remove states, as the
built-in loops do for `SRFPLL.normalisation` and the power-filter `enable`. A module stored under
`peslite/addons/control/` is discovered automatically; the class name is then available directly
under `units.<u>.ctrl.loops` exactly like a built-in loop:

```yaml
loops:
  LaggedSync: {k_p_pu: 6.2832, tau: 0.02}
```

Repeated uses of one class follow the loop naming rules in
[Bridge and PWM](Bridge-and-PWM.md#controller-and-loop-periods).

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

A custom electrical component is a frozen parameter dataclass registered by
`register_component_type`. Its `network()` method declares the physical primitives used by every
circuit model.

```python
from dataclasses import dataclass
from typing import ClassVar

from peslite.addons.components import register_component_type
from peslite.solver import netlist as nl

@register_component_type
@dataclass(frozen=True, kw_only=True)
class SeriesRL:
    bus: str
    r: float
    l: float
    type: str = "series_rl"
    domain_transparent: ClassVar[bool] = True

    def network(self, n):
        middle = n.internal()
        n.add(
            nl.Resistor(n.bus, middle, self.r),
            nl.Inductor(middle, n.ground, self.l, n.state("i")),
        )
        n.port_output("i", "bus", "current")
        n.losses()
```

The file can then use:

```yaml
elements:
  load: {type: series_rl, bus: pcc, r: 1.0, l: 2e-3}
```

Modules under `peslite/addons/components/` are discovered automatically. Built-in and add-on
components use the same registry and assembly-time translation. A `bus` field creates one port;
`bus1`, `bus2` and later numbered fields create multiple ports. Each port starts with zero voltage
and current, with current positive into the component. `n.bus` or `n.busN` is its network node,
`n.ground` is the reference node, and `n.internal()` inherits the component's electrical dimension.
`n.state("name")` registers a zero initial state; pass `initial=` only for a nonzero initial value.
Scalar ports create real states and three-phase ports create complex space-vector states.

Use `n.output`, `n.linear_output` or `n.port_output` to publish values. Ordinary scalar inductor and
capacitor storage is inferred from the declared primitives; `n.losses()` and `n.supplies()` select
the primitives used for energy accounting. Public names are namespaced by the configured component
name, so instances do not share state or output paths.

Configured element buses may also name an assembled Unit default or named port. An optional
`series_connections()` mapping `{bus: inlet_bus}` cuts a node that joins exactly two ports and
places the element between them. Repeated declarations form a series chain in declaration order.
The built-in `Resistor(series=True)` is a stateless reference implementation.

Structural changes remain assembly-time configuration. Mark a field with dataclass metadata
`{"runtime": False}` to reject its `set` events; `{"structural": True}` also excludes a numerical
field from native runtime parameters.

Every entry returned by `signals()` follows the same output path. Complex values become three
phase columns with `_a`, `_b` and `_c` suffixes, while scalar values retain their declared name;
custom signal names do not require recorder-specific code.

The cable add-on is a complete multi-port reference implementation:

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

Custom and built-in components use the same equation tracing; see
[What is specialized](Cpp-Export.md#what-is-specialized) and
[Current limitations](Cpp-Export.md#current-limitations).
