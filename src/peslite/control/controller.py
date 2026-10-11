"""A converter controller in sampled or continuous averaged form.

At each control interrupt the controller takes an ADC sample (:class:`Measurement`, SI), scales it to
the unit's pu bases, runs its loop network (:class:`ControlGraph`: the loops of :mod:`.loops`
connected by typed ports) and its start-up state, and turns the voltage command into duty
ratios. It sees only its samples, host commands and its own state. Protection and physical trip
actions belong to the unit.  The ideal averaging bridge uses the same graph as a continuous ODE;
the other bridge models run it as one closed sampled block.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..solver.model import Bag, ConfigError, gather, scatter
from .blocks import OutputLimit, abc2complex, peak_abs, phases, smoothstep
from .loops import (ANGLE, DUTY_RATIO, FREQUENCY, I, I_AB, LOOP_TYPES, POWER,
                    SI_CURRENT, SI_VOLTAGE, V, V_AB, V_DQ, CurrentLoop,
                    DCVoltageLoop, Loop, SignalType, SyncLaw)
from .modulation import CONFIGURED, OutputStage

__all__ = ["Measurement", "ControlMeasurement", "ControlOutput", "Controller", "Startup",
           "ControlInterface", "CONTROL_INTERFACE", "ControlGraph", "default_wiring",
           "UniteType", "ScalarControlOutput", "ScalarController",
           "make_controller", "make_scalar_controller"]


# ------------------------------------------------------------------ the interface

@dataclass
class Measurement:
    """SI sample at ``t``: peak-scaled alpha-beta AC vectors (V, A) and DC voltage (V).

    Windowed channels hold means over ``(t - T_avg, t]``; ``*_raw`` and ``i_abc`` are instantaneous.
    ``samples`` holds an oversampled update's samples, oldest first, this one last; otherwise empty.
    """

    t: float
    u_g: complex  # PCC voltage (V)
    i_c: complex  # converter current (A), positive out of the bridge
    u_dc: float  # dc-link voltage (V)
    i_abc: NDArray[np.float64]  # instantaneous phase currents (A)
    u_g_raw: Optional[complex] = None  # before window averaging
    i_c_raw: Optional[complex] = None
    u_dc_raw: Optional[float] = None
    extras: dict[str, float] = field(default_factory=dict)
    samples: tuple["Measurement", ...] = ()


@dataclass
class ControlMeasurement:
    """Controller-side sample in pu: alpha-beta AC vectors on peak phase bases, DC voltage on ``vdc_ref``.

    ``t`` is in s.
    """

    t: float
    u_g: complex
    i_c: complex
    u_dc: float
    i_abc: NDArray[np.float64]


@dataclass
class ControlOutput:
    """Result of one controller call.

    ``d_abc`` and ``gates`` go through the PWM registers. ``startup_complete`` reports that the
    controller's start-up ramp has finished; the owning unit decides how to use it.
    ``theta`` (rad) and ``omega`` (rad/s): the controller's synchronization angle and frequency,
    required by synchronous modulators; ``None`` allows only asynchronous modulation.
    """

    d_abc: NDArray[np.float64]  # duty ratios of phases a, b, c in [0, 1]
    log: dict[str, float] | None = None
    theta: float | None = None  # synchronization angle (rad)
    omega: float | None = None  # synchronization frequency (rad/s)
    gates: bool = True
    startup_complete: bool = True


@dataclass
class ScalarControlOutput:
    """Result of one scalar converter control update."""

    duty: float
    log: dict[str, float] | None = None
    theta: float | None = None
    omega: float | None = None
    gates: bool = True
    startup_complete: bool = True


@runtime_checkable
class Controller(Protocol):
    """Sampled controller ``(t, Measurement) -> ControlOutput``.

    Optional host-facing methods are ``command``, ``track``, ``retune``, state access and
    ``summary``; the required target interface is only the sampled call. Protection is deliberately
    outside this interface.
    """

    def __call__(self, t: float, meas: Measurement) -> ControlOutput: ...


@dataclass(frozen=True)
class ControlInterface:
    """Typed directed ports at the boundary of a :class:`ControlGraph`.

    ``inputs`` are sources available to loop input ports. ``outputs`` are controller output ports,
    each driven by one loop output through ``ctrl.outputs``. Direction belongs to the mapping that
    contains a port; its value is the same :class:`SignalType` used by loop port declarations.
    """

    inputs: Mapping[str, SignalType]
    outputs: Mapping[str, SignalType]


_OutputLimit = OutputLimit

# ------------------------------------------------------------------ the start-up sequence

class Startup:
    """Minimal run/ramp state advanced only by this controller's interrupts.

    Only the command, ramp duration and interrupt count are persistent. ``value`` and ``complete``
    describe the current controller call; applying the value to a reference belongs to the loop
    which owns that reference.
    """

    def __init__(self, T: float, run: bool = True, ramp: float = 0.0) -> None:
        self.T = float(T)
        self.run = bool(run)
        self.ramp = float(ramp)
        self.steps = 0
        self.value = 0.0
        self.complete = False

    def command(self, run: bool, ramp: float = 0.0) -> None:
        """Take a host run/stop command; every new run has its own ramp."""
        self.run = bool(run)
        if self.run:
            self.ramp = float(ramp)
            self.steps, self.value, self.complete = 0, 0.0, False
        else:
            self.steps, self.value, self.complete = 0, 0.0, False

    @property
    def active(self) -> bool:
        return self.run

    @property
    def in_progress(self) -> bool:
        return self.run and not self.complete

    def advance(self) -> None:
        """Advance one controller interrupt and expose its start-up multiplier."""
        if not self.in_progress:
            return
        progress = 1.0 if self.ramp <= 0.0 else min(1.0, self.steps * self.T / self.ramp)
        self.value = smoothstep(progress)
        self.complete = progress >= 1.0
        if not self.complete:
            self.steps += 1

    def get_state(self) -> dict[str, Any]:
        return {"run": self.run, "ramp": self.ramp, "steps": self.steps}

    def set_state(self, values: Mapping[str, Any]) -> None:
        unknown = set(values) - {"run", "ramp", "steps"}
        if unknown:
            raise KeyError(f"the start-up sequence has no state(s) {sorted(unknown)}")
        if "run" in values:
            self.run = bool(values["run"])
        if "ramp" in values:
            self.ramp = float(values["ramp"])
        if "steps" in values:
            steps = float(values["steps"])
            if steps < 0.0 or steps != int(steps):
                raise ValueError(f"the start-up sequence: steps must be a whole number >= 0, got {values['steps']}")
            self.steps = min(int(steps), self._end())

    def _end(self) -> int:
        if self.ramp <= 0.0:
            return 0
        n = max(0, math.ceil(self.ramp / self.T) - 2)
        while n * self.T / self.ramp < 1.0:
            n += 1
        return n


class ContinuousStartup:
    """Run/ramp command for a continuous controller, evaluated from absolute time."""

    def __init__(self, run: bool = True, ramp: float = 0.0) -> None:
        self.run = bool(run)
        self.ramp = float(ramp)
        self.start = 0.0
        self.value = 0.0
        self.complete = False

    @property
    def active(self) -> bool:
        return self.run

    @property
    def in_progress(self) -> bool:
        return self.run and not self.complete

    def command(self, run: bool, ramp: float = 0.0, t: float = 0.0) -> None:
        self.run, self.ramp, self.start = bool(run), float(ramp), float(t)
        self.value = 0.0
        self.complete = False

    def at(self, t: float) -> None:
        if not self.run:
            self.value, self.complete = 0.0, False
            return
        progress = 1.0 if self.ramp <= 0.0 else min(1.0, max(0.0, (t - self.start) / self.ramp))
        self.value = smoothstep(progress)
        self.complete = progress >= 1.0

    def get_state(self) -> dict[str, Any]:
        return {"run": self.run, "ramp": self.ramp, "start": self.start}

    def set_state(self, values: Mapping[str, Any]) -> None:
        unknown = set(values) - {"run", "ramp", "start"}
        if unknown:
            raise KeyError(f"the continuous start-up sequence has no state(s) {sorted(unknown)}")
        if "run" in values:
            self.run = bool(values["run"])
        if "ramp" in values:
            self.ramp = float(values["ramp"])
        if "start" in values:
            self.start = float(values["start"])


# ------------------------------------------------------------------ the loop network

# kinds of input source
_HELD, _MEASURED, _CONSTANT = 0, 1, 2

CONTROL_INTERFACE = ControlInterface(
    inputs=MappingProxyType({
        "meas.u_g": V_AB,
        "meas.i_c": I_AB,
        "meas.u_dc": V,
        "references.id_ref_pu": I,
        "references.iq_ref_pu": I,
        "references.theta": ANGLE,
        "references.omega": FREQUENCY,
        "references.p_ref_pu": POWER,
        "references.q_ref_pu": POWER,
        "references.v_ref_pu": V,
        "references.vdc_ref_pu": V,
        "references.zero_v_pu": V_DQ,
    }),
    outputs=MappingProxyType({"u_dq": V_DQ, "theta": ANGLE, "omega": FREQUENCY}),
)


class ControlGraph:
    """The loops of ``cfg.ctrl.loops`` wired by typed ports and run by the control interrupt.

    Loop inputs are held outputs of other loops, the pu measurement (``meas.u_g``, ``.i_c``,
    ``.u_dc``) or the references (``references.<name>``); ``connections`` and ``outputs`` are the
    default wiring, which ``cfg.ctrl.connections`` and ``.outputs`` override. Loops run in signal
    order, every interrupt or every n-th interrupt according to their periods.

    Retuned parameters are queued by :meth:`schedule` and take effect at the next update. Changed
    loops are rebuilt from their new parameters and continue from their named states.
    """

    def __init__(self, cfg, startup, connections, outputs, *, interface=CONTROL_INTERFACE):
        self.cfg, self.startup = cfg, startup
        self.interface = interface
        self._pending: list[Any] = []
        self.on_retune = None
        self.connections = {**connections, **cfg.ctrl.connections}
        self.outputs = {**outputs, **cfg.ctrl.outputs}
        self.nodes = {}
        self.periods = {}
        self.continuous = cfg.bridge.model == "averaging"
        self._T = None if self.continuous else float(cfg.ctrl.period)
        self._t0 = cfg.pwm.grid_offset
        self.values: dict[str, Any] = {}
        self._out_specs: dict[str, tuple] = {}
        self._limits: dict[str, tuple[str, _OutputLimit]] = {}
        self.updated = set()
        refs = cfg.ctrl.references
        self.references = {f.name: getattr(refs, f.name) for f in fields(refs)}
        declared_references = {
            path.partition(".")[2] for path in self.interface.inputs
            if path.startswith("references.")
        }
        if declared_references != set(self.references):
            raise RuntimeError("controller reference fields and input ports differ")
        types = dict(self.interface.inputs)
        for name, loop in cfg.ctrl.loops.items():
            cls = LOOP_TYPES.get(loop.type)
            if cls is None:
                raise ConfigError(f"ctrl.loops.{name}: no loop type {loop.type!r} is registered")
            self.nodes[name] = node = cls(loop, cfg, startup)
            self._out_specs[name] = tuple((port, f"{name}.{port}", spec.complex_value)
                                          for port, spec in node.outputs.items())
            self.periods[name] = loop.period
            self._store(name, node.initial_outputs())
            types.update({f"{name}.{port}": spec for port, spec in node.outputs.items()})
        self._samples = {name: node.sample for name, node in self.nodes.items()}
        dependencies = {name: set() for name in self.nodes}
        targets = set()
        for name, node in self.nodes.items():
            for port, spec in node.inputs.items():
                target = f"{name}.{port}"
                targets.add(target)
                source = self.connections.get(target)
                if source not in types:
                    raise ConfigError(f"ctrl.connections.{target}: missing or unknown source {source!r}")
                if types[source] != spec:
                    raise ConfigError(f"ctrl.connections.{target}: incompatible signal {source!r}: "
                                      f"{types[source]} -> {spec}")
                owner = source.partition(".")[0]
                # A delayed input breaks a cycle only in the sampled implementation.  Continuous
                # averaging has no held previous sample, so its present-time dependency remains.
                if owner in self.nodes and (self.continuous or port not in node.delayed):
                    dependencies[name].add(owner)
        if set(self.connections) - targets:
            raise ConfigError(f"ctrl.connections: unknown input ports {sorted(set(self.connections) - targets)}")
        unknown_outputs = set(self.outputs) - set(self.interface.outputs)
        if unknown_outputs:
            raise ConfigError(f"ctrl.outputs: unknown output ports {sorted(unknown_outputs)}; "
                              f"known: {sorted(self.interface.outputs)}")
        for port, spec in self.interface.outputs.items():
            source = self.outputs.get(port)
            if source not in types or types[source] != spec:
                raise ConfigError(f"ctrl.outputs.{port}: missing or incompatible source {source!r}")
        self.order = []
        if not self.continuous:
            pending = {name: set(deps) for name, deps in dependencies.items()}
            while pending:
                ready = [n for n, deps in pending.items() if not deps]
                if not ready:
                    raise ConfigError(
                        "ctrl.connections: instantaneous cycle in sampled control; "
                        "insert an explicit unit_delay"
                    )
                for name in ready:
                    self.order.append(name)
                    pending.pop(name)
                for deps in pending.values():
                    deps.difference_update(ready)
            self._continuous_plan = tuple((name,) for name in self.order)
            self._continuous_cycles = frozenset()
        else:
            # A continuous control graph may contain algebraic feedback between loops.  Collapse
            # its strongly connected components now: acyclic components retain the ordinary
            # one-pass path, while only a genuinely cyclic component is iterated at RHS time.
            component_plan = self._continuous_components(dependencies)
            self._continuous_plan = component_plan
            self._continuous_cycles = frozenset(
                group for group in component_plan
                if len(group) > 1 or group[0] in dependencies[group[0]]
            )
            self.order = [name for group in component_plan for name in group]
        # Loops whose present outputs or derivatives depend on the DC measurement.  During an
        # automatically solved DC-side algebraic loop, only this downstream slice needs to be
        # reevaluated after the first full controller evaluation.
        self.dc_affected = frozenset()
        affected = set()
        for group in self._continuous_plan:
            sources = (self.connections[f"{name}.{port}"]
                       for name in group for port in self.nodes[name].inputs)
            if any(source == "meas.u_dc" or source.partition(".")[0] in affected
                   for source in sources):
                affected.update(group)
        self.dc_affected = frozenset(affected)
        self._rewire()
        if self.continuous:
            self._prepare_flow()
        self._every = (() if self.continuous else
                       tuple((name, max(1, int(round(T / self._T))))
                             for name, T in self.periods.items()))
        self._every_interrupt = (None if self.continuous else
                                 frozenset(name for name, every in self._every)
                                 if all(every == 1 for _name, every in self._every) else None)

    def limit_output(self, name, port, limit):
        """Attach a controller-owned limit to one sampled loop output."""
        if self.continuous:
            raise RuntimeError("continuous control limits are observed, not applied")
        node = self.nodes.get(name)
        if node is None or port not in node.outputs:
            raise ConfigError(f"controller limit: unknown output {name}.{port}")
        self._limits[name] = (port, limit)
        sample = node.sample

        def limited_sample(t, inputs):
            outputs = sample(t, inputs)
            outputs[port] = limit.sample(t, outputs[port])
            return outputs

        self._samples[name] = limited_sample

    def clear_limit(self, name):
        if self._limits.pop(name, None) is not None:
            self._samples[name] = self.nodes[name].sample

    @staticmethod
    def _continuous_components(dependencies):
        """Return SCCs of a loop dependency graph in condensation-graph signal order."""
        names = tuple(dependencies)
        rank = {name: i for i, name in enumerate(names)}
        successors = {name: [] for name in names}
        for name, deps in dependencies.items():
            for dependency in deps:
                successors[dependency].append(name)

        index = 0
        stack = []
        on_stack = set()
        indices = {}
        low = {}
        components = []

        def visit(name):
            nonlocal index
            indices[name] = low[name] = index
            index += 1
            stack.append(name)
            on_stack.add(name)
            for nxt in successors[name]:
                if nxt not in indices:
                    visit(nxt)
                    low[name] = min(low[name], low[nxt])
                elif nxt in on_stack:
                    low[name] = min(low[name], indices[nxt])
            if low[name] == indices[name]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.append(member)
                    if member == name:
                        break
                components.append(tuple(sorted(component, key=rank.get)))

        for name in names:
            if name not in indices:
                visit(name)

        owner = {name: i for i, group in enumerate(components) for name in group}
        compact_deps = [set() for _ in components]
        for name, deps in dependencies.items():
            here = owner[name]
            compact_deps[here].update(owner[dependency] for dependency in deps
                                      if owner[dependency] != here)
        pending = set(range(len(components)))
        ordered = []
        while pending:
            ready = sorted(
                (i for i in pending if not (compact_deps[i] & pending)),
                key=lambda i: min(rank[name] for name in components[i]),
            )
            if not ready:
                raise RuntimeError("internal error while ordering continuous control components")
            for i in ready:
                ordered.append(components[i])
                pending.remove(i)
        return tuple(ordered)

    def _rewire(self):
        """Resolve loop inputs and graph outputs against the current references."""
        self._boundary_sources = {}
        for path, spec in self.interface.inputs.items():
            owner, _, port = path.partition(".")
            if owner == "meas":
                self._boundary_sources[path] = (_MEASURED, port)
            elif owner == "references":
                value = self.references[port]
                self._boundary_sources[path] = (
                    _CONSTANT, complex(value) if spec.complex_value else value
                )
            else:  # ControlInterface is a package-owned declaration, not user input.
                raise RuntimeError(f"unknown controller input owner {owner!r}")
        self._wiring = {}
        self._immediate = {}
        self._port_source = {}
        for name, node in self.nodes.items():
            wiring = tuple((port,) + self._resolve(self.connections[f"{name}.{port}"]) for port in node.inputs)
            self._wiring[name] = wiring
            self._immediate[name] = tuple(w for w in wiring if w[0] not in node.delayed)
            self._port_source[name] = {w[0]: w[1:] for w in wiring}
        self._latching = frozenset(n for n, node in self.nodes.items() if node.delayed)
        self._output_source = {port: self._resolve(source) for port, source in self.outputs.items()}

    def _prepare_flow(self):
        """Build the fixed buffers and runner for continuous loop-flow evaluations."""
        self._flow_inputs = {
            name: {port: 0.0 for port in node.inputs}
            for name, node in self.nodes.items()
        }
        self._flow_outputs = {
            name: {port: self.values[key] for port, key, _complex in self._out_specs[name]}
            for name in self.nodes
        }
        self._flow_derivatives = {
            name: dict(node.flow_state())
            for name, node in self.nodes.items()
        }
        self._flow_derivative_specs = {
            name: tuple((local, f"{name}.{local}") for local in flow)
            for name, flow in self._flow_derivatives.items()
        }
        self.flow_state_names = tuple(
            key for name in self.order for _local, key in self._flow_derivative_specs[name]
        )
        self._flow_derivative_index = {
            key: index for index, key in enumerate(self.flow_state_names)
        }
        self._flow_cycle_keys = {
            group: tuple(key for name in group for _port, key, _complex in self._out_specs[name])
            for group in self._continuous_cycles
        }
        steps = {
            name: (
                name, node.flow_into, self._wiring[name], self._flow_inputs[name],
                self._flow_outputs[name], self._flow_derivatives[name],
                self._out_specs[name], self._flow_derivative_specs[name],
            )
            for name, node in self.nodes.items()
        }
        self._flow_plan = tuple(
            (group, group in self._continuous_cycles,
             self._flow_cycle_keys.get(group, ()), tuple(steps[name] for name in group))
            for group in self._continuous_plan
        )
        if self._continuous_cycles:
            self._flow_runner = self._dc_flow_runner = None
        else:
            self._flow_runner = self._compile_flow_runner(None)
            self._dc_flow_runner = (
                self._compile_flow_runner(self.dc_affected)
                if self.dc_affected else None
            )

    def _compile_flow_runner(self, selected):
        """Compile an acyclic fixed-wiring graph into one allocation-free Python call."""
        environment = {"_values": self.values}
        body = []
        index = 0
        for group, _cyclic, _keys, steps in self._flow_plan:
            name, evaluate, wiring, inputs, outputs, flow, out_specs, flow_specs = steps[0]
            if selected is not None and name not in selected:
                continue
            suffix = str(index)
            index += 1
            node = self.nodes[name]
            path_ports = node.flow_inputs
            environment[f"_evaluate{suffix}"] = (
                node.flow_path if path_ports is not None else evaluate
            )
            environment[f"_inputs{suffix}"] = inputs
            environment[f"_outputs{suffix}"] = outputs
            environment[f"_flow{suffix}"] = flow
            sources = {}
            for port, kind, key in wiring:
                if kind == _HELD:
                    value = f"_values[{key!r}]"
                elif kind == _MEASURED:
                    value = f"meas.{key}"
                else:
                    constant = f"_constant{suffix}_{len(environment)}"
                    environment[constant] = key
                    value = constant
                sources[port] = value
                if path_ports is not None:
                    continue
                body.append(f"_inputs{suffix}[{port!r}] = {value}")
            if path_ports is None:
                body.append(
                    f"_evaluate{suffix}(t, _inputs{suffix}, _outputs{suffix}, _flow{suffix})"
                )
                body.extend(
                    f"_values[{key!r}] = _outputs{suffix}[{port!r}]"
                    for port, key, _complex_value in out_specs
                )
                body.extend(
                    f"derivatives[{self._flow_derivative_index[key]}] = "
                    f"_flow{suffix}[{local!r}]"
                    for local, key in flow_specs
                )
            else:
                result = f"_result{suffix}"
                body.append(
                    f"{result} = _evaluate{suffix}("
                    + ", ".join(sources[port] for port in path_ports) + ")"
                )
                output_values = {port: f"{result}[{position}]"
                                 for position, (port, _key, _complex_value) in enumerate(out_specs)}
                offset = len(out_specs)
                flow_values = {local: f"{result}[{position}]"
                               for position, (local, _key) in enumerate(flow_specs, start=offset)}
                for port, key, _complex_value in out_specs:
                    body.append(f"_values[{key!r}] = {output_values[port]}")
                for local, key in flow_specs:
                    body.append(
                        f"derivatives[{self._flow_derivative_index[key]}] = "
                        f"{flow_values[local]}"
                    )
        source = "def run(t, meas, derivatives):\n" + "\n".join(
            "    " + line for line in (body or ["pass"])
        ) + "\n"
        exec(compile(source, "<peslite continuous control>", "exec"), environment)
        return environment["run"]

    def schedule(self, cfg):
        """Queue unit parameters to take effect at the next control update."""
        self._pending.append(cfg)

    def _apply_pending(self):
        """Apply queued references and rebuild loops whose parameters changed."""
        while self._pending:
            cfg = self._pending.pop(0)
            ctrl = cfg.ctrl
            if ctrl.references != self.cfg.ctrl.references:
                self.references = {f.name: getattr(ctrl.references, f.name)
                                   for f in fields(ctrl.references)}
                self._rewire()
                if self.continuous:
                    self._prepare_flow()
            changed = [name for name, params in ctrl.loops.items()
                       if params != self.cfg.ctrl.loops[name]]
            self.cfg = cfg
            for name in changed:
                self._retune(name)

    def _retune(self, name):
        """Rebuild one loop and continue from the old instance's state."""
        old = self.nodes[name]
        params = self.cfg.ctrl.loops[name]
        fresh = LOOP_TYPES[params.type](params, self.cfg, self.startup)
        if self.continuous:
            old_schema = set(old.flow_state())
            new_schema = set(fresh.flow_state())
            if new_schema != old_schema:
                raise ConfigError(
                    f"ctrl.loops.{name}: a set event cannot change continuous states "
                    f"from {sorted(old_schema)} to {sorted(new_schema)}"
                )
        try:
            fresh.set_state(old.get_state())
        except (KeyError, ValueError) as exc:
            raise ConfigError(
                f"ctrl.loops.{name}: with the parameters of a set event it has other "
                f"states than before ({exc}), so it cannot continue from them") from None
        fresh.retuned(old)
        self.nodes[name] = fresh
        self._samples[name] = fresh.sample
        if self.continuous:
            self._prepare_flow()
        if self.on_retune is not None:
            self.on_retune(name)

    def _resolve(self, source):
        if source in self._boundary_sources:
            return self._boundary_sources[source]
        return _HELD, source

    def _read(self, kind, key, meas):
        if kind == _HELD:
            return self.values[key]
        if kind == _MEASURED:
            return getattr(meas, key)
        return key

    def _gather(self, wiring, meas):
        values = self.values  # _read, inlined: this runs at every loop update
        out = {}
        for port, kind, key in wiring:
            if kind == _HELD:
                out[port] = values[key]
            elif kind == _MEASURED:
                out[port] = getattr(meas, key)
            else:
                out[port] = key
        return out

    def _store(self, name, outputs):
        specs = self._out_specs[name]
        values = self.values
        isfinite = math.isfinite
        for port, key, complex_value in specs:
            value = outputs[port]
            if complex_value:
                value = complex(value)
                finite = isfinite(value.real) and isfinite(value.imag)
            else:
                value = float(value)
                finite = isfinite(value)
            if not finite:
                raise FloatingPointError(f"control loop {name}.{port} returned a non-finite value")
            values[key] = value

    def due(self, t):
        """Return the loops that run at the control interrupt at ``t``."""
        if self.continuous:
            raise RuntimeError("a continuous control graph has no interrupt grid")
        if self._every_interrupt is not None:
            return self._every_interrupt
        k = int(round((t - self._t0) / self._T))
        return {name for name, every in self._every if k % every == 0}

    # ------------------------------------------------------------ running
    def track(self, meas):
        """Pre-synchronise every loop that supports ``track(inputs)``."""
        done = set()
        for name in self.order:
            track = getattr(self.nodes[name], "track", None)
            out = track(self._gather(self._immediate[name], meas)) if track is not None else None
            if out is not None:
                self._store(name, out)
                done.add(name)
        return done

    def reset_integrators(self):
        """Clear every loop's explicitly declared error integrator."""
        for node in self.nodes.values():
            node.reset_integrator()

    def input(self, name, port, meas):
        """Return the input ``port`` of the loop ``name`` for the pu measurement ``meas``."""
        return self._read(*self._port_source[name][port], meas)

    def sample(self, t, meas, finalize=None):
        """Run the loops due at ``t`` on the pu measurement ``meas``; return their names.

        ``finalize(t, meas, updated)`` runs after them, before the delayed inputs are latched.
        """
        if self._pending:
            self._apply_pending()
        self.updated = updated = set()
        due = self.due(t)
        if not due:
            return updated
        samples, gather_, immediate = self._samples, self._gather, self._immediate
        for name in self.order:
            if name not in due:
                continue
            self._store(name, samples[name](t, gather_(immediate[name], meas)))
            updated.add(name)
        if finalize is not None:
            finalize(t, meas, updated)
        for name in due:
            if name in self._latching:
                self.nodes[name].latch(gather_(self._wiring[name], meas))
        return updated

    def flow(self, t, meas, only=None, derivatives=None):
        """Evaluate continuous loops and return their named state derivatives.

        Acyclic components run once.  A strongly connected component is solved by warm-started
        Gauss--Seidel iteration over its typed scalar/complex outputs.  Consequently an ordinary
        controller pays no algebraic-iteration overhead, while custom continuous loops may use
        direct feedback without placing a delay in either loop.
        """
        if self._pending:
            self._apply_pending()
        named = derivatives is None
        if derivatives is None:
            derivatives = [0.0] * len(self.flow_state_names)
        runner = self._flow_runner
        if runner is not None:
            if only is None:
                runner(t, meas, derivatives)
                return (dict(zip(self.flow_state_names, derivatives))
                        if named else derivatives)
            if only == self.dc_affected and self._dc_flow_runner is not None:
                self._dc_flow_runner(t, meas, derivatives)
                return (dict(zip(self.flow_state_names, derivatives))
                        if named else derivatives)
        values = self.values
        for group, cyclic, keys, steps in self._flow_plan:
            if only is not None and not any(name in only for name in group):
                continue
            if not cyclic:
                _name, evaluate, wiring, inputs, outputs, flow, out_specs, flow_specs = steps[0]
                for port, kind, key in wiring:
                    if kind == _HELD:
                        inputs[port] = values[key]
                    elif kind == _MEASURED:
                        inputs[port] = getattr(meas, key)
                    else:
                        inputs[port] = key
                evaluate(t, inputs, outputs, flow)
                for port, key, _complex_value in out_specs:
                    values[key] = outputs[port]
                for local, key in flow_specs:
                    derivatives[self._flow_derivative_index[key]] = flow[local]
                continue
            previous = tuple(values[key] for key in keys)
            for _iteration in range(16):
                for (_name, evaluate, wiring, inputs, outputs, flow,
                     out_specs, flow_specs) in steps:
                    for port, kind, key in wiring:
                        if kind == _HELD:
                            inputs[port] = values[key]
                        elif kind == _MEASURED:
                            inputs[port] = getattr(meas, key)
                        else:
                            inputs[port] = key
                    evaluate(t, inputs, outputs, flow)
                    for port, key, _complex_value in out_specs:
                        values[key] = outputs[port]
                    for local, key in flow_specs:
                        derivatives[self._flow_derivative_index[key]] = flow[local]
                current = tuple(values[key] for key in keys)
                if all(abs(new - old) <= 1e-10 + 1e-8 * max(1.0, abs(new), abs(old))
                       for new, old in zip(current, previous)):
                    break
                previous = current
            else:
                raise FloatingPointError(
                    "continuous control algebraic loop did not converge after 16 iterations: "
                    + ", ".join(group)
                )
        return dict(zip(self.flow_state_names, derivatives)) if named else derivatives

    def output(self, port, meas):
        """Return the graph output ``port`` (``u_dq``, ``theta`` or ``omega``)."""
        return self._read(*self._output_source[port], meas)

    def describe(self):
        return {"loops": {n: {"type": self.cfg.ctrl.loops[n].type, "period": self.periods[n]}
                          for n in self.order}, "connections": dict(self.connections), "outputs": dict(self.outputs)}

    # ------------------------------------------------------------ states
    def get_state(self):
        state = gather(self.nodes)
        state.update({f"held.{self._state_port(k)}": v for k, v in self.values.items()})
        return state

    def set_state(self, values):
        unknown = set(values) - set(self.get_state())
        if unknown:
            raise KeyError(f"unknown control states {sorted(unknown)}; electrical held outputs and converted states use _pu")
        state_ports = {self._state_port(k): k for k in self.values}
        rest = {}
        for key, value in values.items():
            if key.startswith("held."):
                self.values[state_ports[key[5:]]] = value
            else:
                rest[key] = value
        scatter(self.nodes, rest)
        # held outputs that follow the loaded states, unless given themselves
        for name, node in self.nodes.items():
            if node.outputs_from_state:
                for port, value in node.initial_outputs().items():
                    if f"held.{self._state_port(name + '.' + port)}" not in values:
                        self.values[f"{name}.{port}"] = value

    def _state_port(self, key):
        name, port = key.split(".", 1)
        return key + "_pu" if self.nodes[name].outputs[port].unit.startswith("pu_") else key


# ------------------------------------------------------------------ the default wiring

def default_wiring(cfg, family):
    """Return the default ``(connections, outputs)`` for family ``"gfl"``, ``"gfm"`` or ``"custom"``.

    Each loop is wired by its role (:attr:`peslite.control.loops.Loop.role`), so a registered loop
    type takes the place of the built-in one of its role.
    """
    if family == "custom":
        return {}, {}
    loops = cfg.ctrl.loops
    roles = {name: getattr(LOOP_TYPES.get(p.type), "role", None) for name, p in loops.items()}

    def find(role):
        found = [n for n, r in roles.items() if r == role]
        if len(found) > 1:
            raise ConfigError(f"ctrl.loops: ambiguous {family} role {found}; use ctrl.type = 'custom' "
                              "and explicit connections/outputs for multiple instances of this role")
        return found[0] if len(found) == 1 else None
    pll, sync = find("pll"), find("sync")
    angle = pll if family == "gfl" else sync
    frame = f"{angle}.frame" if angle else "references.theta"
    theta = f"{angle}.theta" if angle else "references.theta"
    omega = f"{angle}.omega" if angle else "references.omega"
    dc, cc, power = find("dc_voltage"), find("current"), find("power")
    va, vi, damping = find("admittance"), find("impedance"), find("damping")
    extra = f"{damping}.extra" if damping else "references.zero_v_pu"
    ports_of = {
        "pll": {"v": "meas.u_g"},
        "dc_voltage": {"u_dc": "meas.u_dc", "vdc_ref": "references.vdc_ref_pu"},
        "current": {"v": "meas.u_g", "i": "meas.i_c", "frame": frame, "omega": omega,
                    "extra": extra, "id_ref": f"{va or dc}.id_ref" if va or dc else "references.id_ref_pu",
                    "iq_ref": f"{va}.iq_ref" if va else "references.iq_ref_pu"},
        "power": {"v": "meas.u_g", "i": "meas.i_c"},
        "sync": {"p": f"{power}.p", "q": f"{power}.q", "v": "meas.u_g", "i": "meas.i_c",
                 "u_dc": "meas.u_dc", "p_ref": "references.p_ref_pu", "q_ref": "references.q_ref_pu",
                 "v_ref": "references.v_ref_pu"},
        "impedance": {"v_ref": f"{sync}.v_ref", "frame": frame, "omega": omega, "i": "meas.i_c",
                      "extra": extra},
        "admittance": {"v_ref": f"{sync}.v_ref", "frame": frame, "omega": omega, "v": "meas.u_g"},
        "damping": {"i": "meas.i_c", "frame": frame},
    }
    wires = {f"{name}.{port}": source for name, role in roles.items()
             for port, source in ports_of.get(role, {}).items()
             if port in LOOP_TYPES[loops[name].type].inputs}
    terminal = vi if family == "gfm" and vi else cc
    return wires, {"u_dq": f"{terminal}.u_dq", "theta": theta, "omega": omega}


# ------------------------------------------------------------------ the controller

class _ContinuousControlState:
    """Dynamic attributes named ``<loop>.<state>`` for Model's packed ODE state."""


class _ContinuousControlInp(Bag):
    __slots__ = ("u_g", "i_c", "u_dc")


class _ContinuousControlOut(Bag):
    __slots__ = ("q",)

class UniteType:
    """Configurable closed controller block with loops, start-up and output stage.

    The host gives it only run/stop commands and parameter updates.  With switching or PWM-period
    averaging it is called at each interrupt.  With ideal averaging it is a continuous subsystem:
    its loop states join the plant ODE and its output is the bridge's modulation space vector.
    """

    outputs_need_inputs = True
    observer = True

    def __init__(self, cfg, pwm_method=None, *, limiter=CONFIGURED):
        self.p = cfg
        self.continuous = cfg.bridge.model == "averaging"
        # A continuous controller already participates in construction-time energy probes, before
        # VSC.start() applies its scenario. Keep its ideal source off until that command arrives.
        self.startup = ContinuousStartup(run=False) if self.continuous else Startup(cfg.ctrl.period)
        wires, outputs = default_wiring(cfg, cfg.ctrl.type)
        self.graph = graph = ControlGraph(cfg, self.startup, wires, outputs)
        self.periods = graph.periods
        self.v_base, self.i_base, self.w0 = cfg.base.v_phase_peak, cfg.base.i_phase_peak, cfg.base.w0
        self.v_dc_base = cfg.dc_base.v
        # Ideal averaging has no PWM/output stage. Its continuous controller drives the bridge
        # voltage ratio directly; sampled bridge models retain modulation and anti-windup here.
        self.stage = (None if self.continuous else
                      OutputStage(cfg, pwm_method=pwm_method, limiter=limiter))
        self.theta, self.omega = self.initial_sync()
        self.u_cmd = 1.0 + 0j
        self.command_theta = self.theta
        self.v_dq = self.i_dq = 0j
        self.last_log = {}
        self.logging = True
        self._in_algebraic_solve = False
        self._algebraic_first = False
        # the loops the controller itself reads, by role
        role = {name: node.role for name, node in graph.nodes.items()}
        def first(r, port=None):
            return next((name for name, value in role.items()
                         if value == r and (port is None or port in graph.nodes[name].inputs)), None)

        self._terminal = graph.outputs["u_dq"].partition(".")[0]
        self._terminal_in_graph = self._terminal in graph.nodes
        graph.on_retune = self._loop_retuned
        self._frame_key = f"{graph.outputs['theta'].partition('.')[0]}.frame"
        self._dc, self._sync = first("dc_voltage"), first("sync")
        self._frame_cc, self._log_cc = first("current", "frame"), first("current", "id_ref")
        self._theta_source = graph._output_source["theta"]
        self._omega_source = graph._output_source["omega"]
        self._command_source = graph._output_source["u_dq"]
        self._frame_source = (graph._port_source[self._frame_cc]["frame"]
                              if self._frame_cc is not None else None)
        self._is_gfl = cfg.ctrl.type == "gfl"
        self._p_key, self._q_key, self._v_ref_key = f"{first('power')}.p", f"{first('power')}.q", f"{self._sync}.v_ref"
        self._terminal_limit = None
        self._terminal_limit_scale = None
        self._flow_limits = {}
        self.alarms: list[str] = []
        self._configure_limit(self._dc)
        self._configure_limit(self._terminal)
        if self.continuous:
            self._build_flow_state()

    def _configure_limit(self, name):
        """Build a sampled output limit or a continuous observation-only limit."""
        if name is None:
            return
        graph = self.graph
        graph.clear_limit(name)
        self._flow_limits.pop(name, None)
        node = graph.nodes.get(name)
        if name == self._terminal:
            self._terminal_limit = None
            self._terminal_limit_scale = None
            if self.stage is not None:
                self.stage._command_saturated = False
        if isinstance(node, DCVoltageLoop):
            limit = _OutputLimit(
                node.limit, lower=node.floor, pi=None if self.continuous else node.pi,
                observer=node.observe_limit,
            )
            if self.continuous:
                self._flow_limits[name] = (f"{name}.id_ref", limit, "ID_REF_LIMIT", node)
            else:
                graph.limit_output(name, "id_ref", limit)
        if (not self.continuous and name == self._terminal
                and isinstance(node, CurrentLoop) and self.stage is not None):
            limit = self.stage.command_limit(1.0)
            if limit is not None:
                output_limit = _OutputLimit(
                    limit, pi=node.pi,
                    active=lambda: self.startup.active,
                )
                graph.limit_output(name, "u_dq", output_limit)
                self._terminal_limit = output_limit
                self._terminal_limit_scale = limit

    def observe_limits(self, t: float) -> None:
        """Check ideal-model commands at a recorded snapshot without altering their values."""
        for path, limit, alarm, node in self._flow_limits.values():
            value = self.graph.values[path]
            exceeded = limit.exceeded(value)
            node.n_updates += 1
            node.limit_exceeded = exceeded
            if not exceeded:
                continue
            node.n_limit_exceeded += 1
            if node.first_limit_t is None:
                node.first_limit_t = t
            if alarm not in self.alarms:
                self.alarms.append(alarm)

    def _build_flow_state(self):
        """Create the fixed ODE state layout from each loop's flow-state declaration."""
        state = _ContinuousControlState()
        names = []
        for name in self.graph.order:
            node = self.graph.nodes[name]
            try:
                if (type(node).flow is Loop.flow
                        and (node.flow_inputs is None
                             or (type(node).flow_path is Loop.flow_path
                                 and type(node).equation is Loop.equation))):
                    raise NotImplementedError(
                        f"loop type {node.type!r} has no continuous-time implementation"
                    )
                if isinstance(node, SyncLaw) and type(node)._flow is SyncLaw._flow:
                    raise NotImplementedError(
                        f"loop type {node.type!r} has no continuous-time flow implementation"
                    )
                values = node.flow_state()
            except NotImplementedError as exc:
                raise ConfigError(f"ctrl.loops.{name}: {exc}") from None
            for local, value in values.items():
                if isinstance(value, bool):
                    raise ConfigError(
                        f"ctrl.loops.{name}: boolean state {local!r} cannot be a continuous ODE state"
                    )
                key = f"{name}.{local}"
                names.append(key)
                setattr(state, key, complex(value) if isinstance(value, complex) else float(value))
        self.state, self.state_names = state, tuple(names)
        if self.state_names != self.graph.flow_state_names:
            raise RuntimeError("continuous controller state and derivative layouts differ")
        self.inp = _ContinuousControlInp(u_g=0j, i_c=0j, u_dc=self.p.dclink.vdc_ref)
        self.out = _ContinuousControlOut(q=0j)
        self._derivatives = [0j if isinstance(getattr(state, name), complex) else 0.0
                             for name in self.state_names]
        self._flow_derivative_index = {
            name: index for index, name in enumerate(self.state_names)
        }
        self._continuous_meas = ControlMeasurement(0.0, 0j, 0j, 1.0, np.zeros(3))
        self._last_control_meas = self._continuous_meas
        self._bind_flow_state()

    def _bind_flow_state(self):
        """Resolve direct loop-state assignments once, including after a loop is retuned."""
        direct = []
        indirect = []
        known = set(self.state_names)
        for name in self.graph.order:
            node = self.graph.nodes[name]
            values = node.flow_state()
            bindings = node.flow_state_bindings()
            if bindings is None or set(bindings) != set(values):
                indirect.append((node, tuple((local, f"{name}.{local}") for local in values)))
                continue
            for local, (owner, attribute) in bindings.items():
                key = f"{name}.{local}"
                if key not in known:
                    raise ConfigError(f"ctrl.loops.{name}: unknown continuous state binding {local!r}")
                direct.append((owner, attribute, key))
        self._continuous_direct = tuple(direct)
        self._continuous_indirect = tuple(indirect)

    def _sync_continuous_nodes(self):
        """Copy the current packed Model state into the loop instances."""
        state = self.state
        for owner, attribute, key in self._continuous_direct:
            setattr(owner, attribute, getattr(state, key))
        for node, bindings in self._continuous_indirect:
            node.set_state({local: getattr(state, key) for local, key in bindings})

    def _store_continuous_nodes(self):
        """Copy loop states changed by a host action (tracking/reset) into Model state."""
        for name in self.graph.order:
            for local, value in self.graph.nodes[name].flow_state().items():
                key = f"{name}.{local}"
                if key in self.state_names:
                    setattr(self.state, key, value)

    def _loop_retuned(self, name=None):
        """Rebuild the loop's limit and continuous state bindings after a set event."""
        self._configure_limit(name)
        if self.continuous:
            if self.graph.flow_state_names != self.state_names:
                raise ConfigError("a set event cannot reorder continuous controller states")
            self._bind_flow_state()

    def describe(self):
        return self.graph.describe()

    def retune(self, cfg, paths, t):
        """Queue new controller parameters for its next interrupt."""
        if any(path.startswith("ctrl.") for path in paths):
            self.graph.schedule(cfg)
        self.p = cfg

    def initial_sync(self):
        values = {**self.graph.values, **{f"references.{k}": v for k, v in self.graph.references.items()}}
        return float(values[self.graph.outputs["theta"]]), float(values[self.graph.outputs["omega"]])

    def command(self, run, ramp=0.0):
        """Take the host's run/stop command."""
        if self.continuous:
            self.command_at(run, ramp, 0.0)
            return
        starting = bool(run) and not self.startup.active
        if starting:
            self.graph.reset_integrators()
        self.startup.command(run, ramp)

    def command_at(self, run, ramp, t, *, reset=True):
        """Take a continuous run/stop command at absolute time ``t``."""
        if not self.continuous:
            self.command(run, ramp)
            return
        starting = bool(run) and not self.startup.active
        if starting and reset:
            self._sync_continuous_nodes()
            self.graph.reset_integrators()
            self._store_continuous_nodes()
        self.startup.command(run, ramp, t)

    def track(self, meas):
        """Pre-synchronise grid-forming laws to an SI terminal sample."""
        if self.continuous:
            self._sync_continuous_nodes()
        if self.graph.track(self._pu(meas)):
            self.theta, self.omega = self.initial_sync()
            self.command_theta = self.theta
        if self.startup.active:
            # A fresh run is tracked after its host command.  Preserve the aligned angle and
            # magnitude, but do not let tracking seed an error integrator before PWM starts.
            self.graph.reset_integrators()
        if self.continuous:
            self._store_continuous_nodes()

    # ------------------------------------------------------------ one sample
    def _pu(self, meas: Measurement) -> ControlMeasurement:
        """The SI sample in the unit's pu bases."""
        if not isinstance(meas, Measurement):
            raise TypeError("the controller takes an SI Measurement, not an already normalized sample")
        return ControlMeasurement(meas.t, meas.u_g / self.v_base, meas.i_c / self.i_base,
                                  meas.u_dc / self.v_dc_base, meas.i_abc / self.i_base)

    def _raw_log(self, meas: Measurement, theta: float) -> dict[str, float]:
        """The pre-average sample in the ``theta`` frame as ``*_raw_pu`` log entries (empty if not averaged)."""
        out: dict[str, float] = {}
        if meas.u_g_raw is not None and (meas.u_g_raw != meas.u_g or meas.i_c_raw != meas.i_c):
            rot = cmath.exp(-1j * theta)
            v, i = meas.u_g_raw * rot, meas.i_c_raw * rot
            out.update({"vd_raw_pu": v.real / self.v_base, "vq_raw_pu": v.imag / self.v_base,
                        "id_raw_pu": i.real / self.i_base, "iq_raw_pu": i.imag / self.i_base})
        if meas.u_dc_raw is not None and meas.u_dc_raw != meas.u_dc:
            out["vdc_raw_pu"] = meas.u_dc_raw / self.v_dc_base
        return out

    def log_names(self) -> tuple[str, ...]:
        """Return the fixed controller-log schema before the first interrupt."""
        names = ["id_pu", "iq_pu", "vd_pu", "vq_pu", "vac_pu", "vdc_pu"]
        if not self.continuous:
            names.append("m_max")
        names.append("in_service")
        if not self.continuous and {"u_g", "i_c"} & set(self.p.meas.average):
            names += ["vd_raw_pu", "vq_raw_pu", "id_raw_pu", "iq_raw_pu"]
        if not self.continuous and "u_dc" in self.p.meas.average:
            names.append("vdc_raw_pu")
        names += (["id_ref_pu", "freq_dev", "angle_rel"] if self._is_gfl else
                  ["p_pu", "q_pu", "p_ref_pu", "v_ref_pu", "freq_dev", "angle_rel"])
        return tuple(names)

    def set_logging(self, enabled: bool) -> None:
        """Enable per-interrupt log dictionaries when a run records or watches them."""
        self.logging = bool(enabled)
    def _accept_command(self, t, meas, updated):
        """Update the held angle, frequency and sampled voltage command."""
        graph = self.graph
        self.theta = float(graph.output("theta", meas))
        self.omega = float(graph.output("omega", meas))
        if self._terminal in updated or (updated and not self._terminal_in_graph):
            self.u_cmd = complex(graph.output("u_dq", meas))
            self.command_theta = self.theta
            if self._terminal_limit is not None:
                self.stage._command_saturated = self._terminal_limit.saturated
            self.stage.modulate(t, self.u_cmd, self.command_theta, meas.u_dc, count=False)

    def __call__(self, t, meas):
        if self.continuous:
            raise RuntimeError("an ideal averaged controller is evaluated by the ODE solver, not sampled")
        control_meas = self._pu(meas)
        startup = self.startup
        if startup.in_progress:
            startup.advance()
        self.stage.new_instant()
        if self._terminal_limit is not None:
            self._terminal_limit.limit = (
                self._terminal_limit_scale * max(0.0, control_meas.u_dc)
            )
        self.graph.sample(t, control_meas, finalize=self._accept_command)
        frame = self.graph.values.get(self._frame_key, self.theta)
        if self._frame_cc is not None:
            frame = self.graph.input(self._frame_cc, "frame", control_meas)
        rot = cmath.exp(-1j * frame)
        self.v_dq, self.i_dq = control_meas.u_g * rot, control_meas.i_c * rot
        gates = startup.active
        duty = self.stage.finish(t, self.u_cmd, self.command_theta, control_meas.u_dc,
                                 count=gates)
        log = None
        if self.logging:
            freq_dev = (self.omega - self.w0) / (2 * math.pi)
            refs = self.graph.references
            log = {"id_pu": self.i_dq.real, "iq_pu": self.i_dq.imag,
                   "vd_pu": self.v_dq.real, "vq_pu": self.v_dq.imag,
                   "vac_pu": abs(self.v_dq), "vdc_pu": control_meas.u_dc,
                   "m_max": peak_abs(self.stage.m_abc),
                   "in_service": 1.0 if gates else 0.0,
                   **self._raw_log(meas, frame)}
            angle_rel = (self.theta - self.w0 * t + math.pi) % (2 * math.pi) - math.pi
            if self._is_gfl:
                id_ref = self.graph.input(self._log_cc, "id_ref", control_meas) if self._log_cc else 0.0
                log.update(id_ref_pu=id_ref, freq_dev=freq_dev, angle_rel=angle_rel)
            else:
                values = self.graph.values
                pr = startup.value * refs["p_ref_pu"]
                log.update(p_pu=values.get(self._p_key, 0.0), q_pu=values.get(self._q_key, 0.0),
                           p_ref_pu=pr, v_ref_pu=values.get(self._v_ref_key, 1.0),
                           freq_dev=freq_dev, angle_rel=angle_rel)
            self.last_log = log
        return ControlOutput(duty, log=log, theta=self.theta, omega=self.omega,
                             gates=gates, startup_complete=startup.complete)

    # ------------------------------------------------------------ continuous ODE
    def begin_algebraic(self):
        """Prepare loop states once before repeated algebraic output evaluations."""
        if self.continuous:
            self._sync_continuous_nodes()
            self._in_algebraic_solve = True
            self._algebraic_first = True

    def end_algebraic(self):
        self._in_algebraic_solve = False

    def set_outputs(self, t):
        """Evaluate continuous measurements, loop outputs and bridge modulation at one ODE stage."""
        if not self.continuous:
            return
        if not self._in_algebraic_solve:
            self._sync_continuous_nodes()
        self.startup.at(t)
        meas = self._continuous_meas
        meas.t = t
        meas.u_g = self.inp.u_g / self.v_base
        meas.i_c = self.inp.i_c / self.i_base
        meas.u_dc = self.inp.u_dc / self.v_dc_base
        only = None
        if self._in_algebraic_solve and not self._algebraic_first:
            only = self.graph.dc_affected
        derivatives = self._derivatives
        self.graph.flow(t, meas, only=only, derivatives=derivatives)
        self._algebraic_first = False
        graph = self.graph
        self.theta = float(graph._read(*self._theta_source, meas))
        self.omega = float(graph._read(*self._omega_source, meas))
        self.u_cmd = complex(graph._read(*self._command_source, meas))
        self.command_theta = self.theta
        frame = graph.values.get(self._frame_key, self.theta)
        if self._frame_source is not None:
            frame = graph._read(*self._frame_source, meas)
        command_rot = complex(math.cos(self.command_theta), math.sin(self.command_theta))
        into_frame = (command_rot.conjugate() if frame == self.command_theta
                      else cmath.exp(-1j * frame))
        self.v_dq, self.i_dq = meas.u_g * into_frame, meas.i_c * into_frame
        u_dc = meas.u_dc * self.v_dc_base
        if self.startup.active and u_dc != 0.0:
            u_ab = self.u_cmd * command_rot * self.v_base
            a, b, c = phases(u_ab)
            gain = 2.0 / u_dc
            # Express the ideal three-leg average on the common bridge-ratio port. The common
            # 0.5 term cancels from the space vector; retaining this operation order also keeps
            # saved-state continuation bit-level stable across solver restarts.
            self.out.q = abc2complex((0.5 * (1.0 + gain * a),
                                      0.5 * (1.0 + gain * b),
                                      0.5 * (1.0 + gain * c)))
        else:
            self.out.q = 0j
        self._last_control_meas = meas

    def rhs(self, t):
        return self._derivatives

    def continuous_log(self, t):
        """Controller observation at an output snapshot of the continuous model."""
        if not self.logging:
            return None
        meas = self._last_control_meas
        freq_dev = (self.omega - self.w0) / (2 * math.pi)
        refs = self.graph.references
        log = {"id_pu": self.i_dq.real, "iq_pu": self.i_dq.imag,
               "vd_pu": self.v_dq.real, "vq_pu": self.v_dq.imag,
               "vac_pu": abs(self.v_dq), "vdc_pu": meas.u_dc,
               "in_service": 1.0 if self.startup.active else 0.0}
        angle_rel = (self.theta - self.w0 * t + math.pi) % (2 * math.pi) - math.pi
        if self._is_gfl:
            id_ref = self.graph.input(self._log_cc, "id_ref", meas) if self._log_cc else 0.0
            log.update(id_ref_pu=id_ref, freq_dev=freq_dev, angle_rel=angle_rel)
        else:
            values = self.graph.values
            log.update(p_pu=values.get(self._p_key, 0.0), q_pu=values.get(self._q_key, 0.0),
                       p_ref_pu=self.startup.value * refs["p_ref_pu"],
                       v_ref_pu=values.get(self._v_ref_key, 1.0),
                       freq_dev=freq_dev, angle_rel=angle_rel)
        self.last_log = log
        return log

    # ------------------------------------------------------------ states and summary
    def get_state(self):
        if self.continuous:
            return gather({"startup": self.startup})
        return {**self.graph.get_state(), "command.u_dq_pu": self.u_cmd,
                "command.theta": self.command_theta, "command.omega": self.omega,
                **gather({"startup": self.startup})}

    def set_state(self, values):
        if self.continuous:
            scatter({"startup": self.startup}, values)
            return
        rest, startup = {}, {}
        for name, value in values.items():
            if name.startswith("startup."):
                startup[name] = value
            elif name == "command.u_dq_pu":
                self.u_cmd = complex(value)
            elif name == "command.theta":
                self.command_theta = float(value)
            elif name == "command.omega":
                self.omega = float(value)
            else:
                rest[name] = value
        self.graph.set_state(rest)
        scatter({"startup": self.startup}, startup)
        self.theta, self.omega = self.initial_sync()

    def summary(self):
        """Return controller and modulation observations."""
        stage = self.stage
        summary = ({
            "modulation_saturation_fraction": None,
            "modulation_saturation_first_t": None,
        } if stage is None else {
            "modulation_saturation_fraction": stage.n_saturated / max(1, stage.n_updates),
            "modulation_saturation_first_t": stage.first_saturation_t,
        })
        if self._dc is not None:
            dc = self.graph.nodes[self._dc]
            summary["id_ref_limit_fraction"] = dc.n_limit_exceeded / max(1, dc.n_updates)
            summary["id_ref_limit_first_t"] = dc.first_limit_t
        if self._sync is not None:
            summary["law"] = self.p.ctrl.loops[self._sync].type
        summary["alarms"] = list(self.alarms)
        return summary


SCALAR_CONTROL_INTERFACE = ControlInterface(
    inputs=MappingProxyType({
        "meas.current": SI_CURRENT,
        "meas.voltage": SI_VOLTAGE,
        "meas.vin": SI_VOLTAGE,
        "references.voltage": SI_VOLTAGE,
    }),
    outputs=MappingProxyType({"duty": DUTY_RATIO}),
)


@dataclass
class _ScalarMeasurement:
    current: float = 0.0
    voltage: float = 0.0
    vin: float = 0.0


class _ScalarControlState:
    """Dynamic attributes containing the scalar graph's continuous PI states."""


class _ScalarControlInp(Bag):
    __slots__ = ("current", "voltage", "vin")


class _ScalarControlOut(Bag):
    __slots__ = ("q",)


class ScalarController:
    """Cascaded scalar controller using the shared control graph and registered loops.

    The adapter changes only the controller boundary from three-phase vectors to scalar SI
    measurements. Loop scheduling, PI states, limits, retuning and continuous flow remain owned by
    :class:`ControlGraph` and the existing loop implementations.
    """

    outputs_need_inputs = True
    observer = True

    def __init__(self, cfg) -> None:
        self.p = cfg
        self.type = cfg.topology
        self.continuous = cfg.bridge.model == "averaging"
        self.startup = (
            ContinuousStartup(run=False) if self.continuous
            else Startup(cfg.ctrl.period, run=False)
        )
        connections = {
            "voltage.reference": "references.voltage",
            "voltage.voltage": "meas.voltage",
            "current.reference": "voltage.current_reference",
            "current.current": "meas.current",
            "current.voltage": "meas.voltage",
            "current.vin": "meas.vin",
        }
        self.graph = ControlGraph(
            cfg,
            self.startup,
            connections,
            {"duty": "current.duty"},
            interface=SCALAR_CONTROL_INTERFACE,
        )
        self.graph.on_retune = self._loop_retuned
        self.periods = self.graph.periods
        self.logging = True
        self.last_log: dict[str, float] = {}
        self._last_measurement = _ScalarMeasurement(vin=cfg.source_voltage)
        self._last_duty = 0.0
        if self.continuous:
            self._build_flow_state()

    def _measurement(self, current: float, voltage: float, vin: float,
                     target: _ScalarMeasurement | None = None) -> _ScalarMeasurement:
        measured_voltage = abs(voltage) if self.p.topology == "buck_boost" else voltage
        measurement = target if target is not None else _ScalarMeasurement()
        measurement.current = float(current)
        measurement.voltage = float(measured_voltage)
        measurement.vin = float(vin)
        return measurement

    def _build_flow_state(self) -> None:
        state = _ScalarControlState()
        names = []
        for name in self.graph.order:
            for local, value in self.graph.nodes[name].flow_state().items():
                key = f"{name}.{local}"
                names.append(key)
                setattr(state, key, complex(value) if isinstance(value, complex) else float(value))
        self.state, self.state_names = state, tuple(names)
        if self.state_names != self.graph.flow_state_names:
            raise RuntimeError("continuous scalar controller state layouts differ")
        self.inp = _ScalarControlInp(current=0.0, voltage=0.0, vin=self.p.source_voltage)
        self.out = _ScalarControlOut(q=0.0)
        self._derivatives = [0.0] * len(self.state_names)
        self._bind_flow_state()

    def _bind_flow_state(self) -> None:
        direct = []
        indirect = []
        for name in self.graph.order:
            node = self.graph.nodes[name]
            values = node.flow_state()
            bindings = node.flow_state_bindings()
            if bindings is None or set(bindings) != set(values):
                indirect.append((node, tuple((local, f"{name}.{local}") for local in values)))
            else:
                direct.extend(
                    (owner, attribute, f"{name}.{local}")
                    for local, (owner, attribute) in bindings.items()
                )
        self._continuous_direct = tuple(direct)
        self._continuous_indirect = tuple(indirect)

    def _sync_continuous_nodes(self) -> None:
        state = self.state
        for owner, attribute, key in self._continuous_direct:
            setattr(owner, attribute, getattr(state, key))
        for node, bindings in self._continuous_indirect:
            node.set_state({local: getattr(state, key) for local, key in bindings})

    def _store_continuous_nodes(self) -> None:
        for name in self.graph.order:
            for local, value in self.graph.nodes[name].flow_state().items():
                setattr(self.state, f"{name}.{local}", value)

    def _loop_retuned(self, _name=None) -> None:
        if self.continuous:
            self._bind_flow_state()

    def command(self, run: bool, ramp: float = 0.0) -> None:
        if self.continuous:
            self.command_at(run, ramp, 0.0)
            return
        starting = bool(run) and not self.startup.active
        if starting:
            self.graph.reset_integrators()
        self.startup.command(run, ramp)

    def command_at(self, run: bool, ramp: float, t: float, *, reset: bool = True) -> None:
        if not self.continuous:
            self.command(run, ramp)
            return
        starting = bool(run) and not self.startup.active
        if starting and reset:
            self._sync_continuous_nodes()
            self.graph.reset_integrators()
            self._store_continuous_nodes()
        self.startup.command(run, ramp, t)

    def __call__(self, t: float, meas) -> ScalarControlOutput:
        if self.continuous:
            raise RuntimeError("an ideal averaged scalar controller is evaluated by the ODE solver")
        measurement = self._measurement(meas.current, meas.voltage, meas.vin)
        if self.startup.in_progress:
            self.startup.advance()
        self.graph.sample(t, measurement)
        duty = float(self.graph.output("duty", measurement))
        self._last_measurement, self._last_duty = measurement, duty
        gates = self.startup.active
        log = self._log(duty, gates) if self.logging else None
        return ScalarControlOutput(
            duty=duty,
            log=log,
            gates=gates,
            startup_complete=self.startup.complete,
        )

    def set_outputs(self, t: float) -> None:
        self._sync_continuous_nodes()
        self.startup.at(t)
        measurement = self._measurement(
            self.inp.current, self.inp.voltage, self.inp.vin, self._last_measurement,
        )
        self.graph.flow(t, measurement, derivatives=self._derivatives)
        duty = float(self.graph.output("duty", measurement))
        self.out.q = duty if self.startup.active else 0.0
        self._last_measurement, self._last_duty = measurement, duty

    def rhs(self, t: float):
        return self._derivatives

    def _log(self, duty: float, active: bool) -> dict[str, float]:
        measurement = self._last_measurement
        self.last_log = {
            "current": measurement.current,
            "voltage": measurement.voltage,
            "voltage_reference": self.graph.references["voltage"],
            "input_voltage": measurement.vin,
            "duty": duty,
            "in_service": 1.0 if active else 0.0,
        }
        return self.last_log

    def continuous_log(self, t: float) -> dict[str, float] | None:
        return self._log(self._last_duty, self.startup.active) if self.logging else None

    @staticmethod
    def log_names() -> tuple[str, ...]:
        return ("current", "voltage", "voltage_reference", "input_voltage", "duty", "in_service")

    def set_logging(self, enabled: bool) -> None:
        self.logging = bool(enabled)

    def retune(self, cfg, paths, t: float) -> None:
        if any(path.startswith("ctrl.") for path in paths):
            self.graph.schedule(cfg)
        self.p = cfg

    def get_state(self) -> dict[str, Any]:
        startup = {f"startup.{name}": value for name, value in self.startup.get_state().items()}
        return startup if self.continuous else {**self.graph.get_state(), **startup}

    def set_state(self, values: Mapping[str, Any]) -> None:
        startup = {name[8:]: value for name, value in values.items() if name.startswith("startup.")}
        self.startup.set_state(startup)
        if not self.continuous:
            self.graph.set_state({name: value for name, value in values.items()
                                  if not name.startswith("startup.")})

    def summary(self) -> dict[str, Any]:
        return {"alarms": []}


def make_scalar_controller(cfg) -> ScalarController:
    """Build the scalar controller of a Buck, Boost or Buck-Boost unit."""
    return ScalarController(cfg)


def make_controller(cfg, **kwargs):
    """Build the :class:`UniteType` controller of a unit from ``cfg.ctrl``."""
    return UniteType(cfg, **kwargs)
