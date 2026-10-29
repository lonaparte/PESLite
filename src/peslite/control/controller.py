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
import numbers
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..solver.model import Bag, ConfigError, _complex, _real, gather, scatter
from ..solver.energy import PowerPort, port_power
from ..solver.timing import control_clock_offset
from .blocks import OutputLimit, abc2complex, peak_abs, phases, smoothstep
from .loops import (ANGLE, DUTY_RATIO, FREQUENCY, HERTZ, I, I_AB, LOOP_TYPES, MODE, MODULATION, POWER,
                    SI_CURRENT, SI_VOLTAGE, SI_POWER, V, V_AB, V_DQ, CurrentLoop,
                    Loop, SignalType, SyncLaw)
from .modulation import CONFIGURED, OutputStage
from .profiles import (DutyReferenceParams, PhaseShiftReferenceParams, ReferenceParams,
                       ResonantReferenceParams,
                       ScalarVoltageReferenceParams, control_type,
                       register_control_type, validate_duty_references,
                       validate_phase_shift_references,
                       validate_resonant_references,
                       validate_scalar_voltage_references)

__all__ = ["Measurement", "ConverterMeasurement", "ControlMeasurement", "ControlOutput", "Controller", "Startup",
           "ControlInterface", "CONTROL_INTERFACE", "ControlGraph", "default_wiring",
           "VSCController", "BridgeControlOutput", "GraphController",
           "make_controller"]


# ------------------------------------------------------------------ the interface

@dataclass
class Measurement:
    """SI sample at ``t``: peak-scaled alpha-beta AC vectors (V, A) and DC voltage (V).

    Windowed channels hold means over ``(t - T_avg, t]``. ``samples`` holds the instantaneous
    ADC conversions in chronological order, including the conversion at ``t``; an instantaneous
    entry has no nested samples. ``i_abc`` is always derived from the final conversion.
    """

    t: float
    u_g: complex  # PCC voltage (V)
    i_c: complex  # converter current (A), positive out of the bridge
    u_dc: float  # dc-link voltage (V)
    i_abc: NDArray[np.float64]  # instantaneous phase currents (A)
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


def _command_value(output, name):
    commands = output.__dict__.get("commands") or ()
    if name in commands:
        return commands[name]
    raise AttributeError(f"{type(output).__name__!r} has no command {name!r}")


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
    commands: dict[str, float] | None = None

    __getattr__ = _command_value


@dataclass(init=False)
class BridgeControlOutput:
    """One canonical command mapping and the common sampled-control lifecycle outputs.

    ``duty`` remains a constructor argument and property for existing scalar controllers; it
    reads and writes the same mapping used by every other modulation command.
    """

    commands: dict[str, float]
    log: dict[str, float] | None
    theta: float | None
    omega: float | None
    gates: bool
    startup_complete: bool

    def __init__(self, duty: float | None = None, log=None, theta=None, omega=None,
                 gates: bool = True, startup_complete: bool = True,
                 commands: Mapping[str, float] | None = None):
        self.commands = commands if isinstance(commands, dict) else dict(commands or {})
        if duty is not None:
            if "duty" in self.commands and self.commands["duty"] != duty:
                raise ValueError("conflicting duty commands")
            self.commands["duty"] = duty
        self.log, self.theta, self.omega = log, theta, omega
        self.gates, self.startup_complete = gates, startup_complete

    @property
    def duty(self):
        return self.commands.get("duty")

    @duty.setter
    def duty(self, value):
        self.commands["duty"] = value

    __getattr__ = _command_value


@runtime_checkable
class Controller(Protocol):
    """Sampled controller ``(t, Measurement) -> ControlOutput``.

    Optional host-facing methods are ``initialize``, ``command``, ``track``, ``retune``, state access and
    ``summary``; the required target interface is only the sampled call. Protection is deliberately
    outside this interface.
    """

    def __call__(self, t: float, meas: Any) -> Any: ...


@dataclass(frozen=True)
class ControlInterface:
    """Typed directed ports at the boundary of a :class:`ControlGraph`.

    ``inputs`` are sources available to loop input ports. ``outputs`` are controller output ports,
    each driven by one loop output through ``ctrl.outputs``. Direction belongs to the mapping that
    contains a port; its value is the same :class:`SignalType` used by loop port declarations.
    ``optional_inputs`` are schema ports retained at runtime only when an active path uses them.
    ``optional_outputs`` are controller outputs that exist only when ``ctrl.outputs`` wires them.
    """

    inputs: Mapping[str, SignalType]
    outputs: Mapping[str, SignalType]
    optional_inputs: frozenset[str] = frozenset()
    optional_outputs: frozenset[str] = frozenset()

    def auxiliary_outputs(self, configured):
        """Configured outputs beyond the required waveform/control boundary."""
        return tuple(sorted(set(configured) - (self.outputs.keys() - self.optional_outputs)))


# ------------------------------------------------------------------ the start-up sequence

class Startup:
    """Minimal run/ramp state advanced only by this controller's interrupts.

    Only the command, ramp duration and interrupt count are persistent. ``value`` and ``complete``
    describe the current controller call; applying the value to a reference belongs to the loop
    which owns that reference.
    """

    def __init__(self, T: float, run: bool = True, ramp: float = 0.0) -> None:
        self.T = _real(T)
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
    """The active loops of ``cfg.ctrl.loops`` wired by typed ports and run by the control interrupt.

    Loop inputs are held outputs of other loops, the pu measurement (``meas.u_g``, ``.i_c``,
    ``.u_dc``) or the references (``references.<name>``); ``connections`` and ``outputs`` are the
    default wiring, which ``cfg.ctrl.connections`` and ``.outputs`` override. Loops run in signal
    order, every interrupt or every n-th interrupt according to their periods.

    Dependencies are traced backward from controller outputs before loop objects are constructed,
    so declarations outside the active graph use no runtime object. Retuned parameters are queued
    by :meth:`schedule` and take effect at the next update. Changed active loops are rebuilt from
    their new parameters and continue from their named states.
    """

    _cycle_iterations = 16
    _cycle_atol = 1e-10
    _cycle_rtol = 1e-8

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
        self._T = None if self.continuous else _real(cfg.ctrl.period)
        self._t0 = control_clock_offset(cfg.ctrl, cfg.pwm)
        self.values: dict[str, Any] = {}
        self._out_specs: dict[str, tuple] = {}
        self._limits: dict[str, tuple[str, OutputLimit]] = {}
        self.updated = set()
        refs = cfg.ctrl.references
        self.references = {f.name: getattr(refs, f.name) for f in fields(refs)}
        declared_references = {
            path.partition(".")[2] for path in self.interface.inputs
            if path.startswith("references.")
        }
        if declared_references != set(self.references):
            raise RuntimeError("controller reference fields and input ports differ")
        classes = {}
        types = dict(self.interface.inputs)
        for name, loop in cfg.ctrl.loops.items():
            cls = LOOP_TYPES.get(loop.type)
            if cls is None:
                raise ConfigError(f"ctrl.loops.{name}: no loop type {loop.type!r} is registered")
            classes[name] = cls
            types.update({f"{name}.{port}": spec for port, spec in cls.outputs.items()})
        targets = {
            f"{name}.{port}": spec
            for name, cls in classes.items()
            for port, spec in cls.inputs.items()
        }
        unknown_connections = set(self.connections) - set(targets)
        if unknown_connections:
            raise ConfigError(
                f"ctrl.connections: unknown input ports {sorted(unknown_connections)}"
            )
        for target, source in self.connections.items():
            if source not in types:
                raise ConfigError(
                    f"ctrl.connections.{target}: missing or unknown source {source!r}"
                )
            if not types[source].compatible(targets[target]):
                raise ConfigError(
                    f"ctrl.connections.{target}: incompatible signal {source!r}: "
                    f"{types[source]} -> {targets[target]}"
                )
        additional = set(self.outputs) - set(self.interface.outputs)
        reserved = set(ControlOutput.__dataclass_fields__) | set(BridgeControlOutput.__dataclass_fields__)
        if additional & (reserved - {"duty"}):
            raise ConfigError(f"ctrl.outputs: command names conflict with control lifecycle fields: "
                              f"{sorted(additional & (reserved - {'duty'}))}")
        for port in additional:
            source = self.outputs[port]
            if source not in types:
                raise ConfigError(f"ctrl.outputs.{port}: missing or unknown source {source!r}")
            if types[source].complex_value:
                raise ConfigError(f"ctrl.outputs.{port}: an auxiliary register needs a scalar signal")
        if additional:
            self.interface = interface = replace(
                interface,
                outputs=MappingProxyType({**interface.outputs,
                                          **{port: types[self.outputs[port]] for port in additional}}),
                optional_outputs=interface.optional_outputs | additional,
            )
        active = set()
        pending_sources = []
        for port, spec in self.interface.outputs.items():
            source = self.outputs.get(port)
            if source is None and port in self.interface.optional_outputs:
                continue
            if source not in types or not types[source].compatible(spec):
                raise ConfigError(f"ctrl.outputs.{port}: missing or incompatible source {source!r}")
            pending_sources.append(source)
        while pending_sources:
            owner = pending_sources.pop().partition(".")[0]
            if owner not in classes or owner in active:
                continue
            active.add(owner)
            for port in classes[owner].inputs:
                target = f"{owner}.{port}"
                source = self.connections.get(target)
                if source not in types:
                    raise ConfigError(
                        f"ctrl.connections.{target}: missing or unknown source {source!r}"
                    )
                pending_sources.append(source)
        self.connections = {
            target: source for target, source in self.connections.items()
            if target.partition(".")[0] in active
        }
        if interface.optional_inputs:
            used = set(self.connections.values()) | set(self.outputs.values())
            self.interface = ControlInterface(
                MappingProxyType({name: spec for name, spec in interface.inputs.items()
                                  if name not in interface.optional_inputs or name in used}),
                interface.outputs,
                interface.optional_inputs,
                interface.optional_outputs,
            )
        for name, loop in cfg.ctrl.loops.items():
            if name not in active:
                continue
            cls = classes[name]
            self.nodes[name] = node = cls(loop, cfg, startup)
            self._out_specs[name] = tuple((port, f"{name}.{port}", spec.complex_value)
                                          for port, spec in node.outputs.items())
            self.periods[name] = loop.period
            self._store(name, node.initial_outputs(), real=_real, complex=_complex)
        self._samples = {name: node.sample for name, node in self.nodes.items()}
        dependencies = {name: set() for name in self.nodes}
        for name, node in self.nodes.items():
            for port, spec in node.inputs.items():
                target = f"{name}.{port}"
                source = self.connections[target]
                owner = source.partition(".")[0]
                if owner in self.nodes:
                    dependencies[name].add(owner)
        self.order = []
        if not self.continuous:
            pending = {name: set(deps) for name, deps in dependencies.items()}
            while pending:
                ready = [n for n, deps in pending.items() if not deps]
                if not ready:
                    raise ConfigError(
                        "ctrl.connections: instantaneous cycle in sampled control; "
                        "store one dependency's previous sample inside its owning loop"
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
                       tuple((name, int(self.period_stride(T, self._T)))
                             for name, T in self.periods.items()))
        self._every_interrupt = (None if self.continuous else
                                 frozenset(name for name, every in self._every)
                                 if all(every == 1 for _name, every in self._every) else None)
        angles = []
        for name, node in self.nodes.items():
            node_states = node.get_state()
            for port, spec in node.outputs.items():
                if spec.unit != "rad":
                    continue
                if port in node_states:
                    angles.append(f"{name}.{port}")
                angles.append(f"held.{self._state_port(f'{name}.{port}')}")
        self.angle_state_names = tuple(angles)

    @staticmethod
    def period_stride(period, interrupt_period):
        """Interrupt count per loop update; configuration validation requires an integer ratio."""
        return np.maximum(1.0, np.rint(period / interrupt_period))

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
                        _CONSTANT, _complex(value) if spec.complex_value else value
                )
            else:  # ControlInterface is a package-owned declaration, not user input.
                raise RuntimeError(f"unknown controller input owner {owner!r}")
        self._wiring = {}
        for name, node in self.nodes.items():
            wiring = tuple((port,) + self._resolve(self.connections[f"{name}.{port}"]) for port in node.inputs)
            self._wiring[name] = wiring
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
            (group, self._flow_cycle_keys.get(group, ()), tuple(steps[name] for name in group))
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
        for group, _keys, steps in self._flow_plan:
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
            changed = [name for name in self.nodes
                       if ctrl.loops[name] != self.cfg.ctrl.loops[name]]
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

    def _store(self, name, outputs, *, real=float, complex=complex):
        specs = self._out_specs[name]
        values = self.values
        isfinite = math.isfinite
        for port, key, complex_value in specs:
            value = outputs[port]
            if complex_value:
                value = complex(value)
                finite = isfinite(value.real) and isfinite(value.imag)
            else:
                value = real(value)
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
            out = track(self._gather(self._wiring[name], meas)) if track is not None else None
            if out is not None:
                self._store(name, out)
                done.add(name)
        return done

    def reset_integrators(self):
        """Clear every loop's explicitly declared error integrator."""
        for node in self.nodes.values():
            node.reset_integrator()

    def sample(self, t, meas, finalize=None):
        """Run the loops due at ``t`` on the pu measurement ``meas``; return their names.

        ``finalize(t, meas, updated)`` runs after the due loops.
        """
        if self._pending:
            self._apply_pending()
        self.updated = updated = set()
        due = self.due(t)
        if not due:
            return updated
        samples, gather_, wiring = self._samples, self._gather, self._wiring
        for name in self.order:
            if name not in due:
                continue
            self._store(name, samples[name](t, gather_(wiring[name], meas)))
            updated.add(name)
        if finalize is not None:
            finalize(t, meas, updated)
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
        for group, keys, steps in self._flow_plan:
            if only is not None and not any(name in only for name in group):
                continue
            # An acyclic component has no feedback keys, so its first pass has converged.
            previous = tuple(values[key] for key in keys)
            for _iteration in range(self._cycle_iterations):
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
                if all(abs(new - old) <= self._cycle_atol + self._cycle_rtol * max(1.0, abs(new), abs(old))
                       for new, old in zip(current, previous)):
                    break
                previous = current
            else:
                raise FloatingPointError(
                    "continuous control algebraic loop did not converge after "
                    f"{self._cycle_iterations} iterations: "
                    + ", ".join(group)
                )
        return dict(zip(self.flow_state_names, derivatives)) if named else derivatives

    def output(self, port, meas):
        """Return the graph output ``port`` (``u_dq``, ``theta`` or ``omega``)."""
        return self._read(*self._output_source[port], meas)

    def describe(self):
        return {"loops": {n: {"period": self.periods[n]}
                          for n in self.order}, "connections": dict(self.connections), "outputs": dict(self.outputs)}

    # ------------------------------------------------------------ states
    def overwritten_state_names(self, period, *, reads=None):
        """Held values overwritten before every possible read during this map.

        Aligned integer-rate consumers run after their producers whenever they are due.
        Host reads are ``(source, positive_integer_stride)`` pairs after graph updates;
        other strides denote a read before updates or at an unknown rate. Boundary outputs
        stay conservative without a host declaration. The producer must update within the
        map, including for an otherwise unused output's own final state.
        No numerical zero test or operating-point-dependent classification is used.
        """
        if self.continuous or self._pending:
            return frozenset()
        strides = dict(self._every)
        consumers = [(source, strides[target.partition(".")[0]])
                     for target, source in self.connections.items()]
        consumers.extend(((source, None) for source in self.outputs.values())
                         if reads is None else reads)
        retained = {
            source for source, stride in consumers
            if source in self.values
            and (not isinstance(stride, numbers.Integral) or isinstance(stride, bool)
                 or stride <= 0 or stride % strides[source.partition(".")[0]] != 0)
        }
        return frozenset(
            f"held.{self._state_port(source)}"
            for name in self.order if period >= self.periods[name]
            for _port, source, _complex in self._out_specs[name]
            if source not in retained
        )

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

def _role_finder(cfg, label: str):
    """Return a resolver which enforces one configured loop per semantic role."""
    roles = {
        name: getattr(LOOP_TYPES.get(params.type), "role", None)
        for name, params in cfg.ctrl.loops.items()
    }

    def find(role):
        found = [name for name, value in roles.items() if value == role]
        if len(found) > 1:
            raise ConfigError(
                f"ctrl.loops: ambiguous {label} role {found}; use explicit "
                "ctrl.connections and ctrl.outputs"
            )
        return found[0] if found else None

    return find

def default_wiring(cfg, family):
    """Return the default ``(connections, outputs)`` for family ``"gfl"``, ``"gfm"`` or ``"custom"``.

    Each loop is wired by its role (:attr:`peslite.control.loops.Loop.role`), so a registered loop
    type takes the place of the built-in one of its role.
    """
    if family == "custom":
        return {}, {}
    loops = cfg.ctrl.loops
    roles = {name: getattr(LOOP_TYPES.get(p.type), "role", None) for name, p in loops.items()}
    find = _role_finder(cfg, family)
    pll, sync = find("pll"), find("sync")
    angle = pll if family == "gfl" else sync
    frame = f"{angle}.frame" if angle else "references.theta"
    theta = f"{angle}.theta" if angle else "references.theta"
    omega = f"{angle}.omega" if angle else "references.omega"
    dc, cc = find("dc_voltage"), find("current")
    va, vi, damping = find("admittance"), find("impedance"), find("damping")
    extra = f"{damping}.extra" if damping else "references.zero_v_pu"
    ports_of = {
        "pll": {"v": "meas.u_g"},
        "dc_voltage": {"voltage": "meas.u_dc", "reference": "references.vdc_ref_pu"},
        "current": {"v": "meas.u_g", "i": "meas.i_c", "frame": frame, "omega": omega,
                    "extra": extra,
                    "id_ref": (f"{va}.id_ref" if va else f"{dc}.current_reference"
                               if dc else "references.id_ref_pu"),
                    "iq_ref": f"{va}.iq_ref" if va else "references.iq_ref_pu"},
        "sync": {"v": "meas.u_g", "i": "meas.i_c",
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


class _GraphRuntime:
    """Lifecycle and continuous-state mechanics shared by control-graph hosts."""

    outputs_need_inputs = True
    observer = True

    @property
    def discrete_state_names(self) -> tuple[str, ...]:
        """Run/ramp bookkeeping is restored, not differentiated as controller memory."""
        return tuple(f"startup.{name}" for name in self.startup.get_state())

    @property
    def angle_state_names(self) -> tuple[str, ...]:
        """Controller-owned state coordinates whose values are angles in radians."""
        known = set(self.state_names) if self.continuous else set(self.get_state())
        names = [name for name in self.graph.angle_state_names if name in known]
        names.extend(
            f"command.{port}"
            for port, spec in self.interface.outputs.items()
            if spec.unit == "rad" and f"command.{port}" in known
        )
        return tuple(names)

    @property
    def interface(self) -> ControlInterface:
        """The effective typed boundary of the active control graph."""
        return self.graph.interface

    def _install_flow_state(self, inp: Any, out: Any) -> None:
        """Create one fixed ODE layout from the graph's declared loop states."""
        state = _ContinuousControlState()
        names = []
        for name in self.graph.order:
            for local, value in self.graph.nodes[name].flow_state().items():
                if isinstance(value, bool):
                    raise ConfigError(
                        f"ctrl.loops.{name}: boolean state {local!r} "
                        "cannot be a continuous ODE state"
                    )
                key = f"{name}.{local}"
                names.append(key)
                setattr(state, key, _complex(value) if isinstance(value, complex) else _real(value))
        self.state, self.state_names = state, tuple(names)
        if self.state_names != self.graph.flow_state_names:
            raise RuntimeError("continuous controller state and derivative layouts differ")
        self.inp, self.out = inp, out
        self._derivatives = [
            0j if isinstance(getattr(state, name), complex) else 0.0
            for name in self.state_names
        ]
        self._bind_flow_state()

    def _bind_flow_state(self) -> None:
        direct = []
        indirect = []
        known = set(self.state_names)
        for name in self.graph.order:
            node = self.graph.nodes[name]
            values = node.flow_state()
            bindings = node.flow_state_bindings()
            if bindings is None or set(bindings) != set(values):
                indirect.append((node, tuple(
                    (local, f"{name}.{local}") for local in values
                )))
                continue
            for local, (owner, attribute) in bindings.items():
                key = f"{name}.{local}"
                if key not in known:
                    raise ConfigError(
                        f"ctrl.loops.{name}: unknown continuous state binding {local!r}"
                    )
                direct.append((owner, attribute, key))
        self._continuous_direct = tuple(direct)
        self._continuous_indirect = tuple(indirect)

    def _sync_continuous_nodes(self) -> None:
        state = self.state
        for owner, attribute, key in self._continuous_direct:
            setattr(owner, attribute, getattr(state, key))
        for node, bindings in self._continuous_indirect:
            node.set_state({local: getattr(state, key) for local, key in bindings})

    def _store_continuous_nodes(self) -> None:
        known = set(self.state_names)
        for name in self.graph.order:
            for local, value in self.graph.nodes[name].flow_state().items():
                key = f"{name}.{local}"
                if key in known:
                    setattr(self.state, key, value)

    def _loop_retuned(self, name=None) -> None:
        """Rebind continuous state after a set event rebuilt loop ``name``."""
        if not self.continuous:
            return
        if self.graph.flow_state_names != self.state_names:
            raise ConfigError("a set event cannot reorder continuous controller states")
        self._bind_flow_state()

    def retune(self, cfg, paths, t: float) -> None:
        """Queue new controller parameters for the next graph update."""
        if any(path.startswith("ctrl.") for path in paths):
            self.graph.schedule(cfg)
        self.p = cfg

    def initialize(self) -> None:
        """Apply initial parameter events before startup and named-state restoration."""
        self.graph._apply_pending()

    def command(self, run: bool, ramp: float = 0.0, t: float = 0.0, *, reset: bool = True) -> None:
        """Take the host's run or stop command; a continuous controller takes it at absolute
        time ``t`` and clears its integrators on a new run only with ``reset``."""
        starting = bool(run) and not self.startup.active
        if not self.continuous:
            if starting:
                self.graph.reset_integrators()
            self.startup.command(run, ramp)
            return
        if starting and reset:
            self._sync_continuous_nodes()
            self.graph.reset_integrators()
            self._store_continuous_nodes()
        self.startup.command(run, ramp, t)

    command_at = command

    def rhs(self, t):
        return self._derivatives

    def set_logging(self, enabled: bool) -> None:
        """Enable per-interrupt log dictionaries when a run records or watches them."""
        self.logging = bool(enabled)


class VSCController(_GraphRuntime):
    """Configurable closed controller block with loops, start-up and output stage.

    The host gives it only run/stop commands and parameter updates.  With switching or PWM-period
    averaging it is called at each interrupt.  With ideal averaging it is a continuous subsystem:
    its loop states join the plant ODE and its output is the bridge's modulation space vector.
    """

    def __init__(self, cfg, pwm_method=None, *, limiter=CONFIGURED):
        self.p = cfg
        self.continuous = cfg.bridge.model == "averaging"
        # A continuous controller already participates in construction-time energy probes, before
        # VSC.start() applies its scenario. Keep its ideal source off until that command arrives.
        self.startup = ContinuousStartup(run=False) if self.continuous else Startup(cfg.ctrl.period)
        profile = control_type(cfg.ctrl.type)
        wires, outputs = profile.wiring(cfg)
        self.graph = graph = ControlGraph(
            cfg, self.startup, wires, outputs, interface=profile.interface,
        )
        self._carried_outputs = profile.interface.auxiliary_outputs(graph.outputs)
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
        # dq frame of the logged signals, as a graph path that C++ export lowers unchanged: the
        # current loop's frame input, else the angle the theta loop used for its latest update.
        # Resolve it when read so reference changes use the same wiring as graph inputs and outputs.
        self._frame_path = (graph.connections[f"{self._frame_cc}.frame"] if self._frame_cc is not None
                            else self._frame_key if self._frame_key in graph.values
                            else graph.outputs["theta"])
        self._is_gfl = profile.grid_following
        if self._is_gfl:
            mode_log_sources = (
                {"id_ref_pu": graph.connections[f"{self._log_cc}.id_ref"]}
                if self._log_cc is not None else {}
            )
        else:
            mode_log_sources = (
                {"p_pu": f"{self._sync}.p", "q_pu": f"{self._sync}.q",
                 "p_ref_pu": "references.p_ref_pu",
                 "v_ref_pu": f"{self._sync}.v_ref"}
                if self._sync is not None else
                {"p_ref_pu": "references.p_ref_pu",
                 "v_ref_pu": "references.v_ref_pu"}
            )
        self._mode_log_sources = MappingProxyType(mode_log_sources)
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
        if (self.continuous and name == self._dc
                and callable(getattr(node, "observe_limit", None))):
            self._flow_limits[name] = (f"{name}.current_reference", "ID_REF_LIMIT", node)
        if (not self.continuous and name == self._terminal
                and isinstance(node, CurrentLoop) and self.stage is not None):
            limit = self.stage.command_limit(1.0)
            if limit is not None:
                output_limit = OutputLimit(
                    limit, pi=node.pi,
                    active=lambda: self.startup.active,
                )
                graph.limit_output(name, "u_dq", output_limit)
                self._terminal_limit = output_limit
                self._terminal_limit_scale = limit

    def observe_limits(self, t: float) -> None:
        """Check ideal-model commands at a recorded snapshot without altering their values."""
        for output_path, alarm, node in self._flow_limits.values():
            output = self.graph.values[output_path]
            node.n_updates += 1
            node.observe_limit(t, node.raw, output, count=True)
            if not node.limit_exceeded:
                continue
            if alarm not in self.alarms:
                self.alarms.append(alarm)

    def _build_flow_state(self):
        """Check each loop's continuous implementation, then create the fixed ODE state layout."""
        for name in self.graph.order:
            node = self.graph.nodes[name]
            if (type(node).flow is Loop.flow
                    and (node.flow_inputs is None
                         or (type(node).flow_path is Loop.flow_path
                             and type(node).equation is Loop.equation))):
                raise ConfigError(f"ctrl.loops.{name}: loop type {node.type!r} "
                                  "has no continuous-time implementation")
            if isinstance(node, SyncLaw) and type(node)._flow is SyncLaw._flow:
                raise ConfigError(f"ctrl.loops.{name}: loop type {node.type!r} "
                                  "has no continuous-time flow implementation")
        self._install_flow_state(
            _ContinuousControlInp(u_g=0j, i_c=0j, u_dc=self.p.dclink.vdc_ref),
            _ContinuousControlOut(q=0j),
        )
        self._continuous_meas = ControlMeasurement(0.0, 0j, 0j, 1.0, np.zeros(3))
        self._last_control_meas = self._continuous_meas

    def _loop_retuned(self, name=None):
        """Rebuild the loop's limit and continuous state bindings after a set event."""
        self._configure_limit(name)
        super()._loop_retuned(name)

    def describe(self):
        return self.graph.describe()

    def initial_sync(self):
        graph = self.graph
        return _real(graph.output("theta", None)), _real(graph.output("omega", None))

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
        reading = meas.samples[-1] if meas.samples else meas
        if reading.u_g != meas.u_g or reading.i_c != meas.i_c:
            rot = cmath.exp(-1j * theta)
            v, i = reading.u_g * rot, reading.i_c * rot
            out.update({"vd_raw_pu": v.real / self.v_base, "vq_raw_pu": v.imag / self.v_base,
                        "id_raw_pu": i.real / self.i_base, "iq_raw_pu": i.imag / self.i_base})
        if reading.u_dc != meas.u_dc:
            out["vdc_raw_pu"] = reading.u_dc / self.v_dc_base
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

    def mode_log_sources(self) -> Mapping[str, str]:
        """Return graph sources for control-family-specific log fields."""
        return self._mode_log_sources

    def _mode_log(self, t: float, meas: ControlMeasurement) -> dict[str, float]:
        """Evaluate the control-family-specific part of one controller log."""
        graph = self.graph
        values = {
            name: graph._read(*graph._resolve(source), meas)
            for name, source in self._mode_log_sources.items()
        }
        if self._is_gfl:
            values.setdefault("id_ref_pu", 0.0)
        else:
            values.setdefault("p_pu", 0.0)
            values.setdefault("q_pu", 0.0)
            values.setdefault("v_ref_pu", graph.references["v_ref_pu"])
            values["p_ref_pu"] *= self.startup.value
        values["freq_dev"] = (self.omega - self.w0) / (2 * math.pi)
        values["angle_rel"] = (
            self.theta - self.w0 * t + math.pi
        ) % (2 * math.pi) - math.pi
        return values

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

    def __call__(self, t, meas):
        if self.continuous:
            raise RuntimeError("an ideal averaged controller is evaluated by the ODE solver, not sampled")
        control_meas = self._pu(meas)
        startup = self.startup
        if startup.in_progress:
            startup.advance()
        if self._terminal_limit is not None:
            self._terminal_limit.limit = (
                self._terminal_limit_scale * max(0.0, control_meas.u_dc)
            )
        graph = self.graph
        graph.sample(t, control_meas, finalize=self._accept_command)
        frame = graph._read(*graph._resolve(self._frame_path), control_meas)
        rot = cmath.exp(-1j * frame)
        self.v_dq, self.i_dq = control_meas.u_g * rot, control_meas.i_c * rot
        gates = startup.active
        duty = self.stage.modulate(t, self.u_cmd, self.command_theta, control_meas.u_dc,
                                   count=gates)
        log = None
        if self.logging:
            log = {"id_pu": self.i_dq.real, "iq_pu": self.i_dq.imag,
                   "vd_pu": self.v_dq.real, "vq_pu": self.v_dq.imag,
                   "vac_pu": abs(self.v_dq), "vdc_pu": control_meas.u_dc,
                   "m_max": peak_abs(self.stage.m_abc),
                   "in_service": 1.0 if gates else 0.0,
                   **self._raw_log(meas, frame)}
            log.update(self._mode_log(t, control_meas))
            self.last_log = log
        commands = ({port: self.graph.output(port, control_meas) for port in self._carried_outputs}
                    if self._carried_outputs else None)
        return ControlOutput(duty, log=log, theta=self.theta, omega=self.omega,
                             gates=gates, startup_complete=startup.complete, commands=commands)

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
        read, source = graph._read, graph._output_source
        self.theta = float(read(*source["theta"], meas))
        self.omega = float(read(*source["omega"], meas))
        self.u_cmd = complex(read(*source["u_dq"], meas))
        self.command_theta = self.theta
        frame = read(*graph._resolve(self._frame_path), meas)
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

    def continuous_log(self, t):
        """Controller observation at an output snapshot of the continuous model."""
        if not self.logging:
            return None
        meas = self._last_control_meas
        log = {"id_pu": self.i_dq.real, "iq_pu": self.i_dq.imag,
               "vd_pu": self.v_dq.real, "vq_pu": self.v_dq.imag,
               "vac_pu": abs(self.v_dq), "vdc_pu": meas.u_dc,
               "in_service": 1.0 if self.startup.active else 0.0}
        log.update(self._mode_log(t, meas))
        self.last_log = log
        return log

    # ------------------------------------------------------------ states and summary
    def overwritten_state_names(self, period):
        graph = self.graph
        # Frames are read after each sample; the voltage command only after its producer runs.
        reads = [(self._frame_key, 1), (graph.connections.get(f"{self._frame_cc}.frame"), 1),
                 (graph.outputs["u_dq"], dict(graph._every).get(self._terminal, 1))]
        # initial_sync() also supplies the bridge before the first graph update.
        reads.extend((graph.outputs[port], None) for port in ("theta", "omega"))
        return graph.overwritten_state_names(period, reads=reads)

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
            summary["id_ref_limit_first_t"] = (
                dc.first_limit_t if math.isfinite(dc.first_limit_t) else None
            )
        if self._sync is not None:
            summary["law"] = self.p.ctrl.loops[self._sync].type
        summary["alarms"] = list(self.alarms)
        return summary


_CONVERTER_MEASUREMENT_INPUTS = {
    "meas.current": SI_CURRENT,
    "meas.voltage": SI_VOLTAGE,
    "meas.vin": SI_VOLTAGE,
    "meas.power": SI_POWER,
}


DC_CONVERTER_INTERFACE = ControlInterface(
    inputs=MappingProxyType({
        **_CONVERTER_MEASUREMENT_INPUTS,
        "meas.duty_feedforward": DUTY_RATIO,
        "references.voltage": SI_VOLTAGE,
    }),
    outputs=MappingProxyType({"duty": DUTY_RATIO}),
    optional_inputs=frozenset(("meas.power",)),
)

RESONANT_CONVERTER_INTERFACE = ControlInterface(
    inputs=MappingProxyType({
        **_CONVERTER_MEASUREMENT_INPUTS,
        "references.modulation": MODULATION,
        "references.frequency": HERTZ,
        "references.phase": ANGLE,
        "references.power": SI_POWER,
    }),
    outputs=MappingProxyType({"duty": DUTY_RATIO}),
    optional_inputs=frozenset(("meas.power",)),
)


OPEN_LOOP_CONVERTER_INTERFACE = ControlInterface(
    inputs=MappingProxyType({
        **_CONVERTER_MEASUREMENT_INPUTS,
        "references.duty": DUTY_RATIO,
    }),
    outputs=MappingProxyType({"duty": DUTY_RATIO}),
    optional_inputs=frozenset(("meas.power",)),
)


PHASE_SHIFT_CONVERTER_INTERFACE = ControlInterface(
    inputs=MappingProxyType({
        **_CONVERTER_MEASUREMENT_INPUTS,
        "references.width": MODULATION,
        "references.frequency": HERTZ,
        "references.phase": ANGLE,
        "references.power": SI_POWER,
        "references.mode": MODE,
    }),
    outputs=MappingProxyType({"width": MODULATION, "phase": ANGLE, "frequency": HERTZ,
                              "mode": MODE}),
    optional_inputs=frozenset(("meas.power",)),
)


def _converter_log_sources(outputs: tuple[str, ...] = ("duty",),
                           **specific: str) -> Mapping[str, str]:
    """Build the common scalar-converter log boundary in display order."""
    return MappingProxyType({
        "current": "meas.current",
        "voltage": "meas.voltage",
        "power": "meas.power",
        **specific,
        **{name: name for name in outputs},
        "in_service": "startup.active",
    })


_DC_LOG_SOURCES = _converter_log_sources(
    voltage_reference="references.voltage",
    input_voltage="meas.vin",
)


_RESONANT_LOG_SOURCES = _converter_log_sources(
    input_voltage="meas.vin",
    modulation="references.modulation",
    frequency="references.frequency",
    phase="references.phase",
    power_reference="references.power",
)


_PHASE_SHIFT_LOG_SOURCES = _converter_log_sources(
    ("width", "phase", "frequency", "mode"),
    input_voltage="meas.vin",
    width_reference="references.width",
    phase_reference="references.phase",
    power_reference="references.power",
)


def _modulation_wiring(cfg):
    """Wire the modulation-role loop: each of its inputs named like a reference of the control
    type reads that reference, and each controller output reads the loop output of its name."""
    find = _role_finder(cfg, cfg.ctrl.type)
    modulation = find("modulation")
    if modulation is None:
        return {}, {}
    interface = control_type(cfg.ctrl.type).interface
    loop = LOOP_TYPES[cfg.ctrl.loops[modulation].type]
    connections = {f"{modulation}.{port}": f"references.{port}" for port in loop.inputs
                   if f"references.{port}" in interface.inputs}
    outputs = {port: f"{modulation}.{port}" for port in interface.outputs if port in loop.outputs}
    return connections, outputs


def _dc_converter_wiring(cfg):
    find = _role_finder(cfg, cfg.ctrl.type)
    voltage, current = find("dc_voltage"), find("current")
    wires = {}
    if voltage is not None:
        wires.update({
            f"{voltage}.reference": "references.voltage",
            f"{voltage}.voltage": "meas.voltage",
        })
    if current is not None:
        wires.update({
            f"{current}.reference": (
                f"{voltage}.current_reference" if voltage is not None else None
            ),
            f"{current}.current": "meas.current",
            f"{current}.feedforward": "meas.duty_feedforward",
        })
    return wires, ({"duty": f"{current}.duty"} if current is not None else {})


@dataclass
class ConverterMeasurement:
    """SI sample at a scalar converter's public control boundary."""

    t: float = 0.0
    current: float = 0.0
    voltage: float = 0.0
    vin: float = 0.0
    duty_feedforward: float = 0.0
    power: float | None = None
    samples: tuple["ConverterMeasurement", ...] = ()


class _ConverterControlInp(Bag):
    __slots__ = ("current", "voltage", "vin", "power_voltage", "power_current", "power")


class GraphController(_GraphRuntime):
    """Run a DC converter's registered loops through the shared control graph.

    This class supplies only the plant boundary and lifecycle. Feedback and open-loop laws are both
    ordinary graph loops; no application-specific controller bypasses the graph.
    """

    def __init__(self, cfg, topology) -> None:
        self.p = cfg
        self.topology = topology
        self.continuous = cfg.bridge.model == "averaging"
        self.startup = (
            ContinuousStartup(run=False) if self.continuous
            else Startup(cfg.ctrl.period, run=False)
        )
        profile = control_type(cfg.ctrl.type)
        connections, outputs = profile.wiring(cfg)
        self.graph = ControlGraph(
            cfg,
            self.startup,
            connections,
            outputs,
            interface=profile.interface,
        )
        self.outputs_need_inputs = any(
            isinstance(source, str) and source.startswith("meas.")
            for source in (*self.graph.connections.values(), *self.graph.outputs.values())
        )
        self.graph.on_retune = self._loop_retuned
        self.periods = self.graph.periods
        self.logging = True
        self.last_log: dict[str, float] = {}
        self._input_values = topology.control_values
        self._needs_power = "meas.power" in self.graph.interface.inputs
        self._bound_power = "power" in cfg.meas.sources
        # Follow overridden default wires when observing a commanded quantity. Observations
        # belong to port wiring and loop declarations, not to application or signal types.
        observed = {source: self.graph.connections[target]
                    for target, source in connections.items()
                    if target in self.graph.connections}
        used = set(self.graph.connections.values()) | set(self.graph.outputs.values())
        self._log_sources = MappingProxyType({
            **{name: observed.get(source, source)
               for name, source in profile.log_sources.items()
               if (not source.startswith("meas.") or source in self.graph.interface.inputs)
               and (not source.startswith("references.") or source in observed or source in used)},
            **{f"{name}.{port}": f"{name}.{port}"
               for name, node in self.graph.nodes.items() for port in node.log_outputs},
        })
        self._logs_loop_outputs = any(
            source in self.graph.values for source in self._log_sources.values()
        )
        self._last_measurement = ConverterMeasurement(vin=cfg.dclink.vdc_ref)
        self._command_ports = tuple(port for port in self.graph.interface.outputs
                                    if port in self.graph.outputs)
        self._last_outputs = {port: (topology.neutral_duty if port == "duty" else 0.0)
                              for port in self._command_ports}
        if self.continuous:
            self._install_flow_state(
                _ConverterControlInp(current=0.0, voltage=0.0, vin=cfg.dclink.vdc_ref,
                                     power_voltage=0.0, power_current=0.0, power=0.0),
                _ContinuousControlOut(q=topology.zero),
            )

    def control_inputs(self, current, voltage, vin, power=None) -> dict[str, Any]:
        """Return canonical measurements and converter feedforward at the control boundary."""
        measurement = self._measurement(current, voltage, vin, power=power)
        return {name: getattr(measurement, name.partition(".")[2])
                for name in self.graph.interface.inputs
                if name.startswith("meas.")}

    def _measurement(self, current: float, voltage: float, vin: float,
                     target: ConverterMeasurement | None = None,
                     power: float | None = None) -> ConverterMeasurement:
        if self._needs_power and power is None:
            raise ValueError("meas.power requires a measured electrical-port power in watts")
        current, voltage, vin, feedforward = self._input_values(current, voltage, vin)
        measurement = target if target is not None else ConverterMeasurement()
        measurement.current = current
        measurement.voltage = voltage
        measurement.vin = vin
        measurement.duty_feedforward = feedforward
        if self._needs_power:
            measurement.power = power
        return measurement

    def bind_power_port(self, port: PowerPort) -> None:
        """Observe outward power with the connected terminal's sign and scale."""
        self._power_port = PowerPort(
            "inp.power_voltage", "inp.power_current", port.scale, -port.sign,
        )

    def _flow_measurement(self) -> ConverterMeasurement:
        power = ((self.inp.power if self._bound_power else port_power(self, self._power_port))
                 if self._needs_power else None)
        return self._measurement(
            self.inp.current, self.inp.voltage, self.inp.vin, self._last_measurement,
            power=power,
        )

    def sample(self, t: float, current: float, voltage: float, vin: float,
               power: float | None = None) -> BridgeControlOutput:
        """Evaluate one sampled control interrupt from physical plant values."""
        if self.continuous:
            raise RuntimeError("an ideal averaged converter controller is evaluated by the ODE solver")
        measurement = self._measurement(current, voltage, vin, self._last_measurement, power=power)
        if self.startup.in_progress:
            self.startup.advance()
        self.graph.sample(t, measurement)
        outputs = {port: float(self.graph.output(port, measurement))
                   for port in self._command_ports}
        self._last_measurement, self._last_outputs = measurement, outputs
        gates = self.startup.active
        log = self._log(outputs, gates) if self.logging else None
        return BridgeControlOutput(
            log=log,
            gates=gates,
            startup_complete=self.startup.complete,
            commands=outputs,
        )

    def __call__(self, t: float, meas) -> BridgeControlOutput:
        return self.sample(t, meas.current, meas.voltage, meas.vin, getattr(meas, "power", None))

    def set_outputs(self, t: float) -> None:
        self._sync_continuous_nodes()
        self.startup.at(t)
        measurement = self._flow_measurement()
        self.graph.flow(t, measurement, derivatives=self._derivatives)
        duty = float(self.graph.output("duty", measurement))
        self.out.q = self.topology.state_from_duty(duty) if self.startup.active else self.topology.zero
        self._last_measurement = measurement
        self._last_outputs["duty"] = duty

    def _log(self, outputs: Mapping[str, float], active: bool) -> dict[str, float]:
        measurement = self._last_measurement
        values = {
            "meas.current": measurement.current,
            "meas.voltage": measurement.voltage,
            "meas.vin": measurement.vin,
            **outputs,
            "startup.active": 1.0 if active else 0.0,
        }
        if self._needs_power:
            values["meas.power"] = measurement.power
        values.update({f"references.{name}": value
                       for name, value in self.graph.references.items()})
        if self._logs_loop_outputs:
            values.update(self.graph.values)
        self.last_log = {name: values[source] for name, source in self._log_sources.items()}
        return self.last_log

    def continuous_log(self, t: float) -> dict[str, float] | None:
        # Non-loading/open-loop outputs may be evaluated before unused inputs are wired. The
        # model has synchronized every input by the time it asks for a snapshot observation.
        if self.logging:
            self._flow_measurement()
        return self._log(self._last_outputs, self.startup.active) if self.logging else None

    def log_names(self) -> tuple[str, ...]:
        return tuple(self._log_sources)

    def log_sources(self) -> Mapping[str, str]:
        """Map log columns to public control ports or controller lifecycle values."""
        return self._log_sources

    def get_state(self) -> dict[str, Any]:
        startup = {f"startup.{name}": value for name, value in self.startup.get_state().items()}
        return startup if self.continuous else {**self.graph.get_state(), **startup}

    def overwritten_state_names(self, period):
        # sample() consumes every boundary output after graph.sample() on each interrupt.
        return self.graph.overwritten_state_names(
            period, reads=((source, 1) for source in self.graph.outputs.values()),
        )

    def set_state(self, values: Mapping[str, Any]) -> None:
        startup = {name[8:]: value for name, value in values.items() if name.startswith("startup.")}
        self.startup.set_state(startup)
        if not self.continuous:
            self.graph.set_state({name: value for name, value in values.items()
                                  if not name.startswith("startup.")})

    def summary(self) -> dict[str, Any]:
        return {"alarms": []}


def make_controller(cfg, **kwargs):
    """Build the controller registered for ``cfg.ctrl.type``."""
    factory = control_type(cfg.ctrl.type).controller_factory
    if factory is None:
        raise ConfigError(f"ctrl.type: {cfg.ctrl.type!r} has no controller factory")
    return factory(cfg, **kwargs)


for _control_name in ("gfl", "gfm", "custom"):
    register_control_type(
        _control_name,
        references=ReferenceParams,
        unit_types=frozenset(("vsc",)),
        interface=CONTROL_INTERFACE,
        wiring=lambda cfg, family=_control_name: default_wiring(cfg, family),
        grid_following=_control_name == "gfl",
        default_bridge="three_phase_bridge",
        controller_factory=VSCController,
    )
register_control_type(
    "dc",
    references=ScalarVoltageReferenceParams,
    unit_types=frozenset(("dc_converter",)),
    interface=DC_CONVERTER_INTERFACE,
    wiring=_dc_converter_wiring,
    log_sources=_DC_LOG_SOURCES,
    default_bridge="half_bridge",
    controller_factory=GraphController,
    reference_validator=validate_scalar_voltage_references,
)
register_control_type(
    "open_loop",
    references=DutyReferenceParams,
    unit_types=frozenset(("dc_converter",)),
    interface=OPEN_LOOP_CONVERTER_INTERFACE,
    wiring=_modulation_wiring,
    log_sources=_converter_log_sources(input_voltage="meas.vin",
                                       duty_reference="references.duty"),
    default_bridge="half_bridge",
    controller_factory=GraphController,
    reference_validator=validate_duty_references,
)
register_control_type(
    "phase_shift",
    references=PhaseShiftReferenceParams,
    unit_types=frozenset(("dc_converter",)),
    interface=PHASE_SHIFT_CONVERTER_INTERFACE,
    wiring=_modulation_wiring,
    log_sources=_PHASE_SHIFT_LOG_SOURCES,
    default_bridge="full_bridge",
    controller_factory=GraphController,
    reference_validator=validate_phase_shift_references,
    modulator="phase_shift",
)
register_control_type(
    "resonant",
    references=ResonantReferenceParams,
    unit_types=frozenset(("dc_converter",)),
    interface=RESONANT_CONVERTER_INTERFACE,
    wiring=_modulation_wiring,
    log_sources=_RESONANT_LOG_SOURCES,
    default_bridge="full_bridge",
    controller_factory=GraphController,
    reference_validator=validate_resonant_references,
)
