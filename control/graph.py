"""Control graph: typed port wiring between loops, held outputs and per-loop clocks."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from ..phs.states import gather, scatter
from ..params import ConfigError, to_dict
from ..assembly.protocols import ControlMeasurement
from .loops import (LOOP_TYPES, V_AB, I_AB, V_DQ, VOLTAGE, CURRENT, ANGLE,
                    FREQUENCY, POWER_PU, DC_VOLTAGE)


def default_wiring(cfg, family):
    """Return the default ``(connections, outputs)`` for family ``"gfl"``, ``"gfm"`` or ``"custom"``."""
    if family == "custom":
        return {}, {}
    loops = cfg.control.loops
    def find(*kinds):
        found = [n for n, p in loops.items() if p.type in kinds]
        if len(found) > 1:
            raise ConfigError(f"control.loops: ambiguous {family} role {found}; use control.type = 'custom' "
                              "and explicit connections/outputs for multiple instances of this role")
        return found[0] if len(found) == 1 else None
    pll = find("srf_pll")
    sync = find("psc", "droop", "vsg", "dvoc", "matching")
    angle = pll if family == "gfl" else sync
    frame = f"{angle}.frame" if angle else "references.theta"
    theta = f"{angle}.theta" if angle else "references.theta"
    omega = f"{angle}.omega" if angle else "references.omega"
    dc, cc, power = find("dc_voltage_pi"), find("dq_current_pi"), find("power")
    va, vi, damping = find("virtual_admittance"), find("virtual_impedance"), find("active_damping")
    extra = f"{damping}.extra" if damping else "references.zero_v_pu"
    wires = {}
    for name, p in loops.items():
        kind = p.type
        ports = {}
        if kind == "srf_pll":
            ports = {"v": "measurement.u_g"}
        elif kind == "dc_voltage_pi":
            ports = {"u_dc": "measurement.u_dc", "vdc_ref": "references.vdc_ref_pu"}
        elif kind == "dq_current_pi":
            ports = {"v": "measurement.u_g", "i": "measurement.i_c", "frame": frame, "omega": omega,
                     "extra": extra, "id_ref": f"{va or dc}.id_ref" if va or dc else "references.id_ref_pu",
                     "iq_ref": f"{va}.iq_ref" if va else "references.iq_ref_pu"}
        elif kind == "power":
            ports = {"v": "measurement.u_g", "i": "measurement.i_c"}
        elif kind in ("psc", "droop", "vsg", "dvoc", "matching"):
            ports = {"p": f"{power}.p", "q": f"{power}.q", "v": "measurement.u_g", "i": "measurement.i_c",
                     "u_dc": "measurement.u_dc", "p_ref": "references.p_ref_pu", "q_ref": "references.q_ref_pu",
                     "v_ref": "references.v_ref_pu", "theta_seed": "references.theta"}
        elif kind in ("virtual_impedance", "virtual_admittance"):
            ports = {"v_ref": f"{sync}.v_ref", "frame": frame, "omega": omega}
            if kind == "virtual_impedance":
                ports.update(i="measurement.i_c", extra=extra)
            else:
                ports.update(v="measurement.u_g")
        elif kind == "active_damping":
            ports = {"i": "measurement.i_c", "frame": frame}
        wires.update({f"{name}.{port}": source for port, source in ports.items()})
    terminal = vi if family == "gfm" and vi else cc
    return wires, {"u_dq": f"{terminal}.u_dq", "theta": theta, "omega": omega}


# Kinds of input source.
_HELD, _MEASURED, _CONSTANT = 0, 1, 2
_MEASUREMENTS = {"measurement.u_g": "u_g", "measurement.i_c": "i_c", "measurement.u_dc": "u_dc"}
_PU_ONLY = "ControlGraph requires ControlMeasurement in pu; convert SI samples at the firmware boundary"


class ControlGraph:
    def __init__(self, cfg, scenario, connections, outputs, overrides=None):
        self.cfg = cfg
        overrides = {} if overrides is None else overrides
        unknown = set(overrides) - set(cfg.control.loops)
        if unknown:
            raise ConfigError(f"loop_overrides: unknown loop instances {sorted(unknown)}")
        self.connections = {**connections, **cfg.control.connections}
        self.outputs = {**outputs, **cfg.control.outputs}
        self.nodes = {}
        self.definitions = {}
        self.periods = {}
        self.ticks = {}
        self.values: dict[str, Any] = {}
        self._out_specs: dict[str, tuple] = {}
        self.updated = set()
        self.references = to_dict(cfg.control.references)
        ref_types = {"id_ref_pu": CURRENT, "iq_ref_pu": CURRENT, "theta": ANGLE, "omega": FREQUENCY,
                     "p_ref_pu": POWER_PU, "q_ref_pu": POWER_PU, "v_ref_pu": VOLTAGE,
                     "vdc_ref_pu": DC_VOLTAGE, "zero_v_pu": V_DQ}
        unknown_refs = set(self.references) - set(ref_types)
        if unknown_refs:
            raise ConfigError(f"control.references: unknown references {sorted(unknown_refs)}")
        types = {"measurement.u_g": V_AB, "measurement.i_c": I_AB, "measurement.u_dc": DC_VOLTAGE,
                 **{f"references.{k}": v for k, v in ref_types.items()}}
        for name, loop in cfg.control.loops.items():
            definition = LOOP_TYPES.get(loop.type)
            if definition is None:
                raise ConfigError(f"control.loops.{name}: no runtime registered for {loop.type!r}")
            if loop.type == "unit_delay":
                signal = {"current_pu": CURRENT, "voltage_pu": VOLTAGE, "dc_voltage_pu": DC_VOLTAGE,
                          "voltage_dq_pu": V_DQ, "voltage_ab_pu": V_AB, "current_ab_pu": I_AB,
                          "angle": ANGLE, "frequency": FREQUENCY, "power_pu": POWER_PU}[loop.signal]
                definition = replace(definition, inputs={"value": signal}, outputs={"value": signal})
            self.definitions[name] = definition
            self._out_specs[name] = tuple((port, f"{name}.{port}", spec.complex_value)
                                          for port, spec in definition.outputs.items())
            self.nodes[name] = node = definition.factory(loop, cfg, scenario)
            if name in overrides:
                if getattr(node, "obj", None) is None:
                    raise ConfigError(f"loop_overrides.{name}: this loop has no replaceable algorithm; "
                                      "register a custom loop type instead")
                node.obj = overrides[name]
            self.periods[name] = loop.period
            self.ticks[name] = 1
            self._store(name, node.initial_outputs())
            types.update({f"{name}.{port}": spec for port, spec in definition.outputs.items()})
        dependencies = {name: set() for name in self.nodes}
        targets = set()
        for name, definition in self.definitions.items():
            for port, spec in definition.inputs.items():
                target = f"{name}.{port}"
                targets.add(target)
                source = self.connections.get(target)
                if source not in types:
                    raise ConfigError(f"control.connections.{target}: missing or unknown source {source!r}")
                if types[source] != spec:
                    raise ConfigError(f"control.connections.{target}: incompatible signal {source!r}: "
                                      f"{types[source]} -> {spec}")
                owner = source.partition(".")[0]
                if owner in self.nodes and port not in definition.delayed_inputs:
                    dependencies[name].add(owner)
        if set(self.connections) - targets:
            raise ConfigError(f"control.connections: unknown input ports {sorted(set(self.connections) - targets)}")
        for port, spec in {"u_dq": V_DQ, "theta": ANGLE, "omega": FREQUENCY}.items():
            source = self.outputs.get(port)
            if source not in types or types[source] != spec:
                raise ConfigError(f"control.outputs.{port}: missing or incompatible source {source!r}")
        if set(self.outputs) - {"u_dq", "theta", "omega"}:
            raise ConfigError("control.outputs: only u_dq, theta and omega are supported")
        self.order = []
        pending = dict(dependencies)
        while pending:
            ready = [n for n, deps in pending.items() if not deps]
            if not ready:
                raise ConfigError("control.connections: instantaneous cycle; insert an explicit unit_delay")
            for name in ready:
                self.order.append(name)
                pending.pop(name)
            for deps in pending.values():
                deps.difference_update(ready)
        # Resolved input sources per loop.
        self._constants = {f"references.{k}": complex(v) if k == "zero_v_pu" else v
                           for k, v in self.references.items()}
        self._wiring = {}
        self._immediate = {}
        self._port_source = {}
        for name, definition in self.definitions.items():
            wiring = tuple((port,) + self._resolve(self.connections[f"{name}.{port}"])
                           for port in definition.inputs)
            self._wiring[name] = wiring
            self._immediate[name] = tuple(w for w in wiring if w[0] not in definition.delayed_inputs)
            self._port_source[name] = {w[0]: w[1:] for w in wiring}
        self._latching = frozenset(n for n, d in self.definitions.items() if d.delayed_inputs)
        self._output_source = {port: self._resolve(source) for port, source in self.outputs.items()}
        self._all_clocked = all(T is not None for T in self.periods.values())
        self._clocked = tuple((n, T) for n, T in self.periods.items() if T)  # the loops with a clock
        self._retime()

    def _resolve(self, source):
        if source in _MEASUREMENTS:
            return _MEASURED, _MEASUREMENTS[source]
        if source in self._constants:
            return _CONSTANT, self._constants[source]
        return _HELD, source

    def _read(self, kind, key, meas):
        if kind == _HELD:
            return self.values[key]
        if kind == _MEASURED:
            return getattr(meas, key)
        return key

    def _store(self, name, outputs):
        specs = self._out_specs[name]
        if len(outputs) != len(specs) or any(port not in outputs for port, _key, _c in specs):
            raise ValueError(f"control loop {name}: outputs must be {sorted(self.definitions[name].outputs)}")
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

    def reset_clocks(self, t):
        self.ticks = {name: int(math.floor(t / T + 1e-9)) + 1 if T else 1
                      for name, T in self.periods.items()}
        self._retime()

    def _retime(self):
        """Recompute :attr:`next_event` from the clocks."""
        ticks = self.ticks
        self._next_event = min([ticks[n] * T for n, T in self._clocked], default=math.inf)

    @property
    def next_event(self):
        """Earliest time (s) at which a clocked loop is due."""
        return self._next_event

    def sources(self, meas):
        if not isinstance(meas, ControlMeasurement):
            raise TypeError("ControlGraph requires ControlMeasurement in pu; convert SI samples at the firmware boundary")
        return {**self.values, "measurement.u_g": meas.u_g, "measurement.i_c": meas.i_c,
                "measurement.u_dc": meas.u_dc,
                **{f"references.{k}": complex(v) if k == "zero_v_pu" else v for k, v in self.references.items()}}

    def _gather(self, wiring, meas):
        values = self.values
        out = {}
        for port, kind, key in wiring:
            if kind == _HELD:
                out[port] = values[key]
            elif kind == _MEASURED:
                out[port] = getattr(meas, key)
            else:
                out[port] = key
        return out

    def inputs(self, name, meas):
        if not isinstance(meas, ControlMeasurement):
            raise TypeError(_PU_ONLY)
        return self._gather(self._wiring[name], meas)

    def input(self, name, port, meas):
        """Return one input of a loop, equal to ``inputs(name, meas)[port]``."""
        if not isinstance(meas, ControlMeasurement):
            raise TypeError(_PU_ONLY)
        return self._read(*self._port_source[name][port], meas)

    def update(self, t, meas, finalize=None):
        self.updated = updated = set()
        periods, ticks = self.periods, self.ticks
        due = {n for n, T in self._clocked if ticks[n] * T <= t + 1e-10}
        if not due and self._all_clocked:
            return updated
        if not isinstance(meas, ControlMeasurement):
            raise TypeError(_PU_ONLY)
        nodes, gather_, immediate = self.nodes, self._gather, self._immediate
        for name in self.order:
            T = periods[name]
            if name not in due and T is not None:
                continue
            self._store(name, nodes[name].update(t, gather_(immediate[name], meas)))
            updated.add(name)
            if T:
                ticks[name] += 1
        if due:
            self._retime()
        if finalize is not None:
            finalize(t, meas, updated)
        for name in due:
            if name in self._latching:
                nodes[name].latch(gather_(self._wiring[name], meas))
        return updated

    def output(self, port, meas):
        if not isinstance(meas, ControlMeasurement):
            raise TypeError(_PU_ONLY)
        return self._read(*self._output_source[port], meas)

    def describe(self):
        return {"loops": {n: {"type": self.cfg.control.loops[n].type, "period": self.periods[n]}
                          for n in self.order}, "connections": dict(self.connections), "outputs": dict(self.outputs)}

    def get_state(self):
        state = gather(self.nodes)
        state.update({f"held.{self._state_port(k)}": v for k, v in self.values.items()})
        state.update({f"clock.{n}": k for n, k in self.ticks.items() if self.periods[n]})
        return state

    def set_state(self, values):
        unknown = set(values) - set(self.get_state())
        if unknown:
            raise KeyError(f"unknown control states {sorted(unknown)}; electrical held outputs and converted states use _pu")
        state_ports = {self._state_port(k): k for k in self.values}
        rest = {}
        for key, value in values.items():
            if key.startswith("held."):
                if key[5:] not in state_ports:
                    raise KeyError(f"unknown held control output {key[5:]}; electrical held values now require the _pu suffix")
                port = state_ports[key[5:]]
                self.values[port] = value
            elif key.startswith("clock."):
                name = key[6:]
                if name not in self.ticks or int(value) != value or value < 1:
                    raise ValueError(f"invalid control clock {key}: {value}")
                self.ticks[name] = int(value)
            else:
                rest[key] = value
        self._retime()
        scatter(self.nodes, rest)
        # Refresh held angle outputs that were not set explicitly.
        for name, node in self.nodes.items():
            if self.cfg.control.loops[name].type in ("srf_pll", "psc", "droop", "vsg", "dvoc", "matching"):
                for port, value in node.initial_outputs().items():
                    if f"held.{self._state_port(name + '.' + port)}" not in values:
                        self.values[f"{name}.{port}"] = value

    def _state_port(self, key):
        name, port = key.split(".", 1)
        return key + "_pu" if self.definitions[name].outputs[port].unit.startswith("pu_") else key
