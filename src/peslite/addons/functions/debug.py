"""File-backed control-graph debugging built on the ordinary PESLite runtime."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import re
from dataclasses import dataclass, field, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from ... import __version__
from ...assembly.params import Params, _unit_items, flat_paths, to_dict, unit_type
from ...assembly.syntax import parse_parameter_mapping
from ...control.controller import ControlGraphOverlay, continuous_control
from ...control.loops import LOOP_TYPES
from ...assembly.control import control_profile, resolved_control_interface
from ...solver.signals import TableSignal
from ...solver.model import ConfigError
from ...solver.simulation import Simulation, validate_model


_SECTIONS = ("debug", "sources", "probe", "analysis", "checks")
_HEADER = re.compile(r"^@([A-Za-z_][\w-]*)\s*(?:#.*)?$")
_NAME = re.compile(r"^[A-Za-z_][\w-]*$")
_PATH = r"[A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)+"
_PATH_NAME = re.compile(rf"^{_PATH}$")
_WIRE = re.compile(rf"^([+-])\s*({_PATH})\s*->\s*({_PATH})\s*$")
_SET = re.compile(rf"^~\s*({_PATH})\s*=\s*(.+)$")
_OPERATORS = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "==": operator.eq, "!=": operator.ne,
}


@dataclass(frozen=True)
class DebugOperation:
    """One connection edit or parameter replacement from ``@debug``."""

    kind: str
    source: str = ""
    target: str = ""
    path: str = ""
    value: Any = None


@dataclass(frozen=True)
class DebugPlan:
    """Parsed, serializable debug overlay; the simulation file remains unchanged."""

    model: str = ""
    state_policy: str = "continue"
    operations: tuple[DebugOperation, ...] = ()
    sources: Mapping[str, Any] = field(default_factory=dict)
    probes: Mapping[str, str] = field(default_factory=dict)
    analyses: Mapping[str, Any] = field(default_factory=dict)
    checks: Mapping[str, Any] = field(default_factory=dict)

    def with_updates(self, *, model: str | None = None,
                     operations: tuple[DebugOperation, ...] = (),
                     probes: Mapping[str, str] | None = None,
                     analyses: tuple[str, ...] = ()) -> "DebugPlan":
        analysis_values = dict(self.analyses)
        for name in analyses:
            analysis_values.setdefault(name, {"type": name})
        return replace(
            self,
            model=self.model if model is None else model,
            operations=(*self.operations, *operations),
            probes={**self.probes, **(probes or {})},
            analyses=analysis_values,
        )

    def to_text(self) -> str:
        lines = [f"model: {_scalar(self.model)}", f"state_policy: {self.state_policy}"]
        lines.extend(("", "@debug"))
        lines.extend(_operation_text(operation) for operation in self.operations)
        for name, values in (("sources", self.sources), ("probe", self.probes),
                             ("analysis", self.analyses), ("checks", self.checks)):
            lines.extend(("", f"@{name}"))
            if values:
                lines.extend(yaml.safe_dump(dict(values), sort_keys=False,
                                            default_flow_style=False).rstrip().splitlines())
        return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True)
class PreparedDebug:
    """Effective parameters and observation names produced from a debug plan."""

    params: Params
    operations: tuple[DebugOperation, ...]
    probes: Mapping[str, tuple[str, ...]]
    state_probes: Mapping[str, str]
    parameter_overrides: Mapping[str, Any]

    @property
    def watch(self) -> tuple[str, ...]:
        return tuple(name for names in self.probes.values() for name in names)


def _scalar(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"debug value is not serializable: {exc}") from None


def _operation_text(operation: DebugOperation) -> str:
    if operation.kind == "set":
        return f"~ {operation.path} = {_scalar(operation.value)}"
    symbol = "+" if operation.kind == "connect" else "-"
    return f"{symbol} {operation.source} -> {operation.target}"


def parse_operation(text: str, *, where: str = "debug operation") -> DebugOperation:
    """Parse one connection or parameter operation."""
    statement = text.strip()
    match = _WIRE.fullmatch(statement)
    if match is not None:
        symbol, source, target = match.groups()
        return DebugOperation("connect" if symbol == "+" else "disconnect",
                              source=source, target=target)
    match = _SET.fullmatch(statement)
    if match is not None:
        path, written = match.groups()
        try:
            value = parse_parameter_mapping(f"value: {written}\n", where)["value"]
        except (KeyError, ValueError) as exc:
            raise ConfigError(f"{where}: invalid parameter value: {exc}") from None
        return DebugOperation("set", path=path, value=value)
    raise ConfigError(f"{where}: expected '+ source -> target', '- source -> target', or '~ path = value'")


def _validate_plan(plan: DebugPlan, where: str = "debug plan") -> DebugPlan:
    """Validate plans from files, the CLI and the Python interface identically."""
    if not isinstance(plan.model, str):
        raise ConfigError(f"{where}: model must be a path string")
    if (not isinstance(plan.state_policy, str)
            or plan.state_policy not in {"continue", "freeze", "tracking"}):
        raise ConfigError(f"{where}: state_policy must be continue, freeze, or tracking")
    for name, values in (("sources", plan.sources), ("probe", plan.probes),
                         ("analysis", plan.analyses), ("checks", plan.checks)):
        if not isinstance(values, Mapping):
            raise ConfigError(f"{where}: @{name} must be a mapping")
    if any(not isinstance(name, str) or _NAME.fullmatch(name) is None
           for name in plan.sources):
        raise ConfigError(f"{where}: @sources names must be simple identifiers")
    if any(not isinstance(name, str) or _NAME.fullmatch(name) is None
           or not isinstance(value, str) for name, value in plan.probes.items()):
        raise ConfigError(f"{where}: @probe aliases need simple names and dotted-path values")
    for operation in plan.operations:
        if not isinstance(operation, DebugOperation):
            raise ConfigError(f"{where}: operations must be DebugOperation values")
        if operation.kind == "set":
            valid = _PATH_NAME.fullmatch(operation.path) is not None
        elif operation.kind in {"connect", "disconnect"}:
            valid = (_PATH_NAME.fullmatch(operation.source) is not None
                     and _PATH_NAME.fullmatch(operation.target) is not None)
        else:
            valid = False
        if not valid:
            raise ConfigError(f"{where}: invalid {operation.kind!r} operation")
    return plan


def parse_debug_text(text: str, path: str | Path = "<string>") -> DebugPlan:
    """Parse a ``.pesdbg`` file without interpreting its referenced simulation."""
    bodies: dict[str, list[tuple[int, str]]] = {name: [] for name in _SECTIONS}
    preamble: list[str] = []
    active: str | None = None
    seen: set[str] = set()
    for number, raw in enumerate(text.removeprefix("\ufeff").splitlines(), 1):
        match = _HEADER.fullmatch(raw)
        if match is not None:
            active = match.group(1)
            if active not in bodies:
                raise ConfigError(f"{path}: unknown @{active} section at line {number}")
            if active in seen:
                raise ConfigError(f"{path}: duplicate @{active} section at line {number}")
            seen.add(active)
            continue
        if active is None:
            preamble.append(raw)
        else:
            bodies[active].append((number, raw))
    header = parse_parameter_mapping("\n".join(preamble) or "{}", path)
    unknown = set(header) - {"model", "state_policy"}
    if unknown:
        raise ConfigError(f"{path}: unknown debug fields {sorted(unknown)}")
    model = header.get("model", "")
    state_policy = header.get("state_policy", "continue")
    operations = []
    for number, raw in bodies["debug"]:
        statement = raw.strip()
        if not statement or statement.startswith("#"):
            continue
        operations.append(parse_operation(statement, where=f"{path}:{number}"))
    mappings = {}
    for section in _SECTIONS[1:]:
        body = "\n".join(raw for _number, raw in bodies[section])
        mappings[section] = parse_parameter_mapping(body or "{}", f"{path} @{section}")
    return _validate_plan(DebugPlan(
        model=model, state_policy=state_policy,
        operations=tuple(operations), sources=mappings["sources"], probes=mappings["probe"],
        analyses=mappings["analysis"], checks=mappings["checks"],
    ), str(path))


def read_debug_plan(path: str | Path) -> DebugPlan:
    source = Path(path)
    try:
        return parse_debug_text(source.read_text(encoding="utf-8"), source)
    except OSError as exc:
        raise ConfigError(f"cannot read debug plan {source}: {exc}") from None


def _controlled_units(p: Params) -> dict[str, Any]:
    return {name: unit for name, unit in _unit_items(p.units) if unit.ctrl is not None}


def _split_control_path(path: str, units: Mapping[str, Any], *, default: str | None = None
                        ) -> tuple[str, str]:
    matches = [name for name in units if path.startswith(name + ".")]
    if matches:
        name = max(matches, key=len)
        return name, path[len(name) + 1:]
    if default is not None:
        return default, path
    if len(units) == 1:
        return next(iter(units)), path
    raise ConfigError(f"control path {path!r} must name one of the units {sorted(units)}")


def _parameter_path(p: Params, written: str) -> str:
    values = flat_paths(to_dict(p, derived=True))
    if written in values:
        return written
    candidates = [path for path in values if path.endswith("." + written)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ConfigError(f"debug parameter {written!r} is unknown")
    raise ConfigError(f"debug parameter {written!r} is ambiguous: {candidates}")


def _number(config: Mapping[str, Any], name: str, default: float | None = None, *,
            positive: bool = False) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"debug source {name!r} must be a finite number")
    value = float(value)
    if positive and value <= 0.0:
        raise ConfigError(f"debug source {name!r} must be positive")
    return value


def _source_signal(config: Any, *, directory: Path, start: float, stop: float,
                   period: float) -> Callable[[float], float]:
    """Build one source using existing table generators where sampling is intrinsic."""
    if isinstance(config, (int, float)) and not isinstance(config, bool):
        config = {"type": "constant", "value": config}
    if not isinstance(config, Mapping):
        raise ConfigError("debug source must be a number or mapping")
    config = dict(config)
    kind = str(config.pop("type", "constant")).casefold()
    if kind == "constant":
        value = _number(config, "value")
        signal = lambda _t, value=value: value
    elif kind == "step":
        before = _number(config, "initial", 0.0)
        after = _number(config, "final", config.get("value"))
        at = _number(config, "time", start)
        signal = lambda t, before=before, after=after, at=at: after if t >= at else before
    elif kind == "ramp":
        initial = _number(config, "initial", 0.0)
        slope = _number(config, "slope")
        begin = _number(config, "start", start)
        end = _number(config, "stop", stop)
        if end < begin:
            raise ConfigError("debug ramp stop must not precede start")
        signal = lambda t, initial=initial, slope=slope, begin=begin, end=end: (
            initial + slope * (min(max(t, begin), end) - begin)
        )
    elif kind == "sine":
        amplitude = _number(config, "amplitude", 1.0)
        frequency = _number(config, "frequency", positive=True)
        phase = _number(config, "phase", 0.0)
        offset = _number(config, "offset", 0.0)
        begin = _number(config, "start", start)
        signal = lambda t, a=amplitude, f=frequency, ph=phase, o=offset, begin=begin: (
            o if t < begin else o + a * math.sin(math.tau * f * (t - begin) + ph)
        )
    elif kind == "pulse":
        low = _number(config, "low", 0.0)
        high = _number(config, "high", 1.0)
        width = _number(config, "width", config.get("duty", 0.5))
        cycle = _number(config, "period", positive=True)
        delay = _number(config, "delay", start)
        if not 0.0 <= width <= 1.0:
            raise ConfigError("debug pulse width must be between 0 and 1")
        signal = lambda t, lo=low, hi=high, w=width, cycle=cycle, delay=delay: (
            lo if t < delay else hi if ((t - delay) % cycle) < w * cycle else lo
        )
    else:
        from .reference_signals import (
            reference_chirp, reference_multitone, reference_prbs,
        )
        from .reference_signals import reference_csv, reference_table

        builders = {"chirp": reference_chirp, "prbs": reference_prbs,
                    "multitone": reference_multitone}
        if kind in builders:
            config.setdefault("start", start)
            config.setdefault("stop", stop)
            config.setdefault("period", period)
            table = builders[kind](**config)
        elif kind == "table":
            table = reference_table(**config)
        elif kind == "csv":
            if "file" in config and "path" in config:
                raise ConfigError("debug CSV source accepts file or path, not both")
            key = "file" if "file" in config else "path"
            file = config.pop(key, None)
            if not isinstance(file, str) or not file:
                raise ConfigError("debug CSV source requires file")
            source = Path(file).expanduser()
            if not source.is_absolute():
                source = directory / source
            table = reference_csv(source, **config)
        else:
            raise ConfigError(
                "debug source type must be constant, step, ramp, sine, pulse, chirp, "
                "prbs, multitone, table, or csv"
            )
        signal = TableSignal(**table)
    unused = set(config) - {
        "value", "initial", "final", "time", "slope", "start", "stop", "amplitude",
        "frequency", "phase", "offset", "low", "high", "width", "duty", "period", "delay",
    }
    if kind in {"constant", "step", "ramp", "sine", "pulse"} and unused:
        raise ConfigError(f"debug {kind} source has unknown fields {sorted(unused)}")
    return signal


def _replace_units(units: Mapping[str, Any], overlays: Mapping[str, ControlGraphOverlay],
                   prefix: str = "") -> dict[str, Any]:
    result = {}
    for name, unit in units.items():
        path = f"{prefix}.{name}" if prefix else name
        spec = unit_type(unit.type)
        changes = {}
        if path in overlays:
            changes["ctrl"] = replace(unit.ctrl, _control_overlay=overlays[path])
        if spec.composite:
            children = {child: getattr(unit, child) for child in spec.children}
            changes.update(_replace_units(children, overlays, path))
        if changes:
            updated = replace(unit, **changes)
            for attribute in ("base", "derived_defaults", "events"):
                object.__setattr__(updated, attribute, getattr(unit, attribute))
            result[name] = updated
        else:
            result[name] = unit
    return result


def _install_overlays(p: Params, overlays: Mapping[str, ControlGraphOverlay]) -> Params:
    return replace(p, units=_replace_units(p.units, overlays))


def prepare_debug(p: Params, plan: DebugPlan, *, directory: str | Path = ".") -> PreparedDebug:
    """Apply a plan before assembly and return the ordinary simulation parameters."""
    _validate_plan(plan)
    overrides = {}
    operations: list[DebugOperation | None] = []
    previous_sets: dict[str, int] = {}
    for operation in plan.operations:
        if operation.kind != "set":
            operations.append(operation)
            continue
        path = _parameter_path(p, operation.path)
        previous = previous_sets.get(path)
        if previous is not None:
            operations[previous] = None
        previous_sets[path] = len(operations)
        operations.append(replace(operation, path=path))
        overrides[path] = operation.value
    canonical_operations = tuple(operation for operation in operations if operation is not None)
    if overrides:
        p = p.replace(**overrides)
    p = p.replace(**{
        "simulation.output.states": False,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
        "simulation.progress.enable": False,
    })
    changes = p.changes
    units = _controlled_units(p)
    if not units and (plan.sources or plan.probes
                      or any(operation.kind != "set" for operation in plan.operations)):
        raise ConfigError("debug control operations require at least one controlled unit")

    descriptions = {}
    for name, unit in units.items():
        profile = control_profile(unit.ctrl)
        is_continuous = continuous_control(unit)
        defaults, outputs = profile.wiring(unit, continuous=is_continuous)
        interface = resolved_control_interface(profile, is_continuous)
        classes = {loop_name: LOOP_TYPES[loop.type]
                   for loop_name, loop in unit.ctrl.loops.items()}
        source_types = dict(interface.inputs)
        source_types.update({f"{loop_name}.{port}": signal
                             for loop_name, cls in classes.items()
                             for port, signal in cls.outputs.items()})
        targets = {f"{loop_name}.{port}": signal
                   for loop_name, cls in classes.items()
                   for port, signal in cls.inputs.items()}
        descriptions[name] = {
            "unit": unit, "connections": {**defaults, **unit.ctrl.connections},
            "outputs": outputs, "sources": source_types, "targets": targets,
            "overrides": {}, "external": {}, "retained": set(), "policies": {},
            "probes": {},
        }

    used_sources: dict[str, set[str]] = {name: set() for name in plan.sources}
    for operation in plan.operations:
        if operation.kind == "set":
            continue
        unit_name, target = _split_control_path(operation.target, units)
        data = descriptions[unit_name]
        if target not in data["targets"]:
            raise ConfigError(f"debug target {operation.target!r} is not a loop input")
        source_name, source = _split_control_path(operation.source, units, default=unit_name)
        if source_name != unit_name:
            raise ConfigError("debug connections cannot cross controller boundaries")
        external_name = source.removesuffix(".out") if source.endswith(".out") else ""
        if external_name in plan.sources:
            source = f"debug.{external_name}"
            signal_type = data["targets"][target]
            previous = data["external"].get(source)
            if previous is not None and previous != signal_type:
                raise ConfigError(f"debug source {external_name!r} is used with incompatible ports")
            data["external"][source] = signal_type
            used_sources[external_name].add(unit_name)
        elif source not in data["sources"]:
            raise ConfigError(f"debug source {operation.source!r} is unknown")
        if operation.kind == "disconnect":
            if external_name:
                raise ConfigError("a debug source cannot be disconnected from the original graph")
            if data["connections"].get(target) != source:
                raise ConfigError(
                    f"debug disconnect {operation.source} -> {operation.target} does not exist"
                )
            data["connections"].pop(target)
            data["overrides"][target] = None
            data["retained"].add(source)
            owner = source.partition(".")[0]
            if owner in unit.ctrl.loops and plan.state_policy != "continue":
                data["policies"][owner] = plan.state_policy
        else:
            if target in data["connections"]:
                raise ConfigError(
                    f"debug target {operation.target!r} is already connected; disconnect it first"
                )
            source_type = data["external"].get(source, data["sources"].get(source))
            if source_type is None or not source_type.compatible(data["targets"][target]):
                raise ConfigError(
                    f"debug connection {operation.source} -> {operation.target} has incompatible ports"
                )
            data["connections"][target] = source
            data["overrides"][target] = source

    probe_names: dict[str, tuple[str, ...]] = {}
    state_probes: dict[str, str] = {}
    for alias, written in plan.probes.items():
        if written.startswith("state."):
            state = written.removeprefix("state.")
            if not state:
                raise ConfigError(f"debug probe {alias!r}: state path is empty")
            state_probes[alias] = state
            continue
        unit_name, source = _split_control_path(written, units)
        data = descriptions[unit_name]
        external_name = source.removesuffix(".out") if source.endswith(".out") else ""
        if external_name in plan.sources:
            source = f"debug.{external_name}"
            if source not in data["external"]:
                raise ConfigError(
                    f"debug probe {alias!r}: source {external_name!r} is not connected"
                )
            used_sources[external_name].add(unit_name)
        resolved = data["connections"].get(source, source)
        signal_type = data["external"].get(resolved, data["sources"].get(resolved))
        if signal_type is None:
            raise ConfigError(f"debug probe {alias!r}: unknown source {written!r}")
        data["probes"][alias] = source
        suffixes = (".re", ".im", ".abs") if signal_type.complex_value else ("",)
        probe_names[alias] = tuple(f"{unit_name}.{alias}{suffix}" for suffix in suffixes)

    unused = [name for name, consumers in used_sources.items() if not consumers]
    if unused:
        raise ConfigError(f"debug sources are not connected or probed: {unused}")
    signal_cache = {}
    overlays = {}
    directory = Path(directory)
    for unit_name, data in descriptions.items():
        if not (data["overrides"] or data["probes"] or data["retained"]):
            continue
        sources = {}
        for local, signal_type in data["external"].items():
            external_name = local.partition(".")[2]
            signal = signal_cache.get(external_name)
            if signal is None:
                try:
                    signal = _source_signal(
                        plan.sources[external_name], directory=directory,
                        start=p.simulation.initial.t, stop=p.simulation.t_end,
                        period=p.simulation.output.period,
                    )
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"debug source {external_name!r}: {exc}") from None
                signal_cache[external_name] = signal
            sources[local] = (signal_type, signal)
        overlays[unit_name] = ControlGraphOverlay(
            connections=data["overrides"], sources=sources,
            probes=data["probes"], retained=frozenset(data["retained"]),
            policies=data["policies"], initial_t=p.simulation.initial.t,
        )
    p = _install_overlays(p, overlays)
    adjusted = tuple(replace(change, params=_install_overlays(change.params, overlays)
                             if change.params is not None else None)
                     for change in changes)
    p.__dict__["changes"] = adjusted
    return PreparedDebug(p, canonical_operations, probe_names, state_probes, overrides)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": _json_safe(value.real), "imaginary": _json_safe(value.imag)}
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_dataclass(value):
        return {name: _json_safe(getattr(value, name))
                for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _signal(data: Mapping[str, list[float]], name: str) -> list[float]:
    if name in data:
        return data[name]
    raise ConfigError(f"debug signal {name!r} is unknown; known: {sorted(data)}")


def _required(config: dict[str, Any], field: str, context: str) -> Any:
    if field not in config:
        raise ConfigError(f"{context} requires {field}")
    return config.pop(field)


def _analyses(plan: DebugPlan, simulation: Simulation,
              series: Mapping[str, list[float]]) -> dict[str, Any]:
    results = {}
    time = series["t"]
    for name, written in plan.analyses.items():
        if isinstance(written, str):
            config = {"type": written}
        elif written is None or isinstance(written, Mapping):
            config = dict(written or {})
        else:
            raise ConfigError(f"debug analysis {name!r} must be text or a mapping")
        kind = str(config.pop("type", name)).casefold()
        if kind == "small_signal":
            from .online_analysis import small_signal
            try:
                result = small_signal(simulation, **config)
            except TypeError as exc:
                raise ConfigError(f"debug analysis {name!r}: {exc}") from None
            results[name] = {
                "type": kind, "at": result.at, "period": result.period,
                "stability": result.stability, "periodic": result.periodic,
                "periodic_error": result.periodic_error,
                "coordinates": result.coordinates, "multipliers": result.multipliers,
                "evaluations": result.evaluations, "wall_time": result.wall_time,
                "diagnostics": result.diagnostics,
            }
            continue
        if kind == "dynamic":
            from .online_analysis import dynamic_performance
            response = _signal(series, _required(
                config, "response", f"debug analysis {name!r}",
            ))
            reference = config.pop("reference", None)
            if isinstance(reference, str):
                reference = _signal(series, reference)
            try:
                metrics = dynamic_performance(
                    time, response, reference=reference, **config,
                )
            except TypeError as exc:
                raise ConfigError(f"debug analysis {name!r}: {exc}") from None
            results[name] = {"type": kind, **metrics.to_dict()}
            continue
        if kind in {"frequency_response", "impedance", "loop_gain"}:
            from .offline_analysis import frequency_response, loop_gain_sweep
            context = f"debug analysis {name!r}"
            input_name = _required(config, "input", context)
            output_name = _required(config, "output", context)
            excitation = _signal(series, input_name)
            response = _signal(series, output_name)
            feedback_sign = config.pop("feedback_sign", None)
            if kind == "loop_gain":
                feedback_sign = -1.0 if feedback_sign is None else feedback_sign
            elif feedback_sign is not None:
                raise ConfigError(f"{context}: feedback_sign is only valid for loop_gain")
            frequencies = config.pop("frequencies", None)
            if frequencies is None:
                frequencies = (_required(config, "frequency", context),)
            elif "frequency" in config:
                raise ConfigError(
                    f"debug analysis {name!r}: use frequency or frequencies, not both"
                )
            elif isinstance(frequencies, (int, float)) and not isinstance(frequencies, bool):
                frequencies = (frequencies,)
            if isinstance(frequencies, (str, bytes)):
                raise ConfigError(f"{context}: frequencies must be numbers")
            try:
                selected = tuple(float(frequency) for frequency in frequencies)
            except (TypeError, ValueError):
                raise ConfigError(f"{context}: frequencies must be numbers") from None
            try:
                measurements = {
                    frequency: frequency_response(
                        time, excitation, response, frequency=frequency, **config,
                    ) for frequency in selected
                }
                if kind == "loop_gain":
                    from .offline_analysis import _signed_return_ratio
                    measurements = {
                        frequency: _signed_return_ratio(
                            measurement, feedback_sign, output_name, input_name,
                        )
                        for frequency, measurement in measurements.items()
                    }
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{context}: {exc}") from None
            if kind == "loop_gain":
                result = loop_gain_sweep(measurements).to_dict()
            else:
                result = (next(iter(measurements.values())).to_dict()
                          if len(measurements) == 1 else
                          {str(frequency): item.to_dict()
                           for frequency, item in measurements.items()})
            results[name] = {"type": kind, "result": result}
            continue
        raise ConfigError(f"debug analysis {name!r}: unsupported type {kind!r}")
    return results


def _checks(plan: DebugPlan, series: Mapping[str, list[float]]) -> tuple[dict[str, Any], bool]:
    results = {}
    passed = True
    metrics = {
        "final": lambda values: values[-1], "min": min, "max": max,
        "mean": lambda values: float(np.mean(values)),
        "rms": lambda values: float(np.sqrt(np.mean(np.square(values)))),
        "max_abs": lambda values: max(map(abs, values)),
    }
    for name, written in plan.checks.items():
        if not isinstance(written, Mapping):
            raise ConfigError(f"debug check {name!r} must be a mapping")
        config = dict(written)
        signal = config.pop("signal", name)
        metric = config.pop("metric", "final")
        comparison = config.pop("operator", config.pop("op", "<="))
        expected = config.pop("value", None)
        tolerance = config.pop("tolerance", 0.0)
        if config or metric not in metrics or comparison not in _OPERATORS:
            raise ConfigError(f"debug check {name!r} has unsupported fields or operation")
        if (isinstance(expected, bool) or not isinstance(expected, (int, float))
                or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or tolerance < 0):
            raise ConfigError(f"debug check {name!r} needs numeric value and nonnegative tolerance")
        actual = metrics[metric](_signal(series, signal))
        limit = float(expected)
        good = (abs(actual - limit) <= tolerance if comparison == "==" else
                abs(actual - limit) > tolerance if comparison == "!=" else
                _OPERATORS[comparison](actual, limit + tolerance if comparison in ("<", "<=")
                                       else limit - tolerance))
        results[name] = {"signal": signal, "metric": metric, "actual": actual,
                         "operator": comparison, "expected": limit,
                         "tolerance": tolerance, "passed": bool(good)}
        passed &= bool(good)
    return results, passed


def _run_id(root: Path) -> str:
    prefix = datetime.now().strftime("%y%m%dT%H%M")
    used = {path.name for path in root.glob(prefix + "-*") if path.is_dir()}
    index = next(index for index in range(1000) if f"{prefix}-{index:03d}" not in used)
    return f"{prefix}-{index:03d}"


def debug_root(config: str | Path, output: str | Path = "output") -> Path:
    return Path(output) / Path(config).stem / "debug"


def debug_history(config: str | Path, output: str | Path = "output") -> list[Path]:
    root = debug_root(config, output)
    return sorted((path for path in root.iterdir() if path.is_dir()), reverse=True) if root.is_dir() else []


def _canonical_plan(plan: DebugPlan, config: Path, directory: Path) -> DebugPlan:
    """Make a saved plan independent of the location from which it was loaded."""
    sources = {}
    for name, written in plan.sources.items():
        if not isinstance(written, Mapping):
            sources[name] = written
            continue
        source = dict(written)
        if str(source.get("type", "constant")).casefold() == "csv":
            keys = [key for key in ("file", "path") if key in source]
            if len(keys) > 1:
                raise ConfigError("debug CSV source accepts file or path, not both")
            if keys:
                value = source[keys[0]]
                if not isinstance(value, str) or not value:
                    raise ConfigError("debug CSV source requires a nonempty path")
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = directory / path
                source[keys[0]] = str(path.resolve())
        sources[name] = source
    return replace(plan, model=str(config), sources=sources)


def _resolve_state_probes(simulation: Simulation, probes: Mapping[str, str]
                          ) -> dict[str, tuple[str, ...]]:
    """Resolve scalar and complex controller-state columns after ordinary assembly."""
    columns = set(simulation.state_names())
    resolved = {}
    for alias, state in probes.items():
        if state in columns:
            resolved[alias] = (state,)
        elif {f"{state}.re", f"{state}.im"} <= columns:
            resolved[alias] = (f"{state}.re", f"{state}.im")
        else:
            raise ConfigError(
                f"debug probe {alias!r}: unknown state {state!r}; known: {sorted(columns)}"
            )
    return resolved


def execute_debug(config: str | Path, p: Params, interface_text: str, plan: DebugPlan, *,
                  plan_directory: str | Path = ".", output: str | Path = "output",
                  timeout: float | None = None, max_rhs: int | None = None) -> tuple[Path, dict]:
    """Run one debug overlay and write its canonical plan and structured evidence."""
    config = Path(config).resolve()
    canonical = _canonical_plan(plan, config, Path(plan_directory).resolve())
    prepared = prepare_debug(p, canonical, directory=config.parent)
    canonical = replace(canonical, operations=prepared.operations)
    simulation = validate_model(prepared.params, interface_text=interface_text)
    state_probes = _resolve_state_probes(simulation, prepared.state_probes)
    series = {"t": []}
    for alias, names in prepared.probes.items():
        for full in names:
            marker = f".{alias}"
            suffix = full[full.rfind(marker) + len(marker):]
            public = alias + suffix
            series[public] = []
    for alias, names in state_probes.items():
        suffixes = (".re", ".im", ".abs") if len(names) == 2 else ("",)
        for suffix in suffixes:
            series[alias + suffix] = []

    def observe(t, states, read_values, *, final, stop_reason):
        values = read_values()
        if series["t"] and math.isclose(series["t"][-1], t, rel_tol=0.0, abs_tol=1e-15):
            index = -1
        else:
            series["t"].append(float(t))
            for name in series:
                if name != "t":
                    series[name].append(math.nan)
            index = -1
        for alias, names in prepared.probes.items():
            for full in names:
                marker = f".{alias}"
                suffix = full[full.rfind(marker) + len(marker):]
                public = alias + suffix
                if full in values:
                    series[public][index] = float(values[full])
        for alias, names in state_probes.items():
            if states is None:
                continue
            if len(names) == 1:
                series[alias][index] = float(states[names[0]])
                continue
            real, imaginary = (float(states[name]) for name in names)
            series[f"{alias}.re"][index] = real
            series[f"{alias}.im"][index] = imaginary
            series[f"{alias}.abs"][index] = math.hypot(real, imaginary)

    root = debug_root(config, output)
    run_dir = root / _run_id(root)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "debug.pesdbg").write_text(canonical.to_text(), encoding="utf-8")
    report = {
        "schema": "peslite.debug/v1", "version": __version__,
        "model": {"path": str(config),
                  "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
        "plan": {"state_policy": canonical.state_policy,
                 "parameter_overrides": prepared.parameter_overrides,
                 "effective_configuration": to_dict(prepared.params, derived=True)},
        "status": "running", "simulation": {}, "data": {}, "analyses": {},
        "metrics": {}, "checks": {}, "evidence": {}, "diagnostics": [],
    }
    try:
        result = simulation.run(
            out_dir=None, observe=observe, times=None, observe_values=prepared.watch,
            name=config.stem, timeout=timeout, max_rhs=max_rhs,
        )
        report["simulation"] = {
            "status": result.summary.get("status", "completed"),
            "t_start": prepared.params.simulation.initial.t,
            "t_stop": result.summary.get("t_stop"), "wall_time": result.wall_time,
            "rhs_evaluations": result.n_rhs, "tripped": result.tripped,
            "stop_reason": result.summary.get("stop_reason") or None,
        }
        report["data"] = series
        report["evidence"] = {
            "sample_count": len(series["t"]),
            "signals": sorted(name for name in series if name != "t"),
            "analyses": list(canonical.analyses),
        }
        report["analyses"] = _analyses(canonical, simulation, series)
        checks, passed = _checks(canonical, series)
        report["checks"] = checks
        report["metrics"] = {
            name: {"value": check["actual"], "signal": check["signal"],
                   "metric": check["metric"]}
            for name, check in checks.items()
        }
        stopped = bool(result.summary.get("stop_reason"))
        report["status"] = "stopped" if stopped else "passed" if passed else "failed"
    except Exception as exc:
        report["status"] = "error"
        report["diagnostics"].append({"severity": "error", "message": str(exc)})
        (run_dir / "report.json").write_text(
            json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        raise
    (run_dir / "report.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return run_dir, report


__all__ = [
    "DebugOperation", "DebugPlan", "PreparedDebug", "debug_history", "debug_root",
    "execute_debug", "parse_debug_text", "parse_operation", "prepare_debug", "read_debug_plan",
]
