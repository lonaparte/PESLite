"""Optional legacy-YAML preview function addon.

This adapter reuses PESLite's parameter parser and topology descriptor.
It is intentionally separate from the browser-only text-to-SVG preview library.
The source buffer is never rewritten; project-local ``PESaddons`` are not
executed merely to inspect a file. An unsupported addon is a diagnostic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from contextvars import Token
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, ScalarNode

from ...gui.workspace import (
    WorkspaceError, add_component, explicit_parameters, patch_scalar, patched_text,
    remove_component,
)


def _point(line: int, column: int = 1) -> dict[str, int]:
    return {"line": line, "column": column}


def _range(line: int, end: int | None = None) -> dict[str, Any]:
    return {"start": _point(line), "end": _point(end or line)}


def _positions(text: str) -> dict[str, dict[str, Any]]:
    """Map every declared YAML mapping key to its original source range."""
    result: dict[str, dict[str, Any]] = {}
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    if not isinstance(root, MappingNode):
        raise WorkspaceError("PESLite YAML requires a top-level mapping")

    def visit(node: MappingNode, prefix: str, ancestors: frozenset[int]) -> None:
        if id(node) in ancestors:
            return
        seen: set[str] = set()
        for key, child in node.value:
            if not isinstance(key, ScalarNode) or key.value in seen:
                continue
            seen.add(key.value)
            name = f"{prefix}.{key.value}" if prefix else key.value
            result[name] = {
                "start": _point(key.start_mark.line + 1, key.start_mark.column + 1),
                "end": _point(child.end_mark.line + 1, child.end_mark.column + 1),
                "startIndex": key.start_mark.index,
                "endIndex": child.end_mark.index,
            }
            if isinstance(child, MappingNode):
                visit(child, name, ancestors | {id(node)})
    visit(root, "", frozenset())
    return result


def _source_location(locations: Mapping[str, dict], key: str) -> dict[str, Any]:
    match = locations.get(key)
    if match:
        return {"start": match["start"], "end": match["end"]}
    return _range(1)


def _section(locations: Mapping[str, dict], text: str, key: str,
             name: str, kind: str, *, parent: str | None = None) -> dict:
    region = locations.get(key)
    source = text[region["startIndex"]:region["endIndex"]] if region else ""
    block = {"id": key, "name": name, "type": kind, "source": source,
             "range": _source_location(locations, key), "children": []}
    if parent:
        block["parent"] = parent
    return block


def _diagnostic(message: str, line: int = 1, column: int = 1) -> dict:
    return {"severity": "error", "message": message,
            "range": {"start": _point(line, column), "end": _point(line, column + 1)}}


def _event_data(config: Mapping, locations: Mapping[str, dict], params: Any,
                components: list[dict], connections: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep all event boundaries, including non-topology parameter changes."""
    raw = config.get("events") or {}
    simulation = config.get("simulation") or {}
    initial = simulation.get("initial") or {}
    start = initial.get("t", 0.0) if isinstance(initial, Mapping) else 0.0
    stop = simulation.get("t_end", 1.0)
    if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
        return [], []
    if not math.isfinite(start) or not math.isfinite(stop) or start >= stop:
        return [], []
    events: list[dict] = []
    boundaries = {float(start), float(stop)}
    node_names: dict[str, str] = {}
    for node in components:
        node_names[node["name"]] = node["id"]
    for connection in connections:
        node_names[connection["name"]] = connection["id"]
    for section, prefix in (("buses", "bus-"), ("sources", "source-"),
                            ("branches", "branch-"), ("units", "unit-"),
                            ("elements", "element-")):
        for node in components + connections:
            if node["id"].startswith(prefix):
                node_names[f"{section}.{node['name']}"] = node["id"]

    if isinstance(raw, Mapping):
        for name, event in raw.items():
            if not isinstance(event, Mapping):
                continue
            time = event.get("t")
            if not isinstance(time, (int, float)) or isinstance(time, bool) or not math.isfinite(time):
                continue
            if start < time < stop:
                boundaries.add(float(time))
            kind = event.get("type")
            if kind in ("connect", "disconnect"):
                target = str(event.get("target", ""))
                events.append({"name": str(name), "type": kind,
                               "target": target, "time": float(time),
                               "line": locations.get(f"events.{name}", {}).get("start", {}).get("line", 1)})

    from ...assembly.events import _initial_connection, switching_schedule
    from ...assembly.params import connection_defaults

    initially_inactive = set()
    for path, connected in connection_defaults(params):
        if not connected:
            name = path.removesuffix(".connected")
            if name in node_names:
                initially_inactive.add(node_names[name])
    # Use PESLite's own event connection schedule for the other switchable
    # objects: a first 'connect' event implies initially disconnected.
    schedule = switching_schedule(params.changes)
    for section in ("sources", "branches", "elements"):
        configs = config.get(section) or {}
        if not isinstance(configs, Mapping):
            continue
        for name, values in configs.items():
            if not isinstance(values, Mapping):
                continue
            target = f"{section}.{name}"
            explicit = values.get("connected")
            active = _initial_connection(schedule.get(target, ()), explicit)
            if not active and target in node_names:
                initially_inactive.add(node_names[target])

    segments = []
    ordered = sorted(enumerate(events), key=lambda item: (item[1]["time"], item[0]))
    times = sorted(boundaries)
    for index, (left, right) in enumerate(zip(times, times[1:])):
        inactive = set(initially_inactive)
        for _sequence, event in ordered:
            if event["time"] > left:
                break
            key = node_names.get(event["target"], event["target"])
            if event["type"] == "disconnect":
                inactive.add(key)
            else:
                inactive.discard(key)
        segments.append({"index": index, "start": left, "end": right,
                         "disconnected": sorted(inactive)})
    return events, segments


def preview(text: str, path: str | Path) -> dict[str, Any]:
    """Return a normalized PESModel and editable field locations for a buffer.

    YAML errors or PESLite validation failures become diagnostics and never
    mutate source text. Host adapters can keep showing the last valid graph.
    """
    path = Path(path)
    try:
        config = yaml.safe_load(text)
        if not isinstance(config, dict):
            raise WorkspaceError("PESLite YAML requires a top-level mapping")
        locations = _positions(text)
        # Do not call load(path): load preparers may execute project-local addons.
        from ...assembly.params import _ASSET_DIRECTORY, from_dict
        from ...gui.descriptor import Topology

        token: Token = _ASSET_DIRECTORY.set(path.resolve().parent)
        try:
            params = from_dict(config)
        finally:
            _ASSET_DIRECTORY.reset(token)
        topology = Topology.build(path.stem, path, params)

        nodes = []
        section_names = (("bus-", "buses"), ("source-", "sources"),
                         ("unit-", "units"), ("element-", "elements"))
        for component in topology.components:
            name = component["name"]
            section = next((section for prefix, section in section_names
                            if component["id"].startswith(prefix)), "")
            location = f"{section}.{name}" if section else ""
            first = locations.get(location) or locations.get(f"{section}.{name.split('.')[0]}") or {}
            nodes.append({"id": component["id"], "name": name,
                          "type": component["className"].removeprefix("peslite."),
                          "firstLine": first.get("start", {}).get("line", 1),
                          "source": _source_location(locations, location),
                          "ports": [], "parameterBlock": location})
        edges = []
        for connection in topology.connections:
            identifier = connection["id"]
            name = connection["name"]
            prefix = "branches" if identifier.startswith("branch-") else "elements"
            reference = (f"{prefix}.{name}" if identifier.startswith(("branch-", "element-"))
                         else "")
            position = locations.get(reference) or {}
            edges.append({"id": identifier, "from": connection["source"],
                          "to": connection["target"], "directed": False,
                          "line": position.get("start", {}).get("line", 1),
                          "source": _source_location(locations, reference)})

        parameters = []
        for section in ("buses", "branches", "sources", "units", "elements"):
            values = config.get(section) or {}
            if not isinstance(values, Mapping):
                continue
            for name in values:
                key = f"{section}.{name}"
                parameters.append(_section(locations, text, key, str(name), section))
        globals_ = [_section(locations, text, name, name, "global")
                    for name in ("base", "simulation", "events", "signals", "meta")
                    if name in locations]
        views = [{"name": "circuit", "nodes": nodes, "edges": edges,
                  "groups": [], "parameters": parameters, "source": _range(1, len(text.splitlines()) or 1)}]

        # Legacy controllers explicitly list their loops within their owning unit.
        # Do not infer wires that do not exist in this file format.
        def add_unit_views(units: Mapping, prefix: str = "") -> None:
            for name, unit in units.items():
                if not isinstance(unit, Mapping):
                    continue
                owner = f"{prefix}.{name}" if prefix else str(name)
                ctrl = unit.get("ctrl") or {}
                loops = ctrl.get("loops", {}) if isinstance(ctrl, Mapping) else {}
                if isinstance(loops, Mapping) and loops:
                    members, blocks = [], []
                    for loop, value in loops.items():
                        key = f"units.{owner}.ctrl.loops.{loop}"
                        position = locations.get(key) or {}
                        members.append({"id": str(loop), "name": str(loop),
                                        "type": str(value.get("type", loop)) if isinstance(value, Mapping)
                                        else str(loop),
                                        "firstLine": position.get("start", {}).get("line", 1),
                                        "source": _source_location(locations, key),
                                        "ports": [], "parameterBlock": key})
                        blocks.append(_section(locations, text, key, str(loop), "loop"))
                    views.append({"name": owner, "nodes": members, "edges": [],
                                  "groups": [], "parameters": blocks,
                                  "source": _source_location(locations, f"units.{owner}")})
                # Child units have their own type and controller. Other nested
                # YAML maps (dclink, filters, loops) are ordinary parameters.
                for child, value in unit.items():
                    if isinstance(value, Mapping) and "type" in value and "ctrl" in value:
                        add_unit_views({child: value}, owner)

        units = config.get("units") or {}
        if isinstance(units, Mapping):
            add_unit_views(units)
        events, timeline = _event_data(config, locations, params,
                                       topology.components, topology.connections)
        model = {"views": views, "globals": globals_, "events": events,
                 "timeline": timeline, "diagnostics": [], "defaultView": "circuit"}
        return {"format": "legacy", "model": model,
                "fields": explicit_parameters(text), "diagnostics": []}
    except (yaml.YAMLError, ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        mark = getattr(exc, "problem_mark", None)
        line, column = (mark.line + 1, mark.column + 1) if mark else (1, 1)
        message = getattr(exc, "problem", None) or str(exc)
        # PESLite's schema errors often start with a dotted parameter path.
        name = re.match(r"^([\w.-]+):", message)
        if name:
            try:
                loc = _positions(text).get(name.group(1))
                if loc:
                    line, column = loc["start"]["line"], loc["start"]["column"]
            except (yaml.YAMLError, WorkspaceError):
                pass
        return {"format": "legacy", "model": None, "fields": [],
                "diagnostics": [_diagnostic(message, line, column)]}


def _validated_edit(text: str, path: str | Path, edit: dict[str, Any]) -> dict[str, Any]:
    """Reject changes invalid under PESLite's existing schema or addon registry."""
    checked = preview(patched_text(text, edit), path)
    if checked["diagnostics"]:
        raise WorkspaceError("invalid graphical edit: " + "; ".join(
            diagnostic["message"] for diagnostic in checked["diagnostics"]))
    return edit


def validated_patch(text: str, path: str | Path, key: str, value: Any) -> dict[str, Any]:
    """Edit an existing scalar; preserve unrelated comments, layout, and YAML."""
    return _validated_edit(text, path, patch_scalar(text, key, value))


def validated_add_component(text: str, path: str | Path, section: str,
                            name: str, definition: str) -> dict[str, Any]:
    """Add an explicitly configured component at one source insertion point."""
    return _validated_edit(text, path, add_component(text, section, name, definition))


def validated_remove_component(text: str, path: str | Path, section: str,
                               name: str) -> dict[str, Any]:
    """Remove only unreferenced and comment-free component declarations."""
    return _validated_edit(text, path, remove_component(text, section, name))


def validated_rewire(text: str, path: str | Path, key: str,
                     target: str) -> dict[str, Any]:
    """Change an explicitly present network terminal to an existing bus.

    No new topology syntax is invented: PESLite's own validator rejects
    incompatible electrical phases and nonexisting connections.
    """
    if not isinstance(target, str) or not target:
        raise WorkspaceError("a destination bus is required")
    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise WorkspaceError(f"invalid YAML: {exc}") from exc
    if not isinstance(config, dict) or target not in (config.get("buses") or {}):
        raise WorkspaceError(f"unknown destination bus: {target}")
    parts = key.split(".")
    if len(parts) < 3 or parts[0] not in {"sources", "units", "elements", "branches"}:
        raise WorkspaceError("only component terminals may be reconnected")
    terminal = parts[-1]
    if terminal not in ({"bus1", "bus2"} if parts[0] == "branches"
                        else {"bus", "bus1", "bus2"}):
        raise WorkspaceError("only explicit bus, bus1 and bus2 terminals may be reconnected")
    return validated_patch(text, path, key, target)
