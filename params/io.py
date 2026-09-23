"""Read and write a converter configuration (hierarchical YAML/JSON)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, fields, is_dataclass, replace
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional, get_args, get_type_hints

from ..phs.protocols import ConfigError

from .base import BaseValues, convert_quantities
from .schema import Params, BusParams, BranchParams, SourceParams, DCSourceParams, UnitParams, build, to_dict
from .loops import LOOP_SCHEMAS, DelayLoopParams, build_loop
from .validate import validate

__all__ = ["load", "dump", "read_initial", "from_dict", "to_dict"]


def _flatten_states(states: Any, prefix: str = "") -> dict:
    """Flatten nested state mappings to dotted keys; lists are kept as values."""
    if not isinstance(states, dict):
        raise ConfigError(f"initial.states{'.' + prefix if prefix else ''}: expected a mapping")
    out: dict = {}
    for key, value in states.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_states(value, name))
        else:
            out[name] = value
    return out


def _numeric_string(value: Any) -> Any:
    """Convert a numeric string to float; return other values unchanged."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _normalize(data: dict) -> dict:
    """Return a normalized copy of the input mapping."""
    data = deepcopy(data)
    if not isinstance(data, dict):
        return data  # type error reported by build()
    init = data.get("initial")
    if isinstance(init, dict) and init.get("states") is not None:
        init["states"] = {key: _numeric_string(value)
                          for key, value in _flatten_states(init["states"]).items()}
    simulation = data.get("simulation")
    solver = simulation.get("solver") if isinstance(simulation, dict) else None
    subsystems = solver.get("subsystems") if isinstance(solver, dict) else None
    if isinstance(subsystems, dict):
        for name, value in subsystems.items():
            if isinstance(value, dict):
                if "step" in value:
                    value["step"] = _numeric_string(value["step"])
            else:
                subsystems[name] = _numeric_string(value)
    return data


def from_dict(data: dict) -> Params:
    """Build, validate and resolve a :class:`Params` tree from a mapping (input is not modified)."""
    p = build(Params, _convert_input_tree(_normalize(data)))
    units = {}
    for name, unit in p.units.items():
        loops = {loop_name: build_loop(cfg, f"units.{name}.control.loops.{loop_name}")
                 for loop_name, cfg in unit.control.loops.items()}
        units[name] = replace(unit, control=replace(unit.control, loops=loops))
    p = replace(p, units=units)
    validate(p)
    return replace(p, units={name: unit.resolved(p.base) for name, unit in p.units.items()})


def _section(data, cls, scales, where):
    if is_dataclass(data):
        data = to_dict(data)
    if not isinstance(data, dict):
        return data  # type error reported by build()
    if cls is DelayLoopParams:
        data = {key: value for key, value in data.items() if value is not None}
        if not cls._signal_quantities(data) and "initial_pu" in data:
            raise ConfigError(f"{where}.initial_pu: angle/frequency delays use initial in rad or rad/s")
    data = convert_quantities(data, cls, scales, where)
    if cls is DCSourceParams and data.get("type") == "voltage" and "v" not in data:
        data["v"] = scales["dc_voltage"]
    hints = get_type_hints(cls)
    for field in fields(cls):
        if not field.init:
            continue
        hint = hints[field.name]
        nested = hint if is_dataclass(hint) else next((arg for arg in get_args(hint) if is_dataclass(arg)), None)
        if nested is not None:
            if field.name in data:
                data[field.name] = _section(data[field.name], nested, scales, f"{where}.{field.name}")
            elif is_dataclass(hint) and field.default_factory is not MISSING:
                data[field.name] = _section({}, nested, scales, f"{where}.{field.name}")
    return data


def _convert_input_tree(data):
    if not isinstance(data, dict):
        return data
    if "base" not in data:
        return data  # missing base reported by build()
    try:
        base = build(BaseValues, data.get("base"))
    except ConfigError as exc:
        message = str(exc)
        raise ConfigError(message if message.startswith("base.") else f"base.{message}") from exc
    for key, cls in (("buses", BusParams), ("branches", BranchParams), ("sources", SourceParams)):
        if isinstance(data.get(key), dict):
            data[key] = {name: _section(cfg, cls, base.parameter_scales(), f"{key}.{name}")
                         for name, cfg in data[key].items()}
    if isinstance(data.get("units"), dict):
        units = {}
        for name, cfg in data["units"].items():
            if not isinstance(cfg, dict):
                units[name] = cfg
                continue
            unit_base = base if cfg.get("s_base") is None else build(BaseValues, {
                "s_base": cfg["s_base"], "v_ll_rms": base.v_ll_rms, "f0": base.f0})
            dc = cfg.get("dclink")
            if not isinstance(dc, dict) or "vdc_ref" not in dc:
                raise ConfigError(f"units.{name}.dclink.vdc_ref: required value missing")
            vdc_ref = _numeric_string(dc["vdc_ref"])
            if (isinstance(vdc_ref, bool) or not isinstance(vdc_ref, (int, float))
                    or not math.isfinite(vdc_ref) or vdc_ref <= 0):
                raise ConfigError(f"units.{name}.dclink.vdc_ref must be finite and positive")
            scales = unit_base.parameter_scales(vdc_ref)
            cfg = _section(cfg, UnitParams, scales, f"units.{name}")
            control = cfg.get("control")
            if isinstance(control, dict) and isinstance(control.get("loops"), dict):
                loops = {}
                for loop_name, loop in control["loops"].items():
                    if is_dataclass(loop):
                        loop = to_dict(loop)
                    cls = LOOP_SCHEMAS.get(loop.get("type")) if isinstance(loop, dict) else None
                    loops[loop_name] = (_section(loop, cls, scales, f"units.{name}.control.loops.{loop_name}")
                                        if cls is not None else loop)
                control["loops"] = loops
            units[name] = cfg
        data["units"] = units
    return data


def load(path: str | Path, initial: str | Path | None = None,
         initial_time: Optional[float] = None, **overrides: Any) -> Params:
    """Load a YAML/JSON configuration, optionally replace its initial state, and apply overrides.

    ``initial``: states CSV (last row, or the row at ``initial_time`` in s) or YAML/JSON.
    ``overrides``: dotted paths, e.g. ``load("case.yaml", **{"simulation.t_end": 5.0})``.
    """
    p = from_dict(read_tree(path))
    if initial is not None:
        init = read_initial(initial, initial_time)
        merged = to_dict(p)
        merged["initial"] = init
        if merged["simulation"]["t_end"] <= init.get("t", 0.0):
            # keep the configured duration after the new start time
            merged["simulation"]["t_end"] = init.get("t", 0.0) + p.simulation.t_end
        p = from_dict(merged)
    if overrides:
        p = p.replace(**overrides)
    return p


def read_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
        raise ConfigError(f"{path}: reading YAML needs PyYAML (pip install PyYAML), or write the "
                          f"configuration as .json") from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_initial(path: str | Path, t: Optional[float] = None) -> dict:
    """Read an ``initial`` mapping (``t``, ``states``) from YAML/JSON or a states CSV.

    CSV: last row, or the row at ``t`` (s); empty/NaN cells are omitted.
    YAML/JSON: the mapping itself or an ``initial`` block; ``t`` overrides its time.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        if len(rows) < 2 or rows[0][0] != "t":
            raise ConfigError(f"{path}: not a states table (first column must be 't')")
        header, body = rows[0], rows[1:]
        if t is None:
            row = body[-1]
        else:
            times = [float(r[0]) for r in body]
            k = min(range(len(times)), key=lambda i: abs(times[i] - t))
            if abs(times[k] - t) > 1e-9 * max(1.0, abs(t)):
                raise ConfigError(f"{path}: no row at t = {t} (nearest: {times[k]})")
            row = body[k]
        states = {}
        for name, cell in zip(header[1:], row[1:]):
            if cell.strip() == "":
                continue
            value = float(cell)
            if not math.isnan(value):
                states[name] = value
        return {"t": float(row[0]), "states": states}
    data = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else read_yaml(path)
    if isinstance(data, dict) and "initial" in data:
        data = data["initial"]
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected an 'initial' mapping")
    if t is not None:
        data = {**data, "t": t}
    return data


def read_tree(path: str | Path) -> dict:
    """Return the top-level mapping of a ``.yaml``/``.yml``/``.json`` configuration file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else read_yaml(path)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def dump(p: Any, path: str | Path) -> None:
    """Write ``p`` as a configuration file (YAML if PyYAML is installed, else JSON)."""
    path = Path(path)
    d = to_dict(p)
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    except ImportError:  # pragma: no cover
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
