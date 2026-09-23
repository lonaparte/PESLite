"""Named-state helpers: dotted-path composition, flattening to real columns and input resolution.

State values are float, complex or bool.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .protocols import Stateful

__all__ = ["join", "gather", "scatter", "flatten", "resolve", "expand_aliases", "assign", "is_unset"]


def join(prefix: str, name: str) -> str:
    """Join two name parts with a dot; an empty part is dropped."""
    if not prefix:
        return name
    return f"{prefix}.{name}" if name else prefix


def gather(parts: Mapping[str, Any]) -> dict[str, Any]:
    """Return the states of the named parts with keys prefixed by the part names.

    Parts without ``get_state`` are skipped; an empty part name adds no prefix.
    """
    out: dict[str, Any] = {}
    for pname, part in parts.items():
        get = getattr(part, "get_state", None)
        if get is None:
            continue
        if pname:
            for key, value in get().items():
                out[f"{pname}.{key}" if key else pname] = value
        else:
            out.update(get())
    return out


def scatter(parts: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    """Pass each part the entries of ``values`` prefixed with its name."""
    if not values:
        return
    for pname, part in parts.items():
        if not isinstance(part, Stateful):
            continue
        own = {key: values[join(pname, key)] for key in part.get_state() if join(pname, key) in values}
        if own:
            part.set_state(own)


def flatten(states: Mapping[str, Any]) -> dict[str, float]:
    """Convert states to real columns: complex ``x`` -> ``x.re``, ``x.im``; flags -> 0.0 / 1.0."""
    out: dict[str, float] = {}
    for key, value in states.items():
        if isinstance(value, complex):
            out[key + ".re"] = value.real
            out[key + ".im"] = value.imag
        elif isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        else:
            out[key] = float(value)
    return out


def is_unset(value: Any) -> bool:
    """Return True if ``value`` is NaN (real or complex), which marks an unset state."""
    if isinstance(value, complex):
        return math.isnan(value.real) or math.isnan(value.imag)
    return isinstance(value, float) and math.isnan(value)


def expand_aliases(given: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    """Rename alias keys (and their ``.re`` / ``.im`` parts) to canonical names."""
    out: dict[str, Any] = {}
    for key, value in given.items():
        for alias, canonical in aliases.items():
            if key == alias or key.startswith(alias + "."):
                key = canonical + key[len(alias):]
                break
        if key in out:
            raise ValueError(f"state {key!r} is given twice (directly and through an alias)")
        out[key] = value
    return out


def _convert(where: str, target: Any, value: Any, presets: Any, name: str = "") -> Any:
    if isinstance(value, str):
        word = value.strip()
        if callable(presets):  # presets(state name, keyword)
            try:
                value = presets(name or where, word)
            except ValueError as exc:
                raise ValueError(f"{where}: {exc.args[0] if exc.args else exc}") from None
        elif word in presets:
            value = presets[word]
        else:
            known = ", ".join(sorted(presets)) or "none"
            raise ValueError(f"{where}: unknown keyword {value!r} (keywords here: {known})")
    if isinstance(value, (list, tuple)):
        if not isinstance(target, complex) or len(value) != 2:
            raise ValueError(f"{where}: a [re, im] pair only fits a complex state")
        return complex(float(value[0]), float(value[1]))
    if isinstance(target, complex):
        return complex(value)
    if isinstance(value, complex):
        raise ValueError(f"{where}: complex value {value} for a real state")
    if isinstance(target, bool):
        return bool(round(float(value)))
    return float(value)


def resolve(template: Mapping[str, Any], given: Mapping[str, Any],
            presets: Any = None, where: str = "initial.states") -> dict[str, Any]:
    """Convert the entries of ``given`` to the types of the matching ``template`` entries.

    ``template``: state name -> current value. ``given``: state names or ``x.re``/``x.im`` parts;
    values are a number, ``[re, im]``, 0/1 or true/false for a flag, or a keyword of ``presets``.
    ``presets``: mapping keyword -> value, or callable ``(state name, keyword) -> value``.
    Raises KeyError for unknown names.
    """
    presets = presets or {}
    out: dict[str, Any] = {}
    parts: dict[str, dict[str, float]] = {}
    for name, value in given.items():
        if name in template:
            out[name] = _convert(f"{where}.{name}", template[name], value, presets, name)
            continue
        base, _, part = name.rpartition(".")
        if part in ("re", "im") and isinstance(template.get(base), complex):
            parts.setdefault(base, {})[part] = _convert(f"{where}.{name}", 0.0, value, presets, base)
            continue
        known = ", ".join(flatten(template))
        raise KeyError(f"{where}: unknown state {name!r}. Known states: {known}")
    for base, pr in parts.items():
        if base in out:
            raise ValueError(f"{where}: {base!r} is given both whole and by its .re/.im parts")
        t = template[base]
        out[base] = complex(pr.get("re", t.real), pr.get("im", t.imag))
    return out


def assign(obj: Any, values: Mapping[str, Any], names: tuple[str, ...]) -> None:
    """Set attributes of ``obj`` from ``values``; keys must be in ``names``."""
    for key, value in values.items():
        if key not in names:
            raise KeyError(f"{type(obj).__name__} has no state {key!r} (states: {', '.join(names)})")
        setattr(obj, key, value)
