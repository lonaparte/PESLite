"""``.cir`` simulation files: a SPICE-style netlist of the case.

A ``.cir`` file describes the topology and every device with netlist cards::

    gfl-example                                      title (line 1)
    .options sbase=2000000.0 vbase=690.0 freq=50.0   base (CodeSim names)
    Cpcc pcc 0 c_pu=0.02 r_d_pu=0.5                  bus capacitor (C card to 0)
    Lline pcc bus2 x_pu=0.05 r_pu=0.005              series R-L branch (L card)
    Xgrid pcc 0 source x_pu=0.4 r_pu=0.04            source, element or unit (X card)
    Xvsc pcc 0 vsc ac_filter.l_f_pu=0.2 pwm.f_sw=20000.0
    + ctrl.type=gfl ctrl.loops.SRFPLL.kp_pu=20.0 ...
    .event connect_vsc type=connect target=vsc t=0.2 ramp=1.0
    .simulation solver.dt=2.5e-05 output.record_every=10
    .tran 0.0005 3.0
    .end

``X`` cards name a PESLite device model (``source``, a unit type such as ``vsc`` or an element
type such as ``load``); nested parameters use dotted keys. A positional value on an R, L or C card
is in SI (``.options units=pu``: per unit on the system base). Values are YAML scalars or flow
collections (``[1, 2]``, ``{}``); ``{expr}`` evaluates an expression over ``.param`` values.

A plain CodeSim netlist (no ``.simulation`` card) is imported with CodeSim's meaning: R, L and C
cards, ``Infbus`` sources behind one R-L branch, ``.options`` and ``.tran``.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

import yaml

from .syntax import SimulationFileError, parse_native_text

CIR_SUFFIX = ".cir"
_PEAK = math.sqrt(2.0 / 3.0)  # phase-peak volts per line-line RMS volt
_OPTIONS = {"sbase": "s_base", "vbase": "v_ll_rms", "freq": "f0"}
_CODESIM_OPTIONS = {"units", "method", "reltol", "abstol", "residualtol"}
_SECTIONS = {".event": "events", ".signal": "signals", ".bus": "buses"}
_IGNORED = {".probe", ".save", ".meas", ".measure", ".title", ".global", ".ac", ".dc", ".noise",
            ".tf", ".four", ".op", ".step", ".temp", ".wave", ".backanno", ".net", ".fra",
            ".ic", ".nodeset", ".savestate", ".loadstate", ".savebias", ".text"}
_WIDTH = 100


# ------------------------------------------------------------------ values

_SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "mil": 25.4e-6, "m": 1e-3,
           "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
_INTEGER = re.compile(r"^[+-]?\d+$")
_NUMBER = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(meg|mil|[tgkmunpf])?(\d*)[a-z]*$")
_PLAIN = re.compile(r"^[A-Za-z_][\w.\-]*$")
_FUNCTIONS = {"sqrt": math.sqrt, "exp": math.exp, "ln": math.log, "log": math.log,
              "log10": math.log10, "sin": math.sin, "cos": math.cos, "tan": math.tan,
              "atan": math.atan, "atan2": math.atan2, "abs": abs, "min": min, "max": max,
              "pow": pow, "hypot": math.hypot}


def _number(token: str) -> float | None:
    """A SPICE number (``1m`` is milli, ``1meg`` mega, ``6k34`` 6340, ``10uF`` 1e-5)."""
    match = _NUMBER.match(token.lower())
    if match is None:
        return None
    mantissa, suffix, decimals = match.groups()
    if suffix is None:
        return float(mantissa)
    return float(f"{mantissa}.{decimals}" if decimals else mantissa) * _SUFFIX[suffix]


def _evaluate(text: str, params: dict[str, float]) -> float:
    """A number, or a ``{...}``/``'...'`` arithmetic expression over ``.param`` values."""
    value = _number(text)
    if value is not None:
        return value
    source = text.strip()
    if source[:1] in "{'" and source[-1:] in "}'":
        source = source[1:-1]
    source = source.replace("^", "**")
    lowered = {key.lower(): item for key, item in params.items()}

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id.lower() in lowered:
                return lowered[node.id.lower()]
            if node.id.lower() == "pi":
                return math.pi
            raise ValueError(f"unknown name {node.id}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = walk(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            operations = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                          ast.Mult: lambda: left * right, ast.Div: lambda: left / right,
                          ast.Pow: lambda: left ** right, ast.Mod: lambda: left % right}
            if type(node.op) in operations:
                return float(operations[type(node.op)]())
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id.lower() in _FUNCTIONS and not node.keywords):
            return float(_FUNCTIONS[node.func.id.lower()](*(walk(arg) for arg in node.args)))
        raise ValueError("unsupported expression")

    try:
        return walk(ast.parse(source, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as exc:
        raise ValueError(f"cannot evaluate {text!r}: {exc}") from None


def _try(value: str, params: dict[str, float]) -> float | None:
    try:
        return _evaluate(value, params)
    except ValueError:
        return None


def _value(token: str, params: dict[str, float]) -> Any:
    """A keyword value: integer, SPICE number, ``{expr}``, or a YAML scalar/flow collection."""
    if _INTEGER.match(token):
        return int(token)
    number = _number(token)
    if number is not None:
        return number
    if token.startswith("{") and token != "{}" and ":" not in token:
        return _evaluate(token, params)
    return yaml.safe_load(token)


def _text(value: Any) -> str:
    """Write a value so that :func:`_value` reads it back unchanged."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ".nan"
        if math.isinf(value):
            return ".inf" if value > 0 else "-.inf"
        return repr(value)
    if isinstance(value, str):
        if _NUMBER.match(value.lower()) and " " not in value:
            return value  # YAML reads 2.0e6 as text; PESLite converts it like a number
        plain = _PLAIN.match(value) and yaml.safe_load(value) == value
        return value if plain else json.dumps(value)
    return yaml.safe_dump(value, default_flow_style=True, width=math.inf).strip()


# ------------------------------------------------------------------ cards

def _card(line: str) -> str:
    """A card without its CodeSim inline comment (``;`` or `` $ ``) outside quotes."""
    quote = ""
    for index, character in enumerate(line):
        if quote:
            if character == quote:
                quote = ""
        elif character in "\"'":
            quote = character
        elif character == ";" or (character == "$" and line[index - 1:index].isspace()
                                   and line[index + 1:index + 2].isspace()):
            return line[:index].strip()
    return line.strip()


def _tokens(card: str) -> tuple[list[str], dict[str, str]]:
    """Positional tokens and ``key=value`` pairs; brackets and quotes stay one token."""
    pieces: list[str] = []
    current, depth, quote = "", 0, ""
    for character in card:
        if quote:
            current += character
            if character == quote:
                quote = ""
            continue
        if character in "\"'":
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        if depth == 0 and (character.isspace() or character in ",="):
            if current:
                pieces.append(current)
            current = ""
            if character == "=":
                pieces.append("=")
            continue
        current += character
    if current:
        pieces.append(current)
    positional: list[str] = []
    named: dict[str, str] = {}
    index = 0
    while index < len(pieces):
        if index + 1 < len(pieces) and pieces[index + 1] == "=":
            named[pieces[index]] = pieces[index + 2] if index + 2 < len(pieces) else ""
            index += 3
        else:
            positional.append(pieces[index])
            index += 1
    return positional, named


def _cards(text: str, path: str | Path) -> tuple[str, list[tuple[int, str]]]:
    """The title and ``(line, card)`` pairs, with continuations joined, up to ``.end``."""
    cards: list[tuple[int, str]] = []
    title: str | None = None
    for number, raw in enumerate(text.removeprefix("\ufeff").splitlines(), start=1):
        if not raw.strip():
            continue
        if title is None:
            title = raw.strip().lstrip("*").strip()
            continue
        if raw.lstrip().startswith("*"):
            continue
        card = _card(raw)
        if not card:
            continue
        if card.startswith("+"):
            if not cards:
                raise SimulationFileError("continuation before any card", path=path, line=number)
            cards[-1] = (cards[-1][0], f"{cards[-1][1]} {card[1:].strip()}")
            continue
        if card.split()[0].lower() == ".end":
            break
        cards.append((number, card))
    return title or "", cards


def _nest(target: dict[str, Any], key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    for parent in parents:
        target = target.setdefault(parent, {})
        if not isinstance(target, dict):
            raise ValueError(f"{parent} is already a value")
    target[leaf] = value


def _flatten(mapping: dict[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Dotted keys; a mapping whose own keys contain dots (paths) stays one flow value."""
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, dict) and value and not any("." in str(item) for item in value):
            yield from _flatten(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


# ------------------------------------------------------------------ reading

def read_cir_text(text: str, path: str | Path = "<string>") -> tuple[dict[str, Any], str]:
    """Return ``(parameters, generated @circuit view)`` of a ``.cir`` netlist."""
    from .addons import register_installed
    from .params import UNIT_TYPES
    from ..components.network import ELEMENT_TYPES
    register_installed()

    title, cards = _cards(text, path)
    native = any(card.split()[0].lower() == ".simulation" for _, card in cards)

    def fail(message: str, number: int) -> SimulationFileError:
        return SimulationFileError(message, path=path, line=number)

    raw_params: dict[str, tuple[int, str]] = {}
    for number, card in cards:
        if card.split()[0].lower() in {".param", ".params"}:
            positional, named = _tokens(card)
            if len(positional) > 1:
                raise fail(".param needs name=value pairs", number)
            raw_params.update({key: (number, value) for key, value in named.items()})
    params: dict[str, float] = {}
    pending = dict(raw_params)
    while pending:  # .param values may be written in any order
        solved = {key: result for key, (_, value) in pending.items()
                  if (result := _try(value, params)) is not None}
        if not solved:
            key, (number, value) = next(iter(pending.items()))
            raise fail(f".param {key}: cannot evaluate {value!r}", number)
        params.update(solved)
        pending = {key: item for key, item in pending.items() if key not in solved}

    tree: dict[str, Any] = {"base": {}, "buses": {}, "branches": {}, "sources": {},
                            "elements": {}, "units": {}}
    units_pu = False
    tran: tuple[float, float] | None = None
    branches: list[dict[str, Any]] = []
    capacitors: dict[str, tuple[float | None, dict[str, Any]]] = {}
    infbus: dict[str, tuple[str, float, float, int]] = {}
    devices: list[tuple[str, str, list[str], dict[str, Any], int]] = []

    def number_of(token: str, line: int) -> float:
        try:
            return _evaluate(token, params)
        except ValueError as exc:
            raise fail(str(exc), line) from None

    def keywords(named: dict[str, str], line: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in named.items():
            try:
                _nest(result, key, _value(item, params))
            except (ValueError, yaml.YAMLError) as exc:
                raise fail(f"{key}: {exc}", line) from None
        return result

    def node(token: str) -> str:
        return "ground" if token.lower() in {"0", "gnd", "ground"} else token

    for number, card in cards:
        positional, named = _tokens(card)
        head = positional[0].lower()
        if head in {".param", ".params"} or head in _IGNORED:
            continue
        if head in {".options", ".option", ".opt"}:
            for key, item in named.items():
                if key.lower() in _OPTIONS:
                    tree["base"][_OPTIONS[key.lower()]] = number_of(item, number)
                elif key.lower() == "units":
                    if item.lower() not in {"si", "pu"}:
                        raise fail(f"units={item} is not supported", number)
                    units_pu = item.lower() == "pu"
                elif key.lower() not in _CODESIM_OPTIONS:
                    tree["base"].update(keywords({key: item}, number))
            continue
        if head == ".tran":
            if len(positional) < 3:
                raise fail(".tran needs tstep and tstop", number)
            tran = (number_of(positional[1], number), number_of(positional[2], number))
            continue
        if head in {".simulation", ".meta"}:
            tree.setdefault(head[1:], {}).update(keywords(named, number))
            continue
        if head in _SECTIONS:
            if len(positional) != 2:
                raise fail(f"{head} needs one name, then key=value pairs", number)
            section = tree.setdefault(_SECTIONS[head], {})
            section[positional[1]] = {**section.get(positional[1], {}), **keywords(named, number)}
            continue
        if head.startswith("."):
            raise fail(f"{head} is not supported in a .cir file", number)
        letter, name = head[0], positional[0][1:]
        if letter in "rl" and len(positional) in {3, 4}:
            other_key = "l" if letter == "r" else "r"
            other = next((item for key, item in named.items() if key.lower() == other_key), None)
            extra = keywords({key: item for key, item in named.items()
                              if key.lower() != other_key}, number)
            value = number_of(positional[3], number) if len(positional) == 4 else None
            second = number_of(other, number) if other is not None else None
            r, l = (value, second) if letter == "r" else (second, value)
            branches.append({"name": name, "a": node(positional[1]), "b": node(positional[2]),
                             "r": r, "l": l, "extra": extra, "line": number,
                             "plain": letter if not named else None})
        elif letter == "c" and len(positional) in {3, 4}:
            a, b = node(positional[1]), node(positional[2])
            if "ground" not in (a, b) or a == b:
                raise fail(f"{positional[0]}: one capacitor node must be 0", number)
            bus = b if a == "ground" else a
            value = number_of(positional[3], number) if len(positional) == 4 else None
            if bus in capacitors:  # CodeSim: capacitors on one bus add up
                previous, extra = capacitors[bus]
                if value is None or previous is None or named or extra:
                    raise fail(f"{positional[0]}: bus {bus} already has a C card", number)
                value += previous
            capacitors[bus] = (value, keywords(named, number))
        elif letter == "x" and len(positional) >= 2:
            model, nodes = positional[-1], [node(item) for item in positional[1:-1]]
            if model.lower() == "infbus":
                if len(nodes) != 2 or nodes[1] != "ground":
                    raise fail(f"{positional[0]}: write X<name> <node> 0 Infbus", number)
                lowered = {key.lower(): item for key, item in named.items()}
                aref = lowered.get("aref", "1")
                if aref.lower().startswith("pwl"):
                    raise fail(f"{positional[0]}: PWL references are not supported", number)
                infbus[nodes[0]] = (name, number_of(aref, number),
                                    number_of(lowered.get("w0", "1"), number), number)
                continue
            devices.append((name, model, nodes, keywords(named, number), number))
        else:
            raise fail(f"{positional[0]}: unsupported card (use R, L, C or X)", number)

    # CodeSim: plain R and plain L cards on one node pair form one series branch
    merged: list[dict[str, Any]] = []
    for branch in branches:
        pair = frozenset((branch["a"], branch["b"]))
        partner = next((item for item in merged if item["plain"] and branch["plain"]
                        and item["plain"] != branch["plain"]
                        and frozenset((item["a"], item["b"])) == pair), None)
        if partner is None:
            merged.append(dict(branch))
            continue
        for key in ("r", "l"):
            if branch[key] is not None:
                partner[key] = branch[key]
        partner["plain"] = "rl"
        if branch["plain"] == "l":
            partner["name"] = branch["name"]

    base = tree["base"]
    if units_pu:  # L as reactance, C as susceptance, on the system base
        if "s_base" not in base or "v_ll_rms" not in base:
            raise SimulationFileError("units=pu needs .options sbase and vbase", path=path)
        z_base = base["v_ll_rms"] ** 2 / base["s_base"]
        w = 2.0 * math.pi * base.get("f0", 50.0)
        for branch in merged:
            branch["r"] = None if branch["r"] is None else branch["r"] * z_base
            branch["l"] = None if branch["l"] is None else branch["l"] * z_base / w
        capacitors = {bus: (None if c is None else c / (z_base * w), extra)
                      for bus, (c, extra) in capacitors.items()}

    view: list[str] = []
    for emf, (name, aref, w0, number) in infbus.items():
        touching = [item for item in merged if emf in (item["a"], item["b"])]
        if len(touching) != 1 or not (touching[0]["l"] or 0.0) > 0.0:
            raise fail(f"x{name}: an Infbus must reach the network through exactly one R-L "
                       "branch", number)
        branch = touching[0]
        merged.remove(branch)
        bus = branch["b"] if branch["a"] == emf else branch["a"]
        tree["sources"][name] = {"bus": bus, "l": branch["l"], "r": branch["r"] or 0.0,
                                 "v": aref * base.get("v_ll_rms", 1.0) * _PEAK,
                                 "f": w0 * base.get("f0", 50.0)}
        view.append(f"{name} -- {bus}")
    for branch in merged:
        name, a, b = branch["name"], branch["a"], branch["b"]
        values = {key: branch[key] for key in ("l", "r") if branch[key] is not None}
        if not native and "ground" in (a, b):  # CodeSim: a branch to ground is a load
            bus = b if a == "ground" else a
            if values.get("l", 0.0) > 0.0:
                tree["elements"][name] = {"type": "load", "bus": bus, **values, **branch["extra"]}
            else:
                tree["elements"][name] = {"type": "resistor", "bus": bus, "r": values.get("r", 0.0)}
            view.append(f"{bus} -- {name}")
        else:
            tree["branches"][name] = {"bus1": a, "bus2": b, **values, **branch["extra"]}
            view.append(" -- ".join(item for item in (a, b) if item != "ground"))
    for bus, (c, extra) in capacitors.items():
        entry = tree["buses"].setdefault(bus, {})
        entry.update({"c": c} if c is not None else {})
        entry.update(extra)
    for name, model, nodes, values, number in devices:
        if len(nodes) == 2 and nodes[1] == "ground" and "bus" not in values:
            values = {"bus": nodes[0], **values}
        elif len(nodes) == 1 and "bus" not in values:
            values = {"bus": nodes[0], **values}
        elif len(nodes) == 2:
            values = {"bus1": nodes[0], "bus2": nodes[1], **values}
        elif nodes:
            raise fail(f"x{name}: write the nodes as <bus> 0, or as bus1=... bus2=...", number)
        if model == "source":
            tree["sources"][name] = values
        elif model in UNIT_TYPES:
            tree["units"][name] = {"type": model, **values}
        elif model in ELEMENT_TYPES:
            tree["elements"][name] = {"type": model, **values}
        else:
            known = ", ".join(sorted({"source", *UNIT_TYPES, *ELEMENT_TYPES}))
            raise fail(f"x{name}: unknown device model {model}; known: {known}", number)
        ends = [values[key] for key in ("bus", "bus1", "bus2")
                if isinstance(values.get(key), str) and values[key] != "ground"]
        view.append(f"{ends[0]} -- {name} -- {ends[1]}" if len(ends) == 2
                    else " -- ".join([name, *ends]))
    connected = {bus for entry in (*tree["branches"].values(), *tree["sources"].values(),
                                   *tree["elements"].values(), *tree["units"].values())
                 for key in ("bus", "bus1", "bus2") if isinstance(bus := entry.get(key), str)}
    for bus in sorted(connected - {"ground"} - set(tree["buses"])):
        if "." not in bus:  # unit.port is a unit's port, not a bus
            tree["buses"][bus] = {}
    shown = {word for line in view for word in line.split(" -- ")}
    view += [bus for bus in tree["buses"] if bus not in shown]

    simulation = tree.setdefault("simulation", {})
    if tran is not None:
        simulation["t_end"] = tran[1]
        if not native:  # CodeSim's output step; a PESLite file sets it in .simulation
            simulation.setdefault("output", {})["period"] = tran[0]
    if not native and title:
        tree["meta"] = {"title": title}
    tree = {key: value for key, value in tree.items() if value or key in {"base", "simulation"}}
    return tree, "@circuit\n" + "\n".join(view) + "\n"


# ------------------------------------------------------------------ writing

def _line(head: str, values: dict[str, Any], positional: tuple[str, ...] = ()) -> list[str]:
    """One card, continued on ``+`` lines when it is long."""
    words = [head, *positional, *(f"{key}={_text(value)}" for key, value in _flatten(values))]
    lines, current = [], words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) > _WIDTH:
            lines.append(current)
            current = "+ " + word
        else:
            current += " " + word
    return [*lines, current]


def write_cir_text(tree: dict[str, Any], title: str | None = None) -> str:
    """A ``.cir`` netlist for a parameter mapping (as read from a ``.pes`` file)."""
    from .. import __version__

    def node(bus: Any) -> str:
        return "0" if bus == "ground" else str(bus)

    def drop(mapping: dict[str, Any] | None, *keys: str) -> dict[str, Any]:
        return {key: value for key, value in (mapping or {}).items()
                if key not in keys and value is not None}

    meta = drop(tree.get("meta"))
    lines = [(title or meta.get("title") or "PESLite case").splitlines()[0],
             f"* PESLite {__version__} netlist. X cards are PESLite device models",
             "* (source, unit and element types); dotted keys set nested parameters."]
    base = drop(tree.get("base"))
    options = {short: base.pop(long) for short, long in _OPTIONS.items() if long in base}
    lines += _line(".options", {**options, **base, "units": "si"})

    for name, bus in (tree.get("buses") or {}).items():
        bus = drop(bus)
        if "c" in bus:
            lines += _line(f"C{name}", drop(bus, "c"), (str(name), "0", _text(bus["c"])))
        elif "c_pu" in bus:
            lines += _line(f"C{name}", bus, (str(name), "0"))
        else:
            lines += _line(".bus", bus, (str(name),))
    for name, branch in (tree.get("branches") or {}).items():
        branch = drop(branch)
        nodes = (node(branch.pop("bus1", "")), node(branch.pop("bus2", "")))
        value = (_text(branch.pop("l")),) if "l" in branch else ()
        lines += _line(f"L{name}", branch, (*nodes, *value))
    devices = [(name, "source", drop(source))
               for name, source in (tree.get("sources") or {}).items()]
    devices += [(name, str(entry.get("type")), drop(entry, "type"))
                for section in ("elements", "units")
                for name, entry in (tree.get(section) or {}).items()]
    for name, model, values in devices:
        nodes = (node(values.pop("bus")), "0") if isinstance(values.get("bus"), str) else ()
        lines += _line(f"X{name}", values, (*nodes, model))
    for directive, section in ((".event", "events"), (".signal", "signals")):
        for name, entry in (tree.get(section) or {}).items():
            lines += _line(directive, drop(entry), (str(name),))
    simulation = drop(tree.get("simulation"))
    t_end = simulation.pop("t_end", None)
    lines += _line(".simulation", simulation)
    if meta:
        lines += _line(".meta", meta)
    if t_end is not None:
        output, solver = simulation.get("output") or {}, simulation.get("solver") or {}
        tstep = output.get("period") or solver.get("dt") or 1e-4
        lines.append(f".tran {_text(float(tstep))} {_text(float(t_end))}")
    lines.append(".end")
    return "\n".join(lines) + "\n"


def pes_text(tree: dict[str, Any], interface: str) -> str:
    """A ``.pes`` file for a parameter mapping and its interface view."""
    body = yaml.safe_dump(tree, sort_keys=False, allow_unicode=True)
    return f"{interface.strip()}\n\n{body}"


def convert_text(text: str, path: str | Path) -> str:
    """A ``.cir`` netlist for a ``.pes`` or ``.cir`` source text."""
    path = Path(path)
    if path.suffix.lower() == CIR_SUFFIX:
        tree, _ = read_cir_text(text, path)
        title = next((line.strip() for line in text.splitlines() if line.strip()), path.stem)
    else:
        tree, _ = parse_native_text(text, path)
        title = path.stem
    return write_cir_text(tree, title)


__all__ = ["CIR_SUFFIX", "convert_text", "pes_text", "read_cir_text", "write_cir_text"]
