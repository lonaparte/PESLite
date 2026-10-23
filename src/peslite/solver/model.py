"""Continuous-time model: what a subsystem is, subsystems connected by signals into one model, and
named states and automatically solved algebraic signal loops.

A subsystem has ``state``, ``inp`` and ``out`` records (:class:`Bag`) and state derivatives; the
:class:`Model` packs the states of its subsystems into a flat real vector (two entries per complex
state) named ``<subsystem>.<state>`` and evaluates their outputs and connections in signal order.
Strongly connected output/input dependencies are detected during assembly and reduced to
internally solved algebraic groups; individual subsystems do not need case-specific break rules.
Every part with a state (a model, a controller, a PWM peripheral, a solver) is :class:`Stateful`: it
gives its state by name, parts are composed by dotted names (``vsc.ctrl.SRFPLL``), flattened to the
real columns of ``states.csv`` and resolved from ``initial.states`` values.
"""

from __future__ import annotations

import keyword
import linecache
import math
import numbers
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from . import energy as _energy

__all__ = ["ConfigError", "Bag", "Empty", "OutputMethod", "Subsystem", "Model", "GroupPlan",
           "Stateful", "StateRegistry", "join", "gather", "scatter", "flatten", "resolve",
           "expand_aliases", "assign"]


# ------------------------------------------------------------------ subsystems

class ConfigError(ValueError):
    """Invalid model or solver configuration."""


def _real(value):
    """Normalize scalar numbers while preserving their arithmetic semantics."""
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, numbers.Real) else float(value)


def _complex(value):
    """Normalize complex numbers while preserving their arithmetic semantics."""
    if hasattr(value, "item"):
        value = value.item()
    return value + 0j if isinstance(value, numbers.Number) else complex(value)


def _integer(value):
    """Integral scalar normalization without erasing a number's arithmetic semantics."""
    if isinstance(value, numbers.Integral):
        return value
    truncate = getattr(value, "__trunc__", None)
    return truncate() if truncate is not None else int(value)


class Bag:
    """Record whose ``__slots__`` fields default to ``0.0`` unless given as keywords (``state``, ``inp``, ``out``)."""

    __slots__ = ()

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name, 0.0))


class Empty(Bag):
    """Record with no fields."""

    __slots__ = ()


@dataclass(frozen=True)
class OutputMethod:
    """One output method of a subsystem and its direct input dependencies.

    ``method``: name of a method taking ``t``; ``inputs``: ``inp`` fields it reads;
    ``outputs``: ``out`` fields it writes. Every output field needs exactly one producer.
    ``rhs`` may be false when no state derivative of the owning subsystem reads the stage.  The
    model still includes it when an RHS-required downstream signal depends on its outputs.
    """

    method: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    rhs: bool = True


@runtime_checkable
class Subsystem(Protocol):
    """Continuous block with ``state``, ``inp`` and ``out`` records and state derivatives.

    ``state_names``: packing order (a complex state takes two real entries);
    ``outputs_need_inputs``: whether ``set_outputs`` reads ``inp``; optional ``output_stages``
    (tuple of :class:`OutputMethod`) replaces ``set_outputs``. A subsystem may declare real-state
    unilateral constraints as ``state_bounds = {name: (lower, upper)}``, using ``None`` for an
    open side.
    ``out`` and connected ``inp`` fields are current when ``rhs`` runs.
    """

    state: Any
    inp: Any
    out: Any
    state_names: ClassVar[tuple[str, ...]]
    outputs_need_inputs: ClassVar[bool]

    def set_outputs(self, t: float) -> None:
        """Update ``out`` from ``state`` (and ``inp`` if ``outputs_need_inputs``)."""

    def rhs(self, t: float) -> Sequence[complex | float]:
        """Time derivatives of the states, in ``state_names`` order."""


# ------------------------------------------------------------------ named states

@runtime_checkable
class Stateful(Protocol):
    """Named state access for initialization and continuation.

    ``get_state()`` returns ``{name: float | complex | bool}`` (``""`` names the part itself);
    ``set_state(values)`` accepts any subset of those names.
    """

    def get_state(self) -> dict[str, Any]: ...

    def set_state(self, values: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class _RegisteredState:
    """One state whose public path and table columns were fixed during assembly."""

    local_name: str
    name: str
    columns: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class _RegisteredPart:
    """One state owner and its fixed local schema."""

    prefix: str
    part: Any
    states: tuple[_RegisteredState, ...]


def _state_kind(value: Any) -> str:
    if isinstance(value, complex):
        return "complex"
    if isinstance(value, bool):
        return "bool"
    return "real"


class StateRegistry:
    """The immutable state-name layout of one assembled simulation.

    Part names and their local state names are read once while the registry is built.  Later reads
    use that registered layout: a part may change state *values*, but it may not add, remove or
    change the representation of states during a run.  This makes ownership and the columns of the
    state table construction-time properties instead of consequences of repeated dictionary merges.
    """

    def __init__(self, parts: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> None:
        groups: list[_RegisteredPart] = []
        names: dict[str, str] = {}
        columns: dict[str, str] = {}
        bools: set[str] = set()
        prefixes: dict[str, str] = {}
        items = parts.items() if isinstance(parts, Mapping) else parts
        for prefix, part in items:
            owner = type(part).__name__
            if prefix in prefixes:
                raise ConfigError(
                    f"state prefix {prefix!r} is assigned to both {prefixes[prefix]!r} and {owner!r}"
                )
            prefixes[prefix] = owner
            get = getattr(part, "get_state", None)
            if get is None:
                continue
            values = get()
            if not isinstance(values, Mapping):
                raise TypeError(f"{type(part).__name__}.get_state() must return a mapping")
            registered: list[_RegisteredState] = []
            owner = prefix or type(part).__name__
            for local_name, value in values.items():
                if not isinstance(local_name, str):
                    raise TypeError(
                        f"{type(part).__name__}.get_state() returned non-string name {local_name!r}"
                    )
                name = join(prefix, local_name)
                if name in names:
                    raise ConfigError(
                        f"state name {name!r} is produced by both {names[name]!r} and {owner!r}"
                    )
                kind = _state_kind(value)
                state_columns = ((name + ".re", name + ".im") if kind == "complex" else (name,))
                for column in state_columns:
                    if column in columns:
                        raise ConfigError(
                            f"state-table column {column!r} is produced by both "
                            f"{columns[column]!r} and {name!r}"
                        )
                    columns[column] = name
                names[name] = owner
                if kind == "bool":
                    bools.add(name)
                registered.append(_RegisteredState(local_name, name, state_columns, kind))
            groups.append(_RegisteredPart(prefix, part, tuple(registered)))
        self._groups = tuple(groups)
        self.names = tuple(names)
        self.columns = tuple(columns)
        self.boolean_names = frozenset(bools)

    def read(self, prefixes: Iterable[str] | None = None) -> dict[str, Any]:
        """Read current values using the paths registered during assembly."""
        selected = None if prefixes is None else set(prefixes)
        out: dict[str, Any] = {}
        for group in self._groups:
            if selected is not None and group.prefix not in selected:
                continue
            values = group.part.get_state()
            for state in group.states:
                out[state.name] = values[state.local_name]
        return out

    @property
    def discrete_names(self) -> frozenset[str]:
        """Discrete coordinates from state values and optional owner lifecycle declarations.

        This view is requested by post-run loop maps, not maintained as another state layout.
        Float-valued lifecycle state such as a protection timer must be declared by its owner;
        boolean and integer state is discrete without a declaration.
        """
        names = set()
        for group in self._groups:
            declared = set(getattr(group.part, "discrete_state_names", ()))
            values = group.part.get_state()
            for state in group.states:
                if state.local_name in declared or isinstance(values[state.local_name], (bool, int)):
                    names.add(state.name)
        return frozenset(names)

    @property
    def linearization_columns(self) -> tuple[str, ...]:
        """Existing columns retained by an owner's optional smooth-coordinate declaration.

        Non-modal history remains in restart and recurrence checks without becoming an
        independent smooth coordinate. Owners without a declaration retain their complete schema.
        """
        columns = []
        for group in self._groups:
            selected = getattr(group.part, "linearization_state_names", None)
            selected = None if selected is None else set(selected)
            for state in group.states:
                if selected is None or state.local_name in selected:
                    columns.extend(state.columns)
        return tuple(columns)

    def periodic_state(self, row: Mapping[str, float]) -> dict[str, float]:
        """Compare existing snapshots in an owner's logical, rather than storage, order.

        An optional ``periodic_state(values)`` projects an owner's local named states onto
        equivalent existing columns, for example a ring buffer in chronological order.
        The original restart row and registered layout are not changed.
        """
        result = dict(row)
        for group in self._groups:
            project = getattr(group.part, "periodic_state", None)
            if project is None:
                continue
            local = {state.local_name: (complex(*(row[key] for key in state.columns))
                                       if state.kind == "complex" else row[state.name])
                     for state in group.states}
            projected = flatten(project(local))
            for state in group.states:
                for key in state.columns:
                    local_key = key[len(group.prefix) + 1:] if group.prefix else key
                    if local_key in projected:
                        result[key] = projected[local_key]
                    else:
                        result.pop(key, None)
        return result

    def read_flat(self, prefixes: Iterable[str] | None = None) -> dict[str, float]:
        """Read the fixed real-valued state-table columns using the registered layout.

        Construction fixes each owner's names, representations and output columns.  The normal
        snapshot path therefore trusts that contract: it avoids rebuilding a set of names or
        rediscovering every value's kind.  A custom owner must not change its state schema after
        the registry has been constructed.
        """
        selected = None if prefixes is None else set(prefixes)
        out: dict[str, float] = {}
        for group in self._groups:
            if selected is not None and group.prefix not in selected:
                continue
            values = group.part.get_state()
            for state in group.states:
                value = values[state.local_name]
                if state.kind == "complex":
                    out[state.columns[0]] = value.real
                    out[state.columns[1]] = value.imag
                elif state.kind == "bool":
                    out[state.columns[0]] = 1.0 if value else 0.0
                else:
                    out[state.columns[0]] = float(value)
        return out

    def load(self, values: Mapping[str, Any], prefixes: Iterable[str] | None = None) -> None:
        """Load a subset of registered values into selected owners."""
        if not values:
            return
        selected = None if prefixes is None else set(prefixes)
        for group in self._groups:
            if selected is not None and group.prefix not in selected:
                continue
            own = {state.local_name: values[state.name]
                   for state in group.states if state.name in values}
            if not own:
                continue
            set_state = getattr(group.part, "set_state", None)
            if set_state is None:
                owner = group.prefix or type(group.part).__name__
                raise ConfigError(f"state owner {owner!r} does not support loading states")
            set_state(own)


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
    owners: dict[str, str] = {}
    for pname, part in parts.items():
        get = getattr(part, "get_state", None)
        if get is None:
            continue
        for key, value in get().items():
            name = join(pname, key)
            if name in out:
                raise ConfigError(
                    f"state name {name!r} is produced by both {owners[name]!r} and {pname!r}"
                )
            out[name] = value
            owners[name] = pname
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
        return _complex(value)
    if isinstance(value, complex):
        raise ValueError(f"{where}: complex value {value} for a real state")
    if isinstance(target, bool):
        return bool(round(float(value)))
    return _real(value)


def resolve(template: Mapping[str, Any], given: Mapping[str, Any],
            presets: Any = None, where: str = "simulation.initial.states") -> dict[str, Any]:
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


def assign(obj: Any, values: Mapping[str, Any], names: tuple[str, ...] | Mapping[str, str]) -> None:
    """Set attributes from named states.

    ``names`` is either a tuple whose state and attribute names agree, or a
    mapping from public state names to internal attribute names.
    """
    attrs = names if isinstance(names, Mapping) else {name: name for name in names}
    for key, value in values.items():
        if key not in attrs:
            raise KeyError(f"{type(obj).__name__} has no state {key!r} (states: {', '.join(attrs)})")
        setattr(obj, attrs[key], value)


# ------------------------------------------------------------------ the model

Connection = tuple[Any, str]


def _copy_value(operation: tuple) -> Any:
    """Value produced by one connection-copy operation, without assigning it."""
    _dst, _field, src, src_field, fanin = operation
    if fanin is None:
        return getattr(src, src_field)
    total = 0
    for out, name, gain in fanin:
        total += gain * getattr(out, name)
    return total


def _run_output_operation(operation: tuple, t: float) -> None:
    """Execute one uncompiled output/copy operation."""
    kind, item = operation
    if kind == "out":
        item(t)
    else:
        dst, field, _src, _src_field, _fanin = item
        setattr(dst, field, _copy_value(item))


def _signal_error(a: Any, b: Any) -> tuple[float, float]:
    """Absolute error and scale of two real, complex or array-valued signals."""
    if isinstance(a, (int, float, complex)) and isinstance(b, (int, float, complex)):
        return abs(a - b), max(1.0, abs(a), abs(b))
    aa, bb = np.asarray(a), np.asarray(b)
    delta = float(np.max(np.abs(aa - bb))) if aa.size else 0.0
    scale = max(1.0,
                float(np.max(np.abs(aa))) if aa.size else 0.0,
                float(np.max(np.abs(bb))) if bb.size else 0.0)
    return delta, scale


class _AlgebraicLoop:
    """Automatically solved strongly connected group of output stages and signal copies.

    A small set of connection copies is selected as tear variables while the model is assembled.
    The remaining operations are acyclic. Scalar groups use warm-started secant iteration;
    general groups use Gauss--Seidel sweeps. No participating subsystem owns the loop solve.
    """

    __slots__ = ("operations", "tears", "owners", "labels", "max_sweeps", "rtol", "atol",
                 "_inverse", "_scalar_sweep")
    step_limit = 10.0

    def __init__(self, operations: tuple, tears: tuple, owners: tuple,
                 labels: tuple[str, ...],
                 max_sweeps: int = 16, rtol: float = 1e-8, atol: float = 1e-10) -> None:
        self.tears = tears
        torn = {id(item) for item in tears}
        self.operations = tuple(operation for operation in operations
                                if not (operation[0] == "copy" and id(operation[1]) in torn))
        self.owners = owners
        self.labels = labels
        self.max_sweeps = max_sweeps
        self.rtol = rtol
        self.atol = atol
        self._inverse: complex | float | None = None
        self._scalar_sweep = self._compile_scalar_sweep() if len(tears) == 1 else None

    def _compile_scalar_sweep(self):
        """Compile the torn scalar assignment and its acyclic operations into one Python call."""
        env: dict[str, Any] = {}
        refs: dict[int, str] = {}

        def ref(obj: Any) -> str:
            key = refs.get(id(obj))
            if key is None:
                key = refs[id(obj)] = f"_{len(refs)}"
                env[key] = obj
            return key

        def get(obj: Any, field: str) -> str:
            return (f"{ref(obj)}.{field}" if field.isidentifier() and not keyword.iskeyword(field)
                    else f"getattr({ref(obj)}, {field!r})")

        def put(obj: Any, field: str, value: str) -> str:
            return (f"{ref(obj)}.{field} = {value}"
                    if field.isidentifier() and not keyword.iskeyword(field)
                    else f"setattr({ref(obj)}, {field!r}, {value})")

        def value(item: tuple) -> str:
            _dst, _field, src, src_field, fanin = item
            if fanin is None:
                return get(src, src_field)
            expression = "0"
            for out, name, gain in fanin:
                expression = f"({expression} + {ref(gain)} * {get(out, name)})"
            return expression

        tear = self.tears[0]
        body = [put(tear[0], tear[1], "x")]
        for kind, item in self.operations:
            if kind == "out":
                body.append(f"{ref(item)}(t)")
            else:
                body.append(put(item[0], item[1], value(item)))
        body.append("return " + value(tear))
        source = "def sweep(t, x):\n" + "\n".join("    " + line for line in body) + "\n"
        exec(compile(source, "<peslite algebraic scalar>", "exec"), env)
        return env["sweep"]

    def _evaluate_scalar(self, t: float, x: complex | float) -> complex | float:
        return self._scalar_sweep(t, x)

    def _close(self, a: Any, b: Any) -> bool:
        error, scale = _signal_error(a, b)
        return error <= self.atol + self.rtol * scale

    @staticmethod
    def _finite_scalar(value: Any) -> bool:
        return (isinstance(value, (int, float, complex))
                and math.isfinite(value.real if isinstance(value, complex) else value)
                and (not isinstance(value, complex) or math.isfinite(value.imag)))

    def _scalar(self, t: float) -> None:
        """Warm-started secant solve for the common one-scalar feedback case."""
        item = self.tears[0]
        dst, field, _src, _src_field, _fanin = item
        x0 = getattr(dst, field)
        y0 = self._evaluate_scalar(t, x0)
        r0 = y0 - x0
        if self._close(y0, x0):
            return

        inverse = self._inverse
        x1 = x0 - inverse * r0 if inverse is not None else y0
        scale = max(1.0, abs(x0), abs(y0))
        if (not self._finite_scalar(x1)) or abs(x1 - x0) > self.step_limit * scale:
            x1 = y0
        for _iteration in range(self.max_sweeps - 1):
            y1 = self._evaluate_scalar(t, x1)
            r1 = y1 - x1
            denominator = r1 - r0
            if denominator != 0:
                estimate = (x1 - x0) / denominator
                if self._finite_scalar(estimate):
                    self._inverse = estimate
            if self._close(y1, x1):
                return
            inverse = self._inverse
            x2 = x1 - inverse * r1 if inverse is not None else y1
            scale = max(1.0, abs(x1), abs(y1))
            if (not self._finite_scalar(x2)) or abs(x2 - x1) > self.step_limit * scale:
                x2 = y1
            x0, r0, x1 = x1, r1, x2
        raise FloatingPointError(
            "algebraic loop did not converge after "
            f"{self.max_sweeps} scalar iterations: " + ", ".join(self.labels)
        )

    def __call__(self, t: float) -> None:
        begun = []
        for owner in self.owners:
            begin = getattr(owner, "begin_algebraic", None)
            if begin is not None:
                begin()
                begun.append(owner)
        try:
            if len(self.tears) == 1:
                item = self.tears[0]
                dst, field, _src, _src_field, _fanin = item
                if isinstance(getattr(dst, field), (int, float, complex)):
                    self._scalar(t)
                    return
            worst = math.inf
            for _sweep in range(self.max_sweeps):
                for item in self.tears:
                    dst, field, _src, _src_field, _fanin = item
                    setattr(dst, field, _copy_value(item))
                for operation in self.operations:
                    _run_output_operation(operation, t)
                worst = 0.0
                converged = True
                for item in self.tears:
                    dst, field, _src, _src_field, _fanin = item
                    current = getattr(dst, field)
                    target = _copy_value(item)
                    error, scale = _signal_error(current, target)
                    worst = max(worst, error / scale)
                    if error > self.atol + self.rtol * scale:
                        converged = False
                if converged:
                    return
            raise FloatingPointError(
                "algebraic loop did not converge after "
                f"{self.max_sweeps} sweeps (relative residual {worst:.3g}): "
                + ", ".join(self.labels)
            )
        finally:
            for owner in begun:
                owner.end_algebraic()


class GroupPlan:
    """Subsystem group and its evaluation plans (see :meth:`Model.groups`).

    ``mask``: the group's entries of the state vector. ``plan``: updates all member inputs.
    ``own``: member outputs and intra-group copies only. ``feed``: copies from outside the group.
    """

    __slots__ = ("members", "mask", "plan", "own", "feed", "inputs", "rhs")

    def __init__(self, members: tuple, mask: NDArray[np.bool_], plan: tuple, own: tuple, feed: tuple,
                 inputs: tuple, rhs: tuple) -> None:
        self.members, self.mask, self.plan, self.own, self.feed, self.inputs, self.rhs = (
            members, mask, plan, own, feed, inputs, rhs)


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def _auto_names(subsystems: Sequence[Any]) -> list[str]:
    """Snake-case names from class names; repeated classes are numbered from 1."""
    base = [_snake(type(s).__name__) for s in subsystems]
    total, seen = Counter(base), Counter()
    names = []
    for b in base:
        seen[b] += 1
        names.append(f"{b}_{seen[b]}" if total[b] > 1 else b)
    return names


def _topological_order(dependencies, successors):
    """Kahn order and the nodes still blocked by cycles (not necessarily in cycles)."""
    indegree = [len(deps) for deps in dependencies]
    ready = deque(node for node, degree in enumerate(indegree) if degree == 0)
    order = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for nxt in successors[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    return order, [node for node, degree in enumerate(indegree) if degree]


def _strong_components(successors, nodes):
    """Iterative Tarjan on the induced subgraph of unsorted nodes only."""
    selected = set(nodes)
    indices, low = {}, {}
    stack, on_stack, components = [], set(), []
    for root in nodes:
        if root in indices:
            continue
        indices[root] = low[root] = len(indices)
        stack.append(root)
        on_stack.add(root)
        frames = [(root, iter(successors[root]))]
        while frames:
            node, edges = frames[-1]
            nxt = next(edges, None)
            if nxt is not None:
                if nxt not in selected:
                    continue
                if nxt not in indices:
                    indices[nxt] = low[nxt] = len(indices)
                    stack.append(nxt)
                    on_stack.add(nxt)
                    frames.append((nxt, iter(successors[nxt])))
                elif nxt in on_stack:
                    low[node] = min(low[node], indices[nxt])
                continue
            frames.pop()
            if frames:
                parent = frames[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == indices[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(tuple(sorted(component)))
    return components


class Model:
    """Connected subsystems evaluated as ``rhs(t, y)`` on a flat real state vector.

    ``subsystems``: sequence (named from class names) or mapping name -> subsystem;
    dots in names form namespaces.
    ``connections``: ``(dst, input) -> (src, output[, gain])`` or a list of these (summed).
    ``zoh_connections``: ``(dst, input) -> label`` for held inputs set by :meth:`set_zoh_input`.
    """

    def __init__(
        self,
        subsystems: Sequence[Subsystem] | Mapping[str, Subsystem],
        connections: Mapping[Connection, Connection | Sequence[Connection]],
        zoh_connections: Mapping[Connection, str] | None = None,
    ) -> None:
        if isinstance(subsystems, Mapping):
            names = [str(n) for n in subsystems]
            subsystems = list(subsystems.values())
        else:
            subsystems = list(subsystems)
            names = _auto_names(subsystems)
        for name in names:
            if not name or any(not part for part in name.split(".")):
                raise ValueError(f"subsystem name {name!r} must be non-empty and have no empty part "
                                 f"(a dot separates namespaces: 'vsc1.branch_f')")
        if len(set(names)) != len(names):
            raise ValueError(f"subsystem names must be unique, got {names}")
        if len(set(map(id, subsystems))) != len(subsystems):
            raise ValueError("a subsystem appears more than once")
        self.names: list[str] = names
        for sub in subsystems:
            if not isinstance(sub, Subsystem):
                raise TypeError(
                    f"{type(sub).__name__} does not satisfy the Subsystem protocol "
                    "(needs state/inp/out, state_names, outputs_need_inputs, set_outputs, rhs)")
        self.subsystems: list[Subsystem] = list(subsystems)
        self.connections = dict(connections)
        self.zoh_connections = dict(zoh_connections or {})
        self._zoh_targets: dict[str, list[tuple[Any, str]]] = {}
        for (dst, name), label in self.zoh_connections.items():
            self._zoh_targets.setdefault(label, []).append((dst.inp, name))
        self._check_connections()
        self._build_state_layout()
        self._build_evaluation_plan()
        self._compile()
        # energy declaration per subsystem (default if none declared)
        self.energy_specs: dict[str, _energy.EnergySpec] = {n: _energy.spec_of(s) for n, s in zip(names, subsystems)}

    # ---------------------------------------------------------------- setup
    def _check_connections(self) -> None:
        known = set(map(id, self.subsystems))
        for (dst, dst_name), src in self.connections.items():
            if id(dst) not in known:
                raise ValueError(f"connection target {type(dst).__name__} is not in the subsystem list")
            if not hasattr(dst.inp, dst_name):
                raise ValueError(f"{type(dst).__name__}.inp has no field {dst_name!r}")
            sources = src if isinstance(src, list) else [src]
            for entry in sources:
                s_obj, s_name = entry[0], entry[1]
                if id(s_obj) not in known:
                    raise ValueError(f"connection source {type(s_obj).__name__} is not in the subsystem list")
                if not hasattr(s_obj.out, s_name):
                    raise ValueError(f"{type(s_obj).__name__}.out has no field {s_name!r}")
        for (dst, dst_name) in self.zoh_connections:
            if not hasattr(dst.inp, dst_name):
                raise ValueError(f"{type(dst).__name__}.inp has no field {dst_name!r}")

    def _build_state_layout(self) -> None:
        """Map every state field to a slice of the flat real vector."""
        ops: list[tuple[Any, str, int, bool]] = []
        index = 0
        for sub in self.subsystems:
            for name in sub.state_names:
                value = getattr(sub.state, name)
                is_complex = isinstance(value, complex)
                ops.append((sub.state, name, index, is_complex))
                index += 2 if is_complex else 1
        self._state_ops = tuple(ops)
        self.n_states = index
        # per-subsystem packing of the rhs return values
        rhs_layout = []
        for sub in self.subsystems:
            entries = []
            for name in sub.state_names:
                for st, nm, idx, is_c in ops:
                    if st is sub.state and nm == name:
                        entries.append((idx, is_c))
                        break
            rhs_layout.append((sub, tuple(entries)))
        self._rhs_layout = tuple(rhs_layout)
        bounds = []
        for sub, entries in rhs_layout:
            declared = getattr(sub, "state_bounds", {})
            if not isinstance(declared, Mapping):
                raise TypeError(f"{type(sub).__name__}.state_bounds must be a mapping")
            unknown = set(declared) - set(sub.state_names)
            if unknown:
                raise ValueError(
                    f"{type(sub).__name__}.state_bounds contains unknown states {sorted(unknown)}"
                )
            for name, limits in declared.items():
                if not isinstance(limits, (tuple, list)) or len(limits) != 2:
                    raise TypeError(
                        f"{type(sub).__name__}.state_bounds[{name!r}] must be (lower, upper)"
                    )
                lower, upper = limits
                lower = None if lower is None else float(lower)
                upper = None if upper is None else float(upper)
                if ((lower is not None and not math.isfinite(lower))
                        or (upper is not None and not math.isfinite(upper))):
                    raise ValueError(
                        f"{type(sub).__name__}.state_bounds[{name!r}] must be finite"
                    )
                if lower is not None and upper is not None and lower > upper:
                    raise ValueError(
                        f"{type(sub).__name__}.state_bounds[{name!r}] has lower > upper"
                    )
                entry = entries[sub.state_names.index(name)]
                if entry[1]:
                    raise TypeError(
                        f"{type(sub).__name__}.state_bounds cannot constrain complex state {name!r}"
                    )
                bounds.append((entry[0], lower, upper))
        self._state_bounds = tuple(bounds)
        # stateful subsystems with complex flags, in vector order
        self._rhs_flags = tuple((sub, tuple(is_c for _idx, is_c in entries))
                                for sub, entries in rhs_layout if entries)
        self._rhs_plan = tuple((sub, flags, all(flags)) for sub, flags in self._rhs_flags)

    def _output_stages(self, sub: Subsystem, name: str):
        """Validate a subsystem's ``output_stages`` and return them, or None if absent."""
        stages = getattr(sub, "output_stages", None)
        if stages is None:
            return None
        if not isinstance(stages, (tuple, list)) or not stages:
            raise ValueError(f"{name}.output_stages must be a nonempty sequence of OutputMethod declarations")
        produced, methods = set(), set()
        for stage in stages:
            if not isinstance(stage, OutputMethod):
                raise TypeError(f"{name}.output_stages: expected OutputMethod, got {type(stage).__name__}")
            where = f"{name}.output_stages[{stage.method!r}]"
            if not isinstance(stage.method, str) or not callable(getattr(sub, stage.method, None)):
                raise ValueError(f"{where}: method must name a callable on the subsystem")
            if stage.method in methods:
                raise ValueError(f"{where}: duplicate stage method")
            methods.add(stage.method)
            for label, fields, record in (("inputs", stage.inputs, sub.inp), ("outputs", stage.outputs, sub.out)):
                if not isinstance(fields, (tuple, list)) or any(not isinstance(f, str) for f in fields):
                    raise ValueError(f"{where}.{label}: expected a sequence of field names")
                if len(set(fields)) != len(fields):
                    raise ValueError(f"{where}.{label}: duplicate field")
                for field in fields:
                    if not hasattr(record, field):
                        raise ValueError(f"{where}.{label}: unknown field {field!r}")
            if not stage.outputs:
                raise ValueError(f"{where}: at least one output is required")
            if produced.intersection(stage.outputs):
                raise ValueError(f"{where}: an output has more than one producer")
            produced.update(stage.outputs)
        outputs = {f for f in dir(sub.out) if not f.startswith("_") and not callable(getattr(sub.out, f))}
        missing = outputs - produced
        if missing:
            raise ValueError(f"{name}.output_stages: outputs without a producer: {sorted(missing)}")
        return stages

    def _build_evaluation_plan(self) -> None:
        """Sort output-method calls and individual input copies by signal dependency."""
        names = {id(s): n for n, s in zip(self.names, self.subsystems)}
        connected = {id(s): [] for s in self.subsystems}
        for sub, field in self.connections:
            connected[id(sub)].append(field)

        operations, owners, dependencies, labels, rhs_required = [], [], [], [], []

        def add(operation, owner, sources, label, rhs=True):
            index = len(operations)
            operations.append(operation)
            owners.append(((owner,), sources))
            dependencies.append(set())
            labels.append(label)
            rhs_required.append(rhs)
            return index

        producers, legacy, stages = {}, {}, []
        for name, sub in zip(self.names, self.subsystems):
            declared = self._output_stages(sub, name)
            if declared is None:
                node = None
                if sub.outputs_need_inputs or sub.out is not sub.state:
                    node = add(("out", sub.set_outputs), sub, (), f"{name}.set_outputs")
                    stages.append((node, sub, connected[id(sub)] if sub.outputs_need_inputs else ()))
                legacy[id(sub)] = node  # state aliases need no output evaluation
            else:
                for stage in declared:
                    node = add(("out", getattr(sub, stage.method)), sub, (),
                               f"{name}.{stage.method}", stage.rhs)
                    stages.append((node, sub, stage.inputs))
                    for field in stage.outputs:
                        producers[id(sub), field] = node

        def producer(sub, field):
            key = (id(sub), field)
            if key in producers:
                return producers[key]
            if id(sub) in legacy:
                return legacy[id(sub)]
            raise ValueError(f"{names[id(sub)]}.out.{field}: no output stage produces this signal")

        copies = {}
        for (dst, field), spec in self.connections.items():
            entries = spec if isinstance(spec, list) else [spec]
            sources = tuple(entry[0] for entry in entries)
            if len(entries) == 1 and (len(entries[0]) < 3 or entries[0][2] == 1.0):
                src, src_field = entries[0][:2]
                operation = (dst.inp, field, src.out, src_field, None)
            else:
                terms = tuple((entry[0].out, entry[1], float(entry[2]) if len(entry) > 2 else 1.0)
                              for entry in entries)
                operation = (dst.inp, field, None, None, terms)
            node = add(("copy", operation), dst, sources, f"{names[id(dst)]}.inp.{field}")
            copies[id(dst), field] = node
            for entry in entries:
                source = producer(entry[0], entry[1])
                if source is not None:
                    dependencies[node].add(source)
        for node, sub, fields in stages:
            for field in fields:
                copy = copies.get((id(sub), field))
                if copy is not None:
                    dependencies[node].add(copy)

        # Reuse the scheduling pass: DAGs need neither SCC analysis nor a condensed graph.
        successors = [[] for _ in operations]
        for node, deps in enumerate(dependencies):
            for dep in deps:
                successors[dep].append(node)
        order, residual = _topological_order(dependencies, successors)
        if residual:
            # Residual nodes can be downstream of a loop. Only cyclic SCCs become solver groups.
            components = [(node,) for node in order] + _strong_components(successors, residual)
            component_of = {node: ci for ci, component in enumerate(components) for node in component}
            compact_ops, compact_owners, compact_deps, compact_rhs = [], [], [], []
            for ci, component in enumerate(components):
                members = set(component)
                cyclic = len(component) > 1 or any(node in dependencies[node] for node in component)
                if not cyclic:
                    node = component[0]
                    operation = operations[node]
                    own_group, srcs = owners[node]
                else:
                    # Retain the existing connection tearing and algebraic solver unchanged.
                    pending = {node: set(dependencies[node]) & members for node in component}
                    internal_order: list[int] = []
                    tears: list[tuple] = []
                    while pending:
                        ready = sorted(node for node, deps in pending.items() if not deps)
                        if not ready:
                            candidates = sorted(node for node in pending if operations[node][0] == "copy")
                            if not candidates:
                                raise ValueError(
                                    "algebraic output cycle has no connected signal to solve: "
                                    + ", ".join(labels[node] for node in component)
                                )
                            torn = candidates[0]
                            pending[torn].clear()
                            tears.append(operations[torn][1])
                            ready = [torn]
                        for node in ready:
                            if node not in pending:
                                continue
                            internal_order.append(node)
                            pending.pop(node)
                            for deps in pending.values():
                                deps.discard(node)
                    own_group = tuple(dict.fromkeys(owner for node in component
                                                     for owner in owners[node][0]))
                    srcs = tuple(dict.fromkeys(source for node in component
                                               for source in owners[node][1]))
                    operation = ("algebraic", _AlgebraicLoop(
                        tuple(operations[node] for node in internal_order), tuple(tears),
                        own_group, tuple(labels[node] for node in component),
                    ))
                compact_ops.append(operation)
                compact_owners.append((own_group, srcs))
                compact_rhs.append(any(rhs_required[node] for node in component))
                compact_deps.append({component_of[dep] for node in component
                                     for dep in dependencies[node] if component_of[dep] != ci})

            operations, owners, dependencies, rhs_required = (
                compact_ops, compact_owners, compact_deps, compact_rhs)
            successors = [[] for _ in operations]
            for node, deps in enumerate(dependencies):
                for dep in deps:
                    successors[dep].append(node)
            order, residual = _topological_order(dependencies, successors)
            if residual:
                raise RuntimeError("internal error while collapsing algebraic output groups")
        # An observational producer is still part of the derivative plan when a required
        # downstream copy or output stage consumes it.  Propagate liveness towards producers in
        # reverse topological order; this retains continuous-control signal paths while pruning
        # unconnected measurements from sampled plant RHS calls.
        for node in reversed(order):
            if rhs_required[node]:
                for dependency in dependencies[node]:
                    rhs_required[dependency] = True
        position = {node: i for i, node in enumerate(order)}
        self._plan = tuple(operations[i] for i in order)
        self._plan_rhs = tuple(rhs_required[i] for i in order)
        self._plan_owners = tuple(owners[i] for i in order)
        self._plan_dependencies = tuple(
            tuple(position[d] for d in sorted(dependencies[i])) for i in order
        )
        self.algebraic_loops = tuple(
            operations[i][1] for i in order if operations[i][0] == "algebraic"
        )
        # Minimal output plan reached by a held-input change.  States, time and ordinary
        # parameters are unchanged in this path, so unrelated subsystem outputs need no refresh.
        zoh_owners = {id(sub) for sub, _field in self.zoh_connections}
        affected: set[int] = set()
        for i, (op_owners, _sources) in enumerate(self._plan_owners):
            if (any(id(owner) in zoh_owners for owner in op_owners)
                    or any(dependency in affected for dependency in self._plan_dependencies[i])):
                affected.add(i)
        self._zoh_plan = tuple(sorted(affected))
        # staged subsystems appear once per output call
        aliases = [s for s in self.subsystems if id(s) in legacy and legacy[id(s)] is None]
        output_order = list(aliases)
        for (kind, _), (op_owners, _) in zip(self._plan, self._plan_owners):
            if kind not in ("out", "algebraic"):
                continue
            for owner in op_owners:
                if owner not in output_order:
                    output_order.append(owner)
        self.output_order = output_order

    def _compile(self) -> None:
        """Generate and compile ``_load``, ``_outputs``, ``rhs_list`` (source in ``compiled_source``)."""
        owners = {id(s): n for n, s in zip(self.names, self.subsystems)}
        env: dict[str, Any] = {}
        refs: dict[int, str] = {}

        def ref(obj: Any) -> str:
            key = refs.get(id(obj))
            if key is None:
                key = refs[id(obj)] = f"_{len(refs)}"
                env[key] = obj
            return key

        def plain(field: str) -> bool:
            return field.isidentifier() and not keyword.iskeyword(field)

        def get(obj: Any, field: str) -> str:
            return f"{ref(obj)}.{field}" if plain(field) else f"getattr({ref(obj)}, {field!r})"

        def put(obj: Any, field: str, value: str) -> str:
            return f"{ref(obj)}.{field} = {value}" if plain(field) else f"setattr({ref(obj)}, {field!r}, {value})"

        load = [put(st, name, f"complex(v[{idx}], v[{idx + 1}])" if is_c else f"v[{idx}]")
                for st, name, idx, is_c in self._state_ops]

        plan, derivative_plan = [], []
        for (kind, item), (op_owners, _srcs), rhs_required in zip(
                self._plan, self._plan_owners, self._plan_rhs):
            if kind in ("out", "algebraic"):
                line = f"{ref(item)}(t)"
            else:
                dst, field, src, src_field, fanin = item
                if fanin is None:
                    value = get(src, src_field)
                else:
                    value = "0"
                    for out, name, gain in fanin:
                        value = f"({value} + {ref(gain)} * {get(out, name)})"
                line = put(dst, field, value)
            comment = ", ".join(owners[id(owner)] for owner in op_owners)
            line = f"{line}  # {comment}"
            plan.append(line)
            if rhs_required:
                derivative_plan.append(line)

        derive, packed = [], []
        for j, (sub, flags, _all_complex) in enumerate(self._rhs_plan):
            got = [f"d{j}_{q}" for q in range(len(flags))]
            derive.append(f"{', '.join(got)}, = {ref(sub)}.rhs(t)  # {owners[id(sub)]}")
            for d, is_c in zip(got, flags):
                packed += [f"{d}.real", f"{d}.imag"] if is_c else [d]

        def function(signature: str, body: list[str]) -> list[str]:
            return [f"def {signature}:"] + ["    " + line for line in body or ["pass"]]

        source = "\n".join(function("_load(v)", load) + function("_outputs(t)", plan)
                           + function("_zoh_outputs(t)", [plan[i] for i in self._zoh_plan])
                           + function("rhs_list(t, v)", load + derivative_plan + derive
                                      + [f"return [{', '.join(packed)}]"])) + "\n"
        filename = f"<peslite model {id(self):#x}>"
        exec(compile(source, filename, "exec"), env)
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        self.compiled_source = source
        self._load, self._outputs, self._zoh_outputs, self.rhs_list = (
            env["_load"], env["_outputs"], env["_zoh_outputs"], env["rhs_list"])

    # -------------------------------------------------------------- groups
    def groups(self, assignment: Mapping[str, str], system: str = "system") -> dict[str, GroupPlan]:
        """Split subsystems into groups and return ``{label: GroupPlan}``.

        ``assignment`` maps subsystem names or namespaces (``"vsc1"`` for ``vsc1.*``) to
        group labels; unassigned subsystems go to ``system``.
        """
        resolved: dict[str, str] = {}
        for key, label in assignment.items():
            members = self.select(key)
            if not members:
                raise KeyError(f"unknown subsystem(s) or namespace(s) [{key!r}]; this model has {self.names}")
            for name in members:
                if resolved.get(name, label) != label:
                    raise KeyError(f"subsystem {name!r} is assigned to both {resolved[name]!r} and {label!r}")
                resolved[name] = label
        labels: dict[str, list] = {}
        for name, sub in zip(self.names, self.subsystems):
            labels.setdefault(resolved.get(name, system), []).append(sub)
        labels.setdefault(system, [])
        return {label: self._group_plan(members) for label, members in labels.items()}

    def _group_plan(self, members: Sequence[Any]) -> GroupPlan:
        member_ids = {id(m) for m in members}
        mask = np.zeros(self.n_states, dtype=bool)
        for st, name, idx, is_c in self._state_ops:
            if any(st is m.state for m in members):
                mask[idx:idx + (2 if is_c else 1)] = True

        def closure(seeds):
            needed = set(seeds)
            pending = list(needed)
            while pending:
                for dep in self._plan_dependencies[pending.pop()]:
                    if dep not in needed:
                        needed.add(dep)
                        pending.append(dep)
            return tuple(self._plan[i] for i in sorted(needed))

        member_ops = [i for i, (owners, _) in enumerate(self._plan_owners)
                      if any(id(owner) in member_ids for owner in owners)]
        plan = closure(member_ops)
        own = tuple(item for item, (owners, srcs) in zip(self._plan, self._plan_owners)
                    if any(id(owner) in member_ids for owner in owners)
                    and (item[0] in ("out", "algebraic")
                         or all(id(x) in member_ids for x in srcs)))
        boundary = [i for i, ((kind, _), (owners, srcs))
                    in enumerate(zip(self._plan, self._plan_owners))
                    if kind == "copy" and any(id(owner) in member_ids for owner in owners)
                    and any(id(x) not in member_ids for x in srcs)]
        inputs = tuple(self._plan[i] for i in boundary)
        # only the stages producing boundary signals
        feed = closure(boundary)
        rhs = tuple((sub, entries) for sub, entries in self._rhs_layout if entries and id(sub) in member_ids)
        return GroupPlan(tuple(members), mask, plan, own, feed, inputs, rhs)

    def run_plan(self, t: float, plan: tuple) -> None:
        """Run a plan (subset of the output plan) on the current subsystem records."""
        for kind, item in plan:
            if kind in ("out", "algebraic"):
                item(t)
            else:
                dst_inp, dst_name, src_out, src_name, fanin = item
                if fanin is None:
                    setattr(dst_inp, dst_name, getattr(src_out, src_name))
                else:
                    total = 0
                    for out, name, gain in fanin:
                        total += gain * getattr(out, name)
                    setattr(dst_inp, dst_name, total)

    def rhs_group(self, t: float, y: NDArray[np.float64], group: GroupPlan, own: bool = False) -> NDArray[np.float64]:
        """Return the derivatives of one group's states (zeros elsewhere) for the full state ``y``.

        ``own=True`` keeps inputs from outside the group unchanged.
        """
        self.set_states(y)
        self.run_plan(t, group.own if own else group.plan)
        dy = np.zeros(self.n_states)
        for sub, entries in group.rhs:
            for d, (idx, is_c) in zip(sub.rhs(t), entries):
                if is_c:
                    dy[idx] = d.real
                    dy[idx + 1] = d.imag
                else:
                    dy[idx] = d
        return dy

    # ------------------------------------------------------------- packing
    def get_initial_values(self) -> NDArray[np.float64]:
        y = np.empty(self.n_states)
        for st, name, idx, is_c in self._state_ops:
            v = getattr(st, name)
            if is_c:
                y[idx] = v.real
                y[idx + 1] = v.imag
            else:
                y[idx] = v
        return y

    def set_states(self, y: NDArray[np.float64]) -> None:
        self._load(y.tolist())

    @property
    def has_state_bounds(self) -> bool:
        """Whether any subsystem declares a real-state constraint."""
        return bool(self._state_bounds)

    def project_states(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project declared real-state bounds in place and return ``y``."""
        for index, lower, upper in self._state_bounds:
            if lower is not None and y[index] < lower:
                y[index] = lower
            elif upper is not None and y[index] > upper:
                y[index] = upper
        return y

    def state_labels(self) -> list[str]:
        """One label per entry of the flat vector (``<subsystem>.<state>[.re|.im]``)."""
        labels = []
        for sname, (sub, entries) in zip(self.names, self._rhs_layout):
            for name, (_idx, is_c) in zip(sub.state_names, entries):
                base = f"{sname}.{name}"
                labels += [base + ".re", base + ".im"] if is_c else [base]
        return labels

    def select(self, prefix: str) -> list[str]:
        """Return the subsystem named ``prefix`` or all subsystems named ``prefix.*``."""
        head = prefix + "."
        return [n for n in self.names if n == prefix or n.startswith(head)]

    def name_of(self, sub: Subsystem) -> str:
        for name, s in zip(self.names, self.subsystems):
            if s is sub:
                return name
        raise KeyError(f"{type(sub).__name__} is not in this model")

    def get_state(self) -> dict[str, complex | float]:
        """Return all continuous states as ``{"<subsystem>.<state>": value}``.

        Reads the subsystem records; call :meth:`sync` or :meth:`set_states` first.
        """
        out: dict[str, complex | float] = {}
        for sname, sub in zip(self.names, self.subsystems):
            st = sub.state
            for name in sub.state_names:
                out[f"{sname}.{name}"] = getattr(st, name)
        return out

    def set_state(self, values: Mapping[str, Any]) -> None:
        """Write named states into the subsystem records (repack with :meth:`get_initial_values`)."""
        index = {f"{sname}.{name}": (sub.state, name)
                 for sname, sub in zip(self.names, self.subsystems) for name in sub.state_names}
        for key, value in values.items():
            if key not in index:
                raise KeyError(f"unknown model state {key!r}; known: {', '.join(index)}")
            st, name = index[key]
            setattr(st, name, complex(value) if isinstance(getattr(st, name), complex) else float(value))

    def state_slice(self, sub: Subsystem, name: str) -> slice:
        """Position of a subsystem's state field in the flat vector."""
        for st, nm, idx, is_c in self._state_ops:
            if st is sub.state and nm == name:
                return slice(idx, idx + (2 if is_c else 1))
        raise KeyError(f"{type(sub).__name__} has no state {name!r}")

    # --------------------------------------------------------- evaluation
    def set_zoh_input(self, label: str, value: Any) -> None:
        """Set a held external input (e.g. the switching state) by label."""
        for inp, name in self._zoh_targets[label]:
            setattr(inp, name, value)

    def evaluate_outputs(self, t: float) -> None:
        """Run the ordered output-method / input-copy plan for the current states."""
        self._outputs(t)

    def rhs(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``dy/dt`` at time ``t`` (s) for the flat real state vector ``y``."""
        return np.array(self.rhs_list(t, y.tolist()))

    def sync_zoh(self, t: float) -> None:
        """Refresh only outputs reached by held inputs after a full sync at the same instant."""
        self._zoh_outputs(t)

    def sync(self, t: float, y: NDArray[np.float64]) -> None:
        """Load ``y`` into the subsystems and refresh all outputs (for sampling/logging)."""
        self._load(y.tolist())
        self._outputs(t)

    # ---------------------------------------------------------------- energy
    def energy_balance(self, t: float, y: NDArray[np.float64]) -> "_energy.EnergyReport":
        """Energy accounting at ``(t, y)`` from the subsystems' declarations (see :mod:`peslite.solver.energy`)."""
        return _energy.balance(self, t, y)

    def ph_report(self, zoh: Mapping[str, Any] | None = None, groups: Mapping[str, Any] | None = None,
                  rtol: float = 1e-8, hold: Mapping[str, float] | None = None) -> "_energy.PHReport":
        """Return the port-Hamiltonian report of the model (:class:`peslite.solver.energy.PHReport`).

        ``groups``: result of :meth:`groups`; ``hold``: ``{"step": dt, "window": W}`` hold times in s.
        """
        return _energy.ph_report(self, zoh=dict(zoh) if zoh else None, rtol=rtol,
                                 groups=dict(groups) if groups else None, hold=dict(hold) if hold else None)
