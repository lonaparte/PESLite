"""The electrical network of a netlist circuit model: one Circuit subsystem and a view per component.

Under ``simulation.solver.circuit_model`` ``statewise`` (dense per-mode state space) or ``nodal``
(sparse factorized derivative MNA), :func:`lower` rewrites the connection graph the System
assembled. Components declaring ``network(nb)`` become primitives of one
:class:`~peslite.solver.netlist.Netlist` between bus nodes; ``network(nb)`` returns
``NotImplemented`` when a component's present configuration has no netlist form. Every other
subsystem attached to such a bus is adapted through the signal paths declared by its physical
port: an output current becomes a current source, or an output voltage becomes a voltage source,
while the conjugate input reads the netlist solution.

The stateless :class:`Circuit` solves the netlist once per model evaluation. Each component is
replaced in the Model by a :class:`NetworkView` with the same name and the same ``state``, ``input``
and ``output`` records. The view's outputs come from the component's
own output equations and its derivatives from the netlist. Connections between components are
dropped; connections from other producers into component inputs are retargeted to Circuit inputs,
copied into the components' ``input`` before solving. Zero-order-hold inputs keep writing into the
components' ``input``; the Circuit also receives each such label, so a held change refreshes it.

Mode changes (breakers, gates, transformer ratios, diodes, set events) are boundaries of the
commutation protocol of :class:`~peslite.assembly.conduction.DiodeConduction`: the Circuit's margin is
negative while the states are inconsistent with the present mode, and ``commutate`` projects them
(charge and flux conservation) and records the impulsive loss and supply for the energy audit. A component
may keep a commutation of its own beside it (a gate clock) and mirror the netlist's diode conduction into
its discrete states (:meth:`NetlistBuilder.commutation`); its states may be stored in other coordinates than
the netlist's (``NetlistBuilder.state(..., transform=T)``).

Under ``nodal`` the sparse solve eliminates the netlist's automatic switch-cluster groups onto the buses
between them (:meth:`~peslite.solver.netlist.Netlist.grouping`); :meth:`Circuit.grouping_report`
describes the groups by subsystem.
"""

from __future__ import annotations

import copy
import itertools
import math
import re
from contextvars import ContextVar
from types import MethodType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ..components.network import Port
from ..solver.energy import Energetic, spec_of
from ..solver.model import (
    Bag, ConfigError, Empty, OutputMethod, _InlineProgram, _compile_straight_line,
)
from ..solver.netlist import (QUANTITIES, CoupledInductors, CurrentSource, Diode, Element, Mode,
                              Netlist, Node, Topology, VoltageSource, _DENSE_GROUP, _Degenerate,
                              _Evaluation, _linear_expression, _numeric_values, _roles, _UnionFind)

__all__ = ["NetlistBuilder", "NetworkPlan", "Circuit", "NetworkView", "circuit_structure",
           "circuit_mode_key", "circuit_linear_blocks", "circuit_linear_modes",
           "circuit_linear_certificates",
           "lower", "bind_device", "compile_network"]


_NETWORK_DECLARATION: ContextVar[bool] = ContextVar(
    "peslite_network_declaration", default=False
)


def _kind(port: Port) -> tuple[int, float]:
    """Node dimension and power scale of a port: alpha-beta (complex) or scalar."""
    return (2 if isinstance(port.voltage, complex) else 1), port.scale


def _ports(sub: Any) -> list[Port]:
    return [port for port in getattr(sub, "ports", ())
            if isinstance(port, Port) and port.physical and port.component is sub]


def _fields(record: Any) -> tuple[str, ...]:
    return tuple(f for f in dir(record) if not f.startswith("_") and not callable(getattr(record, f)))


def _state_dimension(component: Any, name: str) -> int:
    declared = getattr(component, "state_dimensions", {})
    return int(declared.get(name, 2 if isinstance(getattr(component.state, name), complex) else 1))


_IDENTIFIERS = re.compile(r"\b[A-Za-z_]\w*\b")
_ASSIGNMENT = re.compile(r"\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.*)")


def _live_statements(statements: Sequence[str], expressions: Sequence[str]) -> list[str]:
    """Return generated statements needed by the requested expressions."""
    used = {name for expression in expressions for name in _IDENTIFIERS.findall(expression)}
    kept = []
    for statement in reversed(statements):
        match = _ASSIGNMENT.fullmatch(statement)
        if match is None:
            kept.append(statement)
            used.update(_IDENTIFIERS.findall(statement))
            continue
        name, value = match.groups()
        if name not in used:
            continue
        used.discard(name)
        used.update(_IDENTIFIERS.findall(value))
        kept.append(statement)
    return list(reversed(kept))


class _BoundValue:
    """A netlist value read directly from a component record."""

    __slots__ = ("record", "field", "factor", "invert")

    def __init__(self, record: Any, field: str, factor: float = 1.0, invert: bool = False) -> None:
        if invert and factor != 1.0:
            raise ValueError("an inverted bound value cannot also have a numeric factor")
        self.record, self.field, self.factor, self.invert = record, field, factor, invert

    def __call__(self, *_args: Any) -> Any:
        value = getattr(self.record, self.field)
        return not value if self.invert else self.factor * value


class _CallableValue:
    """A netlist callback with explicit component-input dependencies."""

    __slots__ = ("callback", "inputs", "record", "state_dependencies")

    def __init__(self, callback: Callable, inputs: Sequence[str], record: Any = None,
                 state_dependencies: Sequence[tuple[Any, str]] = ()) -> None:
        self.callback, self.inputs, self.record = callback, tuple(inputs), record
        self.state_dependencies = tuple(state_dependencies)

    def __call__(self, *args: Any) -> Any:
        return self.callback(*args)


def _record_dependencies(value: Any) -> Optional[tuple[Any, tuple[str, ...]]]:
    """Record fields read by a declared netlist value, if explicitly known."""
    if isinstance(value, _BoundValue):
        return value.record, (value.field,)
    if isinstance(value, _CallableValue):
        return value.record, value.inputs
    return None


class NetlistBuilder:
    """What a component's ``network(nb)`` declares its primitives with.

    ``node(port)``: the bus node of a port (ground for an unattached voltage-following
    port, a floating node for an unattached driving one); ``internal(dim, scale)``: a new node;
    ``value(record, field, factor=1, invert=False)``: a primitive callback bound directly to a field;
    ``add(*primitives)``; ``state(component, names, transform=None)``: the indices of one state or of
    several (concatenated); with an invertible ``transform`` ``T`` the netlist sees ``T x`` there (for
    example coil currents of stored modal currents) and the component's derivatives are ``T^-1`` of the
    netlist's; ``output(component, field, quantity)``: publish ``quantity`` = ``(kind, target[, port])``,
    ``kind`` one of :data:`~peslite.solver.netlist.QUANTITIES`, as ``component.output.<field>`` in place of
    the component's own output equation for it; ``losses(component)`` and ``supplies(component)``
    derive its dissipated and supplied power from its primitives.

    ``commutation(component, margin=, commutate=, next_time=, after=, diodes=)``: the component keeps a
    discrete state of its own (a gate clock): its view exposes ``margin``, ``commutate`` and ``next_time``
    as ``commutation_margin``, ``commutate`` and ``next_commutation_time`` beside the Circuit's projection;
    ``after(t, before)`` is called whenever the Circuit commutated (``before``: the component's states
    before the jump, by name) and at least once after the conduction of ``diodes`` changed, so that the
    component mirrors the netlist's conduction into its own discrete states.
    """

    def __init__(self, netlist: Netlist, nodes: Optional[dict[int, Optional[Node]]] = None,
                 states: Optional[dict[tuple[int, str], np.ndarray]] = None) -> None:
        self.netlist = netlist
        self.probe = nodes is None  # a declaration: every port gets its own placeholder node
        self._nodes: dict[int, Optional[Node]] = {} if nodes is None else nodes
        self._ports: dict[int, Port] = {}
        self._states = {} if states is None else states
        self._state_owners: dict[tuple[int, str], tuple[Any, str]] = {}
        self.owner: Any = None
        self.owners: dict[int, Any] = {}
        self.owned: dict[int, list[Element]] = {}
        self.outputs: list[tuple[Any, str, tuple[tuple[float, tuple], ...]]] = []
        self.port_outputs: list[tuple[Any, str, Port, str, float]] = []
        self.lossy: set[int] = set()
        self.supplying: set[int] = set()
        self.transforms: list[tuple[np.ndarray, np.ndarray]] = []
        self.hooks: dict[int, dict[str, Any]] = {}
        self.bindings: list[tuple[Callable, tuple[Any, ...]]] = []
        self.mixtures: list[tuple[Callable, Callable]] = []
        self.value_outputs: dict[tuple[int, str], tuple[Any, str]] = {}
        self.value_bindings: dict[tuple[int, str], _BoundValue] = {}

    def node(self, port: Port) -> Optional[Node]:
        key = id(port)
        self._ports[key] = port
        if port.grounded:
            self._nodes[key] = None
        elif key not in self._nodes:
            self._nodes[key] = Node(*_kind(port), name=port.name)
        return self._nodes[key]

    @staticmethod
    def internal(dim: int = 1, scale: float = 1.0) -> Node:
        return Node(dim, scale)

    def value(self, record: Any, field: str, factor: float = 1.0, *,
              invert: bool = False) -> Callable:
        """A callback that reads ``record.field`` when a primitive evaluates it."""
        value = _BoundValue(record, field, factor, invert)
        if self.owner is not None and record is getattr(self.owner, "output", None):
            self.value_outputs[id(record), field] = self.owner, field
        return value

    def function(self, callback: Callable, inputs: Sequence[str] = (),
                 states: Sequence[str] = ()) -> Callable:
        """A callback with named owner input and state dependencies."""
        if not callable(callback):
            raise TypeError("a netlist function value must be callable")
        record = None
        if self.owner is not None:
            known = set(_fields(self.owner.input))
            unknown = set(inputs) - known
            if unknown:
                raise ValueError(f"unknown netlist function input fields {sorted(unknown)}")
            record = self.owner.input
            unknown_states = set(states) - set(self.owner.state_names)
            if unknown_states:
                raise ValueError(f"unknown netlist function states {sorted(unknown_states)}")
        elif states:
            raise ValueError("a netlist function cannot name states without an owner")
        return _CallableValue(
            callback, inputs, record,
            tuple((self.owner, state) for state in states),
        )

    def add(self, *elements: Element) -> Element:
        first = self.netlist.add(*elements)
        self.owners[id(self.owner)] = self.owner
        self.owned.setdefault(id(self.owner), []).extend(elements)
        return first

    def state(self, component: Any, names: str | Sequence[str], transform: Any = None) -> np.ndarray:
        index = []
        for name in [names] if isinstance(names, str) else names:
            key = (id(component), name)
            self._state_owners[key] = component, name
            if key not in self._states:
                if not self.probe:
                    raise ConfigError(f"{type(component).__name__}.{name} is not a state of a netlist component")
                n = _state_dimension(component, name)
                used = sum(v.size for v in self._states.values())
                self._states[key] = np.arange(used, used + n)
            index.append(self._states[key])
        index = np.concatenate(index) if index else np.zeros(0, dtype=np.intp)
        if transform is not None:
            T = _numeric_values(transform)
            if T.shape != (index.size, index.size) or not np.all(np.isfinite(T)) or (
                    index.size and np.linalg.cond(T) > 1e12):
                raise ConfigError(f"{type(component).__name__}: a state transform must be an invertible "
                                  f"{index.size} x {index.size} matrix, got shape {T.shape}")
            self.transforms.append((index, transform))
        return index

    def commutation(self, component: Any, *, margin: Any = None, commutate: Any = None, next_time: Any = None,
                    after: Any = None, diodes: Sequence[Element] = ()) -> None:
        if (margin is None) != (commutate is None):
            raise ConfigError(f"{type(component).__name__}: a netlist commutation needs both margin and commutate")
        self.hooks[id(component)] = {"margin": margin, "commutate": commutate, "next_time": next_time,
                                     "after": after, "diodes": tuple(diodes)}

    def output(self, component: Any, field: str, quantity: tuple) -> None:
        if quantity[0] not in QUANTITIES:
            raise ValueError(f"unknown netlist quantity {quantity[0]!r}; expected one of {QUANTITIES}")
        self.outputs.append((component, field, ((1.0, tuple(quantity)),)))

    def linear_output(self, component: Any, field: str,
                      terms: Sequence[tuple[float, tuple]]) -> None:
        """Publish a linear combination of primitive quantities as one component output."""
        normalized = []
        for factor, quantity in terms:
            if quantity[0] not in QUANTITIES:
                raise ValueError(
                    f"unknown netlist quantity {quantity[0]!r}; expected one of {QUANTITIES}"
                )
            normalized.append((float(factor), tuple(quantity)))
        self.outputs.append((component, field, tuple(normalized)))

    def node_current_output(self, component: Any, field: str, node: Optional[Node],
                            *, exclude: Sequence[Element] = ()) -> None:
        """Publish current injected into ``node`` by the component's declared primitives."""
        omitted = {id(element) for element in exclude}
        terms = []
        for element in self.owned.get(id(component), ()):
            if id(element) in omitted:
                continue
            for port, (start, end) in enumerate(element.ports):
                if start is node:
                    terms.append((-1.0, ("branch_current", element, port)))
                if end is node:
                    terms.append((1.0, ("branch_current", element, port)))
        self.linear_output(component, field, terms)

    def port_output(self, component: Any, field: str, port: Port,
                    quantity: str, factor: float = 1.0) -> None:
        """Publish one acausal port voltage or entering current as a component output."""
        if quantity not in ("voltage", "current"):
            raise ValueError("a port output quantity must be 'voltage' or 'current'")
        if port.component is not component:
            raise ValueError("a published port must belong to its component")
        self.port_outputs.append((component, field, port, quantity, float(factor)))

    def losses(self, component: Any) -> None:
        self.lossy.add(id(component))

    def supplies(self, component: Any) -> None:
        self.supplying.add(id(component))

    def bind(self, callback: Callable, *objects: Any) -> None:
        """Run ``callback(netlist, *objects)`` after placeholders become actual primitives."""
        self.bindings.append((callback, tuple(objects)))

    def mixture(self, cases: Callable, select: Callable) -> None:
        """Average evaluations of this declaration over ``(weight, case)`` values."""
        self.mixtures.append((cases, select))

class NetworkPlan:
    """A component's assembly-time electrical plan, materialized by every circuit model."""

    __slots__ = ("builder", "port_nodes", "internal_nodes", "states", "elements")

    def __init__(self, builder: NetlistBuilder) -> None:
        self.builder = builder
        self.port_nodes = tuple(
            (node, builder._ports[key]) for key, node in builder._nodes.items()
        )
        port_node_ids = {id(node) for node, _port in self.port_nodes}
        self.internal_nodes = tuple(dict.fromkeys(
            node
            for element in builder.netlist.elements
            for branch in element.ports
            for node in branch
            if node is not None and id(node) not in port_node_ids
        ))
        self.states = tuple(
            (*builder._state_owners[key], indices)
            for key, indices in builder._states.items()
        )
        owner_of = {
            id(element): builder.owners[key]
            for key, owned in builder.owned.items()
            for element in owned
        }
        self.elements = tuple(
            (
                element,
                owner_of[id(element)],
                tuple(
                    (attribute, value)
                    for attribute, value in vars(element).items()
                    if isinstance(value, _BoundValue)
                ),
            )
            for element in builder.netlist.elements
        )

    def materialize(self, target: NetlistBuilder) -> None:
        source = self.builder
        node_map: dict[int, Optional[Node]] = {id(None): None}
        for node, port in self.port_nodes:
            node_map[id(node)] = target.node(port)
        for node in self.internal_nodes:
            node_map[id(node)] = Node(node.dim, node.scale, node.name)

        state_map: dict[int, int] = {}
        for owner, name, indices in self.states:
            actual = target.state(owner, name)
            state_map.update((int(old), int(new)) for old, new in zip(indices, actual))
        for indices, transform in source.transforms:
            target.transforms.append((
                np.asarray([state_map[int(index)] for index in indices], dtype=np.intp),
                transform,
            ))

        elements: dict[int, Element] = {}
        for element, owner, bound_values in self.elements:
            cloned = copy.copy(element)
            cloned.ports = tuple(tuple(node_map[id(node)] for node in port) for port in element.ports)
            for attribute, value in bound_values:
                replacement = target.value_bindings.get((id(value.record), value.field))
                if replacement is not None:
                    setattr(cloned, attribute, _BoundValue(
                        replacement.record, replacement.field, value.factor, value.invert
                    ))
            if hasattr(element, "state"):
                cloned.state = np.asarray(
                    [state_map[int(index)] for index in element.state], dtype=np.intp
                )
            elements[id(element)] = cloned
            target.owner = owner
            target.add(cloned)

        def mapped(value: Any) -> Any:
            if isinstance(value, Element):
                return elements[id(value)]
            if isinstance(value, Node):
                return node_map[id(value)]
            if isinstance(value, tuple):
                return tuple(mapped(item) for item in value)
            if isinstance(value, list):
                return [mapped(item) for item in value]
            return value

        target.outputs.extend(
            (component, field,
             tuple((factor, (kind, mapped(value), *rest))
                   for factor, (kind, value, *rest) in terms))
            for component, field, terms in source.outputs
        )
        target.port_outputs.extend(source.port_outputs)
        target.lossy.update(source.lossy)
        target.supplying.update(source.supplying)
        target.mixtures.extend(source.mixtures)
        target.value_outputs.update(source.value_outputs)
        for key, hook in source.hooks.items():
            target.hooks[key] = {
                **hook,
                "diodes": tuple(mapped(item) for item in hook["diodes"]),
            }
        for callback, objects in source.bindings:
            callback(target.netlist, *(mapped(item) for item in objects))


def compile_network(sub: Any) -> Optional[NetworkPlan]:
    """Translate ``sub.network`` once into its assembly plan, or return ``None``."""
    declare = getattr(sub, "network", None)
    if not callable(declare):
        return None
    probe = NetlistBuilder(Netlist("device"))
    probe.owner = sub
    token = _NETWORK_DECLARATION.set(True)
    try:
        if declare(probe) is NotImplemented:
            return None
    finally:
        _NETWORK_DECLARATION.reset(token)
    sub._network_has_losses = id(sub) in probe.lossy
    sub._network_has_supplies = id(sub) in probe.supplying
    return NetworkPlan(probe)


class _Group:
    """One netlist node: the ports a bus (or a port-to-port connection) joins."""

    __slots__ = ("label", "ports", "dim", "scale")

    def __init__(self, label: str, ports: list[Port]) -> None:
        self.label, self.ports = label, ports
        if len({_kind(t)[0] for t in ports}) != 1:
            raise ConfigError(f"bus {label!r}: its ports differ in phase representation")
        self.dim, self.scale = _kind(ports[0])


class Circuit:
    """Stateless subsystem solving the netlist once per model evaluation.

    ``solve`` copies retargeted inputs into the components, gathers their states, evaluates the
    netlist and publishes port voltages and currents; the views return its ``derivatives``.
    ``build()`` recompiles the netlist after a parameter change and keeps the record layout.
    """

    state_names: tuple[str, ...] = ()
    outputs_need_inputs = True
    observer = True  # a calculation: the components keep their energy ports
    grouped = True  # nodal: eliminate the automatic switch-cluster groups onto their interface (Netlist.grouping)

    def __init__(self, form: str) -> None:
        self.form = form
        self.names: dict[int, str] = {}  # id(subsystem) -> its name in the model (components and ports)
        self._owners: dict[int, str] = {}  # id(element) -> the subsystem that declared it
        self.state = Empty()
        self.components: list[Any] = []
        self.declarations: dict[int, NetworkPlan] = {}
        self.groups: list[_Group] = []
        self.port_values: list[tuple[Port, Optional[_Group], str, str]] = []
        self.standalone_ports: tuple[Port, ...] = ()
        self.behavioral_sources: list[tuple[Port, _Group, str, str]] = []
        self.copies: list[tuple[Any, str, str]] = []  # (component, input field, input key)
        self.published: dict[tuple[int, str], str] = {}  # (id(component), out field) -> out key
        self.inputs: dict[str, Any] = {}  # input key -> initial value
        self.outputs: dict[str, Any] = {}  # out key -> initial value
        self._output_producers: dict[str, str] = {}
        self._output_stage_owners: dict[str, tuple[Any, ...]] = {}
        self._output_stage_fields: dict[Callable, tuple[str, ...]] = {}
        self.passthrough: list[tuple[str, str]] = []  # input key -> output key
        self.value_inputs: dict[tuple[int, str], str] = {}  # bound component outputs -> input key
        self.netlist: Optional[Netlist] = None
        self.derivatives: list[float] = []
        self.t = 0.0
        self.impulsive_dissipated_energy = 0.0  # J, dissipated by state jumps at mode changes
        self.impulsive_supplied_energy = 0.0  # J, delivered by sources during those jumps
        self.locate_commutations = False  # only diodes commutate between boundaries
        self.commutation_active = False
        self._consistent: Optional[tuple] = None  # the mode the states were last made consistent with
        self._consistent_constraints: Optional[tuple] = None
        self._loss_elements: dict[int, list[Element]] = {}
        self._supply_elements: dict[int, list[Element]] = {}
        self._mixtures: tuple[tuple[Callable, Callable], ...] = ()
        self._mixture_modes: dict[tuple, tuple[tuple[bool, ...], tuple[float, ...]]] = {}
        self._mixture_loss: dict[int, float] = {}
        self._mixture_supply: dict[int, float] = {}
        self._energy_uses_time = False
        self._names: set[str] = set()
        self._transforms: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...] = ()  # (indices, T, T^-1)
        self._hooks: tuple[tuple[Any, Any], ...] = ()  # (component, after) mirroring the netlist's conduction
        self._watched: tuple[int, ...] = ()  # diodes whose conduction the hooks mirror
        self._mirrored_valid = False  # whether watched diode modes were reported to their hooks
        self._boundary_signature: Optional[tuple] = None
        self._stored: Optional[tuple[list[float], np.ndarray]] = None  # (component states, netlist vector) last stored
        self._margin_evaluator_key: Optional[tuple] = None
        self._margin_evaluator: Optional[Callable[[Sequence[float]], float]] = None
        self._views: tuple["NetworkView", ...] = ()
        self._mode_targets: set[tuple[int, str]] = set()
        self._discrete_mode_targets: set[tuple[int, str]] = set()
        self._held_input_keys: set[str] = set()
        self.requires_topology_freeze = False
        self._direct_dynamic = False
        self._direct_mode_key: Optional[tuple] = None
        self._numeric_stages = False
        self._stage_signature: Optional[tuple] = None
        self._standalone_runtime = False
        self._state_diode_groups: tuple[
            tuple[int, tuple[tuple[int, float], ...]], ...
        ] = ()
        self._device_bindings: dict[tuple, tuple[Topology, Callable, Callable]] = {}
        self._device_runtime: Optional[list] = None  # [netlist, topology, evaluator] once compiled
        self._device_commands: Optional[tuple] = None  # boundary commands the runtime was bound for
        self._structure_plan: Optional["_CircuitPlan"] = None
        self._mode_decompositions: dict[tuple, Optional["_ModeDecomposition"]] = {}
        self.plan_changed = False

    # ---------------------------------------------------------------- assembly
    def key(self, label: str, initial: Any, output: bool) -> str:
        """A new record field for ``label`` (sanitized, unique)."""
        base = re.sub(r"\W", "_", label)
        base = base if base.isidentifier() else "k_" + base
        name, k = base, 1
        while name in self._names:
            k += 1
            name = f"{base}_{k}"
        self._names.add(name)
        (self.outputs if output else self.inputs)[name] = initial
        return name

    def finalize(self) -> None:
        """Create the records, compile the state transfers and the netlist."""
        self.input = type("_CircuitInput", (Bag,), {"__slots__": tuple(self.inputs)})(**self.inputs)
        self.output = type("_CircuitOutput", (Bag,), {"__slots__": tuple(self.outputs)})(**self.outputs)
        layout, index = [], 0
        self._state_index: dict[tuple[int, str], np.ndarray] = {}
        for comp in self.components:
            for name in comp.state_names:
                is_complex = _state_dimension(comp, name) == 2
                self._state_index[id(comp), name] = np.arange(index, index + (2 if is_complex else 1))
                layout.append((comp, name, is_complex, index))
                index += 2 if is_complex else 1
        self.layout, self.n_states = tuple(layout), index
        env: dict[str, Any] = {"input": self.input}
        refs: dict[int, str] = {}

        def ref(obj: Any) -> str:
            if id(obj) not in refs:
                refs[id(obj)] = f"_{len(refs)}"
                env[refs[id(obj)]] = obj
            return refs[id(obj)]

        values = [f"{ref(c.state)}.{n}.real, {ref(c.state)}.{n}.imag" if z else f"{ref(c.state)}.{n}"
                  for c, n, z, _i in layout]
        writes = [f"{ref(c.state)}.{n} = v[{i}] + 1j * v[{i + 1}]" if z else f"{ref(c.state)}.{n} = v[{i}]"
                  for c, n, z, i in layout]
        copies = [f"setattr({ref(c.input)}, {f!r}, input.{k})" for c, f, k in self.copies]
        source = "\n".join(["def gather():", f"    return [{', '.join(values)}]",
                            "def scatter(v):", *("    " + w for w in writes or ["pass"]),
                            "def receive():", *("    " + c for c in copies or ["pass"])]) + "\n"
        exec(compile(source, "<peslite circuit>", "exec"), env)
        self.gather, self.scatter, self.receive = env["gather"], env["scatter"], env["receive"]
        self.derivatives = [0.0] * self.n_states
        self.build()

    def build(self) -> None:
        """Compile the netlist from the components' present parameters (also after a ``set`` event)."""
        old = self.netlist
        rebuilding = old is not None
        if self.netlist is not None:
            refreshed = {id(component): compile_network(component) for component in self.components}
            if any(declaration is None for declaration in refreshed.values()):
                changed = [type(component).__name__ for component in self.components
                           if refreshed[id(component)] is None]
                raise ConfigError(f"netlist form changed after assembly for {changed}")
            self.declarations = refreshed
        net = Netlist(self.form, grouped=self.grouped)
        nodes: dict[int, Optional[Node]] = {}
        node_of: dict[int, Node] = {}
        for group in self.groups:
            node_of[id(group)] = node = Node(group.dim, group.scale, group.label)
            for port in group.ports:
                nodes[id(port)] = node
        for port, group, _voltage_key, _current_key in self.port_values:
            if group is None:
                nodes[id(port)] = None
        nb = NetlistBuilder(net, nodes, self._state_index)
        nb.value_bindings = {
            token: _BoundValue(self.input, key) for token, key in self.value_inputs.items()
        }
        for comp in self.components:
            self.declarations[id(comp)].materialize(nb)
        for port, group, key, kind in self.behavioral_sources:
            nb.owner = port.component
            source = _BoundValue(self.input, key)
            primitive = (
                CurrentSource(node_of[id(group)], None, source)
                if kind == "current" else
                VoltageSource(node_of[id(group)], None, source)
            )
            nb.add(primitive)
        for port in self.standalone_ports:
            if port.standalone == "voltage":
                net.add(VoltageSource(
                    nodes[id(port)], None, lambda _t, port=port: port.voltage
                ))
            elif port.standalone == "current":
                net.add(CurrentSource(
                    None, nodes[id(port)], lambda _t, port=port: port.current
                ))
        declared = set(net.states.tolist())
        missing = sorted(f"{type(c).__name__}.{n}" for c, n, _z, i in self.layout if i not in declared)
        if missing:
            raise ConfigError(f"netlist components declare no primitive for their states {missing}")
        self.netlist = net
        self._margin_evaluator_key = None
        self._margin_evaluator = None
        if old is not None and len(old.conducting) == len(net.conducting):  # a rebuild keeps diode memory
            for diode, previous in zip(net._diodes, old._diodes):
                net.set_conducting(diode, previous.conducting, previous.residual)
        self._nodes = nodes
        self._compile_publish(nb, node_of)
        self._owners = {id(el): self.names.get(key, "?") for key, elements in nb.owned.items() for el in elements}
        self._loss_elements = {key: nb.owned.get(key, []) for key in nb.lossy}
        self._supply_elements = {key: nb.owned.get(key, []) for key in nb.supplying}
        self._energy_uses_time = any(
            isinstance(element, (VoltageSource, CurrentSource))
            and callable(element.value)
            and not isinstance(element.value, _BoundValue)
            for elements in self._supply_elements.values()
            for element in elements
        )
        self._mixtures = tuple(nb.mixtures)
        self._mixture_modes.clear()
        self._mixture_loss.clear()
        self._mixture_supply.clear()
        self.locate_commutations = not self._mixtures and any(isinstance(el, Diode) for el in net.elements)
        self._state_diode_groups = ()
        self._device_bindings.clear()
        self._structure_plan = None
        self._mode_decompositions.clear()
        self._consistent = None  # new parameters (a set event) may move the loops' and cutsets' constraints
        self._consistent_constraints = None
        transforms = []
        for index, value in nb.transforms:
            matrix = _numeric_values(value)
            transforms.append((index, matrix, np.linalg.inv(matrix)))
        self._transforms = tuple(transforms)
        self._stored = None
        hooks = [(comp, nb.hooks[id(comp)]) for comp in self.components if id(comp) in nb.hooks]
        self._hooks = tuple((comp, hook["after"]) for comp, hook in hooks if hook["after"] is not None)
        self._watched = tuple(sorted({net._diodes.index(d) for _c, hook in hooks if hook["after"] is not None
                                      for d in hook["diodes"]}))
        self.commutation_active = self.locate_commutations or bool(self._hooks)
        self._mirrored_valid = False
        self._boundary_signature = None
        self._mode_targets, self._discrete_mode_targets, held_ratios = (
            self._mode_input_targets(net)
        )
        self.requires_topology_freeze = self.commutation_active or held_ratios
        self._direct_dynamic = False
        self._direct_mode_key = None
        self._numeric_stages = False
        self._stage_signature = None
        if self.form == "device":
            (self.scalar_derivatives, self.scalar_outputs,
             self.scalar_output_evaluator) = net.scalar_evaluators()
            compiled = (
                self._compile_equation_stages()
                or (not self._standalone_runtime and self._compile_device_runtime())
            )
        else:
            compiled = self._compile_equation_stages()
        if not compiled:
            self._use_runtime_stages()
        if rebuilding:
            for view in self._views:
                view.update_circuit_dependencies(self)
            self.plan_changed = True

    def _mode_input_targets(
        self, net: Netlist
    ) -> tuple[set[tuple[int, str]], set[tuple[int, str]], bool]:
        """Record fields selecting modes and whether every dynamic ratio is held."""
        all_targets = {
            *((id(component.input), field) for component, field, _key in self.copies),
            *((id(self.input), key) for key in self.inputs),
        }
        held_targets = {
            *((id(self.input), key) for key in self._held_input_keys),
            *((id(component.input), field) for component, field, key in self.copies
              if key in self._held_input_keys),
        }

        def dependencies(callback: Callable) -> Optional[set[tuple[int, str]]]:
            declared = _record_dependencies(callback)
            if declared is None or declared[0] is None:
                return None
            record, fields = declared
            return {(id(record), field) for field in fields}

        discrete: set[tuple[int, str]] = set()
        callbacks = [switch.closed for switch in net._switches]
        callbacks.extend(diode.gate for diode in net._diodes if diode.gate is not None)
        ratios = [transformer.N for transformer in net._transformers
                  if transformer.constant is None]
        for callback in callbacks:
            fields = dependencies(callback)
            discrete.update(all_targets if fields is None else fields)
        structural = set(discrete)
        for callback in ratios:
            fields = dependencies(callback)
            structural.update(all_targets if fields is None else fields)
        ratio_dependencies = [dependencies(callback) for callback in ratios]
        held_ratios = bool(ratios) and all(
            fields is not None and fields <= held_targets
            for fields in ratio_dependencies
        )
        return (structural.intersection(all_targets),
                discrete.intersection(all_targets), held_ratios)

    def _compile_publish(self, nb: NetlistBuilder, node_of: dict[int, Node]) -> None:
        net = self.netlist

        def slot(kind: str, target: Any, port: int = 0) -> Optional[slice]:
            try:
                return net.output_slice(kind, target, port)
            except KeyError:  # a node no primitive uses
                return None

        publications = []  # (out key, dimension, [(coefficient, output slice)]): publish() and the C++ export

        def value(key: str, terms: list[tuple[float, Optional[slice]]], dim: int) -> None:
            terms = [(s, sl) for s, sl in terms if sl is not None]
            publications.append((key, dim, terms))

        port_terms: dict[int, tuple[Optional[slice], list[tuple[float, Optional[slice]]], int]] = {}
        for port, group, voltage_key, current_key in self.port_values:
            node = None if group is None else node_of[id(group)]
            dimension = _kind(port)[0]
            voltage = slot("node_voltage", node)
            value(voltage_key, [(1.0, voltage)], dimension)
            terms = []
            for element in nb.owned.get(id(port.component), ()):
                for branch_port, (start, end) in enumerate(element.ports):
                    if start is node:
                        terms.append((1.0, slot("branch_current", element, branch_port)))
                    if end is node:
                        terms.append((-1.0, slot("branch_current", element, branch_port)))
            value(current_key, terms, dimension)
            port_terms[id(port)] = voltage, terms, dimension
        for component, field, port, quantity, factor in nb.port_outputs:
            voltage, current, dimension = port_terms[id(port)]
            terms = [(factor, voltage)] if quantity == "voltage" else [
                (factor * coefficient, output) for coefficient, output in current
            ]
            value(self.published[id(component), field], terms, dimension)
        for comp, field, declared in nb.outputs:
            terms = [(factor, slot(kind, target, *port))
                     for factor, (kind, target, *port) in declared]
            dimensions = {output.stop - output.start for _factor, output in terms
                          if output is not None}
            if len(dimensions) > 1:
                raise ConfigError(
                    f"{type(comp).__name__}.{field}: combined netlist quantities differ in dimension"
                )
            value(self.published[id(comp), field], terms, dimensions.pop() if dimensions else 1)
        self.publications = tuple(publications)
        selected = (tuple(range(net.n_out)) if self.form == "device" else
                    tuple(sorted({index for _key, _dimension, terms in publications
                                  for _factor, output in terms
                                  for index in range(output.start, output.stop)})))
        self.output_indices = selected
        position = {index: k for k, index in enumerate(selected)}
        lines: list[str] = []
        env: dict[str, Any] = {}
        computed: dict[str, str] = {}
        for key, dimension, terms in publications:
            parts = [
                " + ".join(
                    f"{factor!r} * o[{position[output.start + k]}]"
                    for factor, output in terms
                ) or "0.0"
                for k in range(dimension)
            ]
            expression = (f"({parts[0]}) + 1j * ({parts[1]})" if dimension == 2 else parts[0])
            self._append_publication_writes(key, expression, lines, env, "runtime", computed)
        lines.extend(f"output.{output} = input.{input_}" for input_, output in self.passthrough)
        env.update(input=self.input, output=self.output)
        exec(compile("def publish(o):\n" + "\n".join("    " + line for line in lines or ["pass"]) + "\n",
                     "<peslite circuit outputs>", "exec"), env)
        self.publish = env["publish"]

    def _use_runtime_stages(self) -> None:
        """Evaluate a circuit as one runtime operation when it cannot be split statically."""
        self._output_dependencies = {"publish_outputs": ((self, "solve"),)}
        self._output_producers = dict.fromkeys(self.outputs, "publish_outputs")
        self.output_stages = ((
            OutputMethod("solve", tuple(self.inputs), ()),
            OutputMethod("publish_outputs", (), tuple(self.outputs), rhs=False),
        ) if self.outputs else None)
        self._bind_stage_owners()

    def _bind_stage_owners(self) -> None:
        """Retain stage metadata and assign derivative stages to the views they advance."""
        owners = (self, *self._views)
        self._output_stage_owners = {
            stage.method: owners for stage in self.output_stages or () if stage.rhs
        }
        for stage in self.output_stages or ():
            method = getattr(self, stage.method)
            function = getattr(method, "__func__", method)
            self._output_stage_fields[function] = tuple(stage.outputs)

    def _input_names(
        self, value: Any, copied: Mapping[tuple[int, str], str]
    ) -> Optional[set[str]]:
        """Return declared circuit inputs of a netlist value, or ``None`` if unknown."""
        declared = _record_dependencies(value)
        if declared is None:
            return None
        record, fields = declared
        if record is None:
            return set()
        return {
            key for field in fields
            if (key := field if record is self.input else copied.get((id(record), field)))
            is not None
        }

    def _copied_inputs(self) -> dict[tuple[int, str], str]:
        """Map component input fields copied by this circuit to circuit input names."""
        return {
            (id(component.input), field): key
            for component, field, key in self.copies
        }

    def _program_dependencies(self, program: tuple) -> tuple[str, ...]:
        """Circuit input fields read by one generated scalar program."""
        statements, _result, namespace, expressions = program
        expressions = expressions or ()
        active = "\n".join((*_live_statements(statements, expressions), *expressions))
        copied = self._copied_inputs()
        dependencies = set()
        for name, value in namespace.items():
            if re.search(rf"\b{re.escape(name)}\b", active) is None:
                continue
            names = self._input_names(value, copied)
            if names is None:
                return tuple(self.inputs)
            dependencies.update(names)
        return tuple(name for name in self.inputs if name in dependencies)

    def _structural_dependencies(
        self, topology: Topology
    ) -> Optional[tuple[Callable[[Sequence[int]], tuple[str, ...]], tuple[str, ...]]]:
        """Return conservative circuit-input dependencies from the numeric network structure."""
        reach = topology.input_reach()
        if reach is None:
            return None
        derivatives, outputs, _state_ratios = reach
        net, copied = self.netlist, self._copied_inputs()
        names: list[set[str]] = []
        values = [
            (source.value, getattr(source, "derivative", None))
            for source, _offset in net._sources
        ] + [(transformer.N, None) for transformer in net._transformers]
        for value, derivative in values:
            current = self._input_names(value, copied)
            rate = set() if derivative is None else self._input_names(derivative, copied)
            if current is None or rate is None:
                return None
            names.append(current | rate)

        def ordered(indices: Sequence[int]) -> tuple[str, ...]:
            found = set().union(*(names[int(index)] for index in indices)) if len(indices) else set()
            return tuple(name for name in self.inputs if name in found)

        return (
            lambda selected: ordered(np.flatnonzero(outputs[:, list(selected)].any(axis=1))),
            ordered(np.flatnonzero(derivatives)) if self.n_states else (),
        )

    def _copy_prelude(
        self, dependencies: Sequence[str], namespace: dict[str, Any]
    ) -> list[str]:
        """Copy only the component inputs read by one generated circuit stage."""
        active = set(dependencies)
        body = []
        for index, (component, field, key) in enumerate(self.copies):
            if key not in active:
                continue
            token = f"_circuit_input_{index}"
            namespace[token] = component.input
            body.append(f"{token}.{field} = self.input.{key}")
        return body

    def _transform_expressions(
        self, values: Mapping[int, str], *, inverse: bool = False
    ) -> dict[int, str]:
        """Apply the declared coordinate maps directly to scalar expressions."""
        transformed = dict(values)
        transforms = reversed(self._transforms) if inverse else self._transforms
        for indices, matrix, inverse_matrix in transforms:
            active = inverse_matrix if inverse else matrix
            source = [transformed[int(index)] for index in indices]
            transformed.update(
                (int(index), _linear_expression(row, source))
                for index, row in zip(indices, active)
            )
        return transformed

    def _state_expressions(
        self, namespace: dict[str, Any], prefix: str
    ) -> dict[int, str]:
        """Bind component state records and return their netlist-coordinate expressions."""
        values: dict[int, str] = {}
        for item, (component, name, complex_state, offset) in enumerate(self.layout):
            token = f"{prefix}_{item}"
            namespace[token] = component.state
            values[offset] = f"{token}.{name}.real" if complex_state else f"{token}.{name}"
            if complex_state:
                values[offset + 1] = f"{token}.{name}.imag"
        return self._transform_expressions(values)

    def _derivative_expressions(
        self, values: Sequence[str]
    ) -> dict[int, str]:
        """Map netlist-coordinate derivatives back to component coordinates."""
        if len(values) != len(self.netlist.states):
            raise ValueError("a netlist derivative program returned the wrong state dimension")
        mapped = {
            int(state): value for state, value in zip(self.netlist.states, values)
        }
        return self._transform_expressions(mapped, inverse=True)

    def _bind_program(
        self, program: tuple, dependencies: Sequence[str]
    ) -> tuple[list[str], tuple[str, ...], dict[str, Any]]:
        """Bind generated netlist expressions to the assembled state records."""
        statements, _result, source_namespace, expressions = program
        namespace = {}
        bound_values: dict[str, str] = {}
        for name, value in source_namespace.items():
            if isinstance(value, _BoundValue):
                token = f"_circuit_value_{len(bound_values)}"
                namespace[token] = value.record
                expression = f"{token}.{value.field}"
                if value.invert:
                    expression = f"not {expression}"
                elif value.factor == -1.0:
                    expression = f"-({expression})"
                elif value.factor != 1.0:
                    expression = f"{value.factor!r} * ({expression})"
                bound_values[name] = expression
            else:
                namespace[name] = value.callback if isinstance(value, _CallableValue) else value
        state_values = self._state_expressions(namespace, "_circuit_state")

        def bind(source: str) -> str:
            for name, expression in bound_values.items():
                source = re.sub(rf"\b{re.escape(name)}\((?:t)?\)", expression, source)
            for offset in sorted(state_values, reverse=True):
                source = source.replace(f"y[{offset}]", f"({state_values[offset]})")
            return source[4:] if source.startswith("    ") else source

        body = self._copy_prelude(dependencies, namespace)
        body.extend(bind(statement)
                    for statement in _live_statements(statements, expressions or ()))
        return body, tuple(bind(expression) for expression in expressions or ()), namespace

    def _generated_method(
        self, name: str, body: Sequence[str], namespace: Mapping[str, Any], *,
        inline: bool = True,
    ) -> Callable:
        """Compile one method and expose the same straight-line body to the Model compiler.

        A circuit mode may change coefficients without changing stage dependencies.  In that
        case the Model already holds the bound method, so update that function in place instead
        of rebuilding the scheduling graph.
        """
        env = dict(namespace)
        _compile_straight_line(env, "<peslite direct circuit>", (
            (f"{name}(self, t)", body),
        ))
        compiled = env[name]
        previous = self.__dict__.get(name)
        if isinstance(previous, MethodType):
            function = previous.__func__
            function.__globals__.update(env)
            function.__code__ = compiled.__code__
        else:
            function = compiled
            setattr(self, name, MethodType(function, self))
        if inline:
            identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", "\n".join(body)))
            function._peslite_inline_program = _InlineProgram(
                tuple(body), (), {
                    key: value for key, value in namespace.items() if key in identifiers
                }
            )
        elif hasattr(function, "_peslite_inline_program"):
            del function._peslite_inline_program
        function._peslite_atomic_equation = True
        return getattr(self, name)

    def _append_publications(
        self, keys: Sequence[str], values: Mapping[int, str], body: list[str],
        namespace: dict[str, Any], label: str,
    ) -> None:
        """Append direct circuit-output and physical-port writes for scalar netlist rows."""
        publications = {key: (dimension, terms)
                        for key, dimension, terms in self.publications}

        def scaled(factor: float, expression: str) -> str:
            return f"({expression})" if factor == 1.0 else f"{factor!r} * ({expression})"

        computed: dict[str, str] = {}
        for key in keys:
            dimension, terms = publications[key]
            parts = [
                " + ".join(
                    scaled(factor, values[output.start + part])
                    for factor, output in terms
                ) or "0.0"
                for part in range(dimension)
            ]
            expression = (f"({parts[0]}) + 1j * ({parts[1]})"
                          if dimension == 2 else parts[0])
            self._append_publication_writes(key, expression, body, namespace, label, computed)

    def _append_publication_writes(
        self, key: str, expression: str, body: list[str], namespace: dict[str, Any],
        label: str, computed: dict[str, str],
    ) -> None:
        """Write one public circuit value to every record that declares it.

        The value is evaluated once into a local. ``computed`` maps the expressions already
        evaluated in the same straight-line body to their locals: publications read only states
        and the body's own locals and write only output records, so an equal expression has the
        same value and reuses the local.
        """
        targets: list[tuple[Any, str]] = [(self.output, key)]
        for port, _group, voltage_key, current_key in self.port_values:
            if key == voltage_key:
                targets.append((port, "voltage"))
            if key == current_key:
                targets.append((port, "current"))
        components = {id(component): component for component in self.components}
        for (owner, field), published in self.published.items():
            component = components.get(owner)
            if component is not None and published == key:
                targets.append((component.output, field))
        known = computed.get(expression)
        if known is not None:
            expression = known
        else:
            value = f"_published_{label}_{len(namespace)}"
            body.append(f"{value} = {expression}")
            computed[expression] = value
            expression = value
        for target, field in targets:
            token = f"_circuit_output_{label}_{len(namespace)}"
            namespace[token] = target
            body.append(f"{token}.{field} = {expression}")

    def _stage_programs(
        self, topology: Optional[Topology] = None
    ) -> tuple[Topology, bool, Callable, Callable]:
        """Scalar equation programs shared by evaluation and dependency planning."""
        net = self.netlist
        topology = net.topology() if topology is None else topology
        dynamic = any(transformer.constant is None for transformer in net._transformers)
        if dynamic:
            derivatives = net._dynamic_scalar_function(topology, "derivatives")
            outputs = lambda selected: net._dynamic_scalar_function(  # noqa: E731
                topology, "outputs", selected
            )
        elif self.form == "device":
            derivatives = self.scalar_derivatives
            outputs = self.scalar_output_evaluator
        else:
            derivatives = net._scalar_function(topology, "derivatives")
            outputs = lambda selected: net._scalar_function(  # noqa: E731
                topology, "outputs", tuple(selected)
            )
        return topology, dynamic, derivatives, outputs

    def _mode_dependencies(self, programs: Sequence[Any]) -> Optional[tuple[str, ...]]:
        """Union the inputs read by equivalent equation programs in structural modes."""
        dependencies: set[str] = set()
        for program in programs:
            if program is None or program[3] is None:
                return None
            dependencies.update(self._program_dependencies(program))
        return tuple(key for key in self.inputs if key in dependencies)

    def _output_stage_groups(
        self,
        publications: Mapping[str, tuple[int, Sequence[tuple[float, slice]]]],
        input_dependencies: Callable[[list[int]], Optional[tuple[str, ...]]],
    ) -> Optional[list[tuple[tuple[str, ...], list[str], list[int]]]]:
        """Group public rows by every input their equations may read in any planned mode."""
        groups: dict[tuple[str, ...], tuple[list[str], set[int]]] = {}
        for key in publications:
            keys = [key]
            selected = sorted({
                index for key in keys for _factor, output in publications[key][1]
                for index in range(output.start, output.stop)
            })
            dependencies = input_dependencies(selected)
            if dependencies is None:
                return None
            grouped_keys, rows = groups.setdefault(dependencies, ([], set()))
            grouped_keys.extend(keys)
            rows.update(selected)
        return [
            (dependencies, keys, sorted(rows))
            for dependencies, (keys, rows) in groups.items()
        ]

    def _append_passthrough_stage(
        self, stages: list[OutputMethod], producers: dict[str, str], prefix: str
    ) -> None:
        """Append direct input-to-output copies to a compiled circuit plan."""
        if not self.passthrough:
            return
        name = f"_{prefix}_passthrough"
        body = [f"self.output.{output} = self.input.{input_}"
                for input_, output in self.passthrough]
        self._generated_method(name, body, {})
        inputs = tuple(dict.fromkeys(input_ for input_, _output in self.passthrough))
        outputs = tuple(output for _input, output in self.passthrough)
        stages.append(OutputMethod(name, inputs, outputs, rhs=False))
        producers.update(dict.fromkeys(outputs, name))

    def _compile_equation_stages(
        self, *, dependency_topologies: Optional[Sequence[Topology]] = ()
    ) -> bool:
        """Compile dependency-precise stages for every circuit formulation."""
        net = self.netlist
        if self._mixtures:
            return False
        numeric = self.form != "device"
        if not numeric and net._diodes:
            return False
        publications = {key: (dimension, terms)
                        for key, dimension, terms in self.publications}
        if set(publications) | {output for _input, output in self.passthrough} != set(self.outputs):
            return False
        conservative = dependency_topologies is None
        planned = () if conservative else tuple(dependency_topologies)
        topology = net.topology()
        structural = None
        if numeric and topology.n_z > _DENSE_GROUP:
            structural = tuple(
                self._structural_dependencies(item) for item in (topology, *planned)
            )
            if any(item is None for item in structural):
                structural = None
        if structural:
            dynamic = any(transformer.constant is None for transformer in net._transformers)
            derivative_dependencies = tuple(
                key for key in self.inputs
                if any(key in dependencies for _outputs, dependencies in structural)
            )

            def output_dependencies(selected: list[int]) -> tuple[str, ...]:
                found = {
                    key
                    for outputs, _derivatives in structural
                    for key in outputs(selected)
                }
                return tuple(key for key in self.inputs if key in found)
        else:
            topology, dynamic, derivatives, output_evaluator = self._stage_programs()
            dependency_programs = tuple(self._stage_programs(item) for item in planned)
            output_evaluators = (output_evaluator, *(item[3] for item in dependency_programs))
            derivative_program = getattr(derivatives, "_peslite_scalar_program", None)
            if self.n_states and (derivative_program is None or derivative_program[3] is None):
                return False
            derivative_dependencies = self._mode_dependencies((
                derivative_program,
                *(getattr(item[2], "_peslite_scalar_program", None)
                  for item in dependency_programs),
            )) if self.n_states else ()

            def output_dependencies(selected: list[int]) -> Optional[tuple[str, ...]]:
                programs = [
                    getattr(evaluator(selected), "_peslite_scalar_program", None)
                    for evaluator in output_evaluators
                ]
                return self._mode_dependencies(programs)

        if conservative:
            precise_output_dependencies = output_dependencies

            def output_dependencies(selected: list[int]) -> Optional[tuple[str, ...]]:
                return (None if precise_output_dependencies(selected) is None
                        else tuple(self.inputs))

            if derivative_dependencies is not None and self.n_states:
                derivative_dependencies = tuple(self.inputs)
        output_groups = self._output_stage_groups(publications, output_dependencies)
        if output_groups is None or derivative_dependencies is None:
            return False
        signature = (
            tuple((dependencies, tuple(keys), tuple(selected))
                  for dependencies, keys, selected in output_groups),
            derivative_dependencies,
            tuple(self.passthrough),
        )
        prefix = "numeric" if numeric else "device"
        stages: list[OutputMethod] = []
        producers: dict[str, str] = {}
        for index, (dependencies, keys, selected) in enumerate(output_groups, 1):
            name = f"_{prefix}_outputs_{index}"
            positions = {position: offset for offset, position in enumerate(selected)}
            if structural:
                namespace = {
                    "_circuit_numeric_rows": (
                        lambda t, rows=tuple(selected):
                        net.evaluate_outputs(t, self._states(), rows).tolist()
                    )
                }
                body = self._copy_prelude(dependencies, namespace)
                body.append("_circuit_rows = _circuit_numeric_rows(t)")
                expressions = tuple(
                    f"_circuit_rows[{offset}]" for offset in range(len(selected))
                )
            else:
                program = output_evaluator(selected)._peslite_scalar_program
                body, expressions, namespace = self._bind_program(program, dependencies)
            values = {
                position: expressions[offset]
                for position, offset in positions.items()
            }
            self._append_publications(
                keys, values, body, namespace, f"{prefix}_{index}",
            )
            self._generated_method(name, body, namespace, inline=not numeric)
            stages.append(OutputMethod(name, dependencies, tuple(keys), rhs=False))
            producers.update(dict.fromkeys(keys, name))

        self._append_passthrough_stage(stages, producers, prefix)
        if self.n_states:
            dependencies = derivative_dependencies
            if numeric:
                namespace = {}
                body = self._copy_prelude(dependencies, namespace)
                namespace["_circuit_numeric_derivatives"] = (
                    lambda t: net.evaluate(t, self._states()).tolist()
                )
                body.append("_circuit_values = _circuit_numeric_derivatives(t)")
                values = tuple(
                    f"_circuit_values[{int(state)}]" for state in net.states
                )
            else:
                body, values, namespace = self._bind_program(
                    derivative_program, dependencies
                )
            mapped = self._derivative_expressions(values)
            body.extend(
                f"self.derivatives[{state}] = {value}"
                for state, value in sorted(mapped.items())
            )
            body.append("self.t = t")
            name = f"_{prefix}_derivatives"
            self._generated_method(name, body, namespace, inline=not numeric)
            stages.append(OutputMethod(name, dependencies, (), rhs=True))

        self.output_stages = tuple(stages)
        self._output_dependencies = {}
        self._output_producers = producers
        self._bind_stage_owners()
        self._direct_dynamic = dynamic
        self._direct_mode_key = (
            net._dynamic_structure_key(topology) if dynamic else
            topology.mode.key[:2] if numeric or net._switches else None
        )
        self._numeric_stages = numeric
        self._stage_signature = signature
        return True

    def plan_export(self) -> None:
        """Make one dependency plan safe for every binary-device mode."""
        net = self.netlist
        variables = (
            *(("closed", index) for index in range(len(net._switches))),
            *(("conducting", index) for index in range(len(net._diodes))),
        )
        if not variables:
            return
        topology = net.topology()
        decomposition = _mode_decomposition(self, variables, topology.mode)
        previous = self._stage_signature
        planned = None if decomposition is None else decomposition.topologies
        if not self._compile_equation_stages(dependency_topologies=planned):
            conservative = any(
                stage.rhs and set(stage.inputs) == set(self.inputs)
                for stage in self.output_stages or ()
            )
            if not conservative:
                raise RuntimeError("the circuit has no mode-independent export dependency plan")
            return
        if self._stage_signature != previous:
            for view in self._views:
                view.update_circuit_dependencies(self)
            self.plan_changed = True

    def _refresh_device_evaluator(self, topology: Optional[Topology] = None) -> bool:
        """Select the compiled scalar evaluator for the current structural mode."""
        holder = self._device_runtime
        if holder is None:
            return False
        self.netlist._last = None
        topology = self.netlist.topology() if topology is None else topology
        margin_key = self.netlist.scalar_margin_key(topology)
        if (holder[1] is topology and holder[2] is not None
                and self._margin_evaluator_key == margin_key):
            return True
        dynamic = any(
            transformer.constant is None for transformer in self.netlist._transformers
        )
        key = (topology.mode.key, margin_key[1])
        binding = None if dynamic else self._device_bindings.get(key)
        if binding is None:
            binding = topology, self.netlist.scalar_margin_evaluator(topology), (
                self.netlist._dynamic_scalar_function(topology)
                if dynamic
                else self.netlist._scalar_function(topology, "all")
            )
            if not dynamic:
                self._device_bindings[key] = binding
        topology, margin, evaluate = binding
        holder[:] = self.netlist, topology, evaluate
        self._margin_evaluator_key = margin_key
        self._margin_evaluator = margin
        self._device_commands = self._boundary_mode()
        return True

    def _compile_device_runtime(self) -> bool:
        """Bind the current compiled mode; mode selection remains at physical boundaries."""
        if self._mixtures:
            return False
        if self._device_runtime is not None:
            self._refresh_device_evaluator()
            self._use_runtime_stages()
            return True

        self._device_runtime = [self.netlist, None, None]
        self._refresh_device_evaluator()
        namespace: dict[str, Any] = {
            "_circuit_runtime": self._device_runtime,
            "_circuit_evaluation": _Evaluation,
        }
        state_values = self._state_expressions(namespace, "_circuit_runtime_state")
        states = [state_values[index] for index in range(self.n_states)]
        state_tuple = "(" + ", ".join(states) + ("," if len(states) == 1 else "") + ")"
        body = []
        for item, (component, field, key) in enumerate(self.copies):
            token = f"_circuit_runtime_input_{item}"
            namespace[token] = component.input
            body.append(f"{token}.{field} = self.input.{key}")
        body.append(f"_circuit_result = _circuit_runtime[2](t, {state_tuple})")
        derivatives = self._derivative_expressions(tuple(
            f"_circuit_result[0][{index}]" for index in range(len(self.netlist.states))
        ))
        body.extend(
            f"self.derivatives[{state}] = {value}"
            for state, value in sorted(derivatives.items())
        )
        if self.locate_commutations:
            selected = tuple(states[int(state)] for state in self.netlist.states)
            state_values = "(" + ", ".join(selected) + ("," if len(selected) == 1 else "") + ")"
            body.append(
                "_circuit_runtime[0]._last = _circuit_evaluation("
                f"_circuit_runtime[1], t, {state_values}, _circuit_result[2], "
                "_circuit_result[3], None, _circuit_result[1])"
            )
        body.extend((
            "self._solved_outputs = _circuit_result[1]",
            "self.t = t",
        ))
        self._generated_method("solve", body, namespace)
        self._use_runtime_stages()
        return True

    # ---------------------------------------------------------------- evaluation
    def _states(self) -> np.ndarray:
        """The netlist's state vector: the components' states, in the netlist's coordinates where declared
        (exactly the vector last stored while the states are unchanged, so margins do not move by round-off)."""
        values = self.gather()
        if not self._transforms:
            return np.array(values)
        if self._stored is not None and self._stored[0] == values:
            return self._stored[1].copy()
        y = np.array(values)
        for index, T, _inverse in self._transforms:
            y[index] = T @ y[index]
        return y

    def linear_mode(self) -> tuple[
        tuple, Callable[[], np.ndarray], Optional[tuple[object, float]]
    ]:
        """Return the present linear-mode identity, matrix builder and stability certificate."""
        topology = self._bound_device_topology()
        if topology is None:
            return self.netlist.linear_mode()
        return (topology.state_matrix_key(), lambda: topology.state_space.A,
                topology.state_matrix_stability_certificate())

    def stability_certificate(self) -> Optional[tuple[object, float]]:
        """Return the present mode's stability certificate without its identity or matrix."""
        topology = self._bound_device_topology()
        if topology is None:
            return self.netlist.stability_certificate()
        return topology.state_matrix_stability_certificate()

    def _store(self, y: np.ndarray) -> None:
        """Write a netlist state vector back into the components."""
        if not self._transforms:
            self.scatter(y.tolist())
            return
        net_y = y.copy()
        for index, _T, inverse in reversed(self._transforms):
            y[index] = inverse @ y[index]
        self.scatter(y.tolist())
        self._stored = (self.gather(), net_y)

    def _degenerate(self, exc: _Degenerate, t: float) -> ConfigError:
        owners = sorted({self._owners.get(id(el), "?") for el in exc.elements(self.netlist)})
        return ConfigError(f"t = {t:.9g} s: the netlist circuit has no consistent equations in its present mode "
                           f"(switches, diodes) through {owners}: {exc}")

    def solve(self, t: float) -> None:
        net = self.netlist
        self.receive()
        try:
            if self._mixtures:
                d, outputs = self._evaluate_mixture(t, self._states())
            elif self.form == "device":
                states = self._states() if self._transforms else self.gather()
                d, outputs = net.evaluate_scalar_raw(t, states, record=self.locate_commutations)
            else:
                d, outputs = net.evaluate_with_outputs(t, self._states(), self.output_indices)
        except _Degenerate as exc:
            raise self._degenerate(exc, t) from None
        if self._transforms:
            d = np.asarray(d)
            for index, _T, inverse in reversed(self._transforms):
                d[index] = inverse @ d[index]
            self.derivatives = d.tolist()
        else:
            self.derivatives = list(d)
        self._solved_outputs = outputs
        self.t = t

    def publish_outputs(self, _t: float) -> None:
        """Publish electrical observations from the most recent circuit evaluation."""
        self.publish(self._solved_outputs)

    def _evaluate_mixture(self, t: float, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Average one local netlist over the product of its declared discrete cases."""
        sources = []
        for cases, select in self._mixtures:
            weighted = tuple((float(weight), case) for weight, case in cases() if weight > 0.0)
            if not weighted:
                raise ConfigError("an averaged netlist declaration produced no positive-weight cases")
            total = sum(weight for weight, _case in weighted)
            sources.append((tuple((weight / total, case) for weight, case in weighted), select))
        derivatives = np.zeros(self.n_states)
        outputs = np.zeros(self.netlist.n_out)
        losses = dict.fromkeys(self._loss_elements, 0.0)
        supplies = dict.fromkeys(self._supply_elements, 0.0)
        try:
            for selected in itertools.product(*(weighted for weighted, _select in sources)):
                weight = math.prod(item[0] for item in selected)
                cases = tuple(item[1] for item in selected)
                for case, (_weighted, select) in zip(cases, sources):
                    select(case)
                key = tuple(
                    case if isinstance(case, (str, bytes, int, float, complex, tuple))
                    else tuple(case) for case in cases
                )
                remembered = self._mixture_modes.get(key)
                if remembered is not None:
                    conducting, residual = remembered
                    for diode, on, offset in zip(self.netlist._diodes, conducting, residual):
                        diode.conducting, diode.residual = on, offset
                self.netlist.commutate(t, states)
                derivative, output = self.netlist.evaluate_scalar_raw(t, states, record=True)
                derivatives += weight * np.asarray(derivative)
                outputs += weight * np.asarray(output)
                self._mixture_modes[key] = (
                    self.netlist.conducting,
                    tuple(diode.residual for diode in self.netlist._diodes),
                )
                for key, elements in self._loss_elements.items():
                    losses[key] += weight * self.netlist.dissipated_power(elements)
                for key, elements in self._supply_elements.items():
                    supplies[key] -= weight * sum(
                        self.netlist.power(element) for element in elements
                        if isinstance(element, (VoltageSource, CurrentSource))
                    )
        finally:
            for _weighted, select in sources:
                select(None)
        self._mixture_loss, self._mixture_supply = losses, supplies
        return derivatives, outputs[list(self.output_indices)]

    def set_outputs(self, t: float) -> None:
        self.solve(t)

    def cpp_output_stage(self, generator: Any, method: Any) -> list[str]:
        """Publish the outputs declared by the same dependency stage used in Python."""
        function = getattr(method, "__func__", method)
        try:
            fields = self._output_stage_fields[function]
        except KeyError:
            raise RuntimeError("C++ export received an unknown circuit output stage") from None
        return generator.trace_circuit_outputs(self, fields) if fields else []

    def cpp_commutation_margin(self, generator: Any) -> str:
        """Lower the netlist's diode guard through the standalone C++ backend."""
        if not self.commutating_diodes:
            return "std::numeric_limits<double>::infinity()"
        return generator.trace_circuit_commutation_margin(self)

    def cpp_commutate(self, generator: Any) -> list[str]:
        """Lower the netlist's diode transition through the standalone C++ backend."""
        if not self.commutating_diodes:
            return []
        return generator.trace_circuit_commutation(self)

    @property
    def commutating_diodes(self) -> tuple[Diode, ...]:
        """Diodes whose own conduction state applies whenever their gate is inactive."""
        return tuple(self.netlist._diodes)

    def equation_state_bindings(self) -> dict[str, tuple[Any, str]]:
        """Persistent discrete state owned by the circuit's netlist primitives."""
        return {
            f"diode_{index}_{attribute}": (diode, attribute)
            for index, diode in enumerate(self.commutating_diodes)
            for attribute in ("conducting", "residual")
        } | {
            f"mirrored_{index}": (self.netlist._diodes[index], "mirrored")
            for index in self._watched
        } | {
            attribute: (self, attribute)
            for attribute in (
                "_mirrored_valid", "impulsive_dissipated_energy", "impulsive_supplied_energy"
            )
        }

    def rhs(self, t: float):
        return ()

    def freeze_topology(self) -> None:
        """Keep the present netlist topology throughout one smooth integration attempt."""
        if self._bound_device_topology() is not None:
            return
        self.netlist.freeze_topology()
        if any(transformer.constant is None for transformer in self.netlist._transformers):
            self._refresh_device_evaluator()

    def release_topology(self) -> None:
        """Read topology callbacks again at the next model-changing boundary."""
        self.netlist.release_topology()

    def _bound_device_topology(self) -> Optional[Topology]:
        """Return the fixed topology already selected by the device runtime, if any."""
        runtime = self._device_runtime
        if (runtime is None or runtime[1] is None
                or any(transformer.constant is None for transformer in self.netlist._transformers)):
            return None
        return runtime[1]

    def held_inputs_changed(self, targets: Sequence[tuple[Any, str]]) -> None:
        """Select the compiled mode after held circuit inputs changed at a boundary."""
        if any((id(record), field) in self._mode_targets for record, field in targets):
            boundary = self._boundary_mode()
            previous = self._boundary_signature
            if (previous is not None
                    and boundary[0] == previous[0]
                    and boundary[1] != previous[1]
                    and self._commutate_state_diodes()):
                self._boundary_signature = boundary
                return
            self._refresh_device_evaluator()

    def _refresh_device_mode(self, topology: Optional[Topology] = None) -> None:
        """Bind or regenerate the equations selected by the current structural mode."""
        def mode_key(selected: Topology):
            return (self.netlist._dynamic_structure_key(selected)
                    if self._direct_dynamic else selected.mode.key[:2])

        if self._numeric_stages:
            selected = self.netlist.topology() if topology is None else topology
            if mode_key(selected) != self._direct_mode_key:
                previous = self._stage_signature
                self._compile_equation_stages()
                if self._stage_signature != previous:
                    for view in self._views:
                        view.update_circuit_dependencies(self)
                    self.plan_changed = True
            return
        if self._direct_mode_key is not None:
            self.netlist._last = None
            selected = self.netlist.topology() if topology is None else topology
            if mode_key(selected) == self._direct_mode_key:
                return
            self._compile_equation_stages()
            for view in self._views:
                view.update_circuit_dependencies(self)
            self.plan_changed = True
        else:
            self._refresh_device_evaluator(topology)

    # ---------------------------------------------------------------- mode changes and diodes
    def _pattern(self, mode: Any = None) -> tuple[bool, ...]:
        conducting = (mode or self.netlist.mode()).conducting
        return tuple(conducting[j] for j in self._watched)

    def commutation_margin(self) -> float:
        """Negative while the states are inconsistent with the present mode (it changed at a boundary),
        a diode must commutate or a component has not mirrored the conduction yet; read after the model
        was synchronized. A change of transformer ratios alone keeps constraints that do not involve them."""
        net = self.netlist
        runtime = self._device_runtime
        if (runtime is not None and runtime[1] is not None
                and self._boundary_mode() != self._device_commands):
            self._refresh_device_evaluator()
        evaluation = net._last
        topology = evaluation.topology if evaluation is not None and evaluation.t == self.t else net.topology()
        mode = topology.mode
        if self._hooks and (
            not self._mirrored_valid
            or self._pattern(mode) != tuple(self.netlist._diodes[index].mirrored
                                            for index in self._watched)
        ):
            return -math.inf
        key, last = mode.key, self._consistent
        if key != last:
            try:
                constraints = topology.constraint_key()
                if (constraints is not self._consistent_constraints
                        and not net.consistent(self.t, self._states(), topology)):
                    return -math.inf
            except _Degenerate as exc:
                raise self._degenerate(exc, self.t) from None
            self._consistent = key
            self._consistent_constraints = constraints
        if not self.locate_commutations:
            return math.inf
        evaluated = evaluation is not None and evaluation.t == self.t
        if not evaluated:
            return net.commutation_margin(self.t, self._states())
        margin_key = net.scalar_margin_key(topology)
        if self._margin_evaluator is None or self._margin_evaluator_key != margin_key:
            self._margin_evaluator = net.scalar_margin_evaluator(topology)
            self._margin_evaluator_key = margin_key
        output = evaluation.output if evaluation.output is not None else net.outputs()
        return self._margin_evaluator(output)

    def commutation_state_guards(self) -> Optional[tuple[tuple[int, float], ...]]:
        """Replace diode-output guards by proved signed state guards when possible.

        A conducting diode and every blocking diode tied to the opposite direction of
        the same scalar current can change mode only when that current reaches zero.
        Returning ``None`` keeps the full netlist margin, notably while all diodes in a
        group are blocking and their terminal voltages decide which one turns on.
        """
        if self._mixtures or self._hooks:
            return None
        net = self.netlist
        evaluation = net._last
        topology = self._bound_device_topology()
        if topology is None:
            mode = net.mode()
            topology = (
                evaluation.topology
                if (evaluation is not None and evaluation.t == self.t
                    and evaluation.topology.mode.key == mode.key) else
                net.topology(mode)
            )
        groups, guards = net.state_diode_guard_plan(topology)
        self._state_diode_groups = groups
        return guards

    def commutate(self, t: float) -> None:
        """Find the consistent diode pattern and jump the states into it (charge and flux conservation);
        then let the components mirror the conduction (``NetlistBuilder.commutation``)."""
        boundary = self._boundary_mode()
        previous_boundary = self._boundary_signature
        if (previous_boundary is not None
                and boundary[0] == previous_boundary[0]
                and boundary[1] != previous_boundary[1]
                and self._commutate_state_diodes()):
            self._boundary_signature = boundary
            return
        net, before = self.netlist, self.gather()
        try:
            y, loss = net.commutate(t, self._states())
        except _Degenerate as exc:
            raise self._degenerate(exc, t) from None
        self._store(y)
        self.impulsive_dissipated_energy += loss
        self.impulsive_supplied_energy += net.impulsive_supply
        self._consistent = net.mode().key
        self._consistent_constraints = net.topology().constraint_key()
        for comp, after in self._hooks:
            after(t, {n: complex(before[i], before[i + 1]) if z else before[i]
                      for c, n, z, i in self.layout if c is comp})
        if self._hooks:
            for index, conducting in zip(self._watched, self._pattern()):
                self.netlist._diodes[index].mirrored = conducting
            self._mirrored_valid = True
        self._boundary_signature = boundary
        self._refresh_device_mode()

    def _boundary_mode(self) -> tuple:
        """Discrete commands that can change the netlist constraints at an outer boundary."""
        return (
            tuple(bool(switch.closed()) for switch in self.netlist._switches),
            tuple(None if diode.gate is None else bool(diode.gate())
                  for diode in self.netlist._diodes),
        )

    def reconcile(self, t: float) -> None:
        """Apply one atomic projection after outer events changed electrical device commands."""
        mode = self._boundary_mode()
        if mode != self._boundary_signature:
            self.commutate(t)
        else:
            self._refresh_device_evaluator()

    def _commutate_state_diodes(self) -> bool:
        """Apply a gate-only mode change through proved current relations."""
        if not self._state_diode_groups or self._hooks:
            return False
        topology = self.netlist.select_state_diode_mode(
            self._states(), self._state_diode_groups
        )
        if topology is None:
            return False
        self._consistent = topology.mode.key
        self._consistent_constraints = topology.constraint_key()
        self._refresh_device_mode(topology)
        return True

    def consistent_state(self, t: float) -> None:
        """Project the states onto the present mode without recording a jump (audit states)."""
        y, _loss = self.netlist.project(t, self._states())
        self._store(y)
        topology = self.netlist.topology()
        self._consistent = topology.mode.key
        self._consistent_constraints = topology.constraint_key()
        self._boundary_signature = self._boundary_mode()
        self._refresh_device_mode()

    def initialize_mode(self) -> None:
        """Bind the topology established by initial commands without changing physical states."""
        topology = self.netlist.topology()
        self._boundary_signature = self._boundary_mode()
        self._refresh_device_mode(topology)

    def dissipated_power_of(self, component: Any) -> float:
        """The power dissipated in ``component``'s resistive primitives at the last evaluation."""
        if self._mixtures:
            return self._mixture_loss.get(id(component), 0.0)
        if self.form == "device":
            self.netlist.evaluate_scalar_raw(self.t, self._states(), record=True)
        return self.netlist.dissipated_power(self._loss_elements[id(component)])

    def supplied_power_of(self, component: Any) -> float:
        """The power supplied by ``component`` source primitives at the last evaluation."""
        if self._mixtures:
            return self._mixture_supply.get(id(component), 0.0)
        if self.form == "device":
            self.netlist.evaluate_scalar_raw(self.t, self._states(), record=True)
        return -sum(self.netlist.power(element) for element in self._supply_elements[id(component)]
                    if isinstance(element, (VoltageSource, CurrentSource)))

    def grouping_report(self) -> dict[str, Any]:
        """The netlist's automatic state-space-nodal groups, JSON-ready: per group its member subsystems,
        nodes (an internal node named ``<subsystem>~<k>`` after the subsystem whose primitives it joins),
        switching devices and interface nodes; then the interface and how the nodal solve uses them."""
        net, owners = self.netlist, self._owners
        plan, names, count = net.grouping(), {id(n): n.name for n in net.nodes if n.name}, {}
        for el in net.elements:
            for node in (n for port in el.ports for n in port if n is not None and id(n) not in names):
                owner = owners.get(id(el), "?")
                count[owner] = count.get(owner, 0) + 1
                names[id(node)] = f"{owner}~{count[owner]}"

        def members(elements: Sequence[Element]) -> list[str]:
            return list(dict.fromkeys(owners.get(id(el), "?") for el in elements))

        groups = [{"group": k, "kind": "passive" if plan.passive and k == len(plan.elements) - 1 else "switching",
                   "subsystems": members(elements), "nodes": [names[id(n)] for n in nodes],
                   "switching_devices": devices, "interface_nodes": [names[id(n)] for n in boundary]}
                  for k, (elements, nodes, devices, boundary)
                  in enumerate(zip(plan.elements, plan.nodes, plan.devices, plan.boundary))]
        solve = ("direct scalar equations" if net.form == "device" else
                 "dense state space per mode (groups unused)" if net.form == "statewise" else
                 "grouped: groups factorized per own mode, interface Schur system" if net._grouping_used() else
                 "one sparse system (fewer than two switching groups)")
        return {"circuit_model": net.form,
                "nodal_solve": solve,
                "groups": groups, "interface": {"nodes": [names[id(n)] for n in plan.interface],
                                                "subsystems": members(plan.interface_elements)}}


def _bind_energy(component: Any, circuit: Circuit) -> tuple[Optional[Callable], Optional[Callable]]:
    """Bind one component's declared power laws to its owning circuit."""
    loss = supply = None
    if id(component) in circuit._loss_elements:
        loss = lambda: circuit.dissipated_power_of(component)
        component.dissipated_power = loss
    if id(component) in circuit._supply_elements:
        supply = lambda: circuit.supplied_power_of(component)
        component.supplied_power = supply
    return loss, supply


class NetworkView:
    """A netlist component in the Model: the component's records and output equations, the
    Circuit's derivatives; every other attribute (energy declaration, parameters) is the component's.

    ``outputs``: fields the netlist publishes (``nb.output``) in place of the component's equations.
    A component's own commutation (``NetlistBuilder.commutation``) is exposed beside the Circuit's.
    """

    # The netlist's own diodes, switches and projections take over these component hooks (and the energy of jumps).
    _HIDDEN = frozenset(("state_bounds", "diode_blocks_reverse_current", "commutation_margin", "commutate",
                         "next_commutation_time", "project_state",
                         "impulsive_dissipated_energy", "impulsive_supplied_energy"))

    def __init__(self, component: Any, outputs: Sequence[str] = (),
                 inputs: Sequence[str] = ()) -> None:
        self.component = component
        self.equation_owner = component
        self._loss = self._supply = None
        self.state, self.input, self.output = component.state, component.input, component.output
        self.state_names = tuple(component.state_names)
        stages = _component_output_stages(component)
        kept = []
        for stage, remaining in _component_stages_beside_netlist(component, stages, outputs):
            kept.append(OutputMethod(
                stage.method, tuple(stage.inputs), remaining,
                rhs=not remaining or bool(set(remaining) & set(inputs)),
            ))
            setattr(self, stage.method, getattr(component, stage.method))
        self.declared = tuple(outputs)
        self.output_stages = tuple(kept) if kept or outputs else None
        self._output_dependencies: dict[str, tuple[tuple[Any, str], ...]] = {}
        self._derived_output_methods: tuple[str, ...] = ()
        self.outputs_need_inputs = component.outputs_need_inputs if stages is None else True

    @property
    def has_source(self) -> bool:
        """Whether the component or its assembled netlist owns an internal source."""
        return self._supply is not None or bool(getattr(self.component, "has_source", False))

    def bind(self, circuit: Circuit, fed: Sequence[str]) -> None:
        """Return ``circuit``'s derivatives; ``fed``: the inputs the circuit writes (ordering)."""
        self._circuit = circuit
        self._loss, self._supply = _bind_energy(self.component, circuit)
        entries = [(i, z) for c, _n, z, i in circuit.layout if c is self.component]
        items = "".join(f"d[{i}] + 1j * d[{i + 1}], " if z else f"d[{i}], " for i, z in entries)
        env: dict[str, Any] = {}
        exec(compile(f"def rhs(self, t):\n    d = self._circuit.derivatives\n    return ({items})\n",
                     "<peslite netlist view>", "exec"), env)
        results = tuple(
            f"self._circuit.derivatives[{i}] + 1j * self._circuit.derivatives[{i + 1}]"
            if complex_state else f"self._circuit.derivatives[{i}]"
            for i, complex_state in entries
        )
        env["rhs"]._peslite_inline_program = _InlineProgram((), results, {})
        env["rhs"]._peslite_netlist_rhs = (circuit, tuple(entries))
        self.rhs = MethodType(env["rhs"], self)
        if self.declared:
            derived = tuple(stage.method for stage in self.output_stages or ())
            self._derived_output_methods = derived
            env = {}
            body = "".join(
                f"    self.output.{field} = self._circuit.output."
                f"{circuit.published[id(self.component), field]}\n"
                for field in self.declared
            )
            exec(compile(f"def netlist_outputs(self, t):\n{body}",
                         "<peslite netlist outputs>", "exec"), env)
            env["netlist_outputs"]._peslite_netlist_outputs = (circuit, self.declared)
            self.netlist_outputs = MethodType(env["netlist_outputs"], self)
            self.output_stages = (OutputMethod(
                "netlist_outputs", (), self.declared, rhs=False
            ), *(self.output_stages or ()))
            self.update_circuit_dependencies(circuit)

    def update_circuit_dependencies(self, circuit: Circuit) -> None:
        """Point this view at the circuit stages producing its declared outputs."""
        if not self.declared:
            return
        producers = tuple(dict.fromkeys(
            circuit._output_producers[circuit.published[id(self.component), field]]
            for field in self.declared
        ))
        self._output_dependencies = {
            "netlist_outputs": tuple((circuit, method) for method in producers),
            **{
                method: ((self, "netlist_outputs"),)
                for method in self._derived_output_methods
            },
        }

    def equation_state_bindings(self) -> dict[str, tuple[Any, str]]:
        """State writable by the component and its registered netlist callbacks."""
        component = getattr(self.component, "equation_state_bindings", None)
        bindings = {} if component is None else dict(component())
        circuit = getattr(self, "_circuit", None)
        if circuit is not None and "commutate" in vars(self):
            bindings.update(
                (f"circuit.{name}", binding)
                for name, binding in circuit.equation_state_bindings().items()
            )
        return bindings

    def expose(self, margin: Any, commutate: Any, next_time: Any = None) -> None:
        """Expose the component's own commutation hooks for its netlist form (a gate clock)."""
        self.commutation_margin, self.commutate = margin, commutate
        if next_time is not None:
            self.next_commutation_time = next_time

    def rhs(self, t: float):  # replaced by bind()
        raise RuntimeError("the netlist view is not bound to its circuit")

    def set_outputs(self, t: float) -> None:
        for stage in self.output_stages or ():
            getattr(self, stage.method)(t)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("component", "_loss", "_supply") or name in NetworkView._HIDDEN:
            raise AttributeError(name)
        return getattr(self.component, name)

    def __repr__(self) -> str:
        return f"NetworkView({type(self.component).__name__})"

    @staticmethod
    def of(component: Any, outputs: Sequence[str] = (),
           inputs: Sequence[str] = ()) -> "NetworkView":
        """The view of ``component``; it declares energy exactly when the component does."""
        return (_EnergeticView if isinstance(component, Energetic) else NetworkView)(
            component, outputs, inputs
        )


class _EnergeticView(NetworkView):
    """A view of a component with an energy declaration (protocol members are looked up statically)."""

    @property
    def storage(self):
        return self.component.storage

    @property
    def ports(self):
        return self.component.ports

    def dissipated_power(self) -> float:
        return self.component.dissipated_power() if self._loss is None else self._loss()

    def supplied_power(self) -> float:
        return self.component.supplied_power() if self._supply is None else self._supply()


def _component_output_stages(component: Any) -> Optional[tuple[OutputMethod, ...]]:
    """Return the declared output stages, including the ordinary default stage."""
    stages = getattr(component, "output_stages", None)
    if stages is None and (component.outputs_need_inputs or component.output is not component.state):
        stages = (OutputMethod(
            "set_outputs",
            _fields(component.input) if component.outputs_need_inputs else (),
            _fields(component.output),
        ),)
    return None if stages is None else tuple(stages)


def _component_stages_beside_netlist(
    component: Any,
    stages: Optional[Sequence[OutputMethod]],
    published: Sequence[str],
) -> list[tuple[OutputMethod, tuple[str, ...]]]:
    """Keep component-specific stages beside the netlist-owned output fields."""
    published = set(published)
    return [
        (stage, tuple(field for field in stage.outputs if field not in published))
        for stage in stages or ()
        if not getattr(getattr(component, stage.method), "_peslite_device_adapter", False)
    ]


def _bind_standalone(component: Any, circuit: Circuit, outputs: Sequence[str]) -> None:
    """Map one standalone Circuit's compiled stages back to its component records."""
    component._device_circuit = circuit
    declared = tuple(dict.fromkeys(outputs))
    published = {
        field: circuit.published[id(component), field] for field in declared
    }
    input_fields = {
        key: field
        for owner, field, key in circuit.copies
        if owner is component
    }
    stages = _component_output_stages(component)

    stage_by_name = {
        stage.method: stage for stage in circuit.output_stages or ()
    }

    def stage_plan(stage: OutputMethod) -> tuple[OutputMethod, ...]:
        """Return the Circuit stages required for ``stage`` in execution order."""
        ordered: list[OutputMethod] = []
        visited: set[str] = set()

        def visit(current: OutputMethod) -> None:
            if current.method in visited:
                return
            visited.add(current.method)
            for owner, method in circuit._output_dependencies.get(current.method, ()):
                if owner is not circuit or method not in stage_by_name:
                    raise RuntimeError("a standalone circuit has an external stage dependency")
                visit(stage_by_name[method])
            ordered.append(current)

        visit(stage)
        return tuple(ordered)

    def adapt(
        name: str,
        plan: Sequence[OutputMethod],
        assignments: Sequence[str] = (),
        results: Sequence[str] = (),
    ) -> Callable:
        """Bind already-compiled Circuit stages to the component records."""
        namespace: dict[str, Any] = {"_device_circuit": circuit}
        inputs = tuple(dict.fromkeys(
            field for stage in plan for field in stage.inputs
        ))
        body = [
            f"_device_circuit.input.{key} = self.input.{input_fields[key]}"
            for key in inputs
        ]
        values: tuple[str, ...] = ()
        for index, stage in enumerate(plan):
            method = getattr(circuit, stage.method)
            function = getattr(method, "__func__", method)
            program = getattr(function, "_peslite_inline_program", None)
            if isinstance(program, _InlineProgram):
                rename = {
                    key: f"_device_stage_{index}_{key}"
                    for key in program.namespace
                }
                namespace.update(
                    (rename[key], value) for key, value in program.namespace.items()
                )

                def rewrite(source: str) -> str:
                    source = re.sub(r"\bself\b", "_device_circuit", source)
                    for key in sorted(rename, key=len, reverse=True):
                        source = re.sub(
                            rf"\b{re.escape(key)}\b", rename[key], source
                        )
                    return source

                body.extend(rewrite(line) for line in program.statements)
                values = tuple(rewrite(value) for value in program.results)
            else:
                token = f"_device_stage_{index}"
                namespace[token] = method
                body.append(f"{token}(t)")
                values = ()
        body.extend(assignments)
        values = tuple(results) if results else values
        source = [f"    {line}" for line in body]
        if values:
            source.append(
                "    return (" + ", ".join(values)
                + ("," if len(values) == 1 else "") + ")"
            )
        elif not source:
            source.append("    pass")
        env = dict(namespace)
        exec(compile(
            f"def {name}(self, t):\n" + "\n".join(source) + "\n",
            "<peslite standalone netlist>", "exec",
        ), env)
        function = env[name]
        identifiers = set(_IDENTIFIERS.findall("\n".join((*body, *values))))
        function._peslite_inline_program = _InlineProgram(
            tuple(body), values, {
                key: value for key, value in namespace.items() if key in identifiers
            },
        )
        return function

    produced: dict[str, list[str]] = {}
    for field, key in published.items():
        produced.setdefault(circuit._output_producers[key], []).append(field)

    generated = []
    for index, (producer, fields) in enumerate(produced.items(), 1):
        stage = stage_by_name[producer]
        name = f"_netlist_outputs_{index}"
        assignments = tuple(
            f"self.output.{field} = _device_circuit.output.{published[field]}"
            for field in fields
        )
        plan = stage_plan(stage)
        function = adapt(name, plan, assignments)
        setattr(component, name, MethodType(function, component))
        generated.append(OutputMethod(
            name,
            tuple(input_fields[key] for key in dict.fromkeys(
                field for item in plan for field in item.inputs
            )),
            tuple(fields),
        ))

    kept, observers = [], []
    for stage, remaining in _component_stages_beside_netlist(component, stages, declared):
        rhs = stage.rhs or not remaining
        (kept if rhs else observers).append(OutputMethod(
            stage.method, tuple(stage.inputs), remaining, rhs=rhs
        ))
    component.output_stages = (*kept, *generated, *observers)

    entries = [
        (index, complex_state)
        for owner, _name, complex_state, index in circuit.layout
        if owner is component
    ]
    derivative_stage = next(
        (stage for stage in circuit.output_stages or () if stage.rhs), None
    )
    if entries and derivative_stage is None:
        raise RuntimeError("a stateful standalone circuit has no derivative stage")
    results = tuple(
        f"_device_circuit.derivatives[{index}] + "
        f"1j * _device_circuit.derivatives[{index + 1}]"
        if complex_state else f"_device_circuit.derivatives[{index}]"
        for index, complex_state in entries
    )
    if derivative_stage is None:
        def rhs(_self, _t):
            return ()
        rhs._peslite_inline_program = _InlineProgram((), (), {})
    else:
        rhs = adapt("rhs", stage_plan(derivative_stage), results=results)
    rhs._peslite_netlist_rhs = (circuit, tuple(entries))
    component.rhs = MethodType(rhs, component)

    _bind_energy(component, circuit)
    if circuit.commutation_active:
        component.commutation_margin = circuit.commutation_margin
        component.commutate = circuit.commutate
        component.commutation_active = True
        component.locate_commutations = circuit.locate_commutations


def bind_device(name: str, component: Any, declaration: NetworkPlan,
                names: Optional[dict[int, str]] = None) -> Circuit:
    """Compile one component directly from its declaration, preserving its Model identity."""
    circuit = Circuit("device")
    circuit._standalone_runtime = True
    circuit.names = {id(component): name} if names is None else names
    circuit.components = [component]
    circuit.declarations = {id(component): declaration}
    probe = declaration.builder
    output_fields = [field for owner, field, _quantity in probe.outputs if owner is component]
    output_fields.extend(
        field for owner, field, _port, _quantity, _factor in probe.port_outputs
        if owner is component
    )
    for port in _ports(component):
        if port.grounded:
            voltage_key = circuit.key(
                f"{name}.{port.name}.voltage", 0.0, output=True
            )
            current_key = circuit.key(
                f"{name}.{port.name}.current", port.current, output=True
            )
            circuit.port_values.append((port, None, voltage_key, current_key))
            continue
        group = _Group(f"{name}.{port.name}", [port])
        circuit.groups.append(group)
        voltage_key = circuit.key(
            f"{name}.{port.name}.voltage", port.voltage, output=True
        )
        current_key = circuit.key(
            f"{name}.{port.name}.current", port.current, output=True
        )
        circuit.port_values.append((port, group, voltage_key, current_key))
    circuit.standalone_ports = tuple(
        port for port in _ports(component) if port.standalone is not None
    )
    for field in _fields(component.input):
        key = circuit.key(f"{name}.{field}", getattr(component.input, field), output=False)
        circuit.copies.append((component, field, key))
        circuit.value_inputs[id(component.input), field] = key
    output_fields = list(dict.fromkeys(output_fields))
    for field in output_fields:
        circuit.published[id(component), field] = circuit.key(
            f"{name}.{field}", getattr(component.output, field), output=True
        )
    circuit.finalize()
    _bind_standalone(component, circuit, output_fields)
    return circuit


def _standalone_device(component: Any) -> Circuit:
    circuit = getattr(component, "_device_circuit", None)
    if circuit is not None:
        return circuit
    declaration = compile_network(component)
    if declaration is None:
        raise TypeError(f"{type(component).__name__} has no device netlist equations")
    return bind_device(type(component).__name__, component, declaration)


def prepare_device(component: Any) -> None:
    """Compile a standalone component before Model plans its output dependencies."""
    _standalone_device(component)


def device_rhs(component: Any, t: float):
    """Evaluate a standalone component's derivatives from its netlist declaration."""
    _standalone_device(component)
    return component.rhs(t)


def device_outputs(component: Any, t: float) -> None:
    """Evaluate a standalone component's output stages from its netlist declaration."""
    circuit = _standalone_device(component)
    circuit.release_topology()
    circuit._refresh_device_evaluator()
    for stage in component.output_stages or ():
        getattr(component, stage.method)(t)


def device_dissipated_power(component: Any) -> float:
    circuit = _standalone_device(component)
    return (circuit.dissipated_power_of(component)
            if id(component) in circuit._loss_elements else 0.0)


def device_supplied_power(component: Any) -> float:
    circuit = _standalone_device(component)
    return (circuit.supplied_power_of(component)
            if id(component) in circuit._supply_elements else 0.0)


def device_commutation_margin(component: Any) -> float:
    return _standalone_device(component).commutation_margin()


def device_commutate(component: Any, t: float) -> None:
    _standalone_device(component).commutate(t)


class _StandaloneRuntime:
    """The assembly implementation of the solver's standalone netlist protocol."""

    prepare = staticmethod(prepare_device)
    rhs = staticmethod(device_rhs)
    outputs = staticmethod(device_outputs)
    dissipated_power = staticmethod(device_dissipated_power)
    supplied_power = staticmethod(device_supplied_power)
    commutation_margin = staticmethod(device_commutation_margin)
    commutate = staticmethod(device_commutate)


STANDALONE_RUNTIME = _StandaloneRuntime()


def _find(parent: dict[int, int], a: int) -> int:
    parent.setdefault(a, a)
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _connect_behavioral_group(
    group: Sequence[Port], connections: dict, label: str,
) -> None:
    """Connect one node represented entirely by causal effort/flow signal ports."""
    sources: list[tuple[Port, str, str]] = []
    loads: list[tuple[Port, str, str]] = []
    for port in group:
        effort_record, _, effort_field = port.effort.partition(".")
        flow_record, _, flow_field = port.flow.partition(".")
        if (effort_record, flow_record) == ("output", "input"):
            sources.append((port, effort_field, flow_field))
        elif (effort_record, flow_record) == ("input", "output"):
            loads.append((port, effort_field, flow_field))
        else:
            raise ConfigError(
                f"{label}: a behavioral electrical port needs either "
                "input effort/output flow or output effort/input flow"
            )
    if len(sources) != 1:
        raise ConfigError(
            f"{label}: a behavioral electrical node needs exactly one effort source, "
            f"got {len(sources)}"
        )
    source, effort_field, flow_field = sources[0]

    def assign(target: tuple[Any, str], value: Any) -> None:
        if target in connections:
            raise ConfigError(f"{label}: behavioral port input {target[1]!r} is already driven")
        connections[target] = value

    for load, load_effort, _load_flow in loads:
        assign((load.component, load_effort), (source.component, effort_field))
    flows = [(load.component, load_flow, -1.0) for load, _effort, load_flow in loads]
    if flows:
        assign(
            (source.component, flow_field),
            flows[0] if len(flows) == 1 else flows,
        )


def lower(circuit_model: str, subsystems: Mapping[str, Any], connections: Mapping, zoh: Mapping,
          buses: Mapping[str, Sequence[Port]],
          declarations: Mapping[int, Optional[NetworkPlan]]
          ) -> tuple[dict, dict, dict, Optional[Circuit]]:
    """Rewrite an assembled connection graph for a netlist circuit model.

    ``buses``: the ports of every lowered bus. Returns the subsystems (components replaced by
    views, the Circuit appended), the connections, the zero-order-hold connections and the Circuit
    (``None``, and the graph unchanged, when no component has a netlist form).
    """
    names = {id(sub): name for name, sub in subsystems.items()}
    subs = list(subsystems.values())
    ports = {id(sub): _ports(sub) for sub in subs}
    bus_of: dict[int, str] = {}
    members: list[list[Port]] = []
    for bus, group in buses.items():
        connected = list(dict.fromkeys(group))
        if connected:
            members.append(connected)
        for port in group:
            previous = bus_of.setdefault(id(port), bus)
            if previous != bus:
                raise ConfigError(
                    f"port {type(port.component).__name__}.{port.name} is connected to "
                    f"both {previous!r} and {bus!r}"
                )
            owned = ports.setdefault(id(port.component), [])
            if port not in owned:
                owned.append(port)

    probes = {id(sub): declaration.builder for sub in subs
              if (declaration := declarations.get(id(sub))) is not None}
    circuit_members: list[list[Port]] = []
    direct_members: list[list[Port]] = []
    for group in members:
        (circuit_members if any(id(port.component) in probes for port in group)
         else direct_members).append(group)
    missing = sorted(
        names.get(id(port.component), type(port.component).__name__)
        for group in direct_members for port in group
        if id(port.component) not in probes and not port.behavioral
    )
    if missing:
        raise ConfigError(
            f"electrical components need netlist declarations: {sorted(set(missing))}"
        )
    direct_connections = dict(connections)
    for group in direct_members:
        first = group[0]
        label = next((bus_of[id(port)] for port in group if id(port) in bus_of), None) \
            or f"{names.get(id(first.component), type(first.component).__name__)}.{first.name}"
        _connect_behavioral_group(group, direct_connections, label)
    if not probes:  # nothing to assemble: the circuit model only selects the implicit linear algebra
        return dict(subsystems), direct_connections, dict(zoh), None
    circuit = Circuit(circuit_model)
    circuit.names = names
    circuit.components = [sub for sub in subs if id(sub) in probes]
    circuit.declarations = {id(sub): declarations[id(sub)] for sub in circuit.components}
    value_outputs = {
        token: binding
        for probe in probes.values()
        for token, binding in probe.value_outputs.items()
    }
    for token, (component, field) in value_outputs.items():
        circuit.value_inputs[token] = circuit.key(
            f"{names[id(component)]}.{field}", getattr(component.output, field), output=False
        )
    for sub in circuit.components:
        if hasattr(sub, "netlist_jumps"):
            sub.netlist_jumps = True  # its states jump by the netlist's projection, not by its own reset
    for group in circuit_members:
        first = group[0]
        label = next((bus_of[id(port)] for port in group if id(port) in bus_of), None) \
            or f"{names.get(id(first.component), type(first.component).__name__)}.{first.name}"
        node = _Group(label, group)
        circuit.groups.append(node)
        for port in group:
            owner = names.get(id(port.component), type(port.component).__name__)
            voltage_key = circuit.key(
                f"{owner}.{port.name}.voltage", port.voltage, output=True
            )
            current_key = circuit.key(
                f"{owner}.{port.name}.current", port.current, output=True
            )
            circuit.port_values.append((port, node, voltage_key, current_key))
    for sub in circuit.components:
        grounded = [port for port in ports[id(sub)] if port.grounded]
        if len(grounded) > 1:
            raise ConfigError(
                f"{names[id(sub)]}: a component cannot distinguish multiple grounded ports"
            )
        for port in grounded:
            owner = names[id(sub)]
            voltage_key = circuit.key(
                f"{owner}.{port.name}.voltage", 0.0, output=True
            )
            current_key = circuit.key(
                f"{owner}.{port.name}.current", port.current, output=True
            )
            circuit.port_values.append((port, None, voltage_key, current_key))
    views = {
        id(sub): NetworkView.of(
            sub,
            list(dict.fromkeys(field for c, field, *_rest in (
                *probes[id(sub)].outputs, *probes[id(sub)].port_outputs
            ) if c is sub and (id(c.output), field) not in value_outputs)),
            [field for component, field in value_outputs.values() if component is sub],
        )
        for sub in circuit.components
    }
    for sub in circuit.components:
        hook = probes[id(sub)].hooks.get(id(sub))
        if hook is not None and hook["margin"] is not None:
            views[id(sub)].expose(hook["margin"], hook["commutate"], hook["next_time"])
    for sub in circuit.components:
        declarations_out = (
            *((c, field) for c, field, _q in probes[id(sub)].outputs),
            *((c, field) for c, field, *_rest in probes[id(sub)].port_outputs),
        )
        for c, field in declarations_out:
            initial = getattr(c.output, field)
            circuit.published[id(c), field] = circuit.key(f"{names[id(c)]}.{field}", initial, output=True)
    def mapped(spec):
        def entry(source):
            component, field, *gain = source
            published = circuit.published.get((id(component), field))
            if published is not None:
                return circuit, published, *gain
            if id(component) in views:
                return views[id(component)], field, *gain
            return source

        entries = [entry(source) for source in (
            spec if isinstance(spec, list) else [spec]
        )]
        return entries if isinstance(spec, list) else entries[0]

    # ---------------------------------------------------------------- signal connections
    boundary: dict = {
        (circuit, circuit.value_inputs[token]): (views[id(component)], field)
        for token, (component, field) in value_outputs.items()
    }
    retargeted: dict = {}
    lowered: dict = {}
    for (dst, field), spec in direct_connections.items():
        if id(dst) in views:
            key = circuit.key(f"{names[id(dst)]}.{field}", getattr(dst.input, field), output=False)
            retargeted[(circuit, key)] = mapped(spec)
            circuit.copies.append((dst, field, key))
        else:
            lowered[(dst, field)] = mapped(spec)
    port_sources = {
        id(port): (voltage_key, current_key)
        for port, _group, voltage_key, current_key in circuit.port_values
    }
    group_of = {
        id(port): group
        for port, group, _voltage_key, _current_key in circuit.port_values
    }
    for group in circuit_members:
        for port in group:
            if id(port.component) in probes:
                continue
            effort_record, _, effort_field = port.effort.partition(".")
            flow_record, _, flow_field = port.flow.partition(".")
            owner = names.get(id(port.component), type(port.component).__name__)
            voltage_key, current_key = port_sources[id(port)]
            if (effort_record, flow_record) == ("input", "output"):
                source_field, target_field = flow_field, effort_field
                target_key, kind = voltage_key, "current"
            elif (effort_record, flow_record) == ("output", "input"):
                source_field, target_field = effort_field, flow_field
                target_key, kind = current_key, "voltage"
            else:
                raise ConfigError(
                    f"{owner}.{port.name}: a behavioral electrical port needs either "
                    "input effort/output flow or output effort/input flow"
                )
            key = circuit.key(
                f"{owner}.{port.name}.{kind}",
                getattr(port.component.output, source_field),
                output=False,
            )
            boundary[(circuit, key)] = (port.component, source_field)
            lowered[(port.component, target_field)] = (circuit, target_key)
            circuit.behavioral_sources.append((port, group_of[id(port)], key, kind))
    if any(
        spec_of(port.component).kind == "default"
        for group in members for port in group
        if id(port.component) not in probes
    ):
        circuit.observer = False
    for subsystem in subs:
        observed = getattr(subsystem, "port", None)
        source = port_sources.get(id(observed))
        if source is not None and hasattr(subsystem.input, "voltage") \
                and hasattr(subsystem.input, "current"):
            lowered[(subsystem, "voltage")] = (circuit, source[0])
            lowered[(subsystem, "current")] = (circuit, source[1])
    lowered = {**boundary, **retargeted, **lowered}

    # ---------------------------------------------------------------- held inputs
    held: dict = {}
    held_keys: dict[str, str] = {}
    for (dst, field), label in zoh.items():
        if id(dst) in views:
            held[(views[id(dst)], field)] = label
            if label not in held_keys:
                held_keys[label] = circuit.key(f"held.{label}", getattr(dst.input, field), output=False)
            circuit.copies.append((dst, field, held_keys[label]))
        else:
            held[(dst, field)] = label
    held.update(((circuit, key), label) for label, key in held_keys.items())
    circuit._held_input_keys = set(held_keys.values())

    out = {name: views.get(id(sub), sub) for name, sub in subsystems.items()}
    name, k = "circuit", 1
    while name in out:
        k += 1
        name = f"circuit_{k}"
    out[name] = circuit
    circuit.finalize()
    circuit._views = tuple(views.values())
    circuit._bind_stage_owners()
    for sub in circuit.components:
        views[id(sub)].bind(circuit, ())
    return out, lowered, held, circuit


class _CircuitPlan:
    """Static state mapping and exact-column metadata of one assembled circuit."""

    __slots__ = ("circuit", "local_to_model", "transform", "inverse", "states", "exact",
                 "skipped", "groups", "labels", "interface")

    def __init__(self, circuit: Circuit, local_to_model: np.ndarray, transform: np.ndarray,
                 inverse: np.ndarray, states: np.ndarray, exact: np.ndarray, skipped: np.ndarray,
                 groups: Sequence[set[int]], labels: Sequence[str], interface: set[int]) -> None:
        self.circuit = circuit
        self.local_to_model = local_to_model
        self.transform, self.inverse = transform, inverse
        self.states, self.exact, self.skipped = states, exact, skipped
        self.groups, self.labels, self.interface = tuple(groups), tuple(labels), frozenset(interface)


class _ModeDecomposition:
    """Exact local mode families of independently acting binary-device clusters."""

    __slots__ = (
        "clusters", "owners", "modes", "topologies", "variables", "reference", "structure"
    )

    def __init__(self, clusters: Sequence[Sequence[int]], owners: Sequence[Sequence[int]],
                 modes: Mapping[frozenset[int], Mode], topologies: Sequence[Topology],
                 variables: Sequence[tuple[str, int]], reference: Mode,
                 structure: "_ModeStructure") -> None:
        self.clusters = tuple(tuple(cluster) for cluster in clusters)
        self.owners = tuple(tuple(owner) for owner in owners)
        self.modes = dict(modes)
        self.topologies = tuple(topologies)
        self.variables = tuple(variables)
        self.reference = reference
        self.structure = structure

    def mode(self, changed: frozenset[int], ratios=None) -> Mode:
        """Return the reference mode with the selected binary positions reversed."""
        closed, conducting = list(self.reference.closed), list(self.reference.conducting)
        for variable in changed:
            field, index = self.variables[variable]
            values = closed if field == "closed" else conducting
            values[index] = not values[index]
        return Mode(closed, conducting, self.reference.ratios if ratios is None else ratios)


def circuit_structure(model: Any, circuits: Sequence[Circuit]):
    """Build the common state partition after any circuit formulation has been lowered."""
    from ..solver.implicit import Structure

    circuits = tuple(circuits)
    n = model.n_states
    component_subsystem = {
        id(getattr(subsystem, "equation_owner", subsystem)): subsystem
        for subsystem in model.subsystems
    }
    all_groups: list[set[int]] = []
    all_labels: list[str] = []
    all_interface: set[int] = set()
    for circuit in circuits:
        local_to_model = np.empty(circuit.n_states, dtype=np.intp)
        for component, name, complex_state, local in circuit.layout:
            subsystem = component_subsystem[id(component)]
            positions = tuple(range(*model.state_slice(subsystem, name).indices(n)))
            if len(positions) != (2 if complex_state else 1):
                raise RuntimeError("circuit and model state layouts differ")
            local_to_model[local:local + len(positions)] = positions

        def mapped_states(elements) -> set[int]:
            return {
                int(local_to_model[int(local)])
                for element in elements
                for local in np.asarray(
                    getattr(element, "state", ()), dtype=np.intp
                ).reshape(-1)
                if 0 <= int(local) < local_to_model.size
            }

        grouping = circuit.netlist.grouping()
        groups = [mapped_states(elements) for elements in grouping.elements]
        labels = [", ".join(dict.fromkeys(
            circuit._owners.get(id(element), "?") for element in elements
        )) for elements in grouping.elements]
        interface = mapped_states(grouping.interface_elements)
        interface_nodes = {id(node) for node in grouping.interface}
        interface.update(mapped_states(
            element for element in circuit.netlist.elements
            if any(id(node) in interface_nodes
                   for port in element.ports for node in port if node is not None)
        ))

        local = circuit.n_states
        transform = np.eye(local)
        inverse = np.eye(local)
        for index, matrix, inv in circuit._transforms:
            transform[np.ix_(index, index)] = matrix
            inverse[np.ix_(index, index)] = inv
        unsafe = {
            int(index)
            for value, _offset, _dimension, _name in circuit.netlist._source_slots
            for component, name in getattr(value, "state_dependencies", ())
            for index in circuit._state_index[id(component), name]
        }
        for index, _matrix, _inverse in circuit._transforms:
            if unsafe.intersection(map(int, index)):
                unsafe.update(map(int, index))
        exact = local_to_model[[index for index in range(local) if index not in unsafe]]
        members = {
            id(component_subsystem[id(component)]): component_subsystem[id(component)]
            for component in circuit.components if id(component) in component_subsystem
        }
        if id(circuit) in component_subsystem:
            members[id(circuit)] = component_subsystem[id(circuit)]
        skipped = (np.zeros(0, dtype=np.intp)
                   if model.has_external_state_dependency(tuple(members.values())) else exact)
        circuit._structure_plan = _CircuitPlan(
            circuit, local_to_model, transform, inverse,
            np.asarray(circuit.netlist.states, dtype=np.intp), exact, skipped,
            groups, labels, interface,
        )
        all_groups.extend(groups)
        all_labels.extend(labels)
        all_interface.update(interface)

    occurrences: dict[int, int] = {}
    for group in all_groups:
        for position in group:
            occurrences[position] = occurrences.get(position, 0) + 1
    all_interface.update(position for position, count in occurrences.items() if count > 1)
    grouped = set().union(*all_groups) if all_groups else set()
    all_interface.update(set(range(n)) - grouped)
    groups, labels = [], []
    for group, label in zip(all_groups, all_labels):
        interior = np.array(sorted(group - all_interface), dtype=np.intp)
        if interior.size:
            groups.append(interior)
            labels.append(label)
    return Structure(groups, np.array(sorted(all_interface), dtype=np.intp), labels)


def circuit_mode_key(circuits: Sequence[Circuit]):
    """Current discrete circuit identity, excluding continuously varying ratios."""
    circuits = tuple(circuits)
    if not circuits:
        return None
    return tuple(
        (id(circuit.netlist), (
            circuit._bound_device_topology() or circuit.netlist.topology()
        ).mode.key[:2])
        for circuit in circuits
    )

def circuit_linear_blocks(circuits: Sequence[Circuit]):
    """Return present exact circuit matrices in global model coordinates."""
    blocks = []
    for circuit in circuits:
        plan = circuit._structure_plan
        if plan is None:
            raise RuntimeError("circuit_structure must run before linear block evaluation")
        blocks.append((plan.local_to_model,
                       _placed_linear_matrix(circuit, circuit.netlist.topology().state_space.A),
                       plan.exact, plan.skipped))
    return tuple(blocks)


def _placed_linear_matrix(circuit: Circuit, matrix: np.ndarray) -> np.ndarray:
    """Place a netlist or full-circuit matrix in the assembled circuit state coordinates."""
    plan = circuit._structure_plan
    if plan is None:
        raise RuntimeError("circuit_structure must run before linear matrix evaluation")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape == (circuit.n_states, circuit.n_states):
        placed = matrix
    elif matrix.shape == (plan.states.size, plan.states.size):
        placed = np.zeros((circuit.n_states, circuit.n_states))
        placed[np.ix_(plan.states, plan.states)] = matrix
    else:
        raise ValueError("a circuit linear matrix does not match its state layout")
    return plan.inverse @ placed @ plan.transform


def _placed_certificate(circuit: Circuit, certificate: Optional[tuple[object, float]]):
    # Similarity preserves the spectrum, so the netlist bound also certifies the placed matrix.
    return None if certificate is None else ((id(circuit.netlist), certificate[0]), certificate[1])


def circuit_linear_modes(circuits: Sequence[Circuit]):
    """Return each assembled circuit's present-mode identity and the builder of its exact state
    matrix (a solver builds it only for a mode it has not seen)."""
    return tuple(
        ((id(circuit.netlist), key),
         lambda circuit=circuit, matrix=matrix: _placed_linear_matrix(circuit, matrix()),
         _placed_certificate(circuit, certificate))
        for circuit in circuits
        for key, matrix, certificate in (circuit.linear_mode(),)
    )


def circuit_linear_certificates(circuits: Sequence[Circuit]):
    """Return present-mode certificates without materializing identities or matrices."""
    return tuple(_placed_certificate(circuit, circuit.stability_certificate())
                 for circuit in circuits)


_VOLTAGE_ROLES = ("vsrc", "short", "cap")
_CURRENT_ROLES = ("ind", "isrc", "state_current")


def _binary_role(resistance: Any) -> Optional[str]:
    """Return the branch role of one binary-device resistance."""
    return "short" if resistance == 0.0 else None if resistance == math.inf else "res"


class _ModeStructure:
    """Independent binary-device clusters and their affected rows from connections alone."""

    def __init__(self, net: Netlist, variables: Sequence[tuple[str, int]],
                 reference: Mode) -> None:
        branches, n_vertices = net._branches, net.n_vertices
        self.variables = tuple(variables)
        self.devices = tuple(
            net._switches[index] if field == "closed" else net._diodes[index]
            for field, index in self.variables
        )
        roles, _resistance = _roles(net, reference)
        device_branch = {
            variable: net._ports[id(device)][0]
            for variable, device in enumerate(self.devices)
        }
        free_branch = {branch: variable for variable, branch in device_branch.items()}
        positions = {
            variable: (_binary_role(device.R_on), _binary_role(device.R_off))
            for variable, device in enumerate(self.devices)
        }
        self.roles = roles

        supernode = _UnionFind(n_vertices)
        for index, branch in enumerate(branches):
            if index not in free_branch and roles[index] in _VOLTAGE_ROLES:
                supernode.union(branch.a, branch.b)
        ground = supernode.find(0)
        self.supernode, self.ground = supernode, ground

        edges: list[tuple[set[int], Optional[int], list[int]]] = []
        transformers: dict[int, tuple[set[int], list[int]]] = {}
        for index, branch in enumerate(branches):
            if index in free_branch:
                edges.append(({branch.a, branch.b}, free_branch[index], [index]))
            elif roles[index] == "res":
                edges.append(({branch.a, branch.b}, None, [index]))
            elif roles[index] in ("p", "s"):
                vertices, entries = transformers.setdefault(id(branch.element), (set(), []))
                vertices.update((branch.a, branch.b))
                entries.append(index)
        edges.extend((vertices, None, entries) for vertices, entries in transformers.values())

        part = _UnionFind(n_vertices + len(edges))
        edge_part = []
        for index, (vertices, _device, _branches) in enumerate(edges):
            inner = sorted({supernode.find(vertex) for vertex in vertices} - {ground})
            for other in inner[1:]:
                part.union(inner[0], other)
            edge_part.append(inner[0] if inner else n_vertices + index)
        self.part, self.edges, self.edge_part = part, edges, edge_part

        self.voltage_branches: dict[int, list[int]] = {}
        for index, branch in enumerate(branches):
            if index not in free_branch and roles[index] in _VOLTAGE_ROLES:
                self.voltage_branches.setdefault(supernode.find(branch.a), []).append(index)
        self.branches = branches

        joined = _UnionFind(n_vertices)
        for index, branch in enumerate(branches):
            if index in free_branch:
                if positions[free_branch[index]][1] is not None:
                    joined.union(branch.a, branch.b)
            elif roles[index] in _VOLTAGE_ROLES or roles[index] == "res":
                joined.union(branch.a, branch.b)
        separating: dict[int, set[int]] = {}
        for variable, index in device_branch.items():
            a, b = joined.find(branches[index].a), joined.find(branches[index].b)
            if positions[variable][1] is None and a != b:
                separating.setdefault(a, set()).add(variable)
                separating.setdefault(b, set()).add(variable)
        self.cutsets: dict[int, list[int]] = {}
        grounded = joined.find(0)
        for side, devices in separating.items():
            if side == grounded:
                continue
            crossing = [
                index for index, branch in enumerate(branches)
                if (roles[index] in _CURRENT_ROLES or roles[index] in ("p", "s"))
                and (joined.find(branch.a) == side) != (joined.find(branch.b) == side)
            ]
            parts = {part.find(edge_part[self._device_edge(variable)]) for variable in devices}
            for index in crossing:
                parts.update(self._vertex_part(vertex)
                             for vertex in (branches[index].a, branches[index].b))
            parts.discard(None)
            first, *others = sorted(parts)
            for other in others:
                part.union(first, other)
            self.cutsets.setdefault(first, []).extend(crossing)
        self.positions = positions

        while True:
            short_ends: dict[int, set[int]] = {}
            for index, (vertices, variable, _branches) in enumerate(edges):
                if variable is not None and positions[variable][0] == "short":
                    short_ends.setdefault(part.find(edge_part[index]), set()).update(
                        vertex for vertex in vertices if supernode.find(vertex) == ground
                    )
            joined_any = False
            for root, ends in short_ends.items():
                loop_nodes = {
                    vertex for branch in self._paths(ground, ends)
                    for vertex in (branches[branch].a, branches[branch].b)
                } - {0}
                for index, (vertices, _variable, _branches) in enumerate(edges):
                    if vertices & loop_nodes and part.union(part.find(edge_part[index]), root):
                        joined_any = True
            if not joined_any:
                break

        roots = sorted(
            {part.find(edge_part[self._device_edge(variable)])
             for variable in range(len(self.variables))},
            key=lambda root: min(
                variable for variable in range(len(self.variables))
                if part.find(edge_part[self._device_edge(variable)]) == root
            ),
        )
        self.index = {root: number for number, root in enumerate(roots)}
        self.clusters = [
            [variable for variable in range(len(self.variables))
             if part.find(edge_part[self._device_edge(variable)]) == root]
            for root in roots
        ]

    def _device_edge(self, variable: int) -> int:
        return next(
            index for index, (_vertices, owner, _branches) in enumerate(self.edges)
            if owner == variable
        )

    def _vertex_part(self, vertex: int) -> Optional[int]:
        node = self.supernode.find(vertex)
        return None if node == self.ground else self.part.find(node)

    def _cluster_of_part(self, root: Optional[int]) -> list[int]:
        if root is None:
            return []
        number = self.index.get(self.part.find(root))
        return [] if number is None else [number]

    def _paths(self, node: int, ends: set[int]) -> set[int]:
        """Voltage-fixing branches on paths between attachment vertices."""
        branches = self.voltage_branches.get(node, [])
        if len(ends) < 2 or not branches:
            return set()
        adjacent: dict[int, list[tuple[int, int]]] = {}
        for index in branches:
            branch = self.branches[index]
            adjacent.setdefault(branch.a, []).append((branch.b, index))
            adjacent.setdefault(branch.b, []).append((branch.a, index))
        if len(branches) >= len(adjacent):
            return set(branches)
        start = next(iter(ends))
        parent: dict[int, tuple[int, int]] = {start: (-1, -1)}
        order = [start]
        for vertex in order:
            for other, index in adjacent.get(vertex, ()):
                if other not in parent:
                    parent[other] = vertex, index
                    order.append(other)
        count = {vertex: int(vertex in ends) for vertex in order}
        total = sum(count.values())
        selected = set()
        for vertex in reversed(order[1:]):
            above, index = parent[vertex]
            if 0 < count[vertex] < total:
                selected.add(index)
            count[above] += count[vertex]
        return selected

    def _attachments(self) -> dict[int, dict[int, set[int]]]:
        attached: dict[int, dict[int, set[int]]] = {}
        for index, (vertices, _variable, _branches) in enumerate(self.edges):
            number = self.index.get(self.part.find(self.edge_part[index]))
            if number is None:
                continue
            for vertex in vertices:
                attached.setdefault(number, {}).setdefault(
                    self.supernode.find(vertex), set()
                ).add(vertex)
        for root, crossing in self.cutsets.items():
            number = self.index.get(self.part.find(root))
            if number is None:
                continue
            for index in crossing:
                branch = self.branches[index]
                for vertex in (branch.a, branch.b):
                    attached.setdefault(number, {}).setdefault(
                        self.supernode.find(vertex), set()
                    ).add(vertex)
        return attached

    def _short_loops(self) -> dict[int, dict[int, set[int]]]:
        """Return voltage-defined paths closed by each short-device cluster."""
        ends: dict[int, dict[int, set[int]]] = {}
        for index, (vertices, variable, _branches) in enumerate(self.edges):
            if variable is None or self.positions[variable][0] != "short":
                continue
            cluster = self.index[self.part.find(self.edge_part[index])]
            for vertex in vertices:
                ends.setdefault(cluster, {}).setdefault(
                    self.supernode.find(vertex), set()
                ).add(vertex)
        return {
            cluster: {
                node: self._paths(node, attached)
                for node, attached in by_node.items()
            }
            for cluster, by_node in ends.items()
        }

    def _constrained(self) -> dict[int, set[int]]:
        constrained: dict[int, set[int]] = {}
        for root, crossing in self.cutsets.items():
            number = self.index.get(self.part.find(root))
            if number is not None:
                constrained.setdefault(number, set()).update(crossing)
        return constrained

    def _element_branches(self, net: Netlist, index: int) -> list[int]:
        element = self.branches[index].element
        return list(net._ports[id(element)]) if isinstance(element, CoupledInductors) else [index]

    def equation_owners(self, net: Netlist) -> list[list[int]]:
        """Return the clusters that can change each derivative and output row."""
        branches, roles = self.branches, self.roles
        attached, constrained = self._attachments(), self._constrained()
        voltage_owners: dict[int, set[int]] = {}
        for number, by_node in attached.items():
            for node, ends in by_node.items():
                for index in self._paths(node, ends):
                    voltage_owners.setdefault(index, set()).add(number)
        edge_owners: dict[int, set[int]] = {}
        for index, (_vertices, _variable, entries) in enumerate(self.edges):
            number = self.index.get(self.part.find(self.edge_part[index]))
            if number is not None:
                for entry in entries:
                    edge_owners.setdefault(entry, set()).add(number)

        clamped: dict[int, set[int]] = {}
        for cluster, loops in self._short_loops().items():
            for node, path in loops.items():
                if path:
                    clamped.setdefault(node, set()).add(cluster)

        def node_owners(vertex: int) -> set[int]:
            if vertex == 0:
                return set()
            return clamped.get(self.supernode.find(vertex), set()) | set(
                self._cluster_of_part(self._vertex_part(vertex))
            )

        def voltage_owners_between(a: int, b: int) -> set[int]:
            """Return clusters that can change the voltage between two vertices."""
            if self.supernode.find(a) == self.supernode.find(b):
                return set(clamped.get(self.supernode.find(a), ()))
            return node_owners(a) | node_owners(b)

        def current_owners(index: int) -> set[int]:
            owners: set[int] = set()
            for entry in self._element_branches(net, index):
                branch = branches[entry]
                owners |= voltage_owners_between(branch.a, branch.b)
                owners |= {
                    number for number, entries in constrained.items() if entry in entries
                }
            return owners

        rows: list[set[int]] = [set() for _ in range(net.n_x + net.n_out)]
        binary_branches = {
            entry for _vertices, variable, entries in self.edges
            if variable is not None for entry in entries
        }
        for index, branch in enumerate(branches):
            role = roles[index]
            if index in binary_branches or role in ("res", "p", "s"):
                owners = edge_owners.get(index, set())
            elif role in _VOLTAGE_ROLES:
                owners = voltage_owners.get(index, set())
            elif role in _CURRENT_ROLES:
                owners = current_owners(index)
            else:
                owners = set()
            if branch.x >= 0:
                for offset in range(branch.dim):
                    rows[branch.x + offset] |= owners
            voltage = voltage_owners_between(branch.a, branch.b)
            for offset in range(branch.dim):
                rows[net.n_x + net.n_e + branch.output + offset] |= voltage
                rows[net.n_x + net.n_e + net.n_bd + branch.output + offset] |= owners
        for vertex in range(1, net.n_vertices):
            for offset in range(net._vertex_dim[vertex]):
                rows[net.n_x + int(net._e_off[vertex]) + offset] |= node_owners(vertex)
        return [sorted(owners) for owners in rows]

    def jump_owners(self, net: Netlist, size: int) -> list[list[int]]:
        """Return the clusters that can change projected states and jump energies."""
        branches = self.branches
        entries: list[set[int]] = [set() for _ in range(size + 2)]
        changed: set[int] = set()
        for number, loops in self._short_loops().items():
            for path in loops.values():
                for index in path:
                    if branches[index].x >= 0:
                        for offset in range(branches[index].dim):
                            entries[int(net.states[branches[index].x + offset])].add(number)
                        changed.add(number)
        for number, constrained in self._constrained().items():
            for index in constrained:
                for entry in self._element_branches(net, index):
                    if branches[entry].x >= 0:
                        for offset in range(branches[entry].dim):
                            entries[int(net.states[branches[entry].x + offset])].add(number)
                        changed.add(number)
        entries[size] |= changed
        entries[size + 1] |= changed
        return [sorted(owners) for owners in entries]


def _mode_decomposition(
    circuit: Circuit,
    variables: Sequence[tuple[str, int]],
    reference: Mode,
) -> Optional[_ModeDecomposition]:
    """Find independent binary-device families from the circuit connections."""
    variables = tuple((str(field), int(index)) for field, index in variables)
    if not variables:
        return None
    net = circuit.netlist
    actual = tuple(
        np.asarray(ratio, dtype=float)
        for ratio in (net._ratio_override if net._ratio_override is not None
                      else tuple(transformer.ratio() for transformer in net._transformers))
    )
    ratio_key = tuple((ratio.shape, ratio.tobytes()) for ratio in actual)
    key = variables, reference.closed, reference.conducting, ratio_key
    cache = circuit._mode_decompositions
    if key in cache:
        return cache[key]
    base = Mode(reference.closed, reference.conducting, actual)

    def combinations(cluster: Sequence[int]):
        for bits in range(1, 1 << len(cluster)):
            yield frozenset(
                variable for position, variable in enumerate(cluster) if bits >> position & 1
            )

    try:
        structure = _ModeStructure(net, variables, base)
        modes: dict[frozenset[int], Mode] = {frozenset(): base}
        decomposition = _ModeDecomposition(
            structure.clusters, structure.equation_owners(net), modes, (),
            variables, base, structure,
        )
        for cluster in structure.clusters:
            decomposition.modes.update(
                (changed, decomposition.mode(changed)) for changed in combinations(cluster)
            )
        decomposition.topologies = tuple(
            net._compiled(mode) for mode in decomposition.modes.values()
        )
    except (_Degenerate, ValueError, np.linalg.LinAlgError):
        cache[key] = None
        return None
    cache[key] = decomposition
    return decomposition
