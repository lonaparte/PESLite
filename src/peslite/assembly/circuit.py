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
from collections import OrderedDict
from types import MethodType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ..components.network import Port
from ..solver.energy import Energetic, spec_of
from ..solver.model import Bag, ConfigError, Empty, OutputMethod, _InlineProgram
from ..solver.netlist import (QUANTITIES, CurrentSource, Diode, Element, Netlist, Node, Topology,
                              VoltageSource, _Degenerate, _Evaluation, _linear_expression, _numeric_values)

__all__ = ["NetlistBuilder", "NetworkPlan", "Circuit", "NetworkView", "lower", "lower_devices",
           "bind_device", "compile_network", "circuit_structure"]


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
        self.local = False
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

    def local_model(self) -> None:
        """Compile this signal-controlled declaration locally instead of merging its nodes."""
        self.local = True


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
    if declare(probe) is NotImplemented:
        return None
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
        self._device_stage_pending = False
        self._standalone_runtime = False
        self._state_diode_groups: tuple[
            tuple[int, tuple[tuple[int, float], ...]], ...
        ] = ()
        self._device_bindings: dict[tuple, tuple[Topology, Callable, Callable]] = {}
        self._mode_eigenvalues: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._explicit_stability: "OrderedDict[tuple, tuple[float, float]]" = OrderedDict()
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
        rebuilding = self.netlist is not None
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
        old, self.netlist = self.netlist, net
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
        self._mode_eigenvalues.clear()
        self._explicit_stability.clear()
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
        self._device_stage_pending = False
        if self.form == "device":
            (self.scalar_derivatives, self.scalar_outputs,
             self.scalar_output_evaluator) = net.scalar_evaluators()
            compiled = (
                self._compile_equation_stages(defer_dynamic=not rebuilding)
                or (not self._standalone_runtime
                    and self._compile_device_runtime(lazy=self._device_stage_pending))
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
        lines = []
        for key, dimension, terms in publications:
            parts = [
                " + ".join(
                    f"{factor!r} * o[{position[output.start + k]}]"
                    for factor, output in terms
                ) or "0.0"
                for k in range(dimension)
            ]
            expression = (f"({parts[0]}) + 1j * ({parts[1]})" if dimension == 2 else parts[0])
            lines.append(f"output.{key} = {expression}")
        lines.extend(f"output.{output} = input.{input_}" for input_, output in self.passthrough)
        env = {"input": self.input, "output": self.output}
        for index, (port, _group, voltage_key, current_key) in enumerate(self.port_values):
            env[f"_port_{index}"] = port
            lines.extend((
                f"_port_{index}.voltage = output.{voltage_key}",
                f"_port_{index}.current = output.{current_key}",
            ))
        components = {id(component): component for component in self.components}
        for index, ((owner, field), key) in enumerate(self.published.items()):
            component = components.get(owner)
            if component is None:
                continue
            env[f"_output_{index}"] = component.output
            lines.append(f"_output_{index}.{field} = output.{key}")
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
        """Assign aggregated derivative stages to every component state view they advance."""
        owners = (self, *self._views)
        self._output_stage_owners = {
            stage.method: owners for stage in self.output_stages or () if stage.rhs
        }

    def _program_dependencies(self, program: tuple) -> tuple[str, ...]:
        """Circuit input fields read by one generated scalar program."""
        statements, _result, namespace, expressions = program
        expressions = expressions or ()
        active = "\n".join((*_live_statements(statements, expressions), *expressions))
        copied = {
            (id(component.input), field): key
            for component, field, key in self.copies
        }
        dependencies = set()
        for name, value in namespace.items():
            if re.search(rf"\b{re.escape(name)}\b", active) is None:
                continue
            declared = _record_dependencies(value)
            if declared is None or declared[0] is None:
                continue
            record, fields = declared
            dependencies.update(
                key for field in fields
                if (key := (field if record is self.input else
                            copied.get((id(record), field)))) is not None
            )
        return tuple(name for name in self.inputs if name in dependencies)

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
        self, name: str, body: Sequence[str], namespace: Mapping[str, Any]
    ) -> Callable:
        """Compile one method and expose the same straight-line body to the Model compiler."""
        env = dict(namespace)
        source = f"def {name}(self, t):\n" + "\n".join(
            "    " + line for line in body or ("pass",)
        ) + "\n"
        exec(compile(source, "<peslite direct circuit>", "exec"), env)
        function = env[name]
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", "\n".join(body)))
        function._peslite_inline_program = _InlineProgram(
            tuple(body), (), {
                key: value for key, value in namespace.items() if key in identifiers
            }
        )
        method = MethodType(function, self)
        setattr(self, name, method)
        return method

    def _append_publications(
        self, keys: Sequence[str], values: Mapping[int, str], body: list[str],
        namespace: dict[str, Any], label: str,
    ) -> None:
        """Append direct circuit-output and physical-port writes for scalar netlist rows."""
        publications = {key: (dimension, terms)
                        for key, dimension, terms in self.publications}
        port_writes: dict[str, list[tuple[Port, str]]] = {}
        for port, _group, voltage_key, current_key in self.port_values:
            port_writes.setdefault(voltage_key, []).append((port, "voltage"))
            port_writes.setdefault(current_key, []).append((port, "current"))

        def scaled(factor: float, expression: str) -> str:
            return f"({expression})" if factor == 1.0 else f"{factor!r} * ({expression})"

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
            body.append(f"self.output.{key} = {expression}")
            for port, attribute in port_writes.get(key, ()):
                token = f"_circuit_port_{label}_{len(namespace)}"
                namespace[token] = port
                body.append(f"{token}.{attribute} = self.output.{key}")

    def _stage_programs(self) -> tuple[Topology, bool, Callable, Callable]:
        """Scalar equation programs shared by evaluation and dependency planning."""
        net = self.netlist
        topology = net.topology()
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

    def _output_stage_groups(
        self, output_evaluator: Callable,
        publications: Mapping[str, tuple[int, Sequence[tuple[float, slice]]]],
    ) -> Optional[list[tuple[tuple[str, ...], list[str], list[int]]]]:
        """Group public rows by the circuit inputs their equations read."""
        groups: dict[tuple[str, ...], tuple[list[str], set[int]]] = {}
        for key in publications:
            keys = [key]
            selected = sorted({
                index for key in keys for _factor, output in publications[key][1]
                for index in range(output.start, output.stop)
            })
            program = getattr(output_evaluator(selected), "_peslite_scalar_program", None)
            if program is None or program[3] is None:
                return None
            dependencies = self._program_dependencies(program)
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

    def _compile_equation_stages(self, *, defer_dynamic: bool = False) -> bool:
        """Compile dependency-precise stages for every circuit formulation."""
        net = self.netlist
        if self._mixtures:
            return False
        numeric = self.form != "device"
        dynamic = any(transformer.constant is None for transformer in net._transformers)
        if not numeric:
            fixed = not (net._switches or net._diodes or dynamic)
            boundary_compiled = dynamic and not net._diodes and not self._discrete_mode_targets
            if not (fixed or boundary_compiled):
                return False
            if defer_dynamic and dynamic:
                self._device_stage_pending = True
                return False
        topology, dynamic, derivatives, output_evaluator = self._stage_programs()
        publications = {key: (dimension, terms)
                        for key, dimension, terms in self.publications}
        if set(publications) | {output for _input, output in self.passthrough} != set(self.outputs):
            return False
        output_groups = self._output_stage_groups(output_evaluator, publications)
        derivative_program = getattr(derivatives, "_peslite_scalar_program", None)
        if (output_groups is None or (self.n_states and
                                      (derivative_program is None or derivative_program[3] is None))):
            return False
        derivative_dependencies = (
            self._program_dependencies(derivative_program) if self.n_states else ()
        )
        signature = (
            tuple((dependencies, tuple(keys), tuple(selected))
                  for dependencies, keys, selected in output_groups),
            derivative_dependencies,
            tuple(self.passthrough),
        )
        if numeric and self._numeric_stages and signature == self._stage_signature:
            self._direct_mode_key = topology.mode.key[:2]
            return True

        prefix = "numeric" if numeric else "device"
        stages: list[OutputMethod] = []
        producers: dict[str, str] = {}
        for index, (dependencies, keys, selected) in enumerate(output_groups, 1):
            name = f"_{prefix}_outputs_{index}"
            positions = {position: offset for offset, position in enumerate(selected)}
            program = output_evaluator(selected)._peslite_scalar_program
            body, expressions, namespace = self._bind_program(program, dependencies)
            values = {
                position: expressions[offset]
                for position, offset in positions.items()
            }
            self._append_publications(
                keys, values, body, namespace, f"{prefix}_{index}",
            )
            self._generated_method(name, body, namespace)
            stages.append(OutputMethod(name, dependencies, tuple(keys), rhs=False))
            producers.update(dict.fromkeys(keys, name))

        self._append_passthrough_stage(stages, producers, prefix)
        if self.n_states:
            dependencies = derivative_dependencies
            if numeric:
                namespace = {}
                body = self._copy_prelude(dependencies, namespace)
                namespace["_circuit_numeric_derivatives"] = (
                    lambda t: net.evaluate(t, self._states())
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
            self._generated_method(name, body, namespace)
            stages.append(OutputMethod(name, dependencies, (), rhs=True))

        self.output_stages = tuple(stages)
        self._output_dependencies = {}
        self._output_producers = producers
        self._bind_stage_owners()
        self._direct_dynamic = dynamic
        self._direct_mode_key = topology.mode.key[:2] if numeric or dynamic else None
        self._numeric_stages = numeric
        self._stage_signature = signature
        return True

    def _refresh_device_evaluator(self, topology: Optional[Topology] = None) -> bool:
        """Select the compiled scalar evaluator for the current structural mode."""
        holder = getattr(self, "_device_runtime", None)
        if holder is None:
            return False
        self.netlist._last = None
        topology = self.netlist.topology() if topology is None else topology
        margin_key = self.netlist.scalar_margin_key(topology)
        if (holder[1] is topology and holder[2] is not None
                and getattr(self, "_margin_evaluator_key", None) == margin_key):
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
        return True

    def _compile_device_runtime(self, *, lazy: bool = False) -> bool:
        """Bind the current compiled mode; mode selection remains at physical boundaries."""
        if self._mixtures:
            return False
        holder = getattr(self, "_device_runtime", None)
        if holder is not None:
            self._refresh_device_evaluator()
            self._use_runtime_stages()
            return True

        self._device_runtime = [self.netlist, None, None]
        if lazy:
            holder = self._device_runtime

            def evaluate(t, y):
                topology = self.netlist.topology()
                compiled = self.netlist._dynamic_scalar_function(topology)
                margin = self.netlist.scalar_margin_evaluator(topology)
                holder[:] = self.netlist, topology, compiled
                self._margin_evaluator_key = self.netlist.scalar_margin_key(topology)
                self._margin_evaluator = margin
                return compiled(t, y)

            holder[2] = evaluate
        else:
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

    def cpp_outputs(self, generator: Any) -> list[str]:
        """Lower the same netlist equations through the standalone C++ backend."""
        return generator.trace_circuit_outputs(self)

    def cpp_output_stage(self, generator: Any, method: Any) -> list[str]:
        """Publish the outputs declared by the same dependency stage used in Python."""
        fields = next(stage.outputs for stage in self.output_stages
                      if stage.method == method.__name__)
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
        runtime = getattr(self, "_device_runtime", None)
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
        if self._numeric_stages:
            selected = self.netlist.topology() if topology is None else topology
            if selected.mode.key[:2] != self._direct_mode_key:
                previous = self._stage_signature
                self._compile_equation_stages()
                if self._stage_signature != previous:
                    for view in self._views:
                        view.update_circuit_dependencies(self)
                    self.plan_changed = True
            return
        if self._device_stage_pending:
            self._device_stage_pending = False
            if self._compile_equation_stages():
                self.netlist._last = None
                for view in self._views:
                    view.update_circuit_dependencies(self)
                self.plan_changed = True
                return
        if self._direct_dynamic:
            self.netlist._last = None
            selected = self.netlist.topology() if topology is None else topology
            if selected.mode.key[:2] == self._direct_mode_key:
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
        self._loss = self._supply = None
        self.state, self.input, self.output = component.state, component.input, component.output
        self.state_names = tuple(component.state_names)
        stages = getattr(component, "output_stages", None)
        if stages is None and (component.outputs_need_inputs or component.output is not component.state):
            stages = (OutputMethod("set_outputs", _fields(component.input) if component.outputs_need_inputs else (),
                                   _fields(component.output)),)
        kept = []
        for stage in stages or ():
            replaced = set(stage.outputs) & set(outputs)
            if set(stage.outputs) - replaced:
                # Preserve outputs not supplied by the netlist. The netlist stage publishes its
                # authoritative fields before these derived outputs are evaluated.
                remaining = tuple(field for field in stage.outputs if field not in replaced)
                kept.append(OutputMethod(
                    stage.method, tuple(stage.inputs), remaining,
                    rhs=bool(set(remaining) & set(inputs)),
                ))
                setattr(self, stage.method, getattr(component, stage.method))
        self.declared = tuple(outputs)
        self.output_stages = tuple(kept) if kept or outputs else None
        self._output_dependencies: dict[str, tuple[tuple[Any, str], ...]] = {}
        self._derived_output_methods: tuple[str, ...] = ()
        self.outputs_need_inputs = component.outputs_need_inputs if stages is None else True

    def bind(self, circuit: Circuit, fed: Sequence[str]) -> None:
        """Return ``circuit``'s derivatives; ``fed``: the inputs the circuit writes (ordering)."""
        self._circuit = circuit
        if id(self.component) in circuit._loss_elements:
            self._loss = lambda: circuit.dissipated_power_of(self.component)
        if id(self.component) in circuit._supply_elements:
            self._supply = lambda: circuit.supplied_power_of(self.component)
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


class _DeviceRuntime:
    """Bind one component's runtime equations to its independently compiled netlist.

    The component remains the Model subsystem.  Its ordinary input connections, state record,
    event ownership and public identity therefore stay unchanged; only the electrical output and
    derivative methods are generated from the netlist declaration.
    """

    def __init__(self, component: Any, circuit: Circuit, outputs: Sequence[str]) -> None:
        self.component, self.circuit = component, circuit
        component._device_circuit = circuit
        declared = tuple(dict.fromkeys(outputs))

        stages = getattr(component, "output_stages", None)
        if stages is None and (component.outputs_need_inputs or component.output is not component.state):
            stages = (OutputMethod(
                "set_outputs",
                _fields(component.input) if component.outputs_need_inputs else (),
                _fields(component.output),
            ),)
        direct_inputs = {
            field: tuple(stage.inputs)
            for stage in stages or ()
            for field in stage.outputs
        }
        publish = {field: circuit.published[id(component), field] for field in declared}
        env: dict[str, Any] = {}
        publications = ({key: (dimension, terms) for key, dimension, terms in circuit.publications}
                        if not circuit._transforms else {})
        state_values = []
        if not circuit._transforms:
            for owner, name, complex_state, _index in circuit.layout:
                if owner is not component:
                    continue
                state_values.extend((f"state.{name}.real", f"state.{name}.imag")
                                    if complex_state else (f"state.{name}",))
        state_tuple = "(" + ", ".join(state_values) + ("," if len(state_values) == 1 else "") + ")"

        def substitute(code: str, namespace: Mapping[str, Any], values: Sequence[str]) -> str:
            for index in range(len(values) - 1):
                real, imag = values[index], values[index + 1]
                if real.endswith(".real") and imag == real[:-5] + ".imag":
                    for compose in ("_complex", "complex"):
                        code = code.replace(
                            f"{compose}(y[{index}], y[{index + 1}])", real[:-5]
                        )
            for index in range(len(values) - 1, -1, -1):
                code = code.replace(f"y[{index}]", f"({values[index]})")
            for name, value in namespace.items():
                if isinstance(value, _BoundValue):
                    records = ((component, "self"), (component.input, "self.input"),
                               (component.state, "self.state"), (component.output, "self.output"))
                    prefix = next((label for record, label in records if value.record is record), None)
                    if prefix is not None:
                        expression = f"{prefix}.{value.field}"
                        if value.invert:
                            expression = f"not {expression}"
                        elif value.factor == -1.0:
                            expression = f"-{expression}"
                        elif value.factor != 1.0:
                            expression = f"({value.factor!r} * {expression})"
                        code = re.sub(rf"\b{re.escape(name)}\((?:t)?\)", expression, code)
            return code

        def complex_pair(real: str, imag: str) -> Optional[str]:
            sign = ""
            if real.startswith("-") and imag.startswith("-"):
                sign, real, imag = "-", real[1:], imag[1:]
            return sign + real[:-5] if real.endswith(".real") and imag == real[:-5] + ".imag" else None

        def scaled(factor: float, expression: str) -> str:
            if factor == 1.0:
                return f"({expression})"
            if factor == -1.0:
                if expression.startswith("-"):
                    return f"({expression[1:]})"
                return f"-({expression})"
            return f"{factor!r} * ({expression})"

        direct_values: dict[str, str] = {}

        def runtime_namespace(namespace: Mapping[str, Any]) -> dict[str, Any]:
            """Remove declaration-only callback wrappers from generated runtime code."""
            return {
                name: value.callback if isinstance(value, _CallableValue) else value
                for name, value in namespace.items()
            }

        def replace_direct(expression: str) -> str:
            for name, value in direct_values.items():
                expression = re.sub(rf"\b{re.escape(name)}\b", value, expression)
            return expression

        def replace_direct_statement(statement: str) -> str:
            match = re.fullmatch(r"(\s*[A-Za-z_]\w*\s*=\s*)(.*)", statement)
            return (match.group(1) + replace_direct(match.group(2))
                    if match is not None else replace_direct(statement))

        def generated_inputs(field: str) -> tuple[str, ...]:
            """Direct component inputs used by one compiled netlist output."""
            fallback = direct_inputs.get(field, all_inputs)
            if (circuit._transforms or circuit._mixtures or circuit.standalone_ports
                    or circuit.netlist.has_discrete_topology):
                return fallback
            _dimension, terms = publications[publish[field]]
            selected = sorted(
                index
                for _factor, output in terms
                for index in range(output.start, output.stop)
            )
            program = getattr(
                circuit.scalar_output_evaluator(selected), "_peslite_scalar_program", None
            )
            if program is None:
                return fallback
            statements, _result, namespace, expressions = program
            active = "\n".join((*_live_statements(statements, expressions), *expressions))
            dependencies = set()
            for name, value in namespace.items():
                if re.search(rf"\b{re.escape(name)}\b", active) is None:
                    continue
                declared = _record_dependencies(value)
                if declared is not None:
                    record, fields = declared
                    if record is component.input:
                        dependencies.update(fields)
                elif name != "_complex" and callable(value):
                    return fallback
            return tuple(name for name in all_inputs if name in dependencies)

        def inline_single_use(statements: Sequence[str], expressions: Sequence[str]
                              ) -> tuple[list[str], tuple[str, ...]]:
            """Inline pure generated temporaries used once, without moving callback evaluations."""
            lines, results = list(statements), list(expressions)
            assignment = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*(.*)")
            call = re.compile(r"\b[A-Za-z_]\w*\s*\(")
            changed = True
            while changed:
                changed = False
                for index, statement in enumerate(lines):
                    match = assignment.fullmatch(statement)
                    if match is None:
                        continue
                    name, value = match.groups()
                    if call.search(value):
                        continue
                    if any(
                        (later := assignment.fullmatch(line)) is not None
                        and later.group(1) == name
                        for line in lines[index + 1:]
                    ):
                        continue
                    pattern = re.compile(rf"\b{re.escape(name)}\b")
                    uses = sum(len(pattern.findall(line)) for line in lines[index + 1:])
                    uses += sum(len(pattern.findall(expression)) for expression in results)
                    uses -= sum(
                        complex_pair(results[i], results[i + 1]) in (name, f"-{name}")
                        for i in range(len(results) - 1)
                    )
                    if uses != 1:
                        continue
                    use_line = next(
                        (offset for offset, line in enumerate(lines[index + 1:], index + 1)
                         if pattern.search(line)),
                        len(lines),
                    )
                    if any(call.search(line) for line in lines[index + 1:use_line]):
                        continue
                    replacement = f"({value})"
                    lines[index + 1:] = [
                        pattern.sub(replacement, line) for line in lines[index + 1:]
                    ]
                    results = [pattern.sub(replacement, expression) for expression in results]
                    lines.pop(index)
                    changed = True
                    break
            return lines, tuple(results)

        def output_function(name: str, fields: Sequence[str], *, stamp: bool
                            ) -> tuple[Optional[Callable], tuple[str, ...], set[str]]:
            body = []
            inline_values = None
            calculation = []
            if (circuit._transforms or circuit._mixtures or circuit.standalone_ports
                    or circuit.netlist.has_discrete_topology):
                body.extend((
                    "    self._device_circuit.solve(t)",
                    "    self._device_circuit.publish_outputs(t)",
                ))
                body.extend(
                    f"    self.output.{field} = self._device_circuit.output.{publish[field]}"
                    for field in fields
                )
            else:
                selected = sorted({index for field in fields for _factor, output in publications[publish[field]][1]
                                   for index in range(output.start, output.stop)})
                position = {output: index for index, output in enumerate(selected)}
                evaluator_name = f"_{name}_evaluate"
                evaluator = circuit.scalar_output_evaluator(selected)
                program = getattr(evaluator, "_peslite_scalar_program", None)
                if program is None:
                    env[evaluator_name] = evaluator
                    body.append(
                        f"    values = {evaluator_name}(t, {state_tuple.replace('state.', 'self.state.')})"
                    )
                else:
                    statements, _result, namespace, expressions = program
                    env.update(runtime_namespace(namespace))
                    bound = [value.replace("state.", "self.state.") for value in state_values]
                    bound_statements = [
                        substitute(replace_direct_statement(statement), namespace, bound)
                        for statement in statements
                    ]
                    bound_expressions = tuple(
                        substitute(replace_direct(expression), namespace, bound)
                        for expression in expressions
                    )
                    bound_statements = _live_statements(bound_statements, bound_expressions)
                    bound_statements, inline_values = inline_single_use(
                        bound_statements, bound_expressions
                    )
                    calculation = bound_statements
            assignments = []
            aliases = set()
            for field in fields if not (
                    circuit._transforms or circuit._mixtures or circuit.standalone_ports
                    or circuit.netlist.has_discrete_topology
            ) else ():
                key = publish[field]
                dimension, terms = publications[key]
                paired = None
                if inline_values is not None and dimension == 2 and len(terms) == 1:
                    factor, output = terms[0]
                    paired = complex_pair(
                        inline_values[position[output.start]],
                        inline_values[position[output.start + 1]],
                    )
                if inline_values is not None and dimension == 1 and len(terms) == 1:
                    factor, output = terms[0]
                    value = inline_values[position[output.start]]
                    direct = re.fullmatch(r"\(*\s*([A-Za-z_]\w*)\s*\)*", value)
                    if factor == 1.0 and direct is not None:
                        direct_values.setdefault(direct.group(1), f"self.output.{field}")
                parts = [
                    " + ".join(
                        scaled(factor, inline_values[position[output.start + k]])
                        if inline_values is not None else
                        scaled(factor, f"values[{position[output.start + k]}]")
                               for factor, output in terms) or "0.0"
                    for k in range(dimension)
                ]
                expression = (scaled(terms[0][0], paired) if paired is not None else
                              f"({parts[0]}) + 1j * ({parts[1]})" if dimension == 2 else parts[0])
                direct = re.fullmatch(
                    rf"\(*\s*self\.output\.{re.escape(field)}\s*\)*", expression
                )
                if direct is not None:
                    aliases.add(field)
                else:
                    assignments.append(f"    self.output.{field} = {expression}")
            actual = tuple(field for field in fields if field not in aliases)
            if actual:
                body.extend(calculation)
                if stamp and (circuit._energy_uses_time or circuit.locate_commutations):
                    body.append("    self._device_circuit.t = t")
                body.extend(assignments)
            if not body:
                return None, (), aliases
            exec(compile(f"def {name}(self, t):\n" + "\n".join(body or ["    pass"]) + "\n",
                         "<peslite device netlist>", "exec"), env)
            source = tuple(line[4:] for line in body)
            identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", "\n".join(source)))
            namespace = {
                key: value for key, value in env.items()
                if key in identifiers and key not in (name, "__builtins__")
            }
            env[name]._peslite_inline_program = _InlineProgram(source, (), namespace)
            env[name]._peslite_netlist_outputs = (circuit, actual)
            return MethodType(env[name], component), actual, aliases

        dependencies: dict[tuple[str, ...], list[str]] = {}
        all_inputs = _fields(component.input)
        for field in declared:
            dependencies.setdefault(generated_inputs(field), []).append(field)
        generated = []
        functions = []
        aliases = set()
        dependency_stages = tuple(dependencies.items())
        for index, (inputs_, fields) in enumerate(dependency_stages):
            name = "_netlist_outputs" + (f"_{index + 1}" if index else "")
            function, actual, direct = output_function(
                name, fields, stamp=index == len(dependency_stages) - 1
            )
            aliases.update(direct)
            if function is not None:
                setattr(component, name, function)
                functions.append(function)
                generated.append(OutputMethod(name, inputs_, actual))
        kept, observers = [], []
        for stage in stages or ():
            remaining = tuple(
                field for field in stage.outputs
                if field not in declared or field in aliases
            )
            if remaining:
                target = kept if stage.rhs else observers
                target.append(OutputMethod(
                    stage.method, tuple(stage.inputs), remaining, rhs=stage.rhs
                ))
        self.outputs = tuple(functions)
        component.output_stages = (*kept, *generated, *observers)

        entries = [(name, i, complex_state) for owner, name, complex_state, i in circuit.layout
                   if owner is component]
        items = [f"d[{i}] + 1j * d[{i + 1}]" if complex_state else f"d[{i}]"
                 for _name, i, complex_state in entries]
        rhs_env: dict[str, Any] = {}
        rhs_lines = []
        if not entries:
            rhs_lines.append("    return ()")
        elif (circuit._transforms or circuit._mixtures or circuit.standalone_ports
              or circuit.netlist.has_discrete_topology):
            rhs_lines.extend(("    self._device_circuit.solve(t)",
                              "    d = self._device_circuit.derivatives"))
        else:
            state_values = []
            for name, _index, complex_state in entries:
                state_values.extend((f"self.state.{name}.real", f"self.state.{name}.imag")
                                    if complex_state else (f"self.state.{name}",))
            state_tuple = "(" + ", ".join(state_values) + ("," if len(state_values) == 1 else "") + ")"
            evaluator = circuit.scalar_derivatives
            program = getattr(evaluator, "_peslite_scalar_program", None)
            if program is None:
                rhs_lines.append(f"    d = self._device_circuit.scalar_derivatives(t, {state_tuple})")
            else:
                statements, _result, namespace, expressions = program
                rhs_env.update(runtime_namespace(namespace))
                expressions = tuple(
                    substitute(replace_direct(expression), namespace, state_values)
                    for expression in expressions
                )
                statements = [
                    substitute(replace_direct_statement(statement), namespace, state_values)
                    for statement in statements
                ]
                statements = _live_statements(statements, expressions)
                statements, expressions = inline_single_use(statements, expressions)
                rhs_lines.extend(statements)
                items = []
                for _name, index, complex_state in entries:
                    paired = complex_pair(expressions[index], expressions[index + 1]) \
                        if complex_state else None
                    items.append(f"({paired})" if paired is not None else
                                 f"({expressions[index]}) + 1j * ({expressions[index + 1]})"
                                 if complex_state else f"({expressions[index]})")
        if entries:
            rhs_lines.append("    return (" + ", ".join(items) + ("," if len(items) == 1 else "") + ")")
        exec(compile("def rhs(self, t):\n" + "\n".join(rhs_lines) + "\n",
                     "<peslite device derivatives>", "exec"), rhs_env)
        if entries:
            source = tuple(line[4:] for line in rhs_lines[:-1])
            identifiers = set(re.findall(
                r"\b[A-Za-z_]\w*\b", "\n".join((*source, *items))
            ))
            namespace = {
                key: value for key, value in rhs_env.items()
                if key in identifiers and key not in ("rhs", "__builtins__")
            }
            rhs_env["rhs"]._peslite_inline_program = _InlineProgram(
                source, tuple(items), namespace
            )
        rhs_env["rhs"]._peslite_netlist_rhs = (circuit, tuple((i, z) for _name, i, z in entries))
        self.rhs = MethodType(rhs_env["rhs"], component)
        component.rhs = self.rhs
        if id(component) in circuit._loss_elements:
            component.dissipated_power = lambda: circuit.dissipated_power_of(component)
        if id(component) in circuit._supply_elements:
            component.supplied_power = lambda: circuit.supplied_power_of(component)
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
    output_fields = list(dict.fromkeys(output_fields))
    for field in output_fields:
        circuit.published[id(component), field] = circuit.key(
            f"{name}.{field}", getattr(component.output, field), output=True
        )
    circuit.finalize()
    _DeviceRuntime(component, circuit, output_fields)
    return circuit


def _standalone_device(component: Any) -> Circuit:
    circuit = getattr(component, "_device_circuit", None)
    if circuit is not None:
        return circuit
    declaration = compile_network(component)
    if declaration is None:
        raise TypeError(f"{type(component).__name__} has no device netlist equations")
    return bind_device(type(component).__name__, component, declaration)


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


from ..solver.netlist import register_device_runtime

register_device_runtime(
    prepare=_standalone_device,
    rhs=device_rhs,
    outputs=device_outputs,
    dissipated_power=device_dissipated_power,
    supplied_power=device_supplied_power,
    commutation_margin=device_commutation_margin,
    commutate=device_commutate,
)


def _find(parent: dict[int, int], a: int) -> int:
    parent.setdefault(a, a)
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def lower_devices(subsystems: Mapping[str, Any], connections: Mapping, zoh: Mapping,
                  buses: Mapping[str, Sequence[Port]],
                  declarations: Mapping[int, Optional[NetworkPlan]]) -> tuple[
        dict, dict, dict, tuple[Circuit, ...]]:
    """Compile the complete acausal electrical network into direct scalar equations."""
    out, lowered, held, circuit = lower(
        "device", subsystems, connections, zoh, buses, declarations
    )
    return out, lowered, held, (() if circuit is None else (circuit,))


def lower(circuit_model: str, subsystems: Mapping[str, Any], connections: Mapping, zoh: Mapping,
          buses: Mapping[str, Sequence[Port]],
          declarations: Mapping[int, Optional[NetworkPlan]], *,
          native: Optional[set[int]] = None
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
              if (native is None or id(sub) in native)
              and (declaration := declarations.get(id(sub))) is not None}
    missing = sorted(
        names.get(id(port.component), type(port.component).__name__)
        for group in members for port in group
        if id(port.component) not in probes and not port.behavioral
    )
    if missing:
        raise ConfigError(
            f"electrical components need netlist declarations: {sorted(set(missing))}"
        )
    if not probes:  # nothing to assemble: the circuit model only selects the implicit linear algebra
        return dict(subsystems), dict(connections), dict(zoh), None
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
    for group in members:
        if any(id(t.component) in probes for t in group):
            first = group[0]
            label = next((bus_of[id(t)] for t in group if id(t) in bus_of), None) \
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
    for (dst, field), spec in connections.items():
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
    for group in members:
        for port in group:
            if id(port.component) in probes:
                continue
            effort_record, _, effort_field = port.effort.partition(".")
            flow_record, _, flow_field = port.flow.partition(".")
            owner = names.get(id(port.component), type(port.component).__name__)
            voltage_key, current_key = port_sources[id(port)]
            if (effort_record, flow_record) == ("input", "output"):
                source_field, target_field = flow_field, effort_field
                source_key, target_key, kind = current_key, voltage_key, "current"
            elif (effort_record, flow_record) == ("output", "input"):
                source_field, target_field = effort_field, flow_field
                source_key, target_key, kind = voltage_key, current_key, "voltage"
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


def circuit_structure(model: Any, circuits: Sequence[Circuit]):
    """Map electrical groups to model states; keep shared and non-electrical states on the interface."""
    from ..solver.implicit import Structure

    n = model.n_states
    component_subsystem = {
        id(subsystem.component if isinstance(subsystem, NetworkView) else subsystem): subsystem
        for subsystem in model.subsystems
    }
    group_states: list[set[int]] = []
    group_labels: list[str] = []
    interface: set[int] = set()
    for circuit in circuits:
        local_to_model: dict[int, int] = {}
        for component, name, complex_state, local in circuit.layout:
            subsystem = component_subsystem[id(component)]
            positions = tuple(range(*model.state_slice(subsystem, name).indices(n)))
            if len(positions) != (2 if complex_state else 1):
                raise RuntimeError("circuit and model state layouts differ")
            local_to_model.update(
                (local + offset, position) for offset, position in enumerate(positions)
            )

        def states(elements) -> set[int]:
            return {
                local_to_model[int(local)]
                for element in elements
                for local in np.asarray(getattr(element, "state", ()), dtype=np.intp).reshape(-1)
                if int(local) in local_to_model
            }

        plan = circuit.netlist.grouping()
        for elements in plan.elements:
            group_states.append(states(elements))
            group_labels.append(", ".join(dict.fromkeys(
                circuit._owners.get(id(element), "?") for element in elements
            )))
        interface.update(states(plan.interface_elements))
        interface_nodes = {id(node) for node in plan.interface}
        interface.update(states(
            element for element in circuit.netlist.elements
            if any(id(node) in interface_nodes for port in element.ports for node in port if node is not None)
        ))

    occurrences: dict[int, int] = {}
    for group in group_states:
        for position in group:
            occurrences[position] = occurrences.get(position, 0) + 1
    interface.update(position for position, count in occurrences.items() if count > 1)
    grouped = set().union(*group_states) if group_states else set()
    interface.update(set(range(n)) - grouped)
    groups, labels = [], []
    for group, label in zip(group_states, group_labels):
        interior = np.array(sorted(group - interface), dtype=np.intp)
        if interior.size:
            groups.append(interior)
            labels.append(label)
    return Structure(groups, np.array(sorted(interface), dtype=np.intp), labels)


def circuit_mode_key(model: Any, circuits: Sequence[Circuit]):
    """Return the assembled circuits' exact structural modes for implicit-Jacobian reuse."""
    del model
    if not circuits:
        return None
    return tuple(
        (id(circuit.netlist), (
            circuit._bound_device_topology() or circuit.netlist.topology()
        ).mode.key[:2])
        for circuit in circuits
    )


def circuit_linear_blocks(model: Any, circuits: Sequence[Circuit]):
    """Return exact present-mode circuit state blocks in the model's state coordinates."""
    n = model.n_states
    component_subsystem = {
        id(subsystem.component if isinstance(subsystem, NetworkView) else subsystem): subsystem
        for subsystem in model.subsystems
    }
    blocks = []
    for circuit in circuits:
        member_map = {
            id(component_subsystem[id(component)]): component_subsystem[id(component)]
            for component in circuit.components
            if id(component) in component_subsystem
        }
        if id(circuit) in component_subsystem:
            member_map[id(circuit)] = component_subsystem[id(circuit)]
        members = tuple(member_map.values())
        external_feedback = model.has_external_state_dependency(members)
        local_to_model = np.empty(circuit.n_states, dtype=np.intp)
        for component, name, complex_state, local in circuit.layout:
            subsystem = component_subsystem[id(component)]
            positions = tuple(range(*model.state_slice(subsystem, name).indices(n)))
            if len(positions) != (2 if complex_state else 1):
                raise RuntimeError("circuit and model state layouts differ")
            local_to_model[local:local + len(positions)] = positions
        local = circuit.n_states
        transform = np.eye(local)
        inverse = np.eye(local)
        for index, matrix, inv in circuit._transforms:
            transform[np.ix_(index, index)] = matrix
            inverse[np.ix_(index, index)] = inv
        states = np.asarray(circuit.netlist.states, dtype=np.intp)
        net_matrix = circuit.netlist.topology().state_space.A
        matrix = np.zeros((local, local))
        matrix[np.ix_(states, states)] = net_matrix
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
        skipped = np.zeros(0, dtype=np.intp) if external_feedback else exact
        blocks.append((local_to_model, inverse @ matrix @ transform, exact, skipped))
    return tuple(blocks)


def circuit_explicit_stable(model: Any, method: str, step: float, *,
                            circuits: Sequence[Circuit]) -> bool:
    """Whether every present linear circuit mode lies inside an explicit RK stability region."""
    del model
    from ..solver.integrators import TABLEAUS

    _c, rows, weights = TABLEAUS[method]
    for circuit in circuits:
        topology = circuit.netlist.topology()
        if (circuit.netlist.form == "device"
                and topology._state_space is None
                and any(transformer.constant is None
                        for transformer in circuit.netlist._transformers)):
            # The scalar device equations already consume these run-time ratios directly. Building
            # a dense state space solely for this check would factorize an otherwise unused network.
            continue
        key = topology.mode.key
        result_key = method, key
        bounds = circuit._explicit_stability.get(result_key)
        if bounds is not None:
            circuit._explicit_stability.move_to_end(result_key)
            stable_below, unstable_above = bounds
            if step <= stable_below:
                continue
            if step >= unstable_above:
                return False
        eigenvalues = circuit._mode_eigenvalues.get(key)
        if eigenvalues is None:
            eigenvalues = np.linalg.eigvals(topology.state_space.A)
            circuit._mode_eigenvalues[key] = eigenvalues
            circuit._mode_eigenvalues.move_to_end(key)
            while len(circuit._mode_eigenvalues) > circuit.netlist.cache_size:
                circuit._mode_eigenvalues.popitem(last=False)
        def stable_at(candidate: float) -> bool:
            for eigenvalue in eigenvalues:
                scale = max(1.0, abs(eigenvalue))
                if eigenvalue.real > 1e-12 * scale:
                    continue  # physical instability is not numerical stiffness
                z = candidate * eigenvalue
                stages = [1.0 + 0.0j]
                for row in rows:
                    stages.append(1.0 + z * sum(a * value for a, value in zip(row, stages)))
                amplification = 1.0 + z * sum(
                    b * value for b, value in zip(weights, stages)
                )
                if abs(amplification) > 1.0 + 1e-10:
                    return False
            return True

        stable = stable_at(step)
        previous_stable, previous_unstable = bounds or (0.0, math.inf)
        if stable:
            probe = 1.25 * step
            probe_stable = stable_at(probe)
            bounds = (
                max(previous_stable, probe if probe_stable else step),
                previous_unstable if probe_stable else min(previous_unstable, probe),
            )
        else:
            bounds = previous_stable, min(previous_unstable, step)
        circuit._explicit_stability[result_key] = bounds
        circuit._explicit_stability.move_to_end(result_key)
        while len(circuit._explicit_stability) > circuit.netlist.cache_size:
            circuit._explicit_stability.popitem(last=False)
        if not stable:
            return False
    return True
