"""The electrical network of a netlist circuit model: one Circuit subsystem and a view per component.

Under ``simulation.solver.circuit_model`` ``statewise`` (dense per-mode state space) or ``mna``
(sparse factorized derivative MNA), :func:`lower` rewrites the connection graph the System
assembled. Components declaring ``netlist(nb)`` become primitives of one
:class:`~peslite.solver.netlist.Netlist` between bus nodes; ``netlist(nb)`` returns
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

Under ``mna`` the sparse solve eliminates the netlist's automatic switch-cluster groups onto the buses
between them (:meth:`~peslite.solver.netlist.Netlist.grouping`); :meth:`Circuit.grouping_report`
describes the groups by subsystem.
"""

from __future__ import annotations

import copy
import itertools
import math
import re
from types import MethodType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ..components.network import Port
from ..solver.energy import Energetic, spec_of
from ..solver.model import Bag, ConfigError, Empty, OutputMethod, _InlineProgram
from ..solver.netlist import (QUANTITIES, CurrentSource, Diode, Element, Netlist, Node,
                              VoltageSource, _Degenerate)

__all__ = ["NetlistBuilder", "Circuit", "NetworkView", "lower", "lower_devices",
           "bind_device", "declare_netlist"]


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

    __slots__ = ("callback", "inputs")

    def __init__(self, callback: Callable, inputs: Sequence[str]) -> None:
        self.callback, self.inputs = callback, tuple(inputs)

    def __call__(self, *args: Any) -> Any:
        return self.callback(*args)


class NetlistBuilder:
    """What a component's ``netlist(nb)`` declares its primitives with.

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

    def function(self, callback: Callable, inputs: Sequence[str] = ()) -> Callable:
        """A callback whose direct dependencies are named fields of the owner's input record."""
        if not callable(callback):
            raise TypeError("a netlist function value must be callable")
        if self.owner is not None:
            known = set(_fields(self.owner.input))
            unknown = set(inputs) - known
            if unknown:
                raise ValueError(f"unknown netlist function input fields {sorted(unknown)}")
        return _CallableValue(callback, inputs)

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
            T = np.asarray(transform, dtype=float)
            if T.shape != (index.size, index.size) or not np.all(np.isfinite(T)) or (
                    index.size and np.linalg.cond(T) > 1e12):
                raise ConfigError(f"{type(component).__name__}: a state transform must be an invertible "
                                  f"{index.size} x {index.size} matrix, got shape {T.shape}")
            self.transforms.append((index, T))
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


class _NetlistDeclaration:
    """One pure component declaration, replayed into a local or assembled netlist."""

    def __init__(self, component: Any, builder: NetlistBuilder) -> None:
        self.component, self.builder = component, builder

    def materialize(self, target: NetlistBuilder) -> None:
        source = self.builder
        node_map: dict[int, Optional[Node]] = {id(None): None}
        for key, node in source._nodes.items():
            node_map[id(node)] = target.node(source._ports[key])
        for element in source.netlist.elements:
            for node in (item for port in element.ports for item in port if item is not None):
                node_map.setdefault(id(node), Node(node.dim, node.scale, node.name))

        state_map: dict[int, int] = {}
        for key, indices in source._states.items():
            owner, name = source._state_owners[key]
            actual = target.state(owner, name)
            state_map.update((int(old), int(new)) for old, new in zip(indices, actual))
        for indices, transform in source.transforms:
            target.transforms.append((
                np.asarray([state_map[int(index)] for index in indices], dtype=np.intp),
                transform,
            ))

        elements: dict[int, Element] = {}
        owner_of = {id(element): source.owners[key]
                    for key, owned in source.owned.items() for element in owned}
        for element in source.netlist.elements:
            cloned = copy.copy(element)
            cloned.ports = tuple(tuple(node_map[id(node)] for node in port) for port in element.ports)
            for attribute, value in vars(cloned).items():
                if isinstance(value, _BoundValue):
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
            target.owner = owner_of[id(element)]
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


def declare_netlist(sub: Any) -> Optional[_NetlistDeclaration]:
    """Declare ``sub`` once without changing it, or return ``None`` without a netlist form."""
    declare = getattr(sub, "netlist", None)
    if not callable(declare):
        return None
    probe = NetlistBuilder(Netlist("device"))
    probe.owner = sub
    return None if declare(probe) is NotImplemented else _NetlistDeclaration(sub, probe)


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
    grouped = True  # mna: eliminate the automatic switch-cluster groups onto their interface (Netlist.grouping)

    def __init__(self, form: str) -> None:
        self.form = form
        self.names: dict[int, str] = {}  # id(subsystem) -> its name in the model (components and ports)
        self._owners: dict[int, str] = {}  # id(element) -> the subsystem that declared it
        self.state = Empty()
        self.components: list[Any] = []
        self.declarations: dict[int, _NetlistDeclaration] = {}
        self.groups: list[_Group] = []
        self.port_values: list[tuple[Port, Optional[_Group], str, str]] = []
        self.standalone_ports: tuple[Port, ...] = ()
        self.behavioral_sources: list[tuple[Port, _Group, str, str]] = []
        self.copies: list[tuple[Any, str, str]] = []  # (component, input field, input key)
        self.published: dict[tuple[int, str], str] = {}  # (id(component), out field) -> out key
        self.inputs: dict[str, Any] = {}  # input key -> initial value
        self.outputs: dict[str, Any] = {}  # out key -> initial value
        self.passthrough: list[tuple[str, str]] = []  # input key -> output key
        self.value_inputs: dict[tuple[int, str], str] = {}  # bound component outputs -> input key
        self.netlist: Optional[Netlist] = None
        self.derivatives: list[float] = []
        self.t = 0.0
        self.impulsive_dissipated_energy = 0.0  # J, dissipated by state jumps at mode changes
        self.impulsive_supplied_energy = 0.0  # J, delivered by sources during those jumps
        self.locate_commutations = False  # only diodes commutate between boundaries
        self.commutation_active = False
        self.topology_held = False
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
        self._mirrored: Optional[tuple[bool, ...]] = None  # their conduction when the hooks last ran
        self._boundary_signature: Optional[tuple] = None
        self._stored: Optional[tuple[list[float], np.ndarray]] = None  # (component states, netlist vector) last stored

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
        if self.outputs:
            self.output_stages = (
                OutputMethod("solve", tuple(self.inputs), ()),
                OutputMethod("publish_outputs", (), tuple(self.outputs), rhs=False),
            )
            self._output_dependencies = {"publish_outputs": ((self, "solve"),)}
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
        if self.netlist is not None:
            refreshed = {id(component): declare_netlist(component) for component in self.components}
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
        self._consistent = None  # new parameters (a set event) may move the loops' and cutsets' constraints
        self._consistent_constraints = None
        self._transforms = tuple((index, T, np.linalg.inv(T)) for index, T in nb.transforms)
        self._stored = None
        hooks = [(comp, nb.hooks[id(comp)]) for comp in self.components if id(comp) in nb.hooks]
        self._hooks = tuple((comp, hook["after"]) for comp, hook in hooks if hook["after"] is not None)
        self._watched = tuple(sorted({net._diodes.index(d) for _c, hook in hooks if hook["after"] is not None
                                      for d in hook["diodes"]}))
        self.commutation_active = self.locate_commutations or bool(self._hooks)
        self._mirrored = None
        self._boundary_signature = None
        if self.form == "device":
            (self.scalar_derivatives, self.scalar_outputs,
             self.scalar_output_evaluator) = net.scalar_evaluators()

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
        for index, _T, inverse in self._transforms:
            y[index] = inverse @ y[index]
        self.scatter(y.tolist())
        self._stored = (self.gather(), net_y)

    def _degenerate(self, exc: _Degenerate, t: float) -> ConfigError:
        owners = sorted({self._owners.get(id(el), "?") for el in exc.elements(self.netlist)})
        return ConfigError(f"t = {t:.9g} s: the netlist circuit has no consistent equations in its present mode "
                           f"(switches, diodes) through {owners}: {exc}")

    def solve(self, t: float) -> None:
        self.receive()
        net = self.netlist
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
            for index, _T, inverse in self._transforms:
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
                projected, _loss = self.netlist.commutate(t, states)
                if not np.allclose(projected, states, rtol=1e-12, atol=1e-14):
                    raise ConfigError("an averaged netlist case cannot require an impulsive state projection")
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
        return derivatives, outputs

    def set_outputs(self, t: float) -> None:
        self.solve(t)

    def cpp_outputs(self, generator: Any) -> list[str]:
        """Lower the same netlist equations through the standalone C++ backend."""
        return generator.trace_circuit_outputs(self)

    def cpp_output_stage(self, generator: Any, method: Any) -> list[str]:
        """Evaluate once in generated code; its trace already assigns every public output."""
        return self.cpp_outputs(generator) if method.__name__ == "solve" else []

    def cpp_commutation_margin(self, generator: Any) -> str:
        """Lower the netlist's diode guard through the standalone C++ backend."""
        if not self.cpp_commutating_diodes:
            return "std::numeric_limits<double>::infinity()"
        return generator.trace_circuit_commutation_margin(self)

    def cpp_commutate(self, generator: Any) -> list[str]:
        """Lower the netlist's diode transition through the standalone C++ backend."""
        if not self.cpp_commutating_diodes:
            return []
        return generator.trace_circuit_commutation(self)

    @property
    def cpp_commutating_diodes(self) -> tuple[Diode, ...]:
        """Diodes whose conduction state is independent of a controlled device gate."""
        return tuple(diode for diode in self.netlist._diodes if diode.gate is None)

    def equation_state_bindings(self) -> dict[str, tuple[Any, str]]:
        """Persistent discrete state owned by the circuit's netlist primitives."""
        return {
            f"diode_{index}_{attribute}": (diode, attribute)
            for index, diode in enumerate(self.cpp_commutating_diodes)
            for attribute in ("conducting", "residual")
        }

    def rhs(self, t: float):
        return ()

    def freeze_topology(self) -> None:
        """Keep the present netlist topology throughout one smooth integration attempt."""
        self.netlist.freeze_topology()

    def release_topology(self) -> None:
        """Read topology callbacks again at the next model-changing boundary."""
        self.netlist.release_topology()

    # ---------------------------------------------------------------- mode changes and diodes
    def _pattern(self, mode: Any = None) -> tuple[bool, ...]:
        conducting = (mode or self.netlist.mode()).conducting
        return tuple(conducting[j] for j in self._watched)

    def commutation_margin(self) -> float:
        """Negative while the states are inconsistent with the present mode (it changed at a boundary),
        a diode must commutate or a component has not mirrored the conduction yet; read after the model
        was synchronized. A change of transformer ratios alone keeps constraints that do not involve them."""
        net, y = self.netlist, self._states()
        evaluation = net._last
        topology = evaluation.topology if evaluation is not None and evaluation.t == self.t else net.topology()
        mode = topology.mode
        if self._hooks and self._pattern(mode) != self._mirrored:
            return -math.inf
        key, last = mode.key, self._consistent
        if key != last:
            try:
                constraints = topology.constraint_key()
                if constraints is not self._consistent_constraints and not net.consistent(self.t, y, topology):
                    return -math.inf
            except _Degenerate as exc:
                raise self._degenerate(exc, self.t) from None
            self._consistent = key
            self._consistent_constraints = constraints
        if not self.locate_commutations:
            return math.inf
        return net.commutation_margin(
            self.t, y, evaluated=evaluation is not None and evaluation.t == self.t
        )

    def commutate(self, t: float) -> None:
        """Find the consistent diode pattern and jump the states into it (charge and flux conservation);
        then let the components mirror the conduction (``NetlistBuilder.commutation``)."""
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
            self._mirrored = self._pattern()
        self._boundary_signature = self._boundary_mode()

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

    def consistent_state(self, t: float) -> None:
        """Project the states onto the present mode without recording a jump (audit states)."""
        y, _loss = self.netlist.project(t, self._states())
        self._store(y)
        topology = self.netlist.topology()
        self._consistent = topology.mode.key
        self._consistent_constraints = topology.constraint_key()
        self._boundary_signature = self._boundary_mode()

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
            self._output_dependencies = {
                "netlist_outputs": ((circuit, "publish_outputs"),),
                **{method: ((self, "netlist_outputs"),) for method in derived},
            }

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

        def live_statements(statements: Sequence[str], expressions: Sequence[str]) -> list[str]:
            """Discard generated local assignments made obsolete by an already published output."""
            identifiers = re.compile(r"\b[A-Za-z_]\w*\b")
            assignment = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*(.*)")
            used = {name for expression in expressions for name in identifiers.findall(expression)}
            kept = []
            for statement in reversed(statements):
                match = assignment.fullmatch(statement)
                if match is None:
                    kept.append(statement)
                    used.update(identifiers.findall(statement))
                    continue
                name, value = match.groups()
                if name not in used:
                    continue
                used.remove(name)
                used.update(identifiers.findall(value))
                kept.append(statement)
            return list(reversed(kept))

        def generated_inputs(field: str) -> tuple[str, ...]:
            """Direct component inputs used by one compiled netlist output."""
            fallback = direct_inputs.get(field, all_inputs)
            if circuit._transforms or circuit._mixtures or circuit.standalone_ports:
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
            active = "\n".join((*live_statements(statements, expressions), *expressions))
            dependencies = set()
            for name, value in namespace.items():
                if re.search(rf"\b{re.escape(name)}\b", active) is None:
                    continue
                if isinstance(value, _BoundValue):
                    if value.record is component.input:
                        dependencies.add(value.field)
                elif isinstance(value, _CallableValue):
                    dependencies.update(value.inputs)
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
            if circuit._transforms or circuit._mixtures or circuit.standalone_ports:
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
                    bound_statements = live_statements(bound_statements, bound_expressions)
                    bound_statements, inline_values = inline_single_use(
                        bound_statements, bound_expressions
                    )
                    calculation = bound_statements
            assignments = []
            aliases = set()
            for field in fields if not (
                    circuit._transforms or circuit._mixtures or circuit.standalone_ports
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
        elif circuit._transforms or circuit._mixtures or circuit.standalone_ports:
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
                statements = live_statements(statements, expressions)
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


def bind_device(name: str, component: Any, declaration: _NetlistDeclaration,
                names: Optional[dict[int, str]] = None) -> Circuit:
    """Compile one component directly from its declaration, preserving its Model identity."""
    circuit = Circuit("device")
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
    declaration = declare_netlist(component)
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
    for stage in component.output_stages or ():
        getattr(component, stage.method)(t)


def device_dissipated_power(component: Any) -> float:
    _standalone_device(component)
    return component.dissipated_power()


def device_supplied_power(component: Any) -> float:
    _standalone_device(component)
    return component.supplied_power()


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
                  declarations: Mapping[int, Optional[_NetlistDeclaration]]) -> tuple[
        dict, dict, dict, tuple[Circuit, ...]]:
    """Compile the complete acausal electrical network into direct scalar equations."""
    out, lowered, held, circuit = lower(
        "device", subsystems, connections, zoh, buses, declarations
    )
    return out, lowered, held, (() if circuit is None else (circuit,))


def lower(circuit_model: str, subsystems: Mapping[str, Any], connections: Mapping, zoh: Mapping,
          buses: Mapping[str, Sequence[Port]],
          declarations: Mapping[int, Optional[_NetlistDeclaration]], *,
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
        else:
            held[(dst, field)] = label
    held.update(((circuit, key), label) for label, key in held_keys.items())

    out = {name: views.get(id(sub), sub) for name, sub in subsystems.items()}
    name, k = "circuit", 1
    while name in out:
        k += 1
        name = f"circuit_{k}"
    out[name] = circuit
    circuit.finalize()
    held_targets = {(id(dst.input), field) for dst, field in zoh}
    dynamic_ratios = [
        transformer.N for transformer in circuit.netlist._transformers
        if transformer.constant is None
    ]
    circuit.topology_held = bool(dynamic_ratios) and all(
        isinstance(value, _BoundValue)
        and (id(value.record), value.field) in held_targets
        for value in dynamic_ratios
    )
    for sub in circuit.components:
        views[id(sub)].bind(circuit, ())
    return out, lowered, held, circuit
