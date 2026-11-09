"""Netlist of linear circuit primitives, compiled per switching mode into state equations.

A :class:`Netlist` holds primitives between :class:`Node` objects of dimension ``d`` (ground is
``None``). Capacitor voltages and inductor currents are entries of the caller's flat state vector
``y``, referenced by index arrays; sources read value callbacks of ``t``. A mode is the position of
every switch, the conduction of every diode and the ratio ``N`` of every transformer.

Per mode the network is classified by a normal tree (union-find; priority V > C > R > L > I) and
compiled into the *derivative MNA*: one sparse system ``M z = K_y x + K_u u + K_du du/dt`` whose
solution holds the node voltages, the branch currents and the derivatives of all states. Link
capacitors (C/V loops) and tree inductors (L/I cutsets) are dependent states. Their derivatives
follow from derivative potentials on the V/C forest and from derivative KCL at the super-nodes of
the L/I forest, so effective capacitances and inductances arise from the extra unknowns.
``form="nodal"`` factorizes ``M`` once per mode and back-substitutes per evaluation;
``form="statewise"`` forms ``dx = A x + B u + B1 du/dt``,
``out = C x + D u + D1 du/dt`` once per mode.
Transformer ratios enter ``M`` only, so modes with the same switches and diodes share one structure.

State-space-nodal grouping (:meth:`Netlist.grouping`, sparse form): switch clusters (diodes and
mode-dependent transformers, joined with the switches and transformers they touch) grow over the passive
elements until they meet; the meeting nodes are the interface. ``M`` is then bordered block diagonal: each
group's block is factorized and cached by its content (its own mode), and only the small interface (Schur)
system is factorized per compiled mode. With fewer than two switching groups the network is one system.

:meth:`Netlist.project` makes a state consistent with the mode (charge and flux conservation) and
returns the energy dissipated by the jump; :meth:`Netlist.commutate` finds a consistent diode
conduction pattern. The power of a node's quantities is ``scale * v^T i`` (1.5 for peak-scaled
alpha-beta space vectors); a transformer between nodes of different scales stays lossless.
"""

from __future__ import annotations

import copy
import math
import numbers
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import lapack
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu

__all__ = ["Node", "Element", "Resistor", "Inductor", "CoupledInductors", "Capacitor", "VoltageSource",
           "CurrentSource", "CurrentState", "Dissipation", "Transformer", "Switch", "Diode", "Netlist", "Mode", "Topology", "StateSpace",
           "Grouping", "ConductionError", "FORMS", "QUANTITIES", "device_rhs", "device_outputs",
           "device_dissipated_power", "device_supplied_power",
           "device_commutation_margin", "device_commutate", "prepare_device"]


class _DeviceRuntime(Protocol):
    """Assembly implementation of standalone component execution."""

    def prepare(self, component: Any) -> None: ...
    def rhs(self, component: Any, t: float): ...
    def outputs(self, component: Any, t: float) -> None: ...
    def dissipated_power(self, component: Any) -> float: ...
    def supplied_power(self, component: Any) -> float: ...
    def commutation_margin(self, component: Any) -> float: ...
    def commutate(self, component: Any, t: float) -> None: ...


_DEVICE_RUNTIME: Optional[_DeviceRuntime] = None


def _install_device_runtime(runtime: _DeviceRuntime) -> None:
    """Install the assembly implementation exactly once."""
    global _DEVICE_RUNTIME
    if _DEVICE_RUNTIME is not None and _DEVICE_RUNTIME is not runtime:
        raise RuntimeError("the standalone netlist runtime is already installed")
    _DEVICE_RUNTIME = runtime


def _device_runtime() -> _DeviceRuntime:
    if _DEVICE_RUNTIME is None:
        raise RuntimeError("the standalone netlist runtime has not been initialized")
    return _DEVICE_RUNTIME


def prepare_device(component: Any) -> None:
    """Compile a standalone component before Model plans its output dependencies."""
    _device_runtime().prepare(component)


def device_rhs(component: Any, t: float):
    """Evaluate a standalone component through its declared device netlist."""
    return _device_runtime().rhs(component, t)


def device_outputs(component: Any, t: float) -> None:
    """Publish a standalone component's outputs through its device netlist."""
    _device_runtime().outputs(component, t)


def device_dissipated_power(component: Any) -> float:
    """Return loss derived from a component's resistive primitives."""
    return _device_runtime().dissipated_power(component)


def device_supplied_power(component: Any) -> float:
    """Return supply derived from a component's source primitives."""
    return _device_runtime().supplied_power(component)


def device_commutation_margin(component: Any) -> float:
    """Return the consistency margin of a component's diode network."""
    return _device_runtime().commutation_margin(component)


def device_commutate(component: Any, t: float) -> None:
    """Commutate a standalone component's declared diode network."""
    _device_runtime().commutate(component, t)

Port = Optional["Node"]
Value = Callable[[float], ArrayLike]

FORMS = ("device", "statewise", "nodal")
QUANTITIES = ("node_voltage", "branch_voltage", "branch_current")
_V, _C, _R, _L, _I = range(5)  # tree priority classes
_PRIORITY = {"vsrc": _V, "short": _V, "p": _V, "cap": _C, "res": _R, "ind": _L,
             "isrc": _I, "state_current": _I, "s": _I}
_CACHE = 32  # compiled modes kept
_DELTA_T = 1e-8  # half step (s) of the central difference of a source without a derivative
_MAX_PATTERNS = 64  # diode conduction patterns tried at one instant
_MARGIN_RTOL = 1e-9  # margins within this fraction of the circuit's scale count as zero
_FLOAT_RTOL = 64.0 * np.finfo(float).eps  # equivalent scalar/matrix evaluations of the same zero
_NOOP_RTOL = 1e-13  # projections smaller than this, relative to the states, are no jump
_DENSE_ENTRIES = 1 << 14  # coupling matrices up to this size multiply as dense arrays (no sparse overhead)
_SPARSE_SHARE = 0.1  # large matrices below this nonzero share multiply as sparse arrays
_GROUP_PARTS = 256  # factorized group blocks kept (least recently used first out)
_DENSE_GROUP = 96  # systems and group blocks up to this size are factorized by LAPACK (faster than SuperLU)
_PENDING = object()
_RATIO_RANK = 0.25  # largest low-rank correction relative to the network unknown count
_RATIO_CONDITION = 1e8  # worse-conditioned corrections are factorized normally
_DENSE_REUSE = 2  # direct solves before a repeated dense mode is expanded into state-space equations


@dataclass(frozen=True)
class _Symbolic:
    """One scalar node in a generated arithmetic expression DAG."""

    operation: str
    arguments: tuple[Any, ...]
    _hash: int = field(init=False, repr=False, compare=False)
    __array_priority__ = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.operation, self.arguments)))

    def __hash__(self) -> int:
        return self._hash

    @classmethod
    def value(cls, code: str) -> "_Symbolic":
        return cls("value", (code,))

    def __add__(self, other):
        return _symbolic_binary("+", self, other)

    def __radd__(self, other):
        return _symbolic_binary("+", other, self)

    def __sub__(self, other):
        return _symbolic_binary("-", self, other)

    def __rsub__(self, other):
        return _symbolic_binary("-", other, self)

    def __mul__(self, other):
        return _symbolic_binary("*", self, other)

    def __rmul__(self, other):
        return _symbolic_binary("*", other, self)

    def __truediv__(self, other):
        return _symbolic_binary("/", self, other)

    def __rtruediv__(self, other):
        return _symbolic_binary("/", other, self)

    def __neg__(self):
        if self.operation == "neg":
            return self.arguments[0]
        return _linear_scaled(self, -1.0)

    def __abs__(self):
        return _Symbolic("abs", (self,))

    def __lt__(self, other):
        return _symbolic_relation("<", self, other)

    def __le__(self, other):
        return _symbolic_relation("<=", self, other)

    def __gt__(self, other):
        return _symbolic_relation(">", self, other)

    def __ge__(self, other):
        return _symbolic_relation(">=", self, other)

    def __and__(self, other):
        return _symbolic_relation("and", self, other)

    def __rand__(self, other):
        return _symbolic_relation("and", other, self)

    def __or__(self, other):
        return _symbolic_relation("or", self, other)

    def __ror__(self, other):
        return _symbolic_relation("or", other, self)

    def __bool__(self):
        raise TypeError("a symbolic equation value cannot control a Python branch")

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any):
        if method != "__call__" or kwargs:
            return NotImplemented
        name = ufunc.__name__
        if name in {"add", "subtract", "multiply", "divide", "true_divide"}:
            left, right = inputs
            operation = {"add": "+", "subtract": "-", "multiply": "*",
                         "divide": "/", "true_divide": "/"}[name]
            return _symbolic_binary(operation, left, right)
        if name in {"absolute", "fabs"}:
            value, = inputs
            return abs(value)
        if name in {"maximum", "minimum"}:
            return _symbolic_relation("max" if name == "maximum" else "min", *inputs)
        comparisons = {"less": "<", "less_equal": "<=", "greater": ">",
                       "greater_equal": ">=", "equal": "==", "not_equal": "!="}
        if name in comparisons:
            return _symbolic_relation(comparisons[name], *inputs)
        return NotImplemented

    def __array_function__(self, function: Any, types: Any, args: tuple, kwargs: dict):
        if function.__name__ == "where" and len(args) == 3 and not kwargs:
            condition, when_true, when_false = args
            if isinstance(condition, (bool, np.bool_)):
                return when_true if condition else when_false
            return _Symbolic("select", (condition, when_true, when_false))
        return NotImplemented

    @property
    def real(self):
        return _Symbolic("real", (self,))

    @property
    def imag(self):
        return _Symbolic("imag", (self,))


def _constant(value: Any) -> bool:
    """Only literal numerical types can be folded; scalar subclasses may carry equations."""
    return type(value) in (bool, int, float, complex) or isinstance(value, np.number)


def _literal(value: Any) -> str:
    """Render one scalar constant without leaking a library-specific representation."""
    return repr(value.item() if isinstance(value, np.generic) else value)


def _linear_terms(coefficients: Sequence[float], variables: Sequence[str]) -> list[str]:
    """Render the nonzero terms of one numeric linear expression."""
    terms = []
    for coefficient, variable in zip(coefficients, variables):
        value = float(coefficient)
        if value == 1.0:
            terms.append(f"({variable})")
        elif value == -1.0:
            terms.append(f"-({variable})")
        elif value != 0.0:
            terms.append(f"{value!r} * ({variable})")
    return terms


def _linear_expression(coefficients: Sequence[float], variables: Sequence[str]) -> str:
    """Render one numeric linear expression."""
    return " + ".join(_linear_terms(coefficients, variables)) or "0.0"


def _zero(value: Any) -> bool:
    return _constant(value) and value == 0.0


def _one(value: Any) -> bool:
    return _constant(value) and value == 1.0


def _minus_one(value: Any) -> bool:
    return _constant(value) and value == -1.0


def _negative_of(left: Any, right: Any) -> bool:
    return isinstance(right, _Symbolic) and right.operation == "neg" and right.arguments[0] == left


def _linear_form(value: Any) -> tuple[Any, dict[Any, Any]]:
    """Return the constant and ordered terms of an additive scalar expression."""
    if _constant(value):
        return value, {}
    if isinstance(value, _Symbolic) and value.operation == "linear":
        constant, terms = value.arguments
        return constant, {term: coefficient for coefficient, term in terms}
    return 0.0, {value: 1.0}


def _linear_value(constant: Any, terms: Mapping[Any, Any]) -> Any:
    active = tuple((coefficient, term) for term, coefficient in terms.items()
                   if not _zero(coefficient))
    if not active:
        return constant
    if _zero(constant) and len(active) == 1:
        coefficient, term = active[0]
        if _one(coefficient):
            return term
        if _minus_one(coefficient):
            return _Symbolic("neg", (term,))
    return _Symbolic("linear", (constant, active))


def _linear_sum(left: Any, right: Any, sign: float = 1.0) -> Any:
    left_constant, left_terms = _linear_form(left)
    right_constant, right_terms = _linear_form(right)
    for term, coefficient in right_terms.items():
        left_terms[term] = left_terms.get(term, 0.0) + sign * coefficient
    return _linear_value(left_constant + sign * right_constant, left_terms)


def _linear_scaled(value: Any, factor: Any) -> Any:
    constant, terms = _linear_form(value)
    return _linear_value(factor * constant, {
        term: factor * coefficient for term, coefficient in terms.items()
    })


def _symbolic_binary(operation: str, left: Any, right: Any) -> Any:
    left = left.item() if getattr(left, "shape", None) == () else left
    right = right.item() if getattr(right, "shape", None) == () else right
    if _constant(left) and _constant(right):
        return {"+": lambda: left + right, "-": lambda: left - right,
                "*": lambda: left * right, "/": lambda: left / right}[operation]()
    if operation == "+":
        if _zero(left):
            return right
        if _zero(right):
            return left
        if _negative_of(left, right) or _negative_of(right, left):
            return 0.0
        return _linear_sum(left, right)
    elif operation == "-":
        if _zero(right):
            return left
        if _zero(left):
            return -right
        if isinstance(left, _Symbolic) and isinstance(right, _Symbolic) and left == right:
            return 0.0
        if isinstance(right, _Symbolic) and right.operation == "neg":
            return _symbolic_binary("+", left, right.arguments[0])
        return _linear_sum(left, right, -1.0)
    elif operation == "*":
        if _zero(left) or _zero(right):
            return 0.0
        if _one(left):
            return right
        if _one(right):
            return left
        if _minus_one(left):
            return -right
        if _minus_one(right):
            return -left
        if _constant(left):
            return _linear_scaled(right, left)
        if _constant(right):
            return _linear_scaled(left, right)
    elif operation == "/":
        if _zero(left):
            return 0.0
        if _one(right):
            return left
        if _minus_one(right):
            return -left
        if _constant(right):
            return _linear_scaled(left, 1.0 / right)
    return _Symbolic(operation, (left, right))


def _symbolic_relation(operation: str, left: Any, right: Any) -> Any:
    """Build a scalar comparison, reduction or logical node with constant folding."""
    left = left.item() if getattr(left, "shape", None) == () else left
    right = right.item() if getattr(right, "shape", None) == () else right
    if _constant(left) and _constant(right):
        return {
            "max": lambda: max(left, right), "min": lambda: min(left, right),
            "<": lambda: left < right, "<=": lambda: left <= right,
            ">": lambda: left > right, ">=": lambda: left >= right,
            "==": lambda: left == right, "!=": lambda: left != right,
            "and": lambda: bool(left and right), "or": lambda: bool(left or right),
        }[operation]()
    if operation == "and":
        if left is False or right is False:
            return False
        if left is True:
            return right
        if right is True:
            return left
    elif operation == "or":
        if left is True or right is True:
            return True
        if left is False:
            return right
        if right is False:
            return left
    return _Symbolic(operation, (left, right))


def _symbolic_solve(matrix: Sequence[Sequence[Any] | Mapping[int, Any]], right: list[Any],
                    reference: NDArray[np.float64],
                    varying: Optional[set[tuple[int, int]]] = None, *,
                    gauge: bool = False) -> list[Any]:
    """Eliminate a generated sparse linear system using one valid numerical pivot pattern."""
    n = len(right)
    rows = [
        dict(row) if isinstance(row, Mapping) else
        {column: value for column, value in enumerate(row) if not _zero(value)}
        for row in matrix
    ]
    moving = [
        {
            column for column, value in row.items()
            if ((index, column) in varying if varying is not None else not _constant(value))
        }
        for index, row in enumerate(rows)
    ]
    right = list(right)
    reference = np.array(reference, dtype=float, copy=True)
    pivots: list[tuple[int, int]] = []
    pivot_row = 0
    for column in range(n):
        active = [row for row in range(pivot_row, n)
                  if column in rows[row] and reference[row, column] != 0.0]
        if not active:
            if gauge:
                continue
            raise ValueError(f"the generated netlist equations are singular at column {column}")
        stable = [row for row in active if column not in moving[row]]
        candidates = stable or active
        constants = [row for row in candidates if _constant(rows[row][column])]
        candidates = constants or candidates
        pivot = max(candidates, key=lambda row: abs(reference[row, column]))
        if pivot != pivot_row:
            rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
            moving[pivot_row], moving[pivot] = moving[pivot], moving[pivot_row]
            right[pivot_row], right[pivot] = right[pivot], right[pivot_row]
            reference[[pivot_row, pivot]] = reference[[pivot, pivot_row]]
        source = rows[pivot_row]
        pivot_value = source[column]
        trailing = tuple((entry, value) for entry, value in source.items()
                         if entry > column)
        for row in range(pivot_row + 1, n):
            value = rows[row].pop(column, None)
            if value is None:
                continue
            factor_moves = column in moving[row] or column in moving[pivot_row]
            moving[row].discard(column)
            factor = value / pivot_value
            numeric_factor = reference[row, column] / reference[pivot_row, column]
            reference[row, column] = 0.0
            target = rows[row]
            for entry, pivot_entry in trailing:
                entry_moves = (
                    entry in moving[row]
                    or factor_moves
                    or entry in moving[pivot_row]
                )
                updated = target.get(entry, 0.0) - factor * pivot_entry
                if _zero(updated):
                    target.pop(entry, None)
                    moving[row].discard(entry)
                else:
                    target[entry] = updated
                    if entry_moves:
                        moving[row].add(entry)
                    else:
                        moving[row].discard(entry)
                reference[row, entry] -= numeric_factor * reference[pivot_row, entry]
            right[row] -= factor * right[pivot_row]
        pivots.append((pivot_row, column))
        pivot_row += 1
    solution: list[Any] = [0.0] * n
    for row, column in reversed(pivots):
        residual = right[row] - sum(
            value * solution[entry]
            for entry, value in rows[row].items() if entry > column
        )
        solution[column] = residual / rows[row][column]
    return solution


def _symbolic_source(expressions: Sequence[Any]) -> tuple[list[str], tuple[str, ...]]:
    """Render expression DAGs with shared non-leaf nodes evaluated once."""
    uses: dict[_Symbolic, int] = {}

    def count(value: Any) -> None:
        if not isinstance(value, _Symbolic) or value.operation == "value":
            return
        seen = uses.get(value, 0)
        uses[value] = seen + 1
        if seen:
            return
        if value.operation == "linear":
            for _coefficient, term in value.arguments[1]:
                count(term)
        else:
            for argument in value.arguments:
                count(argument)

    for expression in expressions:
        count(expression)
    shared = {node for node, n in uses.items() if n > 1}
    names: dict[_Symbolic, str] = {}
    lines: list[str] = []

    def render(value: Any, defining: Optional[_Symbolic] = None) -> str:
        if not isinstance(value, _Symbolic):
            return _literal(value)
        if value in names and value is not defining:
            return names[value]
        if value.operation == "value":
            return value.arguments[0]
        children = ([term for _coefficient, term in value.arguments[1]]
                    if value.operation == "linear" else value.arguments)
        for argument in children:
            if isinstance(argument, _Symbolic) and argument in shared and argument not in names:
                name = f"v{len(names)}"
                names[argument] = name
                lines.append(f"{name} = {render(argument, argument)}")
        if value.operation == "linear":
            constant, terms = value.arguments
            parts = ([] if _zero(constant) else [_literal(constant)])
            for coefficient, term in terms:
                expression = render(term)
                parts.append(expression if _one(coefficient) else
                             f"(-({expression}))" if _minus_one(coefficient) else
                             f"({_literal(coefficient)} * {expression})")
            return "(" + " + ".join(parts) + ")"
        arguments = tuple(render(argument) for argument in value.arguments)
        if value.operation == "neg":
            return f"(-({arguments[0]}))"
        if value.operation == "abs":
            return f"abs({arguments[0]})"
        if value.operation in ("real", "imag"):
            return f"({arguments[0]}).{value.operation}"
        if value.operation in ("max", "min"):
            return f"{value.operation}({arguments[0]}, {arguments[1]})"
        if value.operation == "select":
            return f"({arguments[1]} if {arguments[0]} else {arguments[2]})"
        if value.operation in ("and", "or"):
            return f"({arguments[0]} {value.operation} {arguments[1]})"
        return f"({arguments[0]} {value.operation} {arguments[1]})"

    results = tuple(render(expression) for expression in expressions)
    return lines, results


def _maximum(left: Any, right: Any) -> Any:
    """Scalar maximum which remains one arithmetic expression during equation lowering."""
    if type(left) is float and type(right) is float:  # numeric hot path without ABC checks
        return max(left, right)
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        return max(left, right)
    if hasattr(left, "__array_ufunc__") or hasattr(right, "__array_ufunc__"):
        return np.maximum(left, right)
    return 0.5 * (left + right + abs(left - right))


def _minimum(left: Any, right: Any) -> Any:
    """Scalar minimum paired with :func:`_maximum`."""
    if type(left) is float and type(right) is float:  # numeric hot path without ABC checks
        return min(left, right)
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        return min(left, right)
    if hasattr(left, "__array_ufunc__") or hasattr(right, "__array_ufunc__"):
        return np.minimum(left, right)
    return 0.5 * (left + right - abs(left - right))


def _maximum_all(values: Sequence[Any]) -> Any:
    result: Any = 0.0
    for value in values:
        result = _maximum(result, value)
    return result


def _logical_or(left: Any, right: Any) -> Any:
    if isinstance(left, (bool, np.bool_)) and isinstance(right, (bool, np.bool_)):
        return bool(left or right)
    return left | right


def _logical_and(left: Any, right: Any) -> Any:
    if isinstance(left, (bool, np.bool_)) and isinstance(right, (bool, np.bool_)):
        return bool(left and right)
    return left & right


def _select(condition: Any, when_true: Any, when_false: Any) -> Any:
    """Select a scalar value without branching while lowering symbolic equations."""
    if isinstance(condition, (bool, np.bool_)):
        return when_true if condition else when_false
    return np.where(condition, when_true, when_false)


def _complex(real: Any, imag: Any) -> Any:
    """Compose a complex scalar without coercing equation values to Python floats."""
    return real + 1j * imag


class ConductionError(RuntimeError):
    """No consistent diode conduction pattern at an instant (a Zeno or inconsistent circuit)."""


class _Degenerate(ValueError):
    """A loop of voltage-type branches only (``loop``), or a cutset of current-type branches only;
    ``branches``: its branch indices."""

    def __init__(self, message: str, branches: list[int], loop: bool = True) -> None:
        super().__init__(message)
        self.branches, self.loop = branches, loop

    def elements(self, net: "Netlist") -> list["Element"]:
        """The elements of the loop or cutset."""
        return list(dict.fromkeys(net._branches[k].element for k in self.branches))


class Node:
    """A node of dimension ``dim``; the power of its quantities is ``scale * v^T i``."""

    __slots__ = ("dim", "scale", "name")

    def __init__(self, dim: int = 1, scale: float = 1.0, name: str = "") -> None:
        if int(dim) < 1 or not scale > 0.0:
            raise ValueError(f"node {name!r}: dimension must be >= 1 and scale > 0")
        self.dim, self.scale, self.name = int(dim), float(scale), name

    def __repr__(self) -> str:
        return f"Node({self.name!r}, dim={self.dim})"


def _port(a: Port, b: Port) -> tuple[int, float]:
    nodes = [n for n in (a, b) if n is not None]
    if any(not isinstance(n, Node) for n in nodes):
        raise TypeError("branch ports must be Node objects or None (ground)")
    if not nodes or a is b:
        raise ValueError("a branch needs two distinct ports, at most one of them ground")
    if len(nodes) == 2 and (a.dim, a.scale) != (b.dim, b.scale):
        raise ValueError(f"branch ports {a!r} and {b!r} differ in dimension or power scale")
    return nodes[0].dim, nodes[0].scale


def _symmetrizing_scale(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a positive diagonal similarity that balances mutually coupled entries.

    The similarity preserves the spectrum, so its matrix 2-norm remains a rigorous spectral-radius
    bound. Least-squares balancing keeps that bound useful when state scales differ greatly.
    """
    n = matrix.shape[0]
    coupled = (matrix != 0.0) & (matrix.T != 0.0) & np.isfinite(matrix) & np.isfinite(matrix.T)
    rows, cols = np.nonzero(np.triu(coupled, 1))
    if not rows.size:
        return np.ones(n)
    target = 0.5 * (np.log(np.abs(matrix[cols, rows])) - np.log(np.abs(matrix[rows, cols])))
    laplacian = 1e-9 * np.eye(n)
    np.add.at(laplacian, (rows, rows), 1.0)
    np.add.at(laplacian, (cols, cols), 1.0)
    np.add.at(laplacian, (rows, cols), -1.0)
    np.add.at(laplacian, (cols, rows), -1.0)
    rhs = np.zeros(n)
    np.add.at(rhs, rows, target)
    np.add.at(rhs, cols, -target)
    x = np.linalg.solve(laplacian, rhs)
    x -= x.mean()
    scale = np.exp(np.clip(x, -300.0, 300.0))
    return scale if np.all(np.isfinite(scale)) else np.ones(n)


def _scaled_norm(matrix: NDArray[np.float64], left: Optional[NDArray[np.float64]] = None,
                 right: Optional[NDArray[np.float64]] = None) -> float:
    """Return ``|diag(left) matrix diag(right)^-1|_2`` with either scale optional."""
    if left is not None:
        matrix = left[:, None] * matrix
    if right is not None:
        matrix = matrix / right[None, :]
    return float(np.linalg.norm(matrix, 2)) if matrix.size else 0.0


def _coefficient_array(value: Any) -> NDArray:
    """Preserve expression scalars while keeping ordinary coefficient arithmetic numerical."""
    if isinstance(value, np.ndarray) and value.dtype.kind != "O":
        return np.asarray(value, dtype=float)
    values = np.asarray(value, dtype=object)
    return np.asarray(values, dtype=float) if all(_constant(item) for item in values.flat) else values


def _numeric_values(value: Any) -> NDArray[np.float64]:
    """Read numerical coefficients while retaining their optional lowering expressions."""
    if isinstance(value, np.ndarray) and value.dtype.kind != "O":
        return np.asarray(value, dtype=float)
    values = np.asarray(value, dtype=object)
    return np.asarray([
        float(getattr(item, "_peslite_numeric_value", item))
        for item in values.flat
    ]).reshape(values.shape)


def _matrix(value: ArrayLike, d: int, what: str) -> NDArray[np.float64]:
    m = _numeric_values(value)
    if m.ndim == 0:
        m = float(m) * np.eye(d)
    if m.shape != (d, d) or not np.all(np.isfinite(m)):
        raise ValueError(f"{what} must be a finite scalar or {d} x {d} matrix, got shape {m.shape}")
    return m.copy()


def _block_weights(blocks: Sequence[tuple[float, NDArray]]) -> sp.csc_matrix:
    """The same block-diagonal storage weights, retaining equation-valued coefficients."""
    data, rows, columns = [], [], []
    offset = 0
    for scale, matrix in blocks:
        numeric = scale * _numeric_values(matrix)
        row, column = np.nonzero(numeric)
        data.append(numeric[row, column])
        rows.append(row + offset)
        columns.append(column + offset)
        offset += matrix.shape[0]
    result = sp.csc_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(offset, offset),
    )
    if any(matrix.dtype.kind == "O" for _scale, matrix in blocks):
        expressions = np.zeros(result.shape, dtype=object)
        offset = 0
        for scale, matrix in blocks:
            size = matrix.shape[0]
            expressions[offset:offset + size, offset:offset + size] = scale * matrix
            offset += size
        result._expressions = expressions
    return result


def _definite(m: NDArray[np.float64], what: str) -> NDArray[np.float64]:
    if not np.allclose(m, m.T, rtol=1e-12, atol=0.0) or np.linalg.eigvalsh(0.5 * (m + m.T)).min() <= 0.0:
        raise ValueError(f"{what} must be symmetric positive definite")
    return m


def _indices(state: ArrayLike, n: int, what: str) -> NDArray[np.intp]:
    idx = np.asarray(state, dtype=np.intp).reshape(-1)
    if idx.size != n or np.unique(idx).size != n or (n and idx.min() < 0):
        raise ValueError(f"{what} needs {n} distinct nonnegative state indices, got {idx.tolist()}")
    return idx


def _vector(value: ArrayLike, d: int, what: str) -> NDArray[np.float64]:
    if np.iscomplexobj(value) and d == 2 and np.ndim(value) == 0:
        c = complex(value)
        return np.array([c.real, c.imag])
    v = np.asarray(value, dtype=float).reshape(-1)
    if v.size != d:
        raise ValueError(f"{what} returned {v.size} values for a dimension-{d} branch")
    return v


class Element:
    """A netlist primitive. ``ports``: its (a, b) port pairs; positive current flows a -> b
    through the element and its voltage is ``v_a - v_b``."""

    ports: tuple[tuple[Port, Port], ...] = ()
    name: str = ""

    def _coefficient(self, name: str, value: ArrayLike, dimension: Optional[int] = None,
                     what: str = "coefficient"):
        """Keep the declared coefficient beside its normalized numerical value for lowering."""
        self.__dict__.setdefault("_coefficients", {})[name] = value
        return _numeric_values(value).item() if dimension is None else _matrix(value, dimension, what)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class Resistor(Element):
    """``v = R i``; ``R`` a scalar or d x d matrix (positive semidefinite), ``R = 0`` an ideal short."""

    def __init__(self, a: Port, b: Port, R: ArrayLike, name: str = "") -> None:
        d, _ = _port(a, b)
        self.ports, self.name, self.R = ((a, b),), name, self._coefficient("R", R, d, "resistance")
        if np.linalg.eigvalsh(0.5 * (self.R + self.R.T)).min() < -1e-12 * np.abs(self.R).max():
            raise ValueError("resistance must be positive semidefinite")


class Inductor(Element):
    """``v = L di/dt``; the current is the state at the ``d`` indices ``state``."""

    def __init__(self, a: Port, b: Port, L: ArrayLike, state: ArrayLike, name: str = "") -> None:
        d, _ = _port(a, b)
        self.ports, self.name = ((a, b),), name
        self.L = _definite(self._coefficient("L", L, d, "inductance"), "inductance")
        self.state = _indices(state, d, "an inductor")


class CoupledInductors(Element):
    """``v_k = sum_j L_kj di_j/dt`` over ``branches`` ``[(a_k, b_k), ...]``. ``L`` is (k d) x (k d), or
    k x k applied to every component; ``state``: the k d current indices, branch by branch."""

    def __init__(self, branches: Sequence[tuple[Port, Port]], L: ArrayLike, state: ArrayLike,
                 name: str = "") -> None:
        self.ports, self.name = tuple((a, b) for a, b in branches), name
        kinds = {_port(a, b) for a, b in self.ports}
        if len(kinds) != 1:
            raise ValueError("coupled inductors need at least one branch, all of one dimension and scale")
        d, k = kinds.pop()[0], len(self.ports)
        m = _coefficient_array(L)
        if m.shape == (k, k) and d > 1:
            m = np.kron(m, np.eye(d))
        self.L = _definite(self._coefficient("L", m, k * d, "coupled inductance"), "coupled inductance")
        self.state = _indices(state, k * d, "coupled inductors")


class Capacitor(Element):
    """``i = C dv/dt``; the voltage is the state at the ``d`` indices ``state``."""

    def __init__(self, a: Port, b: Port, C: ArrayLike, state: ArrayLike, name: str = "") -> None:
        d, _ = _port(a, b)
        self.ports, self.name = ((a, b),), name
        self.C = _definite(self._coefficient("C", C, d, "capacitance"), "capacitance")
        self.state = _indices(state, d, "a capacitor")


class VoltageSource(Element):
    """``v = value(t)``; ``derivative(t)`` is dv/dt, else a central difference in ``t`` (used only
    where a C/V loop needs it)."""

    def __init__(self, a: Port, b: Port, value: Value, derivative: Optional[Value] = None,
                 name: str = "") -> None:
        _port(a, b)
        self.ports, self.name, self.value, self.derivative = ((a, b),), name, value, derivative


class CurrentSource(Element):
    """``i = value(t)``; ``derivative(t)`` as for :class:`VoltageSource` (used in L/I cutsets)."""

    def __init__(self, a: Port, b: Port, value: Value, derivative: Optional[Value] = None,
                 name: str = "") -> None:
        _port(a, b)
        self.ports, self.name, self.value, self.derivative = ((a, b),), name, value, derivative


class CurrentState(Element):
    """Behavioral current port with ``i = x`` and ``dx/dt = derivative(t)``.

    Unlike an inductor, the port voltage does not determine the derivative. This primitive retains
    a reduced constitutive law as one physical netlist branch instead of an isolated helper circuit.
    """

    def __init__(self, a: Port, b: Port, state: ArrayLike, derivative: Value,
                 name: str = "") -> None:
        d, _ = _port(a, b)
        self.ports, self.name = ((a, b),), name
        self.state = _indices(state, d, "a current state")
        self.value, self.derivative = derivative, None


class Dissipation(Element):
    """A nonnegative power loss ``value(t)`` without an artificial electrical branch."""

    def __init__(self, value: Value, name: str = "") -> None:
        if not callable(value):
            raise TypeError("dissipation must be a callable of time")
        self.name, self.value = name, value


class Transformer(Element):
    """Lossless controlled two-port: ``v_p = N v_s`` and ``i_s = -(scale_p / scale_s) N^T i_p``.

    ``N()`` returns the d_p x d_s ratio and is part of the mode (a bridge: its switching function); a
    constant ``N`` (an array) is checked once. The primary is voltage-type and the secondary
    current-type in the normal tree.
    """

    def __init__(self, p_a: Port, p_b: Port, s_a: Port, s_b: Port,
                 N: Callable[[], ArrayLike] | ArrayLike, name: str = "", *,
                 bounds: Optional[tuple[ArrayLike, ArrayLike]] = None) -> None:
        (dp, scale_p), (ds, scale_s) = _port(p_a, p_b), _port(s_a, s_b)
        self.ports, self.name, self.N = ((p_a, p_b), (s_a, s_b)), name, N
        self.shape, self.current_scale = (dp, ds), scale_p / scale_s
        self.constant = None if callable(N) else self._ratio(N)
        if bounds is None:
            self.bounds = ((self.constant, self.constant)
                           if self.constant is not None else None)
        else:
            lower, upper = (self._ratio(value) for value in bounds)
            if np.any(lower > upper):
                raise ValueError(f"{self.name}: transformer ratio lower bounds exceed upper bounds")
            self.bounds = lower, upper
        if self.bounds is None:
            self._range = None
        else:
            lower, upper = self.bounds
            tolerance = _FLOAT_RTOL * np.maximum(
                1.0, np.maximum(np.abs(lower), np.abs(upper))
            )
            self._range = tuple(zip(
                (lower - tolerance).ravel().tolist(),
                (upper + tolerance).ravel().tolist(),
            ))
        self._cached_scalar: Optional[tuple[numbers.Number, NDArray[np.float64]]] = None

    def _ratio(self, value: ArrayLike) -> NDArray[np.float64]:
        # Common scalar bridge callbacks avoid NumPy's general shape conversion.
        kind = type(value)
        if kind is complex and self.shape == (2, 1):
            real, imag = value.real, value.imag
            if math.isfinite(real) and math.isfinite(imag):
                return np.array(((real,), (imag,)))
        elif (kind is float or kind is int) and self.shape == (1, 1):
            if math.isfinite(value):
                return np.array(((float(value),),))
        if np.ndim(value) == 0:
            if self.shape == (1, 1):
                value = ((value,),)
            elif self.shape == (2, 1) and np.iscomplexobj(value):
                scalar = complex(value)
                value = ((scalar.real,), (scalar.imag,))
        try:
            N = np.array(value, dtype=float).reshape(self.shape)
        except ValueError:
            raise ValueError(f"{self.name}: the ratio must be {self.shape[0]} x {self.shape[1]}") from None
        if not np.isfinite(N).all():
            raise ValueError(f"{self.name}: the ratio is not finite")
        return N

    def within_bounds(self, ratio: NDArray[np.float64]) -> bool:
        """Whether a ratio is covered by the optional stability range declaration."""
        if self._range is None:
            return False
        for value, (lower, upper) in zip(ratio.ravel().tolist(), self._range):
            if not lower <= value <= upper:
                return False
        return True

    def ratio(self) -> NDArray[np.float64]:
        """Return the current ratio, reusing the matrix form of an unchanged scalar callback."""
        if self.constant is not None:
            return self.constant
        value = self.N()
        kind = type(value)
        if kind is complex or kind is float or isinstance(value, numbers.Number):
            cached = self._cached_scalar
            if cached is not None and cached[0] == value:
                return cached[1]
            ratio = self._ratio(value)
            self._cached_scalar = (value, ratio)
            return ratio
        return self._ratio(value)


class Switch(Element):
    """``R_on`` while ``closed()``, else ``R_off``; 0 is an ideal short and ``inf`` an open circuit."""

    def __init__(self, a: Port, b: Port, closed: Callable[[], bool], R_on: float = 0.0,
                 R_off: float = math.inf, name: str = "") -> None:
        _port(a, b)
        if not (0.0 <= R_on <= R_off) or R_on == math.inf:
            raise ValueError("a switch needs 0 <= R_on <= R_off and a finite R_on")
        self.ports, self.name, self.closed = ((a, b),), name, closed
        self.R_on = self._coefficient("R_on", R_on)
        self.R_off = self._coefficient("R_off", R_off)


class Diode(Element):
    """A diode on dimension-1 ports: ``R_on`` while conducting (current a -> b nonnegative), else
    ``R_off`` (voltage nonpositive). ``conducting`` and ``residual`` are its discrete equation state.

    ``gate()``, if given, makes it a semiconductor device position (a controlled switch with this
    antiparallel diode): while the gate is on it conducts in both directions with ``R_on`` and is not
    commutated; when it turns off the diode takes over from its own (blocking) conduction state."""

    def __init__(self, a: Port, b: Port, R_on: float = 0.0, R_off: float = math.inf,
                 name: str = "", gate: Optional[Callable[[], bool]] = None) -> None:
        if _port(a, b)[0] != 1:
            raise ValueError("a diode connects dimension-1 nodes")
        if not (0.0 <= R_on <= R_off) or R_on == math.inf:
            raise ValueError("a diode needs 0 <= R_on <= R_off and a finite R_on")
        self.ports, self.name = ((a, b),), name
        self.R_on = self._coefficient("R_on", R_on)
        self.R_off = self._coefficient("R_off", R_off)
        self.gate = gate
        self.conducting = False
        self.residual = 0.0
        self.mirrored = False


class Mode:
    """Switch positions, diode conduction and transformer ratios; ``key`` identifies the mode."""

    __slots__ = ("closed", "conducting", "ratios", "key")

    def __init__(self, closed: Sequence[bool], conducting: Sequence[bool],
                 ratios: Sequence[NDArray[np.float64]]) -> None:
        self.closed, self.conducting, self.ratios = tuple(closed), tuple(conducting), tuple(ratios)
        self.key = (self.closed, self.conducting,
                    b"".join(r.tobytes() for r in self.ratios) if self.ratios else b"")


@dataclass(frozen=True)
class StateSpace:
    """``dx = A x + B u + B1 du/dt`` and ``out = C x + D u + D1 du/dt`` of one mode.

    ``x``: the netlist states (``Netlist.states`` order), ``u``: the source values
    (``Netlist.sources`` driving-value order), ``out``: node voltages, branch voltages, branch currents
    (``Netlist.output_slice``). Columns of dependent states are zero.
    """

    A: NDArray[np.float64]
    B: NDArray[np.float64]
    B1: NDArray[np.float64]
    C: NDArray[np.float64]
    D: NDArray[np.float64]
    D1: NDArray[np.float64]


@dataclass(frozen=True)
class Grouping:
    """The automatic state-space-nodal partition of a netlist (see :meth:`Netlist.grouping`).

    Per group: ``elements``, its own ``nodes``, ``devices`` (switches, diodes and transformers with a
    mode-dependent ratio) and the ``boundary`` interface nodes it touches. Switching groups come first;
    a last group without switch cluster gathers the passive rest (``passive``). ``interface``: the nodes
    between groups; ``interface_elements``: the elements between them and ground. ``used``: whether the
    sparse solve eliminates the groups onto the interface (at least two switching groups).
    """

    elements: tuple[tuple[Element, ...], ...]
    nodes: tuple[tuple[Node, ...], ...]
    devices: tuple[int, ...]
    boundary: tuple[tuple[Node, ...], ...]
    passive: bool
    interface: tuple[Node, ...]
    interface_elements: tuple[Element, ...]
    used: bool


class _Branch:
    __slots__ = ("element", "port", "a", "b", "dim", "scale", "output", "x", "label")

    def __init__(self, element: Element, port: int, a: int, b: int, dim: int, scale: float, out: int,
                 label: str) -> None:
        self.element, self.port, self.a, self.b, self.dim, self.scale = element, port, a, b, dim, scale
        self.output, self.x, self.label = out, -1, label


class _UnionFind:
    """Union-find over vertices; a component's root is its smallest vertex (ground is 0)."""

    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, v: int) -> int:
        p = self.parent
        while p[v] != v:
            p[v] = p[p[v]]
            v = p[v]
        return v

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


class _Triplets:
    """COO entries of a sparse matrix; duplicates are summed."""

    def __init__(self) -> None:
        self.r: list[int] = []
        self.c: list[int] = []
        self.v: list[float] = []

    def eye(self, r: int, c: int, n: int, s: float = 1.0) -> None:
        self.r.extend(range(r, r + n))
        self.c.extend(range(c, c + n))
        self.v.extend([s] * n)

    def block(self, r: int, c: int, m: NDArray[np.float64]) -> None:
        i, j = np.nonzero(_numeric_values(m))
        self.r.extend((r + i).tolist())
        self.c.extend((c + j).tolist())
        self.v.extend(m[int(row), int(column)] for row, column in zip(i, j))

    def incidence(self, r: int, br: _Branch, offsets: NDArray[np.intp], m: Optional[NDArray] = None,
                  s: float = 1.0) -> None:
        """``s (m) (x_a - x_b)`` for the variables at the branch ports in ``offsets``."""
        for v, sign in ((br.a, s), (br.b, -s)):
            if offsets[v] >= 0:
                if m is None:
                    self.eye(r, int(offsets[v]), br.dim, sign)
                else:
                    self.block(r, int(offsets[v]), sign * m)

    def matrix(self, shape: tuple[int, int]) -> sp.csc_matrix:
        return sp.csc_matrix((_numeric_values(self.v), (self.r, self.c)), shape=shape)


class _RatioSlots:
    """Entries ``coef * N_j[i, k]`` (or of ``N_j^T``) of the network matrix that follow transformer
    ratios: recorded once per switch and diode pattern, filled for each ratio."""

    def __init__(self, shapes: Sequence[tuple[int, int]]) -> None:
        self.shapes = list(shapes)
        self.offsets = np.cumsum([0] + [dp * ds for dp, ds in self.shapes]).tolist()
        self.r: list[int] = []
        self.c: list[int] = []
        self.flat: list[int] = []
        self.coef: list[float] = []

    def ratio(self, r: int, c: int, j: int, coef: float, transpose: bool = False) -> None:
        """``coef * N_j`` (``N_j^T``) at ``(r, c)``."""
        dp, ds = self.shapes[j]
        for i in range(ds if transpose else dp):
            for k in range(dp if transpose else ds):
                self.r.append(r + i)
                self.c.append(c + k)
                self.flat.append(self.offsets[j] + (k * ds + i if transpose else i * ds + k))
                self.coef.append(coef)

    def incidence(self, r: int, br: _Branch, offsets: NDArray[np.intp], j: int, coef: float) -> None:
        """``coef * N_j (x_a - x_b)`` for the variables at the branch ports in ``offsets``."""
        for v, sign in ((br.a, coef), (br.b, -coef)):
            if offsets[v] >= 0:
                self.ratio(r, int(offsets[v]), j, sign)


class _Pattern:
    """A square sparse matrix ``base + ratio slots`` whose CSC structure is formed once; a matrix for
    other ratios only refills the values."""

    def __init__(self, base: _Triplets, slots: _RatioSlots, n: int) -> None:
        r = np.array(base.r + slots.r, dtype=np.intp)
        c = np.array(base.c + slots.c, dtype=np.intp)
        order = np.lexsort((r, c))
        rs, cs = r[order], c[order]
        first = np.ones(r.size, dtype=bool)
        first[1:] = (rs[1:] != rs[:-1]) | (cs[1:] != cs[:-1])
        pos = np.empty(r.size, dtype=np.intp)
        pos[order] = np.cumsum(first) - 1
        self.n, self.nnz = n, int(first.sum())
        self.indices = rs[first].astype(np.int32)
        self.indptr = np.searchsorted(cs[first], np.arange(n + 1)).astype(np.int32)
        nb = len(base.v)
        self.base = np.bincount(pos[:nb], weights=_numeric_values(base.v), minlength=self.nnz)
        expressions: list[Any] = [0.0] * self.nnz
        for position, value in zip(pos[:nb], base.v):
            expressions[int(position)] += value
        self.expressions = tuple(expressions)
        self.slot_pos, self.slot_flat = pos[nb:], np.array(slots.flat, dtype=np.intp)
        self.slot_coef = np.array(slots.coef, dtype=float)

    def values(self, ratios: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
        if not self.slot_pos.size:
            return self.base
        flat = np.concatenate([N.ravel() for N in ratios])
        return self.base + np.bincount(self.slot_pos, weights=self.slot_coef * flat[self.slot_flat],
                                       minlength=self.nnz)

    def value_bounds(
        self,
        bounds: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Componentwise network-matrix bounds for bounded transformer ratios."""
        if not self.slot_pos.size:
            return self.base, self.base
        lower = np.concatenate([value[0].ravel() for value in bounds])
        upper = np.concatenate([value[1].ravel() for value in bounds])
        first = self.slot_coef * lower[self.slot_flat]
        second = self.slot_coef * upper[self.slot_flat]
        return (
            self.base + np.bincount(
                self.slot_pos, weights=np.minimum(first, second), minlength=self.nnz,
            ),
            self.base + np.bincount(
                self.slot_pos, weights=np.maximum(first, second), minlength=self.nnz,
            ),
        )

    def matrix(self, ratios: Sequence[NDArray[np.float64]]) -> sp.csc_matrix:
        return sp.csc_matrix((self.values(ratios), self.indices, self.indptr), shape=(self.n, self.n))

    def dense(self, ratios: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
        return self.dense_values(self.values(ratios))

    def dense_values(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if not hasattr(self, "_flat"):
            columns = np.repeat(np.arange(self.n), np.diff(self.indptr))
            self._flat = self.indices.astype(np.intp) * self.n + columns
        m = np.zeros(self.n * self.n)
        m[self._flat] = values
        return m.reshape(self.n, self.n)

class _DenseSolve:
    """Direct solve for a small dense system, including a permitted free voltage gauge."""

    __slots__ = ("lu", "piv", "matrix", "inverse")

    def __init__(self, m: NDArray[np.float64], what: str, *, gauge: bool = False) -> None:
        self.lu, self.piv, info = lapack.dgetrf(m)
        self.matrix = self.inverse = None
        if info and gauge:
            self.matrix = m
            self.inverse = np.linalg.pinv(m)
        elif info:
            raise ValueError(f"the {what} is singular in this mode (pivot {info} is zero)")

    def solve(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.inverse is not None:
            solution = self.inverse @ b
            if (b.ndim == 1
                    and np.linalg.norm(self.matrix @ solution - b)
                    > 1e-10 * (1.0 + np.linalg.norm(b))):
                raise ValueError("the network equations are inconsistent in this mode")
            return solution
        return lapack.dgetrs(self.lu, self.piv, b)[0]

    def solve_transpose(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve the transposed system from the same factorization."""
        if self.inverse is not None:
            return self.inverse.T @ b
        return lapack.dgetrs(self.lu, self.piv, b, trans=1)[0]

def _factor(m: sp.spmatrix, what: str) -> object:
    try:
        return splu(sp.csc_matrix(m))
    except RuntimeError as exc:
        raise ValueError(f"the {what} is singular in this mode ({exc})") from None


def _solve_transpose(factor: object, b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve a transposed dense, sparse or grouped factorization."""
    solve = getattr(factor, "solve_transpose", None)
    return solve(b) if solve is not None else factor.solve(b, trans="T")


class _RatioPlan:
    """Low-rank transformer-ratio changes shared by one network structure."""

    def __init__(self, pattern: "_Pattern", base_values: NDArray[np.float64], factor: object) -> None:
        positions = np.unique(pattern.slot_pos)
        columns = np.repeat(np.arange(pattern.n), np.diff(pattern.indptr))
        rows, cols = pattern.indices[positions].astype(np.intp), columns[positions]
        self.positions, self.base = positions, base_values[positions]
        self.rows, self.cols = np.unique(rows), np.unique(cols)
        self.row_index = np.searchsorted(self.rows, rows)
        self.col_index = np.searchsorted(self.cols, cols)
        self.factor = factor
        unit = np.zeros((pattern.n, self.rows.size))
        unit[self.rows, np.arange(self.rows.size)] = 1.0
        self.Z = factor.solve(unit)
        self.ZC = self.Z[self.cols]
        self._Zt: Optional[NDArray[np.float64]] = None
        self.solutions: dict[object, NDArray[np.float64]] = {}

    @property
    def Zt(self) -> NDArray[np.float64]:
        """Return ``M0^-T E_C``, formed only when corrected rows need it."""
        if self._Zt is None:
            unit = np.zeros((self.Z.shape[0], self.cols.size))
            unit[self.cols, np.arange(self.cols.size)] = 1.0
            self._Zt = _solve_transpose(self.factor, unit)
        return self._Zt

    def update(self, values: NDArray[np.float64]) -> Optional["_RatioUpdate"]:
        """Return a stable Woodbury update for ``values``, otherwise request refactorization."""
        change = np.zeros((self.rows.size, self.cols.size))
        np.add.at(
            change,
            (self.row_index, self.col_index),
            values[self.positions] - self.base,
        )
        correction = np.eye(self.cols.size) + self.ZC @ change
        if not np.all(np.isfinite(correction)):
            return None
        try:
            inverse = np.linalg.inv(correction)
        except np.linalg.LinAlgError:
            return None
        condition = (
            np.abs(correction).sum(axis=0).max(initial=0.0)
            * np.abs(inverse).sum(axis=0).max(initial=0.0)
        )
        return None if condition > _RATIO_CONDITION else _RatioUpdate(self, change, inverse)

    def shared(self, key: object, right: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return one structure solve shared by all of its ratio copies."""
        solution = self.solutions.get(key)
        if solution is None:
            solution = self.solutions[key] = self.factor.solve(right)
        return solution

    def change_bound(
        self,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Entrywise absolute bound of the low-rank correction over an interval matrix."""
        size = np.maximum(
            np.abs(lower[self.positions] - self.base),
            np.abs(upper[self.positions] - self.base),
        )
        change = np.zeros((self.rows.size, self.cols.size))
        np.add.at(change, (self.row_index, self.col_index), size)
        return change


class _RatioUpdate:
    """Solve a transformer-ratio copy through its structure's Woodbury correction."""

    __slots__ = ("plan", "change", "inverse", "left")

    def __init__(self, plan: _RatioPlan, change: NDArray[np.float64],
                 inverse: NDArray[np.float64]) -> None:
        self.plan, self.change, self.inverse = plan, change, inverse
        self.left = plan.Z @ change

    def solve(self, right: NDArray[np.float64],
              base: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        value = self.plan.factor.solve(right) if base is None else base
        return value - self.left @ (self.inverse @ value[self.plan.cols])

    def solve_transpose(self, right: NDArray[np.float64],
                        base: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        """Solve the transposed corrected system from the same Woodbury factors."""
        value = _solve_transpose(self.plan.factor, right) if base is None else base
        correction = self.inverse.T @ (self.change.T @ value[self.plan.rows])
        return value - self.plan.Zt @ correction


def _compact(matrix: NDArray[np.float64]) -> NDArray[np.float64] | sp.csr_matrix:
    """Store a large mostly-zero equation matrix sparsely."""
    if (matrix.size > _DENSE_ENTRIES
            and np.count_nonzero(matrix) <= _SPARSE_SHARE * matrix.size):
        return sp.csr_matrix(matrix)
    return matrix


class _AffineEquations:
    """Derivative and output rows, optionally corrected for another transformer ratio."""

    __slots__ = ("matrix", "left", "right")

    def __init__(self, matrix: NDArray[np.float64] | sp.csr_matrix,
                 left: Optional[NDArray[np.float64]] = None,
                 right: Optional[NDArray[np.float64]] = None) -> None:
        self.matrix, self.left, self.right = matrix, left, right

    def __call__(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        result = self.matrix @ values
        if self.left is not None:
            result -= self.left @ (self.right @ values)
        return result


class _Layout:
    """The bordered block-diagonal order of one switch and diode pattern: the groups' rows and columns
    first (each block square), the interface last. Per block its CSC structure and the positions of its
    entries in the pattern's values, so other transformer ratios only gather values."""

    def __init__(self, pattern: _Pattern, row: NDArray[np.intp], col: NDArray[np.intp], n_groups: int) -> None:
        G = n_groups
        rb, cb = np.where(row < 0, G, row), np.where(col < 0, G, col)
        self.row_perm = np.argsort(rb, kind="stable")
        self.col_perm = np.argsort(cb, kind="stable")
        self.row_inv = np.argsort(self.row_perm)
        self.col_inv = np.argsort(self.col_perm)
        counts = np.bincount(rb, minlength=G + 1)
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).tolist()
        self.n_i, self.failed = int(counts[G]), False
        r = pattern.indices.astype(np.intp)
        c = np.repeat(np.arange(pattern.n), np.diff(pattern.indptr))
        pr, pc, br, bc = self.row_inv[r], self.col_inv[c], rb[r], cb[c]
        self.groups = [g for g in range(G) if counts[g]]
        self.blocks: dict[tuple[int, int], tuple] = {}
        for a, b in [(g, h) for g in self.groups for h in (g, G)] + [(G, g) for g in self.groups] + [(G, G)]:
            pos = np.flatnonzero((br == a) & (bc == b))
            lr, lc = pr[pos] - self.offsets[a], pc[pos] - self.offsets[b]
            order = np.lexsort((lr, lc))
            shape = (int(counts[a]), int(counts[b]))
            indptr = np.searchsorted(lc[order], np.arange(shape[1] + 1)).astype(np.int32)
            flat = lr[order] * shape[1] + lc[order]
            self.blocks[a, b] = (pos[order], lr[order].astype(np.int32), indptr, shape, flat)
        self.signature = {g: b"|".join(np.array(self.blocks[key][3]).tobytes() + self.blocks[key][1].tobytes()
                                       + self.blocks[key][2].tobytes() for key in ((g, g), (g, G), (G, g)))
                          for g in self.groups}

    def block(self, values: NDArray[np.float64], a: int, b: int) -> sp.csc_matrix:
        pos, indices, indptr, shape, _flat = self.blocks[a, b]
        return sp.csc_matrix((values[pos], indices, indptr), shape=shape)

    def dense(self, values: NDArray[np.float64], a: int, b: int) -> NDArray[np.float64]:
        pos, _indices, _indptr, shape, flat = self.blocks[a, b]
        m = np.zeros(shape)
        m.flat[flat] = values[pos]
        return m


class _GroupedLU:
    """``M`` eliminated group by group onto the interface: ``S = D - sum_g C_g M_gg^-1 B_g``. A group's
    factors ``(lu, W_g = M_gg^-1 B_g, C_g, C_g W_g)`` come from the netlist's cache when its blocks are
    unchanged (its own mode); a block up to ``_DENSE_GROUP`` unknowns is factorized by LAPACK, a larger
    one by SuperLU. ``solve`` is that of SuperLU (one right-hand side or a matrix of them)."""

    def __init__(self, layout: _Layout, values: NDArray[np.float64], net: "Netlist") -> None:
        G, n_i, stats = len(layout.offsets) - 2, layout.n_i, net._stats
        self.layout, self.i0 = layout, layout.offsets[-2]
        schur = layout.dense(values, G, G) if n_i else None
        self.solves, couplings = [], []
        for g in layout.groups:
            gg, gi, ig = (values[layout.blocks[key][0]] for key in ((g, g), (g, G), (G, g)))
            key = (g, layout.signature[g], gg.tobytes(), gi.tobytes(), ig.tobytes())
            part = net._parts.get(key)
            if part is None:
                n_g, what = layout.blocks[g, g][3][0], f"network group {g}"
                lu = (_DenseSolve(layout.dense(values, g, g), what) if n_g <= _DENSE_GROUP
                      else _factor(layout.block(values, g, g), what))
                if n_i:
                    W = lu.solve(layout.dense(values, g, G))
                    C = layout.dense(values, G, g)
                    part = (lu, W, C, C @ W)
                else:
                    part = (lu, None, None, None)
                net._parts[key] = part
                while len(net._parts) > _GROUP_PARTS:
                    net._parts.popitem(last=False)
                stats["groups"][g] += 1
            else:
                net._parts.move_to_end(key)
                stats["reused"][g] += 1
            if n_i:
                schur -= part[3]
                couplings.append(part)
            self.solves.append((slice(layout.offsets[g], layout.offsets[g + 1]), part[0]))
        self.schur = _DenseSolve(schur, "interface system") if n_i else None
        if n_i:
            stats["interface"] += 1
            self.C = np.hstack([part[2] for part in couplings])  # the interface rows over all group unknowns
            self.W = np.vstack([part[1] for part in couplings])

    def solve(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        layout, i0 = self.layout, self.i0
        bp = b[layout.row_perm]
        zp = np.empty_like(bp)
        for sl, factor in self.solves:
            zp[sl] = factor.solve(bp[sl])
        if self.schur is not None:
            zi = self.schur.solve(bp[i0:] - self.C @ zp[:i0])
            zp[:i0] -= self.W @ zi
            zp[i0:] = zi
        return zp[layout.col_inv]

    def solve_transpose(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve the transposed bordered system from the same group factors."""
        layout, i0 = self.layout, self.i0
        bp = b[layout.col_perm]
        zp = np.empty_like(bp)
        if self.schur is not None:
            zi = self.schur.solve_transpose(bp[i0:] - self.W.T @ bp[:i0])
            group_rhs = bp[:i0] - self.C.T @ zi
            zp[i0:] = zi
        else:
            group_rhs = bp[:i0]
        for sl, factor in self.solves:
            zp[sl] = _solve_transpose(factor, group_rhs[sl])
        return zp[layout.row_inv]


class _Projector:
    """Minimum-energy jump ``x+ = x- + W^-1 G^T mult`` onto the mode's loop or cutset constraints.

    The KKT system ``[[W, 0, G_x^T], [0, 0, G_f^T], [G_x, G_f, 0]]`` has free variables ``f``
    (potentials or primary currents); its multipliers are the impulsive charges or flux linkages.
    """

    def __init__(self, W: sp.spmatrix, G_x: sp.spmatrix, G_f: sp.spmatrix, G_u: sp.spmatrix,
                 index: NDArray[np.intp]) -> None:
        self.n, self.n_free = W.shape[0], G_f.shape[1]
        self.W, self.G_u, self.index = sp.csr_matrix(W), sp.csr_matrix(G_u), index

        def signature(matrix: sp.spmatrix) -> tuple:
            value = sp.csr_matrix(matrix)
            value.sort_indices()
            return value.shape, value.indptr.tobytes(), value.indices.tobytes(), value.data.tobytes()

        self.signature = (tuple(map(int, index)), signature(W), signature(G_x),
                          signature(G_f), signature(G_u))
        kkt = sp.bmat([[W, None, G_x.T], [None, sp.csc_matrix((self.n_free, self.n_free)), G_f.T],
                       [G_x, G_f, None]], format="csc")
        self.lu = _factor(kkt, "projection")
        expressions = getattr(W, "_expressions", None)
        if expressions is not None:
            self.W._expressions = expressions
            self._equations = kkt.toarray().astype(object)
            self._equations[:self.n, :self.n] = expressions
        self._affine: Optional[tuple[NDArray[np.float64], ...]] = None
        self._numeric_projection: Optional[Callable] = None

    def _affine_maps(self) -> tuple[NDArray[np.float64], ...]:
        """Dense state/input maps formed once from the projection factorization."""
        if self._affine is None:
            rows = self.n + self.n_free + self.G_u.shape[0]
            x_rhs = np.zeros((rows, self.n))
            x_rhs[:self.n] = self.W.toarray()
            u_rhs = np.zeros((rows, self.G_u.shape[1]))
            u_rhs[self.n + self.n_free:] = self.G_u.toarray()
            x_map, u_map = self.lu.solve(x_rhs), self.lu.solve(u_rhs)
            multiplier = self.n + self.n_free
            self._affine = (x_map[:self.n], u_map[:self.n],
                            x_map[multiplier:], u_map[multiplier:])
        return self._affine

    def _compile_numeric_projection(self) -> Callable:
        """Compile the small affine projection maps into one allocation-light scalar kernel."""
        px, pu, mx, mu = self._affine_maps()
        names_x = [f"x_{index}" for index in range(self.n)]
        names_u = [f"u[{index}]" for index in range(self.G_u.shape[1])]

        lines = [
            *(f"    {name} = y[{int(index)}]" for name, index in zip(names_x, self.index)),
        ]
        for index, (x_row, u_row) in enumerate(zip(px, pu)):
            expression = " + ".join(
                _linear_terms(x_row, names_x) + _linear_terms(u_row, names_u)
            ) or "0.0"
            lines.append(f"    d_{index} = ({expression}) - x_{index}")
        changes = ", ".join(f"abs(d_{index})" for index in range(self.n))
        references = ", ".join(f"abs(x_{index})" for index in range(self.n))
        lines.extend((
            f"    if max(({changes},)) <= _rtol * max(({references},)):",
            *(f"        d_{index} = 0.0" for index in range(self.n)),
        ))
        for index, (x_row, u_row) in enumerate(zip(mx, mu)):
            expression = " + ".join(
                _linear_terms(x_row, names_x) + _linear_terms(u_row, names_u)
            ) or "0.0"
            lines.append(f"    m_{index} = {expression}")
        projected = ", ".join(f"x_{index} + d_{index}" for index in range(self.n))
        multipliers = ", ".join(f"m_{index}" for index in range(mx.shape[0]))
        W = self.W.toarray()
        loss = " + ".join(
            f"{0.5 * float(W[row, column])!r} * d_{row} * d_{column}"
            for row in range(self.n) for column in range(self.n) if W[row, column] != 0.0
        ) or "0.0"
        lines.append(
            f"    return _array(({projected},)), _array(({multipliers},)), {loss}"
        )
        namespace = {"_array": np.asarray, "_rtol": _NOOP_RTOL}
        exec(compile("def project(y, u):\n" + "\n".join(lines) + "\n",
                     "<peslite state projection>", "exec"), namespace)
        return namespace["project"]

    def __call__(self, y: NDArray[np.float64], u: NDArray[np.float64]
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """The projected states, the multipliers and the dissipated energy."""
        x = y[self.index]
        if (self.n + self.G_u.shape[0]) * (self.n + self.G_u.shape[1]) <= _DENSE_ENTRIES:
            if self._numeric_projection is None:
                self._numeric_projection = self._compile_numeric_projection()
            return self._numeric_projection(y, u)
        sol = self.lu.solve(np.concatenate([self.W @ x, np.zeros(self.n_free), self.G_u @ u]))
        projected, multipliers = sol[:self.n], sol[self.n + self.n_free:]
        dx = projected - x
        if float(np.max(np.abs(dx), initial=0.0)) <= _NOOP_RTOL * float(np.max(np.abs(x), initial=0.0)):
            dx[:] = 0.0
        return x + dx, multipliers, 0.5 * float(dx @ (self.W @ dx))

    def significant(self, y: NDArray[np.float64], x: NDArray[np.float64], scale: float) -> bool:
        """Whether the jump from ``y`` to the projected ``x`` exceeds round-off of the circuit's ``scale``."""
        reference = max(scale, float(np.max(np.abs(y[self.index]), initial=0.0)))
        return float(np.max(np.abs(x - y[self.index]), initial=0.0)) > _MARGIN_RTOL * reference

    def expression_solution(self, y: Sequence[Any], u: Sequence[Any]
                            ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Projected states and multipliers from the KKT factorization used by :meth:`__call__`."""
        if hasattr(self, "_equations"):
            weights = self.W._expressions
            x = np.asarray([y[int(index)] for index in self.index], dtype=object)
            right = [*(weights @ x), *([0.0] * self.n_free),
                     *(self.G_u.toarray() @ np.asarray(u, dtype=object))]
            matrix = self._equations.tolist()
            # The reference shares the lowering interface; elimination retains each coefficient.
            reference = self._equations.copy()
            reference[:self.n, :self.n] = self.W.toarray()
            solution = _symbolic_solve(matrix, right, np.asarray(reference, dtype=float))
            return tuple(solution[:self.n]), tuple(solution[self.n + self.n_free:])
        px, pu, mx, mu = self._affine_maps()
        x = np.asarray([y[int(index)] for index in self.index], dtype=object)
        inputs = np.asarray(u, dtype=object)
        return tuple(px @ x + pu @ inputs), tuple(mx @ x + mu @ inputs)

    def values(self, y: Sequence[Any], u: Sequence[Any]
               ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Projected states and multipliers for numeric or equation values."""
        if all(isinstance(y[int(index)], numbers.Real) for index in self.index) \
                and all(isinstance(value, numbers.Real) for value in u):
            projected, multipliers, _loss = self(np.asarray(y, dtype=float), np.asarray(u, dtype=float))
            return tuple(projected), tuple(multipliers)
        return self.expression_solution(y, u)


class Topology:
    """One compiled mode: the normal tree, the dependent states, the factorized derivative MNA and,
    on demand, the dense :class:`StateSpace` and the projections.

    ``roles``: per branch ``"vsrc"``, ``"short"``, ``"p"``, ``"cap"``, ``"res"``, ``"ind"``,
    ``"isrc"``, ``"state_current"``, ``"s"`` or ``None`` (open);
    ``tree``: whether each branch is a tree branch; ``dependent``: per netlist state.
    """

    def __init__(self, net: "Netlist", mode: Mode, *, matrices: Optional[dict] = None) -> None:
        self.mode = mode
        brs, nv = net._branches, net.n_vertices
        nb = len(brs)
        roles, resistance = _roles(net, mode)
        self.roles = roles
        ratio: dict[int, NDArray[np.float64]] = {}
        partner: dict[int, int] = {}
        tindex: dict[int, int] = {}  # branch -> its transformer
        for j, (tr, N) in enumerate(zip(net._transformers, mode.ratios)):
            p, s = net._ports[id(tr)]
            ratio[p] = ratio[s] = N
            partner[p], partner[s] = s, p
            tindex[p] = tindex[s] = j
        self._tindex = tindex
        self.gauge_singular = False
        active = [k for k in range(nb) if roles[k] is not None]

        # ---------------------------------------------------------------- normal tree
        uf = _UnionFind(nv)
        tree = np.zeros(nb, dtype=bool)
        for k in sorted(active, key=lambda k: _PRIORITY[roles[k]]):
            tree[k] = uf.union(brs[k].a, brs[k].b)
        self.tree = tree
        path = _paths(brs, [k for k in active if tree[k]], nv)
        links = [k for k in active if not tree[k]]
        names = lambda ks: ", ".join(brs[j].label for j in ks)  # noqa: E731
        for k in links:
            if _PRIORITY[roles[k]] == _V:
                raise _Degenerate(f"loop of voltage-type branches only (sources, closed ideal switches, conducting "
                                  f"ideal diodes, transformer primaries, zero resistors): {names([k] + path(k))}",
                                  [k] + path(k))
        for k in active:
            if tree[k] and _PRIORITY[roles[k]] == _I:
                cut = [j for j in links if k in path(j)]
                branches = [k] + cut
                if all(roles[index] == "s" for index in branches):
                    self.gauge_singular = True
                else:
                    raise _Degenerate(f"cutset of current-type branches only (current sources, transformer "
                                      f"secondaries): {names(branches)}", branches, loop=False)

        # ---------------------------------------------------------------- dependent states
        primaries = {k for k in active if roles[k] == "p"}
        vc_tree = {k for k in active if tree[k] and roles[k] in ("vsrc", "short", "cap")}
        vd = set(primaries)  # primaries whose voltage derivative follows from sources and capacitors
        while True:
            drop = {p for p in vd if not set(path(partner[p])) <= vc_tree | vd}
            if not drop:
                break
            vd -= drop
        dep_c = [k for k in links if roles[k] == "cap" and set(path(k)).isdisjoint(primaries - vd)]
        cd = set(primaries)  # primaries whose current derivative follows from inductors and current sources
        resistive_links = [k for k in links if roles[k] in ("cap", "res")]
        while True:
            marked = {j for x in resistive_links + [partner[p] for p in primaries - cd] for j in path(x)}
            if cd.isdisjoint(marked):
                break
            cd -= marked
        dep_l = [k for k in active if tree[k] and roles[k] == "ind" and k not in marked]
        dependent = np.zeros(net.n_x, dtype=bool)
        for k in dep_c + dep_l:
            dependent[brs[k].x:brs[k].x + brs[k].dim] = True
        self.dependent = dependent

        # ---------------------------------------------------------------- unknowns
        dim = net._vertex_dim
        cols = net.n_e
        i_off = np.full(nb, -1, dtype=np.intp)
        xd_off = np.full(nb, -1, dtype=np.intp)
        for k in active:
            i_off[k], cols = cols, cols + brs[k].dim
        for k, br in enumerate(brs):
            if br.x >= 0:
                xd_off[k], cols = cols, cols + br.dim
        forest = [k for k in vc_tree if roles[k] != "p"] + sorted(vd)  # derivative V/C forest
        uf_w = _UnionFind(nv)
        for k in forest:
            uf_w.union(brs[k].a, brs[k].b)
        w_off = np.full(nv, -1, dtype=np.intp)
        self._w_base = cols
        if dep_c:
            for v in range(1, nv):
                if uf_w.find(v) != v:
                    w_off[v], cols = cols, cols + dim[v]
        uf_s = _UnionFind(nv)  # super-nodes: tree branches other than dependent inductors and cd primaries
        cut_tree = set(dep_l) | cd if dep_l else set()
        for k in active:
            if tree[k] and k not in cut_tree:
                uf_s.union(brs[k].a, brs[k].b)
        zd_off: dict[int, int] = {}
        if dep_l:
            for p in sorted(cd):
                zd_off[p], cols = cols, cols + brs[p].dim
        n_z = cols

        # ---------------------------------------------------------------- equations
        M, Ky, Ku, Kdu = _Triplets(), _Triplets(), _Triplets(), _Triplets()
        MN = _RatioSlots([tr.shape for tr in net._transformers])  # the ratio entries of M
        e_off = net._e_off
        row = 0
        kcl = np.full(nv, -1, dtype=np.intp)
        for v in range(1, nv):
            kcl[v], row = row, row + dim[v]
            if uf.find(v) == v:  # reference of a floating sub-network
                M.eye(int(kcl[v]), int(e_off[v]), dim[v])
        for k in active:
            br = brs[k]
            for v, s in ((br.a, 1.0), (br.b, -1.0)):
                if v and uf.find(v) != v:
                    M.eye(int(kcl[v]), int(i_off[k]), br.dim, s)
        u_off = net._u_off
        dep_set = set(dep_c) | set(dep_l)
        spans: list[tuple[int, int, int]] = []  # (first row, end row, branch) of each element's equations
        for k in active:
            br, role, d, start = brs[k], roles[k], brs[k].dim, row
            if role in ("vsrc", "short", "res"):
                M.incidence(row, br, e_off)
                if role == "res":
                    M.block(row, int(i_off[k]), -resistance[k])
                if role == "vsrc":
                    Ku.eye(row, u_off[k], d)
                row += d
            elif role == "isrc":
                M.eye(row, int(i_off[k]), d)
                Ku.eye(row, u_off[k], d)
                row += d
            elif role == "state_current":
                M.eye(row, int(i_off[k]), d)
                Ky.eye(row, br.x, d)
                row += d
                M.eye(row, int(xd_off[k]), d)
                Ku.eye(row, u_off[k], d)
                row += d
            elif role == "p":
                M.incidence(row, br, e_off)
                MN.incidence(row, brs[partner[k]], e_off, tindex[k], -1.0)
                row += d
            elif role == "s":
                M.eye(row, int(i_off[k]), d)
                MN.ratio(row, int(i_off[partner[k]]), tindex[k], br.element.current_scale, transpose=True)
                row += d
            elif role == "cap":
                M.eye(row, int(i_off[k]), d)
                M.block(row, int(xd_off[k]), -br.element.C)
                row += d
                if k not in dep_set:
                    M.incidence(row, br, e_off)
                    Ky.eye(row, br.x, d)
                    row += d
            spans.append((start, row, k))
        for el in net._inductors:
            ks = net._ports[id(el)]
            d, start = brs[ks[0]].dim, row
            for m, k in enumerate(ks):
                M.incidence(row, brs[k], e_off)
                for j, kj in enumerate(ks):
                    M.block(row, int(xd_off[kj]),
                            -el.L[m * d:(m + 1) * d, j * d:(j + 1) * d])
                row += d
            for k in ks:
                if k not in dep_set:
                    M.eye(row, int(i_off[k]), d)
                    Ky.eye(row, brs[k].x, d)
                    row += d
            spans.append((start, row, ks[0]))
        # derivative potentials w on the V/C forest: one row per forest branch and dependent capacitor
        self._wrows: list[tuple[int, int]] = []  # (branch, row offset among the w rows)
        needs = np.zeros(len(net._sources), dtype=bool)
        w0 = c0 = row
        if dep_c:
            w0 = row
            for k in sorted(forest) + dep_c:
                br, role = brs[k], roles[k]
                self._wrows.append((k, row - w0))
                M.incidence(row, br, w_off)
                if role == "vsrc":
                    Kdu.eye(row, u_off[k], br.dim)
                elif role == "cap":
                    M.eye(row, int(xd_off[k]), br.dim, -1.0)
                elif role == "p":
                    MN.incidence(row, brs[partner[k]], w_off, tindex[k], -1.0)
                row += br.dim
            consumers = {uf_w.find(brs[k].a) for k in dep_c} | {uf_w.find(brs[partner[p]].a) for p in vd}
            for j, (src, _) in enumerate(net._sources):
                k = net._ports[id(src)][0]
                needs[j] = isinstance(src, VoltageSource) and uf_w.find(brs[k].a) in consumers
        # derivative KCL at the super-nodes of the L/I forest
        self._cuts: list[tuple[int, int, float]] = []  # (row offset among the cut rows, branch, sign)
        sn_off = np.full(nv, -1, dtype=np.intp)
        if dep_l:
            c0 = row
            for v in range(1, nv):
                if uf_s.find(v) == v and uf.find(v) != v:
                    sn_off[v], row = row - c0, row + dim[v]
            for k in active:
                br = brs[k]
                for v, s in ((br.a, 1.0), (br.b, -1.0)):
                    S = uf_s.find(v)
                    if S == uf_s.find(br.b if s > 0 else br.a) or sn_off[S] < 0:
                        continue
                    r, role = c0 + int(sn_off[S]), roles[k]
                    self._cuts.append((int(sn_off[S]), k, s))
                    if role == "ind":
                        M.eye(r, int(xd_off[k]), br.dim, s)
                    elif role == "p":
                        M.eye(r, zd_off[k], br.dim, s)
                    elif role == "s":
                        MN.ratio(r, zd_off[partner[k]], tindex[k], -s * br.element.current_scale, transpose=True)
                    elif role == "isrc":
                        Kdu.eye(r, u_off[k], br.dim, -s)
                        needs[[j for j, (src, _) in enumerate(net._sources) if src is br.element]] = True
                    elif role == "state_current":
                        M.eye(r, int(xd_off[k]), br.dim, s)
                    else:  # pragma: no cover - excluded by the normal tree
                        raise AssertionError(f"{br.label} crosses an L/I cutset")
        if row != n_z:  # pragma: no cover - the classification guarantees a square system
            raise AssertionError(f"derivative MNA has {row} equations for {n_z} unknowns")
        self.n_z = n_z
        self.needs_rate, self.any_rate = needs, bool(needs.any())
        self.component = [uf.find(v) for v in range(nv)]  # sub-network of each vertex (0: grounded)
        self.blocking = [k for k, br in enumerate(brs) if roles[k] is None and isinstance(br.element, Diode)
                         and self.component[br.a] != self.component[br.b]]
        self._rate_columns = np.zeros(net.n_u, dtype=bool)
        for j, (src, o) in enumerate(net._sources):
            self._rate_columns[o:o + brs[net._ports[id(src)][0]].dim] = needs[j]
        self._pattern = _Pattern(M, MN, n_z)
        self._M = self._values = None
        self._state_equations = net.form != "nodal"
        self._dense = self._state_equations and n_z <= _DENSE_GROUP
        self._layout: Optional[_Layout] = None
        if n_z > _DENSE_GROUP and net._grouping_used():
            row_group, col_group = self._labels(net, active, spans, (i_off, xd_off, zd_off, w_off, kcl),
                                                (forest, uf_w, w0), (cut_tree, uf_s, sn_off, c0))
            G = len(net.grouping().elements)
            r, c = self._pattern.indices, np.repeat(np.arange(n_z), np.diff(self._pattern.indptr))
            rg, cg = row_group[r], col_group[c]  # square group blocks coupled only through the interface
            if (np.array_equal(np.bincount(row_group[row_group >= 0], minlength=G),
                               np.bincount(col_group[col_group >= 0], minlength=G))
                    and not np.any((rg >= 0) & (cg >= 0) & (rg != cg))):
                self._layout = _Layout(self._pattern, row_group, col_group, G)
        self.Ky = sp.csr_matrix(Ky.matrix((n_z, net.n_x)))
        self.Ku = sp.csr_matrix(Ku.matrix((n_z, net.n_u)))
        self.Kdu = sp.csr_matrix(Kdu.matrix((n_z, net.n_u)))
        if self._dense:
            self.M = self._pattern.dense(mode.ratios)
            self.lu = _DenseSolve(self.M, "network", gauge=self.gauge_singular) if n_z else None
        else:
            self.lu = self._factorize(net, self._pattern.values(mode.ratios))
        xd_index = np.zeros(net.n_x, dtype=np.intp)
        out = _Triplets()
        for k, br in enumerate(brs):
            if br.x >= 0:
                xd_index[br.x:br.x + br.dim] = xd_off[k] + np.arange(br.dim)
            out.incidence(net.n_e + br.output, br, e_off)
            if i_off[k] >= 0:
                out.eye(net.n_e + net.n_bd + br.output, int(i_off[k]), br.dim)
        out.eye(0, 0, net.n_e)
        self.xd_index = xd_index
        self.S_out = sp.csr_matrix(out.matrix((net.n_out, n_z)))
        if matrices is not None:
            matrices.update(M=M, y=Ky, u=Ku, du=Kdu)
        self._net_shape = (net.n_x, net.n_u)
        self._couplings()
        self._state_space: Optional[StateSpace] = None
        self._evaluations = 0
        self._affine_equations: dict[tuple[bool, tuple[int, ...]], _AffineEquations] = {}
        self._selected_outputs: dict[tuple[int, ...], NDArray[np.float64]] = {}
        self._projectors: Optional[tuple[Optional[_Projector], Optional[_Projector]]] = None
        self._constraint_key: Optional[object] = None
        self._store = (net, dep_c, dep_l, cd, partner, ratio, w_off, sn_off, uf_s)
        self._shared: dict[str, object] = {"ratio_base": self}
        # no transformer primary on the derivative V/C forest and no secondary across an L/I cutset: the loop and
        # cutset constraints (and their projections) do not depend on the transformer ratios
        self.ratio_free = (all(roles[k] != "p" for k, _o in self._wrows)
                           and all(roles[k] != "s" for _o, k, _s in self._cuts))

    def with_ratios(self, mode: Mode) -> "Topology":
        """This compiled mode with other transformer ratios (same switches and diodes): the
        classification and matrix structure are shared; its matrix is factorized on first use."""
        top = copy.copy(self)
        top.mode = mode
        net, dep_c, dep_l, cd, partner, _ratio, w_off, sn_off, uf_s = self._store
        ratio = {k: mode.ratios[j] for k, j in self._tindex.items()}
        top._store = (net, dep_c, dep_l, cd, partner, ratio, w_off, sn_off, uf_s)
        top._M = top._values = None
        top._lu = _PENDING
        top._state_space = top._projectors = top._constraint_key = None
        top._evaluations = 0
        top._affine_equations = {}
        top._selected_outputs = {}
        top.__dict__.pop("_dynamic_structure_key", None)
        return top

    @property
    def M(self) -> NDArray[np.float64] | sp.csc_matrix:
        """The network matrix of the mode: an array in the dense form, CSC (formed on demand) in the sparse."""
        if self._M is None:
            p = self._pattern
            if self._values is None:
                self._values = p.values(self.mode.ratios)
            self._M = (p.dense_values(self._values) if self._dense else
                       sp.csc_matrix((self._values, p.indices, p.indptr), shape=(p.n, p.n)))
        return self._M

    @M.setter
    def M(self, value: NDArray[np.float64] | sp.csc_matrix) -> None:
        self._M = value

    @property
    def lu(self) -> object:
        """Network factors or a stable low-rank update of the structure's factors."""
        if self._lu is _PENDING:
            values = (self._values if self._values is not None else
                      self._pattern.values(self.mode.ratios))
            # A Woodbury solve pays off for large or numeric networks; small device modes refactor.
            worth = self._pattern.n > _DENSE_GROUP or self._store[0].form != "device"
            plan = self._ratio_plan() if worth else None
            update = plan.update(values) if plan is not None else None
            if update is not None:
                self._values, self._M, self._lu = values, None, update
            elif self._dense:
                self._values = values
                self._M = self._pattern.dense_values(values)
                self._lu = (_DenseSolve(self._M, "network", gauge=self.gauge_singular)
                            if self.n_z else None)
            else:
                self._lu = self._factorize(self._store[0], values)
        return self._lu

    @lu.setter
    def lu(self, value: object) -> None:
        self._lu = value

    def _ratio_plan(self) -> Optional[_RatioPlan]:
        """Return the structure's shared low-rank ratio plan when its factors are ready."""
        cached = self._shared.get("ratio_plan", _PENDING)
        if cached is not _PENDING:
            return cached
        base = self._shared["ratio_base"]
        factor = getattr(base, "_lu", _PENDING)
        if factor is _PENDING:
            return None
        pattern = self._pattern
        eligible = (
            factor is not None
            and pattern.slot_pos.size
            and getattr(factor, "inverse", None) is None
        )
        plan = _RatioPlan(pattern, pattern.values(base.mode.ratios), factor) if eligible else None
        if plan is not None and max(plan.rows.size, plan.cols.size) > max(
                4, _RATIO_RANK * pattern.n):
            plan = None
        self._shared["ratio_plan"] = plan
        return plan

    def state_matrix_norm_bound(self) -> Optional[float]:
        """Return a cheap upper bound on the state matrix's spectral radius.

        The bound is the 2-norm of a diagonal similarity of the matrix. A transformer-ratio copy
        uses the structure's Woodbury factors, avoiding the dense state matrix while remaining
        conservative. An unavailable low-rank update asks the caller for the exact matrix.
        """
        base = self._shared["ratio_base"]
        scale = self._stability_scale()
        if self is base or not self._pattern.slot_pos.size:
            return self._base_scaled_norm()
        _ = base.lu
        plan = self._ratio_plan()
        if plan is None:
            return None
        update = plan.update(self._pattern.values(self.mode.ratios))
        if update is None:
            return None
        if "K" not in self._shared:
            self._shared["K"] = sp.hstack([self.Ky, self.Ku, self.Kdu]).toarray()
        n_x = self._net_shape[0]
        base_solution = plan.shared("state_matrix_bound", self._shared["K"])
        left = update.left[self.xd_index]
        right = update.inverse @ base_solution[plan.cols, :n_x]
        return (self._base_scaled_norm()
                + _scaled_norm(left, left=scale) * _scaled_norm(right, right=scale))

    def _stability_scale(self) -> NDArray[np.float64]:
        """Return the structure's cached diagonal similarity for stability bounds."""
        scale = self._shared.get("stability_scale")
        if scale is None:
            scale = self._shared["stability_scale"] = _symmetrizing_scale(
                self._shared["ratio_base"].state_space.A
            )
        return scale

    def _base_scaled_norm(self) -> float:
        """Return ``|diag(d) A diag(d)^-1|_2`` for the structure's base matrix."""
        norm = self._shared.get("base_scaled_norm")
        if norm is None:
            scale = self._stability_scale()
            norm = self._shared["base_scaled_norm"] = _scaled_norm(
                self._shared["ratio_base"].state_space.A, scale, scale
            )
        return norm

    def _state_matrix_range_certificate(self) -> Optional[tuple[object, float]]:
        """Certify one spectral-radius bound over every declared transformer ratio.

        The interval matrix bounds the low-rank ratio correction entry by entry.  If its
        interaction with the base inverse is below one, the Neumann bound proves that every
        matrix in the range is nonsingular and bounds its state matrix in one diagonal similarity.
        """
        base = self._shared["ratio_base"]
        cached = self._shared.get("state_matrix_range_certificate", _PENDING)
        if cached is not _PENDING:
            return cached
        net = base._store[0]
        bounds = tuple(transformer.bounds for transformer in net._transformers)
        if any(bound is None for bound in bounds):
            certificate = None
        elif (not base._pattern.slot_pos.size
              or all(np.array_equal(lower, upper) for lower, upper in bounds)):
            certificate = (base, self._base_scaled_norm())
        else:
            _ = base.lu
            plan = base._ratio_plan()
            certificate = None
            if plan is not None:
                lower, upper = base._pattern.value_bounds(bounds)
                change = plan.change_bound(lower, upper)
                change_norm = float(np.linalg.norm(change, "fro"))
                interaction = float(np.linalg.norm(plan.ZC, 2)) * change_norm
                if interaction < 1.0:
                    if "K" not in self._shared:
                        self._shared["K"] = sp.hstack([
                            base.Ky, base.Ku, base.Kdu,
                        ]).toarray()
                    n_x = base._net_shape[0]
                    solution = plan.shared(
                        "state_matrix_bound", self._shared["K"]
                    )
                    scale = self._stability_scale()
                    correction = (
                        _scaled_norm(plan.Z[base.xd_index], left=scale)
                        * change_norm / (1.0 - interaction)
                        * _scaled_norm(solution[plan.cols, :n_x], right=scale)
                    )
                    bound = self._base_scaled_norm() + correction
                    if math.isfinite(bound):
                        certificate = (base, bound)
        self._shared["state_matrix_range_certificate"] = certificate
        return certificate

    def range_stability_certificate(
        self, ratios: Sequence[NDArray[np.float64]]
    ) -> Optional[tuple[object, float]]:
        """Return the structure's range certificate when it covers ``ratios``."""
        for transformer, ratio in zip(self._store[0]._transformers, ratios):
            if not transformer.within_bounds(ratio):
                return None
        return self._state_matrix_range_certificate()

    def state_matrix_stability_certificate(self) -> Optional[tuple[object, float]]:
        """Return a reusable range certificate, or a bound for this exact ratio mode."""
        certificate = self.range_stability_certificate(self.mode.ratios)
        if certificate is not None:
            return certificate
        bound = self.state_matrix_norm_bound()
        return None if bound is None else (self.state_matrix_key(), bound)

    def _factorize(self, net: "Netlist", values: NDArray[np.float64]) -> object:
        """The factors of the sparse network matrix with entries ``values``: grouped onto the interface
        when this pattern has a block layout, else (or when a group's block is singular) one SuperLU."""
        self._values, self._M = values, None
        if not self.n_z:
            return None
        layout = self._layout
        if layout is not None and not layout.failed:
            try:
                return _GroupedLU(layout, values, net)
            except ValueError:  # a group cannot take its interface voltages as given: solve it whole
                layout.failed = True
        net._stats["network"] += 1
        if self.n_z <= _DENSE_GROUP:  # a small system: LAPACK is faster than SuperLU
            return _DenseSolve(
                self._pattern.dense_values(values), "network", gauge=self.gauge_singular
            )
        if self.gauge_singular:
            return _DenseSolve(self._pattern.dense_values(values), "network", gauge=True)
        return _factor(self.M, "network")

    def _labels(self, net: "Netlist", active: list[int], spans: list[tuple[int, int, int]], offsets: tuple,
                forest_data: tuple, cut_data: tuple) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """The group (-1: interface) of every row and unknown of the derivative MNA.

        Node voltages and node equations follow their node, branch unknowns and equations their element.
        Each equation is paired with an unknown of the same group: the row of a V/C-forest branch with
        the derivative potential of its end away from the forest's root, a super-node's derivative KCL
        with the derivative of the cutset branch towards its parent super-node; a super-node equation
        that crosses into another group goes to the interface together with that derivative."""
        brs, n, dim, e_off = net._branches, self.n_z, net._vertex_dim, net._e_off
        i_off, xd_off, zd_off, w_off, kcl = offsets
        vertex, branch = net._labels
        row, col = np.full(n, -1, dtype=np.intp), np.full(n, -1, dtype=np.intp)
        for v in range(1, net.n_vertices):
            col[e_off[v]:e_off[v] + dim[v]] = row[kcl[v]:kcl[v] + dim[v]] = vertex[v]
            if w_off[v] >= 0:
                col[w_off[v]:w_off[v] + dim[v]] = vertex[v]
        for k in active:
            col[i_off[k]:i_off[k] + brs[k].dim] = branch[k]
        for k in np.flatnonzero(xd_off >= 0):
            col[xd_off[k]:xd_off[k] + brs[k].dim] = branch[k]
        for k, o in zd_off.items():
            col[o:o + brs[k].dim] = branch[k]
        for start, stop, k in spans:
            row[start:stop] = branch[k]
        forest, uf_w, w0 = forest_data
        if self._wrows:
            roots = [v for v in range(net.n_vertices) if uf_w.find(v) == v]
            child = _orient([(k, brs[k].a, brs[k].b) for k in forest], roots)
            for k, o in self._wrows:
                row[w0 + o:w0 + o + brs[k].dim] = vertex[child[k]] if k in child else branch[k]
        cut_tree, uf_s, sn_off, c0 = cut_data
        if self._cuts:
            roots = [v for v in range(net.n_vertices) if uf_s.find(v) == v and sn_off[v] < 0]
            child = _orient([(k, uf_s.find(brs[k].a), uf_s.find(brs[k].b)) for k in cut_tree], roots)
            crossing: dict[int, set[int]] = {}
            for o, k, _s in self._cuts:
                crossing.setdefault(o, set()).add(int(branch[k]))
            for k, S in child.items():
                o, g, d = int(sn_off[S]), int(branch[k]), dim[S]
                if crossing.get(o, set()) - {g, -1}:
                    row[c0 + o:c0 + o + d] = -1
                    x = int(xd_off[k]) if xd_off[k] >= 0 else zd_off[k]
                    col[x:x + brs[k].dim] = -1
                else:
                    row[c0 + o:c0 + o + d] = g
        return row, col

    def _couplings(self) -> None:
        """One right-hand-side product ``K [x; u; du]`` (without ``du`` when no source rate is needed) and
        the output selection, dense while small."""
        K = sp.hstack([self.Ky, self.Ku] + ([self.Kdu] if self.any_rate else []), format="csr")
        small = K.shape[0] * K.shape[1] <= _DENSE_ENTRIES
        self._K = K.toarray() if small else K
        self._S = self.S_out.toarray() if self.S_out.shape[0] * self.S_out.shape[1] <= _DENSE_ENTRIES else self.S_out

    def solve(self, x: NDArray[np.float64], u: NDArray[np.float64], du: NDArray[np.float64]) -> NDArray[np.float64]:
        """The derivative-MNA unknowns ``z`` for states ``x``, inputs ``u`` and their rates ``du``."""
        if not self.n_z:
            return np.zeros(0)
        return self.lu.solve(self._K @ (np.concatenate((x, u, du)) if self.any_rate else np.concatenate((x, u))))

    @property
    def state_space(self) -> StateSpace:
        """The dense state space of the mode, formed from the same factorization on first use."""
        if self._state_space is None:
            n_x, n_u = self._net_shape
            if "K" not in self._shared:
                self._shared["K"] = sp.hstack([self.Ky, self.Ku, self.Kdu]).toarray()
            K = self._shared["K"]
            lu = self.lu
            if isinstance(lu, _RatioUpdate):
                T = lu.solve(K, lu.plan.shared("state_space", K))
            else:
                T = lu.solve(K) if self.n_z else np.zeros((0, K.shape[1]))
            T[:, n_x + n_u:][:, ~self._rate_columns] = 0.0
            X, O = T[self.xd_index], self.S_out @ T
            a, b = slice(0, n_x), slice(n_x, n_x + n_u)
            c = slice(n_x + n_u, n_x + 2 * n_u)
            self._state_space = StateSpace(X[:, a], X[:, b], X[:, c], O[:, a], O[:, b], O[:, c])
        return self._state_space

    def affine_equations(self, selected: Sequence[int], *, derivatives: bool = True
                         ) -> _AffineEquations:
        """Cached derivative and output rows over ``[x, u, du]``."""
        rows = tuple(map(int, selected))
        key = derivatives, rows
        equations = self._affine_equations.get(key)
        if equations is not None:
            return equations
        base = self._shared["ratio_base"]
        update = self.lu if base is not self else None
        if isinstance(update, _RatioUpdate):
            matrix = base.affine_equations(rows, derivatives=derivatives).matrix
            selected_rows, coupling = self._ratio_rows(rows, derivatives, update.plan)
            left = selected_rows @ (update.change @ update.inverse)
            if matrix.shape[0] * matrix.shape[1] <= _DENSE_ENTRIES:
                dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
                equations = _AffineEquations(dense - left @ coupling)
            else:
                equations = _AffineEquations(matrix, left, coupling)
        elif self._state_equations:
            state_space = self.state_space
            blocks = [np.hstack((state_space.A, state_space.B, state_space.B1))] \
                if derivatives else []
            if rows:
                output_rows = list(rows)
                blocks.append(np.hstack((state_space.C[output_rows],
                                         state_space.D[output_rows],
                                         state_space.D1[output_rows])))
            width = self._net_shape[0] + 2 * self._net_shape[1]
            matrix = np.vstack(blocks) if blocks else np.zeros((0, width))
            equations = _AffineEquations(_compact(matrix))
        elif derivatives:
            raise ValueError("nodal affine equations apply only to electrical outputs")
        elif not rows or not self.n_z:
            n_x, n_u = self._net_shape
            equations = _AffineEquations(np.zeros((len(rows), n_x + 2 * n_u)))
        else:
            selected = self.selected_outputs(rows)
            right = selected.T.toarray() if sp.issparse(selected) else selected.T
            dual = _solve_transpose(self.lu, np.asarray(right))
            equations = _AffineEquations(
                _compact(np.asarray(self._all_couplings().T @ dual).T)
            )
        self._affine_equations[key] = equations
        return equations

    def _all_couplings(self) -> sp.csr_matrix:
        """Return the shared right-hand side map over ``[x, u, du]``."""
        couplings = self._shared.get("couplings")
        if couplings is None:
            couplings = self._shared["couplings"] = sp.hstack(
                [self.Ky, self.Ku, self.Kdu], format="csr"
            )
        return couplings

    def _ratio_rows(self, rows: tuple[int, ...], derivatives: bool,
                    plan: _RatioPlan) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the structure rows and couplings needed by every ratio correction."""
        key = "ratio_rows", derivatives, rows
        shared = self._shared.get(key)
        if shared is None:
            parts = [plan.Z[self.xd_index]] if derivatives else []
            if rows:
                parts.append(np.asarray(self.selected_outputs(rows) @ plan.Z))
            selected = np.vstack(parts) if parts else np.zeros((0, plan.Z.shape[1]))
            coupling = np.asarray(self._all_couplings().T @ plan.Zt).T
            shared = self._shared[key] = selected, coupling
        return shared

    def input_reach(
        self,
    ) -> Optional[tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]]:
        """Return structural input reach and which transformer ratios can change the state matrix."""
        reach = self._shared.get("input_reach", _PENDING)
        if reach is not _PENDING:
            return reach
        net, pattern = self._store[0], self._pattern
        n = pattern.n
        rows = pattern.indices.astype(np.intp)
        columns = np.repeat(np.arange(n), np.diff(pattern.indptr))
        matching = csgraph.maximum_bipartite_matching(
            sp.csr_matrix((np.ones(rows.size), (rows, columns)), shape=(n, n)),
            perm_type="column",
        )
        reach = None
        if not np.any(matching < 0):
            graph = sp.csr_matrix(
                (np.ones(rows.size), (columns, matching[rows])),
                shape=(n + 1, n + 1),
            )

            def reached(equations: NDArray[np.intp]) -> NDArray[np.bool_]:
                seeds = np.unique(matching[equations])
                root = sp.csr_matrix(
                    (np.ones(seeds.size), (np.full(seeds.size, n), seeds)),
                    shape=(n + 1, n + 1),
                )
                found = np.zeros(n + 1, dtype=bool)
                found[csgraph.breadth_first_order(
                    graph + root, n, directed=True, return_predecessors=False
                )] = True
                return found[:n]

            Ku, Kdu = self.Ku.tocsc(), self.Kdu.tocsc()
            entered = []
            for source, offset in net._sources:
                stop = offset + net._branch(source).dim
                entered.append(np.concatenate((
                    Ku.indices[Ku.indptr[offset]:Ku.indptr[stop]],
                    Kdu.indices[Kdu.indptr[offset]:Kdu.indptr[stop]],
                )))
            state_rows = self.Ky.tocoo().row.astype(np.intp)
            state_live = reached(state_rows)
            live = reached(np.concatenate([state_rows, *entered]).astype(np.intp))
            owner = np.repeat(
                np.arange(len(self.mode.ratios)),
                [np.size(ratio) for ratio in self.mode.ratios],
            )[pattern.slot_flat]
            slot_rows = rows[pattern.slot_pos]
            slot_live = live[columns[pattern.slot_pos]]
            state_slot_live = state_live[columns[pattern.slot_pos]]
            entered.extend(
                slot_rows[(owner == index) & slot_live]
                for index in range(len(self.mode.ratios))
            )
            outputs = abs(self.S_out).tocsr()
            derivatives = np.zeros(len(entered), dtype=bool)
            reached_outputs = np.zeros((len(entered), self.S_out.shape[0]), dtype=bool)
            for index, equations in enumerate(entered):
                if equations.size:
                    found = reached(equations)
                    derivatives[index] = bool(found[self.xd_index].any())
                    reached_outputs[index] = outputs @ found.astype(float) > 0.0
            state_ratios = np.zeros(len(self.mode.ratios), dtype=bool)
            for index in range(len(state_ratios)):
                equations = slot_rows[(owner == index) & state_slot_live]
                if equations.size:
                    state_ratios[index] = bool(reached(equations)[self.xd_index].any())
            reach = derivatives, reached_outputs, state_ratios
        self._shared["input_reach"] = reach
        return reach

    def state_matrix_key(self, mode: Optional[Mode] = None) -> tuple:
        """Mode identity containing only transformer ratios that can change ``A``.

        ``mode`` may be another ratio mode of this structure; the reach is shared by them.
        """
        mode = self.mode if mode is None else mode
        reach = self.input_reach()
        relevant = (np.ones(len(mode.ratios), dtype=bool)
                    if reach is None else reach[2])
        ratios = b"".join(
            ratio.tobytes() for ratio, used in zip(mode.ratios, relevant) if used
        )
        return mode.closed, mode.conducting, ratios

    def selected_outputs(self, selected: Sequence[int]) -> NDArray[np.float64]:
        """Cached ratio-independent rows of the network output map."""
        key = tuple(map(int, selected))
        output = self._selected_outputs.get(key)
        if output is None:
            shared = self._shared.setdefault("selected_outputs", {})
            output = shared.get(key)
            if output is None:
                output = shared[key] = self._S[list(key)]
            self._selected_outputs[key] = output
        return output

    def projectors(self) -> tuple[Optional[_Projector], Optional[_Projector]]:
        """The capacitor (charge) and inductor (flux) projections, ``None`` where nothing is dependent; shared
        by the modes that differ only in transformer ratios when the constraints do not involve them."""
        if self._projectors is None:
            shared = self._shared.get("projectors") if self.ratio_free else None
            self._projectors = shared or (self._capacitor_projector(), self._inductor_projector())
            if self.ratio_free:
                self._shared["projectors"] = self._projectors
        return self._projectors

    def constraint_key(self) -> object:
        """Exact structural identity of the charge and flux projection equations."""
        if self._constraint_key is None:
            signature = tuple(None if projector is None else projector.signature
                              for projector in self.projectors())
            net = self._store[0]
            key = net._constraints.get(signature)
            if key is None:
                key = object()
                net._constraints[signature] = key
                while len(net._constraints) > net.cache_size:
                    net._constraints.popitem(last=False)
            else:
                net._constraints.move_to_end(signature)
            self._constraint_key = key
        return self._constraint_key

    def _capacitor_projector(self) -> Optional[_Projector]:
        net, dep_c, _l, _cd, partner, ratio, w_off, _sn, _uf = self._store
        if not dep_c:
            return None
        brs = net._branches
        caps = [k for k, br in enumerate(brs) if self.roles[k] == "cap"]
        cx = {k: o for k, o in zip(caps, np.cumsum([0] + [brs[k].dim for k in caps]).tolist())}
        n_c = sum(brs[k].dim for k in caps)
        W = _block_weights([(brs[k].scale, brs[k].element.C) for k in caps])
        n_w = int(np.sum([net._vertex_dim[v] for v in np.flatnonzero(w_off >= 0)]))
        f_off = np.where(w_off >= 0, w_off - self._w_base, -1)
        Gx, Gf, Gu = _Triplets(), _Triplets(), _Triplets()
        for k, r in self._wrows:
            br, role = brs[k], self.roles[k]
            Gf.incidence(r, br, f_off)
            if role == "cap":
                Gx.eye(r, cx[k], br.dim, -1.0)
            elif role == "vsrc":
                Gu.eye(r, net._u_off[k], br.dim)
            elif role == "p":
                Gf.incidence(r, brs[partner[k]], f_off, -ratio[k])
        n_l = self._wrows[-1][1] + brs[self._wrows[-1][0]].dim
        index = np.concatenate([net._branches[k].element.state for k in caps])
        return _Projector(W, Gx.matrix((n_l, n_c)), Gf.matrix((n_l, n_w)), Gu.matrix((n_l, net.n_u)), index)

    def _inductor_projector(self) -> Optional[_Projector]:
        net, _c, dep_l, cd, partner, ratio, _w, sn_off, _uf = self._store
        if not dep_l:
            return None
        brs = net._branches
        inds = [k for k, br in enumerate(brs) if self.roles[k] == "ind"]
        lx = {k: o for k, o in zip(inds, np.cumsum([0] + [brs[k].dim for k in inds]).tolist())}
        n_l = sum(brs[k].dim for k in inds)
        W = _block_weights([(brs[net._ports[id(el)][0]].scale, el.L) for el in net._inductors])
        prims = sorted(cd)
        po = {p: o for p, o in zip(prims, np.cumsum([0] + [brs[p].dim for p in prims]).tolist())}
        n_p = sum(brs[p].dim for p in prims)
        n_s = int(sum(net._vertex_dim[v] for v in np.flatnonzero(sn_off >= 0)))
        Gx, Gf, Gu = _Triplets(), _Triplets(), _Triplets()
        for r, k, s in self._cuts:
            br, role = brs[k], self.roles[k]
            if role == "ind":
                Gx.eye(r, lx[k], br.dim, s)
            elif role == "p":
                Gf.eye(r, po[k], br.dim, s)
            elif role == "s":
                Gf.block(r, po[partner[k]], -s * br.element.current_scale * ratio[k].T)
            else:
                Gu.eye(r, net._u_off[k], br.dim, -s)
        index = np.concatenate([el.state for el in net._inductors])
        # cutsets Q x + Q_p i_p = c with W dx = Q^T phi: the multipliers are impulsive super-node fluxes
        return _Projector(W, -Gx.matrix((n_s, n_l)), -Gf.matrix((n_s, n_p)), -Gu.matrix((n_s, net.n_u)), index)


def _roles(net: "Netlist", mode: Mode) -> tuple[list[Optional[str]], dict[int, NDArray[np.float64]]]:
    closed = {id(s): c for s, c in zip(net._switches, mode.closed)}
    conducting = {id(d): c for d, c in zip(net._diodes, mode.conducting)}
    roles: list[Optional[str]] = []
    resistance: dict[int, NDArray[np.float64]] = {}
    for k, br in enumerate(net._branches):
        el = br.element
        if isinstance(el, Resistor):
            role = "res" if el.R.any() else "short"
            resistance[k] = el.R
        elif isinstance(el, (Switch, Diode)):
            on = closed[id(el)] if isinstance(el, Switch) else conducting[id(el)]
            r = el.R_on if on else el.R_off
            role = "short" if r == 0.0 else None if r == math.inf else "res"
            if role == "res":
                resistance[k] = np.diag(np.asarray([r] * br.dim, dtype=object))
        elif isinstance(el, Transformer):
            role = "p" if br.port == 0 else "s"
        else:
            role = {Inductor: "ind", CoupledInductors: "ind", Capacitor: "cap", VoltageSource: "vsrc",
                    CurrentSource: "isrc", CurrentState: "state_current"}[type(el)]
        roles.append(role)
    return roles, resistance


def _paths(brs: list[_Branch], tree: list[int], nv: int) -> Callable[[int], list[int]]:
    """``path(k)``: the tree branches between the ports of branch ``k`` (its fundamental loop)."""
    adjacent: list[list[tuple[int, int]]] = [[] for _ in range(nv)]
    for k in tree:
        adjacent[brs[k].a].append((brs[k].b, k))
        adjacent[brs[k].b].append((brs[k].a, k))
    up, via, depth = list(range(nv)), [-1] * nv, [-1] * nv
    for root in range(nv):
        if depth[root] >= 0:
            continue
        depth[root], stack = 0, [root]
        while stack:
            v = stack.pop()
            for w, k in adjacent[v]:
                if depth[w] < 0:
                    up[w], via[w], depth[w] = v, k, depth[v] + 1
                    stack.append(w)
    cache: dict[int, list[int]] = {}

    def path(k: int) -> list[int]:
        if k not in cache:
            a, b, out = brs[k].a, brs[k].b, []
            while a != b:
                if depth[a] >= depth[b]:
                    out.append(via[a])
                    a = up[a]
                else:
                    out.append(via[b])
                    b = up[b]
            cache[k] = out
        return cache[k]

    return path


def _orient(edges: list[tuple[int, int, int]], roots: Sequence[int]) -> dict[int, int]:
    """Per edge ``(k, a, b)`` of a forest whose trees each hold one of ``roots``: its end away from the root."""
    adjacent: dict[int, list[tuple[int, int]]] = {}
    for k, a, b in edges:
        adjacent.setdefault(a, []).append((b, k))
        adjacent.setdefault(b, []).append((a, k))
    seen, child = set(roots), {}
    stack = [r for r in roots if r in adjacent]
    while stack:
        v = stack.pop()
        for w, k in adjacent[v]:
            if w not in seen:
                seen.add(w)
                child[k] = w
                stack.append(w)
    return child


def _is_device(el: Element) -> bool:
    """A switch, a diode or a transformer with a mode-dependent ratio."""
    return isinstance(el, (Switch, Diode)) or (isinstance(el, Transformer) and el.constant is None)


def _partition(net: "Netlist") -> tuple[Grouping, NDArray[np.intp], NDArray[np.intp]]:
    """The state-space-nodal groups of ``net``, with the group of every vertex and branch (-1: interface).

    Switches, diodes and transformers joined at their ports form clusters; a cluster with a diode or a
    mode-dependent transformer (a converter's legs or bridge, not a lone breaker) seeds a region. Regions
    grow breadth-first over the other elements, one element per round, never through ground: a node
    reached by two regions in one round, or the busier end of an element joining two regions, is an
    interface node and stops the growth. The groups are the parts the interface nodes separate: one per
    part holding a seed, and one passive group for the rest; elements between interface nodes and ground
    belong to the interface.
    """
    brs, nv, els = net._branches, net.n_vertices, net.elements
    ends = [sorted({v for k in net._ports[id(el)] for v in (brs[k].a, brs[k].b) if v}) for el in els]
    incident: list[list[int]] = [[] for _ in range(nv)]
    for j, vs in enumerate(ends):
        for v in vs:
            incident[v].append(j)
    uf = _UnionFind(nv)
    joint = [j for j, el in enumerate(els) if isinstance(el, (Switch, Diode, Transformer)) and ends[j]]
    for j in joint:
        for v in ends[j][1:]:
            uf.union(ends[j][0], v)
    seeds = {r: s for s, r in enumerate(sorted({uf.find(ends[j][0]) for j in joint
                                                if _is_device(els[j]) and not isinstance(els[j], Switch)}))}
    owner, shared = np.full(nv, -1, dtype=np.intp), np.zeros(nv, dtype=bool)
    cluster = {j for j in joint if uf.find(ends[j][0]) in seeds}
    for j in cluster:
        owner[ends[j]] = seeds[uf.find(ends[j][0])]
    frontier = np.flatnonzero(owner >= 0).tolist()
    while frontier:
        claims: dict[int, set[int]] = {}
        for u in frontier:
            for j in incident[u]:
                if shared[u]:
                    break
                if j in cluster:
                    continue
                for w in ends[j]:
                    if w == u or shared[w]:
                        continue
                    if owner[w] < 0:
                        claims.setdefault(w, set()).add(int(owner[u]))
                    elif owner[w] != owner[u]:
                        x = max((len(incident[u]), u), (len(incident[w]), w))[1]
                        shared[x], owner[x] = True, -1
                        if x == u:
                            break
        frontier = []
        for w, regions in claims.items():
            if len(regions) > 1:
                shared[w] = True
            elif not shared[w]:
                owner[w] = regions.pop()
                frontier.append(w)
    while True:  # the parts the interface separates; an interface node bordering one group only joins it
        part = _UnionFind(nv)
        for vs in ends:
            inner = [v for v in vs if not shared[v]]
            for v in inner[1:]:
                part.union(inner[0], v)
        of = [next((part.find(v) for v in vs if not shared[v]), -1) for vs in ends]
        first: dict[int, int] = {}
        for j, p in enumerate(of):
            first.setdefault(p, j)
        switching = sorted({of[j] for j in cluster if of[j] >= 0}, key=first.__getitem__)
        index = {p: g for g, p in enumerate(switching)}
        element = np.array([index.get(p, len(switching)) if p >= 0 else -1 for p in of], dtype=np.intp)
        lone = [v for v in np.flatnonzero(shared) if len(set(element[incident[v]].tolist()) - {-1}) <= 1]
        if not lone:
            break
        shared[lone] = False
    passive = bool(np.any(element == len(switching)))
    G = len(switching) + passive
    vertex = np.full(nv, -1, dtype=np.intp)
    for v in range(1, nv):
        if not shared[v]:
            vertex[v] = index.get(part.find(v), len(switching))
    branch = np.empty(len(brs), dtype=np.intp)
    for j, el in enumerate(els):
        branch[net._ports[id(el)]] = element[j]
    node = lambda v: net._nodes[v - 1]  # noqa: E731
    members = [tuple(el for el, g in zip(els, element) if g == h) for h in range(G)]
    grouping = Grouping(
        elements=tuple(members),
        nodes=tuple(tuple(node(v) for v in range(1, nv) if vertex[v] == h) for h in range(G)),
        devices=tuple(sum(_is_device(el) for el in m) for m in members),
        boundary=tuple(tuple(node(v) for v in range(1, nv) if shared[v] and any(element[j] == h for j in incident[v]))
                       for h in range(G)),
        passive=passive, interface=tuple(node(v) for v in range(1, nv) if shared[v]),
        interface_elements=tuple(el for el, g in zip(els, element) if g < 0), used=len(switching) >= 2)
    return grouping, vertex, branch


@dataclass
class _Evaluation:
    topology: Topology
    t: Any
    x: NDArray[np.float64]
    u: NDArray[np.float64]
    du: NDArray[np.float64]
    z: Optional[NDArray[np.float64]]
    output: Optional[NDArray[np.float64]] = None


class Netlist:
    """Primitives between nodes, evaluated as ``dy = evaluate(t, y)`` in the current mode.

    ``form``: ``"device"`` (straight-line scalar equations generated per component), ``"statewise"``
    (whole-circuit per-mode state space) or ``"nodal"`` (factorized derivative MNA, one
    back-substitution per evaluation); all give the same result to round-off.
    ``cache``: compiled modes kept (least recently used first out). ``grouped``: let the sparse form
    eliminate the automatic groups (:meth:`grouping`) onto their interface when that helps.
    """

    def __init__(self, form: str = "nodal", cache: int = _CACHE, grouped: bool = True) -> None:
        if form not in FORMS:
            raise ValueError(f"unknown netlist form {form!r}; expected one of {FORMS}")
        self.form, self.cache_size, self.grouped = form, int(cache), bool(grouped)
        self._plan: Optional[tuple[Grouping, NDArray[np.intp], NDArray[np.intp]]] = None
        self._parts: "OrderedDict[tuple, tuple]" = OrderedDict()  # factorized group blocks by content
        self._constraints: "OrderedDict[tuple, object]" = OrderedDict()
        self._stats: dict[str, Any] = {"network": 0, "interface": 0, "groups": [], "reused": []}
        self.elements: list[Element] = []
        self._branches: list[_Branch] = []
        self._ports: dict[int, list[int]] = {}
        self._vertex: dict[int, int] = {}
        self._nodes: list[Node] = []
        self._vertex_dim, self._e_off = [0], np.full(1, -1, dtype=np.intp)
        self._sources: list[tuple[Element, int]] = []
        self._source_slots: list[tuple[Value, int, int, str]] = []  # value, offset, dimension, name
        self._u_off: dict[int, int] = {}
        self._capacitors: list[Capacitor] = []
        self._inductors: list[Element] = []
        self._switches: list[Switch] = []
        self._diodes: list[Diode] = []
        self._diode_branches: Optional[tuple[int, ...]] = None
        self._diode_output_bases: Optional[tuple[int, ...]] = None
        self._transformers: list[Transformer] = []
        self._gates: list[tuple[int, Callable[[], bool]]] = []  # (diode, gate) of device positions
        self._gated_state: tuple[bool, ...] = ()
        self.states = np.zeros(0, dtype=np.intp)
        self._ordered = True  # the states are y[0:n_x] in order
        self.n_e = self.n_bd = self.n_x = self.n_u = 0
        self.impulsive_supply = 0.0
        self._cache: "OrderedDict[tuple, Topology]" = OrderedDict()
        self._structures: "OrderedDict[tuple, Topology]" = OrderedDict()  # by switch and diode pattern
        self._invalid_structures: "OrderedDict[tuple, tuple[str, tuple[int, ...], bool]]" = OrderedDict()
        self._scalar_functions: "OrderedDict[tuple[tuple, str, Optional[tuple[int, ...]]], Callable]" = OrderedDict()
        self._dynamic_scalar_functions: "OrderedDict[tuple, Callable]" = OrderedDict()
        self._dynamic_scalar_equation_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._scalar_margin_functions: "OrderedDict[tuple, Callable]" = OrderedDict()
        self._state_diode_relations: dict[int, Optional[tuple[int, float]]] = {}
        self._state_diode_structures: dict[int, set[tuple]] = {}
        self._state_diode_groups: Optional[
            tuple[tuple[int, tuple[tuple[int, float], ...]], ...]
        ] = None
        self._state_diode_modes: dict[tuple, tuple[tuple[bool, ...], Topology]] = {}
        self._state_diode_guards: dict[tuple, Optional[tuple[tuple[int, float], ...]]] = {}
        self._scalar_topology: Optional[Topology] = None
        self._last: Optional[_Evaluation] = None
        self._mode: Optional[Mode] = None
        self._ratio_override: Optional[tuple[NDArray[np.float64], ...]] = None
        self._frozen_mode: Optional[Mode] = None
        self._frozen_topology: Optional[Topology] = None
        self._frozen_scalar: Optional[Callable] = None
        self._recovery_excluded: set[tuple[bool, ...]] = set()

    # ---------------------------------------------------------------- construction
    @staticmethod
    def node(dim: int = 1, scale: float = 1.0, name: str = "") -> Node:
        return Node(dim, scale, name)

    @property
    def n_vertices(self) -> int:
        return len(self._vertex_dim)

    @property
    def n_out(self) -> int:
        return self.n_e + 2 * self.n_bd

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes)

    @property
    def sources(self) -> list[Element]:
        """Elements supplying driving values in the order of the input vector ``u``."""
        return [src for src, _ in self._sources]

    def _vertex_of(self, node: Port) -> int:
        if node is None:
            return 0
        v = self._vertex.get(id(node))
        if v is None:
            v = self._vertex[id(node)] = len(self._vertex_dim)
            self._nodes.append(node)
            self._vertex_dim.append(node.dim)
            self._e_off = np.append(self._e_off, self.n_e)
            self.n_e += node.dim
        return v

    def add(self, *elements: Element) -> Element:
        """Add primitives; returns the (first) element."""
        for el in elements:
            if not isinstance(el, Element) or any(el is e for e in self.elements):
                raise ValueError(f"{el!r} is not a new netlist element")
            if not el.name:
                el.name = f"{type(el).__name__.lower()}{len(self.elements)}"
            state = getattr(el, "state", None)
            if state is not None and np.intersect1d(state, self.states).size:
                raise ValueError(f"{el.name}: state indices {np.intersect1d(state, self.states).tolist()} are taken")
            self.elements.append(el)
            ids = []
            for port, (a, b) in enumerate(el.ports):
                d, scale = _port(a, b)
                label = (el.name if len(el.ports) == 1 else f"{el.name}.{'ps'[port]}" if isinstance(el, Transformer)
                         else f"{el.name}[{port}]")
                br = _Branch(el, port, self._vertex_of(a), self._vertex_of(b), d, scale, self.n_bd, label)
                self.n_bd += d
                if state is not None:
                    br.x, self.n_x = self.n_x, self.n_x + d
                ids.append(len(self._branches))
                self._branches.append(br)
            self._ports[id(el)] = ids
            if state is not None:
                self.states = np.concatenate([self.states, state])
                self._ordered = bool(np.array_equal(self.states, np.arange(self.states.size)))
                if isinstance(el, Capacitor):
                    self._capacitors.append(el)
                elif not isinstance(el, CurrentState):
                    self._inductors.append(el)
            if isinstance(el, (VoltageSource, CurrentSource, CurrentState)):
                self._u_off[ids[0]] = self.n_u
                self._sources.append((el, self.n_u))
                self._source_slots.append((el.value, self.n_u, self._branches[ids[0]].dim, el.name))
                self.n_u += self._branches[ids[0]].dim
            elif isinstance(el, Switch):
                self._switches.append(el)
            elif isinstance(el, Diode):
                if el.gate is not None:
                    self._gates.append((len(self._diodes), el.gate))
                self._diodes.append(el)
                self._diode_branches = None
                self._diode_output_bases = None
            elif isinstance(el, Transformer):
                self._transformers.append(el)
        self._plan = None
        self._diode_branches = None
        self._diode_output_bases = None
        self.invalidate()
        return elements[0] if elements else None

    def invalidate(self) -> None:
        """Forget every compiled mode (after changing a parameter of an element)."""
        self._cache.clear()
        self._structures.clear()
        self._invalid_structures.clear()
        self._parts.clear()
        self._scalar_functions.clear()
        self._dynamic_scalar_functions.clear()
        self._dynamic_scalar_equation_cache.clear()
        self._scalar_margin_functions.clear()
        self._state_diode_relations.clear()
        self._state_diode_structures.clear()
        self._state_diode_groups = None
        self._state_diode_modes.clear()
        self._state_diode_guards.clear()
        self._scalar_topology = None
        self._last = None
        self._frozen_mode = self._frozen_topology = self._frozen_scalar = None

    # ---------------------------------------------------------------- state-space-nodal groups
    def grouping(self) -> Grouping:
        """The automatic partition into switch-cluster groups and their interface nodes (see
        :func:`_partition`); the sparse form uses it when at least two groups switch."""
        if self._plan is None:
            self._plan = _partition(self)
            n = len(self._plan[0].elements)
            self._stats.update(groups=[0] * n, reused=[0] * n)
        return self._plan[0]

    @property
    def _labels(self) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        self.grouping()
        return self._plan[1], self._plan[2]

    def _grouping_used(self) -> bool:
        return self.grouped and self.form == "nodal" and self.grouping().used

    @property
    def factorizations(self) -> dict[str, Any]:
        """Factorizations so far: ``network`` (the whole sparse system), ``interface`` (Schur systems),
        per group ``groups`` (its block) and ``reused`` (compiled modes that found its block factorized)."""
        return {key: list(value) if isinstance(value, list) else value for key, value in self._stats.items()}

    def _branch(self, element: Element, port: int = 0) -> _Branch:
        ids = self._ports.get(id(element))
        if ids is None or not 0 <= port < len(ids):
            raise KeyError(f"{element!r} has no port {port} in this netlist")
        return self._branches[ids[port]]

    def state_diode_relations(
            self, topology: Optional[Topology] = None
    ) -> tuple[tuple[int, tuple[tuple[int, float], ...]], ...]:
        """Return diode currents proved to be signed scalar inductor states.

        Each item is ``(state_index, ((diode_index, sign), ...))`` and means that a
        conducting diode's current is exactly ``sign * y[state_index]``.  The proof
        uses the compiled equations, so callers need no component- or topology-name
        knowledge.  An incomplete proof returns no relations; ordinary voltage/current
        commutation remains the fallback.
        """
        if (not self._diodes
                or any(relation is None for relation in self._state_diode_relations.values())
                or any(
                    transformer.constant is None for transformer in self._transformers
                )):
            return ()
        topology = self.topology() if topology is None else topology
        structure = topology.mode.key[:2]
        pending = tuple(
            (diode_index, diode)
            for diode_index, diode in enumerate(self._diodes)
            if topology.mode.conducting[diode_index]
            and structure not in self._state_diode_structures.get(diode_index, ())
        )
        if not pending:
            return self._state_diode_groups or ()

        inductor_states = {
            state
            for element in self._inductors
            for branch_index in self._ports[id(element)]
            for branch in (self._branches[branch_index],)
            if branch.x >= 0
            for state in range(branch.x, branch.x + branch.dim)
        }
        if not inductor_states:
            return ()

        state_space = topology.state_space
        for diode_index, diode in pending:
            branch = self.output_slice("branch_current", diode)
            if branch.stop - branch.start != 1:
                self._state_diode_relations[diode_index] = None
                continue
            current = state_space.C[branch.start]
            active = np.flatnonzero(np.abs(current) > 1e-12)
            if (active.size != 1
                    or int(active[0]) not in inductor_states
                    or not np.all(np.abs(state_space.D[branch.start]) <= 1e-12)
                    or not np.all(np.abs(state_space.D1[branch.start]) <= 1e-12)):
                self._state_diode_relations[diode_index] = None
                continue
            state = int(active[0])
            coefficient = float(current[state])
            if not math.isclose(abs(coefficient), 1.0, rel_tol=1e-12, abs_tol=1e-12):
                self._state_diode_relations[diode_index] = None
                continue
            relation = (
                int(self.states[state]), math.copysign(1.0, coefficient)
            )
            previous = self._state_diode_relations.get(diode_index, relation)
            self._state_diode_relations[diode_index] = relation if previous == relation else None
            self._state_diode_structures.setdefault(diode_index, set()).add(structure)
        if (len(self._state_diode_relations) != len(self._diodes)
                or any(relation is None for relation in self._state_diode_relations.values())):
            self._state_diode_groups = None
            return ()
        relations: dict[int, list[tuple[int, float]]] = {}
        for diode_index, relation in self._state_diode_relations.items():
            state, direction = relation
            relations.setdefault(state, []).append((diode_index, direction))
        if self._state_diode_groups is None:
            self._state_diode_groups = tuple(
                (state, tuple(items)) for state, items in relations.items()
            )
        return self._state_diode_groups

    def state_diode_guard_plan(
            self, topology: Topology
    ) -> tuple[
        tuple[tuple[int, tuple[tuple[int, float], ...]], ...],
        Optional[tuple[tuple[int, float], ...]],
    ]:
        """Return proved diode-current groups and their current state guards."""
        groups = self.state_diode_relations(topology)
        if not groups:
            return (), None
        gated = self._gated_state or (False,) * len(self._diodes)
        key = topology.mode.key[:2], gated, groups
        if key in self._state_diode_guards:
            return groups, self._state_diode_guards[key]

        covered: set[int] = set()
        guards = []
        for state, relations in groups:
            natural = [(index, sign) for index, sign in relations if not gated[index]]
            conducting = [
                (index, sign) for index, sign in natural
                if topology.mode.conducting[index]
            ]
            if not conducting:
                if (any(gated[index] for index, _sign in relations)
                        and self.diode_turn_on_excluded(
                            topology, [index for index, _sign in natural]
                        )):
                    covered.update(index for index, _sign in natural)
                    continue
                self._state_diode_guards[key] = None
                return groups, None
            if len(conducting) != 1:
                self._state_diode_guards[key] = None
                return groups, None
            active, direction = conducting[0]
            if any(sign == direction for index, sign in natural if index != active):
                self._state_diode_guards[key] = None
                return groups, None
            guards.append((state, direction))
            covered.update(index for index, _sign in natural)
        result = (
            None
            if any(not gated[index] and index not in covered
                   for index in range(len(self._diodes)))
            else tuple(guards)
        )
        self._state_diode_guards[key] = result
        return groups, result

    def select_state_diode_mode(
            self, y: NDArray[np.float64],
            groups: Sequence[tuple[int, Sequence[tuple[int, float]]]],
    ) -> Optional[Topology]:
        """Select a proved diode pattern directly from nonzero inductor currents.

        This is valid only when every diode belongs to a proved signed-current group,
        no state projection is introduced, and a unique diode carries each ungated
        current direction.  Otherwise the caller retains the ordinary commutation
        search.
        """
        gated = [False] * len(self._diodes)
        for index, gate in self._gates:
            gated[index] = bool(gate())
        self._gated_state = tuple(gated)
        selected: dict[int, bool] = {}
        directions = []
        for state, relations in groups:
            group = tuple(relations)
            gated_group = [index for index, _direction in group if gated[index]]
            if gated_group:
                if len(gated_group) != 1:
                    return None
                directions.append(0.0)
                selected.update((index, False) for index, _direction in group)
                continue
            value = float(y[state])
            if value == 0.0:
                return None
            direction = math.copysign(1.0, value)
            directions.append(direction)
            candidates = [index for index, sign in group if sign == direction]
            if len(candidates) != 1:
                return None
            chosen = candidates[0]
            selected.update((index, index == chosen) for index, _sign in group)
        if len(selected) != len(self._diodes):
            return None

        key = (
            tuple(bool(switch.closed()) for switch in self._switches),
            tuple(gated),
            tuple(directions),
        )
        cached = self._state_diode_modes.get(key)
        if cached is not None:
            pattern, topology = cached
            self.release_topology()
            for diode, conducting in zip(self._diodes, pattern):
                diode.conducting, diode.residual = conducting, 0.0
            self._mode = topology.mode
            return topology

        previous = tuple((diode.conducting, diode.residual) for diode in self._diodes)
        self.release_topology()
        for index, conducting in selected.items():
            diode = self._diodes[index]
            diode.conducting, diode.residual = conducting, 0.0
        expected = self.conducting
        try:
            topology = self.topology()
            if self.conducting != expected or any(
                    projector is not None for projector in topology.projectors()
            ):
                raise ValueError
            self._state_diode_modes[key] = expected, topology
            return topology
        except (_Degenerate, ConductionError, ValueError):
            self.release_topology()
            for diode, (conducting, residual) in zip(self._diodes, previous):
                diode.conducting, diode.residual = conducting, residual
            return None

    def diode_turn_on_excluded(self, topology: Topology, indices: Sequence[int]) -> bool:
        """Whether closing any listed blocking diode makes the present mode singular."""
        for index in indices:
            conducting = list(topology.mode.conducting)
            conducting[index] = True
            try:
                self._compiled(Mode(
                    topology.mode.closed, conducting, topology.mode.ratios
                ))
            except _Degenerate:
                continue
            return False
        return True

    # ---------------------------------------------------------------- modes
    @property
    def conducting(self) -> tuple[bool, ...]:
        """The diodes' own conduction states, in netlist order."""
        return tuple(bool(diode.conducting) for diode in self._diodes)

    @property
    def has_discrete_topology(self) -> bool:
        """Whether switch or diode states can change the circuit equations."""
        return bool(self._switches or self._diodes)

    @property
    def _residual(self) -> NDArray[np.float64]:
        """The diodes' margin residuals, in netlist order."""
        return np.asarray([diode.residual for diode in self._diodes])

    def mode(self) -> Mode:
        """The current switch positions, diode conduction and transformer ratios."""
        if self._frozen_mode is not None:
            return self._frozen_mode
        ratios = (list(self._ratio_override) if self._ratio_override is not None else
                  [tr.ratio() for tr in self._transformers])
        conducting = self.conducting
        if self._gates:  # a gated device position conducts whatever its diode does
            conducting = list(conducting)
            gated = [False] * len(self._diodes)
            for j, gate in self._gates:
                gated[j] = bool(gate())
                if gated[j]:
                    conducting[j] = True
            self._gated_state = tuple(gated)
        closed = tuple(bool(s.closed()) for s in self._switches)
        conducting = tuple(bool(value) for value in conducting)
        last = self._mode
        if (last is not None and last.closed == closed and last.conducting == conducting
                and len(last.ratios) == len(ratios)
                and all(a is b or np.array_equal(a, b) for a, b in zip(last.ratios, ratios))):
            return last
        self._mode = Mode(closed, conducting, ratios)
        return self._mode

    def _gated(self) -> list[bool]:
        """Per diode: gate state read by the latest mode evaluation."""
        return list(self._gated_state or (False,) * len(self._diodes))

    def topology(self, mode: Optional[Mode] = None) -> Topology:
        """The compiled current (or given) mode, from the cache when it was compiled before. A mode
        that differs from a compiled one only in transformer ratios reuses its classification and
        matrix structure (:meth:`Topology.with_ratios`): an averaged bridge's new duty is a refactorization.

        A conducting ideal diode that the current mode puts in a loop of voltage-type branches only (a
        switch closing across a freewheeling diode) is reverse-biased by that loop: it turns off, and the
        conduction margins confirm it."""
        if mode is None and self._frozen_topology is not None:
            return self._frozen_topology
        if mode is None and self._frozen_mode is not None:
            self._frozen_topology = self._compiled(self._frozen_mode)
            return self._frozen_topology
        if mode is not None:
            return self._compiled(mode)
        requested = self.mode()
        topology = self.resolved_topology(
            requested, self._gated(), excluded=self._recovery_excluded
        )
        if topology.mode.conducting != requested.conducting:
            for diode, before, after in zip(
                    self._diodes, requested.conducting, topology.mode.conducting):
                if before != after:
                    diode.conducting, diode.residual = after, 0.0
            self._mode = topology.mode
        return topology

    def resolved_topology(
        self, mode: Mode, gated: Sequence[bool] = (), *,
        excluded: Sequence[tuple[bool, ...]] = (),
    ) -> Topology:
        """Resolve a singular ideal-device pattern by the ordinary diode recovery rule."""
        gated = tuple(gated) or (False,) * len(self._diodes)
        if len(gated) != len(self._diodes):
            raise ValueError("the gated-device pattern has the wrong length")
        seen: set[tuple[bool, ...]] = set()
        excluded = set(excluded)
        for _ in range(_MAX_PATTERNS):
            pattern = mode.conducting
            if pattern in seen:
                break
            seen.add(pattern)
            try:
                return self._compiled(mode)
            except _Degenerate as degenerate:
                candidates = (
                    [j for j, diode in enumerate(self._diodes)
                     if pattern[j] and not gated[j] and diode.R_on == 0.0
                     and self._ports[id(diode)][0] in degenerate.branches]
                    if degenerate.loop else
                    [j for j, diode in enumerate(self._diodes)
                     if not pattern[j] and not gated[j]]
                )
                selected = next((j for j in candidates
                                 if (candidate := tuple(
                                     value if k != j else not value
                                     for k, value in enumerate(pattern)
                                 )) not in seen and candidate not in excluded), None)
                if selected is None:
                    raise
                conducting = list(pattern)
                conducting[selected] = not conducting[selected]
                mode = Mode(mode.closed, conducting, mode.ratios)
        raise ConductionError("no nonsingular diode conduction pattern exists")

    def freeze_topology(self) -> None:
        """Reuse the current mode until the enclosing smooth interval ends.

        A previously resolved switch and diode structure needs only its transformer-ratio copy;
        that copy is created on first topology access, which a conclusive certificate avoids.
        """
        if self._frozen_mode is None:
            requested = self.mode()
            if (requested.closed, requested.conducting) in self._structures:
                self._frozen_mode = requested
            else:
                topology = self.topology()
                self._frozen_mode, self._frozen_topology = topology.mode, topology
            if (self.form == "device"
                    and all(transformer.constant is not None for transformer in self._transformers)):
                key = self._frozen_mode.key, "all", None
                if key in self._scalar_functions:
                    self._frozen_scalar = self._scalar_function(self.topology(), "all")

    def _frozen_range_certificate(self) -> tuple[Optional[Topology], Optional[tuple[object, float]]]:
        """Return the frozen mode's structure and covering range certificate, if available."""
        mode = self._frozen_mode
        if mode is None or self._frozen_topology is not None:
            return None, None
        base = self._structures.get((mode.closed, mode.conducting))
        return base, None if base is None else base.range_stability_certificate(mode.ratios)

    def stability_certificate(self) -> Optional[tuple[object, float]]:
        """Return the present stability certificate without its identity or matrix."""
        _base, certificate = self._frozen_range_certificate()
        if certificate is not None:
            return certificate
        return self.topology().state_matrix_stability_certificate()

    def linear_mode(self) -> tuple[tuple, Callable[[], NDArray[np.float64]],
                                   Optional[tuple[object, float]]]:
        """Return the present state-matrix identity, builder and stability certificate."""
        base, certificate = self._frozen_range_certificate()
        if certificate is not None:
            return (base.state_matrix_key(self._frozen_mode),
                    lambda: self.topology().state_space.A, certificate)
        topology = self.topology()
        return (topology.state_matrix_key(), lambda: topology.state_space.A,
                topology.state_matrix_stability_certificate())

    def release_topology(self) -> None:
        """Allow topology callbacks to be read at the next model-changing boundary."""
        self._frozen_mode = self._frozen_topology = self._frozen_scalar = None

    def _compiled(self, mode: Mode) -> Topology:
        top = self._cache.get(mode.key)
        if top is None:
            structure = mode.key[:2]
            base = self._structures.get(structure)
            if base is None:
                failure = self._invalid_structures.get(structure)
                if failure is not None:
                    self._invalid_structures.move_to_end(structure)
                    message, branches, loop = failure
                    raise _Degenerate(message, list(branches), loop)
                try:
                    top = Topology(self, mode)
                except _Degenerate as exc:
                    self._invalid_structures[structure] = (
                        str(exc), tuple(exc.branches), exc.loop
                    )
                    while len(self._invalid_structures) > self.cache_size:
                        self._invalid_structures.popitem(last=False)
                    raise
                self._structures[structure] = top
                while len(self._structures) > self.cache_size:
                    self._structures.popitem(last=False)
            else:
                self._structures.move_to_end(structure)
                top = base.with_ratios(mode)
            self._cache[mode.key] = top
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(mode.key)
        return top

    def state_space(self, mode: Optional[Mode] = None) -> StateSpace:
        """The dense state space of the current (or given) mode."""
        return self.topology(mode).state_space

    def dependent_states(self, mode: Optional[Mode] = None) -> NDArray[np.intp]:
        """State indices (into ``y``) that the current mode's loops and cutsets make dependent."""
        return self.states[self.topology(mode).dependent]

    def set_conducting(self, diode: Diode, conducting: bool, residual: float = 0.0) -> None:
        """Set a diode's conduction state and the margin residual it had at its last commutation."""
        if diode not in self._diodes:
            raise ValueError("the diode does not belong to this netlist")
        self.release_topology()
        diode.conducting, diode.residual = bool(conducting), min(float(residual), 0.0)

    def conduction(self, diode: Diode) -> tuple[bool, float]:
        """A diode's conduction state (its own, whatever its gate) and margin residual."""
        if diode not in self._diodes:
            raise ValueError("the diode does not belong to this netlist")
        return diode.conducting, diode.residual

    # ---------------------------------------------------------------- evaluation
    def inputs(self, t: float) -> NDArray[np.float64]:
        """The source values ``u`` at ``t``."""
        u = np.empty(self.n_u)
        for value, o, d, name in self._source_slots:
            v = value(t)
            if d == 1 and isinstance(v, (float, int)):
                u[o] = v
            elif d == 2 and type(v) is complex:
                u[o], u[o + 1] = v.real, v.imag
            else:
                u[o:o + d] = _vector(v, d, name)
        return u

    def source_values(self, t: Any) -> tuple[Any, ...]:
        """Source values without coercing equation values to numbers."""
        values: list[Any] = [0.0] * self.n_u
        for value, offset, dimension, _name in self._source_slots:
            source = value(t)
            if dimension == 1:
                values[offset] = source
            elif dimension == 2:
                values[offset], values[offset + 1] = source.real, source.imag
            else:
                values[offset:offset + dimension] = source
        return tuple(values)

    def source_rates(self, t: Any, needed: Optional[Sequence[bool]] = None) -> tuple[Any, ...]:
        """Source derivatives without coercing equation values to numbers (zero where ``needed``
        is false); a source without a derivative is differentiated centrally."""
        rates: list[Any] = [0.0] * self.n_u
        for index, (source, offset) in enumerate(self._sources):
            if needed is not None and not needed[index]:
                continue
            dimension = self._branch(source).dim
            value = (source.derivative(t) if source.derivative is not None else
                     (source.value(t + _DELTA_T) - source.value(t - _DELTA_T)) / (2.0 * _DELTA_T))
            if dimension == 1:
                rates[offset] = value
            elif dimension == 2:
                rates[offset:offset + 2] = (value.real, value.imag)
            else:
                rates[offset:offset + dimension] = value
        return tuple(rates)

    def rates(self, t: float, needed: Optional[NDArray[np.bool_]] = None) -> NDArray[np.float64]:
        """The source derivatives ``du/dt`` at ``t`` (zero where ``needed`` is false)."""
        du = np.zeros(self.n_u)
        for j, (src, o) in enumerate(self._sources):
            if needed is not None and not needed[j]:
                continue
            d = self._branch(src).dim
            if src.derivative is not None:
                du[o:o + d] = _vector(src.derivative(t), d, src.name)
            else:
                hi, lo = t + _DELTA_T, t - _DELTA_T
                du[o:o + d] = (_vector(src.value(hi), d, src.name) - _vector(src.value(lo), d, src.name)) / (hi - lo)
        return du

    @staticmethod
    def _complex_term(coefficient: complex, variable: str) -> str:
        """One compact term of a generated complex-linear expression."""
        if coefficient == 1.0:
            return variable
        if coefficient == -1.0:
            return f"-({variable})"
        if coefficient.imag == 0.0:
            return f"{coefficient.real!r} * ({variable})"
        return f"{coefficient!r} * ({variable})"

    def _scalar_row_expressions(
        self,
        rows: NDArray[np.float64],
        row_indices: Sequence[int],
        paired_rows: set[int],
        variables: Sequence[str],
        complex_variables: Mapping[int, str],
        prefix: str,
    ) -> tuple[list[str], list[str], set[int]]:
        """Generate scalar expressions, retaining two-coordinate electrical quantities as complex values.

        A pair is combined only when every paired-column block has the complex-linear form
        ``[[a, -b], [b, a]]``. Other rows keep the ordinary scalar representation.
        """
        expressions = [""] * len(row_indices)
        statements: list[str] = []
        scalar_columns: set[int] = set()
        positions = {row: position for position, row in enumerate(row_indices)}
        paired_columns = set(complex_variables)
        paired_members = paired_columns | {column + 1 for column in paired_columns}
        processed: set[int] = set()

        for row in sorted(paired_rows):
            first, second = positions.get(row), positions.get(row + 1)
            if first is None or second is None:
                continue
            real, imag = rows[first], rows[second]
            terms: list[str] = []
            compatible = True
            for column in sorted(paired_columns):
                a, minus_b = float(real[column]), float(real[column + 1])
                b, repeated_a = float(imag[column]), float(imag[column + 1])
                scale = max(1.0, abs(a), abs(minus_b), abs(b), abs(repeated_a))
                if abs(a - repeated_a) > 1e-13 * scale or abs(minus_b + b) > 1e-13 * scale:
                    compatible = False
                    break
                coefficient = complex(a, b)
                if coefficient:
                    terms.append(self._complex_term(coefficient, complex_variables[column]))
            if not compatible:
                continue
            for column in range(rows.shape[1]):
                if column in paired_members:
                    continue
                coefficient = complex(float(real[column]), float(imag[column]))
                if coefficient:
                    scalar_columns.add(column)
                    terms.append(self._complex_term(coefficient, variables[column]))
            name = f"_complex_{prefix}_{len(statements)}"
            expression = " + ".join(terms).replace("+ -", "- ") or "0j"
            statements.append(f"    {name} = {expression}")
            expressions[first], expressions[second] = f"{name}.real", f"{name}.imag"
            processed.update((first, second))

        for position, row in enumerate(rows):
            if position in processed:
                continue
            expressions[position] = _linear_expression(row, variables)
            scalar_columns.update(map(int, np.flatnonzero(row)))
        return expressions, statements, scalar_columns

    def _scalar_coordinate_pairs(self) -> tuple[dict[int, str], set[int], set[int]]:
        """Complex-valued column expressions and paired derivative/output row starts."""
        complex_variables: dict[int, str] = {}
        derivative_rows: set[int] = set()
        for branch in self._branches:
            if branch.x >= 0 and branch.dim == 2:
                derivative_rows.add(branch.x)
                complex_variables[branch.x] = (
                    f"_complex(y[{int(self.states[branch.x])}], y[{int(self.states[branch.x + 1])}])"
                )
        for source_index, ((source, _offset), (_value, offset, dimension, _name)) in enumerate(
                zip(self._sources, self._source_slots)):
            if dimension == 2:
                complex_variables[self.n_x + offset] = f"_value_{source_index}"
                complex_variables[self.n_x + self.n_u + offset] = (
                    f"_rate_value_{source_index}" if source.derivative is not None
                    else f"_rate_complex_{source_index}"
                )

        output_rows = {
            int(self._e_off[vertex])
            for vertex in range(1, self.n_vertices)
            if self._vertex_dim[vertex] == 2
        }
        for branch in self._branches:
            if branch.dim == 2:
                output_rows.update((self.n_e + branch.output, self.n_e + self.n_bd + branch.output))
        return complex_variables, derivative_rows, output_rows

    def _compile_scalar(self, top: Topology, part: str = "all",
                        selected: Optional[tuple[int, ...]] = None) -> Callable:
        """Compile one topology to straight-line scalar state and output equations."""
        if part not in ("all", "derivatives", "outputs"):
            raise ValueError(f"unknown device equation part {part!r}")
        if selected is not None and part != "outputs":
            raise ValueError("selected rows apply only to device outputs")
        ss = top.state_space
        env: dict[str, Any] = {"_complex": _complex}
        lines: list[str] = []
        u_names = [f"u{k}" for k in range(self.n_u)]
        du_names = [f"du{k}" for k in range(self.n_u)]
        x_names = [f"x{k}" for k in range(self.n_x)]
        variables = [*x_names, *u_names, *du_names]
        derivative_rows = np.hstack((ss.A, ss.B, ss.B1))
        output_rows = np.hstack((ss.C, ss.D, ss.D1))
        active_output = (tuple(range(self.n_out)) if selected is None else selected)
        if part == "derivatives":
            active_rows = derivative_rows
        elif part == "outputs":
            active_rows = output_rows[list(active_output)]
        else:
            active_rows = np.vstack((derivative_rows, output_rows))
        used = np.any(active_rows != 0.0, axis=0) if active_rows.size else np.zeros(
            self.n_x + 2 * self.n_u, dtype=bool
        )
        complex_variables, derivative_pairs, output_pairs = self._scalar_coordinate_pairs()
        dx: list[str] = []
        outputs: list[str] = []
        expression_lines: list[str] = []
        scalar_columns: set[int] = set()
        if part in ("all", "derivatives"):
            dx, generated, scalar = self._scalar_row_expressions(
                derivative_rows, tuple(range(self.n_x)), derivative_pairs,
                variables, complex_variables, "d",
            )
            expression_lines.extend(generated)
            scalar_columns.update(scalar)
        if part in ("all", "outputs"):
            outputs, generated, scalar = self._scalar_row_expressions(
                output_rows[list(active_output)], active_output, output_pairs,
                variables, complex_variables, "o",
            )
            expression_lines.extend(generated)
            scalar_columns.update(scalar)
        if part == "all":
            scalar_columns.update(range(self.n_x, self.n_x + 2 * self.n_u))

        for j, ((source, offset), (_value, _same, dimension, _name)) in enumerate(
                zip(self._sources, self._source_slots)):
            need_value = part == "all" or bool(np.any(
                used[self.n_x + offset:self.n_x + offset + dimension]
            ))
            need_rate = bool(np.any(used[self.n_x + self.n_u + offset:
                                         self.n_x + self.n_u + offset + dimension]))
            if not need_value and not need_rate:
                continue
            callback = f"_source_{j}"
            env[callback] = source.value
            if need_value:
                value = f"_value_{j}"
                lines.append(f"    {value} = {callback}(t)")
                for k in range(dimension):
                    column = self.n_x + offset + k
                    if column not in scalar_columns:
                        continue
                    part_value = (value if dimension == 1 else
                                  f"{value}.{'real' if k == 0 else 'imag'}" if dimension == 2 else
                                  f"{value}[{k}]")
                    lines.append(f"    u{offset + k} = {part_value}")

            if not need_rate:
                if part == "all":
                    for k in range(dimension):
                        lines.append(f"    du{offset + k} = 0.0")
                continue
            derivative = source.derivative
            if derivative is not None:
                rate = f"_rate_{j}"
                env[rate] = derivative
                rate_value = f"_rate_value_{j}"
                lines.append(f"    {rate_value} = {rate}(t)")
            else:
                before, after = f"_before_{j}", f"_after_{j}"
                lines.extend((f"    {before} = {callback}(t - {_DELTA_T!r})",
                              f"    {after} = {callback}(t + {_DELTA_T!r})"))
                rate_value = f"_rate_complex_{j}" if dimension == 2 else f"_rate_value_{j}"
                lines.append(f"    {rate_value} = ({after} - {before}) / {2.0 * _DELTA_T!r}")
            for k in range(dimension):
                column = self.n_x + self.n_u + offset + k
                if column not in scalar_columns:
                    continue
                part_value = (rate_value if dimension == 1 else
                              f"{rate_value}.{'real' if k == 0 else 'imag'}" if dimension == 2 else
                              f"{rate_value}[{k}]")
                lines.append(f"    du{offset + k} = {part_value}")

        for k, state in enumerate(self.states):
            if k in scalar_columns:
                lines.append(f"    x{k} = y[{int(state)}]")
        lines.extend(expression_lines)

        def tuple_expression(items: Sequence[str]) -> str:
            if not items:
                return "()"
            return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"

        if part == "derivatives":
            result = tuple_expression(dx)
        elif part == "outputs":
            result = tuple_expression(outputs)
        else:
            result = ", ".join(tuple_expression(items) for items in (dx, outputs, u_names, du_names))
        statements = tuple(lines)
        lines.append("    return " + result)
        exec(compile("def evaluate(t, y):\n" + "\n".join(lines) + "\n",
                     "<peslite scalar circuit equations>", "exec"), env)
        evaluate = env["evaluate"]
        evaluate._peslite_scalar_program = (
            statements,
            result,
            {key: value for key, value in env.items()
             if key not in ("evaluate", "__builtins__")},
            tuple(dx if part == "derivatives" else outputs) if part != "all" else None,
        )
        return evaluate

    def _scalar_function(self, top: Topology, part: str,
                         selected: Optional[Sequence[int]] = None) -> Callable:
        selected_key = None if selected is None else tuple(sorted(set(map(int, selected))))
        key = top.mode.key, part, selected_key
        evaluate = self._scalar_functions.get(key)
        if evaluate is None:
            evaluate = self._scalar_functions[key] = self._compile_scalar(top, part, selected_key)
            while len(self._scalar_functions) > 3 * self.cache_size:
                self._scalar_functions.popitem(last=False)
        else:
            self._scalar_functions.move_to_end(key)
        return evaluate

    def _scalar_topology_at(self) -> Topology:
        top = self._scalar_topology
        if top is None:
            top = self.topology()
            if (not self._switches and not self._diodes
                    and all(transformer.constant is not None for transformer in self._transformers)):
                self._scalar_topology = top
        return top

    def evaluate_scalar_derivatives(self, t: float, y: Sequence[float]) -> tuple:
        """Evaluate the generated state equations in the caller's state layout."""
        top = self._scalar_topology_at()
        derivative = (self._dynamic_scalar_function(top)(t, y)[0]
                      if any(transformer.constant is None for transformer in self._transformers)
                      else self._scalar_function(top, "derivatives")(t, y))
        return self._place_scalar_derivatives(derivative, len(y))

    def scalar_evaluators(self) -> tuple[Callable, Callable, Callable]:
        """Compile scalar derivatives, all outputs and a selected-output factory.

        Fixed topologies use the compiled state space; mode-dependent networks retain the ordinary
        topology evaluator. Every circuit therefore uses the same primitive equations.
        """
        if (not self._switches and not self._diodes
                and all(item.constant is not None for item in self._transformers)):
            top = self._scalar_topology_at()
            derivatives = self._scalar_function(top, "derivatives")
            if not self._ordered:
                raw = derivatives
                derivatives = lambda t, y: self._place_scalar_derivatives(raw(t, y), len(y))
            return derivatives, self._scalar_function(top, "outputs"), self.scalar_output_evaluator
        return (self.evaluate_scalar_derivatives, self.evaluate_scalar_outputs,
                self.selected_scalar_output_evaluator)

    def selected_scalar_output_evaluator(self, selected: Sequence[int]) -> Callable:
        """Return a mode-aware evaluator for the requested output rows."""
        rows = tuple(sorted(set(map(int, selected))))

        def evaluate(t: float, y: Sequence[float]) -> tuple:
            top = self._scalar_topology_at()
            if any(transformer.constant is None for transformer in self._transformers):
                outputs = self._dynamic_scalar_function(top)(t, y)[1]
                return tuple(outputs[row] for row in rows)
            return self._scalar_function(top, "outputs", rows)(t, y)

        return evaluate

    def scalar_output_evaluator(self, selected: Sequence[int]) -> Callable:
        """Compile a fixed-topology evaluator returning the requested outputs in the same order."""
        return self._scalar_function(self._scalar_topology_at(), "outputs", selected)

    def evaluate_scalar_outputs(self, t: float, y: Sequence[float]) -> tuple:
        """Evaluate only the generated electrical outputs of the present topology."""
        top = self._scalar_topology_at()
        if any(transformer.constant is None for transformer in self._transformers):
            return self._dynamic_scalar_function(top)(t, y)[1]
        return self._scalar_function(top, "outputs")(t, y)

    def _dynamic_structure_key(self, top: Topology) -> tuple:
        """Return the discrete mode and symbolic sparsity selecting generated equations."""
        key = getattr(top, "_dynamic_structure_key", None)
        if key is None:
            pattern = top._pattern
            symbolic = getattr(pattern, "_symbolic_sparsity", None)
            if symbolic is None:
                present = np.fromiter(
                    (not _zero(value) for value in pattern.expressions),
                    dtype=bool, count=pattern.nnz,
                )
                present[pattern.slot_pos] = True
                pattern._symbolic_sparsity = symbolic = np.packbits(present).tobytes()
            key = top.mode.key[:2], symbolic
            top._dynamic_structure_key = key
        return key

    def _dynamic_scalar_equations(self, top: Topology) -> tuple:
        """Return one symbolic equation set shared by all scalar stages of a structural mode."""
        key = self._dynamic_structure_key(top)
        equations = self._dynamic_scalar_equation_cache.get(key)
        if equations is None:
            equations = self._build_dynamic_scalar_equations(top)
            self._dynamic_scalar_equation_cache[key] = equations
            while len(self._dynamic_scalar_equation_cache) > self.cache_size:
                self._dynamic_scalar_equation_cache.popitem(last=False)
        else:
            self._dynamic_scalar_equation_cache.move_to_end(key)
        return equations

    def _build_dynamic_scalar_equations(self, top: Topology) -> tuple:
        """Build symbolic equations for a mode whose transformer ratios remain run-time values."""
        env: dict[str, Any] = {}
        preamble: list[str] = []
        flat_ratios: list[Any] = []
        for index, transformer in enumerate(self._transformers):
            if transformer.constant is not None:
                flat_ratios.extend(transformer.constant.ravel())
                continue
            callback, value = f"_ratio_{index}", f"_ratio_value_{index}"
            env[callback] = transformer.N
            preamble.append(f"    {value} = {callback}()")
            compact = transformer._cached_scalar is not None
            if compact and transformer.shape == (1, 1):
                flat_ratios.append(_Symbolic.value(value))
            elif compact and transformer.shape == (2, 1):
                scalar = _Symbolic.value(value)
                flat_ratios.extend((scalar.real, scalar.imag))
            else:
                flat_ratios.extend(
                    _Symbolic.value(f"{value}[{row}][{column}]")
                    for row in range(transformer.shape[0])
                    for column in range(transformer.shape[1])
                )

        pattern = top._pattern
        columns = np.repeat(np.arange(pattern.n), np.diff(pattern.indptr))
        entries: list[Any] = list(pattern.expressions)
        for position, flat, coefficient in zip(
                pattern.slot_pos, pattern.slot_flat, pattern.slot_coef):
            entries[int(position)] += coefficient * flat_ratios[int(flat)]
        matrix: list[dict[int, Any]] = [{} for _ in range(pattern.n)]
        for value, row, column in zip(entries, pattern.indices, columns):
            if not _zero(value):
                matrix[int(row)][int(column)] = value

        inputs: list[Any] = [0.0] * self.n_u
        rates: list[Any] = [0.0] * self.n_u
        for index, (source, offset) in enumerate(self._sources):
            branch = self._branch(source)
            callback, value = f"_source_{index}", f"_source_value_{index}"
            env[callback] = source.value
            preamble.append(f"    {value} = {callback}(t)")
            if branch.dim == 1:
                inputs[offset] = _Symbolic.value(value)
            elif branch.dim == 2:
                scalar = _Symbolic.value(value)
                inputs[offset:offset + 2] = (scalar.real, scalar.imag)
            else:
                inputs[offset:offset + branch.dim] = (
                    _Symbolic.value(f"{value}[{part}]") for part in range(branch.dim)
                )
            if not top.needs_rate[index]:
                continue
            if source.derivative is not None:
                derivative = f"_source_rate_{index}"
                rate_value = f"_source_rate_value_{index}"
                env[derivative] = source.derivative
                preamble.append(f"    {rate_value} = {derivative}(t)")
            else:
                before, after = f"_source_before_{index}", f"_source_after_{index}"
                preamble.extend((
                    f"    {before} = {callback}(t - {_DELTA_T!r})",
                    f"    {after} = {callback}(t + {_DELTA_T!r})",
                    f"    _source_rate_value_{index} = "
                    f"({after} - {before}) / {2.0 * _DELTA_T!r}",
                ))
                rate_value = f"_source_rate_value_{index}"
            if branch.dim == 1:
                rates[offset] = _Symbolic.value(rate_value)
            elif branch.dim == 2:
                scalar = _Symbolic.value(rate_value)
                rates[offset:offset + 2] = (scalar.real, scalar.imag)
            else:
                rates[offset:offset + branch.dim] = (
                    _Symbolic.value(f"{rate_value}[{part}]") for part in range(branch.dim)
                )

        states = [_Symbolic.value(f"y[{int(index)}]") for index in self.states]
        variables = [*states, *inputs, *(rates if top.any_rate else ())]
        coefficients = top._K.toarray() if hasattr(top._K, "toarray") else top._K
        right = [sum((coefficient * value for coefficient, value in zip(row, variables)
                      if coefficient != 0.0), 0.0) for row in coefficients]
        solution = (_symbolic_solve(matrix, right, pattern.dense(top.mode.ratios),
                                    gauge=top.gauge_singular)
                    if pattern.n else [])
        derivatives = [solution[int(index)] for index in top.xd_index]
        output_matrix = top.S_out.toarray()
        outputs = [sum((coefficient * value for coefficient, value in zip(row, solution)
                        if coefficient != 0.0), 0.0) for row in output_matrix]
        return (tuple(preamble), env, tuple(inputs), tuple(rates),
                tuple(derivatives), tuple(outputs))

    def _compile_dynamic_scalar(
        self, top: Topology, part: str = "all", selected: Optional[tuple[int, ...]] = None,
    ) -> Callable:
        """Compile one scalar stage from the structural mode's shared symbolic equations."""
        if part not in ("all", "derivatives", "outputs"):
            raise ValueError(f"unknown device equation part {part!r}")
        if selected is not None and part != "outputs":
            raise ValueError("selected rows apply only to device outputs")
        preamble, source_env, inputs, rates, derivatives, outputs = (
            self._dynamic_scalar_equations(top)
        )
        env = dict(source_env)
        if part == "derivatives":
            active_derivatives, active_outputs = derivatives, []
        elif part == "outputs":
            active_derivatives = []
            rows = tuple(range(len(outputs))) if selected is None else selected
            active_outputs = [outputs[index] for index in rows]
        else:
            active_derivatives, active_outputs = derivatives, outputs
        expressions = [*active_derivatives, *active_outputs]
        if part == "all":
            expressions.extend((*inputs, *rates))
        shared, rendered = _symbolic_source(expressions)
        n_d, n_o, n_u = len(active_derivatives), len(active_outputs), len(inputs)
        derivative_text = rendered[:n_d]
        output_text = rendered[n_d:n_d + n_o]
        input_text = rendered[n_d + n_o:n_d + n_o + n_u] if part == "all" else []
        rate_text = rendered[n_d + n_o + n_u:] if part == "all" else []

        def tuple_expression(items: Sequence[str]) -> str:
            return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"

        result = (", ".join(tuple_expression(items) for items in (
            derivative_text, output_text, input_text, rate_text,
        )) if part == "all" else
                  tuple_expression(derivative_text if part == "derivatives" else output_text))
        lines = ["def evaluate(t, y):", *preamble,
                 *(f"    {line}" for line in shared), f"    return {result}"]
        exec(compile("\n".join(lines) + "\n", "<peslite dynamic circuit equations>", "exec"), env)
        evaluate = env["evaluate"]
        evaluate._peslite_scalar_program = (
            tuple((*preamble, *(f"    {line}" for line in shared))),
            result,
            {key: value for key, value in env.items()
             if key not in ("evaluate", "__builtins__")},
            tuple(rendered),
        )
        evaluate._peslite_scalar_parts = (
            n_d, n_o, n_u if part == "all" else 0, len(rates) if part == "all" else 0
        )
        return evaluate

    def _dynamic_scalar_function(
        self, top: Topology, part: str = "all", selected: Optional[Sequence[int]] = None,
    ) -> Callable:
        selected_key = None if selected is None else tuple(sorted(set(map(int, selected))))
        key = self._dynamic_structure_key(top), part, selected_key
        evaluate = self._dynamic_scalar_functions.get(key)
        if evaluate is None:
            evaluate = self._dynamic_scalar_functions[key] = self._compile_dynamic_scalar(
                top, part, selected_key
            )
            while len(self._dynamic_scalar_functions) > 3 * self.cache_size:
                self._dynamic_scalar_functions.popitem(last=False)
        else:
            self._dynamic_scalar_functions.move_to_end(key)
        return evaluate

    def evaluate_scalar_raw(self, t: float, y: Sequence[float], *, record: bool = True) -> tuple[tuple, tuple]:
        """Evaluate generated scalar equations without allocating solver-facing arrays."""
        if any(transformer.constant is None for transformer in self._transformers):
            top = self._frozen_topology or self._scalar_topology_at()
            evaluate = self._dynamic_scalar_function(top)
            derivative, output, inputs, rates = evaluate(t, y)
            if record:
                x = tuple(y[int(index)] for index in self.states)
                self._last = _Evaluation(top, t, x, inputs, rates, None, output)
            return self._place_scalar_derivatives(derivative, len(y)), output
        top = self._frozen_topology or self._scalar_topology_at()
        evaluate = self._frozen_scalar or self._scalar_function(top, "all")
        derivative, output, inputs, rates = evaluate(t, y)
        if record:
            x = tuple(y[int(index)] for index in self.states)
            self._last = _Evaluation(top, t, x, inputs, rates, None, output)
        return self._place_scalar_derivatives(derivative, len(y)), output

    def _place_scalar_derivatives(self, derivative: Sequence[float], size: int) -> tuple:
        """Map primitive-order derivatives onto the state indices supplied by the assembler."""
        if size == self.n_x and self._ordered:
            return tuple(derivative)
        placed = [0.0] * size
        for index, value in zip(self.states, derivative):
            placed[int(index)] = value
        return tuple(placed)

    def evaluate_scalar(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate generated scalar equations for the present topology."""
        y = np.asarray(y, dtype=float)
        derivative, _output = self.evaluate_scalar_raw(t, y)
        return np.asarray(derivative, dtype=float)

    def evaluate(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """``dy/dt`` of the netlist states (zero elsewhere in ``y``) in the current mode."""
        if self.form == "device":
            return self.evaluate_scalar(t, y)
        derivative, _output = self.evaluate_with_outputs(t, y, ())
        return derivative

    def _evaluate_numeric(
        self, t: float, y: NDArray[np.float64], selected: Sequence[int], *, derivatives: bool
    ) -> tuple[NDArray[np.float64], Optional[NDArray[np.float64]], NDArray[np.float64]]:
        """Evaluate one numeric circuit form, optionally including its state derivatives."""
        y = np.asarray(y, dtype=float)
        top = self.topology()
        x, u = y[self.states], self.inputs(t)
        du = self.rates(t, top.needs_rate) if top.any_rate else np.zeros(self.n_u)
        affine = (
            self.form == "statewise"
            and (top._evaluations >= _DENSE_REUSE or top._state_space is not None
                 or isinstance(top.lu, _RatioUpdate))
        ) or (self.form == "nodal" and not derivatives
              and (top._evaluations >= len(selected) or isinstance(top.lu, _RatioUpdate)))
        if affine:
            values = top.affine_equations(selected, derivatives=derivatives)(
                np.concatenate((x, u, du))
            )
            if derivatives:
                dx, output = values[:self.n_x], values[self.n_x:]
            else:
                dx, output = None, values
            z = None
        else:
            z = top.solve(x, u, du)
            dx = z[top.xd_index] if derivatives else None
            output = top.selected_outputs(selected) @ z
        self._last = _Evaluation(top, float(t), x, u, du, z)
        top._evaluations += 1
        return y, dx, np.asarray(output, dtype=float)

    def evaluate_outputs(
        self, t: float, y: NDArray[np.float64], selected: Sequence[int]
    ) -> NDArray[np.float64]:
        """Evaluate selected electrical outputs without also forming state derivatives."""
        if self.form == "device":
            return np.asarray(self.evaluate_scalar_outputs(t, y), dtype=float)[list(selected)]
        return self._evaluate_numeric(t, y, selected, derivatives=False)[2]

    def evaluate_with_outputs(self, t: float, y: NDArray[np.float64], selected: Sequence[int]
                              ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the derivatives and selected electrical outputs in one compiled operation."""
        if self.form == "device":
            derivative, output = self.evaluate_scalar_raw(t, y)
            return np.asarray(derivative, dtype=float), np.asarray(output, dtype=float)[list(selected)]
        y, dx, output = self._evaluate_numeric(t, y, selected, derivatives=True)
        if y.size == self.n_x and self._ordered:
            dy = dx
        else:
            dy = np.zeros_like(y)
            dy[self.states] = dx
        return dy, output

    def outputs(self) -> NDArray[np.float64]:
        """Node voltages, branch voltages and branch currents of the last evaluation (see ``output_slice``)."""
        ev = self._last
        if ev is None:
            raise RuntimeError("the netlist has not been evaluated")
        if ev.output is not None and not isinstance(ev.output, np.ndarray):
            ev.output = np.asarray(ev.output, dtype=float)
        if not isinstance(ev.x, np.ndarray):
            ev.x = np.asarray(ev.x, dtype=float)
        if not isinstance(ev.u, np.ndarray):
            ev.u = np.asarray(ev.u, dtype=float)
        if not isinstance(ev.du, np.ndarray):
            ev.du = np.asarray(ev.du, dtype=float)
        if ev.output is None:
            if ev.z is not None:
                ev.output = ev.topology._S @ ev.z
            else:
                ss = ev.topology.state_space
                ev.output = ss.C @ ev.x + ss.D @ ev.u + ss.D1 @ ev.du
        return ev.output

    def output_slice(self, quantity: str, target: object, port: int = 0) -> slice:
        """Where ``quantity`` (``"node_voltage"`` of a node, ``"branch_voltage"`` or ``"branch_current"``
        of an element's port) lies in ``outputs()`` and in the rows of ``StateSpace.C``."""
        if quantity == "node_voltage":
            v = self._vertex.get(id(target))
            if v is None:
                raise KeyError(f"{target!r} is not a node of this netlist")
            return slice(int(self._e_off[v]), int(self._e_off[v]) + self._vertex_dim[v])
        if quantity not in QUANTITIES:
            raise ValueError(f"unknown quantity {quantity!r}; expected one of {QUANTITIES}")
        br = self._branch(target, port)
        base = self.n_e + (self.n_bd if quantity == "branch_current" else 0)
        return slice(base + br.output, base + br.output + br.dim)

    def node_voltage(self, node: Node) -> NDArray[np.float64]:
        return self.outputs()[self.output_slice("node_voltage", node)]

    def branch_voltage(self, element: Element, port: int = 0) -> NDArray[np.float64]:
        return self.outputs()[self.output_slice("branch_voltage", element, port)]

    def branch_current(self, element: Element, port: int = 0) -> NDArray[np.float64]:
        return self.outputs()[self.output_slice("branch_current", element, port)]

    # ---------------------------------------------------------------- energy
    def energy(self, y: NDArray[np.float64]) -> float:
        """Stored energy ``H = sum scale/2 v^T C v + sum scale/2 i^T L i``."""
        H = 0.0
        for el in self._capacitors + self._inductors:
            x = np.asarray(y, dtype=float)[el.state]
            H += 0.5 * self._branch(el).scale * float(x @ ((el.C if isinstance(el, Capacitor) else el.L) @ x))
        return H

    def power(self, element: Element, port: Optional[int] = None) -> float:
        """Power absorbed by an element (or one port) at the last evaluation, ``scale v^T i``."""
        ports = range(len(element.ports)) if port is None else (port,)
        out = self.outputs()
        return sum(self._branch(element, k).scale * float(out[self.output_slice("branch_voltage", element, k)]
                                                          @ out[self.output_slice("branch_current", element, k)])
                   for k in ports)

    def dissipated_power(self, elements: Optional[Sequence[Element]] = None) -> float:
        """Power dissipated in resistors and resistive switches and diodes (of ``elements``, default
        all) at the last evaluation."""
        self.outputs()
        roles = self._last.topology.roles
        selected = self.elements if elements is None else elements
        return sum(
            self.power(el, port)
            for el in selected
            for port, k in enumerate(self._ports[id(el)])
            if roles[k] == "res"
        ) + sum(
            float(el.value(self._last.t)) for el in selected if isinstance(el, Dissipation)
        )

    def supplied_power(self) -> float:
        """Power delivered by the voltage and current sources at the last evaluation."""
        return -sum(
            self.power(src) for src, _ in self._sources
            if isinstance(src, (VoltageSource, CurrentSource))
        )

    # ---------------------------------------------------------------- jumps and diodes
    def project(self, t: float, y: NDArray[np.float64], *,
                topology: Optional[Topology] = None) -> tuple[NDArray[np.float64], float]:
        """Make ``y`` consistent with the current mode's loops and cutsets.

        Capacitors jump by ``x+ = x- + C^-1 B^T lambda`` with ``B x+ = b`` (charge conservation) and
        inductors by ``x+ = x- + L^-1 Q^T mu`` with ``Q x+ = c`` (flux conservation). Returns the
        projected state and the energy dissipated by the jump, ``1/2 dx^T C dx + 1/2 di^T L di >= 0``;
        it equals ``H(y-) - H(y+)`` plus ``impulsive_supply``, the energy the sources delivered
        during the jump (zero when no source lies in the new loops and cutsets).
        """
        y = np.array(y, dtype=float)
        projectors = [p for p in (topology or self.topology()).projectors() if p is not None]
        self.impulsive_supply = 0.0
        if not projectors:
            return y, 0.0
        u, out, loss = self.inputs(t), y.copy(), 0.0
        for proj in projectors:
            x, _mult, part = proj(y, u)
            out[proj.index] = x
            loss += part
        self.impulsive_supply = self.energy(out) - self.energy(y) + loss
        return out, loss

    def consistent(self, t: float, y: NDArray[np.float64],
                   topology: Optional[Topology] = None) -> bool:
        """Whether ``y`` satisfies the current mode's loop and cutset constraints (to round-off), so
        that :meth:`project` would not jump."""
        projectors = [p for p in (topology or self.topology()).projectors() if p is not None]
        if not projectors:
            return True
        y, u = np.asarray(y, dtype=float), self.inputs(t)
        return not any(p.significant(y, p(y, u)[0], 0.0) for p in projectors)

    def _evaluated(self, t: float, y: NDArray[np.float64]) -> tuple[NDArray[np.float64], Topology]:
        """Return outputs and topology after evaluating the requested state and current mode."""
        ev = self._last
        mode = self.mode()
        if (ev is None or ev.t != t or ev.topology.mode.key != mode.key
                or not np.array_equal(ev.x, np.asarray(y, dtype=float)[self.states])):
            self.evaluate(t, y)
            ev = self._last
        return self.outputs(), ev.topology

    def _scales(self, t: float, y: NDArray[np.float64]) -> tuple[float, float]:
        """Voltage and current scales of the circuit at ``(t, y)`` in the current mode."""
        out, _top = self._evaluated(t, y)
        return (float(np.max(np.abs(out[:self.n_e]), initial=0.0)),
                float(np.max(np.abs(out[self.n_e + self.n_bd:]), initial=0.0)))

    def _blocking_voltages(self, top: Topology, potential: NDArray[np.float64]) -> dict[int, float]:
        """Voltages across blocking diodes between sub-networks, from vertex ``potential``s (gauge 0 at
        each floating sub-network's first node) shifted by the offsets that maximize the smallest
        blocking margin: a floating sub-network takes the midpoint of the offsets that keep its diodes
        to ground (directly or through sub-networks placed before) blocking."""
        comp = top.component
        edges = [(k, comp[self._branches[k].a], comp[self._branches[k].b],
                  potential[self._branches[k].a] - potential[self._branches[k].b]) for k in top.blocking]
        scale = _maximum_all([abs(value) for value in potential])
        fixed = {0: 0.0}
        while True:  # v_D = v + c_a - c_b <= 0
            lo: dict[int, float] = {}
            hi: dict[int, float] = {}
            for _k, ca, cb, v in edges:
                if ca in fixed and cb not in fixed:
                    value = v + fixed[ca]
                    lo[cb] = value if cb not in lo else _maximum(lo[cb], value)
                elif cb in fixed and ca not in fixed:
                    value = fixed[cb] - v
                    hi[ca] = value if ca not in hi else _minimum(hi[ca], value)
            if not lo and not hi:
                break
            for c in set(lo) | set(hi):
                if c in lo and c in hi:
                    fixed[c] = 0.5 * (lo[c] + hi[c])
                elif c in lo:
                    fixed[c] = lo[c] + scale
                else:
                    fixed[c] = hi[c] - scale
        return {k: v + fixed.get(ca, 0.0) - fixed.get(cb, 0.0) for k, ca, cb, v in edges}

    def scalar_margins(
        self, t: Any, y: Sequence[Any]
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        """Diode margins, tolerances and scales from the current scalar topology.

        This is the equation-level counterpart of :meth:`_margins`: it avoids numeric coercion so
        exporters can lower the same mode, source and projection laws without a second device model.
        """
        if not self._diodes:
            return (), (), ()
        top = self.topology()
        out = self.scalar_output_values(t, y, topology=top)
        scales = self._margin_scales(out)
        return (self._diode_margins(top, out),
                tuple(_MARGIN_RTOL * scale for scale in scales), scales)

    def scalar_output_values(
        self, t: Any, y: Sequence[Any], *, topology: Optional[Topology] = None
    ) -> tuple[Any, ...]:
        """All electrical outputs without coercing numeric or equation-valued scalars."""
        top = self.topology() if topology is None else topology
        return (self._dynamic_scalar_function(top)(t, y)[1]
                if any(transformer.constant is None for transformer in self._transformers)
                else self._scalar_function(top, "outputs")(t, y))

    def _scalar_port_power(self, out: Sequence[Any], element: Element, port: int) -> Any:
        """Power absorbed at one element port from a scalar output vector."""
        branch = self._branch(element, port)
        voltage = self.n_e + branch.output
        current = voltage + self.n_bd
        return branch.scale * sum(
            out[voltage + index] * out[current + index]
            for index in range(branch.dim)
        )

    def scalar_power(self, out: Sequence[Any], elements: Sequence[Element]) -> Any:
        """Power absorbed by netlist elements from one scalar output vector."""
        total: Any = 0.0
        for element in elements:
            for port in range(len(element.ports)):
                total += self._scalar_port_power(out, element, port)
        return total

    def scalar_dissipated_power(
        self, out: Sequence[Any], elements: Sequence[Element], *, t: Any = 0.0,
        topology: Optional[Topology] = None,
    ) -> Any:
        """Resistive power of selected elements from one scalar output vector."""
        roles = (self.topology() if topology is None else topology).roles
        return sum(
            self._scalar_port_power(out, element, port)
            for element in elements
            for port, branch in enumerate(self._ports[id(element)])
            if roles[branch] == "res"
        ) + sum(
            element.value(t) for element in elements if isinstance(element, Dissipation)
        )

    def _output_scales(self, out: Sequence[Any]) -> tuple[Any, Any]:
        """Voltage and current scales of one electrical output vector."""
        if isinstance(out, np.ndarray):
            return (float(np.max(np.abs(out[:self.n_e]), initial=0.0)),
                    float(np.max(np.abs(out[self.n_e + self.n_bd:]), initial=0.0)))
        return (_maximum_all([abs(value) for value in out[:self.n_e]]),
                _maximum_all([abs(value) for value in out[self.n_e + self.n_bd:]]))

    def _diode_margins(self, top: Topology, out: Sequence[Any]) -> tuple[Any, ...]:
        """Interpret diode branch outputs in the current conduction state."""
        if self._diode_branches is None:
            self._diode_branches = tuple(self._ports[id(diode)][0] for diode in self._diodes)
        branches = self._diode_branches
        if self._diode_output_bases is None:
            self._diode_output_bases = tuple(
                self.n_e + self._branches[index].output for index in branches
            )
        base = self._diode_output_bases
        gated = self._gated()
        if not top.blocking:
            return tuple(
                _select(
                    gated[k], math.inf,
                    _select(diode.conducting, out[index + self.n_bd], -out[index]),
                )
                for k, (diode, index) in enumerate(zip(self._diodes, base))
            )
        voltage = [out[index] for index in base]
        potential: list[Any] = [0.0] * self.n_vertices
        for vertex in range(1, self.n_vertices):
            potential[vertex] = out[int(self._e_off[vertex])]
        blocking = self._blocking_voltages(top, potential)
        voltage = [blocking.get(k, value) for k, value in zip(branches, voltage)]
        current = [out[index + self.n_bd] for index in base]
        margins = [_select(diode.conducting, current[k], -voltage[k])
                   for k, diode in enumerate(self._diodes)]
        return tuple(_select(gated[k], math.inf, value) for k, value in enumerate(margins))

    def _margin_scales(self, out: Sequence[Any]) -> tuple[Any, ...]:
        """Physical scale of each diode margin in its present conduction state."""
        voltage_scale, current_scale = self._output_scales(out)
        gated = self._gated()
        return tuple(_select(
            gated[k], 0.0, _select(diode.conducting, current_scale, voltage_scale)
        ) for k, diode in enumerate(self._diodes))

    def _margin_tolerances(self, out: Sequence[Any]) -> tuple[Any, ...]:
        """Scale-dependent tolerance of each diode margin."""
        return tuple(_MARGIN_RTOL * scale for scale in self._margin_scales(out))

    def projected_values(self, t: Any, y: Sequence[Any]) -> tuple[Any, ...]:
        """State values after the current mode's charge and flux projection."""
        return self.projected_jump(t, y)[0]

    @staticmethod
    def _quadratic(values: Sequence[Any], matrix: Any) -> Any:
        """Return ``values.T @ matrix @ values`` without coercing equation scalars."""
        matrix = getattr(matrix, "_expressions", matrix)
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        return sum(
            coefficient * values[row] * values[column]
            for row in range(len(values))
            for column in range(len(values))
            if not _zero(coefficient := dense[row, column])
        )

    def _scalar_energy(self, y: Sequence[Any]) -> Any:
        """Stored energy for numeric or equation-valued states."""
        total: Any = 0.0
        for element in self._capacitors + self._inductors:
            branch = self._branch(element)
            values = [y[int(index)] for index in element.state]
            matrix = element.C if isinstance(element, Capacitor) else element.L
            total += 0.5 * branch.scale * self._quadratic(values, matrix)
        return total

    def projected_jump(self, t: Any, y: Sequence[Any]) -> tuple[tuple[Any, ...], Any, Any]:
        """Projected states, dissipated jump energy and impulsive source supply."""
        out = list(y)
        inputs = self.source_values(t)
        loss: Any = 0.0
        for projector in (item for item in self.topology().projectors() if item is not None):
            projected = projector.values(y, inputs)[0]
            before = [y[int(index)] for index in projector.index]
            delta = [value - previous for value, previous in zip(projected, before)]
            loss += 0.5 * self._quadratic(delta, projector.W)
            for index, value in zip(projector.index, projected):
                out[int(index)] = value
        supply = self._scalar_energy(out) - self._scalar_energy(y) + loss
        return tuple(out), loss, supply

    def projection_inconsistent(self, t: Any, y: Sequence[Any]) -> Any:
        """Whether the present topology requires a charge- or flux-conserving state jump."""
        inputs, inconsistent = self.source_values(t), False
        for projector in (item for item in self.topology().projectors() if item is not None):
            projected = projector.values(y, inputs)[0]
            reference = _maximum_all([abs(y[int(index)]) for index in projector.index])
            tolerance = _MARGIN_RTOL * reference
            for index, value in zip(projector.index, projected):
                inconsistent = _logical_or(inconsistent, abs(value - y[int(index)]) > tolerance)
        return inconsistent

    def _outputs_in_topology(
        self, topology: Topology, t: float, y: Sequence[float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate all electrical outputs and inputs in an already selected topology."""
        if self.form == "device":
            evaluator = (self._dynamic_scalar_function(topology)
                         if any(item.constant is None for item in self._transformers)
                         else self._scalar_function(topology, "all"))
            result = evaluator(t, y)
            return np.asarray(result[1], dtype=float), np.asarray(result[2], dtype=float)
        x, inputs = np.asarray(y, dtype=float)[self.states], self.inputs(t)
        rates = self.rates(t, topology.needs_rate) if topology.any_rate else np.zeros(self.n_u)
        if self.form == "statewise":
            state_space = topology.state_space
            output = state_space.C @ x + state_space.D @ inputs
            if topology.any_rate:
                output += state_space.D1 @ rates
            return np.asarray(output, dtype=float), inputs
        return np.asarray(topology._S @ topology.solve(x, inputs, rates), dtype=float), inputs

    def _scalar_impulse_flags(
        self, top: Topology, y: Sequence[Any], inputs: Sequence[Any], output: Sequence[Any],
        projections: Optional[Mapping[int, tuple[Sequence[Any], Sequence[Any]]]] = None,
    ) -> tuple[Any, ...]:
        """Per-diode reverse impulse flags from one already evaluated scalar topology."""
        capacitor, inductor = top.projectors()
        if capacitor is None and inductor is None:
            return (False,) * len(self._diodes)
        voltage_scale, current_scale = self._output_scales(output)
        projections = {} if projections is None else projections
        flags: list[Any] = [False] * len(self._diodes)
        gated = self._gated()

        def projected(projector: _Projector) -> tuple[Sequence[Any], Sequence[Any]]:
            return projections.get(id(projector), projector.values(y, inputs))

        if capacitor is not None:
            values, multipliers = projected(capacitor)
            reference = _maximum(
                voltage_scale, _maximum_all([abs(y[int(index)]) for index in capacitor.index])
            )
            changed: Any = False
            for index, value in zip(capacitor.index, values):
                changed = _logical_or(
                    changed, abs(value - y[int(index)]) > _MARGIN_RTOL * reference
                )
            tolerance = _MARGIN_RTOL * _maximum_all([abs(value) for value in multipliers])
            rows = dict(top._wrows)
            for index, diode in enumerate(self._diodes):
                branch = self._ports[id(diode)][0]
                if top.roles[branch] == "short" and not gated[index]:
                    flags[index] = _logical_or(
                        flags[index],
                        _logical_and(changed, multipliers[rows[branch]] < -tolerance),
                    )
        if inductor is not None:
            values, multipliers = projected(inductor)
            reference = _maximum(
                current_scale, _maximum_all([abs(y[int(index)]) for index in inductor.index])
            )
            changed = False
            for index, value in zip(inductor.index, values):
                changed = _logical_or(
                    changed, abs(value - y[int(index)]) > _MARGIN_RTOL * reference
                )
            tolerance = _MARGIN_RTOL * _maximum_all([abs(value) for value in multipliers])
            _net, _c, _l, _cd, _p, _r, _w, offsets, supernodes = top._store
            potential = [multipliers[offsets[root]] if offsets[root] >= 0 else 0.0
                         for root in map(supernodes.find, range(self.n_vertices))]
            blocking = self._blocking_voltages(top, potential)
            for index, diode in enumerate(self._diodes):
                branch = self._ports[id(diode)][0]
                if top.roles[branch] is None:
                    primitive = self._branches[branch]
                    voltage = blocking.get(
                        branch, potential[primitive.a] - potential[primitive.b]
                    )
                    flags[index] = _logical_or(
                        flags[index], _logical_and(changed, voltage > tolerance)
                    )
        return tuple(flags)

    def impulse_flags(self, t: Any, y: Sequence[Any], *,
                      topology: Optional[Topology] = None) -> tuple[Any, ...]:
        """Per-diode reverse impulse flags for a jump into the present topology."""
        top = topology or self.topology()
        capacitor, inductor = top.projectors()
        if capacitor is None and inductor is None:
            return (False,) * len(self._diodes)
        if all(isinstance(value, numbers.Real) for value in y):
            y = np.asarray(y, dtype=float)
            output, inputs = self._outputs_in_topology(top, float(t), y)
        else:
            inputs = self.source_values(t)
            output = self._scalar_function(top, "outputs")(t, y)
        return self._scalar_impulse_flags(top, y, inputs, output)

    def _margin_values(self, t: float, y: NDArray[np.float64], *, evaluated: bool = False
                       ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Conduction margins (on: current, off: -voltage) and evaluated outputs."""
        if evaluated and self._last is not None:
            top = self._last.topology
            out = self._last.output if self._last.output is not None else self.outputs()
        else:
            out, top = self._evaluated(t, y)
        return np.asarray(self._diode_margins(top, out), dtype=float), out

    def _margins(
        self, t: float, y: NDArray[np.float64], topology: Optional[Topology] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Conduction margins, comparison tolerances and their physical scales."""
        if not self._diodes:
            empty = np.zeros(0)
            return empty, empty, empty
        if topology is None:
            margin, out = self._margin_values(t, y)
        else:
            out, _inputs = self._outputs_in_topology(topology, t, y)
            margin = np.asarray(self._diode_margins(topology, out), dtype=float)
        scale = np.asarray(self._margin_scales(out), dtype=float)
        return margin, _MARGIN_RTOL * scale, scale

    def scalar_margin_key(self, top: Topology) -> tuple:
        """Structural equation and gate state that select one diode guard."""
        gated = self._gated_state or (False,) * len(self._diodes)
        return top.mode.key[:2], gated

    def scalar_margin_evaluator(self, top: Topology) -> Callable[[Sequence[float]], float]:
        """Compile the current mode's diode guard from an already evaluated output vector."""
        if self._diode_branches is None:
            self._diode_branches = tuple(
                self._ports[id(diode)][0] for diode in self._diodes
            )
        if self._diode_output_bases is None:
            self._diode_output_bases = tuple(
                self.n_e + self._branches[index].output
                for index in self._diode_branches
            )
        key = self.scalar_margin_key(top)
        gated = key[1]
        cached = self._scalar_margin_functions.get(key)
        if cached is not None:
            self._scalar_margin_functions.move_to_end(key)
            return cached
        active = [
            (diode, base) for index, (diode, base) in enumerate(
                zip(self._diodes, self._diode_output_bases)
            ) if not gated[index]
        ]
        if not active:
            evaluator = lambda _output: math.inf
        elif top.blocking:
            evaluator = lambda output: float(min(
                margin - diode.residual for margin, diode in zip(
                    self._diode_margins(top, output), self._diodes
                )
            ))
        else:
            namespace: dict[str, Any] = {}
            expressions = []
            for index, (diode, base) in enumerate(active):
                name = f"_diode_{index}"
                namespace[name] = diode
                value = (f"output[{base + self.n_bd}]"
                         if diode.conducting else f"-output[{base}]")
                expressions.append(f"({value} - {name}.residual)")
            expression = (expressions[0] if len(expressions) == 1
                          else "min(" + ", ".join(expressions) + ")")
            source = f"def margin(output):\n    return float({expression})\n"
            exec(compile(source, "<peslite diode margin>", "exec"), namespace)
            evaluator = namespace["margin"]
        self._scalar_margin_functions[key] = evaluator
        while len(self._scalar_margin_functions) > self.cache_size:
            self._scalar_margin_functions.popitem(last=False)
        return evaluator

    def commutation_margin(self, t: float, y: NDArray[np.float64], *, evaluated: bool = False) -> float:
        """Minimum diode margin in the current mode (on: current, off: -voltage), relative to the
        residual each margin had at the last commutation; negative when a diode must commutate."""
        if not self._diodes:
            return math.inf
        if evaluated and self._last is not None:
            top = self._last.topology
            out = self._last.output if self._last.output is not None else self.outputs()
        else:
            out, top = self._evaluated(t, y)
        if not top.blocking:
            if self._diode_output_bases is None:
                self._diode_branches = tuple(
                    self._ports[id(diode)][0] for diode in self._diodes
                )
                self._diode_output_bases = tuple(
                    self.n_e + self._branches[index].output
                    for index in self._diode_branches
                )
            gated = self._gated_state or (False,) * len(self._diodes)
            return float(min(
                math.inf if gated[index] else (
                    out[base + self.n_bd] if diode.conducting else -out[base]
                ) - diode.residual
                for index, (diode, base) in enumerate(
                    zip(self._diodes, self._diode_output_bases)
                )
            ))
        return float(min(
            margin - diode.residual
            for margin, diode in zip(self._diode_margins(top, out), self._diodes)
        ))

    def _impulses(self, t: float, y: NDArray[np.float64], topology: Topology) -> NDArray[np.intp]:
        """Diodes that the jump into the current mode would drive in reverse: a negative impulsive
        charge through a conducting ideal diode or a positive impulsive voltage across a blocking one."""
        if not self._diodes:
            return np.zeros(0, dtype=np.intp)
        return np.flatnonzero(np.asarray(
            self.impulse_flags(t, y, topology=topology), dtype=bool
        ))

    def commutate(self, t: float, y: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        """Flip diodes until the conduction pattern is consistent at ``(t, y)``; returns the projected
        state and the dissipated energy (see :meth:`project`).

        A pattern is consistent when the jump into it drives no diode in reverse and every margin
        of the projected state is nonnegative (within round-off; the residual is kept so that
        ``commutation_margin`` starts nonnegative). The starting pattern is left when a margin fell below
        its residual and checked again to round-off if the search returns to it (a grazing zero). Raises
        :class:`ConductionError` when a pattern repeats or none is found within the bound.
        """
        y0, start = np.array(y, dtype=float), list(self.conducting)
        seen: dict[tuple, bool] = {}  # pattern -> checked to the margin tolerance (the start: only strictly)
        try:
            try:
                for attempt in range(_MAX_PATTERNS):
                    pattern = tuple(self.conducting)
                    if seen.get(pattern, False):
                        break
                    seen[pattern] = attempt > 0
                    self._recovery_excluded = set(seen)
                    topology = self.topology()
                    bad = self._impulses(t, y0, topology)
                    if not bad.size:
                        y1, loss = self.project(t, y0, topology=topology)
                        margin, tol, scale = self._margins(t, y1, topology)
                        bad = np.flatnonzero(
                            margin - self._residual < 0.0 if attempt == 0 else margin < -tol
                        )
                        if not bad.size:
                            if attempt:
                                residual = np.minimum(margin, 0.0) - _FLOAT_RTOL * scale
                                for diode, value in zip(self._diodes, residual):
                                    diode.residual = float(value)
                            return y1, loss
                    for k in bad:
                        diode = self._diodes[k]
                        diode.conducting = not diode.conducting
            except ValueError:
                for diode, conducting in zip(self._diodes, start):
                    diode.conducting = conducting
                raise
            current = self.conducting
            for diode, conducting in zip(self._diodes, start):
                diode.conducting = conducting
            names = sorted(d.name for d, a, b in zip(self._diodes, start, current) if a != b) or \
                [d.name for d in self._diodes]
            raise ConductionError(
                f"t = {t:.12g} s: no consistent diode conduction pattern after {len(seen)} "
                f"patterns; diodes {names} keep commutating (Zeno or inconsistent circuit)"
            )
        finally:
            self._recovery_excluded.clear()
