"""Parameter-specialised standalone C++17 simulator backend.

The backend emits one translation unit.  The resolved configuration, state layout, connection
plan and schedules are embedded in that unit; the generated executable needs only the C++17
standard library and never calls Python.
"""

from __future__ import annotations

import copy
import json
import math
import numbers
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..components.converter import _BridgeModel, DCLink, DCVoltageSource
from ..components.network import RCNode, RLBranch, ThreePhaseSource
from ..components.pwm import CarrierComparison, SynchronousCarrier, ZOH
from ..control.blocks import OutputLimit
from ..control.controller import GraphController, VSCController
from ..control.loops import DCVoltageControl, DVOC, SyncLaw, VSG
from .conduction import DiodeConduction
from .params import _unit_items, _unit_leaves, _written_out, dumps
from .exporter import ExportResult, _exporter

__all__ = []


def _number(value: float) -> str:
    """A finite C++ double literal preserving the Python value."""
    value = float(value)
    if not math.isfinite(value):
        if math.isnan(value):
            return "std::numeric_limits<double>::quiet_NaN()"
        return ("-" if value < 0.0 else "") + "std::numeric_limits<double>::infinity()"
    text = repr(value)
    return text if any(ch in text for ch in ".eE") else text + ".0"


def _cpp_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def _identifier(value: str) -> str:
    value = re.sub(r"\W", "_", value)
    if not value or value[0].isdigit():
        value = "_" + value
    return value


def _fields(record: Any) -> tuple[str, ...]:
    """Record fields in their declared order, including dynamic controller-state records."""
    names: list[str] = []
    for cls in reversed(type(record).__mro__):
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.extend(name for name in slots if name not in names)
    dynamic = vars(record) if hasattr(record, "__dict__") else {}
    names.extend(name for name in dynamic if name not in names)
    return tuple(names)


def _complex(value: complex) -> str:
    value = complex(value)
    return f"Complex{{{_number(value.real)}, {_number(value.imag)}}}"


class _Expression:
    """One scalar expression traced from an ordinary custom subsystem equation.

    This exists only while generating C++; generated simulators contain the resulting static
    expression and no tracing or dispatch machinery.
    """

    __slots__ = ("code", "kind")
    __array_priority__ = 1000

    def __init__(self, code: str, kind: str) -> None:
        self.code, self.kind = code, kind

    @staticmethod
    def of(value: Any) -> "_Expression":
        if isinstance(value, _Expression):
            return value
        if isinstance(value, bool):
            return _Expression("true" if value else "false", "bool")
        if isinstance(value, numbers.Real):
            return _Expression(_number(float(value)), "real")
        if isinstance(value, numbers.Complex):
            return _Expression(_complex(complex(value)), "complex")
        raise TypeError(f"unsupported equation value {value!r} ({type(value).__name__})")

    @staticmethod
    def _result(left: "_Expression", right: "_Expression") -> str:
        return "complex" if "complex" in (left.kind, right.kind) else "real"

    def _binary(self, other: Any, op: str) -> "_Expression":
        right = self.of(other)
        if op in ("+", "-") and right.code == "0.0":
            return self
        if op == "+" and self.code == "0.0":
            return right
        if op == "*" and right.code == "1.0":
            return self
        if op == "*" and self.code == "1.0":
            return right
        if op == "*" and (self.code == "0.0" or right.code == "0.0"):
            return _Expression("0.0", self._result(self, right))
        return _Expression(f"(({self.code}) {op} ({right.code}))", self._result(self, right))

    def _rbinary(self, other: Any, op: str) -> "_Expression":
        left = self.of(other)
        if op == "+" and left.code == "0.0":
            return self
        if op == "*" and left.code == "1.0":
            return self
        if op == "*" and (left.code == "0.0" or self.code == "0.0"):
            return _Expression("0.0", self._result(left, self))
        return _Expression(f"(({left.code}) {op} ({self.code}))", self._result(left, self))

    def __add__(self, other: Any): return self._binary(other, "+")
    def __radd__(self, other: Any): return self._rbinary(other, "+")
    def __sub__(self, other: Any): return self._binary(other, "-")
    def __rsub__(self, other: Any): return self._rbinary(other, "-")
    def __mul__(self, other: Any): return self._binary(other, "*")
    def __rmul__(self, other: Any): return self._rbinary(other, "*")
    def __truediv__(self, other: Any): return self._binary(other, "/")
    def __rtruediv__(self, other: Any): return self._rbinary(other, "/")

    def __pow__(self, exponent: Any):
        power = self.of(exponent)
        if power.kind == "real" and power.code == "2.0":
            if self.code.startswith("std::abs(") and self.code.endswith(")"):
                return _Expression(f"std::norm({self.code[9:-1]})", "real")
            return _Expression(f"(({self.code}) * ({self.code}))", self.kind)
        return _Expression(f"std::pow(({self.code}), ({power.code}))", self.kind)

    def __rpow__(self, base: Any):
        left = self.of(base)
        return _Expression(f"std::pow(({left.code}), ({self.code}))", self._result(left, self))

    def __neg__(self): return _Expression(f"-({self.code})", self.kind)
    def __pos__(self): return self
    def __abs__(self): return _Expression(f"std::abs({self.code})", "real")

    @property
    def real(self): return _Expression(f"std::real({self.code})", "real")

    @property
    def imag(self): return _Expression(f"std::imag({self.code})", "real")

    def conjugate(self): return _Expression(f"std::conj({self.code})", self.kind)

    def _compare(self, other: Any, op: str) -> "_Expression":
        right = self.of(other)
        if op not in ("==", "!=") and "complex" in (self.kind, right.kind):
            raise TypeError("complex equation values cannot be ordered")
        return _Expression(f"(({self.code}) {op} ({right.code}))", "bool")

    def __lt__(self, other: Any): return self._compare(other, "<")
    def __le__(self, other: Any): return self._compare(other, "<=")
    def __gt__(self, other: Any): return self._compare(other, ">")
    def __ge__(self, other: Any): return self._compare(other, ">=")
    def __eq__(self, other: Any): return self._compare(other, "==")
    def __ne__(self, other: Any): return self._compare(other, "!=")

    def __bool__(self) -> bool:
        control = _TRACE_CONTROL.get()
        if control is None:
            raise TypeError("a run-time equation value cannot control a Python branch")
        condition = self if self.kind == "bool" else _Expression(
            f"(std::abs({self.code}) != 0.0)", "bool"
        )
        index = control.index
        control.index += 1
        if index >= len(control.decisions):
            raise _TraceBranch(condition)
        return control.decisions[index]

    def __float__(self) -> float:
        raise TypeError("a run-time equation value cannot be converted to a Python float")

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any):
        if method != "__call__" or kwargs:
            return NotImplemented
        name = getattr(ufunc, "__name__", "")
        if name in {"add", "subtract", "multiply", "divide", "true_divide", "power"}:
            left, right = inputs
            operations = {"add": "+", "subtract": "-", "multiply": "*",
                          "divide": "/", "true_divide": "/"}
            if name == "power":
                return self.of(left).__pow__(right)
            return self.of(left)._binary(right, operations[name])
        value = self.of(inputs[0])
        if name in {"absolute", "fabs"}:
            return abs(value)
        if name in {"conjugate", "conj"}:
            return value.conjugate()
        if name == "real":
            return value.real
        if name == "imag":
            return value.imag
        if name == "square":
            return value ** 2
        functions = {
            "sin": "std::sin", "cos": "std::cos", "tan": "std::tan",
            "exp": "std::exp", "log": "std::log", "sqrt": "std::sqrt",
            "sinh": "std::sinh", "cosh": "std::cosh", "tanh": "std::tanh",
        }
        if name in functions:
            kind = "complex" if value.kind == "complex" else "real"
            return _Expression(f"{functions[name]}({value.code})", kind)
        if name == "arctan2":
            left, right = (self.of(item) for item in inputs)
            return _Expression(f"std::atan2({left.code}, {right.code})", "real")
        if name in {"maximum", "minimum"}:
            left, right = (self.of(item) for item in inputs)
            function = "std::max" if name == "maximum" else "std::min"
            return _Expression(
                f"{function}({left.code}, {right.code})",
                self._result(left, right),
            )
        return NotImplemented


@dataclass
class _TraceControl:
    decisions: tuple[bool, ...]
    index: int = 0


class _TraceBranch(Exception):
    def __init__(self, condition: _Expression) -> None:
        self.condition = condition


_TRACE_CONTROL: ContextVar[_TraceControl | None] = ContextVar(
    "peslite_cpp_trace_control", default=None
)


@dataclass(frozen=True)
class _RuntimeParameter:
    path: str
    member: str
    default: float
    positive: bool = False
    above: float | None = None


class _RuntimeParameters:
    """The explicitly selected scalar parameters retained by a generated executable."""

    def __init__(self, p: Any, requested: tuple[str, ...]) -> None:
        available: dict[str, _RuntimeParameter] = {}

        def add(path: str, value: Any, *, positive: bool = False,
                above: float | None = None) -> None:
            if value is not None and not isinstance(value, bool):
                available[path] = _RuntimeParameter(
                    path, "param_" + _identifier(path), float(value), positive, above)

        simulation = p.simulation
        add("simulation.t_end", simulation.t_end, above=simulation.initial.t)
        add("simulation.output.period", simulation.output.period, positive=True)
        solver = simulation.solver
        if solver.type == "fixed":
            add("simulation.solver.dt", solver.dt, positive=True)
        elif solver.type == "adaptive":
            add("simulation.solver.rtol", solver.rtol, positive=True)
            add("simulation.solver.atol", solver.atol, positive=True)
            add("simulation.solver.max_step", solver.max_step, positive=True)
        for name, source in p.sources.items():
            add(f"sources.{name}.v", source.v)
            add(f"sources.{name}.f", source.f if source.f is not None else p.base.f0,
                positive=True)
            add(f"sources.{name}.angle", source.angle)
        for unit_name, unit in _unit_items(p.units):
            references = unit.ctrl.references
            for field in fields(references):
                add(f"units.{unit_name}.ctrl.references.{field.name}",
                    getattr(references, field.name))

        if "all" in requested:
            selected = list(available)
        else:
            selected = list(dict.fromkeys(requested))
        unknown = [path for path in selected if path not in available]
        if unknown:
            known = ", ".join(available)
            raise ValueError(
                f"parameter {unknown[0]!r} is not variable"
                + (f"; available: {known}" if known else "")
            )
        self.available = available
        self.selected = tuple(available[path] for path in selected)
        self.by_path = {item.path: item for item in self.selected}

    def value(self, path: str, value: Any) -> str:
        item = self.by_path.get(path)
        return f"parameters.{item.member}" if item is not None else _number(value)

    def declarations(self) -> str:
        members = "\n    ".join(
            f"double {item.member} = {_number(item.default)};" for item in self.selected
        )
        return f"struct RuntimeParameters {{\n    {members}\n}};" if members else "struct RuntimeParameters {};"

    def setters(self) -> str:
        cases: list[str] = []
        for item in self.selected:
            checks = []
            if item.positive:
                checks.append("!(value > 0.0)")
            if item.above is not None:
                checks.append(f"!(value > {_number(item.above)})")
            if item.path != "simulation.solver.max_step":
                checks.append("!std::isfinite(value)")
            condition = " || ".join(checks)
            validation = (
                f" if ({condition}) throw std::runtime_error(\"invalid value for {item.path}\");"
                if condition else ""
            )
            cases.append(
                f"if (path == {_cpp_string(item.path)}) {{ const double value = parse_number(text);"
                f"{validation} parameters.{item.member} = value; return true; }}"
            )
        return "\n    ".join(cases)

    def listing(self) -> str:
        return "\n    ".join(
            f"std::cout << {_cpp_string(item.path + '  value=')} << parameters.{item.member} << '\\n';"
            for item in self.selected
        )

    def resolved_template(self, p: Any) -> tuple[str, list[tuple[str, str]]]:
        if not self.selected:
            return dumps(p), []
        tree = _written_out(p)
        replacements: list[tuple[str, str]] = []
        for index, item in enumerate(self.selected):
            marker = f"__PESLITE_RUNTIME_PARAMETER_{index}__"
            node = tree
            keys = item.path.split(".")
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = marker
            replacements.append((marker, f"parameters.{item.member}"))
        try:
            import yaml  # type: ignore
            body = yaml.safe_dump(tree, sort_keys=False)
        except ImportError:  # pragma: no cover
            body = json.dumps(tree, indent=2) + "\n"
        return body, replacements


class _ModelGenerator:
    """Turn Model's fixed state and connection plans into straight-line C++."""

    def __init__(self, simulation: Any, runtime: _RuntimeParameters) -> None:
        self.simulation = simulation
        self.p = simulation.p
        self.runtime = runtime
        self.system = simulation.system
        self.model = self.system.model
        self.subsystem_name = {id(sub): name for name, sub in zip(self.model.names, self.model.subsystems)}
        self.prefix = {id(sub): f"s{index}" for index, sub in enumerate(self.model.subsystems)}
        self.record_fields: dict[int, dict[str, str]] = {}
        self.record_types: dict[tuple[int, str], str] = {}
        self._unit_control_inputs: dict[int, dict[str, _Expression]] = {}
        self._solver_observations: dict[tuple[int, str], tuple[str, Any, str]] = {}
        self.state_index: dict[tuple[int, str], tuple[int, bool]] = {
            (id(state), name): (index, is_complex)
            for state, name, index, is_complex in self.model._state_ops
        }
        self.zoh_fields: dict[tuple[int, str], str] = {}
        for label, targets in self.model._zoh_targets.items():
            member = f"hold_{_identifier(label)}"
            for record, field in targets:
                self.zoh_fields[id(record), field] = member
        self._discover_records()
        self.enabled_by = {
            id(subsystem): f"u_{_identifier(name)}_gates"
            for name, unit in self.system.units.items()
            for subsystem in unit.enabled_subsystems()
        }
        self.continuous_units = {
            id(unit.ctrl): _make_unit_generator(name, unit, self)
            for name, unit in self.system.units.items() if unit.continuous
        }

    def unit_control_inputs(self, unit: Any) -> dict[str, _Expression]:
        """Trace one unit's public control boundary once per export."""
        key = id(unit)
        if key not in self._unit_control_inputs:
            self._unit_control_inputs[key] = self.trace_element_values(unit, unit.control_inputs)
        return self._unit_control_inputs[key]

    def _discover_records(self) -> None:
        for sub in self.model.subsystems:
            tag = self.prefix[id(sub)]
            for role in ("state", "inp", "out"):
                record = getattr(sub, role)
                mapped = self.record_fields.setdefault(id(record), {})
                for field in _fields(record):
                    name = mapped.setdefault(field, f"{tag}_{role}_{_identifier(field)}")
                    value = getattr(record, field)
                    self.record_types[id(record), field] = (
                        "Complex" if isinstance(value, complex) else
                        "bool" if isinstance(value, bool) else "double"
                    )

    def value(self, record: Any, field: str) -> str:
        try:
            return self.record_fields[id(record)][field]
        except KeyError as exc:  # an exporter bug, not a configuration restriction
            raise RuntimeError(f"C++ export: unregistered record field {field!r}") from exc

    def _literal(self, value: Any) -> str:
        if isinstance(value, complex):
            return _complex(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        return _number(value)

    @staticmethod
    def _normalise_trace(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _ModelGenerator._normalise_trace(item)
                    for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(_ModelGenerator._normalise_trace(item) for item in value)
        if isinstance(value, (_Expression, numbers.Number)):
            return _Expression.of(value)
        if hasattr(value, "shape") and hasattr(value, "flat"):
            return tuple(_ModelGenerator._normalise_trace(item) for item in value.flat)
        return _Expression.of(value)

    @staticmethod
    def _merge_trace(condition: _Expression, when_false: Any, when_true: Any) -> Any:
        if isinstance(when_false, dict) and isinstance(when_true, dict):
            if when_false.keys() != when_true.keys():
                raise TypeError("a symbolic branch changes the equation output names")
            return {key: _ModelGenerator._merge_trace(condition, when_false[key], when_true[key])
                    for key in when_false}
        if isinstance(when_false, tuple) and isinstance(when_true, tuple):
            if len(when_false) != len(when_true):
                raise TypeError("a symbolic branch changes the number of state derivatives")
            return tuple(_ModelGenerator._merge_trace(condition, left, right)
                         for left, right in zip(when_false, when_true))
        left, right = _Expression.of(when_false), _Expression.of(when_true)
        if (left.code, left.kind) == (right.code, right.kind):
            return left
        kind = "complex" if "complex" in (left.kind, right.kind) else (
            "bool" if left.kind == right.kind == "bool" else "real"
        )
        if kind == "complex":
            if left.kind != "complex":
                left = _Expression(f"Complex{{{left.code}, 0.0}}", "complex")
            if right.kind != "complex":
                right = _Expression(f"Complex{{{right.code}, 0.0}}", "complex")
        return _Expression(
            f"(({condition.code}) ? ({right.code}) : ({left.code}))", kind
        )

    def _trace(self, run: Any, decisions: tuple[bool, ...] = ()) -> Any:
        """Trace pure scalar arithmetic, exploring state-dependent Python branches."""
        if len(decisions) > 12:
            raise TypeError("more than 12 state-dependent branch decisions in one equation")
        token = _TRACE_CONTROL.set(_TraceControl(decisions))
        branch: _TraceBranch | None = None
        try:
            return self._normalise_trace(run())
        except _TraceBranch as exc:
            branch = exc
        finally:
            _TRACE_CONTROL.reset(token)
        assert branch is not None
        when_false = self._trace(run, (*decisions, False))
        when_true = self._trace(run, (*decisions, True))
        return self._merge_trace(branch.condition, when_false, when_true)

    def _equation_clone(self, owner: Any,
                        attributes: dict[str, _Expression] | None = None) -> Any:
        clone = copy.copy(owner)
        for name, value in (vars(owner).items() if hasattr(owner, "__dict__") else ()):
            if getattr(value, "__self__", None) is owner:
                setattr(clone, name, self._rebind(value, clone))
        for role in ("state", "inp", "out"):
            original = getattr(owner, role)
            record = copy.copy(original)
            for field in _fields(original):
                kind = self.record_types[id(original), field]
                setattr(record, field, _Expression(
                    self.value(original, field),
                    "complex" if kind == "Complex" else "bool" if kind == "bool" else "real",
                ))
            setattr(clone, role, record)
        for name, value in (attributes or {}).items():
            setattr(clone, name, value)
        return clone

    @staticmethod
    def _rebind(method: Any, owner: Any) -> Any:
        function = getattr(method, "__func__", None)
        if function is None:
            raise TypeError("equation callable is not a bound Python method")
        return function.__get__(owner, type(owner))

    def trace_outputs(self, method: Any,
                      attributes: dict[str, _Expression] | None = None) -> list[str]:
        owner = method.__self__
        fields = _fields(owner.out)

        def run():
            clone = self._equation_clone(owner, attributes)
            self._rebind(method, clone)(_Expression("t", "real"))
            return tuple(getattr(clone.out, field) for field in fields)

        values = self._trace(run)
        lines: list[str] = []
        for field, value in zip(fields, values):
            original = self.value(owner.out, field)
            if value.code != original:
                lines.append(f"{original} = {value.code};")
        if not lines and fields:
            raise TypeError("output equation did not assign an output field")
        return lines

    def trace_rhs(self, sub: Any, method: Any = None,
                  attributes: dict[str, _Expression] | None = None) -> list[str]:
        method = sub.rhs if method is None else method

        def run():
            clone = self._equation_clone(sub, attributes)
            return tuple(self._rebind(method, clone)(_Expression("t", "real")))

        return [value.code for value in self._trace(run)]

    def trace_scalar_method(self, sub: Any, name: str,
                            attributes: dict[str, _Expression] | None = None) -> str:
        method = getattr(sub, name)

        def run():
            clone = self._equation_clone(sub, attributes)
            return self._rebind(method, clone)()

        return self._trace(run).code

    def trace_solver_method(self, sub: Any, name: str,
                            attributes: dict[str, _Expression] | None = None) -> str:
        """Trace a scalar capability for solver scope, where states are read from ``y``."""
        code = self.trace_scalar_method(sub, name, attributes)
        for (record_id, field), (index, is_complex) in self.state_index.items():
            local = self.record_fields[record_id][field]
            value = (f"Complex{{y[{index}], y[{index + 1}]}}" if is_complex
                     else f"y[{index}]")
            code = code.replace(local, value)
        for (record_id, field), held in self.zoh_fields.items():
            code = code.replace(self.record_fields[record_id][field], held)
        for subsystem in self.model.subsystems:
            subsystem_name = self.subsystem_name[id(subsystem)]
            for role in ("inp", "out"):
                record = getattr(subsystem, role)
                for field in _fields(record):
                    local = self.record_fields[id(record)][field]
                    if local not in code:
                        continue
                    code = code.replace(
                        local,
                        self.solver_observation(record, field, subsystem_name, role),
                    )
        return code

    def solver_observation(self, record: Any, field: str,
                           subsystem_name: str | None = None,
                           role: str | None = None) -> str:
        """Return a persistent observation usable by solver helpers outside ``evaluate``."""
        key = (id(record), field)
        if key not in self._solver_observations:
            if subsystem_name is None or role is None:
                owner = next(
                    (sub for sub in self.model.subsystems
                     if record is getattr(sub, "inp") or record is getattr(sub, "out")),
                    None,
                )
                if owner is None:
                    raise RuntimeError("solver observation record has no subsystem owner")
                subsystem_name = self.subsystem_name[id(owner)]
                role = "inp" if record is owner.inp else "out"
            observation_name = f"solver.{subsystem_name}.{role}.{field}"
            self._solver_observations[key] = (observation_name, record, field)
        return "obs_" + _identifier(self._solver_observations[key][0])

    def _diode_conduction(self, sub: RLBranch, *, solver_scope: bool = False) -> tuple[str, str]:
        """Render the shared averaged half-bridge diode law for C++."""
        conduction = sub.conduction
        if conduction is None:
            raise RuntimeError("an averaged diode expression needs a conduction law")
        if solver_scope:
            index, is_complex = self.state_index[id(sub.state), "i"]
            if is_complex:
                raise TypeError("averaged diode conduction requires a scalar current")
            current = f"y[{index}]"
            held = self.zoh_fields.get((id(sub.inp), "q"))
            q = held if held is not None else self.solver_observation(sub.inp, "q")
            value = lambda field: self.solver_observation(sub.inp, field)
        else:
            current = self.value(sub.state, "i")
            q = self.value(sub.inp, "q")
            value = lambda field: self.value(sub.inp, field)
        result = self._trace(lambda: conduction.equations(
            _Expression(current, "real"),
            _Expression(q, "real"),
            _Expression(value("u1"), "real"),
            _Expression(value("u2"), "real"),
            _Expression(value("u_dc"), "real"),
            sub.L,
            sub.R,
        ))
        return result[4].code, result[5].code

    def _rl_derivative(self, sub: RLBranch, current: str, value: Any) -> str:
        return (
            f"({value('u1')} - {value('u2')} - {_number(sub.R)} * {current}) / "
            f"{_number(sub.L)}"
        )

    def trace_element_signals(self, element: Any) -> dict[str, _Expression]:
        return self.trace_element_values(element, element.signals)

    def trace_element_values(self, element: Any, read: Any) -> dict[str, _Expression]:
        """Trace values returned by an assembly object from its subsystem records."""
        # An assembly object's public boundary may read a connected network subsystem which it
        # does not own (for example a converter reads its terminal bus voltage). Rebind every
        # assembled record so this generic tracing path follows those references as well.
        subsystems = tuple(self.model.subsystems)

        def run():
            saved: list[tuple[Any, str, Any]] = []
            try:
                for sub in subsystems:
                    for role in ("state", "inp", "out"):
                        record = getattr(sub, role)
                        for field in _fields(record):
                            value = getattr(record, field)
                            saved.append((record, field, value))
                            kind = self.record_types[id(record), field]
                            setattr(record, field, _Expression(
                                self.value(record, field),
                                "complex" if kind == "Complex" else
                                "bool" if kind == "bool" else "real",
                            ))
                return read()
            finally:
                for record, field, value in reversed(saved):
                    setattr(record, field, value)

        return self._trace(run)

    def declarations(self) -> list[str]:
        lines: list[str] = []
        declared: set[str] = set()
        for sub in self.model.subsystems:
            for record in (sub.state, sub.inp, sub.out):
                for field in _fields(record):
                    name = self.value(record, field)
                    if name in declared:
                        continue
                    declared.add(name)
                    state = self.state_index.get((id(record), field))
                    kind = self.record_types[id(record), field]
                    if state is not None:
                        index, is_complex = state
                        initial = (f"Complex{{y[{index}], y[{index + 1}]}}" if is_complex
                                   else f"y[{index}]")
                    elif (id(record), field) in self.zoh_fields:
                        initial = self.zoh_fields[id(record), field]
                    else:
                        initial = self._literal(getattr(record, field))
                    lines.append(f"{kind} {name} = {initial};")
        for generator in self.continuous_units.values():
            lines.extend(generator.declarations())
        return lines

    def _copy_expression(self, item: tuple) -> str:
        _dst, _field, source, source_field, fanin = item
        if fanin is None:
            return self.value(source, source_field)
        terms = [f"({_number(gain)} * {self.value(out, name)})" for out, name, gain in fanin]
        return " + ".join(terms) if terms else "0.0"

    def _source_values(self, source: ThreePhaseSource) -> tuple[str, str]:
        entry = next(((name, item) for name, item in self.system.sources.items()
                      if item.emf is source), None)
        wrapper = entry[1] if entry is not None else None
        if wrapper is None:
            return _number(source.e_peak), "0.0"
        source_name = entry[0]
        scenario = wrapper.scenario

        def piecewise(starts, expressions):
            result = expressions[0]
            for start, expression in zip(starts[1:], expressions[1:]):
                result = f"(t >= {_number(start)} ? {expression} : {result})"
            return result

        magnitudes = [_number(value) for value in scenario._v]
        magnitudes[0] = self.runtime.value(f"sources.{source_name}.v", scenario._v[0])
        magnitude = piecewise(scenario._v_from, magnitudes)
        phases = []
        accumulated = "0.0"
        previous_start = 0.0
        previous_frequency = ""
        for index, (start, (frequency, angle, _phase)) in enumerate(
                zip(scenario._angle_from, scenario._angle)):
            if index:
                duration = start - previous_start
                accumulated = (
                    f"({accumulated} + 2.0 * pi * ({previous_frequency} - "
                    f"{_number(scenario.f0)}) * {_number(duration)})"
                )
            if index == 0:
                frequency_value = self.runtime.value(f"sources.{source_name}.f", frequency)
                angle_value = self.runtime.value(f"sources.{source_name}.angle", angle)
            else:
                frequency_value = _number(frequency)
                angle_value = _number(angle)
            origin = max(start, 0.0)
            phases.append(
                f"({accumulated} + {angle_value} + 2.0 * pi * "
                f"({frequency_value} - {_number(scenario.f0)}) * (t - {_number(origin)}))"
            )
            previous_start, previous_frequency = origin, frequency_value
        return magnitude, piecewise(scenario._angle_from, phases)

    def _output(self, method: Any) -> list[str]:
        owner = method.__self__
        name = method.__name__
        inp, out, state = owner.inp, owner.out, owner.state
        if isinstance(owner, ThreePhaseSource):
            magnitude, phi = self._source_values(owner)
            return [
                f"{self.value(out, 'phi')} = {phi};",
                f"{self.value(out, 'theta')} = {_number(owner.w0)} * t + {self.value(out, 'phi')};",
                f"{self.value(out, 'e_g')} = {magnitude} * Complex{{std::cos({self.value(out, 'theta')}), std::sin({self.value(out, 'theta')})}};",
            ]
        if isinstance(owner, RLBranch):
            return [f"{self.value(out, 'i')} = {self.value(state, 'i')};"]
        if isinstance(owner, RCNode):
            return [f"{self.value(out, 'u')} = {self.value(state, 'u_C')} + {_number(owner.R_d)} * {self.value(inp, 'i')};"]
        if isinstance(owner, _BridgeModel):
            if name == "set_dc_current":
                return self.trace_outputs(method)
            if name == "set_ac_voltage":
                return [f"{self.value(out, 'u_c')} = {self.value(inp, 'q')} * {self.value(inp, 'u_dc')};"]
        if isinstance(owner, DCLink):
            return self._dclink_output(owner, name)
        generator = self.continuous_units.get(id(owner))
        if generator is not None:
            return generator.output_lines()
        hook = getattr(owner, "cpp_outputs", None)
        if hook is not None:
            return list(hook(self))
        label = self.subsystem_name.get(id(owner), type(owner).__name__)
        try:
            enabled = self.enabled_by.get(id(owner))
            attributes = ({"enabled": _Expression(enabled, "bool")}
                          if enabled is not None else None)
            return self.trace_outputs(method, attributes)
        except Exception as exc:
            raise TypeError(
                f"C++ export: cannot lower the output equation of subsystem {label!r} "
                f"({type(owner).__name__}): {exc}"
            ) from exc

    def _dclink_output(self, link: DCLink, name: str) -> list[str]:
        inp, out, state = link.inp, link.out, link.state
        i_dc = self.value(inp, "i_dc")
        u_dc, u_c, i_src = (self.value(out, field) for field in ("u_dc", "u_C", "i_src"))
        cap, source = link.capacitor, link.source
        if name == "set_stored_voltage":
            value = self.value(state, "u_C") if cap is not None else _number(source.u_dc)
            return [f"{u_c} = {value};"]
        if cap is None:
            return [f"{u_dc} = {u_c} = {_number(source.u_dc)} - {_number(source.R)} * {i_dc};",
                    f"{i_src} = {i_dc};"]
        stored = self.value(state, "u_C")
        if source is None:
            current = "0.0"
        elif hasattr(source, "u_dc"):
            current = (f"(dclink_tripped_{self.prefix[id(link)]} ? 0.0 : "
                       f"({_number(source.u_dc)} - {stored} + {_number(cap.R_esr)} * {i_dc}) / "
                       f"{_number(source.R + cap.R_esr)})")
        else:
            ramp = f"source_ramp_{self.prefix[id(link)]}(t)"
            current = (f"(dclink_tripped_{self.prefix[id(link)]} ? 0.0 : "
                       f"{ramp} * {_number(source.i_nom)} + {_number(source.k_dc)} * "
                       f"({_number(source.u_ref)} - {stored}))")
        return [f"{i_src} = {current};", f"{u_c} = {stored};",
                f"{u_dc} = {stored} + {_number(cap.R_esr)} * ({i_src} - {i_dc});"]

    def _operation(self, operation: tuple, indent: str = "") -> list[str]:
        kind, item = operation
        if kind == "out":
            return [indent + line for line in self._output(item)]
        if kind == "copy":
            return [indent + f"{self.value(item[0], item[1])} = {self._copy_expression(item)};"]
        if kind == "algebraic":
            return self._algebraic(item, indent)
        raise RuntimeError(f"C++ export: unknown model operation {kind!r}")

    def _algebraic(self, loop: Any, indent: str) -> list[str]:
        lines = [indent + "for (int algebraic_sweep = 0; algebraic_sweep < 16; ++algebraic_sweep) {"]
        for tear in loop.tears:
            lines.append(indent + "    " + f"{self.value(tear[0], tear[1])} = {self._copy_expression(tear)};")
        for operation in loop.operations:
            lines.extend(self._operation(operation, indent + "    "))
        checks = []
        for index, tear in enumerate(loop.tears):
            current = self.value(tear[0], tear[1])
            target = self._copy_expression(tear)
            kind = self.record_types[id(tear[0]), tear[1]]
            delta = f"std::abs({current} - ({target}))"
            scale = f"std::max({{1.0, std::abs({current}), std::abs({target})}})"
            checks.append(f"({delta} <= 1e-10 + 1e-8 * {scale})")
        lines.append(indent + "    if (" + " && ".join(checks or ["true"]) + ") break;")
        lines.append(indent + "}")
        return lines

    def _rhs(self, sub: Any) -> list[str]:
        state, inp, out = sub.state, sub.inp, sub.out
        if isinstance(sub, RLBranch):
            i = self.value(state, "i")
            zero = ("Complex{0.0, 0.0}"
                    if self.record_types[id(state), "i"] == "Complex" else "0.0")
            value = lambda field: self.value(inp, field)
            raw = (self._diode_conduction(sub)[0] if sub.conduction is not None
                   else self._rl_derivative(sub, i, value))
            if sub.unilateral:
                raw = f"(({i} <= 0.0 && ({raw}) < 0.0) ? 0.0 : ({raw}))"
            derivative = f"(breaker_open_{self.prefix[id(sub)]} ? {zero} : ({raw}))"
            return [derivative]
        if isinstance(sub, RCNode):
            return [f"{self.value(inp, 'i')} / {_number(sub.C)}"]
        if isinstance(sub, (ThreePhaseSource, _BridgeModel)):
            return []
        if isinstance(sub, DCLink):
            if sub.capacitor is None:
                return []
            return [f"({self.value(out, 'i_src')} - {self.value(inp, 'i_dc')}) / {_number(sub.capacitor.C)}"]
        generator = self.continuous_units.get(id(sub))
        if generator is not None:
            return generator.derivatives()
        enabled = self.enabled_by.get(id(sub))
        if (enabled is not None and hasattr(sub, "_active_rhs")
                and hasattr(sub, "_inactive_rhs")):
            active = self.trace_rhs(sub, sub._active_rhs)
            disabled = self.trace_rhs(sub, sub._inactive_rhs)
            return [f"({enabled} ? ({on}) : ({off}))"
                    for on, off in zip(active, disabled)]
        if (getattr(sub, "continuous", False) and hasattr(sub, "graph")
                and hasattr(sub, "state_names")):
            hook = getattr(sub, "cpp_continuous_rhs", None)
            if hook is not None:
                return list(hook(self))
            label = self.subsystem_name.get(id(sub), "controller")
            raise TypeError(f"C++ export: continuous controller {label!r} is not yet lowered")
        hook = getattr(sub, "cpp_rhs", None)
        if hook is not None:
            return list(hook(self))
        label = self.subsystem_name.get(id(sub), type(sub).__name__)
        try:
            return self.trace_rhs(sub)
        except Exception as exc:
            raise TypeError(
                f"C++ export: cannot lower the state equation of subsystem {label!r} "
                f"({type(sub).__name__}): {exc}"
            ) from exc

    def body(self, derivatives: bool = True) -> list[str]:
        lines = self.declarations()
        for operation, rhs_required in zip(self.model._plan, self.model._plan_rhs):
            if derivatives and not rhs_required:
                continue
            lines.extend(self._operation(operation))
        if not derivatives:
            return lines
        for sub, entries in self.model._rhs_layout:
            derivatives = self._rhs(sub)
            if len(derivatives) != len(entries):
                raise RuntimeError(
                    f"C++ export: {self.subsystem_name[id(sub)]} emitted {len(derivatives)} "
                    f"derivatives for {len(entries)} states"
                )
            for expression, (index, is_complex) in zip(derivatives, entries):
                if is_complex:
                    temp = f"d_{self.prefix[id(sub)]}_{index}"
                    lines += [f"const Complex {temp} = {expression};",
                              f"dy[{index}] = std::real({temp});",
                              f"dy[{index + 1}] = std::imag({temp});"]
                else:
                    lines.append(f"dy[{index}] = {expression};")
        return lines

    def members(self) -> list[str]:
        lines = []
        for label, targets in self.model._zoh_targets.items():
            record, field = targets[0]
            value = getattr(record, field)
            lines.append(
                f"{self.record_types[id(record), field]} hold_{_identifier(label)} = {self._literal(value)};"
            )
        for sub in self.model.subsystems:
            if isinstance(sub, RLBranch):
                lines.append(f"bool breaker_open_{self.prefix[id(sub)]} = {'true' if sub.breaker_open else 'false'};")
            if isinstance(sub, DCLink):
                lines.append(f"bool dclink_tripped_{self.prefix[id(sub)]} = {'true' if sub.tripped else 'false'};")
                source = sub.source
                if source is not None and hasattr(source, "i_nom"):
                    unit = next((unit for unit in self.system.units.values()
                                 if unit.dclink is sub), None)
                    expression = "1.0"
                    if unit is not None and unit.scenario._commands:
                        expression = "0.0" if unit.scenario._commands[0][1] else "1.0"
                        for at, on, ramp in unit.scenario._commands:
                            value = ("0.0" if not on else "1.0" if ramp <= 0.0 else
                                     f"(t >= {_number(at + ramp)} ? 1.0 : "
                                     f"smoothstep((t - {_number(at)}) / {_number(ramp)}))")
                            expression = f"(t >= {_number(at)} ? {value} : {expression})"
                    lines.append(
                        f"double source_ramp_{self.prefix[id(sub)]}(double t) const noexcept "
                        f"{{ return {expression}; }}"
                    )
        for generator in self.continuous_units.values():
            lines.extend(generator.members())
        return lines

    def observations(self) -> tuple[list[str], list[str]]:
        """Members and assignments for values consumed by scheduling and public output."""
        members: list[str] = []
        assigns: list[str] = []
        seen: set[str] = set()

        def add(name: str, expression: str, kind: str) -> None:
            member = f"obs_{_identifier(name)}"
            if member not in seen:
                seen.add(member)
                members.append(f"{kind} {member}{{}};")
            assigns.append(f"{member} = {expression};")

        for name, node in self.system.buses.items():
            add(f"{name}.u", self.value(node.out, "u"),
                self.record_types[id(node.out), "u"])
        for name, branch in self.system.branches.items():
            add(f"{name}.i", self.value(branch.out, "i"),
                self.record_types[id(branch.out), "i"])
        for name, source in self.system.sources.items():
            add(f"{name}.i", self.value(source.branch.out, "i"), "Complex")
            add(f"{name}.angle", self.value(source.emf.out, "phi"), "double")
        for name, unit in self.system.units.items():
            if getattr(unit, "branch_f", None) is not None and hasattr(unit, "dclink"):
                add(f"{name}.i_c", self.value(unit.branch_f.out, "i"),
                    self.record_types[id(unit.branch_f.out), "i"])
                add(f"{name}.u_g", self.value(unit.bus.out, "u"),
                    self.record_types[id(unit.bus.out), "u"])
                add(f"{name}.u_dc", self.value(unit.dclink.out, "u_dc"), "double")
                add(f"{name}.i_dc", self.value(unit.dclink.inp, "i_dc"), "double")
            else:
                for signal, value in self.trace_element_signals(unit).items():
                    kind = "Complex" if value.kind == "complex" else (
                        "bool" if value.kind == "bool" else "double"
                    )
                    add(signal, value.code, kind)
            if hasattr(unit.ctrl, "log_sources"):
                for port, value in self.unit_control_inputs(unit).items():
                    kind = "Complex" if value.kind == "complex" else (
                        "bool" if value.kind == "bool" else "double"
                    )
                    add(f"{name}.ctrl.{port}", value.code, kind)
        for name, element in self.system.named_elements.items():
            signals = getattr(element, "signals", None)
            if signals is not None:
                values = signals()
                if hasattr(element, "branch") and set(values) == {f"{name}.i"}:
                    add(f"{name}.i", self.value(element.branch.out, "i"), "Complex")
                    continue
                try:
                    traced = self.trace_element_signals(element)
                except Exception as exc:
                    if self.p.simulation.output.signals:
                        raise TypeError(
                            f"C++ export: cannot lower signals of element {name!r} "
                            f"({type(element).__name__}): {exc}"
                        ) from exc
                    continue
                for signal, value in traced.items():
                    kind = "Complex" if value.kind == "complex" else (
                        "bool" if value.kind == "bool" else "double"
                    )
                    add(signal, value.code, kind)
        for source in self.system.composite_signal_sources:
            for signal, value in self.trace_element_signals(source).items():
                kind = "Complex" if value.kind == "complex" else (
                    "bool" if value.kind == "bool" else "double"
                )
                add(signal, value.code, kind)
        for name, record, field in self._solver_observations.values():
            add(name, self.value(record, field), self.record_types[id(record), field])
        return members, assigns


class _ControlGenerator:
    """Shared names and configuration expressions for generated unit controllers."""

    def __init__(self, name: str, unit: Any, model: _ModelGenerator, prefix: str) -> None:
        self.name, self.unit, self.model = name, unit, model
        self.tag = _identifier(name)
        self.cfg, self.ctrl = unit.cfg, unit.ctrl
        self.graph = self.ctrl.graph
        self.prefix = f"{prefix}_{self.tag}_"

    @property
    def p(self):
        return self.model.p

    def m(self, name: str) -> str:
        return self.prefix + _identifier(name)

    def g(self, key: str) -> str:
        return self.m("g_" + key)

    def node(self, name: str, field: str) -> str:
        return self.m(f"n_{name}_{field}")

    def d(self, name: str, field: str) -> str:
        return self.m(f"d_{name}_{field}")

    def _sampled_loop_members(self) -> list[str]:
        """Declare sampled loop state and carried runtime values from their own contracts."""
        lines: list[str] = []
        for loop_name, node in self.graph.nodes.items():
            for local, value in node.get_state().items():
                kind = "Complex" if isinstance(value, complex) else "double"
                lines.append(
                    f"{kind} {self.node(loop_name, local)} = {self._literal(value)};"
                )
            if isinstance(node, DCVoltageControl):
                lines.append(
                    f"bool {self.node(loop_name, 'limit_exceeded')} = "
                    f"{'true' if node.limit_exceeded else 'false'};"
                )
            for carried in node.carried:
                value = getattr(node, carried)
                if carried == "first_limit_t":
                    lines.append(
                        f"double {self.node(loop_name, carried)} = "
                        f"{_number(value if value is not None else math.nan)};"
                    )
                    continue
                kind = ("bool" if isinstance(value, bool) else
                        "std::size_t" if isinstance(value, int) else
                        "Complex" if isinstance(value, complex) else "double")
                lines.append(
                    f"{kind} {self.node(loop_name, carried)} = {self._literal(value)};"
                )
        return lines

    def _effective_command(self, on: bool) -> str:
        """C++ expression for a host command after the unit trip latch is applied."""
        return f"!{self.m('tripped')}" if on else "false"

    def _controller_command_lines(self, on: bool, ramp: float, t: str = "t") -> list[str]:
        """Lower the controller's shared run/stop and start-up state transition."""
        effective = self._effective_command(on)
        lines: list[str] = []
        if on:
            lines += [f"if ({effective} && !{self.m('startup_run')}) {{",
                      *["    " + line for line in self._reset_integrators()], "}"]
        lines.append(f"{self.m('startup_run')} = {effective};")
        if self.continuous:
            lines += [f"{self.m('startup_ramp')} = {_number(ramp)};",
                      f"{self.m('startup_start')} = {t};"]
        else:
            if on:
                lines.append(
                    f"if ({effective}) {self.m('startup_ramp')} = {_number(ramp)};"
                )
            lines += [f"{self.m('startup_steps')} = 0;",
                      f"{self.m('startup_value')} = 0.0;",
                      f"{self.m('startup_complete')} = false;"]
        return lines

    @staticmethod
    def _literal(value: Any) -> str:
        if isinstance(value, complex):
            return _complex(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        return _number(value)

    def _reference(self, name: str) -> str:
        values = [(self.p.simulation.initial.t, getattr(self.cfg.ctrl.references, name))]
        for change in self.p.changes:
            current = getattr(change.params.unit(self.name).ctrl.references, name)
            if current != values[-1][1]:
                values.append((change.t, current))
        path = f"units.{self.name}.ctrl.references.{name}"
        expression = self.model.runtime.value(path, values[0][1])
        for at, value in values[1:]:
            expression = f"(t >= {_number(at)} ? {self._literal(value)} : {expression})"
        return expression

    def input(self, loop: str, port: str) -> str:
        return self.source(self.graph.connections[f"{loop}.{port}"])

    def derivatives(self) -> list[str]:
        return [self.d(*name.split(".", 1)) for name in self.ctrl.state_names]

    def _state(self, loop: str, local: str) -> str:
        if self.continuous:
            return self.model.value(self.ctrl.state, f"{loop}.{local}")
        return self.state_expression(self.ctrl, f"{loop}.{local}")

    @staticmethod
    def _local(block_name: str, local: str) -> str:
        return block_name if not local else f"{block_name}.{local}"

    @staticmethod
    def _copy_block(block: Any) -> Any:
        """Copy a dynamic block without invoking factories implemented through ``__new__``."""
        cloned = object.__new__(type(block))
        if hasattr(block, "__dict__"):
            vars(cloned).update(vars(block))
        for cls in reversed(type(block).__mro__):
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if hasattr(block, name):
                    setattr(cloned, name, getattr(block, name))
        return cloned

    def _clone_loop(self, loop_name: str, node: Any) -> Any:
        clone = copy.copy(node)
        if node.startup is not None:
            startup = copy.copy(node.startup)
            startup.run = _Expression(self.m("startup_run"), "bool")
            startup.value = _Expression(self.m("startup_value"), "real")
            clone.startup = startup

        bindings = node.flow_state_bindings()
        if bindings is None:
            raise TypeError(
                f"C++ export: {self.name}.{loop_name} does not expose direct state bindings"
            )
        owners = {id(owner): owner for owner, _attribute in bindings.values()
                  if owner is not node}
        cloned_owners = {key: self._copy_block(owner) for key, owner in owners.items()}
        for public, (owner, attribute) in bindings.items():
            target = clone if owner is node else cloned_owners[id(owner)]
            value = node.get_state()[public]
            kind = "complex" if isinstance(value, complex) else "real"
            setattr(target, attribute, _Expression(self._state(loop_name, public), kind))

        clone._state_blocks = {
            name: cloned_owners.get(id(block), self._copy_block(block))
            for name, block in node._state_blocks.items()
        }
        for attribute, value in vars(node).items():
            replacement = cloned_owners.get(id(value))
            if getattr(value, "__self__", None) is node:
                setattr(clone, attribute, self.model._rebind(value, clone))
            elif replacement is not None:
                setattr(clone, attribute, replacement)
            elif isinstance(value, OutputLimit):
                limit = copy.copy(value)
                if limit.pi is not None and id(limit.pi) in cloned_owners:
                    limit.pi = cloned_owners[id(limit.pi)]
                setattr(clone, attribute, limit)

        if not self.continuous:
            for name in node.carried:
                if name == "first_limit_t":
                    continue
                value = getattr(node, name)
                kind = "bool" if isinstance(value, bool) else "real"
                setattr(clone, name, _Expression(self.node(loop_name, name), kind))
        return clone

    def _trace_loop(self, loop_name: str, node: Any) -> tuple[tuple[_Expression, ...],
                                                               tuple[str, ...], bool, bool]:
        inputs = tuple(_Expression(self.input(loop_name, port),
                                   "complex" if spec.complex_value else "real")
                       for port, spec in node.inputs.items())
        input_map = dict(zip(node.inputs, inputs))
        state_names = tuple(node.flow_state() if self.continuous else node.get_state())
        carried = (() if self.continuous else
                   tuple(name for name in node.carried if name != "first_limit_t"))
        internal_limit = isinstance(node, DCVoltageControl)
        external_limit = not self.continuous and loop_name in self.graph._limits

        def run():
            clone = self._clone_loop(loop_name, node)
            external_saturated: Any = False
            if self.continuous:
                if node.flow_inputs is None:
                    outputs, derivatives = clone.flow(_Expression("t", "real"), input_map)
                    values = tuple(outputs[name] for name in node.outputs) + tuple(
                        derivatives[name] for name in node.flow_state()
                    )
                else:
                    values = tuple(clone.flow_path(*inputs))
            else:
                outputs = clone.sample(_Expression("t", "real"), input_map)
                if external_limit:
                    port, original = self.graph._limits[loop_name]
                    limit = copy.copy(original)
                    if limit.pi is node.pi:
                        limit.pi = clone.pi
                    if original is self.ctrl._terminal_limit:
                        limit.limit = self.ctrl.stage.command_limit(
                            _Expression(self.source("meas.u_dc"), "real")
                        )
                    limit.active = lambda: clone.startup.active
                    limit.observer = None
                    outputs[port] = limit.sample(_Expression("t", "real"), outputs[port])
                    external_saturated = limit.saturated
                state = clone.get_state()
                values = tuple(outputs[name] for name in node.outputs) + tuple(
                    state[name] for name in state_names
                )
            values += tuple(getattr(clone, name) for name in carried)
            if internal_limit:
                values += (clone.limit.saturated,)
            if external_limit:
                values += (external_saturated,)
            return values

        return tuple(self.model._trace(run)), carried, internal_limit, external_limit

    def _loop_lines(self, loop_name: str, node: Any) -> list[str]:
        values, carried, internal_limit, external_limit = self._trace_loop(loop_name, node)
        temporary = tuple(
            self.m(f"value_{loop_name}_{index}") for index in range(len(values))
        )
        lines = [f"const auto {name} = {value.code};"
                 for name, value in zip(temporary, values)]
        n_outputs = len(node.outputs)
        state_names = tuple(node.flow_state() if self.continuous else node.get_state())
        n_states = len(state_names)
        lines += [f"{self.g(loop_name + '.' + port)} = {value};"
                  for port, value in zip(node.outputs, temporary[:n_outputs])]
        state_values = temporary[n_outputs:n_outputs + n_states]
        if self.continuous:
            lines += [f"{self.d(loop_name, local)} = {value};"
                      for local, value in zip(state_names, state_values)]
        else:
            lines += [f"{self._state(loop_name, local)} = {value};"
                      for local, value in zip(state_names, state_values)]
        offset = n_outputs + n_states
        lines += [f"{self.node(loop_name, name)} = {value};"
                  for name, value in zip(carried,
                                         temporary[offset:offset + len(carried)])]
        offset += len(carried)
        if internal_limit:
            target = (self.m(loop_name + "_limit_active") if self.continuous
                      else self.node(loop_name, "limit_exceeded"))
            lines.append(f"{target} = {temporary[offset]};")
            offset += 1
            if not self.continuous and "first_limit_t" in node.carried:
                first = self.node(loop_name, "first_limit_t")
                lines.append(
                    f"if ({target} && !std::isfinite({first})) {first} = t;"
                )
        if external_limit:
            lines.append(f"{self.m('command_saturated')} = {temporary[offset]};")
        if self.continuous:
            return lines
        period = self.graph.periods[loop_name]
        every = max(1, int(round(period / self.cfg.ctrl.period)))
        if every == 1:
            return lines
        guard = f"{self.m('pwm_k')} % {every} == 0"
        return [f"if ({guard}) {{", *["    " + line for line in lines], "}"]

    def _reset_integrators(self) -> list[str]:
        """Trace each loop's own reset operation and assign only states it changes."""
        lines: list[str] = []
        for loop_name, node in self.graph.nodes.items():
            state_names = tuple(node.get_state())

            def run():
                clone = self._clone_loop(loop_name, node)
                clone.reset_integrator()
                state = clone.get_state()
                return tuple(state[name] for name in state_names)

            values = tuple(self.model._trace(run))
            for local, value in zip(state_names, values):
                current = self._state(loop_name, local)
                if value.code == current:
                    continue
                if self.continuous:
                    index, is_complex = self.model.state_index[
                        (id(self.ctrl.state), f"{loop_name}.{local}")
                    ]
                    if is_complex:
                        lines += [f"y[{index}] = std::real({value.code});",
                                  f"y[{index + 1}] = std::imag({value.code});"]
                    else:
                        lines.append(f"y[{index}] = {value.code};")
                else:
                    lines.append(f"{current} = {value.code};")
        return lines

    def limit_observation(self) -> list[str]:
        return []

    @staticmethod
    def adc_methods() -> str:
        return ""

    @staticmethod
    def adc_accumulate_call(_duration: str) -> str:
        return ""

    @staticmethod
    def adc_seed_call() -> str:
        return ""

    def controller_summary(self, stream: str = "summary") -> list[str]:
        return []

    def summary_lines(self, stream: str = "summary") -> list[str]:
        return [*self.protection_summary(stream), *self.controller_summary(stream)]

    def _topology_expression(self, method: Any, values: tuple[str, ...]) -> str:
        result = self.model._trace(
            lambda: method(tuple(_Expression(value, "real") for value in values))
        )
        return result.code

    def _load_modulation(self, hold: str, switching: bool) -> list[str]:
        """Map active compares to the topology's one held bridge input."""
        bridge = self.bridge
        if switching:
            modulator = bridge.modulator
            if isinstance(modulator, SynchronousCarrier):
                frequency = (
                    f"({_number(modulator.pulse_ratio)} * {self.m('omega')} / (2.0 * pi))"
                )
                phase = (
                    f"({_number(modulator.pulse_ratio)} * {self.m('theta')} / (2.0 * pi) + "
                    f"{_number(modulator.phase)} - t * {frequency})"
                )
            elif isinstance(modulator, CarrierComparison):
                frequency = _number(modulator.f_sw)
                phase = _number(modulator.phase)
            else:
                raise TypeError(
                    f"C++ export: unsupported PWM modulator {type(modulator).__name__}"
                )
            mapped = (
                f"make_switches_{self.tag}(t, {_number(bridge.load_period)}, "
                f"{frequency}, {phase}, {hold});"
            )
        else:
            values = tuple(f"{self.m('active')}[{index}]" for index in range(self.width))
            mapped = f"{hold} = {self._topology_expression(self.topology.pwm_state, values)};"
        return [f"        {mapped}",
                f"        {self.m('load_end')} = t + {_number(bridge.load_period)};"]

    def switching_method(self) -> str:
        """Generate one topology-driven carrier-comparison implementation."""
        if isinstance(self.bridge.modulator, ZOH):
            return ""
        hold = self._hold()
        values = tuple(f"q[{index}]" for index in range(self.width))
        encoded = self._topology_expression(self.topology.switching_state, values)
        q_kind = "Complex" if isinstance(self.topology.zero, complex) else "double"
        return f'''void make_switches_{self.tag}(double t, double span, double frequency,
        double phase, {q_kind}& held) {{
    if (!(frequency > 0.0)) throw std::runtime_error("PWM frequency is not positive");
    {self.m('switches')}.clear();
    {self.m('switch_index')} = 0;
    std::array<double, {self.width}> modulation{{}};
    std::array<double, {self.width}> q{{}};
    for (int k = 0; k < {self.width}; ++k)
        modulation[k] = std::clamp(2.0 * {self.m('active')}[k] - 1.0, -1.0, 1.0);
    double position = carrier_position(t, frequency, phase);
    const double carrier0 = carrier_value(position);
    for (int k = 0; k < {self.width}; ++k)
        q[k] = position < 0.5 ? (modulation[k] > carrier0) : (modulation[k] >= carrier0);
    held = {encoded};
    std::vector<std::tuple<double, int, double>> edges;
    double covered = 0.0;
    for (int piece = 0; piece < 64 && covered < span - 1e-13; ++piece) {{
        position = carrier_position(t + covered, frequency, phase);
        const bool rising = position < 0.5;
        const double length = std::min(((rising ? 0.5 : 1.0) - position) / frequency,
                                       span - covered);
        const double c0 = carrier_value(position);
        const double slope = rising ? 4.0 * frequency : -4.0 * frequency;
        for (int k = 0; k < {self.width}; ++k) {{
            const double x = (modulation[k] - c0) / slope;
            if (x > 0.0 && x < length)
                edges.emplace_back(covered + x, k, rising ? 0.0 : 1.0);
        }}
        covered += length;
    }}
    std::sort(edges.begin(), edges.end());
    double previous = 0.0;
    for (const auto& edge : edges) {{
        const double at = std::get<0>(edge);
        q[std::get<1>(edge)] = std::get<2>(edge);
        if (at - previous > 1e-12) {{
            {self.m('switches')}.emplace_back(t + at, {encoded});
            previous = at;
        }} else if (!{self.m('switches')}.empty()) {{
            {self.m('switches')}.back().second = {encoded};
        }}
    }}
    {self.m('next_switch')} = {self.m('switches')}.empty()
        ? std::numeric_limits<double>::infinity() : {self.m('switches')}[0].first;
}}
'''

    def _pwm_members(self) -> list[str]:
        """Declare the shared sampled bridge timer, register and edge state."""
        bridge = self.bridge
        q_kind = "Complex" if isinstance(self.topology.zero, complex) else "double"
        schedule = ", ".join(
            "{" + _number(at) + ", " + self._literal(q) + "}" for at, q in bridge.schedule
        )
        return [
            f"long long {self.m('pwm_k')} = {int(bridge.k)};",
            f"long long {self.m('pwm_j')} = {int(bridge.j)};",
            f"double {self.m('t_interrupt')} = {_number(bridge.t_interrupt)};",
            f"double {self.m('t_load')} = {_number(bridge.t_load)};",
            f"std::array<double, {self.width + 1}> {self.m('active')}"
            f"{{{', '.join(_number(x) for x in bridge.active)}}};",
            f"std::array<double, {self.width + 1}> {self.m('shadow')}"
            f"{{{', '.join(_number(x) for x in bridge.shadow)}}};",
            f"std::vector<std::pair<double, {q_kind}>> {self.m('switches')}"
            f"{{{schedule}}};",
            f"std::size_t {self.m('switch_index')} = 0;",
            f"double {self.m('next_switch')} = {_number(bridge.next_switch)};",
            f"double {self.m('load_end')} = {_number(bridge.load_end)};",
        ]

    def _pwm_write_lines(self, on: str) -> list[str]:
        bridge = self.bridge
        return [
            f"        for (int k = 0; k < {self.width}; ++k) "
            f"{self.m('shadow')}[k] = {self.m('pending_d')}[k];",
            f"        {self.m('shadow')}[{self.width}] = {on};",
            f"        ++{self.m('pwm_k')}; {self.m('t_interrupt')} = "
            f"{_number(bridge.offset)} + {self.m('pwm_k')} * {_number(bridge.period)};",
        ]

    def _pwm_load_lines(self, hold: str, switching: bool, gates: str,
                        tail: tuple[str, ...] = ()) -> list[str]:
        bridge = self.bridge
        return [
            "    if (load_due) {",
            f"        const double completed = {_number(bridge.offset)} + "
            f"({self.m('pwm_k')} - 1) * {_number(bridge.period)} + "
            f"{_number(bridge.computation)};",
            f"        if (completed <= t + time_eps) {self.m('active')} = {self.m('shadow')};",
            *self._load_modulation(hold, switching),
            f"        ++{self.m('pwm_j')}; {self.m('t_load')} = {_number(bridge.offset)} + "
            f"{self.m('pwm_j')} * {_number(bridge.load_period)};",
            f"        {self.m('gates')} = {gates};",
            *tail,
            "    }",
        ]

    def _pwm_switch_lines(self, hold: str) -> list[str]:
        return [
            f"    while ({self.m('switch_index')} < {self.m('switches')}.size() && "
            f"std::abs({self.m('switches')}[{self.m('switch_index')}].first - t) < time_eps) {{",
            f"        {hold} = {self.m('switches')}[{self.m('switch_index')}++].second;",
            "    }",
            f"    {self.m('next_switch')} = {self.m('switch_index')} < "
            f"{self.m('switches')}.size() ? "
            f"{self.m('switches')}[{self.m('switch_index')}].first : "
            "std::numeric_limits<double>::infinity();",
        ]


class _ProtectionGenerator:
    """Shared allocation-light lowering of one unit's protection subsystem."""

    def protection_members(self) -> list[str]:
        protection = self.unit.protection
        stats = protection.stats
        period = protection.T
        window = max(1, int(self.cfg.protection.rocof.window / period + 0.5))

        def observed(value: Any) -> str:
            return _number(math.nan if value is None else value)

        return [
            f"bool {self.m('sampled_trip_due')} = false;",
            f"double {self.m('overcurrent_first_t')} = {observed(stats.overcurrent_first_t)};",
            f"double {self.m('undervoltage_first_t')} = {observed(stats.undervoltage_first_t)};",
            f"double {self.m('overvoltage_first_t')} = {observed(stats.overvoltage_first_t)};",
            f"double {self.m('frequency_first_t')} = {observed(stats.frequency_first_t)};",
            f"double {self.m('dc_voltage_first_t')} = {observed(stats.dc_voltage_first_t)};",
            f"double {self.m('rocof_first_t')} = {observed(stats.rocof_first_t)};",
            f"double {self.m('rocof_max')} = {_number(stats.rocof_max)};",
            f"double {self.m('vac_min')} = {observed(stats.vac_min_pu)};",
            f"double {self.m('vac_max')} = {observed(stats.vac_max_pu)};",
            f"std::array<double, {window}> {self.m('rocof_buffer')}{{}};",
            f"std::size_t {self.m('rocof_index')} = 0;",
            f"bool {self.m('rocof_full')} = false;",
            f"std::vector<std::string> {self.m('alarms')}"
            "{" + ", ".join(_cpp_string(value) for value in stats.alarms) + "};",
        ]

    def _trip_body(self) -> list[str]:
        branch = f"breaker_open_{self.model.prefix[id(self.unit.branch_f)]}"
        link = f"dclink_tripped_{self.model.prefix[id(self.unit.dclink)]}"
        return [f"{self.m('startup_run')} = false;",
                f"{self.m('gates')} = false;",
                f"{self.m('breaker')} = false;",
                f"{branch} = true;",
                f"{link} = true;"]

    def protection_methods(self) -> str:
        """Fast and sampled protection, plus the one physical trip action."""
        cfg = self.cfg.protection
        period = self.unit.protection.T
        window = max(1, int(cfg.rocof.window / period + 0.5))
        current = f"obs_{_identifier(self.name + '.i_c')}"

        def enabled(item: Any) -> str:
            return "true" if item.enable else "false"

        def limit(item: Any, field: str) -> str:
            value = getattr(item, field)
            return _number(0.0 if value is None else value)

        def first(name: str) -> str:
            return self.m(name + "_first_t")

        def timed(item: Any, condition: str, name: str, alarm: str, timer: str,
                  cause: str) -> list[str]:
            if not item.enable:
                return []
            return [
                f"if ({condition}) {{",
                f"    if (!std::isfinite({first(name)})) {first(name)} = t;",
                f"    alarm_{self.tag}({_cpp_string(alarm)});",
                f"    {self.m('prot_' + timer)} += {_number(period)};",
                "} else {",
                f"    {self.m('prot_' + timer)} = 0.0;",
                "}",
                f"if ({condition} && {self.m('prot_' + timer)} >= {_number(cfg.hold)}) {{",
                f"    latch_{self.tag}(t, {_cpp_string(cause)});",
                f"    {self.m('sampled_trip_due')} = true;",
                "    return true;",
                "}",
            ]

        sampled: list[str] = [
            f"bool sampled_protect_{self.tag}(double t, double vac, double freq_dev, double vdc_error, bool startup_complete) {{",
            f"    if (!{self.m('tripped')}) {{",
            f"        const bool had_old = {self.m('rocof_full')};",
            f"        const double old = {self.m('rocof_buffer')}[{self.m('rocof_index')}];",
            f"        {self.m('rocof_buffer')}[{self.m('rocof_index')}] = freq_dev;",
            f"        {self.m('rocof_index')} = ({self.m('rocof_index')} + 1) % {window};",
            f"        if ({self.m('rocof_index')} == 0) {self.m('rocof_full')} = true;",
            "        if (had_old) {",
            f"            const double rocof = (freq_dev - old) / {_number(window * period)};",
            f"            {self.m('rocof_max')} = std::max({self.m('rocof_max')}, std::abs(rocof));",
            f"            if ({enabled(cfg.rocof)} && std::abs(rocof) >= {limit(cfg.rocof, 'limit')} && startup_complete) {{",
            f"                if (!std::isfinite({first('rocof')})) {first('rocof')} = t;",
            f"                alarm_{self.tag}(\"ROCOF\");",
            "            }",
            "        }",
            "    }",
            f"    if ({self.m('tripped')} || !startup_complete) return false;",
        ]
        sampled += ["    " + line for line in timed(
            cfg.dc_voltage,
            f"std::abs(vdc_error) >= {limit(cfg.dc_voltage, 'limit_pu')}",
            "dc_voltage", "VDC_BAND", "hold_vdc", "vdc")]
        sampled += ["    " + line for line in timed(
            cfg.frequency,
            f"std::abs(freq_dev) >= {limit(cfg.frequency, 'limit')}",
            "frequency", "FREQ_BAND", "hold_freq", "freq")]
        sampled += [
            f"    {self.m('vac_min')} = std::isfinite({self.m('vac_min')}) ? std::min({self.m('vac_min')}, vac) : vac;",
            f"    {self.m('vac_max')} = std::isfinite({self.m('vac_max')}) ? std::max({self.m('vac_max')}, vac) : vac;",
        ]
        sampled += ["    " + line for line in timed(
            cfg.undervoltage,
            f"vac <= {limit(cfg.undervoltage, 'limit_pu')}",
            "undervoltage", "VAC_UNDER", "hold_uv", "vac_uv")]
        sampled += ["    " + line for line in timed(
            cfg.overvoltage,
            f"vac >= {limit(cfg.overvoltage, 'limit_pu')}",
            "overvoltage", "VAC_OVER", "hold_ov", "vac_ov")]
        sampled += ["    return false;", "}"]

        apply = [f"void apply_trip_{self.tag}() {{", *["    " + line for line in self._trip_body()], "}"]
        fast = [f"bool protect_{self.tag}(double t) {{"]
        if not self.unit.continuous:
            fast += [
                f"    const bool edge = std::abs({self.m('next_switch')} - t) < time_eps || std::abs({self.m('load_end')} - t) < time_eps;",
                "    if (!edge) return false;",
            ]
        fast += [
            f"    const auto current = phases({current});",
            f"    const double peak = std::max({{std::abs(current[0]), std::abs(current[1]), std::abs(current[2])}}) / {_number(self.cfg.base.i_phase_peak)};",
            f"    if (std::isfinite(peak)) {self.m('max_current')} = std::max({self.m('max_current')}, peak);",
            f"    if ({self.m('tripped')} || !{enabled(cfg.overcurrent)} || !(peak > {limit(cfg.overcurrent, 'limit_pu')})) return false;",
            f"    if (!std::isfinite({first('overcurrent')})) {first('overcurrent')} = t;",
            f"    alarm_{self.tag}(\"OVERCURRENT\");",
            f"    latch_{self.tag}(t, \"overcurrent\", true);",
            f"    apply_trip_{self.tag}();",
            "    return true;",
            "}",
        ]
        helpers = [
            f"void alarm_{self.tag}(std::string_view value) {{",
            f"    if (std::find({self.m('alarms')}.begin(), {self.m('alarms')}.end(), value) == {self.m('alarms')}.end())",
            f"        {self.m('alarms')}.emplace_back(value);",
            "}",
            f"bool latch_{self.tag}(double t, std::string_view cause, bool fault = false) {{",
            f"    if (fault) {self.m('fault')} = true;",
            f"    if ({self.m('tripped')}) return false;",
            f"    {self.m('tripped')} = true;",
            f"    {self.m('trip_time')} = t;",
            f"    {self.m('trip_cause')} = cause;",
            f"    alarm_{self.tag}(std::string(\"TRIP_\") + std::string(cause == \"vac_uv\" ? \"VAC_UV\" : cause == \"vac_ov\" ? \"VAC_OV\" : cause == \"vdc\" ? \"VDC\" : cause == \"freq\" ? \"FREQ\" : \"OVERCURRENT\"));",
            "    return true;",
            "}",
        ]
        return "\n\n".join("\n".join(section) for section in (helpers, apply, sampled, fast))

    def protection_summary(self, stream: str = "summary") -> list[str]:
        """C++ statements which append this unit's protection result to a JSON object."""
        lines = [
            f'{stream} << ",\\n  \\"{self.name}.tripped\\": " << ({self.m("tripped")} ? 1 : 0);',
            f'{stream} << ",\\n  \\"{self.name}.trip_time\\": "; if (std::isfinite({self.m("trip_time")})) {stream} << {self.m("trip_time")}; else {stream} << "null";',
            f'{stream} << ",\\n  \\"{self.name}.trip_cause\\": "; if ({self.m("trip_cause")}.empty()) {stream} << "null"; else {stream} << \'"\' << {self.m("trip_cause")} << \'"\';',
            f'{stream} << ",\\n  \\"{self.name}.max_current_pu\\": " << {self.m("max_current")};',
            f'{stream} << ",\\n  \\"{self.name}.rocof_max\\": " << {self.m("rocof_max")};',
        ]
        for key in ("overcurrent", "undervoltage", "overvoltage", "frequency", "dc_voltage", "rocof"):
            value = self.m(key + "_first_t")
            lines.append(
                f'{stream} << ",\\n  \\"{self.name}.{key}_first_t\\": "; '
                f'if (std::isfinite({value})) {stream} << {value}; else {stream} << "null";'
            )
        for key, value in (("vac_min_pu", self.m("vac_min")), ("vac_max_pu", self.m("vac_max"))):
            lines.append(
                f'{stream} << ",\\n  \\"{self.name}.{key}\\": "; '
                f'if (std::isfinite({value})) {stream} << {value}; else {stream} << "null";'
            )
        lines += [
            f'{stream} << ",\\n  \\"{self.name}.alarms\\": [";',
            f'for (std::size_t i = 0; i < {self.m("alarms")}.size(); ++i) {{ if (i) {stream} << ", "; {stream} << \'"\' << {self.m("alarms")}[i] << \'"\'; }}',
            f'{stream} << "]";',
        ]
        return lines


class _ACControlGenerator(_ProtectionGenerator, _ControlGenerator):
    """Summary contract shared by the optimized sampled and continuous AC paths."""

    def __init__(self, name: str, unit: Any, model: _ModelGenerator, prefix: str) -> None:
        _ControlGenerator.__init__(self, name, unit, model, prefix)

    def _mode_log_lines(self, indent: str) -> list[str]:
        """Generate the control-family-specific suffix of the AC controller log."""
        sources = self.ctrl.mode_log_sources()
        if self.ctrl._is_gfl:
            idref = self.source(sources["id_ref_pu"]) if "id_ref_pu" in sources else "0.0"
            return [f"{indent}{self.m('log_id_ref_pu')} = {idref};"]
        return [
            f"{indent}{self.m('log_p_pu')} = "
            f"{self.source(sources['p_pu']) if 'p_pu' in sources else '0.0'};",
            f"{indent}{self.m('log_q_pu')} = "
            f"{self.source(sources['q_pu']) if 'q_pu' in sources else '0.0'};",
            f"{indent}{self.m('log_p_ref_pu')} = {self.m('startup_value')} * "
            f"{self.source(sources['p_ref_pu'])};",
            f"{indent}{self.m('log_v_ref_pu')} = "
            f"{self.source(sources['v_ref_pu'])};",
        ]

    def protection_method(self) -> str:
        return self.protection_methods()

    def controller_summary(self, stream: str = "summary") -> list[str]:
        name = self.name
        if self.unit.continuous:
            lines = [
                f'{stream} << ",\\n  \\"{name}.modulation_saturation_fraction\\": null,'
                f'\\n  \\"{name}.modulation_saturation_first_t\\": null";'
            ]
        else:
            lines = [
                f'{stream} << ",\\n  \\"{name}.modulation_saturation_fraction\\": " << '
                f'(static_cast<double>({self.m("mod_saturated")}) / '
                f'std::max<std::size_t>(1, {self.m("mod_updates")}));',
                f'{stream} << ",\\n  \\"{name}.modulation_saturation_first_t\\": "; '
                f'if (std::isfinite({self.m("mod_first_t")})) '
                f'{stream} << {self.m("mod_first_t")}; else {stream} << "null";',
            ]
        loop_name = self.ctrl._dc
        if loop_name is not None:
            if self.unit.continuous:
                checks = self.m(loop_name + "_limit_checks")
                exceeded = self.m(loop_name + "_limit_exceeded")
                first = self.m(loop_name + "_limit_first_t")
            else:
                checks = self.node(loop_name, "n_updates")
                exceeded = self.node(loop_name, "n_limit_exceeded")
                first = self.node(loop_name, "first_limit_t")
            lines += [
                f'{stream} << ",\\n  \\"{name}.id_ref_limit_fraction\\": " << '
                f'(static_cast<double>({exceeded}) / std::max<std::size_t>(1, {checks}));',
                f'{stream} << ",\\n  \\"{name}.id_ref_limit_first_t\\": "; '
                f'if (std::isfinite({first})) {stream} << {first}; else {stream} << "null";',
            ]
        sync_node = self.graph.nodes.get(self.ctrl._sync)
        if sync_node is not None:
            lines.append(
                f'{stream} << ",\\n  \\"{name}.law\\": \\"{sync_node.type}\\"";'
            )
        return lines


class _ContinuousUnitGenerator(_ACControlGenerator):
    """Lower an ideal averaged controller into the plant RHS without a sampled/PWM path."""

    def __init__(self, name: str, unit: Any, model: _ModelGenerator) -> None:
        super().__init__(name, unit, model, "c")
        self.continuous = True

    def state(self, loop: str, local: str) -> str:
        return self.model.value(self.ctrl.state, f"{loop}.{local}")

    def source(self, path: str) -> str:
        if path == "meas.u_g":
            return self.m("meas_u_g")
        if path == "meas.i_c":
            return self.m("meas_i_c")
        if path == "meas.u_dc":
            return self.m("meas_u_dc")
        if path.startswith("references."):
            return self._reference(path[11:])
        return self.g(path)

    def declarations(self) -> list[str]:
        lines = [
            f"Complex {self.m('meas_u_g')}{{}};",
            f"Complex {self.m('meas_i_c')}{{}};",
            f"double {self.m('meas_u_dc')}{{}};",
            f"double {self.m('startup_value')}{{}};",
        ]
        for key, value in self.graph.values.items():
            kind = "Complex" if isinstance(value, complex) else "double"
            lines.append(f"{kind} {self.g(key)} = {self._literal(value)};")
        for loop_name, node in self.graph.nodes.items():
            for local, value in node.flow_state().items():
                kind = "Complex" if isinstance(value, complex) else "double"
                lines.append(f"{kind} {self.d(loop_name, local)}{{}};")
        return lines

    def members(self) -> list[str]:
        startup = self.ctrl.startup
        lines = [
            f"bool {self.m('tripped')} = {'true' if self.unit.tripped else 'false'};",
            f"bool {self.m('fault')} = {'true' if self.unit.protection.fault else 'false'};",
            f"double {self.m('trip_time')} = std::numeric_limits<double>::quiet_NaN();",
            f"std::string {self.m('trip_cause')};",
            f"double {self.m('max_current')} = 0.0;",
            *[f"double {self.m('prot_' + key)} = {_number(value)};"
              for key, value in self.unit.protection.get_state().items()],
            *self.protection_members(),
            f"bool {self.m('breaker')} = {'true' if self.unit.breaker else 'false'};",
            f"bool {self.m('gates')} = {'true' if self.unit.gates else 'false'};",
            f"bool {self.m('startup_run')} = {'true' if startup.run else 'false'};",
            f"double {self.m('startup_ramp')} = {_number(startup.ramp)};",
            f"double {self.m('startup_start')} = {_number(startup.start)};",
            f"bool {self.m('track_due')} = false;",
            f"double {self.m('theta')} = {_number(self.ctrl.theta)};",
            f"double {self.m('omega')} = {_number(self.ctrl.omega)};",
            f"Complex {self.m('u_cmd')} = {_complex(self.ctrl.u_cmd)};",
            *[f"double {self.m('log_' + key)} = std::numeric_limits<double>::quiet_NaN();"
              for key in self.ctrl.log_names()],
        ]
        for loop_name, node in self.graph.nodes.items():
            if isinstance(node, SyncLaw):
                lines.append(f"double {self.m(loop_name + '_v_mag')} = {_number(node.v_mag)};")
            elif isinstance(node, DCVoltageControl):
                lines += [
                    f"std::size_t {self.m(loop_name + '_limit_checks')} = {int(node.n_updates)};",
                    f"std::size_t {self.m(loop_name + '_limit_exceeded')} = {int(node.n_limit_exceeded)};",
                    f"double {self.m(loop_name + '_limit_first_t')} = "
                    f"{_number(node.first_limit_t if node.first_limit_t is not None else math.nan)};",
                    f"bool {self.m(loop_name + '_limit_active')} = false;",
                ]
        return lines

    def limit_observation(self) -> list[str]:
        """Observe ideal-controller limits once per output snapshot, outside the RHS."""
        lines: list[str] = []
        for loop_name, node in self.graph.nodes.items():
            if not isinstance(node, DCVoltageControl):
                continue
            checks = self.m(loop_name + "_limit_checks")
            exceeded = self.m(loop_name + "_limit_exceeded")
            first = self.m(loop_name + "_limit_first_t")
            active = self.m(loop_name + "_limit_active")
            lines += [
                f"++{checks};",
                f"if ({active}) {{",
                f"    ++{exceeded};",
                f"    if (!std::isfinite({first})) {first} = t;",
                f"    alarm_{self.tag}(\"ID_REF_LIMIT\");",
                "}",
            ]
        return lines

    def state_expression(self, local: str) -> str:
        if local == "tripped":
            return self.m("tripped")
        if local == "fault":
            return self.m("fault")
        if local.startswith("prot."):
            return self.m("prot_" + local[5:])
        if local == "startup.run":
            return self.m("startup_run")
        if local == "startup.ramp":
            return self.m("startup_ramp")
        if local == "startup.start":
            return self.m("startup_start")
        raise KeyError(local)

    @staticmethod
    def _rotate(value: str, frame: str) -> str:
        return f"({value} * Complex{{std::cos({frame}), -std::sin({frame})}})"

    def output_lines(self) -> list[str]:
        lines: list[str] = [
            f"{self.m('meas_u_g')} = {self.model.value(self.ctrl.inp, 'u_g')} / {_number(self.cfg.base.v_phase_peak)};",
            f"{self.m('meas_i_c')} = {self.model.value(self.ctrl.inp, 'i_c')} / {_number(self.cfg.base.i_phase_peak)};",
            f"{self.m('meas_u_dc')} = {self.model.value(self.ctrl.inp, 'u_dc')} / {_number(self.cfg.dc_base.v)};",
            f"{self.m('startup_value')} = !{self.m('startup_run')} ? 0.0 : ({self.m('startup_ramp')} <= 0.0 ? 1.0 : smoothstep((t - {self.m('startup_start')}) / {self.m('startup_ramp')}));",
        ]
        for group in self.graph._continuous_plan:
            cyclic = group in self.graph._continuous_cycles
            if cyclic:
                lines.append("for (int control_sweep = 0; control_sweep < 16; ++control_sweep) {")
            for loop_name in group:
                emitted = self._loop_lines(loop_name, self.graph.nodes[loop_name])
                lines += (["    " + line for line in emitted] if cyclic else emitted)
            if cyclic:
                # The outer Model algebraic solver also checks its electrical tear.  Fixed built-in
                # controller SCCs are rare; sixteen allocation-free Gauss--Seidel sweeps match the
                # configured Python bound without constructing a dynamic convergence vector.
                lines.append("}")
        theta = self.source(self.graph.outputs["theta"])
        omega = self.source(self.graph.outputs["omega"])
        command = self.source(self.graph.outputs["u_dq"])
        q = self.model.value(self.ctrl.out, "q")
        lines += [
            f"{self.m('theta')} = {theta};",
            f"{self.m('omega')} = {omega};",
            f"{self.m('u_cmd')} = {command};",
            f"if ({self.m('startup_run')} && {self.model.value(self.ctrl.inp, 'u_dc')} != 0.0) {{",
            f"    {q} = {command} * Complex{{std::cos({theta}), std::sin({theta})}} * "
            f"{_number(self.cfg.base.v_phase_peak)} / {self.model.value(self.ctrl.inp, 'u_dc')};",
            "} else {",
            f"    {q} = Complex{{0.0, 0.0}};",
            "}",
        ]
        lines += self._log_lines()
        return lines

    def _log_lines(self) -> list[str]:
        current_name = self.ctrl._frame_cc
        frame = (self.input(current_name, "frame") if current_name is not None
                 else self.m("theta"))
        current = f"({self.m('meas_i_c')} * Complex{{std::cos({frame}), -std::sin({frame})}})"
        voltage = f"({self.m('meas_u_g')} * Complex{{std::cos({frame}), -std::sin({frame})}})"
        log_i, log_v = self.m("log_i_value"), self.m("log_v_value")
        lines = [f"const Complex {log_i} = {current};", f"const Complex {log_v} = {voltage};",
                 f"{self.m('log_id_pu')} = std::real({log_i});", f"{self.m('log_iq_pu')} = std::imag({log_i});",
                 f"{self.m('log_vd_pu')} = std::real({log_v});", f"{self.m('log_vq_pu')} = std::imag({log_v});",
                 f"{self.m('log_vac_pu')} = std::abs({log_v});", f"{self.m('log_vdc_pu')} = {self.m('meas_u_dc')};",
                 f"{self.m('log_in_service')} = {self.m('startup_run')} ? 1.0 : 0.0;",
                 f"{self.m('log_freq_dev')} = ({self.m('omega')} - {_number(self.cfg.base.w0)}) / (2.0 * pi);",
                 f"{self.m('log_angle_rel')} = std::remainder({self.m('theta')} - {_number(self.cfg.base.w0)} * t, 2.0 * pi);"]
        lines += self._mode_log_lines("")
        return lines

    def command_lines(self, on: bool, ramp: float) -> list[str]:
        effective = self._effective_command(on)
        lines = [
            (f"{self.m('track_due')} = {effective} && "
             f"!({self.m('breaker')} && {self.m('gates')} && !{self.m('tripped')});")
            if on else f"{self.m('track_due')} = false;"
        ]
        lines += self._controller_command_lines(on, ramp)
        lines += [
            f"{self.m('breaker')} = {effective};",
            f"{self.m('gates')} = {effective};",
            f"breaker_open_{self.model.prefix[id(self.unit.branch_f)]} = "
            f"!({self.m('breaker')} && {self.m('gates')} && !{self.m('tripped')});",
        ]
        return lines

    def sense_method(self) -> str:
        observed = f"obs_{_identifier(self.name + '.u_g')}"
        dc = f"obs_{_identifier(self.name + '.u_dc')}"
        lines = [f"void sense_{self.tag}(double t) {{",
                 f"    if ({self.m('track_due')}) {{"]
        for loop_name, node in self.graph.nodes.items():
            if not isinstance(node, SyncLaw):
                continue
            index, _ = self.model.state_index[(id(self.ctrl.state), f"{loop_name}.theta")]
            magnitude = f"std::abs({observed}) / {_number(self.cfg.base.v_phase_peak)}"
            lines += [f"        if (std::abs({observed}) > 0.1) {{",
                      f"            y[{index}] += std::remainder(std::arg({observed}) - y[{index}], 2.0 * pi);",
                      f"            {self.m(loop_name + '_v_mag')} = {magnitude};"]
            if isinstance(node, (DVOC, VSG)) and (
                    not isinstance(node, VSG) or node.t_q > 0.0):
                v_index, _ = self.model.state_index[
                    (id(self.ctrl.state), f"{loop_name}.v_mag_pu")
                ]
                lines.append(f"            y[{v_index}] = {magnitude};")
            if isinstance(node, VSG):
                dw_index, _ = self.model.state_index[
                    (id(self.ctrl.state), f"{loop_name}.dw_pu")
                ]
                lines.append(f"            y[{dw_index}] = 0.0;")
            lines.append("        }")
        lines += [f"        {self.m('track_due')} = false;", "    }",
                  f"    sampled_protect_{self.tag}(t, std::abs({observed}) / {_number(self.cfg.base.v_phase_peak)},",
                  f"        ({self.m('omega')} - {_number(self.cfg.base.w0)}) / (2.0 * pi),",
                  f"        {dc} / {_number(self.cfg.dc_base.v)} - {self._reference('vdc_ref_pu')},",
                  f"        {self.m('startup_run')} && ({self.m('startup_ramp')} <= 0.0 || "
                  f"t - {self.m('startup_start')} >= {self.m('startup_ramp')} - time_eps));",
                  "}"]
        return "\n".join(lines)

    def actuate_method(self) -> str:
        return f'''bool actuate_{self.tag}(double) {{
    if (!{self.m('sampled_trip_due')}) return false;
    {self.m('sampled_trip_due')} = false;
    apply_trip_{self.tag}();
    return true;
}}'''

    def protection_method(self) -> str:
        return self.protection_methods()

class _UnitGenerator(_ACControlGenerator):
    """Lower one sampled controller/PWM unit to allocation-free specialised methods."""

    def __init__(self, name: str, unit: Any, model: _ModelGenerator) -> None:
        super().__init__(name, unit, model, "u")
        self.continuous = False
        self.bridge = unit.bridge
        self.topology = self.bridge.topology
        self.width = len(self.topology.duty_names)

    def _hold(self) -> str:
        return self.model.zoh_fields[id(self.bridge.inp), "q"]

    def source(self, path: str) -> str:
        if path == "meas.u_g":
            return "meas_u_g"
        if path == "meas.i_c":
            return "meas_i_c"
        if path == "meas.u_dc":
            return "meas_u_dc"
        if path.startswith("references."):
            return self._reference(path[11:])
        return self.g(path)

    def _adc_raw(self, channel: str) -> str:
        return f"obs_{_identifier(self.name + '.' + channel)}"

    def _adc_value(self, channel: str) -> str:
        adc = self.unit.adc
        assert adc is not None
        if channel in adc.channels:
            return f"({self.m('adc_acc_' + channel)} / {_number(adc.length)})"
        return self._adc_raw(channel)

    def _adc_members(self) -> list[str]:
        adc = self.unit.adc
        if adc is None or (not adc.averaging and adc.samples == 1):
            return []
        lines = [f"long long {self.m('adc_n_samp')} = {int(adc.n_samp)};",
                 f"bool {self.m('adc_window_open')} = {'true' if adc.window_open else 'false'};"]
        for channel in adc.channels:
            value = adc.accumulated[channel]
            kind = "Complex" if isinstance(value, complex) else "double"
            lines += [f"{kind} {self.m('adc_acc_' + channel)} = {self._literal(value)};",
                      f"{kind} {self.m('adc_last_' + channel)} = {self._literal(adc._last[channel])};"]
        return lines

    def adc_methods(self) -> str:
        adc = self.unit.adc
        if adc is None or (not adc.averaging and adc.samples == 1):
            return ""
        methods: list[str] = []
        if adc.averaging:
            accumulate = [f"void adc_accumulate_{self.tag}(double dt) {{", "    if (dt <= 0.0) return;"]
            seed = [f"void adc_seed_{self.tag}() {{"]
            for channel in adc.channels:
                raw = self._adc_raw(channel)
                accumulate += [
                    f"    const auto now_{channel} = {raw};",
                    f"    if ({self.m('adc_window_open')}) {self.m('adc_acc_' + channel)} += 0.5 * dt * ({self.m('adc_last_' + channel)} + now_{channel});",
                    f"    {self.m('adc_last_' + channel)} = now_{channel};",
                ]
                seed.append(f"    {self.m('adc_last_' + channel)} = {raw};")
            accumulate.append("}")
            seed.append("}")
            methods += ["\n".join(accumulate), "\n".join(seed)]
        return "\n\n".join(methods)

    def adc_accumulate_call(self, duration: str) -> str:
        adc = self.unit.adc
        return (f"adc_accumulate_{self.tag}({duration});"
                if adc is not None and adc.averaging else "")

    def adc_seed_call(self) -> str:
        adc = self.unit.adc
        return (f"adc_seed_{self.tag}();" if adc is not None and adc.averaging else "")

    def members(self) -> list[str]:
        unit, ctrl, graph = self.unit, self.ctrl, self.graph
        bridge = unit.bridge
        lines = [
            f"bool {self.m('tripped')} = {'true' if unit.tripped else 'false'};",
            f"bool {self.m('fault')} = {'true' if unit.protection.fault else 'false'};",
            f"double {self.m('trip_time')} = std::numeric_limits<double>::quiet_NaN();",
            f"std::string {self.m('trip_cause')};",
            f"double {self.m('max_current')} = 0.0;",
            *[f"double {self.m('prot_' + key)} = {_number(value)};"
              for key, value in unit.protection.get_state().items()],
            *self.protection_members(),
            f"bool {self.m('breaker')} = {'true' if unit.breaker else 'false'};",
            f"bool {self.m('gates')} = {'true' if unit.gates else 'false'};",
            f"bool {self.m('startup_run')} = {'true' if ctrl.startup.run else 'false'};",
            f"double {self.m('startup_ramp')} = {_number(ctrl.startup.ramp)};",
            f"long long {self.m('startup_steps')} = {int(ctrl.startup.steps)};",
            f"double {self.m('startup_value')} = {_number(ctrl.startup.value)};",
            f"bool {self.m('startup_complete')} = {'true' if ctrl.startup.complete else 'false'};",
            f"double {self.m('theta')} = {_number(ctrl.theta)};",
            f"double {self.m('omega')} = {_number(ctrl.omega)};",
            f"Complex {self.m('u_cmd')} = {_complex(ctrl.u_cmd)};",
            f"double {self.m('command_theta')} = {_number(ctrl.command_theta)};",
            f"std::array<double, 3> {self.m('pending_d')}{{{', '.join(_number(x) for x in bridge.shadow[:3])}}};",
            f"bool {self.m('pending_on')} = false;",
            f"bool {self.m('pending_valid')} = false;",
            f"double {self.m('last_ctrl_t')} = -std::numeric_limits<double>::infinity();",
            f"double {self.m('last_m_max')} = 0.0;",
            f"bool {self.m('command_saturated')} = false;",
            f"std::size_t {self.m('mod_updates')} = {int(getattr(ctrl.stage, 'n_updates', 0))};",
            f"std::size_t {self.m('mod_saturated')} = {int(getattr(ctrl.stage, 'n_saturated', 0))};",
            f"double {self.m('mod_first_t')} = {_number(getattr(ctrl.stage, 'first_saturation_t', math.nan) or math.nan)};",
            *[f"double {self.m('log_' + key)} = std::numeric_limits<double>::quiet_NaN();"
              for key in ctrl.log_names()],
            *self._adc_members(),
        ]
        for key, value in graph.values.items():
            lines.append(f"{'Complex' if isinstance(value, complex) else 'double'} {self.g(key)} = {self._literal(value)};")
        lines += self._sampled_loop_members()
        if not unit.continuous:
            lines += self._pwm_members()
        return lines

    def state_expression(self, part: Any, local: str) -> str:
        if part is self.unit:
            if local == "tripped":
                return self.m("tripped")
            if local == "fault":
                return self.m("fault")
            if local.startswith("prot."):
                return self.m("prot_" + local[5:])
        if part is self.unit.bridge:
            bank, _, field = local.rpartition(".")
            word = self.m("shadow" if bank == "shadow" else "active")
            index = (*self.topology.duty_names, "on").index(field)
            return f"{word}[{index}]"
        if part is self.unit.adc and local.startswith("x_"):
            return self.m("adc_acc_" + local[2:])
        if part is self.ctrl:
            if local.startswith("held."):
                public = local[5:]
                for key in self.graph.values:
                    if self.graph._state_port(key) == public:
                        return self.g(key)
            if local == "command.u_dq_pu":
                return self.m("u_cmd")
            if local == "command.theta":
                return self.m("command_theta")
            if local == "command.omega":
                return self.m("omega")
            if local == "startup.run":
                return self.m("startup_run")
            if local == "startup.ramp":
                return self.m("startup_ramp")
            if local == "startup.steps":
                return self.m("startup_steps")
            loop_name, _, public = local.partition(".")
            node = self.graph.nodes.get(loop_name)
            if node is not None and public in node.get_state():
                return self.node(loop_name, public)
        raise KeyError(local)

    def controller_method(self) -> str:
        lines = [f"void controller_{self.tag}(double t) {{",
                 f"    const Complex meas_u_g = {self._adc_value('u_g')} / {_number(self.cfg.base.v_phase_peak)};",
                 f"    const Complex meas_i_c = {self._adc_value('i_c')} / {_number(self.cfg.base.i_phase_peak)};",
                 f"    const double meas_u_dc = {self._adc_value('u_dc')} / {_number(self.cfg.dc_base.v)};",
                 f"    if ({self.m('startup_run')} && !{self.m('startup_complete')}) {{",
                 f"        const double progress = {self.m('startup_ramp')} <= 0.0 ? 1.0 : std::min(1.0, {self.m('startup_steps')} * {_number(self.cfg.ctrl.period)} / {self.m('startup_ramp')});",
                 f"        {self.m('startup_value')} = smoothstep(progress);",
                 f"        {self.m('startup_complete')} = progress >= 1.0;",
                 f"        if (!{self.m('startup_complete')}) ++{self.m('startup_steps')};",
                 "    }"]
        for loop_name in self.graph.order:
            lines.extend("    " + line for line in
                         self._loop_lines(loop_name, self.graph.nodes[loop_name]))
        theta_source = self.source(self.graph.outputs["theta"])
        omega_source = self.source(self.graph.outputs["omega"])
        command_source = self.source(self.graph.outputs["u_dq"])
        lines += [f"    {self.m('theta')} = {theta_source};",
                  f"    {self.m('omega')} = {omega_source};",
                  f"    {self.m('u_cmd')} = {command_source};",
                  f"    {self.m('command_theta')} = {self.m('theta')};"]
        lines += self._modulation_lines(indent="    ")
        lines += self._log_lines("    ")
        lines += [f"    sampled_protect_{self.tag}(t, std::abs(meas_u_g),",
                  f"        ({self.m('omega')} - {_number(self.cfg.base.w0)}) / (2.0 * pi),",
                  f"        meas_u_dc - {self._reference('vdc_ref_pu')}, {self.m('startup_complete')});",
                  f"    {self.m('pending_on')} = {self.m('startup_run')};",
                  f"    {self.m('pending_valid')} = true;",
                  f"    {self.m('last_ctrl_t')} = t;",
                  "}"]
        return "\n".join(lines)

    def _log_lines(self, indent: str) -> list[str]:
        current_name = self.ctrl._frame_cc
        frame = (self.input(current_name, "frame") if current_name is not None
                 else self.m("theta"))
        current = f"(meas_i_c * Complex{{std::cos({frame}), -std::sin({frame})}})"
        voltage = f"(meas_u_g * Complex{{std::cos({frame}), -std::sin({frame})}})"
        lines = [f"{indent}const Complex log_i = {current};",
                 f"{indent}const Complex log_v = {voltage};",
                 f"{indent}{self.m('log_id_pu')} = std::real(log_i);",
                 f"{indent}{self.m('log_iq_pu')} = std::imag(log_i);",
                 f"{indent}{self.m('log_vd_pu')} = std::real(log_v);",
                 f"{indent}{self.m('log_vq_pu')} = std::imag(log_v);",
                 f"{indent}{self.m('log_vac_pu')} = std::abs(log_v);",
                 f"{indent}{self.m('log_vdc_pu')} = meas_u_dc;",
                 f"{indent}{self.m('log_m_max')} = {self.m('last_m_max')};",
                 f"{indent}{self.m('log_in_service')} = {self.m('startup_run')} ? 1.0 : 0.0;",
                 f"{indent}{self.m('log_freq_dev')} = ({self.m('omega')} - {_number(self.cfg.base.w0)}) / (2.0 * pi);",
                 f"{indent}{self.m('log_angle_rel')} = std::remainder({self.m('theta')} - {_number(self.cfg.base.w0)} * t, 2.0 * pi);"]
        lines += self._mode_log_lines(indent)
        return lines

    def _modulation_lines(self, indent: str) -> list[str]:
        stage = self.ctrl.stage
        command = self.m("u_cmd")
        theta = self.m("command_theta")
        udc = f"obs_{_identifier(self.name + '.u_dc')}"
        if not stage._trusted_pwm or not stage._standard_limiter:
            raise TypeError(
                "C++ export: a custom VSC output stage has no equation tracer contract"
            )
        traced = self.model._trace(lambda: stage.equation(
            _Expression(command, "complex"),
            _Expression(f"Complex{{std::cos({theta}), std::sin({theta})}}", "complex"),
            _Expression(udc, "real"),
        ))
        modulation, output_saturated = traced
        lines = [
                  f"{indent}std::array<double, 3> modulation{{"
                  f"{', '.join(value.code for value in modulation)}}};",
                  f"{indent}const double peak_m = std::max({{std::abs(modulation[0]), std::abs(modulation[1]), std::abs(modulation[2])}});",
                  f"{indent}const bool output_saturated = {output_saturated.code};",
                  f"{indent}const bool saturated = {self.m('command_saturated')} || output_saturated;",
                  f"{indent}{self.m('last_m_max')} = peak_m;",
                  f"{indent}++{self.m('mod_updates')};",
                  f"{indent}if (saturated) {{ ++{self.m('mod_saturated')}; if (!std::isfinite({self.m('mod_first_t')})) {self.m('mod_first_t')} = t; }}",
                  f"{indent}for (int k = 0; k < 3; ++k) {self.m('pending_d')}[k] = 0.5 * (1.0 + modulation[k]);"]
        return lines

    def event_methods(self) -> str:
        bridge = self.unit.bridge
        hold = self.model.zoh_fields[id(bridge.inp), "q"]
        switching = not isinstance(bridge.modulator, ZOH)
        next_values = [self.m('t_interrupt'), self.m('t_load'), self.m('next_switch')]
        adc = self.unit.adc
        if adc is not None and adc.averaging and not adc._full:
            next_values.append(
                f"({self.m('adc_window_open')} ? std::numeric_limits<double>::infinity() : "
                f"{self.m('t_interrupt')} - {_number(adc.length)})"
            )
        if adc is not None and adc.samples > 1:
            next_values.append(
                f"({self.m('adc_n_samp')} < {adc.samples} ? {self.m('t_interrupt')} - {_number(bridge.period)} + "
                f"{self.m('adc_n_samp')} * {_number(adc.sample_period)} : std::numeric_limits<double>::infinity())"
            )
        next_expression = "std::min({" + ", ".join(next_values) + "})"
        lines = [
            f"double next_{self.tag}() const noexcept {{ return {next_expression}; }}",
            f"void sense_{self.tag}(double t) {{",
            f"    const bool control_due = std::abs(t - {self.m('t_interrupt')}) < time_eps;",
        ]
        if adc is not None and adc.samples > 1:
            lines += [
                f"    if (!control_due && {self.m('adc_n_samp')} < {adc.samples}) {{",
                f"        const double sample_at = {self.m('t_interrupt')} - {_number(bridge.period)} + {self.m('adc_n_samp')} * {_number(adc.sample_period)};",
                f"        if (std::abs(t - sample_at) < time_eps) ++{self.m('adc_n_samp')};",
                "    }",
            ]
        lines += [
            f"    if (control_due) controller_{self.tag}(t);",
            "}",
            f"bool actuate_{self.tag}(double t) {{",
            f"    const bool tripped_now = {self.m('sampled_trip_due')};",
            f"    if (tripped_now) {{ {self.m('sampled_trip_due')} = false; apply_trip_{self.tag}(); }}",
            f"    const bool control_due = std::abs(t - {self.m('t_interrupt')}) < time_eps;",
            f"    const bool load_due = std::abs(t - {self.m('t_load')}) < time_eps;",
        ]
        write = self._pwm_write_lines(
            f"{self.m('pending_on')} && !{self.m('tripped')}"
        )
        if bridge.computation == 0.0:
            lines += [f"    if (control_due && {self.m('pending_valid')}) {{",
                      *write, "    }"]
        lines += self._pwm_load_lines(
            hold,
            switching,
            f"{self.m('active')}[{self.width}] != 0.0 && !{self.m('tripped')}",
            (f"        breaker_open_{self.model.prefix[id(self.unit.branch_f)]} = "
             f"!({self.m('breaker')} && {self.m('gates')} && !{self.m('tripped')});",),
        )
        if bridge.computation > 0.0:
            lines += [f"    if (control_due && {self.m('pending_valid')}) {{",
                      *write, "    }"]
        if switching:
            lines += self._pwm_switch_lines(hold)
        if adc is not None:
            if adc.averaging:
                full = adc._full
                lines += [
                    "    if (control_due) {",
                    *[f"        {self.m('adc_acc_' + channel)} = {self._literal(adc._zero[channel])};"
                      for channel in adc.channels],
                    f"        {self.m('adc_window_open')} = {'true' if full else 'false'};",
                    f"        {self.m('adc_n_samp')} = 1;",
                    "    }",
                ]
                if not full:
                    lines += [
                        f"    const double window_at = {self.m('t_interrupt')} - {_number(adc.length)};",
                        f"    if (!{self.m('adc_window_open')} && std::abs(t - window_at) < time_eps) {{",
                        *[f"        {self.m('adc_acc_' + channel)} = {self._literal(adc._zero[channel])};"
                          for channel in adc.channels],
                        f"        {self.m('adc_window_open')} = true;",
                        "    }",
                    ]
            elif adc.samples > 1:
                lines += [f"    if (control_due) {self.m('adc_n_samp')} = 1;"]
        lines += [f"    if (control_due) {self.m('pending_valid')} = false;",
                  "    return tripped_now;", "}"]
        return "\n".join(lines)

    def command_lines(self, on: bool, ramp: float) -> list[str]:
        effective = self._effective_command(on)
        lines = [f"{self.m('breaker')} = {effective};"]
        lines += self._controller_command_lines(on, ramp)
        lines.append(
            f"breaker_open_{self.model.prefix[id(self.unit.branch_f)]} = "
            f"!({self.m('breaker')} && {self.m('gates')} && !{self.m('tripped')});"
        )
        return lines

class _GraphUnitGenerator(_ControlGenerator):
    """Lower a unit through its public control and bridge ports.

    This path has no knowledge of a converter family.  The control interface supplies measured
    and commanded ports, registered control loops supply their equations and state, and the
    bridge topology supplies its duty-register width and switching-state encoding.
    """

    def __init__(self, name: str, unit: Any, model: _ModelGenerator) -> None:
        super().__init__(name, unit, model, "u")
        self.bridge = unit.bridge
        self.topology = self.bridge.topology
        self.width = len(self.topology.duty_names)
        self.continuous = unit.continuous
        self.maxima = unit.control_maxima()
        self._control_inputs = model.unit_control_inputs(unit)
        graph_inputs = {path for path in self.graph.interface.inputs if path.startswith("meas.")}
        if set(self._control_inputs) != graph_inputs:
            raise TypeError(
                f"C++ export: {name}.control_inputs() exposes "
                f"{sorted(self._control_inputs)}, expected {sorted(graph_inputs)}"
            )

    @staticmethod
    def _measurement_name(path: str) -> str:
        return "meas_" + _identifier(path.partition(".")[2])

    def source(self, path: str) -> str:
        if path.startswith("meas."):
            name = self._measurement_name(path)
            return self.m(name) if self.continuous else name
        if path.startswith("references."):
            return self._reference(path[11:])
        if path in self.graph.interface.outputs:
            return self.source(self.graph.outputs[path])
        if path == "startup.active":
            return self.m("startup_run")
        return self.g(path)

    def declarations(self) -> list[str]:
        if not self.continuous:
            return []
        lines: list[str] = []
        for path, value in self._control_inputs.items():
            kind = "Complex" if value.kind == "complex" else "double"
            lines.append(f"{kind} {self.m(self._measurement_name(path))}{{}};")
        for key, value in self.graph.values.items():
            kind = "Complex" if isinstance(value, complex) else "double"
            lines.append(f"{kind} {self.g(key)} = {self._literal(value)};")
        for loop_name, node in self.graph.nodes.items():
            for local, value in node.flow_state().items():
                kind = "Complex" if isinstance(value, complex) else "double"
                lines.append(f"{kind} {self.d(loop_name, local)}{{}};")
        return lines

    def members(self) -> list[str]:
        startup = self.ctrl.startup
        trip_cause = self.unit.trip_cause
        lines = [
            f"bool {self.m('tripped')} = {'true' if self.unit.tripped else 'false'};",
            f"bool {self.m('breaker')} = {'true' if self.unit.breaker else 'false'};",
            f"bool {self.m('gates')} = {'true' if self.unit.gates else 'false'};",
            f"std::string {self.m('trip_cause')} = {_cpp_string(trip_cause or '')};",
            f"bool {self.m('startup_run')} = {'true' if startup.run else 'false'};",
            f"double {self.m('startup_ramp')} = {_number(startup.ramp)};",
            *[f"double {self.m('log_' + key)} = std::numeric_limits<double>::quiet_NaN();"
              for key in self.ctrl.log_names()],
        ]
        lines += [f"double {self.m('maximum_' + key)} = {_number(initial)};"
                  for key, (_port, _scale, initial) in self.maxima.items()]
        if self.continuous:
            lines += [
                f"double {self.m('startup_start')} = {_number(startup.start)};",
                f"double {self.m('startup_value')} = {_number(startup.value)};",
            ]
            lines += [f"bool {self.m(loop_name + '_limit_active')} = false;"
                      for loop_name, node in self.graph.nodes.items()
                      if isinstance(node, DCVoltageControl)]
            return lines
        lines += [
            f"long long {self.m('startup_steps')} = {int(startup.steps)};",
            f"double {self.m('startup_value')} = {_number(startup.value)};",
            f"bool {self.m('startup_complete')} = {'true' if startup.complete else 'false'};",
            f"std::array<double, {self.width}> {self.m('pending_d')}{{}};",
            f"bool {self.m('pending_on')} = false;",
            f"bool {self.m('pending_valid')} = false;",
        ]
        for key, value in self.graph.values.items():
            kind = "Complex" if isinstance(value, complex) else "double"
            lines.append(f"{kind} {self.g(key)} = {self._literal(value)};")
        lines += self._sampled_loop_members()
        lines += [
            *self._pwm_members(),
        ]
        return lines

    def _measurement_lines(self, *, members: bool) -> list[str]:
        lines = []
        for path, value in self._control_inputs.items():
            target = self.m(self._measurement_name(path)) if members else self._measurement_name(path)
            expression = (value.code if members else
                          f"obs_{_identifier(self.name + '.ctrl.' + path)}")
            declaration = "" if members else (
                "Complex " if value.kind == "complex" else "double "
            )
            lines.append(f"{declaration}{target} = {expression};")
        return lines

    def _log_lines(self) -> list[str]:
        sources = self.ctrl.log_sources()
        return [f"{self.m('log_' + key)} = {self.source(sources[key])};"
                for key in self.ctrl.log_names()]

    def output_lines(self) -> list[str]:
        lines = self._measurement_lines(members=True)
        lines += [
            f"{self.m('startup_value')} = !{self.m('startup_run')} ? 0.0 : "
            f"({self.m('startup_ramp')} <= 0.0 ? 1.0 : smoothstep("
            f"std::clamp((t - {self.m('startup_start')}) / {self.m('startup_ramp')}, 0.0, 1.0)));"
        ] if self.continuous else []
        for group in self.graph._continuous_plan:
            cyclic = group in self.graph._continuous_cycles
            if cyclic:
                lines.append("for (int control_sweep = 0; control_sweep < 16; ++control_sweep) {")
            for loop_name in group:
                emitted = self._loop_lines(loop_name, self.graph.nodes[loop_name])
                lines += (["    " + line for line in emitted] if cyclic else emitted)
            if cyclic:
                lines.append("}")
        outputs = tuple(self.graph.interface.outputs)
        if len(outputs) != 1:
            raise TypeError(
                f"C++ export: a port-driven bridge controller needs one output, got {outputs}"
            )
        command = _Expression(self.source(outputs[0]), "real")
        state = self.model._trace(lambda: self.topology.state_from_duty(command))
        q = self.model.value(self.ctrl.out, "q")
        lines.append(
            f"{q} = {self.m('startup_run')} ? {state.code} : {self._literal(self.topology.zero)};"
        )
        lines += self._log_lines()
        return lines

    def _command_duties(self) -> tuple[_Expression, ...]:
        output = SimpleNamespace(**{
            port: _Expression(self.source(port),
                              "complex" if spec.complex_value else "real")
            for port, spec in self.graph.interface.outputs.items()
        })
        return tuple(_Expression.of(value) for value in self.topology.command_duties(output))

    def controller_method(self) -> str:
        lines = [f"void controller_{self.tag}(double t) {{"]
        lines += ["    " + line for line in self._measurement_lines(members=False)]
        lines += [
            f"    if ({self.m('startup_run')} && !{self.m('startup_complete')}) {{",
            f"        const double progress = {self.m('startup_ramp')} <= 0.0 ? 1.0 : "
            f"std::min(1.0, {self.m('startup_steps')} * {_number(self.cfg.ctrl.period)} / "
            f"{self.m('startup_ramp')});",
            f"        {self.m('startup_value')} = smoothstep(progress);",
            f"        {self.m('startup_complete')} = progress >= 1.0;",
            f"        if (!{self.m('startup_complete')}) ++{self.m('startup_steps')};",
            "    }",
        ]
        for loop_name in self.graph.order:
            lines += ["    " + line for line in
                      self._loop_lines(loop_name, self.graph.nodes[loop_name])]
        for index, duty in enumerate(self._command_duties()):
            lines.append(f"    {self.m('pending_d')}[{index}] = {duty.code};")
        lines += ["    " + line for line in self._log_lines()]
        lines += [
            f"    {self.m('pending_on')} = {self.m('startup_run')};",
            f"    {self.m('pending_valid')} = true;",
            "}",
        ]
        return "\n".join(lines)

    def _hold(self) -> str:
        connections = self.unit.zoh_connections()
        labels = set(connections.values())
        target = next(
            ((subsystem, field) for subsystem, field in connections
             if subsystem is self.bridge and field == "q"),
            None,
        )
        if target is None or len(labels) != 1:
            raise TypeError(
                f"C++ export: sampled unit {self.name!r} needs one shared held bridge command"
            )
        subsystem, field = target
        record = subsystem.inp
        return self.model.zoh_fields[id(record), field]

    def event_methods(self) -> str:
        hold = self._hold()
        bridge = self.bridge
        switching = not isinstance(bridge.modulator, ZOH)
        lines = [
            f"double next_{self.tag}() const noexcept {{ return std::min({{{self.m('t_interrupt')}, "
            f"{self.m('t_load')}, {self.m('next_switch')}}}); }}",
            f"void sense_{self.tag}(double t) {{",
            f"    if (std::abs(t - {self.m('t_interrupt')}) < time_eps) controller_{self.tag}(t);",
            "}",
            f"bool actuate_{self.tag}(double t) {{",
            f"    const bool control_due = std::abs(t - {self.m('t_interrupt')}) < time_eps;",
            f"    const bool load_due = std::abs(t - {self.m('t_load')}) < time_eps;",
        ]
        write = self._pwm_write_lines(
            f"{self.m('pending_on')} && {self.m('breaker')} && !{self.m('tripped')}"
        )
        if bridge.computation == 0.0:
            lines += [f"    if (control_due && {self.m('pending_valid')}) {{",
                      *write, "    }"]
        if self.unit.branch_f is None:
            tail = (f"        if (!{self.m('gates')}) "
                    f"{hold} = {self._literal(self.topology.zero)};",)
        else:
            tail = (
                f"        breaker_open_{self.model.prefix[id(self.unit.branch_f)]} = "
                f"!({self.m('breaker')} && {self.m('gates')} && !{self.m('tripped')});",
            )
        lines += self._pwm_load_lines(
            hold,
            switching,
            f"{self.m('active')}[{self.width}] != 0.0 && "
            f"{self.m('breaker')} && !{self.m('tripped')}",
            tail,
        )
        if bridge.computation > 0.0:
            lines += [f"    if (control_due && {self.m('pending_valid')}) {{",
                      *write, "    }"]
        if switching:
            lines += self._pwm_switch_lines(hold)
            if self.unit.branch_f is None:
                lines.append(
                    f"    if (!{self.m('gates')}) "
                    f"{hold} = {self._literal(self.topology.zero)};"
                )
        lines += [f"    if (control_due) {self.m('pending_valid')} = false;",
                  "    return false;", "}"]
        return "\n".join(lines)

    def state_expression(self, *args: Any) -> str:
        if self.continuous:
            local = args[0]
            if local == "tripped":
                return self.m("tripped")
            if local == "startup.run":
                return self.m("startup_run")
            if local == "startup.ramp":
                return self.m("startup_ramp")
            if local == "startup.start":
                return self.m("startup_start")
            raise KeyError(local)
        part, local = args
        if part is self.unit:
            if local == "tripped":
                return self.m("tripped")
        if part is self.bridge:
            bank, _, field = local.rpartition(".")
            word = self.m("shadow" if bank == "shadow" else "active")
            index = (*self.topology.duty_names, "on").index(field)
            return f"{word}[{index}]"
        if part is self.ctrl:
            if local.startswith("held."):
                public = local[5:]
                for key in self.graph.values:
                    if self.graph._state_port(key) == public:
                        return self.g(key)
            if local == "startup.run":
                return self.m("startup_run")
            if local == "startup.ramp":
                return self.m("startup_ramp")
            if local == "startup.steps":
                return self.m("startup_steps")
            loop_name, _, public = local.partition(".")
            node = self.graph.nodes.get(loop_name)
            if node is not None and public in node.get_state():
                return self.node(loop_name, public)
        raise KeyError(local)

    def command_lines(self, on: bool, ramp: float, t: str = "t") -> list[str]:
        effective = self._effective_command(on)
        lines = [f"{self.m('breaker')} = {effective};"]
        lines += self._controller_command_lines(on, ramp, t)
        if self.continuous:
            lines.append(f"{self.m('gates')} = {effective};")
            if self.unit.branch_f is not None:
                lines.append(
                    f"breaker_open_{self.model.prefix[id(self.unit.branch_f)]} = "
                    f"!({self.m('breaker')} && {self.m('gates')} && !{self.m('tripped')});"
                )
        else:
            lines.append(
                f"{self.m('gates')} = {self.m('active')}[{self.width}] != 0.0 && "
                f"{self.m('breaker')} && !{self.m('tripped')};"
            )
            if self.unit.branch_f is None:
                lines.append(
                    f"if (!{self.m('gates')}) "
                    f"{self._hold()} = {self._literal(self.topology.zero)};"
                )
            else:
                lines.append(
                    f"breaker_open_{self.model.prefix[id(self.unit.branch_f)]} = "
                    f"!({self.m('breaker')} && {self.m('gates')} && !{self.m('tripped')});"
                )
        return lines

    def sense_method(self) -> str:
        return f"void sense_{self.tag}(double) {{}}"

    def actuate_method(self) -> str:
        return f"bool actuate_{self.tag}(double) {{ return false; }}"

    def protection_method(self) -> str:
        lines = [f"bool protect_{self.tag}(double) {{"]
        for key, (port, _scale, _initial) in self.maxima.items():
            observed = f"obs_{_identifier(self.name + '.ctrl.' + port)}"
            member = self.m("maximum_" + key)
            lines.append(f"    {member} = std::max({member}, std::abs({observed}));")
        lines += ["    return false;", "}"]
        return "\n".join(lines)

    def protection_summary(self, stream: str = "summary") -> list[str]:
        lines = [
            f'{stream} << ",\\n  \\"{self.name}.trip_cause\\": "; if ({self.m("trip_cause")}.empty()) {stream} << "null"; else {stream} << \'"\' << {self.m("trip_cause")} << \'"\';',
        ]
        lines += [
            f'{stream} << ",\\n  \\"{self.name}.{key}\\": " << '
            f'{self.m("maximum_" + key)} / {_number(scale)};'
            for key, (_port, scale, _initial) in self.maxima.items()
        ]
        lines.append(f'{stream} << ",\\n  \\"{self.name}.alarms\\": []";')
        return lines

def _make_unit_generator(name: str, unit: Any, model: _ModelGenerator) -> Any:
    """Select an optimized controller lowering or the public-port graph lowering."""
    if isinstance(unit.ctrl, GraphController):
        return _GraphUnitGenerator(name, unit, model)
    if isinstance(unit.ctrl, VSCController):
        return (_ContinuousUnitGenerator(name, unit, model) if unit.continuous
                else _UnitGenerator(name, unit, model))
    raise TypeError(
        f"C++ export: controller {type(unit.ctrl).__name__!r} has no generator"
    )


class _CppGenerator:
    """Generate a straight-line C++ model for one assembled simulation."""

    def __init__(self, simulation: Any, name: str, variables: tuple[str, ...]) -> None:
        self.simulation = simulation
        self.p = simulation.p
        self.system = simulation.system
        self.model = self.system.model
        self.name = name
        self.runtime = _RuntimeParameters(self.p, variables)

    def _state_layout(self, model: _ModelGenerator, sampled: dict[str, _ControlGenerator],
                      continuous: dict[str, _ControlGenerator]) -> tuple[list[str], list[str]]:
        """Public state-table columns and their fixed C++ value expressions."""
        columns: list[str] = []
        expressions: list[str] = []
        ode = {name: f"y[{index}]" for index, name in enumerate(self.model.state_labels())}
        for group in self.simulation._states._groups:
            part = group.part
            for state in group.states:
                if part is self.model:
                    values = [ode[column] for column in state.columns]
                else:
                    generator = next((item for name, item in sampled.items()
                                      if part in (item.unit, item.ctrl, item.unit.bridge,
                                                  item.unit.adc)), None)
                    if generator is not None:
                        value = generator.state_expression(part, state.local_name)
                    else:
                        generator_c = next((item for name, item in continuous.items()
                                            if part in (item.unit, item.ctrl, item.unit.bridge)), None)
                        if generator_c is not None:
                            value = generator_c.state_expression(state.local_name)
                        elif not group.states:
                            continue
                        else:
                            owner = group.prefix or type(part).__name__
                            raise TypeError(
                                f"C++ export: state owner {owner!r} ({type(part).__name__}) "
                                "does not expose a generated state mapping"
                            )
                    if state.kind == "complex":
                        values = [f"std::real({value})", f"std::imag({value})"]
                    elif state.kind == "bool":
                        values = [f"({value} ? 1.0 : 0.0)"]
                    else:
                        values = [value]
                columns.extend(state.columns)
                expressions.extend(values)
        return columns, expressions

    def _actions(self, model: _ModelGenerator, sampled: dict[str, _ControlGenerator],
                 continuous: dict[str, _ControlGenerator]) -> tuple[list[float], str]:
        """Remaining file events, specialised to direct member assignments."""
        t0 = self.p.simulation.initial.t
        actions = [action for action in self.simulation._actions() if action[0] > t0 + 1e-10]
        cases: list[str] = []
        for index, (_at, _order, action) in enumerate(actions):
            lines: list[str] = []
            kind = getattr(action, "type", None)
            if kind in ("connect", "disconnect"):
                on = kind == "connect"
                target = action.target
                if target in sampled:
                    lines = sampled[target].command_lines(on, getattr(action, "ramp", 0.0))
                elif target in continuous:
                    lines = continuous[target].command_lines(on, getattr(action, "ramp", 0.0))
                elif leaves := _unit_leaves(self.p.units, target):
                    generators = [sampled.get(leaf) or continuous.get(leaf) for leaf in leaves]
                    if any(generator is None for generator in generators):
                        raise TypeError(f"C++ export: composite unit {target!r} is incomplete")
                    lines = [
                        line
                        for generator in generators
                        for line in generator.command_lines(on, getattr(action, "ramp", 0.0))
                    ]
                elif target in self.system.sources:
                    branch = self.system.sources[target].branch
                    lines = [f"breaker_open_{model.prefix[id(branch)]} = {'false' if on else 'true'};"]
                elif target in self.system.branches:
                    branch = self.system.branches[target]
                    lines = [f"breaker_open_{model.prefix[id(branch)]} = {'false' if on else 'true'};"]
                elif target in self.system.named_elements:
                    element = self.system.named_elements[target]
                    breakers = getattr(element, "breakers", ())
                    if hasattr(element, "branch"):
                        breakers = (element.branch,)
                    lines = [f"breaker_open_{model.prefix[id(branch)]} = {'false' if on else 'true'};"
                             for branch in breakers]
                    if not lines:
                        raise TypeError(f"C++ export: element event target {target!r} has no static breaker")
                else:
                    raise TypeError(f"C++ export: unknown event target {target!r}")
            elif hasattr(action, "paths"):
                # References and source waveforms are compiled as time expressions. Other built-in
                # component coefficients are checked below until their piecewise emitters are used.
                handled = all(
                    (path.startswith("sources.") and path.rsplit(".", 1)[-1] in {"v", "f", "angle"})
                    or (path.startswith("units.") and ".ctrl.references." in path)
                    for path in action.paths
                )
                if not handled:
                    raise TypeError(
                        f"C++ export: set event {action.event!r} changes unsupported paths "
                        f"{list(action.paths)}"
                    )
            else:
                raise TypeError(f"C++ export: custom event {kind!r} needs a C++ lowering hook")
            cases += [f"        case {index}:", *["            " + line for line in lines],
                      "            break;"]
        times = [item[0] for item in actions]
        return times, "\n".join(cases)

    @staticmethod
    def _phase_expressions(value: str) -> tuple[str, str, str]:
        return (f"std::real({value})",
                f"(-0.5 * std::real({value}) + 0.86602540378443864676 * std::imag({value}))",
                f"(-0.5 * std::real({value}) - 0.86602540378443864676 * std::imag({value}))")

    def _signal_layout(self, sampled: dict[str, _ControlGenerator],
                       continuous: dict[str, _ControlGenerator]) -> tuple[list[str], list[str]]:
        """The exact public plant.csv schema and expressions used by Python's recorder."""
        columns: list[str] = []
        expressions: list[str] = []
        self.model.sync(self.p.simulation.initial.t, self.model.get_initial_values())
        signal_values = self.system.signals(self.p.simulation.initial.t)
        for key, value in signal_values.items():
            head, _, what = key.rpartition(".")
            if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
                stem = {"u_g": "v", "i_c": "i_conv", "u": "v"}.get(what, what)
                prefix = f"{head}." if head else ""
                columns.extend(f"{prefix}{stem}_{phase}" for phase in "abc")
                expressions.extend(self._phase_expressions(f"obs_{_identifier(key)}"))
            elif not ((head in self.system.units and what in ("u_dc", "i_dc"))
                      or (head in self.p.sources and what == "angle")):
                columns.append(key)
                expressions.append(f"obs_{_identifier(key)}")
        for name in self.system.units:
            for what in ("u_dc", "i_dc"):
                key = f"{name}.{what}"
                if key in signal_values:
                    columns.append(key)
                    expressions.append(f"obs_{_identifier(key)}")
        for name in self.p.sources:
            columns.append(f"{name}.angle")
            expressions.append(f"obs_{_identifier(name + '.angle')}")
        for name in self.system.units:
            generator = sampled.get(name) or continuous[name]
            for key in generator.ctrl.log_names():
                columns.append(f"{name}.{key}")
                expressions.append(generator.m("log_" + key))
        return columns, expressions

    @staticmethod
    def _csv_header(names: list[str]) -> str:
        return _cpp_string("t" + "".join("," + name for name in names) + "\n")

    @staticmethod
    def _csv_values(stream: str, expressions: list[str], indent: str = "") -> str:
        lines = [f"{indent}{stream} << t;"]
        lines += [f"{indent}{stream} << ',' << ({expression});" for expression in expressions]
        lines.append(f"{indent}{stream} << '\\n';")
        return "\n".join(lines)

    def _energy_method(self, model: _ModelGenerator) -> tuple[str, list[str]]:
        """Lower the model's compiled energy audit to one fixed-size C++ sample."""
        from ..solver.energy import compile_balance

        if self.p.simulation.energy_check == "off":
            return "", []
        plan = compile_balance(self.model)
        lines = ["auto energy_sample(double t) {", "    State dy{};",
                 *["    " + line for line in model.body()],
                 "    double total_energy = 0.0, total_supplied = 0.0, total_dissipated = 0.0;",
                 "    double tellegen = 0.0, max_residual = 0.0, scale = 1e-300;"]
        columns = ["energy_J", "power_supplied_W", "power_dissipated_W",
                   "tellegen_W", "balance_residual_W"]
        values: list[str] = []

        def product(record_a: Any, field_a: str, record_b: Any, field_b: str) -> str:
            a, b = model.value(record_a, field_a), model.value(record_b, field_b)
            kinds = (model.record_types[id(record_a), field_a],
                     model.record_types[id(record_b), field_b])
            return f"std::real(({a}) * std::conj({b}))" if "Complex" in kinds else f"(({a}) * ({b}))"

        into_default = ["0.0" for _ in range(plan.n_defaults)]
        default_terms: list[list[str]] = [[] for _ in range(plan.n_defaults)]
        for transfer in plan.transfers:
            term = product(transfer.other_record, transfer.other_field,
                           transfer.source_record, transfer.source_field)
            default_terms[transfer.default_index].append(
                f"-({_number(transfer.coefficient)}) * ({term})"
            )
        into_default = [" + ".join(terms) if terms else "0.0" for terms in default_terms]

        def powers(check: Any) -> tuple[str, str]:
            sub = check.subsystem
            if isinstance(sub, RLBranch):
                current = model.value(sub.state, "i")
                squared = (f"std::norm({current})" if sub.phases == 3
                           else f"({current}) * ({current})")
                return "0.0", f"{_number(sub.power_scale * sub.R)} * {squared}"
            if isinstance(sub, RCNode):
                current = model.value(sub.inp, "i")
                squared = (f"std::norm({current})" if sub.phases == 3
                           else f"({current}) * ({current})")
                return "0.0", f"{_number(sub.power_scale * sub.R_d)} * {squared}"
            if isinstance(sub, ThreePhaseSource):
                supplied = product(sub.out, "e_g", sub.inp, "i")
                return f"1.5 * ({supplied})", "0.0"
            if isinstance(sub, DCLink):
                i_dc = model.value(sub.inp, "i_dc")
                i_src = model.value(sub.out, "i_src")
                supplied = (f"{_number(sub.source.u_dc)} * {i_src}"
                            if isinstance(sub.source, DCVoltageSource)
                            else f"{model.value(sub.out, 'u_dc')} * {i_src}")
                losses = []
                if sub.capacitor is not None:
                    losses.append(f"{_number(sub.capacitor.R_esr)} * ({i_src} - {i_dc}) * ({i_src} - {i_dc})")
                if isinstance(sub.source, DCVoltageSource):
                    used = i_src if sub.capacitor is not None else i_dc
                    losses.append(f"{_number(sub.source.R)} * {used} * {used}")
                return supplied, " + ".join(losses) if losses else "0.0"
            hook = getattr(sub, "cpp_energy", None)
            if hook is not None:
                supplied, dissipated = hook(model)
                return str(supplied), str(dissipated)
            try:
                return (model.trace_scalar_method(sub, "supplied_power"),
                        model.trace_scalar_method(sub, "dissipated_power"))
            except Exception as exc:
                raise TypeError(
                    f"C++ export: cannot lower the energy equation of {check.name!r} "
                    f"({type(sub).__name__}): {exc}"
                ) from exc

        declared_index = 0
        for check in plan.subsystems:
            kind = check.spec.kind
            if kind in ("observer", "dirac"):
                continue
            tag = f"energy_{declared_index}"
            declared_index += 1
            if kind == "default":
                p_in = into_default[check.default_index]
                lines += [f"    const double {tag}_in = {p_in};",
                          f"    const double {tag}_supplied = -{tag}_in;",
                          f"    tellegen += {tag}_in; total_supplied += {tag}_supplied;",
                          f"    scale = std::max(scale, std::max(std::abs({tag}_in), std::abs({tag}_supplied)));" ]
                columns.append(f"{check.name}.power_supplied_W")
                values.append(f"{tag}_supplied")
                continue
            energy_terms, rate_terms = [], []
            for storage in check.storage:
                state = model.value(storage.record, storage.field)
                if storage.complex_value:
                    derivative = f"Complex{{dy[{storage.index}], dy[{storage.index + 1}]}}"
                    norm = f"std::norm({state})"
                    rate = f"std::real(({state}) * std::conj({derivative}))"
                else:
                    derivative = f"dy[{storage.index}]"
                    norm = f"({state}) * ({state})"
                    rate = f"({state}) * ({derivative})"
                energy_terms.append(f"0.5 * {_number(storage.coefficient)} * ({norm})")
                rate_terms.append(f"{_number(storage.coefficient)} * ({rate})")
            port_terms = [
                f"{_number(port.coefficient)} * ({product(port.effort_record, port.effort_field, port.flow_record, port.flow_field)})"
                for port in check.ports
            ]
            supplied, dissipated = powers(check)
            lines += [
                f"    const double {tag}_energy = {' + '.join(energy_terms) if energy_terms else '0.0'};",
                f"    const double {tag}_in = {' + '.join(port_terms) if port_terms else '0.0'};",
                f"    const double {tag}_supplied = {supplied};",
                f"    const double {tag}_dissipated = {dissipated};",
                f"    const double {tag}_rate = {' + '.join(rate_terms) if rate_terms else '0.0'};",
                f"    const double {tag}_residual = {tag}_in + {tag}_supplied - {tag}_dissipated - {tag}_rate;",
                f"    total_energy += {tag}_energy; total_supplied += {tag}_supplied; total_dissipated += {tag}_dissipated;",
                f"    tellegen += {tag}_in; max_residual = std::max(max_residual, std::abs({tag}_residual));",
                f"    scale = std::max({{scale, std::abs({tag}_in), std::abs({tag}_supplied), std::abs({tag}_dissipated)}});",
            ]
            columns.append(f"{check.name}.energy_J")
            values.append(f"{tag}_energy")
        all_values = ["total_energy", "total_supplied", "total_dissipated", "tellegen",
                      "max_residual", *values, "std::abs(tellegen) / scale", "max_residual / scale"]
        lines += [f"    return std::array<double, {len(all_values)}>{{{{{', '.join(all_values)}}}}};", "}"]
        return "\n".join(lines), columns

    def source(self) -> str:
        """Return the complete generated translation unit."""
        model = _ModelGenerator(self.simulation, self.runtime)
        sampled = {name: _make_unit_generator(name, unit, model)
                   for name, unit in self.system.units.items() if not unit.continuous}
        continuous = {name: model.continuous_units[id(unit.ctrl)]
                      for name, unit in self.system.units.items() if unit.continuous}
        unit_names = tuple(self.system.units)
        state_names = self.model.state_labels()
        initial = self.model.get_initial_values().tolist()
        public_names, state_values = self._state_layout(model, sampled, continuous)
        labels = ",\n        ".join(_cpp_string(name) for name in public_names)
        values = ", ".join(_number(value) for value in initial)
        projection_lines = []
        for index, lower, upper in self.model._state_bounds:
            if lower is not None:
                projection_lines.append(f"y[{index}] = std::max(y[{index}], {_number(lower)});")
            if upper is not None:
                projection_lines.append(f"y[{index}] = std::min(y[{index}], {_number(upper)});")
        projection_body = "\n        ".join(projection_lines)
        projection_method = (
            f"    inline void project_state() {{\n        {projection_body}\n    }}\n"
            if projection_lines else ""
        )
        projection_call = "project_state();" if projection_lines else ""
        diode_conduction = DiodeConduction(self.model)
        diode_current_states = diode_conduction.current_states
        diode_blocking_guards = []
        for subsystem, _index in diode_current_states:
            enabled = model.enabled_by.get(id(subsystem))
            attributes = ({"enabled": _Expression(enabled, "bool")}
                          if enabled is not None else None)
            guard = model.trace_solver_method(
                subsystem, "diode_blocks_reverse_current", attributes
            )
            diode_blocking_guards.append(guard)
        diode_step_limits = []
        for subsystem in diode_conduction.resolution_subsystems:
            if isinstance(subsystem, RLBranch) and subsystem.conduction is not None:
                _derivative, in_dcm = model._diode_conduction(
                    subsystem, solver_scope=True
                )
                diode_step_limits.append(
                    f"({in_dcm} ? {_number(subsystem.conduction.period / 10.0)} "
                    ": std::numeric_limits<double>::infinity())"
                )
            else:
                enabled = model.enabled_by.get(id(subsystem))
                attributes = ({"enabled": _Expression(enabled, "bool")}
                              if enabled is not None else None)
                diode_step_limits.append(model.trace_solver_method(
                    subsystem, "diode_maximum_step", attributes
                ))
        raw_projection_call = "" if diode_conduction else projection_call
        member_lines = model.members()
        for generator in sampled.values():
            member_lines.extend(generator.members())
        members = "\n    ".join(member_lines)
        rhs = "\n        ".join(model.body())
        sync_body = "\n        ".join(model.body(derivatives=False))
        observation_members, observation_assignments = model.observations()
        observations = "\n        ".join(observation_assignments)
        members = "\n    ".join([members, *observation_members])
        methods = []
        for generator in sampled.values():
            methods += [generator.controller_method(), generator.switching_method(),
                        generator.event_methods(), generator.adc_methods(),
                        generator.protection_method()]
        methods += [method for generator in continuous.values()
                    for method in (generator.sense_method(), generator.actuate_method(),
                                   generator.protection_method())]
        energy_method, energy_columns = self._energy_method(model)
        if energy_method:
            methods.append(energy_method)
        methods_text = "\n\n    ".join(method.replace("\n", "\n    ") for method in methods if method)
        next_units = ", ".join(f"next_{generator.tag}()" for generator in sampled.values())
        sense = []
        for name in unit_names:
            generator = sampled.get(name) or continuous.get(name)
            sense += [f"sense_{generator.tag}(t);", "sync(t);"]
        actuate = [f"trip_now = actuate_{(sampled.get(name) or continuous[name]).tag}(t) || trip_now;"
                   for name in unit_names]
        protect = [f"trip_now = protect_{(sampled.get(name) or continuous[name]).tag}(t) || trip_now;"
                   for name in unit_names]
        adc_accumulate = [generator.adc_accumulate_call("t - interval_start")
                          for generator in sampled.values()]
        adc_accumulate = [line for line in adc_accumulate if line]
        adc_seed = [generator.adc_seed_call() for generator in sampled.values()]
        adc_seed = [line for line in adc_seed if line]
        action_times, action_cases = self._actions(model, sampled, continuous)
        times = ", ".join(_number(value) for value in action_times)
        snapshot_values = "\n            ".join(f"states << ',' << ({value});" for value in state_values)
        output = self.p.simulation.output
        resolved_template, resolved_replacements = self.runtime.resolved_template(self.p)
        resolved_text = _cpp_string(resolved_template)
        resolved_replace = "\n    ".join(
            f"replace_all(text, {_cpp_string(marker)}, yaml_number({expression}));"
            for marker, expression in resolved_replacements
        )
        plant_names, plant_values = self._signal_layout(sampled, continuous)
        plant_snapshot = (self._csv_values("plant", plant_values, "                ")
                          if output.signals else "")
        signal_declarations: list[str] = []
        signal_headers: list[str] = []
        sampled_ctrl_rows: list[str] = []
        continuous_ctrl_rows: list[str] = []
        if output.signals:
            signal_declarations += ["std::ofstream plant(out / \"plant.csv\");",
                                    "if (!plant) throw std::runtime_error(\"cannot open plant.csv\");",
                                    "plant << std::setprecision(17); "]
            signal_headers.append(f"plant << {self._csv_header(plant_names)};")
            for name in unit_names:
                generator = sampled.get(name) or continuous[name]
                stream = "ctrl_" + _identifier(name)
                log_names = list(generator.ctrl.log_names())
                signal_declarations += [f"std::ofstream {stream}(out / {_cpp_string('ctrl.' + name + '.csv')});",
                                        f"if (!{stream}) throw std::runtime_error(\"cannot open ctrl.{name}.csv\");",
                                        f"{stream} << std::setprecision(17);"]
                signal_headers.append(f"{stream} << {self._csv_header(log_names)};")
                expressions = [generator.m("log_" + key) for key in log_names]
                row = self._csv_values(stream, expressions, "                ")
                if name in sampled:
                    sampled_ctrl_rows += [
                        f"if (std::abs(t - {generator.m('t_interrupt')}) < time_eps && "
                        f"{generator.m('pwm_k')} % {max(1, output.record_every)} == 0) {{",
                        row,
                        "}",
                    ]
                else:
                    continuous_ctrl_rows += [
                        f"if (snapshot_index % {max(1, output.record_every)} == 0) {{",
                        row,
                        "}",
                    ]
        signal_declarations_text = "\n        ".join(signal_declarations)
        signal_headers_text = "\n        ".join(signal_headers)
        sampled_ctrl_text = "\n            ".join(sampled_ctrl_rows)
        continuous_ctrl_text = "\n                ".join(continuous_ctrl_rows)
        continuous_limit_text = "\n            ".join(
            line for generator in continuous.values() for line in generator.limit_observation()
        )
        energy_on = bool(energy_method)
        energy_declarations: list[str] = []
        energy_header = ""
        energy_row = ""
        if energy_on and output.energy:
            energy_declarations = [
                'std::ofstream energy(out / "energy.csv");',
                'if (!energy) throw std::runtime_error("cannot open energy.csv");',
                'energy << std::setprecision(17);',
            ]
            energy_header = f"energy << {self._csv_header(energy_columns)};"
            energy_row = self._csv_values(
                "energy", [f"energy_values[{index}]" for index in range(len(energy_columns))],
                "                    ")
        energy_declarations_text = "\n        ".join(energy_declarations)
        energy_snapshot = ""
        if energy_on:
            energy_snapshot = "\n".join([
                f"if (snapshot_index % {max(1, self.p.simulation.solver.phs_check_step)} == 0 || final) {{",
                "    const auto energy_values = energy_sample(t);",
                *([energy_row] if energy_row else []),
                f"    energy_tellegen_max_rel = std::max(energy_tellegen_max_rel, energy_values[{len(energy_columns)}]);",
                f"    energy_balance_max_rel = std::max(energy_balance_max_rel, energy_values[{len(energy_columns) + 1}]);",
                "}",
            ])
        global_tripped = " || ".join(
            (sampled.get(name) or continuous[name]).m("tripped") for name in unit_names
        ) or "false"
        summary_rows: list[str] = []
        for name in unit_names:
            generator = sampled.get(name) or continuous[name]
            summary_rows += generator.summary_lines()
        summary_text = "\n        ".join(summary_rows)
        ph_summary = ""
        if self.simulation.ph_report is not None:
            report = self.simulation.ph_report
            fragment = (",\n  \"ph_verdict\": " + json.dumps(report.verdict)
                        + ",\n  \"ph_defaulted\": " + json.dumps(list(report.defaulted))
                        + ",\n  \"ph_report\": " + json.dumps(report.to_dict(), separators=(",", ":")))
            ph_summary = f"summary << {_cpp_string(fragment)};"
        energy_summary = ""
        if energy_on:
            problems = _cpp_string(json.dumps(list(self.simulation.energy_problems)))
            energy_summary = (
                'summary << ",\\n  \\\"energy_tellegen_max_rel\\\": " << energy_tellegen_max_rel'
                ' << ",\\n  \\\"energy_balance_max_rel\\\": " << energy_balance_max_rel'
                f' << ",\\n  \\\"energy_problems\\\": " << {problems};'
            )
        solver = self.p.simulation.solver
        fixed = solver.type == "fixed" and solver.method in ("euler", "heun", "rk4")
        adaptive = solver.type == "adaptive" and solver.method == "DP45"
        if not (fixed or adaptive):
            raise TypeError(
                f"C++ export supports fixed euler/heun/rk4 and adaptive DP45, not "
                f"{solver.type}/{solver.method}"
            )
        runtime_declarations = self.runtime.declarations()
        runtime_setters = self.runtime.setters()
        runtime_listing = self.runtime.listing()
        fixed_dt = self.runtime.value("simulation.solver.dt", solver.dt)
        adaptive_max_step = self.runtime.value("simulation.solver.max_step", solver.max_step)
        adaptive_atol = self.runtime.value("simulation.solver.atol", solver.atol)
        adaptive_rtol = self.runtime.value("simulation.solver.rtol", solver.rtol)
        run_end = self.runtime.value("simulation.t_end", self.p.simulation.t_end)
        output_period_value = self.runtime.value("simulation.output.period", output.period)
        if fixed:
            if solver.method == "euler":
                step_code = '''    inline void step(double& t, double h) {
        State a{};
        rhs(t, y, a);
        for (std::size_t i = 0; i < state_count; ++i) y[i] += h * a[i];
        t += h;
    }'''
            elif solver.method == "heun":
                step_code = '''    inline void step(double& t, double h) {
        State a{}, b{}, x{};
        rhs(t, y, a);
        for (std::size_t i = 0; i < state_count; ++i) x[i] = y[i] + h * a[i];
        rhs(t + h, x, b);
        for (std::size_t i = 0; i < state_count; ++i) y[i] += 0.5 * h * (a[i] + b[i]);
        t += h;
    }'''
            else:
                step_code = '''    inline void step(double& t, double h) {
        State a{}, b{}, c{}, d{}, x{};
        rhs(t, y, a);
        for (std::size_t i = 0; i < state_count; ++i) x[i] = y[i] + 0.5 * h * a[i];
        rhs(t + 0.5 * h, x, b);
        for (std::size_t i = 0; i < state_count; ++i) x[i] = y[i] + 0.5 * h * b[i];
        rhs(t + 0.5 * h, x, c);
        for (std::size_t i = 0; i < state_count; ++i) x[i] = y[i] + h * c[i];
        rhs(t + h, x, d);
        for (std::size_t i = 0; i < state_count; ++i)
            y[i] += (h / 6.0) * (a[i] + 2.0 * (b[i] + c[i]) + d[i]);
        t += h;
    }'''
            solver_code = step_code + f'''

    void integrate(double& t, double target) {{
        const double span = target - t;
        if (span <= 0.0) return;
        const std::size_t count = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(span / {fixed_dt} - 1e-9)));
        const double h = span / static_cast<double>(count);
        for (std::size_t k = 0; k < count; ++k) step(t, h);
        {raw_projection_call}
        t = target;
    }}'''
        else:
            solver_code = f'''    void integrate(double& t, double target) {{
        const double span = target - t;
        if (span <= 0.0) return;
        double h = std::isfinite(adaptive_h) ? adaptive_h : span;
        h = std::min(h, std::min(span, {adaptive_max_step}));
        State k1{{}}, k2{{}}, k3{{}}, k4{{}}, k5{{}}, k6{{}}, k7{{}}, x{{}}, y1{{}};
        rhs(t, y, k1);
        const double end_epsilon = 1e-15 * std::max(1.0, std::abs(target));
        while (t < target - end_epsilon) {{
            if (t + h > target) h = target - t;
            for (std::size_t i = 0; i < state_count; ++i) x[i] = y[i] + h * (1.0 / 5.0) * k1[i];
            rhs(t + h * (1.0 / 5.0), x, k2);
            for (std::size_t i = 0; i < state_count; ++i) x[i] = y[i] + h * ((3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i]);
            rhs(t + h * (3.0 / 10.0), x, k3);
            for (std::size_t i = 0; i < state_count; ++i) x[i] = y[i] + h * ((44.0 / 45.0) * k1[i] - (56.0 / 15.0) * k2[i] + (32.0 / 9.0) * k3[i]);
            rhs(t + h * (4.0 / 5.0), x, k4);
            for (std::size_t i = 0; i < state_count; ++i) x[i] = y[i] + h * ((19372.0 / 6561.0) * k1[i] - (25360.0 / 2187.0) * k2[i] + (64448.0 / 6561.0) * k3[i] - (212.0 / 729.0) * k4[i]);
            rhs(t + h * (8.0 / 9.0), x, k5);
            for (std::size_t i = 0; i < state_count; ++i) x[i] = y[i] + h * ((9017.0 / 3168.0) * k1[i] - (355.0 / 33.0) * k2[i] + (46732.0 / 5247.0) * k3[i] + (49.0 / 176.0) * k4[i] - (5103.0 / 18656.0) * k5[i]);
            rhs(t + h, x, k6);
            for (std::size_t i = 0; i < state_count; ++i) y1[i] = y[i] + h * ((35.0 / 384.0) * k1[i] + (500.0 / 1113.0) * k3[i] + (125.0 / 192.0) * k4[i] - (2187.0 / 6784.0) * k5[i] + (11.0 / 84.0) * k6[i]);
            rhs(t + h, y1, k7);
            double error = 0.0;
            for (std::size_t i = 0; i < state_count; ++i) {{
                const double estimate = std::abs(h * ((71.0 / 57600.0) * k1[i] - (71.0 / 16695.0) * k3[i] + (71.0 / 1920.0) * k4[i] - (17253.0 / 339200.0) * k5[i] + (22.0 / 525.0) * k6[i] - (1.0 / 40.0) * k7[i]));
                error = std::max(error, estimate / ({adaptive_atol} + {adaptive_rtol} * std::max(std::abs(y[i]), std::abs(y1[i]))));
            }}
            if (error <= 1.0 || h <= 1e-15) {{
                t += h;
                y = y1;
                k1 = k7;
                double factor = error == 0.0 ? 5.0 : 0.9 * std::pow(error, -0.14) * std::pow(adaptive_error, 0.08);
                factor = std::clamp(factor, 0.2, 5.0);
                adaptive_error = std::max(error, 1e-4);
                h = std::min(h * factor, {adaptive_max_step});
            }} else {{
                ++n_rejected;
                h *= std::max(0.1, 0.9 * std::pow(error, -0.25));
            }}
        }}
        adaptive_h = h;
        {raw_projection_call}
        t = target;
    }}'''
        advance_method = "integrate"
        diode_declaration = ""
        diode_guard_methods = "\n".join(
            f"    inline bool diode_blocks_reverse_current_{index}() {{ return {guard}; }}"
            for index, guard in enumerate(diode_blocking_guards)
        )
        diode_limit_methods = "\n".join(
            f"    inline double diode_maximum_step_{index}() {{ return {limit}; }}"
            for index, limit in enumerate(diode_step_limits)
        )
        diode_methods = "\n".join(filter(None, (diode_guard_methods, diode_limit_methods)))
        diode_projection_call = "sim.project_state();" if projection_lines else ""
        diode_step_update = " ".join(
            f"diode_step = std::min(diode_step, sim.diode_maximum_step_{index}());"
            for index in range(len(diode_step_limits))
        )
        diode_advance_body = ""
        if diode_step_limits and not diode_current_states:
            solver_code = solver_code.replace(
                "void integrate(double& t, double target)",
                "void integrate_solver(double& t, double target, double maximum_step)",
            )
            if fixed:
                solver_code = solver_code.replace(
                    f"span / {fixed_dt} - 1e-9",
                    f"span / std::min({fixed_dt}, maximum_step) - 1e-9",
                )
            else:
                solver_code = solver_code.replace(
                    f"std::min(span, {adaptive_max_step})",
                    f"std::min(span, std::min({adaptive_max_step}, maximum_step))",
                ).replace(
                    f"std::min(h * factor, {adaptive_max_step})",
                    f"std::min(h * factor, std::min({adaptive_max_step}, maximum_step))",
                )
            diode_advance_body = f'''
            double diode_step = target - t;
            {diode_step_update}
            if (diode_step <= 0.0) throw std::runtime_error("diode conduction step is below time resolution");
            sim.integrate_solver(t, target, diode_step);
            {diode_projection_call}
            t = target;'''
        elif diode_current_states:
            declarations = "\n            ".join(
                f"double root_{index} = std::numeric_limits<double>::infinity();"
                for _subsystem, index in diode_current_states
            )
            searches = []
            zeroes = []
            for guard_index, ((_subsystem, index), _guard) in enumerate(
                    zip(diode_current_states, diode_blocking_guards)):
                searches += [
                    f"if (sim.diode_blocks_reverse_current_{guard_index}() && initial[{index}] > 0.0 && trial[{index}] < 0.0) {{",
                    "    double lower = begin, upper = end;",
                    "    for (int iteration = 0; iteration < 24; ++iteration) {",
                    "        const double middle = lower + 0.5 * (upper - lower);",
                    "        sim.y = initial; double probe = begin; sim.integrate_solver(probe, middle);",
                    f"        if (sim.y[{index}] > 0.0) lower = middle; else upper = middle;",
                    "    }",
                    f"    root_{index} = lower + 0.5 * (upper - lower);",
                    f"    first = std::min(first, root_{index});",
                    "}",
                ]
                zeroes.append(
                    f"if (std::abs(root_{index} - first) <= 2e-14) sim.y[{index}] = 0.0;"
                )
            solver_code = solver_code.replace(
                "void integrate(double& t, double target)",
                "void integrate_solver(double& t, double target)",
            )
            diode_advance_body = f'''
            double diode_step = target - t;
            {diode_step_update}
            while (t < target - 1e-15) {{
                const double begin = t;
                const double end = std::min(target, begin + diode_step);
                if (end <= begin) throw std::runtime_error("diode conduction step is below time resolution");
                const State initial = sim.y;
                sim.integrate_solver(t, end);
                const State trial = sim.y;
                double first = std::numeric_limits<double>::infinity();
                {declarations}
                {' '.join(searches)}
                if (std::isfinite(first)) {{
                    sim.y = initial; t = begin; sim.integrate_solver(t, first);
                    {' '.join(zeroes)}
                }} else {{
                    sim.y = trial; t = end;
                }}
            }}
            {diode_projection_call}
            t = target;'''
        if diode_advance_body:
            solver_code += f'''

{diode_methods}

    class DiodeConduction {{
        Simulator& sim;

    public:
        explicit DiodeConduction(Simulator& owner) : sim(owner) {{}}

        void advance(double& t, double target) {{
{diode_advance_body}
        }}
    }};'''
            advance_method = "diode_conduction.advance"
            diode_declaration = "DiodeConduction diode_conduction(*this);"
        adaptive_members = ('''std::size_t n_rejected = 0;
    double adaptive_h = std::numeric_limits<double>::quiet_NaN();
    double adaptive_error = 1.0;''' if adaptive else "")
        # The execution backend is deliberately emitted as one TU: whole-program optimisation can
        # inline the configured RHS and remove unused output paths without crossing a library ABI.
        return f'''// Generated by PESLite.  Do not edit: regenerate from the source simulation file.
// Standalone C++17; no Python, NumPy, SciPy or YAML dependency at run time.
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace peslite_generated {{

using Complex = std::complex<double>;
constexpr std::size_t state_count = {len(state_names)};
using State = std::array<double, state_count>;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double time_eps = 1e-10;

{runtime_declarations}

inline std::string_view trim(std::string_view text) {{
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) text.remove_prefix(1);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.remove_suffix(1);
    return text;
}}

inline double parse_number(std::string_view text) {{
    text = trim(text);
    if (text.size() >= 2 && ((text.front() == static_cast<char>(39) && text.back() == static_cast<char>(39)) ||
                             (text.front() == static_cast<char>(34) && text.back() == static_cast<char>(34)))) {{
        text.remove_prefix(1); text.remove_suffix(1);
    }}
    if (text == ".inf" || text == "+.inf") return std::numeric_limits<double>::infinity();
    if (text == "-.inf") return -std::numeric_limits<double>::infinity();
    if (text == ".nan") return std::numeric_limits<double>::quiet_NaN();
    std::string value(text);
    std::size_t used = 0;
    double result = 0.0;
    try {{ result = std::stod(value, &used); }}
    catch (const std::exception&) {{ throw std::runtime_error("invalid numeric value '" + value + "'"); }}
    if (used != value.size()) throw std::runtime_error("invalid numeric value '" + value + "'");
    return result;
}}

inline bool try_set_parameter(RuntimeParameters& parameters, std::string_view path,
                              std::string_view text) {{
    path = trim(path);
    text = trim(text);
    {runtime_setters}
    return false;
}}

inline void set_parameter(RuntimeParameters& parameters, std::string_view assignment) {{
    const auto equal = assignment.find('=');
    if (equal == std::string_view::npos || equal == 0 || equal + 1 == assignment.size())
        throw std::runtime_error("--set expects PATH=VALUE");
    const auto path = assignment.substr(0, equal);
    if (!try_set_parameter(parameters, path, assignment.substr(equal + 1)))
        throw std::runtime_error("parameter '" + std::string(trim(path)) + "' is not variable");
}}

inline std::string yaml_key(std::string_view value) {{
    value = trim(value);
    if (value.size() >= 2 && ((value.front() == static_cast<char>(39) && value.back() == static_cast<char>(39)) ||
                              (value.front() == static_cast<char>(34) && value.back() == static_cast<char>(34))))
        value = value.substr(1, value.size() - 2);
    return std::string(value);
}}

inline void load_config(RuntimeParameters& parameters, const std::filesystem::path& file) {{
    std::ifstream input(file);
    if (!input) throw std::runtime_error("cannot open config '" + file.string() + "'");
    std::vector<std::pair<std::size_t, std::string>> levels;
    std::size_t ignored = 0;
    std::string line;
    while (std::getline(input, line)) {{
        std::size_t indent = 0;
        while (indent < line.size() && line[indent] == ' ') ++indent;
        std::string_view content = trim(std::string_view(line).substr(indent));
        if (content.empty() || content.front() == '#' || content == "---" ||
                content == "..." || content.front() == '-') continue;
        const auto colon = content.find(':');
        if (colon == std::string_view::npos) continue;
        const std::string key = yaml_key(content.substr(0, colon));
        std::string_view value = trim(content.substr(colon + 1));
        while (!levels.empty() && levels.back().first >= indent) levels.pop_back();
        std::string path;
        for (const auto& level : levels) {{ if (!path.empty()) path += '.'; path += level.second; }}
        if (!path.empty()) path += '.';
        path += key;
        if (value.empty()) {{ levels.emplace_back(indent, key); continue; }}
        const auto comment = value.find(" #");
        if (comment != std::string_view::npos) value = trim(value.substr(0, comment));
        if (!try_set_parameter(parameters, path, value)) ++ignored;
    }}
    if (ignored)
        std::cerr << "peslite: warning: " << ignored << " parameter"
                  << (ignored == 1 ? " in '" : "s in '") << file.string()
                  << (ignored == 1 ? "' is" : "' are")
                  << " not variable and were ignored\\n";
}}

inline void list_parameters(const RuntimeParameters& parameters) {{
    {runtime_listing}
}}

inline void replace_all(std::string& text, std::string_view from, const std::string& to) {{
    std::size_t position = 0;
    while ((position = text.find(from, position)) != std::string::npos) {{
        text.replace(position, from.size(), to);
        position += to.size();
    }}
}}

inline std::string yaml_number(double value) {{
    if (std::isnan(value)) return ".nan";
    if (std::isinf(value)) return value < 0.0 ? "-.inf" : ".inf";
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}}

inline std::string resolved_parameters(const RuntimeParameters& parameters) {{
    std::string text = {resolved_text};
    {resolved_replace}
    return text;
}}

inline double smoothstep(double x) noexcept {{
    x = std::clamp(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}}

inline Complex abc_to_complex(double a, double b, double c) noexcept {{
    constexpr double root3_2 = 0.866025403784438646763723170752936;
    return (2.0 / 3.0) * Complex{{a - 0.5 * (b + c), root3_2 * (b - c)}};
}}

inline std::array<double, 3> phases(Complex value) noexcept {{
    constexpr double root3_2 = 0.866025403784438646763723170752936;
    return {{std::real(value), -0.5 * std::real(value) + root3_2 * std::imag(value),
            -0.5 * std::real(value) - root3_2 * std::imag(value)}};
}}

inline double carrier_position(double t, double frequency, double phase) noexcept {{
    double value = std::fmod(t * frequency + phase, 1.0);
    if (value < 0.0) value += 1.0;
    if (value > 1.0 - 1e-9 || value < 1e-9) return 0.0;
    if (std::abs(value - 0.5) < 1e-9) return 0.5;
    return value;
}}

inline double carrier_value(double position) noexcept {{
    return position < 0.5 ? -1.0 + 4.0 * position : 3.0 - 4.0 * position;
}}

constexpr std::array<std::string_view, {len(public_names)}> state_names{{
        {labels}
}};

constexpr State initial_state{{{values}}};

class Simulator {{
public:
    RuntimeParameters parameters;
    State y = initial_state;
    std::size_t n_rhs = 0;
    {adaptive_members}
    {members}

    explicit Simulator(RuntimeParameters configured = {{}}) : parameters(std::move(configured)) {{}}

    inline void evaluate(double t, const State& y, State& dy) {{
        {rhs}
    }}

    inline void rhs(double t, const State& y, State& dy) {{
        ++n_rhs;
        evaluate(t, y, dy);
    }}

{projection_method}
    inline void sync(double t) {{
        {sync_body}
        {observations}
    }}

{solver_code}

    {methods_text}

    void apply_action(std::size_t action, double t) {{
        switch (action) {{
{action_cases}
        default: break;
        }}
    }}

    int run(const std::filesystem::path& out) {{
        const auto wall_start = std::chrono::steady_clock::now();
        std::filesystem::create_directories(out);
        std::ofstream states;
        if constexpr ({'true' if output.states else 'false'}) {{
            states.open(out / "states.csv");
            if (!states) throw std::runtime_error("cannot open states.csv");
            states << std::setprecision(17) << "t";
            for (const auto name : state_names) states << ',' << name;
            states << '\\n';
        }}
        {signal_declarations_text}
        {signal_headers_text}
        {energy_declarations_text}
        {energy_header}
        double t = {_number(self.p.simulation.initial.t)};
        {diode_declaration}
        {projection_call}
        const double end = {run_end};
        const double output_period = {output_period_value};
        long long output_index = std::max<long long>(0, static_cast<long long>(std::ceil(t / output_period - 1e-9)));
        double next_output = output_index * output_period;
        constexpr std::array<double, {len(action_times)}> action_times{{{times}}};
        std::size_t action_index = 0;
        std::size_t snapshot_index = 0;
        double energy_tellegen_max_rel = 0.0;
        double energy_balance_max_rel = 0.0;
        auto snapshot = [&](bool final = false) {{
            sync(t);
            {continuous_limit_text}
            if constexpr ({'true' if output.signals else 'false'}) {{
                {continuous_ctrl_text}
                {plant_snapshot}
            }}
            if constexpr ({'true' if output.states else 'false'}) {{
                states << t;
                {snapshot_values}
                states << '\\n';
            }}
            {energy_snapshot}
            ++snapshot_index;
        }};
        while (true) {{
            double boundary = std::min(end, next_output);
            {'boundary = std::min(boundary, std::min({' + next_units + '}));' if next_units else ''}
            if (action_index < action_times.size()) boundary = std::min(boundary, action_times[action_index]);
            const double interval_start = t;
            if (boundary > t + time_eps) {{
                {advance_method}(t, boundary);
            }} else {{
                t = boundary;
            }}
            sync(t);
            {' '.join(adc_accumulate)}
            while (action_index < action_times.size() && action_times[action_index] <= t + time_eps) {{
                apply_action(action_index, t);
                ++action_index;
            }}
            sync(t);
            bool trip_now = false;
            {' '.join(protect)}
            if constexpr ({'true' if self.p.simulation.stop_on_trip else 'false'}) {{
                if (trip_now) {{ snapshot(true); break; }}
            }}
            {' '.join(sense)}
            if constexpr ({'true' if output.signals else 'false'}) {{
                {sampled_ctrl_text}
            }}
            {' '.join(actuate)}
            if constexpr ({'true' if self.p.simulation.stop_on_trip else 'false'}) {{
                if (trip_now) {{ snapshot(true); break; }}
            }}
            {'sync(t); ' + ' '.join(adc_seed) if adc_seed else ''}
            if (t >= end - time_eps) {{ snapshot(true); break; }}
            if (std::abs(next_output - t) < time_eps) {{
                snapshot();
                ++output_index;
                next_output = output_index * output_period;
            }}
        }}
        std::ofstream summary(out / "summary.json");
        const double wall_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();
        summary << std::setprecision(17)
                << "{{\\n  \\"wall_time\\": " << wall_time << ','
                << "\\n  \\"n_rhs\\": " << n_rhs
                << ",\\n  \\"t_start\\": {_number(self.p.simulation.initial.t)},"
                << "\\n  \\"t_stop\\": " << t
                << ",\\n  \\"tripped\\": " << ({global_tripped} ? 1 : 0);
        {summary_text}
        {ph_summary}
        {energy_summary}
        summary << "\\n}}\\n";
        std::ofstream resolved(out / "simulation.pes");
        resolved << resolved_parameters(parameters);
        return 0;
    }}
}};

int run(const std::filesystem::path& out, RuntimeParameters parameters = {{}}) {{
    return Simulator{{std::move(parameters)}}.run(out);
}}

}}  // namespace peslite_generated

int main(int argc, char** argv) {{
    try {{
        peslite_generated::RuntimeParameters parameters;
        std::filesystem::path out = {_cpp_string('output/' + self.name)};
        std::filesystem::path config;
        bool out_given = false;
        bool config_given = false;
        bool list_params = false;
        std::vector<std::string> assignments;
        for (int index = 1; index < argc; ++index) {{
            const std::string_view argument(argv[index]);
            if (argument == "--set") {{
                if (++index >= argc) throw std::runtime_error("--set expects PATH=VALUE");
                assignments.emplace_back(argv[index]);
            }} else if (argument.substr(0, 6) == "--set=") {{
                assignments.emplace_back(argument.substr(6));
            }} else if (argument == "--config") {{
                if (++index >= argc) throw std::runtime_error("--config expects PESFILE");
                config = argv[index]; config_given = true;
            }} else if (argument.substr(0, 9) == "--config=") {{
                config = std::string(argument.substr(9)); config_given = true;
            }} else if (argument == "--out") {{
                if (++index >= argc) throw std::runtime_error("--out expects DIRECTORY");
                out = argv[index]; out_given = true;
            }} else if (argument.substr(0, 6) == "--out=") {{
                out = std::string(argument.substr(6)); out_given = true;
            }} else if (argument == "--list-params") {{
                list_params = true;
            }} else if (argument == "--help" || argument == "-h") {{
                std::cout << "usage: peslite [--config PESFILE] [--set PATH=VALUE]... "
                             "[--out DIRECTORY] [--list-params]\\n";
                return 0;
            }} else if (!argument.empty() && argument.front() != '-' && !out_given) {{
                out = std::string(argument); out_given = true;
            }} else {{
                throw std::runtime_error("unknown argument '" + std::string(argument) + "'");
            }}
        }}
        if (config_given) peslite_generated::load_config(parameters, config);
        for (const auto& assignment : assignments)
            peslite_generated::set_parameter(parameters, assignment);
        if (list_params) {{ peslite_generated::list_parameters(parameters); return 0; }}
        return peslite_generated::run(out, std::move(parameters));
    }} catch (const std::exception& error) {{
        std::cerr << "peslite: " << error.what() << '\\n';
        return 1;
    }}
}}
'''


@_exporter("cpp")
def _export_cpp(simulation: Any, out_dir: Path, name: str,
                variables: tuple[str, ...], reusable: bool) -> ExportResult:
    """Write one self-contained, parameter-specialised C++17 translation unit."""
    # Execute only the ordinary initialisation path.  A caller-owned Simulation is copied to avoid
    # mutation; the Simulation already created internally by export(Params, ...) can be reused.
    # Initialisation replaces diagnostic values temporarily used by the port-Hamiltonian structure
    # probe with the real initial held inputs and controller state.
    from ..solver.simulation import Simulation

    prepared = simulation if reusable else Simulation(simulation.p)
    if prepared.p.simulation.solver.subsystems:
        raise TypeError(
            "C++ export does not yet lower multirate subsystem schedules; use a single-rate "
            "fixed or DP45 solver for the exported simulator"
        )
    t0 = prepared.p.simulation.initial.t
    model = prepared.system.model
    y = prepared._apply_initial(t0)
    synced_at: float | None = None

    def outputs(t: float) -> None:
        nonlocal synced_at
        if synced_at != t:
            model.sync(t, y)
            synced_at = t

    def hold(label: str, value: Any) -> None:
        model.set_zoh_input(label, value)

    @contextmanager
    def change():
        nonlocal y, synced_at
        yield
        y = model.get_initial_values()
        synced_at = None

    plant = SimpleNamespace(outputs=outputs, hold=hold, change=change)
    for unit in prepared.system.units.values():
        unit.begin(t0, plant, unit.name in prepared._continued,
                   unit.name in prepared._windows_given)
    y = model.get_initial_values()
    model.sync(t0, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    source = out_dir / "peslite.cpp"
    source.write_text(_CppGenerator(prepared, name, variables).source(), encoding="utf-8")
    return ExportResult("cpp", out_dir, (source,))
