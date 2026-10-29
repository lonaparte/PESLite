"""Parameter-specialised standalone C++17 simulator backend.

The backend emits one translation unit.  The resolved configuration, state layout, connection
plan and schedules are embedded in that unit; the generated executable needs only the C++17
standard library and never calls Python.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import numbers
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import cached_property
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..components.converter import DCLink
from ..components.network import ACVoltageSource, RLBranch
from ..components.pwm import (TIME_EPS, _EPS, _SNAP, CarrierComparison, SynchronousCarrier, PhaseClock,
                               PhaseShiftModulator, ZOH)
from ..control.profiles import PHASE_SHIFT_MODES
from ..control.blocks import OutputLimit
from ..control.controller import GraphController, VSCController
from ..solver.energy import re_product
from ..solver.integrators import DormandPrince45, FixedStepSolver, TABLEAUS
from ..solver.model import _strong_components, commutation_tolerance
from .conduction import DiodeConduction, _LOCATOR_ITERATIONS, _MAX_COMMUTATIONS
from .protection import ProtectionStats
from .params import (BaseValues, SourceParams, UnitParams, _PARAMETER_RECORDER, _event_assignments,
                     _written_out, connection_defaults, dumps, parameter_leaves)
from .exporter import ExportResult, _exporter, _signal_names
from .system import _source_scenario
from .types import unit_type
from .unit import Plant

__all__ = []


_ZERO_COEFFICIENTS: ContextVar[dict | None] = ContextVar("peslite_cpp_zero_coefficients", default=None)
_CONDITIONAL_DEPENDENCIES: ContextVar[tuple | None] = ContextVar(
    "peslite_cpp_conditional_dependencies", default=None)
_SCALAR_TERMS: ContextVar[tuple | None] = ContextVar("peslite_cpp_scalar_terms", default=None)
_TRACE_WORK_BUDGET = (1 << 13) - 1  # bound work instead of the depth of a narrow equation

_MOVING_WINDOW = r'''template <std::size_t N> struct MovingWindow {
    std::array<double, N> buf{};
    std::size_t index = 0, count = 0;
    double push(double value) {
        const double old = count == N ? buf[index] : 0.0;
        buf[index] = value;
        index = (index + 1) % N;
        count = std::min(N, count + 1);
        return old;
    }
};'''

_CONTROL_MATH = r'''inline double smoothstep(double x) noexcept {
    x = std::clamp(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

inline Complex abc_to_complex(double a, double b, double c) noexcept {
    constexpr double root3_2 = 0.866025403784438646763723170752936;
    return (2.0 / 3.0) * Complex{a - 0.5 * (b + c), root3_2 * (b - c)};
}

inline std::array<double, 3> phases(Complex value) noexcept {
    constexpr double root3_2 = 0.866025403784438646763723170752936;
    return {std::real(value), -0.5 * std::real(value) + root3_2 * std::imag(value),
            -0.5 * std::real(value) - root3_2 * std::imag(value)};
}'''


def _specialized_zero(value):
    coefficients = _ZERO_COEFFICIENTS.get()
    if (coefficients is not None and isinstance(value, _InitializationExpression)
            and value.default == 0.0):
        coefficients[value.code] = value
        return True
    return False


_ALGEBRAIC_SOLVER = r'''
#if defined(_MSC_VER)
#define PESLITE_FORCEINLINE __forceinline
#define PESLITE_INLINE_LAMBDA
#elif defined(__GNUC__) || defined(__clang__)
#define PESLITE_FORCEINLINE inline __attribute__((always_inline))
#define PESLITE_INLINE_LAMBDA __attribute__((always_inline))
#else
#define PESLITE_FORCEINLINE inline
#define PESLITE_INLINE_LAMBDA
#endif

template <typename Value, typename Sweep>
PESLITE_FORCEINLINE void solve_algebraic(Sweep sweep, Value& current, Value& inverse, bool& has_inverse,
                            int iterations, double rtol, double atol, double step_limit,
                            const char* error) {
    auto finite = [](Value x) {
        return std::isfinite(std::real(x)) && std::isfinite(std::imag(x));
    };
    auto close = [&](Value a, Value b) {
        return std::abs(a - b) <= atol + rtol * std::max({1.0, std::abs(a), std::abs(b)});
    };
    Value x0 = current, y0 = sweep(x0), r0 = y0 - x0;
    if (close(y0, x0)) return;
    Value x1 = has_inverse ? x0 - inverse * r0 : y0;
    if (!finite(x1) || std::abs(x1 - x0) > step_limit * std::max({1.0, std::abs(x0), std::abs(y0)}))
        x1 = y0;
    for (int iteration = 0; iteration < iterations - 1; ++iteration) {
        const Value y1 = sweep(x1), r1 = y1 - x1, denominator = r1 - r0;
        if (denominator != Value{}) {
            const Value estimate = (x1 - x0) / denominator;
            if (finite(estimate)) { inverse = estimate; has_inverse = true; }
        }
        if (close(y1, x1)) { current = x1; return; }
        Value x2 = has_inverse ? x1 - inverse * r1 : y1;
        if (!finite(x2) || std::abs(x2 - x1) > step_limit * std::max({1.0, std::abs(x1), std::abs(y1)}))
            x2 = y1;
        x0 = x1; r0 = r1; x1 = x2;
    }
    throw std::runtime_error(error);
}
'''


_CONDUCTION_KERNEL = r'''
    class DiodeConduction {
        Simulator& sim;
        static constexpr std::size_t commutation_count = @@COMMUTATIONS@@;
        using Guards = std::array<bool, @@CURRENTS@@>;

        static double tolerance(double t) { return @@TOLERANCE@@; }

        void raw_advance(double begin, double end, const State& initial) {
            sim.y = initial;
            double at = begin;
            @@RAW_ADVANCE@@
        }

        double margin(double t, const State& state, const Guards& guarded) {
            sim.y = state;
            if constexpr (commutation_count != 0) sim.sync(t);
            double value = std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < commutation_count; ++k)
                value = std::min(value, sim.commutation_margin(k));
            @@CURRENT_MARGINS@@
            return value;
        }

        double next_time() {
            double next = std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < commutation_count; ++k)
                next = std::min(next, sim.commutation_next(k));
            return next;
        }

    public:
        explicit DiodeConduction(Simulator& owner) : sim(owner) {}

        void synchronize(double t) {
            if constexpr (commutation_count == 0) return;
            for (int iteration = 0; iteration < @@MAX_COMMUTATIONS@@; ++iteration) {
                sim.sync(t);
                std::array<bool, commutation_count> pending{};
                bool any = false;
                for (std::size_t k = 0; k < commutation_count; ++k) {
                    pending[k] = sim.commutation_margin(k) < 0.0
                              || sim.commutation_next(k) <= t + tolerance(t);
                    any = any || pending[k];
                }
                if (!any) return;
                for (std::size_t k = 0; k < commutation_count; ++k) {
                    if (!pending[k]) continue;
                    sim.sync(t);
                    if (sim.commutation_next(k) <= t + tolerance(t)
                            || sim.commutation_margin(k) < 0.0)
                        sim.commutate(k, t);
                }
            }
            throw std::runtime_error("commutations cannot settle without time advancing");
        }

        void advance(double& t, double target) {
            synchronize(t);
            int stalled = 0;
            while (t < target - tolerance(target)) {
                const double begin = t;
                const double scheduled = next_time();
                double end = std::min({target, begin + (@@PROBE@@), scheduled});
                const State initial = sim.y;
                const Guards guarded{{@@GUARDS@@}};
                raw_advance(begin, end, initial);
                State trial = sim.y;
                double upper_margin = margin(end, trial, guarded);
                if (upper_margin < 0.0) {
                    double lower = begin, upper = end;
                    double lower_margin = margin(begin, initial, guarded);
                    bool bisect_next = false;
                    for (int iteration = 0; iteration < @@LOCATOR_ITERATIONS@@; ++iteration) {
                        const double resolution = tolerance(upper), width = upper - lower;
                        if (width <= resolution) break;
                        double middle = lower + 0.5 * width;
                        bool interpolated = false;
                        if (!bisect_next && lower_margin > 0.0 && upper_margin < 0.0) {
                            const double candidate = std::max(lower + resolution, std::min(
                                lower + width * lower_margin / (lower_margin - upper_margin),
                                upper - resolution));
                            if (lower < candidate && candidate < upper) {
                                middle = candidate; interpolated = true;
                            }
                        }
                        bisect_next = interpolated;
                        raw_advance(begin, middle, initial);
                        double value = margin(middle, sim.y, guarded);
                        const bool invalid = value < 0.0;
                        if (invalid) { upper = middle; upper_margin = value; trial = sim.y; }
                        else { lower = middle; lower_margin = value; }
                        if (interpolated) {
                            const double adjacent = middle + (invalid ? -resolution : resolution);
                            if (lower < adjacent && adjacent < upper) {
                                raw_advance(begin, adjacent, initial);
                                value = margin(adjacent, sim.y, guarded);
                                if (value < 0.0) { upper = adjacent; upper_margin = value; trial = sim.y; }
                                else { lower = adjacent; lower_margin = value; }
                            }
                        }
                    }
                    end = upper;
                    @@CLAMP_CURRENTS@@
                }
                sim.y = trial;
                if (upper_margin < 0.0 || scheduled <= end) synchronize(end);
                if (end == target) { @@PROJECT@@ t = target; return; }
                stalled = end - begin <= tolerance(begin) ? stalled + 1 : 0;
                if (stalled >= @@MAX_COMMUTATIONS@@)
                    throw std::runtime_error("conduction events cannot advance time");
                t = end;
            }
            @@PROJECT@@
            t = target;
        }
    };
'''


def _number(value: float) -> str:
    """A finite C++ double literal preserving the Python value."""
    if _specialized_zero(value):
        return "0.0"
    if isinstance(value, _InitializationExpression):
        return value.runtime.cached(value).code
    if isinstance(value, _Expression):
        return value.code
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


def _uses_implementation(obj: Any, implementation: type, *operations: str) -> bool:
    """Check only replaced execution paths and the implementation methods they reference."""
    if not isinstance(obj, implementation):
        return False
    pending = list(operations or ("__call__",))
    checked = set()
    while pending:
        name = pending.pop()
        if name in checked:
            continue
        checked.add(name)
        expected = inspect.getattr_static(implementation, name, None)
        # Special methods use Python's type-level lookup, ordinary methods use the instance.
        actual = inspect.getattr_static(type(obj) if name.startswith("__") else obj, name, None)
        if getattr(actual, "__func__", actual) is not getattr(expected, "__func__", expected):
            return False
        function = expected.fget if isinstance(expected, property) else getattr(expected, "__func__", expected)
        codes = [function.__code__] if hasattr(function, "__code__") else []
        while codes:
            code = codes.pop()
            for field in code.co_names:
                member = inspect.getattr_static(implementation, field, None)
                if callable(member) or isinstance(member, (property, staticmethod, classmethod)):
                    pending.append(field)
            codes.extend(value for value in code.co_consts if inspect.iscode(value))
    return True


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
    if isinstance(value, _Expression):
        return value.code
    value = complex(value)
    return f"Complex{{{_number(value.real)}, {_number(value.imag)}}}"


def _literal(value: Any) -> str:
    if isinstance(value, complex):
        return _complex(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return _number(value)


_CPP_TYPES = {"bool": "bool", "complex": "Complex", "real": "double"}  # by ``_Expression`` kind


def _weighted_sum(weights, names) -> str:
    """Unroll a shared RK coefficient row; no coefficient lookup reaches the executable."""
    return " + ".join(name if weight == 1.0 else f"{_number(weight)} * {name}"
                      for weight, name in zip(weights, names) if weight) or "0.0"


def _assign_values(prefix, assignments):
    """Evaluate a simultaneous state update before writing any of its destinations."""
    assignments = tuple(assignments)
    temporary = tuple(f"{prefix}_{index}" for index in range(len(assignments)))
    return ([f"const auto {name} = {value.code};"
             for name, (_target, value) in zip(temporary, assignments)]
            + [f"{target} = {name};" for (target, _value), name in zip(assignments, temporary)])


def _fixed_step(method: str) -> str:
    c, a, b = TABLEAUS[method]
    names = tuple(f"k{index}" for index in range(len(c)))
    lines = ["    inline void step(double& t, double h) {",
             "        State " + ", ".join(f"{name}{{}}" for name in names)
             + (", x{};" if len(c) > 1 else ";"),
             f"        rhs(t, y, {names[0]});"]
    for index in range(1, len(c)):
        lines += ["        for (std::size_t i = 0; i < state_count; ++i) "
                  f"x[i] = y[i] + h * ({_weighted_sum(a[index - 1], (name + '[i]' for name in names))});",
                  f"        rhs(t + {_number(c[index])} * h, x, {names[index]});"]
    factor = min(abs(value) for value in b if value)
    weighted = _weighted_sum((value / factor for value in b),
                             (name + "[i]" for name in names))
    lines += ["        for (std::size_t i = 0; i < state_count; ++i) "
              f"y[i] += (h / {_number(1.0 / factor)}) * ({weighted});",
              "        t += h;", "    }"]
    return "\n".join(lines)


class _Expression:
    """One scalar expression traced from an ordinary custom subsystem equation.

    This exists only while generating C++; generated simulators contain the resulting static
    expression and no tracing or dispatch machinery.
    """

    __array_priority__ = 1000

    def __init__(self, code: str, kind: str) -> None:
        pool = _SCALAR_TERMS.get()
        if (pool is not None and kind != "bool" and len(code) > 48 and "/" not in code
                and any(char in code for char in "(+-*?{")):
            calls = set(re.findall(r"([A-Za-z_]\w*(?:::\w+)*)\s*\(", code))
            if calls <= {"std::real", "std::imag", "std::conj", "std::abs", "std::min", "std::max"}:
                prefix, terms, runtime = pool
                conditional = _CONDITIONAL_DEPENDENCIES.get()
                pure_parameter = (conditional is not None
                                  and runtime.parameter_dependencies((code,))[code] is not None)
                if not pure_parameter and (conditional is None
                        or not any(fragment in code for fragment in conditional[1])):
                    if code not in terms:
                        name = terms[code] = f"{prefix}{len(terms)}"
                        runtime._expression_dependencies[name] = code
                    code = terms[code]
        self.code, self.kind = code, kind

    def item(self):
        """NumPy scalar extraction preserves the already-scalar symbolic expression."""
        return self

    @staticmethod
    def of(value: Any) -> "_Expression":
        if _specialized_zero(value):
            return _Expression("0.0", value.kind)
        if isinstance(value, _Expression):
            return value
        if getattr(value, "shape", None) == ():
            return _Expression.of(value.item())
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
        if _specialized_zero(self):
            return self.of(self)._binary(other, op)
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
        code = f"(({self.code}) {op} ({right.code}))"
        trace = _CONDITIONAL_DEPENDENCIES.get() if op == "*" else None
        if trace is not None:
            candidates = ((self.code, right.code, "0.0"), (right.code, self.code, "0.0"))
            parameters = trace[0].parameter_dependencies((self.code, right.code))
            contributions = tuple(item for item in candidates if parameters[item[0]])
            if contributions:
                trace[1][code] = contributions
        return _Expression(code, self._result(self, right))

    def _rbinary(self, other: Any, op: str) -> "_Expression":
        return self.of(other)._binary(self, op)

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
        if self.kind == right.kind == "bool" and self.code == right.code:
            return _Expression("true" if op in ("==", "<=", ">=") else "false", "bool")
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
        if self.kind == "bool" and self.code in ("true", "false"):
            return self.code == "true"
        control = _TRACE_CONTROL.get()
        if control is None:
            raise TypeError("a run-time equation value cannot control a Python branch")
        condition = self if self.kind == "bool" else _Expression(
            f"(std::abs({self.code}) != 0.0)", "bool"
        )
        if condition.code in control.known:
            return control.known[condition.code]
        index = control.index
        control.index += 1
        if index >= len(control.decisions):
            raise _TraceBranch(condition)
        value = control.decisions[index]
        control.known[condition.code] = value
        return value

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
            "isfinite": "std::isfinite", "isnan": "std::isnan",
            "rint": "std::nearbyint", "floor": "std::floor", "ceil": "std::ceil",
        }
        if name in functions:
            kind = ("bool" if name in {"isfinite", "isnan"} else
                    "complex" if value.kind == "complex" else "real")
            return _Expression(f"{functions[name]}({value.code})", kind)
        if name == "arctan2":
            left, right = (self.of(item) for item in inputs)
            return _Expression(f"std::atan2({left.code}, {right.code})", "real")
        if name == "copysign":
            left, right = (self.of(item) for item in inputs)
            return _Expression(f"std::copysign({left.code}, {right.code})", "real")
        if name in {"maximum", "minimum"}:
            left, right = (self.of(item) for item in inputs)
            function = "std::max" if name == "maximum" else "std::min"
            return _Expression(
                f"{function}({left.code}, {right.code})",
                self._result(left, right),
            )
        comparisons = {"less": "<", "less_equal": "<=", "greater": ">",
                       "greater_equal": ">=", "equal": "==", "not_equal": "!="}
        if name in comparisons:
            return self.of(inputs[0])._compare(inputs[1], comparisons[name])
        return NotImplemented

    def __array_function__(self, function, types, args, kwargs):
        if function.__name__ in {"real", "imag"} and len(args) == 1 and not kwargs:
            return getattr(self.of(args[0]), function.__name__)
        if function.__name__ == "where" and len(args) == 3 and not kwargs:
            condition, left, right = (self.of(value) for value in args)
            return _Expression(f"(({condition.code}) ? ({left.code}) : ({right.code}))",
                               self._result(left, right))
        return NotImplemented


class _InitializationExpression(_Expression):
    """Numerical initialization with both its default and its generated expression.

    Construction follows the ordinary Python factories. A branch or an array-size conversion
    records a checked specialization condition; numerical coefficients remain expressions.
    """

    @staticmethod
    def make(expression, default, runtime):
        if expression.kind != "complex":
            value = float.__new__(_ParameterValue, default)
        else:
            value = complex.__new__(_ParameterComplex, default)
        value.code, value.kind = expression.code, expression.kind
        value.default, value.runtime = default, runtime
        return value

    @staticmethod
    def nominal(value):
        return value.default if isinstance(value, _InitializationExpression) else value

    def wrap(self, expression, default):
        return self.make(expression, default, self.runtime)

    def _binary(self, other, op):
        expression = super()._binary(other, op)
        if (_TRACE_CONTROL.get() is not None
                or isinstance(other, _Expression) and not isinstance(other, _InitializationExpression)):
            return expression
        left, right = self.default, self.nominal(other)
        operations = {"+": lambda: left + right, "-": lambda: left - right,
                      "*": lambda: left * right, "/": lambda: left / right}
        return self.wrap(expression, operations[op]())

    def _rbinary(self, other, op):
        if isinstance(other, _Expression) and not isinstance(other, _InitializationExpression):
            return super()._rbinary(other, op)
        return self.wrap(self.of(other), self.nominal(other))._binary(self, op)

    def __pow__(self, other):
        return self.wrap(super().__pow__(other), self.default ** self.nominal(other))

    def __rpow__(self, other):
        return self.wrap(super().__rpow__(other), self.nominal(other) ** self.default)

    def __neg__(self): return self.wrap(super().__neg__(), -self.default)
    def __abs__(self): return self.wrap(super().__abs__(), abs(self.default))

    @property
    def real(self): return self.wrap(super().real, self.default.real)

    @property
    def imag(self): return self.wrap(super().imag, self.default.imag)

    def conjugate(self): return self.wrap(super().conjugate(), self.default.conjugate())

    def _compare(self, other, op):
        if other is None and op in {"==", "!="}:
            return op == "!="
        expression = super()._compare(other, op)
        if isinstance(other, _Expression) and not isinstance(other, _InitializationExpression):
            return expression
        left, right = self.default, self.nominal(other)
        operations = {"<": lambda: left < right, "<=": lambda: left <= right,
                      ">": lambda: left > right, ">=": lambda: left >= right,
                      "==": lambda: left == right, "!=": lambda: left != right}
        return self.wrap(expression, operations[op]())

    def __bool__(self):
        expected = bool(self.default)
        condition = self.code if self.kind == "bool" else f"(({self.code}) != 0.0)"
        # Construction already guarded layout-changing choices. Preserve that same choice
        # while tracing equations instead of exploring an incompatible state layout.
        if _TRACE_CONTROL.get() is not None and (condition, expected) not in self.runtime.guards:
            return super().__bool__()
        self.runtime.guards.setdefault((condition, expected), None)
        return expected

    def __round__(self, digits=None):
        if digits is not None:
            factor = 10.0 ** digits
            return round(self * factor) / factor
        value = round(self.default)
        if not self.runtime.checking_layout:
            return self.wrap(_Expression(f"std::nearbyint({self.code})", "real"), value)
        self.runtime.guards.setdefault((f"(std::nearbyint({self.code}) == {value})", True), None)
        return value

    def __int__(self):
        value = int(self.default)
        if self.runtime.checking_layout:
            self.runtime.guards.setdefault((f"(std::trunc({self.code}) == {value})", True), None)
        return value

    def __trunc__(self):
        return self.wrap(_Expression(f"std::trunc({self.code})", "real"), int(self.default))

    def __floor__(self):
        return self.wrap(_Expression(f"std::floor({self.code})", "real"), math.floor(self.default))

    def __mod__(self, other):
        right = self.of(other)
        return self.wrap(_Expression(
            f"(({self.code}) - ({right.code}) * std::floor(({self.code}) / ({right.code})))",
            "real"), self.default % self.nominal(other))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        expression = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if expression is NotImplemented:
            return expression
        if _TRACE_CONTROL.get() is not None or any(isinstance(value, _Expression) and not isinstance(value, _InitializationExpression)
               for value in inputs):
            return expression
        default = ufunc(*(self.nominal(value) for value in inputs), **kwargs)
        return self.wrap(expression, default.item() if hasattr(default, "item") else default)

    def __array_function__(self, function, types, args, kwargs):
        expression = super().__array_function__(function, types, args, kwargs)
        if expression is NotImplemented:
            return expression
        default = function(*(self.nominal(value) for value in args), **kwargs)
        return self.wrap(expression, default.item() if hasattr(default, "item") else default)

    def __str__(self): return self.code
    def __format__(self, spec): return format(self.default, spec) if spec else self.code


class _ParameterValue(_InitializationExpression, float):
    __hash__ = float.__hash__


class _ParameterComplex(_InitializationExpression, complex):
    __hash__ = complex.__hash__


@dataclass
class _TraceControl:
    decisions: tuple[bool, ...]
    index: int = 0
    known: dict[str, bool] = field(default_factory=dict)


class _TraceBranch(Exception):
    def __init__(self, condition: _Expression) -> None:
        self.condition = condition


@dataclass
class _TraceFailure:
    error: ValueError


_TRACE_CONTROL: ContextVar[_TraceControl | None] = ContextVar(
    "peslite_cpp_trace_control", default=None
)


@dataclass(frozen=True)
class _RuntimeParameter:
    path: str
    member: str
    default: float | int | bool
    derived: bool = False


class _RuntimeParameters:
    """Scalar parameters and unit commands selected through the shared export interface."""

    def __init__(self, p: Any, requested: tuple[str, ...], *, exclude=()) -> None:
        available = {path: _RuntimeParameter(path, "param_" + _identifier(path), value, derived)
                     for path, value, derived in parameter_leaves(p)}
        if requested:
            available.update((path, _RuntimeParameter(path, "param_" + _identifier(path), value))
                             for path, value in connection_defaults(p))

        selected = list(available) if "all" in requested else []
        selected += [path for prefix in requested if prefix != "all" for path in available
                     if path == prefix or path.startswith(prefix + ".")]
        selected += [prefix for prefix in requested if prefix != "all" and not any(
            path == prefix or path.startswith(prefix + ".") for path in available)]
        selected = list(dict.fromkeys(selected))
        unknown = [path for path in selected if path not in available]
        if unknown:
            known = ", ".join(available)
            raise ValueError(
                f"parameter {unknown[0]!r} is not variable"
                + (f"; available: {known}" if known else "")
            )
        self.available = available
        self.selected = tuple(available[path] for path in selected if path not in exclude)
        if len({item.member for item in self.selected}) != len(self.selected):
            raise ValueError("exported parameter names collide after C++ identifier normalization")
        self.by_path = {item.path: item for item in self.selected}
        self.guards = {}
        self.initializers = {}
        self._expression_dependencies = {}
        self._equation_counter = 0
        self._parameter_dependencies = {}
        self.copies = {}
        self.object_updates = {}
        self.object_copies = {}
        self._contains_parameters = {}
        self.checking_layout = False
        self.base = getattr(p, "base", None)
        self.p = self.copy(p)
        if self.selected:
            assignments, changes, before = {}, [], self.p
            for change in p.changes:
                after = self.copy(change.params) if change.params is not None else before
                paths = list(change.paths)
                values = dict(_event_assignments(before, p.events[change.event]))
                for path, value in values.items():
                    if path not in change.connections:
                        assignments[path] = value
                    if (path in self.by_path or isinstance(value, _InitializationExpression)) and path not in paths:
                        paths.append(path)
                if change.params is not None:
                    for path, value in assignments.items():
                        self.put(after, path, value)
                changes.append(replace(change, params=after if change.params is not None else None,
                                       paths=tuple(paths), values=values))
                before = after
            self.p.__dict__["changes"] = tuple(changes)
        self.initial_p = self.p
        for change in self.p.changes:
            if (change.t <= p.simulation.initial.t + TIME_EPS
                    and any(path not in change.connections for path in change.paths)):
                self.initial_p = change.params
        self.initial_copies = {}
        if self.initial_p is not self.p:
            self._initial_copies(p, self.initial_p)
        self.checking_layout = True

    def _initial_copies(self, original, initialized):
        """Bind construction inputs to the ordinary parameter state at the initial instant."""
        if is_dataclass(original):
            self.initial_copies[id(original)] = initialized
            for item in fields(original):
                if item.name not in {"events", "changes"}:
                    self._initial_copies(getattr(original, item.name), getattr(initialized, item.name))
        elif isinstance(original, dict):
            self.initial_copies[id(original)] = initialized
            for key in original.keys() & initialized.keys():
                self._initial_copies(original[key], initialized[key])

    def construction_parameters(self, value):
        if id(value) in self.initial_copies:
            return self.initial_copies[id(value)]
        if isinstance(value, (tuple, list)):
            return type(value)(self.construction_parameters(item) for item in value)
        return self.copy(value)

    @staticmethod
    def at(value, path):
        for field in path.split("."):
            value = value[field] if isinstance(value, dict) else getattr(value, field)
        return value

    @staticmethod
    def put(value, path, replacement):
        *head, tail = path.split(".")
        for field in head:
            value = value[field] if isinstance(value, dict) else getattr(value, field)
        if isinstance(value, dict):
            value[tail] = replacement
        else:
            object.__setattr__(value, tail, replacement)

    def cached(self, value):
        if value.code.startswith("parameters.") or value.code.startswith("initial_"):
            return value
        key = value.code, value.kind
        member = self.initializers.setdefault(key, f"initial_{len(self.initializers)}")
        self._expression_dependencies.setdefault(member, value.code)
        return value.wrap(_Expression(member, value.kind), value.default)

    def initialization_members(self):
        return [f"{_CPP_TYPES[kind]} {member} = {code};"
                for (code, kind), member in self.initializers.items()]

    @cached_property
    def _parameter_members(self):
        members = {item.member: item.path for item in self.selected}
        members.update({"given_" + item.member: item.path for item in self.selected if item.derived})
        return members

    def expression_symbols(self, code):
        """Yield leaf symbols through shared scalar expressions without expanding strings."""
        seen, pending = set(), [code]
        while pending:
            text = pending.pop()
            for name in re.findall(r"\b[A-Za-z_]\w*(?:::\w+)*\b", text):
                if name in self._expression_dependencies:
                    if name not in seen:
                        seen.add(name)
                        pending.append(self._expression_dependencies[name])
                else:
                    yield name

    def parameter_dependencies(self, expressions):
        """Resolve pure parameter coefficients through the scalar dependency DAG.

        Unknown run-time symbols are rejected rather than treated as constants. No parameter
        values are tried, and cached initializers are visited at most once per expression.
        """
        members = self._parameter_members
        def resolve(code):
            paths = set()
            for name in self.expression_symbols(code):
                if name in members:
                    paths.add(members[name])
                elif name not in {"parameters", "true", "false", "Complex"} and not name.startswith("std::"):
                    return None
            return tuple(sorted(paths))

        for code in expressions:
            if code not in self._parameter_dependencies:
                self._parameter_dependencies[code] = resolve(code)
        return {code: self._parameter_dependencies[code] for code in expressions}

    def validation(self):
        return "\n        ".join(
            f"if ({condition}) throw std::runtime_error("
            f"{_cpp_string('invalid parameter combination or changed structural specialization: ' + code)});"
            for code, condition in self.validation_conditions())

    def validation_conditions(self):
        return [(code, f"{'!' if expected else ''}({code})")
                for code, expected in self.guards]

    def contains_parameters(self, value):
        if isinstance(value, _InitializationExpression):
            return True
        if id(value) in self._contains_parameters:
            return self._contains_parameters[id(value)]
        self._contains_parameters[id(value)] = False
        if isinstance(value, dict):
            values = value.values()
        elif isinstance(value, (tuple, list)):
            values = value
        elif inspect.isroutine(value) or isinstance(value, type):
            return False
        else:
            values = (getattr(value, field) for field in _fields(value))
        result = any(self.contains_parameters(item) for item in values)
        self._contains_parameters[id(value)] = result
        return result

    def pair(self, original, initialized, seen=None):
        """Find initialized parameter expressions without replacing live state or wiring."""
        if seen is None:
            self._contains_parameters.clear()
            seen = set()
        if isinstance(initialized, _InitializationExpression):
            return self.cached(initialized)
        if not self.contains_parameters(initialized):
            return original
        if isinstance(original, (tuple, list)) and isinstance(initialized, type(original)):
            if len(original) != len(initialized):
                raise TypeError("parameter initialization changes a collection's layout")
            return type(original)(self.pair(left, right, seen)
                                  for left, right in zip(original, initialized))
        if isinstance(original, dict) and isinstance(initialized, dict):
            if original.keys() != initialized.keys():
                raise TypeError("parameter initialization changes an object's fields")
            return {key: self.pair(value, initialized[key], seen)
                    for key, value in original.items()}
        if type(original) is not type(initialized) or inspect.isroutine(original):
            return original
        if not hasattr(original, "__dict__") and not hasattr(type(original), "__slots__"):
            return original
        if id(original) in seen:
            return original
        seen.add(id(original))
        changes = self.object_updates.setdefault(id(original), {})
        constructor_parameters = inspect.signature(type(original).__init__).parameters
        for name in _fields(original):
            if not hasattr(initialized, name):
                continue
            left, right = getattr(original, name), getattr(initialized, name)
            if not self.contains_parameters(right):
                continue
            if (isinstance(right, _InitializationExpression) and name in constructor_parameters
                    and isinstance(left, numbers.Number) and left != right.default):
                if is_dataclass(original) and not original.__dataclass_params__.frozen:
                    # Mutable value records may now hold an initial observation, not the
                    # constructor's placeholder. Preserve that live value.
                    continue
                raise TypeError(
                    f"{type(original).__name__}.{name}: a factory changed this initialized "
                    "parameter outside its recorded construction; it cannot be silently replaced"
                )
            if inspect.isroutine(left) or isinstance(left, type):
                continue
            try:
                changes[name] = self.pair(left, right, seen)
            except TypeError as exc:
                raise TypeError(f"{type(original).__name__}.{name}: {exc}") from exc
        return original

    def object(self, original):
        """A shallow parameterized view; state records and model identities stay unchanged."""
        if isinstance(original, _InitializationExpression):
            return self.cached(original)
        if id(original) in self.copies:
            return self.copies[id(original)]
        if isinstance(original, tuple):
            return tuple(self.object(value) for value in original)
        if isinstance(original, list):
            return [self.object(value) for value in original]
        if isinstance(original, dict):
            return {key: self.object(value) for key, value in original.items()}
        if id(original) not in self.object_updates:
            return original
        if id(original) in self.object_copies:
            return self.object_copies[id(original)]
        clone = _ControlGenerator._copy_block(original)
        self.object_copies[id(original)] = clone
        updates = self.object_updates[id(original)]
        for name in _fields(original):
            if name not in {"state", "inp", "out"}:
                value = updates.get(name, getattr(original, name))
                object.__setattr__(clone, name, self.object(value))
        for name in _fields(original):
            value = getattr(original, name)
            if getattr(value, "__self__", None) is original:
                object.__setattr__(clone, name, value.__func__.__get__(clone, type(clone)))
        return clone

    def initial_value(self, record, field):
        return self.object_updates.get(id(record), {}).get(field, getattr(record, field))

    def copy(self, value, path=""):
        if path in self.by_path and not isinstance(self.by_path[path].default, bool):
            item = self.by_path[path]
            return _InitializationExpression.make(
                _Expression("parameters." + item.member, "real"), item.default, self)
        if id(value) in self.copies:
            return self.copies[id(value)]
        if is_dataclass(value):
            clone = copy.copy(value)
            self.copies[id(value)] = clone
            if hasattr(clone, "__dict__"):
                for parent in type(clone).__mro__:
                    for key, descriptor in vars(parent).items():
                        if isinstance(descriptor, cached_property):
                            vars(clone).pop(key, None)
            for item in fields(value):
                if item.init:
                    child = path if item.name == "_unit_fields" else (
                        f"{path}.{item.name}" if path else item.name)
                    object.__setattr__(clone, item.name, self.copy(getattr(value, item.name), child))
            if isinstance(clone, BaseValues):
                clone.__post_init__()
            elif isinstance(clone, (UnitParams, SourceParams)):
                defaults = sorted(value.derived_defaults, key=lambda key: (key.count("."), key))
                base = self.copy(value.base if isinstance(value, UnitParams) else self.base)
                # Event schedules are structural metadata. Their parameter writes are traced
                # separately at root paths; copying snapshots here would bind them as unit fields.
                args = (base, value.events) if isinstance(value, UnitParams) else (base,)
                for derived in defaults:
                    self.put(clone, derived, None)
                clone = clone.resolved(*args)
                for index, derived in enumerate(defaults):
                    key = f"{path}.{derived}"
                    item = self.by_path.get(key)
                    if item is not None:
                        fallback = self.at(clone, derived)
                        code = (f"(parameters.given_{item.member} ? parameters.{item.member} : "
                                f"({_Expression.of(fallback).code}))")
                        self.put(clone, derived, _InitializationExpression.make(
                            _Expression(code, "real"), item.default, self))
                    for remaining in defaults[index + 1:]:
                        self.put(clone, remaining, None)
                    clone = clone.resolved(*args)
                self.copies[id(value)] = clone
            elif hasattr(clone, "__post_init__"):
                clone.__post_init__()
            return clone
        if isinstance(value, dict):
            clone = {key: self.copy(item, f"{path}.{key}" if path else key)
                     for key, item in value.items()}
            self.copies[id(value)] = clone
            return clone
        if isinstance(value, tuple):
            return tuple(self.copy(item) for item in value)
        if isinstance(value, list):
            return [self.copy(item) for item in value]
        return value

    def value(self, path: str, value: Any) -> str:
        item = self.by_path.get(path)
        if item is None:
            return _number(value)
        if isinstance(item.default, bool):
            return "parameters." + item.member
        stored = self.at(self.initial_p, path)
        return stored.code if isinstance(stored, _Expression) else _number(stored)

    def declarations(self) -> str:
        members = "\n    ".join(
            f"{'long long' if isinstance(item.default, int) else 'double'} "
            f"{item.member} = {_number(item.default)};"
            + (f"\n    bool given_{item.member} = false;"
               if item.derived or isinstance(item.default, bool) else "")
            for item in self.selected
        )
        return f"struct RuntimeParameters {{\n    {members}\n}};" if members else "struct RuntimeParameters {};"

    def setters(self) -> str:
        cases: list[str] = []
        for item in self.selected:
            condition = self.invalid_value(item, "value")
            validation = (
                f" if ({condition}) throw std::runtime_error(\"invalid value for {item.path}\");"
                if condition else ""
            )
            cases.append(
                f"if (path == {_cpp_string(item.path)}) {{ const double value = parse_number(text);"
                f"{validation} parameters.{item.member} = value;"
                + (f" parameters.given_{item.member} = true;"
                   if item.derived or isinstance(item.default, bool) else "")
                + " return true; }"
            )
        return "\n    ".join(cases)

    @staticmethod
    def invalid_value(item, value):
        checks = [f"!std::isfinite({value})" if math.isfinite(item.default)
                  else f"std::isnan({value})"]
        if isinstance(item.default, int):
            checks.append(f"std::trunc({value}) != {value}")
        if isinstance(item.default, bool):
            checks.append(f"({value} != 0 && {value} != 1)")
        return " || ".join(checks)

    def listing(self) -> str:
        lines = []
        for item in self.selected:
            value = ("parameters." + item.member if isinstance(item.default, bool)
                     else _Expression.of(self.at(self.p, item.path)).code)
            lines.append(f"std::cout << {_cpp_string(item.path + '  value=')} << {value} << '\\n';")
        return "\n    ".join(lines)

    def resolved_template(self) -> tuple[str, list[tuple[str, str]]]:
        if not self.selected:
            return dumps(self.p), []
        replacements: list[tuple[str, str]] = []

        def written(value):
            if isinstance(value, _InitializationExpression):
                marker = f"__PESLITE_RUNTIME_PARAMETER_{len(replacements)}__"
                replacements.append((marker, f"yaml_number({value.code})"))
                return marker
            if isinstance(value, dict):
                return {key: written(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [written(item) for item in value]
            return value

        tree = written(_written_out(self.p))
        for item in self.selected:
            if isinstance(item.default, bool):
                marker = f"__PESLITE_RUNTIME_PARAMETER_{len(replacements)}__"
                original = self.at(self.p, item.path)
                fallback = "null" if original is None else str(int(original))
                replacements.append((marker, f"parameters.given_{item.member} ? "
                                     f"yaml_number(parameters.{item.member}) : {_cpp_string(fallback)}"))
                self.put(tree, item.path, marker)
        try:
            import yaml  # type: ignore
            body = yaml.safe_dump(tree, sort_keys=False)
        except ImportError:  # pragma: no cover
            body = json.dumps(tree, indent=2) + "\n"
            for marker, _expression in replacements:
                body = body.replace(json.dumps(marker), marker)
        return body, replacements


class _ModelGenerator:
    """Turn Model's fixed state and connection plans into straight-line C++."""

    def __init__(self, simulation: Any, runtime: _RuntimeParameters) -> None:
        runtime = runtime if runtime is not None else _RuntimeParameters(simulation.p, ())
        self.simulation = simulation
        self.p = runtime.p
        self.runtime = runtime
        self.system = simulation.system
        self.model = self.system.model
        self.subsystem_name = {id(sub): name for name, sub in zip(self.model.names, self.model.subsystems)}
        self.prefix = {id(sub): f"s{index}" for index, sub in enumerate(self.model.subsystems)}
        self.record_fields: dict[int, dict[str, str]] = {}
        self.record_types: dict[tuple[int, str], str] = {}
        self._unit_control_inputs: dict[int, dict[str, _Expression]] = {}
        self._unit_measurements: dict[int, dict[str, _Expression]] = {}
        self._solver_observations: dict[tuple[int, str], tuple[str, Any, str]] = {}
        self._algebraic_index = {id(loop): index
                                 for index, loop in enumerate(self.model.algebraic_loops)}
        self._algebraic_plans = {}
        self.diagnostics = []
        self.state_index: dict[tuple[int, str], tuple[int, bool]] = {
            (id(state), name): (index, is_complex)
            for state, name, index, is_complex in self.model._state_ops
        }
        self.zoh_fields: dict[tuple[int, str], str] = {}
        self.element_parameters: dict[str, tuple[str, Any, str]] = {}
        self._parameter_attributes: dict[int, dict[str, _Expression]] = {}
        self.equation_bindings = {}
        self.equation_members = {}
        self.equation_references = {}
        self._equation_method_plans = {}
        for sub in self.model.subsystems:
            read = getattr(sub, "equation_state_bindings", None)
            if read is None:
                continue
            bindings = self.equation_bindings[id(sub)] = read()
            for owner, attribute in bindings.values():
                key = id(owner), attribute
                value = getattr(owner, attribute)
                self.equation_members.setdefault(key, (
                    f"{self.prefix[id(sub)]}_memory_{len(self.equation_members)}",
                    "bool" if isinstance(value, bool) else
                    "complex" if isinstance(value, complex) else "real", owner, attribute,
                ))
        self.storage_coefficients = {
            (id(owner.state), storage.state): storage.scale * storage.value
            for owner in self.model.subsystems
            for storage in getattr(runtime.object(owner), "storage", ())
        }
        changed_paths = {path for change in self.p.changes for path in change.paths}
        for name, element in self.system.named_elements.items():
            for field, (subsystem, attribute) in getattr(
                    element, "parameter_bindings", lambda: {})().items():
                path = f"elements.{name}.{field}"
                if path in changed_paths:
                    member = "element_" + _identifier(name + "_" + field)
                    self.element_parameters[path] = (member, subsystem, attribute)
                    self._parameter_attributes.setdefault(id(subsystem), {})[attribute] = (
                        _Expression(member, "real")
                    )
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
            for name, unit in self.system.units.items() if unit.continuous and unit.ctrl is not None
        }
        # Retain exactly the record fields read at the start of the next iterative solve.
        self._algebraic_fields = {}
        for loop in self.model.algebraic_loops:
            if self._algebraic_plan(loop) is not None:
                continue
            for record, name in self._algebraic_seeds(loop):
                self._algebraic_fields[id(record), name] = record
        for key in self.state_index.keys() | self.zoh_fields.keys():
            self._algebraic_fields.pop(key, None)

    def breaker_lines(self, branch: RLBranch, open_expression: str) -> list[str]:
        """Render the state transition implemented by ``RLBranch`` breaker methods."""
        flag = self._equation_attributes(branch)["breaker_open"].code
        index, is_complex = self.state_index[id(branch.state), "i"]
        zero = [f"y[{index}] = 0.0;"]
        if is_complex:
            zero.append(f"y[{index + 1}] = 0.0;")
        return [
            f"{flag} = {open_expression};",
            f"if ({flag}) {{",
            *["    " + line for line in zero],
            "}",
        ]

    def source_ramp_members(self, link: DCLink) -> tuple[str, str, str] | None:
        """Physical source ramp state and law, independent of restored controller startup state."""
        if link.source is None or not hasattr(link.source, "i_nom"):
            return None
        tag = self.prefix[id(link)]
        return "source_start_" + tag, "source_duration_" + tag, "source_ramp_" + tag

    def unit_control_inputs(self, unit: Any) -> dict[str, _Expression]:
        """Trace one unit's public control boundary once per export."""
        key = id(unit)
        if key not in self._unit_control_inputs:
            self._unit_control_inputs[key] = self.trace_element_values(unit, unit.control_inputs)
        return self._unit_control_inputs[key]

    def unit_measurements(self, unit: Any) -> dict[str, _Expression]:
        """Trace the physical channels read by one unit's ADC once per export."""
        key = id(unit)
        if key not in self._unit_measurements:
            self._unit_measurements[key] = self.trace_element_values(unit, unit.ports.read)
        return self._unit_measurements[key]

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
                        "bool" if isinstance(value, bool) else
                        "complex" if isinstance(value, complex) else "real"
                    )

    def value(self, record: Any, field: str) -> str:
        try:
            return self.record_fields[id(record)][field]
        except KeyError as exc:  # an exporter bug, not a configuration restriction
            raise RuntimeError(f"C++ export: unregistered record field {field!r}") from exc

    def state_value(self, key: tuple[int, str]) -> str:
        """Read one declared ODE state in solver scope, outside output evaluation."""
        index, is_complex = self.state_index[key]
        return f"Complex{{y[{index}], y[{index + 1}]}}" if is_complex else f"y[{index}]"

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
        if isinstance(when_false, _TraceFailure) and isinstance(when_true, _TraceFailure):
            return when_false
        def failure_like(value, failure):
            if isinstance(value, dict):
                return {key: failure_like(item, failure) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(failure_like(item, failure) for item in value)
            kind = _Expression.of(value).kind
            return _Expression(f"([]() -> {_CPP_TYPES[kind]} {{ throw std::runtime_error("
                               f"{_cpp_string(str(failure.error))}); }}())", kind)
        if isinstance(when_false, _TraceFailure):
            when_false = failure_like(when_true, when_false)
        if isinstance(when_true, _TraceFailure):
            when_true = failure_like(when_false, when_true)
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
        if left.kind == right.kind == "bool":
            if (left.code, right.code) == ("false", "true"):
                return condition
            if (left.code, right.code) == ("true", "false"):
                return _Expression(f"!({condition.code})", "bool")
        kind = "complex" if "complex" in (left.kind, right.kind) else (
            "bool" if left.kind == right.kind == "bool" else "real"
        )
        if kind == "complex":
            if left.kind != "complex":
                left = _Expression(f"Complex{{{left.code}, 0.0}}", "complex")
            if right.kind != "complex":
                right = _Expression(f"Complex{{{right.code}, 0.0}}", "complex")
        code = f"(({condition.code}) ? ({right.code}) : ({left.code}))"
        trace = _CONDITIONAL_DEPENDENCIES.get()
        if trace is not None and trace[0].parameter_dependencies((condition.code,))[condition.code]:
            common = (set(trace[0].expression_symbols(left.code))
                      & set(trace[0].expression_symbols(right.code)))
            # The condition itself and reads shared by both branches remain unconditional.
            replacement = " ".join(sorted(common)) + " " + condition.code
            trace[1][code] = ((condition.code, right.code, replacement),
                             (f"!({condition.code})", left.code, replacement))
        return _Expression(code, kind)

    def _trace(self, run: Any, decisions: tuple[bool, ...] = (), *, budget=None) -> Any:
        """Trace pure scalar arithmetic, exploring state-dependent Python branches."""
        budget = [0] if budget is None else budget
        budget[0] += 1
        if budget[0] > _TRACE_WORK_BUDGET:
            raise TypeError("equation tracing exceeds its branch-work budget")
        token = _TRACE_CONTROL.set(_TraceControl(decisions))
        branch: _TraceBranch | None = None
        try:
            return self._normalise_trace(run())
        except _TraceBranch as exc:
            branch = exc
        except ValueError as exc:
            if not decisions:
                raise
            return _TraceFailure(exc)
        finally:
            _TRACE_CONTROL.reset(token)
        assert branch is not None
        when_false = self._trace(run, (*decisions, False), budget=budget)
        when_true = self._trace(run, (*decisions, True), budget=budget)
        result = self._merge_trace(branch.condition, when_false, when_true)
        if not decisions and isinstance(result, _TraceFailure):
            raise result.error
        return result

    def _equation_method_plan(self, owner, bindings):
        """Use ordinary annotated method boundaries to keep independent branches separate."""
        if id(owner) not in self._equation_method_plans:
            objects = {id(owner): owner, **{id(part): part for part, _attribute in bindings.values()}}
            pending = ['rhs', 'set_outputs', 'set_ac_voltage', 'set_dc_current', 'get_state',
                       'commutation_margin', 'commutate', 'next_commutation_time', 'project_state',
                       'diagnostics']
            visited, plan = set(), []
            while pending:
                name = pending.pop()
                if name in visited:
                    continue
                visited.add(name)
                for part in objects.values():
                    method = getattr(part, name, None)
                    if not inspect.ismethod(method):
                        continue
                    function = method.__func__
                    codes = [function.__code__]
                    while codes:
                        code = codes.pop()
                        pending.extend(code.co_names)
                        codes.extend(value for value in code.co_consts if inspect.iscode(value))
                    annotation = inspect.signature(function).return_annotation
                    returns_none = annotation in (None, type(None), 'None')
                    scalar = annotation in (float, int, bool, complex, tuple, 'float', 'int',
                                            'bool', 'complex', 'tuple')
                    if (returns_none or scalar) and hasattr(part, '__dict__'):
                        plan.append((id(part), name, function, returns_none))
            self._equation_method_plans[id(owner)] = tuple(plan)
        return self._equation_method_plans[id(owner)]

    def _equation_clone(self, owner: Any,
                        attributes: dict[str, _Expression] | None = None, *, bound=None) -> Any:
        clone = copy.copy(self.runtime.object(owner))
        records = {}
        for name, value in (vars(owner).items() if hasattr(owner, "__dict__") else ()):
            if getattr(value, "__self__", None) is owner:
                setattr(clone, name, self._rebind(value, clone))
        for role in ("state", "inp", "out"):
            original = getattr(owner, role)
            record = copy.copy(original)
            for field in _fields(original):
                setattr(record, field, _Expression(self.value(original, field),
                                                   self.record_types[id(original), field]))
            setattr(clone, role, record)
            records[id(original)] = record
        # Composite parts may alias a parent's state record (for example a DC capacitor).
        # Preserve that alias in the equation copy without cloning the connected model graph.
        for name, value in (list(vars(clone).items()) if hasattr(clone, "__dict__") else ()):
            record = records.get(id(getattr(value, "state", None)))
            if record is not None:
                part = copy.copy(value)
                part.state = record
                for role in ("inp", "out"):
                    inner = getattr(value, role, None)
                    if inner is not None:
                        setattr(part, role, records.get(id(inner), copy.copy(inner)))
                setattr(clone, name, part)
        bindings = self.equation_bindings.get(id(owner), {})
        copies = {id(owner): clone}
        for part, _attribute in bindings.values():
            if id(part) not in copies:
                copies[id(part)] = copy.copy(self.runtime.object(part))
        references = {**self.equation_references, **copies, **records}

        def rebind(original, value):
            if id(original) in references:
                return references[id(original)]
            if (isinstance(original, (tuple, list)) and isinstance(value, type(original))
                    and len(original) == len(value)):
                return type(value)(rebind(left, right) for left, right in zip(original, value))
            if (isinstance(original, dict) and isinstance(value, dict)
                    and original.keys() == value.keys()):
                return {key: rebind(left, value[key]) for key, left in original.items()}
            method_owner = getattr(original, "__self__", None)
            if id(method_owner) in references:
                return self._rebind(original, references[id(method_owner)])
            return value

        for original in {id(owner): owner, **{id(part): part for part, _attribute in bindings.values()}}.values():
            target = copies[id(original)]
            for name in _fields(original):
                value = getattr(target, name)
                setattr(target, name, rebind(getattr(original, name), value))
        for (part_id, attribute), (member, kind, _part, _attribute) in self.equation_members.items():
            if part_id in copies:
                setattr(copies[part_id], attribute, _Expression(member, kind))
        if bindings:
            tracked = [(copies[part_id], attribute)
                       for part_id, attribute in self.equation_members if part_id in copies]
            tracked += [(record, field) for record in records.values() for field in _fields(record)]
            cache_objects = {id(value) for value in (*copies.values(), *records.values(), *references.values())}
            for part_id, name, method, returns_none in self._equation_method_plan(owner, bindings):
                target = copies[part_id]
                method_cache = {}
                def scoped(receiver, *args, _method=method, _void=returns_none,
                           _tracked=tuple(tracked), _cache=method_cache, **kwargs):
                    before = [(part, attribute, getattr(part, attribute)) for part, attribute in _tracked]
                    snapshot = copy.copy(receiver)
                    mutable = {attribute for part, attribute in _tracked if part is receiver}
                    def restore_transients():
                        for field in _fields(snapshot):
                            if field not in mutable:
                                setattr(receiver, field, getattr(snapshot, field))
                    def run():
                        restore_transients()
                        for part, attribute, value in before:
                            setattr(part, attribute, value)
                        result = _method(receiver, *args, **kwargs)
                        return ((0.0 if _void else result),
                                tuple(getattr(part, attribute) for part, attribute in _tracked))
                    def key(value):
                        if isinstance(value, (_Expression, numbers.Number)):
                            expression = _Expression.of(value)
                            return expression.kind, expression.code
                        if isinstance(value, (tuple, list)):
                            return tuple(key(item) for item in value)
                        if isinstance(value, dict):
                            return tuple((key(name), key(item)) for name, item in value.items())
                        if value is None or isinstance(value, (str, bytes)):
                            return type(value), value
                        if getattr(value, 'shape', None) == ():
                            return key(value.item())
                        if hasattr(value, 'shape') and hasattr(value, 'flat'):
                            return value.shape, tuple(key(item) for item in value.flat)
                        if id(value) in cache_objects:
                            return id(value)
                        raise TypeError('opaque equation argument')
                    try:
                        cache_key = (key(receiver), tuple(key(value) for _part, _attribute, value in before),
                                     tuple(map(key, args)), tuple((name, key(value)) for name, value in kwargs.items()))
                    except TypeError:
                        # Untracked mutable arguments cannot prove a reusable equation result.
                        cached = self._trace(run)
                    else:
                        cached = _cache.get(cache_key)
                        if cached is None:
                            cached = _cache[cache_key] = self._trace(run)
                    result, values = cached
                    restore_transients()
                    for (part, attribute), value in zip(_tracked, values):
                        setattr(part, attribute, value)
                    return None if _void else result
                setattr(target, name, scoped.__get__(target, type(target)))
        if bound is not None:
            bound.update(copies)
        for name, value in {**self._parameter_attributes.get(id(owner), {}),
                            **(attributes or {})}.items():
            setattr(clone, name, value)
        return clone

    @staticmethod
    def _rebind(method: Any, owner: Any) -> Any:
        function = getattr(method, "__func__", None)
        if function is None:
            raise TypeError("equation callable is not a bound Python method")
        return function.__get__(owner, type(owner))

    @contextmanager
    def _shared_terms(self):
        prefix = f"equation_{self.runtime._equation_counter}_"
        self.runtime._equation_counter += 1
        terms = {}
        token = _SCALAR_TERMS.set((prefix, terms, self.runtime))
        try:
            yield prefix, terms
        finally:
            _SCALAR_TERMS.reset(token)

    @staticmethod
    def _term_declarations(prefix, terms, lines):
        needed = set(re.findall(rf"\b{prefix}\d+\b", "\n".join(lines)))
        declarations = []
        for code, name in reversed(tuple(terms.items())):
            if name in needed:
                needed.update(re.findall(rf"\b{prefix}\d+\b", code))
                declarations.append(f"const auto {name} = {code};")
        return [*reversed(declarations), *lines]

    def trace_outputs(self, method: Any,
                      attributes: dict[str, _Expression] | None = None) -> list[str]:
        owner = method.__self__
        fields = _fields(owner.out)

        def run():
            bound = {}
            clone = self._equation_clone(owner, attributes, bound=bound)
            self._rebind(method, clone)(_Expression("t", "real"))
            return (tuple(getattr(clone.out, field) for field in fields),
                    tuple(getattr(bound[id(part)], attribute)
                          for part, attribute in self.equation_bindings.get(id(owner), {}).values()))

        # Share total scalar operations. The same dependency DAG resolves parameter aliases;
        # conditional parameter terms, guarded divisions and partial functions retain scope.
        with self._shared_terms() as (prefix, terms):
            values, memory = self._trace(run)
        lines: list[str] = []
        for field, value in zip(fields, values):
            original = self.value(owner.out, field)
            if value.code != original:
                lines.append(f"{original} = {value.code};")
        for (part, attribute), value in zip(self.equation_bindings.get(id(owner), {}).values(), memory):
            target = self.equation_members[id(part), attribute][0]
            if value.code != target:
                lines.append(f"{target} = {value.code};")
        if not lines and fields:
            raise TypeError("output equation did not assign an output field")
        return self._term_declarations(prefix, terms, lines)

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

        with self._shared_terms() as (prefix, terms):
            value = self._trace(run)
        lines = self._term_declarations(prefix, terms, [f"return {value.code};"])
        return value.code if len(lines) == 1 else f"([&]() {{ {' '.join(lines)} }}())"

    def trace_solver_method(self, sub: Any, name: str,
                            attributes: dict[str, _Expression] | None = None) -> str:
        """Trace a scalar capability for solver scope, where states are read from ``y``."""
        return self.solver_expression(self.trace_scalar_method(sub, name, attributes))

    def solver_expression(self, code):
        """Translate an equation expression to persistent solver-scope values."""
        for record_id, field in self.state_index:
            local = self.record_fields[record_id][field]
            value = self.state_value((record_id, field))
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

    def trace_transition(self, sub, name, *, timed=True):
        """Trace the ordinary state-changing callback, including ODE-state resets."""
        bindings = self.equation_bindings.get(id(sub), {})

        def run():
            bound = {}
            clone = self._equation_clone(sub, bound=bound)
            self._rebind(getattr(sub, name), clone)(*([_Expression("t", "real")] if timed else []))
            return (*[getattr(bound[id(part)], attribute) for part, attribute in bindings.values()],
                    *[getattr(clone.state, field) for field in sub.state_names])

        with self._shared_terms() as (prefix, terms):
            values = self._trace(run)
        assignments = [(self.equation_members[id(part), attribute][0], value)
                       for (part, attribute), value in zip(bindings.values(), values)]
        offset = len(bindings)
        for field, value in zip(sub.state_names, values[offset:]):
            key = id(sub.state), field
            index, complex_value = self.state_index[key]
            if complex_value:
                assignments.extend(((f"y[{index}]", value.real), (f"y[{index + 1}]", value.imag)))
            else:
                assignments.append((f"y[{index}]", value))
        assignments = [(target, value) for target, value in assignments
                       if target != self.solver_expression(value.code)]
        lines = _assign_values(f"transition_{self.prefix[id(sub)]}", assignments)
        return [self.solver_expression(line) for line in
                self._term_declarations(prefix, terms, lines)]

    def named_equation_state(self, part):
        def run():
            clone = self._equation_clone(part)
            return clone.get_state()
        return self._trace(run)

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
                            setattr(record, field, _Expression(
                                self.value(record, field), self.record_types[id(record), field]))
                return read()
            finally:
                for record, field, value in reversed(saved):
                    setattr(record, field, value)

        return self._trace(run)

    @staticmethod
    def _algebraic_seeds(loop):
        """The record values carried between successive iterative solves."""
        for dst, field, src, source_field, fanin in loop.tears:
            yield from (((dst, field),) if len(loop.tears) == 1 else
                        ((src, source_field),) if fanin is None else
                        tuple((record, name) for record, name, _gain in fanin))

    def declarations(self, persistent_fields=None) -> list[str]:
        persistent_fields = self._algebraic_fields if persistent_fields is None else persistent_fields
        lines: list[str] = []
        declared: set[str] = set()
        for sub in self.model.subsystems:
            for record in (sub.state, sub.inp, sub.out):
                for field in _fields(record):
                    name = self.value(record, field)
                    if name in declared:
                        continue
                    declared.add(name)
                    if (id(record), field) in persistent_fields:
                        continue
                    state = self.state_index.get((id(record), field))
                    kind = _CPP_TYPES[self.record_types[id(record), field]]
                    if state is not None:
                        initial = self.state_value((id(record), field))
                    elif (id(record), field) in self.zoh_fields:
                        initial = self.zoh_fields[id(record), field]
                    else:
                        initial = _literal(getattr(record, field))
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

    def _source_values(self, source: ACVoltageSource) -> tuple[str, str]:
        entry = next(((name, item) for name, item in self.system.sources.items()
                      if item.emf is source), None)
        wrapper = entry[1] if entry is not None else None
        if wrapper is None:
            return _number(source.e_peak), "0.0"
        source_name = entry[0]
        scenario = _source_scenario(self.p, f"sources.{source_name}",
                                    lambda q: q.sources[source_name])

        def piecewise(starts, expressions):
            result = expressions[0]
            for start, expression in zip(starts[1:], expressions[1:]):
                result = f"(event_time >= {_number(start)} ? {expression} : {result})"
            return result

        magnitudes = [_number(value) for value in scenario._v]
        magnitude = piecewise(scenario._v_from, magnitudes)
        phases = []
        for start, (frequency, angle, phase) in zip(scenario._angle_from, scenario._angle):
            frequency, angle = _number(frequency), _number(angle)
            phases.append(f"({_number(phase)} + {angle} + 2.0 * pi * ({frequency} - "
                          f"{_number(scenario.f0)}) * (t - {_number(max(start, 0.0))}))")
        return magnitude, piecewise(scenario._angle_from, phases)

    def _source_equations(self, source, time="t"):
        """The native source adapter shared by execution and initial-state presets."""
        magnitude, phi = self._source_values(source)
        phi = re.sub(r"\bt\b", f"({time})", phi)
        magnitude = re.sub(r"\bt\b", f"({time})", magnitude)
        theta = f"({_number(self.runtime.object(source).w0)} * ({time}) + ({phi}))"
        voltage = (f"({magnitude}) * std::cos({theta})" if source.phases == 1 else
                   f"({magnitude}) * Complex{{std::cos({theta}), std::sin({theta})}}")
        return {"phi": _Expression(phi, "real"), "theta": _Expression(theta, "real"),
                "e_g": _Expression(voltage, "real" if source.phases == 1 else "complex")}

    def _output(self, method: Any) -> list[str]:
        owner = method.__self__
        out = owner.out
        hook = getattr(owner, "cpp_outputs", None)
        if hook is not None:
            return list(hook(self))
        if (isinstance(owner, ACVoltageSource)
                and ACVoltageSource.uses_standard_outputs(method)):
            return [f"{self.value(out, field)} = {value.code};"
                    for field, value in self._source_equations(owner).items()]
        generator = self.continuous_units.get(id(owner))
        if generator is not None:
            return generator.output_lines()
        label = self.subsystem_name.get(id(owner), type(owner).__name__)
        try:
            return self.trace_outputs(method, self._equation_attributes(owner))
        except Exception as exc:
            raise TypeError(
                f"C++ export: cannot lower the output equation of subsystem {label!r} "
                f"({type(owner).__name__}): {exc}"
            ) from exc

    def _equation_attributes(self, owner: Any) -> dict:
        """Bind dynamic lifecycle values without restating component equations."""
        attributes = {}
        enabled = self.enabled_by.get(id(owner))
        if enabled is not None:
            attributes["enabled"] = _Expression(enabled, "bool")
        if isinstance(owner, RLBranch):
            attributes["breaker_open"] = _Expression(f"breaker_open_{self.prefix[id(owner)]}", "bool")
        if isinstance(owner, DCLink):
            attributes["tripped"] = _Expression(f"dclink_tripped_{self.prefix[id(owner)]}", "bool")
            source = self.runtime.object(owner.source)
            ramp = self.source_ramp_members(owner)
            if ramp is not None:
                source = copy.copy(source)
                source.ramp = lambda t: _Expression(f"{ramp[2]}({t.code})", "real")
                attributes["source"] = source
        return attributes

    def _operation(self, operation: tuple, indent: str = "",
                   direct_guards: dict[int, str] | None = None) -> list[str]:
        kind, item = operation
        if kind == "out":
            return [indent + line for line in self._output(item)]
        if kind == "copy":
            return [indent + f"{self.value(item[0], item[1])} = {self._copy_expression(item)};"]
        if kind == "algebraic":
            return self._algebraic(item, indent, direct_guards)
        raise RuntimeError(f"C++ export: unknown model operation {kind!r}")

    def _algebraic(self, loop: Any, indent: str,
                   direct_guards: dict[int, str] | None = None) -> list[str]:
        straight = self._algebraic_plan(loop)
        if straight is not None:
            return [indent + line for line in straight]
        if self.runtime.selected:
            coefficients = {}
            token = _ZERO_COEFFICIENTS.set(coefficients)
            try:
                straight = self._algebraic_plan(loop)
            finally:
                _ZERO_COEFFICIENTS.reset(token)
            if straight is not None and coefficients:
                condition = " && ".join(f"(({code}) == 0.0)" for code in coefficients)
                guard = self.runtime.cached(_InitializationExpression.make(
                    _Expression(condition, "bool"), True, self.runtime)).code
                if direct_guards is not None:
                    direct_guards[id(loop)] = guard
                    return [indent + line for line in straight]
                return [indent + f"if ({guard}) {{",
                        *[indent + "    " + line for line in straight],
                        indent + "} else {", *self._algebraic_solve(loop, indent + "    "),
                        indent + "}"]
        return self._algebraic_solve(loop, indent)

    def _algebraic_solve(self, loop: Any, indent: str) -> list[str]:
        """The ordinary iterative path for genuine feedback or unrecognized equation code."""
        label = _cpp_string("algebraic loop did not converge: " + ", ".join(loop.labels))
        if len(loop.tears) == 1:
            tear, = loop.tears
            index = self._algebraic_index[id(loop)]
            current = self.value(tear[0], tear[1])
            lines = [indent + f"solve_algebraic([&](auto x) PESLITE_INLINE_LAMBDA {{",
                     indent + f"    {current} = x;"]
            for operation in loop.operations:
                lines.extend(self._operation(operation, indent + "    "))
            lines += [indent + f"    return {self._copy_expression(tear)};",
                      indent + f"}}, {current}, algebraic_inverse_{index}, "
                      f"algebraic_has_inverse_{index}, {loop.max_sweeps}, "
                      f"{_number(loop.rtol)}, {_number(loop.atol)}, "
                      f"{_number(loop.step_limit)}, {label});"]
            return lines
        lines = [indent + "{", indent + "bool algebraic_converged = false;",
                 indent + f"for (int algebraic_sweep = 0; algebraic_sweep < {loop.max_sweeps}; ++algebraic_sweep) {{"]
        for tear in loop.tears:
            lines.append(indent + "    " + f"{self.value(tear[0], tear[1])} = {self._copy_expression(tear)};")
        for operation in loop.operations:
            lines.extend(self._operation(operation, indent + "    "))
        checks = []
        for index, tear in enumerate(loop.tears):
            current = self.value(tear[0], tear[1])
            target = self._copy_expression(tear)
            delta = f"std::abs({current} - ({target}))"
            scale = f"std::max({{1.0, std::abs({current}), std::abs({target})}})"
            checks.append(f"({delta} <= {_number(loop.atol)} + {_number(loop.rtol)} * {scale})")
        lines += [indent + "    if (" + " && ".join(checks or ["true"])
                  + ") { algebraic_converged = true; break; }",
                  indent + "}",
                  indent + f"if (!algebraic_converged) throw std::runtime_error({label});",
                  indent + "}"]
        return lines

    def _algebraic_plan(self, loop: Any) -> tuple[str, ...] | None:
        """Remove only feedback dependencies eliminated by parameter specialization.

        The runtime model conservatively retains input dependencies even when a coefficient is
        zero. Traced scalar assignments can then form a DAG instead of a genuine algebraic loop.
        A standard topological sort emits that DAG once; other code keeps the shared solver.
        """
        coefficients = _ZERO_COEFFICIENTS.get()
        key = id(loop), coefficients is not None
        if key in self._algebraic_plans:
            plan, required_zeroes = self._algebraic_plans[key]
            if coefficients is not None:
                coefficients.update(required_zeroes)
            return plan
        self._algebraic_plans[key] = None, {}
        diagnose = bool(self.runtime.selected) and coefficients is None
        plan = self._assignment_plan(loop, diagnose)
        self._algebraic_plans[key] = plan, dict(coefficients or {})
        return plan

    def _assignment_plan(self, loop: Any, diagnose: bool) -> tuple[str, ...] | None:
        """Order the same scalar assignments for diagnosis and compact code emission."""
        lines = []
        for kind, item in loop.operations:
            if kind == "algebraic":
                return None
            if kind == "out":
                owner = item.__self__
                if getattr(owner, "cpp_outputs", None) is not None:
                    return None
            # Observe the existing trace once, only for retained-parameter candidate groups.
            conditional = {} if diagnose else None
            token = _CONDITIONAL_DEPENDENCIES.set(
                (self.runtime, conditional) if conditional is not None else None)
            try:
                emitted = self._operation((kind, item))
            finally:
                _CONDITIONAL_DEPENDENCIES.reset(token)
            lines.extend((line, conditional) for line in emitted)
        lines.extend((f"{self.value(tear[0], tear[1])} = {self._copy_expression(tear)};", None)
                     for tear in loop.tears)
        assignments = {}
        conditional_by_target = {}
        for line, conditional in lines:
            match = re.fullmatch(r"(?:const (?:auto|double|Complex) )?(\w+) = (.+);", line)
            if match is None or match[1] in assignments:
                return None
            assignments[match[1]] = (line, set(re.findall(r"\b\w+\b", match[2])))
            if conditional:
                conditional_by_target[match[1]] = match[2], conditional
        targets = set(assignments)
        sorter = TopologicalSorter({
            name: dependencies & targets
            for name, (_line, dependencies) in assignments.items()
        })
        try:
            order = sorter.static_order()
            plan = tuple(assignments[name][0] for name in order)
        except CycleError:
            if conditional_by_target:
                self._parameter_dependencies(loop, assignments, conditional_by_target, sorter)
            return None
        return plan

    def _parameter_dependencies(self, loop, assignments, conditional_by_target, sorter):
        """Associate conditional edges with parameters, only inside actual scalar SCCs."""
        expressions = {coefficient for _rhs, conditional in conditional_by_target.values()
                       for contributions in conditional.values()
                       for coefficient, _signal, _replacement in contributions}
        parameters = self.runtime.parameter_dependencies(expressions)
        if not any(parameters.values()):
            return
        residual = set(assignments)
        # CycleError leaves the sorter's ready prefix usable; do not build or sort another graph.
        while sorter.is_active():
            ready = sorter.get_ready()
            residual.difference_update(ready)
            sorter.done(*ready)
        names = [name for name in assignments if name in residual]
        index = {name: i for i, name in enumerate(names)}
        successors = [[] for _ in names]
        for name in names:
            for dependency in assignments[name][1] & residual:
                successors[index[dependency]].append(index[name])
        components = _strong_components(successors, range(len(names)))
        component_of = {names[node]: ci for ci, component in enumerate(components)
                        if len(component) > 1 or component[0] in successors[component[0]]
                        for node in component}
        public = {self.value(getattr(owner, role), field):
                  f"{self.subsystem_name[id(owner)]}.{role}.{field}"
                  for owner in loop.owners for role in ("state", "inp", "out")
                  for field in _fields(getattr(owner, role))}
        affected = {}
        for target, (rhs, fragments) in conditional_by_target.items():
            if target not in component_of:
                continue
            conditional = []
            for fragment, contributions in fragments.items():
                if fragment not in rhs:
                    continue
                for coefficient, signal, replacement in contributions:
                    paths = parameters[coefficient]
                    dependencies = ((set(self.runtime.expression_symbols(signal)) & residual)
                                    - set(self.runtime.expression_symbols(replacement)))
                    if paths and dependencies:
                        conditional.append((fragment, coefficient, paths, dependencies, replacement))
            unguarded = rhs
            replacements = {item[0]: item[4] for item in conditional}
            for fragment in sorted(replacements, key=len, reverse=True):
                unguarded = unguarded.replace(fragment, replacements[fragment])
            unconditional = set(re.findall(r"\b\w+\b", unguarded))
            for _fragment, coefficient, paths, dependencies, _replacement in conditional:
                for dependency in dependencies - unconditional:
                    if component_of.get(dependency) == component_of[target]:
                        affected.setdefault((coefficient, paths), set()).add(
                            (public.get(dependency, dependency), public.get(target, target)))
        for (coefficient, paths), dependencies in sorted(affected.items()):
            self.diagnostics.append({
                "code": "conditional-algebraic-dependency",
                "parameters": paths,
                "condition": f"(({coefficient}) == 0.0)",
                "dependencies": tuple(sorted(dependencies)),
                "loop": loop.labels,
                "message": "Retained parameters " + ", ".join(paths)
                + " control conditional dependencies in an algebraic loop; changing them "
                "may affect simulation cost (static structure only, not a timing prediction).",
            })

    def _rhs(self, sub: Any) -> list[str]:
        hook = getattr(sub, "cpp_rhs", None)
        if hook is not None:
            return list(hook(self))
        generator = self.continuous_units.get(id(sub))
        if generator is not None:
            return generator.derivatives()
        if (getattr(sub, "continuous", False) and hasattr(sub, "graph")
                and hasattr(sub, "state_names")):
            label = self.subsystem_name.get(id(sub), "controller")
            raise TypeError(f"C++ export: continuous controller {label!r} is not yet lowered")
        label = self.subsystem_name.get(id(sub), type(sub).__name__)
        try:
            method = sub.rhs
            if (isinstance(sub, RLBranch) and getattr(method, "__func__", None) in
                    (RLBranch._rhs_bilateral, RLBranch._rhs_unilateral,
                     RLBranch._rhs_open, RLBranch._rhs_conduction)):
                method = RLBranch.rhs.__get__(sub)
            return self.trace_rhs(sub, method, self._equation_attributes(sub))
        except Exception as exc:
            raise TypeError(
                f"C++ export: cannot lower the state equation of subsystem {label!r} "
                f"({type(sub).__name__}): {exc}"
            ) from exc

    def body(self, derivatives: bool = True, *,
             direct_guards: dict[int, str] | None = None) -> list[str]:
        """Emit shared equations, optionally collecting prerequisites for direct-only paths."""
        lines = []
        for operation, rhs_required in zip(self.model._plan, self.model._plan_rhs):
            if derivatives and not rhs_required:
                continue
            lines.extend(self._operation(operation, direct_guards=direct_guards))
        for sub, entries in (self.model._rhs_layout if derivatives else ()):
            expressions = self._rhs(sub)
            if len(expressions) != len(entries):
                raise RuntimeError(
                    f"C++ export: {self.subsystem_name[id(sub)]} emitted {len(expressions)} "
                    f"derivatives for {len(entries)} states"
                )
            for expression, (index, is_complex) in zip(expressions, entries):
                if is_complex:
                    temp = f"d_{self.prefix[id(sub)]}_{index}"
                    lines += [f"const Complex {temp} = {expression};",
                              f"dy[{index}] = std::real({temp});",
                              f"dy[{index + 1}] = std::imag({temp});"]
                else:
                    lines.append(f"dy[{index}] = {expression};")
        persistent_fields = None
        if direct_guards is not None:
            # A proven DAG needs no warm-start memory. Keep seeds only for actual feedback.
            persistent_fields = {(id(record), name) for loop in self.model.algebraic_loops
                                 if id(loop) not in direct_guards
                                 for record, name in self._algebraic_seeds(loop)}
            persistent_fields.intersection_update(self._algebraic_fields)
        return self.declarations(persistent_fields) + lines

    def members(self, controllers=()) -> list[str]:
        lines = [f"{_CPP_TYPES[self.record_types[key]]} {self.value(record, key[1])} = "
                 f"{_literal(getattr(record, key[1]))};"
                 for key, record in self._algebraic_fields.items()]
        lines += [f"{_CPP_TYPES[kind]} {name} = {_literal(getattr(self.runtime.object(owner), attribute))};"
                  for name, kind, owner, attribute in self.equation_members.values()]
        for loop in self.model.algebraic_loops:
            if len(loop.tears) == 1 and self._algebraic_plan(loop) is None:
                tear, = loop.tears
                index = self._algebraic_index[id(loop)]
                kind = _CPP_TYPES[self.record_types[id(tear[0]), tear[1]]]
                inverse = loop._inverse
                lines += [f"{kind} algebraic_inverse_{index} = "
                          f"{_literal(0.0 if inverse is None else inverse)};",
                          f"bool algebraic_has_inverse_{index} = "
                          f"{'false' if inverse is None else 'true'};"]
        for member, subsystem, attribute in self.element_parameters.values():
            lines.append(f"double {member} = "
                         f"{_number(getattr(self.runtime.object(subsystem), attribute))};")
        for label, targets in self.model._zoh_targets.items():
            record, field = targets[0]
            value = getattr(record, field)
            lines.append(
                f"{_CPP_TYPES[self.record_types[id(record), field]]} hold_{_identifier(label)} = {_literal(value)};"
            )
        for sub in self.model.subsystems:
            attributes = self._equation_attributes(sub)
            for name in ("breaker_open", "tripped"):
                if name in attributes:
                    lines.append(f"bool {attributes[name].code} = {_literal(bool(getattr(sub, name)))};")
            if "source" in attributes:
                start, duration, law = self.source_ramp_members(sub)
                control = next((control for control in controllers
                                if control.unit.dclink is sub), None)
                expression = "1.0"
                if control is not None:
                    _, since, ramp = control.unit.scenario.at(self.p.simulation.initial.t)
                    since = since if math.isfinite(since) else self.p.simulation.initial.t
                    lines.append(f"double {start} = {_number(since)}, {duration} = {_number(ramp)};")
                    expression = (f"(!{control.m('breaker')} ? 0.0 : {duration} <= 0.0 ? 1.0 : "
                                  f"smoothstep((t - {start}) / {duration}))")
                lines.append(f"double {law}(double t) const noexcept {{ return {expression}; }}")
        for generator in self.continuous_units.values():
            lines.extend(generator.members())
        return lines

    def observations(self, generators: dict) -> tuple[list[str], list[str]]:
        """Members and assignments for values consumed by scheduling and public output."""
        members: list[str] = []
        assigns: list[str] = []
        seen: set[str] = set()
        self._observed_values = {}

        def add(name: str, expression: str, kind: str) -> None:
            member = f"obs_{_identifier(name)}"
            if member not in seen:
                seen.add(member)
                members.append(f"{_CPP_TYPES[kind]} {member}{{}};")
            assigns.append(f"{member} = {expression};")
            self._observed_values.setdefault(expression, member)

        self._add_observation = add

        for name, node in self.system.buses.items():
            add(f"{name}.u", self.value(node.out, "u"),
                self.record_types[id(node.out), "u"])
        for name, branch in self.system.branches.items():
            add(f"{name}.i", self.value(branch.out, "i"),
                self.record_types[id(branch.out), "i"])
        for name, source in self.system.sources.items():
            add(f"{name}.i", self.value(source.branch.out, "i"), "complex")
            add(f"{name}.angle", self.value(source.emf.out, "phi"), "real")
        for name, unit in self.system.units.items():
            if getattr(unit, "branch_f", None) is not None and hasattr(unit, "dclink"):
                add(f"{name}.i_c", self.value(unit.branch_f.out, "i"),
                    self.record_types[id(unit.branch_f.out), "i"])
                add(f"{name}.u_g", self.value(unit.bus.out, "u"),
                    self.record_types[id(unit.bus.out), "u"])
                add(f"{name}.u_dc", self.value(unit.dclink.out, "u_dc"), "real")
                add(f"{name}.i_dc", self.value(unit.dclink.inp, "i_dc"), "real")
            else:
                for signal, value in self.trace_element_signals(unit).items():
                    add(signal, value.code, value.kind)
            if hasattr(unit.ctrl, "log_sources"):
                for port, value in self.unit_control_inputs(unit).items():
                    add(f"{name}.ctrl.{port}", value.code, value.kind)
            if not unit.continuous or unit.adc is not None or unit.cfg.meas.sources:
                for channel, value in self.unit_measurements(unit).items():
                    add(f"{name}.adc.{channel}", value.code, value.kind)
            if unit.adc is not None and unit.adc.rms:
                generator = generators[name]
                for channel in unit.adc.rms:
                    add(f"{name}.meas.{channel}_rms", generator._rms_value(channel), "real")
            if unit.adc is not None and unit.adc.clocked:
                generator = generators[name]
                for channel, value in unit.adc.values.items():
                    public = channel + "_mean" if channel in unit.adc.channels else channel
                    add(f"{name}.meas.{public}", generator.m("adc_held_" + channel),
                        _Expression.of(value).kind)
        for name, element in self.system.named_elements.items():
            signals = getattr(element, "signals", None)
            if signals is not None:
                values = signals()
                if hasattr(element, "branch") and set(values) == {f"{name}.i"}:
                    add(f"{name}.i", self.value(element.branch.out, "i"), "complex")
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
                    add(signal, value.code, value.kind)
        for source in self.system.composite_signal_sources:
            for signal, value in self.trace_element_signals(source).items():
                add(signal, value.code, value.kind)
        for name, record, field in self._solver_observations.values():
            add(name, self.value(record, field), self.record_types[id(record), field])
        return members, assigns

    def observed(self, value):
        """Reuse an existing observation store for a traced lifecycle input."""
        if value.code not in self._observed_values:
            name = f"initial_{len(self._observed_values)}"
            self._add_observation(name, value.code, value.kind)
        return _Expression(self._observed_values[value.code], value.kind)


_SHORT_INTERVALS = r'''template <typename Value>
std::vector<std::pair<double, Value>> merge_short_intervals(double span, double minimum_interval,
        Value value, const std::vector<std::pair<double, Value>>& changes) {
    std::vector<std::pair<double, Value>> result;
    double previous = 0.0;
    for (const auto& change : changes) {
        if (change.first - previous > minimum_interval) {
            result.emplace_back(change.first - previous, value);
            previous = change.first;
        }
        value = change.second;
    }
    if (span - previous > minimum_interval || result.empty())
        result.emplace_back(span - previous, value);
    else result.back().first += span - previous;
    return result;
}
'''


_PHASE_SHIFT_KERNEL = r'''struct PhaseShiftPattern {
    double width, phase;
    int mode;
};
struct PhaseShiftModulatorState {
    using Edges = std::vector<std::pair<double, int>>;
    double angle, time, frequency;
    double minimum_interval;

    static const Edges& pulses(int mode) {
        switch (mode) {
@@PULSE_CASES@@
        default: throw std::runtime_error("invalid phase-shift mode");
        }
    }
    static int level(const PhaseShiftPattern& pattern, double y) {
        if (pattern.width <= 0.0) return 0;
        const double half = 0.5 * pi * pattern.width;
        for (const auto& pulse : pulses(pattern.mode)) {
            double offset = std::remainder(y - pulse.first, 2.0 * pi);
            if (offset == pi) offset = -pi;
            if (-half <= offset && offset < half) return pulse.second;
        }
        return 0;
    }
    static std::pair<int, Edges> edges(const PhaseShiftPattern& pattern, double theta,
            double omega, double first, double last, double snap) {
        const double start = theta + omega * first - pattern.phase;
        int initial = level(pattern, start + omega * snap);
        if (pattern.width <= 0.0 || last - snap <= first + snap) return {initial, {}};
        const double period = 2.0 * pi / omega;
        const double half = 0.5 * pi * pattern.width;
        std::vector<double> times;
        for (const auto& pulse : pulses(pattern.mode)) {
            for (double bound : {pulse.first - half, pulse.first + half}) {
                double x = std::fmod(bound - start, 2.0 * pi);
                if (x < 0.0) x += 2.0 * pi;
                x /= omega;
                if (x <= snap) x += period;
                while (first + x <= last - snap) {
                    times.push_back(first + x);
                    x += period;
                }
            }
        }
        std::sort(times.begin(), times.end());
        Edges result;
        int previous = initial;
        for (double at : times) {
            const int after = level(pattern, theta + omega * at - pattern.phase + omega * snap);
            if (after != previous) {
                result.emplace_back(at, after);
                previous = after;
            }
        }
        return {initial, result};
    }
    Edges load(double t, double span, double width, double phase, double next_frequency, double mode) {
        if (!(std::isfinite(width) && std::isfinite(phase) && std::isfinite(next_frequency)
                && next_frequency > 0.0 && std::isfinite(mode) && mode == std::trunc(mode)))
            throw std::runtime_error("invalid phase-shift registers");
        pulses(static_cast<int>(mode));
        angle = std::fmod(angle + 2.0 * pi * frequency * (t - time), 2.0 * pi);
        if (angle < 0.0) angle += 2.0 * pi;
        time = t;
        frequency = next_frequency;
        const double omega = 2.0 * pi * frequency;
        const double period = 1.0 / frequency;
        const double snap = @@SNAP@@ * period;
        PhaseShiftPattern next{std::clamp(width, 0.0, 1.0), phase, static_cast<int>(mode)};
        auto all = edges(next, angle, omega, 0.0, span, snap);
        return merge_short_intervals(span, minimum_interval, all.first, all.second);
    }
};
'''.replace("@@SNAP@@", _number(_SNAP))


_PULSE_SEQUENCE_KERNEL = r'''template <std::size_t N, std::size_t G> struct PulseSequenceState {
    using Word = std::array<double, N>;
    using Sequence = std::vector<std::pair<double, Word>>;
    std::array<int, N> groups;
    Word entry{}, next{};
    std::array<double, G> remaining{}, next_remaining{};
    double time;
    bool restored;

    bool different(const Word& a, const Word& b, int group) const {
        for (std::size_t k = 0; k < N; ++k)
            if (groups[k] == group && a[k] != b[k]) return true;
        return false;
    }
    bool active(const Word& word, int group) const {
        for (std::size_t k = 0; k < N; ++k)
            if (groups[k] == group && word[k] != 0.0) return true;
        return false;
    }
    void assign(Word& target, const Word& source, int group) const {
        for (std::size_t k = 0; k < N; ++k) if (groups[k] == group) target[k] = source[k];
    }
    static Word at(const Sequence& parts, double time) {
        Word value = parts.front().second;
        for (std::size_t k = 1; k < parts.size() && parts[k].first <= time; ++k)
            value = parts[k].second;
        return value;
    }
    Sequence load(double t, double span, double period, const Sequence& raw,
                  bool enabled, double minimum_interval) {
        if (time != t) {
            if (!restored) { entry = next; remaining = next_remaining; }
            time = t;
            restored = false;
        }
        if (!enabled) {
            next.fill(0.0); next_remaining.fill(0.0);
            return {{span, Word{}}};
        }
        struct Change { double time; int group; Word value; };
        std::vector<Change> changes;
        Word initial{}, next_word{};
        const Word zero{};
        const double snap = @@SNAP@@ * period;
        for (std::size_t group = 0; group < G; ++group) {
            const double free = active(entry, group) ? std::max(0.0, remaining[group]) * period : 0.0;
            assign(initial, free > snap ? entry : zero, group);
            Word after{};
            double left = 0.0;
            if (free >= span - snap) {
                after = free > span + snap ? entry : zero;
                left = std::max(0.0, (free - span) / period);
            } else {
                if (free > snap) changes.push_back({free, static_cast<int>(group), zero});
                Sequence parts;
                double cursor = 0.0;
                for (const auto& segment : raw) {
                    if (parts.empty() || different(parts.back().second, segment.second, group))
                        parts.emplace_back(cursor, segment.second);
                    cursor += segment.first;
                }
                double begin = std::numeric_limits<double>::infinity();
                Word value{};
                if (parts.size() == 1) { begin = free; value = parts.front().second; }
                else {
                    for (std::size_t k = 1; k < parts.size(); ++k)
                        if (active(parts[k].second, group) && parts[k].first >= free - snap) {
                            begin = parts[k].first; value = parts[k].second; break;
                        }
                    if (free <= snap && active(parts.front().second, group)
                            && different(parts.front().second, at(parts, period - snap), group)) {
                        begin = 0.0; value = parts.front().second;
                    }
                    begin = std::max(free, begin);
                }
                if (begin < span - snap) {
                    if (begin <= snap) assign(initial, value, group);
                    else changes.push_back({begin, static_cast<int>(group), value});
                    for (std::size_t k = 1; k < parts.size(); ++k)
                        if (parts[k].first > begin + snap && parts[k].first < span - snap)
                            changes.push_back({parts[k].first, static_cast<int>(group), parts[k].second});
                    after = at(parts, span + snap);
                    if (active(after, group))
                        for (std::size_t k = 1; k < parts.size(); ++k)
                            if (parts[k].first > span + snap) {
                                left = (parts[k].first - span) / period; break;
                            }
                }
            }
            assign(next_word, after, group);
            next_remaining[group] = left;
        }
        next = next_word;
        std::sort(changes.begin(), changes.end(), [](const Change& a, const Change& b) {
            return a.time < b.time || (a.time == b.time && a.group < b.group);
        });
        Sequence words;
        Word word = initial;
        for (const auto& change : changes) {
            assign(word, change.value, change.group);
            words.emplace_back(change.time, word);
        }
        return merge_short_intervals(span, minimum_interval, initial, words);
    }
};
'''.replace("@@SNAP@@", _number(_SNAP))


def _phase_shift_kernel():
    cases = []
    for mode, pulses in PHASE_SHIFT_MODES.items():
        values = ", ".join("{" + _number(centre) + ", " + str(level) + "}"
                           for centre, level in pulses)
        cases.append(f"        case {mode}: {{ static const Edges values{{{values}}}; return values; }}")
    return _SHORT_INTERVALS, _PHASE_SHIFT_KERNEL.replace("@@PULSE_CASES@@", "\n".join(cases))


class _CarrierModulatorGenerator:
    def phase_clock(self, generator):
        modulator = generator.bridge.modulator
        if isinstance(modulator, SynchronousCarrier):
            return PhaseClock(
                _Expression(f"({generator.m('t_load')} - {_number(generator.bridge.load_period)})", "real"),
                _Expression(generator.m('carrier_angle'), "real"),
                _Expression(f"(2.0 * pi * {generator.m('carrier_frequency')})", "real"))
        period = generator.bridge.carrier_period
        return PhaseClock(0.0, 2.0 * math.pi * getattr(modulator, "phase", 0.0),
                          2.0 * math.pi / period)

    @staticmethod
    def shared_code():
        return ()

    def members(self, generator):
        if not isinstance(generator.bridge.modulator, SynchronousCarrier):
            return []
        bridge = generator.bridge
        modulator = bridge.modulator
        initial = generator.model.simulation._initial_values(generator.p.simulation.initial.t)
        prefix = f"{generator.name}.{bridge.state_prefix}."
        angle = (_number(modulator._angle) if prefix + "angle" in initial else
                 f"std::fmod(std::fmod({_number(modulator.pulse_ratio)} * {generator.m('theta')} + "
                 f"2.0 * pi * {_number(modulator.phase)}, 2.0 * pi) + 2.0 * pi, 2.0 * pi)")
        frequency = (_number(modulator._frequency) if prefix + "frequency" in initial else
                     f"{_number(modulator.pulse_ratio)} * {generator.m('omega')} / (2.0 * pi)")
        return [f"double {generator.m('sync_theta')} = {generator.m('theta')};",
                f"double {generator.m('sync_omega')} = {generator.m('omega')};",
                f"double {generator.m('sync_time')} = {_number(bridge.t_sync)};",
                f"double {generator.m('carrier_angle')} = {angle};",
                f"double {generator.m('carrier_frequency')} = {frequency};"]

    def write_sync(self, generator):
        if not isinstance(generator.bridge.modulator, SynchronousCarrier):
            return []
        return [f"        {generator.m('sync_theta')} = {generator.m('theta')};",
                f"        {generator.m('sync_omega')} = {generator.m('omega')};",
                f"        {generator.m('sync_time')} = t;"]

    def state_expression(self, generator, local):
        if isinstance(generator.bridge.modulator, SynchronousCarrier):
            if local in ("angle", "frequency"):
                return generator.m("carrier_" + local)
        return None

    def load(self, generator, hold: str, switching: bool) -> list[str]:
        """Map active compares to the topology's one held bridge input."""
        bridge = generator.bridge
        clock_lines = []
        if switching:
            modulator = bridge.modulator
            if isinstance(modulator, SynchronousCarrier):
                rate = f"{_number(modulator.pulse_ratio)} * {generator.m('sync_omega')}"
                angle = (f"{_number(modulator.pulse_ratio)} * "
                         f"({generator.m('sync_theta')} + {generator.m('sync_omega')} * "
                         f"(t - {generator.m('sync_time')})) + 2.0 * pi * {_number(modulator.phase)}")
                clock_lines = [f"        {generator.m('carrier_angle')} = std::fmod({angle}, 2.0 * pi);",
                               f"        if ({generator.m('carrier_angle')} < 0.0) "
                               f"{generator.m('carrier_angle')} += 2.0 * pi;",
                               f"        {generator.m('carrier_frequency')} = ({rate}) / (2.0 * pi);"]
                frequency = generator.m('carrier_frequency')
                phase = f"({generator.m('carrier_angle')} / (2.0 * pi) - t * {frequency})"
            elif isinstance(modulator, CarrierComparison):
                frequency = _number(modulator.f_sw)
                phase = _number(modulator.phase)
            else:
                raise TypeError(
                    f"C++ export: unsupported PWM modulator {type(modulator).__name__}"
                )
            mapped = (
                f"make_switches_{generator.tag}(t, {_number(bridge.load_period)}, "
                f"{frequency}, {phase}, {hold});"
            )
        else:
            values = tuple(f"{generator.m('active')}[{index}]" for index in range(generator.bridge.modulation_width))
            mapped = f"{hold} = {generator._topology_expression(generator.topology.pwm_state, values)};"
        return [*clock_lines, f"        {mapped}",
                f"        {generator.m('load_end')} = t + {_number(bridge.load_period)};"]

    def method(self, generator) -> str:
        """Generate one topology-driven carrier-comparison implementation."""
        if isinstance(generator.bridge.modulator, ZOH):
            return ""
        hold = generator._hold()
        values = tuple(f"q[{index}]" for index in range(generator.bridge.modulation_width))
        encoded = generator._topology_expression(generator.topology.switching_state, values)
        q_kind = _CPP_TYPES[_Expression.of(generator.topology.zero).kind]
        minimum_interval = _number(generator.bridge.modulator.min_interval)
        pulse = generator.bridge.pulse_update
        switches = "edges_out" if pulse is not None else generator.m("switches")
        method = "raw_switches" if pulse is not None else "make_switches"
        extra = ""
        index_reset = f"{generator.m('switch_index')} = 0;"
        finish = (f"{generator.m('next_switch')} = {switches}.empty() ? "
                  f"std::numeric_limits<double>::infinity() : {switches}[0].first;")
        if pulse is not None:
            q_kind = f"std::array<double, {generator.bridge.modulation_width}>"
            encoded = "q"
            extra = f", std::vector<std::pair<double, {q_kind}>>& edges_out"
            index_reset, finish = "", ""
        result = f'''void {method}_{generator.tag}(double t, double span, double frequency,
        double phase, {q_kind}& held{extra}) {{
    if (!(frequency > 0.0)) throw std::runtime_error("PWM frequency is not positive");
    {switches}.clear();
    {index_reset}
    std::array<double, {generator.bridge.modulation_width}> modulation{{}};
    std::array<double, {generator.bridge.modulation_width}> q{{}};
    for (int k = 0; k < {generator.bridge.modulation_width}; ++k)
        modulation[k] = std::clamp(2.0 * {generator.m('active')}[k] - 1.0, -1.0, 1.0);
    double position = carrier_position(t, frequency, phase);
    const double carrier0 = carrier_value(position);
    for (int k = 0; k < {generator.bridge.modulation_width}; ++k)
        q[k] = position < 0.5 ? (modulation[k] > carrier0) : (modulation[k] >= carrier0);
    held = {encoded};
    std::vector<std::tuple<double, int, double>> edges;
    double covered = 0.0;
    for (int piece = 0; piece < 64 && covered < span - {_number(_EPS)}; ++piece) {{
        position = carrier_position(t + covered, frequency, phase);
        const bool rising = position < 0.5;
        const double length = std::min(((rising ? 0.5 : 1.0) - position) / frequency,
                                       span - covered);
        const double c0 = carrier_value(position);
        const double slope = rising ? 4.0 * frequency : -4.0 * frequency;
        for (int k = 0; k < {generator.bridge.modulation_width}; ++k) {{
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
        if (at - previous > {minimum_interval}) {{
            {switches}.emplace_back(t + at, {encoded});
            previous = at;
        }} else if (!{switches}.empty()) {{
            {switches}.back().second = {encoded};
        }} else {{
            held = {encoded};
        }}
    }}
    if (!{switches}.empty() && span - previous <= {minimum_interval})
        {switches}.pop_back();
    {finish}
}}
'''
        if pulse is not None:
            width = len(pulse.entry)
            kind = _CPP_TYPES[_Expression.of(generator.topology.zero).kind]
            result += f'''void make_switches_{generator.tag}(double t, double span, double frequency,
        double phase, {kind}& held) {{
    using Word = std::array<double, {width}>;
    Word initial{{}};
    std::vector<std::pair<double, Word>> edges;
    const double period = 1.0 / frequency;
    const double horizon = span + period;
    raw_switches_{generator.tag}(t, horizon, frequency, phase, initial, edges);
    std::vector<std::pair<double, Word>> raw;
    double previous = t;
    for (const auto& edge : edges) {{
        raw.emplace_back(edge.first - previous, initial);
        previous = edge.first; initial = edge.second;
    }}
    raw.emplace_back(t + horizon - previous, initial);
{_pulse_application(generator, "raw", "period")}
}}
'''
        return result



class _PhaseShiftModulatorGenerator:
    shared_code = staticmethod(_phase_shift_kernel)

    def write_sync(self, generator):
        return []

    @staticmethod
    def _member(generator):
        return generator.m("phase_modulator")

    def members(self, generator):
        mod = generator.bridge.modulator
        return [f"PhaseShiftModulatorState {self._member(generator)}{{"
                f"{_number(mod._theta)}, {_number(mod._t)}, {_number(mod._f)}, "
                f"{_number(mod.min_interval)}" + "};"]

    def state_expression(self, generator, local):
        return self._member(generator) + ".angle" if local == "angle" else None

    def phase_clock(self, generator):
        member = self._member(generator)
        return PhaseClock(_Expression(member + ".time", "real"),
                          _Expression(member + ".angle", "real"),
                          _Expression(f"(2.0 * pi * {member}.frequency)", "real"))

    def load(self, generator, hold, switching):
        return [f"        make_switches_{generator.tag}(t, {_number(generator.bridge.load_period)}, {hold});",
                f"        {generator.m('load_end')} = t + {_number(generator.bridge.load_period)};"]

    def method(self, generator):
        kind = _CPP_TYPES[_Expression.of(generator.topology.zero).kind]
        indices = [generator.bridge.register_names.index(name)
                   for name in generator.bridge.modulator.pattern_registers]
        values = ", ".join(f"{generator.m('active')}[{index}]" for index in indices)
        encoded = generator._topology_expression(generator.topology.switching_state,
                                                 ("(level == 1 ? 1.0 : 0.0)",
                                                  "(level == -1 ? 1.0 : 0.0)"))
        if generator.bridge.pulse_update is not None:
            frequency = f"{generator.m('active')}[{indices[2]}]"
            return f'''void make_switches_{generator.tag}(double t, double span, {kind}& held) {{
    const double period = 1.0 / ({frequency});
    auto candidate = {self._member(generator)}.load(t, span + period, {values});
    std::vector<std::pair<double, std::array<double, 2>>> raw;
    for (const auto& item : candidate) {{
        const int level = item.second;
        raw.push_back({{item.first, {{(level == 1 ? 1.0 : 0.0), (level == -1 ? 1.0 : 0.0)}}}});
    }}
{_pulse_application(generator, "raw", "period")}
}}
'''
        return f'''void make_switches_{generator.tag}(double t, double span, {kind}& held) {{
    auto segments = {self._member(generator)}.load(t, span, {values});
{_sequence_schedule(generator, encoded, 'level')}
}}
'''


def _pulse_application(generator, raw, period):
    pulse = generator.bridge.pulse_update
    values = tuple(f"word[{index}]" for index in range(len(pulse.entry)))
    encoded = generator._topology_expression(generator.topology.switching_state, values)
    return f'''    auto segments = {generator.m('pulse_sequence')}.load(t, span, {period}, {raw},
        {generator.m('active')}[{generator.width}] != 0.0,
        {_number(generator.bridge.modulator.min_interval)});
{_sequence_schedule(generator, encoded, 'word')}'''


def _sequence_schedule(generator, encoded, value):
    """Encode waveform segments and install their edges through one scheduling path."""
    return f'''    {generator.m('switches')}.clear();
    {generator.m('switch_index')} = 0;
    auto encode = [](const auto& {value}) {{ return {encoded}; }};
    held = encode(segments.front().second);
    double at = t;
    for (std::size_t k = 0; k + 1 < segments.size(); ++k) {{
        at += segments[k].first;
        auto value = encode(segments[k + 1].second);
        if (at <= t + time_eps) held = value;
        else {generator.m('switches')}.emplace_back(at, value);
    }}
    {generator.m('next_switch')} = {generator.m('switches')}.empty()
        ? std::numeric_limits<double>::infinity() : {generator.m('switches')}[0].first;'''


_NATIVE_MODULATORS = {
    CarrierComparison: _CarrierModulatorGenerator,
    SynchronousCarrier: _CarrierModulatorGenerator,
    ZOH: _CarrierModulatorGenerator,
    PhaseShiftModulator: _PhaseShiftModulatorGenerator,
}


class _ControlGenerator:
    """Shared names and configuration expressions for generated unit controllers."""

    def __init__(self, name: str, unit: Any, model: _ModelGenerator, prefix: str) -> None:
        self.name, self.unit, self.model = name, unit, model
        self.tag = _identifier(name)
        self.cfg, self.ctrl = model.runtime.copy(unit.cfg), model.runtime.object(unit.ctrl)
        self.graph = getattr(self.ctrl, "graph", None)
        modulator = getattr(unit.bridge, "modulator", None)
        self.modulation = next((adapter() for cls, adapter in _NATIVE_MODULATORS.items()
                                if modulator is not None and _uses_implementation(modulator, cls)), None)
        if modulator is not None and self.modulation is None:
            raise TypeError(
                f"C++ export: modulator {type(modulator).__name__!r} has no native adapter"
            )
        self.prefix = f"{prefix}_{self.tag}_"
        if self.modulation is not None:
            bridge = copy.copy(unit.bridge)
            bridge.phase_clock = lambda: self.modulation.phase_clock(self)
            bridge.register = lambda name: _Expression(
                f"{self.m('active')}[{unit.bridge.register_names.index(name)}]", "real")
            model.equation_references[id(unit.bridge)] = bridge
        reference_prefix = f"units.{name}.ctrl.references."
        self._changing_references = {
            path[len(reference_prefix):]
            for change in self.p.changes for path in change.paths
            if path.startswith(reference_prefix)
        }

    @property
    def p(self):
        return self.model.p

    @property
    def adc(self):
        return self.model.runtime.object(self.unit.adc)

    def _adc_state_expression(self, local: str) -> str | None:
        """Map the common ADC state contract to this unit's generated members."""
        if local == "integration_time":
            return self.m("adc_integration_time")
        if local.startswith("x_"):
            return self.m("adc_acc_" + local[2:])
        if local.startswith("last_"):
            return self.m("adc_last_" + local[5:])
        if local.startswith("held_"):
            return self.m("adc_held_" + local[5:])
        return None

    @property
    def protection(self):
        return self.model.runtime.object(self.unit.protection)

    def m(self, name: str) -> str:
        return self.prefix + _identifier(name)

    def g(self, key: str) -> str:
        return self.m("g_" + key)

    def node(self, name: str, field: str) -> str:
        return self.m(f"n_{name}_{field}")

    def d(self, name: str, field: str) -> str:
        return self.m(f"d_{name}_{field}")

    def _loop_members(self) -> list[str]:
        """Declare loop-owned values; continuous ODE states already live in ``y``."""
        lines: list[str] = []
        for loop_name, node in self.graph.nodes.items():
            states = {} if self.continuous else node.get_state()
            for local, value in states.items():
                lines.append(f"{_CPP_TYPES[_Expression.of(value).kind]} "
                             f"{self.node(loop_name, local)} = {_literal(value)};")
            for carried in node.carried:
                value = _Expression.of(getattr(node, carried))
                lines.append(f"{_CPP_TYPES[value.kind]} {self.node(loop_name, carried)} = {value.code};")
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

    def _startup_value_line(self) -> str:
        """The continuous ``ContinuousStartup.at`` value, refreshed at every output evaluation."""
        return (f"{self.m('startup_value')} = !{self.m('startup_run')} ? 0.0 : ({self.m('startup_ramp')} <= 0.0 ? 1.0 : "
                f"smoothstep((t - {self.m('startup_start')}) / {self.m('startup_ramp')}));")

    def _startup_complete(self) -> str:
        """Continuous start-up completion as passed to sampled protection."""
        return (f"{self.m('startup_run')} && ({self.m('startup_ramp')} <= 0.0 || "
                f"t - {self.m('startup_start')} >= {self.m('startup_ramp')} - time_eps)")

    def _reference(self, name: str) -> str:
        if name in self._changing_references:
            return self.m("reference_" + name)
        path = f"units.{self.name}.ctrl.references.{name}"
        return self.model.runtime.value(path, getattr(self.cfg.ctrl.references, name))

    def reference_members(self) -> list[str]:
        """Only event-modified references need storage, updated at the actual event boundary.

        Time expressions would change an RK endpoint before the event is applied and can miss a
        coincident sampled interrupt through rounding. Unchanged references remain constants.
        """
        if self.graph is None:
            return []
        return [
            f"double {self.m('reference_' + name)} = "
            f"{self.model.runtime.value(f'units.{self.name}.ctrl.references.{name}', value)};"
            for name, value in self.graph.references.items()
            if name in self._changing_references
        ]

    def log_observation_lines(self) -> list[str]:
        """Optional log refresh after every model output has been synchronized."""
        return []

    def input(self, loop: str, port: str) -> str:
        return self.source(self.graph.connections[f"{loop}.{port}"])

    @staticmethod
    def _measurement_name(path: str) -> str:
        return "meas_" + _identifier(path.partition(".")[2])

    def source(self, path: str) -> str:
        """The generated value read through one control-graph source path."""
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

    def _sampled_loop_lines(self) -> list[str]:
        """The sampled startup and graph update, independent of the execution target."""
        lines = [
            f"if ({self.m('startup_run')} && !{self.m('startup_complete')}) {{",
            f"    const double progress = {self.m('startup_ramp')} <= 0.0 ? 1.0 : "
            f"std::min(1.0, {self.m('startup_steps')} * {_number(self.cfg.ctrl.period)} / "
            f"{self.m('startup_ramp')});",
            f"    {self.m('startup_value')} = smoothstep(progress);",
            f"    {self.m('startup_complete')} = progress >= 1.0;",
            f"    if (!{self.m('startup_complete')}) ++{self.m('startup_steps')};",
            "}",
        ]
        for name in self.graph.order:
            lines += self._loop_lines(name, self.graph.nodes[name])
        return lines

    def _sampled_control_inputs(self) -> dict[str, _Expression]:
        channels = {name: _Expression(self._adc_value(name), value.kind)
                    for name, value in self.model.unit_measurements(self.unit).items()}
        unit = copy.copy(self.unit)
        unit.cfg, unit.ctrl = self.cfg, self.ctrl
        return self.model._trace(lambda: unit.control_inputs_from(channels))

    def _sampled_measurement_lines(self) -> list[str]:
        return [f"const {_CPP_TYPES[value.kind]} "
                f"meas_{_identifier(path[5:])} = {value.code};"
                for path, value in self._sampled_control_inputs().items()]

    def _modulation_values(self, command: str, theta: str, udc: str):
        stage = self.ctrl.stage
        if not stage._trusted_pwm or not stage._standard_limiter:
            raise TypeError("C++ export: a custom VSC output stage has no equation tracer contract")
        return self.model._trace(lambda: stage.equation(
            _Expression(command, "complex"),
            _Expression(f"Complex{{std::cos({theta}), std::sin({theta})}}", "complex"),
            _Expression(udc, "real"),
        ))

    def _sampled_due(self, name: str) -> str:
        every = self.graph.period_stride(self.graph.periods[name], self.cfg.ctrl.period)
        if isinstance(every, _InitializationExpression):
            return f"{self.m('pwm_k')} % static_cast<long long>({_number(every)}) == 0"
        return "true" if int(every) == 1 else f"{self.m('pwm_k')} % {int(every)} == 0"

    def _sampled_vsc_command_lines(self) -> list[str]:
        """Match VSCController._accept_command, including the held command's coordinate frame."""
        lines = [f"{self.m('theta')} = {self.source(self.graph.outputs['theta'])};",
                 f"{self.m('omega')} = {self.source(self.graph.outputs['omega'])};"]
        due = ([self._sampled_due(self.ctrl._terminal)] if self.ctrl._terminal_in_graph else
               list(dict.fromkeys(self._sampled_due(name) for name in self.graph.order)))
        condition = "true" if "true" in due else " || ".join(f"({item})" for item in due) or "false"
        commit = [f"{self.m('u_cmd')} = {self.source(self.graph.outputs['u_dq'])};",
                  f"{self.m('command_theta')} = {self.m('theta')};"]
        return lines + (commit if condition == "true" else
                        [f"if ({condition}) {{", *["    " + line for line in commit], "}"])

    def derivatives(self) -> list[str]:
        return [self.d(*name.split(".", 1)) for name in self.ctrl.state_names]

    def _state(self, loop: str, local: str) -> str:
        if self.continuous:
            return self.model.value(self.ctrl.state, f"{loop}.{local}")
        return self.state_expression(self.ctrl, f"{loop}.{local}")

    def state_expression(self, part: Any, local: str) -> str:
        """Map one named state of the unit, its bridge, ADC or controller to generated storage."""
        if self.continuous or part is self.unit:
            protected = self.protection_state(local)
            if protected is not None:
                return protected
        if part is self.unit.adc:
            expression = self._adc_state_expression(local)
            if expression is not None:
                return expression
        startup = local.startswith("startup.") and local[8:] in self.ctrl.startup.get_state()
        if self.continuous:
            if startup:
                return self.m(local)
            raise KeyError(local)
        if part is self.bridge or part is self.unit.bridge:
            expression = self._timing_state_expression(local)
            if expression is not None:
                return expression
            bank, _, field = local.rpartition(".")
            word = self.m("shadow" if bank == "shadow" else "active")
            index = (*self.bridge.register_names, "on").index(field)
            return f"{word}[{index}]"
        if part is self.ctrl or part is self.unit.ctrl:
            if local.startswith("held."):
                public = local[5:]
                for key in self.graph.values:
                    if self.graph._state_port(key) == public:
                        return self.g(key)
            if startup:
                return self.m(local)
            loop_name, _, public = local.partition(".")
            node = self.graph.nodes.get(loop_name)
            if node is not None and public in node.get_state():
                return self.node(loop_name, public)
        raise KeyError(local)

    @staticmethod
    def _copy_block(block: Any) -> Any:
        """Copy a dynamic block without invoking factories implemented through ``__new__``."""
        cloned = object.__new__(type(block))
        if hasattr(block, "__dict__"):
            vars(cloned).update(vars(block))
        for name in _fields(block):
            if hasattr(block, name) and not hasattr(cloned, name):
                setattr(cloned, name, getattr(block, name))
        return cloned

    def _clone_loop(self, loop_name: str, node: Any, *, state_values: dict | None = None) -> Any:
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
            code = self._state(loop_name, public) if state_values is None else state_values[public]
            setattr(target, attribute, _Expression(code, kind))

        clone._state_blocks = {
            name: (cloned_owners[id(block)] if id(block) in cloned_owners
                   else self._copy_block(block))
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

        for name in node.carried:
            value = getattr(node, name)
            member = self.node(loop_name, name)
            setattr(clone, name, _Expression(member, _Expression.of(value).kind))
        return clone

    def _trace_loop(self, loop_name: str, node: Any) -> tuple[tuple[_Expression, ...],
                                                               tuple[str, ...], bool, bool]:
        inputs = tuple(_Expression(self.input(loop_name, port),
                                   "complex" if spec.complex_value else "real")
                       for port, spec in node.inputs.items())
        input_map = dict(zip(node.inputs, inputs))
        state_names = tuple(node.flow_state() if self.continuous else node.get_state())
        carried = () if self.continuous else node.carried
        internal_limit = self.continuous and loop_name in getattr(self.ctrl, "_flow_limits", {})
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
        targets = [self.g(loop_name + '.' + port) for port in node.outputs]
        state_names = tuple(node.flow_state() if self.continuous else node.get_state())
        targets += [self.d(loop_name, local) if self.continuous else self._state(loop_name, local)
                    for local in state_names]
        targets += [self.node(loop_name, name) for name in carried]
        if internal_limit:
            targets.append(self.node(loop_name, "limit_exceeded"))
        if external_limit:
            targets.append(self.m('command_saturated'))
        lines = self._assign_values(f"value_{loop_name}", zip(targets, values, strict=True))
        if self.continuous:
            return lines
        guard = self._sampled_due(loop_name)
        if guard == "true":
            return lines
        return [f"if ({guard}) {{", *["    " + line for line in lines], "}"]

    def _assign_values(self, prefix: str, assignments) -> list[str]:
        """Evaluate every new value before writing any destination, including aliased states."""
        return _assign_values(self.m(prefix), assignments)

    def _flow_values(self, *, members: bool, values=None) -> list[str]:
        """Only actual control-feedback outputs need warm-start storage across RHS calls."""
        cyclic = {key for keys in self.graph._flow_cycle_keys.values() for key in keys}
        values = self.graph.values if values is None else values
        return [f"{_CPP_TYPES[_Expression.of(value).kind]} "
                f"{self.g(key)} = {_literal(value)};"
                for key, value in values.items() if (key in cyclic) == members]

    def declarations(self) -> list[str]:
        """RHS locals of a continuous controller: acyclic outputs and loop-state derivatives."""
        lines = self._flow_values(members=False)
        for loop_name, node in self.graph.nodes.items():
            lines += [f"{_CPP_TYPES[_Expression.of(value).kind]} {self.d(loop_name, local)}{{}};"
                      for local, value in node.flow_state().items()]
        return lines

    def _flow_lines(self) -> list[str]:
        """Lower the same controller SCC plan for every continuous control boundary."""
        lines = []
        graph = self.graph
        for group in graph._continuous_plan:
            emitted = [line for name in group
                       for line in self._loop_lines(name, graph.nodes[name])]
            if group not in graph._continuous_cycles:
                lines.extend(emitted)
                continue
            keys = graph._flow_cycle_keys[group]
            lines += ["{", "bool control_converged = false;",
                      f"for (int control_sweep = 0; control_sweep < {graph._cycle_iterations}; ++control_sweep) {{"]
            lines += [f"    const auto previous_{i} = {self.g(key)};"
                      for i, key in enumerate(keys)]
            lines += ["    " + line for line in emitted]
            checks = [f"std::abs({self.g(key)} - previous_{i}) <= {_number(graph._cycle_atol)} + "
                      f"{_number(graph._cycle_rtol)} * std::max({{1.0, std::abs({self.g(key)}), "
                      f"std::abs(previous_{i})}})" for i, key in enumerate(keys)]
            lines += ["    if (" + " && ".join(checks or ["true"])
                      + ") { control_converged = true; break; }", "}",
                      "if (!control_converged) throw std::runtime_error("
                      + _cpp_string("continuous control algebraic loop did not converge: "
                                    + ", ".join(group)) + ");", "}"]
        return lines

    def _reset_integrators(self) -> list[str]:
        """Trace the loop's reset and commit declared state and carried values together."""
        lines: list[str] = []
        for loop_name, node in self.graph.nodes.items():
            state_names = tuple(node.get_state())
            state_values = {
                local: (self.model.state_value((id(self.ctrl.state), f"{loop_name}.{local}"))
                        if self.continuous else self._state(loop_name, local))
                for local in state_names
            }

            def run():
                clone = self._clone_loop(loop_name, node, state_values=state_values)
                clone.reset_integrator()
                state = clone.get_state()
                return (tuple(state[name] for name in state_names)
                        + tuple(getattr(clone, name) for name in node.carried))

            values = tuple(self.model._trace(run))
            assignments = self._state_assignments(loop_name, state_values, values)
            for name, value in zip(node.carried, values[len(state_names):]):
                current = self.node(loop_name, name)
                if value.code != current:
                    assignments.append((current, value))
            lines += self._assign_values(f"reset_{loop_name}", assignments)
        return lines

    def _state_assignments(self, loop_name, state_values, values):
        """Commit traced loop state through the existing sampled or ODE storage layout."""
        assignments = []
        for (local, current), value in zip(state_values.items(), values):
            if value.code == current:
                continue
            if self.continuous:
                index, is_complex = self.model.state_index[id(self.ctrl.state), f"{loop_name}.{local}"]
                assignments += ([(f"y[{index}]", value.real), (f"y[{index + 1}]", value.imag)]
                                if is_complex else [(f"y[{index}]", value)])
            else:
                assignments.append((current, value))
        return assignments

    def tracking_method(self):
        """One lowering of the existing tracking operation for cold starts and reconnects."""
        if not hasattr(self.ctrl, "track"):
            return ""
        unit = copy.copy(self.unit)
        unit.cfg, unit.ctrl = self.cfg, self.ctrl
        channels = {key: self.model.observed(value)
                    for key, value in self.model.unit_measurements(self.unit).items()}
        measurements = self.model._trace(lambda: unit.control_inputs_from(channels))
        outputs = self._tracked_outputs()
        values = dict(self.graph.values)
        held_outputs = {}
        for target, source in outputs:
            if source in values:
                held_outputs.setdefault(source, _Expression(target, _Expression.of(values[source]).kind))
        values.update(held_outputs)
        cyclic = ({key for keys in self.graph._flow_cycle_keys.values() for key in keys}
                  if self.continuous else set())
        lines, tracked = [], []
        for name in self.graph.order:
            node = self.graph.nodes[name]
            if not hasattr(node, "track"):
                continue
            inputs = {}
            for port, spec in node.inputs.items():
                source = self.graph.connections[f"{name}.{port}"]
                if (self.continuous and source in self.graph.values
                        and source not in cyclic and source not in held_outputs):
                    values[source] = self.model.observed(_Expression(
                        self.source(source), "complex" if spec.complex_value else "real"))
                code = (measurements[source].code if source.startswith("meas.")
                        else self.source(source))
                inputs[port] = _Expression(code, "complex" if spec.complex_value else "real")
            state_values = {
                local: (self.model.state_value((id(self.ctrl.state), f"{name}.{local}"))
                        if self.continuous else self._state(name, local))
                for local in node.get_state()
            }

            def run():
                clone = self._clone_loop(name, node, state_values=state_values)
                out = clone.track(inputs)
                return (*[out[port] if out is not None else _Expression(self.g(f"{name}.{port}"),
                             "complex" if spec.complex_value else "real")
                          for port, spec in node.outputs.items()],
                        *clone.get_state().values(),
                        *[getattr(clone, field) for field in node.carried], out is not None)

            traced = tuple(self.model._trace(run))
            n, m = len(node.outputs), len(state_values)
            assignments = [(self.g(f"{name}.{port}"), value)
                           for port, value in zip(node.outputs, traced[:n])]
            assignments += self._state_assignments(name, state_values, traced[n:n + m])
            assignments += [(self.node(name, field), value)
                            for field, value in zip(node.carried, traced[n + m:-1])]
            flag = self.m("tracked_" + name)
            lines.append(f"bool {flag} = false;")
            lines += self._assign_values("track_" + name, (*assignments, (flag, traced[-1])))
            tracked.append(flag)
        if tracked:
            lines += [f"if ({' || '.join(tracked)}) {{",
                      *[f"{target} = {self.source(source)};" for target, source in outputs], "}"]
        lines += [f"if ({self.m('startup_run')}) {{", *self._reset_integrators(), "}"]
        locals_ = self._flow_values(members=False, values=values) if self.continuous else []
        return (f"void track_{self.tag}() {{\n"
                + "\n".join("    " + line for line in (*locals_, *lines)) + "\n}")

    def _tracked_outputs(self):
        return []

    def limit_observation(self) -> list[str]:
        return []

    def _adc_source(self, channel: str) -> str:
        return f"obs_{_identifier(self.name + '.adc.' + channel)}"

    def _adc_reading(self, channel: str) -> str:
        adc = self.adc
        return (self.m("adc_held_" + channel)
                if adc is not None and adc.independent else self._adc_source(channel))

    def _adc_mean(self, channel: str) -> str | None:
        adc = self.adc
        assert adc is not None
        if channel in adc.channels:
            return f"({self.m('adc_acc_' + channel)} / {_number(adc.length)})"
        return None

    def _adc_value(self, channel: str) -> str:
        """Value presented to a controller at its interrupt."""
        return self._adc_mean(channel) or self._adc_reading(channel)

    def _adc_conversion(self, channel: str) -> str:
        """Value produced by a conversion which itself closes the averaging window."""
        return self._adc_mean(channel) or self._adc_source(channel)

    def _adc_members(self) -> list[str]:
        adc = self.adc
        lines = []
        if adc is not None and adc.independent:
            held = self.name not in self.model.simulation._windows_given
            index = adc._start_index(self.p.simulation.initial.t, held)
            lines.append(f"long long {self.m('adc_clock_index')} = static_cast<long long>({_number(index)});")
            for channel, value in adc.values.items():
                lines.append(f"{_CPP_TYPES[_Expression.of(value).kind]} "
                             f"{self.m('adc_held_' + channel)} = {_literal(value)};")
        if adc is not None:
            # Oversample scheduling and history layout are specialized; numerical periods may
            # change together, but not the number of samples represented by this executable.
            round(adc.period / adc.sample_period)
        if adc is None or (not adc.accumulating and (adc.independent or adc.samples == 1)):
            return lines
        if adc.accumulating:
            lines.append(
                f"double {self.m('adc_integration_time')} = "
                f"{_number(adc.integration_time)};"
            )
        if not adc.independent:
            lines.append(f"long long {self.m('adc_n_samp')} = {int(adc.n_samp)};")
        lines += [f"bool {self.m('adc_window_open')} = "
                 f"{'true' if adc.window_open else 'false'};"]
        for channel in adc.channels:
            value = adc.accumulated[channel]
            kind = _CPP_TYPES[_Expression.of(value).kind]
            lines += [
                f"{kind} {self.m('adc_acc_' + channel)} = {_literal(value)};",
                f"{kind} {self.m('adc_last_' + channel)} = "
                f"{_literal(adc._last[channel])};",
            ]
        for channel, window in adc.rms.items():
            round(self.cfg.meas.rms[channel] / adc.period)
            member = self.m("adc_rms_" + channel)
            buf = ", ".join(_number(value) for value in window.window.buf)
            lines.append(f"RMSWindow<{window.window.n}> {member}{{"
                         f"{{{{{buf}}}, {window.window.idx}, {window.count}}}, "
                         f"{_number(window.area)}, {_number(window.duration)}, "
                         f"{_number(window.last)}, {_number(window.total)}}};")
        return lines

    def _rms_state(self, local: str) -> str:
        channel, _, field = local.removeprefix("rms_").partition(".")
        member = self.m("adc_rms_" + channel)
        if field.startswith("sample_"):
            return f"{member}.window.buf[{int(field[7:])}]"
        if field in ("index", "count"):
            return f"{member}.window.{field}"
        return f"{member}.{field}"

    def adc_initialization_lines(self) -> list[str]:
        """Reuse ADC startup with the retained timer and freshly seeded physical readings."""
        adc = self.adc
        if (adc is None or (not self.model.runtime.selected and not adc.independent)
                or self.continuous and not adc.clocked
                or not adc.accumulating and (adc.independent or adc.samples == 1)):
            return []
        clone = copy.copy(adc)
        restored = self.name in self.model.simulation._windows_given
        clone.accumulated, clone._last = {}, {}
        for channel in adc.channels:
            kind = "complex" if isinstance(adc._zero[channel], complex) else "real"
            clone.accumulated[channel] = (_Expression(self.m('adc_acc_' + channel), kind)
                                          if restored else adc._zero[channel])
            clone._last[channel] = _Expression(self.m('adc_last_' + channel), kind)
        bridge = self.bridge
        t0 = self.p.simulation.initial.t
        if adc.clocked:
            clone.start(t0, held=not restored)
        else:
            index = bridge.control_clock.index_after(t0)
            clone.start(t0, bridge.interrupt(index - 1), bridge.interrupt(index), held=not restored)
        return [*([f"{self.m('adc_n_samp')} = {clone.n_samp};"] if not adc.independent else []),
                f"{self.m('adc_window_open')} = {_literal(clone.window_open)};",
                *(f"{self.m('adc_acc_' + channel)} = {_literal(value)};"
                  for channel, value in clone.accumulated.items())]

    def _rms_value(self, channel: str) -> str:
        return f"{self.m('adc_rms_' + channel)}.value()"

    def _rms_square(self, channel: str, *, direct: bool = False) -> str:
        reading = self.model.unit_measurements(self.unit)[channel]
        value = reading if direct else _Expression(self._adc_source(channel), reading.kind)
        return self.model._trace(lambda: re_product(value, value)).code

    def adc_seed_lines(self) -> list[str]:
        adc = self.adc
        if adc is None or not adc.accumulating:
            return []
        readings = self.model.unit_measurements(self.unit)
        return [
            *(f"{self.m('adc_last_' + channel)} = {readings[channel].code};"
              for channel in adc.channels),
            *(f"{self.m('adc_rms_' + channel)}.last = "
              f"{self._rms_square(channel, direct=True)};" for channel in adc.rms),
        ]

    def _rms_sample_lines(self, *, due=None) -> list[str]:
        adc = self.adc
        if adc is None or not adc.rms:
            return []
        lines = [f"{self.m('adc_rms_' + channel)}.sample({_number(adc.period)});"
                 for channel in adc.rms]
        if self.unit.continuous:
            period = _number(adc.period)
            due = due or f"std::abs(t - std::round(t / {period}) * {period}) < time_eps"
            return [f"const bool rms_fresh = {due};",
                    "if (rms_fresh) {",
                    *["    " + line for line in lines], "}"]
        return lines

    def _rms_protection_argument(self) -> str:
        adc = self.adc
        return ", rms_fresh" if self.unit.continuous and adc is not None and adc.rms else ""

    def adc_methods(self) -> str:
        adc = self.adc
        if adc is None or not adc.accumulating:
            return ""
        lines = [f"void adc_accumulate_{self.tag}(double t) {{",
                 f"    const double dt = t - {self.m('adc_integration_time')};",
                 "    if (dt <= 0.0) return;"]
        for channel in adc.channels:
            reading = self._adc_source(channel)
            lines += [
                f"    const auto now_{channel} = {reading};",
                f"    if ({self.m('adc_window_open')}) "
                f"{self.m('adc_acc_' + channel)} += 0.5 * dt * "
                f"({self.m('adc_last_' + channel)} + now_{channel});",
                f"    {self.m('adc_last_' + channel)} = now_{channel};",
            ]
        for channel in adc.rms:
            member = self.m("adc_rms_" + channel)
            lines.append(f"    {member}.accumulate(dt, {self._rms_square(channel)});")
        lines.append(f"    {self.m('adc_integration_time')} = t;")
        return "\n".join([*lines, "}"])

    def adc_accumulate_call(self, t: str) -> str:
        adc = self.adc
        return (f"adc_accumulate_{self.tag}({t});"
                if adc is not None and adc.accumulating else "")

    def _adc_next_values(self) -> list[str]:
        adc = self.adc
        if adc is None:
            return []
        if adc.clocked:
            interrupt = (f"({_number(adc.sampling_clock.offset)} + "
                         f"{self.m('adc_clock_index')} * {_number(adc.sample_period)})")
            return [interrupt, *([
                f"({self.m('adc_window_open')} ? std::numeric_limits<double>::infinity() : "
                f"{interrupt} - {_number(adc.length)})"
            ] if adc.averaging and not adc._full else [])]
        values: list[str] = []
        if adc.independent:
            values.append(
                f"({_number(adc.sampling_clock.offset)} + {self.m('adc_clock_index')} * "
                f"{_number(adc.sample_period)})"
            )
        if adc.averaging and not adc._full:
            values.append(
                f"({self.m('adc_window_open')} ? std::numeric_limits<double>::infinity() : "
                f"{self.m('t_interrupt')} - {_number(adc.length)})"
            )
        if adc.samples > 1 and not adc.independent:
            values.append(
                f"({self.m('adc_n_samp')} < {adc.samples} ? "
                f"{self.m('t_interrupt')} - {_number(self.bridge.period)} + "
                f"{self.m('adc_n_samp')} * {_number(adc.sample_period)} : "
                "std::numeric_limits<double>::infinity())"
            )
        return values

    def _adc_sense_lines(self) -> list[str]:
        adc = self.adc
        if adc is None or (adc.samples == 1 and not adc.independent):
            return []
        if adc.independent:
            interrupt = (f"({_number(adc.sampling_clock.offset)} + "
                         f"{self.m('adc_clock_index')} * {_number(adc.sample_period)})")
            return [
                f"const bool adc_due = std::abs(t - {interrupt}) < time_eps;",
                "if (adc_due) {",
                *[f"    {self.m('adc_held_' + channel)} = {self._adc_source(channel)};"
                  for channel in adc.values],
                f"    ++{self.m('adc_clock_index')};",
                "}",
            ]
        return [
            f"if (!control_due && {self.m('adc_n_samp')} < {adc.samples}) {{",
            f"    const double sample_at = {self.m('t_interrupt')} "
            f"- {_number(self.bridge.period)} + {self.m('adc_n_samp')} "
            f"* {_number(adc.sample_period)};",
            f"    if (std::abs(t - sample_at) < time_eps) ++{self.m('adc_n_samp')};",
            "}",
        ]

    def _adc_actuate_lines(self, due="control_due", interrupt=None) -> list[str]:
        adc = self.adc
        if adc is None:
            return []
        interrupt = self.m('t_interrupt') if interrupt is None else interrupt
        lines: list[str] = []
        if adc.averaging:
            lines += [
                f"if ({due}) {{",
                *[f"    {self.m('adc_acc_' + channel)} = {_literal(adc._zero[channel])};"
                  for channel in adc.channels],
                f"    {self.m('adc_window_open')} = {'true' if adc._full else 'false'};",
                *([f"    {self.m('adc_n_samp')} = 1;"] if not adc.independent else []),
                "}",
            ]
            if not adc._full:
                lines += [
                    f"const double window_at = {interrupt} - {_number(adc.length)};",
                    f"if (!{self.m('adc_window_open')} && "
                    "std::abs(t - window_at) < time_eps) {",
                    *[f"    {self.m('adc_acc_' + channel)} = "
                      f"{_literal(adc._zero[channel])};"
                      for channel in adc.channels],
                    f"    {self.m('adc_window_open')} = true;",
                    "}",
                ]
        elif adc.samples > 1 and not adc.independent:
            lines.append(f"if ({due}) {self.m('adc_n_samp')} = 1;")
        return lines

    def controller_summary(self, stream: str = "summary") -> list[str]:
        return []

    def summary_lines(self, stream: str = "summary") -> list[str]:
        return [*self.protection_summary(stream), *self.controller_summary(stream)]

    def _topology_expression(self, method: Any, values: tuple[str, ...]) -> str:
        result = self.model._trace(
            lambda: method(tuple(_Expression(value, "real") for value in values))
        )
        return result.code

    def shared_code(self):
        return (*(self.modulation.shared_code() if self.modulation is not None else ()),
                *((_SHORT_INTERVALS, _PULSE_SEQUENCE_KERNEL) if self.bridge.pulse_update is not None else ()))

    def _pulse_members(self):
        pulse = self.bridge.pulse_update
        if pulse is None:
            return []
        literal = lambda values: "{" + ", ".join(_number(x) for x in values) + "}"
        groups = [next(k for k, group in enumerate(pulse.groups) if index in group)
                  for index in range(len(pulse.entry))]
        return [f"PulseSequenceState<{len(pulse.entry)}, {len(pulse.groups)}> "
                f"{self.m('pulse_sequence')}{{{{{', '.join(map(str, groups))}}}, {literal(pulse.entry)}, "
                f"{literal(pulse.next)}, {literal(pulse.remaining)}, {literal(pulse.next_remaining)}, "
                f"{_number(pulse.time) if pulse.time is not None else '-std::numeric_limits<double>::infinity()'}, "
                f"{str(pulse.restored).lower()}" + "};"]

    def _timing_state_expression(self, local):
        if self.bridge.pulse_update is None or not local.startswith("pulse."):
            return self.modulation.state_expression(self, local)
        field, _, index = local.removeprefix("pulse.").rpartition("_")
        member = "entry" if field == "q" else "remaining"
        return f"{self.m('pulse_sequence')}.{member}[{int(index)}]"

    def _load_modulation(self, hold, switching):
        return self.modulation.load(self, hold, switching)

    def switching_method(self):
        return self.modulation.method(self)

    @property
    def width(self) -> int:
        return len(self.bridge.register_names)

    def _pwm_members(self) -> list[str]:
        """Declare the shared sampled bridge timer, register and edge state."""
        bridge = self.bridge
        if any(value != self.topology.zero for _, value in bridge.schedule):
            # Restored register values may already have produced a fixed initial edge sequence.
            # Its constructor inputs must remain in that specialization until it is re-exported.
            for name in inspect.signature(type(bridge).__init__).parameters:
                value = getattr(bridge, name, None)
                if isinstance(value, _InitializationExpression):
                    bool(value == value.default)
        t0 = self.p.simulation.initial.t
        control_index = bridge.control_clock.index_after(t0)
        load_index = bridge.load_clock.index_after(t0)
        q_kind = _CPP_TYPES[_Expression.of(self.topology.zero).kind]
        schedule = ", ".join(
            "{" + _number(at) + ", " + _literal(q) + "}" for at, q in bridge.schedule
        )
        return [
            *self.modulation.members(self),
            *self._pulse_members(),
            f"long long {self.m('pwm_k')} = static_cast<long long>({_number(control_index)});",
            f"long long {self.m('pwm_j')} = static_cast<long long>({_number(load_index)});",
            f"double {self.m('t_interrupt')} = {_number(bridge.interrupt(control_index))};",
            f"double {self.m('t_load')} = {_number(bridge.load_time(load_index))};",
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
            *self.modulation.write_sync(self),
            f"        ++{self.m('pwm_k')}; {self.m('t_interrupt')} = "
            f"{_number(bridge.control_clock.offset)} + {self.m('pwm_k')} * "
            f"{_number(bridge.period)};",
        ]

    def _pwm_load_lines(self, hold: str, switching: bool, gates: str,
                        tail: tuple[str, ...] = ()) -> list[str]:
        bridge = self.bridge
        return [
            "    if (load_due) {",
            f"        const double completed = {_number(bridge.control_clock.offset)} + "
            f"({self.m('pwm_k')} - 1) * {_number(bridge.period)} + "
            f"{_number(bridge.computation)};",
            f"        if (completed <= t + time_eps) {self.m('active')} = {self.m('shadow')};",
            *self._load_modulation(hold, switching),
            f"        ++{self.m('pwm_j')}; {self.m('t_load')} = "
            f"{_number(bridge.load_clock.offset)} + "
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

    _FIRST_TIMES = tuple(item.name for item in fields(ProtectionStats) if item.name.endswith("_first_t"))

    def _observes_rocof(self) -> bool:
        return "frequency" in unit_type(self.cfg.type).protection_criteria

    def unit_members(self) -> list[str]:
        """Runtime state owned by the common non-composite Unit contract."""
        unit = self.unit
        trip = unit.protection.trip
        variable = f"units.{self.name}.connected" in self.model.runtime.by_path
        connected = self.initial_connection() if variable else _literal(unit.breaker)
        gates = _literal(unit.gates)
        if variable:
            gates = self.initial_run() if self.continuous else _literal(unit.bridge.on)
            if self.continuous or unit.branch_f is None:
                gates = f"({gates}) && {self.m('breaker')}"
            gates = f"({gates}) && !{self.m('tripped')}"
        return [
            f"bool {self.m('tripped')} = {'true' if unit.tripped else 'false'};",
            f"bool {self.m('fault')} = {'true' if unit.protection.fault else 'false'};",
            f"double {self.m('trip_time')} = "
            f"{_number(trip.t if trip is not None else math.nan)};",
            f"std::string {self.m('trip_cause')} = "
            f"{_cpp_string(trip.cause if trip is not None else '')};",
            f"double {self.m('max_current')} = "
            f"{_number(unit.protection.stats.max_current_pu)};",
            *[f"double {self.m('prot_' + key)} = {_number(value)};"
              for key, value in unit.protection.get_state().items()],
            *self.protection_members(),
            f"bool {self.m('breaker')} = ({connected}) && !{self.m('tripped')};",
            f"bool {self.m('gates')} = {gates};",
        ]

    def initial_connection(self) -> str:
        """The initial command, with past event writes taking their normal precedence."""
        path = f"units.{self.name}.connected"
        if any(path in change.connections and change.t <= self.p.simulation.initial.t + TIME_EPS
               for change in self.p.changes):
            return _literal(self.unit._initially_connected)
        return self.model.runtime.value(path, self.unit._initially_connected)

    def initial_run(self) -> str:
        if self.ctrl is None:
            return f"({self.initial_connection()}) && !{self.m('tripped')}"
        if (f"units.{self.name}.connected" not in self.model.runtime.by_path
                or f"{self.name}.ctrl.startup.run" in self.p.simulation.initial.states):
            return _literal(self.ctrl.startup.run)
        return f"({self.initial_connection()}) && !{self.m('tripped')}"

    def protection_state(self, local: str) -> str | None:
        if local == "tripped":
            return self.m("tripped")
        if local == "fault":
            return self.m("fault")
        if local.startswith("prot."):
            return self.m("prot_" + local[5:])
        return None

    def _pending_trip_lines(self) -> list[str]:
        return [
            f"const bool tripped_now = {self.m('sampled_trip_due')};",
            f"if (tripped_now) {{ {self.m('sampled_trip_due')} = false; "
            f"apply_trip_{self.tag}(); }}",
        ]

    def _terminal_lines(self) -> list[str]:
        """Apply the generated equivalent of ``Unit._update_terminal`` to the filter branch."""
        branch = self.unit.branch_f
        if branch is None:
            return []
        return self.model.breaker_lines(
            branch, f"!({self.m('breaker')} && {self.m('gates')} && !{self.m('tripped')})")

    def actuate_method(self) -> str:
        """Apply a sampled trip for a continuous converter."""
        return f'''bool actuate_{self.tag}(double) {{
    if (!{self.m('sampled_trip_due')}) return false;
    {self.m('sampled_trip_due')} = false;
    apply_trip_{self.tag}();
    return true;
}}'''

    def protection_members(self) -> list[str]:
        protection = self.protection
        stats = protection.stats

        def observed(value: Any) -> str:
            return _number(math.nan if value is None else value)

        return [
            f"bool {self.m('sampled_trip_due')} = false;",
            *(f"double {self.m(name)} = {observed(getattr(stats, name))};" for name in self._FIRST_TIMES),
            f"double {self.m('rocof_max')} = {_number(stats.rocof_max)};",
            f"double {self.m('terminal_voltage_min')} = "
            f"{observed(stats.terminal_voltage_min_pu)};",
            f"double {self.m('terminal_voltage_max')} = "
            f"{observed(stats.terminal_voltage_max_pu)};",
            *([f"MovingWindow<{protection._rocof_count}> {self.m('rocof_window')};"]
              if self._observes_rocof() else []),
            *self.alarm_members(),
        ]

    def alarm_members(self) -> list[str]:
        return [f"std::vector<std::string> {self.m('alarms')}"
                "{" + ", ".join(_cpp_string(value) for value in self.protection.stats.alarms) + "};"]

    def _trip_body(self) -> list[str]:
        link = self.model._equation_attributes(self.unit.dclink)["tripped"].code
        lines = ([f"{self.m('startup_run')} = false;"] if self.ctrl is not None else []) + [
                 f"{self.m('gates')} = false;",
                 f"{self.m('breaker')} = false;"]
        if self.unit.branch_f is not None:
            lines += self._terminal_lines()
        elif not self.unit.continuous:
            lines.append(f"{self._hold()} = {_literal(self.unit.bridge.topology.zero)};")
        lines.append(f"{link} = true;")
        return lines

    def protection_method(self, *, sampled_only: bool = False) -> str:
        """Fast and sampled protection, plus the one physical trip action."""
        cfg = self.cfg.protection
        period = self.protection.T
        has_frequency = self._observes_rocof()

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
            f"bool sampled_protect_{self.tag}(double t, double terminal_voltage, "
            "double freq_dev, double vdc_error, bool startup_complete, bool rms_fresh = true) {",
        ]
        if has_frequency:
            sampled += [
                f"    if (!{self.m('tripped')}) {{",
                f"        const bool had_old = {self.m('rocof_window')}.count == {self.protection._rocof_count};",
                f"        const double old = {self.m('rocof_window')}.push(freq_dev);",
                "        if (had_old) {",
                f"            const double rocof = (freq_dev - old) / {_number(self.protection._rocof_window)};",
                f"            {self.m('rocof_max')} = std::max({self.m('rocof_max')}, std::abs(rocof));",
                f"            if ({enabled(cfg.rocof)} && std::abs(rocof) >= {limit(cfg.rocof, 'limit')} && startup_complete) {{",
                f"                if (!std::isfinite({first('rocof')})) {first('rocof')} = t;",
                f"                alarm_{self.tag}(\"ROCOF\");",
                "            }",
                "        }",
                "    }",
            ]
        sampled.append(f"    if ({self.m('tripped')} || !startup_complete) return false;")
        if cfg.rms_current.enable:
            channel = unit_type(self.cfg.type).current_channel
            rms = f"({self._rms_value(channel)} / {_number(self.protection.i_base)})"
            criterion = timed(
                cfg.rms_current,
                f"std::isfinite({rms}) && {rms} > {limit(cfg.rms_current, 'limit_pu')}",
                "rms_current", "RMS_OVERCURRENT", "hold_rms", "rms_current")
            if self.unit.continuous:
                criterion = ["if (rms_fresh) {",
                             *["    " + line for line in criterion], "}"]
            sampled += ["    " + line for line in criterion]
        sampled += ["    " + line for line in timed(
            cfg.dc_voltage,
            f"std::abs(vdc_error) >= {limit(cfg.dc_voltage, 'limit_pu')}",
            "dc_voltage", "VDC_BAND", "hold_vdc", "vdc")]
        sampled += ["    " + line for line in timed(
            cfg.frequency,
            f"std::abs(freq_dev) >= {limit(cfg.frequency, 'limit')}",
            "frequency", "FREQ_BAND", "hold_freq", "freq")]
        sampled += [
            f"    {self.m('terminal_voltage_min')} = "
            f"std::isfinite({self.m('terminal_voltage_min')}) ? "
            f"std::min({self.m('terminal_voltage_min')}, terminal_voltage) : terminal_voltage;",
            f"    {self.m('terminal_voltage_max')} = "
            f"std::isfinite({self.m('terminal_voltage_max')}) ? "
            f"std::max({self.m('terminal_voltage_max')}, terminal_voltage) : terminal_voltage;",
        ]
        sampled += ["    " + line for line in timed(
            cfg.undervoltage,
            f"terminal_voltage <= {limit(cfg.undervoltage, 'limit_pu')}",
            "undervoltage", "VOLTAGE_UNDER", "hold_uv", "voltage_uv")]
        sampled += ["    " + line for line in timed(
            cfg.overvoltage,
            f"terminal_voltage >= {limit(cfg.overvoltage, 'limit_pu')}",
            "overvoltage", "VOLTAGE_OVER", "hold_ov", "voltage_ov")]
        sampled += ["    return false;", "}"]

        apply = [f"void apply_trip_{self.tag}() {{", *["    " + line for line in self._trip_body()], "}"]
        sections = [self._protection_helpers(), apply, sampled]
        if not sampled_only:
            current = f"obs_{_identifier(self.name + '.i_c')}"
            current_width = len(self.unit._protection_currents())
            fast = [f"bool protect_{self.tag}(double t) {{"]
            if not self.unit.continuous:
                fast += [
                    f"    const bool edge = std::abs({self.m('next_switch')} - t) < time_eps || std::abs({self.m('load_end')} - t) < time_eps;",
                    "    if (!edge) return false;",
                ]
            if current_width == 3:
                fast += [
                    f"    const auto current = phases({current});",
                    "    const double peak_current = std::max({std::abs(current[0]), "
                    "std::abs(current[1]), std::abs(current[2])});",
                ]
            else:
                fast.append(f"    const double peak_current = std::abs({current});")
            fast += [
                f"    const double peak = peak_current / {_number(self.protection.i_base)};",
                f"    if (std::isfinite(peak)) {self.m('max_current')} = std::max({self.m('max_current')}, peak);",
                f"    if ({self.m('tripped')} || !{enabled(cfg.overcurrent)} || !(peak > {limit(cfg.overcurrent, 'limit_pu')})) return false;",
                f"    if (!std::isfinite({first('overcurrent')})) {first('overcurrent')} = t;",
                f"    alarm_{self.tag}(\"OVERCURRENT\");",
                f"    latch_{self.tag}(t, \"overcurrent\", true);",
                f"    apply_trip_{self.tag}();",
                "    return true;",
                "}",
            ]
            sections.append(fast)
        return "\n\n".join("\n".join(section) for section in sections)

    def _protection_helpers(self) -> list[str]:
        return [
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
            "    std::string alarm = \"TRIP_\";",
            "    for (const char c : cause) alarm += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));",
            f"    alarm_{self.tag}(alarm);",
            "    return true;",
            "}",
        ]

    def protection_summary(self, stream: str = "summary") -> list[str]:
        """C++ statements which append this unit's protection result to a JSON object."""
        lines = [
            f'{stream} << ",\\n  \\"{self.name}.tripped\\": " << ({self.m("tripped")} ? 1 : 0);',
            f'{stream} << ",\\n  \\"{self.name}.trip_time\\": "; if (std::isfinite({self.m("trip_time")})) {stream} << {self.m("trip_time")}; else {stream} << "null";',
            f'{stream} << ",\\n  \\"{self.name}.trip_cause\\": "; if ({self.m("trip_cause")}.empty()) {stream} << "null"; else {stream} << \'"\' << {self.m("trip_cause")} << \'"\';',
            f'{stream} << ",\\n  \\"{self.name}.max_current_pu\\": " << {self.m("max_current")};',
            f'{stream} << ",\\n  \\"{self.name}.rocof_max\\": " << {self.m("rocof_max")};',
        ]
        for name in self._FIRST_TIMES:
            value = self.m(name)
            lines.append(
                f'{stream} << ",\\n  \\"{self.name}.{name}\\": "; '
                f'if (std::isfinite({value})) {stream} << {value}; else {stream} << "null";'
            )
        for key, value in (
            ("terminal_voltage_min_pu", self.m("terminal_voltage_min")),
            ("terminal_voltage_max_pu", self.m("terminal_voltage_max")),
        ):
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
    """Shared AC initialization and summaries for sampled and continuous control."""

    def _tracked_outputs(self):
        return [(self.m('theta'), self.graph.outputs['theta']),
                (self.m('omega'), self.graph.outputs['omega']),
                *([] if self.continuous else
                  [(self.m('command_theta'), self.graph.outputs['theta'])])]

    def _frequency_deviation(self) -> str:
        return f"({self.m('omega')} - {_number(self.cfg.base.w0)}) / (2.0 * pi)"

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

    def declarations(self) -> list[str]:
        return [f"Complex {self.m('meas_u_g')}{{}};", f"Complex {self.m('meas_i_c')}{{}};",
                f"double {self.m('meas_u_dc')}{{}};", f"double {self.m('startup_value')}{{}};",
                *super().declarations()]

    def members(self) -> list[str]:
        startup = self.ctrl.startup
        lines = [
            *self.unit_members(),
            *self._flow_values(members=True),
            *self._loop_members(),
            f"bool {self.m('startup_run')} = {self.initial_run()};",
            f"double {self.m('startup_ramp')} = {_number(startup.ramp)};",
            f"double {self.m('startup_start')} = {_number(startup.start)};",
            f"bool {self.m('track_due')} = false;",
            f"double {self.m('theta')} = {_number(self.ctrl.theta)};",
            f"double {self.m('omega')} = {_number(self.ctrl.omega)};",
            f"Complex {self.m('u_cmd')} = {_complex(self.ctrl.u_cmd)};",
            *[f"double {self.m('log_' + key)} = std::numeric_limits<double>::quiet_NaN();"
              for key in self.ctrl.log_names()],
        ]
        lines += self._adc_members()
        return lines

    def limit_observation(self) -> list[str]:
        """Observe ideal-controller limits once per output snapshot, outside the RHS."""
        lines: list[str] = []
        for loop_name, (_path, alarm, _node) in self.ctrl._flow_limits.items():
            checks = self.node(loop_name, "n_updates")
            exceeded = self.node(loop_name, "n_limit_exceeded")
            first = self.node(loop_name, "first_limit_t")
            active = self.node(loop_name, "limit_exceeded")
            lines += [
                f"++{checks};",
                f"if ({active}) {{",
                f"    ++{exceeded};",
                f"    if (!std::isfinite({first})) {first} = t;",
                f"    alarm_{self.tag}({_cpp_string(alarm)});",
                "}",
            ]
        return lines

    def output_lines(self) -> list[str]:
        lines: list[str] = [
            f"{self.m('meas_u_g')} = {self.model.value(self.ctrl.inp, 'u_g')} / {_number(self.cfg.base.v_phase_peak)};",
            f"{self.m('meas_i_c')} = {self.model.value(self.ctrl.inp, 'i_c')} / {_number(self.cfg.base.i_phase_peak)};",
            f"{self.m('meas_u_dc')} = {self.model.value(self.ctrl.inp, 'u_dc')} / {_number(self.cfg.dc_base.v)};",
            self._startup_value_line(),
        ]
        lines += self._flow_lines()
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
        frame = self.source(self.ctrl._frame_path)
        current = f"({self.m('meas_i_c')} * Complex{{std::cos({frame}), -std::sin({frame})}})"
        voltage = f"({self.m('meas_u_g')} * Complex{{std::cos({frame}), -std::sin({frame})}})"
        log_i, log_v = self.m("log_i_value"), self.m("log_v_value")
        lines = [f"const Complex {log_i} = {current};", f"const Complex {log_v} = {voltage};",
                 f"{self.m('log_id_pu')} = std::real({log_i});", f"{self.m('log_iq_pu')} = std::imag({log_i});",
                 f"{self.m('log_vd_pu')} = std::real({log_v});", f"{self.m('log_vq_pu')} = std::imag({log_v});",
                 f"{self.m('log_vac_pu')} = std::abs({log_v});", f"{self.m('log_vdc_pu')} = {self.m('meas_u_dc')};",
                 f"{self.m('log_in_service')} = {self.m('startup_run')} ? 1.0 : 0.0;",
                 f"{self.m('log_freq_dev')} = {self._frequency_deviation()};",
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
        ]
        lines += self._terminal_lines()
        return lines

    def sense_method(self) -> str:
        observed = f"obs_{_identifier(self.name + '.u_g')}"
        dc = f"obs_{_identifier(self.name + '.u_dc')}"
        if self.cfg.meas.sources:
            observed, dc = self._adc_reading("u_g"), self._adc_reading("u_dc")
        lines = [f"bool sense_{self.tag}(double t) {{",
                 "    bool changed = false;",
                 *["    " + line for line in self._rms_sample_lines()],
                 f"    if ({self.m('track_due')}) {{"]
        lines += [f"        track_{self.tag}();", "        changed = true;"]
        lines += [f"        {self.m('track_due')} = false;", "    }",
                  f"    sampled_protect_{self.tag}(t, std::abs({observed}) / {_number(self.cfg.base.v_phase_peak)},",
                  f"        {self._frequency_deviation()},",
                  f"        {dc} / {_number(self.cfg.dc_base.v)} - {self._reference('vdc_ref_pu')},",
                  f"        {self._startup_complete()}{self._rms_protection_argument()});",
                  "    return changed;",
                  "}"]
        return "\n".join(lines)

class _UnitGenerator(_ACControlGenerator):
    """Lower one sampled controller/PWM unit to allocation-free specialised methods."""

    def __init__(self, name: str, unit: Any, model: _ModelGenerator) -> None:
        super().__init__(name, unit, model, "u")
        self.continuous = False
        self.bridge = model.runtime.object(unit.bridge)
        self.topology = self.bridge.topology

    def _hold(self) -> str:
        return self.model.zoh_fields[id(self.bridge.inp), "q"]

    def members(self) -> list[str]:
        unit, ctrl, graph = self.unit, self.ctrl, self.graph
        bridge = unit.bridge
        lines = [
            *self.unit_members(),
            f"bool {self.m('startup_run')} = {self.initial_run()};",
            f"double {self.m('startup_ramp')} = {_number(ctrl.startup.ramp)};",
            f"long long {self.m('startup_steps')} = {int(ctrl.startup.steps)};",
            f"double {self.m('startup_value')} = {_number(ctrl.startup.value)};",
            f"bool {self.m('startup_complete')} = {'true' if ctrl.startup.complete else 'false'};",
            f"double {self.m('theta')} = {_number(ctrl.theta)};",
            f"double {self.m('omega')} = {_number(ctrl.omega)};",
            f"Complex {self.m('u_cmd')} = {_complex(ctrl.u_cmd)};",
            f"double {self.m('command_theta')} = {_number(ctrl.command_theta)};",
            f"std::array<double, {self.width}> {self.m('pending_d')}{{{', '.join(_number(x) for x in bridge.shadow[:self.width])}}};",
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
            lines.append(f"{_CPP_TYPES[_Expression.of(value).kind]} {self.g(key)} = {_literal(value)};")
        lines += self._loop_members()
        lines += self._pwm_members()
        return lines

    def state_expression(self, part: Any, local: str) -> str:
        held = {"command.u_dq_pu": "u_cmd", "command.theta": "command_theta", "command.omega": "omega"}
        if local in held and (part is self.ctrl or part is self.unit.ctrl):
            return self.m(held[local])
        return super().state_expression(part, local)

    def controller_method(self) -> str:
        lines = [f"void controller_{self.tag}(double t) {{"]
        lines += ["    " + line for line in self._sampled_measurement_lines()]
        lines += ["    " + line for line in self._sampled_loop_lines()]
        lines += ["    " + line for line in self._sampled_vsc_command_lines()]
        lines += self._modulation_lines(indent="    ")
        lines += self._log_lines("    ")
        lines += ["    " + line for line in self._rms_sample_lines()]
        lines += [f"    sampled_protect_{self.tag}(t, std::abs(meas_u_g),",
                  f"        {self._frequency_deviation()},",
                  f"        meas_u_dc - {self._reference('vdc_ref_pu')}, {self.m('startup_complete')});",
                  f"    {self.m('pending_on')} = {self.m('startup_run')};",
                  f"    {self.m('pending_valid')} = true;",
                  f"    {self.m('last_ctrl_t')} = t;",
                  "}"]
        return "\n".join(lines)

    def _log_lines(self, indent: str) -> list[str]:
        frame = self.source(self.ctrl._frame_path)
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
                 f"{indent}{self.m('log_freq_dev')} = {self._frequency_deviation()};",
                 f"{indent}{self.m('log_angle_rel')} = std::remainder({self.m('theta')} - {_number(self.cfg.base.w0)} * t, 2.0 * pi);"]
        lines += self._mode_log_lines(indent)
        return lines

    def _modulation_lines(self, indent: str) -> list[str]:
        command = self.m("u_cmd")
        theta = self.m("command_theta")
        udc = f"obs_{_identifier(self.name + '.u_dc')}"
        modulation, output_saturated = self._modulation_values(command, theta, udc)
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
        lines += [f"{indent}{self.m('pending_d')}[{index}] = {self.source(self.graph.outputs[name])};"
                  for index, name in enumerate(self.bridge.register_names[3:], 3)]
        return lines

    def event_methods(self) -> str:
        bridge = self.bridge
        hold = self.model.zoh_fields[id(bridge.inp), "q"]
        switching = not isinstance(bridge.modulator, ZOH)
        next_values = [
            self.m("t_interrupt"), self.m("t_load"), self.m("next_switch"),
            *self._adc_next_values(),
        ]
        next_expression = "std::min({" + ", ".join(next_values) + "})"
        lines = [
            f"double next_{self.tag}() const noexcept {{ return {next_expression}; }}",
            f"bool sense_{self.tag}(double t) {{",
            f"    const bool control_due = std::abs(t - {self.m('t_interrupt')}) < time_eps;",
        ]
        lines += ["    " + line for line in self._adc_sense_lines()]
        lines += [
            f"    if (control_due) controller_{self.tag}(t);",
            "    return false;",
            "}",
            f"bool actuate_{self.tag}(double t) {{",
            *["    " + line for line in self._pending_trip_lines()],
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
            tuple("        " + line for line in self._terminal_lines()),
        )
        if bridge.computation > 0.0:
            lines += [f"    if (control_due && {self.m('pending_valid')}) {{",
                      *write, "    }"]
        if switching:
            lines += self._pwm_switch_lines(hold)
        lines += ["    " + line for line in self._adc_actuate_lines()]
        lines += [f"    if (control_due) {self.m('pending_valid')} = false;",
                  "    return tripped_now;", "}"]
        return "\n".join(lines)

    def command_lines(self, on: bool, ramp: float) -> list[str]:
        effective = self._effective_command(on)
        lines = [f"{self.m('breaker')} = {effective};"]
        lines += self._controller_command_lines(on, ramp)
        lines += self._terminal_lines()
        return lines

class _ElectricalUnitGenerator(_ProtectionGenerator, _ControlGenerator):
    """Unit lifecycle and owned ADC timing, independent of a control graph."""

    continuous = True

    def __init__(self, name, unit, model):
        super().__init__(name, unit, model, "u")
        self.bridge = model.runtime.object(unit.bridge)
        self.topology = self.bridge.topology

    def members(self):
        return [*self.unit_members(), *self._adc_members()]

    def state_expression(self, _part: Any, local: str) -> str:
        protected = self.protection_state(local)
        if protected is not None:
            return protected
        expression = self._adc_state_expression(local)
        if expression is not None:
            return expression
        raise KeyError(local)

    def sense_method(self):
        lines = [f"bool sense_{self.tag}(double t) {{"]
        adc = self.adc
        if adc is not None and adc.clocked:
            interrupt = (f"({_number(adc.sampling_clock.offset)} + "
                         f"{self.m('adc_clock_index')} * {_number(adc.sample_period)})")
            lines += [f"    const bool adc_due = std::abs(t - {interrupt}) < time_eps;",
                      "    if (adc_due) {",
                      *[f"        {self.m('adc_held_' + channel)} = {self._adc_conversion(channel)};"
                        for channel in adc.values],
                      f"        ++{self.m('adc_clock_index')};", "    }"]
        lines += self._rms_sample_lines(due="adc_due" if adc is not None and adc.clocked else None)
        voltage = f"obs_{_identifier(self.name + '.u_g')}"
        dc = f"obs_{_identifier(self.name + '.u_dc')}"
        lines += [f"    sampled_protect_{self.tag}(t, std::abs({voltage}) / {_number(self.cfg.dc_base.v)}, "
                  f"0.0, {dc} / {_number(self.cfg.dc_base.v)} - 1.0, true{self._rms_protection_argument()});",
                  "    return false;", "}"]
        return "\n".join(lines)

    def actuate_method(self):
        lines = [f"bool actuate_{self.tag}(double t) {{", *self._pending_trip_lines()]
        adc = self.adc
        if adc is not None and adc.clocked:
            interrupt = (f"({_number(adc.sampling_clock.offset)} + "
                         f"{self.m('adc_clock_index')} * {_number(adc.sample_period)})")
            lines += [f"const bool adc_due = std::abs(t - ({interrupt} - {_number(adc.sample_period)})) < time_eps;",
                      *self._adc_actuate_lines("adc_due", interrupt)]
        return "\n".join([*lines, "return tripped_now;", "}"])

    def command_lines(self, on, ramp, t="t"):
        effective = self._effective_command(on)
        return [f"{self.m('breaker')} = {effective};", f"{self.m('gates')} = {effective};",
                *self._terminal_lines()]


class _GraphUnitGenerator(_ElectricalUnitGenerator):
    """Lower a graph-controlled unit through its public control and bridge ports.

    The unit adapter supplies measured ports, registered control loops supply their equations and
    state, and the bridge topology supplies its duty-register width and switching-state encoding.
    """

    def __init__(self, name: str, unit: Any, model: _ModelGenerator) -> None:
        super().__init__(name, unit, model)
        self.continuous = unit.continuous
        self._control_inputs = model.unit_control_inputs(unit)
        graph_inputs = {path for path in self.graph.interface.inputs if path.startswith("meas.")}
        if set(self._control_inputs) != graph_inputs:
            raise TypeError(
                f"C++ export: {name}.control_inputs() exposes "
                f"{sorted(self._control_inputs)}, expected {sorted(graph_inputs)}"
            )
        if self.continuous:
            # Equations must read the controller's wired inputs at its scheduled ODE stage.
            # Physical-port observations are read only after the complete model is synchronized.
            controller = model._equation_clone(self.ctrl)
            controller._last_measurement = copy.copy(self.ctrl._last_measurement)

            def read():
                measurement = controller._flow_measurement()
                return {path: getattr(measurement, path.partition(".")[2])
                        for path in self._control_inputs}

            self._flow_inputs = model._trace(read)

    def _stage_condition(self, active: str) -> str:
        """Add the only extra enable required by a direct, breaker-less power port."""
        terms = [active]
        if self.unit.branch_f is None:
            terms.append(self.m("breaker"))
        terms.append(f"!{self.m('tripped')}")
        return " && ".join(terms)

    def declarations(self) -> list[str]:
        return [*(f"{_CPP_TYPES[value.kind]} {self.m(self._measurement_name(path))}{{}};"
                  for path, value in self._control_inputs.items()), *super().declarations()]

    def members(self) -> list[str]:
        startup = self.ctrl.startup
        lines = [
            *self.unit_members(),
            *self._loop_members(),
            f"bool {self.m('startup_run')} = {self.initial_run()};",
            f"double {self.m('startup_ramp')} = {_number(startup.ramp)};",
            *[f"double {self.m('log_' + key)} = std::numeric_limits<double>::quiet_NaN();"
              for key in self.ctrl.log_names()],
        ]
        if self.continuous:
            lines += [
                *self._flow_values(members=True),
                f"double {self.m('startup_start')} = {_number(startup.start)};",
                f"double {self.m('startup_value')} = {_number(startup.value)};",
            ]
            lines += self._adc_members()
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
            lines.append(f"{_CPP_TYPES[_Expression.of(value).kind]} {self.g(key)} = {_literal(value)};")
        lines += [*self._pwm_members(), *self._adc_members()]
        return lines

    def _log_lines(self) -> list[str]:
        sources = self.ctrl.log_sources()
        return [f"{self.m('log_' + key)} = {self.source(sources[key])};"
                for key in self.ctrl.log_names()
                if not (self.continuous and sources[key].startswith("meas."))]

    def log_observation_lines(self) -> list[str]:
        # An open-loop controller may run before its unused measurement inputs are wired.
        # Read those logs from final model observations, not from its early output evaluation.
        return [
            f"{self.m('log_' + key)} = obs_{_identifier(self.name + '.ctrl.' + source)};"
            for key, source in self.ctrl.log_sources().items()
            if source.startswith("meas.")
        ]

    def output_lines(self) -> list[str]:
        lines = [f"{self.m(self._measurement_name(path))} = {value.code};"
                 for path, value in self._flow_inputs.items()]
        lines.append(self._startup_value_line())
        lines += self._flow_lines()
        outputs = tuple(self.graph.interface.outputs)
        if len(outputs) != 1:
            raise TypeError(
                f"C++ export: a port-driven bridge controller needs one output, got {outputs}"
            )
        command = _Expression(self.source(outputs[0]), "real")
        state = self.model._trace(lambda: self.topology.state_from_duty(command))
        q = self.model.value(self.ctrl.out, "q")
        lines.append(
            f"{q} = {self.m('startup_run')} ? {state.code} : {_literal(self.topology.zero)};"
        )
        lines += self._log_lines()
        return lines

    def _command_registers(self) -> tuple[_Expression, ...]:
        output = SimpleNamespace(**{
            port: _Expression(self.source(port),
                              "complex" if self.graph.interface.outputs[port].complex_value else "real")
            for port in self.graph.outputs
        })
        return tuple(_Expression.of(value) for value in self.bridge.command_registers(output))

    def controller_method(self) -> str:
        lines = [f"void controller_{self.tag}(double t) {{"]
        lines += ["    " + line for line in self._rms_sample_lines()]
        lines += ["    " + line for line in self._sampled_measurement_lines()]
        lines += ["    " + line for line in self._sampled_loop_lines()]
        for index, duty in enumerate(self._command_registers()):
            lines.append(f"    {self.m('pending_d')}[{index}] = {duty.code};")
        lines += ["    " + line for line in self._log_lines()]
        lines += [
            f"    sampled_protect_{self.tag}(t, "
            f"std::abs({self._measurement_name('meas.voltage')}) / "
            f"{_number(self.cfg.dc_base.v)}, 0.0, "
            f"obs_{_identifier(self.name + '.u_dc')} / "
            f"{_number(self.cfg.dc_base.v)} - 1.0, {self.m('startup_complete')});",
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
             if subsystem is self.unit.bridge and field == "q"),
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
        next_values = [
            self.m("t_interrupt"), self.m("t_load"), self.m("next_switch"),
            *self._adc_next_values(),
        ]
        lines = [
            f"double next_{self.tag}() const noexcept {{ return "
            f"std::min({{{', '.join(next_values)}}}); }}",
            f"bool sense_{self.tag}(double t) {{",
            f"    const bool control_due = "
            f"std::abs(t - {self.m('t_interrupt')}) < time_eps;",
            *["    " + line for line in self._adc_sense_lines()],
            f"    if (control_due) controller_{self.tag}(t);",
            "    return false;",
            "}",
            f"bool actuate_{self.tag}(double t) {{",
            *["    " + line for line in self._pending_trip_lines()],
            f"    const bool control_due = std::abs(t - {self.m('t_interrupt')}) < time_eps;",
            f"    const bool load_due = std::abs(t - {self.m('t_load')}) < time_eps;",
        ]
        write = self._pwm_write_lines(self._stage_condition(self.m("pending_on")))
        if bridge.computation == 0.0:
            lines += [f"    if (control_due && {self.m('pending_valid')}) {{",
                      *write, "    }"]
        if self.unit.branch_f is None:
            tail = (f"        if (!{self.m('gates')}) "
                    f"{hold} = {_literal(self.topology.zero)};",)
        else:
            tail = tuple("        " + line for line in self._terminal_lines())
        lines += self._pwm_load_lines(
            hold,
            switching,
            self._stage_condition(f"{self.m('active')}[{self.width}] != 0.0"),
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
                    f"{hold} = {_literal(self.topology.zero)};"
                )
        lines += ["    " + line for line in self._adc_actuate_lines()]
        lines += [f"    if (control_due) {self.m('pending_valid')} = false;",
                  "    return tripped_now;", "}"]
        return "\n".join(lines)

    state_expression = _ControlGenerator.state_expression  # not the controller-less mapping

    def command_lines(self, on: bool, ramp: float, t: str = "t") -> list[str]:
        effective = self._effective_command(on)
        lines = [f"{self.m('breaker')} = {effective};"]
        lines += self._controller_command_lines(on, ramp, t)
        if self.continuous:
            lines.append(f"{self.m('gates')} = {effective};")
        else:
            active = f"{self.m('active')}[{self.width}] != 0.0"
            lines.append(
                f"{self.m('gates')} = {self._stage_condition(active)};"
            )
            if self.unit.branch_f is None:
                lines.append(
                    f"if (!{self.m('gates')}) "
                    f"{self._hold()} = {_literal(self.topology.zero)};"
                )
        return lines + self._terminal_lines()

    def sense_method(self) -> str:
        voltage = f"obs_{_identifier(self.name + '.ctrl.meas.voltage')}"
        dc = f"obs_{_identifier(self.name + '.u_dc')}"
        return f'''bool sense_{self.tag}(double t) {{
    {chr(10).join(self._rms_sample_lines())}
    sampled_protect_{self.tag}(t, std::abs({voltage}) / {_number(self.cfg.dc_base.v)},
        0.0, {dc} / {_number(self.cfg.dc_base.v)} - 1.0,
        {self._startup_complete()}{self._rms_protection_argument()});
    return false;
}}'''

def _controller_implementation(controller: Any, *operations: str) -> type:
    """Check the execution callbacks replaced by a native target and their dependencies."""
    # These are the existing host execution callbacks replaced by native code. Follow their
    # implementation dependencies; metadata readers invoked during export remain ordinary Python.
    operations = operations or ("__call__", "command", "command_at", "track", "retune", "set_logging",
                                "set_outputs", "rhs", "continuous_log", "observe_limits", "summary")
    for implementation in (GraphController, VSCController):
        if _uses_implementation(controller, implementation, *operations):
            return implementation
    raise TypeError(
        f"C++ export: controller {type(controller).__name__!r} has no generator"
    )


def _make_unit_generator(name: str, unit: Any, model: _ModelGenerator) -> Any:
    """Select an optimized controller lowering or the public-port graph lowering."""
    if unit.ctrl is None:
        return _ElectricalUnitGenerator(name, unit, model)
    if _controller_implementation(unit.ctrl) is GraphController:
        return _GraphUnitGenerator(name, unit, model)
    return (_ContinuousUnitGenerator(name, unit, model) if unit.continuous
            else _UnitGenerator(name, unit, model))


class _CppGenerator:
    """Generate a straight-line C++ model for one assembled simulation."""

    def __init__(self, simulation: Any, name: str, variables: tuple[str, ...],
                 initializations=(), *, exclude=()) -> None:
        self.simulation = simulation
        self.p = simulation.p
        self.system = simulation.system
        self.model = self.system.model
        self.name = name
        self.runtime = _RuntimeParameters(self.p, variables, exclude=exclude)
        if self.runtime.selected:
            from .validate import validate

            restored = simulation._initial_values(simulation.p.simulation.initial.t)
            restored_parts = {
                id(group.part): {state.local_name: restored[state.name]
                                 for state in group.states if state.name in restored}
                for group in simulation._states._groups
            }
            self.runtime.checking_layout = False
            try:
                validate(self.runtime.p)
                for change in self.runtime.p.changes:
                    if change.params is not None:
                        validate(change.params)
                for original, factory, parameters in initializations:
                    initialized = factory(self.runtime.construction_parameters(parameters))
                    # Parameterized factories obey the same named-state restoration contract
                    # as the live owners, including outputs derived from restored loop state.
                    if values := restored_parts.get(id(original)):
                        initialized.set_state(values)
                    self.runtime.pair(original, initialized)
            finally:
                self.runtime.checking_layout = True
        self.p = self.runtime.p

    def _initial_values(self, model):
        """Resolve the ordinary initial-state contract against parameterized source values."""
        if not self.runtime.selected:
            return self.simulation._initial_values(self.p.simulation.initial.t)
        system = copy.copy(self.system)
        system.units = {}
        for name, unit in self.system.units.items():
            view = copy.copy(unit)
            view.cfg = self.runtime.copy(unit.cfg)
            system.units[name] = view
        system.sources = {}
        t0 = self.p.simulation.initial.t
        for name, source in self.system.sources.items():
            view = copy.copy(source)
            if ACVoltageSource.uses_standard_outputs(source.emf.set_outputs):
                expression = model._source_equations(source.emf, _number(t0))["e_g"]
            else:
                def read():
                    clone = model._equation_clone(source.emf)
                    clone.set_outputs(_Expression(_number(t0), "real"))
                    return clone.out.e_g
                expression = model._trace(read)
            value = _InitializationExpression.make(expression, source.emf.out.e_g, self.runtime)
            # The normal preset resolver asks the source for its value once. Its AOT source
            # adapter has already evaluated that same expression at the initial instant.
            view.emf = SimpleNamespace(set_outputs=lambda _t: None,
                                       out=SimpleNamespace(e_g=value))
            system.sources[name] = view
        simulation = copy.copy(self.simulation)
        simulation.system, simulation.p = system, self.p
        return simulation._initial_values(t0)

    def _state_layout(self, model: _ModelGenerator,
                      generators: dict[str, _ControlGenerator]) -> tuple[list[str], list[str]]:
        """Public state-table columns and their fixed C++ value expressions."""
        columns: list[str] = []
        expressions: list[str] = []
        ode = {name: f"y[{index}]" for index, name in enumerate(self.model.state_labels())}
        equation_states = {id(group.part): model.named_equation_state(group.part)
                           for group in self.simulation._states._groups
                           if id(group.part) in model.equation_bindings}
        for group in self.simulation._states._groups:
            part = group.part
            for state in group.states:
                if part is self.model:
                    values = [ode[column] for column in state.columns]
                else:
                    statistics = next((item for item in generators.values()
                                       if part is item.unit.adc), None)
                    generator = next((item for item in generators.values()
                                      if part in (item.unit, item.unit.ctrl, item.unit.bridge,
                                                  item.unit.adc)), None)
                    if id(part) in equation_states:
                        value = model.solver_expression(equation_states[id(part)][state.local_name].code)
                    elif statistics is not None and state.local_name.startswith("rms_"):
                        value = statistics._rms_state(state.local_name)
                    elif generator is not None:
                        value = generator.state_expression(part, state.local_name)
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

    def _actions(self, model: _ModelGenerator,
                 generators: dict[str, _ControlGenerator]) -> tuple[list[float], str]:
        """Remaining file events, specialised to direct member assignments."""
        t0 = self.p.simulation.initial.t
        actions = [action for action in self.simulation._actions() if action[0] > t0 + TIME_EPS]
        changes = {change.event: change for change in self.p.changes}
        cases: list[str] = []
        for index, (_at, _order, action) in enumerate(actions):
            action = changes[action.event]
            lines: list[str] = []
            for path in action.connections:
                section, _, target = path.removesuffix(".connected").partition(".")
                on = action.values[path]
                if section == "units":
                    generator = generators.get(target)
                    if generator is None:
                        raise TypeError(f"C++ export: connection variable {path!r} has no controller")
                    source_ramp = model.source_ramp_members(generator.unit.dclink)
                    if source_ramp is not None:
                        start, duration, _law = source_ramp
                        lines += [f"{start} = t;", f"{duration} = {_number(action.ramp)};"]
                    lines += generator.command_lines(on, action.ramp)
                elif section == "sources":
                    branch = self.system.sources[target].branch
                    lines += model.breaker_lines(branch, "false" if on else "true")
                elif section == "branches":
                    branch = self.system.branches[target]
                    lines += model.breaker_lines(branch, "false" if on else "true")
                else:
                    element = self.system.named_elements[target]
                    breakers = getattr(element, "breakers", ())
                    if hasattr(element, "branch"):
                        breakers = (element.branch,)
                    if not breakers:
                        raise TypeError(f"C++ export: connection variable {path!r} has no static breaker")
                    lines += [
                        line
                        for branch in breakers
                        for line in model.breaker_lines(branch, "false" if on else "true")
                    ]
            for path in action.paths:
                if path in action.connections:
                    continue
                if path in model.element_parameters:
                    member, _, _ = model.element_parameters[path]
                    _, element, field = path.split(".")
                    value = getattr(action.params.elements[element], field)
                    lines.append(f"{member} = {_number(value)};")
                    continue
                owner, separator, reference = path.removeprefix("units.").partition(".ctrl.references.")
                if separator and path.startswith("units."):
                    generator = generators[owner]
                    value = getattr(action.params.unit(owner).ctrl.references, reference)
                    lines.append(f"{generator._reference(reference)} = {_literal(value)};")
                elif not (path.startswith("sources.") and path.rsplit(".", 1)[-1] in {"v", "f", "angle"}):
                    raise TypeError(f"C++ export: event {action.event!r} writes unsupported variable {path!r}")
            cases += [f"        case {index}:", *["            " + line for line in lines],
                      "            break;"]
        times = [item[0] for item in actions]
        return times, "\n".join(cases)

    def _signal_layout(self, generators: dict[str, _ControlGenerator]) -> tuple[list[str], list[str]]:
        """The exact public plant.csv schema and expressions used by Python's recorder."""
        from ..solver.simulation import _plant_output_row

        t = self.p.simulation.initial.t
        self.model.sync(t, self.model.get_initial_values())
        signal_values = self.system.signals(t)
        expressions = {}
        for key, value in signal_values.items():
            if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
                expressions.update((name, f"phases(obs_{_identifier(key)})[{index}]")
                                   for index, name in enumerate(_signal_names(key, 3)))
            else:
                expressions[key] = f"obs_{_identifier(key)}"
        control_names = {}
        for name, generator in generators.items():
            if generator.ctrl is None:
                continue
            control_names[name] = tuple(generator.ctrl.log_names())
            expressions.update((f"{name}.{key}", generator.m("log_" + key))
                               for key in control_names[name])
        schema = _plant_output_row(self.p, t, signal_values, {}, control_names)
        columns = [name for name in schema if name != "t"]
        return columns, [expressions[name] for name in columns]

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
            return f"std::real(({a}) * std::conj({b}))" if "complex" in kinds else f"(({a}) * ({b}))"

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
            try:
                attributes = model._equation_attributes(sub)
                return (model.trace_scalar_method(sub, "supplied_power", attributes),
                        model.trace_scalar_method(sub, "dissipated_power", attributes))
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
                coefficient = model.storage_coefficients.get(
                    (id(storage.record), storage.field), storage.coefficient)
                energy_terms.append(f"0.5 * {_number(coefficient)} * ({norm})")
                rate_terms.append(f"{_number(coefficient)} * ({rate})")
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
            if hasattr(check.subsystem, "impulsive_dissipated_energy"):
                impulse = model._trace(lambda: model._equation_clone(
                    check.subsystem).impulsive_dissipated_energy)
                columns.append(f"{check.name}.dissipated_impulsive_J")
                values.append(impulse.code)
        all_values = ["total_energy", "total_supplied", "total_dissipated", "tellegen",
                      "max_residual", *values, "std::abs(tellegen) / scale", "max_residual / scale"]
        lines += [f"    return std::array<double, {len(all_values)}>{{{{{', '.join(all_values)}}}}};", "}"]
        return "\n".join(lines), columns

    def source(self, model: _ModelGenerator | None = None) -> str:
        """Return the complete generated translation unit."""
        if model is None:
            model = _ModelGenerator(self.simulation, self.runtime)
        self.diagnostics = model.diagnostics
        sampled = {name: _make_unit_generator(name, unit, model)
                   for name, unit in self.system.units.items() if not unit.continuous}
        continuous = {name: (model.continuous_units[id(unit.ctrl)] if unit.ctrl is not None
                             else _make_unit_generator(name, unit, model))
                      for name, unit in self.system.units.items() if unit.continuous}
        generators = {**sampled, **continuous}
        for sub in self.model.subsystems:
            if id(sub) in model.equation_bindings and id(sub) not in model.equation_references:
                model.equation_references[id(sub)] = model._equation_clone(sub)
        unit_names = tuple(self.system.units)
        state_names = self.model.state_labels()
        initial = self.model.get_initial_values().tolist()
        initialized = self._initial_values(model)
        state_names_by_record = {(id(sub.state), field): f"{name}.{field}"
                                 for name, sub in zip(self.model.names, self.model.subsystems)
                                 for field in sub.state_names}
        for record, field, index, complex_value in self.model._state_ops:
            key = state_names_by_record[id(record), field]
            value = initialized.get(key, self.runtime.initial_value(record, field))
            initial[index] = value.real if complex_value else value
            if complex_value:
                initial[index + 1] = value.imag
        public_names, state_values = self._state_layout(model, generators)
        labels = ",\n        ".join(_cpp_string(name) for name in public_names)
        values = ", ".join(_number(value) for value in initial)
        projection_lines = []
        for index, lower, upper in self.model._state_bounds:
            if lower is not None:
                projection_lines.append(f"y[{index}] = std::max(y[{index}], {_number(lower)});")
            if upper is not None:
                projection_lines.append(f"y[{index}] = std::min(y[{index}], {_number(upper)});")
        for project in self.model._state_projections:
            projection_lines.extend(model.trace_transition(project.__self__, "project_state", timed=False))
        projection_body = "\n        ".join(projection_lines)
        projection_method = (
            f"    inline void project_state() {{\n        {projection_body}\n    }}\n"
            if projection_lines else ""
        )
        projection_call = "project_state();" if projection_lines else ""
        diode_conduction = DiodeConduction(self.model, self.simulation.solver)
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
            enabled = model.enabled_by.get(id(subsystem))
            attributes = ({"enabled": _Expression(enabled, "bool")}
                          if enabled is not None else None)
            diode_step_limits.append(model.trace_solver_method(
                subsystem, "diode_maximum_step", attributes
            ))
        commutation_margins, commutation_times, commutation_transitions = [], [], []
        for sub in diode_conduction.commutating:
            commutation_margins.append(model.trace_solver_method(sub, "commutation_margin"))
            commutation_times.append(model.trace_solver_method(sub, "next_commutation_time")
                                    if callable(getattr(sub, "next_commutation_time", None))
                                    else "std::numeric_limits<double>::infinity()")
            commutation_transitions.append(model.trace_transition(sub, "commutate"))
        raw_projection_call = "" if diode_conduction else projection_call
        member_lines = model.members(generators.values())
        for generator in (*sampled.values(), *(item for item in continuous.values() if item.ctrl is None)):
            member_lines.extend(generator.members())
        for generator in generators.values():
            member_lines.extend(generator.reference_members())
        members = "\n    ".join(member_lines)
        direct_guards = {}
        rhs = "\n        ".join(model.body(direct_guards=direct_guards))
        fallback_method = ""
        if direct_guards:
            # Dispatch before declaring equation locals: capturing them in a per-loop fallback
            # forces stack stores even when the direct path never needs an iterative solve.
            fallback_rhs = "\n        ".join(model.body())
            fallback_method = ("    PESLITE_NOINLINE void evaluate_feedback(double t, "
                               "const State& y, State& dy) {\n        "
                               + fallback_rhs + "\n    }\n")
            guard = " && ".join(dict.fromkeys(direct_guards.values()))
            rhs = f"if (!({guard})) {{ evaluate_feedback(t, y, dy); return; }}\n        " + rhs
        sync_body = "\n        ".join(model.body(derivatives=False))
        observation_members, observation_assignments = model.observations(generators)
        tracking = {generator.name: generator.tracking_method()
                    for generator in generators.values()}
        initial_tracking = [f"if ({generator.initial_connection()}) {{ "
                            f"sync({_number(self.p.simulation.initial.t)}); track_{generator.tag}(); }}"
                            for generator in generators.values()
                            if tracking[generator.name] and generator.name not in self.simulation._continued]
        observations = "\n        ".join(observation_assignments)
        members = "\n    ".join([members, *observation_members])
        methods = list(tracking.values())
        for generator in sampled.values():
            methods += [generator.controller_method(), generator.switching_method(),
                        generator.event_methods(), generator.adc_methods(),
                        generator.protection_method()]
        methods += [method for generator in continuous.values()
                    for method in (generator.sense_method(), generator.actuate_method(),
                                   generator.protection_method(), generator.adc_methods())]
        energy_method, energy_columns = self._energy_method(model)
        if energy_method:
            methods.append(energy_method)
        methods_text = "\n\n    ".join(method.replace("\n", "\n    ") for method in methods if method)
        next_units = ", ".join([
            *(f"next_{generator.tag}()" for generator in sampled.values()),
            *(value for generator in continuous.values() for value in generator._adc_next_values()
              if generator.adc is not None and generator.adc.clocked),
        ])
        sense = [f"if (sense_{generators[name].tag}(t)) sync(t);" for name in unit_names]
        actuate = [f"trip_now = actuate_{generators[name].tag}(t) || trip_now;" for name in unit_names]
        protect = [f"trip_now = protect_{generators[name].tag}(t) || trip_now;" for name in unit_names]
        adc_accumulate = [generator.adc_accumulate_call("t")
                          for generator in generators.values()]
        adc_accumulate = [line for line in adc_accumulate if line]
        adc_seed = [line for generator in generators.values()
                    for line in generator.adc_seed_lines()]
        adc_initialization = [line for generator in generators.values()
                              for line in generator.adc_initialization_lines()]
        connection_initialization = [line for generator in generators.values()
                                     if f"units.{generator.name}.connected" in self.runtime.by_path
                                     for line in generator._terminal_lines()]
        if adc_initialization and adc_seed:
            adc_initialization.insert(0, f"sync<true>({_number(self.p.simulation.initial.t)});")
        # Exact-edge loops have a larger event body; keep their observation refresh out of line.
        sync_qualifier = ("PESLITE_NOINLINE" if any(unit.bridge.has_switching_events
                          for unit in self.system.units.values()) else "inline")
        sync_template = "template <bool seed_adc = false>" if adc_seed else ""
        sync_update = observations
        if adc_seed:
            # Share one model body; each instantiation retains only its required stores.
            sync_update = ("if constexpr (seed_adc) {\n" + "\n".join(adc_seed)
                           + "\n} else {\n" + observations + "\n}")
        action_times, action_cases = self._actions(model, generators)
        times = ", ".join(_number(value) for value in action_times)
        snapshot_values = "\n            ".join(f"states << ',' << ({value});" for value in state_values)
        output = self.p.simulation.output
        resolved_template, resolved_replacements = self.runtime.resolved_template()
        resolved_text = _cpp_string(resolved_template)
        resolved_replace = "\n    ".join(
            f"replace_all(text, {_cpp_string(marker)}, {expression});"
            for marker, expression in resolved_replacements
        )
        plant_names, plant_values = self._signal_layout(generators)
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
                generator = generators[name]
                if generator.ctrl is None:
                    continue
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
                        f"{generator.m('pwm_k')} % std::max<long long>(1, {output.record_every}) == 0) {{",
                        row,
                        "}",
                    ]
                else:
                    continuous_ctrl_rows += [
                        *generator.log_observation_lines(),
                        f"if (snapshot_index % std::max<long long>(1, {output.record_every}) == 0) {{",
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
                f"if (snapshot_index % {self.p.simulation.solver.phs_check_step} == 0 || final) {{",
                "    const auto energy_values = energy_sample(t);",
                *([energy_row] if energy_row else []),
                f"    energy_tellegen_max_rel = std::max(energy_tellegen_max_rel, energy_values[{len(energy_columns)}]);",
                f"    energy_balance_max_rel = std::max(energy_balance_max_rel, energy_values[{len(energy_columns) + 1}]);",
                "}",
            ])
        global_tripped = " || ".join(
            generators[name].m("tripped") for name in unit_names
        ) or "false"
        summary_rows: list[str] = []
        for name in unit_names:
            generator = generators[name]
            summary_rows += generator.summary_lines()
        for name, sub in zip(self.model.names, self.model.subsystems):
            if callable(getattr(sub, "diagnostics", None)):
                diagnostics = model._trace(lambda: model._equation_clone(sub).diagnostics())
                summary_rows += [f'summary << {_cpp_string("," + chr(10) + "  " + json.dumps(name + "." + key) + ": ")} '
                                 f'<< {model.solver_expression(value.code)};'
                                 for key, value in diagnostics.items()]
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
        fixed = solver.type == "fixed"  # _export_cpp admits only fixed RK and DP45 solvers
        runtime_declarations = self.runtime.declarations()
        runtime_setters = self.runtime.setters()
        runtime_listing = self.runtime.listing()
        fixed_dt = self.runtime.value("simulation.solver.dt", solver.dt)
        adaptive_max_step = self.runtime.value("simulation.solver.max_step", solver.max_step)
        adaptive_atol = self.runtime.value("simulation.solver.atol", solver.atol)
        adaptive_rtol = self.runtime.value("simulation.solver.rtol", solver.rtol)
        adaptive_safety = "" if fixed else _number(self.simulation.solver.safety)
        run_end = self.runtime.value("simulation.t_end", self.p.simulation.t_end)
        output_period_value = self.runtime.value("simulation.output.period", output.period)
        integration_name = "integrate_solver" if diode_conduction else "integrate"
        local_limit = bool(diode_step_limits)
        integration_args = (", double maximum_step = std::numeric_limits<double>::infinity()"
                            if local_limit else "")
        if local_limit:
            fixed_dt = f"std::min({fixed_dt}, maximum_step)"
            adaptive_max_step = f"std::min({adaptive_max_step}, maximum_step)"
        if fixed:
            step_code = _fixed_step(solver.method)
            solver_code = step_code + f'''

    void {integration_name}(double& t, double target{integration_args}) {{
        const double span = target - t;
        if (span <= 0.0) return;
        const std::size_t count = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(span / {fixed_dt} - 1e-9)));
        const double h = span / static_cast<double>(count);
        for (std::size_t k = 0; k < count; ++k) step(t, h);
        {raw_projection_call}
        t = target;
    }}'''
        else:
            actual = self.simulation.solver
            c, a, b, e = actual._c, actual._a, actual._b, actual._e
            names = tuple(f"k{index + 1}[i]" for index in range(len(c)))
            stages = []
            for index in range(1, len(c) - 1):
                stages += ["            for (std::size_t i = 0; i < state_count; ++i) "
                           f"x[i] = y[i] + h * ({_weighted_sum(a[index], names)});",
                           f"            rhs(t + h * {_number(c[index])}, x, k{index + 1});"]
            solver_code = f'''    void {integration_name}(double& t, double target{integration_args}) {{
        const double span = target - t;
        if (span <= 0.0) return;
        double h = std::isfinite(adaptive_h) ? adaptive_h : span;
        h = std::min(h, std::min(span, {adaptive_max_step}));
        State k1{{}}, k2{{}}, k3{{}}, k4{{}}, k5{{}}, k6{{}}, k7{{}}, x{{}}, y1{{}};
        rhs(t, y, k1);
        const double end_epsilon = 1e-15 * std::max(1.0, std::abs(target));
        while (t < target - end_epsilon) {{
            if (t + h > target) h = target - t;
{chr(10).join(stages)}
            for (std::size_t i = 0; i < state_count; ++i) y1[i] = y[i] + h * ({_weighted_sum(b, names)});
            rhs(t + h, y1, k7);
            double error = 0.0;
            for (std::size_t i = 0; i < state_count; ++i) {{
                const double estimate = std::abs(h * ({_weighted_sum(e, names)}));
                error = std::max(error, estimate / ({adaptive_atol} + {adaptive_rtol} * std::max(std::abs(y[i]), std::abs(y1[i]))));
            }}
            if (error <= 1.0 || h <= 1e-15) {{
                t += h;
                y = y1;
                k1 = k7;
                double factor = error == 0.0 ? 5.0 : {adaptive_safety} * std::pow(error, -0.14) * std::pow(adaptive_error, 0.08);
                factor = std::clamp(factor, 0.2, 5.0);
                adaptive_error = std::max(error, 1e-4);
                h = std::min(h * factor, {adaptive_max_step});
            }} else {{
                ++n_rejected;
                h *= std::max(0.1, {adaptive_safety} * std::pow(error, -0.25));
            }}
        }}
        adaptive_h = h;
        {raw_projection_call}
        t = target;
    }}'''
        advance_method = "integrate"
        diode_declaration = ""
        conduction_sync = ""
        if diode_conduction:
            callbacks = [f"bool diode_blocks_reverse_current_{k}() {{ return {guard}; }}"
                         for k, guard in enumerate(diode_blocking_guards)]
            callbacks += [f"double diode_maximum_step_{k}() {{ return {value}; }}"
                          for k, value in enumerate(diode_step_limits)]
            for name, expressions in (("commutation_margin", commutation_margins),
                                      ("commutation_next", commutation_times)):
                callbacks += [f"double {name}(std::size_t index) {{ switch (index) {{",
                              *[f"case {k}: return {expression};"
                                for k, expression in enumerate(expressions)],
                              "default: return std::numeric_limits<double>::infinity(); } }"]
            callbacks += ["void commutate(std::size_t index, double t) { switch (index) {",
                          *[f"case {k}: {{ " + " ".join(lines) + "; break; }"
                            for k, lines in enumerate(commutation_transitions)],
                          "default: break; } }"]
            raw = "sim.integrate_solver(at, end);"
            if local_limit:
                raw = ("sim.sync(begin); double maximum_step = std::numeric_limits<double>::infinity(); "
                       + " ".join(f"maximum_step = std::min(maximum_step, sim.diode_maximum_step_{k}());"
                                  for k in range(len(diode_step_limits)))
                       + ' if (!(maximum_step > 0.0)) throw std::runtime_error("diode conduction step is below time resolution");'
                       + " sim.integrate_solver(at, end, maximum_step);")
            tolerance = model._trace(lambda: commutation_tolerance(_Expression("t", "real"))).code
            probe = (self.runtime.value("simulation.solver.dt", solver.dt) if fixed else
                     self.runtime.value("simulation.solver.max_step", solver.max_step))
            callbacks.append(f"double commutation_probe() const {{ return {probe}; }}")
            substitutions = {
                "COMMUTATIONS": str(len(commutation_margins)),
                "CURRENTS": str(len(diode_current_states)),
                "TOLERANCE": tolerance,
                "RAW_ADVANCE": raw,
                "CURRENT_MARGINS": " ".join(
                    f"if (guarded[{k}]) value = std::min(value, state[{index}]);"
                    for k, (_sub, index) in enumerate(diode_current_states)),
                "CLAMP_CURRENTS": " ".join(
                    f"if (guarded[{k}] && trial[{index}] < 0.0) trial[{index}] = 0.0;"
                    for k, (_sub, index) in enumerate(diode_current_states)),
                "GUARDS": ", ".join(f"sim.diode_blocks_reverse_current_{k}() && initial[{index}] > 0.0"
                                    for k, (_sub, index) in enumerate(diode_current_states)),
                "MAX_COMMUTATIONS": str(_MAX_COMMUTATIONS),
                "LOCATOR_ITERATIONS": str(_LOCATOR_ITERATIONS),
                "PROBE": "sim.commutation_probe()" if commutation_margins else "std::numeric_limits<double>::infinity()",
                "PROJECT": "sim.project_state();" if projection_lines else "",
            }
            kernel = _CONDUCTION_KERNEL
            for key, value in substitutions.items():
                kernel = kernel.replace("@@" + key + "@@", value)
            solver_code += "\n" + "\n".join(callbacks) + "\n" + kernel
            advance_method = "diode_conduction.advance"
            diode_declaration = "DiodeConduction diode_conduction(*this);"
            if commutation_margins:
                conduction_sync = "diode_conduction.synchronize(t);"
        adaptive_members = (f'''std::size_t n_rejected = 0;
    double adaptive_h = {_number(math.nan if self.simulation.solver._h is None else self.simulation.solver._h)};
    double adaptive_error = {_number(self.simulation.solver._err_prev)};''' if not fixed else "")
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

// Keep observation refreshes out of the numerical hot loop's instruction footprint.
#if defined(_MSC_VER)
#define PESLITE_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define PESLITE_NOINLINE __attribute__((noinline))
#else
#define PESLITE_NOINLINE
#endif

namespace peslite_generated {{

using Complex = std::complex<double>;
constexpr std::size_t state_count = {len(state_names)};
using State = std::array<double, state_count>;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double time_eps = {_number(TIME_EPS)};

{_ALGEBRAIC_SOLVER if any(len(loop.tears) == 1 for loop in self.model.algebraic_loops) else ""}

{_MOVING_WINDOW}

template <std::size_t N> struct RMSWindow {{
    MovingWindow<N> window;
    double area = 0.0, duration = 0.0, last = 0.0, total = 0.0;
    mutable double cached_value = std::numeric_limits<double>::quiet_NaN();
    void accumulate(double dt, double square) {{
        area += 0.5 * dt * (last + square);
        duration += dt;
        last = square;
    }}
    void sample(double period) {{
        if (duration >= period * (1.0 - 1e-9)) {{
            const double mean = area / duration;
            total += mean - window.push(mean);
            cached_value = std::numeric_limits<double>::quiet_NaN();
        }}
        area = duration = 0.0;
    }}
    double value() const {{
        if (window.count != N) return std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(cached_value)) cached_value = std::sqrt(std::max(0.0, total / N));
        return cached_value;
    }}
}};

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

{_CONTROL_MATH}

{"".join(dict.fromkeys(code for generator in sampled.values() for code in generator.shared_code()))}

inline double carrier_position(double t, double frequency, double phase) noexcept {{
    double value = std::fmod(t * frequency + phase, 1.0);
    if (value < 0.0) value += 1.0;
    if (value > 1.0 - {_number(_SNAP)} || value < {_number(_SNAP)}) return 0.0;
    if (std::abs(value - 0.5) < {_number(_SNAP)}) return 0.5;
    return value;
}}

inline double carrier_value(double position) noexcept {{
    return position < 0.5 ? -1.0 + 4.0 * position : 3.0 - 4.0 * position;
}}

constexpr std::array<std::string_view, {len(public_names)}> state_names{{
        {labels}
}};

class Simulator {{
public:
    RuntimeParameters parameters;
    double event_time = {_number(self.p.simulation.initial.t)};
    {chr(10).join('    ' + line for line in self.runtime.initialization_members())}
    State y{{{values}}};
    std::size_t n_rhs = 0;
    {adaptive_members}
    {members}

    explicit Simulator(RuntimeParameters configured = {{}}) : parameters(std::move(configured)) {{
        {self.runtime.validation()}
        {chr(10).join(connection_initialization)}
        {'DiodeConduction initial_conduction(*this); initial_conduction.synchronize(' + _number(self.p.simulation.initial.t) + ');' if conduction_sync else ''}
        {projection_call}
        {chr(10).join(initial_tracking)}
        {chr(10).join(adc_initialization)}
    }}

    inline void evaluate(double t, const State& y, State& dy) {{
        {rhs}
    }}

{fallback_method}
    inline void rhs(double t, const State& y, State& dy) {{
        ++n_rhs;
        evaluate(t, y, dy);
    }}

{projection_method}
    {sync_template}
    {sync_qualifier} void sync(double t) {{
        {sync_body}
        {sync_update}
    }}

{solver_code}

    {methods_text}

    void apply_action(std::size_t action, double t) {{
        event_time = t;
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
        {conduction_sync}
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
            double physical_boundary = end;
            {'physical_boundary = std::min(physical_boundary, next_output);' if continuous else ''}
            {'physical_boundary = std::min(physical_boundary, std::min({' + next_units + '}));' if next_units else ''}
            if (action_index < action_times.size())
                physical_boundary = std::min(physical_boundary, action_times[action_index]);
            const double boundary = std::min(physical_boundary, next_output);
            if (boundary > t + time_eps) {{
                {advance_method}(t, boundary);
            }} else {{
                t = boundary;
            }}
            sync(t);
            const bool physical_due = physical_boundary <= t + time_eps;
            if (physical_due) {{
                {' '.join(adc_accumulate)}
                bool actions_applied = false;
                while (action_index < action_times.size() && action_times[action_index] <= t + time_eps) {{
                    apply_action(action_index, t);
                    actions_applied = true;
                    ++action_index;
                }}
                if (actions_applied) sync(t);
                {conduction_sync}
                bool trip_now = false;
                {' '.join(protect)}
                if constexpr ({'true' if self.p.simulation.stop_on_trip else 'false'}) {{
                    if (trip_now) {{ snapshot(true); break; }}
                }}
                if (trip_now) sync(t);
                {' '.join(sense)}
                if constexpr ({'true' if output.signals else 'false'}) {{
                    {sampled_ctrl_text}
                }}
                {' '.join(actuate)}
                {conduction_sync}
                if constexpr ({'true' if self.p.simulation.stop_on_trip else 'false'}) {{
                    if (trip_now) {{ snapshot(true); break; }}
                }}
                {'sync<true>(t);' if adc_seed else ''}
            }}
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


def _prepare_export(simulation, variables, reusable, *, params=None):
    """Rebuild privately and record the ordinary factories for either C++ target."""
    from ..solver.simulation import Simulation

    if not simulation._rebuildable:
        raise TypeError(
            "C++ export cannot safely reconstruct supplied instances; use parts factories instead"
        )
    initializations = []
    recorder = lambda part, factory, parameters: initializations.append((part, factory, parameters))
    token = _PARAMETER_RECORDER.set(recorder if variables else None)
    try:
        p = simulation.p if params is None else params
        prepared = (simulation if reusable and not variables and p is simulation.p else
                    Simulation(p, parts=simulation._factories))
    finally:
        _PARAMETER_RECORDER.reset(token)
    return prepared, initializations


def _parameter_generator(prepared, name, variables, initializations, *, constants=None):
    """Combine target constants and common specialization in one shared parameter rebuild."""
    common = "common" in variables
    variables = tuple("all" if path == "common" else path for path in variables)
    generator = _CppGenerator(prepared, name, variables, initializations)
    excluded = set(constants(generator.runtime)) if constants is not None else set()
    model = None
    diagnostics = []
    if common:
        model = _ModelGenerator(prepared, generator.runtime)
        sensitive = {path for item in model.diagnostics for path in item["parameters"]}
        if sensitive:
            diagnostics.append({
                "code": "common-constants",
                "parameters": tuple(sorted(sensitive)),
                "message": "common keeps detected structure-sensitive parameters constant: "
                + ", ".join(sorted(sensitive)),
            })
            excluded.update(sensitive)
    if excluded:
        generator = _CppGenerator(prepared, name, variables, initializations, exclude=excluded)
        model = None
    return generator, model, diagnostics


@_exporter("cpp")
def _export_cpp(simulation: Any, out_dir: Path, name: str,
                variables: tuple[str, ...], reusable: bool) -> ExportResult:
    """Write one self-contained, parameter-specialised C++17 translation unit."""
    prepared, initializations = _prepare_export(simulation, variables, reusable)
    if prepared.p.simulation.solver.subsystems:
        raise TypeError(
            "C++ export does not yet lower multirate subsystem schedules; use a single-rate "
            "fixed or DP45 solver for the exported simulator"
        )
    settings = prepared.p.simulation.solver
    actual = prepared.solver
    supported = (_uses_implementation(actual, FixedStepSolver)
                 and settings.type == "fixed"
                 and actual.method == settings.method and actual.dt == settings.dt)
    supported = supported or (_uses_implementation(actual, DormandPrince45)
                              and settings.type == "adaptive" and settings.method == "DP45"
                              and (actual.rtol, actual.atol, actual.max_step)
                              == (settings.rtol, settings.atol, settings.max_step))
    if not supported:
        raise TypeError("C++ export needs a supported solver matching simulation.solver; "
                        "a solver replacement cannot be silently substituted")
    t0 = prepared.p.simulation.initial.t
    model = prepared.system.model
    y = prepared._apply_initial(t0)
    # Native initialization settles natural modes before applying state projections. Building
    # timer metadata must not project an uninitialized mode and alter another initial state.
    plant = Plant(model, y)
    # Native control initialization always runs in the executable, including constant exports.
    for unit in prepared.system.units.values():
        unit.begin_timing(t0, plant, unit.name in prepared._continued,
                          unit.name in prepared._windows_given)
    plant.read_state()
    plant.outputs(t0)

    generator, model, diagnostics = _parameter_generator(prepared, name, variables, initializations)
    code = generator.source(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = out_dir / "peslite.cpp"
    source.write_text(code, encoding="utf-8")
    return ExportResult("cpp", out_dir, (source,), tuple(diagnostics + generator.diagnostics))
