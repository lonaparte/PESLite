"""Scalar lowering shares arithmetic without evaluating inactive branches or opaque calls."""

import itertools
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from peslite.assembly.cpp import _Expression, _ModelGenerator
from peslite.solver.netlist import _Symbolic, _symbolic_source
from test_cpp_export import _compiler

pytestmark = pytest.mark.long


def _function(name, equation, statements=None):
    generator = object.__new__(_ModelGenerator)
    generator.runtime = SimpleNamespace(_equation_counter=0, _expression_dependencies={})
    with generator._shared_terms() as (prefix, terms):
        value = equation()
    lines = [f"return {value.code};"] if statements is None else statements(value.code)
    body = generator._term_declarations(prefix, terms, lines)
    return f"double {name}(double input_value) {{\n" + "\n".join(body) + "\n}"


def _rows(tmp_path, compiler, functions, main):
    source = tmp_path / "expressions.cpp"
    source.write_text('''#include <cfenv>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
using Complex = std::complex<double>;
int calls = 0;
double probe(double value) { ++calls; return value; }
double next_value() { return ++calls; }
''' + "\n".join(functions) + '''
int main() {
    std::cout << std::setprecision(17);
''' + main + "}\n", encoding="utf-8")
    binary = tmp_path / "expressions"
    subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True, timeout=60)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)
    return [[float(value) for value in row.split()] for row in result.stdout.splitlines()]


def _symbolic_function(name, expression):
    shared, (result,) = _symbolic_source((expression,), language="cpp")
    declarations = [f"const auto {line};" for line in shared]
    return f"double {name}(double input_value) {{\n" + "\n".join(
        (*declarations, f"return {result};")
    ) + "\n}"


def test_shared_expressions_keep_linear_work_and_guarded_evaluation(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")

    def chain(depth):
        value = _Expression("probe(input_value)", "real", pure=True)
        for _ in range(depth):
            value = (value + value) / (value + 2.0)
        return value

    def guarded():
        value = _Expression("input_value", "real")
        return np.where(value > 2.0, np.sqrt(value - 2.0) / (value + value), 7.0)

    def opaque():
        value = _Expression("next_value()", "real")
        return value + value

    def partial():
        value = _Expression("input_value", "real")
        return np.sqrt((value - 2.0) * (value + 1.0) * (value + 2.0)) / (value + value)

    functions = [
        _function("chain8", lambda: chain(8)),
        _function("chain12", lambda: chain(12)),
        _function("guarded", guarded),
        _function("statement_guard", partial,
                  lambda value: ["if (input_value > 2.0) {", f"return {value};", "}", "return 7.0;"]),
        _function("opaque", opaque),
    ]
    rows = _rows(tmp_path, compiler, functions, '''
    for (int depth : {8, 12}) {
        calls = 0;
        const double value = depth == 8 ? chain8(1.5) : chain12(1.5);
        std::cout << value << " " << calls << "\\n";
    }
    std::feclearexcept(FE_ALL_EXCEPT);
    const double guarded_value = guarded(0.0);
    std::cout << guarded_value << " " << std::fetestexcept(FE_INVALID | FE_DIVBYZERO) << "\\n";
    std::cout << guarded(3.0) << "\\n";
    std::feclearexcept(FE_ALL_EXCEPT);
    const double statement_value = statement_guard(0.0);
    std::cout << statement_value << " " << std::fetestexcept(FE_INVALID | FE_DIVBYZERO) << "\\n";
    calls = 0;
    const double opaque_value = opaque(0.0);
    std::cout << opaque_value << " " << calls << "\\n";
''')
    for row, depth in zip(rows[:2], (8, 12)):
        expected = 1.5
        for _ in range(depth):
            expected = (expected + expected) / (expected + 2.0)
        assert row[0] == pytest.approx(expected, rel=1e-12)
    assert rows[1][1] <= 2 * rows[0][1]
    assert rows[2] == [7.0, 0.0]
    assert rows[3][0] == pytest.approx(1.0 / 6.0)
    assert rows[4] == [7.0, 0.0]
    assert rows[5] == [3.0, 2.0]


def test_shared_symbolic_dag_generates_valid_cpp_for_scalar_operations(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    value = _Symbolic.value("input_value")
    complex_value = _Symbolic.value("Complex{input_value, 2.0}")
    condition = (value > 0.0) | (value < -2.0)
    expression = np.where(
        condition,
        np.maximum(2.0 * value, 1.0),
        np.minimum(complex_value.real + complex_value.imag, 4.0),
    )
    rows = _rows(
        tmp_path, compiler, (_symbolic_function("dag", expression),),
        'for (double value : {-1.0, 0.25, 2.0}) std::cout << dag(value) << "\\n";\n',
    )
    assert rows == [[1.0], [1.0], [4.0]]


def test_guarded_terms_are_set_up_on_the_branch_that_uses_them(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    depth = 200

    def nested():
        value = _Expression("input_value", "real")
        slow = _Expression("probe(input_value)", "real", pure=True)
        partial = np.sqrt((value - 2.0) * (slow + value) * (slow + 2.0))
        inner = np.where(value > 3.0, partial / (value + value), partial + partial)
        return np.where(value > 2.0, inner * (value - 1.0), 7.0)

    def chain():
        value = _Expression("input_value", "real")
        result = np.sqrt(value - float(depth)) / (value + value)
        for level in reversed(range(depth)):
            result = np.where(value > float(level),
                              np.sqrt(value - float(level)) / (value + value) + result,
                              float(level))
        return result

    functions = {"nested": _function("nested", nested), "chain": _function("chain", chain)}
    assert "_ready" in functions["nested"]
    assert "_ready" not in functions["nested"].partition("?")[0]
    assert max(itertools.accumulate(
        (char in "([{") - (char in ")]}") for char in functions["chain"])) < 256

    rows = _rows(tmp_path, compiler, functions.values(), '''
    for (double input : {0.0, 2.5, 4.0}) {
        calls = 0;
        std::feclearexcept(FE_ALL_EXCEPT);
        const double value = nested(input);
        std::cout << value << " " << calls << " "
                  << std::fetestexcept(FE_INVALID | FE_DIVBYZERO) << "\\n";
    }
    std::cout << chain(-1.0) << " " << chain(50.5) << " " << chain(500.0) << "\\n";
''')

    def expected(value, level=0):
        if level == depth:
            return np.sqrt(value - depth) / (value + value)
        if value <= level:
            return float(level)
        return np.sqrt(value - level) / (value + value) + expected(value, level + 1)

    assert rows[0] == [7.0, 0.0, 0.0]
    assert rows[1] == [pytest.approx(2.0 * np.sqrt(11.25) * 1.5, rel=1e-14), 2.0, 0.0]
    assert rows[2] == [pytest.approx(np.sqrt(96.0) / 8.0 * 3.0, rel=1e-14), 2.0, 0.0]
    assert rows[3] == [pytest.approx(expected(value), rel=1e-12)
                       for value in (-1.0, 50.5, 500.0)]
