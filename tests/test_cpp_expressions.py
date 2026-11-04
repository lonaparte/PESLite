"""Scalar lowering shares arithmetic without evaluating inactive branches or opaque calls."""

import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from peslite.assembly.cpp import _Expression, _ModelGenerator
from test_cpp_export import _compiler

pytestmark = pytest.mark.full


def _function(name, equation, statements=None):
    generator = object.__new__(_ModelGenerator)
    generator.runtime = SimpleNamespace(_equation_counter=0, _expression_dependencies={})
    with generator._shared_terms() as (prefix, terms):
        value = equation()
    lines = [f"return {value.code};"] if statements is None else statements(value.code)
    body = generator._term_declarations(prefix, terms, lines)
    return f"double {name}(double input_value) {{\n" + "\n".join(body) + "\n}"


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
}
''', encoding="utf-8")
    binary = tmp_path / "expressions"
    subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True, timeout=30)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)
    rows = [[float(value) for value in row.split()] for row in result.stdout.splitlines()]
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
