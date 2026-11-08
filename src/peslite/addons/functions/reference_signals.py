"""Construction helpers for native table-based controller reference signals."""

from __future__ import annotations

import ast
import csv
import math
from pathlib import Path
from typing import Iterable

from ...control.signals import TableSignal


_FUNCTIONS = {
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "cos": math.cos,
    "cosh": math.cosh,
    "exp": math.exp,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "max": max,
    "min": min,
    "sin": math.sin,
    "sinh": math.sinh,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "tanh": math.tanh,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)
_UNARY = (ast.UAdd, ast.USub)


def _expression(source: str):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("expression must be nonempty text")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load)):
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("expression constants must be numbers")
            continue
        if isinstance(node, ast.Name):
            if node.id not in {"t", *_FUNCTIONS, *_CONSTANTS}:
                raise ValueError(f"expression name {node.id!r} is not available")
            continue
        if isinstance(node, ast.BinOp) and isinstance(node.op, _BINARY):
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _FUNCTIONS or node.keywords:
                raise ValueError(f"expression function {node.func.id!r} is not available")
            continue
        if isinstance(node, (*_BINARY, *_UNARY)):
            continue
        raise ValueError(f"expression operation {type(node).__name__!r} is not available")
    code = compile(tree, "<PESLite reference expression>", "eval")
    namespace = {"__builtins__": {}, **_FUNCTIONS, **_CONSTANTS}

    def evaluate(t: float) -> float:
        try:
            value = eval(code, namespace, {"t": t})
            value = float(value)
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise ValueError(f"expression failed at t = {t:.12g}: {exc}") from None
        if not math.isfinite(value):
            raise ValueError(f"expression returned a non-finite value at t = {t:.12g}")
        return value

    return evaluate


def reference_table(time: Iterable[float], value: Iterable[float], *, repeat: bool = False,
                    interpolation: str = "linear", extrapolation: str = "hold",
                    time_scale: float = 1.0, time_offset: float = 0.0,
                    value_scale: float = 1.0, value_offset: float = 0.0) -> dict:
    """Return one serializable native table after applying playback coordinates."""
    factors = (time_scale, time_offset, value_scale, value_offset)
    if (any(isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(item) for item in factors) or time_scale <= 0.0):
        raise ValueError("time_scale must be positive and all scales and offsets must be finite")
    try:
        times = [time_offset + time_scale * float(item) for item in time]
        values = [value_offset + value_scale * float(item) for item in value]
    except (TypeError, ValueError):
        raise ValueError("time and value must contain numbers") from None
    table = {
        "type": "table",
        "time": times,
        "value": values,
        "repeat": repeat,
        "interpolation": interpolation,
        "extrapolation": extrapolation,
    }
    TableSignal(**table)
    return table


def reference_csv(path: str | Path, *, time_column: str = "t", value_column: str = "value",
                  repeat: bool = False, interpolation: str = "linear",
                  extrapolation: str = "hold", time_scale: float = 1.0,
                  time_offset: float = 0.0, value_scale: float = 1.0,
                  value_offset: float = 0.0) -> dict:
    """Load two named CSV columns and return a native reference table."""
    path = Path(path).expanduser()
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from None
    if not rows:
        raise ValueError(f"{path}: CSV contains no data rows")
    fields = set(rows[0])
    missing = {time_column, value_column} - fields
    if missing:
        raise ValueError(f"{path}: missing CSV column(s) {sorted(missing)}")
    try:
        times = [float(row[time_column]) for row in rows]
        values = [float(row[value_column]) for row in rows]
    except (TypeError, ValueError):
        raise ValueError(f"{path}: selected CSV columns must contain numbers") from None
    return reference_table(
        times, values, repeat=repeat, interpolation=interpolation,
        extrapolation=extrapolation, time_scale=time_scale, time_offset=time_offset,
        value_scale=value_scale, value_offset=value_offset,
    )


def reference_expression(expression: str, *, start: float, stop: float, period: float,
                         repeat: bool = False, interpolation: str = "linear",
                         extrapolation: str = "hold", time_scale: float = 1.0,
                         time_offset: float = 0.0, value_scale: float = 1.0,
                         value_offset: float = 0.0) -> dict:
    """Sample a safe scalar expression of ``t`` and return a native reference table."""
    for name, item in (("start", start), ("stop", stop), ("period", period)):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{name} must be finite")
    if stop <= start or period <= 0.0:
        raise ValueError("stop must be greater than start and period must be positive")
    evaluate = _expression(expression)
    count = int(math.floor((stop - start) / period))
    times = [start + index * period for index in range(count + 1)]
    tolerance = 1e-12 * max(1.0, abs(start), abs(stop))
    if stop - times[-1] > tolerance:
        times.append(stop)
    else:
        times[-1] = stop
    values = [evaluate(t) for t in times]
    return reference_table(
        times, values, repeat=repeat, interpolation=interpolation,
        extrapolation=extrapolation, time_scale=time_scale, time_offset=time_offset,
        value_scale=value_scale, value_offset=value_offset,
    )


__all__ = ["reference_table", "reference_csv", "reference_expression"]
