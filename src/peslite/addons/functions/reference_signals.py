"""Construction helpers for native table-based controller reference signals."""

from __future__ import annotations

import ast
import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable

from ...solver.signals import TableSignal


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

# Primitive right-shift Galois LFSR feedback masks (period 2**order - 1).
_FEEDBACK = {
    2: 0x3, 3: 0x6, 4: 0xC, 5: 0x14, 6: 0x30, 7: 0x60,
    8: 0xB8, 9: 0x110, 10: 0x240, 11: 0x500, 12: 0x829,
    13: 0x100D, 14: 0x2015, 15: 0x6000, 16: 0xB400,
}


def _sample_grid(start: float, stop: float, period: float) -> list[float]:
    """Return a bounded, strictly increasing grid including both endpoints."""
    for name, item in (("start", start), ("stop", stop), ("period", period)):
        if (isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(item)):
            raise ValueError(f"{name} must be finite")
    if stop <= start or period <= 0.0:
        raise ValueError("stop must be greater than start and period must be positive")
    count = int(math.floor((stop - start) / period))
    if count > 1_000_000:
        raise ValueError("the reference table would exceed one million samples")
    times = [start + index * period for index in range(count + 1)]
    tolerance = 1e-12 * max(1.0, abs(start), abs(stop))
    if stop - times[-1] > tolerance:
        times.append(stop)
    else:
        times[-1] = stop
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("the requested sample grid is not representable")
    return times


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
    if isinstance(time, (str, bytes)) or isinstance(value, (str, bytes)):
        raise ValueError("time and value must contain numbers")
    try:
        raw_time, raw_value = tuple(time), tuple(value)
    except (TypeError, ValueError):
        raise ValueError("time and value must contain numbers") from None
    if any(isinstance(item, bool) for item in (*raw_time, *raw_value)):
        raise ValueError("time and value must contain numbers")
    try:
        times = [time_offset + time_scale * float(item) for item in raw_time]
        values = [value_offset + value_scale * float(item) for item in raw_value]
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
    evaluate = _expression(expression)
    times = _sample_grid(start, stop, period)
    values = [evaluate(t) for t in times]
    return reference_table(
        times, values, repeat=repeat, interpolation=interpolation,
        extrapolation=extrapolation, time_scale=time_scale, time_offset=time_offset,
        value_scale=value_scale, value_offset=value_offset,
    )


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def reference_prbs(*, start: float, stop: float, period: float, order: int = 7,
                   seed: int = 1, low: float = -1.0, high: float = 1.0,
                   repeat: bool = False, time_scale: float = 1.0,
                   time_offset: float = 0.0, value_scale: float = 1.0,
                   value_offset: float = 0.0) -> dict:
    """Generate a maximal-length binary LFSR sequence with zero-order hold."""
    if isinstance(order, bool) or not isinstance(order, int) or order not in _FEEDBACK:
        raise ValueError("order must be an integer between 2 and 16")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= seed < (1 << order):
        raise ValueError("seed must be a nonzero LFSR state below 2**order")
    low, high = _finite("low", low), _finite("high", high)
    if low == high:
        raise ValueError("low and high must differ")
    times = _sample_grid(start, stop, period)
    state = seed
    values = []
    for _ in range(len(times) - 1):
        values.append(high if state & 1 else low)
        state = (state >> 1) ^ (_FEEDBACK[order] if state & 1 else 0)
    values.append(values[0] if repeat else values[-1])
    return reference_table(
        times, values, repeat=repeat, interpolation="zero_order_hold",
        time_scale=time_scale, time_offset=time_offset,
        value_scale=value_scale, value_offset=value_offset,
    )


def reference_multitone(*, start: float, stop: float, period: float,
                        frequencies: Sequence[float], amplitudes: Sequence[float] | None = None,
                        phases: Sequence[float] | None = None, offset: float = 0.0,
                        repeat: bool = False, time_scale: float = 1.0,
                        time_offset: float = 0.0, value_scale: float = 1.0,
                        value_offset: float = 0.0) -> dict:
    """Sample a sum of sinusoidal reference signals using time relative to ``start``."""
    times = _sample_grid(start, stop, period)
    try:
        freq = tuple(_finite("frequency", f) for f in frequencies)
        amp = (1.0,) * len(freq) if amplitudes is None else tuple(
            _finite("amplitude", a) for a in amplitudes)
        phase = (0.0,) * len(freq) if phases is None else tuple(
            _finite("phase", p) for p in phases)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid multitone parameters: {exc}") from None
    if not freq or len(freq) != len(amp) or len(freq) != len(phase):
        raise ValueError("frequencies, amplitudes, and phases must have matching nonzero lengths")
    if any(f <= 0 or f >= 0.5 / period for f in freq) or len(set(freq)) != len(freq):
        raise ValueError("frequencies must be distinct and strictly below Nyquist")
    if repeat and any(not math.isclose((stop - start) * f, round((stop - start) * f),
                                       abs_tol=1e-9, rel_tol=1e-9) for f in freq):
        raise ValueError("repeated multitone tables require integer cycles of every tone")
    offset = _finite("offset", offset)
    values = [offset + sum(a * math.sin(math.tau * f * (t - start) + p)
                           for f, a, p in zip(freq, amp, phase)) for t in times]
    if repeat:
        values[-1] = values[0]
    return reference_table(
        times, values, repeat=repeat, interpolation="linear",
        time_scale=time_scale, time_offset=time_offset,
        value_scale=value_scale, value_offset=value_offset,
    )


def reference_chirp(*, start: float, stop: float, period: float,
                    start_frequency: float, stop_frequency: float, amplitude: float = 1.0,
                    phase: float = 0.0, offset: float = 0.0,
                    time_scale: float = 1.0, time_offset: float = 0.0,
                    value_scale: float = 1.0, value_offset: float = 0.0) -> dict:
    """Generate a linear-frequency sweep without introducing runtime state."""
    times = _sample_grid(start, stop, period)
    f0 = _finite("start_frequency", start_frequency)
    f1 = _finite("stop_frequency", stop_frequency)
    amplitude, phase, offset = (_finite("amplitude", amplitude),
                                _finite("phase", phase), _finite("offset", offset))
    if f0 < 0 or f1 < 0 or max(f0, f1) >= 0.5 / period:
        raise ValueError("chirp frequencies must be nonnegative and strictly below Nyquist")
    slope = (f1 - f0) / (stop - start)
    values = [offset + amplitude * math.sin(
        phase + math.tau * (f0 * (t - start) + 0.5 * slope * (t - start) ** 2))
        for t in times]
    return reference_table(
        times, values, interpolation="linear", time_scale=time_scale,
        time_offset=time_offset, value_scale=value_scale, value_offset=value_offset,
    )


__all__ = ["reference_table", "reference_csv", "reference_expression", "reference_prbs",
           "reference_multitone", "reference_chirp"]
