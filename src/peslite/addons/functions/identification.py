"""Reproducible identification excitations using the native reference table schema."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .reference_signals import _sample_grid, reference_table

__all__ = ["reference_prbs", "reference_multitone", "reference_chirp"]

# Primitive right-shift Galois LFSR feedback masks (period 2**order - 1).
_FEEDBACK = {
    2: 0x3, 3: 0x6, 4: 0xC, 5: 0x14, 6: 0x30, 7: 0x60,
    8: 0xB8, 9: 0x110, 10: 0x240, 11: 0x500, 12: 0x829,
    13: 0x100D, 14: 0x2015, 15: 0x6000, 16: 0xB400,
}


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
    """Generate a maximal-length binary LFSR sequence with zero-order hold.

    ``period`` is one bit interval. A full sequence repeats after ``2**order-1``
    intervals. The nonzero ``seed`` fixes the phase, without global RNG state.
    ``repeat`` repeats exactly the provided table window, which need not contain
    a complete LFSR cycle.
    """
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
    """Sample a sum of sinusoids using time relative to ``start``.

    Frequencies are in Hz, phases in radians. Repeated tables require every
    tone to complete an integer number of cycles within the requested window.
    """
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
        values[-1] = values[0]  # Avoid a floating-point seam on periodic playback.
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
    """Generate a linear-frequency sweep in Hz without introducing runtime state."""
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
