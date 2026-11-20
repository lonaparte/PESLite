"""Shared finite-difference linearization of an application's complete loop map."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..defaults import DEFAULT_LINEARIZATION_RELATIVE_STEP

__all__ = ["LoopMap", "LoopComparison", "LoopLinearization", "coordinate_width",
           "active_coordinates", "coordinate_difference", "compare_loop_states",
           "linearize_loop"]


@runtime_checkable
class LoopMap(Protocol):
    """One-period advance and the existing named-state coordinate transformations.

    Coordinates are ``(kind, names, scale)`` groups. ``vector`` and ``fixed`` have two entries,
    ``triple`` has three, and other kinds have one. ``angle`` is in radians; ``frozen`` and
    ``mode`` are restored but not perturbed. ``scale`` supplies a finite-difference step floor.
    The optional ``view`` argument belongs to split-error analysis: it offsets a held state in
    the integration (``b``) or end-of-period sampler (``c``).
    """

    period: float

    def coordinates(self, names: list[str], *, linearization: bool = True
                    ) -> list[tuple[str, list[str], float]]: ...

    def to_frame(self, row: dict[str, float], t: float, cols: list) -> np.ndarray: ...

    def from_frame(self, x: np.ndarray, t: float, cols: list,
                   row: dict[str, float]) -> dict[str, float]: ...

    def advance(self, row: dict[str, float], t: float,
                view: tuple[str, float, str] | None = None) -> dict[str, float]: ...

    def periodic_state(self, row: dict[str, float]) -> dict[str, float]: ...


def coordinate_width(kind: str) -> int:
    return 2 if kind in ("vector", "fixed") else 3 if kind == "triple" else 1


def active_coordinates(cols: list) -> tuple[int, ...]:
    """Indices of the differentiable entries in the existing coordinate layout."""
    indices = []
    j = 0
    for kind, _names, _scale in cols:
        width = coordinate_width(kind)
        if kind not in ("frozen", "mode"):
            indices.extend(range(j, j + width))
        j += width
    return tuple(indices)


def coordinate_difference(a: np.ndarray, b: np.ndarray, cols: list) -> np.ndarray:
    """Subtract coordinate vectors, respecting the local difference of angular coordinates."""
    delta = a - b
    j = 0
    for kind, _names, _scale in cols:
        if kind == "angle":
            delta[j] = math.remainder(float(delta[j]), math.tau)
        j += coordinate_width(kind)
    return delta


@dataclass(frozen=True)
class LoopComparison:
    """Scaled recurrence comparison in a loop map's physical coordinates."""

    error: float
    worst_coordinate: str | None
    changed_modes: tuple[str, ...]
    initial: np.ndarray
    final: np.ndarray
    scales: np.ndarray
    coordinates: tuple[str, ...]


def compare_loop_states(loop: LoopMap, initial: dict[str, float], final: dict[str, float],
                        start: float, end: float) -> LoopComparison:
    """Compare two logical loop states with shared frame, scale and mode semantics."""
    first = loop.periodic_state(initial)
    last = loop.periodic_state(final)
    cols = loop.coordinates(list(first), linearization=False)
    a = loop.to_frame(first, start, cols)
    b = loop.to_frame(last, end, cols)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("loop states must be finite and have matching coordinates")
    scales = (np.concatenate([np.full(coordinate_width(kind), floor)
                              for kind, _names, floor in cols])
              if cols else np.empty(0))
    scales = np.maximum(scales, np.maximum(np.abs(a), np.abs(b)))
    names: list[str] = []
    changed_modes: list[str] = []
    index = 0
    for kind, group, floor in cols:
        names.extend(group)
        if kind == "angle":
            scales[index] = floor
        elif kind in ("frozen", "mode") and a[index] != b[index]:
            changed_modes.append(group[0])
        index += coordinate_width(kind)
    active = active_coordinates(cols)
    if not active:
        return LoopComparison(0.0, None, tuple(changed_modes), a, b, scales, tuple(names))
    selected = list(active)
    residual = np.abs(coordinate_difference(b, a, cols))[selected] / scales[selected]
    worst = int(np.argmax(residual))
    return LoopComparison(float(residual[worst]), names[active[worst]], tuple(changed_modes),
                          a, b, scales, tuple(names))


@dataclass(frozen=True)
class LoopLinearization:
    """Full restart-coordinate map; ``active`` selects its differentiable subspace."""

    phi: np.ndarray
    initial: np.ndarray
    final: np.ndarray
    active: tuple[int, ...]
    changed_modes: tuple[str, ...]
    evaluations: int


def linearize_loop(loop: LoopMap, row: dict[str, float], t0: float, cols: list,
                   *, relative_step: float = DEFAULT_LINEARIZATION_RELATIVE_STEP,
                   baseline: dict[str, float] | None = None) -> LoopLinearization:
    """Central differences of the complete map, including one reusable baseline evaluation.

    Fixed columns retain the historical identity convention for split-error analysis. Modal
    analysis must select both rows and columns with ``active`` instead of interpreting those
    bookkeeping entries as dynamic modes. A caller may validate a baseline first and pass its
    state row here to avoid repeating that evaluation; ``evaluations`` counts only actual calls.
    Discrete endpoints are compared with the baseline: nominal clock/lifecycle evolution is not
    a perturbation-dependent mode change. A periodic-state caller checks baseline recurrence.
    Optional ``loop.overwritten_columns(t0)`` proves zero columns without perturbation runs.
    Their rows, coordinates and zero multipliers remain present; unknown maps stay exhaustive.
    """
    if not math.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    x0 = loop.to_frame(row, t0, cols)
    changed: set[str] = set()
    fixed = []
    j = 0
    for kind, names, _floor in cols:
        width = coordinate_width(kind)
        if kind in ("frozen", "mode"):
            fixed.append((j, names[0]))
        j += width

    def converted(values):
        out = loop.to_frame(values, t0 + loop.period, cols)
        if out.shape != x0.shape or not np.all(np.isfinite(out)):
            raise ValueError("loop advance returned a non-finite or inconsistent state vector")
        return out

    if not np.all(np.isfinite(x0)):
        raise ValueError("linearization requires finite initial states")
    base = converted(loop.advance(dict(row), t0) if baseline is None else baseline)
    phi = np.eye(len(x0))
    evaluations = int(baseline is None)
    declare = getattr(loop, "overwritten_columns", None)
    overwritten = frozenset() if declare is None else declare(t0)
    j = 0
    for kind, names, floor in cols:
        if kind not in ("frozen", "mode"):
            if all(name in overwritten for name in names):
                phi[:, j:j + coordinate_width(kind)] = 0.0
                j += coordinate_width(kind)
                continue
            for w in range(coordinate_width(kind)):
                value = float(x0[j + w])
                if kind == "angle":
                    value = math.remainder(value, math.tau)
                h = relative_step * max(abs(value), floor)
                if not math.isfinite(h) or h <= 0.0:
                    raise ValueError("each differentiable coordinate needs a positive finite step scale")
                outs = []
                for sign in (1.0, -1.0):
                    xp = x0.copy()
                    xp[j + w] += sign * h
                    outs.append(converted(loop.advance(loop.from_frame(xp, t0, cols, row), t0)))
                    changed.update(name for index, name in fixed
                                   if outs[-1][index] != base[index])
                    evaluations += 1
                phi[:, j + w] = coordinate_difference(outs[0], outs[1], cols) / (2.0 * h)
        j += coordinate_width(kind)
    return LoopLinearization(phi, x0, base, active_coordinates(cols), tuple(sorted(changed)), evaluations)
