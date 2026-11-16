"""Small dependency-free differentiation tools shared by model adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["directional_derivative"]


def directional_derivative(
    function: Callable[[np.ndarray], Any],
    point: Sequence[complex | float] | np.ndarray,
    direction: Sequence[complex | float] | np.ndarray,
) -> np.ndarray:
    """Return the centered numerical derivative of ``function`` along ``direction``.

    The step limits the largest relative coordinate perturbation. This keeps one shared,
    unit-independent rule for constitutive models whose analytic derivative is unavailable.
    """
    point = np.asarray(point)
    direction = np.asarray(direction)
    if point.shape != direction.shape:
        raise ValueError("point and direction must have the same shape")
    relative_speed = np.max(
        np.abs(direction) / np.maximum(1.0, np.abs(point)), initial=0.0
    )
    if relative_speed == 0.0:
        value = np.asarray(function(point))
        return np.zeros(value.shape, dtype=np.result_type(value, float))
    step = np.sqrt(np.finfo(float).eps) / relative_speed
    before = np.asarray(function(point - step * direction))
    after = np.asarray(function(point + step * direction))
    if before.shape != after.shape:
        raise ValueError("function output shape changed during differentiation")
    dtype = np.result_type(before, after, float)
    return (after.astype(dtype, copy=False) - before.astype(dtype, copy=False)) / (2.0 * step)
