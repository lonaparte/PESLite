"""Linear magnetic-circuit reduction to winding inductance matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...solver import ConfigError
from .magport import write_magport

__all__ = ["export_magnetic_circuit", "reluctance_inductance"]


def _matrix(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a finite numeric matrix") from None
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ConfigError(f"{name} must be a finite numeric matrix")
    return array


def _reluctance(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError("reluctance must be a finite numeric vector or matrix") from None
    if array.ndim == 1:
        array = np.diag(array)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not len(array):
        raise ConfigError("reluctance must be a nonempty vector or square matrix")
    if not np.isfinite(array).all() or not np.allclose(array, array.T, rtol=1e-12, atol=0.0):
        raise ConfigError("reluctance must be finite and symmetric")
    if np.linalg.eigvalsh(array).min() <= 0.0:
        raise ConfigError("reluctance must be positive definite")
    return array


def _flux_response(reluctance: Any, coupling: Any,
                   incidence: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    matrix = _reluctance(reluctance)
    turns = _matrix(coupling, "coupling")
    branches = len(matrix)
    if turns.shape[0] != branches or not turns.shape[1]:
        raise ConfigError(
            f"coupling must have shape ({branches}, windings) with at least one winding"
        )
    if incidence is None:
        return np.linalg.solve(matrix, turns), matrix

    constraints = _matrix(incidence, "incidence")
    if constraints.shape[1] != branches:
        raise ConfigError(f"incidence must have {branches} columns")
    _u, singular, right = np.linalg.svd(constraints, full_matrices=True)
    tolerance = singular.max(initial=0.0) * max(constraints.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tolerance))
    cycles = right[rank:].T
    if not cycles.shape[1]:
        raise ConfigError("incidence leaves no independent magnetic flux path")
    reduced = cycles.T @ matrix @ cycles
    return cycles @ np.linalg.solve(reduced, cycles.T @ turns), matrix


def _inductance(turns: np.ndarray, response: np.ndarray) -> np.ndarray:
    matrix = turns.T @ response
    return 0.5 * (matrix + matrix.T)


def reluctance_inductance(reluctance: Any, coupling: Any,
                          incidence: Any | None = None) -> np.ndarray:
    """Reduce a linear reluctance network to a reciprocal winding inductance matrix."""
    response, _matrix_value = _flux_response(reluctance, coupling, incidence)
    turns = _matrix(coupling, "coupling")
    return _inductance(turns, response)


def export_magnetic_circuit(
    file: str | Path,
    *,
    winding_names: Sequence[str],
    reluctance: Any,
    coupling: Any,
    incidence: Any | None = None,
    field_points: Any | None = None,
    field_per_flux: Any | None = None,
) -> Path:
    """Reduce a linear reluctance network and write its result through MagPort."""
    response, matrix = _flux_response(reluctance, coupling, incidence)
    turns = _matrix(coupling, "coupling")
    names = tuple(winding_names)
    if turns.shape[1] != len(names):
        raise ConfigError(
            f"coupling has {turns.shape[1]} windings, but {len(names)} names were supplied"
        )
    inductance = _inductance(turns, response)
    basis = None
    if field_per_flux is not None:
        try:
            locations = None if field_points is None else np.asarray(field_points, dtype=float)
            branch_field = np.asarray(field_per_flux, dtype=float)
        except (TypeError, ValueError):
            raise ConfigError("field data must be numeric") from None
        expected = ((len(locations), 3, len(matrix))
                    if locations is not None and locations.ndim > 0 else None)
        if (expected is None or branch_field.shape != expected
                or not np.isfinite(branch_field).all()):
            raise ConfigError(
                "field_per_flux requires field_points and shape (points, 3, branches)"
            )
        basis = np.einsum("pcb,bn->pcn", branch_field, response)
    physical = {
        "reluctance_network": {
            "reluctance": matrix.tolist(),
            "coupling": turns.tolist(),
            "incidence": None if incidence is None else _matrix(incidence, "incidence").tolist(),
        }
    }
    return write_magport(
        file,
        winding_names=names,
        inductance=inductance,
        physical=physical,
        field_points=field_points,
        field_basis=basis,
    )
