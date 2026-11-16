"""Offline magnetic-circuit reduction and MagPort exchange helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...solver import ConfigError
from ..components.magcircuit import _linear_circuit, _matrix
from .magport import write_magport

__all__ = ["export_magnetic_circuit", "reluctance_inductance"]


def reluctance_inductance(
    reluctance: Any, coupling: Any, incidence: Any | None = None
) -> np.ndarray:
    """Return the exact winding inductance of a linear magnetic circuit."""
    inductance, _response, _matrix_value = _linear_circuit(
        reluctance, coupling, incidence
    )
    return inductance


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
    """Write the exact linear reduction of one magnetic circuit through MagPort."""
    inductance, response, matrix = _linear_circuit(
        reluctance, coupling, incidence
    )
    turns = _matrix(coupling, "coupling")
    names = tuple(winding_names)
    if turns.shape[1] != len(names):
        raise ConfigError(
            f"coupling has {turns.shape[1]} windings, but {len(names)} names were supplied"
        )
    basis = None
    if field_per_flux is not None:
        try:
            locations = None if field_points is None else np.asarray(field_points, dtype=float)
            branch_field = np.asarray(field_per_flux, dtype=float)
        except (TypeError, ValueError):
            raise ConfigError("field data must be numeric") from None
        expected = (
            (len(locations), 3, len(matrix))
            if locations is not None and locations.ndim > 0 else None
        )
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
            "incidence": (
                None if incidence is None else _matrix(incidence, "incidence").tolist()
            ),
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
