"""Offline magnetic-model import, reluctance reduction and field reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...components.magnetic import MagneticModel
from ...solver import ConfigError

__all__ = [
    "MagneticFieldResult",
    "build_reluctance_model",
    "import_magnetic_model",
    "magnetic_field",
    "reluctance_inductance",
]


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
    """Return branch flux per winding current and the normalized reluctance matrix."""
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
    tolerance = (singular.max(initial=0.0) * max(constraints.shape)
                 * np.finfo(float).eps)
    rank = int(np.count_nonzero(singular > tolerance))
    cycles = right[rank:].T
    if not cycles.shape[1]:
        raise ConfigError("incidence leaves no independent magnetic flux path")
    reduced = cycles.T @ matrix @ cycles
    response = cycles @ np.linalg.solve(reduced, cycles.T @ turns)
    return response, matrix


def reluctance_inductance(reluctance: Any, coupling: Any,
                          incidence: Any | None = None) -> np.ndarray:
    """Reduce a linear reluctance network to its reciprocal winding inductance matrix."""
    response, _matrix_value = _flux_response(reluctance, coupling, incidence)
    turns = _matrix(coupling, "coupling")
    inductance = turns.T @ response
    return 0.5 * (inductance + inductance.T)


def import_magnetic_model(
    file: str | Path,
    *,
    winding_names: Sequence[str],
    inductance: Any,
    physical: Mapping[str, Any] | None = None,
    field_points: Any | None = None,
    field_basis: Any | None = None,
) -> Path:
    """Import analytical, FEM or other external results into the common model archive."""
    model = MagneticModel(
        winding_names=tuple(winding_names),
        inductance=inductance,
        physical={} if physical is None else physical,
        field_points=field_points,
        field_basis=field_basis,
    )
    return model.save(file)


def build_reluctance_model(
    file: str | Path,
    *,
    winding_names: Sequence[str],
    reluctance: Any,
    coupling: Any,
    incidence: Any | None = None,
    field_points: Any | None = None,
    field_per_flux: Any | None = None,
) -> Path:
    """Reduce one linear reluctance network and write the common model archive."""
    response, matrix = _flux_response(reluctance, coupling, incidence)
    turns = _matrix(coupling, "coupling")
    names = tuple(winding_names)
    if turns.shape[1] != len(names):
        raise ConfigError(
            f"coupling has {turns.shape[1]} windings, but {len(names)} names were supplied"
        )
    inductance = 0.5 * (turns.T @ response + response.T @ turns)
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
    return import_magnetic_model(
        file,
        winding_names=names,
        inductance=inductance,
        physical=physical,
        field_points=field_points,
        field_basis=basis,
    )


@dataclass(frozen=True)
class MagneticFieldResult:
    """Magnetic flux density reconstructed at named spatial points over simulation time."""

    time: np.ndarray
    points: np.ndarray
    flux_density: np.ndarray


def _field_basis(model: MagneticModel, points: Any | None) -> tuple[np.ndarray, np.ndarray]:
    if model.field_points is None:
        raise ConfigError("magnetic model does not contain a spatial field map")
    if points is None:
        return model.field_points, model.field_basis
    requested = np.asarray(points, dtype=float)
    if requested.ndim == 1:
        requested = requested.reshape(1, -1)
    if requested.ndim != 2 or requested.shape[1] != 3 or not np.isfinite(requested).all():
        raise ConfigError("points must have shape (points, 3) and finite coordinates")
    try:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import QhullError

        interpolator = LinearNDInterpolator(
            model.field_points, model.field_basis.reshape(len(model.field_points), -1)
        )
        basis = np.asarray(interpolator(requested), dtype=float)
    except (ValueError, QhullError) as exc:
        raise ConfigError(f"cannot interpolate the magnetic field map: {exc}") from None
    if (basis.shape != (len(requested), 3 * len(model.winding_names))
            or not np.isfinite(basis).all()):
        raise ConfigError("requested points lie outside the magnetic field map")
    return requested, basis.reshape(len(requested), 3, len(model.winding_names))


def magnetic_field(result: Any, model_file: str | Path, *, element: str,
                   points: Any | None = None) -> MagneticFieldResult:
    """Reconstruct ``B(x, y, z, t)`` from recorded winding currents and a field map."""
    model = MagneticModel.load(model_file)
    locations, basis = _field_basis(model, points)
    columns = result.columns()
    paths = tuple(f"{element}.i_{name}" for name in model.winding_names)
    missing = [path for path in paths if path not in columns]
    if missing:
        raise ConfigError(f"simulation result is missing winding currents {missing}")
    currents = np.column_stack([columns[path] for path in paths])
    values = np.einsum("tn,pcn->tpc", currents, basis)
    return MagneticFieldResult(
        time=np.asarray(result.t, dtype=float).copy(),
        points=np.asarray(locations, dtype=float).copy(),
        flux_density=values,
    )
