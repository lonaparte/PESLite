"""External magnetic data exchange and field reconstruction."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..components.magport import (
    _FORMAT,
    _VERSION,
    _MagPortData,
    _data,
    _load,
)
from ...components.magnetic import MagneticDescription
from ...solver import ConfigError

__all__ = ["MagneticFieldResult", "write_magport", "magnetic_field"]


def _write(file: str | Path, data: _MagPortData) -> Path:
    path = Path(file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {
        "format": np.asarray(_FORMAT),
        "version": np.asarray(_VERSION),
        "winding_names": np.asarray(data.winding_names),
        "physical": np.asarray(json.dumps(data.physical, allow_nan=False, sort_keys=True,
                                           separators=(",", ":"))),
    }
    if data.inductance is not None:
        entries["inductance"] = data.inductance
    else:
        entries["flux_linkage"] = data.flux_linkage
        entries["current"] = data.current
    if data.magnetic is not None:
        entries["magnetic"] = np.asarray(json.dumps(
            asdict(data.magnetic), allow_nan=False, sort_keys=True, separators=(",", ":")
        ))
    if data.field_points is not None:
        entries["field_points"] = data.field_points
        entries["field_basis"] = data.field_basis
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **entries)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def write_magport(
    file: str | Path,
    *,
    winding_names: Sequence[str],
    inductance: Any | None = None,
    flux_linkage: Any | None = None,
    current: Any | None = None,
    physical: Mapping[str, Any] | None = None,
    magnetic: MagneticDescription | Mapping[str, Any] | None = None,
    field_points: Any | None = None,
    field_basis: Any | None = None,
) -> Path:
    """Write a linear or sampled nonlinear constitutive law in the MagPort format."""
    return _write(file, _data(
        winding_names, inductance, flux_linkage, current,
        physical, magnetic, field_points, field_basis,
    ))


@dataclass(frozen=True)
class MagneticFieldResult:
    """Magnetic flux density reconstructed at spatial points over simulation time."""

    time: np.ndarray
    points: np.ndarray
    flux_density: np.ndarray


def _field_basis(data: _MagPortData, points: Any | None) -> tuple[np.ndarray, np.ndarray]:
    if data.field_points is None:
        raise ConfigError("magport file does not contain a spatial field map")
    if points is None:
        return data.field_points, data.field_basis
    requested = np.asarray(points, dtype=float)
    if requested.ndim == 1:
        requested = requested.reshape(1, -1)
    if requested.ndim != 2 or requested.shape[1] != 3 or not np.isfinite(requested).all():
        raise ConfigError("points must have shape (points, 3) and finite coordinates")
    try:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import QhullError

        interpolator = LinearNDInterpolator(
            data.field_points, data.field_basis.reshape(len(data.field_points), -1)
        )
        basis = np.asarray(interpolator(requested), dtype=float)
    except (ValueError, QhullError) as exc:
        raise ConfigError(f"cannot interpolate the magnetic field map: {exc}") from None
    expected = (len(requested), 3 * len(data.winding_names))
    if basis.shape != expected or not np.isfinite(basis).all():
        raise ConfigError("requested points lie outside the magnetic field map")
    return requested, basis.reshape(len(requested), 3, len(data.winding_names))


def magnetic_field(result: Any, model_file: str | Path, *, element: str,
                   points: Any | None = None) -> MagneticFieldResult:
    """Reconstruct ``B(x, y, z, t)`` from recorded winding currents and a field map."""
    data = _load(model_file)
    locations, basis = _field_basis(data, points)
    columns = result.columns()
    paths = tuple(f"{element}.i_{name}" for name in data.winding_names)
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
