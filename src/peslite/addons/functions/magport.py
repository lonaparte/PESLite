"""External magnetic data exchange and field reconstruction."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...components.magnetic import _inductance_decomposition, _validate_winding_names
from ...solver import ConfigError

__all__ = ["MagneticFieldResult", "write_magport", "magnetic_field"]


_FORMAT = "peslite-magport"
_VERSION = 1
_REQUIRED = frozenset({"format", "version", "winding_names", "inductance"})
_OPTIONAL = frozenset({"physical", "field_points", "field_basis"})


@dataclass(frozen=True)
class _MagPortData:
    winding_names: tuple[str, ...]
    inductance: np.ndarray
    physical: dict[str, Any]
    field_points: np.ndarray | None
    field_basis: np.ndarray | None


def _data(
    winding_names: Sequence[str],
    inductance: Any,
    physical: Mapping[str, Any] | None = None,
    field_points: Any | None = None,
    field_basis: Any | None = None,
) -> _MagPortData:
    names = _validate_winding_names(winding_names)
    matrix, _modes, _shapes = _inductance_decomposition(inductance, len(names))

    try:
        encoded = json.dumps({} if physical is None else physical,
                             allow_nan=False, sort_keys=True)
        source = json.loads(encoded)
    except (TypeError, ValueError):
        raise ConfigError("physical must contain finite JSON-compatible data") from None
    if not isinstance(source, dict):
        raise ConfigError("physical must be a mapping")

    points, basis = field_points, field_basis
    if (points is None) != (basis is None):
        raise ConfigError("field_points and field_basis must be provided together")
    if points is not None:
        try:
            points = np.asarray(points, dtype=float)
            basis = np.asarray(basis, dtype=float)
        except (TypeError, ValueError):
            raise ConfigError("field data must be numeric") from None
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ConfigError("field_points must have shape (points, 3)")
        expected = (len(points), 3, len(names))
        if basis.shape != expected:
            raise ConfigError(f"field_basis must have shape {expected}")
        if not np.isfinite(points).all() or not np.isfinite(basis).all():
            raise ConfigError("field data must be finite")
        points.setflags(write=False)
        basis.setflags(write=False)
    return _MagPortData(names, matrix, source, points, basis)


def _scalar(value: np.ndarray, name: str) -> Any:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ConfigError(f"{name} must be a scalar")
    return array.item()


def _vector_strings(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in ("U", "S"):
        raise ConfigError(f"{name} must be a string vector")
    return array.astype(str)


def _write(file: str | Path, data: _MagPortData) -> Path:
    path = Path(file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {
        "format": np.asarray(_FORMAT),
        "version": np.asarray(_VERSION),
        "winding_names": np.asarray(data.winding_names),
        "inductance": data.inductance,
        "physical": np.asarray(json.dumps(data.physical, allow_nan=False, sort_keys=True,
                                           separators=(",", ":"))),
    }
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


def _load(file: str | Path) -> _MagPortData:
    path = Path(file).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"magport file does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            missing = _REQUIRED - keys
            unknown = keys - _REQUIRED - _OPTIONAL
            if missing or unknown:
                details = []
                if missing:
                    details.append(f"missing entries {sorted(missing)}")
                if unknown:
                    details.append(f"unknown entries {sorted(unknown)}")
                raise ConfigError("; ".join(details))
            format_name = _scalar(archive["format"], "format")
            version = _scalar(archive["version"], "version")
            if format_name != _FORMAT:
                raise ConfigError(f"unknown magport format {format_name!r}")
            if type(version) not in (int, np.int32, np.int64) or int(version) != _VERSION:
                raise ConfigError(f"unsupported magport version {version!r}")
            names = tuple(str(name) for name in _vector_strings(
                archive["winding_names"], "winding_names"
            ))
            physical = {}
            if "physical" in keys:
                text = _scalar(archive["physical"], "physical")
                if not isinstance(text, str):
                    raise ConfigError("physical must be a JSON string")
                try:
                    physical = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ConfigError(f"physical contains invalid JSON: {exc.msg}") from None
            return _data(
                names,
                archive["inductance"].copy(),
                physical,
                archive["field_points"].copy() if "field_points" in keys else None,
                archive["field_basis"].copy() if "field_basis" in keys else None,
            )
    except ConfigError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigError(f"cannot load magport file {path}: {exc}") from None


def write_magport(
    file: str | Path,
    *,
    winding_names: Sequence[str],
    inductance: Any,
    physical: Mapping[str, Any] | None = None,
    field_points: Any | None = None,
    field_basis: Any | None = None,
) -> Path:
    """Write external inductance and optional field data in the MagPort format."""
    return _write(file, _data(
        winding_names, inductance, physical, field_points, field_basis
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
