"""Nonlinear magnetic constitutive models connected through ordinary electrical ports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .magnetic import (
    MagComponent,
    MagneticDescription,
    MagneticParams,
    _MagneticElement,
    _inductance_decomposition,
    _magnetic_description,
    _validate_winding_names,
)
from ...components.network import register_component_type
from ...solver import Bag, ConfigError, Port

__all__ = [
    "LinearMagneticLaw",
    "MagneticLaw",
    "MagPort",
    "MagPortElement",
    "MagPortModel",
    "TabulatedMagneticLaw",
]


_FORMAT = "peslite-magport"
_VERSION = 2
_REQUIRED = frozenset({"format", "version", "winding_names"})
_LINEAR = frozenset({"inductance"})
_NONLINEAR = frozenset({"flux_linkage", "current"})
_OPTIONAL = frozenset({
    "physical", "magnetic", "field_points", "field_basis"
}) | _LINEAR | _NONLINEAR


@dataclass(frozen=True)
class _MagPortData:
    winding_names: tuple[str, ...]
    inductance: np.ndarray | None
    flux_linkage: np.ndarray | None
    current: np.ndarray | None
    physical: dict[str, Any]
    magnetic: MagneticDescription | None
    field_points: np.ndarray | None
    field_basis: np.ndarray | None


def _current_map(
    winding_names: Sequence[str], flux_linkage: Any, current: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one sampled explicit current map and return immutable arrays."""
    count = len(tuple(winding_names))
    try:
        flux = np.asarray(flux_linkage, dtype=float)
        currents = np.asarray(current, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError("flux_linkage and current samples must be numeric") from None
    if count == 1 and flux.ndim == currents.ndim == 1:
        flux, currents = flux[:, None], currents[:, None]
    if flux.ndim != 2 or flux.shape[1] != count or len(flux) < count + 1:
        raise ConfigError(
            f"flux_linkage samples must have shape (samples, {count}) with at least "
            f"{count + 1} samples"
        )
    if currents.shape != flux.shape:
        raise ConfigError(f"current samples must have shape {flux.shape}")
    if not np.isfinite(flux).all() or not np.isfinite(currents).all():
        raise ConfigError("flux_linkage and current samples must be finite")
    if count == 1:
        order = np.argsort(flux[:, 0])
        flux, currents = flux[order], currents[order]
        if np.any(np.diff(flux[:, 0]) <= 0.0):
            raise ConfigError("single-winding flux_linkage samples must be unique")
    elif np.linalg.matrix_rank(flux - flux[0]) != count:
        raise ConfigError("flux_linkage samples must span the winding space")
    flux, currents = flux.copy(), currents.copy()
    flux.setflags(write=False)
    currents.setflags(write=False)
    return flux, currents


def _data(
    winding_names: Sequence[str],
    inductance: Any | None = None,
    flux_linkage: Any | None = None,
    current: Any | None = None,
    physical: Mapping[str, Any] | None = None,
    magnetic: MagneticDescription | Mapping[str, Any] | None = None,
    field_points: Any | None = None,
    field_basis: Any | None = None,
) -> _MagPortData:
    names = _validate_winding_names(winding_names)
    linear = inductance is not None
    nonlinear = flux_linkage is not None or current is not None
    if linear == nonlinear:
        raise ConfigError(
            "provide either inductance or both flux_linkage and current samples"
        )
    if linear:
        matrix, _modes, _shapes = _inductance_decomposition(inductance, len(names))
        flux_samples = current_samples = None
    else:
        if flux_linkage is None or current is None:
            raise ConfigError("flux_linkage and current samples must be provided together")
        flux_samples, current_samples = _current_map(names, flux_linkage, current)
        matrix = None

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
    description = _magnetic_description(magnetic)
    if description is not None:
        description.matrices(names)
    return _MagPortData(
        names, matrix, flux_samples, current_samples, source, description, points, basis
    )


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
            constitutive = keys & (_LINEAR | _NONLINEAR)
            if constitutive not in (_LINEAR, _NONLINEAR):
                raise ConfigError(
                    "magport file must contain either inductance or both "
                    "flux_linkage and current"
                )
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
            magnetic = None
            if "magnetic" in keys:
                text = _scalar(archive["magnetic"], "magnetic")
                if not isinstance(text, str):
                    raise ConfigError("magnetic must be a JSON string")
                try:
                    magnetic = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ConfigError(f"magnetic contains invalid JSON: {exc.msg}") from None
            return _data(
                names,
                inductance=(archive["inductance"].copy()
                            if "inductance" in keys else None),
                flux_linkage=(archive["flux_linkage"].copy()
                              if "flux_linkage" in keys else None),
                current=(archive["current"].copy() if "current" in keys else None),
                physical=physical,
                magnetic=magnetic,
                field_points=(archive["field_points"].copy()
                              if "field_points" in keys else None),
                field_basis=(archive["field_basis"].copy()
                             if "field_basis" in keys else None),
            )
    except ConfigError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigError(f"cannot load magport file {path}: {exc}") from None


class MagneticLaw(Protocol):
    """Explicit magnetic constitutive law used by :class:`MagPortModel`.

    ``currents`` maps winding flux linkages and optional internal magnetic states to winding
    currents. PESLite owns and integrates every declared state. A stateful law also provides
    ``derivatives``. A conservative law may provide ``stored_energy`` and
    ``stored_energy_rate``; the electrical adapter then exposes an exact energy declaration.
    """

    winding_names: tuple[str, ...]
    initial_state: Mapping[str, float]

    def currents(
        self, t: float, flux_linkage: np.ndarray, state: Mapping[str, float]
    ) -> Sequence[float]: ...

class LinearMagneticLaw:
    """Reciprocal constant-inductance constitutive law ``i = L^-1 lambda``."""

    initial_state: Mapping[str, float] = {}

    def __init__(self, winding_names: Sequence[str], inductance: Any) -> None:
        self.winding_names = _validate_winding_names(winding_names)
        matrix, _modes, _shapes = _inductance_decomposition(
            inductance, len(self.winding_names)
        )
        self.inductance = matrix
        self._inverse = np.linalg.solve(matrix, np.eye(len(matrix)))

    def currents(
        self, _t: float, flux_linkage: np.ndarray, _state: Mapping[str, float]
    ) -> np.ndarray:
        return self._inverse @ flux_linkage

    def stored_energy(
        self, _t: float, flux_linkage: np.ndarray, _state: Mapping[str, float]
    ) -> float:
        return float(0.5 * flux_linkage @ self._inverse @ flux_linkage)

    @staticmethod
    def stored_energy_rate(
        _t: float,
        _flux_linkage: np.ndarray,
        currents: np.ndarray,
        _state: Mapping[str, float],
        flux_rate: np.ndarray,
        _state_rate: Mapping[str, float],
    ) -> float:
        return float(currents @ flux_rate)

class TabulatedMagneticLaw:
    """Memoryless nonlinear current map sampled over winding flux-linkage space."""

    initial_state: Mapping[str, float] = {}

    def __init__(self, winding_names: Sequence[str], flux_linkage: Any, current: Any) -> None:
        self.winding_names = _validate_winding_names(winding_names)
        count = len(self.winding_names)
        self.flux_linkage, self.current = _current_map(
            self.winding_names, flux_linkage, current
        )
        if count == 1:
            from scipy.interpolate import PchipInterpolator

            self._interpolator = PchipInterpolator(
                self.flux_linkage[:, 0], self.current[:, 0], extrapolate=True
            )
            self._energy_integral = self._interpolator.antiderivative()
            self._energy_origin = float(self._energy_integral(0.0))
            self._extrapolator = None
            self.stored_energy = self._stored_energy
            self.stored_energy_rate = self._stored_energy_rate
        else:
            try:
                from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
                from scipy.spatial import QhullError

                self._interpolator = LinearNDInterpolator(
                    self.flux_linkage, self.current, fill_value=np.nan
                )
                self._extrapolator = NearestNDInterpolator(
                    self.flux_linkage, self.current
                )
            except (ValueError, QhullError) as exc:
                raise ConfigError(f"cannot construct the magnetic current map: {exc}") from None
            self._energy_integral = None
            self._energy_origin = 0.0

    def currents(
        self, _t: float, flux_linkage: np.ndarray, _state: Mapping[str, float]
    ) -> np.ndarray:
        argument = flux_linkage[0] if len(self.winding_names) == 1 else flux_linkage
        value = np.asarray(self._interpolator(argument), dtype=float).reshape(-1)
        if self._extrapolator is not None and not np.isfinite(value).all():
            value = np.asarray(self._extrapolator(argument), dtype=float).reshape(-1)
        if value.shape != (len(self.winding_names),) or not np.isfinite(value).all():
            raise ConfigError("magnetic current interpolation returned invalid values")
        return value

    def _stored_energy(
        self, _t: float, flux_linkage: np.ndarray, _state: Mapping[str, float]
    ) -> float:
        return float(self._energy_integral(flux_linkage[0]) - self._energy_origin)

    @staticmethod
    def _stored_energy_rate(
        _t: float,
        _flux_linkage: np.ndarray,
        currents: np.ndarray,
        _state: Mapping[str, float],
        flux_rate: np.ndarray,
        _state_rate: Mapping[str, float],
    ) -> float:
        return float(currents @ flux_rate)

def _numeric_vector(values: Any, size: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError(f"{label} must return {size} finite numeric values") from None
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ConfigError(f"{label} must return {size} finite numeric values")
    return vector


class MagPortModel(MagComponent):
    """Dedicated magnetic state subsystem with two electrical terminals per winding."""

    outputs_need_inputs = False

    def __init__(
        self,
        law: MagneticLaw,
        resistance: Sequence[float],
        terminals: Sequence[tuple[bool, bool]],
        magnetic: MagneticDescription | None = None,
    ) -> None:
        super().__init__(magnetic)
        if not callable(getattr(law, "currents", None)):
            raise TypeError("a magnetic law must define currents(t, flux_linkage, state)")
        self.law = law
        self.names = _validate_winding_names(law.winding_names)
        count = len(self.names)
        if len(resistance) != count or len(terminals) != count:
            raise ConfigError("every winding must define resistance and terminals")
        if any(not math.isfinite(value) or value < 0.0 for value in resistance):
            raise ConfigError("winding resistance must be finite and >= 0")
        self.resistance = np.asarray(resistance, dtype=float)

        flux_names = tuple(f"lambda_{name}" for name in self.names)
        initial = dict(getattr(law, "initial_state", {}))
        if any(not isinstance(name, str) or not name.isidentifier() for name in initial):
            raise ConfigError("magnetic internal state names must be identifiers")
        if set(flux_names) & set(initial):
            raise ConfigError("magnetic internal states must not shadow winding flux linkages")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in initial.values()):
            raise ConfigError("magnetic internal state values must be finite numbers")
        self.flux_names = flux_names
        self.internal_state_names = tuple(initial)
        if self.internal_state_names and not callable(getattr(law, "derivatives", None)):
            raise TypeError("a stateful magnetic law must define derivatives")
        self.state_names = (*flux_names, *self.internal_state_names)
        state_type = type("_MagPortState", (Bag,), {"__slots__": self.state_names})
        self.state = state_type(**{
            **dict.fromkeys(flux_names, 0.0),
            **{name: float(value) for name, value in initial.items()},
        })

        input_names = tuple(
            field
            for name in self.names
            for field in (f"u_positive_{name}", f"u_negative_{name}")
        )
        output_names = tuple(
            field
            for name in self.names
            for field in (f"i_positive_{name}", f"i_negative_{name}")
        )
        input_type = type("_MagPortInput", (Bag,), {"__slots__": input_names})
        output_type = type("_MagPortOutput", (Bag,), {"__slots__": output_names})
        self.input = input_type(**dict.fromkeys(input_names, 0.0))
        self.output = output_type(**dict.fromkeys(output_names, 0.0))

        ports = []
        for name, (positive_ground, negative_ground) in zip(self.names, terminals):
            ports.append((
                Port(
                    self,
                    f"{name}.positive",
                    grounded=bool(positive_ground),
                    effort=f"input.u_positive_{name}",
                    flow=f"output.i_positive_{name}",
                ),
                Port(
                    self,
                    f"{name}.negative",
                    grounded=bool(negative_ground),
                    effort=f"input.u_negative_{name}",
                    flow=f"output.i_negative_{name}",
                ),
            ))
        self.winding_ports = tuple(ports)
        self.ports = tuple(port for pair in ports for port in pair)
        energy = callable(getattr(law, "stored_energy", None))
        energy_rate = callable(getattr(law, "stored_energy_rate", None))
        if energy != energy_rate:
            raise TypeError(
                "a magnetic energy declaration needs stored_energy and stored_energy_rate"
            )
        if energy:
            self.storage = ()
            self.energy_states = self.state_names

    def _flux_linkage(self) -> np.ndarray:
        return np.fromiter(
            (getattr(self.state, name) for name in self.flux_names),
            dtype=float, count=len(self.flux_names),
        )

    def _internal_state(self) -> dict[str, float]:
        return {
            name: float(getattr(self.state, name))
            for name in self.internal_state_names
        }

    def _currents(self) -> np.ndarray:
        return np.fromiter(
            (getattr(self.output, f"i_positive_{name}") for name in self.names),
            dtype=float, count=len(self.names),
        )

    def set_outputs(self, t: float) -> None:
        currents = _numeric_vector(
            self.law.currents(t, self._flux_linkage(), self._internal_state()),
            len(self.names),
            "magnetic law currents",
        )
        for name, current in zip(self.names, currents):
            setattr(self.output, f"i_positive_{name}", float(current))
            setattr(self.output, f"i_negative_{name}", -float(current))

    def rhs(self, t: float) -> tuple[float, ...]:
        currents = self._currents()
        voltage = np.fromiter(
            (
                getattr(self.input, f"u_positive_{name}")
                - getattr(self.input, f"u_negative_{name}")
                for name in self.names
            ),
            dtype=float,
            count=len(self.names),
        )
        flux_rate = voltage - self.resistance * currents
        if not self.internal_state_names:
            return tuple(float(value) for value in flux_rate)
        derivative = self.law.derivatives(
            t, self._flux_linkage(), currents, self._internal_state()
        )
        if not isinstance(derivative, Mapping) or set(derivative) != set(self.internal_state_names):
            raise ConfigError(
                "magnetic law derivatives must return every declared internal state exactly once"
            )
        internal = _numeric_vector(
            [derivative[name] for name in self.internal_state_names],
            len(self.internal_state_names),
            "magnetic law derivatives",
        )
        return (*map(float, flux_rate), *map(float, internal))

    def stored_energy(self, t: float) -> float:
        value = self.law.stored_energy(  # type: ignore[attr-defined]
            t, self._flux_linkage(), self._internal_state()
        )
        if not math.isfinite(float(value)):
            raise ConfigError("magnetic stored energy must be finite")
        return float(value)

    def stored_energy_rate(
        self, t: float, derivatives: Mapping[str, complex | float]
    ) -> float:
        flux_rate = np.fromiter(
            (float(derivatives[name]) for name in self.flux_names),
            dtype=float,
            count=len(self.flux_names),
        )
        state_rate = {
            name: float(derivatives[name]) for name in self.internal_state_names
        }
        value = self.law.stored_energy_rate(  # type: ignore[attr-defined]
            t,
            self._flux_linkage(),
            self._currents(),
            self._internal_state(),
            flux_rate,
            state_rate,
        )
        if not math.isfinite(float(value)):
            raise ConfigError("magnetic stored-energy rate must be finite")
        return float(value)

    def dissipated_power(self) -> float:
        currents = self._currents()
        loss = float(np.dot(self.resistance, currents * currents))
        material = getattr(self.law, "dissipated_power", None)
        if material is not None:
            loss += float(material(
                self._flux_linkage(), currents, self._internal_state()
            ))
        if not math.isfinite(loss):
            raise ConfigError("magnetic dissipated power must be finite")
        return loss

    @staticmethod
    def supplied_power() -> float:
        return 0.0

    def magnetic_signals(self) -> dict[str, float]:
        signals = {}
        for name in self.names:
            signals[f"i_{name}"] = getattr(self.output, f"i_positive_{name}")
            signals[f"u_{name}"] = (
                getattr(self.input, f"u_positive_{name}")
                - getattr(self.input, f"u_negative_{name}")
            )
        return signals


class MagPortElement(_MagneticElement):
    """Assembly adapter from any magnetic law to standard PESLite winding ports."""

    def __init__(self, name: str, cfg: Any, buses: dict, law: MagneticLaw,
                 magnetic: MagneticDescription | None = None) -> None:
        _names, resistance, terminals = self._winding_data(
            name, cfg.windings, law.winding_names, buses
        )
        description = cfg.magnetic if magnetic is None else magnetic
        self._attach(
            name, cfg,
            MagPortModel(law, resistance, terminals, description),
        )


@register_component_type
class MagPort(MagPortElement):
    """Connect a file-backed linear or nonlinear magnetic law to electrical windings."""

    @dataclass(frozen=True, kw_only=True)
    class Params(MagneticParams):
        file: str = field(metadata={"asset": True, "runtime": False})
        type: str = "magport"

        def __post_init__(self) -> None:
            super().__post_init__()
            if not isinstance(self.file, str) or not self.file:
                raise ConfigError("file must be a nonempty path")

    type = "magport"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        data = _load(cfg.file)
        law = (
            LinearMagneticLaw(data.winding_names, data.inductance)
            if data.inductance is not None else
            TabulatedMagneticLaw(data.winding_names, data.flux_linkage, data.current)
        )
        if cfg.magnetic is not None and data.magnetic is not None:
            raise ConfigError("magnetic may be provided by the configuration or file, not both")
        super().__init__(name, cfg, buses, law, cfg.magnetic or data.magnetic)
