"""Nonlinear magnetic constitutive models connected through ordinary electrical ports."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .magcomponent import (
    MagComponent,
    MagneticDescription,
    MagneticParams,
    _MagneticElement,
    _magnetic_description,
    _validate_winding_names,
)
from .magcomponent import inductance_modes
from ...components.network import register_component_type
from ...solver import Bag, ConfigError, OutputMethod, Port, directional_derivative
from ...solver.model import _record_type

__all__ = [
    "MAGPORT_FORMAT",
    "MAGPORT_VERSION",
    "MagneticLaw",
    "MagPort",
    "MagPortData",
    "MagPortModel",
    "TabulatedEnergyLaw",
    "TabulatedMagneticLaw",
    "load_magport_data",
    "magport_data",
]


MAGPORT_FORMAT = "peslite-magport"
MAGPORT_VERSION = 2
_REQUIRED = frozenset({"format", "version", "winding_names"})
_LINEAR = frozenset({"inductance"})
_CURRENT_MAP = frozenset({"flux_linkage", "current"})
_ENERGY_MAP = frozenset({"flux_linkage", "stored_energy"})
_OPTIONAL = frozenset({
    "physical", "magnetic", "field_points", "field_basis"
}) | _LINEAR | _CURRENT_MAP | _ENERGY_MAP


@dataclass(frozen=True)
class MagPortData:
    winding_names: tuple[str, ...]
    inductance: np.ndarray | None
    flux_linkage: np.ndarray | None
    current: np.ndarray | None
    stored_energy: np.ndarray | None
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


def _energy_map(
    winding_names: Sequence[str], flux_linkage: Any, stored_energy: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Validate samples of one scalar magnetic-energy potential on a Cartesian grid."""
    count = len(tuple(winding_names))
    try:
        flux = np.asarray(flux_linkage, dtype=float)
        energy = np.asarray(stored_energy, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError("flux_linkage and stored_energy samples must be numeric") from None
    if count == 1 and flux.ndim == 1:
        flux = flux[:, None]
    if flux.ndim != 2 or flux.shape[1] != count:
        raise ConfigError(f"flux_linkage samples must have shape (samples, {count})")
    if energy.shape != (len(flux),):
        raise ConfigError(f"stored_energy samples must have shape ({len(flux)},)")
    if not np.isfinite(flux).all() or not np.isfinite(energy).all():
        raise ConfigError("flux_linkage and stored_energy samples must be finite")
    axes = tuple(np.unique(flux[:, index]) for index in range(count))
    if any(len(axis) < 2 for axis in axes) or math.prod(map(len, axes)) != len(flux):
        raise ConfigError("stored_energy samples must cover a complete Cartesian flux grid")
    positions = tuple({float(value): index for index, value in enumerate(axis)} for axis in axes)
    grid = np.full(tuple(map(len, axes)), np.nan)
    for point, value in zip(flux, energy):
        index = tuple(positions[axis][float(coordinate)]
                      for axis, coordinate in enumerate(point))
        if math.isfinite(grid[index]):
            raise ConfigError("stored_energy flux grid contains a duplicate point")
        grid[index] = value
    if not np.isfinite(grid).all():
        raise ConfigError("stored_energy samples must cover a complete Cartesian flux grid")
    ordered_flux = np.asarray(tuple(np.ndindex(grid.shape)), dtype=int)
    ordered_flux = np.column_stack([
        axes[axis][ordered_flux[:, axis]] for axis in range(count)
    ])
    ordered_energy = grid.ravel().copy()
    ordered_flux.setflags(write=False)
    ordered_energy.setflags(write=False)
    return ordered_flux, ordered_energy


def magport_data(
    winding_names: Sequence[str],
    inductance: Any | None = None,
    flux_linkage: Any | None = None,
    current: Any | None = None,
    stored_energy: Any | None = None,
    physical: Mapping[str, Any] | None = None,
    magnetic: MagneticDescription | Mapping[str, Any] | None = None,
    field_points: Any | None = None,
    field_basis: Any | None = None,
) -> MagPortData:
    names = _validate_winding_names(winding_names)
    linear = inductance is not None
    current_map = current is not None
    energy_map = stored_energy is not None
    if sum((linear, current_map, energy_map)) != 1:
        raise ConfigError(
            "provide exactly one of inductance, current samples or stored_energy samples"
        )
    if linear:
        if flux_linkage is not None:
            raise ConfigError("flux_linkage belongs to a sampled constitutive law")
        matrix, _modes, _shapes = inductance_modes(inductance, len(names))
        flux_samples = current_samples = energy_samples = None
    elif current_map:
        if flux_linkage is None:
            raise ConfigError("flux_linkage and current samples must be provided together")
        flux_samples, current_samples = _current_map(names, flux_linkage, current)
        energy_samples = None
        matrix = None
    else:
        if flux_linkage is None:
            raise ConfigError("flux_linkage and stored_energy samples must be provided together")
        flux_samples, energy_samples = _energy_map(names, flux_linkage, stored_energy)
        current_samples = None
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
    return MagPortData(
        names, matrix, flux_samples, current_samples, energy_samples,
        source, description, points, basis
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


def load_magport_data(file: str | Path) -> MagPortData:
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
            if format_name != MAGPORT_FORMAT:
                raise ConfigError(f"unknown magport format {format_name!r}")
            if (type(version) not in (int, np.int32, np.int64)
                    or int(version) != MAGPORT_VERSION):
                raise ConfigError(f"unsupported magport version {version!r}")
            constitutive = keys & (_LINEAR | _CURRENT_MAP | _ENERGY_MAP)
            if constitutive not in (_LINEAR, _CURRENT_MAP, _ENERGY_MAP):
                raise ConfigError(
                    "magport file must contain inductance, a flux_linkage/current map, "
                    "or a flux_linkage/stored_energy map"
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
            return magport_data(
                names,
                inductance=(archive["inductance"].copy()
                            if "inductance" in keys else None),
                flux_linkage=(archive["flux_linkage"].copy()
                              if "flux_linkage" in keys else None),
                current=(archive["current"].copy() if "current" in keys else None),
                stored_energy=(archive["stored_energy"].copy()
                               if "stored_energy" in keys else None),
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
    A dissipative law may provide ``dissipated_power`` using the same state and rate arguments.
    """

    winding_names: tuple[str, ...]
    initial_state: Mapping[str, float]

    def currents(
        self, t: float, flux_linkage: np.ndarray, state: Mapping[str, float]
    ) -> Sequence[float]: ...


class TabulatedEnergyLaw:
    """Conservative current map derived from one sampled scalar magnetic-energy potential."""

    initial_state: Mapping[str, float] = {}

    def __init__(
        self, winding_names: Sequence[str], flux_linkage: Any, stored_energy: Any
    ) -> None:
        self.winding_names = _validate_winding_names(winding_names)
        self.flux_linkage, self.energy = _energy_map(
            self.winding_names, flux_linkage, stored_energy
        )
        count = len(self.winding_names)
        self._axes = tuple(np.unique(self.flux_linkage[:, index]) for index in range(count))
        self._grid = self.energy.reshape(tuple(map(len, self._axes)))
        if count == 1:
            from scipy.interpolate import PchipInterpolator

            self._interpolator = PchipInterpolator(
                self._axes[0], self.energy, extrapolate=False
            )
            self._gradient = self._interpolator.derivative()
        else:
            self._interpolator = self._gradient = None

    def _evaluate(self, flux_linkage: np.ndarray) -> tuple[float, np.ndarray]:
        point = np.asarray(flux_linkage, dtype=float)
        count = len(self.winding_names)
        if point.shape != (count,) or not np.isfinite(point).all():
            raise ConfigError(f"flux linkage must contain {count} finite values")
        if count == 1:
            energy = float(self._interpolator(point[0]))
            current = float(self._gradient(point[0]))
            if not math.isfinite(energy) or not math.isfinite(current):
                raise ConfigError(
                    f"flux linkage {point.tolist()} lies outside the sampled magnetic energy map"
                )
            return energy, np.asarray((current,))

        lower = []
        fraction = []
        width = []
        for value, axis in zip(point, self._axes):
            if value < axis[0] or value > axis[-1]:
                raise ConfigError(
                    f"flux linkage {point.tolist()} lies outside the sampled magnetic energy map"
                )
            index = min(max(int(np.searchsorted(axis, value, side="right")) - 1, 0),
                        len(axis) - 2)
            step = float(axis[index + 1] - axis[index])
            lower.append(index)
            width.append(step)
            fraction.append(float((value - axis[index]) / step))

        energy = 0.0
        current = np.zeros(count)
        for corner in itertools.product((0, 1), repeat=count):
            index = tuple(lower[axis] + corner[axis] for axis in range(count))
            value = float(self._grid[index])
            factors = tuple(
                fraction[axis] if corner[axis] else 1.0 - fraction[axis]
                for axis in range(count)
            )
            energy += value * math.prod(factors)
            for axis in range(count):
                derivative = (1.0 if corner[axis] else -1.0) / width[axis]
                current[axis] += value * derivative * math.prod(
                    factors[other] for other in range(count) if other != axis
                )
        return energy, current

    def currents(
        self, _t: float, flux_linkage: np.ndarray, _state: Mapping[str, float]
    ) -> np.ndarray:
        return self._evaluate(flux_linkage)[1]

    def stored_energy(
        self, _t: float, flux_linkage: np.ndarray, _state: Mapping[str, float]
    ) -> float:
        return self._evaluate(flux_linkage)[0]

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
            self.stored_energy = self._stored_energy
            self.stored_energy_rate = self._stored_energy_rate
        else:
            try:
                from scipy.interpolate import LinearNDInterpolator
                from scipy.spatial import QhullError

                self._interpolator = LinearNDInterpolator(
                    self.flux_linkage, self.current, fill_value=np.nan
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
        if value.shape != (len(self.winding_names),) or not np.isfinite(value).all():
            raise ConfigError(
                f"flux linkage {flux_linkage.tolist()} lies outside the sampled magnetic current map"
            )
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
        temperature_coefficient_per_K: float = 0.0,
        temperature_reference_K: float = 298.15,
    ) -> None:
        super().__init__(
            magnetic,
            tuple(resistance),
            temperature_coefficient_per_K,
            temperature_reference_K,
        )
        if not callable(getattr(law, "currents", None)):
            raise TypeError("a magnetic law must define currents(t, flux_linkage, state)")
        self.law = law
        self.names = _validate_winding_names(law.winding_names)
        count = len(self.names)
        if len(resistance) != count or len(terminals) != count:
            raise ConfigError("every winding must define resistance and terminals")
        if any(not math.isfinite(value) or value < 0.0 for value in resistance):
            raise ConfigError("winding resistance must be finite and >= 0")
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
        state_type = _record_type("_MagPortState", self.state_names)
        self.state = state_type(**{
            **dict.fromkeys(flux_names, 0.0),
            **{name: float(value) for name, value in initial.items()},
        })

        input_names = (
            *tuple(
                field
                for name in self.names
                for field in (f"u_positive_{name}", f"u_negative_{name}")
            ),
            "temperature_K",
        )
        current_outputs = tuple(
            field
            for name in self.names
            for field in (f"i_positive_{name}", f"i_negative_{name}")
        )
        current_rate_outputs = tuple(
            field
            for name in self.names
            for field in (f"_di_positive_{name}", f"_di_negative_{name}")
        )
        voltage_inputs = tuple(
            field
            for name in self.names
            for field in (f"u_positive_{name}", f"u_negative_{name}")
        )
        rate_inputs = (*voltage_inputs, *(("temperature_K",)
                                         if self.temperature_coefficient_per_K != 0.0 else ()))
        loss_outputs = ("copper_loss_W", "core_loss_W", "loss_W")
        loss_inputs = ((*voltage_inputs, "temperature_K")
                       if callable(getattr(law, "dissipated_power", None)) else
                       (("temperature_K",)
                        if self.temperature_coefficient_per_K != 0.0 else ()))
        output_names = (*current_outputs, *current_rate_outputs, *loss_outputs)
        input_type = _record_type("_MagPortInput", input_names)
        output_type = _record_type("_MagPortOutput", output_names)
        self.input = input_type(
            **{**dict.fromkeys(input_names, 0.0),
               "temperature_K": self.temperature_reference_K}
        )
        self.output = output_type(**dict.fromkeys(output_names, 0.0))
        temperature_stage = (() if self.temperature_coefficient_per_K == 0.0 else (
            OutputMethod("prepare_temperature", ("temperature_K",), ()),
        ))
        self.output_stages = (
            *temperature_stage,
            OutputMethod("set_currents", (), current_outputs),
            OutputMethod(
                "set_current_rates",
                rate_inputs,
                current_rate_outputs,
            ),
            OutputMethod(
                "set_loss_outputs",
                loss_inputs,
                loss_outputs,
                rhs=False,
            ),
        )
        self._output_dependencies = {
            "set_current_rates": ((self, "set_currents"),),
            "set_loss_outputs": ((self, "set_currents"),),
        }

        ports = []
        for name, (positive_ground, negative_ground) in zip(self.names, terminals):
            ports.append((
                Port(
                    self,
                    f"{name}.positive",
                    grounded=bool(positive_ground),
                    effort=f"input.u_positive_{name}",
                    flow=f"output.i_positive_{name}",
                    flow_rate=f"output._di_positive_{name}",
                ),
                Port(
                    self,
                    f"{name}.negative",
                    grounded=bool(negative_ground),
                    effort=f"input.u_negative_{name}",
                    flow=f"output.i_negative_{name}",
                    flow_rate=f"output._di_negative_{name}",
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

    def prepare_temperature(self, _t: float) -> None:
        pass

    def set_currents(self, t: float) -> None:
        currents = _numeric_vector(
            self.law.currents(t, self._flux_linkage(), self._internal_state()),
            len(self.names),
            "magnetic law currents",
        )
        for name, current in zip(self.names, currents):
            setattr(self.output, f"i_positive_{name}", float(current))
            setattr(self.output, f"i_negative_{name}", -float(current))

    def set_outputs(self, t: float) -> None:
        self.set_currents(t)
        self.set_current_rates(t)
        self.set_loss_outputs(t)

    def _state_derivatives(self, t: float) -> np.ndarray:
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
        flux_rate = voltage - self._effective_resistance() * currents
        if not self.internal_state_names:
            return flux_rate
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
        return np.concatenate((flux_rate, internal))

    def set_current_rates(self, t: float) -> None:
        state_rate = self._state_derivatives(t)
        flux = self._flux_linkage()
        internal = np.fromiter(
            (getattr(self.state, name) for name in self.internal_state_names),
            dtype=float,
            count=len(self.internal_state_names),
        )
        point = np.concatenate(([t], flux, internal))
        direction = np.concatenate(([1.0], state_rate))
        count = len(self.names)

        def evaluate(candidate: np.ndarray) -> np.ndarray:
            values = self.law.currents(
                float(candidate[0]),
                candidate[1:1 + count],
                dict(zip(self.internal_state_names, candidate[1 + count:])),
            )
            return _numeric_vector(values, count, "magnetic law currents")

        rates = directional_derivative(evaluate, point, direction)
        if rates.shape != (count,) or not np.isfinite(rates).all():
            raise ConfigError("magnetic law current rate must contain finite numeric values")
        for name, rate in zip(self.names, rates):
            setattr(self.output, f"_di_positive_{name}", float(rate))
            setattr(self.output, f"_di_negative_{name}", -float(rate))

    def rhs(self, t: float) -> tuple[float, ...]:
        return tuple(map(float, self._state_derivatives(t)))

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

    def _winding_currents(self) -> np.ndarray:
        return self._currents()

    def _material_loss(self, t: float) -> float:
        material = getattr(self.law, "dissipated_power", None)
        if material is None:
            return 0.0
        derivatives = self._state_derivatives(t)
        count = len(self.names)
        return material(
            t,
            self._flux_linkage(),
            self._currents(),
            self._internal_state(),
            derivatives[:count],
            dict(zip(self.internal_state_names, derivatives[count:])),
            float(self.input.temperature_K),
        )

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
        signals.update(
            copper_loss_W=self.output.copper_loss_W,
            core_loss_W=self.output.core_loss_W,
            loss_W=self.output.loss_W,
        )
        return signals


def _attach_magnetic_law(
    element: _MagneticElement,
    name: str,
    cfg: Any,
    buses: dict,
    law: MagneticLaw,
    magnetic: MagneticDescription | None = None,
) -> None:
    """Attach one magnetic law through the common winding and signal contracts."""
    _names, resistance, endpoints = element._winding_data(
        name, cfg.windings, law.winding_names, buses
    )
    description = cfg.magnetic if magnetic is None else magnetic
    element._attach(
        name,
        cfg,
        MagPortModel(
            law,
            resistance,
            tuple(tuple(endpoint.grounded for endpoint in pair) for pair in endpoints),
            description,
            cfg.temperature_coefficient_per_K,
            cfg.temperature_reference_K,
        ),
        endpoints,
    )


@register_component_type
class MagPort(_MagneticElement):
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

    def __init__(
        self,
        name: str,
        cfg: Any,
        buses: dict,
        _params: Any,
        *,
        law: MagneticLaw | None = None,
        magnetic: MagneticDescription | None = None,
    ) -> None:
        if law is not None:
            _attach_magnetic_law(self, name, cfg, buses, law, magnetic)
            return
        data = load_magport_data(cfg.file)
        if cfg.magnetic is not None and data.magnetic is not None:
            raise ConfigError("magnetic may be provided by the configuration or file, not both")
        magnetic = cfg.magnetic or data.magnetic
        if data.inductance is not None:
            self._assemble(name, cfg, data.winding_names, data.inductance, buses, magnetic)
            return
        law = (TabulatedEnergyLaw(data.winding_names, data.flux_linkage, data.stored_energy)
               if data.stored_energy is not None else
               TabulatedMagneticLaw(data.winding_names, data.flux_linkage, data.current))
        _attach_magnetic_law(self, name, cfg, buses, law, magnetic)
