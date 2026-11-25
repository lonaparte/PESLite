"""MagComponent contract and its coupled-coil implementation."""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar, Mapping, Optional, Sequence

import numpy as np

from ...assembly.params import _parameterized, register_unit_type, unit_type
from ...assembly.system import _UnitAssembly, _assemble_unit, _part
from ...components.network import GROUND, Element, NetworkComponent, resolve_endpoints
from ...components.registry import register_component_type
from ...components.pwm import PhaseClock
from ...solver import Bag, ConfigError, Empty, OutputMethod, Port, StoragePort
from ...solver import record_type
from ...solver.model import LocatedZero, _real, commutation_tolerance

__all__ = [
    "MagComponent",
    "MagneticBranchDescription",
    "MagneticDescription",
    "MagneticParams",
    "WindingParams",
    "CoupledCoils",
    "CoupledCoilsModel",
    "CompensatedLink",
    "SwitchedCompensatedLink",
    "SwitchControlledCapacitor",
    "make_wpt_link",
    "SCCParams",
    "SSLinkParams",
    "LCCLinkParams",
    "LCCSLinkParams",
    "MultiSSLinkParams",
]

SIDES = ("primary", "secondary")


def two_winding_modes(primary: Any, secondary: Any, mutual: Any):
    """Diagonalize two windings while retaining symbolic scalar arithmetic."""
    nominal = tuple(getattr(value, "_peslite_numeric_value", value)
                    for value in (primary, secondary, mutual))
    if all(isinstance(value, (int, float, np.number)) for value in nominal):
        values = tuple(float(value) for value in nominal)
        if (not all(map(np.isfinite, values)) or values[0] <= 0.0 or values[1] <= 0.0
                or values[0] * values[1] <= values[2] ** 2):
            raise ConfigError("inductance must be positive definite")
    mean = 0.5 * (primary + secondary)
    half_difference = 0.5 * (primary - secondary)
    radius = np.sqrt(half_difference ** 2 + mutual ** 2)
    distinct = radius > 0.0
    denominator = np.where(distinct, 2.0 * radius, 1.0)
    cosine = np.sqrt(np.maximum(0.0, (radius + half_difference) / denominator))
    sine = np.copysign(
        np.sqrt(np.maximum(0.0, (radius - half_difference) / denominator)), mutual,
    )
    cosine, sine = np.where(distinct, cosine, 1.0), np.where(distinct, sine, 0.0)
    modes = (_real(mean - radius), _real(mean + radius))
    shapes = ((_real(-sine), _real(cosine)),
              (_real(cosine), _real(sine)))
    return modes, shapes


def inductance_modes(inductance: Any, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and diagonally decompose one numeric coupled-inductance matrix."""
    try:
        matrix = np.asarray(inductance, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError("inductance must be a numeric square matrix") from None
    if matrix.shape != (count, count) or not np.isfinite(matrix).all():
        raise ConfigError(f"inductance must be a finite {count} x {count} matrix")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0):
        raise ConfigError("inductance must be symmetric")
    modes, shapes = np.linalg.eigh(matrix)
    if modes.min() <= 0.0:
        raise ConfigError("inductance must be positive definite")
    matrix.setflags(write=False)
    return matrix, modes, shapes


class LinearEnergyLaw:
    """Analytic quadratic energy law for a linear coupled-inductance matrix."""

    initial_state: Mapping[str, float] = {}

    def __init__(self, winding_names: Sequence[str], inductance: Any) -> None:
        self.winding_names = tuple(winding_names)
        if len(self.winding_names) == 2:
            try:
                primary, mutual = inductance[0]
                other_mutual, secondary = inductance[1]
            except (TypeError, ValueError, IndexError):
                raise ConfigError("inductance must be a 2 x 2 matrix") from None
            values = (primary, mutual, other_mutual, secondary)
            modes, shapes = two_winding_modes(primary, secondary, mutual)
            if (all(isinstance(value, (int, float, np.number)) for value in values)
                    and not any(hasattr(value, "_peslite_numeric_value") for value in values)):
                self.inductance = inductance_modes(inductance, 2)[0]
                self.modes = np.asarray(modes, dtype=float)
                self.shapes = np.asarray(shapes, dtype=float)
            else:
                self.inductance = inductance
                self.modes, self.shapes = modes, shapes
        else:
            self.inductance, self.modes, self.shapes = inductance_modes(
                inductance, len(self.winding_names)
            )

    def currents(
        self, _t: float, flux_linkage: np.ndarray, _state: Mapping[str, float]
    ) -> np.ndarray:
        flux = np.asarray(flux_linkage, dtype=float)
        if flux.shape != (len(self.winding_names),) or not np.isfinite(flux).all():
            raise ConfigError(
                f"flux linkage must contain {len(self.winding_names)} finite values"
            )
        return self.shapes @ ((self.shapes.T @ flux) / self.modes)

    def stored_energy(
        self, t: float, flux_linkage: np.ndarray, state: Mapping[str, float]
    ) -> float:
        return float(0.5 * np.asarray(flux_linkage, dtype=float)
                     @ self.currents(t, flux_linkage, state))

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

@dataclass(frozen=True, kw_only=True)
class WindingParams:
    """The electrical terminals and copper resistance of one named winding."""

    positive: str
    negative: str
    r: float = 0.0

    def __post_init__(self) -> None:
        if not self.positive or not self.negative or self.positive == self.negative:
            raise ConfigError("positive and negative must be distinct, nonempty terminals")
        if not math.isfinite(self.r) or self.r < 0.0:
            raise ConfigError(f"r must be finite and >= 0, got {self.r}")


def _validate_winding_names(winding_names: Any) -> tuple[str, ...]:
    names = tuple(winding_names)
    if not names:
        raise ConfigError("winding_names must contain at least one name")
    if any(not isinstance(name, str) or not name.isidentifier() for name in names):
        raise ConfigError("winding_names must contain identifiers")
    if len(set(names)) != len(names):
        raise ConfigError("winding_names must contain unique names")
    return names


def _validate_scoped_names(values: Any, path: str) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(
        not isinstance(name, str) or not name or "." in name for name in names
    ):
        raise ConfigError(f"{path} names must be nonempty strings without dots")
    if len(set(names)) != len(names):
        raise ConfigError(f"{path} names must be unique")
    return names


def _winding_buses(windings: dict[str, WindingParams]) -> dict[str, str]:
    return {
        f"windings.{name}.{side}": getattr(winding, side)
        for name, winding in windings.items()
        for side in ("positive", "negative")
    }


def _positive(value: float | None, name: str) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        raise ConfigError(f"{name} must be finite and > 0")


def _model_mapping(value: Mapping[str, Any] | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must contain finite JSON-compatible data") from None


@dataclass(frozen=True, kw_only=True)
class MagneticBranchDescription:
    """Physical description of one named magnetic path.

    A branch has one constitutive description: direct reluctance, geometry with a linear
    permeability or B-H curve, or explicit flux/MMF samples. Loss and history mappings are
    provider data; the electrical solver uses them only when its selected magnetic law declares
    the corresponding dynamics or dissipation.
    """

    reluctance: float | None = None
    area: float | None = None
    path_length: float | None = None
    mu_r: float | None = None
    bh_curve: list[list[float]] | None = None
    flux: list[float] | None = None
    mmf: list[float] | None = None
    hysteresis: Mapping[str, Any] | None = None
    core_loss: Mapping[str, Any] | None = None
    eddy_current: Mapping[str, Any] | None = None

    _provider_fields: ClassVar[tuple[str, ...]] = (
        "hysteresis", "core_loss", "eddy_current",
    )

    def __post_init__(self) -> None:
        _positive(self.reluctance, "reluctance")
        _positive(self.area, "area")
        _positive(self.path_length, "path_length")
        _positive(self.mu_r, "mu_r")
        geometry = self.area is not None or self.path_length is not None or self.mu_r is not None
        curve = self.bh_curve is not None
        samples = self.flux is not None or self.mmf is not None
        modes = int(self.reluctance is not None) + int(geometry or curve) + int(samples)
        if modes != 1:
            raise ConfigError(
                "a magnetic branch needs exactly one of reluctance, material geometry, "
                "or flux/MMF samples"
            )
        if geometry or curve:
            if self.area is None or self.path_length is None:
                raise ConfigError("material geometry needs area and path_length")
            if (self.mu_r is None) == (self.bh_curve is None):
                raise ConfigError("material geometry needs exactly one of mu_r or bh_curve")
        if self.bh_curve is not None:
            values = self._curve(self.bh_curve, "bh_curve")
            if np.any(np.diff(values[:, 0]) <= 0.0) or np.any(np.diff(values[:, 1]) <= 0.0):
                raise ConfigError("bh_curve B and H values must be strictly increasing")
            object.__setattr__(self, "bh_curve", values.tolist())
        if samples:
            if self.flux is None or self.mmf is None:
                raise ConfigError("flux and mmf samples must be provided together")
            try:
                pairs = np.column_stack((self.flux, self.mmf))
            except (TypeError, ValueError):
                raise ConfigError("flux and mmf samples must have equal lengths") from None
            values = self._curve(pairs, "flux/MMF samples")
            if np.any(np.diff(values[:, 0]) <= 0.0) or np.any(np.diff(values[:, 1]) <= 0.0):
                raise ConfigError("flux and mmf samples must be strictly increasing")
            object.__setattr__(self, "flux", values[:, 0].tolist())
            object.__setattr__(self, "mmf", values[:, 1].tolist())
        for name, value in self.provider_data().items():
            _model_mapping(value, name)

    def provider_data(self) -> dict[str, Mapping[str, Any]]:
        """Return configured loss and history provider data by field name."""
        return {
            name: value
            for name in self._provider_fields
            if (value := getattr(self, name)) is not None
        }

    @staticmethod
    def _curve(value: Any, name: str) -> np.ndarray:
        try:
            curve = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ConfigError(f"{name} must contain numeric pairs") from None
        if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
            raise ConfigError(f"{name} must contain at least two [x, y] pairs")
        if not np.isfinite(curve).all():
            raise ConfigError(f"{name} must contain finite values")
        return curve

    def constitutive_data(self) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
        """Return direct reluctance or monotone flux/MMF samples in SI units."""
        if self.reluctance is not None:
            return self.reluctance, None, None
        if self.mu_r is not None:
            mu0 = 4.0e-7 * math.pi
            return self.path_length / (mu0 * self.mu_r * self.area), None, None
        if self.bh_curve is not None:
            curve = np.asarray(self.bh_curve, dtype=float)
            return None, curve[:, 0] * self.area, curve[:, 1] * self.path_length
        return None, np.asarray(self.flux, dtype=float), np.asarray(self.mmf, dtype=float)


@dataclass(frozen=True, kw_only=True)
class MagneticDescription:
    """Optional physical magnetic domain accompanying an electrical component model."""

    branches: dict[str, MagneticBranchDescription] = field(
        metadata={"entries": MagneticBranchDescription, "structural": True}
    )
    nodes: dict[str, dict[str, float]] = field(
        default_factory=dict, metadata={"structural": True}
    )
    coupling: dict[str, dict[str, float]] = field(
        default_factory=dict, metadata={"structural": True}
    )

    def __post_init__(self) -> None:
        branch_names = _validate_scoped_names(self.branches, "magnetic.branches")
        known = set(branch_names)
        normalized = {}
        for section, values in (("nodes", self.nodes), ("coupling", self.coupling)):
            if not isinstance(values, dict):
                raise ConfigError(f"magnetic.{section} must be a mapping")
            normalized[section] = {}
            for name, coefficients in values.items():
                _validate_scoped_names((name,), f"magnetic.{section}")
                if not isinstance(coefficients, dict):
                    raise ConfigError(f"magnetic.{section}.{name} must be a mapping")
                if section == "coupling" and name not in known:
                    raise ConfigError(f"magnetic.coupling: unknown branch {name!r}")
                if section == "nodes":
                    unknown = set(coefficients) - known
                    if unknown:
                        raise ConfigError(
                            f"magnetic.nodes.{name}: unknown branches {sorted(unknown)}"
                        )
                converted = {}
                for target, coefficient in coefficients.items():
                    if not isinstance(target, str) or not target or "." in target:
                        raise ConfigError(
                            f"magnetic.{section}.{name} targets must be nonempty strings "
                            "without dots"
                        )
                    try:
                        value = float(coefficient)
                    except (TypeError, ValueError):
                        raise ConfigError(
                            f"magnetic.{section}.{name}.{target} must be finite"
                        ) from None
                    if isinstance(coefficient, bool) or not math.isfinite(value):
                        raise ConfigError(
                            f"magnetic.{section}.{name}.{target} must be finite"
                        )
                    converted[target] = value
                normalized[section][name] = converted
        object.__setattr__(self, "nodes", normalized["nodes"])
        object.__setattr__(self, "coupling", normalized["coupling"])

    def matrices(
        self, winding_names: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return branch-to-winding coupling and optional node-incidence matrices."""
        windings = _validate_winding_names(winding_names)
        unknown = {
            winding
            for coefficients in self.coupling.values()
            for winding in coefficients
            if winding not in windings
        }
        if unknown:
            raise ConfigError(f"magnetic.coupling: unknown windings {sorted(unknown)}")
        branch_names = tuple(self.branches)
        coupling = np.asarray([
            [self.coupling.get(branch, {}).get(winding, 0.0) for winding in windings]
            for branch in branch_names
        ], dtype=float)
        incidence = None if not self.nodes else np.asarray([
            [coefficients.get(branch, 0.0) for branch in branch_names]
            for coefficients in self.nodes.values()
        ], dtype=float)
        return coupling, incidence


def _magnetic_description(value: Any) -> MagneticDescription | None:
    """Build the physical magnetic domain from its portable mapping representation."""
    if value is None or isinstance(value, MagneticDescription):
        return value
    if not isinstance(value, Mapping):
        raise ConfigError("magnetic must be a mapping")
    unknown = set(value) - {"branches", "nodes", "coupling"}
    if unknown:
        raise ConfigError(f"magnetic: unknown keys {sorted(unknown)}")
    branches = value.get("branches")
    if not isinstance(branches, Mapping):
        raise ConfigError("magnetic.branches must be a mapping")
    built = {}
    for name, branch in branches.items():
        if not isinstance(branch, Mapping):
            raise ConfigError(f"magnetic.branches.{name} must be a mapping")
        try:
            built[str(name)] = MagneticBranchDescription(**branch)
        except TypeError as exc:
            raise ConfigError(f"magnetic.branches.{name}: {exc}") from None
    return MagneticDescription(
        branches=built,
        nodes=dict(value.get("nodes", {})),
        coupling=dict(value.get("coupling", {})),
    )


@dataclass(frozen=True, kw_only=True)
class MagneticParams:
    """Common winding configuration for magnetic element adapters."""

    windings: dict[str, WindingParams] = field(metadata={"structural": True})
    magnetic: MagneticDescription | None = field(
        default=None, metadata={"structural": True}
    )
    temperature_source: str | None = field(
        default=None, metadata={"runtime": False}
    )
    temperature_coefficient_per_K: float = 0.0
    temperature_reference_K: float = 298.15

    def __post_init__(self) -> None:
        names = _validate_winding_names(self.windings)
        if (not math.isfinite(self.temperature_coefficient_per_K)
                or not math.isfinite(self.temperature_reference_K)
                or self.temperature_reference_K <= 0.0):
            raise ConfigError("magnetic temperature parameters must be finite and physical")
        if (self.temperature_source is not None
                and (not isinstance(self.temperature_source, str)
                     or not self.temperature_source)):
            raise ConfigError("temperature_source must name a physical output")
        if self.magnetic is not None:
            self.magnetic.matrices(names)

    def connection_buses(self) -> dict[str, str]:
        return _winding_buses(self.windings)


class MagComponent(NetworkComponent):
    """Common magnetic component contract and cross-domain coupling signals."""

    def __init__(
        self,
        magnetic: MagneticDescription | None = None,
        resistance: tuple[float, ...] = (),
        temperature_coefficient_per_K: float = 0.0,
        temperature_reference_K: float = 298.15,
    ) -> None:
        self.magnetic = magnetic
        self.resistance = tuple(_real(value) for value in resistance)
        self.temperature_coefficient_per_K = _real(temperature_coefficient_per_K)
        self.temperature_reference_K = _real(temperature_reference_K)

    def _effective_resistance(self) -> np.ndarray:
        if self.temperature_coefficient_per_K == 0.0:
            return self.resistance
        temperature = float(self.input.temperature_K)
        factor = 1.0 + self.temperature_coefficient_per_K * (
            temperature - self.temperature_reference_K
        )
        if (not math.isfinite(temperature) or temperature <= 0.0
                or not math.isfinite(factor) or factor < 0.0):
            raise ConfigError("magnetic winding resistance became nonphysical")
        return tuple(value * factor for value in self.resistance)

    def _winding_currents(self) -> np.ndarray:
        raise NotImplementedError

    def _material_loss(self, _t: float) -> float:
        return 0.0

    def _loss_parts(self, t: float) -> tuple[Any, Any]:
        current = self._winding_currents()
        copper = sum(
            resistance * value * value
            for resistance, value in zip(self._effective_resistance(), current)
        )
        material = self._material_loss(t)
        if isinstance(copper, numbers.Number) and isinstance(material, numbers.Number):
            if (not math.isfinite(copper) or not math.isfinite(material)
                    or copper < 0.0 or material < 0.0):
                raise ConfigError("magnetic losses must be nonnegative finite W")
        return copper, material

    def set_loss_outputs(self, t: float) -> None:
        copper, material = self._loss_parts(t)
        self.output.copper_loss_W = copper
        self.output.core_loss_W = material
        self.output.loss_W = copper + material

    def dissipated_power(self) -> float:
        return self.output.loss_W


class _MagneticInput(Bag):
    __slots__ = ("temperature_K",)


class CoupledCoilsModel(MagComponent):
    """Constant winding-inductance matrix implementation of :class:`MagComponent`."""

    outputs_need_inputs: ClassVar[bool] = False

    def __init__(self, winding_names: tuple[str, ...], inductance: Any,
                 resistance: tuple[float, ...],
                 terminals: tuple[tuple[bool, bool], ...],
                 magnetic: MagneticDescription | None = None,
                 temperature_coefficient_per_K: float = 0.0,
                 temperature_reference_K: float = 298.15) -> None:
        super().__init__(
            magnetic,
            resistance,
            temperature_coefficient_per_K,
            temperature_reference_K,
        )
        names = _validate_winding_names(winding_names)
        if len(resistance) != len(names) or len(terminals) != len(names):
            raise ConfigError("every winding must define resistance and terminals")
        self.law = LinearEnergyLaw(names, inductance)

        self.names = names
        self.inductance = self.law.inductance
        self._shapes = self.law.shapes
        self.state_names = tuple(f"i_mode_{index + 1}" for index in range(len(names)))
        self.state = record_type("_MagneticState", self.state_names)(
            **dict.fromkeys(self.state_names, 0.0)
        )
        output_names = (
            *tuple(label for name in names for label in (f"i_{name}", f"u_{name}")),
            "copper_loss_W", "core_loss_W", "loss_W",
        )
        self.output = record_type("_MagneticOutput", output_names)(
            **dict.fromkeys(output_names, 0.0)
        )
        self.input = _MagneticInput(temperature_K=self.temperature_reference_K)
        self.output_stages = (
            OutputMethod(
                "set_loss_outputs",
                (("temperature_K",)
                 if self.temperature_coefficient_per_K != 0.0 else ()),
                ("copper_loss_W", "core_loss_W", "loss_W"),
                rhs=False,
            ),
        )
        self.storage = tuple(StoragePort(name, _real(mode), kind="inductor")
                             for name, mode in zip(self.state_names, self.law.modes))

        ground = Port(self, GROUND, grounded=True) if any(any(pair) for pair in terminals) else None
        ports = []
        for name, (positive_ground, negative_ground) in zip(names, terminals):
            ports.append((
                ground if positive_ground else Port(self, f"{name}.positive", standalone="voltage"),
                ground if negative_ground else Port(self, f"{name}.negative", standalone="voltage"),
            ))
        self.winding_ports = tuple(ports)
        self.ports = tuple(dict.fromkeys(port for pair in ports for port in pair))

    def network(self, nb: Any) -> None:
        from ...solver import netlist as nl

        temperature_dependent = self.temperature_coefficient_per_K != 0.0
        branches = []
        for index, ((positive, negative), resistance) in enumerate(
            zip(self.winding_ports, self.resistance)
        ):
            high, low = nb.node(positive), nb.node(negative)
            if resistance:
                inner = nb.internal()
                if temperature_dependent:
                    drop = nb.function(
                        partial(self._winding_drop, index),
                        ("temperature_K",),
                        states=self.state_names,
                    )
                    nb.add(nl.VoltageSource(high, inner, drop))
                else:
                    nb.add(nl.Resistor(high, inner, resistance))
                high = inner
            branches.append((high, low))
        coupled = nb.add(nl.Inductor(
            L=self.inductance,
            state=nb.state(self, self.state_names, transform=self._shapes),
            branches=branches,
        ))
        for index, (name, (positive, negative)) in enumerate(zip(self.names,
                                                                  self.winding_ports)):
            nb.output(self, f"i_{name}", ("branch_current", coupled, index))
            terminal_voltage = []
            if not positive.grounded:
                terminal_voltage.append((1.0, ("node_voltage", nb.node(positive))))
            if not negative.grounded:
                terminal_voltage.append((-1.0, ("node_voltage", nb.node(negative))))
            nb.linear_output(self, f"u_{name}", terminal_voltage)
        if temperature_dependent and any(self.resistance):
            nb.add(nl.Dissipation(nb.function(
                self._copper_loss,
                ("temperature_K",),
                states=self.state_names,
            )))
        nb.losses(self)

    def _winding_currents(self) -> np.ndarray:
        modal = np.asarray([getattr(self.state, name) for name in self.state_names])
        return self._shapes @ modal

    def _winding_drop(self, index: int, _t: float) -> float:
        return self._effective_resistance()[index] * self._winding_currents()[index]

    def _copper_loss(self, _t: float) -> float:
        current = self._winding_currents()
        return sum(
            resistance * value * value
            for resistance, value in zip(self._effective_resistance(), current)
        )

    def magnetic_signals(self) -> dict[str, float]:
        """Return the public winding currents and terminal voltages."""
        return {
            field: getattr(self.output, field)
            for field in type(self.output).__slots__
        }


class _MagneticElement(Element):
    """Shared winding validation and assembly for magnetic component implementations."""

    @staticmethod
    def _winding_data(
        name: str,
        windings: dict[str, WindingParams],
        winding_names: tuple[str, ...],
        buses: dict,
    ) -> tuple[tuple[str, ...], tuple[float, ...], tuple[tuple[Any, Any], ...]]:
        names = _validate_winding_names(windings)
        if names != tuple(winding_names):
            raise ConfigError(
                f"elements.{name}: winding order {names!r} does not match "
                f"model order {tuple(winding_names)!r}"
            )
        endpoints = resolve_endpoints(
            _winding_buses(windings), buses, f"elements.{name}", same_phases=True
        )
        if any(endpoint.phases != 1 for endpoint in endpoints.values()):
            raise ConfigError(f"elements.{name}: magnetic windings require scalar endpoints")
        pairs = tuple(
            (endpoints[f"windings.{label}.positive"],
             endpoints[f"windings.{label}.negative"])
            for label in names
        )
        return (
            names,
            tuple(winding.r for winding in windings.values()),
            pairs,
        )

    def _attach(self, name: str, cfg: Any, model: Any, endpoints: Any) -> None:
        self.name, self.cfg, self.model = name, cfg, model
        self._winding_endpoints = endpoints

    def _assemble(self, name: str, cfg: Any, winding_names: tuple[str, ...], inductance: Any,
                  buses: dict, magnetic: MagneticDescription | None) -> None:
        """Assemble a constant winding-inductance law through the shared netlist model."""
        names, resistance, endpoints = self._winding_data(
            name, cfg.windings, winding_names, buses
        )
        self._attach(
            name, cfg,
            CoupledCoilsModel(
                names,
                inductance,
                resistance,
                tuple(tuple(endpoint.grounded for endpoint in pair) for pair in endpoints),
                magnetic,
                cfg.temperature_coefficient_per_K,
                cfg.temperature_reference_K,
            ),
            endpoints,
        )

    def connections(self) -> dict:
        source = self.cfg.temperature_source
        return {(self.model, "temperature_K"): source} if source is not None else {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return tuple(
            (endpoint.name, port)
            for endpoints, ports in zip(self._winding_endpoints, self.model.winding_ports)
            for endpoint, port in zip(endpoints, ports)
            if not endpoint.grounded
        )

    def signals(self) -> dict[str, float]:
        return {
            f"{self.name}.{field}": value
            for field, value in self.model.magnetic_signals().items()
        }


@register_component_type
class CoupledCoils(_MagneticElement):
    """Configuration adapter for :class:`CoupledCoilsModel`."""

    @dataclass(frozen=True, kw_only=True)
    class Params(MagneticParams):
        inductance: list[list[float]] = field(metadata={"structural": True})
        type: str = "coupled_coils"

        def __post_init__(self) -> None:
            super().__post_init__()
            matrix, _modes, _shapes = inductance_modes(
                self.inductance, len(self.windings)
            )
            object.__setattr__(self, "inductance", matrix.tolist())

    type = "coupled_coils"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self._assemble(name, cfg, tuple(cfg.windings), cfg.inductance, buses, cfg.magnetic)

class _Side:
    """Static description of one side: its coils and optional LCC front end."""

    __slots__ = ("name", "coils", "lcc", "compensation_inductance", "compensation_resistance",
                 "parallel_capacitance", "parallel_resistance")

    def __init__(self, name: str, coils: tuple[int, ...], lcc: bool = False,
                 compensation_inductance: float = 0.0, compensation_resistance: float = 0.0,
                 parallel_capacitance: float = 0.0, parallel_resistance: float = 0.0) -> None:
        self.name, self.coils, self.lcc = name, coils, lcc
        self.compensation_inductance = compensation_inductance
        self.compensation_resistance = compensation_resistance
        self.parallel_capacitance = parallel_capacitance
        self.parallel_resistance = parallel_resistance


class CompensatedLink(MagComponent):
    """Coupled coils with per-coil series capacitors and per-side series or LCC compensation."""

    outputs_need_inputs: ClassVar[bool] = False
    def __init__(self, *, inductance, coil_sides: tuple[str, ...], capacitance, resistance,
                 sides: dict[str, _Side],
                 switched: tuple["SwitchControlledCapacitor", ...] = ()) -> None:
        super().__init__(resistance=tuple(resistance))
        n = len(coil_sides)
        self.switched = tuple(switched)
        self._scc_names = tuple(scc.state_name for scc in switched)
        self.coil_sides = tuple(coil_sides)
        self.sides = sides
        law = LinearEnergyLaw(tuple(f"coil_{index + 1}" for index in range(n)), inductance)
        self._inductance = tuple(law.modes)
        self._shapes = tuple(tuple(row) for row in law.shapes)
        self.capacitance = tuple(_real(c) for c in capacitance)

        # state layout: LCC front ends, modal currents, coil capacitors (stable public names)
        front = [side for side in SIDES if sides[side].lcc]
        multi = any(len(sides[side].coils) > 1 for side in SIDES)
        self._cap_names = tuple(self._capacitor_name(j, multi) for j in range(n))
        self.state_names = (
            tuple(f"i_{side}_comp" for side in front)
            + tuple(f"u_{side}_parallel" for side in front)
            + tuple(f"i_mode_{m + 1}" for m in range(n))
            + self._cap_names + self._scc_names
        )
        self._mode_names = tuple(f"i_mode_{m + 1}" for m in range(n))
        self.state = record_type("_LinkState", self.state_names)()
        self.storage = (
            tuple(StoragePort(f"i_{side}_comp", sides[side].compensation_inductance,
                              kind="inductor") for side in front)
            + tuple(StoragePort(f"u_{side}_parallel", sides[side].parallel_capacitance,
                                kind="capacitor") for side in front)
            + tuple(StoragePort(name, self._inductance[m], kind="inductor")
                    for m, name in enumerate(self._mode_names))
            + tuple(StoragePort(name, c, kind="capacitor")
                    for name, c in zip(self._cap_names, self.capacitance))
            + tuple(StoragePort(scc.state_name, scc.capacitance, kind="capacitor")
                    for scc in switched)
        )
        self._coil_outputs = (tuple(f"i_{self.coil_sides[j]}_coil_{self._module(j) + 1}"
                                    for j in range(n)) if multi else ())
        capacitor_outputs = (tuple(f"u_{side}_parallel" for side in front)
                             + self._cap_names + self._scc_names)
        outputs = ("i_primary", "i_secondary", "i_primary_coil", "i_secondary_coil",
                   *self._coil_outputs, *capacitor_outputs,
                   *(f"{scc.side}_scc_on" for scc in switched))
        self.output = record_type("_LinkOutput", outputs)()
        self.output_stages = ()
        self.input = Empty()
        self.primary_port = Port(
            self, "primary", electrical_domain="ac", standalone="voltage",
        )
        self.secondary_port = Port(
            self, "secondary", electrical_domain="ac", standalone="voltage",
        )
        self.ports = (self.primary_port, self.secondary_port)

    def _module(self, coil: int) -> int:
        return self.sides[self.coil_sides[coil]].coils.index(coil)

    def _capacitor_name(self, coil: int, multi: bool) -> str:
        side = self.coil_sides[coil]
        name = f"u_{side}_series" if self.sides[side].lcc else f"u_{side}_cap"
        return f"{name}_{self._module(coil) + 1}" if multi else name

    def coil_currents(self) -> tuple[float, ...]:
        """Coil currents obtained from the stored modal coordinates."""
        modal = tuple(getattr(self.state, name) for name in self._mode_names)
        return tuple(sum(coefficient * current for coefficient, current in zip(row, modal))
                     for row in self._shapes)

    def _winding_currents(self) -> np.ndarray:
        return np.asarray(self.coil_currents())

    def network(self, nb):
        """Declare the compensated coupled-coil network in coil coordinates."""
        from ...solver import netlist as nl

        nb.port_output(self, "i_primary", self.primary_port, "current")
        nb.port_output(self, "i_secondary", self.secondary_port, "current")
        count = len(self.coil_sides)
        shapes = nl._coefficient_array(self._shapes).reshape(count, count)
        inductance = shapes @ np.diag(nl._coefficient_array(self._inductance)) @ shapes.T
        coils = [nb.internal() for _ in range(count)]
        coupled = nb.add(nl.Inductor(
            L=0.5 * (inductance + inductance.T),
            state=nb.state(self, self._mode_names, transform=shapes),
            branches=[(coil, None) for coil in coils],
        ))
        for side in SIDES:
            nb.linear_output(
                self,
                f"i_{side}_coil",
                [(1.0, ("branch_current", coupled, index))
                 for index in self.sides[side].coils],
            )
        for index, name in enumerate(self._coil_outputs):
            nb.output(self, name, ("branch_current", coupled, index))

        def series(node, resistance):
            if not resistance:
                return node
            inner = nb.internal()
            nb.add(nl.Resistor(node, inner, resistance))
            return inner

        switched = {scc.coil: scc for scc in self.switched}
        devices = []
        for side in SIDES:
            spec = self.sides[side]
            node = nb.node(getattr(self, f"{side}_port"))
            if spec.lcc:
                parallel = nb.internal()
                parallel_capacitor = nl.Capacitor(
                    series(parallel, spec.parallel_resistance),
                    None,
                    spec.parallel_capacitance,
                    nb.state(self, f"u_{side}_parallel"),
                )
                nb.add(
                    nl.Inductor(
                        series(node, spec.compensation_resistance),
                        parallel,
                        spec.compensation_inductance,
                        nb.state(self, f"i_{side}_comp"),
                    ),
                    parallel_capacitor,
                )
                nb.output(self, f"u_{side}_parallel", ("branch_voltage", parallel_capacitor))
                node = parallel
            for index in spec.coils:
                scc = switched.get(index)
                terminal = nb.internal() if scc is not None else coils[index]
                capacitor = nb.add(nl.Capacitor(
                    series(node, self.resistance[index]),
                    terminal,
                    self.capacitance[index],
                    nb.state(self, self._cap_names[index]),
                ))
                nb.output(self, self._cap_names[index], ("branch_voltage", capacitor))
                if scc is not None:
                    middle = nb.internal()
                    pair = (
                        nl.Diode(middle, terminal, gate=lambda s=scc: s.positive_gate),
                        nl.Diode(middle, coils[index], gate=lambda s=scc: s.negative_gate),
                    )
                    switched_capacitor = nl.Capacitor(
                            terminal,
                            coils[index],
                            scc.capacitance,
                            nb.state(self, scc.state_name),
                    )
                    nb.add(switched_capacitor, *pair)
                    nb.output(self, scc.state_name, ("branch_voltage", switched_capacitor))
                    devices.append((scc, pair))
        nb.losses(self)
        if devices:
            nb.bind(self._bind_netlist_devices, tuple(devices))
            nb.commutation(
                self,
                margin=self._schedule_margin,
                commutate=self._clock_commutate,
                next_time=self.next_commutation_time,
                after=self._mirror_conduction,
                diodes=[device for _scc, pair in devices for device in pair],
            )

def _link_spec(cfg: Any) -> dict[str, Any]:
    """Normalize one link type into coil/side form; a compensation inductor marks an LCC side."""
    if cfg.type == "ss_multi":
        n = len(cfg.primary_capacitance)
        return {
            "inductance": cfg.inductance,
            "coil_sides": ("primary",) * n + ("secondary",) * n,
            "capacitance": (*cfg.primary_capacitance, *cfg.secondary_capacitance),
            "resistance": (*cfg.primary_resistance, *cfg.secondary_resistance),
            "sides": {"primary": _Side("primary", tuple(range(n))),
                      "secondary": _Side("secondary", tuple(range(n, 2 * n)))},
        }
    matrix = ((cfg.primary_inductance, cfg.mutual_inductance),
              (cfg.mutual_inductance, cfg.secondary_inductance))
    sides, capacitance = {}, []
    for index, side in enumerate(SIDES):
        if hasattr(cfg, f"{side}_compensation_inductance"):
            sides[side] = _Side(
                side, (index,), True,
                getattr(cfg, f"{side}_compensation_inductance"),
                getattr(cfg, f"{side}_compensation_resistance"),
                getattr(cfg, f"{side}_parallel_capacitance"),
                getattr(cfg, f"{side}_parallel_resistance"),
            )
            capacitance.append(getattr(cfg, f"{side}_series_capacitance"))
        else:
            sides[side] = _Side(side, (index,))
            capacitance.append(getattr(cfg, f"{side}_capacitance"))
    return {
        "inductance": matrix,
        "coil_sides": SIDES,
        "capacitance": tuple(capacitance),
        "resistance": (cfg.primary_resistance, cfg.secondary_resistance),
        "sides": sides,
    }


def make_wpt_link(cfg: Any) -> CompensatedLink:
    """Build a compensated link with any configured switch-controlled capacitors."""
    spec = _link_spec(cfg)
    switched = []
    for side in SIDES:
        scc = getattr(cfg, f"{side}_scc", None)
        if scc is None:
            continue
        coils = spec["sides"][side].coils
        if len(coils) != 1:
            raise ValueError(f"a switch-controlled capacitor needs one coil on the {side} side")
        switched.append(SwitchControlledCapacitor(
            side, coils[0], scc.capacitance, scc.angle, scc.drive, scc.sync, scc.sync_angle,
            scc.command))
    if not switched:
        return CompensatedLink(**spec)
    return SwitchedCompensatedLink(**spec, switched=tuple(switched))



# ------------------------------------------------------------------ switch-controlled capacitors

_CURRENT_TOL = 1e-9
_VOLTAGE_TOL = 1e-9


class SwitchControlledCapacitor:
    """A capacitor shunted by two source-to-source MOSFETs with their body diodes.

    Positive branch current needs the positive-direction MOSFET channel and the other
    device's channel or body diode; negative current needs the opposite channel. Gate
    commands therefore do not directly declare whether the whole branch conducts.

    ``drive='common'`` applies the same pulse to both gates. ``drive='alternating'``
    uses two pulses of width ``pi + angle``, displaced by pi. Their overlap has the
    commanded nominal conduction angle; body diodes can conduct outside that overlap.
    Every actual voltage-clamping transition accounts for its capacitor discharge energy.
    """

    def __init__(self, side: str, coil: int, capacitance: float, angle: float,
                 drive: str = "common", sync: str = "current", sync_angle: float = 0.0,
                 command: bool = False) -> None:
        if drive not in ("common", "alternating"):
            raise ValueError(f"unknown SCC gate drive {drive!r}")
        if sync not in ("current", "bridge"):
            raise ValueError(f"unknown SCC clock source {sync!r}")
        self.side, self.coil = side, coil
        self.capacitance = _real(capacitance)
        self.angle = self.theta = _real(angle)
        self.drive, self.sync = drive, sync
        self.sync_angle = _real(sync_angle)
        self.command = bool(command)
        self.state_name = f"u_{side}_scc"
        self.bridge = None
        self.started = self.referenced = self.on = False
        self.gates = (False, False)
        self._gate_cache = None
        self.zero = LocatedZero()
        self.half = 0.0
        self._reference = 0.0
        self._since = None
        self.switchings = self.hard_turn_ons = 0
        self.max_turn_on_voltage = self.impulsive_energy = 0.0
        self._edge = -math.inf

    def __copy__(self):
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        result._gate_cache = None
        return result

    @property
    def gates(self):
        return self.positive_gate, self.negative_gate

    @gates.setter
    def gates(self, values):
        self.positive_gate, self.negative_gate = values

    def equation_state_bindings(self):
        return {"sign": (self.zero, "sign"), "i_ref": (self.zero, "residual"),
                **{name: (self, name) for name in (
                    "started", "referenced", "on", "positive_gate", "negative_gate", "theta",
                    "half", "_reference", "switchings", "hard_turn_ons", "max_turn_on_voltage",
                    "impulsive_energy", "_edge")}}

    def bind_bridge(self, bridge: Any) -> None:
        self.bridge = bridge

    def _command(self) -> float:
        value = self.bridge.register("scc_angle") if self.command else self.angle
        if not np.isfinite(value):
            raise ValueError(f"{self.side} SCC angle must be finite")
        return min(math.pi, max(0.0, _real(value)))

    def _clock(self):
        if self.sync == "bridge":
            clock = self.bridge.phase_clock()
            return PhaseClock(clock.time, clock.angle - self.bridge.phase_reference() - self.sync_angle,
                              clock.rate)
        if not self.referenced or self.half <= 0.0:
            return None
        return PhaseClock(self._reference, math.pi if self.zero.sign < 0.0 else 0.0,
                          math.pi / self.half)

    def _pulses(self, theta):
        if self.drive == "alternating":
            return (0.0, math.pi), 0.5 * (math.pi + theta), 2.0 * math.pi
        return (0.5 * math.pi,), 0.5 * theta, math.pi

    def _gate_word(self, now, clock, theta) -> tuple:
        """Gate word and its validity interval from the existing periodic clock."""
        centres, width, period = self._pulses(theta)
        if width == 0.0:
            return (False, False), -math.inf, math.inf
        if width >= 0.5 * period:
            return (True, True), -math.inf, math.inf
        if clock is None:
            return (False, False), -math.inf, math.inf
        angle = clock.at(now)
        gates, starts, ends = [], [], []
        for centre in centres:
            cycle = _real(np.floor((angle - centre + width) / period))
            middle = clock.time + (centre + period * cycle - clock.angle) / clock.rate
            start, end = middle - width / clock.rate, middle + width / clock.rate
            active = now < end
            gates.append(active)
            starts.append(_real(np.where(active, start, end)))
            ends.append(_real(np.where(active, end, start + period / clock.rate)))
        word = tuple(gates) if len(gates) == 2 else tuple(gates) * 2
        return word, max(starts), min(ends)

    def _gate_state(self, t):
        clock = self._clock()
        theta = self._command() if self.sync == "bridge" else self.theta
        key, now = (clock, theta), t + commutation_tolerance(t)
        cached = self._gate_cache
        if cached is not None and cached[0] == key and cached[1] <= now < cached[2]:
            return cached[3]
        word, start, end = self._gate_word(now, clock, theta)
        # Derived waveform cache only; accepted switch state and restart data stay separate.
        self._gate_cache = key, start, end, word
        return word

    def next_time(self, t: float) -> float:
        """Use the same absolute edge times and due comparison as the dispatcher."""
        expected = self._gate_state(t)
        if not self.started or self.gates != expected:
            return t
        return self._gate_cache[2]

    def _tracks_current(self) -> bool:
        return self.sync == "current" or (self.on & (self.gates[0] != self.gates[1]))

    def _flow(self, current) -> float:
        return (self.zero.sign if self.zero.sign and abs(self.zero.offset(current)) <= _CURRENT_TOL
                else current)

    def margin(self, t: float, current: float, voltage: float) -> float:
        if not self.started or self.gates != self._gate_state(t):
            return -1.0
        margin = self.zero.margin(current, _CURRENT_TOL) if self._tracks_current() else math.inf
        positive, negative = self.gates
        if self.on:
            if not (positive or negative):
                return -1.0
            if positive != negative:
                direction = 1.0 if positive else -1.0
                if self.zero.sign != direction:
                    return -1.0
                margin = min(margin, self.zero.margin(current))
            return margin
        if positive and negative:
            return -1.0
        if positive or negative:
            direction = 1.0 if positive else -1.0
            margin = min(margin, _VOLTAGE_TOL - direction * voltage)
            # Complementarity at zero voltage selects the path carrying the coil current.
            if (abs(voltage) <= _VOLTAGE_TOL
                    and direction * self._flow(current) > 0):
                return -1.0
        return margin

    def commutate(self, link: Any, t: float, current: float, voltage: float) -> None:
        self._synchronize_current(t, current)
        self.gates = self._gate_state(t)
        self._set_conduction(link, current, voltage)

    def clock(self, t: float, current: float) -> None:
        """Advance the gate clock while the netlist owns conduction."""
        self._synchronize_current(t, current)
        before, self.gates = self.gates, self._gate_state(t)
        if self.gates != before:
            self._edge = t

    def schedule_margin(self, t: float, current: float) -> float:
        """Return the validity margin of the gate schedule and current clock."""
        if not self.started or self.gates != self._gate_state(t):
            return -1.0
        return self.zero.margin(current, _CURRENT_TOL) if self.sync == "current" else math.inf

    def mirror(self, t: float, conducting: bool, voltage: float) -> None:
        """Mirror netlist conduction into the public SCC state and diagnostics."""
        positive, negative = self.gates
        if conducting:
            if positive != negative and self.zero.sign == 0.0:
                self.zero.sign = 1.0 if positive else -1.0
            if not self.on:
                self.switchings += 1
                if (abs(voltage) > _VOLTAGE_TOL
                        and abs(t - self._edge) <= commutation_tolerance(t)):
                    self.hard_turn_ons += 1
                self.max_turn_on_voltage = max(self.max_turn_on_voltage, abs(voltage))
                self.impulsive_energy += 0.5 * self.capacitance * voltage * voltage
        elif self.on:
            self.switchings += 1
        self.on = bool(conducting)

    def _synchronize_current(self, t: float, current: float) -> None:
        if not self.started:
            self.started = True
            if self._since is not None:
                self._reference, self._since = t - self._since, None
        if self.zero.sign == 0.0 or not self._tracks_current():
            if abs(current) > _CURRENT_TOL:
                self.zero.sign = _real(np.copysign(1.0, current))
        elif self.zero.margin(current) < 0.0:
            self.zero.locate(current, -self.zero.sign)
            if self.sync == "current":
                if self.referenced:
                    self.half = t - self._reference
                self._reference, self.referenced = t, True
                self.theta = self._command()
    def _set_conduction(self, link: Any, current: float, voltage: float) -> None:
        positive, negative = self.gates
        direction = 1.0 if positive else -1.0
        flow = self._flow(current)
        conducting = (positive and negative) or ((positive or negative) and (
            direction * voltage > _VOLTAGE_TOL
            or (abs(voltage) <= _VOLTAGE_TOL and direction * flow > 0.0)))
        if conducting:
            if positive != negative and self.zero.sign == 0.0:
                self.zero.sign = direction
            if not self.on:
                self.switchings += 1
            if abs(voltage) > _VOLTAGE_TOL:
                self.hard_turn_ons += 1
            self.max_turn_on_voltage = max(self.max_turn_on_voltage, abs(voltage))
            self.impulsive_energy += 0.5 * self.capacitance * voltage * voltage
            setattr(link.state, self.state_name, 0.0)
        elif self.on:
            self.switchings += 1
        self.on = conducting

    def get_state(self, t: float) -> dict[str, Any]:
        state = {"on": self.on, "positive_gate": self.gates[0], "negative_gate": self.gates[1],
                 "sign": _real(self.zero.sign), "i_ref": _real(self.zero.residual)}
        if self.sync == "current":
            since = t - self._reference if self.started and self.referenced else self._since or 0.0
            state.update(referenced=self.referenced, since=_real(since), half=_real(self.half),
                         theta=_real(self.theta))
        return state

    def set_state(self, values: Mapping[str, Any]) -> None:
        unknown = set(values) - set(self.get_state(0.0))
        if unknown:
            raise KeyError(f"{self.side} SCC: no state(s) {sorted(unknown)}")
        for name in ("on", "referenced"):
            if name in values:
                setattr(self, name, bool(values[name]))
        self.gates = (bool(values.get("positive_gate", self.gates[0])),
                      bool(values.get("negative_gate", self.gates[1])))
        for name in ("half", "theta"):
            if name in values:
                setattr(self, name, float(values[name]))
        if "sign" in values:
            self.zero.sign = float(values["sign"])
        if "i_ref" in values:
            self.zero.residual = float(values["i_ref"])
        if "since" in values:
            self._since, self.started = float(values["since"]), False

    def diagnostics(self):
        return {"switchings": self.switchings, "hard_turn_ons": self.hard_turn_ons,
                "max_turn_on_voltage": self.max_turn_on_voltage,
                "turn_on_energy": self.impulsive_energy}


class SwitchedCompensatedLink(CompensatedLink):
    """A compensated link with switch-controlled capacitors, a commutating subsystem.

    Its validity margin, commutations and scheduled gate instants come from its capacitors; the
    commutation framework writes back the capacitor voltage zeroed at a turn-on. The switch
    states and schedules are named states (``<side>_scc.*``) of the link.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._t = 0.0
        self._netlist_devices = None
        self.output_stages = (OutputMethod(
            "set_outputs", inputs=(),
            outputs=tuple(f"{scc.side}_scc_on" for scc in self.switched),
        ),)

    def set_outputs(self, t: float) -> None:
        self._t = t
        for scc in self.switched:
            setattr(self.output, f"{scc.side}_scc_on", _real(scc.on))

    def _scc_values(self):
        coils = self.coil_currents()
        for scc in self.switched:
            yield scc, coils[scc.coil], _real(getattr(self.state, scc.state_name))

    def commutation_margin(self) -> float:
        return min(scc.margin(self._t, current, voltage)
                   for scc, current, voltage in self._scc_values())

    def commutate(self, t: float) -> None:
        for scc, current, voltage in list(self._scc_values()):
            scc.commutate(self, t, current, voltage)

    def next_commutation_time(self) -> float:
        return min(scc.next_time(self._t) for scc in self.switched)

    def _schedule_margin(self) -> float:
        coils = self.coil_currents()
        return min(
            scc.schedule_margin(self._t, coils[scc.coil]) for scc in self.switched
        )

    def _clock_commutate(self, t: float) -> None:
        coils = self.coil_currents()
        netlist, devices = self._netlist_devices
        for scc, pair in devices:
            before = scc.gates
            scc.clock(t, coils[scc.coil])
            flow = scc._flow(coils[scc.coil])
            for index, (device, direction) in enumerate(zip(pair, (-1.0, 1.0))):
                if (scc.on and before[index] and not scc.gates[index]
                        and scc.gates[1 - index] and direction * flow > 0.0):
                    netlist.set_conducting(device, True)

    def _mirror_conduction(self, t: float, before: Mapping[str, float]) -> None:
        netlist, devices = self._netlist_devices
        for scc, (positive, negative) in devices:
            conducting = (
                (scc.positive_gate | netlist.conduction(positive)[0])
                & (scc.negative_gate | netlist.conduction(negative)[0])
            )
            scc.mirror(t, conducting, _real(before[scc.state_name]))

    def _restore_conduction(self) -> None:
        if self._netlist_devices is None:
            return
        netlist, devices = self._netlist_devices
        for scc, (positive, negative) in devices:
            netlist.set_conducting(positive, scc.on and not scc.positive_gate)
            netlist.set_conducting(negative, scc.on and not scc.negative_gate)

    def equation_state_bindings(self):
        return {"time": (self, "_t"),
                **{f"{scc.side}_scc.{name}": binding for scc in self.switched
                   for name, binding in scc.equation_state_bindings().items()}}

    @property
    def impulsive_dissipated_energy(self) -> float:
        """Energy (J) dissipated by capacitor discharges at switch closings."""
        return sum(scc.impulsive_energy for scc in self.switched)

    def diagnostics(self) -> dict[str, float]:
        return {f"{scc.side}_scc.{key}": value for scc in self.switched
                for key, value in scc.diagnostics().items()}

    # -- named switch state (a state part of the link) -----------------------------------------
    def get_state(self) -> dict[str, Any]:
        return {f"{scc.side}_scc.{key}": value for scc in self.switched
                for key, value in scc.get_state(self._t).items()}

    def set_state(self, values: Mapping[str, Any]) -> None:
        for scc in self.switched:
            prefix = f"{scc.side}_scc."
            own = {key[len(prefix):]: value for key, value in values.items()
                   if key.startswith(prefix)}
            if own:
                scc.set_state(own)
        unknown = [key for key in values
                   if not any(key.startswith(f"{scc.side}_scc.") for scc in self.switched)]
        if unknown:
            raise KeyError(f"switched link: no state(s) {sorted(unknown)}")
        self._restore_conduction()

    @property
    def discrete_state_names(self) -> tuple[str, ...]:
        return tuple(f"{scc.side}_scc.{name}" for scc in self.switched
                     for name in ("sign",))

    # Current-derived clock memory affects future gates and remains in the smooth map.
    @property
    def linearization_state_names(self):
        return tuple(f"{scc.side}_scc.{name}" for scc in self.switched if scc.sync == "current"
                     for name in ("since", "half", "theta"))

@dataclass(frozen=True)
class SCCParams:
    """Capacitor shunted by a source-to-source MOSFET pair, with explicit gate drive.

    ``common`` drives both gates together; ``alternating`` uses overlapping pulses displaced
    by pi. ``angle`` is their nominal conduction/overlap angle. Device channels and body diodes
    determine the actual branch conduction; zero-voltage turn-on is not imposed artificially.
    """

    capacitance: float
    angle: float = 0.0
    drive: str = "common"
    sync: str = "current"
    sync_angle: float = 0.0
    command: bool = False

    _choices = {"drive": ("common", "alternating"), "sync": ("current", "bridge")}


@dataclass(frozen=True, kw_only=True)
class _CoilPairLinkParams:
    """Common coupled-coil parameters of a two-winding compensated link."""

    type: str
    primary_inductance: float
    secondary_inductance: float
    mutual_inductance: float
    primary_resistance: float = 0.0
    secondary_resistance: float = 0.0
    primary_scc: Optional[SCCParams] = None
    secondary_scc: Optional[SCCParams] = None


@dataclass(frozen=True, kw_only=True)
class SSLinkParams(_CoilPairLinkParams):
    """Series-series compensated two-winding inductive link in SI units."""

    type: str = "ss"
    primary_capacitance: float
    secondary_capacitance: float

    _choices = {"type": ("ss",)}


@dataclass(frozen=True, kw_only=True)
class LCCLinkParams(_CoilPairLinkParams):
    """Symmetric-form LCC-LCC compensated two-winding inductive link in SI units.

    Each port has a series compensation inductor, a shunt compensation capacitor and a
    series coil capacitor.  The two coil inductances are coupled by ``mutual_inductance``.
    """

    type: str = "lcc"
    primary_series_capacitance: float
    secondary_series_capacitance: float
    primary_compensation_inductance: float
    secondary_compensation_inductance: float
    primary_parallel_capacitance: float
    secondary_parallel_capacitance: float
    primary_compensation_resistance: float = 0.0
    secondary_compensation_resistance: float = 0.0
    primary_parallel_resistance: float = 0.0  # ESR in series with the shunt capacitor
    secondary_parallel_resistance: float = 0.0

    _choices = {"type": ("lcc",)}


@dataclass(frozen=True, kw_only=True)
class LCCSLinkParams(_CoilPairLinkParams):
    """LCC-compensated primary and series-compensated secondary of a two-winding link (SI).

    The primary port has a series compensation inductor, a shunt capacitor (optional ESR) and a
    series coil capacitor; the secondary coil has a series capacitor only.
    """

    type: str = "lcc_s"
    primary_series_capacitance: float
    primary_compensation_inductance: float
    primary_parallel_capacitance: float
    secondary_capacitance: float
    primary_compensation_resistance: float = 0.0
    primary_parallel_resistance: float = 0.0

    _choices = {"type": ("lcc_s",)}


@dataclass(frozen=True, kw_only=True)
class MultiSSLinkParams:
    """N series-series modules with inter- and cross-coupling, in SI units.

    ``inductance`` is the symmetric 2N x 2N inductance matrix of the coils in the order
    Tx1..TxN, Rx1..RxN. Every Tx coil (with its series capacitor and resistance) connects in
    parallel to the primary port, every Rx coil to the secondary port.
    """

    type: str = "ss_multi"
    inductance: list
    primary_capacitance: list
    secondary_capacitance: list
    primary_resistance: Optional[list] = None  # default: zero per module
    secondary_resistance: Optional[list] = None

    _choices = {"type": ("ss_multi",)}

    def __post_init__(self) -> None:
        modules = len(self.primary_capacitance) if isinstance(
            self.primary_capacitance, (list, tuple)) else 0
        for name in ("primary_resistance", "secondary_resistance"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, [0.0] * modules)


@dataclass(frozen=True)
class _WPTUnitFields:
    """Fields unique to a composite wireless-power-transfer unit."""

    primary: Any = field(metadata={"unit": True})
    secondary: Any = field(metadata={"unit": True})
    link: Any = field(metadata={"types": {"ss": SSLinkParams, "lcc": LCCLinkParams,
                                          "lcc_s": LCCSLinkParams,
                                          "ss_multi": MultiSSLinkParams}})




class _WPTSignals:
    """Expose shared link quantities without making the composite a scheduled unit."""

    def __init__(self, name: str, link: Any) -> None:
        self.name = name
        self.link = link
        self._outputs = tuple(type(link.output).__slots__)

    def signals(self) -> dict[str, float]:
        out = self.link.output
        return {f"{self.name}.link.{name}": getattr(out, name) for name in self._outputs}




def _assemble_wpt(name: str, cfg: Any, system: "System",
                  parts: Mapping[str, Any]) -> _UnitAssembly:
    if _part(parts, name) is not None:
        raise ConfigError(
            f"parts.{name}: a composite WPT unit is assembled from its primary and secondary parts"
        )
    link = _parameterized(make_wpt_link, cfg.link)
    assembly = _UnitAssembly(
        subsystems={f"{name}.link": link},
        signal_sources=(_WPTSignals(name, link),),
        default_port=unit_type(cfg.type).default_port,
    )
    for side in unit_type(cfg.type).children:
        child = _assemble_unit(f"{name}.{side}", getattr(cfg, side), system, parts)
        physical_port = getattr(link, f"{side}_port")
        child.attach(physical_port, f"units.{name}.{side}", "power")
        for scc in link.switched:
            if scc.side == side:
                scc.bind_bridge(child.units[child.port("power").owner].bridge)
        assembly.ports[side] = child.port()
        assembly.include(child)
    default = assembly.port()
    assembly.unit_port_bindings[f"{name}.link"] = (default,)
    return assembly


def _wpt(u, base, where: str, *, nested: bool = False) -> None:
    """Check a WPT composite and its shared compensated magnetic link."""
    for side in unit_type(u.type).children:
        child = getattr(u, side)
        child_where = f"{where}.{side}"
        spec = unit_type(child.type, f"{child_where}.type")
        if spec.phases != 1 or spec.composite:
            raise ConfigError(
                f"{child_where}: a WPT child needs a scalar, non-composite unit"
            )
        spec.validator(child, base, child_where, nested=True)
    link = u.link
    _switch_controlled_capacitors(u, where)
    if link.type == "ss_multi":
        _multi_link(link, f"{where}.link")
        return
    positive = ["primary_inductance", "secondary_inductance"]
    nonnegative = ["primary_resistance", "secondary_resistance"]
    for side in unit_type(u.type).children:
        if hasattr(link, f"{side}_compensation_inductance"):
            positive += [f"{side}_series_capacitance", f"{side}_compensation_inductance",
                         f"{side}_parallel_capacitance"]
            nonnegative += [f"{side}_compensation_resistance", f"{side}_parallel_resistance"]
        else:
            positive.append(f"{side}_capacitance")
    for name in positive:
        value = getattr(link, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ConfigError(f"{where}.link.{name} must be finite and positive")
    for name in nonnegative:
        value = getattr(link, name)
        if not math.isfinite(value) or value < 0.0:
            raise ConfigError(f"{where}.link.{name} must be finite and nonnegative")
    if not math.isfinite(link.mutual_inductance):
        raise ConfigError(f"{where}.link.mutual_inductance must be finite")
    determinant = (
        link.primary_inductance * link.secondary_inductance
        - link.mutual_inductance * link.mutual_inductance
    )
    if determinant <= 0.0:
        raise ConfigError(
            f"{where}.link: the coupled-inductor matrix must be positive definite "
            "(mutual_inductance^2 < primary_inductance * secondary_inductance)"
        )


def _switch_controlled_capacitors(u, where: str) -> None:
    """Check the switch-controlled capacitors of a WPT link and their same-side converters."""
    for side in unit_type(u.type).children:
        child = getattr(u, side)
        ctrl = child.ctrl
        clocked = ctrl is not None and child.bridge.pwm_timed and child.pwm is not None
        commanded = ctrl is not None and "scc_angle" in ctrl.outputs
        scc = getattr(u.link, f"{side}_scc", None)
        at = f"{where}.link.{side}_scc"
        if scc is None:
            if commanded:
                raise ConfigError(f"{where}.{side}.ctrl.outputs.scc_angle: the link has no "
                                  f"{side}_scc with command: 1")
            continue
        if not math.isfinite(scc.capacitance) or scc.capacitance <= 0.0:
            raise ConfigError(f"{at}.capacitance must be finite and positive")
        if not math.isfinite(scc.angle) or not 0.0 <= scc.angle <= math.pi:
            raise ConfigError(f"{at}.angle must be in [0, pi]")
        if not math.isfinite(scc.sync_angle):
            raise ConfigError(f"{at}.sync_angle must be finite")
        if scc.sync == "bridge" and not clocked:
            raise ConfigError(f"{at}.sync: 'bridge' needs a clocked bridge on the {side} side")
        if scc.command and not (clocked and commanded):
            raise ConfigError(f"{at}.command needs a clocked controller output scc_angle "
                              f"on the {side} side (ctrl.outputs.scc_angle)")
        if commanded and not scc.command:
            raise ConfigError(f"{where}.{side}.ctrl.outputs.scc_angle: {at}.command is off")
        if scc.command and scc.angle != 0.0:
            raise ConfigError(f"{at}.angle is not used with command: the angle is the "
                              "controller output scc_angle (0 before its first write)")


def _multi_link(link, where: str) -> None:
    """Check the coil matrix and per-module values of a multi-module series-series link."""
    names = ("primary_capacitance", "secondary_capacitance",
             "primary_resistance", "secondary_resistance")
    try:
        matrix = np.asarray(link.inductance, dtype=float)
        vectors = {name: np.asarray(getattr(link, name), dtype=float) for name in names}
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: inductance must be a numeric matrix and the per-module "
                          "values numeric lists") from None
    modules = vectors["primary_capacitance"].size
    if modules < 1 or any(vector.shape != (modules,) for vector in vectors.values()):
        raise ConfigError(f"{where}: {', '.join(names)} must be lists of one value per module")
    if matrix.shape != (2 * modules, 2 * modules):
        raise ConfigError(f"{where}.inductance must be a {2 * modules} x {2 * modules} matrix "
                          f"for {modules} module(s)")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0):
        raise ConfigError(f"{where}.inductance must be finite and symmetric")
    if np.linalg.eigvalsh(matrix).min() <= 0.0:
        raise ConfigError(f"{where}.inductance must be positive definite")
    for name in names:
        vector = vectors[name]
        positive = name.endswith("capacitance")
        if not np.all(np.isfinite(vector)) or np.any(vector <= 0.0 if positive else vector < 0.0):
            raise ConfigError(f"{where}.{name} must be finite and "
                              f"{'positive' if positive else 'nonnegative'}")



# WPT is installed and registered with the standard MagComponent addon.
register_unit_type(
    "wpt",
    params=_WPTUnitFields,
    children=("primary", "secondary"),
    default_port="primary",
    phases=1,
    validator=_wpt,
    assembler=_assemble_wpt,
)
