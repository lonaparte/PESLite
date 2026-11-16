"""Magnetic components and their constant-inductance coupled-coil implementation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar, Mapping

import numpy as np

from ...components.network import Element, NetworkComponent, register_component_type
from ...solver import Bag, ConfigError, OutputMethod, Port, StoragePort

__all__ = [
    "MagComponent",
    "MagneticBranchDescription",
    "MagneticDescription",
    "MagneticParams",
    "WindingParams",
    "CoupledCoils",
    "CoupledCoilsModel",
]


GROUND = "ground"


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
        f"windings.{name}.{side}": bus
        for name, winding in windings.items()
        for side in ("positive", "negative")
        if (bus := getattr(winding, side)) != GROUND
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


def _inductance_decomposition(
    inductance: Any, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        self.resistance = np.asarray(resistance, dtype=float)
        self.temperature_coefficient_per_K = float(temperature_coefficient_per_K)
        self.temperature_reference_K = float(temperature_reference_K)

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
        return self.resistance * factor

    def _winding_currents(self) -> np.ndarray:
        raise NotImplementedError

    def _material_loss(self) -> float:
        return 0.0

    def _loss_parts(self) -> tuple[float, float]:
        current = self._winding_currents()
        copper = float(np.dot(self._effective_resistance(), current * current))
        material = float(self._material_loss())
        if (not math.isfinite(copper) or not math.isfinite(material)
                or copper < 0.0 or material < 0.0):
            raise ConfigError("magnetic losses must be nonnegative finite W")
        return copper, material

    def set_loss_outputs(self, _t: float) -> None:
        copper, material = self._loss_parts()
        self.output.copper_loss_W = copper
        self.output.core_loss_W = material
        self.output.loss_W = copper + material

    def dissipated_power(self) -> float:
        return sum(self._loss_parts())


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
        matrix, modes, shapes = _inductance_decomposition(inductance, len(names))

        self.names = names
        self.inductance = matrix
        self._shapes = shapes
        self.state_names = tuple(f"i_mode_{index + 1}" for index in range(len(names)))
        self.state = type("_MagneticState", (Bag,), {"__slots__": self.state_names})(
            **dict.fromkeys(self.state_names, 0.0)
        )
        output_names = (
            *tuple(label for name in names for label in (f"i_{name}", f"u_{name}")),
            "copper_loss_W", "core_loss_W", "loss_W",
        )
        self.output = type("_MagneticOutput", (Bag,), {"__slots__": output_names})(
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
        self.storage = tuple(StoragePort(name, float(mode), kind="inductor")
                             for name, mode in zip(self.state_names, modes))

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
        coupled = nb.add(nl.CoupledInductors(
            branches,
            self.inductance,
            nb.state(self, self.state_names, transform=self._shapes),
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
        if temperature_dependent and np.any(self.resistance):
            nb.add(nl.Dissipation(nb.function(
                self._copper_loss,
                ("temperature_K",),
                states=self.state_names,
            )))
        nb.losses(self)

    def _winding_currents(self) -> np.ndarray:
        modal = np.fromiter(
            (getattr(self.state, name) for name in self.state_names),
            dtype=float,
            count=len(self.state_names),
        )
        return self._shapes @ modal

    def _winding_drop(self, index: int, _t: float) -> float:
        return float(self._effective_resistance()[index] * self._winding_currents()[index])

    def _copper_loss(self, _t: float) -> float:
        current = self._winding_currents()
        return float(np.dot(self._effective_resistance(), current * current))

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
    ) -> tuple[tuple[str, ...], tuple[float, ...], tuple[tuple[bool, bool], ...]]:
        names = _validate_winding_names(windings)
        if names != tuple(winding_names):
            raise ConfigError(
                f"elements.{name}: winding order {names!r} does not match "
                f"model order {tuple(winding_names)!r}"
            )
        for label, winding in windings.items():
            for side in ("positive", "negative"):
                bus = getattr(winding, side)
                if bus == GROUND:
                    continue
                if bus not in buses:
                    raise ConfigError(
                        f"elements.{name}.windings.{label}.{side}: unknown bus {bus!r}"
                    )
                if buses[bus].phases != 1:
                    raise ConfigError(
                        f"elements.{name}.windings.{label}.{side}: "
                        "magnetic windings require scalar buses"
                    )
        return (
            names,
            tuple(winding.r for winding in windings.values()),
            tuple((winding.positive == GROUND, winding.negative == GROUND)
                  for winding in windings.values()),
        )

    def _attach(self, name: str, cfg: Any, model: Any) -> None:
        self.name, self.cfg, self.model = name, cfg, model

    def _assemble(self, name: str, cfg: Any, winding_names: tuple[str, ...], inductance: Any,
                  buses: dict, magnetic: MagneticDescription | None) -> None:
        """Assemble a constant winding-inductance law through the shared netlist model."""
        names, resistance, terminals = self._winding_data(
            name, cfg.windings, winding_names, buses
        )
        self._attach(
            name, cfg,
            CoupledCoilsModel(
                names,
                inductance,
                resistance,
                terminals,
                magnetic,
                cfg.temperature_coefficient_per_K,
                cfg.temperature_reference_K,
            ),
        )

    def subsystems(self) -> dict[str, Any]:
        return {self.name: self.model}

    def connections(self) -> dict:
        source = self.cfg.temperature_source
        return {(self.model, "temperature_K"): source} if source is not None else {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return tuple(
            (bus, port)
            for winding, pair in zip(self.cfg.windings.values(), self.model.winding_ports)
            for bus, port in zip((winding.positive, winding.negative), pair)
            if bus != GROUND
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
            matrix, _modes, _shapes = _inductance_decomposition(
                self.inductance, len(self.windings)
            )
            object.__setattr__(self, "inductance", matrix.tolist())

    type = "coupled_coils"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self._assemble(name, cfg, tuple(cfg.windings), cfg.inductance, buses, cfg.magnetic)
