"""Magnetic windings represented by the ordinary PESLite component and netlist contracts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np

from ..solver import Bag, ConfigError, Empty, Port, StoragePort
from .network import Element, NetworkComponent, _register_assembled_type

__all__ = ["MagComponent", "MagneticModel", "WindingParams"]


GROUND = "ground"
_MODEL_FORMAT = "peslite-magnetic-model"
_MODEL_VERSION = 1
_MODEL_REQUIRED = frozenset({"format", "version", "winding_names", "inductance"})
_MODEL_OPTIONAL = frozenset({"physical", "field_points", "field_basis"})


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


def _validate_windings(windings: dict[str, WindingParams]) -> tuple[str, ...]:
    names = tuple(windings)
    if not names:
        raise ConfigError("windings must define at least one winding")
    if any(not name.isidentifier() for name in names):
        raise ConfigError("winding names must be Python identifiers")
    return names


def _connection_buses(windings: dict[str, WindingParams]) -> dict[str, str]:
    return {
        f"windings.{name}.{side}": bus
        for name, winding in windings.items()
        for side in ("positive", "negative")
        if (bus := getattr(winding, side)) != GROUND
    }


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
    return matrix, modes, shapes


@dataclass(frozen=True, kw_only=True)
class MagneticModel:
    """Validated electrical realization with optional external magnetic result data."""

    winding_names: tuple[str, ...]
    inductance: np.ndarray
    physical: dict[str, Any] = field(default_factory=dict)
    field_points: np.ndarray | None = None
    field_basis: np.ndarray | None = None
    _modes: np.ndarray = field(init=False, repr=False, compare=False)
    _shapes: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = tuple(self.winding_names)
        if not names or len(set(names)) != len(names):
            raise ConfigError("winding_names must contain unique names")
        if any(not isinstance(name, str) or not name.isidentifier() for name in names):
            raise ConfigError("winding_names must contain identifiers")
        matrix, modes, shapes = _inductance_decomposition(self.inductance, len(names))
        matrix.setflags(write=False)
        modes.setflags(write=False)
        shapes.setflags(write=False)

        try:
            encoded = json.dumps(self.physical, allow_nan=False, sort_keys=True)
            physical = json.loads(encoded)
        except (TypeError, ValueError):
            raise ConfigError("physical must contain finite JSON-compatible data") from None
        if not isinstance(physical, dict):
            raise ConfigError("physical must be a mapping")

        points, basis = self.field_points, self.field_basis
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

        object.__setattr__(self, "winding_names", names)
        object.__setattr__(self, "inductance", matrix)
        object.__setattr__(self, "physical", MappingProxyType(physical))
        object.__setattr__(self, "field_points", points)
        object.__setattr__(self, "field_basis", basis)
        object.__setattr__(self, "_modes", modes)
        object.__setattr__(self, "_shapes", shapes)

    @classmethod
    def linear(cls, winding_names: tuple[str, ...], inductance: Any) -> "MagneticModel":
        """Create the basic in-memory realization used by coupled coils."""
        return cls(winding_names=winding_names, inductance=inductance)

    def save(self, file: str | Path) -> Path:
        """Atomically write the portable, data-only magnetic-model archive."""
        path = Path(file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = {
            "format": np.asarray(_MODEL_FORMAT),
            "version": np.asarray(_MODEL_VERSION),
            "winding_names": np.asarray(self.winding_names),
            "inductance": self.inductance,
            "physical": np.asarray(json.dumps(dict(self.physical), allow_nan=False,
                                               sort_keys=True, separators=(",", ":"))),
        }
        if self.field_points is not None:
            entries["field_points"] = self.field_points
            entries["field_basis"] = self.field_basis
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

    @classmethod
    def load(cls, file: str | Path) -> "MagneticModel":
        """Load and strictly validate one portable magnetic-model archive."""
        path = Path(file).expanduser().resolve()
        if not path.is_file():
            raise ConfigError(f"magnetic model file does not exist: {path}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = frozenset(archive.files)
                missing = _MODEL_REQUIRED - keys
                unknown = keys - _MODEL_REQUIRED - _MODEL_OPTIONAL
                if missing or unknown:
                    details = []
                    if missing:
                        details.append(f"missing entries {sorted(missing)}")
                    if unknown:
                        details.append(f"unknown entries {sorted(unknown)}")
                    raise ConfigError("; ".join(details))
                format_name = _scalar(archive["format"], "format")
                version = _scalar(archive["version"], "version")
                if format_name != _MODEL_FORMAT:
                    raise ConfigError(f"unknown magnetic model format {format_name!r}")
                if type(version) not in (int, np.int32, np.int64) or int(version) != _MODEL_VERSION:
                    raise ConfigError(f"unsupported magnetic model version {version!r}")
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
                points = archive["field_points"].copy() if "field_points" in keys else None
                basis = archive["field_basis"].copy() if "field_basis" in keys else None
                return cls(
                    winding_names=names,
                    inductance=archive["inductance"].copy(),
                    physical=physical,
                    field_points=points,
                    field_basis=basis,
                )
        except ConfigError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise ConfigError(f"cannot load magnetic model {path}: {exc}") from None


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


@dataclass(frozen=True, kw_only=True)
class _ModelFileParams:
    file: str = field(metadata={"asset": True, "runtime": False})

    def __post_init__(self) -> None:
        if not isinstance(self.file, str) or not self.file:
            raise ConfigError("file must be a nonempty path")


class MagComponent(NetworkComponent):
    """Ordinary PESLite component backed by one validated magnetic model."""

    outputs_need_inputs: ClassVar[bool] = False

    def __init__(self, model: MagneticModel,
                 resistance: tuple[float, ...],
                 terminals: tuple[tuple[bool, bool], ...]) -> None:
        names, matrix = model.winding_names, model.inductance
        modes, shapes = model._modes, model._shapes
        self.magnetic = model
        self.names = names
        self.resistance = resistance
        self.inductance = matrix
        self._shapes = tuple(tuple(float(value) for value in row) for row in shapes)
        self.state_names = tuple(f"i_mode_{index + 1}" for index in range(len(names)))
        self.state = type("_MagneticState", (Bag,), {"__slots__": self.state_names})(
            **dict.fromkeys(self.state_names, 0.0)
        )
        output_names = tuple(label for name in names for label in (f"i_{name}", f"u_{name}"))
        self.output = type("_MagneticOutput", (Bag,), {"__slots__": output_names})(
            **dict.fromkeys(output_names, 0.0)
        )
        self.input = Empty()
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
        from ..solver import netlist as nl

        branches = []
        for (positive, negative), resistance in zip(self.winding_ports, self.resistance):
            high, low = nb.node(positive), nb.node(negative)
            if resistance:
                inner = nb.internal()
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
        nb.losses(self)


@_register_assembled_type
class _MagneticElement(Element):
    """Full configuration entry for the ordinary magnetic component."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        windings: dict[str, WindingParams] = field(metadata={"structural": True})
        model: _ModelFileParams = field(metadata={"structural": True})
        type: str = "magnetic_component"

        def __post_init__(self) -> None:
            _validate_windings(self.windings)

        def connection_buses(self) -> dict[str, str]:
            return _connection_buses(self.windings)

    type = "magnetic_component"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        self._assemble(name, cfg, cfg.windings, MagneticModel.load(cfg.model.file), buses)

    def _assemble(self, name: str, cfg: Any, windings: dict[str, WindingParams],
                  model: MagneticModel, buses: dict) -> None:
        self.name, self.cfg = name, cfg
        names = _validate_windings(windings)
        if names != model.winding_names:
            raise ConfigError(
                f"elements.{name}: winding order {names!r} does not match "
                f"model order {model.winding_names!r}"
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
        self.model = MagComponent(
            model,
            tuple(winding.r for winding in windings.values()),
            tuple((winding.positive == GROUND, winding.negative == GROUND)
                  for winding in windings.values()),
        )

    def subsystems(self) -> dict[str, MagComponent]:
        return {self.name: self.model}

    @staticmethod
    def connections() -> dict:
        return {}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return tuple(
            (bus, port)
            for winding, pair in zip(self.cfg.windings.values(), self.model.winding_ports)
            for bus, port in zip((winding.positive, winding.negative), pair)
            if bus != GROUND
        )

    def signals(self) -> dict[str, float]:
        return {f"{self.name}.{field}": getattr(self.model.output, field)
                for field in type(self.model.output).__slots__}
