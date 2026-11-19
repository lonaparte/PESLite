"""External thermal models connected through the thermalport addon."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Protocol, Sequence

import numpy as np

from ...components.network import Element, register_component_type
from ...solver import Bag, ConfigError
from .thermalnetwork import _finite, _record

__all__ = ["ThermalLaw", "ThermalPort", "ThermalPortModel", "load_thermalport"]

_FORMAT = "peslite-thermalport"
_VERSION = 1
_FIELDS = {
    "format",
    "version",
    "units",
    "heat_names",
    "temperature_names",
    "A_per_s",
    "B_K_per_J",
    "C",
    "D_K_W",
    "ambient_K",
    "initial",
    "signals",
}
_RESERVED_SIGNALS = frozenset(("heat_total_W",))


def _names(values: Any, label: str) -> tuple[str, ...]:
    names = tuple(values) if not isinstance(values, (str, bytes)) else ()
    if not names or any(not isinstance(name, str) or not name.isidentifier() for name in names):
        raise ValueError(f"{label} must contain one or more identifiers")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must contain unique names")
    return names


class ThermalLaw(Protocol):
    """Named heat-input and temperature-output constitutive law."""

    heat_names: tuple[str, ...]
    temperature_names: tuple[str, ...]
    initial_state: Mapping[str, float]

    def rhs(
        self, t: float, state: np.ndarray, heat_W: Mapping[str, float]
    ) -> np.ndarray: ...

    def temperatures_K(
        self, t: float, state: np.ndarray, heat_W: Mapping[str, float]
    ) -> Mapping[str, float]: ...


class ThermalPortModel:
    """Adapt one external thermal law to PESLite states and named signals."""

    outputs_need_inputs: ClassVar[bool] = True
    observer: ClassVar[bool] = True

    def __init__(self, law: ThermalLaw) -> None:
        required = ("rhs", "temperatures_K")
        if any(not callable(getattr(law, name, None)) for name in required):
            raise TypeError("a thermal law must define rhs() and temperatures_K()")
        self.heat_names = _names(getattr(law, "heat_names", ()), "heat_names")
        self.temperature_names = _names(
            getattr(law, "temperature_names", ()), "temperature_names"
        )
        signal_names = tuple(getattr(law, "signal_names", ()))
        if any(not isinstance(name, str) or not name.isidentifier() for name in signal_names):
            raise ValueError("thermal signal names must be identifiers")
        if len(set(signal_names)) != len(signal_names):
            raise ValueError("thermal signal names must be unique")
        if signal_names and not callable(getattr(law, "signals", None)):
            raise TypeError("a thermal law with signal_names must define signals()")
        self.heat_fields = tuple(f"heat_{name}_W" for name in self.heat_names)
        self.temperature_fields = tuple(
            f"temperature_{name}_K" for name in self.temperature_names
        )
        reserved = set(self.heat_fields) | set(self.temperature_fields) | {"heat_total_W"}
        overlap = reserved.intersection(signal_names)
        if overlap:
            raise ValueError(f"thermal signals shadow standard outputs: {sorted(overlap)}")
        self.signal_names = signal_names

        initial = dict(getattr(law, "initial_state", {}))
        if (
            any(not isinstance(name, str) or not name.isidentifier() for name in initial)
            or any(
                isinstance(value, bool) or not math.isfinite(float(value))
                for value in initial.values()
            )
        ):
            raise ValueError("thermal-law initial states must be named and finite")
        self.law = law
        self.state_names = tuple(initial)
        self.state = _record("_ThermalPortState", self.state_names)(
            **{name: float(value) for name, value in initial.items()}
        )
        self.input = _record("_ThermalPortInput", self.heat_fields)(
            **dict.fromkeys(self.heat_fields, 0.0)
        )
        output_names = (
            *self.temperature_fields,
            *self.heat_fields,
            "heat_total_W",
            *self.signal_names,
        )
        self.output_names = output_names
        self.output = _record("_ThermalPortOutput", output_names)()
        self.set_outputs(0.0)

    def _state_vector(self) -> np.ndarray:
        return np.fromiter(
            (getattr(self.state, name) for name in self.state_names),
            dtype=float,
            count=len(self.state_names),
        )

    def _heat(self) -> dict[str, float]:
        heat = {
            name: _finite(getattr(self.input, field), field)
            for name, field in zip(self.heat_names, self.heat_fields)
        }
        if any(value < 0.0 for value in heat.values()):
            raise ValueError("thermalport heat inputs must be nonnegative")
        return heat

    @staticmethod
    def _mapping(
        value: Any, names: tuple[str, ...], label: str, *, positive: bool = False
    ) -> dict[str, float]:
        if not isinstance(value, Mapping) or set(value) != set(names):
            raise ValueError(f"{label} must return exactly {names}")
        result = {name: _finite(value[name], f"{label}.{name}") for name in names}
        if positive and any(number <= 0.0 for number in result.values()):
            raise ValueError(f"{label} must contain positive values")
        return result

    def set_outputs(self, t: float) -> None:
        state = self._state_vector()
        heat = self._heat()
        temperatures = self._mapping(
            self.law.temperatures_K(t, state, heat),
            self.temperature_names,
            "temperatures_K",
            positive=True,
        )
        signals = (
            self._mapping(self.law.signals(t, state, heat), self.signal_names, "signals")
            if self.signal_names
            else {}
        )
        for field, name in zip(self.temperature_fields, self.temperature_names):
            setattr(self.output, field, temperatures[name])
        for field, name in zip(self.heat_fields, self.heat_names):
            setattr(self.output, field, heat[name])
        self.output.heat_total_W = sum(heat.values())
        for name, value in signals.items():
            setattr(self.output, name, value)

    def rhs(self, t: float) -> tuple[float, ...]:
        derivative = np.asarray(
            self.law.rhs(t, self._state_vector(), self._heat()),
            dtype=float,
        )
        if derivative.shape != (len(self.state_names),) or not np.isfinite(derivative).all():
            raise ValueError("thermal law rhs must match its finite state vector")
        return tuple(float(value) for value in derivative)


class _StateSpaceThermalLaw:
    """Stable MIMO continuous-time thermal law in SI units."""

    def __init__(
        self,
        *,
        heat_names: Any,
        temperature_names: Any,
        A_per_s: Any,
        B_K_per_J: Any,
        C: Any,
        D_K_W: Any,
        ambient_K: Any,
        initial: Any,
        signals: Any,
    ) -> None:
        self.heat_names = _names(heat_names, "heat_names")
        self.temperature_names = _names(temperature_names, "temperature_names")
        try:
            A = np.asarray(A_per_s, dtype=float)
            B = np.asarray(B_K_per_J, dtype=float)
            output = np.asarray(C, dtype=float)
            direct = np.asarray(D_K_W, dtype=float)
            ambient = np.asarray(ambient_K, dtype=float)
            initial_values = np.asarray(initial, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("thermalport matrices and initial state must be numeric") from None
        states = A.shape[0] if A.ndim == 2 else 0
        inputs = len(self.heat_names)
        outputs = len(self.temperature_names)
        if (
            states < 1
            or A.shape != (states, states)
            or B.shape != (states, inputs)
            or output.shape != (outputs, states)
            or direct.shape != (outputs, inputs)
            or ambient.shape != (outputs,)
            or initial_values.shape != (states,)
            or not all(
                np.isfinite(array).all()
                for array in (A, B, output, direct, ambient, initial_values)
            )
        ):
            raise ValueError(
                "thermalport needs finite A(N,N), B(N,M), C(P,N), D(P,M), "
                "ambient(P), and initial(N)"
            )
        if np.max(np.real(np.linalg.eigvals(A))) >= 0.0:
            raise ValueError("thermalport A must be strictly stable")
        if np.any(ambient <= 0.0):
            raise ValueError("thermalport ambient temperatures must be positive")

        if not isinstance(signals, Mapping):
            raise ValueError("thermalport signals must be a mapping")
        signal_names = tuple(signals)
        if any(
            not isinstance(name, str)
            or not name.isidentifier()
            or name in _RESERVED_SIGNALS
            for name in signal_names
        ):
            raise ValueError("thermalport signal names must be nonreserved identifiers")
        signal_state = []
        signal_heat = []
        signal_offset = []
        for name, description in signals.items():
            if not isinstance(description, Mapping) or set(description) != {
                "state",
                "heat",
                "offset",
            }:
                raise ValueError(f"thermalport signal {name!r} needs state, heat, and offset")
            try:
                state_gain = np.asarray(description["state"], dtype=float)
                heat_gain = np.asarray(description["heat"], dtype=float)
                offset = float(description["offset"])
            except (TypeError, ValueError):
                raise ValueError(f"thermalport signal {name!r} must be numeric") from None
            if (
                state_gain.shape != (states,)
                or heat_gain.shape != (inputs,)
                or not np.isfinite(state_gain).all()
                or not np.isfinite(heat_gain).all()
                or not math.isfinite(offset)
            ):
                raise ValueError(f"thermalport signal {name!r} has inconsistent dimensions")
            signal_state.append(state_gain)
            signal_heat.append(heat_gain)
            signal_offset.append(offset)

        self.A_per_s = A
        self.B_K_per_J = B
        self.C = output
        self.D_K_W = direct
        self.ambient_K = ambient
        self.signal_names = signal_names
        self._signal_state = np.asarray(signal_state, dtype=float).reshape(-1, states)
        self._signal_heat = np.asarray(signal_heat, dtype=float).reshape(-1, inputs)
        self._signal_offset = np.asarray(signal_offset, dtype=float)
        self.initial_state = {
            f"state_{index}": float(value)
            for index, value in enumerate(initial_values)
        }

    def _heat_vector(self, heat_W: Mapping[str, float]) -> np.ndarray:
        return np.fromiter(
            (heat_W[name] for name in self.heat_names),
            dtype=float,
            count=len(self.heat_names),
        )

    def temperatures_K(
        self, _t: float, state: np.ndarray, heat_W: Mapping[str, float]
    ) -> dict[str, float]:
        value = self.ambient_K + self.C @ state + self.D_K_W @ self._heat_vector(heat_W)
        return dict(zip(self.temperature_names, map(float, value)))

    def rhs(
        self, _t: float, state: np.ndarray, heat_W: Mapping[str, float]
    ) -> np.ndarray:
        return self.A_per_s @ state + self.B_K_per_J @ self._heat_vector(heat_W)

    def signals(
        self, _t: float, state: np.ndarray, heat_W: Mapping[str, float]
    ) -> dict[str, float]:
        value = (
            self._signal_state @ state
            + self._signal_heat @ self._heat_vector(heat_W)
            + self._signal_offset
        )
        return dict(zip(self.signal_names, map(float, value)))


def _load(file: str | Path) -> ThermalLaw:
    path = Path(file)
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigError(f"cannot load thermalport file {path}: {exc}") from None
    if not isinstance(data, dict) or set(data) != _FIELDS:
        raise ConfigError(f"thermalport file must contain exactly {sorted(_FIELDS)}")
    if data["format"] != _FORMAT or data["version"] != _VERSION or data["units"] != "SI":
        raise ConfigError(
            "thermalport file must declare the supported format, version, and SI units"
        )
    try:
        return _StateSpaceThermalLaw(
            heat_names=data["heat_names"],
            temperature_names=data["temperature_names"],
            A_per_s=data["A_per_s"],
            B_K_per_J=data["B_K_per_J"],
            C=data["C"],
            D_K_W=data["D_K_W"],
            ambient_K=data["ambient_K"],
            initial=data["initial"],
            signals=data["signals"],
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from None


def load_thermalport(file: str | Path) -> ThermalPortModel:
    """Load an external thermalport file as a PESLite runtime model."""
    return ThermalPortModel(_load(file))


@register_component_type
class ThermalPort(Element):
    """Connect an external thermal law to configured heat sources."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        sources: dict[str, str] = field(metadata={"structural": True})
        file: str = field(metadata={"asset": True, "runtime": False})
        type: str = "thermalport"

        def __post_init__(self) -> None:
            if (
                not isinstance(self.sources, Mapping)
                or not self.sources
                or any(
                    not isinstance(name, str)
                    or not name.isidentifier()
                    or not isinstance(path, str)
                    or not path
                    for name, path in self.sources.items()
                )
            ):
                raise ConfigError("sources must map identifiers to physical output paths")
            if not isinstance(self.file, str) or not self.file:
                raise ConfigError("file must be a nonempty path")

    type = "thermalport"

    def __init__(
        self,
        name: str,
        cfg: Any,
        _buses: dict,
        _params: Any,
        *,
        law: ThermalLaw | None = None,
    ) -> None:
        self.name, self.cfg = name, cfg
        self.model = ThermalPortModel(_load(cfg.file) if law is None else law)
        if set(cfg.sources) != set(self.model.heat_names):
            raise ConfigError(
                f"thermalport sources {tuple(cfg.sources)!r} do not match "
                f"file inputs {self.model.heat_names!r}"
            )

    def subsystems(self) -> dict[str, ThermalPortModel]:
        return {self.name: self.model}

    def connections(self) -> dict:
        return {
            (self.model, field): self.cfg.sources[name]
            for name, field in zip(self.model.heat_names, self.model.heat_fields)
        }

    @staticmethod
    def port_bindings() -> tuple:
        return ()

    def signals(self) -> dict[str, float]:
        return {
            f"{self.name}.{field}": getattr(self.model.output, field)
            for field in self.model.output_names
        }
