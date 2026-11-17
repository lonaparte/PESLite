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
from .thermal.network import _finite, _record

__all__ = ["ThermalLaw", "ThermalPort", "ThermalPortModel", "load_thermalport"]

_FORMAT = "peslite-thermalport"
_VERSION = 1
_FIELDS = {
    "format",
    "version",
    "units",
    "A_per_s",
    "B_K_per_J",
    "C",
    "D_K_W",
    "ambient_K",
    "initial",
}


class ThermalLaw(Protocol):
    """Thermal constitutive law evaluated through a heat/temperature boundary."""

    initial_state: Mapping[str, float]

    def rhs(self, t: float, state: np.ndarray, heat_W: float) -> np.ndarray: ...

    def temperature_K(self, t: float, state: np.ndarray, heat_W: float) -> float: ...


class _ThermalPortInput(Bag):
    __slots__ = ("heat_W",)


class _ThermalPortOutput(Bag):
    __slots__ = ("temperature_K", "heat_W")


class ThermalPortModel:
    """Adapt one external thermal law to PESLite states and signals."""

    outputs_need_inputs: ClassVar[bool] = True
    observer: ClassVar[bool] = True

    def __init__(self, law: ThermalLaw) -> None:
        if not callable(getattr(law, "rhs", None)) or not callable(
            getattr(law, "temperature_K", None)
        ):
            raise TypeError("a thermal law must define rhs() and temperature_K()")
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
        self.input = _ThermalPortInput(heat_W=0.0)
        self.output = _ThermalPortOutput()
        self.set_outputs(0.0)

    def _state_vector(self) -> np.ndarray:
        return np.fromiter(
            (getattr(self.state, name) for name in self.state_names),
            dtype=float,
            count=len(self.state_names),
        )

    def _heat(self) -> float:
        heat = _finite(self.input.heat_W, "heat_W")
        if heat < 0.0:
            raise ValueError("thermalport heat must be nonnegative")
        return heat

    def set_outputs(self, t: float) -> None:
        heat = self._heat()
        temperature = _finite(
            self.law.temperature_K(t, self._state_vector(), heat),
            "temperature_K",
        )
        if temperature <= 0.0:
            raise ValueError("thermalport temperature must be positive")
        self.output.temperature_K = temperature
        self.output.heat_W = heat

    def rhs(self, t: float) -> tuple[float, ...]:
        derivative = np.asarray(
            self.law.rhs(t, self._state_vector(), self._heat()),
            dtype=float,
        )
        if derivative.shape != (len(self.state_names),) or not np.isfinite(derivative).all():
            raise ValueError("thermal law rhs must match its finite state vector")
        return tuple(float(value) for value in derivative)


class _StateSpaceThermalLaw:
    """Stable continuous-time state-space thermal law in SI units."""

    def __init__(
        self,
        A_per_s: Any,
        B_K_per_J: Any,
        C: Any,
        D_K_W: float,
        *,
        ambient_K: float,
        initial: Sequence[float],
    ) -> None:
        try:
            A = np.asarray(A_per_s, dtype=float)
            B = np.asarray(B_K_per_J, dtype=float)
            output = np.asarray(C, dtype=float)
            initial_values = np.asarray(initial, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("thermalport matrices and initial state must be numeric") from None
        if (
            A.ndim != 2
            or A.shape[0] < 1
            or A.shape[0] != A.shape[1]
            or B.shape != (A.shape[0],)
            or output.shape != (A.shape[0],)
            or initial_values.shape != (A.shape[0],)
            or not all(np.isfinite(array).all() for array in (A, B, output, initial_values))
        ):
            raise ValueError("thermalport needs finite A(N,N), B(N), C(N), and initial(N)")
        if np.max(np.real(np.linalg.eigvals(A))) >= 0.0:
            raise ValueError("thermalport A must be strictly stable")
        if np.any(B < 0.0) or np.any(output < 0.0):
            raise ValueError("thermalport thermal gain coefficients must be nonnegative")
        direct = _finite(D_K_W, "D_K_W")
        ambient = _finite(ambient_K, "ambient_K")
        if direct < 0.0 or ambient <= 0.0:
            raise ValueError("thermalport direct gain and ambient must be physical")
        self.A_per_s = A
        self.B_K_per_J = B
        self.C = output
        self.D_K_W = direct
        self.ambient_K = ambient
        self.initial_state = {
            f"state_{index}": float(value)
            for index, value in enumerate(initial_values)
        }

    def temperature_K(self, _t: float, state: np.ndarray, heat_W: float) -> float:
        return self.ambient_K + float(self.C @ state) + self.D_K_W * heat_W

    def rhs(self, _t: float, state: np.ndarray, heat_W: float) -> np.ndarray:
        return self.A_per_s @ state + self.B_K_per_J * heat_W


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
            data["A_per_s"],
            data["B_K_per_J"],
            data["C"],
            data["D_K_W"],
            ambient_K=data["ambient_K"],
            initial=data["initial"],
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from None


def load_thermalport(file: str | Path) -> ThermalPortModel:
    """Load an external thermalport file as a PESLite runtime model."""
    return ThermalPortModel(_load(file))


@register_component_type
class ThermalPort(Element):
    """Connect an external thermal law to one configured heat source."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        heat_source: str = field(metadata={"runtime": False})
        file: str = field(metadata={"asset": True, "runtime": False})
        type: str = "thermalport"

        def __post_init__(self) -> None:
            if not isinstance(self.heat_source, str) or not self.heat_source:
                raise ConfigError("heat_source must name an existing physical output")
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

    def subsystems(self) -> dict[str, ThermalPortModel]:
        return {self.name: self.model}

    def connections(self) -> dict:
        return {(self.model, "heat_W"): self.cfg.heat_source}

    @staticmethod
    def port_bindings() -> tuple:
        return ()

    def signals(self) -> dict[str, float]:
        return {
            f"{self.name}.temperature_K": self.model.output.temperature_K,
            f"{self.name}.heat_W": self.model.output.heat_W,
        }
