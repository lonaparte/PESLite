"""Semiconductor heat accounting for the thermal addon.

Estimated losses are not an electrical energy sink. Switching energies must come from
explicitly supplied, *committed* transitions, never from sampled gate traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = [
    "LossTable", "SemiconductorLossModel", "SwitchingEvent", "EventLoss",
    "DeviceLossReport", "device_loss_report", "recorded_device_loss",
    "compare_devices", "circuit_heat_report",
]

_AXES = frozenset(("current_A", "voltage_V", "temperature_K"))
_EVENT_KINDS = frozenset(("turn_on", "turn_off", "reverse_recovery"))


def _values(name: str, values: Any, count: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
    if count is not None and result.size != count:
        raise ValueError(f"{name} must have {count} samples")
    return result


def _scalar(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be {'nonnegative and ' if nonnegative else ''}finite")
    return result


@dataclass(frozen=True)
class LossTable:
    """Nonnegative loss data on named, strictly increasing SI axes.

    Axes may be any nonempty subset of ``current_A``, ``voltage_V`` and
    ``temperature_K``. Multilinear interpolation is used inside the grid.
    Outside it, the default ``out_of_range='error'`` rejects the lookup;
    ``'clamp'`` uses the nearest boundary rather than extrapolating.
    Current and voltage are treated as magnitudes.
    """

    axes: Mapping[str, Sequence[float]]
    values: Any
    out_of_range: str = "error"
    _names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _interpolator: Any = field(init=False, repr=False, compare=False)
    _grid: tuple[np.ndarray, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.axes or any(name not in _AXES for name in self.axes):
            raise ValueError("loss table needs named current_A, voltage_V or temperature_K axes")
        if self.out_of_range not in ("error", "clamp"):
            raise ValueError("out_of_range must be 'error' or 'clamp'")
        grid = []
        for name, points in self.axes.items():
            axis = _values(f"axis {name}", points)
            if axis.size < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError(f"axis {name} needs at least two strictly increasing points")
            if (np.any(axis < 0) if name != "temperature_K" else np.any(axis <= 0)):
                raise ValueError(f"axis {name} must use physical nonnegative SI values")
            axis = axis.copy()
            axis.setflags(write=False)
            grid.append(axis)
        try:
            values = np.array(self.values, dtype=float, copy=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("loss table values must be finite and nonnegative") from exc
        if values.shape != tuple(len(axis) for axis in grid):
            raise ValueError("loss table values must match the axis grid shape")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("loss table values must be finite and nonnegative")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "_names", tuple(self.axes))
        object.__setattr__(self, "_grid", tuple(grid))
        object.__setattr__(self, "_interpolator", RegularGridInterpolator(
            grid, values, method="linear", bounds_error=self.out_of_range == "error",
            fill_value=None,
        ))

    def evaluate(self, *, current_A: Any = 0.0, voltage_V: Any = 0.0,
                 temperature_K: Any = 298.15) -> float | np.ndarray:
        """Evaluate one or broadcasted lookup points using SI quantities."""
        sources = {"current_A": current_A, "voltage_V": voltage_V,
                   "temperature_K": temperature_K}
        try:
            columns = np.broadcast_arrays(*(
                np.asarray(sources[name], dtype=float) for name in self._names
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError("loss table coordinates must be broadcastable numeric arrays") from exc
        coordinates = []
        for name, column, grid in zip(self._names, columns, self._grid):
            if not np.isfinite(column).all():
                raise ValueError(f"{name} lookup coordinates must be finite")
            if name == "temperature_K" and np.any(column <= 0):
                raise ValueError("temperature_K must be positive")
            magnitude = column if name == "temperature_K" else np.abs(column)
            if self.out_of_range == "clamp":
                magnitude = np.clip(magnitude, grid[0], grid[-1])
            coordinates.append(magnitude.ravel())
        points = np.stack(coordinates, axis=-1)
        try:
            lookup = self._interpolator(points).reshape(columns[0].shape)
        except ValueError as exc:
            raise ValueError("loss table lookup is outside the declared axes") from exc
        return float(lookup) if lookup.ndim == 0 else lookup


@dataclass(frozen=True)
class SemiconductorLossModel:
    """Optional conduction voltage (V) and transition energies (J) for one device.

    Missing fields mean that loss mechanism is unavailable, not that it is known
    to be zero. The caller supplies known conduction intervals and transitions.
    """

    conduction_voltage_V: float | LossTable | None = None
    turn_on_energy_J: float | LossTable | None = None
    turn_off_energy_J: float | LossTable | None = None
    reverse_recovery_energy_J: float | LossTable | None = None
    turn_on_includes_reverse_recovery: bool = False

    def __post_init__(self) -> None:
        for name in ("conduction_voltage_V", "turn_on_energy_J", "turn_off_energy_J",
                     "reverse_recovery_energy_J"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, LossTable):
                object.__setattr__(self, name, _scalar(name, value, nonnegative=True))
        if type(self.turn_on_includes_reverse_recovery) is not bool:
            raise ValueError("turn_on_includes_reverse_recovery must be bool")


def _lookup(value: float | LossTable | None, current: Any, voltage: Any,
            temperature: float) -> Any:
    return (value.evaluate(current_A=current, voltage_V=voltage,
                           temperature_K=temperature)
            if isinstance(value, LossTable) else 0.0 if value is None else value)


@dataclass(frozen=True)
class SwitchingEvent:
    """One confirmed device transition, with pre-transition absolute operating values.

    These events must originate from accepted/committed simulation transitions or
    an independent verified event log. Snapshot-detected edges are not sufficient.
    """

    time_s: float
    kind: str
    current_A: float
    voltage_V: float

    def __post_init__(self) -> None:
        if self.kind not in _EVENT_KINDS:
            raise ValueError(f"unknown switching event kind: {self.kind!r}")
        for name in ("time_s", "current_A", "voltage_V"):
            object.__setattr__(self, name, _scalar(name, getattr(self, name)))


@dataclass(frozen=True)
class EventLoss:
    event: SwitchingEvent
    energy_J: float


@dataclass(frozen=True)
class DeviceLossReport:
    """Recorded continuous loss powers and distinct transition energies."""

    time_s: np.ndarray
    accounted_conduction_W: np.ndarray
    estimated_conduction_W: np.ndarray
    event_losses: tuple[EventLoss, ...]

    @property
    def instantaneous_conduction_W(self) -> np.ndarray:
        return self.accounted_conduction_W + self.estimated_conduction_W

    def to_dict(self) -> dict[str, float | int]:
        """Unit-labelled, JSON-compatible heat quantities for one device."""
        elapsed = float(self.time_s[-1] - self.time_s[0])
        dt = np.diff(self.time_s)

        def integrate(power: np.ndarray) -> float:
            return float(np.sum(0.5 * dt * (power[1:] + power[:-1])))

        accounted = integrate(self.accounted_conduction_W)
        conduction = integrate(self.estimated_conduction_W)
        switching = {kind: sum(event.energy_J for event in self.event_losses
                               if event.event.kind == kind) for kind in _EVENT_KINDS}
        additional = conduction + sum(switching.values())
        return {
            "duration_s": elapsed,
            "accounted_conduction_J": accounted,
            "estimated_conduction_J": conduction,
            "turn_on_J": switching["turn_on"],
            "turn_off_J": switching["turn_off"],
            "reverse_recovery_J": switching["reverse_recovery"],
            "additional_estimated_J": additional,
            "attributed_heat_J": accounted + additional,
            "accounted_conduction_W": accounted / elapsed,
            "estimated_conduction_W": conduction / elapsed,
            "turn_on_W": switching["turn_on"] / elapsed,
            "turn_off_W": switching["turn_off"] / elapsed,
            "reverse_recovery_W": switching["reverse_recovery"] / elapsed,
            "additional_estimated_W": additional / elapsed,
            "attributed_heat_W": (accounted + additional) / elapsed,
            "event_count": len(self.event_losses),
        }


def device_loss_report(
    time_s: Sequence[float], current_A: Sequence[float], model: SemiconductorLossModel,
    *, voltage_V: Sequence[float] | float = 0.0,
    temperature_K: float = 298.15,
    conduction_fraction: Sequence[float] | float = 1.0,
    accounted_conduction_power_W: Sequence[float] | None = None,
    events: Sequence[SwitchingEvent] = (),
) -> DeviceLossReport:
    """Estimate heat using a trace and independently confirmed device events.

    ``accounted_conduction_power_W`` denotes conduction loss already dissipated
    by the electrical model; it is mutually exclusive with a voltage-drop lookup.
    Transition energies remain *estimated* and never alter the solver energy.
    Event losses are impulses (J), not artificial spikes in sampled power (W).
    """
    if not isinstance(model, SemiconductorLossModel):
        raise TypeError("model must be a SemiconductorLossModel")
    t = _values("time_s", time_s).copy()
    if t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("time_s requires at least two strictly increasing samples")
    current = _values("current_A", current_A, len(t))
    if np.isscalar(voltage_V):
        voltage = np.full_like(t, _scalar("voltage_V", voltage_V))
    else:
        voltage = _values("voltage_V", voltage_V, len(t))
    temperature = _scalar("temperature_K", temperature_K)
    if temperature <= 0:
        raise ValueError("temperature_K must be positive")
    if np.isscalar(conduction_fraction):
        fraction = np.full_like(t, _scalar("conduction_fraction", conduction_fraction))
    else:
        fraction = _values("conduction_fraction", conduction_fraction, len(t))
    if np.any((fraction < 0) | (fraction > 1)):
        raise ValueError("conduction_fraction must lie between 0 and 1")
    if accounted_conduction_power_W is None:
        accounted = np.zeros_like(t)
    else:
        if model.conduction_voltage_V is not None:
            raise ValueError("a conduction lookup and electrically accounted conduction overlap")
        accounted = _values("accounted_conduction_power_W", accounted_conduction_power_W,
                            len(t)).copy()
        if np.any(accounted < 0):
            raise ValueError("accounted conduction power must be nonnegative")
    estimated = (fraction * np.abs(current) * _lookup(
        model.conduction_voltage_V, current, voltage, temperature
    ))
    estimated = np.broadcast_to(np.asarray(estimated, dtype=float), t.shape).copy()
    event_losses: list[EventLoss] = []
    seen: set[tuple[str, float]] = set()
    curves = {"turn_on": model.turn_on_energy_J, "turn_off": model.turn_off_energy_J,
              "reverse_recovery": model.reverse_recovery_energy_J}
    for event in events:
        if not isinstance(event, SwitchingEvent):
            raise TypeError("events must be SwitchingEvent instances")
        if not t[0] <= event.time_s <= t[-1]:
            raise ValueError("switching events must lie within the recorded trace")
        if (event.kind, event.time_s) in seen:
            raise ValueError("duplicate switching event would double-count its energy")
        seen.add((event.kind, event.time_s))
        if event.kind == "reverse_recovery" and model.turn_on_includes_reverse_recovery:
            raise ValueError("reverse-recovery energy is already included in turn-on data")
        curve = curves[event.kind]
        if curve is None:
            raise ValueError(f"no {event.kind} energy data for this device")
        energy = _lookup(curve, event.current_A, event.voltage_V, temperature)
        event_losses.append(EventLoss(event, float(energy)))
    for array in (t, accounted, estimated):
        array.setflags(write=False)
    return DeviceLossReport(t, accounted, estimated, tuple(event_losses))


def recorded_device_loss(
    result: Any, current_column: str, model: SemiconductorLossModel, *,
    voltage_column: str | None = None,
    conduction_fraction_column: str | None = None,
    temperature_K: float = 298.15,
    events: Sequence[SwitchingEvent] = (),
) -> DeviceLossReport:
    """Read named circuit columns from an existing streamed ``SimulationResult``.

    Requires ``simulation.output.signals: 1``. The caller must select an actual
    *device* current rather than assuming the bridge or line current equals it.
    Neither CSV samples nor controller signals are used to infer switch events.
    """
    plant = getattr(result, "plant", None)
    if not isinstance(plant, Mapping):
        raise TypeError("result must expose PESLite plant columns")

    def column(name: str) -> np.ndarray:
        if name not in plant:
            raise ValueError(f"plant column {name!r} is unavailable; enable output.signals")
        return plant[name]

    return device_loss_report(
        result.t, column(current_column), model,
        voltage_V=column(voltage_column) if voltage_column else 0.0,
        conduction_fraction=(column(conduction_fraction_column)
                             if conduction_fraction_column else 1.0),
        temperature_K=temperature_K, events=events,
    )


def compare_devices(
    time_s: Sequence[float], current_A: Sequence[float],
    candidates: Mapping[str, SemiconductorLossModel], **conditions: Any,
) -> dict[str, DeviceLossReport]:
    """Apply identical observed conditions to alternative device loss models."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    return {name: device_loss_report(time_s, current_A, model, **conditions)
            for name, model in candidates.items()}


def circuit_heat_report(
    devices: Mapping[str, DeviceLossReport], *,
    energy_closure: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Combine device reports without double-counting electrically modeled losses.

    When ``energy_closure`` is supplied, circuit-wide realized dissipation is
    reported separately from *additional estimates*: their sum is deliberately
    not claimed to be a proven physical total, since loss mechanisms can overlap.
    """
    if not devices:
        raise ValueError("devices must not be empty")
    values = list(devices.values())
    reference = values[0].time_s
    if any(not np.array_equal(item.time_s, reference) for item in values[1:]):
        raise ValueError("device reports must share the same time samples")
    reports = {name: report.to_dict() for name, report in devices.items()}
    accounted = sum(report["accounted_conduction_J"] for report in reports.values())
    additional = sum(report["additional_estimated_J"] for report in reports.values())
    duration = float(reference[-1] - reference[0])
    total = {
        "accounted_device_conduction_J": accounted,
        "additional_estimated_device_J": additional,
        "attributed_device_heat_J": accounted + additional,
        "attributed_device_heat_W": (accounted + additional) / duration,
    }
    if energy_closure is not None:
        try:
            dissipated = _scalar("energy_closure.dissipated", energy_closure["dissipated"],
                                 nonnegative=True)
            impulsive = _scalar("energy_closure.impulsive", energy_closure["impulsive"],
                                nonnegative=True)
        except KeyError as exc:
            raise ValueError("energy_closure must include dissipated and impulsive") from exc
        total["circuit_electrical_dissipation_J"] = dissipated + impulsive
        total["circuit_additional_estimate_J"] = additional
    return {"devices": reports, "totals": total}
