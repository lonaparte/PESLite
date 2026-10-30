"""The registered event types of a simulation file and their time-dependent scenarios.

An event is one entry of the top-level ``events`` mapping: ``{type, t, ...}``. Events which act
on one named object use ``target``. The built-in types are ``connect``, ``disconnect`` and ``set``;
custom types are classes registered with :func:`register_event_type`.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Callable, ClassVar, Iterable, Mapping, Optional

__all__ = [
    "NAMED", "SWITCHABLE", "SWITCHING", "Event", "EVENT_TYPES", "register_event_type",
    "Connect", "Disconnect", "Set", "events_for", "switching_schedule",
    "Scenario", "SourceScenario",
]

NAMED = ("buses", "branches", "sources", "units", "elements")
SWITCHABLE = ("branches", "sources", "units", "elements")
SWITCHING = ("connect", "disconnect")


class Event:
    """Base class of a registered event type.

    A type supplies ``type``, a frozen ``Params`` dataclass containing at least ``type`` and ``t``,
    and ``variables(event, params)`` returning the parameter or connection-command paths to write.
    Bindings are resolved and checked during assembly, not by arbitrary runtime callbacks.
    """

    type: ClassVar[str]
    Params: ClassVar[type]

    @staticmethod
    def variables(event: Any, params: Any) -> Mapping[str, Any]:
        raise NotImplementedError


EVENT_TYPES: dict[str, type] = {}


def register_event_type(cls: type) -> type:
    """Register an :class:`Event` class under ``cls.type`` and return the class."""
    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not name:
        raise TypeError("an event type needs a name: the class attribute 'type'")
    if name in EVENT_TYPES:
        raise ValueError(f"event type {name!r} is already registered")
    params = getattr(cls, "Params", None)
    if not is_dataclass(params) or not {"type", "t"} <= set(params.__dataclass_fields__):
        raise TypeError(f"event type {name!r}: Params must be a dataclass with type and t fields")
    if params.__dataclass_fields__["type"].default != name:
        raise TypeError(f"event type {name!r}: Params.type must default to {name!r}")
    variables = getattr(cls, "variables", None)
    if not callable(variables) or variables is Event.variables:
        raise TypeError(f"event type {name!r}: it needs variables(event, params), declaring its writes")
    if hasattr(cls, "apply"):
        raise TypeError(f"event type {name!r}: use variable bindings, not an apply() callback")
    EVENT_TYPES[name] = cls
    return cls


@register_event_type
class Connect(Event):
    """Connect ``target`` at ``t``; ``ramp`` is the optional ramp duration in seconds."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "connect"
        target: str
        t: float
        ramp: float = 0.0

    type = "connect"

    @staticmethod
    def variables(event: Any, params: Any) -> Mapping[str, Any]:
        section, = (section for section in params.sections_of(event.target) if section in SWITCHABLE)
        return {f"{section}.{event.target}.connected": True}


@register_event_type
class Disconnect(Event):
    """Disconnect ``target`` at ``t``."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "disconnect"
        target: str
        t: float

    type = "disconnect"

    @staticmethod
    def variables(event: Any, params: Any) -> Mapping[str, Any]:
        return {path: False for path in Connect.variables(event, params)}


@register_event_type
class Set(Event):
    """Write checked parameter and connection-command variables at ``t``."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "set"
        t: float
        set: dict = field(default_factory=dict)
        ramp: float = 0.0

    type = "set"

    @staticmethod
    def variables(event: Any, params: Any) -> Mapping[str, Any]:
        return event.set


def events_for(events: Iterable[Any], target: str) -> list:
    """Return checked events writing to a fully scoped target or one of its children."""
    found = [event for event in events if any(path.startswith(target + ".") for path in event.values)]
    return sorted(found, key=lambda event: event.t)


def switching_schedule(events: Iterable[Any]) -> dict[str, list[tuple[float, bool]]]:
    """Return each target's time-ordered ``(time, connected)`` steps."""
    out: dict[str, list[tuple[float, bool]]] = {}
    for event in events:
        for path in event.connections:
            out.setdefault(path.removesuffix(".connected"), []).append((event.t, event.values[path]))
    return {target: sorted(steps) for target, steps in out.items()}


def _initial_connection(steps, initial=None):
    return initial if initial is not None else not steps[0][1] if steps else True


class Scenario:
    """Read-only connection commands for one target over absolute simulation time.

    It does not advance ramps or own component state; the converter unit turns the selected
    command into physical connection and controller start-up actions.
    """

    def __init__(self, events: Iterable[Any] = (), target: Optional[str] = None,
                 *, initial: Optional[bool] = None) -> None:
        events = sorted(events, key=lambda event: event.t)
        paths = {path for event in events for path in event.connections
                 if target is None or path == target + ".connected"}
        if len(paths) > 1:
            raise ValueError("a connection scenario needs one fully scoped target")
        self._commands = tuple((event.t, event.values[path], event.ramp)
                               for event in events for path in event.connections if path in paths)
        self._initial = _initial_connection(self._commands, initial)

    def at(self, t: float) -> tuple[bool, float, float]:
        """Return ``(connected, last_connect_time, ramp)`` in force at ``t``."""
        connected = self._initial
        since, ramp = -math.inf, 0.0
        for event_t, on, requested_ramp in self._commands:
            if event_t > t + 1e-10:
                break
            connected = on
            if on:
                since, ramp = event_t, requested_ramp
        return connected, since, ramp

    def connected(self, t: float) -> bool:
        return self.at(t)[0]

    def since(self, t: float) -> tuple[float, float]:
        """Return the time and ramp of the connection in force at ``t``."""
        return self.at(t)[1:]

class SourceScenario:
    """A source's magnitude, frequency and angle between successive parameter changes."""

    def __init__(self, cfg: Any, f0: float, steps: Iterable[tuple[float, Any]] = ()) -> None:
        self.f0 = f0
        frequency = lambda value: value.f if value.f is not None else f0  # noqa: E731
        self._v_from, self._v = [-math.inf], [cfg.v]
        changes = [(-math.inf, frequency(cfg), cfg.angle)]
        for event_t, value in sorted(steps, key=lambda step: step[0]):
            # A currently redundant assignment can become meaningful after an initial override.
            self._v_from.append(event_t)
            self._v.append(value.v)
            changes.append((event_t, frequency(value), value.angle))

        self._angle_from: list[float] = []
        self._angle: list[tuple[float, float, float]] = []
        phase = 0.0
        for index, (event_t, freq, angle) in enumerate(changes):
            if index:
                previous = changes[index - 1]
                phase += 2.0 * math.pi * (previous[1] - f0) * (event_t - max(previous[0], 0.0))
            self._angle_from.append(event_t)
            self._angle.append((freq, angle, phase))

    def law(self, t: float) -> tuple[float, Optional[Callable[[float], float]]]:
        """Return the magnitude and relative-angle law in force at ``t``."""
        voltage = self._v[bisect_right(self._v_from, t) - 1]
        index = bisect_right(self._angle_from, t) - 1
        frequency, angle, phase = self._angle[index]
        slope = 2.0 * math.pi * (frequency - self.f0)
        start = max(self._angle_from[index], 0.0)
        if slope == 0.0:
            return voltage, None if phase + angle == 0.0 else (lambda _t: phase + angle)
        return voltage, lambda at: phase + slope * (at - start) + angle

    def magnitude(self, t: float) -> float:
        """Return the magnitude in force at ``t`` for the current source component."""
        return self.law(t)[0]

    def angle(self, t: float) -> float:
        """Return the relative source angle at ``t``."""
        _voltage, law = self.law(t)
        return 0.0 if law is None else law(t)
