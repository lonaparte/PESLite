"""The PWM peripheral of a converter: timer, duty registers, carrier and modulators.

A modulator is called at every load of the compare registers for the interval until the next load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Callable, Iterator, Mapping, Optional, Protocol, NamedTuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..control.profiles import PHASE_SHIFT_MODES, control_type
from ..solver.model import ConfigError, _integer, _real
from ..solver.timing import TIME_EPS, PeriodicClock

__all__ = ["PWM", "PhaseClock", "PulseSequence", "SwitchingSequence",
           "Modulator", "CarrierComparison",
           "SynchronousCarrier", "ZOH", "PhaseShiftModulator",
           "phase_shift_level", "carrier", "carrier_position", "MODULATOR_TYPES",
           "make_modulator"]


class PhaseClock(NamedTuple):
    """An affine view of an existing clock, with a stable epoch between clock updates."""
    time: float
    angle: float
    rate: float

    def at(self, t):
        return self.angle + self.rate * (t - self.time)


@dataclass
class SwitchingSequence:
    """Piecewise-constant switching pattern from one compare-register load to the next.

    ``dt``: interval lengths (s) summing to the interval between compare-register loads.
    ``q_abc[i]``: phase states in interval ``i`` (0/1 when switching, fractional when averaged).
    """

    dt: NDArray[np.float64]
    q_abc: NDArray[np.float64]


def _switching_sequence(span, min_interval, word, changes) -> SwitchingSequence:
    """Apply time-ordered ``(time, positions, values)`` changes to ``word`` over ``span``.

    An interval not longer than ``min_interval`` is absorbed into the next one; a vanishing last
    interval into the previous one.
    """
    durations, states, previous = [], [], 0.0
    for at, positions, values in changes:
        if at - previous > min_interval:
            durations.append(at - previous)
            states.append(word.copy())
            previous = at
        for k, value in zip(positions, values):
            word[k] = value
    if span - previous > min_interval or not durations:
        durations.append(span - previous)
        states.append(word.copy())
    else:
        durations[-1] += span - previous
    return SwitchingSequence(np.asarray(durations), np.asarray(states))


@runtime_checkable
class Modulator(Protocol):
    """Convert loaded commands into a SwitchingSequence spanning ``T`` seconds.

    Optional ``theta`` (rad) and ``omega`` (rad/s) set synchronous carrier timing;
    asynchronous modulators ignore them.
    Complete-pulse updates also require a periodic ``phase_clock`` view accepting prospective
    ``registers``; the ordinary interval path needs no additional interface.
    """

    carried: tuple[str, ...] = ()  # controller outputs passed after the modulated registers
    config_fields = frozenset(("carrier_phase", "sync"))  # the pwm options the modulator reads

    def __call__(self, t: float, T: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence: ...


class PulseSequence:
    """Preserve complete pulses in a periodic switching sequence.

    Groups identify atomic output pulses: one pole per group for carrier comparison,
    both poles together for a phase-shift bridge. Only the accepted pulse value and its
    remaining phase are retained. Register values and waveform equations stay with the modulator.
    """

    def __init__(self, modulator, width: int):
        if not callable(getattr(modulator, "phase_clock", None)):
            raise TypeError("complete-pulse updates require a periodic modulator phase clock")
        self.modulator = modulator
        self.groups = getattr(modulator, "pulse_groups", tuple((k,) for k in range(width)))
        self.entry = self.next = (0.0,) * width
        self.remaining = self.next_remaining = (0.0,) * len(self.groups)
        self.time = None
        self.restored = False

    @property
    def discrete_state_names(self):
        return (*getattr(self.modulator, "discrete_state_names", ()),
                *(f"pulse.q_{k}" for k in range(len(self.entry))))

    @property
    def angle_state_names(self):
        return getattr(self.modulator, "angle_state_names", ())

    def get_state(self):
        read = getattr(self.modulator, "get_state", None)
        state = dict(read()) if read is not None else {}
        state.update((f"pulse.q_{k}", _real(value)) for k, value in enumerate(self.entry))
        state.update((f"pulse.remaining_{k}", _real(value))
                     for k, value in enumerate(self.remaining))
        return state

    def set_state(self, values):
        unknown = set(values) - self.get_state().keys()
        if unknown:
            raise KeyError(f"pulse update: unknown state(s) {sorted(unknown)}")
        self.entry = tuple(float(values.get(f"pulse.q_{k}", value))
                           for k, value in enumerate(self.entry))
        self.remaining = tuple(float(values.get(f"pulse.remaining_{k}", value))
                               for k, value in enumerate(self.remaining))
        other = {key: value for key, value in values.items() if not key.startswith("pulse.")}
        if other:
            self.modulator.set_state(other)
        self.time, self.restored = None, True

    @staticmethod
    def _at(parts, time):
        value = parts[0][1]
        for at, after in parts[1:]:
            if at > time:
                break
            value = after
        return value

    def __call__(self, t, span, values, theta=None, omega=None, *, enabled=True):
        clock = self.modulator.phase_clock(t, theta, omega, registers=values)
        period = 2.0 * math.pi / clock.rate
        raw = self.modulator(t, span + period, values, theta, omega)
        if self.time != t:
            if not self.restored:
                self.entry, self.remaining = self.next, self.next_remaining
            self.time, self.restored = t, False
        if not enabled:
            self.next = (0.0,) * len(self.entry)
            self.next_remaining = (0.0,) * len(self.groups)
            return SwitchingSequence(np.asarray([span]), np.asarray([self.next]))
        times = np.r_[0.0, np.cumsum(raw.dt)[:-1]]
        snap = _SNAP * period
        initial = [0.0] * len(self.entry)
        next_word, remaining, changes = initial.copy(), [], []
        for index, group in enumerate(self.groups):
            zero = (0.0,) * len(group)
            old = tuple(self.entry[k] for k in group)
            free = max(0.0, self.remaining[index]) * period if any(old) else 0.0
            level = old if free > snap else zero
            for k, value in zip(group, level):
                initial[k] = value
            if free >= span - snap:
                after = old if free > span + snap else zero
                left = max(0.0, (free - span) / period)
            else:
                if free > snap:
                    changes.append((free, index, zero))
                parts = []
                for at, word in zip(times, raw.q_abc):
                    value = tuple(float(word[k]) for k in group)
                    if not parts or parts[-1][1] != value:
                        parts.append((float(at), value))
                if len(parts) == 1:
                    begin, value = free, parts[0][1]
                else:
                    starts = [(at, value) for at, value in parts[1:]
                              if any(value) and at >= free - snap]
                    if (free <= snap and any(parts[0][1])
                            and parts[0][1] != self._at(parts, period - snap)):
                        starts.insert(0, (0.0, parts[0][1]))
                    begin, value = starts[0] if starts else (math.inf, zero)
                    begin = max(free, begin)
                after, left = zero, 0.0
                if begin < span - snap:
                    if begin <= snap:
                        for k, value_k in zip(group, value):
                            initial[k] = value_k
                    else:
                        changes.append((begin, index, value))
                    changes.extend((at, index, value) for at, value in parts[1:]
                                   if begin + snap < at < span - snap)
                    after = self._at(parts, span + snap)
                    if any(after):
                        end = next((at for at, _ in parts[1:] if at > span + snap), None)
                        left = (end - span) / period if end is not None else 0.0
            for k, value in zip(group, after):
                next_word[k] = value
            remaining.append(left)
        self.next, self.next_remaining = tuple(next_word), tuple(remaining)
        return _switching_sequence(span, self.modulator.min_interval, initial, (
            (at, self.groups[index], value) for at, index, value in sorted(changes)))


class PWM:
    """Carrier timer, compare registers and switching schedule.

    Interrupts occur every ``period`` from ``offset``. Compare registers load every
    ``load_period``. An interrupt writes commands and the PWM enable to the shadow registers;
    a load copies them to the active registers only after ``computation`` seconds. Until then
    the shadow still holds the replaced word, which a load takes instead. The enable is
    carried through exactly the same register path as the commands.
    """

    def __init__(self, period: float, load_period: float, control_offset: float,
                 load_offset: float, computation: float,
                 carrier_period: float, modulator: Modulator, *,
                 pattern_sync: str = "interval",
                 duty_names: tuple[str, ...],
                 switching_state: Callable[[Any], Any],
                 pwm_state: Callable[[Any], Any]) -> None:
        if not math.isfinite(computation) or computation < 0.0:
            raise ValueError(f"the computation time must be finite and >= 0, got {computation}")
        self.control_clock = PeriodicClock(_real(period), _real(control_offset))
        self.load_clock = PeriodicClock(_real(load_period), _real(load_offset))
        self.period, self.load_period = self.control_clock.period, self.load_clock.period
        self.computation, self.carrier_period = _real(computation), _real(carrier_period)
        self._names = (*duty_names, "on")
        if any(not name or "." in name for name in self._names):
            raise ValueError("PWM register names must be nonempty path segments")
        if len(set(self._names)) != len(self._names):
            raise ValueError("PWM register names must be unique")
        self._width = len(self._names) - 1
        self.modulation_width = self._width - len(modulator.carried)
        self._switching_state, self._pwm_state = switching_state, pwm_state
        if pattern_sync not in ("interval", "pulse"):
            raise ValueError(f"pattern_sync must be 'interval' or 'pulse', got {pattern_sync!r}")
        self.pulse_update = (PulseSequence(modulator, getattr(modulator, "switching_legs", self.modulation_width))
                             if pattern_sync == "pulse" and not isinstance(modulator, ZOH) else None)
        self.modulator = modulator
        registers = {*self._names, *(f"shadow.{name}" for name in self._names)}
        if overlap := registers & self._modulator_state().keys():
            raise ValueError(f"PWM registers conflict with modulator states: {sorted(overlap)}")
        self.k = self.j = 0
        self.t_interrupt, self.t_load = self.interrupt(0), self.load_time(0)
        self.active = self.shadow = (0.0,) * (self._width + 1)
        self.completed = self.shadow
        self.sync: tuple[float | None, float | None] = (None, None)
        self.t_sync = 0.0
        self.schedule: list[tuple[float, Any]] = []
        self.next_switch = math.inf
        self.load_end = math.inf

    @property
    def register_names(self) -> tuple[str, ...]:
        return self._names[:-1]

    def phase_clock(self) -> PhaseClock:
        read = getattr(self.modulator, "phase_clock", None)
        if read is not None:
            return read(self.t_sync, *self.sync)
        return PhaseClock(0.0, 0.0, 2.0 * math.pi / self.carrier_period)

    def phase_reference(self) -> float:
        name = getattr(self.modulator, "phase_register", None)
        return self.register(name) if name is not None else 0.0

    @property
    def modulator(self) -> Modulator:
        return self._modulator

    @modulator.setter
    def modulator(self, value: Modulator) -> None:
        self._modulator = value
        if self.pulse_update is not None:
            self.pulse_update.modulator = value
        self._sequence = self._pulse_sequence if self.pulse_update is not None else value
        self._switching = (
            self._pwm_interval if isinstance(value, ZOH) else self._switching_interval
        )

    def interrupt(self, k: int) -> float:
        return self.control_clock.at(k)

    def load_time(self, j: int) -> float:
        return self.load_clock.at(j)

    def after(self, t: float) -> int:
        """Index of the first control interrupt after ``t``."""
        return int(self.control_clock.index_after(t))

    def control_count(self, t_a: float, t_b: float) -> int:
        """Number of control interrupts in ``[t_a, t_b]``, including both ends."""
        return max(0, self.after(t_b) - int(self.control_clock.index_at_or_after(t_a)))

    def describe(self) -> str:
        double = self.load_period < 0.75 * self.carrier_period
        return (f"control {1e-3 / self.period:g} kHz, computation {self.computation * 1e6:g} us, "
                f"{'double' if double else 'single'} update")

    def _word(self, registers, on: bool) -> tuple[float, ...]:
        # Registers are scalar values, not a numerical vector operation. Preserve each
        # scalar's arithmetic semantics so ordinary construction also binds export parameters.
        values = registers.tolist() if isinstance(registers, np.ndarray) else list(registers)
        if len(values) != self._width:
            raise ValueError(f"expected {self._width} command registers, got {len(values)}")
        values.append(1.0 if on else 0.0)
        return tuple(values)

    def reset(self, d_abc: NDArray[np.float64], on: bool = False) -> None:
        """Put commands and enable in both register banks; the default is blocked."""
        word = self._word(d_abc, on)
        self.active = self.shadow = self.completed = word

    def tick(self, t: float, theta: float | None = None, omega: float | None = None) -> None:
        """Advance the control-interrupt timer and its synchronous-carrier reference."""
        self.sync, self.t_sync = (theta, omega), t
        self.k += 1
        self.t_interrupt = self.interrupt(self.k)

    def write(self, t: float, d_abc: NDArray[np.float64], on: bool = True,
              theta: float | None = None,
              omega: float | None = None) -> None:
        """Advance an interrupt and write its computed commands and enable to shadow."""
        self.tick(t, theta, omega)
        # computation < period: the replaced word's computation has completed.
        self.completed, self.shadow = self.shadow, self._word(d_abc, on)

    @property
    def duty(self) -> NDArray[np.float64]:
        """Duty ratios in the active compare registers."""
        return np.asarray(self.active[:-1], dtype=float)

    @property
    def on(self) -> bool:
        """PWM enable in the active compare registers."""
        return bool(self.active[-1])

    def start(self, t: float, sync: tuple[float | None, float | None], continued: bool):
        """Resume the timer and switching sequence at ``t`` after named states were loaded."""
        self.k = int(self.control_clock.index_after(t))
        self.j = int(self.load_clock.index_after(t))
        self.t_interrupt, self.t_load = self.interrupt(self.k), self.load_time(self.j)
        # The word the latest interrupt replaced, once a load has followed its completion.
        self.completed = self.active
        last = self.interrupt(self.k - 1)
        self.sync, self.t_sync = sync, min(last, t)
        q = self._switching(self.load_time(self.j - 1), self.active[:-1], t)
        if not continued:
            self.t_sync = t
        return q

    def load(self, t: float):
        """Load shadow into active registers and schedule switching until the next load."""
        completed = self.interrupt(self.k - 1) + self.computation
        # A load within the latest computation takes the completed word it replaces.
        self.active = self.shadow if completed <= t + TIME_EPS else self.completed
        q = self._switching(t, self.active[:-1], t)
        self.j += 1
        self.t_load = self.load_time(self.j)
        return q

    def _pwm_interval(self, t_l: float, d: NDArray[np.float64], _t_now: float):
        self.load_end = t_l + self.load_period
        self.schedule.clear()
        self.next_switch = math.inf
        values = np.asarray(d[:self.modulation_width], dtype=float)
        return self._pwm_state(values)

    def _switching_interval(self, t_l: float, d: NDArray[np.float64], t_now: float):
        theta, omega = self.sync
        if theta is not None and omega is not None and t_l != self.t_sync:
            theta = theta + omega * (t_l - self.t_sync)
        values = np.asarray(d[:self.modulation_width], dtype=float)
        seq = self._sequence(t_l, self.load_period, values, theta, omega)
        dt = np.asarray(seq.dt, dtype=float).tolist()
        states = np.asarray(seq.q_abc, dtype=float).tolist()
        self.load_end = t_l + self.load_period
        schedule: list[tuple[float, Any]] = []
        encode = self._switching_state
        q_now, t_i = encode(states[0]), t_l
        for i in range(len(dt) - 1):
            t_i += dt[i]
            q = encode(states[i + 1])
            if t_i <= t_now + TIME_EPS:
                q_now = q
            else:
                schedule.append((t_i, q))
        self.schedule = schedule
        self.next_switch = schedule[0][0] if schedule else math.inf
        return q_now

    def _pulse_sequence(self, t, span, values, theta, omega):
        return self.pulse_update(t, span, values, theta, omega, enabled=self.on)

    def fast_edge(self, t: float) -> bool:
        """Whether the old switching or held-input interval ends at ``t``."""
        return abs(self.next_switch - t) < TIME_EPS or abs(self.load_end - t) < TIME_EPS

    def switches(self, t: float) -> list[Any]:
        """Return switching states due at ``t`` in order."""
        due = []
        while self.schedule and abs(self.schedule[0][0] - t) < TIME_EPS:
            due.append(self.schedule.pop(0)[1])
        self.next_switch = self.schedule[0][0] if self.schedule else math.inf
        return due

    def register(self, name: str) -> float:
        """Active value of one compare register."""
        return float(self.active[self._names.index(name)])

    def _modulator_state(self) -> dict[str, Any]:
        read = getattr(self.pulse_update or self._modulator, "get_state", None)
        return dict(read()) if read is not None else {}

    def _declared_registers(self, kind: str) -> tuple[str, ...]:
        """Register and modulator states a modulator declares as ``kind`` (discrete, angle)."""
        registers = getattr(self._modulator, f"{kind}_registers", ())
        own = getattr(self.pulse_update or self._modulator, f"{kind}_state_names", ())
        return (*registers, *(f"shadow.{name}" for name in registers), *own)

    @property
    def discrete_state_names(self) -> tuple[str, ...]:
        return self._declared_registers("discrete")

    @property
    def angle_state_names(self) -> tuple[str, ...]:
        return self._declared_registers("angle")

    def get_state(self) -> dict[str, Any]:
        state = {f"{head}{name}": (bool(word[k]) if name == "on" else _real(word[k]))
                 for head, word in (("", self.active), ("shadow.", self.shadow))
                 for k, name in enumerate(self._names)}
        state.update(self._modulator_state())
        return state

    def set_state(self, values: Mapping[str, Any]) -> None:
        own = set(self._modulator_state())
        names = set(self.get_state())
        unknown = set(values) - names
        if unknown:
            raise KeyError(f"PWM registers: no state(s) {sorted(unknown)}; known: {sorted(names)}")
        if own & set(values):
            (self.pulse_update or self._modulator).set_state(
                {key: value for key, value in values.items() if key in own})
        active, shadow = list(self.active), list(self.shadow)
        for key, value in values.items():
            if key in own:
                continue
            head, _, name = key.rpartition(".")
            k = self._names.index(name)
            target = {"": active, "shadow": shadow}[head]
            target[k] = (1.0 if value else 0.0) if name == "on" else float(value)
        self.active, self.shadow = tuple(active), tuple(shadow)


# ------------------------------------------------------------------ the triangular carrier

_SNAP = 1e-9  # vertex snapping tolerance (carrier periods)


def carrier_position(t: float, f_sw: float, phase: float) -> float:
    """Return the carrier position in [0, 1) periods; ``f_sw`` in Hz, ``phase`` in periods."""
    u = math.fmod(t * f_sw + phase, 1.0)
    if u < 0.0:
        u += 1.0
    if u > 1.0 - _SNAP or u < _SNAP:
        return 0.0
    if abs(u - 0.5) < _SNAP:
        return 0.5
    return u


def _value(u: float) -> float:
    """The carrier value in [-1, 1] at position ``u`` (periods): rising from -1 at 0, falling from 1 at 1/2."""
    return -1.0 + 4.0 * u if u < 0.5 else 3.0 - 4.0 * u


def carrier(t: float, f_sw: float, phase: float = 0.0) -> float:
    """Return the triangular carrier value in [-1, 1]; ``phase`` in carrier periods.

    With ``phase = 0`` the carrier is at -1 and rising at ``t = 0``.
    """
    return _value(carrier_position(t, f_sw, phase))


def _pieces(t0: float, span: float, f_sw: float, phase: float, tol: float,
            limit: int) -> Iterator[tuple[float, float, bool, float, float]]:
    """The monotonic pieces of the carrier over ``[t0, t0 + span]``: ``(offset, length, rising,
    value at the start, slope)``, at most ``limit`` of them."""
    covered = 0.0
    for _ in range(limit):
        if covered >= span - tol:
            return
        u = carrier_position(t0 + covered, f_sw, phase)
        rising = u < 0.5
        seg = min(((0.5 if rising else 1.0) - u) / f_sw, span - covered)
        yield covered, seg, rising, _value(u), 4.0 * f_sw if rising else -4.0 * f_sw
        covered += seg


# ------------------------------------------------------------------ modulators

_EPS = 1e-13


class CarrierComparison(Modulator):
    """Triangular-carrier comparison with exact switching instants.

    ``f_sw`` in Hz, ``phase`` in carrier periods, ``min_interval`` in s.
    """

    supported_models = frozenset(("switching",))

    @classmethod
    def from_config(cls, pwm, f0, references, carried):
        result = cls(pwm.f_sw, pwm.carrier_phase)
        result.carried = carried
        return result

    def phase_clock(self, t, theta=None, omega=None, *, registers=None):
        return PhaseClock(0.0, 2.0 * math.pi * self.phase, 2.0 * math.pi * self.f_sw)


    def __init__(self, f_sw: float, phase: float = 0.0, min_interval: float = 1e-12) -> None:
        self.f_sw = _real(f_sw)
        self.phase = _real(phase)
        self.min_interval = min_interval

    @staticmethod
    def state_fractions(duties):
        """Integrate a complete common-carrier cycle without allocating a timer or events.

        Sorting duty thresholds gives at most n+1 states instead of enumerating 2**n gate
        combinations. Scalar operations retain parameter expressions for standalone export.
        """
        ordered = list(duties)
        for end in range(len(ordered) - 1, 0, -1):
            for index in range(end):
                first, second = ordered[index:index + 2]
                ordered[index] = np.minimum(first, second).item()
                ordered[index + 1] = np.maximum(first, second).item()
        bounds = (0.0, *ordered, 1.0)
        return tuple((upper - lower,
                      tuple(np.where(value > 0.5 * (lower + upper), 1.0, 0.0).item()
                            for value in duties))
                     for lower, upper in zip(bounds[:-1], bounds[1:]))

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        m = np.clip(2.0 * np.asarray(d_abc, dtype=float) - 1.0, -1.0, 1.0).tolist()
        width = len(m)
        # initial switching state at t
        u_t = carrier_position(t, self.f_sw, self.phase)
        c_t = _value(u_t)
        if u_t < 0.5:
            q = [1.0 if m[k] > c_t else 0.0 for k in range(width)]
        else:
            q = [1.0 if m[k] >= c_t else 0.0 for k in range(width)]
        events: list[tuple[float, tuple[int], tuple[float]]] = []
        for start, seg, rising, c0, slope in _pieces(t, T_s, self.f_sw, self.phase, _EPS, 64):
            for k in range(width):
                x = (m[k] - c0) / slope
                if 0.0 < x < seg:
                    # rising carrier crosses m from below -> switch off; falling -> on
                    events.append((start + x, (k,), (0.0 if rising else 1.0,)))
        events.sort(key=lambda e: e[0])
        return _switching_sequence(T_s, self.min_interval, q, events)


class SynchronousCarrier(Modulator):
    """Carrier comparison with the carrier locked to the controller angle.

    Carrier position is ``pulse_ratio * theta / (2 pi) + phase`` (periods);
    requires ``theta`` (rad) and ``omega`` (rad/s).
    """

    supported_models = frozenset(("switching",))

    @classmethod
    def from_config(cls, pwm, f0, references, carried):
        result = cls(round(pwm.f_sw / f0), pwm.carrier_phase)
        result.carried = carried
        return result

    angle_state_names = ("angle",)

    def phase_clock(self, t, theta=None, omega=None, *, registers=None):
        if self._time is not None and (registers is None or self._time == t):
            return PhaseClock(self._time, self._angle, 2.0 * math.pi * self._frequency)
        if ((theta is None and "angle" not in self._restored)
                or (omega is None and "frequency" not in self._restored)):
            raise ValueError("the synchronous carrier needs an angle and angular rate")
        angle = (self._angle if "angle" in self._restored else
                 self.pulse_ratio * theta + 2.0 * math.pi * self.phase)
        rate = (2.0 * math.pi * self._frequency if "frequency" in self._restored else
                self.pulse_ratio * omega)
        return PhaseClock(t, angle, rate)

    def __init__(self, pulse_ratio: int, phase: float = 0.0, min_interval: float = 1e-12) -> None:
        self.pulse_ratio = _integer(pulse_ratio)
        self.phase = _real(phase)
        self.min_interval = min_interval
        self._angle = self._frequency = 0.0
        self._time = None
        self._restored = set()

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        clock = self.phase_clock(t, theta, omega, registers=d_abc)
        frequency = clock.rate / (2.0 * math.pi)
        if frequency <= 0.0:
            raise ValueError(f"synchronous modulation needs omega > 0, got {omega}")
        self._angle = clock.angle % (2.0 * math.pi)
        self._frequency, self._time = frequency, t
        self._restored.clear()
        position = self._angle / (2.0 * math.pi)
        return CarrierComparison(frequency, position - t * frequency, self.min_interval)(
            t, T_s, d_abc)

    def get_state(self):
        return {"angle": float(self._angle), "frequency": float(self._frequency)}

    def set_state(self, values):
        if unknown := set(values) - {"angle", "frequency"}:
            raise KeyError(f"synchronous carrier: unknown state(s) {sorted(unknown)}")
        if "angle" in values:
            self._angle = float(values["angle"]) % (2.0 * math.pi)
        if "frequency" in values:
            self._frequency = float(values["frequency"])
        self._restored.update(values)
        self._time = None


class ZOH(Modulator):
    """PWM-period averaging: hold the duty ratios as the switching state for the period."""

    supported_models = frozenset(("pwm_averaging", "averaging"))

    @classmethod
    def from_config(cls, pwm, f0, references, carried):
        result = cls()
        result.carried = carried
        return result

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        d = np.clip(np.asarray(d_abc, dtype=float), 0.0, 1.0)
        return SwitchingSequence(np.array([T_s]), d[None, :])



# ------------------------------------------------------------------ square-wave phase-shift modulation

_LEGS = {1: (1.0, 0.0), 0: (0.0, 0.0), -1: (0.0, 1.0)}


def pulse_active(angle: float, centre: float, half_width: float) -> bool:
    """Half-open cyclic pulse, shared by bridge modulation and gate-pattern control."""
    offset = math.remainder(angle - centre, 2.0 * math.pi)
    if offset == math.pi:
        offset = -math.pi
    return -half_width <= offset < half_width


def _phase_shift_boundaries(width: float, mode: int) -> tuple[float, ...]:
    """Angles (from the phase register) at which a pattern may change its level."""
    h = 0.5 * math.pi * width
    return tuple(edge for centre, _level in PHASE_SHIFT_MODES[mode]
                 for edge in (centre - h, centre + h)) if h > 0.0 else ()


def phase_shift_level(width: float, mode: int, y: float) -> int:
    """Signed bridge level at angle ``y`` from the phase register.

    Each pulse covers ``[-width pi/2, width pi/2)`` relative to its mode's centre.
    """
    h = 0.5 * math.pi * width
    if h <= 0.0:
        return 0
    for centre, level in PHASE_SHIFT_MODES[mode]:
        if pulse_active(y, centre, h):
            return level
    return 0


class PhaseShiftModulator:
    """Square-wave phase-shift modulation of a full bridge with exact edges.

    The compare registers are ``width`` (pulse width in half periods, clipped to [0, 1]),
    ``phase`` (rad; centre of the positive pulse on the angle axis), ``frequency`` (Hz) and
    ``mode`` (:data:`PHASE_SHIFT_MODES`: 0 bipolar pulses, 1 positive pulse only,
    3 zero level, 4 negative pulse only). They pass the
    same shadow/active register path, computation delay and enable as the duty ratios of a
    carrier modulator.

    The angle axis ``theta`` is continuous: every load continues it from the previous load with
    the previous active frequency, then runs the newly loaded one. The frequency therefore may
    differ from the PWM timer and may change at every load without a phase jump. A fresh run
    starts the axis at ``2 pi f t``, so converters sharing a time base share the axis and their
    ``phase`` registers set their relative phase directly.

    Edges within ``1e-9`` of a period after a load are applied at the load; edges within the same
    distance before the end of an interval belong to the next interval (the load that coincides
    with a switching edge is neither lost nor applied twice).

    ``carried`` names further controller outputs (for example ``scc_angle``) that pass the same
    register path without shaping the pattern; other subsystems read their active values.

    Complete-pulse updates are handled by the shared PWM sequence policy.
    """

    pattern_registers = ("width", "phase", "frequency", "mode")
    discrete_registers = ("mode",)
    angle_registers = ("phase",)
    config_fields = frozenset()
    pulse_groups = ((0, 1),)
    supported_models = frozenset(("switching",))
    switching_legs = 2
    phase_register = "phase"

    @classmethod
    def from_config(cls, pwm, f0, references, carried):
        return cls(references.frequency, carried=carried)

    def phase_clock(self, t, theta=None, omega=None, *, registers=None):
        if registers is None:
            return PhaseClock(self._t or 0.0, self._theta, 2.0 * math.pi * self.frequency)
        frequency = self._registers(registers)[2]
        return PhaseClock(t, self._start_angle(t, frequency), 2.0 * math.pi * frequency)

    def _start_angle(self, t: float, frequency: float) -> float:
        """Angle of an interval starting at ``t``: restored, first-anchored or continued."""
        return (self._theta if self._restored else 2.0 * math.pi * frequency * t
                if self._t is None else self.angle(t))

    def __init__(self, frequency: float, min_interval: float = 1e-12,
                 carried: tuple[str, ...] = ()) -> None:
        frequency = _real(frequency)
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError(f"the phase-shift frequency must be finite and positive, got {frequency}")
        self.nominal_frequency = frequency
        self.min_interval = min_interval
        self.carried = tuple(carried)
        self.registers = self.pattern_registers + self.carried
        self._t: float | None = None       # start of the current interval (None: not anchored)
        self._theta = 0.0                  # angle at that start (rad, wrapped to [0, 2 pi))
        self._f = frequency                # active frequency of the current interval (Hz)
        self._restored = False

    # -- registers -------------------------------------------------------------------------------
    def command_registers(self, output: Any) -> tuple[float, ...]:
        """The register word written by one controller output."""
        return tuple(getattr(output, name) for name in self.registers)

    def neutral_registers(self) -> tuple[float, ...]:
        """Registers before the first controller write: zero width at the nominal frequency."""
        return (0.0, 0.0, self.nominal_frequency, 0.0) + (0.0,) * len(self.carried)

    def _registers(self, values: NDArray[np.float64]) -> tuple[float, float, float, int]:
        for name, value in zip(self.registers, values):
            if not math.isfinite(value):
                raise ValueError(f"phase-shift {name} register must be finite, got {value}")
        width, phase, frequency, mode = (float(value) for value in values[:4])
        if frequency <= 0.0:
            raise ValueError(f"phase-shift frequency register must be positive, got {frequency}")
        if mode not in PHASE_SHIFT_MODES:
            raise ValueError(f"phase-shift mode register must be one of "
                             f"{sorted(PHASE_SHIFT_MODES)}, got {mode}")
        return min(1.0, max(0.0, width)), phase, frequency, int(mode)

    # -- the angle axis --------------------------------------------------------------------------
    def angle(self, t: float) -> float:
        """Angle axis (rad, unwrapped from the current interval start) at ``t``."""
        if self._t is None:
            return 2.0 * math.pi * self._f * t
        return self._theta + 2.0 * math.pi * self._f * (t - self._t)

    @property
    def frequency(self) -> float:
        """Active angle-axis frequency (Hz)."""
        return self._f

    # -- edges -----------------------------------------------------------------------------------
    @staticmethod
    def _edges(pattern, theta0: float, omega: float, x0: float, x1: float, snap: float):
        """Level just after ``x0`` and the level changes in ``(x0 + snap, x1 - snap]``, times
        from the interval start with ``theta(x) = theta0 + omega x``. An edge within ``snap``
        after ``x0`` is part of the initial level."""
        width, phase, mode = pattern
        start = theta0 + omega * x0 - phase
        first = phase_shift_level(width, mode, start + omega * snap)
        bounds = _phase_shift_boundaries(width, mode)
        if not bounds or x1 - snap <= x0 + snap:
            return first, []
        period = 2.0 * math.pi / omega
        times = []
        for bound in bounds:
            x = math.fmod(bound - start, 2.0 * math.pi)
            if x < 0.0:
                x += 2.0 * math.pi
            x /= omega
            if x <= snap:
                x += period
            while x0 + x <= x1 - snap:
                times.append(x0 + x)
                x += period
        times.sort()
        changes, level = [], first
        for x in times:
            after = phase_shift_level(width, mode, theta0 + omega * x - phase + omega * snap)
            if after != level:
                changes.append((x, after))
                level = after
        return first, changes

    # -- one register interval -------------------------------------------------------------------
    def __call__(self, t: float, T_s: float, values: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        width, phase, frequency, mode = self._registers(values)
        theta0 = self._start_angle(t, frequency) % (2.0 * math.pi)
        self._t, self._theta, self._f, self._restored = t, theta0, frequency, False
        level, changes = self._edges(
            (width, phase, mode), theta0, 2.0 * math.pi * frequency, 0.0, T_s, _SNAP / frequency)
        return _switching_sequence(T_s, self.min_interval, list(_LEGS[level]),
                                   ((x, self.pulse_groups[0], _LEGS[after]) for x, after in changes))

    # -- named state -----------------------------------------------------------------------------
    angle_state_names = ("angle",)

    def get_state(self) -> dict[str, Any]:
        return {"angle": float(self._theta)}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if unknown := set(values) - {"angle"}:
            raise KeyError(f"phase-shift modulator: unknown state(s) {sorted(unknown)}")
        if "angle" in values:
            self._theta = float(values["angle"]) % (2.0 * math.pi)
            self._restored = True


MODULATOR_TYPES = {"phase_shift": PhaseShiftModulator}


def modulator_type(pwm, model, kind=None):
    """Select the same implementation for validation, construction and export metadata."""
    if kind is not None:
        return MODULATOR_TYPES[kind]
    if model != "switching":
        return ZOH
    return SynchronousCarrier if pwm.sync == "synchronous" else CarrierComparison


def validate_modulator(unit, topology, where):
    if unit.pwm is None:
        return
    cls = modulator_type(unit.pwm, unit.bridge.model, control_type(unit.ctrl.type).modulator)
    if unit.bridge.model not in cls.supported_models:
        raise ConfigError(f"{where}.bridge.model: {cls.__name__} requires resolved switching edges")
    legs = getattr(cls, "switching_legs", None)
    if legs is not None and len(topology.voltage_factors) != legs:
        raise ConfigError(f"{where}.bridge: {cls.__name__} requires {legs} switching legs")
    for item in fields(unit.pwm):
        if (item.metadata.get("modulator") and item.name not in cls.config_fields
                and getattr(unit.pwm, item.name) != item.default):
            raise ConfigError(f"{where}.pwm.{item.name} does not apply to {cls.__name__}")


def make_modulator(pwm: Any, f0: float, bridge: Any, kind: str | None = None,
                   references: Any = None, carried: tuple[str, ...] = ()) -> Modulator:
    cls = modulator_type(pwm, bridge.model, kind)
    return cls.from_config(pwm, f0, references, tuple(carried))


def make_unit_modulator(cfg):
    spec = control_type(cfg.ctrl.type)
    return make_modulator(cfg.pwm, cfg.base.f0, cfg.bridge, spec.modulator, cfg.ctrl.references,
                          spec.interface.auxiliary_outputs(cfg.ctrl.outputs))
