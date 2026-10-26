"""The PWM peripheral of a converter: timer, duty registers, carrier and modulators.

A modulator is called at every load of the compare registers for the interval until the next load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..control.blocks import abc2complex
from ..solver.model import _integer, _real

__all__ = ["PWM", "TIME_EPS", "SwitchingSequence", "Modulator", "CarrierComparison",
           "SynchronousCarrier", "ZOH", "carrier", "carrier_position", "make_modulator"]


@dataclass
class SwitchingSequence:
    """Piecewise-constant switching pattern from one compare-register load to the next.

    ``dt``: interval lengths (s) summing to the interval between compare-register loads.
    ``q_abc[i]``: phase states in interval ``i`` (0/1 when switching, fractional when averaged).
    """

    dt: NDArray[np.float64]
    q_abc: NDArray[np.float64]


@runtime_checkable
class Modulator(Protocol):
    """Convert loaded duty ratios into a SwitchingSequence spanning ``T`` seconds.

    Optional ``theta`` (rad) and ``omega`` (rad/s) set synchronous carrier timing; asynchronous modulators ignore them.
    """

    def __call__(self, t: float, T: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence: ...


TIME_EPS = 1e-10


class PWM:
    """Carrier timer, PWM duty registers and switching schedule.

    Interrupts occur every ``period`` from ``offset``. Compare registers load every
    ``load_period``. An interrupt writes duty ratios and the PWM enable to the shadow registers;
    a load copies them to the active registers only after ``computation`` seconds. The enable is
    carried through exactly the same register path as the duty ratios.
    """

    _NAMES = ("d_a", "d_b", "d_c", "on")

    def __init__(self, period: float, load_period: float, offset: float, computation: float,
                 carrier_period: float, modulator: Modulator, *,
                 duty_names: tuple[str, ...] | None = None,
                 switching_state: Callable[[Any], Any] | None = None,
                 pwm_state: Callable[[Any], Any] | None = None) -> None:
        if not math.isfinite(computation) or computation < 0.0:
            raise ValueError(f"the computation time must be finite and >= 0, got {computation}")
        self.period, self.load_period, self.offset = _real(period), _real(load_period), _real(offset)
        self.computation, self.carrier_period = _real(computation), _real(carrier_period)
        self._names = (*duty_names, "on") if duty_names is not None else self._NAMES
        self._width = len(self._names) - 1
        self._switching_state = switching_state or abc2complex
        self._pwm_state = pwm_state or ZOH.value
        self.modulator = modulator
        self.k = self.j = 0
        self.t_interrupt, self.t_load = self.interrupt(0), self.load_time(0)
        self.active, self.shadow = np.zeros(self._width + 1), np.zeros(self._width + 1)
        self.sync: tuple[float | None, float | None] = (None, None)
        self.t_sync = 0.0
        self.schedule: list[tuple[float, Any]] = []
        self.next_switch = math.inf
        self.load_end = math.inf

    @property
    def modulator(self) -> Modulator:
        return self._modulator

    @modulator.setter
    def modulator(self, value: Modulator) -> None:
        self._modulator = value
        self._switching = (
            self._pwm_interval if type(value) is ZOH else self._switching_interval
        )

    def interrupt(self, k: int) -> float:
        return self.offset + k * self.period

    def load_time(self, j: int) -> float:
        return self.offset + j * self.load_period

    @property
    def t_next(self) -> float:
        """Next control interrupt (temporary name retained while the run loop is migrated)."""
        return self.t_interrupt

    def after(self, t: float, period: Optional[float] = None) -> int:
        """Index of the first timer point after ``t``; a point at ``t`` is not after it."""
        return int(self.index_after(t, period))

    def index_after(self, t: float, period: Optional[float] = None):
        """Scalar timer-index equation shared by initialization and standalone execution."""
        T = self.period if period is None else period
        return np.floor((t - self.offset + TIME_EPS) / T) + 1

    def count(self, t_a: float, t_b: float) -> int:
        """Number of control interrupts in ``[t_a, t_b]``, including both ends."""
        first = math.ceil((t_a - self.offset - TIME_EPS) / self.period)
        return max(0, self.after(t_b) - first)

    def describe(self) -> str:
        double = self.load_period < 0.75 * self.carrier_period
        return (f"control {1e-3 / self.period:g} kHz, computation {self.computation * 1e6:g} us, "
                f"{'double' if double else 'single'} update")

    def _word(self, duties: NDArray[np.float64], on: bool) -> NDArray[np.float64]:
        values = np.asarray(duties, dtype=float).ravel()
        if values.size != self._width:
            raise ValueError(f"expected {self._width} duty ratios, got {values.size}")
        word = np.empty(self._width + 1, dtype=float)
        word[:-1] = values
        word[-1] = 1.0 if on else 0.0
        return word

    def reset(self, d_abc: NDArray[np.float64], on: bool = False) -> None:
        """Put duty ratios and enable in both register banks; the default is blocked."""
        word = self._word(d_abc, on)
        self.active, self.shadow = word.copy(), word.copy()

    def tick(self, t: float, theta: float | None = None, omega: float | None = None) -> None:
        """Advance the control-interrupt timer and its synchronous-carrier reference."""
        self.sync, self.t_sync = (theta, omega), t
        self.k += 1
        self.t_interrupt = self.interrupt(self.k)

    def write_shadow(self, d_abc: NDArray[np.float64], on: bool = True) -> None:
        """Write duty ratios and enable directly to an immediately eligible shadow register."""
        self.shadow = self._word(d_abc, on)

    def write(self, t: float, d_abc: NDArray[np.float64], on: bool = True,
              theta: float | None = None,
              omega: float | None = None) -> None:
        """Advance an interrupt and write its computed duty ratios and enable to shadow."""
        self.tick(t, theta, omega)
        self.shadow = self._word(d_abc, on)

    @property
    def duty(self) -> NDArray[np.float64]:
        """Duty ratios in the active compare registers."""
        return self.active[:-1].copy()

    @property
    def on(self) -> bool:
        """PWM enable in the active compare registers."""
        return bool(self.active[-1])

    def start(self, t: float, sync: tuple[float | None, float | None], continued: bool):
        """Resume the timer and switching sequence at ``t`` after named states were loaded."""
        self.k, self.j = self.after(t), self.after(t, self.load_period)
        self.t_interrupt, self.t_load = self.interrupt(self.k), self.load_time(self.j)
        last = self.interrupt(self.k - 1)
        self.sync, self.t_sync = sync, min(last, t)
        q = self._switching(self.load_time(self.j - 1), self.active[:-1], t)
        if not continued:
            self.t_sync = t
        return q

    def load(self, t: float):
        """Load shadow into active registers and schedule switching until the next load."""
        completed = self.interrupt(self.k - 1) + self.computation
        if completed <= t + TIME_EPS:
            # Writes replace the shadow word, so the active bank can take ownership without a
            # copy. Named-state writes detach the banks before mutating either one in place.
            self.active = self.shadow
        q = self._switching(t, self.active[:-1], t)
        self.j += 1
        self.t_load = self.load_time(self.j)
        return q

    def _pwm_interval(self, t_l: float, d: NDArray[np.float64], _t_now: float):
        self.load_end = t_l + self.load_period
        self.schedule.clear()
        self.next_switch = math.inf
        return self._pwm_state(d)

    def _switching_interval(self, t_l: float, d: NDArray[np.float64], t_now: float):
        theta, omega = self.sync
        if theta is not None and omega is not None and t_l != self.t_sync:
            theta = theta + omega * (t_l - self.t_sync)
        seq = self._modulator(t_l, self.load_period, d, theta, omega)
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

    @property
    def next_edge(self) -> float:
        """Next switching or held-input edge relevant to fast protection."""
        return min(self.next_switch, self.load_end)

    def edge(self, t: float) -> bool:
        """Whether the old switching or held-input interval ends at ``t``."""
        return abs(self.next_switch - t) < TIME_EPS or abs(self.load_end - t) < TIME_EPS

    def switches(self, t: float) -> list[Any]:
        """Return switching states due at ``t`` in order."""
        due = []
        while self.schedule and abs(self.schedule[0][0] - t) < TIME_EPS:
            due.append(self.schedule.pop(0)[1])
        self.next_switch = self.schedule[0][0] if self.schedule else math.inf
        return due

    def get_state(self) -> dict[str, Any]:
        return {f"{head}{name}": (bool(word[k]) if name == "on" else float(word[k]))
                for head, word in (("", self.active), ("shadow.", self.shadow))
                for k, name in enumerate(self._names)}

    def set_state(self, values: Mapping[str, Any]) -> None:
        names = set(self.get_state())
        unknown = set(values) - names
        if unknown:
            raise KeyError(f"PWM registers: no state(s) {sorted(unknown)}; known: {sorted(names)}")
        if self.active is self.shadow:
            # load() may transfer ownership without copying.  Named-state writes address the two
            # register banks independently, so detach them before changing either bank in place.
            self.active = self.active.copy()
        for key, value in values.items():
            head, _, name = key.rpartition(".")
            k = self._names.index(name)
            target = {"": self.active, "shadow": self.shadow}[head]
            target[k] = (1.0 if value else 0.0) if name == "on" else float(value)


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


class CarrierComparison:
    """Triangular-carrier comparison with exact switching instants.

    ``f_sw`` in Hz, ``phase`` in carrier periods, ``min_interval`` in s.
    """

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
        events: list[tuple[float, int, float]] = []
        for start, seg, rising, c0, slope in _pieces(t, T_s, self.f_sw, self.phase, _EPS, 64):
            for k in range(width):
                x = (m[k] - c0) / slope
                if 0.0 < x < seg:
                    # rising carrier crosses m from below -> switch off; falling -> on
                    events.append((start + x, k, 0.0 if rising else 1.0))
        events.sort(key=lambda e: e[0])
        dts: list[float] = []
        states: list[list[float]] = []
        t_prev = 0.0
        for time, k, new_state in events:
            if time - t_prev > self.min_interval:
                dts.append(time - t_prev)
                states.append(list(q))
                t_prev = time
            q[k] = new_state
        if T_s - t_prev > self.min_interval or not dts:
            dts.append(T_s - t_prev)
            states.append(list(q))
        else:  # absorb a vanishing last interval into the previous one
            dts[-1] += T_s - t_prev
        return SwitchingSequence(np.asarray(dts), np.asarray(states))


class SynchronousCarrier:
    """Carrier comparison with the carrier locked to the controller angle.

    Carrier position is ``pulse_ratio * theta / (2 pi) + phase`` (periods);
    requires ``theta`` (rad) and ``omega`` (rad/s).
    """

    def __init__(self, pulse_ratio: int, phase: float = 0.0, min_interval: float = 1e-12) -> None:
        self.pulse_ratio = _integer(pulse_ratio)
        self.phase = _real(phase)
        self.min_interval = min_interval

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        if theta is None or omega is None:
            raise ValueError("synchronous modulation needs the controller's angle: this controller "
                             "reports no theta/omega in its ControlOutput")
        f = self.pulse_ratio * omega / (2.0 * np.pi)  # carrier frequency (Hz)
        if f <= 0.0:
            raise ValueError(f"synchronous modulation needs omega > 0, got {omega}")
        position = self.pulse_ratio * theta / (2.0 * np.pi) + self.phase  # carrier position at t (periods)
        return CarrierComparison(f, position - t * f, self.min_interval)(t, T_s, d_abc)


class ZOH:
    """PWM-period averaging: hold the duty ratios as the switching state for the period."""

    @staticmethod
    def value(d_abc: NDArray[np.float64]) -> complex:
        """Return the held space vector without constructing a one-segment schedule."""
        d_a = min(1.0, max(0.0, float(d_abc[0])))
        d_b = min(1.0, max(0.0, float(d_abc[1])))
        d_c = min(1.0, max(0.0, float(d_abc[2])))
        return abc2complex((d_a, d_b, d_c))

    def __call__(self, t: float, T_s: float, d_abc: NDArray[np.float64],
                 theta: float | None = None, omega: float | None = None) -> SwitchingSequence:
        d = np.clip(np.asarray(d_abc, dtype=float), 0.0, 1.0)
        return SwitchingSequence(np.array([T_s]), d.reshape(1, 3))


def make_modulator(pwm: Any, f0: float, bridge: Any) -> Modulator:
    """Return the exact-switching or PWM-period-averaged modulator selected by one unit."""
    if bridge.model == "switching":
        if pwm.sync == "synchronous":
            return SynchronousCarrier(round(pwm.f_sw / f0), pwm.carrier_phase)
        return CarrierComparison(pwm.f_sw, pwm.carrier_phase)
    return ZOH()
