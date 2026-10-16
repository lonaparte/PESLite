"""Shared converter measurements: named SI samples, window means and optional RMS statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .pwm import TIME_EPS
from ..control.blocks import MovingWindow
from ..solver.energy import PowerPort, port_power, re_product
from ..solver.model import Bag, Empty

__all__ = ["ADCSample", "MeasurementPorts", "ADC"]


class _PortMeasurementInp(Bag):
    __slots__ = ("voltage", "current")


class _PortMeasurementOut(Bag):
    __slots__ = ("voltage", "current", "power")


class _PortMeasurement:
    """Non-loading observation of an already assembled electrical terminal."""

    observer = True
    outputs_need_inputs = True
    state_names = ()

    def __init__(self, terminal) -> None:
        port = terminal.power_port
        self.ports = (PowerPort("inp.voltage", "inp.current", port.scale, -port.sign),)
        self.state = Empty()
        self.inp = _PortMeasurementInp()
        self.out = _PortMeasurementOut()

    def set_outputs(self, t: float) -> None:
        self.out.voltage = self.inp.voltage
        self.out.current = self.inp.current
        self.out.power = port_power(self, self.ports[0])

    def rhs(self, t: float):
        return ()


class _RMSWindow:
    """Sliding mean-square over complete sampling periods, integrated before sampling."""

    def __init__(self, period: float, length: float) -> None:
        count = round(length / period)
        if count < 1 or not math.isclose(length, count * period, rel_tol=1e-9):
            raise ValueError("an RMS window must contain whole sampling periods")
        self.period = period
        self.window = MovingWindow(count)
        self.area = self.duration = self.last = self.total = 0.0
        self.count = 0
        self._value: float | None = None

    def accumulate(self, dt: float, value: complex | float) -> None:
        square = re_product(value, value)
        self.area += 0.5 * dt * (self.last + square)
        self.duration += dt
        self.last = square

    def sample(self) -> None:
        # Do not manufacture a full period from a partial initial interval.
        if self.duration >= self.period * (1.0 - 1e-9):
            value = self.area / self.duration
            old = self.window.push(value)
            self.total += value - (0.0 if old is None else old)
            self.count = min(self.window.n, self.count + 1)
            self._value = None
        self.area = self.duration = 0.0

    @property
    def value(self) -> float | None:
        if self._value is None and self.window.full:
            self._value = math.sqrt(max(0.0, self.total / self.count))
        return self._value

    def get_state(self) -> dict[str, float]:
        return {"area": self.area, "duration": self.duration, "last": self.last,
                "total": self.total, "count": self.count, "index": self.window.idx,
                **{f"sample_{i}": value for i, value in enumerate(self.window.buf)}}

    def set_state(self, values: Mapping[str, Any]) -> None:
        self._value = None
        for name in ("area", "duration", "last", "total"):
            if name in values:
                setattr(self, name, float(values[name]))
        count = int(values.get("count", self.count))
        index = int(values.get("index", self.window.idx))
        if (count != values.get("count", count) or index != values.get("index", index)
                or not 0 <= count <= self.window.n or not 0 <= index < self.window.n):
            raise ValueError("invalid RMS window position")
        self.count, self.window.idx = count, index
        self.window.full = count == self.window.n
        for i in range(self.window.n):
            if f"sample_{i}" in values:
                self.window.buf[i] = float(values[f"sample_{i}"])


@dataclass(frozen=True, slots=True)
class ADCSample:
    """One generic converter sample before its unit-specific control adaptation.

    ``values`` is the published control value. ``samples`` contains the instantaneous
    conversions in chronological order; its final entry is the conversion at ``t``. An
    instantaneous entry has an empty ``samples`` tuple.
    """

    t: float
    values: Mapping[str, complex | float]
    samples: tuple["ADCSample", ...] = ()


@dataclass(frozen=True, slots=True)
class MeasurementPorts:
    """Named, non-loading zero-argument readers at a converter's control boundary."""

    readers: Mapping[str, Callable[[], complex | float]]

    def read(self) -> dict[str, complex | float]:
        """The instantaneous values of the ADC channels."""
        return {name: read() for name, read in self.readers.items()}

    def value(self, name: str) -> complex | float:
        """Read one named channel."""
        return self.readers[name]()


class ADC:
    """Sampler triggered every control ``period`` and optionally between interrupts.

    The named channels in ``channels`` are averaged over a window of ``length`` s ending at the
    interrupt; all other ports are instantaneous.
    Named state per averaged channel ``c``: ``x_c``, the integral accumulated in the open window.
    Optional RMS windows integrate squared signals on the same timeline and retain only their
    configured number of complete control/protection periods; they do not alter sample clocks.
    """

    def __init__(self, ports: MeasurementPorts, period: float, sample_period: float,
                 length: Optional[float] = None,
                 channels: Mapping[str, complex | float] | None = None,
                 rms: Mapping[str, float] | None = None) -> None:
        self.ports = ports
        channels = dict(channels or {})
        if channels and (length is None or length <= 0.0):
            raise ValueError(f"an averaging window must be longer than zero, got {length}")
        self.length = float(length) if channels else None  # None: instantaneous sampling
        self.channels = tuple(channels)
        self._zero = channels
        self.accumulated = dict(channels)
        self._last = dict(channels)  # previous sample
        self.period, self.sample_period = period, sample_period
        self.samples = max(1, int(round(period / sample_period)))
        self.peeks: list[ADCSample] = []  # instantaneous samples since the last interrupt
        self.n_samp = 1  # sample 0 of a period is its control interrupt
        # the next interrupt's window is open (a full-period window opens at the interrupt itself)
        self.window_open = self.length is None or self._full
        self.rms = {channel: _RMSWindow(period, window)
                    for channel, window in (rms or {}).items()}
        self._integrated = tuple(
            (channel, channel in channels, self.rms.get(channel))
            for channel in dict.fromkeys((*self.channels, *self.rms))
        )

    @property
    def accumulating(self) -> bool:
        return self.averaging or bool(self.rms)

    @property
    def _full(self) -> bool:
        return self.length >= self.period * (1.0 - 1e-12)

    @property
    def averaging(self) -> bool:
        return bool(self.channels)

    def start(self, t: float, last: float, next_: float, held: bool) -> None:
        """Resume sampling at ``t`` between the interrupts ``last`` and ``next_``.

        ADC integrals are named states, while the position of the sampler and whether the next
        averaging window is open follow from ``t`` and its timer grid.  Samples already taken
        between interrupts are not states; validation therefore permits a continued oversampled
        run only at an interrupt.  For a new run, ``held`` makes a window which began before
        ``t`` count the value at ``t`` over that preceding part.
        """
        self.n_samp = max(1, int(math.floor((t - last + TIME_EPS) / self.sample_period)) + 1)
        length = self.length
        self.window_open = length is None or self._full or t >= next_ - length - TIME_EPS
        if length is not None and held and self.window_open:
            self.held_before(max(0.0, t - (next_ - length)))
        self.peeks = []
        if self.n_samp > 1:  # a new run between interrupts: earlier samples use its start values
            values = self.ports.read()
            self.peeks = [ADCSample(last + i * self.sample_period, values)
                          for i in range(1, self.n_samp)]

    # ------------------------------------------------------------ the window
    def seed(self) -> None:
        """Start the next integration interval from the values after changes at this instant."""
        for channel, averaged, window in self._integrated:
            value = self.ports.readers[channel]()
            if averaged:
                self._last[channel] = value
            if window is not None:
                window.last = re_product(value, value)

    def held_before(self, duration: float) -> None:
        """Count the current values as held for ``duration`` seconds before the run starts."""
        for channel in self.channels:
            self.accumulated[channel] += self._last[channel] * duration

    def accumulate(self, dt: float) -> None:
        """Add the trapezoidal integral over the interval of length ``dt`` (s) ending now."""
        if dt <= 0.0:
            return
        half = 0.5 * dt
        for channel, averaged, window in self._integrated:
            value = self.ports.readers[channel]()
            if averaged:
                if self.window_open:
                    self.accumulated[channel] += half * (self._last[channel] + value)
                self._last[channel] = value
            if window is not None:
                window.accumulate(dt, value)

    def open(self) -> None:
        """Start the window ending at the next control interrupt."""
        self.accumulated = dict(self._zero)
        self.window_open = True

    def t_window(self, t_next: float) -> float:
        """The opening of the window ending at interrupt ``t_next`` (s); ``inf`` when open."""
        return math.inf if self.window_open else t_next - self.length

    # ------------------------------------------------------------ the samples
    def t_sample(self, start: float) -> float:
        """The next sample between two interrupts in the period starting at ``start``; ``inf`` if none."""
        return (start + self.n_samp * self.sample_period) if self.n_samp < self.samples else math.inf

    def peek(self, t: float) -> None:
        """Take an oversample between two control interrupts."""
        self.peeks.append(ADCSample(t, self.ports.read()))
        self.n_samp += 1

    def sample(self, t: float) -> ADCSample:
        """Take the sample at the control interrupt ``t``, with the preceding oversamples,
        and rearm the averaging window."""
        meas = self.measure(t)
        self.sample_rms()
        if self.length is not None:
            self.window_open = self._full
            if self.window_open:
                self.open()
            else:
                self.accumulated = dict(self._zero)
        meas = ADCSample(meas.t, meas.values, tuple(self.peeks) + meas.samples)
        self.peeks, self.n_samp = [], 1
        return meas

    def sample_rms(self, t: float | None = None) -> bool:
        """Finish one statistics period without changing instantaneous control measurements."""
        if t is not None and abs(t - round(t / self.period) * self.period) >= TIME_EPS:
            return False
        for window in self.rms.values():
            window.sample()
        return True

    def measure(self, t: float) -> ADCSample:
        """Read every named channel at ``t`` and apply configured window means."""
        reading = self.ports.read()
        mean = {channel: self.accumulated[channel] / self.length for channel in self.channels}
        values = {channel: mean.get(channel, value) for channel, value in reading.items()}
        return ADCSample(t=t, values=values, samples=(ADCSample(t=t, values=reading),))

    # ------------------------------------------------------------ its states
    def get_state(self) -> dict[str, Any]:
        return {**{f"x_{channel}": value for channel, value in self.accumulated.items()},
                **{f"rms_{channel}.{key}": value for channel, window in self.rms.items()
                   for key, value in window.get_state().items()}}

    def set_state(self, values: Mapping[str, Any]) -> None:
        for channel, zero in self._zero.items():
            cast = complex if isinstance(zero, complex) else float
            key = f"x_{channel}"
            if key in values:
                self.accumulated[channel] = cast(values[key])
        for channel, window in self.rms.items():
            prefix = f"rms_{channel}."
            window.set_state({key[len(prefix):]: value for key, value in values.items()
                              if key.startswith(prefix)})
