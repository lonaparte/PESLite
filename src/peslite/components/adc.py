"""The common converter ADC: instantaneous named samples or window means in SI units."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .pwm import TIME_EPS

__all__ = ["ADCSample", "MeasurementPorts", "ADC"]


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
    """Named zero-argument readers at a converter's control boundary."""

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
    """

    def __init__(self, ports: MeasurementPorts, period: float, sample_period: float,
                 length: Optional[float] = None,
                 channels: Mapping[str, complex | float] | None = None) -> None:
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
        values = self.ports.read()
        self._last = {channel: values[channel] for channel in self.channels}

    def held_before(self, duration: float) -> None:
        """Count the current values as held for ``duration`` seconds before the run starts."""
        for channel in self.channels:
            self.accumulated[channel] += self._last[channel] * duration

    def accumulate(self, dt: float) -> None:
        """Add the trapezoidal integral over the interval of length ``dt`` (s) ending now."""
        if dt <= 0.0 or not self.channels:
            return
        values = self.ports.read()
        half = 0.5 * dt
        for channel in self.channels:
            now = values[channel]
            if self.window_open:
                self.accumulated[channel] += half * (self._last[channel] + now)
            self._last[channel] = now

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
        if self.length is not None:
            self.window_open = self._full
            if self.window_open:
                self.open()
            else:
                self.accumulated = dict(self._zero)
        meas = ADCSample(meas.t, meas.values, tuple(self.peeks) + meas.samples)
        self.peeks, self.n_samp = [], 1
        return meas

    def measure(self, t: float) -> ADCSample:
        """Read every named channel at ``t`` and apply configured window means."""
        reading = self.ports.read()
        mean = {channel: self.accumulated[channel] / self.length for channel in self.channels}
        values = {channel: mean.get(channel, value) for channel, value in reading.items()}
        return ADCSample(t=t, values=values, samples=(ADCSample(t=t, values=reading),))

    # ------------------------------------------------------------ its states
    def get_state(self) -> dict[str, Any]:
        return {f"x_{channel}": value for channel, value in self.accumulated.items()}

    def set_state(self, values: Mapping[str, Any]) -> None:
        for channel, zero in self._zero.items():
            cast = complex if isinstance(zero, complex) else float
            key = f"x_{channel}"
            if key in values:
                self.accumulated[channel] = cast(values[key])
