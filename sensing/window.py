"""Averaging ADC window: the mean ``(X(t) - X(t - length)) / length`` of integrated channels."""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["SamplingWindow"]


class SamplingWindow:
    """Averaging window of ``length`` s over named channels, typed by their zero values.

    Example: ``SamplingWindow(50e-6, {"v": 0j, "i": 0j, "dc": 0.0})``.
    Named states per channel ``c``: ``x_c`` (integral) and ``x_c_open`` (integral at window start).
    """

    def __init__(self, length: float, channels: Mapping[str, complex | float]) -> None:
        if length <= 0.0:
            raise ValueError(f"a sampling window must be longer than zero, got {length}")
        if not channels:
            raise ValueError("a sampling window needs at least one channel")
        self.length = float(length)
        self.channels = tuple(channels)
        self._zero = dict(channels)
        self.integral = dict(channels)  # the running integrals
        self.opened = dict(channels)  # integrals at window start
        self._last = dict(channels)  # previous sample

    def __contains__(self, channel: str) -> bool:
        """Return whether ``channel`` is averaged by this window."""
        return channel in self._zero

    # ------------------------------------------------------------ the window
    def seed(self, values: Mapping[str, complex | float]) -> None:
        """Set the initial sample values at the start of a run."""
        self._last = {c: values[c] for c in self.channels}

    def accumulate(self, dt: float, values: Mapping[str, complex | float]) -> None:
        """Add the trapezoidal integral over ``dt`` (s) ending at ``values``."""
        if dt <= 0.0:
            return
        half = 0.5 * dt
        for c in self.channels:
            now = values[c]
            self.integral[c] += half * (self._last[c] + now)
            self._last[c] = now

    def open(self) -> None:
        """Latch the integrals as the start of the next window."""
        self.opened = dict(self.integral)

    def mean(self) -> dict[str, complex | float]:
        """Return the mean of every channel over the window ending now."""
        length, opened = self.length, self.opened
        return {c: (self.integral[c] - opened[c]) / length for c in self.channels}

    # ------------------------------------------------------------ its states
    def get_state(self) -> dict[str, Any]:
        s: dict[str, Any] = {f"x_{c}": v for c, v in self.integral.items()}
        s.update({f"x_{c}_open": v for c, v in self.opened.items()})
        return s

    def set_state(self, values: Mapping[str, Any]) -> None:
        for channel, zero in self._zero.items():
            cast = complex if isinstance(zero, complex) else float
            for key, into in ((f"x_{channel}", self.integral), (f"x_{channel}_open", self.opened)):
                if key in values:
                    into[channel] = cast(values[key])
