"""All protection decisions and the trip latch of one converter unit.

Protection is owned by the unit.  Fast criteria and sampled criteria run on different clocks, but
they feed the same latch and the same physical trip action.  Quantities passed to this module are
already in pu except for time (s) and frequency deviation (Hz).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Mapping, Optional, Sequence

from ..control.blocks import HoldTimer, MovingWindow, peak_abs
from ..solver.model import _real, gather, scatter

__all__ = ["Protection", "ProtectionSample", "TripEvent", "ProtectionStats"]


@dataclass(slots=True)
class TripEvent:
    t: float
    cause: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ProtectionSample:
    """Protection-rate observations available to common converter protection."""

    terminal_voltage_pu: float | None = None
    frequency_deviation: float | None = None
    dc_voltage_error_pu: float | None = None
    startup_complete: bool = True
    rms_current_pu: float | None = None


@dataclass(slots=True)
class ProtectionStats:
    """Protection observations; absent crossings and samples are ``None``."""

    overcurrent_first_t: Optional[float] = None
    rms_current_first_t: Optional[float] = None
    undervoltage_first_t: Optional[float] = None
    overvoltage_first_t: Optional[float] = None
    frequency_first_t: Optional[float] = None
    dc_voltage_first_t: Optional[float] = None
    rocof_first_t: Optional[float] = None
    max_current_pu: float = 0.0
    rocof_max: float = 0.0
    terminal_voltage_min_pu: Optional[float] = None
    terminal_voltage_max_pu: Optional[float] = None
    alarms: list[str] = field(default_factory=list)


class Protection:
    """Protection subsystem of a unit.

    :meth:`check_fast` runs at PWM switching/load edges. :meth:`check_sampled` runs once per
    controller interrupt.  Both latch the same :attr:`trip`; the owning unit performs the physical
    gate, breaker and DC-source action.
    """

    def __init__(self, cfg: Any, T: float, i_base: float, *, rms_available: bool = False) -> None:
        self.cfg, self.T, self.i_base = cfg, _real(T), _real(i_base)
        self.trip: TripEvent | None = None
        self.fault = False
        self.stats = ProtectionStats()
        self._timers = {"hold_uv": HoldTimer(T), "hold_ov": HoldTimer(T),
                        "hold_freq": HoldTimer(T), "hold_vdc": HoldTimer(T)}
        if rms_available:
            self._timers["hold_rms"] = HoldTimer(T)
        self._configure(cfg)

    def _configure(self, cfg: Any) -> None:
        """Cache the enabled paths and rebuild a changed ROCOF window."""
        old_window = getattr(self, "_requested_rocof_window", None)
        if old_window != cfg.rocof.window:
            self._rocof = None
            vars(self).pop("_rocof_count", None)
            vars(self).pop("_rocof_window", None)
            self._requested_rocof_window = cfg.rocof.window
        self.fast_enabled = bool(cfg.overcurrent.enable)
        self.needs_frequency = bool(cfg.frequency.enable or cfg.rocof.enable)
        self.rms_enabled = bool(getattr(getattr(cfg, "rms_current", None), "enable", False))
        if self.rms_enabled and "hold_rms" not in self._timers:
            self._timers["hold_rms"] = HoldTimer(self.T)

    def retune(self, cfg: Any) -> None:
        """Apply new settings without changing latches, timers or observations."""
        self._configure(cfg)
        self.cfg = cfg

    @cached_property
    def _rocof_count(self):
        return max(1, int(self._requested_rocof_window / self.T + 0.5))

    @cached_property
    def _rocof_window(self):
        return self._rocof_count * self.T

    @property
    def tripped(self) -> bool:
        return self.trip is not None

    def get_state(self) -> dict[str, Any]:
        """Return sampled-protection timer states; the owning unit exposes the latches."""
        return gather(self._timers)

    def set_state(self, values: Mapping[str, Any]) -> None:
        scatter(self._timers, values)

    def restore_latches(self, tripped: bool | None = None, fault: bool | None = None) -> None:
        """Restore the two unit-level latches from named initial states."""
        if fault is not None:
            self.fault = bool(fault)
        if tripped is True and self.trip is None:
            self.trip = TripEvent(math.nan, "restored", "latched trip loaded with the initial values")
        elif tripped is False:
            self.trip = None

    def _raise(self, name: str) -> None:
        if name not in self.stats.alarms:
            self.stats.alarms.append(name)

    def latch(self, t: float, cause: str, detail: str = "", *,
              alarm: str | None = None, fault: bool = False) -> bool:
        """Latch a trip and return whether this call was the first one."""
        if alarm is not None:
            self._raise(alarm)
        if fault:
            self.fault = True
        if self.trip is not None:
            return False
        self.trip = TripEvent(t, cause, detail)
        self._raise("TRIP_" + cause.upper())
        return True

    # ---------------------------------------------------------- fast path
    def check_fast(self, t: float, currents: Sequence[float]) -> bool:
        """Observe converter currents at a bridge edge and latch the fast over-current trip."""
        peak = peak_abs(currents) / self.i_base
        if not math.isfinite(peak):
            return False
        stats = self.stats
        stats.max_current_pu = max(stats.max_current_pu, peak)
        cfg = self.cfg.overcurrent
        if self.tripped or not (self.fast_enabled and peak > cfg.limit_pu):
            return False
        if stats.overcurrent_first_t is None:
            stats.overcurrent_first_t = t
        return self.latch(t, "overcurrent", "instantaneous converter-current limit",
                          alarm="OVERCURRENT", fault=True)

    # ---------------------------------------------------------- sampled path
    def _timed(self, t: float, met: bool, criterion: str, alarm: str, timer: str) -> bool:
        if met and getattr(self.stats, f"{criterion}_first_t") is None:
            setattr(self.stats, f"{criterion}_first_t", t)
            self._raise(alarm)
        held = self._timers[timer].update(met)
        return met and held >= self.cfg.hold

    def check_sampled(self, t: float, sample: ProtectionSample) -> bool:
        """Check ADC/controller-rate criteria and return whether they first tripped now."""
        cfg, stats = self.cfg, self.stats
        freq_dev = sample.frequency_deviation
        if not self.tripped and freq_dev is not None:
            if self._rocof is None:
                self._rocof = MovingWindow(self._rocof_count)
            old = self._rocof.push(freq_dev)
            if old is not None:
                rocof = (freq_dev - old) / self._rocof_window
                stats.rocof_max = max(stats.rocof_max, abs(rocof))
                if (cfg.rocof.enable and abs(rocof) >= cfg.rocof.limit
                        and sample.startup_complete):
                    if stats.rocof_first_t is None:
                        stats.rocof_first_t = t
                    self._raise("ROCOF")
        if self.tripped or not sample.startup_complete:
            return False
        rms = sample.rms_current_pu
        if self.rms_enabled and rms is not None and self._timed(
                t, rms > cfg.rms_current.limit_pu,
                "rms_current", "RMS_OVERCURRENT", "hold_rms"):
            return self.latch(t, "rms_current", f"RMS current={rms:.4f} pu held {cfg.hold} s")
        vdc_error = sample.dc_voltage_error_pu
        if cfg.dc_voltage.enable and vdc_error is not None and self._timed(
                t, abs(vdc_error) >= cfg.dc_voltage.limit_pu,
                "dc_voltage", "VDC_BAND", "hold_vdc"):
            return self.latch(t, "vdc", f"|vdc error|={abs(vdc_error):.4f} pu held {cfg.hold} s")
        if cfg.frequency.enable and freq_dev is not None and self._timed(
                t, abs(freq_dev) >= cfg.frequency.limit,
                "frequency", "FREQ_BAND", "hold_freq"):
            return self.latch(t, "freq", f"|f - f0|={abs(freq_dev):.4f} Hz held {cfg.hold} s")
        voltage = sample.terminal_voltage_pu
        if voltage is None:
            return False
        stats.terminal_voltage_min_pu = (
            voltage if stats.terminal_voltage_min_pu is None
            else min(stats.terminal_voltage_min_pu, voltage)
        )
        stats.terminal_voltage_max_pu = (
            voltage if stats.terminal_voltage_max_pu is None
            else max(stats.terminal_voltage_max_pu, voltage)
        )
        if cfg.undervoltage.enable and self._timed(
                t, voltage <= cfg.undervoltage.limit_pu,
                "undervoltage", "VOLTAGE_UNDER", "hold_uv"):
            return self.latch(t, "voltage_uv", f"|v|={voltage:.4f} pu <= {cfg.undervoltage.limit_pu} pu "
                              f"held {cfg.hold} s")
        if cfg.overvoltage.enable and self._timed(
                t, voltage >= cfg.overvoltage.limit_pu,
                "overvoltage", "VOLTAGE_OVER", "hold_ov"):
            return self.latch(t, "voltage_ov", f"|v|={voltage:.4f} pu >= {cfg.overvoltage.limit_pu} pu "
                              f"held {cfg.hold} s")
        return False

    def summary(self) -> dict[str, Any]:
        """Return the unified fast/sampled protection result."""
        stats, trip = self.stats, self.trip
        return {
            "tripped": int(trip is not None),
            "trip_time": trip.t if trip is not None and math.isfinite(trip.t) else None,
            "trip_cause": trip.cause if trip is not None else None,
            "max_current_pu": stats.max_current_pu,
            "overcurrent_first_t": stats.overcurrent_first_t,
            "rocof_max": stats.rocof_max,
            "terminal_voltage_min_pu": stats.terminal_voltage_min_pu,
            "terminal_voltage_max_pu": stats.terminal_voltage_max_pu,
            **{f"{criterion}_first_t": getattr(stats, f"{criterion}_first_t")
               for criterion in ("rms_current", "undervoltage", "overvoltage", "frequency", "dc_voltage", "rocof")},
            "alarms": list(stats.alarms),
        }
