"""Protection relay: over-current, ac-voltage, frequency and dc-voltage trips and a ROCOF alarm.

Trips are enabled from ``arm_time`` (s); timed criteria must hold for ``hold_s`` (s).
Named states: ``tripped`` and the hold timers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..params import ProtectionParams
from ..firmware.blocks import HoldTimer, MovingWindow
from ..phs.states import gather, scatter

__all__ = ["Protection", "TripEvent", "ProtectionStats"]


@dataclass
class TripEvent:
    t: float
    cause: str
    detail: str = ""


@dataclass
class ProtectionStats:
    i_alarm_first_t: float = -1.0
    i_alarm_steps: int = 0
    vac_uv_first_t: float = -1.0
    vac_ov_first_t: float = -1.0
    freq_band_first_t: float = -1.0
    vdc_band_first_t: float = -1.0
    rocof_first_t: float = -1.0
    rocof_max_hz_s: float = 0.0
    max_current_pu: float = 0.0
    vac_min_pu: float = -1.0
    vac_max_pu: float = -1.0
    alarms: list[str] = field(default_factory=list)


class Protection:
    def __init__(self, cfg: ProtectionParams, T: float, arm_time: float) -> None:
        self.cfg, self.T, self.arm_time = cfg, T, arm_time
        self.trip: TripEvent | None = None
        self.stats = ProtectionStats()
        self._t_uv, self._t_ov = HoldTimer(T), HoldTimer(T)
        self._t_freq, self._t_vdc = HoldTimer(T), HoldTimer(T)
        n = max(1, int(cfg.rocof_window_s / T + 0.5))
        self._rocof = MovingWindow(n)
        self._rocof_window = n * T

    @property
    def tripped(self) -> bool:
        return self.trip is not None

    def _timers(self) -> dict[str, HoldTimer]:
        return {"hold_uv": self._t_uv, "hold_ov": self._t_ov, "hold_freq": self._t_freq, "hold_vdc": self._t_vdc}

    def get_state(self) -> dict[str, Any]:
        return {"tripped": self.tripped, **gather(self._timers())}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if "tripped" in values:
            if values["tripped"] and self.trip is None:
                self.trip = TripEvent(math.nan, "restored", "latched trip loaded with the initial values")
            elif not values["tripped"]:
                self.trip = None
        scatter(self._timers(), values)

    def _raise(self, name: str) -> None:
        if name not in self.stats.alarms:
            self.stats.alarms.append(name)

    def _do_trip(self, t: float, cause: str, detail: str) -> None:
        if self.trip is None:
            self.trip = TripEvent(t, cause, detail)
            self._raise("TRIP_" + cause.upper())

    # ---------------------------------------------------------------- fast
    def check_current(self, t: float, peak_pu: float) -> bool:
        """Check the phase-current peak ``peak_pu`` (pu); return ``True`` if this call trips."""
        self.stats.max_current_pu = max(self.stats.max_current_pu, peak_pu)
        if self.cfg.i_alarm_pu > 0.0 and peak_pu > self.cfg.i_alarm_pu:
            if self.stats.i_alarm_steps == 0:
                self.stats.i_alarm_first_t = t
                self._raise("OVERCURRENT")
            self.stats.i_alarm_steps += 1
            if not self.tripped and t >= self.arm_time:
                self._do_trip(t, "overcurrent", f"|i|={peak_pu:.4f} pu > {self.cfg.i_alarm_pu} pu")
                return True
        return False

    # ---------------------------------------------------------- sampled
    def check_sampled(self, t: float, vac_pu: float, freq_dev_hz: float, vdc_err_pu: float) -> None:
        """Check the timed criteria and the ROCOF alarm; call once per control period.

        ``vac_pu`` ac voltage magnitude (pu), ``freq_dev_hz`` frequency deviation (Hz),
        ``vdc_err_pu`` dc-voltage error (pu).
        """
        cfg, armed = self.cfg, t >= self.arm_time
        if not self.tripped:
            old = self._rocof.push(freq_dev_hz)
            if old is not None:
                rocof = (freq_dev_hz - old) / self._rocof_window
                self.stats.rocof_max_hz_s = max(self.stats.rocof_max_hz_s, abs(rocof))
                if cfg.rocof_alarm_hz_s > 0.0 and abs(rocof) >= cfg.rocof_alarm_hz_s and armed:
                    if self.stats.rocof_first_t < 0.0:
                        self.stats.rocof_first_t = t
                    self._raise("ROCOF")
        if self.tripped or not armed:
            return
        if cfg.vdc_band_pu > 0.0:
            in_band = abs(vdc_err_pu) >= cfg.vdc_band_pu
            if in_band and self.stats.vdc_band_first_t < 0.0:
                self.stats.vdc_band_first_t = t
                self._raise("VDC_BAND")
            if self._t_vdc.update(in_band) >= cfg.hold_s:
                self._do_trip(t, "vdc", f"|vdc error|={abs(vdc_err_pu):.4f} pu held {cfg.hold_s} s")
                return
        if cfg.freq_band_hz > 0.0:
            in_band = abs(freq_dev_hz) >= cfg.freq_band_hz
            if in_band and self.stats.freq_band_first_t < 0.0:
                self.stats.freq_band_first_t = t
                self._raise("FREQ_BAND")
            if self._t_freq.update(in_band) >= cfg.hold_s:
                self._do_trip(t, "freq", f"|f_pll - f0|={abs(freq_dev_hz):.4f} Hz held {cfg.hold_s} s")
                return
        if self.stats.vac_min_pu < 0.0 or vac_pu < self.stats.vac_min_pu:
            self.stats.vac_min_pu = vac_pu
        self.stats.vac_max_pu = max(self.stats.vac_max_pu, vac_pu)
        if cfg.vac_uv_pu > 0.0:
            under = vac_pu <= cfg.vac_uv_pu
            if under and self.stats.vac_uv_first_t < 0.0:
                self.stats.vac_uv_first_t = t
                self._raise("VAC_UNDER")
            if self._t_uv.update(under) >= cfg.hold_s:
                self._do_trip(t, "vac_uv", f"|v|={vac_pu:.4f} pu <= {cfg.vac_uv_pu} pu held {cfg.hold_s} s")
                return
        if cfg.vac_ov_pu > 0.0:
            over = vac_pu >= cfg.vac_ov_pu
            if over and self.stats.vac_ov_first_t < 0.0:
                self.stats.vac_ov_first_t = t
                self._raise("VAC_OVER")
            if self._t_ov.update(over) >= cfg.hold_s:
                self._do_trip(t, "vac_ov", f"|v|={vac_pu:.4f} pu >= {cfg.vac_ov_pu} pu held {cfg.hold_s} s")
                return
