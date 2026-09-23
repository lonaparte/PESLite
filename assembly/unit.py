"""Converter unit (DC link, bridge, R-L filter, sampler, control chain), its firmware and controller.

Plant quantities are in SI; the controller works in the unit's pu bases.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Callable, Mapping, Optional

import numpy as np

from ..phs.states import gather, scatter
from ..control import ControlGraph, default_wiring
from ..firmware import ComputationDelay, ModulationLimiter
from ..modulation import make_modulator, make_pwm_method
from ..params import SimulationParams, UnitParams
from ..power import Bridge, RLBranch, make_dclink
from .protocols import Controller, Delay, Measurement, Modulator, ControlOutput, ControlMeasurement, PWMMethod, SignalLimiter
from .events import UnitScenario
from ..sensing import MeasurementPorts, Sampler, SamplingWindow
from ..protection import Protection
from ..firmware.blocks import peak_abs
from ..firmware.transforms import abc2complex, complex2abc

__all__ = ["Unit", "ConverterFirmware", "UniteType", "make_controller"]


class Unit:
    """One converter unit built from its parameter section, connected to ``bus``."""

    def __init__(self, name: str, cfg: UnitParams, sim: SimulationParams, bus: Any,
                 ctrl: Optional[Controller] = None, modulator: Optional[Modulator] = None,
                 delay: Optional[Delay] = None) -> None:
        self.name = name
        self.cfg = cfg
        self.bus = bus
        self.scenario = sc = UnitScenario(cfg)
        base = cfg.base

        # ------------------------------------------------------------ power
        self.dclink = make_dclink(cfg.dclink, ramp=sc.startup)
        self.bridge = Bridge()
        self.branch_f = RLBranch(cfg.ac_filter.l_f, cfg.ac_filter.r_f)

        # ------------------------------------------------------------ sensing
        # ADC: instantaneous samples or window means; the window holds only the averaged channels
        meas = cfg.measurement
        channels: dict[str, complex | float] = {}
        if meas.average == "window":
            channels["v"], channels["i"] = 0j, 0j
        if meas.u_dc == "window":
            channels["dc"] = 0.0
        self.window: Optional[SamplingWindow] = SamplingWindow(
            meas.window_s if meas.window_s is not None else cfg.pwm.update_period, channels
        ) if channels else None
        self.T_avg: Optional[float] = self.window.length if self.window is not None else None
        self.ports = MeasurementPorts(
            u_g=lambda: bus.out.u, i_c=lambda: self.branch_f.out.i,
            i_c_state=lambda: self.branch_f.state.i, u_dc=lambda: self.dclink.out.u_dc,
            i_dc=lambda: self.dclink.inp.i_dc)
        self.sampler = Sampler(self.ports, self.window)
        self.breakers = [self.branch_f, self.dclink]
        self.breaker_open = False

        # ------------------------------------------------------------ control
        self.ctrl = ctrl if ctrl is not None else make_controller(cfg, sc)
        self.modulator = modulator if modulator is not None else make_modulator(cfg.pwm, base.f0, sim)
        self.delay = delay if delay is not None else ComputationDelay(cfg.delay.steps)
        c = cfg.control
        self.T_s = cfg.pwm.update_period                      # PWM publication interval
        self.samples_per_update = int(c.samples_per_update)
        self.T_samp = self.T_s / self.samples_per_update   # sampling period
        self.zoh = f"{name}.q"                   # model label of the bridge's held switching state

    # ---------------------------------------------------------------- assembly
    def subsystems(self) -> dict[str, Any]:
        """Return this unit's blocks keyed by namespaced name."""
        out = {f"{self.name}.branch_f": self.branch_f, f"{self.name}.bridge": self.bridge,
               f"{self.name}.dclink": self.dclink}
        return out

    def connections(self) -> dict:
        """Return this unit's internal wiring and its filter branch's connection to the bus."""
        out = {
            (self.branch_f, "u_to"): (self.bus, "u"),
            **self.bridge.connections(self.dclink, self.branch_f),
        }
        return out

    def zoh_connections(self) -> dict:
        return self.bridge.zoh_connections(self.zoh)

    @property
    def bus_name(self) -> str:
        return self.cfg.bus

    @property
    def injection(self) -> tuple:
        """Current this unit feeds into its bus: the filter branch current, positive into the node."""
        return (self.branch_f, "i")

    def aliases(self) -> dict[str, str]:
        """Return the aliases ``<unit>.i_c`` (filter current) and ``<unit>.u_dc`` (dc-link voltage)."""
        out = {f"{self.name}.i_c": f"{self.name}.branch_f.i"}
        if self.cfg.dclink.capacitor is not None:
            out[f"{self.name}.u_dc"] = f"{self.name}.dclink.u_C"
        return out

    def presets(self, t: float) -> dict[str, Any]:
        """Return the ``initial.states`` keywords of this unit: ``rated`` (dc voltage, V)."""
        return {"rated": self.cfg.dclink.vdc_ref}

    # ---------------------------------------------------------------- states
    def get_state(self) -> dict[str, Any]:
        return {"breaker_open": self.breaker_open}

    def set_state(self, values: Mapping[str, Any]) -> None:
        if values.get("breaker_open", False) and not self.breaker_open:
            self.trip()

    # ---------------------------------------------------------------- the loop
    def measure(self, t: float) -> Measurement:
        """Return the sampled measurement (SI) at ``t``; model outputs must be synced to ``t``."""
        return self.sampler.measure(t)

    def open_window(self) -> None:
        """Open the next averaging window at the current instant; outputs must be synced."""
        self.sampler.open_window()

    def _measured(self) -> dict[str, complex | float]:
        ports = self.ports
        return {"v": ports.u_g(), "i": ports.i_c(), "dc": ports.u_dc()}

    def seed_window(self) -> None:
        """Seed the averaging window at the start instant; outputs must be synced."""
        if self.window is not None:
            self.window.seed(self._measured())

    def accumulate(self, dt: float) -> None:
        """Advance the averaging window over the interval of length ``dt`` (s) ending now."""
        if self.window is not None:
            self.window.accumulate(dt, self._measured())

    def phase_currents(self) -> np.ndarray:
        """Return the instantaneous phase currents (A) from the state."""
        return self.sampler.phase_currents()

    def trip(self) -> None:
        """Open this unit's breakers (filter branch and dc source); other units keep running.

        The caller must repack the solver state vector afterwards.
        """
        self.breaker_open = True
        for breaker in self.breakers:
            breaker.open_breaker()

    def signals(self) -> dict[str, float | complex]:
        """Return this unit's recorded signals (SI); outputs must be synced first."""
        ports = self.ports
        return {f"{self.name}.i_c": ports.i_c(), f"{self.name}.u_g": ports.u_g(),
                f"{self.name}.u_dc": ports.u_dc(), f"{self.name}.i_dc": ports.i_dc()}


class _ConfiguredLimiter:
    """Distinguish an omitted limiter (use the configuration) from explicit None."""


_DEFAULT_LIMITER = _ConfiguredLimiter()


def _signals(value: Any, component: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError
        m = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(_signals_message(component)) from exc
    if m.shape != (3,) or not all(map(math.isfinite, m.tolist())):
        raise ValueError(_signals_message(component))
    return m


def _signals_message(component: str) -> str:
    return f"{component} must return three finite real modulating signals with shape (3,)"


class ConverterFirmware:
    """Controller firmware: SI-to-pu conversion, protection, PWM method, signal limiter and anti-windup.

    pwm_method: ``None`` uses ``cfg.pwm.method``.
    limiter: omitted uses ``ModulationLimiter(cfg.pwm.modulation_limit)``; ``None`` disables the limit
    and its anti-windup.
    """

    def __init__(self, cfg: UnitParams, scenario: UnitScenario, u_init_ab: complex,
                 pwm_method: Optional[PWMMethod] = None, *,
                 limiter: SignalLimiter | None | _ConfiguredLimiter = _DEFAULT_LIMITER) -> None:
        self.T_s = cfg.pwm.update_period
        self.v_base, self.i_base = cfg.base.v_phase_peak, cfg.base.i_phase_peak
        self.v_dc_base = cfg.dc_base.v
        self.u_dc_ref = cfg.dclink.vdc_ref
        self.vdc_ref_pu = cfg.control.references.vdc_ref_pu
        self.pwm_method = make_pwm_method(cfg.pwm.method) if pwm_method is None else pwm_method
        self.limiter = ModulationLimiter(cfg.pwm.modulation_limit) if limiter is _DEFAULT_LIMITER else limiter
        if not callable(self.pwm_method):
            raise TypeError("pwm_method must be callable")
        if self.limiter is not None and not callable(self.limiter):
            raise TypeError("limiter must be callable or None")
        self.protection = Protection(cfg.protection, self.T_s, scenario.arm_time)
        self.n_updates = 0
        self.n_saturated = 0
        self.first_saturation_t = -1.0
        self.saturated = False
        self._memo = None  # cached evaluation for the current control instant
        # start-up modulation from u_init_ab and the rated dc voltage
        self.align_startup(u_init_ab, self.u_dc_ref)

    # ------------------------------------------------------------ interface
    def initial_duty(self) -> np.ndarray:
        return 0.5 * (1.0 + self.m_init)

    def align_startup(self, u_ab: complex, u_dc: float) -> None:
        """Set the start-up modulation that reproduces ``u_ab`` (V, alpha-beta) from the dc voltage ``u_dc`` (V)."""
        self.modulate(0.0, u_ab / self.v_base, 0.0, u_dc / self.v_dc_base, count=False)
        self.m_init = self.m_abc.copy()

    def _parts(self) -> dict[str, Any]:
        return {"prot": self.protection}

    def get_state(self) -> dict[str, Any]:
        return gather(self._parts())

    def set_state(self, values: Mapping[str, Any]) -> None:
        scatter(self._parts(), values)

    @property
    def tripped(self) -> bool:
        return self.protection.tripped

    def new_instant(self) -> None:
        """Start a new control instant (clear the cached evaluation)."""
        self._memo = None

    def raw_log(self, meas: Measurement, theta: float) -> dict[str, float]:
        """Return the pre-average sample in the ``theta`` frame as ``*_raw_pu`` log entries (empty if not averaged)."""
        out: dict[str, float] = {}
        if meas.u_g_raw is not None and (meas.u_g_raw != meas.u_g or meas.i_c_raw != meas.i_c):
            rot = cmath.exp(-1j * theta)
            v, i = meas.u_g_raw * rot, meas.i_c_raw * rot
            out.update({"vd_raw_pu": v.real / self.v_base, "vq_raw_pu": v.imag / self.v_base,
                        "id_raw_pu": i.real / self.i_base, "iq_raw_pu": i.imag / self.i_base})
        if meas.u_dc_raw is not None and meas.u_dc_raw != meas.u_dc:
            out["vdc_raw_pu"] = meas.u_dc_raw / self.v_dc_base
        return out

    def fast_check(self, t: float, i_abc: np.ndarray) -> bool:
        """Check the instantaneous over-current criterion between samples; ``i_abc`` in A."""
        peak = peak_abs(i_abc) / self.i_base
        return self.protection.check_current(t, peak)

    # -------------------------------------------------------- per sample
    def measure(self, meas: Measurement) -> ControlMeasurement:
        """Convert an SI :class:`Measurement` into a pu :class:`ControlMeasurement`."""
        if not isinstance(meas, Measurement):
            raise TypeError("firmware.measure requires an SI Measurement, not an already normalized sample")
        return ControlMeasurement(meas.t, meas.u_g / self.v_base, meas.i_c / self.i_base,
                                  meas.u_dc / self.v_dc_base, meas.i_abc / self.i_base)

    def sense(self, meas: ControlMeasurement, theta: float) -> tuple[complex, complex]:
        """Return voltage and current (pu) rotated into the frame of angle ``theta`` (rad)."""
        rot = cmath.exp(-1j * theta)
        return meas.u_g * rot, meas.i_c * rot

    def protect(self, t: float, meas: ControlMeasurement, vac_pu: float, freq_dev_hz: float) -> bool:
        """Run all protection criteria for this sample; return the latched trip state."""
        self.protection.check_current(t, peak_abs(meas.i_abc))
        vdc_err_pu = meas.u_dc - self.vdc_ref_pu
        self.protection.check_sampled(t, vac_pu, freq_dev_hz, vdc_err_pu)
        return self.protection.tripped

    def modulate(self, t: float, u_cmd_dq: complex, theta: float, u_dc: float, cc: Any = None,
                 extra_dq: Optional[complex] = None, *, count: bool = True) -> np.ndarray:
        """Convert a dq voltage command to duty ratios (rotate, PWM method, limiter) with current-loop anti-windup.

        u_cmd_dq, extra_dq: voltage command (pu, AC base); theta: frame angle (rad); u_dc: dc voltage (pu, DC base).
        cc: current controller for anti-windup (``"conditional"`` or ``"backcalc"``).
        count: ``False`` leaves the publication and saturation counters unchanged.
        """
        rot = complex(math.cos(theta), math.sin(theta))
        u_dc = u_dc * self.v_dc_base  # PWM receives DC volts; loop feedback stays pu
        u_dq = u_cmd_dq if extra_dq is None else u_cmd_dq + extra_dq

        memo_key = (theta, u_dc)

        def evaluate(command: complex) -> tuple[np.ndarray, bool]:
            # reuse the cached result for the same command at this instant
            memo = self._memo
            if memo is not None and memo[0] == command and memo[1] == memo_key:
                return memo[2], memo[3]
            m, saturated = _evaluate(command)
            self._memo = (command, memo_key, m, saturated)
            return m, saturated

        def _evaluate(command: complex) -> tuple[np.ndarray, bool]:
            u_ab = command * rot * self.v_base
            if u_dc > 0.0:
                m = _signals(self.pwm_method(u_ab, u_dc), "pwm_method")
            else:
                u = complex2abc(u_ab)
                peak = float(np.max(np.abs(u)))
                m = u / peak if peak > 0.0 else np.zeros(3)
            if self.limiter is None:
                return m, False
            limited = _signals(self.limiter(m.copy()), "limiter")
            saturated = limited.tolist() != m.tolist() or (u_dc <= 0.0 and u_ab != 0j)  # both finite, shape (3,)
            return limited, saturated

        m_abc, saturated = evaluate(u_dq)
        if cc is not None and cc.antiwindup == "conditional" and saturated:
            u_dq = cc.rollback() if extra_dq is None else cc.rollback() + extra_dq
            m_abc, saturated = evaluate(u_dq)
        if cc is not None and cc.antiwindup == "backcalc" and saturated:
            u_lim_dq = abc2complex(m_abc) * u_dc / (2.0 * self.v_base) * complex(math.cos(theta), -math.sin(theta))
            cc.backcalculate(u_dq, u_lim_dq)
        self.m_abc, self.saturated = m_abc, saturated
        if count:
            self.n_updates += 1
            if saturated:
                self.n_saturated += 1
                if self.first_saturation_t < 0.0:
                    self.first_saturation_t = t
        return 0.5 * (1.0 + self.m_abc)

    @property
    def saturation_fraction(self) -> float:
        return self.n_saturated / max(1, self.n_updates)

    def summary(self) -> dict[str, float | str]:
        st = self.protection.stats
        trip = self.protection.trip
        return {
            "tripped": float(trip is not None),
            "trip_time_s": trip.t if trip else -1.0,
            "trip_cause": trip.cause if trip else "none",
            "max_current_pu": st.max_current_pu,
            "modulation_saturation_fraction": self.saturation_fraction,
            "modulation_saturation_first_t_s": self.first_saturation_t,
            "rocof_max_hz_s": st.rocof_max_hz_s,
            "vac_min_pu": st.vac_min_pu,
            "vac_max_pu": st.vac_max_pu,
            "alarms": "|".join(st.alarms) if st.alarms else "none",
        }


class UniteType:
    """Configurable controller of one unit: a control-loop graph plus the common firmware output stage.

    cfg.control.type: ``"gfl"``, ``"gfm"`` or ``"custom"`` (no default wiring).
    loop_overrides: replacement algorithms by configured loop name, e.g. ``{"pll": my_pll}``.
    ``update()`` advances the due loops; ``__call__`` is the PWM output event.
    """
    def __init__(self, cfg, scenario=None, pwm_method=None, *, limiter=_DEFAULT_LIMITER,
                 loop_overrides=None):
        self.p = cfg
        self.scenario = scenario if scenario is not None else UnitScenario(cfg)
        wires, outputs = default_wiring(cfg, cfg.control.type)
        self.graph = ControlGraph(cfg, self.scenario, wires, outputs, loop_overrides)
        self.periods = self.graph.periods
        self.T_s = cfg.pwm.update_period
        self.fw = ConverterFirmware(cfg, self.scenario, complex(cfg.base.v_phase_peak),
                                    pwm_method=pwm_method, limiter=limiter)
        self.v_base, self.i_base, self.w0 = cfg.base.v_phase_peak, cfg.base.i_phase_peak, cfg.base.w0
        self.theta, self.omega = self.initial_sync()
        self.u_cmd = 1.0 + 0j
        self.command_theta = self.theta
        self.v_dq = self.i_dq = 0j
        self.last_log = {}
        for name, node in self.graph.nodes.items():
            kind = cfg.control.loops[name].type
            alias = {"srf_pll": "pll", "dq_current_pi": "cc", "dc_voltage_pi": "dc"}.get(kind)
            if kind in ("psc", "droop", "vsg", "dvoc", "matching"):
                alias = "law"
            if alias:
                setattr(self, alias, node.obj)
        # graph structure lookups
        graph, loops = self.graph, cfg.control.loops
        self._freezable = tuple(n for n in graph.nodes.values() if hasattr(n, "frozen"))
        self._terminal = graph.outputs["u_dq"].partition(".")[0]
        self._terminal_key = f"{self._terminal}.u_dq"
        self._terminal_node = graph.nodes.get(self._terminal)
        self._terminal_in_graph = self._terminal in graph.nodes
        self._terminal_is_cc = (self._terminal_node is not None
                                and loops[self._terminal].type == "dq_current_pi")
        self._frame_key = f"{graph.outputs['theta'].partition('.')[0]}.frame"
        self._frame_cc = next((n for n in graph.nodes if loops[n].type == "dq_current_pi"), None)
        self._is_gfl = cfg.control.type == "gfl"
        self._log_cc = next((n for n, p in loops.items() if p.type == "dq_current_pi"), None)
        power = next((n for n, p in loops.items() if p.type == "power"), None)
        sync = next((n for n, p in loops.items() if p.type in ("psc", "droop", "vsg", "dvoc", "matching")), None)
        self._p_key, self._q_key, self._v_ref_key = f"{power}.p", f"{power}.q", f"{sync}.v_ref"

    def describe(self):
        return self.graph.describe()

    @property
    def next_event(self):
        return self.graph.next_event

    def reset_clocks(self, t):
        self.graph.reset_clocks(t)

    def initial_sync(self):
        values = {**self.graph.values, **{f"references.{k}": v for k, v in self.graph.references.items()}}
        return float(values[self.graph.outputs["theta"]]), float(values[self.graph.outputs["omega"]])

    def initial_duty(self):
        return self.fw.initial_duty()

    def align_startup(self, u_ab, u_dc):
        self.fw.align_startup(u_ab, u_dc)
        self.theta, self.omega = self.initial_sync()
        self.command_theta = self.theta
        self.u_cmd = u_ab / self.v_base * cmath.exp(-1j * self.command_theta)

    @property
    def tripped(self):
        return self.fw.tripped

    def fast_check(self, t, i_abc):
        return self.fw.fast_check(t, i_abc)

    def update(self, t, meas):
        return self._update_pu(t, self.fw.measure(meas))

    def _update_pu(self, t, control_meas):
        self.fw.new_instant()
        tripped = self.tripped
        for node in self._freezable:
            node.frozen = tripped
        return self.graph.update(t, control_meas, finalize=self._accept_command)

    def _accept_command(self, t, meas, updated):
        """Update the held angle, frequency and voltage command, applying anti-windup feedback."""
        graph = self.graph
        self.theta = float(graph.output("theta", meas))
        self.omega = float(graph.output("omega", meas))
        if self._terminal in updated or (updated and not self._terminal_in_graph):
            self.u_cmd = complex(graph.output("u_dq", meas))
            self.command_theta = self.theta
            node = self._terminal_node
            cc = node.obj if self._terminal_is_cc else None
            extra = node.extra if cc is not None else 0j
            self.fw.modulate(t, self.u_cmd - extra, self.command_theta, meas.u_dc,
                             cc=cc, extra_dq=extra, count=False)
            if cc is not None and cc.antiwindup == "conditional":
                self.u_cmd = node.after_rollback()
                graph.values[self._terminal_key] = self.u_cmd

    def __call__(self, t, meas):
        control_meas = self.fw.measure(meas)  # SI -> pu
        self._update_pu(t, control_meas)
        frame = self.graph.values.get(self._frame_key, self.theta)
        if self._frame_cc is not None:
            frame = self.graph.input(self._frame_cc, "frame", control_meas)
        self.v_dq, self.i_dq = self.fw.sense(control_meas, frame)
        freq_dev_hz = (self.omega - self.w0) / (2 * math.pi)
        tripped = self.fw.protect(t, control_meas, abs(self.v_dq), freq_dev_hz)
        duty = self.fw.modulate(t, 0j if tripped else self.u_cmd, self.command_theta, control_meas.u_dc)
        log = {"id_pu": self.i_dq.real, "iq_pu": self.i_dq.imag,
               "vd_pu": self.v_dq.real, "vq_pu": self.v_dq.imag,
               "vac_pu": abs(self.v_dq), "vdc_pu": control_meas.u_dc,
               "m_max": peak_abs(self.fw.m_abc), "in_service": 0.0 if tripped else 1.0,
               **self.fw.raw_log(meas, frame)}
        if self._is_gfl:
            cc_name = self._log_cc
            id_ref = self.graph.input(cc_name, "id_ref", control_meas) if cc_name else 0.0
            log.update(idref_pu=id_ref, pll_freq_dev_hz=freq_dev_hz,
                       pll_angle_rel=(self.theta - self.w0 * t + math.pi) % (2 * math.pi) - math.pi)
        else:
            values = self.graph.values
            refs = self.graph.references
            pr, qr, vr = self.scenario.setpoints(t, refs["p_ref_pu"], refs["q_ref_pu"], refs["v_ref_pu"])
            log.update(p_pu=values.get(self._p_key, 0.0), q_pu=values.get(self._q_key, 0.0),
                       p_ref_pu=pr, v_mag_pu=abs(self.v_dq),
                       v_ref_pu=values.get(self._v_ref_key, 1.0),
                       freq_dev_hz=freq_dev_hz,
                       angle_rel=(self.theta - self.w0 * t + math.pi) % (2 * math.pi) - math.pi)
        self.last_log = log
        return ControlOutput(duty, tripped, log, theta=self.theta, omega=self.omega)

    def get_state(self):
        return {**self.graph.get_state(), **self.fw.get_state(), "command.u_dq_pu": self.u_cmd,
                "command.theta": self.command_theta, "command.omega": self.omega}

    def set_state(self, values):
        rest, fw = {}, {}
        for name, value in values.items():
            if name.startswith("prot."):
                fw[name] = value
            elif name == "command.u_dq_pu":
                self.u_cmd = complex(value)
            elif name == "command.theta":
                self.command_theta = float(value)
            elif name == "command.omega":
                self.omega = float(value)
            else:
                rest[name] = value
        self.graph.set_state(rest)
        self.fw.set_state(fw)
        self.theta, self.omega = self.initial_sync()

    def summary(self):
        summary = self.fw.summary()
        if hasattr(self, "dc"):
            name = next(n for n, p in self.p.control.loops.items() if p.type == "dc_voltage_pi")
            summary["idref_limit_fraction"] = self.dc.n_clamped / max(1, self.graph.nodes[name].n_updates)
        if hasattr(self, "law"):
            summary["law"] = next(p.type for p in self.p.control.loops.values() if p.type in ("psc", "droop", "vsg", "dvoc", "matching"))
        return summary


def make_controller(cfg, scenario=None, **kwargs):
    """Build a UniteType instance using cfg.control.type and its loops."""
    return UniteType(cfg, scenario=scenario, **kwargs)
