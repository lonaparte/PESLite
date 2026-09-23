"""Cross-field checks of a constructed parameter tree (read-only).

Covers topology, DC link, control, timing, solver settings and initial values.
"""

from __future__ import annotations

import math
import warnings

from ..phs.protocols import ConfigError
from ..phs.solvers import ADAPTIVE_METHODS, FIXED_METHODS

from .schema import Params

__all__ = ["validate"]


def _whole(r: float) -> bool:
    """Is ``r`` a whole number of at least one?"""
    n = round(r)
    return n >= 1 and abs(r - n) <= 1e-9 * max(1.0, r)


def _misaligned(name: str, r: float, where: str, both_ways: bool = False) -> None:
    """Warn if a ratio of periods is not a whole number (optionally in either direction)."""
    if _whole(r) or (both_ways and _whole(1.0 / r)):
        return
    warnings.warn(f"{where}: {name} = {r:.6g} is not a whole number"
                  f"{' in either direction' if both_ways else ''}, so the two grids do not line "
                  f"up; the periods are used as configured", stacklevel=3)


def _dclink(cfg, where):
    cap, source = cfg.capacitor, cfg.source
    if not math.isfinite(cfg.vdc_ref) or cfg.vdc_ref <= 0:
        raise ConfigError(f"{where}.vdc_ref must be finite and positive")
    if cap is not None:
        if not math.isfinite(cap.c) or cap.c <= 0:
            raise ConfigError(f"{where}.capacitor.c must be finite and positive")
        if not math.isfinite(cap.r_esr) or cap.r_esr < 0:
            raise ConfigError(f"{where}.capacitor.r_esr must be finite and nonnegative")
    for key in ("i", "k", "v", "r"):
        if not math.isfinite(getattr(source, key)):
            raise ConfigError(f"{where}.source.{key} must be finite")
    if source.v < 0 or source.r < 0:
        raise ConfigError(f"{where}.source.v and r must be nonnegative")
    allowed = {"current": {"i", "k"}, "voltage": {"v", "r"}, "none": set()}[source.type]
    for key, default in {"i": 0.0, "k": 0.0, "v": 0.0, "r": 0.0}.items():
        if key not in allowed and getattr(source, key) != default:
            raise ConfigError(f"{where}.source.{key} is not used by source.type = {source.type!r}")
    if cap is None and source.type != "voltage":
        raise ConfigError(f"{where}: without a capacitor, a voltage source is required")
    if cap is not None and source.type == "voltage" and cap.r_esr + source.r <= 0:
        raise ConfigError(f"{where}: a voltage source with a capacitor requires positive source resistance or ESR")


def _pwm(w, base, where: str) -> None:
    """Check switching frequency, update period and synchronous pulse ratio."""
    if not math.isfinite(w.f_sw) or w.f_sw <= 0.0:
        raise ConfigError(f"{where}.f_sw must be finite and positive, got {w.f_sw}")
    T_pwm = w.effective_update_period
    if not math.isfinite(T_pwm) or T_pwm <= 0:
        raise ConfigError(f"{where}.update_period must be finite and positive")
    if w.sync == "synchronous":
        ratio = w.f_sw / base.f0
        if abs(ratio - round(ratio)) > 1e-9 * max(1.0, ratio):
            raise ConfigError(f"{where}.sync = 'synchronous' needs an integer pulse ratio "
                              f"pwm.f_sw / base.f0, got {w.f_sw} / {base.f0} = {ratio}")
    _misaligned("pwm.switching_period / pwm.update_period", w.switching_period / T_pwm,
                f"{where}.update_period", both_ways=True)


def _control(c, dclink, where: str) -> None:
    """Check typed control loops and their requirements on the DC side."""
    if not c.loops:
        raise ConfigError(f"{where}.loops must contain at least one loop")
    for name, cfg in c.loops.items():
        if name in {"measurement", "references", "held", "clock", "command", "prot"}:
            raise ConfigError(f"{where}.loops.{name}: reserved loop name")
        T = cfg.period
        if T is not None and (not math.isfinite(T) or T <= 0):
            raise ConfigError(f"{where}.loops.{name}.period must be finite and positive")
        if T is None and cfg.type != "virtual_impedance":
            raise ConfigError(f"{where}.loops.{name}.period is required")
        for key, value in vars(cfg).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ConfigError(f"{where}.loops.{name}.{key} must be finite")
        if cfg.type == "virtual_admittance" and cfg.x_v_pu <= 0:
            raise ConfigError(f"{where}.loops.{name}.x_v_pu must be positive")
        if cfg.type == "matching" and dclink.capacitor is None:
            raise ConfigError(f"{where}: matching control needs dclink.capacitor")
        if cfg.type == "matching" and cfg.k_theta_pu is None and c.references.vdc_ref_pu <= 0:
            raise ConfigError(f"{where}.references.vdc_ref_pu must be positive for automatic matching gain")
    for name, value in vars(c.references).items():
        if name == "omega" and value is None:
            continue
        if not isinstance(name, str) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(f"{where}.references: expected finite scalar references")


def _sampling(c, m, T_pwm: float, where: str) -> None:
    """Check the ADC sampling period and averaging window."""
    if c.sampling_period is not None and (not math.isfinite(c.sampling_period) or c.sampling_period <= 0):
        raise ConfigError(f"{where}.control.sampling_period must be positive")
    if c.sampling_period is not None and c.sampling_period > T_pwm * (1 + 1e-9):
        raise ConfigError(f"{where}.control.sampling_period is longer than pwm.update_period")
    sampling_period = c.sampling_period if c.sampling_period is not None else T_pwm
    _misaligned("pwm.update_period / control.sampling_period", T_pwm / sampling_period,
                f"{where}.control.sampling_period")
    if m.window_s is not None and m.window_s <= 0.0:
        raise ConfigError(f"{where}.measurement.window_s must be > 0, got {m.window_s}")
    if m.average == "window" or m.u_dc == "window":
        window = m.window_s if m.window_s is not None else T_pwm
        if window > T_pwm * (1.0 + 1e-9):
            raise ConfigError(
                f"{where}.measurement.window_s = {window} is longer than the PWM update period {T_pwm}")
        if sampling_period < T_pwm * (1.0 - 1e-9):
            raise ConfigError(
                f"{where}: measurement.average = 'window' cannot be combined with a sampling period "
                f"shorter than the PWM update period")


def _unit(u, base, where: str) -> None:
    """Check one converter unit."""
    if u.s_base is not None and (not math.isfinite(u.s_base) or u.s_base <= 0.0):
        raise ConfigError(f"{where}.s_base must be finite and positive, got {u.s_base}")
    if u.ac_filter.l_f <= 0.0:
        raise ConfigError(f"{where}.ac_filter.l_f must be > 0")
    _dclink(u.dclink, f"{where}.dclink")
    _pwm(u.pwm, base, f"{where}.pwm")
    _control(u.control, u.dclink, f"{where}.control")
    _sampling(u.control, u.measurement, u.pwm.effective_update_period, where)
    if u.delay.steps < 0:
        raise ConfigError(f"{where}.delay.steps must be >= 0")


def _network(p: Params) -> None:
    """Check element identities, network values and bus references."""
    if not p.buses:
        raise ConfigError("buses: a system needs at least one bus")
    if not p.units:
        raise ConfigError("units: a system needs at least one converter")
    seen: dict[str, str] = {}
    for kind in ("buses", "branches", "sources", "units"):
        for name in getattr(p, kind):
            if name in seen:
                raise ConfigError(f"{kind}.{name}: the name is already used by {seen[name]}; element "
                                  f"names are unique across buses, branches, sources and units")
            seen[name] = f"{kind}.{name}"
    for name, bus in p.buses.items():
        if bus.c <= 0.0:
            raise ConfigError(f"buses.{name}.c must be > 0: the shunt capacitor is what makes the "
                              f"bus voltage a state (a bus without one would need a node-voltage solver)")
    def _bus_of(where: str, bus: str) -> None:
        if bus not in p.buses:
            raise ConfigError(f"{where}: unknown bus {bus!r}; known: {sorted(p.buses)}")
    for name, br in p.branches.items():
        _bus_of(f"branches.{name}.from_bus", br.from_bus)
        _bus_of(f"branches.{name}.to_bus", br.to_bus)
        if br.from_bus == br.to_bus:
            raise ConfigError(f"branches.{name}: both ends are {br.from_bus!r}")
        if br.l <= 0.0:
            raise ConfigError(f"branches.{name}.l must be > 0")
    for name, src in p.sources.items():
        _bus_of(f"sources.{name}.bus", src.bus)
        if src.l <= 0.0:
            raise ConfigError(f"sources.{name}.l must be > 0: a source is an emf behind an impedance")
    for name, u in p.units.items():
        _bus_of(f"units.{name}.bus", u.bus)


def _solver(s, bridge: str) -> None:
    """Check solver methods, bridge compatibility and subsystem step ratios."""
    if s.type == "fixed" and s.method not in FIXED_METHODS:
        raise ConfigError(f"simulation.solver.method {s.method!r} is not one of {FIXED_METHODS}")
    if s.type == "adaptive" and s.method not in ADAPTIVE_METHODS:
        raise ConfigError(f"simulation.solver.method {s.method!r} is not one of {ADAPTIVE_METHODS}")
    if bridge == "step_averaged" and s.type != "fixed":
        raise ConfigError("simulation.bridge = 'step_averaged' requires the fixed-step solver")
    if s.sweeps < 1:
        raise ConfigError("simulation.solver.sweeps must be >= 1")
    if s.linearisations < 0:
        raise ConfigError("simulation.solver.linearisations must be >= 0")
    for name, value in s.subsystems.items():
        where = f"simulation.solver.subsystems.{name}"
        if s.type != "fixed":
            raise ConfigError(f"{where}: subsystems on their own step need the fixed-step solver")
        step, method = value, None
        if isinstance(value, dict):
            unknown = set(value) - {"step", "method"}
            if unknown or "step" not in value:
                raise ConfigError(f"{where}: expected {{step, method}} (step required), got {sorted(value)}")
            step, method = value["step"], value.get("method")
        if (isinstance(step, bool) or not isinstance(step, (int, float))
                or not math.isfinite(step) or step <= 0):
            raise ConfigError(f"{where}.step: expected a positive number (relative to dt), got {step!r}")
        if step >= 1 and abs(step - round(step)) > 1e-9:
            raise ConfigError(f"{where}.step: a step longer than dt must be an integer multiple, got {step}")
        if step < 1 and abs(1 / step - round(1 / step)) > 1e-9:
            raise ConfigError(f"{where}.step: a step shorter than dt must be 1/N with N an integer, got {step}")
        if method is not None and method not in FIXED_METHODS + ADAPTIVE_METHODS:
            raise ConfigError(f"{where}.method: {method!r} is not one of {FIXED_METHODS + ADAPTIVE_METHODS}")


def _initial(p: Params) -> None:
    """Check the start/end times and the shape of initial state values."""
    t0 = p.initial.t
    if t0 < 0.0:
        raise ConfigError("initial.t must be >= 0")
    for name, u in p.units.items():  # t0 must be on every unit's PWM update grid
        T = u.pwm.effective_update_period
        if abs(round(t0 / T) * T - t0) > 1e-9 * max(1.0, t0):
            raise ConfigError(f"initial.t = {t0} is not on the PWM update grid of {name!r} (a multiple of "
                              f"units.{name}.pwm.update_period = {T})")
    if p.simulation.t_end <= t0:
        raise ConfigError(f"simulation.t_end = {p.simulation.t_end} must be after initial.t = {t0}")
    for key, value in p.initial.states.items():
        if isinstance(value, str):
            continue  # a keyword, resolved when the plant is built
        elif isinstance(value, (list, tuple)):
            if len(value) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                raise ConfigError(f"initial.states.{key}: expected [re, im], got {value!r}")
        elif not isinstance(value, (int, float)):
            raise ConfigError(f"initial.states.{key}: expected a number, [re, im], a flag or a keyword, "
                              f"got {value!r}")


def validate(p: Params) -> Params:
    """Check a :class:`Params` tree and return it unchanged; raise ConfigError if invalid."""
    _network(p)
    for name, unit in p.units.items():
        _unit(unit, p.base, f"units.{name}")
        if unit.pwm.sync == "synchronous" and p.simulation.bridge != "switching":
            raise ConfigError(f"units.{name}.pwm.sync = 'synchronous' needs simulation.bridge = "
                              f"'switching' (got {p.simulation.bridge!r}): an averaged bridge has no carrier")
    _solver(p.simulation.solver, p.simulation.bridge)
    _initial(p)
    return p
