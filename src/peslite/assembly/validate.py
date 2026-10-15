"""Checks across the sections of a constructed parameter tree (read-only).

Covers topology, dc link, control, timing, solver settings and initial values; the construction
(:func:`peslite.assembly.params.from_dict`) has checked each field on its own.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import fields, is_dataclass

from typing import TYPE_CHECKING, Optional

from ..components.network import element_buses
from ..solver.integrators import ADAPTIVE_METHODS, FIXED_METHODS
from ..solver.model import ConfigError
from ..solver.multirate import parse_step
from .events import NAMED, SWITCHABLE, SWITCHING

if TYPE_CHECKING:
    from .params import Params

__all__ = ["validate"]


def _whole(r: float) -> bool:
    """Is ``r`` a whole number of at least one?"""
    n = round(r)
    return n >= 1 and abs(r - n) <= 1e-9 * max(1.0, r)


def _switched(section, where: str) -> None:
    """Validate a section controlled by an ``enable`` switch."""
    if not section.enable:
        return
    for f in fields(section):
        value = getattr(section, f.name)
        if f.name == "enable" or not f.init:
            continue
        if value is None:
            raise ConfigError(f"{where}.{f.name}: required when enable is 1")
        if isinstance(value, (int, float)) and (not math.isfinite(value) or value <= 0):
            raise ConfigError(f"{where}.{f.name} must be finite and positive, got {value}")


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


def _pwm(w, base, where: str, continuous: bool = False) -> None:
    """Check switching frequency, phase and synchronous pulse ratio."""
    if not math.isfinite(w.f_sw) or w.f_sw <= 0.0:
        raise ConfigError(f"{where}.f_sw must be finite and positive, got {w.f_sw}")
    if not math.isfinite(w.carrier_phase):
        raise ConfigError(f"{where}.carrier_phase must be finite, got {w.carrier_phase}")
    if not continuous and w.sync == "synchronous":
        ratio = w.f_sw / base.f0
        if abs(ratio - round(ratio)) > 1e-9 * max(1.0, ratio):
            raise ConfigError(f"{where}.sync = 'synchronous' needs an integer pulse ratio "
                              f"pwm.f_sw / base.f0, got {w.f_sw} / {base.f0} = {ratio}")


def _period(c, pwm) -> float:
    """The control period (s), defaulting to one carrier period."""
    return c.period if c.period is not None else pwm.switching_period


def _timing(c, pwm, where: str, continuous: bool = False) -> None:
    """Check the carrier-triggered control interrupt and computation time."""
    if continuous:
        if c.period is not None and (not math.isfinite(c.period) or c.period <= 0.0):
            raise ConfigError(f"{where}.period must be finite and positive, got {c.period}")
        if c.computation is not None and (not math.isfinite(c.computation) or c.computation < 0.0):
            raise ConfigError(
                f"{where}.computation must be finite and nonnegative, got {c.computation}"
            )
        return
    T_c = _period(c, pwm)
    if not math.isfinite(T_c) or T_c <= 0.0:
        raise ConfigError(f"{where}.period must be finite and positive, got {T_c}")
    half = pwm.switching_period / 2.0
    if not _whole(T_c / half):
        raise ConfigError(f"{where}.period = {T_c:.6g} s is not a multiple of half the carrier period "
                          f"({half:.6g} s at pwm.f_sw = {pwm.f_sw:g} Hz): the carrier triggers the "
                          f"interrupt at its valleys and peaks")
    tau = 1.0e-6 if c.computation is None else c.computation
    if not math.isfinite(tau) or tau < 0.0 or tau >= T_c:
        raise ConfigError(f"{where}.computation = {tau} s must be at least 0 and shorter than the "
                          f"control period {T_c:.6g} s")


def _control(c, dclink, pwm, where: str, continuous: bool = False) -> None:
    """Check typed control loops and their requirements on the DC side."""
    if not c.loops:
        raise ConfigError(f"{where}.loops must contain at least one loop")
    T_c = None if continuous else _period(c, pwm)
    for name, cfg in c.loops.items():
        if name in {"meas", "references", "held", "command", "prot"}:
            raise ConfigError(f"{where}.loops.{name}: reserved loop name")
        T = cfg.period
        if T is not None and (not math.isfinite(T) or T <= 0):
            raise ConfigError(f"{where}.loops.{name}.period must be finite and positive")
        if not continuous and T is not None and not _whole(T / T_c):
            raise ConfigError(f"{where}.loops.{name}.period = {T:.6g} s is not a multiple of "
                              f"ctrl.period = {T_c:.6g} s: a loop runs at every n-th control interrupt")
        for key, value in vars(cfg).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ConfigError(f"{where}.loops.{name}.{key} must be finite")
            if is_dataclass(value) and hasattr(value, "enable"):
                _switched(value, f"{where}.loops.{name}.{key}")
        if cfg.type == "matching" and (dclink is None or dclink.capacitor is None):
            raise ConfigError(f"{where}: matching control needs dclink.capacitor")
        if cfg.type == "matching" and cfg.k_theta_pu is None and c.references.vdc_ref_pu <= 0:
            raise ConfigError(f"{where}.references.vdc_ref_pu must be positive for automatic matching gain")
    for name, value in vars(c.references).items():
        if name == "omega" and value is None:
            continue
        if not isinstance(name, str) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(f"{where}.references: expected finite scalar references")


def _measurement(m, T_c: Optional[float], where: str) -> None:
    """Check ADC oversampling and averaging windows within one control period."""
    if not isinstance(m.average, (list, tuple)):
        raise ConfigError(f"{where}.average: expected a list of channels, got {m.average!r}")
    unknown = set(m.average) - {"u_g", "i_c", "u_dc"}
    if unknown:
        raise ConfigError(f"{where}.average: unknown channel(s) {sorted(unknown)}; "
                          "known: ['i_c', 'u_dc', 'u_g']")
    if len(set(m.average)) != len(m.average):
        raise ConfigError(f"{where}.average: channels must not be repeated")
    for key in ("period", "window"):
        value = getattr(m, key)
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ConfigError(f"{where}.{key} must be finite and positive, got {value}")
    if T_c is None:
        return
    for key in ("period", "window"):
        value = getattr(m, key)
        if value is None:
            continue
        if value > T_c * (1.0 + 1e-9):
            raise ConfigError(f"{where}.{key} = {value:.6g} s is longer than ctrl.period = {T_c:.6g} s")
    if m.period is not None and not _whole(T_c / m.period):
        raise ConfigError(f"{where}.period = {m.period:.6g} s does not divide ctrl.period = "
                          f"{T_c:.6g} s into whole samples")
    if m.average and m.period is not None and m.period < T_c * (1.0 - 1e-9):
        raise ConfigError(f"{where}: window averaging cannot be combined "
                          f"with oversampling (period shorter than ctrl.period)")
    if not m.average and m.window is not None:
        warnings.warn(f"{where}.window has no effect: no quantity is window-averaged", stacklevel=4)


def _unit(u, base, where: str) -> None:
    """Check one converter unit."""
    if u.s_base is not None and (not math.isfinite(u.s_base) or u.s_base <= 0.0):
        raise ConfigError(f"{where}.s_base must be finite and positive, got {u.s_base}")
    if u.ac_filter.l_f <= 0.0:
        raise ConfigError(f"{where}.ac_filter.l_f must be > 0")
    if u.bridge.type != "three_phase_bridge":
        raise ConfigError(f"{where}.bridge.type: a VSC requires 'three_phase_bridge'")
    if not u.bridge.upper_controlled or not u.bridge.lower_controlled:
        raise ConfigError(
            f"{where}.bridge: controlled-device fields apply only to a half bridge"
        )
    _dclink(u.dclink, f"{where}.dclink")
    continuous = u.bridge.model == "averaging"
    _pwm(u.pwm, base, f"{where}.pwm", continuous)
    _timing(u.ctrl, u.pwm, f"{where}.ctrl", continuous)
    _control(u.ctrl, u.dclink, u.pwm, f"{where}.ctrl", continuous)
    _measurement(u.meas, None if continuous else _period(u.ctrl, u.pwm), f"{where}.meas")
    protection = u.protection
    for name in ("overcurrent", "undervoltage", "overvoltage", "frequency", "dc_voltage", "rocof"):
        _switched(getattr(protection, name), f"{where}.protection.{name}")
    if not math.isfinite(protection.hold) or protection.hold < 0:
        raise ConfigError(f"{where}.protection.hold must be finite and >= 0, got {protection.hold}")
    if not math.isfinite(protection.rocof.window) or protection.rocof.window <= 0:
        raise ConfigError(f"{where}.protection.rocof.window must be finite and positive, "
                          f"got {protection.rocof.window}")


def _dc_converter(u, base, where: str, *, nested: bool = False) -> None:
    """Check one open scalar converter."""
    if u.s_base is not None and (not math.isfinite(u.s_base) or u.s_base <= 0.0):
        raise ConfigError(f"{where}.s_base must be finite and positive, got {u.s_base}")
    _dclink(u.dclink, f"{where}.dclink")
    if u.bridge.type not in {"half_bridge", "full_bridge"}:
        raise ConfigError(
            f"{where}.bridge.type: a DC converter requires 'half_bridge' or 'full_bridge'"
        )
    if u.bridge.type != "half_bridge" and (
            not u.bridge.upper_controlled or not u.bridge.lower_controlled):
        raise ConfigError(
            f"{where}.bridge: controlled-device fields apply only to a half bridge"
        )
    if (u.bus is None) != (u.filter is None):
        raise ConfigError(f"{where}: bus and filter must be specified together")
    if not nested and u.bus is None:
        raise ConfigError(f"{where}.bus: a standalone DC converter needs a scalar bus")
    if u.filter is not None and u.filter.l_f <= 0.0:
        raise ConfigError(f"{where}.filter.l_f must be > 0")
    if u.ctrl.type == "dc":
        if nested:
            raise ConfigError(f"{where}.ctrl.type: a directly connected WPT port requires 'resonant'")
        if not math.isfinite(u.ctrl.references.voltage) or u.ctrl.references.voltage <= 0.0:
            raise ConfigError(f"{where}.ctrl.references.voltage must be finite and positive")
    continuous = u.bridge.model == "averaging"
    _pwm(u.pwm, base, f"{where}.pwm", continuous)
    if u.pwm.sync != "asynchronous":
        raise ConfigError(f"{where}.pwm.sync: scalar converters require asynchronous PWM")
    _timing(u.ctrl, u.pwm, f"{where}.ctrl", continuous)
    if u.ctrl.type == "dc":
        _control(u.ctrl, u.dclink, u.pwm, f"{where}.ctrl", continuous)
        for name, loop in u.ctrl.loops.items():
            if hasattr(loop, "minimum") and loop.minimum > loop.maximum:
                raise ConfigError(
                    f"{where}.ctrl.loops.{name}.minimum must not exceed maximum"
                )
    else:
        if u.bridge.type != "full_bridge":
            raise ConfigError(f"{where}.ctrl.type: resonant control requires a full bridge")
        if u.ctrl.loops:
            raise ConfigError(f"{where}.ctrl.loops: resonant control has no feedback loops")
        ref = u.ctrl.references
        if not math.isfinite(ref.modulation) or not 0.0 <= ref.modulation <= 1.0:
            raise ConfigError(f"{where}.ctrl.references.modulation must be in [0, 1]")
        if not math.isfinite(ref.frequency) or ref.frequency <= 0.0:
            raise ConfigError(f"{where}.ctrl.references.frequency must be finite and positive")
        if not math.isfinite(ref.phase):
            raise ConfigError(f"{where}.ctrl.references.phase must be finite")


def _wpt(u, base, where: str) -> None:
    """Check a WPT composite and its shared compensated magnetic link."""
    if u.s_base is not None and (not math.isfinite(u.s_base) or u.s_base <= 0.0):
        raise ConfigError(f"{where}.s_base must be finite and positive, got {u.s_base}")
    for side in ("primary", "secondary"):
        child = getattr(u, side)
        child_where = f"{where}.{side}"
        if child.type != "dc_converter":
            raise ConfigError(
                f"{child_where}.type: a WPT port must be 'dc_converter', got {child.type!r}"
            )
        _dc_converter(child, base, child_where, nested=True)
    link = u.link
    positive = [
        "primary_inductance", "secondary_inductance",
    ]
    if link.type == "ss":
        positive += ["primary_capacitance", "secondary_capacitance"]
    else:
        positive += [
            "primary_series_capacitance", "secondary_series_capacitance",
            "primary_compensation_inductance", "secondary_compensation_inductance",
            "primary_parallel_capacitance", "secondary_parallel_capacitance",
        ]
    for name in positive:
        value = getattr(link, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ConfigError(f"{where}.link.{name} must be finite and positive")
    for name in ("mutual_inductance", "primary_resistance", "secondary_resistance"):
        value = getattr(link, name)
        if not math.isfinite(value) or value < 0.0:
            raise ConfigError(f"{where}.link.{name} must be finite and nonnegative")
    if link.type == "lcc":
        for name in ("primary_compensation_resistance", "secondary_compensation_resistance"):
            value = getattr(link, name)
            if not math.isfinite(value) or value < 0.0:
                raise ConfigError(f"{where}.link.{name} must be finite and nonnegative")
    determinant = (
        link.primary_inductance * link.secondary_inductance
        - link.mutual_inductance * link.mutual_inductance
    )
    if determinant <= 0.0:
        raise ConfigError(
            f"{where}.link: the coupled-inductor matrix must be positive definite "
            "(mutual_inductance^2 < primary_inductance * secondary_inductance)"
        )


def _network(p: Params) -> None:
    """Check element identities, network values and bus references."""
    if not p.units:
        raise ConfigError("units: a system needs at least one converter")
    ac_units = {name: unit for name, unit in p.runtime_units.items() if unit.type == "vsc"}
    if (ac_units or p.branches or p.sources or p.elements) and not p.buses:
        raise ConfigError("buses: a connected electrical network needs at least one bus")
    seen: dict[str, str] = {}
    for kind in NAMED:
        for name in getattr(p, kind):
            if name in seen:
                raise ConfigError(f"{kind}.{name}: the name is already used by {seen[name]}; element "
                                  f"names are unique across buses, branches, sources, units and elements")
            seen[name] = f"{kind}.{name}"
    for name, bus in p.buses.items():
        if bus.c <= 0.0:
            raise ConfigError(f"buses.{name}.c must be > 0: the shunt capacitor is what makes the "
                              f"bus voltage a state (a bus without one would need a node-voltage solver)")
        if bus.phases not in (1, 3):
            raise ConfigError(f"buses.{name}.phases must be 1 or 3")
    def _bus_of(where: str, bus: str) -> None:
        if bus not in p.buses:
            raise ConfigError(f"{where}: unknown bus {bus!r}; known: {sorted(p.buses)}")
    for name, br in p.branches.items():
        _bus_of(f"branches.{name}.bus1", br.bus1)
        _bus_of(f"branches.{name}.bus2", br.bus2)
        if br.bus1 == br.bus2:
            raise ConfigError(f"branches.{name}: both ends are {br.bus1!r}")
        if p.buses[br.bus1].phases != p.buses[br.bus2].phases:
            raise ConfigError(f"branches.{name}: both buses must have the same phase count")
        if br.l <= 0.0:
            raise ConfigError(f"branches.{name}.l must be > 0")
    for name, src in p.sources.items():
        _bus_of(f"sources.{name}.bus", src.bus)
        if p.buses[src.bus].phases != 3:
            raise ConfigError(f"sources.{name}.bus: a three-phase source needs a three-phase bus")
        if src.f is not None and (not math.isfinite(src.f) or src.f <= 0.0):
            raise ConfigError(f"sources.{name}.f must be finite and positive, got {src.f}")
        if not math.isfinite(src.angle):
            raise ConfigError(f"sources.{name}.angle must be finite")
        if src.l <= 0.0:
            raise ConfigError(f"sources.{name}.l must be > 0: a source is an emf behind an impedance")
    for name, u in ac_units.items():
        _bus_of(f"units.{name}.bus", u.bus)
        if p.buses[u.bus].phases != 3:
            raise ConfigError(f"units.{name}.bus: a VSC needs a three-phase bus")
    for name, u in p.units.items():
        if u.type != "dc_converter":
            continue
        if u.bus is None:
            continue
        _bus_of(f"units.{name}.bus", u.bus)
    for name, element in p.elements.items():
        for key, bus in element_buses(element).items():
            _bus_of(f"elements.{name}.{key}", bus)
        for key, value in vars(element).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ConfigError(f"elements.{name}.{key} must be finite")


def _solver(s) -> None:
    """Check solver methods and subsystem step ratios."""
    if s.type == "fixed" and s.method not in FIXED_METHODS:
        raise ConfigError(f"simulation.solver.method {s.method!r} is not one of {FIXED_METHODS}")
    if s.type == "adaptive" and s.method not in ADAPTIVE_METHODS:
        raise ConfigError(f"simulation.solver.method {s.method!r} is not one of {ADAPTIVE_METHODS}")
    if s.sweeps < 1:
        raise ConfigError("simulation.solver.sweeps must be >= 1")
    if s.linearisations < 0:
        raise ConfigError("simulation.solver.linearisations must be >= 0")
    if s.write_length < 1:
        raise ConfigError("simulation.solver.write_length must be >= 1")
    if s.phs_check_step < 1:
        raise ConfigError("simulation.solver.phs_check_step must be >= 1")
    for name, value in s.subsystems.items():
        where = f"simulation.solver.subsystems.{name}"
        if s.type != "fixed":
            raise ConfigError(f"{where}: subsystems on their own step need the fixed-step solver")
        if isinstance(value, dict) and (set(value) - {"step", "method"} or "step" not in value):
            raise ConfigError(f"{where}: expected {{step, method}} (step required), got {sorted(value)}")
        try:
            _ratio, method = parse_step(value)
        except ValueError as exc:
            raise ConfigError(f"{where}: {exc}") from None
        if method is not None and method not in FIXED_METHODS + ADAPTIVE_METHODS:
            raise ConfigError(f"{where}.method: {method!r} is not one of {FIXED_METHODS + ADAPTIVE_METHODS}")


def _events(p: Params) -> None:
    """Check event targets, times, and each target's connect/disconnect sequence."""
    switched: dict[str, tuple[str, bool]] = {}
    at: dict[tuple[str, float], str] = {}
    for name, event in sorted(p.events.items(), key=lambda item: item[1].t):
        where = f"events.{name}"
        for key, value in vars(event).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ConfigError(f"{where}.{key} must be finite")
        target = getattr(event, "target", None)
        section = p.section_of(target) if isinstance(target, str) else None
        if target is not None and section is None:
            known = sorted(name for named in NAMED for name in getattr(p, named))
            raise ConfigError(f"{where}.target: unknown name {target!r}; known: {known}")
        if event.type not in SWITCHING:
            continue
        if section not in SWITCHABLE:
            raise ConfigError(f"{where}.target: {target!r} is a bus, which has no breaker; {event.type} "
                              f"switches a unit, source, branch or element")
        on = event.type == "connect"
        affected = p.unit_groups.get(target, (target,))
        for actual in affected:
            previous = switched.get(actual)
            if previous is not None and previous[1] == on:
                detail = (f"{target!r} is already {event.type}ed"
                          if actual == target else
                          f"{target!r} also switches {actual!r}, which is already {event.type}ed")
                raise ConfigError(f"{where}: {detail} by events.{previous[0]}")
            if (actual, event.t) in at:
                detail = (f"{target!r} is also switched at t = {event.t}"
                          if actual == target else
                          f"{target!r} also switches {actual!r} at t = {event.t}")
                raise ConfigError(f"{where}: {detail} by events.{at[(actual, event.t)]}")
            at[(actual, event.t)] = name
            switched[actual] = (name, on)


def _initial(p: Params) -> None:
    """Check the start/end times and the shape of initial state values."""
    t0 = p.simulation.initial.t
    if t0 < 0.0:
        raise ConfigError("simulation.initial.t must be >= 0")
    for name, u in p.runtime_units.items():
        if u.type != "vsc":
            continue
        if u.bridge.model == "averaging":
            continue
        m, T_c = u.meas, _period(u.ctrl, u.pwm)
        continued = any(key.startswith((f"{name}.ctrl.", f"{name}.pwm.",
                                         f"{name}.bridge.", f"{name}.meas."))
                        for key in p.simulation.initial.states)
        if not continued or m.period is None or m.period >= T_c * (1.0 - 1e-9):
            continue
        k = round((t0 - u.pwm.grid_offset) / T_c)
        if abs(u.pwm.grid_offset + k * T_c - t0) > 1e-10:
            raise ConfigError(f"simulation.initial.t = {t0} is between two control interrupts of {name!r}, "
                              f"whose oversampled ADC (units.{name}.meas.period) took samples before it "
                              f"that are not states: continue from an interrupt")
    if p.simulation.t_end <= t0:
        raise ConfigError(f"simulation.t_end = {p.simulation.t_end} must be after simulation.initial.t = {t0}")
    for key, value in p.simulation.initial.states.items():
        if isinstance(value, str):
            continue  # a keyword, resolved when the plant is built
        elif isinstance(value, (list, tuple)):
            if len(value) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                raise ConfigError(f"simulation.initial.states.{key}: expected [re, im], got {value!r}")
        elif not isinstance(value, (int, float)):
            raise ConfigError(f"simulation.initial.states.{key}: expected a number, [re, im], a flag or a keyword, "
                              f"got {value!r}")


def _progress(progress) -> None:
    """Check progress settings; watched names are resolved at the first progress line."""
    _switched(progress, "simulation.progress")
    watch = progress.watch
    if not isinstance(watch, list) or not all(isinstance(name, str) and name for name in watch):
        raise ConfigError(f"simulation.progress.watch: expected a list of quantity names, got {watch!r}")


def validate(p: Params) -> Params:
    """Check a :class:`Params` tree and return it unchanged; raise ConfigError if invalid."""
    _network(p)
    checks = {
        "vsc": _unit,
        "dc_converter": _dc_converter,
        "wpt": _wpt,
    }
    for name, unit in p.units.items():
        check = checks[unit.type]
        check(unit, p.base, f"units.{name}")
    _solver(p.simulation.solver)
    _events(p)
    _progress(p.simulation.progress)
    _initial(p)
    return p
