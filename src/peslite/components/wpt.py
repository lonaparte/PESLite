"""Compensated magnetic links for wireless power transfer.

One implementation covers every supported link. A link couples ``n`` coils through a symmetric
positive-definite inductance matrix; every coil carries a series capacitor and resistance and
belongs to the primary or the secondary side. A side is either *series* compensated (its coils
connect in parallel directly to the side's port) or *LCC* compensated (a series compensation
inductor from the port to a shunt-capacitor node, to which the side's coils connect in parallel).
``ss`` (series-series), ``lcc`` (LCC-LCC), ``lcc_s`` (LCC-series) and ``ss_multi`` (several
series-series modules with inter- and cross-coupling) are configurations of this one model.

The coupled-inductor matrix is diagonalized once during construction. The ODE stores modal
currents with the diagonal :class:`StoragePort` energy representation; there is no per-step matrix
inversion and no special solver path. A two-coil link uses the closed-form diagonalization, so its
arithmetic stays scalar for parameterized standalone backends.

A diode bridge which blocks on one port needs the port voltage that keeps the port current
constant (its *hold voltage*). The link evaluates it from its own equations through the output
``u_hold_<side>``, published when terminal assembly requests that capability.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Mapping
from functools import lru_cache

import numpy as np

from ..solver.energy import PowerPort, StoragePort
from ..solver.model import Bag, LocatedZero, OutputMethod, _real, commutation_tolerance
from .network import Terminal
from .pwm import PhaseClock

__all__ = ["CompensatedLink", "SwitchedCompensatedLink", "SwitchControlledCapacitor",
           "make_wpt_link"]

SIDES = ("primary", "secondary")


def _two_coil_modes(primary: float, secondary: float, mutual: float):
    """Closed-form orthogonal diagonalization of a two-winding inductance matrix."""
    angle = 0.5 * np.arctan2(2.0 * mutual, primary - secondary)
    cosine, sine = np.cos(angle), np.sin(angle)
    mean = 0.5 * (primary + secondary)
    radius = np.sqrt((0.5 * (primary - secondary)) ** 2 + mutual ** 2)
    first_sign = np.where(abs(sine) >= abs(cosine), np.where(sine > 0.0, -1.0, 1.0), 1.0)
    second_sign = np.where(abs(sine) > abs(cosine), np.where(sine < 0.0, -1.0, 1.0), 1.0)
    inductance = (_real(mean - radius), _real(mean + radius))
    # rows: coils (primary, secondary); columns: modes
    shapes = ((_real(-sine * first_sign), _real(cosine * second_sign)),
              (_real(cosine * first_sign), _real(sine * second_sign)))
    return inductance, shapes


@lru_cache(maxsize=32)
def _equation_methods(coil_sides, lcc_sides, cap_names, coil_outputs, capacitor_outputs, switched):
    """Compile a bounded set of structural layouts, never numerical values or model instances."""
    n = len(coil_sides)
    mode_names = tuple(f"i_mode_{m + 1}" for m in range(n))
    side_coils = {side: tuple(j for j, owner in enumerate(coil_sides) if owner == side)
                  for side in SIDES}
    coil_lines = ["s = self.state"]
    for j in range(n):
        coefficients = ", ".join(f"a{j}_{m}" for m in range(n))
        coil_lines.append(f"{coefficients}, = self._shapes[{j}]")
        expression = " + ".join(f"a{j}_{m} * s.{name}"
                                for m, name in enumerate(mode_names))
        coil_lines.append(f"i{j} = {expression}")
    sums = {side: " + ".join(f"i{j}" for j in side_coils[side]) for side in SIDES}
    output_lines = [*coil_lines, "out = self.out"]
    if switched:
        output_lines.append("self._t = t")
    output_lines.extend(f"out.{side}_scc_on = _real(self.switched[{k}].on)"
                        for k, (coil, name, side) in enumerate(switched))
    for side in SIDES:
        output_lines.append(f"out.i_{side}_coil = {sums[side]}")
        current = f"s.i_{side}_comp" if side in lcc_sides else f"out.i_{side}_coil"
        output_lines.append(f"out.i_{side} = {current}")
    output_lines.extend(f"out.{name} = i{j}" for j, name in enumerate(coil_outputs))
    output_lines.extend(f"out.{name} = s.{name}" for name in capacitor_outputs)
    specs = [f"p_{side} = self.sides[{side!r}]" for side in SIDES if side in lcc_sides]
    nodes = {}
    front_currents, front_voltages = [], []
    losses = [f"self.resistance[{j}] * i{j} * i{j}" for j in range(n)]
    for side in SIDES:
        if side in lcc_sides:
            spec = f"p_{side}"
            shunt = f"(s.i_{side}_comp - ({sums[side]}))"
            nodes[side] = f"s.u_{side}_parallel + {spec}.parallel_resistance * {shunt}"
            front_currents.append(f"(inp.u_{side} - node_{side} - "
                                  f"{spec}.compensation_resistance * s.i_{side}_comp) "
                                  f"/ {spec}.compensation_inductance")
            front_voltages.append(f"{shunt} / {spec}.parallel_capacitance")
            losses.extend((f"{spec}.compensation_resistance * s.i_{side}_comp * s.i_{side}_comp",
                           f"{spec}.parallel_resistance * {shunt} * {shunt}"))
        else:
            nodes[side] = f"inp.u_{side}"
    voltages = [f"v{j} = node_{side} - s.{cap_names[j]} - self.resistance[{j}] * i{j}"
                for j, side in enumerate(coil_sides)]
    for coil, name, side in switched:
        voltages[coil] += f" - s.{name}"
    rhs_lines = [*coil_lines, "inp = self.inp", *specs,
                 *(f"node_{side} = {value}" for side, value in nodes.items()), *voltages]
    modes = ["(" + " + ".join(f"a{j}_{m} * v{j}" for j in range(n))
             + f") / self._inductance[{m}]" for m in range(n)]
    derivatives = [*front_currents, *front_voltages, *modes,
                   *(f"i{j} / self.capacitance[{j}]" for j in range(n)),
                   *(f"0.0 if self.switched[{k}].on else i{coil} / self.switched[{k}].capacitance"
                     for k, (coil, name, side) in enumerate(switched))]
    rhs_lines.append("return (" + ", ".join(derivatives) + ",)")
    coil_result = [*coil_lines, "return (" + ", ".join(f"i{j}" for j in range(n)) + ",)"]
    functions = [("rhs", "self, t", rhs_lines),
                 ("set_outputs", "self, t", output_lines),
                 ("coil_currents", "self", coil_result),
                 ("_losses", "self", [*coil_lines, *specs,
                                       "return " + " + ".join(losses)])]
    for side in SIDES:
        lines = [*coil_lines, "inp = self.inp", *specs]
        if (side in lcc_sides):
            lines.append(f"return ({nodes[side]}) + "
                         f"p_{side}.compensation_resistance * s.i_{side}_comp")
        else:
            lines.extend(f"node_{key} = {value if key != side else '0.0'}"
                         for key, value in nodes.items())
            lines.extend(voltages)
            rate = " + ".join(f"self._hold_weights[{side!r}][{j}] * v{j}" for j in range(n))
            lines.append(f"return -({rate}) / self._port_gain[{side!r}]")
        functions.append((f"_hold_{side}", "self", lines))
    namespace = {"_real": _real}
    methods = []
    for name, signature, lines in functions:
        source = f"def {name}({signature}):\n" + "\n".join("    " + line for line in lines)
        exec(compile(source + "\n", "<peslite compensated link>", "exec"), namespace)
        methods.append((name, namespace[name]))
    return tuple(methods)



class _Side:
    """Static description of one side: its coils and optional LCC front end."""

    __slots__ = ("name", "coils", "lcc", "compensation_inductance", "compensation_resistance",
                 "parallel_capacitance", "parallel_resistance")

    def __init__(self, name: str, coils: tuple[int, ...], lcc: bool = False,
                 compensation_inductance: float = 0.0, compensation_resistance: float = 0.0,
                 parallel_capacitance: float = 0.0, parallel_resistance: float = 0.0) -> None:
        self.name, self.coils, self.lcc = name, coils, lcc
        self.compensation_inductance = compensation_inductance
        self.compensation_resistance = compensation_resistance
        self.parallel_capacitance = parallel_capacitance
        self.parallel_resistance = parallel_resistance


class CompensatedLink:
    """Coupled coils with per-coil series capacitors and per-side series or LCC compensation."""

    outputs_need_inputs: ClassVar[bool] = False
    ports: ClassVar[tuple[PowerPort, ...]] = (
        PowerPort("inp.u_primary", "out.i_primary"),
        PowerPort("inp.u_secondary", "out.i_secondary"),
    )

    def __init__(self, *, inductance, coil_sides: tuple[str, ...], capacitance, resistance,
                 sides: dict[str, _Side],
                 switched: tuple["SwitchControlledCapacitor", ...] = ()) -> None:
        n = len(coil_sides)
        self.switched = tuple(switched)
        self._scc_names = tuple(scc.state_name for scc in switched)
        self.coil_sides = tuple(coil_sides)
        self.sides = sides
        if n == 2:
            self._inductance, shapes = _two_coil_modes(
                inductance[0][0], inductance[1][1], inductance[0][1])
        else:
            matrix = np.asarray(inductance, dtype=float)
            values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
            self._inductance = tuple(float(v) for v in values)
            shapes = tuple(tuple(float(vectors[j, m]) for m in range(n)) for j in range(n))
        self._shapes = shapes                      # coil current j = sum_m shapes[j][m] mode_m
        self.capacitance = tuple(_real(c) for c in capacitance)
        self.resistance = tuple(_real(r) for r in resistance)

        # state layout: LCC front ends, modal currents, coil capacitors (stable public names)
        front = [side for side in SIDES if sides[side].lcc]
        multi = any(len(sides[side].coils) > 1 for side in SIDES)
        self._cap_names = tuple(self._capacitor_name(j, multi) for j in range(n))
        self.state_names = (
            tuple(f"i_{side}_comp" for side in front)
            + tuple(f"u_{side}_parallel" for side in front)
            + tuple(f"i_mode_{m + 1}" for m in range(n))
            + self._cap_names + self._scc_names
        )
        self._mode_names = tuple(f"i_mode_{m + 1}" for m in range(n))
        self.state = type("_LinkState", (Bag,), {"__slots__": self.state_names})()
        self.storage = (
            tuple(StoragePort(f"i_{side}_comp", sides[side].compensation_inductance,
                              kind="inductor") for side in front)
            + tuple(StoragePort(f"u_{side}_parallel", sides[side].parallel_capacitance,
                                kind="capacitor") for side in front)
            + tuple(StoragePort(name, self._inductance[m], kind="inductor")
                    for m, name in enumerate(self._mode_names))
            + tuple(StoragePort(name, c, kind="capacitor")
                    for name, c in zip(self._cap_names, self.capacitance))
            + tuple(StoragePort(scc.state_name, scc.capacitance, kind="capacitor")
                    for scc in switched)
        )
        self._coil_outputs = (tuple(f"i_{self.coil_sides[j]}_coil_{self._module(j) + 1}"
                                    for j in range(n)) if multi else ())
        capacitor_outputs = (tuple(f"u_{side}_parallel" for side in front)
                             + self._cap_names + self._scc_names)
        self._capacitor_outputs = capacitor_outputs
        outputs = ("i_primary", "i_secondary", "i_primary_coil", "i_secondary_coil",
                   *self._coil_outputs, *capacitor_outputs,
                   *(f"{scc.side}_scc_on" for scc in switched))
        self.hold_sides = ()
        self._outputs = outputs
        self.out = type("_LinkOut", (Bag,), {"__slots__": outputs})()
        self.output_stages = (OutputMethod("set_outputs", inputs=(), outputs=outputs),)
        self.inp = type("_LinkInp", (Bag,), {"__slots__": ("u_primary", "u_secondary")})()
        self.primary_terminal = Terminal(
            self, "u_primary", "i_primary", 1, electrical_domain="ac"
        )
        self.secondary_terminal = Terminal(
            self, "u_secondary", "i_secondary", 1, electrical_domain="ac"
        )
        self._compile_equations()

    def hold_voltage_source(self, terminal: Terminal, *, current_constraint=None) -> tuple:
        """Declare a blocking-voltage output for the actual connected terminal."""
        side = terminal.voltage.removeprefix("u_")
        if side not in self.sides:
            raise ValueError(f"unknown link terminal {terminal.voltage!r}")
        if current_constraint is not None:
            if not hasattr(self, "_current_constraints"):
                self._current_constraints = {}
                self._current_projections = {}
                self.project_state = self._project_currents
            self._current_constraints[side] = current_constraint
        self._prepare_holds()
        if side not in self.hold_sides:
            self.hold_sides = (*self.hold_sides, side)
            names = (*self._outputs, *(f"u_hold_{key}" for key in self.hold_sides))
            previous = self.out
            self.out = type("_LinkOut", (Bag,), {"__slots__": names})(
                **{key: getattr(previous, key) for key in type(previous).__slots__})
            other = SIDES[1 - SIDES.index(side)]
            inputs = () if self.sides[side].lcc or self.sides[other].lcc else (f"u_{other}",)
            self.output_stages = (*self.output_stages, OutputMethod(
                f"set_hold_{side}", inputs=inputs, outputs=(f"u_hold_{side}",)))
        return self, f"u_hold_{side}"

    def __copy__(self):
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        if hasattr(result, "_current_projections"):
            result._current_projections = {}
        if hasattr(result, "_hold_weights"):
            del result._hold_weights
            result._prepare_holds()
        return result

    def _project_currents(self) -> None:
        """Enforce active port constraints jointly in the stored modal coordinates."""
        targets = {}
        for side, read in self._current_constraints.items():
            value = read()
            if value is None:
                continue
            if self.sides[side].lcc:
                setattr(self.state, f"i_{side}_comp", value)
            else:
                targets[side] = value
        if not targets:
            return
        active = tuple(targets)
        if active not in self._current_projections:
            # Weighted orthogonal rows implement the same minimum-energy projection without
            # a numerical matrix solve. Coefficients are cached; scalar parameter expressions
            # retain their arithmetic through construction and standalone export.
            basis = []
            for index, side in enumerate(active):
                row = [sum(self._shapes[j][m] for j in self.sides[side].coils)
                       for m in range(len(self._mode_names))]
                weights = [1.0 if k == index else 0.0 for k in range(len(active))]
                for orthogonal, direction, previous in basis:
                    overlap = sum(a * b for a, b in zip(row, direction))
                    row = [a - overlap * b for a, b in zip(row, orthogonal)]
                    weights = [a - overlap * b for a, b in zip(weights, previous)]
                inverse = [a / inductance for a, inductance in zip(row, self._inductance)]
                norm = sum(a * b for a, b in zip(row, inverse))
                basis.append((tuple(row), tuple(a / norm for a in inverse), tuple(weights)))
            self._current_projections[active] = tuple(basis)
        basis = self._current_projections[active]
        old = tuple(getattr(self.state, name) for name in self._mode_names)
        currents = [0.0] * len(old) if len(active) == len(old) else list(old)
        values = tuple(targets.values())
        for row, direction, weights in basis:
            requested = sum(a * b for a, b in zip(weights, values))
            if len(active) != len(old):
                requested -= sum(a * b for a, b in zip(row, old))
            currents = [value + gain * requested for value, gain in zip(currents, direction)]
        for name, value in zip(self._mode_names, currents):
            setattr(self.state, name, _real(value))

    def _compile_equations(self) -> None:
        """Bind the shared structural template to this instance's live coefficients.

        The same equations serve Python, parameter tracing and C++ export. Only layout is cached;
        no numerical coefficients, physical state or bound instance escapes into the cache.
        """
        methods = _equation_methods(
            self.coil_sides, tuple(side for side in SIDES if self.sides[side].lcc),
            self._cap_names, self._coil_outputs, self._capacitor_outputs,
            tuple((scc.coil, scc.state_name, scc.side) for scc in self.switched))
        for name, method in methods:
            setattr(self, name, method.__get__(self, type(self)))

    def _module(self, coil: int) -> int:
        return self.sides[self.coil_sides[coil]].coils.index(coil)

    def _capacitor_name(self, coil: int, multi: bool) -> str:
        side = self.coil_sides[coil]
        name = f"u_{side}_series" if self.sides[side].lcc else f"u_{side}_cap"
        return f"{name}_{self._module(coil) + 1}" if multi else name

    def _prepare_holds(self) -> None:
        """Build inverse-port coefficients only when a connected port needs blocking."""
        if hasattr(self, "_hold_weights"):
            return
        n, shapes = len(self._mode_names), self._shapes
        inverse = tuple(tuple(sum(shapes[j][m] * shapes[k][m] / self._inductance[m]
                                  for m in range(n)) for k in range(n)) for j in range(n))
        self._hold_weights = {
            side: tuple(sum(inverse[j][k] for j in self.sides[side].coils) for k in range(n))
            for side in SIDES}
        self._port_gain = {
            side: sum(self._hold_weights[side][k] for k in self.sides[side].coils)
            for side in SIDES}

    def supplied_power(self) -> float:
        return 0.0

    def dissipated_power(self) -> float:
        return self._losses()

    # ------------------------------------------------------------------ blocking ports
    def set_hold_primary(self, t: float) -> None:
        self.out.u_hold_primary = self._hold_primary()

    def set_hold_secondary(self, t: float) -> None:
        self.out.u_hold_secondary = self._hold_secondary()


def _link_spec(cfg: Any) -> dict[str, Any]:
    """Normalize one configured link type into the general coil/side description."""
    kind = cfg.type
    if kind == "ss_multi":
        n = len(cfg.primary_capacitance)
        return {
            "inductance": cfg.inductance,
            "coil_sides": ("primary",) * n + ("secondary",) * n,
            "capacitance": (*cfg.primary_capacitance, *cfg.secondary_capacitance),
            "resistance": (*cfg.primary_resistance, *cfg.secondary_resistance),
            "sides": {"primary": _Side("primary", tuple(range(n))),
                      "secondary": _Side("secondary", tuple(range(n, 2 * n)))},
        }
    matrix = ((cfg.primary_inductance, cfg.mutual_inductance),
              (cfg.mutual_inductance, cfg.secondary_inductance))
    lcc = {"ss": (False, False), "lcc": (True, True), "lcc_s": (True, False)}[kind]
    sides, capacitance = {}, []
    for index, side in enumerate(SIDES):
        if lcc[index]:
            sides[side] = _Side(
                side, (index,), True,
                getattr(cfg, f"{side}_compensation_inductance"),
                getattr(cfg, f"{side}_compensation_resistance"),
                getattr(cfg, f"{side}_parallel_capacitance"),
                getattr(cfg, f"{side}_parallel_resistance"),
            )
            capacitance.append(getattr(cfg, f"{side}_series_capacitance"))
        else:
            sides[side] = _Side(side, (index,))
            capacitance.append(getattr(cfg, f"{side}_capacitance"))
    return {
        "inductance": matrix,
        "coil_sides": SIDES,
        "capacitance": tuple(capacitance),
        "resistance": (cfg.primary_resistance, cfg.secondary_resistance),
        "sides": sides,
    }


def make_wpt_link(cfg: Any) -> CompensatedLink:
    """Build a compensated link with any configured switch-controlled capacitors."""
    spec = _link_spec(cfg)
    switched = []
    for side in SIDES:
        scc = getattr(cfg, f"{side}_scc", None)
        if scc is None:
            continue
        coils = spec["sides"][side].coils
        if len(coils) != 1:
            raise ValueError(f"a switch-controlled capacitor needs one coil on the {side} side")
        switched.append(SwitchControlledCapacitor(
            side, coils[0], scc.capacitance, scc.angle, scc.drive, scc.sync, scc.sync_angle,
            scc.command))
    if not switched:
        return CompensatedLink(**spec)
    return SwitchedCompensatedLink(**spec, switched=tuple(switched))



# ------------------------------------------------------------------ switch-controlled capacitors

_CURRENT_TOL = 1e-9
_VOLTAGE_TOL = 1e-9


class SwitchControlledCapacitor:
    """A capacitor shunted by two source-to-source MOSFETs with their body diodes.

    Positive branch current needs the positive-direction MOSFET channel and the other
    device's channel or body diode; negative current needs the opposite channel. Gate
    commands therefore do not directly declare whether the whole branch conducts.

    ``drive='common'`` applies the same pulse to both gates. ``drive='alternating'``
    uses two pulses of width ``pi + angle``, displaced by pi. Their overlap has the
    commanded nominal conduction angle; body diodes can conduct outside that overlap.
    Every actual voltage-clamping transition accounts for its capacitor discharge energy.
    """

    def __init__(self, side: str, coil: int, capacitance: float, angle: float,
                 drive: str = "common", sync: str = "current", sync_angle: float = 0.0,
                 command: bool = False) -> None:
        if drive not in ("common", "alternating"):
            raise ValueError(f"unknown SCC gate drive {drive!r}")
        if sync not in ("current", "bridge"):
            raise ValueError(f"unknown SCC clock source {sync!r}")
        self.side, self.coil = side, coil
        self.capacitance = _real(capacitance)
        self.angle = self.theta = _real(angle)
        self.drive, self.sync = drive, sync
        self.sync_angle = _real(sync_angle)
        self.command = bool(command)
        self.state_name = f"u_{side}_scc"
        self.bridge = None
        self.started = self.referenced = self.on = False
        self.gates = (False, False)
        self._gate_cache = None
        self.zero = LocatedZero()
        self.half = 0.0
        self._reference = 0.0
        self._since = None
        self.switchings = self.hard_turn_ons = 0
        self.max_turn_on_voltage = self.impulsive_energy = 0.0

    def __copy__(self):
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        result._gate_cache = None
        return result

    @property
    def gates(self):
        return self.positive_gate, self.negative_gate

    @gates.setter
    def gates(self, values):
        self.positive_gate, self.negative_gate = values

    def equation_state_bindings(self):
        return {"sign": (self.zero, "sign"), "i_ref": (self.zero, "residual"),
                **{name: (self, name) for name in (
                    "started", "referenced", "on", "positive_gate", "negative_gate", "theta",
                    "half", "_reference", "switchings", "hard_turn_ons", "max_turn_on_voltage",
                    "impulsive_energy")}}

    def bind_bridge(self, bridge: Any) -> None:
        self.bridge = bridge

    def _command(self) -> float:
        value = self.bridge.register("scc_angle") if self.command else self.angle
        if not np.isfinite(value):
            raise ValueError(f"{self.side} SCC angle must be finite")
        return min(math.pi, max(0.0, _real(value)))

    def _clock(self):
        if self.sync == "bridge":
            clock = self.bridge.phase_clock()
            return PhaseClock(clock.time, clock.angle - self.bridge.phase_reference() - self.sync_angle,
                              clock.rate)
        if not self.referenced or self.half <= 0.0:
            return None
        return PhaseClock(self._reference, math.pi if self.zero.sign < 0.0 else 0.0,
                          math.pi / self.half)

    def _pulses(self, theta):
        if self.drive == "alternating":
            return (0.0, math.pi), 0.5 * (math.pi + theta), 2.0 * math.pi
        return (0.5 * math.pi,), 0.5 * theta, math.pi

    def _gate_word(self, now, clock, theta) -> tuple:
        """Gate word and its validity interval from the existing periodic clock."""
        centres, width, period = self._pulses(theta)
        if width == 0.0:
            return (False, False), -math.inf, math.inf
        if width >= 0.5 * period:
            return (True, True), -math.inf, math.inf
        if clock is None:
            return (False, False), -math.inf, math.inf
        angle = clock.at(now)
        gates, starts, ends = [], [], []
        for centre in centres:
            cycle = _real(np.floor((angle - centre + width) / period))
            middle = clock.time + (centre + period * cycle - clock.angle) / clock.rate
            start, end = middle - width / clock.rate, middle + width / clock.rate
            active = now < end
            gates.append(active)
            starts.append(_real(np.where(active, start, end)))
            ends.append(_real(np.where(active, end, start + period / clock.rate)))
        word = tuple(gates) if len(gates) == 2 else tuple(gates) * 2
        return word, max(starts), min(ends)

    def _gate_state(self, t):
        clock = self._clock()
        theta = self._command() if self.sync == "bridge" else self.theta
        key, now = (clock, theta), t + commutation_tolerance(t)
        cached = self._gate_cache
        if cached is not None and cached[0] == key and cached[1] <= now < cached[2]:
            return cached[3]
        word, start, end = self._gate_word(now, clock, theta)
        # Derived waveform cache only; accepted switch state and restart data stay separate.
        self._gate_cache = key, start, end, word
        return word

    def next_time(self, t: float) -> float:
        """Use the same absolute edge times and due comparison as the dispatcher."""
        expected = self._gate_state(t)
        if not self.started or self.gates != expected:
            return t
        return self._gate_cache[2]

    def _tracks_current(self) -> bool:
        return self.sync == "current" or (self.on and self.gates[0] != self.gates[1])

    def _flow(self, current) -> float:
        return self.zero.sign if abs(current) <= _CURRENT_TOL and self.zero.sign else current

    def margin(self, t: float, current: float, voltage: float) -> float:
        if not self.started or self.gates != self._gate_state(t):
            return -1.0
        margin = self.zero.margin(current, _CURRENT_TOL) if self._tracks_current() else math.inf
        positive, negative = self.gates
        if self.on:
            if not (positive or negative):
                return -1.0
            if positive != negative:
                direction = 1.0 if positive else -1.0
                if self.zero.sign != direction:
                    return -1.0
                margin = min(margin, self.zero.margin(current))
            return margin
        if positive and negative:
            return -1.0
        if positive or negative:
            direction = 1.0 if positive else -1.0
            margin = min(margin, _VOLTAGE_TOL - direction * voltage)
            # Complementarity at zero voltage selects the path carrying the coil current.
            if (abs(voltage) <= _VOLTAGE_TOL
                    and direction * self._flow(current) > 0):
                return -1.0
        return margin

    def commutate(self, link: Any, t: float, current: float, voltage: float) -> None:
        self._synchronize_current(t, current)
        self.gates = self._gate_state(t)
        self._set_conduction(link, current, voltage)

    def _synchronize_current(self, t: float, current: float) -> None:
        if not self.started:
            self.started = True
            if self._since is not None:
                self._reference, self._since = t - self._since, None
        if self.zero.sign == 0.0:
            if abs(current) > _CURRENT_TOL:
                self.zero.sign = _real(np.copysign(1.0, current))
        elif self._tracks_current() and self.zero.margin(current) < 0.0:
            self.zero.locate(current, -self.zero.sign)
            if self.sync == "current":
                if self.referenced:
                    self.half = t - self._reference
                self._reference, self.referenced = t, True
                self.theta = self._command()
        elif not self._tracks_current() and abs(current) > _CURRENT_TOL:
            self.zero.sign = _real(np.copysign(1.0, current))
    def _set_conduction(self, link: Any, current: float, voltage: float) -> None:
        positive, negative = self.gates
        direction = 1.0 if positive else -1.0
        flow = self._flow(current)
        conducting = (positive and negative) or ((positive or negative) and (
            direction * voltage > _VOLTAGE_TOL
            or (abs(voltage) <= _VOLTAGE_TOL and direction * flow > 0.0)))
        if conducting:
            if positive != negative and self.zero.sign == 0.0:
                self.zero.sign = direction
            if not self.on:
                self.switchings += 1
            if abs(voltage) > _VOLTAGE_TOL:
                self.hard_turn_ons += 1
            self.max_turn_on_voltage = max(self.max_turn_on_voltage, abs(voltage))
            self.impulsive_energy += 0.5 * self.capacitance * voltage * voltage
            setattr(link.state, self.state_name, 0.0)
        elif self.on:
            self.switchings += 1
        self.on = conducting

    def get_state(self, t: float) -> dict[str, Any]:
        state = {"on": self.on, "positive_gate": self.gates[0], "negative_gate": self.gates[1],
                 "sign": _real(self.zero.sign), "i_ref": _real(self.zero.residual)}
        if self.sync == "current":
            since = t - self._reference if self.started and self.referenced else self._since or 0.0
            state.update(referenced=self.referenced, since=_real(since), half=_real(self.half),
                         theta=_real(self.theta))
        return state

    def set_state(self, values: Mapping[str, Any]) -> None:
        unknown = set(values) - set(self.get_state(0.0))
        if unknown:
            raise KeyError(f"{self.side} SCC: no state(s) {sorted(unknown)}")
        for name in ("on", "referenced"):
            if name in values:
                setattr(self, name, bool(values[name]))
        self.gates = (bool(values.get("positive_gate", self.gates[0])),
                      bool(values.get("negative_gate", self.gates[1])))
        for name in ("half", "theta"):
            if name in values:
                setattr(self, name, float(values[name]))
        if "sign" in values:
            self.zero.sign = float(values["sign"])
        if "i_ref" in values:
            self.zero.residual = float(values["i_ref"])
        if "since" in values:
            self._since, self.started = float(values["since"]), False

    def diagnostics(self):
        return {"switchings": self.switchings, "hard_turn_ons": self.hard_turn_ons,
                "max_turn_on_voltage": self.max_turn_on_voltage,
                "turn_on_energy": self.impulsive_energy}


class SwitchedCompensatedLink(CompensatedLink):
    """A compensated link with switch-controlled capacitors, a commutating subsystem.

    Its validity margin, commutations and scheduled gate instants come from its capacitors; the
    commutation framework writes back the capacitor voltage zeroed at a turn-on. The switch
    states and schedules are named states (``<side>_scc.*``) of the link.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._t = 0.0

    def _scc_values(self):
        coils = self.coil_currents()
        for scc in self.switched:
            yield scc, coils[scc.coil], _real(getattr(self.state, scc.state_name))

    def commutation_margin(self) -> float:
        return min(scc.margin(self._t, current, voltage)
                   for scc, current, voltage in self._scc_values())

    def commutate(self, t: float) -> None:
        for scc, current, voltage in list(self._scc_values()):
            scc.commutate(self, t, current, voltage)

    def next_commutation_time(self) -> float:
        return min(scc.next_time(self._t) for scc in self.switched)

    def equation_state_bindings(self):
        return {"time": (self, "_t"),
                **{f"{scc.side}_scc.{name}": binding for scc in self.switched
                   for name, binding in scc.equation_state_bindings().items()}}

    @property
    def impulsive_dissipated_energy(self) -> float:
        """Energy (J) dissipated by capacitor discharges at switch closings."""
        return sum(scc.impulsive_energy for scc in self.switched)

    def diagnostics(self) -> dict[str, float]:
        return {f"{scc.side}_scc.{key}": value for scc in self.switched
                for key, value in scc.diagnostics().items()}

    # -- named switch state (a state part of the link) -----------------------------------------
    def get_state(self) -> dict[str, Any]:
        return {f"{scc.side}_scc.{key}": value for scc in self.switched
                for key, value in scc.get_state(self._t).items()}

    def set_state(self, values: Mapping[str, Any]) -> None:
        for scc in self.switched:
            prefix = f"{scc.side}_scc."
            own = {key[len(prefix):]: value for key, value in values.items()
                   if key.startswith(prefix)}
            if own:
                scc.set_state(own)
        unknown = [key for key in values
                   if not any(key.startswith(f"{scc.side}_scc.") for scc in self.switched)]
        if unknown:
            raise KeyError(f"switched link: no state(s) {sorted(unknown)}")

    @property
    def discrete_state_names(self) -> tuple[str, ...]:
        return tuple(f"{scc.side}_scc.{name}" for scc in self.switched
                     for name in ("sign",))

    # Current-derived clock memory affects future gates and remains in the smooth map.
    @property
    def linearization_state_names(self):
        return tuple(f"{scc.side}_scc.{name}" for scc in self.switched if scc.sync == "current"
                     for name in ("since", "half", "theta"))
