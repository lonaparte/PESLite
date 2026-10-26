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

from typing import Any, ClassVar
from functools import lru_cache

import numpy as np

from ..solver.energy import PowerPort, StoragePort
from ..solver.model import Bag, OutputMethod, _real
from .network import Terminal

__all__ = ["CompensatedLink", "make_wpt_link"]

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
def _equation_methods(coil_sides, lcc_sides, cap_names, coil_outputs, capacitor_outputs):
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
    rhs_lines = [*coil_lines, "inp = self.inp", *specs,
                 *(f"node_{side} = {value}" for side, value in nodes.items()), *voltages]
    modes = ["(" + " + ".join(f"a{j}_{m} * v{j}" for j in range(n))
             + f") / self._inductance[{m}]" for m in range(n)]
    derivatives = [*front_currents, *front_voltages, *modes,
                   *(f"i{j} / self.capacitance[{j}]" for j in range(n))]
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
    namespace = {}
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
                 sides: dict[str, _Side]) -> None:
        n = len(coil_sides)
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
            + self._cap_names
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
        )
        self._coil_outputs = (tuple(f"i_{self.coil_sides[j]}_coil_{self._module(j) + 1}"
                                    for j in range(n)) if multi else ())
        capacitor_outputs = tuple(f"u_{side}_parallel" for side in front) + self._cap_names
        self._capacitor_outputs = capacitor_outputs
        outputs = ("i_primary", "i_secondary", "i_primary_coil", "i_secondary_coil",
                   *self._coil_outputs, *capacitor_outputs)
        self.hold_sides = ()
        self._outputs = outputs
        self.out = type("_LinkOut", (Bag,), {"__slots__": outputs})()
        self.output_stages = (OutputMethod("set_outputs", inputs=(), outputs=outputs),)
        self.inp = type("_LinkInp", (Bag,), {"__slots__": ("u_primary", "u_secondary")})()
        self.primary_terminal = Terminal(self, "u_primary", "i_primary", 1)
        self.secondary_terminal = Terminal(self, "u_secondary", "i_secondary", 1)
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
            rows = np.array([[sum(self._shapes[j][m] for j in self.sides[side].coils)
                              for m in range(len(self._mode_names))] for side in active])
            inverse = rows.T / np.asarray(self._inductance)[:, None]
            correction = np.linalg.solve(rows @ inverse, inverse.T).T
            self._current_projections[active] = rows, correction
        rows, correction = self._current_projections[active]
        currents = np.array([getattr(self.state, name) for name in self._mode_names])
        values = np.array(tuple(targets.values()))
        if len(active) == len(currents):
            # A fully constrained state has no free component to preserve. Direct
            # reconstruction also preserves an exactly zero current exactly.
            currents = correction @ values
        else:
            currents -= correction @ (rows @ currents - values)
        for name, value in zip(self._mode_names, currents):
            setattr(self.state, name, float(value))

    def _compile_equations(self) -> None:
        """Bind the shared structural template to this instance's live coefficients.

        The same equations serve Python, parameter tracing and C++ export. Only layout is cached;
        no numerical coefficients, physical state or bound instance escapes into the cache.
        """
        methods = _equation_methods(
            self.coil_sides, tuple(side for side in SIDES if self.sides[side].lcc),
            self._cap_names, self._coil_outputs, self._capacitor_outputs)
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

    def port_hold_voltage(self, side: str) -> float:
        """Port voltage at which the port current of ``side`` has zero rate of change."""
        self._prepare_holds()
        return getattr(self, f"_hold_{side}")()


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
    """Build the configured compensated magnetic link through its ordinary terminal interface."""
    return CompensatedLink(**_link_spec(cfg))
