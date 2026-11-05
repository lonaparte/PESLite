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

"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Mapping

import numpy as np

from ..solver.energy import Port, StoragePort
from ..solver.model import Bag, Empty, LocatedZero, OutputMethod, _real, commutation_tolerance
from .network import NetworkComponent
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


class CompensatedLink(NetworkComponent):
    """Coupled coils with per-coil series capacitors and per-side series or LCC compensation."""

    outputs_need_inputs: ClassVar[bool] = False
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
        outputs = ("i_primary", "i_secondary", "i_primary_coil", "i_secondary_coil",
                   *self._coil_outputs, *capacitor_outputs,
                   *(f"{scc.side}_scc_on" for scc in switched))
        self.output = type("_LinkOutput", (Bag,), {"__slots__": outputs})()
        self.output_stages = (OutputMethod("set_outputs", inputs=(), outputs=outputs),)
        self.input = Empty()
        self.primary_port = Port(
            self, "primary", electrical_domain="ac", standalone="voltage",
        )
        self.secondary_port = Port(
            self, "secondary", electrical_domain="ac", standalone="voltage",
        )
        self.ports = (self.primary_port, self.secondary_port)

    def _module(self, coil: int) -> int:
        return self.sides[self.coil_sides[coil]].coils.index(coil)

    def _capacitor_name(self, coil: int, multi: bool) -> str:
        side = self.coil_sides[coil]
        name = f"u_{side}_series" if self.sides[side].lcc else f"u_{side}_cap"
        return f"{name}_{self._module(coil) + 1}" if multi else name

    def set_outputs(self, t: float) -> None:
        self._t = t
        for scc in self.switched:
            setattr(self.output, f"{scc.side}_scc_on", _real(scc.on))

    def coil_currents(self) -> tuple[float, ...]:
        """Coil currents obtained from the stored modal coordinates."""
        modal = tuple(getattr(self.state, name) for name in self._mode_names)
        return tuple(sum(coefficient * current for coefficient, current in zip(row, modal))
                     for row in self._shapes)

    def network(self, nb):
        """Declare the compensated coupled-coil network in coil coordinates."""
        from ..solver import netlist as nl

        nb.port_output(self, "i_primary", self.primary_port, "current")
        nb.port_output(self, "i_secondary", self.secondary_port, "current")
        count = len(self.coil_sides)
        shapes = nl._coefficient_array(self._shapes).reshape(count, count)
        inductance = shapes @ np.diag(nl._coefficient_array(self._inductance)) @ shapes.T
        coils = [nb.internal() for _ in range(count)]
        coupled = nb.add(nl.CoupledInductors(
            [(coil, None) for coil in coils],
            0.5 * (inductance + inductance.T),
            nb.state(self, self._mode_names, transform=shapes),
        ))
        for side in SIDES:
            nb.linear_output(
                self,
                f"i_{side}_coil",
                [(1.0, ("branch_current", coupled, index))
                 for index in self.sides[side].coils],
            )
        for index, name in enumerate(self._coil_outputs):
            nb.output(self, name, ("branch_current", coupled, index))

        def series(node, resistance):
            if not resistance:
                return node
            inner = nb.internal()
            nb.add(nl.Resistor(node, inner, resistance))
            return inner

        switched = {scc.coil: scc for scc in self.switched}
        devices = []
        for side in SIDES:
            spec = self.sides[side]
            node = nb.node(getattr(self, f"{side}_port"))
            if spec.lcc:
                parallel = nb.internal()
                parallel_capacitor = nl.Capacitor(
                    series(parallel, spec.parallel_resistance),
                    None,
                    spec.parallel_capacitance,
                    nb.state(self, f"u_{side}_parallel"),
                )
                nb.add(
                    nl.Inductor(
                        series(node, spec.compensation_resistance),
                        parallel,
                        spec.compensation_inductance,
                        nb.state(self, f"i_{side}_comp"),
                    ),
                    parallel_capacitor,
                )
                nb.output(self, f"u_{side}_parallel", ("branch_voltage", parallel_capacitor))
                node = parallel
            for index in spec.coils:
                scc = switched.get(index)
                terminal = nb.internal() if scc is not None else coils[index]
                capacitor = nb.add(nl.Capacitor(
                    series(node, self.resistance[index]),
                    terminal,
                    self.capacitance[index],
                    nb.state(self, self._cap_names[index]),
                ))
                nb.output(self, self._cap_names[index], ("branch_voltage", capacitor))
                if scc is not None:
                    middle = nb.internal()
                    pair = (
                        nl.Diode(middle, terminal, gate=lambda s=scc: s.positive_gate),
                        nl.Diode(middle, coils[index], gate=lambda s=scc: s.negative_gate),
                    )
                    switched_capacitor = nl.Capacitor(
                            terminal,
                            coils[index],
                            scc.capacitance,
                            nb.state(self, scc.state_name),
                    )
                    nb.add(switched_capacitor, *pair)
                    nb.output(self, scc.state_name, ("branch_voltage", switched_capacitor))
                    devices.append((scc, pair))
        nb.losses(self)
        if devices:
            nb.bind(self._bind_netlist_devices, tuple(devices))
            nb.commutation(
                self,
                margin=self._schedule_margin,
                commutate=self._clock_commutate,
                next_time=self.next_commutation_time,
                after=self._mirror_conduction,
                diodes=[device for _scc, pair in devices for device in pair],
            )

    def _bind_netlist_devices(self, netlist, devices) -> None:
        self._netlist_devices = netlist, tuple(devices)

def _link_spec(cfg: Any) -> dict[str, Any]:
    """Normalize one link type into coil/side form; a compensation inductor marks an LCC side."""
    if cfg.type == "ss_multi":
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
    sides, capacitance = {}, []
    for index, side in enumerate(SIDES):
        if hasattr(cfg, f"{side}_compensation_inductance"):
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
        self._edge = -math.inf

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
                    "impulsive_energy", "_edge")}}

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
        return self.sync == "current" or (self.on & (self.gates[0] != self.gates[1]))

    def _flow(self, current) -> float:
        return (self.zero.sign if self.zero.sign and abs(self.zero.offset(current)) <= _CURRENT_TOL
                else current)

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

    def clock(self, t: float, current: float) -> None:
        """Advance the gate clock while the netlist owns conduction."""
        self._synchronize_current(t, current)
        before, self.gates = self.gates, self._gate_state(t)
        if self.gates != before:
            self._edge = t

    def schedule_margin(self, t: float, current: float) -> float:
        """Return the validity margin of the gate schedule and current clock."""
        if not self.started or self.gates != self._gate_state(t):
            return -1.0
        return self.zero.margin(current, _CURRENT_TOL) if self.sync == "current" else math.inf

    def mirror(self, t: float, conducting: bool, voltage: float) -> None:
        """Mirror netlist conduction into the public SCC state and diagnostics."""
        positive, negative = self.gates
        if conducting:
            if positive != negative and self.zero.sign == 0.0:
                self.zero.sign = 1.0 if positive else -1.0
            if not self.on:
                self.switchings += 1
                if (abs(voltage) > _VOLTAGE_TOL
                        and abs(t - self._edge) <= commutation_tolerance(t)):
                    self.hard_turn_ons += 1
                self.max_turn_on_voltage = max(self.max_turn_on_voltage, abs(voltage))
                self.impulsive_energy += 0.5 * self.capacitance * voltage * voltage
        elif self.on:
            self.switchings += 1
        self.on = bool(conducting)

    def _synchronize_current(self, t: float, current: float) -> None:
        if not self.started:
            self.started = True
            if self._since is not None:
                self._reference, self._since = t - self._since, None
        if self.zero.sign == 0.0 or not self._tracks_current():
            if abs(current) > _CURRENT_TOL:
                self.zero.sign = _real(np.copysign(1.0, current))
        elif self.zero.margin(current) < 0.0:
            self.zero.locate(current, -self.zero.sign)
            if self.sync == "current":
                if self.referenced:
                    self.half = t - self._reference
                self._reference, self.referenced = t, True
                self.theta = self._command()
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
        self._netlist_devices = None

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

    def _schedule_margin(self) -> float:
        coils = self.coil_currents()
        return min(
            scc.schedule_margin(self._t, coils[scc.coil]) for scc in self.switched
        )

    def _clock_commutate(self, t: float) -> None:
        coils = self.coil_currents()
        netlist, devices = self._netlist_devices
        for scc, pair in devices:
            before = scc.gates
            scc.clock(t, coils[scc.coil])
            flow = scc._flow(coils[scc.coil])
            for index, (device, direction) in enumerate(zip(pair, (-1.0, 1.0))):
                if (scc.on and before[index] and not scc.gates[index]
                        and scc.gates[1 - index] and direction * flow > 0.0):
                    netlist.set_conducting(device, True)

    def _mirror_conduction(self, t: float, before: Mapping[str, float]) -> None:
        netlist, devices = self._netlist_devices
        for scc, (positive, negative) in devices:
            conducting = (
                (scc.positive_gate | netlist.conduction(positive)[0])
                & (scc.negative_gate | netlist.conduction(negative)[0])
            )
            scc.mirror(t, conducting, _real(before[scc.state_name]))

    def _restore_conduction(self) -> None:
        if self._netlist_devices is None:
            return
        netlist, devices = self._netlist_devices
        for scc, (positive, negative) in devices:
            netlist.set_conducting(positive, scc.on and not scc.positive_gate)
            netlist.set_conducting(negative, scc.on and not scc.negative_gate)

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
        self._restore_conduction()

    @property
    def discrete_state_names(self) -> tuple[str, ...]:
        return tuple(f"{scc.side}_scc.{name}" for scc in self.switched
                     for name in ("sign",))

    # Current-derived clock memory affects future gates and remains in the smooth map.
    @property
    def linearization_state_names(self):
        return tuple(f"{scc.side}_scc.{name}" for scc in self.switched if scc.sync == "current"
                     for name in ("since", "half", "theta"))
