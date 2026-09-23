"""Port-Hamiltonian energy declarations, power-balance checks and structure reports.

Checks ``P_in + P_supplied = dH/dt + P_dissipated`` per subsystem and a zero sum of
connection-port powers; undeclared subsystems get their net power from neighbouring ports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .protocols import Energetic, PowerPort, StoragePort

__all__ = ["EnergySpec", "EnergyReport", "SubsystemEnergy", "PHReport", "SubsystemStatus", "CutStatus", "signal",
           "spec_of", "default_spec", "energy_of", "power_in", "balance", "declared", "verify", "ph_report"]


def signal(sub: Any, name: str) -> Any:
    """Read ``"inp.x"`` / ``"out.x"`` / ``"state.x"`` from a subsystem's records."""
    where, _, attr = name.partition(".")
    return getattr(getattr(sub, where), attr)


@dataclass(frozen=True)
class EnergySpec:
    """Effective energy declaration of a subsystem (explicit or default).

    ``kind``: ``"declared"``, ``"default"``, ``"dirac"`` or ``"observer"``.
    ``dissipated`` and ``supplied``: callables of the subsystem returning W
    (for ``"default"``, supplied power is inferred from the neighbours).
    """

    kind: str
    storage: tuple[StoragePort, ...] = ()
    ports: tuple[PowerPort, ...] = ()
    dissipated: Optional[Any] = None
    supplied: Optional[Any] = None
    has_source: bool = False

    @property
    def accounted(self) -> bool:
        return self.kind in ("declared", "default")


def default_spec() -> EnergySpec:
    """Return the declaration of an undeclared subsystem: no storage or dissipation, net power inferred."""
    return EnergySpec("default", (), (), lambda sub: 0.0, lambda sub: 0.0, has_source=True)


def spec_of(sub: Any) -> EnergySpec:
    """The effective declaration of a subsystem (explicit, or the default)."""
    if getattr(sub, "observer", False):
        return EnergySpec("observer")
    if getattr(sub, "dirac", False):
        return EnergySpec("dirac")
    if isinstance(sub, Energetic):
        return EnergySpec("declared", tuple(getattr(sub, "storage", ())), tuple(getattr(sub, "ports", ())),
                          type(sub).dissipated_power, type(sub).supplied_power,
                          has_source=bool(getattr(sub, "has_source", False)))
    return default_spec()


def declared(sub: Any) -> bool:
    """Whether the subsystem declares its energy structure itself."""
    return spec_of(sub).kind == "declared"


def energy_of(sub: Any, spec: Optional[EnergySpec] = None) -> float:
    """Stored energy from the storage declarations (J)."""
    spec = spec or spec_of(sub)
    total = 0.0
    for st in spec.storage:
        x = getattr(sub.state, st.state)
        total += 0.5 * st.scale * st.value * (abs(x) ** 2)
    return total


def energy_rate(sub: Any, derivatives: dict[str, Any], spec: EnergySpec) -> float:
    """``dH/dt`` from the state derivatives ``{state name: value}``."""
    total = 0.0
    for st in spec.storage:
        x, dx = getattr(sub.state, st.state), derivatives[st.state]
        total += st.scale * st.value * (x * np.conj(dx)).real if isinstance(x, complex) else st.scale * st.value * x * dx
    return float(total)


def _re_product(e: Any, f: Any) -> float:
    return float((e * np.conj(f)).real) if isinstance(e, complex) or isinstance(f, complex) else float(e * f)


def port_power(sub: Any, port: PowerPort) -> float:
    return port.sign * port.scale * _re_product(signal(sub, port.effort), signal(sub, port.flow))


def power_in(sub: Any, spec: Optional[EnergySpec] = None) -> float:
    """Power entering the subsystem through its connection ports (W)."""
    spec = spec or spec_of(sub)
    return sum(port_power(sub, port) for port in spec.ports)


def port_terms(model: Any, sub: Any, port: PowerPort) -> list[tuple[Any, float]]:
    """Split a port's power by the subsystem at the other end of each connection.

    Returns ``[(source subsystem, power in W), ...]``; empty when neither port signal is an input.
    """
    for name, is_flow in ((port.flow, True), (port.effort, False)):
        where, _, attr = name.partition(".")
        if where != "inp":
            continue
        sources = model.connections.get((sub, attr))
        if sources is None:
            return []
        entries = sources if isinstance(sources, list) else [sources]
        other = signal(sub, port.effort if is_flow else port.flow)
        terms = []
        for entry in entries:
            src, out_name = entry[0], entry[1]
            gain = float(entry[2]) if len(entry) > 2 else 1.0
            piece = gain * getattr(src.out, out_name)
            p = _re_product(other, piece) if is_flow else _re_product(piece, other)
            terms.append((src, port.sign * port.scale * p))
        return terms
    return []


@dataclass
class SubsystemEnergy:
    name: str
    energy: float  # J
    power_in: float  # W, through the connection ports
    supplied: float  # W, from internal sources (inferred for a default declaration)
    dissipated: float  # W
    energy_rate: float  # dH/dt, W
    residual: float  # power_in + supplied - dissipated - energy_rate, W
    kind: str = "declared"


@dataclass
class EnergyReport:
    """Energy accounting at one instant: per subsystem and over the model."""

    t: float
    subsystems: list[SubsystemEnergy] = field(default_factory=list)
    dirac: list[str] = field(default_factory=list)

    @property
    def defaulted(self) -> list[str]:
        """Subsystems accounted with the default declaration."""
        return [s.name for s in self.subsystems if s.kind == "default"]

    @property
    def energy(self) -> float:
        return sum(s.energy for s in self.subsystems)

    @property
    def supplied(self) -> float:
        return sum(s.supplied for s in self.subsystems)

    @property
    def dissipated(self) -> float:
        return sum(s.dissipated for s in self.subsystems)

    @property
    def tellegen(self) -> float:
        """Sum of the connection-port powers (W); zero for a power-conserving wiring."""
        return sum(s.power_in for s in self.subsystems)

    @property
    def scale(self) -> float:
        """Power scale for relative residuals: the largest port, supplied or dissipated power (W)."""
        return max([abs(s.power_in) for s in self.subsystems] + [abs(s.supplied) for s in self.subsystems]
                   + [abs(s.dissipated) for s in self.subsystems] + [1e-300])

    @property
    def max_residual(self) -> float:
        return max([abs(s.residual) for s in self.subsystems] + [0.0])

    def columns(self) -> dict[str, float]:
        """Flat real columns for a table row."""
        cols = {"energy_J": self.energy, "power_supplied_W": self.supplied, "power_dissipated_W": self.dissipated,
                "tellegen_W": self.tellegen, "balance_residual_W": self.max_residual}
        for s in self.subsystems:
            if s.kind == "default":
                cols[f"{s.name}.power_supplied_W"] = s.supplied  # inferred net injection
            else:
                cols[f"{s.name}.energy_J"] = s.energy
        return cols


def balance(model: Any, t: float, y: np.ndarray) -> EnergyReport:
    """Energy accounting of ``model`` at ``(t, y)``; leaves the records synced to ``(t, y)``."""
    dy = model.rhs(t, y)  # syncs the records and gives the derivatives
    specs = model.energy_specs
    report = EnergyReport(t)
    into_default: dict[int, float] = {}  # power pushed into each defaulted subsystem by declared neighbours
    for name, sub in zip(model.names, model.subsystems):
        spec = specs[name]
        if spec.kind == "declared":
            for port in spec.ports:
                for other, p in port_terms(model, sub, port):
                    if specs[model.name_of(other)].kind == "default":
                        into_default[id(other)] = into_default.get(id(other), 0.0) - p
    for name, sub in zip(model.names, model.subsystems):
        spec = specs[name]
        if spec.kind == "observer":
            continue
        if spec.kind == "dirac":
            report.dirac.append(name)
            continue
        if spec.kind == "default":
            p_in = into_default.get(id(sub), 0.0)
            report.subsystems.append(SubsystemEnergy(name, 0.0, p_in, -p_in, 0.0, 0.0, 0.0, "default"))
            continue
        derivs: dict[str, Any] = {}
        for st in spec.storage:
            sl = model.state_slice(sub, st.state)
            vals = dy[sl]
            derivs[st.state] = complex(vals[0], vals[1]) if sl.stop - sl.start == 2 else float(vals[0])
        p_in = power_in(sub, spec)
        supplied = float(spec.supplied(sub))
        dissipated = float(spec.dissipated(sub))
        rate = energy_rate(sub, derivs, spec)
        report.subsystems.append(SubsystemEnergy(name, energy_of(sub, spec), p_in, supplied, dissipated, rate,
                                                 p_in + supplied - dissipated - rate))
    return report


def verify(model: Any, rtol: float = 1e-8, n_states: int = 4, seed: int = 0,
           zoh: Optional[dict[str, Any]] = None) -> list[str]:
    """Check the declarations at random states and return the problems found (empty if consistent).

    Restores the model's states afterwards. ``zoh``: held inputs to set by label for the check.
    """
    saved = model.get_initial_values()
    rng = np.random.default_rng(seed)
    problems: list[str] = []
    try:
        for label, value in (zoh or {}).items():
            model.set_zoh_input(label, value)
        for k in range(n_states):
            y = rng.normal(size=model.n_states) * 100.0
            rep = balance(model, 0.01 * (k + 1), y)
            tol = rtol * rep.scale
            for s in rep.subsystems:
                if abs(s.residual) > tol:
                    problems.append(f"{s.name}: power balance off by {s.residual:.3e} W "
                                    f"(P_in {s.power_in:.3e}, supplied {s.supplied:.3e}, dissipated {s.dissipated:.3e}, "
                                    f"dH/dt {s.energy_rate:.3e})")
            if abs(rep.tellegen) > tol:
                problems.append(f"Tellegen: connection-port powers sum to {rep.tellegen:.3e} W "
                                f"(declared: {[s.name for s in rep.subsystems if s.kind == 'declared']}, "
                                f"dirac: {rep.dirac}, default: {rep.defaulted})")
            if problems:
                break
    finally:
        model.set_states(saved)
    return sorted(set(problems))


# ------------------------------------------------------------------ the pH report
@dataclass
class SubsystemStatus:
    """How one subsystem stands in the energy structure."""

    name: str
    kind: str  # "storage" | "static" | "dirac" | "observer" | "default"
    states: list[str]
    storages: list[dict]
    ports: list[str]
    balance_residual_rel: Optional[float]  # worst |residual| / power scale over the test states (declared only)
    role: str  # "passive" | "source" | "lossless" | "sensing" | "black box"
    notes: list[str] = field(default_factory=list)


@dataclass
class CutStatus:
    """A multirate group's held states and their energy coverage.

    ``coupling``: ``"window"`` (extrapolated states), ``"step"`` (interpolated states)
    or ``"averaged"`` (window-averaged inputs, no held states).
    """

    group: str
    members: list[str]
    held: list[str]  # states of other groups read by this group's right-hand sides
    storage: list[str]  # those that are declared storage efforts
    unaccounted: list[str]  # held states of subsystems on the default declaration
    admissible: bool
    coupling: str = "step"
    hold: dict = field(default_factory=dict)  # held state -> hold time (s)


@dataclass
class PHReport:
    """Energy declarations, check residuals and multirate cuts of a model.

    ``verdict``: ``"port-hamiltonian"`` (all blocks declare and pass), ``"defaulted"``
    (some use inferred power) or ``"inconsistent"`` (a check failed).
    """

    subsystems: list[SubsystemStatus]
    tellegen_residual_rel: Optional[float]
    problems: list[str]
    cuts: list[CutStatus] = field(default_factory=list)
    n_states: int = 0
    n_states_covered: int = 0
    warnings: list[str] = field(default_factory=list)  # non-fatal remarks on cuts

    @property
    def defaulted(self) -> list[str]:
        """Subsystems running on the default declaration."""
        return [s.name for s in self.subsystems if s.kind == "default"]

    @property
    def verdict(self) -> str:
        if self.problems:
            return "inconsistent"
        return "defaulted" if self.defaulted else "port-hamiltonian"

    @property
    def is_port_hamiltonian(self) -> bool:
        return self.verdict == "port-hamiltonian"

    @property
    def coverage(self) -> float:
        """Fraction of state entries in declared, dirac or observer subsystems."""
        return self.n_states_covered / self.n_states if self.n_states else 1.0

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "coverage": self.coverage,
            "tellegen_residual_rel": self.tellegen_residual_rel,
            "problems": list(self.problems),
            "subsystems": [{"name": s.name, "kind": s.kind, "role": s.role, "states": s.states,
                            "storages": s.storages, "ports": s.ports, "balance_residual_rel": s.balance_residual_rel,
                            "notes": s.notes} for s in self.subsystems],
            "warnings": list(self.warnings),
            "cuts": [{"group": c.group, "members": c.members, "coupling": c.coupling, "held": c.held,
                      "storage": c.storage, "unaccounted": c.unaccounted, "admissible": c.admissible,
                      "hold_s": c.hold} for c in self.cuts],
        }

    def __str__(self) -> str:
        lines = [f"port-Hamiltonian structure: {self.verdict} (declared state coverage {self.coverage:.0%})"]
        for s in self.subsystems:
            extra = ""
            if s.kind == "storage":
                extra = "; " + ", ".join(f"{st['state']}: {st['value']:.4g} x{st['scale']:g}" for st in s.storages)
            res = f", balance {s.balance_residual_rel:.1e}" if s.balance_residual_rel is not None else ""
            lines.append(f"  {s.name:12s} {s.kind:10s} {s.role:9s} states {s.states}{extra}{res}")
            for n in s.notes:
                lines.append(f"  {'':12s} note: {n}")
        if self.tellegen_residual_rel is not None:
            lines.append(f"  Tellegen residual {self.tellegen_residual_rel:.1e} (relative to the power scale)")
        for c in self.cuts:
            if c.coupling == "averaged":
                lines.append(f"  cut {c.group!r} {c.members}: receives the window averages of its inputs "
                             f"(sampled live from {c.held}); holds nothing")
                continue
            lines.append(f"  cut {c.group!r} {c.members} ({c.coupling}): holds {c.held}; storage {c.storage}; "
                         f"unaccounted {c.unaccounted}; admissible: {c.admissible}")
            if c.hold:
                held = ", ".join(f"{state} {t:.3g} s" for state, t in c.hold.items())
                lines.append(f"  {'':12s} held for: {held} (bound = hold x largest rate, evaluated by the solver)")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for p in self.problems:
            lines.append(f"  PROBLEM: {p}")
        return "\n".join(lines)


def _state_labels(model: Any, sub: Any) -> list[str]:
    name = model.name_of(sub)
    return [f"{name}.{st}" for st in sub.state_names]


def ph_report(model: Any, zoh: Optional[dict[str, Any]] = None, rtol: float = 1e-8, seed: int = 0,
              groups: Optional[dict[str, Any]] = None, hold: Optional[dict[str, float]] = None) -> PHReport:
    """Build the :class:`PHReport` of a model and, if given, of its multirate groups.

    Evaluates the declarations at random states and restores the model afterwards.
    ``groups``: ``{label: GroupPlan}`` from :meth:`Model.groups`;
    ``hold``: ``{"step": dt, "window": W}`` hold times in s.
    """
    saved = model.get_initial_values()
    rng = np.random.default_rng(seed)
    statuses: dict[str, SubsystemStatus] = {}
    n_covered = 0
    for name, sub in zip(model.names, model.subsystems):
        states = _state_labels(model, sub)
        n = sum(2 if isinstance(getattr(sub.state, s), complex) else 1 for s in sub.state_names)
        spec = model.energy_specs[name]
        if spec.kind == "observer":
            kind, role = "observer", "sensing"
            n_covered += n
        elif spec.kind == "dirac":
            kind, role = "dirac", "lossless"
            n_covered += n
        elif spec.kind == "declared":
            kind = "storage" if spec.storage else "static"
            role = "source" if spec.has_source else "passive"
            n_covered += n
        else:
            kind, role = "default", "black box"
        storages = [{"state": st.state, "value": st.value, "scale": st.scale, "flows": list(st.flows)}
                    for st in spec.storage]
        ports = [f"{p.effort} x {p.flow} ({p.scale:g}, {p.sign:+g})" for p in spec.ports]
        if kind == "default":
            neighbours: set[str] = set()
            for (dst, _inp), src_spec in model.connections.items():
                entries = src_spec if isinstance(src_spec, list) else [src_spec]
                if dst is sub:
                    neighbours.update(model.name_of(e[0]) for e in entries)
                elif any(e[0] is sub for e in entries):
                    neighbours.add(model.name_of(dst))
            ports = [f"inferred from the ports of {sorted(neighbours)}" if neighbours else "none (unconnected)"]
        statuses[name] = SubsystemStatus(name, kind, states, storages, ports, None, role)
    problems: list[str] = []
    tellegen_worst: Optional[float] = None
    try:
        for label, value in (zoh or {}).items():
            model.set_zoh_input(label, value)
        for k in range(4):
            y = rng.normal(size=model.n_states) * 100.0
            rep = balance(model, 0.01 * (k + 1), y)
            scale = rep.scale
            for s in rep.subsystems:
                st = statuses[s.name]
                if s.kind == "default":
                    continue  # closes by construction
                rel = abs(s.residual) / scale
                st.balance_residual_rel = rel if st.balance_residual_rel is None else max(st.balance_residual_rel, rel)
                if s.supplied != 0.0 and st.role == "passive":
                    st.role = "source"
                if s.dissipated < -rtol * scale and "dissipation < 0" not in st.notes:
                    st.notes.append("dissipation < 0")
                    problems.append(f"{s.name}: negative dissipation {s.dissipated:.3e} W")
                if rel > rtol:
                    problems.append(f"{s.name}: power balance off by {s.residual:.3e} W")
            t_rel = abs(rep.tellegen) / scale
            tellegen_worst = t_rel if tellegen_worst is None else max(tellegen_worst, t_rel)
            if t_rel > rtol:
                problems.append(f"Tellegen: connection-port powers sum to {rep.tellegen:.3e} W")
    finally:
        model.set_states(saved)
    for st in statuses.values():
        if st.kind == "default":
            st.notes.append("default declaration: no storage, no dissipation; its net power is inferred from the "
                            "neighbours' ports and reported as supplied; its states are not storage efforts, so an "
                            "interface through them is measured but not bounded")
    cuts: list[CutStatus] = []
    if groups:
        by_id = {id(s): n for n, s in zip(model.names, model.subsystems)}
        label_of = {id(m): label for label, gp in groups.items() for m in gp.members}
        for label, gp in groups.items():
            member_ids = {id(m) for m in gp.members}
            held_subs: dict[int, Any] = {}
            owners = {id(op): info for op, info in zip(model._plan, model._plan_owners)}
            for op in gp.plan:
                kind, item = op
                owner, srcs = owners[id(op)]
                if kind == "out" and id(owner) not in member_ids:
                    held_subs[id(owner)] = owner
                if kind == "copy":
                    for src in srcs:
                        if id(src) not in member_ids:
                            held_subs[id(src)] = src
            held, storage, unaccounted = [], [], []
            hold_of: dict[str, float] = {}
            for sub in held_subs.values():
                if getattr(sub, "observer", False) or not sub.state_names:
                    continue
                labels = _state_labels(model, sub)
                held += labels
                spec = model.energy_specs[by_id[id(sub)]]
                ports = {st.state for st in spec.storage}
                # hold time: a window for windowed-group states, a step otherwise
                t_hold = None
                if hold and label != "outer":
                    on_window = label_of.get(id(sub), "system") == "outer"
                    t_hold = hold.get("window" if on_window else "step")
                for st_name, lab in zip(sub.state_names, labels):
                    (storage if st_name in ports else unaccounted).append(lab)
                    if t_hold is not None:
                        hold_of[lab] = t_hold
            coupling = "averaged" if label == "outer" else (
                "window" if any(label_of.get(id(s), "system") == "outer" for s in held_subs.values()) else "step")
            cuts.append(CutStatus(label, [by_id[id(m)] for m in gp.members], held, storage, unaccounted,
                                  admissible=not unaccounted or label == "outer", coupling=coupling,
                                  hold=hold_of))
    warns: list[str] = []
    for c in cuts:
        if c.coupling != "averaged" and not any(m.state_names for m in groups[c.group].members):
            warns.append(f"group {c.group!r} has no states: the split only refines the others' steps")
    report = PHReport(list(statuses.values()), tellegen_worst, sorted(set(problems)), cuts,
                      n_states=model.n_states, n_states_covered=n_covered, warnings=warns)
    return report
