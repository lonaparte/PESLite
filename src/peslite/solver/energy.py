"""Port-Hamiltonian energy declarations, power-balance checks and structure reports.

Checks ``P_in + P_supplied = dH/dt + P_dissipated`` per subsystem and a zero sum of
connection-port powers; undeclared subsystems get their net power from neighbouring ports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Protocol, runtime_checkable

import numpy as np

__all__ = ["StoragePort", "PowerPort", "Energetic", "EnergySpec", "EnergyReport", "SubsystemEnergy", "PHReport", "SubsystemStatus", "CutStatus", "signal",
           "re_product", "spec_of", "energy_of", "power_in", "balance", "ph_report"]


# ------------------------------------------------------------------ declarations

@dataclass(frozen=True)
class StoragePort:
    """Linear storage with ``H = scale * value * |state|^2 / 2``.

    ``state``: name of the effort state; ``value``: capacitance or inductance;
    ``scale``: power-convention factor of the signal; ``flows``: external conjugate-flow
    inputs as ``("inp.name", sign)`` pairs; ``kind``: ``"capacitor"`` or ``"inductor"``.
    """

    state: str
    value: float
    scale: float = 1.0
    flows: tuple[tuple[str, float], ...] = ()
    kind: str = "capacitor"


@dataclass(frozen=True)
class PowerPort:
    """Connection port with power entering the subsystem ``P_in = sign * scale * Re(effort * conj(flow))``.

    ``effort`` and ``flow`` name signals of the subsystem: ``"inp.x"``, ``"out.x"`` or ``"state.x"``.
    """

    effort: str
    flow: str
    scale: float = 1.0
    sign: float = 1.0


@runtime_checkable
class Energetic(Protocol):
    """Energy declaration of a subsystem, checked as ``P_in + P_supplied = dH/dt + P_dissipated``.

    Members: ``storage``, ``ports``, ``dissipated_power()`` (W, >= 0) and ``supplied_power()``
    (W, zero if passive). Optional flags: ``dirac`` (lossless interconnection),
    ``observer`` (non-loading reader), ``has_source`` (internal source).
    """

    storage: ClassVar[tuple[StoragePort, ...]]
    ports: ClassVar[tuple[PowerPort, ...]]

    def dissipated_power(self) -> float: ...

    def supplied_power(self) -> float: ...


# ------------------------------------------------------------------ accounting

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


def spec_of(sub: Any) -> EnergySpec:
    """The effective declaration of a subsystem (explicit, or the default).

    The default, for an undeclared subsystem: no storage or dissipation, net power inferred.
    """
    if getattr(sub, "observer", False):
        return EnergySpec("observer")
    if getattr(sub, "dirac", False):
        return EnergySpec("dirac")
    if isinstance(sub, Energetic):
        return EnergySpec("declared", tuple(getattr(sub, "storage", ())), tuple(getattr(sub, "ports", ())),
                          type(sub).dissipated_power, type(sub).supplied_power,
                          has_source=bool(getattr(sub, "has_source", False)))
    return EnergySpec("default", (), (), lambda sub: 0.0, lambda sub: 0.0, has_source=True)


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


def re_product(e: Any, f: Any) -> float:
    """``Re(e conj(f))`` for real or complex signals."""
    return float((e * f.conjugate()).real) if isinstance(e, complex) or isinstance(f, complex) else float(e * f)


def port_power(sub: Any, port: PowerPort) -> float:
    return port.sign * port.scale * re_product(signal(sub, port.effort), signal(sub, port.flow))


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
            p = re_product(other, piece) if is_flow else re_product(piece, other)
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


@dataclass(frozen=True)
class _StorageCheck:
    """Pre-resolved access to one storage state and its derivative-vector entries."""

    record: Any
    field: str
    index: int
    complex_value: bool
    coefficient: float


@dataclass(frozen=True)
class _PowerCheck:
    """Pre-resolved access to one power port."""

    effort_record: Any
    effort_field: str
    flow_record: Any
    flow_field: str
    coefficient: float


@dataclass(frozen=True)
class _DefaultTransfer:
    """One declared-port contribution used to infer a default subsystem's injection."""

    default_index: int
    other_record: Any
    other_field: str
    source_record: Any
    source_field: str
    coefficient: float


@dataclass(frozen=True)
class _SubsystemCheck:
    name: str
    subsystem: Any
    spec: EnergySpec
    default_index: Optional[int]
    storage: tuple[_StorageCheck, ...]
    ports: tuple[_PowerCheck, ...]


@dataclass(frozen=True)
class _BalancePlan:
    """Static topology and state-vector accesses for repeated energy audits of one model."""

    model: Any
    subsystems: tuple[_SubsystemCheck, ...]
    transfers: tuple[_DefaultTransfer, ...]
    n_defaults: int

    def __call__(self, t: float, y: np.ndarray) -> EnergyReport:
        dy = self.model.rhs(t, y)
        # The derivative plan may omit observational output stages.  Energy ports and
        # supplied/dissipated-power methods may still read those outputs, so complete the
        # public-output synchronization promised by balance() before evaluating power.
        self.model.evaluate_outputs(t)
        into_default = [0.0] * self.n_defaults
        for term in self.transfers:
            other = getattr(term.other_record, term.other_field)
            source = getattr(term.source_record, term.source_field)
            into_default[term.default_index] -= term.coefficient * re_product(other, source)

        report = EnergyReport(t)
        for check in self.subsystems:
            spec = check.spec
            if spec.kind == "observer":
                continue
            if spec.kind == "dirac":
                report.dirac.append(check.name)
                continue
            if spec.kind == "default":
                p_in = into_default[check.default_index]  # type: ignore[index]
                report.subsystems.append(
                    SubsystemEnergy(check.name, 0.0, p_in, -p_in, 0.0, 0.0, 0.0, "default")
                )
                continue

            energy = rate = 0.0
            for storage in check.storage:
                x = getattr(storage.record, storage.field)
                if storage.complex_value:
                    dx = complex(dy[storage.index], dy[storage.index + 1])
                    rate += storage.coefficient * (x * dx.conjugate()).real
                else:
                    dx = float(dy[storage.index])
                    rate += storage.coefficient * x * dx
                energy += 0.5 * storage.coefficient * (abs(x) ** 2)
            p_in = sum(
                port.coefficient * re_product(
                    getattr(port.effort_record, port.effort_field),
                    getattr(port.flow_record, port.flow_field),
                )
                for port in check.ports
            )
            supplied = float(spec.supplied(check.subsystem))
            dissipated = float(spec.dissipated(check.subsystem))
            report.subsystems.append(SubsystemEnergy(
                check.name, energy, p_in, supplied, dissipated, rate,
                p_in + supplied - dissipated - rate,
            ))
        return report


def compile_balance(model: Any) -> _BalancePlan:
    """Resolve all static names, connections and state slices used by repeated audits."""
    specs = model.energy_specs
    default_index = {
        id(sub): index
        for index, sub in enumerate(
            sub for name, sub in zip(model.names, model.subsystems)
            if specs[name].kind == "default"
        )
    }

    def signal_ref(sub: Any, name: str) -> tuple[Any, str]:
        where, _, field = name.partition(".")
        return getattr(sub, where), field

    transfers: list[_DefaultTransfer] = []
    checks: list[_SubsystemCheck] = []
    for name, sub in zip(model.names, model.subsystems):
        spec = specs[name]
        storage_checks = []
        for storage in spec.storage:
            state_slice = model.state_slice(sub, storage.state)
            storage_checks.append(_StorageCheck(
                sub.state, storage.state, state_slice.start,
                state_slice.stop - state_slice.start == 2,
                storage.scale * storage.value,
            ))
        power_checks = []
        for port in spec.ports:
            effort_record, effort_field = signal_ref(sub, port.effort)
            flow_record, flow_field = signal_ref(sub, port.flow)
            power_checks.append(_PowerCheck(
                effort_record, effort_field, flow_record, flow_field,
                port.sign * port.scale,
            ))
            for input_name, other_name in ((port.flow, port.effort),
                                           (port.effort, port.flow)):
                where, _, input_field = input_name.partition(".")
                if where != "inp":
                    continue
                sources = model.connections.get((sub, input_field))
                if sources is None:
                    break
                entries = sources if isinstance(sources, list) else [sources]
                other_record, other_field = signal_ref(sub, other_name)
                for entry in entries:
                    source, source_field = entry[0], entry[1]
                    index = default_index.get(id(source))
                    if index is None:
                        continue
                    gain = float(entry[2]) if len(entry) > 2 else 1.0
                    transfers.append(_DefaultTransfer(
                        index, other_record, other_field, source.out, source_field,
                        port.sign * port.scale * gain,
                    ))
                break
        checks.append(_SubsystemCheck(
            name, sub, spec, default_index.get(id(sub)),
            tuple(storage_checks), tuple(power_checks),
        ))
    return _BalancePlan(model, tuple(checks), tuple(transfers), len(default_index))


def balance(model: Any, t: float, y: np.ndarray) -> EnergyReport:
    """Energy accounting of ``model`` at ``(t, y)``; leaves the records synced to ``(t, y)``."""
    plan = getattr(model, "_energy_balance_plan", None)
    if plan is None:
        plan = model._energy_balance_plan = compile_balance(model)
    return plan(t, y)


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
            if getattr(model, "has_state_bounds", False):
                y = model.project_states(y)
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
                op_owners, srcs = owners[id(op)]
                if kind in ("out", "algebraic"):
                    for owner in op_owners:
                        if id(owner) not in member_ids:
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
