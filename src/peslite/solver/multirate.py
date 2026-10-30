"""Multirate fixed-step solver, and :func:`make_solver`, which builds the solver of the settings.

Named subsystems step at ``dt/N`` or advance over windows of ``M*dt``. Windowed subsystems advance on
window-averaged inputs while the rest sees their extrapolated states; with every step equal to
``dt``, :func:`make_solver` returns the single-rate integrator itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from .energy import re_product, signal
from .integrators import RHS, TABLEAUS, SolverStep, combine, integrator, stage_state, subintervals
from .model import ConfigError

__all__ = ["MultirateSolver", "HeldStorage", "make_solver", "parse_step"]

_EPS = 1e-12


def make_solver(cfg: Any, model: Optional[Any] = None, ratings: Any = None):
    """Build the solver of the ``simulation.solver`` settings: single-rate, or multirate when
    ``cfg.subsystems`` names subsystems of ``model`` on their own step.

    ``ratings``: mapping storage scale -> (rated effort, rated flow), or a callable
    ``(subsystem_name, storage) -> (effort, flow)``.
    """
    if cfg.type == "fixed" and cfg.subsystems:
        if model is None:
            raise ConfigError("simulation.solver.subsystems needs the model whose subsystems it names")
        try:
            parsed = {name: parse_step(value)[0] for name, value in cfg.subsystems.items()}
            if any(ratio != 1.0 for ratio in parsed.values()):
                return MultirateSolver(model, cfg.subsystems, cfg.dt, cfg.method, cfg.sweeps, cfg.rtol, cfg.atol,
                                       ratings)
            unknown = [name for name in parsed if not model.select(name)]  # every step is dt: single-rate
            if unknown:
                raise KeyError(f"unknown subsystem(s) or namespace(s) {unknown}; this model has {model.names}")
        except (KeyError, ValueError) as exc:
            raise ConfigError(f"simulation.solver.subsystems: {exc.args[0]}") from None
    return integrator(cfg.method, cfg.dt, cfg.rtol, cfg.atol, cfg.max_step, cfg.warm_start)


def parse_step(value: Any) -> tuple[float, Optional[str]]:
    """Parse a step (number relative to ``dt`` or ``{"step": ..., "method": ...}``) into ``(ratio, method)``."""
    method = None
    if isinstance(value, Mapping):
        if "step" not in value:
            raise ValueError("a mapping needs a 'step'")
        method = value.get("method")
        value = value["step"]
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value <= 0.0):
        raise ValueError(f"the step must be a positive number (relative to dt), got {value!r}")
    r = float(value)
    if r >= 1.0:
        if abs(r - round(r)) > 1e-9:
            raise ValueError(f"a step longer than dt must be an integer multiple, got {r}")
        return float(round(r)), method
    inv = 1.0 / r
    if abs(inv - round(inv)) > 1e-9:
        raise ValueError(f"a step shorter than dt must be 1/N with N an integer, got {r}")
    return 1.0 / round(inv), method


def rated_effort(model: Any, ratings: Any) -> Optional[NDArray[np.float64]]:
    """Rated effort per state entry of ``model``'s declared storages (``0.0`` where unrated).

    ``ratings``: mapping storage scale -> (rated effort, rated flow), or a callable
    ``(subsystem_name, storage) -> (effort, flow)``; ``None`` without ratings.
    """
    if not ratings:
        return None
    rated = np.zeros(model.n_states)
    for sub in model.subsystems:
        for st in getattr(sub, "storage", ()):
            rating = ratings(model.name_of(sub), st) if callable(ratings) else ratings.get(st.scale)
            if rating:
                rated[model.state_slice(sub, st.state)] = rating[0] if st.kind == "capacitor" else rating[1]
    return rated


@dataclass(frozen=True)
class HeldStorage:
    """A declared storage held over a window by :class:`MultirateSolver`.

    ``states``: its slice of the model state vector; ``columns``: its columns in each
    :attr:`MultirateSolver.window_log` entry; ``rated``: rated effort (``0.0`` if unrated);
    ``kind``, ``scale``, ``value``: copied from its :class:`StoragePort`.
    """

    label: str
    states: slice
    columns: tuple[int, ...]
    rated: float
    kind: str
    scale: float
    value: float


class MultirateSolver:
    """Explicit RK with system step ``dt`` (s) and, per named subsystem, a step relative to it.

    ``steps``: ``{name: ratio}`` or ``{name: {"step": ratio, "method": ...}}``, ratio ``1/N`` or an
    integer ``M``; ``method``: ``"euler"``, ``"heun"`` or ``"rk4"``; ``sweeps``: coupling sweeps
    for ``1/N`` steps (>= 1); ``rtol`` / ``atol``: for an adaptive windowed method; ``ratings``:
    as in :func:`make_solver`. ``n_system``, ``n_inner`` and ``n_outer`` count RHS evaluations;
    ``groups``: the ``{label: GroupPlan}`` of the split (see :meth:`~peslite.solver.model.Model.groups`).
    """

    def __init__(self, model: Any, steps: Mapping[str, Any], dt: float, method: str = "rk4",
                 sweeps: int = 2, rtol: float = 1e-6, atol: float = 1e-9,
                 ratings: Any = None) -> None:
        if method not in TABLEAUS:
            raise ValueError(f"unknown fixed-step method {method!r}")
        if int(sweeps) < 1:
            raise ValueError("sweeps must be >= 1")
        self.model = model
        self.dt = float(dt)
        self.method = method
        self.sweeps = int(sweeps)
        # storage scale -> (rated effort, rated flow), or a callable (subsystem name, storage) -> that pair
        self.ratings = ratings if callable(ratings) else dict(ratings or {})
        self.n_rhs = 0
        self.n_system = 0
        self.n_inner = 0
        self.n_outer = 0

        inner: dict[str, float] = {}
        outer: dict[str, float] = {}
        outer_method: Optional[str] = None
        for name, value in steps.items():
            try:
                r, meth = parse_step(value)
            except ValueError as exc:
                raise ValueError(f"subsystem {name!r}: {exc}") from None
            if r < 1.0:
                if meth is not None and meth != method:
                    raise ValueError(f"subsystem {name!r}: a step inside the system step shares the system "
                                     f"method ({method!r}); it cannot use {meth!r}")
                inner[name] = r
            elif r > 1.0:
                outer[name] = r
                if meth is not None:
                    if outer_method is not None and meth != outer_method:
                        raise ValueError("subsystems on windows share one method")
                    outer_method = meth
        self.ratio = int(round(1.0 / min(inner.values()))) if inner else 1  # steps per system step
        self.window_steps = int(round(min(outer.values()))) if outer else 1  # system steps per window
        assignment = {n: "inner" for n in inner}
        assignment.update({n: "outer" for n in outer})
        self.groups = groups = model.groups(assignment)  # validates the names
        self.system = groups["system"]
        self.inner = groups.get("inner")
        self.outer = groups.get("outer")
        self._mask_inner = self.inner.mask if self.inner is not None else np.zeros(model.n_states, dtype=bool)
        self._mask_outer = self.outer.mask if self.outer is not None else np.zeros(model.n_states, dtype=bool)
        self._mask_moving = ~self._mask_outer  # what the system step advances (system and inner states)
        self._outer_labels = [lab for lab, m in zip(model.state_labels(), self._mask_outer) if m]
        # windowed-group inputs whose window sums are restart state: (label, input record, field), plan order
        owner = {id(sub.inp): model.name_of(sub) for sub in self.outer.members} if self.outer is not None else {}
        self._input_labels = [(f"{owner[id(inp)]}.{field}", inp, field)
                              for _kind, (inp, field, *_rest) in (self.outer.inputs if self.outer is not None else ())]
        self.window = self.window_steps * self.dt
        self.outer_method = outer_method or method
        self._outer_solver = (integrator(self.outer_method, self.window, rtol, atol)
                              if self.outer is not None else None)
        # window bookkeeping (outer group)
        self._anchor: Optional[NDArray[np.float64]] = None  # outer states at the last window close
        self._slope: Optional[NDArray[np.float64]] = None
        self._t_anchor = 0.0
        self._window_end = math.nan  # the close of the open window, also after one closed early
        self._rate_sampled_at = math.nan  # the anchor time whose live outer rates were sampled
        self._acc: dict[tuple[int, str], Any] = {}
        self._acc_energy: dict[int, float] = {}  # storage state index -> int scale * e_seen * f dt over the window
        self._acc_flow: dict[int, Any] = {}  # storage state index -> running int f dt over the window (all stages)
        self._excursion: dict[int, float] = {}  # storage state index -> largest |running int f dt| in the window
        self._flow_track: dict[int, list[tuple[float, Any]]] = {}  # storage state index -> (tau, int f dt) per step
        self._last_returned: Optional[NDArray[np.float64]] = None
        # interface indicators
        self._peak = np.zeros(model.n_states)  # running peak of |state| per entry
        self._abs_err = np.zeros(model.n_states)  # running max of |predicted - actual| per entry
        self._rate_peak = np.zeros(model.n_states)  # largest |rhs| evaluated per entry
        self._bound_abs = np.zeros(model.n_states)  # running max of the per-hold bound per entry
        self.bound_violations = 0  # holds whose reached value exceeded their bound (per entry)
        self.coarse_hold: Optional[tuple[float, str, float]] = None  # first window hold with bound > 0.1 rated
        # per window: (t_close, |state change|, |prediction error|, ripple), arrays over the outer entries
        self.window_log: list[tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []
        self._energy_residual = 0.0
        self._win_abs: dict[int, float] = {}  # storage index -> worst |residual| of one window (J)
        self._exc_abs: dict[int, float] = {}  # storage index -> worst sub-window change of the effort
        self._windows = 0
        self._steps = 0
        self.parameters_changed()  # _storage_of: state index -> (subsystem, StoragePort)
        self._coupling = np.ones(model.n_states, dtype=bool)  # states that feed back (observers do not)
        for sub in model.subsystems:
            if getattr(sub, "observer", False):
                for name in sub.state_names:
                    self._coupling[model.state_slice(sub, name)] = False
        # rated effort per declared storage entry (0 elsewhere); None without ratings
        self._rated_ref: Optional[NDArray[np.float64]] = rated_effort(model, self.ratings)

    # ------------------------------------------------------------- interface
    def __call__(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64]) -> SolverStep:
        return self.advance(f, t0, t1, y0)

    def advance(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64], *,
                max_step: float = math.inf) -> SolverStep:
        """Advance with a local limit applied coherently to every system and inner step."""
        if max_step <= 0.0:
            raise ValueError("max_step must be positive")
        if t1 - t0 <= 0.0:
            return SolverStep(t1, y0, self.n_rhs)
        if self.outer is None:
            y = self._advance(t0, t1, y0, max_step)
            return SolverStep(t1, y, self.n_rhs)
        # re-anchor if the caller changed the outer states, then split the interval at window ends
        y = np.array(y0, dtype=float)
        if self._anchor is None or (self._last_returned is not None
                                    and not np.array_equal(y0[self._mask_outer], self._last_returned[self._mask_outer])):
            if self._anchor is None or not self._window_end > t0 + _EPS:
                self._window_end = t0 + self.window
            anchor = y0[self._mask_outer].copy()
            self._open_window(t0, anchor, np.zeros_like(anchor))
        t = t0
        while t < t1 - _EPS:
            self._sample_outer_rate(t, y)
            t_close = self._window_end
            t_stop = min(t1, t_close)
            y = self._advance(t, t_stop, y, max_step)
            t = t_stop
            if t >= t_close - _EPS:
                self._close_window(y, None if t_close == self._t_anchor + self.window else t_close)
                self._window_end = self._t_anchor + self.window
                y[self._mask_outer] = self._anchor
        self._last_returned = y.copy()
        return SolverStep(t1, y, self.n_rhs)

    def validate_local_step(self, providers: tuple[Any, ...]) -> None:
        """Reject state-dependent local limits assigned to a window-averaged group."""
        if self.outer is None:
            return
        names = []
        for provider in providers:
            owner = getattr(provider, "local_step_state", provider)
            if owner not in self.model.subsystems:
                continue
            mask = np.zeros(self.model.n_states, dtype=bool)
            for state in getattr(owner, "state_names", ()):
                mask[self.model.state_slice(owner, state)] = True
            if np.any(mask & self._mask_outer):
                names.append(self.model.name_of(owner))
        if names:
            raise ConfigError(
                "state-dependent local step limits cannot be assigned to a windowed "
                f"subsystem: {sorted(set(names))}"
            )

    def settle(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Finish deferred work at ``t`` before the model changes there.

        A shortened window uses its actual duration. The next window still
        closes on the original grid, and the prediction based on the old model
        is discarded even when ``t`` already lies on that grid.
        """
        if self.outer is None or self._anchor is None:
            return y
        y = np.array(y, dtype=float)
        if t > self._t_anchor + _EPS:
            self._close_window(y, t)
            y[self._mask_outer] = self._anchor
        self._slope = np.zeros_like(self._anchor)
        self._rate_sampled_at = math.nan
        self._last_returned = y.copy()
        return y

    def parameters_changed(self) -> None:
        """Refresh storage declarations after a ``set`` event changed parameters."""
        self._storage_of = {
            self.model.state_slice(sub, st.state).start: (sub, st)
            for sub in self.model.subsystems for st in getattr(sub, "storage", ())
        }

    def _sample_outer_rate(self, t: float, y: NDArray[np.float64]) -> None:
        """Evaluate the windowed group's rates once per window to update the rate peaks."""
        assert self.outer is not None
        if self._rate_sampled_at == self._t_anchor:
            return
        self._rate_sampled_at = self._t_anchor
        self.n_outer += 1
        self.n_rhs += 1
        self._note_rate(self.model.rhs_group(t, y, self.outer))

    def _note_rate(self, k: NDArray[np.float64]) -> None:
        np.maximum(self._rate_peak, np.abs(k), out=self._rate_peak)

    def _note_error(self, mask: NDArray[np.bool_], start: NDArray[np.float64], predicted: NDArray[np.float64],
                    actual: NDArray[np.float64], hold: float, t: float, window: bool = False) -> None:
        """Record the prediction error and bound of a hold of ``mask`` entries over ``hold`` s ending at ``t``."""
        np.maximum(self._peak, np.abs(actual), out=self._peak)
        held = mask & self._coupling
        err = np.where(held, np.abs(predicted - actual), 0.0)
        np.maximum(self._abs_err, err, out=self._abs_err)
        assumed = np.abs(predicted - start) / hold
        bound = np.where(held, hold * (self._rate_peak + assumed), 0.0)
        np.maximum(self._bound_abs, bound, out=self._bound_abs)
        self.bound_violations += int(np.count_nonzero(err > bound * (1.0 + 1e-9) + 1e-300))
        if window and self.coarse_hold is None and self._rated_ref is not None:
            rel = np.where(self._rated_ref > 0.0, bound / np.maximum(self._rated_ref, 1e-300), 0.0)
            worst = int(np.argmax(rel))
            if rel[worst] > 0.1:
                self.coarse_hold = (t, self.model.state_labels()[worst], float(rel[worst]))

    def _reference(self) -> NDArray[np.float64]:
        """Per-entry reference: the rated effort where given, otherwise the entry's peak."""
        ref = self._peak.copy()
        if self._rated_ref is not None:
            rated = self._rated_ref > 0.0
            ref[rated] = self._rated_ref[rated]
        return ref

    @property
    def interface(self) -> dict[str, float]:
        """Coupling indicators of the run, relative to rated effort where given, else to the state's peak.

        Keys: ``max_rel_error``, ``excursion_rel_max``, ``energy_residual`` (J), ``energy_window_rel_max``,
        ``windows``, ``steps``, ``kappa`` (hold bound) and ``within_bound`` (1 if no hold exceeded its
        bound); with ratings ``max_error_rated``, ``excursion_rated`` and, once a window hold's bound
        exceeds 0.1 of the rated effort, ``coarse_hold`` (the first such hold, see :attr:`coarse_hold`).
        """
        ref = self._reference()
        seen = ref > 0.0
        ref = np.maximum(ref, 1e-300)
        rel = (self._abs_err / np.maximum(self._peak, 1e-300))[self._peak > 0.0]
        win_rel = exc_rel = err_rated = exc_rated = 0.0
        for idx, (sub, st) in self._storage_of.items():
            sl = self.model.state_slice(sub, st.state)
            reached = float(np.sqrt(np.sum(self._peak[sl] ** 2)))  # peak effort magnitude
            if reached > 0.0:
                win_rel = max(win_rel, self._win_abs.get(idx, 0.0) / (0.5 * st.scale * st.value * reached ** 2))
                exc_rel = max(exc_rel, float(self._exc_abs.get(idx, 0.0) / reached))
            if self._rated_ref is not None and self._rated_ref[sl.start] > 0.0:
                err_rated = max(err_rated, float(np.sqrt(np.sum(self._abs_err[sl] ** 2)) / ref[sl.start]))
                exc_rated = max(exc_rated, float(self._exc_abs.get(idx, 0.0) / ref[sl.start]))
        out = {"max_rel_error": float(np.max(rel)) if rel.size else 0.0,
               "excursion_rel_max": exc_rel,
               "energy_residual": self._energy_residual, "energy_window_rel_max": win_rel,
               "windows": self._windows, "steps": self._steps}
        if self.ratings:
            out["max_error_rated"] = err_rated  # worst held-effort error / rated effort
            out["excursion_rated"] = exc_rated  # worst sub-window change / rated effort
        bound_rel = (self._bound_abs / ref)[seen]  # hold bound relative to the reference
        out["kappa"] = float(np.max(bound_rel)) if bound_rel.size else 0.0
        out["within_bound"] = int(self.bound_violations == 0)
        if self.coarse_hold is not None:
            t_c, label, rel = self.coarse_hold
            out["coarse_hold"] = f"{label} at t = {t_c:.6g} s: {rel:.3g} of rating"
        return out

    @property
    def held_storages(self) -> tuple[HeldStorage, ...]:
        """Declared storages held over a window (empty without a windowed group); see :class:`HeldStorage`."""
        out = []
        for idx, (sub, st) in self._storage_of.items():
            if not self._mask_outer[idx]:
                continue
            sl = self.model.state_slice(sub, st.state)
            labels = self.model.state_labels()
            label = labels[sl.start][:-3] if sl.stop - sl.start == 2 else labels[sl.start]
            pos = {int(i): k for k, i in enumerate(np.flatnonzero(self._mask_outer))}
            out.append(HeldStorage(label=label, states=sl, columns=tuple(pos[j] for j in range(sl.start, sl.stop)),
                                   rated=float(self._rated_ref[sl.start]) if self._rated_ref is not None else 0.0,
                                   kind=st.kind, scale=st.scale, value=st.value))
        return tuple(out)

    # ------------------------------------------------------ Stateful (windows)
    def get_state(self) -> dict[str, Any]:
        """Window bookkeeping (``nan`` before the first window) for exact continuation from any instant.

        ``acc.<input>``: the open window's input integral, complex for a complex input.
        """
        if self.outer is None:
            return {}
        if self._anchor is None:
            s: dict[str, Any] = {"t_anchor": math.nan, "window_end": math.nan}
            for lab in self._outer_labels:
                s[f"anchor.{lab}"] = math.nan
                s[f"slope.{lab}"] = math.nan
        else:
            s = {"t_anchor": self._t_anchor, "window_end": self._window_end}
            for lab, a, sl in zip(self._outer_labels, self._anchor, self._slope):
                s[f"anchor.{lab}"] = float(a)
                s[f"slope.{lab}"] = float(sl)
        for lab, inp, field in self._input_labels:
            s[f"acc.{lab}"] = self._acc.get((id(inp), field), 0.0 * getattr(inp, field))
        return s

    def set_state(self, values: Mapping[str, Any]) -> None:
        if self.outer is None:
            return
        unknown = set(values) - set(self.get_state())
        if unknown:
            raise KeyError(f"MultirateSolver has no state(s) {sorted(unknown)}")
        t_anchor = values.get("t_anchor", math.nan)
        self._rate_sampled_at = math.nan
        if t_anchor != t_anchor:  # nan: no window in progress
            self._anchor = self._slope = None
            return
        anchor = np.zeros(len(self._outer_labels))
        slope = np.zeros(len(self._outer_labels))
        for i, lab in enumerate(self._outer_labels):
            anchor[i] = float(values.get(f"anchor.{lab}", math.nan))
            slope[i] = float(values.get(f"slope.{lab}", 0.0))
        if np.any(np.isnan(anchor)):
            self._anchor = self._slope = None
            return
        self._open_window(float(t_anchor), anchor, slope)
        self._acc = {(id(inp), field): values[f"acc.{lab}"]
                     for lab, inp, field in self._input_labels if f"acc.{lab}" in values}
        end = float(values.get("window_end", math.nan))
        self._window_end = end if end > t_anchor else t_anchor + self.window
        self._last_returned = None  # a loaded anchor is trusted on the next call

    # ------------------------------------------------------ the system step
    def _advance(self, t0: float, t1: float, y0: NDArray[np.float64],
                 max_step: float = math.inf) -> NDArray[np.float64]:
        """System (and inner) steps over ``[t0, t1]``; outer states extrapolated."""
        span = t1 - t0
        n = subintervals(span, min(self.dt, max_step))
        H = span / n
        y, t = y0, t0
        for _ in range(n):
            y = self._macro_step(t, H, y)
            t += H
        return y

    def _outer_at(self, t: float) -> Optional[NDArray[np.float64]]:
        if self._anchor is None:
            return None
        return self._anchor + (t - self._t_anchor) * self._slope

    def _place_outer(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """The state with the outer entries at their extrapolated values for time ``t``."""
        if self._anchor is None:
            return y
        out = np.array(y, dtype=float)
        out[self._mask_outer] = self._outer_at(t)
        return out

    def _f_system(self, t: float, y: NDArray[np.float64], weight: float) -> NDArray[np.float64]:
        """System derivatives; ``weight`` (b_j H) accumulates the outer inputs' quadrature."""
        self.n_system += 1
        self.n_rhs += 1
        k = self.model.rhs_group(t, y, self.system)
        self._note_rate(k)
        if self.outer is not None and weight != 0.0:
            self.outer.run_feed(t)
            for _kind, (dst_inp, dst_name, _s, _n, _f) in self.outer.inputs:
                key = (id(dst_inp), dst_name)
                value = getattr(dst_inp, dst_name)
                self._acc[key] = self._acc.get(key, 0.0) + weight * value
            # energy delivered into each windowed storage at the extrapolated effort
            for idx, (sub, st) in self._storage_of.items():
                if not self._mask_outer[idx]:
                    continue
                e_seen = getattr(sub.state, st.state)  # extrapolated effort
                flow = 0.0
                for name, sign in st.flows:
                    flow += sign * signal(sub, name)
                self._acc_energy[idx] = self._acc_energy.get(idx, 0.0) + weight * st.scale * re_product(e_seen, flow)
                # running flow integral
                run = self._acc_flow.get(idx, 0.0) + weight * flow
                self._acc_flow[idx] = run
                self._excursion[idx] = max(self._excursion.get(idx, 0.0), abs(run))
        return k

    def _f_inner(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        self.n_inner += 1
        self.n_rhs += 1
        k = self.model.rhs_group(t, y, self.inner)
        self._note_rate(k)
        return k

    def _macro_step(self, t0: float, H: float, y0: NDArray[np.float64]) -> NDArray[np.float64]:
        c, a, b = TABLEAUS[self.method]
        m = self.ratio
        h = H / m
        moving = self._mask_moving
        last_sweep = self.sweeps == 1 or self.inner is None
        # -- compound step: shared stages, inner group with step h, the rest with H
        ks: list[NDArray[np.float64]] = []
        k_sys: list[NDArray[np.float64]] = []
        for j in range(len(c)):
            if j == 0:
                y_i = y_s = y0
            else:
                y_i = stage_state(y0, h, a[j - 1], ks)
                y_s = stage_state(y0, H, a[j - 1], ks)
            k_s = self._f_system(t0 + c[j] * H, self._place_outer(t0 + c[j] * H, y_s), b[j] * H if last_sweep else 0.0)
            k_sys.append(k_s)
            if self.inner is not None:
                k_i = self._f_inner(t0 + c[j] * h, self._place_outer(t0 + c[j] * h, y_i))
                ks.append(k_i + k_s)
            else:
                ks.append(k_s)
        y1 = combine(self.method, y0, H, ks)  # system states at t0 + H (inner entries: discarded)
        if self.inner is None:
            self._track_flow(t0 + H)
            return np.where(moving, y1, y0) if self._anchor is not None else y1
        mask_i = self._mask_inner
        # -- remaining inner steps on Hermite-interpolated system states
        k_end = self._f_system(t0 + H, self._place_outer(t0 + H, y1), 0.0)
        traj = self._inner_steps(t0, h, m, y0, y1, k_sys[0], k_end, first=combine(self.method, y0, h, ks))
        seen = y0 + H * (ks[-2] if len(ks) > 1 else ks[0])  # inner states seen by the last system stage
        # -- coupling sweeps
        for sweep in range(2, self.sweeps + 1):
            seen = traj[-1]
            last_sweep = sweep == self.sweeps
            ks2: list[NDArray[np.float64]] = []
            for j in range(len(c)):
                tj = t0 + c[j] * H
                if j == 0:
                    y_s = y0
                else:
                    y_s = np.where(mask_i, self._inner_at(traj, t0, h, tj), stage_state(y0, H, a[j - 1], ks2))
                ks2.append(self._f_system(tj, self._place_outer(tj, y_s), b[j] * H if last_sweep else 0.0))
            y1 = np.where(mask_i, traj[-1], combine(self.method, y0, H, ks2))
            k_end = self._f_system(t0 + H, self._place_outer(t0 + H, y1), 0.0)
            traj = self._inner_steps(t0, h, m, y0, y1, ks2[0], k_end)
        out = np.where(mask_i, traj[-1], y1)
        # inner states seen by the system vs. their final values
        self._note_error(mask_i, y0, seen, out, H, t0 + H)
        self._steps += 1
        self._track_flow(t0 + H)
        return np.where(moving, out, y0) if self._anchor is not None else out

    def _track_flow(self, t: float) -> None:
        """Record the running flow integral of each windowed storage at the end of a system step."""
        if self._anchor is None:
            return
        for idx in self._acc_flow:
            self._flow_track.setdefault(idx, []).append((t - self._t_anchor, self._acc_flow[idx]))

    def _inner_steps(self, t0: float, h: float, m: int, y0: NDArray[np.float64], y1: NDArray[np.float64],
                     k0: NDArray[np.float64], k1: NDArray[np.float64],
                     first: Optional[NDArray[np.float64]] = None) -> list[NDArray[np.float64]]:
        """Run ``m`` inner steps on interpolated system states; return the ``m + 1`` step-end states."""
        c, a, _b = TABLEAUS[self.method]
        mask_i = self._mask_inner
        H = m * h
        s0 = np.where(mask_i, 0.0, y0)
        s1 = np.where(mask_i, 0.0, y1)
        m0 = np.where(mask_i, 0.0, H * k0)
        m1 = np.where(mask_i, 0.0, H * k1)

        def system_at(t: float) -> NDArray[np.float64]:
            u = (t - t0) / H
            u2, u3 = u * u, u * u * u
            return ((2 * u3 - 3 * u2 + 1) * s0 + (u3 - 2 * u2 + u) * m0
                    + (-2 * u3 + 3 * u2) * s1 + (u3 - u2) * m1)

        traj = [y0]
        y, t = y0, t0
        start = 0
        if first is not None:
            y = np.where(mask_i, first, y1)
            traj.append(y)
            t, start = t0 + h, 1
        for _ in range(start, m):
            k_list: list[NDArray[np.float64]] = []
            for j in range(len(c)):
                tj = t + c[j] * h
                if j == 0:
                    yj = np.where(mask_i, y, system_at(tj))
                else:
                    yj = np.where(mask_i, stage_state(y, h, a[j - 1], k_list), system_at(tj))
                k_list.append(self._f_inner(tj, self._place_outer(tj, yj)))
            y = np.where(mask_i, combine(self.method, y, h, k_list), y)
            traj.append(y)
            t += h
        return traj

    @staticmethod
    def _inner_at(traj: list, t0: float, h: float, t: float) -> NDArray[np.float64]:
        """Inner states at ``t`` from the step-end trajectory (exact on the grid, linear between)."""
        x = (t - t0) / h
        i = int(math.floor(x + 1e-9))
        i = min(max(i, 0), len(traj) - 1)
        frac = x - i
        if frac <= 1e-9 or i == len(traj) - 1:
            return traj[i]
        return traj[i] + frac * (traj[i + 1] - traj[i])

    # -------------------------------------------------------------- windows
    def _close_window(self, y: NDArray[np.float64], t_end: Optional[float] = None) -> None:
        """Advance the outer group over its averaged inputs and re-anchor.

        ``t_end`` ends a window shortened by :meth:`settle`.
        """
        assert self.outer is not None and self._anchor is not None and self._outer_solver is not None
        if t_end is None:
            duration, t_end = self.window, self._t_anchor + self.window
        else:
            duration = t_end - self._t_anchor
        t0, t1 = self._t_anchor, t_end
        self.model.set_states(y)
        for _kind, (dst_inp, dst_name, _s, _n, _f) in self.outer.inputs:  # inputs frozen at their averages
            key = (id(dst_inp), dst_name)
            if key in self._acc:
                setattr(dst_inp, dst_name, self._acc[key] / duration)
        y_start = np.array(y, dtype=float)
        y_start[self._mask_outer] = self._anchor

        def f_outer(t: float, yy: NDArray[np.float64]) -> NDArray[np.float64]:
            self.n_outer += 1
            self.n_rhs += 1
            k = self.model.rhs_group(t, yy, self.outer, own=True)
            self._note_rate(k)
            return k

        y_end = self._outer_solver(f_outer, t0, t1, y_start).y
        new = y_end[self._mask_outer]
        predicted = self._outer_at(t1)
        # interface indicators
        full_pred = np.array(y, dtype=float)
        full_pred[self._mask_outer] = predicted
        full_new = np.array(y, dtype=float)
        full_new[self._mask_outer] = new
        self._note_error(self._mask_outer, y_start, full_pred, full_new, duration, t1, window=True)
        residual = 0.0
        for idx, (sub, st) in self._storage_of.items():
            if not self._mask_outer[idx]:
                continue
            sl = self.model.state_slice(sub, st.state)
            is_c = sl.stop - sl.start == 2
            e_old = complex(y_start[sl.start], y_start[sl.start + 1]) if is_c else y_start[sl.start]
            e_new = complex(full_new[sl.start], full_new[sl.start + 1]) if is_c else full_new[sl.start]
            flow = 0.0
            for name, sign in st.flows:
                where, _, attr = name.partition(".")
                flow += sign * self._acc.get((id(getattr(sub, where)), attr), 0.0)
            # energy residual: delivered minus accepted by the storage
            r = self._acc_energy.get(idx, 0.0) - st.scale * re_product(0.5 * (e_old + e_new), flow)
            residual += r
            self._win_abs[idx] = max(self._win_abs.get(idx, 0.0), abs(r))
            # sub-window excursion of the effort
            self._exc_abs[idx] = max(self._exc_abs.get(idx, 0.0), self._excursion.get(idx, 0.0) / st.value)
        self._energy_residual += residual
        self._windows += 1
        # ripple: deviation of the running flow integral from its linear trend over the window
        ripple = np.zeros(int(np.count_nonzero(self._mask_outer)))
        outer_index = {idx: pos for pos, idx in enumerate(np.flatnonzero(self._mask_outer))}
        for idx, (sub, st) in self._storage_of.items():
            if self._mask_outer[idx]:
                track = self._flow_track.get(idx, [])
                total = track[-1][1] if track else 0.0
                stray = max((abs(f - (tau / duration) * total) for tau, f in track), default=0.0) / st.value
                sl = self.model.state_slice(sub, st.state)
                for j in range(sl.start, sl.stop):
                    ripple[outer_index[j]] = stray
        self.window_log.append((t1, np.abs(new - self._anchor), np.abs(predicted - new), ripple))
        self._open_window(t1, new.copy(), (new - self._anchor) / duration)

    def _open_window(self, t: float, anchor: NDArray[np.float64], slope: NDArray[np.float64]) -> None:
        """Anchor the outer states at ``t`` with ``slope`` and start the window's sums empty."""
        self._t_anchor, self._anchor, self._slope = t, anchor, slope
        self._acc, self._acc_energy, self._acc_flow, self._excursion, self._flow_track = {}, {}, {}, {}, {}
