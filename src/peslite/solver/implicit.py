"""Implicit fixed-step integrators and circuit-model linear algebra.

``trapezoidal`` (Tustin) and ``radau`` (three-stage Radau IIA, order 5) solve their stage equations
with a simplified Newton iteration. Its matrix ``I - h (A x J)`` is factorized once per Jacobian and
step and reused while the iteration contracts. The Jacobian ``J`` of ``f`` is formed by finite
differences of ``f`` itself, so these methods need no model-specific derivative code.

The circuit model decides how ``J`` is stored and the linear equations are solved:

* ``device``: one dense ``J``, recomputed when the iteration stops contracting;
* ``statewise``: one dense ``J`` per equation mode, cached and found again by a directional
  probe (the topology is recognized from the equations, not from component internals);
* ``mna``: state-space nodal. The states split automatically into groups coupled only through
  interface variables at the network buses; each group caches its own Jacobian columns per
  topology and is eliminated onto the interface (Schur complement), whose small equation is
  solved jointly.

All three solve the same equations to the same Newton tolerance; they differ in work, not result.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .integrators import IMPLICIT_FIXED_METHODS, RHS, SolverStep, subintervals

__all__ = ["ImplicitSolver", "LinearSystem", "Structure", "circuit_structure", "linear_system",
           "ModeJacobian"]

_S6 = math.sqrt(6.0)
# name -> (A, c, e0): stage equations Z = h A F(Z) + h e0 f(t0, y0), y1 = y0 + Z[-1]
_TABLEAUX: dict[str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = {
    "trapezoidal": (np.array([[0.5]]), np.array([1.0]), np.array([0.5])),
    "radau": (np.array([[(88 - 7 * _S6) / 360, (296 - 169 * _S6) / 1800, (-2 + 3 * _S6) / 225],
                        [(296 + 169 * _S6) / 1800, (88 + 7 * _S6) / 360, (-2 - 3 * _S6) / 225],
                        [(16 - _S6) / 36, (16 + _S6) / 36, 1 / 9]]),
              np.array([(4 - _S6) / 10, (4 + _S6) / 10, 1.0]),
              np.zeros(3)),
}
if tuple(_TABLEAUX) != IMPLICIT_FIXED_METHODS:
    raise RuntimeError("implicit method names and tableaus do not match")

_NEWTON_ITERATIONS = 8
_THETA_MAX = 0.5  # contraction above which the Jacobian is refreshed
_NEWTON_SCALE = 1e-3  # Newton error target relative to atol + rtol |y|
_MAX_SPLITS = 12
_MODE_CACHE = 64  # topologies kept per group
_PROBE_RTOL = 1e-6
_LU = None


def _lu_backend():
    """Load SciPy's reusable LU functions only when an implicit step is first solved."""
    global _LU
    if _LU is None:
        from scipy.linalg import lu_factor, lu_solve

        _LU = lu_factor, lu_solve
    return _LU


def _increment(y: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.5e-8 * np.maximum(np.abs(y), 1.0)


class Structure:
    """State groups of the state-space nodal method: ``groups`` interior index arrays and
    ``interface`` indices. Interior states of different groups never appear in each other's
    derivatives; each group couples to the others only through ``interface``."""

    def __init__(self, groups: Sequence[NDArray[np.intp]], interface: NDArray[np.intp],
                 labels: Sequence[str] = ()) -> None:
        self.groups = [np.asarray(g, dtype=np.intp) for g in groups if len(g)]
        self.interface = np.asarray(interface, dtype=np.intp)
        self.labels = list(labels) if labels else [f"group {k}" for k in range(len(self.groups))]

    @classmethod
    def single(cls, n: int) -> "Structure":
        return cls([np.arange(n)], np.zeros(0, dtype=np.intp), ["system"])

    def merge(self, a: int, b: int) -> None:
        """Merge group ``b`` into group ``a`` (a numerically detected direct coupling)."""
        self.groups[a] = np.sort(np.concatenate([self.groups[a], self.groups[b]]))
        self.labels[a] = f"{self.labels[a]} + {self.labels[b]}"
        del self.groups[b], self.labels[b]


class _Block:
    """Jacobian columns of one group (or of the interface) and their per-topology cache."""

    def __init__(self, cols: NDArray[np.intp], n: int, seed: int) -> None:
        self.cols = cols
        rng = np.random.default_rng(seed)
        self.direction = np.zeros(n)
        self.direction[cols] = rng.uniform(0.5, 1.5, cols.size) * rng.choice((-1.0, 1.0), cols.size)
        self.cache: "OrderedDict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]" = OrderedDict()
        self.version = 0
        self._keys = 0

    def store(self, columns: NDArray[np.float64], signature: NDArray[np.float64]) -> None:
        self._keys += 1
        self.cache[self._keys] = (columns.copy(), signature.copy())
        while len(self.cache) > _MODE_CACHE:
            self.cache.popitem(last=False)

    def find(self, probe: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        for key, (columns, signature) in reversed(self.cache.items()):
            if _same(probe, signature):
                self.cache.move_to_end(key)
                return columns
        return None


def _coupled(J: NDArray[np.float64], rows: NDArray[np.intp], cols: NDArray[np.intp]) -> bool:
    """Whether the derivatives of ``rows`` depend on ``cols`` beyond finite-difference noise."""
    scale = np.max(np.abs(J[rows]), axis=1, initial=0.0)
    return bool(np.any(np.abs(J[np.ix_(rows, cols)]) > 1e-6 * scale[:, None]))


def _same(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    scale = max(float(np.max(np.abs(a), initial=0.0)), float(np.max(np.abs(b), initial=0.0)))
    return float(np.max(np.abs(a - b), initial=0.0)) <= _PROBE_RTOL * scale + 1e-300


class LinearSystem:
    """Jacobian storage and the Newton-matrix solves of one circuit model.

    ``structure``: the groups (``Structure.single`` for one dense block); ``cache``: keep each
    block's columns per topology and recognize a topology again by a directional probe.
    """

    def __init__(self, structure: Optional[Structure] = None, cache: bool = False) -> None:
        self.structure = structure
        self.cache = cache
        self.J: Optional[NDArray[np.float64]] = None
        self.n_rhs = 0
        self.n_jacobians = 0  # Jacobian columns formed by finite differences
        self.n_factorizations = 0
        self._blocks: list[_Block] = []
        self._factors: dict[Any, Any] = {}
        self._parts: dict[Any, Any] = {}  # per-group factorizations, reused while their entries hold

    # ---------------------------------------------------------------- Jacobian
    def invalidate(self) -> None:
        """Forget every Jacobian and factorization (parameters changed)."""
        self.J = None
        self._factors.clear()
        self._parts.clear()
        for block in self._blocks:
            block.cache.clear()
            block.version += 1

    def _setup(self, n: int) -> None:
        if self.structure is None or sum(g.size for g in self.structure.groups) + self.structure.interface.size != n:
            self.structure = Structure.single(n)
        parts = list(self.structure.groups)
        if self.structure.interface.size:
            parts.append(self.structure.interface)
        self._blocks = [_Block(cols, n, seed) for seed, cols in enumerate(parts)]

    def _columns(self, f: RHS, t: float, y: NDArray[np.float64], f0: NDArray[np.float64],
                 cols: NDArray[np.intp]) -> NDArray[np.float64]:
        out = np.empty((y.size, cols.size))
        dy = _increment(y)
        for k, j in enumerate(cols):
            yp = y.copy()
            yp[j] += dy[j]
            out[:, k] = (f(t, yp) - f0) / dy[j]
        self.n_rhs += cols.size
        self.n_jacobians += cols.size
        return out

    def ensure(self, f: RHS, t: float, y: NDArray[np.float64], f0: NDArray[np.float64]) -> None:
        """Form the Jacobian at ``(t, y)`` unless one is held."""
        if self.J is not None and self.J.shape[0] == y.size:
            return
        if not self._blocks or self._blocks[0].direction.size != y.size:
            self._setup(y.size)
        self.J = np.empty((y.size, y.size))
        for block in self._blocks:
            self._refresh_block(f, t, y, f0, block)
        self._check_structure()

    def update(self, f: RHS, t: float, y: NDArray[np.float64], f0: NDArray[np.float64]) -> None:
        """The iteration stopped contracting: find the current topology or form ``J`` again."""
        if self.J is None:
            return self.ensure(f, t, y, f0)
        if not (self.cache and self.lookup(f, t, y, f0)):
            # the equations are nonlinear within the topology, or nothing is cached
            for block in self._blocks:
                self._refresh_block(f, t, y, f0, block)
        self._check_structure()

    def lookup(self, f: RHS, t: float, y: NDArray[np.float64], f0: NDArray[np.float64]) -> bool:
        """Probe every block; take a changed one from its cache or form it. Whether any changed."""
        changed = False
        for block in self._blocks:
            probe = self._probe(f, t, y, f0, block)
            if _same(probe, self.J[:, block.cols] @ block.direction[block.cols]):
                continue
            found = block.find(probe)
            if found is not None:
                self.J[:, block.cols] = found
                block.version += 1
            else:
                self._refresh_block(f, t, y, f0, block, probe)
            changed = True
        return changed

    def _probe(self, f: RHS, t: float, y: NDArray[np.float64], f0: NDArray[np.float64],
               block: _Block) -> NDArray[np.float64]:
        step = 1e-7 * max(1.0, float(np.max(np.abs(y[block.cols]), initial=0.0)))
        self.n_rhs += 1
        return (f(t, y + step * block.direction) - f0) / step

    def _refresh_block(self, f: RHS, t: float, y: NDArray[np.float64], f0: NDArray[np.float64],
                       block: _Block, probe: Optional[NDArray[np.float64]] = None) -> None:
        columns = self._columns(f, t, y, f0, block.cols)
        self.J[:, block.cols] = columns
        block.version += 1
        if self.cache:
            block.store(columns, columns @ block.direction[block.cols] if probe is None else probe)

    def _check_structure(self) -> None:
        """Merge groups whose interior states turn out to be coupled directly."""
        s = self.structure
        if s is None or len(s.groups) < 2:
            return
        merged = False
        a = 0
        while a < len(s.groups):
            b = a + 1
            while b < len(s.groups):
                ga, gb = s.groups[a], s.groups[b]
                if _coupled(self.J, ga, gb) or _coupled(self.J, gb, ga):
                    s.merge(a, b)
                    merged = True
                else:
                    b += 1
            a += 1
        if merged:
            parts = list(s.groups) + ([s.interface] if s.interface.size else [])
            self._blocks = [_Block(cols, self.J.shape[0], seed) for seed, cols in enumerate(parts)]
            self._factors.clear()
            self._parts.clear()
            if self.cache:
                for block in self._blocks:
                    block.store(self.J[:, block.cols], self.J[:, block.cols] @ block.direction[block.cols])

    def jacobian(self) -> NDArray[np.float64]:
        return self.J

    # ---------------------------------------------------------------- solves
    def factor(self, h: float, method: str) -> "_Factor":
        A = _TABLEAUX[method][0]
        s = self.structure
        key = (h, method, tuple(block.version for block in self._blocks))
        factor = self._factors.get(key)
        if factor is None:
            if len(self._factors) > 16:
                self._factors.clear()
            factor = self._factors[key] = _Factor(self.J, s, A, h, self, method)
        return factor


class _Factor:
    """``M = I - h kron(A, J)`` eliminated group by group onto the interface.

    A group whose Jacobian entries (its own block and its coupling to the interface) are unchanged
    reuses its factorization and its Schur contribution from ``owner``.
    """

    def __init__(self, J: NDArray[np.float64], structure: Structure, A: NDArray[np.float64],
                 h: float, owner: LinearSystem, method: str) -> None:
        lu_factor, self._lu_solve = _lu_backend()
        s = A.shape[0]
        self.s = s
        self.groups = structure.groups
        self.interface = structure.interface
        iface = self.interface
        n_i = iface.size
        self.parts = []
        schur = np.eye(s * n_i) - h * np.kron(A, J[np.ix_(iface, iface)]) if n_i else None
        for k, g in enumerate(self.groups):
            entries = (J[np.ix_(g, g)], J[np.ix_(g, iface)], J[np.ix_(iface, g)])
            key = (h, method, k, g.size)
            held = owner._parts.get(key)
            if held is not None and all(np.array_equal(a, b) for a, b in zip(held[0], entries)):
                part = held[1]
            else:
                lu = lu_factor(np.eye(s * g.size) - h * np.kron(A, entries[0]), check_finite=False)
                owner.n_factorizations += 1
                if n_i:
                    M_ig = -h * np.kron(A, entries[2])
                    W = self._lu_solve(lu, -h * np.kron(A, entries[1]), check_finite=False)
                    part = (g, lu, W, M_ig, M_ig @ W)
                else:
                    part = (g, lu, None, None, None)
                if len(owner._parts) > 64:
                    owner._parts.clear()
                owner._parts[key] = (tuple(a.copy() for a in entries), part)
            if n_i:
                schur -= part[4]
            self.parts.append(part[:4])
        self.schur = lu_factor(schur, check_finite=False) if n_i else None
        if n_i:
            owner.n_factorizations += 1

    def solve(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``M x = r`` for the stage-major residual ``r`` of shape ``(s, n)``."""
        lu_solve = self._lu_solve
        x = np.empty_like(r)
        if self.schur is None:
            for g, lu, _w, _m in self.parts:
                x[:, g] = lu_solve(lu, r[:, g].ravel(), check_finite=False).reshape(self.s, g.size)
            return x
        iface = self.interface
        rhs_i = r[:, iface].ravel().copy()
        partial = []
        for g, lu, W, M_ig in self.parts:
            y_g = lu_solve(lu, r[:, g].ravel(), check_finite=False)
            rhs_i -= M_ig @ y_g
            partial.append(y_g)
        x_i = lu_solve(self.schur, rhs_i, check_finite=False)
        x[:, iface] = x_i.reshape(self.s, iface.size)
        for (g, _lu, W, _m), y_g in zip(self.parts, partial):
            x[:, g] = (y_g - W @ x_i).reshape(self.s, g.size)
        return x


class ImplicitSolver:
    """Fixed-step ``trapezoidal`` or ``radau`` with maximum step ``dt`` (s), equal sub-steps that
    land exactly on the interval end, and a simplified Newton iteration to ``1e-3 (atol + rtol |y|)``.

    ``linear``: the :class:`LinearSystem` of the circuit model (a dense one by default).
    """

    def __init__(self, dt: float, method: str = "trapezoidal", rtol: float = 1e-6, atol: float = 1e-9,
                 linear: Optional[LinearSystem] = None) -> None:
        if method not in _TABLEAUX:
            raise ValueError(f"unknown implicit method {method!r}")
        self.dt = float(dt)
        self.method = method
        self.rtol, self.atol = float(rtol), float(atol)
        self.linear = linear if linear is not None else LinearSystem()
        self._A, self._c, self._e0 = _TABLEAUX[method]
        self.n_rhs = 0
        self.n_newton = 0
        self.n_splits = 0

    def __call__(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64]) -> SolverStep:
        return self.advance(f, t0, t1, y0)

    def parameters_changed(self) -> None:
        self.linear.invalidate()

    def advance(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64], *,
                max_step: float = math.inf) -> SolverStep:
        """Integrate with an optional per-call maximum step without changing configuration."""
        span = t1 - t0
        if span <= 0.0:
            return SolverStep(t1, y0, 0)
        step = min(self.dt, max_step)
        if step <= 0.0:
            raise ValueError("max_step must be positive")
        n = subintervals(span, step)
        h = span / n
        y = np.array(y0, dtype=float)
        lin = self.linear
        before = lin.n_rhs
        f0 = f(t0, y)
        self.n_rhs += 1
        lin.ensure(f, t0, y, f0)
        for k in range(n):
            y, f0 = self._step(f, t0 + k * h, y, h, f0, 0)
        self.n_rhs += lin.n_rhs - before
        return SolverStep(t1, y, self.n_rhs)

    def _step(self, f: RHS, t: float, y: NDArray[np.float64], h: float, f0: NDArray[np.float64],
              depth: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        for _attempt in range(2):
            result = self._newton(f, t, y, h, f0)
            if result is not None:
                return result
            self.linear.update(f, t, y, f0)
        if depth >= _MAX_SPLITS:
            raise RuntimeError(f"{self.method}: the Newton iteration does not converge at t = {t:.12g} s")
        self.n_splits += 1
        y_mid, f_mid = self._step(f, t, y, 0.5 * h, f0, depth + 1)
        return self._step(f, t + 0.5 * h, y_mid, 0.5 * h, f_mid, depth + 1)

    def _newton(self, f: RHS, t: float, y: NDArray[np.float64], h: float,
                f0: NDArray[np.float64]) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        A, c, e0 = self._A, self._c, self._e0
        s = c.size
        factor = self.linear.factor(h, self.method)
        explicit = h * np.outer(e0, f0)
        Z = (h * c)[:, None] * f0[None, :]
        scale = _NEWTON_SCALE * (self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y + Z[-1])))
        previous = None
        for _ in range(_NEWTON_ITERATIONS):
            F = np.array([f(t + c[i] * h, y + Z[i]) for i in range(s)])
            self.n_rhs += s
            self.n_newton += 1
            G = Z - h * (A @ F) - explicit
            dZ = -factor.solve(G)
            if not np.all(np.isfinite(dZ)):
                return None
            norm = float(np.max(np.abs(dZ) / scale))
            if previous is None:
                # accept the predictor only if its residual is negligible whatever the matrix
                if norm <= 1e-2 and float(np.max(np.abs(G) / scale)) <= 1e-2:
                    return y + Z[-1], F[-1]
            else:
                theta = norm / previous if previous > 0.0 else 0.0
                if theta >= _THETA_MAX:
                    return None
                if norm / (1.0 - theta) <= 1.0:
                    return y + Z[-1], F[-1]
            previous = norm
            Z = Z + dZ
        return None


class ModeJacobian:
    """``jac(t, y)`` for SciPy's implicit methods from a circuit model's cached Jacobians."""

    def __init__(self, f: RHS, linear: LinearSystem) -> None:
        self.f = f
        self.linear = linear
        self._calls = 0

    @property
    def n_rhs(self) -> int:
        """Right-hand-side evaluations made for the Jacobians."""
        return self._calls + self.linear.n_rhs

    def __call__(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.asarray(y, dtype=float)
        f0 = self.f(t, y)
        self._calls += 1
        if self.linear.J is None or self.linear.J.shape[0] != y.size:
            self.linear.ensure(self.f, t, y, f0)
        elif self.linear.lookup(self.f, t, y, f0):
            self.linear._check_structure()
        return self.linear.J.copy()


def circuit_structure(model: Any) -> Structure:
    """The automatic state-space-nodal grouping of ``model``.

    Network buses are the subsystems that drive their terminal voltage from a state (a shunt
    capacitor). Removing them splits the remaining subsystems into connected groups. A bus
    attached to one group only belongs to it; a bus between groups is an interface node. The
    interface holds the interface nodes' states and every state a group reads from another group
    (for example the branch currents that set a damped bus voltage); the rest of each group is
    its interior.
    """
    subs = list(model.subsystems)
    n = model.n_states
    index = {id(s): np.array(sorted(i for name in s.state_names
                                    for i in range(*model.state_slice(s, name).indices(n))), dtype=np.intp)
             for s in subs}
    candidates = {id(s) for s in subs
                  if getattr(getattr(s, "terminal", None), "drives_voltage", False) and index[id(s)].size}
    adjacent: dict[int, set[int]] = {id(s): set() for s in subs}
    for (dst, _field), spec in model.connections.items():
        for entry in (spec if isinstance(spec, list) else [spec]):
            src = entry[0]
            if src is not dst:
                adjacent[id(dst)].add(id(src))
                adjacent[id(src)].add(id(dst))
    component: dict[int, int] = {}
    count = 0
    for s in subs:
        if id(s) in candidates or id(s) in component:
            continue
        stack = [id(s)]
        component[id(s)] = count
        while stack:
            for other in adjacent[stack.pop()]:
                if other not in candidates and other not in component:
                    component[other] = count
                    stack.append(other)
        count += 1
    nodes = []
    for s in subs:
        if id(s) not in candidates:
            continue
        near = {component[x] for x in adjacent[id(s)] if x in component}
        if len(near) == 1 and not (adjacent[id(s)] & candidates):
            component[id(s)] = near.pop()
        else:
            nodes.append(s)
    members: dict[int, list[Any]] = {}
    for s in subs:
        if id(s) in component:
            members.setdefault(component[id(s)], []).append(s)
    interface = set()
    for s in nodes:
        interface.update(index[id(s)].tolist())
    names = {id(s): model.name_of(s) for s in subs}
    plans = model.groups({names[id(m)]: f"g{k}" for k, group in members.items() for m in group})
    owners = {id(op): info for op, info in zip(model._plan, model._plan_owners)}
    for k, group in members.items():
        inside = {id(m) for m in group}
        for op in plans[f"g{k}"].plan:
            kind = op[0]
            op_owners, srcs = owners[id(op)]
            outside = [o for o in (op_owners if kind in ("out", "algebraic") else srcs if kind == "copy" else ())
                       if id(o) not in inside]
            for other in outside:
                interface.update(index[id(other)].tolist())
    groups, labels = [], []
    for k, group in sorted(members.items()):
        cols = sorted(set(i for m in group for i in index[id(m)].tolist()) - interface)
        if cols:
            groups.append(np.array(cols, dtype=np.intp))
            labels.append(", ".join(names[id(m)] for m in group))
    return Structure(groups, np.array(sorted(interface), dtype=np.intp), labels)


def linear_system(circuit_model: str, model: Any = None) -> LinearSystem:
    """The Jacobian storage and solves of ``circuit_model``."""
    if circuit_model == "device":
        return LinearSystem()
    if circuit_model == "statewise":
        return LinearSystem(cache=True)
    if circuit_model == "mna":
        if model is None:
            raise ValueError("the mna circuit model needs an assembled model")
        return LinearSystem(circuit_structure(model), cache=True)
    raise ValueError(f"unknown circuit model {circuit_model!r}")



