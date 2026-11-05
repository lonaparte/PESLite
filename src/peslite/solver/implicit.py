"""Implicit fixed-step integrators and shared Jacobian linear algebra.

``trapezoidal`` (Tustin) and ``radau`` (three-stage Radau IIA, order 5) solve their stage equations
with a simplified Newton iteration. The small RK tableau is diagonalized so its Newton equation uses
one shifted state solve per stage instead of one stage-expanded system. Exact present-mode netlist
blocks form the electrical part of the Jacobian; state columns with behavioral or cross-subsystem
feedback and non-electrical states use finite differences of the complete ``f``.

The assembled model may supply state groups coupled through interface states. Each group caches
Jacobian columns and Newton factorizations; without a partition the same code uses one dense block.
Assembly also supplies discrete mode identities, so returning modes reuse their matrices directly.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .integrators import IMPLICIT_FIXED_METHODS, RHS, SolverStep, subintervals

__all__ = ["ImplicitSolver", "LinearSystem", "Structure", "linear_system",
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
_JACOBIAN_STEP = 1.5e-8
_LU = None


def _lu_backend():
    """Load SciPy's reusable LU functions only when an implicit step is first solved."""
    global _LU
    if _LU is None:
        from scipy.linalg import lu_factor, lu_solve

        _LU = lu_factor, lu_solve
    return _LU


def _increment(y: NDArray[np.float64]) -> NDArray[np.float64]:
    return _JACOBIAN_STEP * np.maximum(np.abs(y), 1.0)


class Structure:
    """State partition for block elimination: ``groups`` interior index arrays and
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
        self.cache: "OrderedDict[int, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]" = OrderedDict()
        self.version = 0
        self._revision = 0

    def touch(self) -> None:
        """Assign a new identity to the active matrix contents."""
        self._revision += 1
        self.version = self._revision

    def store(self, columns: NDArray[np.float64], signature: NDArray[np.float64],
              exact: NDArray[np.float64]) -> None:
        self.touch()
        self.cache[self.version] = (columns.copy(), signature.copy(), exact.copy())
        while len(self.cache) > _MODE_CACHE:
            self.cache.popitem(last=False)

    def find(self, probe: NDArray[np.float64], exact: NDArray[np.float64]
             ) -> Optional[tuple[int, NDArray[np.float64]]]:
        for key, (columns, signature, stored_exact) in reversed(self.cache.items()):
            if _same(probe, signature) and np.array_equal(exact, stored_exact):
                self.cache.move_to_end(key)
                return key, columns
        return None


def _coupled(J: NDArray[np.float64], rows: NDArray[np.intp], cols: NDArray[np.intp]) -> bool:
    """Whether the derivatives of ``rows`` depend on ``cols`` beyond finite-difference noise."""
    scale = np.max(np.abs(J[rows]), axis=1, initial=0.0)
    return bool(np.any(np.abs(J[np.ix_(rows, cols)]) > 1e-6 * scale[:, None]))


def _same(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    scale = max(float(np.max(np.abs(a), initial=0.0)), float(np.max(np.abs(b), initial=0.0)))
    return float(np.max(np.abs(a - b), initial=0.0)) <= _PROBE_RTOL * scale + 1e-300


class LinearSystem:
    """Jacobian storage and Newton-matrix solves, independent of circuit representation.

    ``structure``: the groups (``Structure.single`` for one dense block); ``cache``: keep each
    block's columns per topology and recognize a topology again by a directional probe.
    ``exact_blocks`` yields state indices, their exact matrix, exact columns and columns which need
    no finite difference; separating the last two retains feedback outside an exact netlist block.
    """

    def __init__(self, structure: Optional[Structure] = None, cache: bool = False,
                 mode_key: Optional[Callable[[], Any]] = None,
                 exact_blocks: Optional[Callable[[], Sequence[tuple]]] = None) -> None:
        self.structure = structure
        self.cache = cache
        self.mode_key = mode_key
        self.exact_blocks = exact_blocks
        self.J: Optional[NDArray[np.float64]] = None
        self.n_rhs = 0
        self.n_jacobians = 0  # Jacobian columns formed by finite differences
        self.n_factorizations = 0
        self._blocks: list[_Block] = []
        self._factors: dict[Any, Any] = {}
        self._parts: dict[Any, Any] = {}  # per-group factorizations, reused while their entries hold
        self._mode: Any = None
        self._modes: "OrderedDict[Any, NDArray[np.float64]]" = OrderedDict()
        self._exact = np.zeros(0, dtype=bool)

    # ---------------------------------------------------------------- Jacobian
    def invalidate(self) -> None:
        """Forget every Jacobian and factorization (parameters changed)."""
        self.J = None
        self._factors.clear()
        self._parts.clear()
        self._mode = None
        self._modes.clear()
        for block in self._blocks:
            block.cache.clear()
            block.touch()

    def _setup(self, n: int) -> None:
        self._factors.clear()
        self._parts.clear()
        if (self.structure is None or len(self.structure.groups) < 2
                or sum(g.size for g in self.structure.groups) + self.structure.interface.size != n):
            self.structure = Structure.single(n)
        parts = list(self.structure.groups)
        if self.structure.interface.size:
            parts.append(self.structure.interface)
        self._blocks = [_Block(cols, n, seed) for seed, cols in enumerate(parts)]
        self._exact = np.zeros(n, dtype=bool)

    def _refresh_exact(self, *, clear: bool = False) -> bool:
        """Place exact netlist blocks into ``J``; return whether any stored coefficient changed."""
        if self.J is None or self.exact_blocks is None:
            return False
        previous = self.J.copy() if not clear else None
        if clear:
            self.J.fill(0.0)
        self._exact[:] = False
        for block in self.exact_blocks():
            indices, matrix = block[:2]
            indices = np.asarray(indices, dtype=np.intp)
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (indices.size, indices.size):
                raise ValueError("an exact linear block must be square on its state indices")
            entries = indices if len(block) == 2 else np.asarray(block[2], dtype=np.intp)
            positions = {int(index): position for position, index in enumerate(indices)}
            try:
                columns = [positions[int(index)] for index in entries]
            except KeyError:
                raise ValueError("exact block columns must belong to its state indices") from None
            self.J[np.ix_(indices, entries)] = matrix[:, columns]
            skipped = entries if len(block) < 4 else np.asarray(block[3], dtype=np.intp)
            if any(int(index) not in positions for index in skipped):
                raise ValueError("skipped exact columns must belong to their state block")
            self._exact[skipped] = True
        changed = previous is not None and not np.array_equal(previous, self.J)
        if changed:
            for block in self._blocks:
                if np.any(self._exact[block.cols]):
                    block.touch()
        return changed

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
        if not self._blocks or self._blocks[0].direction.size != y.size:
            self._setup(y.size)
        mode = self.mode_key() if self.mode_key is not None else None
        if self.J is not None and self.J.shape[0] == y.size and mode == self._mode:
            self._refresh_exact()
            return
        if mode is not None and mode in self._modes:
            self.J = self._modes[mode].copy()
            self._modes.move_to_end(mode)
            self._mode = mode
            for block in self._blocks:
                block.touch()
            self._refresh_exact()
            self._modes[mode] = self.J.copy()
            return
        self.J = np.empty((y.size, y.size))
        self._refresh_exact(clear=True)
        for block in self._blocks:
            self._refresh_block(f, t, y, f0, block)
        self._refresh_exact()
        self._check_structure()
        self._store_mode(mode)

    def update(self, f: RHS, t: float, y: NDArray[np.float64], f0: NDArray[np.float64]) -> None:
        """The iteration stopped contracting: find the current topology or form ``J`` again."""
        if self.J is None:
            return self.ensure(f, t, y, f0)
        self._refresh_exact()
        mode = self.mode_key() if self.mode_key is not None else None
        if mode is not None:
            for block in self._blocks:
                self._refresh_block(f, t, y, f0, block)
            self._refresh_exact()
            self._mode = mode
            self._store_mode(mode)
        elif not (self.cache and self.lookup(f, t, y, f0)):
            # the equations are nonlinear within the topology, or nothing is cached
            for block in self._blocks:
                self._refresh_block(f, t, y, f0, block)
        self._check_structure()

    def _store_mode(self, mode: Any) -> None:
        """Keep a Jacobian under an assembly-provided exact mode identity."""
        if mode is None or self.J is None:
            return
        self._mode = mode
        self._modes[mode] = self.J.copy()
        self._modes.move_to_end(mode)
        while len(self._modes) > _MODE_CACHE:
            self._modes.popitem(last=False)

    def lookup(self, f: RHS, t: float, y: NDArray[np.float64], f0: NDArray[np.float64]) -> bool:
        """Probe every block; take a changed one from its cache or form it. Whether any changed."""
        changed = self._refresh_exact()
        for block in self._blocks:
            finite = block.cols[~self._exact[block.cols]]
            if not finite.size:
                continue
            probe = self._probe(f, t, y, f0, block)
            if _same(probe, self.J[:, finite] @ block.direction[finite]):
                continue
            exact = block.cols[self._exact[block.cols]]
            found = block.find(probe, self.J[:, exact])
            if found is not None:
                block.version, columns = found
                self.J[:, finite] = columns
            else:
                self._refresh_block(f, t, y, f0, block, probe)
            changed = True
        self._refresh_exact()
        return changed

    def _probe(self, f: RHS, t: float, y: NDArray[np.float64], f0: NDArray[np.float64],
               block: _Block) -> NDArray[np.float64]:
        step = 1e-7 * max(1.0, float(np.max(np.abs(y[block.cols]), initial=0.0)))
        direction = block.direction.copy()
        direction[self._exact] = 0.0
        self.n_rhs += 1
        return (f(t, y + step * direction) - f0) / step

    def _refresh_block(self, f: RHS, t: float, y: NDArray[np.float64], f0: NDArray[np.float64],
                       block: _Block, probe: Optional[NDArray[np.float64]] = None) -> None:
        finite = block.cols[~self._exact[block.cols]]
        columns = self._columns(f, t, y, f0, finite)
        self.J[:, finite] = columns
        if self.cache:
            exact = block.cols[self._exact[block.cols]]
            block.store(
                columns,
                columns @ block.direction[finite] if probe is None else probe,
                self.J[:, exact],
            )
        else:
            block.touch()

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
            self._setup(self.J.shape[0])
            self._refresh_exact()
            if self.cache:
                for block in self._blocks:
                    finite = block.cols[~self._exact[block.cols]]
                    exact = block.cols[self._exact[block.cols]]
                    block.store(
                        self.J[:, finite],
                        self.J[:, finite] @ block.direction[finite],
                        self.J[:, exact],
                    )

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


class _ShiftFactor:
    """Factor ``I - shift J`` group by group onto the interface."""

    def __init__(self, J: NDArray[np.float64], structure: Structure, shift: complex,
                 owner: LinearSystem, key: tuple) -> None:
        lu_factor, self._lu_solve = _lu_backend()
        self.groups = structure.groups
        self.interface = structure.interface
        iface = self.interface
        n_i = iface.size
        self.parts = []
        dtype = complex if np.iscomplexobj(shift) else float
        schur = np.eye(n_i, dtype=dtype) - shift * J[np.ix_(iface, iface)] if n_i else None
        for k, g in enumerate(self.groups):
            entries = (J[np.ix_(g, g)], J[np.ix_(g, iface)], J[np.ix_(iface, g)])
            part_key = (*key, k, g.size)
            held = owner._parts.get(part_key)
            if held is not None and all(np.array_equal(a, b) for a, b in zip(held[0], entries)):
                part = held[1]
            else:
                lu = lu_factor(np.eye(g.size, dtype=dtype) - shift * entries[0], check_finite=False)
                owner.n_factorizations += 1
                if n_i:
                    M_ig = -shift * entries[2]
                    W = self._lu_solve(lu, -shift * entries[1], check_finite=False)
                    part = (g, lu, W, M_ig, M_ig @ W)
                else:
                    part = (g, lu, None, None, None)
                if len(owner._parts) > 64:
                    owner._parts.clear()
                owner._parts[part_key] = (tuple(a.copy() for a in entries), part)
            if n_i:
                schur -= part[4]
            self.parts.append(part[:4])
        self.schur = lu_factor(schur, check_finite=False) if n_i else None
        if n_i:
            owner.n_factorizations += 1

    def solve(self, r: NDArray) -> NDArray:
        """Solve one shifted state system."""
        lu_solve = self._lu_solve
        x = np.empty_like(r)
        if self.schur is None:
            for g, lu, _w, _m in self.parts:
                x[g] = lu_solve(lu, r[g], check_finite=False)
            return x
        iface = self.interface
        rhs_i = r[iface].copy()
        partial = []
        for g, lu, W, M_ig in self.parts:
            y_g = lu_solve(lu, r[g], check_finite=False)
            rhs_i -= M_ig @ y_g
            partial.append(y_g)
        x_i = lu_solve(self.schur, rhs_i, check_finite=False)
        x[iface] = x_i
        for (g, _lu, W, _m), y_g in zip(self.parts, partial):
            x[g] = y_g - W @ x_i
        return x


class _Factor:
    """Diagonalize the small Runge--Kutta tableau and solve one shifted state system per stage.

    Radau therefore factors three ``n``-state systems instead of one ``3n`` system. Each shifted
    system uses the same netlist-derived group/interface partition and cached Schur elimination.
    """

    def __init__(self, J: NDArray[np.float64], structure: Structure, A: NDArray[np.float64],
                 h: float, owner: LinearSystem, method: str) -> None:
        self.s = A.shape[0]
        values, vectors = np.linalg.eig(A)
        self.vectors = vectors
        self.inverse = np.linalg.inv(vectors)
        self.parts: list[_ShiftFactor] = [None] * self.s
        self.real: set[int] = set()
        self.conjugate: set[int] = set()
        pending = set(range(self.s))
        factor_index = 0
        while pending:
            stage = min(pending)
            pending.remove(stage)
            value = values[stage]
            if abs(value.imag) <= 1e-14 * max(1.0, abs(value.real)):
                shift: complex | float = h * float(value.real)
                self.real.add(stage)
            else:
                partner = next(
                    (index for index in pending
                     if abs(values[index] - value.conjugate())
                     <= 1e-12 * max(1.0, abs(value))),
                    None,
                )
                if partner is None:
                    raise RuntimeError("a real implicit tableau has an unpaired complex eigenvalue")
                pending.remove(partner)
                shift = h * value
            factor = _ShiftFactor(
                J, structure, shift, owner, (h, method, factor_index)
            )
            factor_index += 1
            self.parts[stage] = factor
            if not isinstance(shift, float):
                self.parts[partner] = factor
                self.conjugate.add(partner)

    def solve(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve ``(I - h A kron J) x = r`` in the tableau eigenbasis."""
        transformed = self.inverse @ np.asarray(r, dtype=complex)
        for stage, factor in enumerate(self.parts):
            if stage in self.real:
                transformed[stage] = factor.solve(transformed[stage].real)
            elif stage in self.conjugate:
                transformed[stage] = np.conj(factor.solve(np.conj(transformed[stage])))
            else:
                transformed[stage] = factor.solve(transformed[stage])
        result = self.vectors @ transformed
        return np.asarray(result.real, dtype=float)


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
        self.linear = linear if linear is not None else linear_system()
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
    """``jac(t, y)`` for SciPy's implicit methods from the shared Jacobian cache."""

    def __init__(self, f: Optional[RHS], linear: LinearSystem) -> None:
        self.f = f
        self.linear = linear
        self._calls = 0

    def bind(self, f: RHS) -> None:
        """Use the actual integrated RHS, discarding caches if the callable changed."""
        if f != self.f:
            self.linear.invalidate()
            self.f = f

    @property
    def n_rhs(self) -> int:
        """Right-hand-side evaluations made for the Jacobians."""
        return self._calls + self.linear.n_rhs

    def __call__(self, t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.asarray(y, dtype=float)
        f0 = self.f(t, y)
        self._calls += 1
        previous_mode = self.linear._mode
        self.linear.ensure(self.f, t, y, f0)
        if previous_mode == self.linear._mode and self.linear.lookup(self.f, t, y, f0):
            self.linear._check_structure()
        return self.linear.J.copy()


def linear_system(model: Any = None) -> LinearSystem:
    """Use the assembled state partition when supplied, otherwise one dense block."""
    structure = getattr(model, "linear_structure", None)
    mode_key = getattr(model, "linear_mode_key", None)
    exact_blocks = getattr(model, "linear_blocks", None)
    return LinearSystem(
        structure() if callable(structure) else None,
        cache=True,
        mode_key=mode_key if callable(mode_key) else None,
        exact_blocks=exact_blocks if callable(exact_blocks) else None,
    )
