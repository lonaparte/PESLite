"""Nonlinear magnetic-circuit laws connected through the MagPort state boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .magnetic import (
    MagneticBranchDescription,
    MagneticParams,
    _validate_winding_names,
)
from ...components.network import register_component_type
from ...solver import ConfigError
from .magport import MagPortElement

__all__ = ["MagCircuit", "MagCircuitLaw"]


def _matrix(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a finite numeric matrix") from None
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ConfigError(f"{name} must be a finite numeric matrix")
    return array


def _reluctance(value: Any, branches: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError("reluctance must be a finite numeric vector or matrix") from None
    if array.ndim == 1:
        array = np.diag(array)
    expected = len(array) if branches is None else branches
    if array.shape != (expected, expected) or not expected:
        raise ConfigError(f"reluctance must have shape ({expected}, {expected})")
    if not np.isfinite(array).all() or not np.allclose(array, array.T, rtol=1e-12, atol=0.0):
        raise ConfigError("reluctance must be finite and symmetric")
    if np.linalg.eigvalsh(array).min() <= 0.0:
        raise ConfigError("reluctance must be positive definite")
    return array


def _flux_basis(incidence: Any | None, branches: int) -> np.ndarray:
    if incidence is None:
        return np.eye(branches)
    constraints = _matrix(incidence, "incidence")
    if constraints.shape[1] != branches:
        raise ConfigError(f"incidence must have {branches} columns")
    _left, singular, right = np.linalg.svd(constraints, full_matrices=True)
    tolerance = singular.max(initial=0.0) * max(constraints.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tolerance))
    basis = right[rank:].T
    if not basis.shape[1]:
        raise ConfigError("incidence leaves no independent magnetic flux path")
    return basis


def _linear_circuit(
    reluctance: Any, coupling: Any, incidence: Any | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    turns = _matrix(coupling, "coupling")
    if not turns.shape[0] or not turns.shape[1]:
        raise ConfigError("coupling must have at least one branch and winding")
    matrix = _reluctance(reluctance, turns.shape[0])
    basis = _flux_basis(incidence, len(matrix))
    reduced = basis.T @ matrix @ basis
    response = basis @ np.linalg.solve(reduced, basis.T @ turns)
    inductance = turns.T @ response
    inductance = 0.5 * (inductance + inductance.T)
    if np.linalg.eigvalsh(inductance).min() <= 0.0:
        raise ConfigError("magnetic circuit does not produce a positive-definite winding law")
    return inductance, response, matrix


class MagCircuitLaw:
    """Winding law obtained from magnetic flux continuity and branch MMF relations.

    A numeric ``reluctance`` gives the exact linear solution. A callable ``mmf`` receives
    ``(t, branch_flux, state)`` and may implement nonlinear or state-dependent material behavior.
    The optional ``differential`` callable returns its branch-MMF Jacobian. Optional internal state
    is advanced by ``state_derivatives(t, branch_flux, winding_current, state)``. Conservative
    nonlinear laws may provide stored-energy callbacks for exact energy accounting.
    """

    def __init__(
        self,
        winding_names: Sequence[str],
        coupling: Any,
        *,
        incidence: Any | None = None,
        reluctance: Any | None = None,
        mmf: Callable[[float, np.ndarray, Mapping[str, float]], Sequence[float]] | None = None,
        differential: Callable[
            [float, np.ndarray, Mapping[str, float]], Any
        ] | None = None,
        initial_state: Mapping[str, float] | None = None,
        state_derivatives: Callable[
            [float, np.ndarray, np.ndarray, Mapping[str, float]], Mapping[str, float]
        ] | None = None,
        stored_energy: Callable[
            [float, np.ndarray, Mapping[str, float]], float
        ] | None = None,
        stored_energy_rate: Callable[
            [
                float,
                np.ndarray,
                np.ndarray,
                Mapping[str, float],
                np.ndarray,
                Mapping[str, float],
            ],
            float,
        ] | None = None,
    ) -> None:
        self.winding_names = _validate_winding_names(winding_names)
        self.coupling = _matrix(coupling, "coupling")
        branches, windings = self.coupling.shape
        if not branches or windings != len(self.winding_names):
            raise ConfigError(
                f"coupling must have shape (branches, {len(self.winding_names)})"
            )
        if np.linalg.matrix_rank(self.coupling) != windings:
            raise ConfigError("coupling must independently excite every winding")
        self.flux_basis = _flux_basis(incidence, branches)
        self._linkage_map = self.coupling.T @ self.flux_basis
        if np.linalg.matrix_rank(self._linkage_map) != windings:
            raise ConfigError(
                "flux constraints and coupling must independently define every winding linkage"
            )
        self.initial_state = dict(initial_state or {})
        self._state_derivatives = state_derivatives
        if bool(self.initial_state) != (state_derivatives is not None):
            raise ConfigError(
                "magnetic-circuit internal states require state_derivatives and vice versa"
            )
        if (reluctance is None) == (mmf is None):
            raise ConfigError("provide exactly one of reluctance or mmf")
        if stored_energy is None and stored_energy_rate is not None:
            raise ConfigError("stored_energy_rate requires stored_energy")
        if stored_energy is not None and self.initial_state and stored_energy_rate is None:
            raise ConfigError(
                "state-dependent magnetic energy requires stored_energy_rate"
            )

        self._mmf = mmf
        self._differential = differential
        self._stored_energy = stored_energy
        self._stored_energy_rate = stored_energy_rate
        self._linear_current: np.ndarray | None = None
        self._linear_flux: np.ndarray | None = None
        self._guess = np.zeros(self.flux_basis.shape[1] + windings)
        self._last_flux = np.zeros(branches)
        self._last_key: tuple | None = None
        self._last_current = np.zeros(windings)
        if reluctance is not None:
            inductance, response, matrix = _linear_circuit(
                reluctance, self.coupling, incidence
            )
            self.reluctance = matrix
            self._linear_current = np.linalg.solve(
                inductance, np.eye(len(inductance))
            )
            self._linear_flux = response @ self._linear_current
            self._stored_energy = lambda _t, flux, _state: float(
                0.5 * flux @ matrix @ flux
            )
        else:
            self.reluctance = None
            self._validate_nonlinear_origin()
        if self._stored_energy is not None:
            self.stored_energy = self._energy_value
            self.stored_energy_rate = self._energy_rate

    def _constrained_guess(
        self, t: float, flux_linkage: np.ndarray, state: Mapping[str, float]
    ) -> np.ndarray:
        """Project the previous flux onto the requested linkage before nonlinear iteration."""
        paths = self.flux_basis.shape[1]
        q = self._guess[:paths].copy()
        linkage_error = flux_linkage - self._linkage_map @ q
        gram = self._linkage_map @ self._linkage_map.T
        q += self._linkage_map.T @ np.linalg.solve(gram, linkage_error)
        flux = self.flux_basis @ q
        magnetic_drive = self.flux_basis.T @ self._branch_mmf(t, flux, state)
        current_drive = self.flux_basis.T @ self.coupling
        current = np.linalg.lstsq(current_drive, magnetic_drive, rcond=None)[0]
        return np.concatenate((q, current))

    def _validate_nonlinear_origin(self) -> None:
        flux = np.zeros(self.coupling.shape[0])
        self._branch_mmf(0.0, flux, self.initial_state)
        if self._differential is not None:
            self._branch_jacobian(0.0, flux, self.initial_state)

    def _branch_mmf(
        self, t: float, flux: np.ndarray, state: Mapping[str, float]
    ) -> np.ndarray:
        try:
            value = np.asarray(self._mmf(t, flux, state), dtype=float)  # type: ignore[misc]
        except (TypeError, ValueError):
            raise ConfigError("magnetic-circuit mmf must return a finite branch vector") from None
        if value.shape != flux.shape or not np.isfinite(value).all():
            raise ConfigError("magnetic-circuit mmf must return a finite branch vector")
        return value

    def _branch_jacobian(
        self, t: float, flux: np.ndarray, state: Mapping[str, float]
    ) -> np.ndarray:
        try:
            value = np.asarray(self._differential(t, flux, state), dtype=float)  # type: ignore[misc]
        except (TypeError, ValueError):
            raise ConfigError(
                "magnetic-circuit differential must return a finite branch matrix"
            ) from None
        if value.ndim == 1:
            value = np.diag(value)
        expected = (len(flux), len(flux))
        if value.shape != expected or not np.isfinite(value).all():
            raise ConfigError(
                f"magnetic-circuit differential must have shape {expected}"
            )
        return value

    def currents(
        self, t: float, flux_linkage: np.ndarray, state: Mapping[str, float]
    ) -> np.ndarray:
        if self._linear_current is not None:
            current = self._linear_current @ flux_linkage
            self._last_flux = self._linear_flux @ flux_linkage
            self._last_current = current.copy()
            return current
        key = (
            float(t),
            *map(float, flux_linkage),
            *((name, float(state[name])) for name in self.initial_state),
        )
        if key == self._last_key:
            return self._last_current.copy()

        basis, turns = self.flux_basis, self.coupling
        paths = basis.shape[1]

        def residual(value: np.ndarray) -> np.ndarray:
            flux = basis @ value[:paths]
            current = value[paths:]
            return np.concatenate((
                basis.T @ (self._branch_mmf(t, flux, state) - turns @ current),
                turns.T @ flux - flux_linkage,
            ))

        jacobian = None
        if self._differential is not None:
            def jacobian(value: np.ndarray) -> np.ndarray:
                flux = basis @ value[:paths]
                top_left = basis.T @ self._branch_jacobian(t, flux, state) @ basis
                return np.block([
                    [top_left, -basis.T @ turns],
                    [turns.T @ basis, np.zeros((len(self.winding_names),
                                                len(self.winding_names)))],
                ])

        from scipy.optimize import root

        initial = self._constrained_guess(t, flux_linkage, state)
        scale = 1.0 + np.linalg.norm(flux_linkage, ord=np.inf)
        if np.linalg.norm(residual(initial), ord=np.inf) <= 1e-9 * scale:
            solution_value = initial
        else:
            solution = root(residual, initial, jac=jacobian)
            if not solution.success or not np.isfinite(solution.x).all():
                raise ConfigError(f"magnetic-circuit solve failed: {solution.message}")
            solution_value = solution.x
        if np.linalg.norm(residual(solution_value), ord=np.inf) > 1e-9 * scale:
            raise ConfigError("magnetic-circuit solve did not satisfy its constitutive equations")
        self._guess = solution_value.copy()
        self._last_flux = basis @ solution_value[:paths]
        self._last_current = solution_value[paths:].copy()
        self._last_key = key
        return self._last_current.copy()

    def derivatives(
        self,
        t: float,
        flux_linkage: np.ndarray,
        currents: np.ndarray,
        state: Mapping[str, float],
    ) -> Mapping[str, float]:
        if self._state_derivatives is None:
            return {}
        self.currents(t, flux_linkage, state)
        return self._state_derivatives(t, self._last_flux.copy(), currents, state)

    def _energy_value(
        self, t: float, flux_linkage: np.ndarray, state: Mapping[str, float]
    ) -> float:
        self.currents(t, flux_linkage, state)
        return float(self._stored_energy(t, self._last_flux.copy(), state))

    def _energy_rate(
        self,
        t: float,
        flux_linkage: np.ndarray,
        currents: np.ndarray,
        state: Mapping[str, float],
        flux_rate: np.ndarray,
        state_rate: Mapping[str, float],
    ) -> float:
        if self._stored_energy_rate is not None:
            self.currents(t, flux_linkage, state)
            return float(self._stored_energy_rate(
                t,
                self._last_flux.copy(),
                currents,
                state,
                flux_rate,
                state_rate,
            ))
        return float(currents @ flux_rate)


def _branch_law(
    branches: Mapping[str, MagneticBranchDescription],
) -> tuple[Callable | None, Callable | None, np.ndarray | None, Callable | None]:
    values = tuple(branches.values())
    constitutive = tuple(branch.constitutive_data() for branch in values)
    if all(reluctance is not None for reluctance, _flux, _mmf in constitutive):
        return None, None, np.diag([
            reluctance for reluctance, _flux, _mmf in constitutive
        ]), None

    from scipy.interpolate import PchipInterpolator

    laws = []
    slopes = []
    energies = []
    for reluctance, flux, branch_mmf in constitutive:
        if reluctance is not None:
            reluctance = float(reluctance)
            laws.append(lambda value, reluctance=reluctance: reluctance * value)
            slopes.append(lambda _value, reluctance=reluctance: reluctance)
            energies.append(
                lambda value, reluctance=reluctance: 0.5 * reluctance * value * value
            )
        else:
            curve = PchipInterpolator(flux, branch_mmf, extrapolate=True)
            derivative = curve.derivative()
            integral = curve.antiderivative()
            origin = float(integral(0.0))
            laws.append(curve)
            slopes.append(derivative)
            energies.append(
                lambda value, integral=integral, origin=origin: integral(value) - origin
            )

    def mmf(_t: float, flux: np.ndarray, _state: Mapping[str, float]) -> np.ndarray:
        return np.fromiter(
            (law(value) for law, value in zip(laws, flux)),
            dtype=float, count=len(laws),
        )

    def differential(
        _t: float, flux: np.ndarray, _state: Mapping[str, float]
    ) -> np.ndarray:
        return np.fromiter(
            (slope(value) for slope, value in zip(slopes, flux)),
            dtype=float, count=len(slopes),
        )

    def stored_energy(
        _t: float, flux: np.ndarray, _state: Mapping[str, float]
    ) -> float:
        return float(sum(energy(value) for energy, value in zip(energies, flux)))

    return mmf, differential, None, stored_energy


@register_component_type
class MagCircuit(MagPortElement):
    """Magnetic branch network evaluated through a dedicated nonlinear magnetic law."""

    @dataclass(frozen=True, kw_only=True)
    class Params(MagneticParams):
        type: str = "magcircuit"

        def __post_init__(self) -> None:
            super().__post_init__()
            if self.magnetic is None:
                raise ConfigError("magcircuit requires the magnetic domain")
            coupling, _incidence = self.magnetic.matrices(tuple(self.windings))
            if np.linalg.matrix_rank(coupling) != len(self.windings):
                raise ConfigError("magnetic coupling must independently excite every winding")

    type = "magcircuit"

    def __init__(self, name: str, cfg: Any, buses: dict, _params: Any) -> None:
        magnetic = cfg.magnetic
        coupling, incidence = magnetic.matrices(tuple(cfg.windings))
        mmf, differential, reluctance, stored_energy = _branch_law(magnetic.branches)
        law = MagCircuitLaw(
            tuple(cfg.windings),
            coupling,
            incidence=incidence,
            reluctance=reluctance,
            mmf=mmf,
            differential=differential,
            stored_energy=stored_energy,
        )
        super().__init__(name, cfg, buses, law)
