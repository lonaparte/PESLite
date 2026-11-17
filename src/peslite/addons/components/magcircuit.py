"""Nonlinear magnetic-circuit laws connected through the magnetic-law state boundary."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .magcomponent import MagneticParams, _MagneticElement, _validate_winding_names
from ...components.network import register_component_type
from ...solver import ConfigError, directional_derivative
from .magport import _attach_magnetic_law

__all__ = ["MagCircuit", "MagCircuitLaw"]

_NEWTON_ITERATIONS = 50
_NEWTON_TOLERANCE = 1.0e-9


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
    """Winding law obtained from magnetic flux continuity and nonlinear branch MMF relations.

    ``mmf`` receives ``(t, branch_flux, state)`` and may implement nonlinear or state-dependent
    material behavior. ``differential`` returns its branch-MMF Jacobian. A linear magnetic circuit
    is reduced to its winding-inductance matrix and assembled through the ordinary netlist model.
    Optional internal state is advanced by
    ``state_derivatives(t, branch_flux, winding_current, state)``. Conservative nonlinear laws may
    provide stored-energy callbacks for exact energy accounting. A material-loss callback receives
    the solved branch flux and its rate, and returns the total magnetic loss in W.
    """

    def __init__(
        self,
        winding_names: Sequence[str],
        coupling: Any,
        *,
        incidence: Any | None = None,
        mmf: Callable[[float, np.ndarray, Mapping[str, float]], Sequence[float]],
        differential: Callable[[float, np.ndarray, Mapping[str, float]], Any],
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
        dissipated_power: Callable[
            [float, np.ndarray, np.ndarray, np.ndarray, Mapping[str, float], float], float
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
        left, singular, right = np.linalg.svd(self._linkage_map, full_matrices=True)
        tolerance = (
            singular.max(initial=0.0)
            * max(self._linkage_map.shape)
            * np.finfo(float).eps
        )
        if len(singular) < windings or singular[-1] <= tolerance:
            raise ConfigError(
                "flux constraints and coupling must independently define every winding linkage"
            )
        # Path fluxes q = linked @ lambda + free @ y satisfy every winding linkage exactly.
        # The remaining equations balance MMF only in the unconstrained flux directions.
        self._linked = (right[:windings].T / singular) @ left.T
        self._free = right[windings:].T
        self._drive_current = self._linked.T
        self._free_reduction = self.flux_basis @ self._free
        self.initial_state = dict(initial_state or {})
        self._state_derivatives = state_derivatives
        if bool(self.initial_state) != (state_derivatives is not None):
            raise ConfigError(
                "magnetic-circuit internal states require state_derivatives and vice versa"
            )
        if not callable(mmf) or not callable(differential):
            raise ConfigError("mmf and differential must be callable")
        if stored_energy is None and stored_energy_rate is not None:
            raise ConfigError("stored_energy_rate requires stored_energy")
        if stored_energy is not None and self.initial_state and stored_energy_rate is None:
            raise ConfigError(
                "state-dependent magnetic energy requires stored_energy_rate"
            )
        if dissipated_power is not None and not callable(dissipated_power):
            raise ConfigError("dissipated_power must be callable")

        self._mmf = mmf
        self._differential = differential
        self._stored_energy = stored_energy
        self._stored_energy_rate = stored_energy_rate
        self._dissipated_power = dissipated_power
        self._free_flux = np.zeros(self._free.shape[1])
        self._last_flux = np.zeros(branches)
        self._last_key: tuple | None = None
        self._last_current = np.zeros(windings)
        self._validate_nonlinear_origin()
        if self._stored_energy is not None:
            self.stored_energy = self._energy_value
            self.stored_energy_rate = self._energy_rate
        if self._dissipated_power is not None:
            self.dissipated_power = self._loss_value

    def _drive(self, t: float, flux: np.ndarray, state: Mapping[str, float]) -> np.ndarray:
        """Return MMF around each independent flux path."""
        return self.flux_basis.T @ self._branch_mmf(t, flux, state)

    def _free_stiffness(
        self, t: float, flux: np.ndarray, state: Mapping[str, float]
    ) -> np.ndarray:
        """Differentiate free-path MMF with respect to free-path flux."""
        return self._free_reduction.T @ self._branch_jacobian(
            t, flux, state
        ) @ self._free_reduction

    def _validate_nonlinear_origin(self) -> None:
        flux = np.zeros(self.coupling.shape[0])
        self._branch_mmf(0.0, flux, self.initial_state)
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
        key = (
            float(t),
            *map(float, flux_linkage),
            *((name, float(state[name])) for name in self.initial_state),
        )
        if key == self._last_key:
            return self._last_current.copy()

        linked = self._linked @ flux_linkage
        free_flux = self._free_flux
        paths = linked + self._free @ free_flux
        flux = self.flux_basis @ paths
        drive = self._drive(t, flux, state)
        residual = self._free.T @ drive
        for _ in range(_NEWTON_ITERATIONS):
            residual_norm = float(np.abs(residual).max(initial=0.0))
            drive_norm = float(np.abs(drive).max(initial=0.0))
            if residual_norm <= _NEWTON_TOLERANCE * drive_norm:
                break
            try:
                step = np.linalg.solve(
                    self._free_stiffness(t, flux, state), residual
                )
            except np.linalg.LinAlgError:
                raise ConfigError(
                    "magnetic-circuit differential is singular on its free flux paths"
                ) from None
            if not np.isfinite(step).all():
                raise ConfigError(
                    "magnetic-circuit differential produced a non-finite Newton step"
                )
            for _ in range(_NEWTON_ITERATIONS):
                trial_free_flux = free_flux - step
                trial_paths = linked + self._free @ trial_free_flux
                trial_flux = self.flux_basis @ trial_paths
                trial_drive = self._drive(t, trial_flux, state)
                trial_residual = self._free.T @ trial_drive
                if np.abs(trial_residual).max(initial=0.0) < residual_norm:
                    break
                step *= 0.5
            else:
                raise ConfigError(
                    "magnetic-circuit solve cannot reduce its path MMF residual"
                )
            free_flux, paths, flux = trial_free_flux, trial_paths, trial_flux
            drive, residual = trial_drive, trial_residual
        else:
            raise ConfigError("magnetic-circuit solve did not converge")
        self._free_flux = free_flux
        self._last_flux = flux
        self._last_current = self._drive_current @ drive
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

    def _branch_flux_rate(
        self,
        t: float,
        state: Mapping[str, float],
        flux_linkage_rate: np.ndarray,
        state_rate: Mapping[str, float],
    ) -> np.ndarray:
        """Map the winding and material-state rates to the solved branch flux rates."""
        flux = self._last_flux
        base = self.flux_basis @ self._linked @ flux_linkage_rate
        if not self._free.shape[1]:
            return base

        names = tuple(self.initial_state)
        point = np.asarray([t, *(state[name] for name in names)], dtype=float)
        direction = np.asarray([1.0, *(state_rate[name] for name in names)], dtype=float)

        def source(candidate: np.ndarray) -> np.ndarray:
            values = dict(zip(names, candidate[1:]))
            return self._branch_mmf(float(candidate[0]), flux, values)

        imposed = directional_derivative(source, point, direction)
        jacobian = self._branch_jacobian(t, flux, state)
        rhs = -self._free_reduction.T @ (jacobian @ base + imposed)
        try:
            free_rate = np.linalg.solve(self._free_stiffness(t, flux, state), rhs)
        except np.linalg.LinAlgError:
            raise ConfigError(
                "magnetic-circuit differential is singular on its free flux paths"
            ) from None
        return base + self._free_reduction @ free_rate

    def _loss_value(
        self,
        t: float,
        flux_linkage: np.ndarray,
        currents: np.ndarray,
        state: Mapping[str, float],
        flux_linkage_rate: np.ndarray,
        state_rate: Mapping[str, float],
        temperature_K: float,
    ) -> float:
        self.currents(t, flux_linkage, state)
        branch_rate = self._branch_flux_rate(t, state, flux_linkage_rate, state_rate)
        return self._dissipated_power(  # type: ignore[misc]
            t,
            self._last_flux.copy(),
            branch_rate,
            currents.copy(),
            state,
            temperature_K,
        )


def _branch_law(
    constitutive: Sequence[tuple[float | None, np.ndarray | None, np.ndarray | None]],
) -> tuple[Callable, Callable, Callable]:
    """Return branch MMF, its Jacobian and stored energy."""
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

    return mmf, differential, stored_energy


@register_component_type
class MagCircuit(_MagneticElement):
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
        names = tuple(cfg.windings)
        coupling, incidence = magnetic.matrices(names)
        unused = sorted(
            f"elements.{name}.magnetic.branches.{branch_name}.{field_name}"
            for branch_name, branch in magnetic.branches.items()
            for field_name in branch.provider_data()
        )
        if unused:
            warnings.warn(
                "magcircuit retains but does not evaluate " + ", ".join(unused)
                + "; these fields do not affect the simulation",
                stacklevel=3,
            )
        constitutive = tuple(
            branch.constitutive_data() for branch in magnetic.branches.values()
        )
        if all(reluctance is not None for reluctance, _flux, _mmf in constitutive):
            inductance, _response, _matrix = _linear_circuit(
                [reluctance for reluctance, _flux, _mmf in constitutive],
                coupling,
                incidence,
            )
            self._assemble(name, cfg, names, inductance, buses, magnetic)
            return
        mmf, differential, stored_energy = _branch_law(constitutive)
        law = MagCircuitLaw(
            names,
            coupling,
            incidence=incidence,
            mmf=mmf,
            differential=differential,
            stored_energy=stored_energy,
        )
        _attach_magnetic_law(self, name, cfg, buses, law)
