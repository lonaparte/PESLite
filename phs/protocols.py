"""Runtime-checkable kernel protocols (subsystems, energy declarations, solvers, named state).

Also provides the storage and port declarations and :class:`ConfigError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["ConfigError", "Subsystem", "OutputStage", "StoragePort", "PowerPort", "Energetic", "Solver", "SolverStep", "Stateful", "RHS"]

class ConfigError(ValueError):
    """Invalid model or solver configuration."""


RHS = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]

@runtime_checkable
class Stateful(Protocol):
    """Named state access for initialization and continuation.

    ``get_state()`` returns ``{name: float | complex | bool}`` (``""`` names the part itself);
    ``set_state(values)`` accepts any subset of those names.
    """

    def get_state(self) -> dict[str, Any]: ...

    def set_state(self, values: Mapping[str, Any]) -> None: ...


# -------------------------------------------------------------------------- energy

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


# --------------------------------------------------------------------------- plant

@dataclass(frozen=True)
class OutputStage:
    """One output method of a subsystem and its direct input dependencies.

    ``method``: name of a method taking ``t``; ``inputs``: ``inp`` fields it reads;
    ``outputs``: ``out`` fields it writes. Every output field needs exactly one producer.
    """

    method: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@runtime_checkable
class Subsystem(Protocol):
    """Continuous block with ``state``, ``inp`` and ``out`` records and state derivatives.

    ``state_names``: packing order (a complex state takes two real entries);
    ``outputs_need_inputs``: whether ``set_outputs`` reads ``inp``; optional ``output_stages``
    (tuple of :class:`OutputStage`) replaces ``set_outputs``.
    ``out`` and connected ``inp`` fields are current when ``rhs`` runs.
    """

    state: Any
    inp: Any
    out: Any
    state_names: ClassVar[tuple[str, ...]]
    outputs_need_inputs: ClassVar[bool]

    def set_outputs(self, t: float) -> None:
        """Update ``out`` from ``state`` (and ``inp`` if ``outputs_need_inputs``)."""

    def rhs(self, t: float) -> Sequence[complex | float]:
        """Time derivatives of the states, in ``state_names`` order."""

@dataclass
class SolverStep:
    """Result of integrating one interval; ``y`` is the state at ``t``."""

    t: float
    y: NDArray[np.float64]
    n_rhs: int = 0

@runtime_checkable
class Solver(Protocol):
    """Integrate ``y' = f(t, y)`` from ``t0`` to ``t1`` and return ``SolverStep``."""

    def __call__(self, f: RHS, t0: float, t1: float, y0: NDArray[np.float64]) -> SolverStep: ...
