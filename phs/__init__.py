"""Simulation kernel: subsystem protocols, connected models, named states and energy checks.

``Model`` connects subsystems; ``solvers`` provides fixed-step, adaptive and multirate integration.
"""

from . import energy, solvers, states
from .containers import Bag, Empty
from .model import GroupPlan, Model
from .protocols import (
    RHS,
    ConfigError,
    Energetic,
    OutputStage,
    PowerPort,
    Solver,
    SolverStep,
    Stateful,
    StoragePort,
    Subsystem,
)
from .solvers import (
    ADAPTIVE_METHODS,
    FIXED_METHODS,
    AdaptiveSolver,
    DormandPrince45,
    FixedStepSolver,
    HeldStorage,
    MultirateSolver,
    make_solver,
)

__all__ = [
    "Model", "GroupPlan", "Bag", "Empty", "ConfigError",
    "Subsystem", "OutputStage", "Energetic", "StoragePort", "PowerPort", "Solver", "SolverStep", "Stateful", "RHS",
    "FixedStepSolver", "AdaptiveSolver", "DormandPrince45", "MultirateSolver", "HeldStorage",
    "make_solver", "FIXED_METHODS", "ADAPTIVE_METHODS",
    "energy", "solvers", "states",
]
