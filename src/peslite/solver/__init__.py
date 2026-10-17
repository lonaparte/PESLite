"""The solver: the port-Hamiltonian model kernel, the integrators and the run of a simulation.

``model``: subsystems connected into one model, and named states; ``energy``: their energy
declarations and accounting; ``integrators`` and ``multirate``: the solvers; ``splitbound``: the error
of a multirate split; ``simulation``: the run of an assembled system, what it produces, and the
command line.
"""

from .model import Bag, ConfigError, Empty, GroupPlan, Model, OutputMethod, StateRegistry, Stateful, Subsystem
from .energy import Energetic, PowerPort, StoragePort
from .integrators import (ADAPTIVE_METHODS, FIXED_METHODS, RHS, AdaptiveSolver, DormandPrince45, FixedStepSolver,
                          Solver, SolverStep)
from .multirate import HeldStorage, MultirateSolver, make_solver

__all__ = ["Model", "GroupPlan", "Bag", "Empty", "ConfigError", "Subsystem", "OutputMethod", "Energetic",
           "StoragePort", "PowerPort", "Solver", "SolverStep", "Stateful", "StateRegistry", "RHS",
           "FixedStepSolver", "AdaptiveSolver", "DormandPrince45", "MultirateSolver", "HeldStorage",
           "make_solver", "FIXED_METHODS", "ADAPTIVE_METHODS"]
