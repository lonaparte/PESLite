"""The solver: the port-Hamiltonian model kernel, the integrators and the run of a simulation.

``model``: subsystems connected into one model, and named states; ``energy``: their energy
declarations and accounting; ``integrators`` and ``multirate``: the solvers; ``splitbound``: the error
of a multirate split; ``simulation``: the run of an assembled system, what it produces, and the
command line.
"""

from .model import (Bag, BlackBoxSubsystem, ConfigError, Empty, GroupPlan, Model, OutputMethod,
                    SolverCheck, SolverRegion, SpectralBound, StateRegistry, Stateful, Subsystem,
                    record_type)
from .differentiation import directional_derivative
from .energy import Energetic, Port, StoragePort
from .integrators import (ADAPTIVE_METHODS, CIRCUIT_MODELS, FIXED_METHODS, RHS, AdaptiveSolver, AutoSolver,
                          DormandPrince45, FixedStepSolver, Solver, SolverStep, make_solver)
from .multirate import HeldStorage, MultirateSolver
from .signals import TableSignal

__all__ = ["Model", "GroupPlan", "Bag", "Empty", "ConfigError", "Subsystem", "BlackBoxSubsystem",
           "SolverCheck", "SolverRegion", "SpectralBound", "record_type",
           "OutputMethod", "directional_derivative", "Energetic",
           "StoragePort", "Port", "Solver", "SolverStep", "Stateful", "StateRegistry", "RHS",
           "FixedStepSolver", "AdaptiveSolver", "AutoSolver", "DormandPrince45", "MultirateSolver", "HeldStorage",
           "make_solver", "FIXED_METHODS", "ADAPTIVE_METHODS", "CIRCUIT_MODELS", "TableSignal"]
