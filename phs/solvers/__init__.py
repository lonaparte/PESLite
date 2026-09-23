"""Integrators with the signature ``solver(f, t0, t1, y0) -> SolverStep``.

Fixed-step (Euler, Heun, RK4), adaptive (SciPy or built-in DP45) and multirate solvers,
and :func:`make_solver` to build one from solver settings.
"""

FIXED_METHODS = ("euler", "heun", "rk4")
ADAPTIVE_METHODS = ("RK45", "DOP853", "Radau", "BDF", "LSODA", "RK23", "DP45")

from .adaptive import AdaptiveSolver, DormandPrince45
from .factory import make_solver
from .fixed import FixedStepSolver
from .multirate import HeldStorage, MultirateSolver

__all__ = ["FIXED_METHODS", "ADAPTIVE_METHODS",
           "FixedStepSolver", "AdaptiveSolver", "DormandPrince45", "MultirateSolver", "HeldStorage",
           "make_solver"]
