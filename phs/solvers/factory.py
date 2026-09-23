"""Build the solver described by the ``simulation.solver`` settings."""

from __future__ import annotations

from typing import Any, Optional

from ..protocols import ConfigError
from .adaptive import AdaptiveSolver, DormandPrince45
from .fixed import FixedStepSolver
from .multirate import MultirateSolver

__all__ = ["make_solver"]


def make_solver(cfg: Any, model: Optional[Any] = None, ratings: Any = None):
    """Build a solver from its type, method, tolerances and step settings.

    ``model`` is required when ``cfg.subsystems`` is set (multirate). ``ratings``: mapping
    storage scale -> (rated effort, rated flow), or a callable ``(subsystem_name, storage) -> (effort, flow)``.
    """
    if cfg.type == "fixed":
        if cfg.subsystems:
            if model is None:
                raise ConfigError("simulation.solver.subsystems needs the model whose subsystems it names")
            try:
                return MultirateSolver(model, cfg.subsystems, cfg.dt, cfg.method, cfg.sweeps, cfg.rtol, cfg.atol,
                                       ratings)
            except (KeyError, ValueError) as exc:
                raise ConfigError(f"simulation.solver.subsystems: {exc.args[0]}") from None
        return FixedStepSolver(cfg.dt, cfg.method)
    if cfg.method == "DP45":
        return DormandPrince45(cfg.rtol, cfg.atol, cfg.max_step)
    return AdaptiveSolver(cfg.method, cfg.rtol, cfg.atol, cfg.max_step, cfg.warm_start)
