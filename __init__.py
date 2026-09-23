"""Power-electronics converter simulation built on the :mod:`peslite.phs` model and solver kernel.

Plant quantities are in SI; controllers work in each unit's pu bases.
"""

from .phs.model import Model
from .phs.solvers import AdaptiveSolver, DormandPrince45, FixedStepSolver, MultirateSolver, make_solver

from . import (phs, assembly, control, firmware, modulation, params, power, protection, results,
               sensing)
from .assembly import (UniteType, ConverterFirmware, System, Unit,
                       SourceScenario, UnitScenario, make_controller)
from .firmware import ComputationDelay
from .modulation import ZOH, CarrierComparison, SampledCarrier, StepAveragedCarrier, make_modulator
from .params import Params, load
from .simulation import Simulation, main
from .results import SimulationResult
from .sensing import MeasurementPorts

__version__ = "0.1.0"

__all__ = [
    "assembly", "control", "firmware", "modulation", "params", "power", "protection",
    "results", "sensing", "simulation", "phs",
    "Params", "load", "System", "Unit", "MeasurementPorts", "Model", "SourceScenario", "UnitScenario",
    "Simulation", "SimulationResult", "UniteType", "main",
    "make_controller", "make_modulator", "make_solver", "ConverterFirmware",
    "ComputationDelay", "CarrierComparison", "ZOH", "StepAveragedCarrier", "SampledCarrier",
    "FixedStepSolver", "AdaptiveSolver", "DormandPrince45", "MultirateSolver",
    "__version__",
]
