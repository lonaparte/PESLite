"""Power-electronics converter simulation on a port-Hamiltonian model kernel.

The package has five parts: :mod:`.components` (power circuit and converter hardware),
:mod:`.control` (the converter's controller), :mod:`.assembly` (a system built from a simulation
file), :mod:`.solver` (the model kernel, integrators and run), and :mod:`.addons` (custom extension
entry points and optional functions). Plant quantities are in SI; controllers work in each unit's
pu bases.
"""

from . import solver, control, components, assembly
from .solver import (AdaptiveSolver, DormandPrince45, FixedStepSolver, Model, MultirateSolver, make_solver)
from .control import OutputStage, VSCController, make_controller, register_loop_type
from .components import (ADC, ADCSample, PWM, PWMBridge, ZOH, AveragingBridge, SwitchingBridge,
                         ThreePhaseBridge, CarrierComparison, FullBridge, HalfBridge,
                         MeasurementPorts, make_bridge,
                         make_modulator, register_element_type)
from .assembly import (DCConverter, Params, Protection, ProtectionSample,
                       SourceScenario, System, Unit, VSC,
                       dump, dumps, load, register_event_type)
from .solver.simulation import Simulation, SimulationResult, main
from .assembly.exporter import ExportResult, export
from . import addons
from .assembly.params import _register_load_preparer

_register_load_preparer(addons._discover_for)

__version__ = "0.4.0"

__all__ = [
    "solver", "control", "components", "assembly", "addons",
    "Params", "load", "dump", "dumps", "System", "Unit", "VSC",
    "DCConverter", "Protection", "ProtectionSample", "MeasurementPorts", "Model", "SourceScenario",
    "Simulation", "SimulationResult", "ExportResult", "export", "VSCController", "main", "make_controller", "make_modulator", "make_solver",
    "register_loop_type", "register_element_type", "register_event_type",
    "OutputStage", "ADC", "ADCSample", "PWM", "SwitchingBridge", "PWMBridge", "AveragingBridge",
    "ThreePhaseBridge", "make_bridge",
    "HalfBridge", "FullBridge",
    "CarrierComparison", "ZOH",
    "FixedStepSolver", "AdaptiveSolver", "DormandPrince45", "MultirateSolver", "__version__",
]
