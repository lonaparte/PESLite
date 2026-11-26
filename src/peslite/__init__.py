"""Power-electronics converter simulation on a port-Hamiltonian model kernel.

The package has five parts: :mod:`.components` (power circuit and converter hardware),
:mod:`.control` (the converter's controller), :mod:`.assembly` (a system built from a simulation
file), :mod:`.solver` (the model kernel, integrators and run), and :mod:`.addons` (extensions).
Plant quantities are in SI; controllers work in each unit's pu bases.
"""

from . import solver, control, components, assembly
from .solver import (AdaptiveSolver, BlackBoxSubsystem, DormandPrince45, FixedStepSolver, Model,
                     MultirateSolver, make_solver)
from .control import OutputStage, VSCController, register_loop_type
from .components import (ADC, ADCSample, Element, PWM, PWMAveragingModel, ZOH, AveragingModel, SwitchingModel,
                         ThreePhaseBridge, CarrierComparison, FullBridge, HalfBridge,
                         MeasurementPorts, make_bridge,
                         make_modulator, register_component_type)
from .assembly import (DCConverter, Params, Protection, ProtectionSample,
                       SourceScenario, System, Unit, VSC,
                       dump, dumps, load, register_event_type)
from .assembly.control import make_controller
from .solver.simulation import Simulation, SimulationResult, validate_model
from .assembly.exporter import ExportResult, export
from . import addons

__version__ = "0.6.0"


def main(argv=None):
    """Run the command-line interface without loading it during normal imports."""
    from .cli import main as cli_main

    return cli_main(argv)

__all__ = [
    "solver", "control", "components", "assembly", "addons",
    "Params", "load", "dump", "dumps", "System", "Unit", "VSC",
    "DCConverter", "Protection", "ProtectionSample", "MeasurementPorts", "Model", "BlackBoxSubsystem",
    "SourceScenario",
    "Simulation", "SimulationResult", "ExportResult", "export", "VSCController", "main", "make_controller", "make_modulator", "make_solver",
    "register_loop_type", "register_component_type", "register_event_type", "validate_model", "Element",
    "OutputStage", "ADC", "ADCSample", "PWM", "SwitchingModel", "PWMAveragingModel", "AveragingModel",
    "ThreePhaseBridge", "make_bridge",
    "HalfBridge", "FullBridge",
    "CarrierComparison", "ZOH",
    "FixedStepSolver", "AdaptiveSolver", "DormandPrince45", "MultirateSolver", "__version__",
]
