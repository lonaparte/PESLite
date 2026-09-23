"""Configuration schemas, SI/pu input conversion and file I/O.

Electrical inputs without suffix are SI values; ``_pu`` names are per unit.
Plant quantities are stored in SI, controller quantities in pu.
"""

from ..phs.protocols import ConfigError
from .schema import to_dict
from ..phs.solvers import ADAPTIVE_METHODS, FIXED_METHODS

from .base import BaseValues, DCBase
from .io import dump, from_dict, load, read_initial
from .schema import (
    BusParams,
    BranchParams,
    SourceParams,
    ACFilterParams,
    DCLinkParams,
    DCCapacitorParams,
    DCSourceParams,
    PWMParams,
    PLLParams,
    CurrentLoopParams,
    DCVoltageLoopParams,
    PSCParams,
    DroopParams,
    VSGParams,
    DVOCParams,
    MatchingParams,
    GFMParams,
    ControlParams,
    ReferenceParams,
    DelayParams,
    ProtectionParams,
    StartupParams,
    SourceEventParams,
    SetpointStepParams,
    PLLGainStepParams,
    UnitEventParams,
    UnitParams,

    SolverParams,
    LogParams,
    SimulationParams,
    Params,
)
from .loops import (PLLLoopParams, CCLoopParams, DVCLoopParams, PowerLoopParams,
                    PSCLoopParams, DroopLoopParams, VSGLoopParams, DVOCLoopParams,
                    MatchingLoopParams, ImpedanceLoopParams, AdmittanceLoopParams,
                    DampingLoopParams, DelayLoopParams)

__all__ = [
    "Params", "BaseValues", "DCBase", "load", "dump", "read_initial", "from_dict", "to_dict",
    "ConfigError", "FIXED_METHODS", "ADAPTIVE_METHODS",
    "BusParams",
    "BranchParams",
    "SourceParams",
    "ACFilterParams",
    "DCLinkParams",
    "DCCapacitorParams", "DCSourceParams",
    "PWMParams",
    "PLLParams",
    "CurrentLoopParams",
    "DCVoltageLoopParams",
    "PSCParams",
    "DroopParams",
    "VSGParams",
    "DVOCParams",
    "MatchingParams",
    "GFMParams",
    "ControlParams", "ReferenceParams",
    "DelayParams", "PLLLoopParams", "CCLoopParams", "DVCLoopParams", "PowerLoopParams",
    "PSCLoopParams", "DroopLoopParams", "VSGLoopParams", "DVOCLoopParams", "MatchingLoopParams",
    "ImpedanceLoopParams", "AdmittanceLoopParams", "DampingLoopParams", "DelayLoopParams",
    "ProtectionParams",
    "StartupParams",
    "SourceEventParams",
    "SetpointStepParams",
    "PLLGainStepParams",
    "UnitEventParams",
    "UnitParams",
    "SolverParams",
    "LogParams",
    "SimulationParams",
]
