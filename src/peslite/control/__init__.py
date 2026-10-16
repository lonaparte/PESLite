"""The controller of a converter unit.

``loops``: what each loop type computes; ``controller``: the controller as one block (its
interface, the loop network that connects and runs the loops, the gfl/gfm wiring,
:class:`VSCController`); ``modulation``: the output stage,
from the voltage command to duty ratios; ``blocks``: transforms, filters and timers they are built of.
Grid-control signals use pu and scalar converter signals use SI; shared scalar loops bind to the
coordinates of their containing graph. Time is in s, angles in rad and frequencies in rad/s. The controller depends on nothing but the
solver's kernel: the hardware adapts to its interface (:class:`Measurement`, :class:`ControlOutput`).
"""

from ..solver import ConfigError
from .blocks import Filter, Integrator, OutputLimit, PI, complex_power, unit_delay
from .loops import (ANGLE, FREQUENCY, I, I_AB, I_DQ, LOOP_TYPES, POWER, V, V_AB, V_DQ,
                    DUTY_RATIO, HERTZ, MODULATION, SI_CURRENT, SI_VOLTAGE, SI_POWER, DCVoltageControl,
                    InductorCurrentControl, PowerControl, PhasePowerControl, ResonantModulation,
                    Loop, SignalType, SyncLaw, register_loop_type)
from .modulation import ModulationLimiter, OutputStage
from .controller import (CONTROL_INTERFACE, ControlGraph, ControlInterface, ControlMeasurement,
                         BridgeControlOutput, ControlOutput, Controller, ConverterMeasurement,
                         GraphController, Measurement,
                         Startup, VSCController, default_wiring,
                         make_controller)

__all__ = ["Loop", "SyncLaw", "Integrator", "PI", "OutputLimit", "Filter",
           "complex_power", "unit_delay", "SignalType", "ConfigError",
           "SI_CURRENT", "SI_VOLTAGE", "SI_POWER", "DUTY_RATIO", "MODULATION", "HERTZ",
           "DCVoltageControl", "InductorCurrentControl", "PowerControl", "PhasePowerControl", "ResonantModulation",
           "I_AB", "V_AB", "I_DQ", "V_DQ", "I", "V",
           "ANGLE", "FREQUENCY", "POWER",
           "LOOP_TYPES", "register_loop_type", "ControlInterface", "CONTROL_INTERFACE",
           "ControlGraph", "default_wiring",
           "ModulationLimiter", "OutputStage", "Measurement", "ControlMeasurement", "ControlOutput",
           "BridgeControlOutput", "ConverterMeasurement", "Controller", "Startup", "VSCController", "GraphController",
           "make_controller"]
