"""Assembly of a system from a simulation file.

``params``: the file parameters, including runtime changes; ``validate``: cross-section checks;
``events``: event types and scenarios; ``protection``: the unified protection subsystem;
``unit``: a converter unit; ``system``: the network, registered elements and event operations as
one model.
"""

from .params import (BaseValues, ConfigError, ControlParams, Params, UnitParams,
                     dump, dumps, from_dict, load, register_unit_type,
                     read_initial, to_dict)
from .syntax import (SimulationFileError, parse_simulation_text,
                              read_simulation_file, register_simulation_reader)
from .events import (EVENT_TYPES, Event, Scenario, SourceScenario,
                     register_event_type)
from .protection import Protection, ProtectionSample
from .unit import DCConverter, Unit, VSC
from .system import System
from .params import _DCConverterUnitFields, _VSCUnitFields
from .system import _assemble_dc_converter, _assemble_vsc
from .validate import _dc_converter, _unit
from ..control.controller import GraphController, VSCController
from .control import control_profile
from . import circuit as _circuit
from ..solver.netlist import _install_device_runtime

_install_device_runtime(_circuit.STANDALONE_RUNTIME)

register_unit_type(
    "vsc",
    params=_VSCUnitFields,
    default_bridge="three_phase_bridge",
    default_port="dc",
    ports=("dc", "power"),
    phases=3,
    bridge_types=frozenset(("three_phase_bridge",)),
    measurement_channels=frozenset(("u_g", "i_c", "u_dc")),
    current_channel="i_c",
    protection_criteria=frozenset((
        "overcurrent", "rms_current", "undervoltage", "overvoltage", "frequency", "dc_voltage", "rocof",
    )),
    controller_factory=VSCController,
    validator=_unit,
    assembler=_assemble_vsc,
)
register_unit_type(
    "dc_converter",
    params=_DCConverterUnitFields,
    default_bridge=lambda control: control_profile(control).default_bridge,
    default_port="dc",
    ports=("dc", "power"),
    phases=1,
    bridge_types=frozenset(("half_bridge", "full_bridge")),
    measurement_channels=frozenset(("current", "voltage", "vin", "power")),
    current_channel="current",
    protection_criteria=frozenset((
        "overcurrent", "rms_current", "undervoltage", "overvoltage", "dc_voltage",
    )),
    controller_factory=GraphController,
    validator=_dc_converter,
    assembler=_assemble_dc_converter,
)
__all__ = ["Params", "BaseValues", "ConfigError", "load", "dump", "dumps", "from_dict", "to_dict", "read_initial",
           "Event", "EVENT_TYPES", "register_event_type", "Scenario",
           "SourceScenario", "Protection", "ProtectionSample", "Unit", "VSC", "DCConverter",
           "UnitParams", "ControlParams",
           "SimulationFileError", "parse_simulation_text", "read_simulation_file",
           "register_simulation_reader",
           "System"]
