"""Assembly of a system from a simulation file.

``params``: the file parameters, including runtime changes; ``validate``: cross-section checks;
``events``: event types and scenarios; ``protection``: the unified protection subsystem;
``unit``: a converter unit; ``system``: the network, registered elements and event operations as
one model.
"""

from .params import (BaseValues, ConfigError, ControlParams, LCCLinkParams, Params,
                     SSLinkParams, UnitParams,
                     dump, dumps, from_dict, load,
                     read_initial, to_dict)
from .events import (EVENT_TYPES, Event, Scenario, SourceScenario,
                     register_event_type)
from .protection import Protection
from .unit import DCConverter, Unit, VSC
from .system import System
from .params import _DCConverterUnitFields, _VSCUnitFields, _WPTUnitFields
from .system import _assemble_dc_converter, _assemble_vsc, _assemble_wpt
from .types import register_unit_type
from .validate import _dc_converter, _unit, _wpt
from ..control.profiles import control_type

register_unit_type(
    "vsc",
    params=_VSCUnitFields,
    default_bridge="three_phase_bridge",
    phases=3,
    bridge_types=frozenset(("three_phase_bridge",)),
    defaults_grid_frequency=True,
    validator=_unit,
    assembler=_assemble_vsc,
)
register_unit_type(
    "dc_converter",
    params=_DCConverterUnitFields,
    default_bridge=lambda control: control_type(control.type).default_bridge,
    phases=1,
    bridge_types=frozenset(("half_bridge", "full_bridge")),
    validator=_dc_converter,
    assembler=_assemble_dc_converter,
    direct_power_port=True,
)
register_unit_type(
    "wpt",
    params=_WPTUnitFields,
    children=("primary", "secondary"),
    validator=_wpt,
    assembler=_assemble_wpt,
)

__all__ = ["Params", "BaseValues", "ConfigError", "load", "dump", "dumps", "from_dict", "to_dict", "read_initial",
           "Event", "EVENT_TYPES", "register_event_type", "Scenario",
           "SourceScenario", "Protection", "Unit", "VSC", "DCConverter",
           "UnitParams", "ControlParams",
           "SSLinkParams", "LCCLinkParams", "System"]
