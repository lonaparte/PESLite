"""Reusable switched DC-converter comparisons against PySpice 1.5 and ngspice.

PySpice is optional because its Python package also needs the system-level ngspice shared
library. Each case is the single source of the parameters passed to both simulators.
"""

from __future__ import annotations

import ctypes.util
import importlib.metadata
import math
import os
from dataclasses import dataclass

import numpy as np
import pytest

import peslite
from peslite.assembly.params import from_dict


pytest.importorskip("PySpice", reason="install the optional dependency with peslite[pyspice]")
if not (os.environ.get("NGSPICE_LIBRARY_PATH") or ctypes.util.find_library("ngspice")):
    pytest.skip("the PySpice comparisons need the ngspice shared library", allow_module_level=True)

from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


PYSPICE_VERSION = "1.5"
MAX_SUPPORTED_NGSPICE_VERSION = 34
pytestmark = [pytest.mark.slow, pytest.mark.pyspice]

try:
    NGSPICE_VERSION = int(NgSpiceShared.new_instance().ngspice_version)
except (OSError, ValueError) as error:
    pytest.skip(f"ngspice is not usable through PySpice: {error}", allow_module_level=True)
if NGSPICE_VERSION > MAX_SUPPORTED_NGSPICE_VERSION:
    pytest.skip(
        f"PySpice {PYSPICE_VERSION} supports ngspice through version "
        f"{MAX_SUPPORTED_NGSPICE_VERSION}, found {NGSPICE_VERSION}",
        allow_module_level=True,
    )


@dataclass(frozen=True)
class DCCase:
    """Equivalent switched DC-converter parameters in SI units."""

    topology: str
    input_voltage: float
    duty: float
    inductance: float
    inductor_resistance: float
    capacitance: float
    load: float
    switching_frequency: float = 20_000.0
    on_resistance: float = 1e-6
    source_resistance: float = 0.1
    input_capacitance: float = 100e-6
    duration: float = 1e-3
    step: float = 0.125e-6

    @property
    def period(self) -> float:
        return 1.0 / self.switching_frequency

    def parameter_map(self) -> dict[str, tuple[str, str, float]]:
        """Return the corresponding public inputs and their shared SI value."""
        load_paths = {
            "buck": ("elements.load.r", "Rload"),
            "boost": ("-units.converter.dclink.source.i", "Iload"),
            "buck_boost": ("elements.load.r", "Rload"),
        }
        capacitor_paths = {
            "buck": ("buses.output.c", "Coutput"),
            "boost": ("units.converter.dclink.capacitor.c", "Coutput"),
            "buck_boost": ("buses.output.c", "Coutput"),
        }
        peslite_load, pyspice_load = load_paths[self.topology]
        peslite_capacitor, pyspice_capacitor = capacitor_paths[self.topology]
        input_path = ("elements.supply.v" if self.topology == "boost"
                      else "units.converter.dclink.vdc_ref")
        mapping = {
            "input_voltage": (input_path, "Vinput", self.input_voltage),
            "duty": ("units.converter.ctrl.references.duty", "Bgate duty", self.duty),
            "inductance": ("units.converter.filter.l_f", "Lfilter", self.inductance),
            "inductor_resistance": (
                "units.converter.filter.r_f", "Rfilter", self.inductor_resistance,
            ),
            "capacitance": (peslite_capacitor, pyspice_capacitor, self.capacitance),
            "load": (peslite_load, pyspice_load, self.load),
            "switching_frequency": (
                "units.converter.pwm.f_sw", "Bgate frequency", self.switching_frequency,
            ),
            "on_resistance": (
                "units.converter.bridge.on_resistance", "SwitchModel.ron", self.on_resistance,
            ),
        }
        if self.topology == "boost":
            mapping.update({
                "source_resistance": (
                    "elements.supply.r", "Rsource", self.source_resistance,
                ),
                "input_capacitance": (
                    "buses.input.c", "Cinput", self.input_capacitance,
                ),
            })
        return mapping

    def peslite_params(self):
        bridge = {
            "model": "switching",
            "upper_controlled": True,
            "lower_controlled": True,
            "on_resistance": self.on_resistance,
        }
        buses = {}
        elements = {}
        dclink = {"vdc_ref": self.input_voltage, "source": {"type": "voltage"}}
        bus = "output"
        output_state = "output.u_C"

        if self.topology == "boost":
            bus = "input"
            bridge["reverse_port"] = True
            buses[bus] = {"phases": 1, "c": self.input_capacitance}
            elements["supply"] = {
                "type": "voltage_source", "bus": bus,
                "v": self.input_voltage, "r": self.source_resistance,
            }
            dclink = {
                "vdc_ref": self.input_voltage / (1.0 - self.duty),
                "capacitor": {"c": self.capacitance},
                "source": {"type": "current", "i": -self.load},
            }
            output_state = "converter.dclink.u_C"
        else:
            buses[bus] = {"phases": 1, "c": self.capacitance}
            elements["load"] = {"type": "resistor", "bus": bus, "r": self.load}
            if self.topology == "buck_boost":
                bridge["external_terminal"] = "lower_rail"

        return from_dict({
            "base": {"s_base": 500.0},
            "buses": buses,
            "elements": elements,
            "units": {
                "converter": {
                    "type": "dc_converter",
                    "bus": bus,
                    "filter": {
                        "l_f": self.inductance,
                        "r_f": self.inductor_resistance,
                    },
                    "dclink": dclink,
                    "bridge": bridge,
                    "pwm": {"f_sw": self.switching_frequency},
                    "ctrl": {
                        "type": "open_loop",
                        "period": self.period,
                        "computation": 0.0,
                        "loops": {"DutyModulation": {}},
                        "references": {"duty": self.duty},
                    },
                },
            },
            "simulation": {
                "t_end": self.duration,
                "initial": {"states": {output_state: 0.0}},
                "energy_check": "off",
                "progress": {"enable": False},
                "solver": {
                    "type": "fixed", "method": "rk4", "dt": self.step,
                    "linearisations": 0,
                },
                "output": {
                    "period": self.step,
                    "states": True,
                    "signals": False,
                    "energy": False,
                },
            },
        })

    def pyspice_circuit(self) -> Circuit:
        circuit = Circuit(f"PESLite {self.topology} reference")
        ground = circuit.gnd
        gate = "gate"
        gate_bar = "gate_bar"
        angular_frequency = 2.0 * math.pi * self.switching_frequency
        threshold = math.cos(math.pi * self.duty)
        circuit.raw_spice += (
            f"Bgate {gate} 0 V=u(cos({angular_frequency:.17g}*time)-{threshold:.17g})\n"
            f"Bgate_bar {gate_bar} 0 V=1-v({gate})\n"
        )
        circuit.model(
            "switch", "SW", ron=self.on_resistance, roff=1e12, vt=0.5, vh=0.0,
        )

        if self.topology == "buck":
            circuit.V("input", "vin", ground, self.input_voltage)
            circuit.VCS("upper", "vin", "switch", gate, ground, model="switch")
            circuit.VCS("lower", "switch", ground, gate_bar, ground, model="switch")
            circuit.L("filter", "switch", "filter_r", self.inductance)
            circuit.R("filter", "filter_r", "output",
                      self.inductor_resistance)
        elif self.topology == "boost":
            circuit.V("input", "supply", ground, self.input_voltage)
            circuit.R("source", "supply", "input", self.source_resistance)
            circuit.C("input", "input", ground, self.input_capacitance)
            circuit.L("filter", "input", "filter_r", self.inductance)
            circuit.R("filter", "filter_r", "switch", self.inductor_resistance)
            circuit.VCS("lower", "switch", ground, gate, ground, model="switch")
            circuit.VCS("upper", "output", "switch", gate_bar, ground, model="switch")
        elif self.topology == "buck_boost":
            circuit.V("input", "vin", ground, self.input_voltage)
            circuit.VCS("upper", "vin", "switch", gate, ground, model="switch")
            circuit.VCS("lower", "switch", "output", gate_bar, ground, model="switch")
            circuit.L("filter", "switch", "filter_r", self.inductance)
            circuit.R("filter", "filter_r", ground, self.inductor_resistance)
        else:
            raise ValueError(f"unknown DC topology {self.topology!r}")

        circuit.C("output", "output", ground, self.capacitance)
        if self.topology == "boost":
            circuit.I("load", "output", ground, self.load)
        else:
            circuit.R("load", "output", ground, self.load)
        return circuit


CASES = (
    DCCase("buck", 48.0, 0.4, 1e-3, 0.05, 470e-6, 6.0),
    DCCase("boost", 24.0, 0.5, 500e-6, 0.05, 470e-6, 1.0),
    DCCase("buck_boost", 24.0, 0.4, 500e-6, 0.05, 470e-6, 12.0),
)


def _interpolate(t, values, at):
    return np.interp(at, t, values)


def compare(case: DCCase, out_dir) -> dict[str, float]:
    """Run both simulators and return scale-normalized maximum state errors."""
    result = peslite.Simulation(case.peslite_params()).run(out_dir=out_dir)

    simulator = case.pyspice_circuit().simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(
        step_time=case.step,
        end_time=case.duration,
        max_time=case.step,
        use_initial_condition=True,
    )
    reference_t = np.asarray(analysis.time, dtype=float)
    reference = {
        "inductor_current": np.asarray(analysis.branches["lfilter"], dtype=float),
        "output_voltage": np.asarray(analysis.nodes["output"], dtype=float),
    }
    actual = {
        "inductor_current": np.asarray(result.states["converter.branch_f.i"]),
        "output_voltage": np.asarray(result.states[
            "converter.dclink.u_C" if case.topology == "boost" else "output.u_C"
        ]),
    }
    errors = {}
    for name in actual:
        expected = _interpolate(reference_t, reference[name], np.asarray(result.t))
        scale = max(1.0, float(np.max(np.abs(expected))))
        errors[name] = float(np.max(np.abs(actual[name] - expected)) / scale)
    return errors


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.topology)
def test_switched_dc_converter_matches_pyspice(case, tmp_path):
    assert importlib.metadata.version("PySpice") == PYSPICE_VERSION
    assert all(peslite_path and pyspice_path and math.isfinite(value)
               for peslite_path, pyspice_path, value in case.parameter_map().values())

    errors = compare(case, tmp_path / case.topology)
    assert errors == pytest.approx({name: 0.0 for name in errors}, abs=1e-3)
