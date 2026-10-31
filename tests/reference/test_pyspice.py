"""Reusable DC-converter and WPT-control comparisons against PySpice and ngspice.

Each case object is the single source of the physical, modulation and numerical inputs
used by PESLite and by the SPICE circuit.
"""

from __future__ import annotations

import ctypes.util
import importlib.metadata
import math
import os
from dataclasses import dataclass
from time import perf_counter

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
pytestmark = [pytest.mark.full, pytest.mark.pyspice]

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

    name: str
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
        """Return corresponding public PESLite and SPICE inputs."""
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
            "model": "switching", "upper_controlled": True, "lower_controlled": True,
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
                    "type": "dc_converter", "bus": bus,
                    "filter": {
                        "l_f": self.inductance, "r_f": self.inductor_resistance,
                    },
                    "dclink": dclink,
                    "bridge": bridge,
                    "pwm": {"f_sw": self.switching_frequency},
                    "ctrl": {
                        "type": "open_loop", "period": self.period, "computation": 0.0,
                        "loops": {"DutyModulation": {}},
                        "references": {"duty": self.duty},
                    },
                },
            },
            "simulation": {
                "t_end": self.duration,
                "initial": {"states": {output_state: 0.0}},
                "energy_check": "off", "progress": {"enable": False},
                "solver": {
                    "type": "fixed", "method": "rk4", "dt": self.step,
                    "linearisations": 0,
                },
                "output": {
                    "period": self.step, "states": True, "signals": False,
                    "energy": False,
                },
            },
        })

    def pyspice_circuit(self) -> Circuit:
        circuit = Circuit(f"PESLite {self.name} reference")
        ground = circuit.gnd
        gate, gate_bar = "gate", "gate_bar"
        angular_frequency = 2.0 * math.pi * self.switching_frequency
        threshold = math.cos(math.pi * self.duty)
        circuit.raw_spice += (
            f"Bgate {gate} 0 V=u(cos({angular_frequency:.17g}*time)-{threshold:.17g})\n"
            f"Bgate_bar {gate_bar} 0 V=1-v({gate})\n"
        )
        circuit.model("switch", "SW", ron=self.on_resistance, roff=1e12, vt=0.5, vh=0.0)
        if self.topology == "buck":
            circuit.V("input", "vin", ground, self.input_voltage)
            circuit.VCS("upper", "vin", "switch", gate, ground, model="switch")
            circuit.VCS("lower", "switch", ground, gate_bar, ground, model="switch")
            circuit.L("filter", "switch", "filter_r", self.inductance)
            circuit.R("filter", "filter_r", "output", self.inductor_resistance)
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

DC_CASES = (
    DCCase("buck", "buck", 48.0, 0.4, 1e-3, 0.05, 470e-6, 6.0),
    DCCase("boost", "boost", 24.0, 0.5, 500e-6, 0.05, 470e-6, 1.0),
    DCCase("buck_boost_duty_0_40", "buck_boost", 24.0, 0.4,
           500e-6, 0.05, 470e-6, 12.0),
    DCCase("buck_boost_duty_0_65", "buck_boost", 36.0, 0.65,
           750e-6, 0.08, 330e-6, 18.0),
)


SS = {
    "type": "ss",
    "primary_inductance": 100e-6,
    "secondary_inductance": 100e-6,
    "mutual_inductance": 30e-6,
    "primary_capacitance": 1.0 / ((2.0 * math.pi * 85e3) ** 2 * 100e-6),
    "secondary_capacitance": 1.0 / ((2.0 * math.pi * 85e3) ** 2 * 100e-6),
    "primary_resistance": 0.3,
    "secondary_resistance": 0.3,
}

LCC = {
    "type": "lcc",
    "primary_inductance": 200.7e-6,
    "secondary_inductance": 203.5e-6,
    "mutual_inductance": 30e-6,
    "primary_series_capacitance": 26.87e-9,
    "secondary_series_capacitance": 28.98e-9,
    "primary_compensation_inductance": 66.39e-6,
    "secondary_compensation_inductance": 68.97e-6,
    "primary_parallel_capacitance": 54.83e-9,
    "secondary_parallel_capacitance": 51.67e-9,
    "primary_resistance": 0.8,
    "secondary_resistance": 0.8,
    "primary_compensation_resistance": 0.3,
    "secondary_compensation_resistance": 0.3,
    "primary_parallel_resistance": 0.02,
    "secondary_parallel_resistance": 0.05,
}

LCC_S = {
    "type": "lcc_s",
    "primary_inductance": 200.7e-6,
    "secondary_inductance": 203.5e-6,
    "mutual_inductance": 30e-6,
    "primary_series_capacitance": 26.87e-9,
    "primary_compensation_inductance": 66.39e-6,
    "primary_parallel_capacitance": 54.83e-9,
    "secondary_capacitance": 1.0 / ((2.0 * math.pi * 85e3) ** 2 * 203.5e-6),
    "primary_resistance": 0.8,
    "secondary_resistance": 0.8,
    "primary_compensation_resistance": 0.3,
    "primary_parallel_resistance": 0.02,
}

LINKS = {link["type"]: link for link in (SS, LCC, LCC_S)}


@dataclass(frozen=True)
class WPTCase:
    """One phase-shift operating point of a compensated WPT link."""

    name: str
    topology: str
    frequency: float = 85e3
    primary_voltage: float = 400.0
    secondary_voltage: float = 220.0
    primary_width: float = 0.8
    secondary_width: float = 0.8
    primary_phase: float = 0.0
    secondary_phase: float = math.pi / 2.0
    primary_mode: int = 0
    secondary_mode: int = 0
    duration: float = 80e-6
    step: float = 25e-9

    @property
    def link(self) -> dict:
        return LINKS[self.topology]

    def parameter_map(self) -> dict[str, tuple[str, str, float]]:
        """Return corresponding public PESLite and SPICE inputs."""
        link = self.link
        mapping = {
            "primary_voltage": (
                "units.charger.primary.dclink.vdc_ref", "Bprimary amplitude",
                self.primary_voltage,
            ),
            "secondary_voltage": (
                "units.charger.secondary.dclink.vdc_ref", "Bsecondary amplitude",
                self.secondary_voltage,
            ),
            "primary_frequency": (
                "units.charger.primary.ctrl.references.frequency",
                "Bprimary angular frequency",
                self.frequency,
            ),
            "secondary_frequency": (
                "units.charger.secondary.ctrl.references.frequency",
                "Bsecondary angular frequency", self.frequency,
            ),
            "primary_width": (
                "units.charger.primary.ctrl.references.width", "Bprimary pulse width",
                self.primary_width,
            ),
            "secondary_width": (
                "units.charger.secondary.ctrl.references.width", "Bsecondary pulse width",
                self.secondary_width,
            ),
            "primary_phase": (
                "units.charger.primary.ctrl.references.phase", "Bprimary pulse phase",
                self.primary_phase,
            ),
            "secondary_phase": (
                "units.charger.secondary.ctrl.references.phase", "Bsecondary pulse phase",
                self.secondary_phase,
            ),
            "primary_mode": (
                "units.charger.primary.ctrl.references.mode", "Bprimary pulse mode",
                self.primary_mode,
            ),
            "secondary_mode": (
                "units.charger.secondary.ctrl.references.mode", "Bsecondary pulse mode",
                self.secondary_mode,
            ),
        }
        mapping.update({
            name: (f"units.charger.link.{name}", f"SPICE link {name}", value)
            for name, value in link.items() if name != "type"
        })
        return mapping

    @staticmethod
    def _converter(voltage, width, frequency, phase, mode):
        return {
            "type": "dc_converter",
            "dclink": {
                "vdc_ref": voltage,
                "source": {"type": "voltage", "v": voltage},
            },
            "pwm": {"f_sw": frequency, "pattern_sync": "interval"},
            "bridge": {"model": "switching"},
            "ctrl": {
                "type": "phase_shift",
                "computation": 0.0,
                "loops": {"PhaseShiftModulation": {}},
                "references": {
                    "width": width,
                    "frequency": frequency,
                    "phase": phase,
                    "mode": mode,
                },
            },
        }

    def peslite_params(self):
        return from_dict({
            "base": {"s_base": 5000.0, "v_ll_rms": 400.0, "f0": 50.0},
            "units": {
                "charger": {
                    "type": "wpt",
                    "primary": self._converter(
                        self.primary_voltage, self.primary_width, self.frequency,
                        self.primary_phase, self.primary_mode,
                    ),
                    "secondary": self._converter(
                        self.secondary_voltage, self.secondary_width, self.frequency,
                        self.secondary_phase, self.secondary_mode,
                    ),
                    "link": self.link,
                },
            },
            "simulation": {
                "t_end": self.duration,
                "energy_check": "off",
                "progress": {"enable": False},
                "solver": {
                    "type": "fixed", "method": "rk4", "dt": self.step,
                    "linearisations": 0,
                },
                "output": {
                    "period": self.step, "states": False, "signals": True,
                    "energy": False,
                },
            },
        })

    @staticmethod
    def _level_expression(width, phase, mode, frequency):
        half_width = 0.5 * math.pi * width
        if mode == 3 or width == 0.0:
            return "0"
        position = f"cos({2.0 * math.pi * frequency:.17g}*time-({phase:.17g}))"
        edge = math.cos(half_width)
        positive = f"u(({position})-({edge:.17g}))"
        negative = f"u(-({position})-({edge:.17g}))"
        return {0: f"({positive})-({negative})", 1: positive, 4: f"-({negative})"}[mode]

    def _bridge(self, circuit, side, voltage, width, phase, mode):
        level = self._level_expression(width, phase, mode, self.frequency)
        circuit.raw_spice += f"B{side} {side}_source 0 V=({voltage:.17g})*({level})\n"

    @staticmethod
    def _series_side(circuit, side, source, inductance, capacitance, resistance):
        circuit.C(f"{side}_series", source, f"{side}_cap", capacitance)
        circuit.R(f"{side}_coil", f"{side}_cap", f"{side}_ind", resistance)
        circuit.L(side, f"{side}_ind", circuit.gnd, inductance)

    @staticmethod
    def _lcc_side(circuit, side, source, *, inductance, series_capacitance,
                  resistance, compensation_inductance, compensation_resistance,
                  parallel_capacitance, parallel_resistance):
        circuit.R(f"{side}_comp", source, f"{side}_comp_r", compensation_resistance)
        circuit.L(f"{side}_comp", f"{side}_comp_r", f"{side}_node",
                  compensation_inductance)
        circuit.R(f"{side}_parallel", f"{side}_node", f"{side}_parallel_c",
                  parallel_resistance)
        circuit.C(f"{side}_parallel", f"{side}_parallel_c", circuit.gnd,
                  parallel_capacitance)
        WPTCase._series_side(
            circuit, side, f"{side}_node", inductance, series_capacitance, resistance,
        )

    def pyspice_circuit(self) -> Circuit:
        circuit = Circuit(f"PESLite {self.name} WPT reference")
        self._bridge(
            circuit, "primary", self.primary_voltage, self.primary_width,
            self.primary_phase, self.primary_mode,
        )
        self._bridge(
            circuit, "secondary", self.secondary_voltage, self.secondary_width,
            self.secondary_phase, self.secondary_mode,
        )
        link = self.link
        for side in ("primary", "secondary"):
            common = {
                "inductance": link[f"{side}_inductance"],
                "resistance": link[f"{side}_resistance"],
            }
            if self.topology == "lcc" or (self.topology == "lcc_s" and side == "primary"):
                self._lcc_side(
                    circuit, side, f"{side}_source", **common,
                    series_capacitance=link[f"{side}_series_capacitance"],
                    compensation_inductance=link[f"{side}_compensation_inductance"],
                    compensation_resistance=link[f"{side}_compensation_resistance"],
                    parallel_capacitance=link[f"{side}_parallel_capacitance"],
                    parallel_resistance=link[f"{side}_parallel_resistance"],
                )
            else:
                self._series_side(
                    circuit, side, f"{side}_source",
                    capacitance=link[f"{side}_capacitance"], **common,
                )
        coupling = link["mutual_inductance"] / math.sqrt(
            link["primary_inductance"] * link["secondary_inductance"]
        )
        circuit.K("link", "Lprimary", "Lsecondary", coupling)
        return circuit

WPT_CASES = (
    WPTCase("ss_phase_0_35", "ss", secondary_phase=0.35),
    WPTCase("ss_phase_1_57", "ss", secondary_phase=math.pi / 2.0),
    WPTCase("lcc_width_0_60", "lcc", secondary_width=0.6),
    WPTCase("lcc_width_0_90", "lcc", secondary_width=0.9),
    WPTCase("lcc_s_frequency_78k", "lcc_s", frequency=78e3),
    WPTCase("lcc_s_frequency_92k", "lcc_s", frequency=92e3),
)


def _interpolate(t, values, at):
    return np.interp(at, t, values)


def _normalized_errors(actual, expected):
    errors = {}
    for name, values in actual.items():
        scale = max(1.0, float(np.max(np.abs(expected[name]))))
        difference = np.abs(values - expected[name]) / scale
        errors[name] = {
            "mean": float(np.mean(difference)),
            "maximum": float(np.max(difference)),
        }
    return errors


def compare_dc(case: DCCase, out_dir):
    """Measure direct differences between equivalent PESLite and PySpice models."""
    peslite_simulation = peslite.Simulation(case.peslite_params())
    started = perf_counter()
    result = peslite_simulation.run(out_dir=out_dir)
    peslite_seconds = perf_counter() - started
    simulator = case.pyspice_circuit().simulator(temperature=25, nominal_temperature=25)
    started = perf_counter()
    analysis = simulator.transient(
        step_time=case.step,
        end_time=case.duration,
        max_time=case.step,
        use_initial_condition=True,
    )
    pyspice_seconds = perf_counter() - started
    reference_t = np.asarray(analysis.time, dtype=float)
    pyspice = {
        "inductor_current": np.asarray(analysis.branches["lfilter"], dtype=float),
        "output_voltage": np.asarray(analysis.nodes["output"], dtype=float),
    }
    peslite_time = np.asarray(result.t)
    actual = {
        "inductor_current": np.asarray(result.states["converter.branch_f.i"]),
        "output_voltage": np.asarray(result.states[
            "converter.dclink.u_C" if case.topology == "boost" else "output.u_C"
        ]),
    }
    paired = {
        name: _interpolate(reference_t, values, peslite_time)
        for name, values in pyspice.items()
    }
    errors = _normalized_errors(actual, paired)
    return errors, {"peslite": peslite_seconds, "pyspice": pyspice_seconds}


def compare_wpt(case: WPTCase, out_dir):
    """Measure direct differences between equivalent PESLite and PySpice WPT models."""
    peslite_simulation = peslite.Simulation(case.peslite_params())
    started = perf_counter()
    result = peslite_simulation.run(out_dir=out_dir)
    peslite_seconds = perf_counter() - started

    simulator = case.pyspice_circuit().simulator(temperature=25, nominal_temperature=25)
    simulator.options(reltol=1e-8, abstol=1e-10, chgtol=1e-16)
    started = perf_counter()
    analysis = simulator.transient(
        step_time=case.step,
        end_time=case.duration,
        max_time=case.step,
        use_initial_condition=True,
    )
    pyspice_seconds = perf_counter() - started
    reference_t = np.asarray(analysis.time, dtype=float)
    columns = result.columns()
    actual_t = np.asarray(result.t)
    actual = {
        "primary_current": np.asarray(columns["charger.link.i_primary_coil"]),
        "secondary_current": np.asarray(columns["charger.link.i_secondary_coil"]),
    }
    pyspice = {
        "primary_current": np.asarray(analysis.branches["lprimary"], dtype=float),
        "secondary_current": np.asarray(analysis.branches["lsecondary"], dtype=float),
    }
    paired = {
        name: _interpolate(reference_t, values, actual_t)
        for name, values in pyspice.items()
    }
    errors = _normalized_errors(actual, paired)
    return errors, {"peslite": peslite_seconds, "pyspice": pyspice_seconds}


def _check_errors(errors, limit, timings, record_property):
    for signal, values in errors.items():
        record_property(f"{signal}_mean_normalized_difference", values["mean"])
        record_property(f"{signal}_maximum_normalized_difference", values["maximum"])
    mean = float(np.mean([values["mean"] for values in errors.values()]))
    maximum = max(values["maximum"] for values in errors.values())
    record_property("mean_normalized_difference", mean)
    record_property("maximum_normalized_difference", maximum)
    record_property("peslite_simulation_seconds", timings["peslite"])
    record_property("pyspice_simulation_seconds", timings["pyspice"])
    record_property(
        "simulation_time_difference_seconds",
        timings["peslite"] - timings["pyspice"],
    )
    record_property(
        "peslite_to_pyspice_simulation_time_ratio",
        timings["peslite"] / timings["pyspice"],
    )
    assert maximum <= limit, errors


@pytest.mark.parametrize("case", DC_CASES, ids=lambda case: case.name)
def test_switched_dc_converter_matches_pyspice(case, tmp_path, record_property):
    assert importlib.metadata.version("PySpice") == PYSPICE_VERSION
    assert all(peslite_path and pyspice_path and math.isfinite(value)
               for peslite_path, pyspice_path, value in case.parameter_map().values())
    errors, timings = compare_dc(case, tmp_path / case.name)
    _check_errors(errors, 5e-4, timings, record_property)


@pytest.mark.parametrize("case", WPT_CASES, ids=lambda case: case.name)
def test_wpt_control_matches_pyspice(case, tmp_path, record_property):
    assert importlib.metadata.version("PySpice") == PYSPICE_VERSION
    assert all(peslite_path and pyspice_path and math.isfinite(value)
               for peslite_path, pyspice_path, value in case.parameter_map().values())

    errors, timings = compare_wpt(case, tmp_path / case.name)
    _check_errors(errors, 5e-4, timings, record_property)
