"""Reusable LCL-network comparisons against motulator 0.21.0.

The case object is the single source of every physical and numerical parameter. Its
``parameter_map`` method makes the corresponding PESLite and motulator inputs explicit.
"""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass

import numpy as np
import pytest

import peslite
from peslite.assembly.params import from_dict
from peslite.control import ANGLE, FREQUENCY, V_DQ, Loop, register_loop_type

pytest.importorskip(
    "motulator", reason='install the reference dependencies with peslite[reference]'
)
from motulator.common import control as motulator_control
from motulator.common import utils as motulator_utils
from motulator.common.model import SolverCfg, ZOH
from motulator.grid import model as motulator_model


MOTULATOR_VERSION = "0.21.0"
pytestmark = [pytest.mark.slow, pytest.mark.motulator]


@register_loop_type
class _ReferenceVoltage(Loop):
    """Sampled sinusoidal voltage command used by both reference models."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        amplitude_pu: float
        frequency: float
        phase: float = 0.0
        period: float | None = None
        type: str = "test_motulator_reference_voltage"

    type = "test_motulator_reference_voltage"
    inputs = {}
    outputs = {"u_dq": V_DQ, "theta": ANGLE, "omega": FREQUENCY}

    def initial_outputs(self):
        return {
            "u_dq": complex(self.cfg.amplitude_pu),
            "theta": self.cfg.phase,
            "omega": 2.0 * math.pi * self.cfg.frequency,
        }

    def sample(self, t, inputs):
        del inputs
        return {
            "u_dq": complex(self.cfg.amplitude_pu),
            "theta": 2.0 * math.pi * self.cfg.frequency * t + self.cfg.phase,
            "omega": 2.0 * math.pi * self.cfg.frequency,
        }


class _MotulatorVoltage(motulator_control.ControlSystem):
    """Apply the same sinusoidal voltage through motulator's zero-delay averaged PWM."""

    def __init__(self, case: "LCLCase") -> None:
        super().__init__()
        self.case = case

    def __call__(self, mdl):
        case = self.case
        angle = 2.0 * math.pi * case.converter_frequency * self.t + case.converter_phase
        voltage = case.converter_voltage * np.exp(1j * angle)
        duty = 0.5 + motulator_utils.complex2abc(voltage / case.dc_voltage)
        self.t += case.control_period
        return case.control_period, duty


@dataclass(frozen=True)
class LCLCase:
    """One set of equivalent PESLite and motulator LCL-network parameters."""

    name: str
    converter_voltage: float
    converter_frequency: float
    converter_phase: float
    source_voltage: float
    source_frequency: float
    source_phase: float
    dc_voltage: float = 650.0
    filter_inductance: float = 2.5e-3
    filter_resistance: float = 0.18
    grid_inductance: float = 3.5e-3
    grid_resistance: float = 0.27
    bus_capacitance: float = 40e-6
    control_period: float = 50e-6
    t_end: float = 4e-3

    @property
    def voltage_base(self) -> float:
        return math.sqrt(2.0 / 3.0) * 400.0

    @property
    def initial_bus_voltage(self) -> complex:
        return self.source_voltage * np.exp(1j * self.source_phase)

    def parameter_map(self) -> dict[str, tuple[str, str, float | complex]]:
        """Return the corresponding public inputs and their shared SI value."""
        return {
            "dc_voltage": (
                "units.converter.dclink.vdc_ref",
                "VoltageSourceConverter.u_dc",
                self.dc_voltage,
            ),
            "filter_inductance": (
                "units.converter.ac_filter.l_f",
                "LCLFilter.L_fc",
                self.filter_inductance,
            ),
            "filter_resistance": (
                "units.converter.ac_filter.r_f",
                "LCLFilter.R_fc",
                self.filter_resistance,
            ),
            "grid_inductance": (
                "sources.grid.l",
                "LCLFilter.L_fg",
                self.grid_inductance,
            ),
            "grid_resistance": (
                "sources.grid.r",
                "LCLFilter.R_fg",
                self.grid_resistance,
            ),
            "bus_capacitance": (
                "buses.pcc.c",
                "LCLFilter.C_f",
                self.bus_capacitance,
            ),
            "source_voltage": (
                "sources.grid.v",
                "ThreePhaseSource.e_g",
                self.source_voltage,
            ),
            "source_frequency": (
                "sources.grid.f",
                "ThreePhaseSource.w_g / (2*pi)",
                self.source_frequency,
            ),
            "source_phase": (
                "sources.grid.angle",
                "ThreePhaseSource.phi",
                self.source_phase,
            ),
            "control_period": (
                "units.converter.ctrl.period",
                "ControlSystem.T_s",
                self.control_period,
            ),
            "converter_voltage": (
                "units.converter.ctrl.loops._ReferenceVoltage.amplitude_pu * base.v_phase_peak",
                "_MotulatorVoltage.converter_voltage",
                self.converter_voltage,
            ),
            "converter_frequency": (
                "units.converter.ctrl.loops._ReferenceVoltage.frequency",
                "_MotulatorVoltage.converter_frequency",
                self.converter_frequency,
            ),
            "converter_phase": (
                "units.converter.ctrl.loops._ReferenceVoltage.phase",
                "_MotulatorVoltage.converter_phase",
                self.converter_phase,
            ),
            "initial_bus_voltage": (
                "simulation.initial.states.pcc.u_C",
                "LCLFilter.u_f0_ab",
                self.initial_bus_voltage,
            ),
        }

    def peslite_params(self):
        return from_dict({
            "base": {"s_base": 10e3, "v_ll_rms": 400.0, "f0": 50.0},
            "buses": {
                "pcc": {"c": self.bus_capacitance, "r_d": 0.0, "phases": 3},
            },
            "sources": {
                "grid": {
                    "bus": "pcc",
                    "l": self.grid_inductance,
                    "r": self.grid_resistance,
                    "v": self.source_voltage,
                    "f": self.source_frequency,
                    "angle": self.source_phase,
                },
            },
            "units": {
                "converter": {
                    "type": "vsc",
                    "bus": "pcc",
                    "ac_filter": {
                        "l_f": self.filter_inductance,
                        "r_f": self.filter_resistance,
                    },
                    "dclink": {
                        "vdc_ref": self.dc_voltage,
                        "source": {"type": "voltage", "v": self.dc_voltage},
                    },
                    "bridge": {"model": "pwm_averaging"},
                    "pwm": {
                        "f_sw": 1.0 / self.control_period,
                        "method": "spwm",
                        "update": "single",
                    },
                    "ctrl": {
                        "type": "custom",
                        "period": self.control_period,
                        "computation": 0.0,
                        "loops": {
                            "_ReferenceVoltage": {
                                "amplitude_pu": self.converter_voltage / self.voltage_base,
                                "frequency": self.converter_frequency,
                                "phase": self.converter_phase,
                            },
                        },
                        "outputs": {
                            "u_dq": "_ReferenceVoltage.u_dq",
                            "theta": "_ReferenceVoltage.theta",
                            "omega": "_ReferenceVoltage.omega",
                        },
                    },
                },
            },
            "simulation": {
                "t_end": self.t_end,
                "initial": {
                    "states": {
                        "pcc.u_C": [
                            float(self.initial_bus_voltage.real),
                            float(self.initial_bus_voltage.imag),
                        ],
                    },
                },
                "energy_check": "off",
                "progress": {"enable": False},
                "solver": {
                    "type": "fixed",
                    "method": "rk4",
                    "dt": self.control_period / 20.0,
                    "linearisations": 0,
                },
                "output": {"period": self.control_period},
            },
        })

    def motulator_system(self):
        converter = motulator_model.VoltageSourceConverter(
            u_dc=self.dc_voltage,
            modulator=ZOH(delay=0),
        )
        ac_filter = motulator_model.LCLFilter(
            L_fc=self.filter_inductance,
            R_fc=self.filter_resistance,
            L_fg=self.grid_inductance,
            R_fg=self.grid_resistance,
            C_f=self.bus_capacitance,
            u_f0_ab=self.initial_bus_voltage,
        )
        source = motulator_model.ThreePhaseSource(
            w_g=2.0 * math.pi * self.source_frequency,
            e_g=self.source_voltage,
            phi=self.source_phase,
        )
        return motulator_model.GridConverterSystem(converter, ac_filter, source)


CASES = (
    LCLCase(
        name="stationary_voltage_step",
        converter_voltage=120.0,
        converter_frequency=0.0,
        converter_phase=0.35,
        source_voltage=0.0,
        source_frequency=50.0,
        source_phase=0.0,
    ),
    LCLCase(
        name="sinusoidal_voltage_difference",
        converter_voltage=260.0,
        converter_frequency=50.0,
        converter_phase=0.08,
        source_voltage=230.0,
        source_frequency=50.0,
        source_phase=-0.04,
    ),
)


def _interpolate(t, values, at):
    return np.interp(at, t, np.real(values)) + 1j * np.interp(at, t, np.imag(values))


def _complex_state(result, name):
    return (np.asarray(result.states[name + ".re"])
            + 1j * np.asarray(result.states[name + ".im"]))


def compare(case: LCLCase, out_dir) -> dict[str, float]:
    """Run both simulators and return scale-normalized maximum errors."""
    peslite_result = peslite.Simulation(case.peslite_params()).run(out_dir=out_dir)

    motulator_system = case.motulator_system()
    motulator_simulation = motulator_model.Simulation(
        motulator_system,
        _MotulatorVoltage(case),
        show_progress=False,
        cfg=SolverCfg(
            method="DOP853",
            rtol=1e-11,
            atol=1e-12,
            max_step=case.control_period / 20.0,
        ),
    )
    # motulator stores interval starts and omits the last endpoint; one extra control interval
    # makes the PESLite end time available without changing any compared value.
    motulator_result = motulator_simulation.simulate(
        t_stop=case.t_end + case.control_period, N_eval=1
    )
    assert motulator_result.success

    t = np.asarray(peslite_result.t)
    reference_t = np.asarray(motulator_result.mdl.t)
    pairs = {
        "converter_current": (
            _complex_state(peslite_result, "converter.branch_f.i"),
            np.asarray(motulator_result.mdl.ac_filter.i_c_ab),
        ),
        "bus_voltage": (
            _complex_state(peslite_result, "pcc.u_C"),
            np.asarray(motulator_result.mdl.ac_filter.u_f_ab),
        ),
        "grid_current": (
            -_complex_state(peslite_result, "grid.branch.i"),
            np.asarray(motulator_result.mdl.ac_filter.i_g_ab),
        ),
    }
    errors = {}
    for name, (actual, reference) in pairs.items():
        expected = _interpolate(reference_t, reference, t)
        scale = max(1.0, float(np.max(np.abs(expected))))
        errors[name] = float(np.max(np.abs(actual - expected)) / scale)
    return errors


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_lcl_network_matches_motulator(case, tmp_path):
    assert importlib.metadata.version("motulator") == MOTULATOR_VERSION
    assert all(peslite_path and motulator_path and np.isfinite(value)
               for peslite_path, motulator_path, value in case.parameter_map().values())

    errors = compare(case, tmp_path / case.name)
    assert errors == pytest.approx({name: 0.0 for name in errors}, abs=5e-6)
