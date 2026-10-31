"""Reusable converter-control comparisons against motulator 0.21.0.

Motulator supplies an independent converter, network, integration and control-event
runtime. Small reference controllers implement the same published equations configured
in PESLite, so unlike similarly named library controllers the compared laws are exact.
The cases cover three GFM laws, GFL current control and a parallel GFM/GFL system, with
two parameter sets for each family.
"""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import InitVar, dataclass, field
from time import perf_counter

import numpy as np
import pytest

import peslite
from peslite.assembly.params import from_dict

pytest.importorskip(
    "motulator", reason="install the reference dependencies with peslite[reference]"
)
from motulator.common import control as motulator_control
from motulator.common import model as common_model
from motulator.common import utils as motulator_utils
from motulator.common.model import SolverCfg, ZOH
from motulator.grid import model as motulator_model


MOTULATOR_VERSION = "0.21.0"
pytestmark = [pytest.mark.full, pytest.mark.motulator]

S_BASE = 10e3
V_LL_RMS = 400.0
V_BASE = math.sqrt(2.0 / 3.0) * V_LL_RMS
I_BASE = S_BASE / (1.5 * V_BASE)
W0 = 2.0 * math.pi * 50.0
V_DC = 650.0
T_CONTROL = 50e-6
L_FILTER = 2.5e-3
R_FILTER = 0.18
L_GRID = 3.5e-3
R_GRID = 0.27
C_BUS = 40e-6
V_GRID = 230.0


@dataclass(frozen=True)
class ControlCase:
    """One single-converter control law and parameter set."""

    name: str
    family: str
    parameters: dict[str, float]
    p_ref_pu: float = 0.12
    q_ref_pu: float = 0.0
    id_ref_pu: float = 0.12
    iq_ref_pu: float = 0.0
    source_voltage: float = V_GRID
    source_phase: float = 0.0
    t_end: float = 4e-3

    @property
    def v_ref_pu(self) -> float:
        return self.source_voltage / V_BASE

    def parameter_map(self) -> dict[str, tuple[str, str, float]]:
        """Return corresponding public PESLite and reference-controller inputs."""
        mapping = {
            "source_voltage": ("sources.grid.v", "ThreePhaseSource.e_g", self.source_voltage),
            "source_phase": ("sources.grid.angle", "ThreePhaseSource.phi", self.source_phase),
            "filter_inductance": ("units.converter.ac_filter.l_f", "LCLFilter.L_fc", L_FILTER),
            "filter_resistance": ("units.converter.ac_filter.r_f", "LCLFilter.R_fc", R_FILTER),
            "grid_inductance": ("sources.grid.l", "LCLFilter.L_fg", L_GRID),
            "grid_resistance": ("sources.grid.r", "LCLFilter.R_fg", R_GRID),
            "bus_capacitance": ("buses.pcc.c", "LCLFilter.C_f", C_BUS),
            "control_period": ("units.converter.ctrl.period", "ControlSystem.T_s", T_CONTROL),
        }
        if self.family == "gfl":
            paths = {
                "pll_kp": "SRFPLL.kp_pu",
                "pll_ki": "SRFPLL.ki_pu",
                "current_kp": "CurrentLoop.kp_pu",
                "current_ki": "CurrentLoop.ki_pu",
            }
        else:
            loop = {"psc": "PSC", "vsg": "VSG", "dvoc": "DVOC"}[self.family]
            paths = {name: f"{loop}.{name}" for name in self.parameters}
        for name, value in self.parameters.items():
            mapping[name] = (
                f"units.converter.ctrl.loops.{paths[name]}",
                f"ReferenceControl.{self.family}.{name}", value,
            )
        return mapping

    def control(self) -> dict:
        references = {
            "p_ref_pu": self.p_ref_pu,
            "q_ref_pu": self.q_ref_pu,
            "v_ref_pu": self.v_ref_pu,
            "id_ref_pu": self.id_ref_pu,
            "iq_ref_pu": self.iq_ref_pu,
        }
        if self.family == "gfl":
            loops = {
                "SRFPLL": {
                    "kp_pu": self.parameters["pll_kp"],
                    "ki_pu": self.parameters["pll_ki"],
                    "normalisation": "amplitude",
                },
                "CurrentLoop": {
                    "bandwidth": 200.0,
                    "kp_pu": self.parameters["current_kp"],
                    "ki_pu": self.parameters["current_ki"],
                    "antiwindup": False,
                },
            }
            control_type = "gfl"
        else:
            loop_name = {"psc": "PSC", "vsg": "VSG", "dvoc": "DVOC"}[self.family]
            loop = dict(self.parameters)
            loops = {loop_name: loop, "VirtualImpedance": {}}
            control_type = "gfm"
        return {
            "type": control_type,
            "period": T_CONTROL,
            "computation": 0.0,
            "loops": loops,
            "references": references,
        }

    def peslite_params(self):
        return _peslite_params({"converter": self.control()}, self.source_voltage,
                               self.source_phase, self.t_end)

    def motulator_system(self):
        converter = motulator_model.VoltageSourceConverter(u_dc=V_DC, modulator=ZOH(delay=0))
        ac_filter = motulator_model.LCLFilter(
            L_fc=L_FILTER, R_fc=R_FILTER, L_fg=L_GRID, R_fg=R_GRID, C_f=C_BUS,
            u_f0_ab=self.source_voltage * np.exp(1j * self.source_phase),
        )
        source = motulator_model.ThreePhaseSource(
            w_g=W0, e_g=self.source_voltage, phi=self.source_phase,
        )
        return motulator_model.GridConverterSystem(converter, ac_filter, source)


def _unit(control):
    return {
        "type": "vsc",
        "bus": "pcc",
        "ac_filter": {"l_f": L_FILTER, "r_f": R_FILTER},
        "dclink": {"vdc_ref": V_DC, "source": {"type": "voltage", "v": V_DC}},
        "bridge": {"model": "pwm_averaging"},
        "pwm": {"f_sw": 1.0 / T_CONTROL, "method": "spwm", "update": "single"},
        "ctrl": control,
    }


def _peslite_params(controls, source_voltage, source_phase, t_end):
    return from_dict({
        "base": {"s_base": S_BASE, "v_ll_rms": V_LL_RMS, "f0": 50.0},
        "buses": {"pcc": {"c": C_BUS, "r_d": 0.0, "phases": 3}},
        "sources": {
            "grid": {
                "bus": "pcc", "l": L_GRID, "r": R_GRID,
                "v": source_voltage, "f": 50.0, "angle": source_phase,
            },
        },
        "units": {name: _unit(control) for name, control in controls.items()},
        "simulation": {
            "t_end": t_end,
            "initial": {
                "states": {
                    "pcc.u_C": [
                        source_voltage * math.cos(source_phase),
                        source_voltage * math.sin(source_phase),
                    ],
                },
            },
            "energy_check": "off",
            "progress": {"enable": False},
            "solver": {
                "type": "fixed", "method": "rk4", "dt": T_CONTROL / 20.0,
                "linearisations": 0,
            },
            "output": {"period": T_CONTROL},
        },
    })


class _ReferenceControl(motulator_control.ControlSystem):
    """Control laws equivalent to the configured PESLite sampled loops."""

    def __init__(self, case: ControlCase, current_name: str | None = None) -> None:
        super().__init__()
        self.case = case
        self.current_name = current_name
        self.theta = 0.0
        self.omega = W0
        self.v_mag = case.v_ref_pu
        self.dw_pu = 0.0
        self.pll_integral = 0.0
        self.pll_voltage = 1.0
        self.current_integral = 0j

    @staticmethod
    def _tustin(integral, error):
        middle = integral + 0.5 * T_CONTROL * error
        return middle, middle + 0.5 * T_CONTROL * error

    def _measure(self, mdl):
        if hasattr(mdl, "network"):
            state = mdl.network.state
            current = getattr(state, self.current_name)
            return state.u_bus, current
        return mdl.ac_filter.state.u_f_ab, mdl.ac_filter.state.i_c_ab

    def _gfm(self, voltage, current):
        case = self.case
        p = voltage * current.conjugate() / S_BASE * 1.5
        p_pu, q_pu = p.real, p.imag
        i_dq = current / I_BASE * np.exp(-1j * self.theta)
        p_ref, q_ref = case.p_ref_pu, case.q_ref_pu
        v_ref = case.v_ref_pu
        cfg = case.parameters
        if case.family == "psc":
            self.omega = W0 + cfg["k_p_pu"] * (p_ref - p_pu)
            self.theta += T_CONTROL * self.omega
            error = v_ref - abs(voltage) / V_BASE
            middle, new_integral = self._tustin(getattr(self, "voltage_integral", 0.0), error)
            self.voltage_integral = new_integral
            self.v_mag = v_ref + cfg.get("k_v", 0.0) * error + cfg.get("k_vi", 0.0) * middle
        elif case.family == "vsg":
            derivative = (p_ref - p_pu - cfg["d_p_pu"] * self.dw_pu) / (2.0 * cfg["h"])
            self.dw_pu += T_CONTROL * derivative
            self.omega = W0 * (1.0 + self.dw_pu)
            self.theta += T_CONTROL * self.omega
            command = v_ref + cfg["k_q_pu"] * (q_ref - q_pu)
            if cfg.get("t_q", 0.0) > 0.0:
                self.v_mag += T_CONTROL / cfg["t_q"] * (command - self.v_mag)
            else:
                self.v_mag = command
        elif case.family == "dvoc":
            magnitude = max(self.v_mag, 1e-12)
            current_ref = (p_ref - 1j * q_ref) * magnitude / (v_ref * v_ref)
            rotation = np.exp(1j * cfg["kappa"])
            derivative = (
                cfg["eta_pu"] * rotation * (current_ref - i_dq)
                + cfg["eta_pu"] * cfg["alpha_pu"]
                * (1.0 - magnitude * magnitude / (v_ref * v_ref)) * magnitude
            )
            self.omega = W0 + derivative.imag / magnitude
            self.theta += T_CONTROL * self.omega
            self.v_mag += T_CONTROL * derivative.real
        else:
            raise ValueError(f"unknown GFM family {case.family!r}")
        return self.v_mag * V_BASE * np.exp(1j * self.theta)

    def _gfl(self, voltage, current):
        cfg = self.case.parameters
        frame = self.theta
        rotated_voltage = voltage / V_BASE * np.exp(-1j * frame)
        error = (rotated_voltage.imag / self.pll_voltage
                 if self.pll_voltage > 0.0 else 0.0)
        middle, self.pll_integral = self._tustin(self.pll_integral, error)
        self.omega = W0 + cfg["pll_kp"] * error + cfg["pll_ki"] * middle
        self.theta += T_CONTROL * self.omega
        self.pll_voltage += T_CONTROL * cfg["pll_kp"] * (
            rotated_voltage.real - self.pll_voltage
        )

        i_dq = current / I_BASE * np.exp(-1j * frame)
        reference = self.case.id_ref_pu + 1j * self.case.iq_ref_pu
        current_error = reference - i_dq
        middle, self.current_integral = self._tustin(self.current_integral, current_error)
        x_pu = L_FILTER * W0 / (V_BASE / I_BASE)
        r_pu = R_FILTER / (V_BASE / I_BASE)
        feedforward = rotated_voltage + (r_pu + 1j * self.omega / W0 * x_pu) * i_dq
        command = (
            feedforward + cfg["current_kp"] * current_error
            + cfg["current_ki"] * middle
        )
        return command * V_BASE * np.exp(1j * self.theta)

    def __call__(self, mdl):
        voltage, current = self._measure(mdl)
        command = self._gfl(voltage, current) if self.case.family == "gfl" else self._gfm(
            voltage, current
        )
        duty = 0.5 + motulator_utils.complex2abc(command / V_DC)
        self.t += T_CONTROL
        return T_CONTROL, duty


SINGLE_CASES = (
    ControlCase("psc_gain_4", "psc", {"k_p_pu": 4.0, "k_v": 0.0, "k_vi": 0.0}),
    ControlCase("psc_gain_8", "psc", {"k_p_pu": 8.0, "k_v": 0.0, "k_vi": 0.0}),
    ControlCase("vsg_inertia_0_20", "vsg", {
        "h": 0.20, "d_p_pu": 12.0, "k_q_pu": 0.04, "t_q": 0.0,
    }),
    ControlCase("vsg_inertia_0_45", "vsg", {
        "h": 0.45, "d_p_pu": 12.0, "k_q_pu": 0.04, "t_q": 0.0,
    }),
    ControlCase("dvoc_eta_5", "dvoc", {
        "eta_pu": 5.0, "alpha_pu": 20.0, "kappa": math.pi / 2.0,
    }),
    ControlCase("dvoc_eta_9", "dvoc", {
        "eta_pu": 9.0, "alpha_pu": 20.0, "kappa": math.pi / 2.0,
    }),
    ControlCase("gfl_current_0_08", "gfl", {
        "pll_kp": 30.0, "pll_ki": 600.0, "current_kp": 0.12, "current_ki": 3.0,
    }, id_ref_pu=0.08),
    ControlCase("gfl_current_0_16", "gfl", {
        "pll_kp": 30.0, "pll_ki": 600.0, "current_kp": 0.12, "current_ki": 3.0,
    }, id_ref_pu=0.16),
)


@dataclass
class _NetworkInputs:
    u_1: complex = 0j
    u_2: complex = 0j
    e_g: complex = 0j


@dataclass
class _NetworkStates:
    i_1: complex = 0j
    i_2: complex = 0j
    u_bus: complex = V_GRID + 0j
    i_grid: complex = 0j


@dataclass
class _NetworkOutputs:
    i_1: complex = 0j
    i_2: complex = 0j
    u_bus: complex = V_GRID + 0j
    i_grid: complex = 0j


@dataclass
class _NetworkTimeSeries(common_model.SubsystemTimeSeries):
    t: InitVar[np.ndarray]
    subsystem: InitVar["_ParallelNetwork"]
    i_1: np.ndarray = field(default_factory=lambda: np.array([]))
    i_2: np.ndarray = field(default_factory=lambda: np.array([]))
    u_bus: np.ndarray = field(default_factory=lambda: np.array([]))
    i_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    u_1: np.ndarray = field(default_factory=lambda: np.array([]))
    u_2: np.ndarray = field(default_factory=lambda: np.array([]))
    e_g: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self, t, subsystem):
        del t
        for name in ("i_1", "i_2", "u_bus", "i_grid"):
            setattr(self, name, np.asarray(subsystem._history[name]))

    def set_outputs(self, t, subsystem):
        del t, subsystem

    def compute_derived_signals(self, t, subsystem):
        del t, subsystem


class _ParallelNetwork(common_model.Subsystem):
    """Two converter filters feeding one capacitive PCC and one grid branch."""

    def __init__(self, grid_scale: float):
        self.L_grid = grid_scale * L_GRID
        self.R_grid = grid_scale * R_GRID
        self.inp = _NetworkInputs()
        self.state = _NetworkStates()
        self.out = _NetworkOutputs()

    def set_outputs(self, t):
        del t
        for name in ("i_1", "i_2", "u_bus", "i_grid"):
            setattr(self.out, name, getattr(self.state, name))

    def rhs(self, t):
        del t
        state, inp = self.state, self.inp
        return [
            (inp.u_1 - state.u_bus - R_FILTER * state.i_1) / L_FILTER,
            (inp.u_2 - state.u_bus - R_FILTER * state.i_2) / L_FILTER,
            (state.i_1 + state.i_2 - state.i_grid) / C_BUS,
            (state.u_bus - inp.e_g - self.R_grid * state.i_grid) / self.L_grid,
        ]

    def create_time_series(self, t):
        return _NetworkTimeSeries(t, self)


class _ParallelConverterModel(common_model.Model):
    def __init__(self, converter, network, source, port):
        current = f"i_{port}"
        converter.ac_load = network
        super().__init__(
            converter,
            {"converter": converter, "network": network, "source": source},
            {
                ("converter", "i_c_ab"): ("network", current),
                ("network", f"u_{port}"): ("converter", "u_c_ab"),
                ("network", "e_g"): ("source", "e_g_ab"),
            },
        )
        self.network = network


@dataclass(frozen=True)
class ParallelCase:
    name: str
    grid_scale: float
    p_ref_pu: float
    id_ref_pu: float
    t_end: float = 4e-3

    def parameter_map(self) -> dict[str, tuple[str, str, float]]:
        return {
            "grid_inductance": (
                "sources.grid.l", "ParallelNetwork.L_grid", self.grid_scale * L_GRID,
            ),
            "grid_resistance": (
                "sources.grid.r", "ParallelNetwork.R_grid", self.grid_scale * R_GRID,
            ),
            "forming_power_reference": (
                "units.forming.ctrl.references.p_ref_pu",
                "ReferenceControl.forming.p_ref_pu", self.p_ref_pu,
            ),
            "following_current_reference": (
                "units.following.ctrl.references.id_ref_pu",
                "ReferenceControl.following.id_ref_pu", self.id_ref_pu,
            ),
        }

    def controls(self):
        forming = ControlCase(
            self.name + "_forming", "psc",
            {"k_p_pu": 6.0, "k_v": 0.0, "k_vi": 0.0},
            p_ref_pu=self.p_ref_pu,
        )
        following = ControlCase(
            self.name + "_following", "gfl",
            {"pll_kp": 30.0, "pll_ki": 600.0,
             "current_kp": 0.12, "current_ki": 3.0},
            id_ref_pu=self.id_ref_pu,
        )
        return forming, following

    def motulator_system(self):
        network = _ParallelNetwork(self.grid_scale)
        source = motulator_model.ThreePhaseSource(w_g=W0, e_g=V_GRID, phi=0.0)
        models = {}
        for name, port in (("forming", "1"), ("following", "2")):
            converter = motulator_model.VoltageSourceConverter(u_dc=V_DC, modulator=ZOH(delay=0))
            models[name] = _ParallelConverterModel(converter, network, source, port)
        return motulator_model.CompositeModel(models)


PARALLEL_CASES = (
    ParallelCase("parallel_stiff_grid", grid_scale=0.7, p_ref_pu=0.08, id_ref_pu=0.06),
    ParallelCase("parallel_weak_grid", grid_scale=1.8, p_ref_pu=0.14, id_ref_pu=0.10),
)


def _interpolate(t, values, at):
    return np.interp(at, t, np.real(values)) + 1j * np.interp(at, t, np.imag(values))


def _complex_state(result, name):
    return np.asarray(result.states[name + ".re"]) + 1j * np.asarray(result.states[name + ".im"])


def _errors(actual_t, pairs, reference_t):
    errors = {}
    for name, (actual, reference) in pairs.items():
        expected = _interpolate(reference_t, reference, actual_t)
        scale = max(1.0, float(np.max(np.abs(expected))))
        difference = np.abs(actual - expected) / scale
        errors[name] = {
            "mean": float(np.mean(difference)),
            "maximum": float(np.max(difference)),
        }
    return errors


def _check_errors(errors, limit, timings, record_property):
    for signal, values in errors.items():
        record_property(f"{signal}_mean_normalized_difference", values["mean"])
        record_property(f"{signal}_maximum_normalized_difference", values["maximum"])
    mean = float(np.mean([values["mean"] for values in errors.values()]))
    maximum = max(values["maximum"] for values in errors.values())
    record_property("mean_normalized_difference", mean)
    record_property("maximum_normalized_difference", maximum)
    record_property("peslite_simulation_seconds", timings["peslite"])
    record_property("motulator_simulation_seconds", timings["motulator"])
    record_property(
        "simulation_time_difference_seconds",
        timings["peslite"] - timings["motulator"],
    )
    record_property(
        "peslite_to_motulator_simulation_time_ratio",
        timings["peslite"] / timings["motulator"],
    )
    assert maximum <= limit, errors


def compare_single(case: ControlCase, out_dir):
    peslite_simulation = peslite.Simulation(case.peslite_params())
    started = perf_counter()
    peslite_result = peslite_simulation.run(out_dir=out_dir)
    peslite_seconds = perf_counter() - started
    system = case.motulator_system()
    motulator_simulation = motulator_model.Simulation(
        system, _ReferenceControl(case), show_progress=False,
        cfg=SolverCfg(method="DOP853", rtol=1e-10, atol=1e-12,
                      max_step=T_CONTROL / 20.0),
    )
    started = perf_counter()
    result = motulator_simulation.simulate(t_stop=case.t_end + T_CONTROL, N_eval=1)
    motulator_seconds = perf_counter() - started
    assert result.success
    pairs = {
        "converter_current": (
            _complex_state(peslite_result, "converter.branch_f.i"),
            np.asarray(result.mdl.ac_filter.i_c_ab),
        ),
        "bus_voltage": (
            _complex_state(peslite_result, "pcc.u_C"),
            np.asarray(result.mdl.ac_filter.u_f_ab),
        ),
        "grid_current": (
            -_complex_state(peslite_result, "grid.branch.i"),
            np.asarray(result.mdl.ac_filter.i_g_ab),
        ),
    }
    errors = _errors(np.asarray(peslite_result.t), pairs, np.asarray(result.mdl.t))
    return errors, {"peslite": peslite_seconds, "motulator": motulator_seconds}


def compare_parallel(case: ParallelCase, out_dir):
    forming, following = case.controls()
    tree = {
        "base": {"s_base": S_BASE, "v_ll_rms": V_LL_RMS, "f0": 50.0},
        "buses": {"pcc": {"c": C_BUS, "r_d": 0.0}},
        "sources": {"grid": {
            "bus": "pcc", "l": case.grid_scale * L_GRID,
            "r": case.grid_scale * R_GRID, "v": V_GRID,
        }},
        "units": {"forming": _unit(forming.control()), "following": _unit(following.control())},
        "simulation": {
            "t_end": case.t_end,
            "initial": {"states": {"pcc.u_C": [V_GRID, 0.0]}},
            "energy_check": "off", "progress": {"enable": False},
            "solver": {"type": "fixed", "method": "rk4", "dt": T_CONTROL / 20.0,
                       "linearisations": 0},
            "output": {"period": T_CONTROL},
        },
    }
    peslite_simulation = peslite.Simulation(from_dict(tree))
    started = perf_counter()
    peslite_result = peslite_simulation.run(out_dir=out_dir)
    peslite_seconds = perf_counter() - started
    system = case.motulator_system()
    controls = {
        "forming": _ReferenceControl(forming, "i_1"),
        "following": _ReferenceControl(following, "i_2"),
    }
    motulator_simulation = motulator_model.Simulation(
        system, controls, show_progress=False,
        cfg=SolverCfg(method="DOP853", rtol=1e-10, atol=1e-12,
                      max_step=T_CONTROL / 20.0),
    )
    started = perf_counter()
    result = motulator_simulation.simulate(t_stop=case.t_end + T_CONTROL, N_eval=1)
    motulator_seconds = perf_counter() - started
    assert result.success
    network = result.mdl.forming.network
    pairs = {
        "forming_current": (
            _complex_state(peslite_result, "forming.branch_f.i"), np.asarray(network.i_1),
        ),
        "following_current": (
            _complex_state(peslite_result, "following.branch_f.i"), np.asarray(network.i_2),
        ),
        "bus_voltage": (
            _complex_state(peslite_result, "pcc.u_C"), np.asarray(network.u_bus),
        ),
        "grid_current": (
            -_complex_state(peslite_result, "grid.branch.i"), np.asarray(network.i_grid),
        ),
    }
    errors = _errors(np.asarray(peslite_result.t), pairs, np.asarray(result.mdl.t))
    return errors, {"peslite": peslite_seconds, "motulator": motulator_seconds}


@pytest.mark.parametrize("case", SINGLE_CASES, ids=lambda case: case.name)
def test_single_converter_control_matches_motulator(case, tmp_path, record_property):
    assert importlib.metadata.version("motulator") == MOTULATOR_VERSION
    assert all(peslite_path and motulator_path and np.isfinite(value)
               for peslite_path, motulator_path, value in case.parameter_map().values())
    errors, timings = compare_single(case, tmp_path / case.name)
    _check_errors(errors, 5e-6, timings, record_property)


@pytest.mark.parametrize("case", PARALLEL_CASES, ids=lambda case: case.name)
def test_parallel_gfm_gfl_system_matches_motulator(case, tmp_path, record_property):
    assert importlib.metadata.version("motulator") == MOTULATOR_VERSION
    assert all(peslite_path and motulator_path and np.isfinite(value)
               for peslite_path, motulator_path, value in case.parameter_map().values())
    errors, timings = compare_parallel(case, tmp_path / case.name)
    _check_errors(errors, 5e-6, timings, record_property)
