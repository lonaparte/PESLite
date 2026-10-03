"""Add-on discovery plus a custom PLL assembled and run with built-in controller loops."""

import importlib

import numpy as np
import peslite
import pytest
from peslite.addons import components as addon_components
from peslite.addons import control as addon_control
from peslite.addons.components.cable import CableModel
from peslite.addons.control.voltage_adaptive_pll import VoltageAdaptivePLL
from peslite.components import ELEMENT_TYPES
from peslite.control import LOOP_TYPES, Loop


def test_control_addon_path_registers_loops(tmp_path, monkeypatch):
    path = tmp_path / "control"
    path.mkdir()
    (path / "test_controller_addon.py").write_text(
        """from dataclasses import dataclass
from peslite.addons.control import Loop, register_loop_type

@register_loop_type
class AddonLoop(Loop):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float | None = None
        type: str = "addon_test_loop"

    type = "addon_test_loop"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(addon_control, "__path__", [str(path)])
    importlib.invalidate_caches()

    imported = addon_control.discover()

    assert [module.__name__ for module in imported] == [
        "peslite.addons.control.test_controller_addon"
    ]
    assert LOOP_TYPES["addon_test_loop"].__module__ == imported[0].__name__


def test_component_addon_path_registers_elements(tmp_path, monkeypatch):
    path = tmp_path / "components"
    path.mkdir()
    (path / "test_component_addon.py").write_text(
        """from dataclasses import dataclass
from peslite.addons.components import Element, register_element_type

@register_element_type
class AddonElement(Element):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "addon_test_element"
        bus: str

    type = "addon_test_element"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(addon_components, "__path__", [str(path)])
    importlib.invalidate_caches()

    imported = addon_components.discover()

    assert [module.__name__ for module in imported] == [
        "peslite.addons.components.test_component_addon"
    ]
    assert ELEMENT_TYPES["addon_test_element"].__module__ == imported[0].__name__


def test_functions_hold_the_plotting_addon():
    assert peslite.addons.functions.plot_csv is peslite.addons.plot_csv
    assert peslite.addons.functions.plot_result is peslite.addons.plot_result


def test_addon_entry_points_match_builtin_public_apis():
    assert set(peslite.control.__all__) < set(addon_control.__all__)
    assert set(peslite.components.__all__) < set(addon_components.__all__)
    assert addon_control.LOOP_TYPES is LOOP_TYPES
    assert addon_components.ELEMENT_TYPES is ELEMENT_TYPES


def test_cable_model_has_two_inward_ports_and_closes_its_power_balance():
    cable = CableModel(
        length_km=2.0,
        r_per_km=0.08,
        l_per_km=0.6e-3,
        g_per_km=2.0e-7,
        c_per_km=0.3e-6,
        sections=2,
        voltage0=0j,
    )
    cable.inp.u1 = 8000.0 + 200.0j
    cable.inp.u2 = 7600.0 - 100.0j
    cable.state.iL1 = 20.0 + 3.0j
    cable.state.iL2 = 18.0 + 2.0j
    cable.state.iL3 = 16.0 + 1.0j
    cable.state.uC1 = 7900.0 + 100.0j
    cable.state.uC2 = 7750.0
    cable.set_outputs(0.0)
    derivatives = cable.rhs(0.0)

    stored_rate = 0.0
    for storage, derivative in zip(cable.storage, derivatives):
        value = getattr(cable.state, storage.state)
        stored_rate += storage.scale * storage.value * (value * derivative.conjugate()).real
    power_in = 1.5 * (
        (cable.inp.u1 * cable.out.i1.conjugate()).real
        + (cable.inp.u2 * cable.out.i2.conjugate()).real
    )

    assert cable.terminal1.direction == cable.terminal2.direction == 1
    assert cable.out.i1 == cable.state.iL1
    assert cable.out.i2 == -cable.state.iL3
    assert sum(cable.series_r) == pytest.approx(0.16)
    assert sum(cable.series_l) == pytest.approx(1.2e-3)
    assert power_in == pytest.approx(stored_rate + cable.dissipated_power())


def test_pesaddons_beside_simulation_file_merges_project_paths(examples, tmp_path):
    root = tmp_path / "project"
    control = root / "PESaddons/control"
    components = root / "PESaddons/components"
    functions = root / "PESaddons/functions"
    control.mkdir(parents=True)
    components.mkdir()
    functions.mkdir()
    (control / "project_local_pll.py").write_text(
        """import cmath
from dataclasses import dataclass
from peslite.addons.control import (ANGLE, FREQUENCY, V_AB, Integrator, Loop,
                                    register_loop_type)

@register_loop_type
class ProjectLocalPLL(Loop):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        kp_pu: float
        ki_pu: float
        period: float | None = None
        type: str = "project_local_pll"

    type = "project_local_pll"
    role = "pll"
    inputs = {"v": V_AB}
    outputs = {"theta": ANGLE, "frame": ANGLE, "omega": FREQUENCY}
    flow_inputs = ("v",)

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.w0 = unit.base.w0
        self.angle = self.state_block("theta", Integrator(self.block_period))
        self.integral = self.state_block("integral_pu", Integrator(self.block_period))
        self.omega = self.w0

    def initial_outputs(self):
        return {"theta": self.angle.value, "frame": self.angle.value, "omega": self.omega}

    def equation(self, voltage):
        frame = self.angle.value
        error = (voltage * cmath.exp(-1j * frame)).imag
        self.omega = self.w0 + self.cfg.kp_pu * error + self.cfg.ki_pu * self.integral(error)
        theta = self.angle(self.omega)
        return theta, frame, self.omega
""",
        encoding="utf-8",
    )
    (components / "project_local_element.py").write_text(
        """from dataclasses import dataclass
from peslite.addons.components import Load, register_element_type

@register_element_type
class ProjectLocalElement(Load):
    @dataclass(frozen=True, kw_only=True)
    class Params(Load.Params):
        type: str = "project_local_element"

    type = "project_local_element"
""",
        encoding="utf-8",
    )
    (functions / "project_local_function.py").write_text("VALUE = 42\n", encoding="utf-8")

    case = root / "local-case.pes"
    case.write_text(
        (examples / "gfl-example.pes").read_text(encoding="utf-8").replace(
            "type: srf_pll", "type: project_local_pll", 1
        ) + "\nelements:\n  project_load: {type: project_local_element, bus: pcc, l: 0.001, r: 1.0}\n",
        encoding="utf-8",
    )

    params = peslite.load(case)
    simulation = peslite.Simulation(params)
    local_function = importlib.import_module(
        "peslite.addons.functions.project_local_function"
    )

    assert type(simulation.unit().ctrl.graph.nodes["pll"]).__name__ == "ProjectLocalPLL"
    assert params.elements["project_load"].type == "project_local_element"
    assert local_function.VALUE == 42
    assert str((root / "PESaddons").resolve()) in peslite.addons.__path__
    assert str(control.resolve()) in addon_control.__path__
    assert str(components.resolve()) in addon_components.__path__
    assert str(functions.resolve()) in peslite.addons.functions.__path__


@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_custom_pll_assembles_with_builtin_loops_and_runs(examples, tmp_path, model):
    params = peslite.load(
        examples / "custom-pll-example.pes",
        **{
            "simulation.t_end": 0.004,
            "simulation.solver.linearisations": 0,
            "units.vsc.bridge.model": model,
        },
    )
    simulation = peslite.Simulation(params)
    nodes = simulation.unit().ctrl.graph.nodes

    assert set(nodes) == {"pll", "cc", "dvc"}
    assert isinstance(nodes["pll"], VoltageAdaptivePLL)
    assert type(nodes["pll"]).sample is Loop.sample
    assert type(nodes["pll"]).flow_path is Loop.flow_path
    assert set(nodes["pll"]._state_blocks) == {"theta", "omega_g", "u_g_pu"}
    assert nodes["cc"].type == "dq_current_pi"
    assert nodes["dvc"].type == "dc_voltage_pi"

    result = simulation.run(out_dir=tmp_path / model)
    theta = result.states["vsc.ctrl.pll.theta"]
    assert np.isfinite(theta).all()
    assert theta[-1] > 1.0
    assert not result.tripped


@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_custom_cable_gfl_case_runs_in_every_bridge_mode(examples, tmp_path, model):
    solver = ("adaptive", "DP45") if model == "averaging" else ("fixed", "rk4")
    params = peslite.load(
        examples / "cable-gfl-example.pes",
        **{
            "simulation.t_end": 0.004,
            "simulation.solver.type": solver[0],
            "simulation.solver.method": solver[1],
            "simulation.solver.linearisations": 0,
            "simulation.energy_check": "strict",
            "simulation.output.signals": 1,
            "units.vsc.bridge.model": model,
        },
    )
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path / model)

    assert simulation.ph_report.verdict == "port-hamiltonian"
    assert not result.tripped
    assert np.isfinite(result.states["cable1.iL1.re"]).all()
    assert np.isfinite(result.states["cable1.uC2.im"]).all()
    assert {"cable1.i1_a", "cable1.i2_a"} <= result.plant.keys()
