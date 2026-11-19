"""Add-on discovery plus a custom PLL assembled and run with built-in controller loops."""

import importlib
import inspect
from pathlib import Path

import numpy as np
import peslite
import pytest
from peslite.addons import components as addon_components
from peslite.addons import control as addon_control
from peslite.addons.components.cable import (BergeronCableModel, Cable,
                                             LumpedCableModel)
from peslite.assembly.params import from_dict, runtime_changeable
from peslite.addons.control.voltage_adaptive_pll import VoltageAdaptivePLL
from peslite.addons.control.predictive import (PredictiveCurrentControl,
                                               ThreePhasePredictiveCurrentControl)
from peslite.components.network import ELEMENT_TYPES
from peslite.control import LOOP_TYPES, Loop
from peslite.solver import ConfigError
from peslite.solver.history import DelayHistory


def test_control_addon_path_registers_loops(tmp_path, monkeypatch):
    path = tmp_path / "control"
    path.mkdir()
    (path / "_private.py").write_text(
        "raise RuntimeError('private add-on helpers must not be discovered')\n",
        encoding="utf-8",
    )
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
from peslite.addons.components import register_component_type
from peslite.solver import netlist as nl

@register_component_type
@dataclass(frozen=True, kw_only=True)
class AddonElement:
    bus: str
    type: str = "addon_test_element"

    def network(self, n):
        n.add(nl.Resistor(n.bus, n.ground, 1.0))
        n.losses()
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(addon_components, "__path__", [str(path)])
    importlib.invalidate_caches()

    imported = addon_components.discover()

    assert [module.__name__ for module in imported] == [
        "peslite.addons.components.test_component_addon"
    ]
    assert ELEMENT_TYPES["addon_test_element"].params.__module__ == imported[0].__name__


def test_functions_have_one_lazy_public_namespace():
    functions = peslite.addons.functions
    assert peslite.addons.__all__ == ["control", "components", "formats", "functions"]
    assert not hasattr(peslite.addons, "plot_csv")
    assert set(functions.__all__) == set(functions._EXPORTS)
    assert len(set(functions._EXPORTS.values())) == len(functions._EXPORTS)

    local = {}
    for name, (module_name, attribute) in functions._EXPORTS.items():
        value = getattr(functions, name)
        qualified = (module_name if "." in module_name
                     else f"{functions.__name__}.{module_name}")
        assert callable(value)
        assert value is getattr(importlib.import_module(qualified), attribute)
        if "." not in module_name:
            local.setdefault(module_name, set()).add(attribute)

    directory = Path(functions.__file__).parent
    for path in directory.glob("*.py"):
        if path.stem.startswith("_"):
            continue
        module = importlib.import_module(f"{functions.__name__}.{path.stem}")
        public_functions = {
            name for name in module.__all__
            if inspect.isfunction(getattr(module, name, None))
            and getattr(module, name).__module__ == module.__name__
        }
        assert public_functions == local.get(path.stem, set())


def test_missing_function_provider_has_one_facade_error(monkeypatch):
    functions = peslite.addons.functions
    monkeypatch.setitem(functions._EXPORTS, "missing_function", ("missing_provider", "run"))
    with pytest.raises(functions.FunctionUnavailableError, match="missing_function"):
        functions._resolve("missing_function")


def test_mpc_is_a_control_addon():
    assert LOOP_TYPES[PredictiveCurrentControl.type] is PredictiveCurrentControl
    assert (
        LOOP_TYPES[ThreePhasePredictiveCurrentControl.type]
        is ThreePhasePredictiveCurrentControl
    )
    assert "CCSMPC" not in peslite.control.__all__
    assert "PredictiveCurrentControl" not in peslite.control.__all__


def test_addon_entry_points_match_builtin_public_apis():
    assert set(peslite.control.__all__) < set(addon_control.__all__)
    assert set(peslite.components.__all__) < set(addon_components.__all__)
    assert addon_control.LOOP_TYPES is LOOP_TYPES
    assert not {"Element", "NetworkComponent", "ELEMENT_TYPES"} & set(
        addon_components.__all__
    )


def test_cable_model_has_two_inward_ports_and_closes_its_power_balance():
    cable = LumpedCableModel(
        length_km=2.0,
        r_per_km=0.08,
        l_per_km=0.6e-3,
        g_per_km=2.0e-7,
        c_per_km=0.3e-6,
        sections=2,
        voltage0=0j,
    )
    cable.port1.voltage = 8000.0 + 200.0j
    cable.port2.voltage = 7600.0 - 100.0j
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
    power_in = cable.port1.power() + cable.port2.power()

    assert cable.port1.component is cable and cable.port2.component is cable
    assert cable.output.i1 == cable.state.iL1
    assert cable.output.i2 == -cable.state.iL3
    assert sum(cable.series_r) == pytest.approx(0.16)
    assert sum(cable.series_l) == pytest.approx(1.2e-3)
    assert power_in == pytest.approx(stored_rate + cable.dissipated_power())


def test_delay_history_interpolates_committed_values_and_restores_checkpoints():
    history = DelayHistory(1.0, 0.0)
    history.accept(0.0, 0.0)
    history.accept(0.5, 1.0)
    checkpoint = history.checkpoint()
    history.accept(1.0, 1.0)

    assert history(0.25) == pytest.approx(0.5)
    assert history.derivative(0.5) == pytest.approx(2.0)
    assert history.integral_squared(1.0) == pytest.approx(2.0 / 3.0)
    history.accept(1.5, 2.0)
    assert history.derivative(0.5) == pytest.approx(2.0)

    history.restore(checkpoint)
    assert history.times == [0.0, 0.5]
    assert history.values == [0.0, 1.0]


def test_cable_defaults_to_bergeron_and_rejects_unsupported_shunt_loss():
    values = {
        "bus1": "left", "bus2": "right", "length_km": 5.0,
        "r_per_km": 0.05, "l_per_km": 0.4e-3, "c_per_km": 0.2e-6,
    }
    assert Cable.Params(**values).model == "bergeron"
    with pytest.raises(ConfigError, match="requires g_per_km=0"):
        Cable.Params(**values, g_per_km=1.0e-8)


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
from peslite.addons.components import register_component_type
from peslite.solver import netlist as nl

@register_component_type
@dataclass(frozen=True, kw_only=True)
class ProjectLocalElement:
    bus: str
    l: float
    r: float
    type: str = "project_local_element"

    def network(self, n):
        middle = n.internal()
        n.add(
            nl.Resistor(n.bus, middle, self.r),
            nl.Inductor(middle, n.ground, self.l, n.state("i")),
        )
        n.port_output("i", "bus", "current")
        n.losses()
""",
        encoding="utf-8",
    )
    (functions / "project_local_function.py").write_text("VALUE = 42\n", encoding="utf-8")

    case = root / "local-case.pes"
    case.write_text(
        (examples / "gfl-example.pes").read_text(encoding="utf-8").replace(
            "SRFPLL:", "ProjectLocalPLL:", 1
        ) + "\nelements:\n  project_load: {type: project_local_element, bus: pcc, l: 0.001, r: 1.0}\n",
        encoding="utf-8",
    )

    params = peslite.load(case)
    simulation = peslite.Simulation(params)
    local_function = importlib.import_module(
        "peslite.addons.functions.project_local_function"
    )

    assert type(simulation.unit().ctrl.graph.nodes["ProjectLocalPLL"]).__name__ == "ProjectLocalPLL"
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

    expected = {"VoltageAdaptivePLL", "CurrentLoop", "DCVoltageControl"}
    if model != "averaging":
        expected.add("ThreePhaseModulation")
    assert set(nodes) == expected
    assert isinstance(nodes["VoltageAdaptivePLL"], VoltageAdaptivePLL)
    assert type(nodes["VoltageAdaptivePLL"]).sample is Loop.sample
    assert type(nodes["VoltageAdaptivePLL"]).flow_path is Loop.flow_path
    assert set(nodes["VoltageAdaptivePLL"]._state_blocks) == {"theta", "omega_g", "u_g_pu"}
    assert nodes["CurrentLoop"].type == "dq_current_pi"
    assert nodes["DCVoltageControl"].type == "dc_voltage_control"

    result = simulation.run(out_dir=tmp_path / model)
    theta = result.states["vsc.ctrl.VoltageAdaptivePLL.theta"]
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

    assert not runtime_changeable("elements.cable1.sections", params)
    assert simulation.ph_report.verdict == "port-hamiltonian"
    assert not result.tripped
    assert np.isfinite(result.states["cable1.iL1.re"]).all()
    assert np.isfinite(result.states["cable1.uC2.im"]).all()
    assert {"cable1.i1_a", "cable1.i2_a"} <= result.plant.keys()


def test_custom_cable_parameters_can_change_at_an_event(examples, tmp_path):
    params = peslite.load(
        examples / "cable-gfl-example.pes",
        **{
            "simulation.t_end": 0.001,
            "simulation.solver.type": "adaptive",
            "simulation.solver.method": "DP45",
            "simulation.solver.linearisations": 0,
            "simulation.output.signals": 0,
            "units.vsc.bridge.model": "averaging",
        },
    )
    initial = params.replace(**{"elements.cable1.r_per_km": 0.2})
    event = params.replace(**{
        "events.cable_r": {
            "type": "set",
            "t": 0.0,
            "set": {"elements.cable1.r_per_km": 0.2},
        }
    })
    finals = [
        peslite.Simulation(case).run(out_dir=tmp_path / str(index)).final_states()
        for index, case in enumerate((initial, event))
    ]
    assert finals[1] == pytest.approx(finals[0], rel=1e-9)


def test_bergeron_cable_runs_with_adaptive_steps_and_strict_energy(examples, tmp_path):
    params = peslite.load(
        examples / "cable-gfl-example.pes",
        **{
            "elements.cable1.model": "bergeron",
            "elements.cable1.g_per_km": 0.0,
            "elements.cable1.r_per_km": 0.0,
            "units.vsc.bridge.model": "averaging",
            "simulation.t_end": 2.0e-4,
            "simulation.solver.type": "adaptive",
            "simulation.solver.method": "DP45",
            "simulation.solver.max_step": 50.0e-6,
            "simulation.solver.linearisations": 0,
            "simulation.energy_check": "strict",
            "simulation.output.period": 25.0e-6,
            "simulation.output.signals": 1,
        },
    )
    simulation = peslite.Simulation(params)
    cable = simulation.system.named_elements["cable1"].model
    result = simulation.run(out_dir=tmp_path / "bergeron")

    assert isinstance(cable, BergeronCableModel)
    assert cable.segment_delay < params.simulation.solver.max_step
    assert not any(name.startswith("cable1.") for name in result.states)
    assert np.isfinite(result.plant["cable1.i1_a"]).all()
    assert np.isfinite(result.plant["cable1.i2_a"]).all()
    assert simulation.ph_report.verdict == "port-hamiltonian"


@pytest.mark.long
def test_lumped_cable_converges_to_bergeron_terminal_behavior(tmp_path):
    inductance_per_km = 0.4e-3
    capacitance_per_km = 0.2e-6
    surge_impedance = np.sqrt(inductance_per_km / capacitance_per_km)

    def run(model, sections=4):
        line = {
            "type": "cable", "model": model,
            "bus1": "send", "bus2": "receive", "length_km": 5.0,
            "r_per_km": 0.0, "l_per_km": inductance_per_km,
            "c_per_km": capacitance_per_km,
        }
        if model == "lumped":
            line["sections"] = sections
        params = from_dict({
            "base": {"s_base": 1.0e6, "v_ll_rms": 10.0e3, "f0": 50.0},
            "buses": {"send": {"phases": 3}, "receive": {"phases": 3}},
            "sources": {
                "grid": {"bus": "send", "l": 1.0e-6, "r": surge_impedance},
            },
            "elements": {"line": line},
            "simulation": {
                "t_end": 3.0e-4,
                "solver": {
                    "type": "fixed", "method": "rk4", "dt": 1.0e-7,
                    "linearisations": 0,
                },
                "energy_check": "off",
                "progress": {"enable": False},
                "output": {
                    "period": 2.0e-6, "precision": 17,
                    "states": False, "signals": True, "energy": False,
                },
            },
        })
        simulation = peslite.Simulation(params)
        result = simulation.run(out_dir=tmp_path / f"{model}-{sections}")
        trajectory = np.column_stack([
            result.plant[name]
            for name in ("receive.u_a", "line.i1_a")
        ])
        return simulation, np.asarray(result.t), trajectory

    reference_simulation, times, reference = run("bergeron")
    travel_time = reference_simulation.system.named_elements["line"].model.travel_time
    before_wavefront = times < 0.9 * travel_time
    assert np.ptp(reference[before_wavefront, 0]) < 1.0e-10

    comparison = times >= 2.0 * travel_time
    scale = np.maximum(1.0, np.max(np.abs(reference[comparison]), axis=0))
    rms_errors, maximum_errors = [], []
    for sections in (2, 8, 32):
        _simulation, actual_times, actual = run("lumped", sections)
        np.testing.assert_array_equal(actual_times, times)
        normalized = (actual[comparison] - reference[comparison]) / scale
        rms_errors.append(float(np.sqrt(np.mean(normalized ** 2))))
        maximum_errors.append(float(np.max(np.abs(normalized))))

    assert rms_errors[0] > rms_errors[1] > rms_errors[2]
    assert rms_errors[2] < 2.0e-6
    assert maximum_errors[2] < 2.0e-5
