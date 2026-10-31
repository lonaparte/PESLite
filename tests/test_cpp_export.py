"""Standalone C++ export: API, CLI layout and numerical smoke tests."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import numpy as np
import yaml

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import from_dict
from peslite.components import Bag, Element, Terminal, register_element_type
from peslite.solver import BlackBoxSubsystem
from peslite.solver.simulation import convert_main
from test_measurement_network import load_case

pytestmark = pytest.mark.full


class _ConditionalState(Bag):
    __slots__ = ("i",)


class _ConditionalInput(Bag):
    __slots__ = ("u",)


class _ConditionalOutput(Bag):
    __slots__ = ("i",)


class _ConditionalModel:
    state_names = ("i",)
    outputs_need_inputs = False

    def __init__(self, inductance, resistance_low, resistance_high, threshold):
        self.inductance = inductance
        self.resistance_low = resistance_low
        self.resistance_high = resistance_high
        self.threshold = threshold
        self.state = _ConditionalState(i=0j)
        self.inp = _ConditionalInput(u=0j)
        self.out = _ConditionalOutput(i=0j)
        self.terminal = Terminal(self, "u", "i", 1)

    def set_outputs(self, t):
        self.out.i = self.state.i

    def rhs(self, t):
        resistance = (self.resistance_high
                      if abs(self.state.i) > self.threshold else self.resistance_low)
        return ((self.inp.u - resistance * self.state.i) / self.inductance,)


@register_element_type
class _ConditionalElement(Element):
    type = "test_conditional_element"

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bus: str
        inductance: float
        resistance_low: float
        resistance_high: float
        threshold: float
        type: str = "test_conditional_element"

    def __init__(self, name, cfg, buses, p):
        self.name, self.cfg = name, cfg
        self.model = _ConditionalModel(
            cfg.inductance, cfg.resistance_low, cfg.resistance_high, cfg.threshold
        )

    def subsystems(self):
        return {self.name: self.model}

    def connections(self):
        return {}

    def terminals(self):
        return ((self.cfg.bus, self.model.terminal),)

    def signals(self):
        return {f"{self.name}.i": self.model.out.i,
                f"{self.name}.i_mag": abs(self.model.out.i)}


@register_element_type
class _BlackBoxElement(Element):
    type = "test_black_box"

    @dataclass(frozen=True, kw_only=True)
    class Params:
        decay: float
        initial: float
        type: str = "test_black_box"

    def __init__(self, name, cfg, buses, p):
        self.name = name
        self.model = BlackBoxSubsystem(
            lambda _t, state, _inputs: state,
            inputs=(),
            outputs=("y",),
            state={"x": cfg.initial},
            rhs=lambda _t, state, _inputs: -cfg.decay * state,
            feedthrough=(),
        )

    def subsystems(self):
        return {self.name: self.model}

    def connections(self):
        return {}

    def terminals(self):
        return ()

    def signals(self):
        return {f"{self.name}.y": self.model.out.y}


@register_element_type
class _DiscreteBlackBoxElement(Element):
    type = "test_discrete_black_box"

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float
        increment: float
        initial: float
        type: str = "test_discrete_black_box"

    def __init__(self, name, cfg, buses, p):
        self.name = name
        self.model = BlackBoxSubsystem(
            lambda _t, state, _inputs: state,
            inputs=(),
            outputs=("y",),
            state={"hidden": cfg.initial},
            update=lambda _t, state, _inputs: state + cfg.increment,
            period=cfg.period,
            feedthrough=(),
        )

    def subsystems(self):
        return {self.name: self.model}

    def connections(self):
        return {}

    def terminals(self):
        return ()

    def signals(self):
        return {f"{self.name}.y": self.model.out.y}


def _compiler() -> str | None:
    return next((path for name in ("c++", "g++", "clang++")
                 if (path := shutil.which(name)) is not None), None)


def _summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _final_csv(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {key: float(value) for key, value in list(csv.DictReader(stream))[-1].items()}


def _run_generated(params, project: Path, compiler: str):
    peslite.export(params, "cpp", project, name=project.name)
    source = project / "peslite.cpp"
    executable = project / "peslite"
    subprocess.run(
        [compiler, "-O3", "-DNDEBUG", "-std=c++17", str(source), "-o", str(executable)],
        check=True,
    )
    cpp_out, python_out = project / "cpp-output", project / "python-output"
    subprocess.run([str(executable), str(cpp_out)], check=True)
    peslite.Simulation(params).run(out_dir=python_out)
    return source, cpp_out, python_out


def _quick(mode: str):
    solver = ("adaptive", "DP45") if mode == "averaging" else ("fixed", "rk4")
    return peslite.load(EXAMPLES / "gfl-example.pes").replace(**{
        "units.vsc.bridge.model": mode,
        "simulation.solver.type": solver[0],
        "simulation.solver.method": solver[1],
        "simulation.solver.subsystems": {},
        "simulation.t_end": 0.202,
        "simulation.output.period": 1e-3,
        "simulation.energy_check": "off",
    })


def test_sampled_measurements_reuse_the_parameterized_unit_adapter(monkeypatch):
    from peslite.assembly.cpp import (_Expression, _ModelGenerator, _RuntimeParameters,
                                      _make_unit_generator)

    sim = peslite.Simulation(_quick("pwm_averaging"))
    unit = sim.units["vsc"]
    original_cfg, original_ctrl = unit.cfg, unit.ctrl
    runtime = _RuntimeParameters(sim.p, ("base.v_ll_rms", "units.vsc.dclink.vdc_ref"))
    generator = _make_unit_generator("vsc", unit, _ModelGenerator(sim, runtime))
    original = type(unit).control_inputs_from
    calls = []

    def observe_adapter(self, values):
        calls.append((self.cfg, self.ctrl))
        return original(self, values)

    monkeypatch.setattr(type(unit), "control_inputs_from", observe_adapter)
    inputs = generator._sampled_control_inputs()
    assert len(calls) == 1
    assert calls[0][0] is generator.cfg
    assert calls[0][1] is generator.ctrl
    assert unit.cfg is original_cfg and unit.ctrl is original_ctrl
    scales = {"u_g": generator.cfg.base.v_phase_peak,
              "i_c": generator.cfg.base.i_phase_peak, "u_dc": generator.cfg.dc_base.v}
    for channel, scale in scales.items():
        reading = _Expression(generator._adc_value(channel), inputs["meas." + channel].kind)
        assert inputs["meas." + channel].code == (reading / scale).code
    assert "parameters.param_base_v_ll_rms" in inputs["meas.u_g"].code
    assert "parameters.param_units_vsc_dclink_vdc_ref" in inputs["meas.u_dc"].code
    assert generator.cfg.dc_base.i.default == pytest.approx(
        unit.cfg.base.s_base / unit.cfg.dclink.vdc_ref)


def test_export_preserves_system_factories_without_mutating_the_caller(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    calls = []

    def make_system(params):
        system = peslite.System(params)
        branch = system.units["vsc"].branch_f
        branch.retune(1.8 * branch.L, branch.R)
        calls.append(system)
        return system

    simulation = peslite.Simulation(_quick("pwm_averaging"), parts={"system": make_system})
    initial = simulation.system.model.get_initial_values().copy()
    peslite.export(simulation, "cpp", tmp_path, name="factory")
    assert len(calls) == 2 and calls[0] is simulation.system
    np.testing.assert_array_equal(simulation.system.model.get_initial_values(), initial)
    assert simulation.result is None
    executable = tmp_path / "peslite"
    subprocess.run([compiler, "-O3", "-std=c++17", str(tmp_path / "peslite.cpp"),
                    "-o", str(executable)], check=True)
    subprocess.run([str(executable), str(tmp_path / "native")], check=True)
    expected = simulation.run(out_dir=tmp_path / "python").final_states()
    actual = _final_csv(tmp_path / "native" / "states.csv")
    np.testing.assert_allclose([actual[key] for key in expected], list(expected.values()),
                               rtol=2e-9, atol=2e-8)


@pytest.mark.parametrize("mode", ("switching", "pwm_averaging"))
def test_generated_cpp_matches_current_directed_pwm_dead_time(mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = _quick(mode).replace(**{
        "units.vsc.pwm.dead_time": 1.0e-6,
        "simulation.t_end": 0.2003,
        "simulation.output.period": 5.0e-6,
        "simulation.output.signals": True,
    })

    source, cpp_out, python_out = _run_generated(
        params, tmp_path / f"dead-time-{mode}", compiler
    )

    for filename in ("states.csv", "plant.csv"):
        actual = np.genfromtxt(cpp_out / filename, delimiter=",", names=True)
        expected = np.genfromtxt(python_out / filename, delimiter=",", names=True)
        assert actual.dtype.names == expected.dtype.names
        for column in expected.dtype.names:
            np.testing.assert_allclose(actual[column], expected[column], rtol=2e-7, atol=2e-7,
                                       equal_nan=True, err_msg=f"{mode}:{filename}:{column}")
    text = source.read_text(encoding="utf-8")
    assert ("DeadTimeSequenceState" in text) == (mode == "switching")


def test_export_rejects_unrebuildable_instances_before_creating_files(tmp_path):
    params = _quick("pwm_averaging")
    simulation = peslite.Simulation(params, system=peslite.System(params))
    destination = tmp_path / "export"
    with pytest.raises(TypeError, match="supplied instances"):
        peslite.export(simulation, "cpp", destination)
    assert not destination.exists()


def test_export_rejects_a_silently_changed_solver_factory(tmp_path):
    from peslite.solver.integrators import FixedStepSolver

    simulation = peslite.Simulation(_quick("pwm_averaging"), parts={
        "solver": lambda params: FixedStepSolver(params.simulation.solver.dt * 2.0),
    })
    with pytest.raises(TypeError, match="solver replacement"):
        peslite.export(simulation, "cpp", tmp_path / "export")


def test_export_does_not_silently_lower_a_controller_subclass_as_its_parent(tmp_path):
    class ModifiedController(peslite.VSCController):
        def __call__(self, t, measurement):
            raise RuntimeError("replacement controller boundary")

    simulation = peslite.Simulation(_quick("pwm_averaging"),
                                   parts={"vsc.ctrl": ModifiedController})
    destination = tmp_path / "export"
    with pytest.raises(TypeError, match="ModifiedController.*no generator"):
        peslite.export(simulation, "cpp", destination)
    assert not destination.exists()


def test_export_rejects_a_changed_modulator_instead_of_using_its_parent(tmp_path):
    from peslite.components.pwm import CarrierComparison

    class ModifiedCarrier(CarrierComparison):
        def __call__(self, t, span, duties, theta=None, omega=None):
            return super().__call__(t, span, 1.0 - duties, theta, omega)

    simulation = peslite.Simulation(_quick("switching"), parts={
        "vsc.modulator": lambda cfg: ModifiedCarrier(cfg.pwm.f_sw),
    })
    destination = tmp_path / "export"
    with pytest.raises(TypeError, match="modulator.*no native adapter"):
        peslite.export(simulation, "cpp", destination)
    assert not destination.exists()


@pytest.mark.parametrize("minimum_interval", [0.0, 2e-6, 20e-6])
def test_native_adapters_accept_inherited_behavior_and_read_modulator_settings(minimum_interval, tmp_path):
    from peslite.components.pwm import CarrierComparison
    from peslite.solver.integrators import FixedStepSolver

    class InheritedController(peslite.VSCController):
        def describe(self):
            return 'Presentation-only controller description'

    class InheritedCarrier(CarrierComparison):
        pass

    class InheritedSolver(FixedStepSolver):
        pass

    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = _quick("switching").replace(**{
        "simulation.t_end": 2e-4,
        "events.connect_vsc.t": 0.0,
        "events.connect_vsc.ramp": 0.0,
    })
    simulation = peslite.Simulation(params, parts={
        "vsc.ctrl": InheritedController,
        "vsc.modulator": lambda cfg: InheritedCarrier(cfg.pwm.f_sw, min_interval=minimum_interval),
        "solver": lambda cfg: InheritedSolver(cfg.simulation.solver.dt, cfg.simulation.solver.method),
    })
    destination = tmp_path / "inherited"
    peslite.export(simulation, "cpp", destination)
    executable = destination / "native"
    subprocess.run([compiler, "-O3", "-std=c++17", str(destination / "peslite.cpp"),
                    "-o", str(executable)], check=True)
    subprocess.run([str(executable), str(destination / "output")], check=True)
    expected = simulation.run(out_dir=tmp_path / "python").final_states()
    actual = _final_csv(destination / "output/states.csv")
    np.testing.assert_allclose([actual[key] for key in expected], list(expected.values()),
                               rtol=2e-9, atol=2e-8)


@pytest.mark.parametrize('location', ['instance', 'class'])
def test_native_dp45_uses_coefficients_from_the_actual_solver(location, tmp_path):
    from peslite.solver.integrators import DormandPrince45

    compiler = _compiler()
    if compiler is None:
        pytest.skip('no C++17 compiler installed')

    class TunedDP45(DormandPrince45):
        _e = tuple(16.0 * value for value in DormandPrince45._e)

    def solver_factory(cfg):
        settings = cfg.simulation.solver
        cls = TunedDP45 if location == 'class' else DormandPrince45
        solver = cls(settings.rtol, settings.atol, settings.max_step)
        if location == 'instance':
            solver._e = TunedDP45._e
        return solver

    params = _quick('averaging').replace(**{
        'simulation.t_end': 0.004,
        'events.connect_vsc.t': 0.0,
        'events.connect_vsc.ramp': 0.0,
        'simulation.solver.linearisations': 0,
    })
    simulation = peslite.Simulation(params, parts={'solver': solver_factory})
    project = tmp_path / location
    peslite.export(simulation, 'cpp', project)
    executable = project / 'native'
    subprocess.run([compiler, '-O3', '-std=c++17', str(project / 'peslite.cpp'),
                    '-o', str(executable)], check=True)
    subprocess.run([str(executable), str(project / 'output')], check=True)
    result = simulation.run(out_dir=tmp_path / 'python')
    actual, expected = _final_csv(project / 'output/states.csv'), result.final_states()
    np.testing.assert_allclose([actual[key] for key in expected], list(expected.values()),
                               rtol=2e-9, atol=2e-8)
    assert _summary(project / 'output')['n_rhs'] == result.n_rhs


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_fixed_solver_lowering_uses_the_shared_tableau(method, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = _quick("pwm_averaging").replace(**{
        "simulation.solver.method": method,
        "simulation.solver.dt": 2e-6,
        "simulation.t_end": 0.2002,
    })
    _source, native, python = _run_generated(params, tmp_path / method, compiler)
    actual, expected = _final_csv(native / "states.csv"), _final_csv(python / "states.csv")
    np.testing.assert_allclose(list(actual.values()), list(expected.values()),
                               rtol=2e-9, atol=2e-8)


def test_native_continuous_control_feedback_checks_convergence(tmp_path):
    from test_control import _AffineFeedback  # noqa: F401 - Register the feedback loop used below.
    from peslite.assembly.params import read_tree

    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    tree = read_tree(EXAMPLES / "gfl-example.pes")
    tree["units"]["vsc"]["bridge"] = {"model": "averaging"}
    tree["units"]["vsc"]["ctrl"] = {
        "type": "custom",
        "loops": {"_AffineFeedback[a]": {"gain": 0.25, "bias": 1.0},
                  "_AffineFeedback[b]": {"gain": 0.5, "bias": 2.0}},
        "connections": {"_AffineFeedback[a].x": "_AffineFeedback[b].y",
                        "_AffineFeedback[b].x": "_AffineFeedback[a].y"},
        "outputs": {"u_dq": "_AffineFeedback[a].y", "theta": "references.theta",
                    "omega": "references.omega"},
    }
    tree["simulation"].update(t_end=1e-4, energy_check="off")
    tree["simulation"]["solver"].update(type="fixed", method="rk4", dt=1e-6,
                                        subsystems={}, linearisations=0)
    source, native, python = _run_generated(from_dict(tree), tmp_path / "feedback", compiler)
    assert "if (!control_converged) throw" in source.read_text()
    actual, expected = _final_csv(native / "states.csv"), _final_csv(python / "states.csv")
    np.testing.assert_allclose(list(actual.values()), list(expected.values()),
                               rtol=2e-9, atol=2e-8)


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
@pytest.mark.parametrize("link", ["ss", "lcc"])
def test_native_named_load_step_and_rms_match_python(mode, link, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    tree = load_case(mode, link=link)
    tree["simulation"]["output"].update(states=True, signals=True, energy=True)
    for side in ("primary", "secondary"):
        rms = tree["units"]["charger"][side]["meas"]["rms"]
        rms.update(voltage=rms["current"], power=rms["current"])
    if link == "lcc":
        tree["units"]["charger"]["secondary"]["meas"].update(
            sources={"power": "load.power"}, average=["power"],
        )
        # An independent ADC clock publishes its held conversions next to the RMS signals.
        tree["units"]["charger"]["primary"]["meas"]["offset"] = 0.25e-6
    tree["events"] = {"step": {"type": "set", "t": 15e-6, "set": {"elements.load.r": 3.0}}}
    source, native, python = _run_generated(from_dict(tree), tmp_path / "loaded", compiler)
    assert "_rocof_window;" not in source.read_text()
    for filename in ("states.csv", "plant.csv", "energy.csv"):
        actual = np.genfromtxt(native / filename, delimiter=",", names=True)
        expected = np.genfromtxt(python / filename, delimiter=",", names=True)
        assert set(actual.dtype.names) == set(expected.dtype.names)
        for column in expected.dtype.names:
            np.testing.assert_allclose(actual[column], expected[column],
                                       rtol=2e-6, atol=2e-6, equal_nan=True,
                                       err_msg=f"{filename}:{column}")


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging"])
def test_native_initialization_reuses_same_time_held_input_refresh(mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    tree = load_case(mode, rms=False)
    tree["simulation"].update(t_end=10e-6, initial={"t": 0.4e-6, "states": {
        "charger.primary.pwm.on": True,
        "charger.primary.pwm.d_left": 0.8, "charger.primary.pwm.d_right": 0.2,
        "charger.secondary.pwm.on": True,
        "charger.secondary.pwm.d_left": 0.2, "charger.secondary.pwm.d_right": 0.8,
    }})
    tree["simulation"]["output"].update(states=True, signals=True, record_every=1)
    for side in ("primary", "secondary"):
        tree["units"]["charger"][side]["meas"] = {"average": ["voltage"]}
    _, native, python = _run_generated(from_dict(tree), tmp_path / "initial", compiler)
    for side in ("primary", "secondary"):
        filename = f"ctrl.charger.{side}.csv"
        a = np.genfromtxt(native / filename, delimiter=",", names=True)
        b = np.genfromtxt(python / filename, delimiter=",", names=True)
        assert a.dtype.names == b.dtype.names
        for name in a.dtype.names:
            np.testing.assert_allclose(a[name], b[name], rtol=2e-6, atol=2e-6,
                                       equal_nan=True, err_msg=f"{filename}:{name}")


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging", "mixed"])
@pytest.mark.parametrize("side", ["primary", "secondary"])
def test_native_rms_trip_matches_python_on_either_independent_side(mode, side, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    tree = load_case("pwm_averaging" if mode == "mixed" else mode)
    if mode == "mixed":
        tree["units"]["charger"]["primary"]["bridge"]["model"] = "averaging"
    tree["simulation"]["output"].update(states=True, signals=True)
    tree["units"]["charger"][side]["protection"] = {
        "rms_current": {"enable": True, "limit_pu": 0.001},
        "hold": 4.5e-6 if mode == "mixed" else 0.0,
    }
    _, native, python = _run_generated(from_dict(tree), tmp_path / "trip", compiler)
    a, b = _summary(native), _summary(python)
    for field in ("trip_cause", "trip_time", "tripped", "rms_current_first_t", "alarms"):
        key = f"charger.{side}.{field}"
        assert a[key] == b[key]


def test_native_loaded_example_regulates_power_across_the_load_step(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / "wpt-load-example.pes")
    _, native, python = _run_generated(params, tmp_path / "load-example", compiler)
    for output in (native, python):
        with (output / "plant.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for start, end in ((0.0025, 0.003), (0.0055, 0.006)):
            powers = [float(row["load.power"]) for row in rows if start <= float(row["t"]) < end]
            assert sum(powers) / len(powers) == pytest.approx(500.0, rel=0.02)
        summary = _summary(output)
        assert not summary["charger.primary.tripped"]
        assert not summary["charger.secondary.tripped"]


def test_native_power_measurement_can_select_another_named_port(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    tree = load_case("averaging", link="lcc")
    tree["simulation"]["output"].update(states=True, signals=True)
    tree["units"]["charger"]["secondary"]["meas"]["sources"] = {
        "power": "charger.primary.dc.power",
    }
    _, native, python = _run_generated(from_dict(tree), tmp_path / "port", compiler)
    a, b = _final_csv(native / "states.csv"), _final_csv(python / "states.csv")
    np.testing.assert_allclose(list(a.values()), list(b.values()), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("mode, method", [("switching", "spwm"), ("pwm_averaging", "spwm"),
                                          ("averaging", "spwm"), ("switching", "svpwm")])
def test_native_vsc_uses_the_same_measurement_binding_and_rms(mode, method, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / "gfl-example.pes").replace(**{
        "units.vsc.bridge.model": mode,
        "units.vsc.pwm.method": method,
        "units.vsc.meas.sources": {"u_g": "grid.emf.e_g", "i_c": "vsc.branch_f.i"},
        "units.vsc.meas.rms": {"i_c": 1e-3},
        "units.vsc.protection.overcurrent.enable": False,
        "units.vsc.protection.undervoltage.enable": False,
        "units.vsc.protection.overvoltage.enable": False,
        "events.connect_vsc.t": 0.0,
        "events.connect_vsc.ramp": 0.0,
        "simulation.t_end": 1.5e-3,
        "simulation.solver.dt": 2e-6,
        "simulation.solver.linearisations": 0,
        "simulation.solver.subsystems": {},
        "simulation.output.period": 0.1e-3,
        "simulation.output.signals": True,
    })
    _, native, python = _run_generated(params, tmp_path / "vsc-rms", compiler)
    for filename in ("states.csv", "plant.csv"):
        a, b = _final_csv(native / filename), _final_csv(python / filename)
        assert a.keys() == b.keys()
        np.testing.assert_allclose(list(a.values()), list(b.values()), rtol=2e-6, atol=2e-6)


def _quick_gfm(mode: str):
    solver = ("adaptive", "DP45") if mode == "averaging" else ("fixed", "rk4")
    return peslite.load(EXAMPLES / "gfm-psc-example.pes").replace(**{
        "units.vsc.bridge.model": mode,
        "units.vsc.ctrl.loops.PSC.power_filter.enable": 1,
        "units.vsc.ctrl.loops.PSC.power_filter.bandwidth": 5.0,
        "events.connect_vsc.t": 0.0,
        "events.connect_vsc.ramp": 0.0,
        "simulation.solver.type": solver[0],
        "simulation.solver.method": solver[1],
        "simulation.solver.subsystems": {},
        "simulation.t_end": 0.004,
        "simulation.output.period": 1e-3,
        "simulation.output.signals": True,
        "simulation.energy_check": "off",
    })


def test_python_export_api_uses_export_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = peslite.export(_quick("pwm_averaging"), "cpp")
    assert result.out_dir == Path("export/run")
    assert tuple(path.name for path in result.files) == ("peslite.cpp",)
    assert tuple(path.name for path in result.out_dir.iterdir()) == ("peslite.cpp",)
    assert "Standalone C++17" in (result.out_dir / "peslite.cpp").read_text()


def test_cli_export_preserves_mode_name_and_emits_one_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert peslite.main([str(EXAMPLES / "gfl-example.pes"), "--averaging",
                         "--export", "cpp"]) == 0
    project = tmp_path / "export/gfl-example-averaging"
    source = project / "peslite.cpp"
    assert tuple(project.iterdir()) == (source,)
    text = source.read_text()
    assert "void integrate(double& t, double target)" in text
    assert "integrate_fixed" not in text
    assert "integrate_adaptive" not in text


def test_convert_alias_and_all_variables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert convert_main([str(EXAMPLES / "gfl-example.pes"), "all"]) == 0
    source = (tmp_path / "export/gfl-example/peslite.cpp").read_text()
    assert "param_simulation_t_end" in source
    assert "param_units_vsc_ctrl_references_p_ref_pu" in source
    assert '"__PESLITE_RUNTIME_PARAMETER_1__"' in source
    assert '"__PESLITE_RUNTIME_PARAMETER_10__"' in source


def test_export_rejects_a_non_variable_parameter(tmp_path):
    with pytest.raises(ValueError, match="is not variable"):
        peslite.export(_quick("pwm_averaging"), "cpp", tmp_path,
                       variables=["units.vsc.bridge.model"])


@pytest.mark.parametrize("mode", ("switching", "pwm_averaging", "averaging"))
def test_generated_cpp_runs_all_bridge_modes(mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = _quick(mode).replace(**{"simulation.output.signals": True})
    project = tmp_path / mode
    _source, cpp_out, python_out = _run_generated(params, project, compiler)

    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    assert _summary(cpp_out).keys() == _summary(python_out).keys()
    scale = max(1.0, max(abs(value) for value in expected.values()))
    tolerance = 2e-5 if mode == "averaging" else 2e-8
    assert max(abs(actual[key] - expected[key]) for key in expected) <= tolerance * scale

    python_signals = _final_csv(python_out / "plant.csv")
    cpp_signals = _final_csv(cpp_out / "plant.csv")
    assert python_signals.keys() == cpp_signals.keys()
    assert {f"{path}_{phase}" for path in ("pcc.u", "vsc.u_g", "vsc.i_c") for phase in "abc"} <= python_signals.keys()
    assert not any(name.startswith(("pcc.v_", "vsc.v_", "vsc.i_conv_")) for name in python_signals)
    signal_scale = max(1.0, max(abs(value) for value in python_signals.values()))
    assert max(abs(cpp_signals[key] - python_signals[key]) for key in python_signals) <= tolerance * signal_scale


@pytest.mark.parametrize("mode", ("switching", "pwm_averaging", "averaging"))
def test_generated_cpp_runs_gfm_function_owned_power_filter(mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = _quick_gfm(mode)
    _source, cpp_out, python_out = _run_generated(params, tmp_path / mode, compiler)

    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    scale = max(1.0, max(abs(value) for value in expected.values()))
    tolerance = 2e-5 if mode == "averaging" else 2e-8
    assert max(abs(actual[key] - expected[key]) for key in expected) <= tolerance * scale
    expected = _final_csv(python_out / "ctrl.vsc.csv")
    actual = _final_csv(cpp_out / "ctrl.vsc.csv")
    assert max(abs(actual[key] - expected[key]) for key in ("id_pu", "iq_pu", "vd_pu", "vq_pu")) <= tolerance


@pytest.mark.parametrize("bridge_type,source,reference,load", (
    ("half_bridge", 48.0, 24.0, 6.0),
    ("half_bridge", 48.0, 24.0, 600.0),
    ("full_bridge", 48.0, 12.0, 6.0),
))
@pytest.mark.parametrize("mode", ("switching", "pwm_averaging", "averaging"))
def test_generated_cpp_uses_public_ports_for_scalar_converters(
        bridge_type, source, reference, load, mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    changes = {
        "units.converter.bridge.type": bridge_type,
        "units.converter.bridge.lower_controlled": bridge_type == "full_bridge",
        "units.converter.dclink.vdc_ref": source,
        "units.converter.dclink.source.v": source,
        "elements.load.r": load,
        "units.converter.bridge.model": mode,
        "units.converter.ctrl.references.voltage": reference,
        "simulation.t_end": 0.004,
        "simulation.output.states": True,
        "simulation.output.signals": True,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    }
    if mode == "averaging":
        changes.update({
            "simulation.solver.type": "adaptive",
            "simulation.solver.method": "DP45",
            "simulation.solver.max_step": 1e-4,
        })
    params = peslite.load(EXAMPLES / "buck-example.pes").replace(**changes)
    project = tmp_path / f"{bridge_type}-{mode}"
    source_path, cpp_out, python_out = _run_generated(params, project, compiler)
    source = source_path.read_text(encoding="utf-8")
    has_diode_stage = bridge_type == "half_bridge"
    assert ("class DiodeConduction" in source) is has_diode_stage
    assert ("diode_conduction.advance(t, boundary)" in source) is has_diode_stage

    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    assert _summary(cpp_out).keys() == _summary(python_out).keys()
    scale = max(1.0, max(abs(value) for value in expected.values()))
    # Repeated light-load diode zero crossings are independently root-located by the two
    # runtimes; their accumulated difference remains below six parts per million.
    tolerance = 6e-6 if bridge_type == "half_bridge" and load >= 600.0 else 2e-8
    assert max(abs(actual[key] - expected[key]) for key in expected) <= tolerance * scale


def test_generated_cpp_reuses_common_adc_and_protection_for_dc_converter(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / "buck-example.pes").replace(**{
        "units.converter.meas.average": ["current", "voltage", "vin"],
        "units.converter.meas.window": 50.0e-6,
        "units.converter.protection.overcurrent.enable": 1,
        "units.converter.protection.overcurrent.limit_pu": 0.05,
        "simulation.stop_on_trip": 0,
        "simulation.t_end": 0.001,
        "simulation.output.states": True,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    })
    _source, cpp_out, python_out = _run_generated(
        params, tmp_path / "dc-common-runtime", compiler
    )

    actual = _final_csv(cpp_out / "states.csv")
    expected = _final_csv(python_out / "states.csv")
    assert actual.keys() == expected.keys()
    assert max(abs(actual[key] - expected[key]) for key in actual) < 1e-8
    assert _summary(cpp_out).keys() == _summary(python_out).keys()
    assert _summary(cpp_out)["converter.trip_cause"] == "overcurrent"


def test_generated_cpp_matches_independent_control_adc_and_pwm_offsets(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / "gfl-example.pes").replace(**{
        "units.vsc.ctrl.offset": 10.0e-6,
        "units.vsc.meas.period": 25.0e-6,
        "units.vsc.meas.offset": 5.0e-6,
        "units.vsc.meas.average": ["u_g", "i_c"],
        "units.vsc.meas.window": 25.0e-6,
        "events.connect_vsc.t": 0.0,
        "events.connect_vsc.ramp": 0.5e-3,
        "simulation.t_end": 0.001,
        "simulation.output.period": 50.0e-6,
        "simulation.output.states": True,
        "simulation.output.signals": True,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    })
    _source, cpp_out, python_out = _run_generated(
        params, tmp_path / "independent-clock-offsets", compiler
    )

    for filename in ("states.csv", "plant.csv"):  # plant.csv includes the *_raw_pu logs
        actual = _final_csv(cpp_out / filename)
        expected = _final_csv(python_out / filename)
        assert actual.keys() == expected.keys()
        np.testing.assert_allclose(
            [actual[name] for name in actual],
            [expected[name] for name in expected],
            rtol=2e-8,
            atol=2e-9,
        )


@pytest.mark.parametrize("mode", ("switching", "pwm_averaging", "averaging"))
def test_generated_cpp_matches_python_for_adaptive_dcm(mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / "buck-example.pes").replace(**{
        "units.converter.bridge.model": mode,
        "elements.load.r": 600.0,
        "simulation.solver.type": "adaptive",
        "simulation.solver.method": "DP45",
        "simulation.solver.max_step": 1e-4,
        "simulation.t_end": 0.005,
        "simulation.output.period": 5e-6,
        "simulation.output.states": True,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    })
    source, cpp_out, python_out = _run_generated(
        params, tmp_path / f"adaptive-{mode}", compiler
    )

    assert source.read_text(encoding="utf-8").count("class DiodeConduction") == 1
    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    scale = max(1.0, max(abs(value) for value in expected.values()))
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 2e-8 * scale


@pytest.mark.parametrize(("example", "mode"), (
    ("boost-example.pes", "switching"),
    ("boost-example.pes", "pwm_averaging"),
    ("boost-example.pes", "averaging"),
    ("buck-boost-example.pes", "switching"),
    ("buck-boost-example.pes", "pwm_averaging"),
    ("buck-boost-example.pes", "averaging"),
))
def test_generated_cpp_matches_open_dc_converter_compositions(
        example, mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / example)
    changes = {
        f"units.{name}.bridge.model": mode
        for name in params.units
    }
    changes.update({
        "simulation.solver.type": "adaptive" if mode == "averaging" else "fixed",
        "simulation.solver.method": "DP45" if mode == "averaging" else "rk4",
        "simulation.t_end": 0.002,
        "simulation.output.states": True,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    })
    if mode == "averaging":
        changes["simulation.solver.max_step"] = 1.0e-4
    params = params.replace(**changes)
    project = tmp_path / f"{Path(example).stem}-{mode}"
    _source, cpp_out, python_out = _run_generated(params, project, compiler)

    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    scale = max(1.0, max(abs(value) for value in expected.values()))
    # Independent floating-point step decisions can differ at a DCM boundary while the resulting
    # open-converter trajectories remain within two tenths of a part per million.
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 2e-7 * scale


@pytest.mark.parametrize("mode", ("switching", "pwm_averaging", "averaging"))
def test_generated_cpp_matches_composite_wpt_with_independent_clocks(mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    changes = {
        "units.charger.primary.bridge.model": mode,
        "units.charger.secondary.bridge.model": mode,
        "simulation.t_end": 20.0e-6,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    }
    if mode == "averaging":
        changes.update({
            "simulation.solver.type": "adaptive",
            "simulation.solver.method": "DP45",
            "simulation.solver.max_step": 50.0e-9,
        })
    params = peslite.load(EXAMPLES / "wpt-example.pes").replace(**changes)
    _source, cpp_out, python_out = _run_generated(
        params, tmp_path / f"wpt-{mode}", compiler
    )

    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    scale = max(1.0, max(abs(value) for value in expected.values()))
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 2e-8 * scale
    assert _final_csv(cpp_out / "plant.csv").keys() == _final_csv(
        python_out / "plant.csv"
    ).keys()


@pytest.mark.parametrize("case", ("ss", "lcc"))
@pytest.mark.parametrize("mode", ("switching", "pwm_averaging", "averaging"))
def test_generated_cpp_matches_wpt_power_feedback_and_reference_step(case, mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / f"wpt-{case}-example.pes").replace(**{
        "units.charger.primary.bridge.model": mode,
        "units.charger.secondary.bridge.model": mode,
        "events.power_step.t": 40.0e-6,
        "simulation.t_end": 80.0e-6,
        "simulation.output.period": 1.0e-6,
        "simulation.output.record_every": 1,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    })
    _source, cpp_out, python_out = _run_generated(
        params, tmp_path / f"wpt-power-{case}-{mode}", compiler,
    )
    for filename in ("states.csv", "plant.csv", "ctrl.charger.primary.csv",
                     "ctrl.charger.secondary.csv"):
        with (python_out / filename).open(newline="", encoding="utf-8") as stream:
            expected = list(csv.DictReader(stream))
        with (cpp_out / filename).open(newline="", encoding="utf-8") as stream:
            actual = list(csv.DictReader(stream))
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert left.keys() == right.keys()
            for key in left:
                assert bool(left[key]) == bool(right[key]), (filename, key)
                if left[key]:
                    actual_value, expected_value = float(left[key]), float(right[key])
                    if np.isnan(expected_value):
                        assert np.isnan(actual_value), (filename, key)
                    else:
                        assert actual_value == pytest.approx(
                            expected_value, rel=2e-7, abs=2e-8,
                        ), (filename, key)


@pytest.mark.parametrize("case", ("buck", "boost", "buck-boost"))
@pytest.mark.parametrize("mode", ("switching", "pwm_averaging", "averaging"))
def test_generated_cpp_uses_electrical_port_power_in_scalar_feedback(case, mode, tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / f"{case}-example.pes").replace(**{
        "units.converter.bridge.model": mode,
        "units.converter.meas.average": ["power"],
        "units.converter.ctrl": {
            "type": "resonant",
            "references": {"modulation": 0.5, "frequency": 2000.0, "phase": 0.3,
                           "power": -120.0 if case == "boost" else 120.0},
            "loops": {
                "ResonantModulation": {},
                "PowerControl": {"kp": 0.001, "ki": 0.1, "minimum": 0.0, "maximum": 0.9},
            },
            "connections": {
                "PowerControl.reference": "references.power",
                "PowerControl.power": "meas.power",
                "PowerControl.feedforward": "references.modulation",
                "ResonantModulation.modulation": "PowerControl.modulation",
            },
        },
        "simulation.t_end": 100.0e-6,
        "simulation.solver.type": "fixed",
        "simulation.solver.method": "rk4",
        "simulation.solver.dt": 0.5e-6,
        "simulation.solver.linearisations": 0,
        "simulation.output.period": 5.0e-6,
        "simulation.output.record_every": 1,
        "simulation.output.states": True,
        "simulation.output.signals": True,
        "simulation.output.energy": False,
        "simulation.energy_check": "off",
    })
    source, cpp_out, python_out = _run_generated(
        params, tmp_path / f"port-power-{case}-{mode}", compiler,
    )
    for filename in ("states.csv", "plant.csv", "ctrl.converter.csv"):
        actual = np.genfromtxt(cpp_out / filename, delimiter=",", names=True)
        expected = np.genfromtxt(python_out / filename, delimiter=",", names=True)
        assert actual.dtype.names == expected.dtype.names
        for column in actual.dtype.names:
            scale = max(1.0, float(np.nanmax(np.abs(expected[column]))))
            np.testing.assert_allclose(actual[column], expected[column], rtol=2e-7,
                                       atol=2e-8 * scale, err_msg=f"{filename}:{column}")
    if mode == "averaging":
        # One declaration and one snapshot assignment, with no RHS-time log write.
        assert source.read_text(encoding="utf-8").count("u_converter_log_power =") == 2


def test_generated_cpp_matches_composite_wpt_switching_events(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    tree = yaml.safe_load((EXAMPLES / "wpt-example.pes").read_text(encoding="utf-8"))
    for side in ("primary", "secondary"):
        tree["units"]["charger"][side]["bridge"] = {"model": "pwm_averaging"}
    tree["events"] = {
        "stop": {"type": "disconnect", "target": "charger", "t": 8.0e-6},
        "restart": {
            "type": "connect",
            "target": "charger",
            "t": 12.0e-6,
            "ramp": 2.0e-6,
        },
    }
    tree["simulation"]["t_end"] = 20.0e-6
    tree["simulation"]["energy_check"] = "off"
    tree["simulation"]["output"].update({"states": 1, "energy": 0})
    params = from_dict(tree)
    _source, cpp_out, python_out = _run_generated(
        params, tmp_path / "wpt-events", compiler
    )

    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    scale = max(1.0, max(abs(value) for value in expected.values()))
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 2e-8 * scale


def test_generated_cpp_runs_custom_cable_equations(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = peslite.load(EXAMPLES / "cable-gfl-example.pes", **{
        "simulation.t_end": 0.01,
        "simulation.solver.linearisations": 0,
        "simulation.energy_check": "strict",
        "simulation.output.signals": 1,
    })
    project = tmp_path / "cable"
    _source, cpp_out, python_out = _run_generated(params, project, compiler)

    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 2e-8 * max(
        1.0, max(abs(value) for value in expected.values())
    )
    expected_plant = _final_csv(python_out / "plant.csv")
    actual_plant = _final_csv(cpp_out / "plant.csv")
    assert actual_plant.keys() == expected_plant.keys()
    assert {"cable1.i1_a", "cable1.i2_a"} <= actual_plant.keys()


def test_generated_cpp_runs_black_box_outputs_and_peslite_state_equations(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = from_dict({
        "base": {"s_base": 1.0},
        "elements": {
            "model": {"type": "test_black_box", "decay": 4.0, "initial": 3.0},
        },
        "simulation": {
            "t_end": 0.02,
            "solver": {"type": "fixed", "method": "rk4", "dt": 1.0e-4,
                       "linearisations": 0},
            "output": {"period": 1.0e-3, "signals": True},
            "energy_check": "off",
        },
    })

    source, cpp_out, python_out = _run_generated(params, tmp_path / "black-box", compiler)

    cpp_state = _final_csv(cpp_out / "states.csv")
    python_state = _final_csv(python_out / "states.csv")
    assert cpp_state["model.x"] == pytest.approx(python_state["model.x"], rel=2e-10)
    assert cpp_state["model.x"] == pytest.approx(3.0 * np.exp(-0.08), rel=1e-10)
    cpp_plant = _final_csv(cpp_out / "plant.csv")
    python_plant = _final_csv(python_out / "plant.csv")
    assert cpp_plant["model.y"] == pytest.approx(python_plant["model.y"], rel=2e-10)
    text = source.read_text(encoding="utf-8")
    assert "dy[0]" in text and "model.y" in text


@pytest.mark.parametrize(("initial", "t_end", "expected"), [
    ({}, 2.1e-3, 3.0),
    ({"t": 2.0e-3, "states": {"model.hidden": 7.0}}, 3.1e-3, 8.0),
])
def test_generated_cpp_runs_discrete_black_box_state_updates(tmp_path, initial, t_end, expected):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = from_dict({
        "base": {"s_base": 1.0},
        "elements": {
            "model": {"type": "test_discrete_black_box", "period": 1.0e-3,
                      "increment": 1.0, "initial": 0.0},
        },
        "simulation": {
            "t_end": t_end,
            "initial": initial,
            "solver": {"type": "fixed", "method": "rk4", "dt": 1.0e-4,
                       "linearisations": 0},
            "output": {"period": 1.0e-3, "signals": True},
            "energy_check": "off",
        },
    })

    source, cpp_out, python_out = _run_generated(
        params, tmp_path / f"discrete-black-box-{expected:g}", compiler
    )

    cpp_state = _final_csv(cpp_out / "states.csv")
    python_state = _final_csv(python_out / "states.csv")
    assert cpp_state["model.hidden"] == pytest.approx(expected)
    assert cpp_state["model.hidden"] == pytest.approx(python_state["model.hidden"])
    cpp_plant = _final_csv(cpp_out / "plant.csv")
    python_plant = _final_csv(python_out / "plant.csv")
    assert cpp_plant["model.y"] == pytest.approx(python_plant["model.y"])
    assert "update_discrete" in source.read_text(encoding="utf-8")


def test_generic_cpp_fallback_lowers_custom_equations_and_state_branches(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    tree = yaml.safe_load((EXAMPLES / "gfl-example.pes").read_text(encoding="utf-8"))
    tree["elements"] = {"conditional": {
        "type": "test_conditional_element",
        "bus": "pcc",
        "inductance": 0.01,
        "resistance_low": 10.0,
        "resistance_high": 20.0,
        "threshold": 50.0,
    }}
    tree["simulation"]["t_end"] = 0.002
    tree["simulation"]["solver"]["linearisations"] = 0
    tree["simulation"]["energy_check"] = "off"
    tree["simulation"].setdefault("output", {})["signals"] = 1
    case = tmp_path / "conditional.pes"
    case.write_text(yaml.safe_dump(tree, sort_keys=False), encoding="utf-8")
    params = peslite.load(case)
    project = tmp_path / "conditional"
    source_path, cpp_out, python_out = _run_generated(params, project, compiler)
    source = source_path.read_text(encoding="utf-8")
    assert "std::abs" in source and "?" in source
    assert "20.0" in source and "10.0" in source

    expected = _final_csv(python_out / "states.csv")
    actual = _final_csv(cpp_out / "states.csv")
    assert actual.keys() == expected.keys()
    assert max(abs(actual[key] - expected[key]) for key in expected) <= 2e-8 * max(
        1.0, max(abs(value) for value in expected.values())
    )
    expected_plant = _final_csv(python_out / "plant.csv")
    actual_plant = _final_csv(cpp_out / "plant.csv")
    assert actual_plant.keys() == expected_plant.keys()
    assert "conditional.i_mag" in actual_plant


def test_cpp_export_rejects_an_already_run_simulation(tmp_path):
    simulation = peslite.Simulation(_quick("pwm_averaging"))
    simulation.run(out_dir=tmp_path / "output")
    with pytest.raises(RuntimeError, match="already-run"):
        simulation.export("cpp", tmp_path / "export")


def test_generated_cpp_runtime_parameters(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    params = _quick("pwm_averaging")
    project = tmp_path / "runtime-parameters"
    peslite.export(params, "cpp", project, variables=[
        "simulation.t_end", "units.vsc.ctrl.references.p_ref_pu",
    ])
    executable = project / "peslite"
    subprocess.run([compiler, "-O3", "-DNDEBUG", "-std=c++17",
                    str(project / "peslite.cpp"), "-o", str(executable)], check=True)

    listed = subprocess.run([executable, "--list-params"], check=True,
                            capture_output=True, text=True).stdout
    assert "simulation.t_end" in listed
    assert "units.vsc.ctrl.references.p_ref_pu" in listed

    output = project / "changed-output"
    subprocess.run([executable,
                    "--set", "simulation.t_end=0.204",
                    "--set", "units.vsc.ctrl.references.p_ref_pu=0.25",
                    "--out", output], check=True)
    resolved = peslite.load(output / "simulation.pes")
    assert resolved.simulation.t_end == pytest.approx(0.204)
    assert resolved.unit("vsc").ctrl.references.p_ref_pu == pytest.approx(0.25)

    config = project / "runtime.pes"
    peslite.dump(params.replace(**{
        "simulation.t_end": 0.203,
        "units.vsc.ctrl.references.p_ref_pu": 0.4,
    }), config)
    configured_output = project / "configured-output"
    configured = subprocess.run([
        executable, "--config", config,
        "--set", "simulation.t_end=0.205",
        "--out", configured_output,
    ], capture_output=True, text=True)
    assert configured.returncode == 0
    assert "warning:" in configured.stderr
    configured_params = peslite.load(configured_output / "simulation.pes")
    assert configured_params.simulation.t_end == pytest.approx(0.205)
    assert configured_params.unit("vsc").ctrl.references.p_ref_pu == pytest.approx(0.4)

    rejected = subprocess.run(
        [executable, "--set", "units.vsc.ctrl.references.q_ref_pu=0.2"],
        capture_output=True, text=True,
    )
    assert rejected.returncode == 1
    assert "parameter 'units.vsc.ctrl.references.q_ref_pu' is not variable" in rejected.stderr
