"""Addon manifest, assembly and registration boundaries."""

import importlib
from pathlib import Path

import peslite
import pytest
from peslite.addons import components as addon_components
from peslite.addons import control as addon_control
from peslite.assembly import addons as addon_assembly
from peslite.addons.control.mpc import (PredictiveCurrentControl,
                                        ThreePhasePredictiveCurrentControl)
from peslite.components.network import ELEMENT_TYPES
from peslite.control import LOOP_TYPES, Loop


def test_control_addon_path_registers_loops(tmp_path, monkeypatch):
    path = tmp_path / "control"
    path.mkdir()
    (path / "_private.py").write_text(
        "raise RuntimeError('private add-on helpers must not be discovered')\n",
        encoding="utf-8",
    )
    (path / "test_controller_addon.py").write_text(
        """from dataclasses import dataclass
from peslite.control import Loop, register_loop_type

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

    imported = addon_assembly._import_modules(addon_control.__name__, addon_control.__path__)

    assert [module.__name__ for module in imported] == [
        "peslite.addons.control.test_controller_addon"
    ]
    assert LOOP_TYPES["addon_test_loop"].__module__ == imported[0].__name__


def test_component_addon_path_registers_elements(tmp_path, monkeypatch):
    path = tmp_path / "components"
    path.mkdir()
    (path / "test_component_addon.py").write_text(
        """from dataclasses import dataclass
from peslite.components import register_component_type
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

    imported = addon_assembly._import_modules(
        addon_components.__name__, addon_components.__path__
    )

    assert [module.__name__ for module in imported] == [
        "peslite.addons.components.test_component_addon"
    ]
    assert ELEMENT_TYPES["addon_test_element"].params.__module__ == imported[0].__name__


def test_function_addons_use_their_explicit_modules():
    functions = importlib.import_module("peslite.addons.functions")
    assert not hasattr(peslite.addons, "plot_csv")
    assert not hasattr(functions, "__all__")
    plotting = importlib.import_module("peslite.addons.functions.plotting")
    assert callable(plotting.plot_csv)


def test_simulation_readers_are_declared_and_loaded_through_addon_assembly():
    declared = [name for name, spec in addon_assembly.ADDONS.items()
                if spec.simulation_reader]
    readers = addon_assembly.simulation_readers()

    assert declared == ["legacy_yaml"]
    assert [reader.__module__ for reader in readers] == [
        "peslite.addons.functions.legacy_yaml"
    ]


def test_every_installed_addon_module_is_declared_once():
    root = Path(addon_assembly.__file__).parents[1] / "addons"
    declared = {category: {} for category in ("control", "components", "functions")}
    for name, spec in addon_assembly.ADDONS.items():
        for field, category in (("control", "control"), ("component", "components"),
                                ("function", "functions")):
            module = spec.module(field)
            if module is not None:
                assert module == name
                assert module not in declared[category]
                declared[category][module] = name
    for category, modules in declared.items():
        files = {
            path.stem for path in (root / category).glob("*.py")
            if path.stem != "__init__" and not path.stem.startswith("_")
        }
        assert files == set(modules)
        assert not any(path.is_dir() for path in (root / category).iterdir())


def test_missing_addon_module_has_one_assembly_error(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(addon_assembly, "import_module", missing)
    with pytest.raises(addon_assembly.AddonUnavailableError, match="plotting"):
        addon_assembly.import_addon("plotting", "function")


def test_mpc_is_a_control_addon():
    assert LOOP_TYPES[PredictiveCurrentControl.type] is PredictiveCurrentControl
    assert (
        LOOP_TYPES[ThreePhasePredictiveCurrentControl.type]
        is ThreePhasePredictiveCurrentControl
    )
    assert "CCSMPC" not in peslite.control.__all__
    assert "PredictiveCurrentControl" not in peslite.control.__all__


def test_addon_namespaces_do_not_duplicate_builtin_public_apis():
    assert not hasattr(addon_control, "__all__")
    assert not hasattr(addon_components, "__all__")
    assert not hasattr(addon_control, "LOOP_TYPES")
    assert not hasattr(addon_components, "register_component_type")
    assert not hasattr(addon_control, "discover")
    assert not hasattr(addon_components, "discover")


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
from peslite.control import (ANGLE, FREQUENCY, V_AB, Integrator, Loop,
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
from peslite.components import register_component_type
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
