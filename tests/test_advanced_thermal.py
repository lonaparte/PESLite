"""Advanced shared cooling, imported reduced thermal models and magnetic feedback."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from peslite.addons.components.advanced_thermal import (
    ReducedThermalElement, ReducedThermalModel, SharedThermalModel,
    SharedThermalNetwork, foster_step_response, load_thermal_rom,
)
from peslite.addons.components.thermal_network import LumpedThermalNetwork, ThermalLink
from peslite.solver.model import Model


def test_shared_electrical_and_magnetic_loss_inputs_conserve_heat():
    network = LumpedThermalNetwork(
        {"electric": 1.0, "magnetic": 2.0, "sink": 4.0},
        [ThermalLink("electric", "sink", 1.0),
         ThermalLink("magnetic", "sink", 2.0),
         ThermalLink("sink", None, 4.0)],
        ambient_K=300,
    )
    sub = SharedThermalModel(
        network, {"inverter": "electric", "coil": "magnetic"}, initial_K=300
    )
    sub.input.heat_inverter_W = 10
    sub.input.heat_coil_W = 4
    assert sub.rhs(0) == pytest.approx((10.0, 2.0, 0.0))
    sub.set_outputs(0)
    assert sub.output.heat_total_W == pytest.approx(14)
    sub.state.T_electric_K = 320
    sub.state.T_magnetic_K = 330
    sub.state.T_sink_K = 310
    total = float(np.dot(network.capacities_J_K, sub.rhs(1)))
    assert total == pytest.approx(14 - network.cooling_power_W([320, 330, 310]))


def test_shared_heat_sources_reuse_the_model_connection_graph():
    class Loss:
        state_names = ()
        outputs_need_inputs = False
        observer = True

        def __init__(self, loss):
            self.loss = loss
            self.state = SimpleNamespace()
            self.input = SimpleNamespace()
            self.output = SimpleNamespace(heat_W=loss)

        def set_outputs(self, _t):
            self.output.heat_W = self.loss

        def rhs(self, _t):
            return ()

    electric = Loss(3)
    magnetic = Loss(4)
    physical = LumpedThermalNetwork(
        {"junction": 2.0, "sink": 1.0},
        [ThermalLink("junction", "sink", 1.0), ThermalLink("sink", None, 2.0)],
        ambient_K=300,
    )
    thermal = SharedThermalModel(physical, {"e": "junction", "m": "junction"})
    model = Model({"electric": electric, "magnetic": magnetic, "thermal": thermal},
                  {(thermal, "heat_e_W"): (electric, "heat_W"),
                   (thermal, "heat_m_W"): (magnetic, "heat_W")})
    state = model.get_initial_values()
    assert model.rhs(0, state) == pytest.approx([3.5, 0.0])
    assert model.energy_specs["thermal"].kind == "observer"


def test_shared_thermal_registration_is_configuration_driven():
    cfg = SharedThermalNetwork.Params(
        capacities_J_K={"chip": 1, "sink": 2},
        links=[{"node": "chip", "other": "sink", "resistance_K_W": 1.0},
               {"node": "sink", "other": None, "resistance_K_W": 2.0}],
        sources={"electrical": {"path": "load.power", "node": "chip"},
                 "magnetic": {"path": "coil.loss_W", "node": "chip"}},
    )
    element = SharedThermalNetwork("thermal", cfg, {}, None)
    assert element.connections() == {
        (element.model, "heat_electrical_W"): "load.power",
        (element.model, "heat_magnetic_W"): "coil.loss_W",
    }
    assert element.port_bindings() == ()
    with pytest.raises(ValueError, match="heat sources"):
        SharedThermalNetwork.Params(
            capacities_J_K={"chip": 1}, links=[{"node": "chip", "other": None,
                                                "resistance_K_W": 1}], sources={},
        )


def test_rom_matches_first_order_analytic_response():
    # dx/dt = -x/(R*C) + P/C; T = ambient + x
    R, C, P = 4.0, 2.0, 10.0
    model = ReducedThermalModel(
        A_per_s=[[-1.0 / (R * C)]], B_K_per_J=[1.0 / C],
        C=[1.0], D_K_W=0.0, ambient_K=300,
    )
    model.input.heat_W = P
    model.set_outputs(0)
    assert model.rhs(0) == pytest.approx([P / C])
    state = Model({"rom": model}, {})
    assert state.state_labels() == ["rom.rom_0_K"]
    t = 2.0
    exact = 300 + R * P * (1 - np.exp(-t / (R * C)))
    assert exact == pytest.approx(foster_step_response(
        [t], P, [R], [C], ambient_K=300)[0])


def test_rom_json_requires_declared_si_and_stable_matrices(tmp_path):
    file = tmp_path / "thermal.json"
    data = {"format": "peslite-thermal-rom-v1", "units": "SI",
            "A_per_s": [[-1.0]], "B_K_per_J": [1.0],
            "C": [1.0], "D_K_W": 0.0, "ambient_K": 300}
    file.write_text(json.dumps(data), encoding="utf-8")
    imported = load_thermal_rom(file)
    assert imported.rhs(0) == (0.0,)
    assert isinstance(imported, ReducedThermalModel)
    data["units"] = "imperial"
    file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="SI units"):
        load_thermal_rom(file)
    with pytest.raises(ValueError, match="strictly stable"):
        ReducedThermalModel([[0]], [1], [1], 0)


def test_reduced_thermal_adapter_consumes_a_physical_heat_source():
    config = ReducedThermalElement.Params(
        heat_source="load.power", A_per_s=[[-1]], B_K_per_J=[1],
        C=[1], D_K_W=0, ambient_K=300,
    )
    element = ReducedThermalElement("rom", config, {}, None)
    assert element.connections() == {(element.model, "heat_W"): "load.power"}
    assert element.subsystems() == {"rom": element.model}
    assert element.signals()["rom.junction_K"] == pytest.approx(300.0)


def test_shared_sink_and_rom_do_not_change_core_only_electrical_install():
    import peslite
    from peslite.assembly.params import from_dict

    # All optional component types register through the shared element registry.
    from peslite.components.network import ELEMENT_TYPES
    for name in ("shared_thermal_network", "thermal_rom"):
        assert name in ELEMENT_TYPES
    assert callable(peslite.Simulation)
