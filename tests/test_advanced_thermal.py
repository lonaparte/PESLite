"""Advanced shared cooling, imported reduced thermal models and magnetic feedback."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from peslite.addons.components.advanced_thermal import (
    ReducedThermalElement, ReducedThermalModel, SharedThermalModel,
    SharedThermalNetwork, ThermalPortModel, foster_step_response,
    load_thermal_rom,
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


def test_thermal_port_uses_an_independent_constitutive_law():
    class Law:
        initial_state = {"temperature_K": 300.0}

        @staticmethod
        def rhs(_t, state, heat_W):
            return np.array([heat_W - (state[0] - 300.0)])

        @staticmethod
        def temperature_K(_t, state, _heat_W):
            return state[0]

    model = ThermalPortModel(Law())
    model.input.heat_W = 5.0
    model.set_outputs(0.0)
    assert model.output.temperature_K == pytest.approx(300.0)
    assert model.rhs(0.0) == pytest.approx((5.0,))


def test_thermal_port_allows_a_stateless_constitutive_law():
    class Law:
        initial_state = {}

        @staticmethod
        def rhs(_t, _state, _heat_W):
            return np.empty(0)

        @staticmethod
        def temperature_K(_t, _state, heat_W):
            return 300.0 + 0.5 * heat_W

    model = ThermalPortModel(Law())
    model.input.heat_W = 4.0
    circuit = Model({"thermal": model}, {})
    circuit.sync(0.0, np.empty(0))
    assert model.output.temperature_K == pytest.approx(302.0)
    assert circuit.rhs(0.0, np.empty(0)).size == 0


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
    assert element.signals()["rom.temperature_K"] == pytest.approx(300.0)


def test_shared_sink_and_rom_do_not_change_core_only_electrical_install():
    import peslite
    from peslite.assembly.params import from_dict

    # All optional component types register through the shared element registry.
    from peslite.components.network import ELEMENT_TYPES
    for name in ("shared_thermal_network", "thermal_rom"):
        assert name in ELEMENT_TYPES
    assert callable(peslite.Simulation)


def test_magnetic_and_thermal_ports_couple_without_owning_each_other(tmp_path):
    import peslite
    from peslite.assembly.params import from_dict

    case = {
        "base": {"s_base": 1.0},
        "buses": {"node": {"phases": 1}},
        "elements": {
            "magnet": {
                "type": "coupled_coils",
                "windings": {
                    "coil": {"positive": "node", "negative": "ground", "r": 1.0},
                },
                "inductance": [[0.01]],
                "temperature_source": "thermal.temperature_K",
                "temperature_coefficient_per_K": 0.004,
                "temperature_reference_K": 300.0,
            },
            "load": {"type": "resistor", "bus": "node", "r": 1.0},
            "thermal": {
                "type": "thermal_rom",
                "heat_source": "magnet.loss_W",
                "A_per_s": [[-1.0]],
                "B_K_per_J": [1.0],
                "C": [1.0],
                "D_K_W": 0.0,
                "ambient_K": 300.0,
            },
        },
        "simulation": {
            "t_end": 1.0e-3,
            "initial": {"states": {"magnet.i_mode_1": 1.0}},
            "solver": {"dt": 1.0e-5, "linearisations": 0},
            "output": {"period": 1.0e-4, "precision": 17, "signals": True},
        },
    }
    result = peslite.Simulation(from_dict(case)).run(out_dir=tmp_path)
    assert np.max(result.plant["magnet.loss_W"]) > 0.0
    assert result.plant["thermal.temperature_K"][-1] > 300.0
