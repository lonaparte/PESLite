"""Shared cooling and external thermalport models."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from peslite.addons.components.thermal import (
    LumpedThermalNetwork, SharedThermalModel, SharedThermalNetwork, ThermalLink,
)
from peslite.addons.components.thermalport import (
    ThermalPort, ThermalPortModel,
)
from peslite.addons.functions import foster_step_response, load_thermalport
from peslite.solver.model import Model


def _thermalport_file(path, *, A=None, B=None, C=None, D=0.0, ambient=300.0):
    data = {
        "format": "peslite-thermalport",
        "version": 1,
        "units": "SI",
        "heat_names": ["source"],
        "temperature_names": ["junction"],
        "A_per_s": [[-1.0]] if A is None else A,
        "B_K_per_J": [[1.0]] if B is None else B,
        "C": [[1.0]] if C is None else C,
        "D_K_W": [[D]],
        "ambient_K": [ambient],
        "initial": [0.0],
        "signals": {},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


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


def test_named_thermalport_replaces_a_multi_source_network(tmp_path):
    file = tmp_path / "shared.json"
    file.write_text(json.dumps({
        "format": "peslite-thermalport",
        "version": 1,
        "units": "SI",
        "heat_names": ["electric", "magnetic"],
        "temperature_names": ["junction", "sink"],
        "A_per_s": [[-1.0, 1.0], [0.5, -0.75]],
        "B_K_per_J": [[1.0, 0.0], [0.0, 0.5]],
        "C": [[1.0, 0.0], [0.0, 1.0]],
        "D_K_W": [[0.0, 0.0], [0.0, 0.0]],
        "ambient_K": [300.0, 300.0],
        "initial": [0.0, 0.0],
        "signals": {
            "cooling_W": {
                "state": [0.0, 0.5],
                "heat": [0.0, 0.0],
                "offset": 0.0,
            },
        },
    }), encoding="utf-8")
    cfg = ThermalPort.Params(
        sources={"electric": "load.power", "magnetic": "magnet.loss_W"},
        file=str(file),
    )
    element = ThermalPort("thermal", cfg, {}, None)
    model = element.model
    model.input.heat_electric_W = 10.0
    model.input.heat_magnetic_W = 4.0
    assert model.rhs(0.0) == pytest.approx((10.0, 2.0))
    model.state.state_0 = 20.0
    model.state.state_1 = 10.0
    model.set_outputs(0.0)
    assert model.output.temperature_junction_K == pytest.approx(320.0)
    assert model.output.temperature_sink_K == pytest.approx(310.0)
    assert model.output.heat_total_W == pytest.approx(14.0)
    assert model.output.cooling_W == pytest.approx(5.0)
    assert element.connections() == {
        (model, "heat_electric_W"): "load.power",
        (model, "heat_magnetic_W"): "magnet.loss_W",
    }


def test_thermalport_matches_first_order_analytic_response(tmp_path):
    # dx/dt = -x/(R*C) + P/C; T = ambient + x
    R, C, P = 4.0, 2.0, 10.0
    model = load_thermalport(_thermalport_file(
        tmp_path / "thermal.json",
        A=[[-1.0 / (R * C)]],
        B=[[1.0 / C]],
        C=[[1.0]],
    ))
    model.input.heat_source_W = P
    model.set_outputs(0)
    assert model.rhs(0) == pytest.approx([P / C])
    state = Model({"rom": model}, {})
    assert state.state_labels() == ["rom.state_0"]
    t = 2.0
    exact = 300 + R * P * (1 - np.exp(-t / (R * C)))
    assert exact == pytest.approx(foster_step_response(
        [t], P, [R], [C], ambient_K=300)[0])


def test_thermal_port_uses_an_independent_constitutive_law():
    class Law:
        heat_names = ("source",)
        temperature_names = ("junction",)
        signal_names = ()
        initial_state = {"temperature_K": 300.0}

        @staticmethod
        def rhs(_t, state, heat_W):
            return np.array([heat_W["source"] - (state[0] - 300.0)])

        @staticmethod
        def temperatures_K(_t, state, _heat_W):
            return {"junction": state[0]}

        @staticmethod
        def signals(_t, _state, _heat_W):
            return {}

    model = ThermalPortModel(Law())
    model.input.heat_source_W = 5.0
    model.set_outputs(0.0)
    assert model.output.temperature_junction_K == pytest.approx(300.0)
    assert model.rhs(0.0) == pytest.approx((5.0,))


def test_thermal_port_allows_a_stateless_constitutive_law():
    class Law:
        heat_names = ("source",)
        temperature_names = ("junction",)
        initial_state = {}

        @staticmethod
        def rhs(_t, _state, _heat_W):
            return np.empty(0)

        @staticmethod
        def temperatures_K(_t, _state, heat_W):
            return {"junction": 300.0 + 0.5 * heat_W["source"]}

    model = ThermalPortModel(Law())
    model.input.heat_source_W = 4.0
    circuit = Model({"thermal": model}, {})
    circuit.sync(0.0, np.empty(0))
    assert model.output.temperature_junction_K == pytest.approx(302.0)
    assert circuit.rhs(0.0, np.empty(0)).size == 0


def test_thermalport_requires_declared_si_and_stable_matrices(tmp_path):
    file = _thermalport_file(tmp_path / "thermal.json")
    imported = load_thermalport(file)
    assert imported.rhs(0) == (0.0,)
    assert isinstance(imported, ThermalPortModel)
    data = json.loads(file.read_text(encoding="utf-8"))
    data["units"] = "imperial"
    file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="format, version, and SI units"):
        load_thermalport(file)
    data["units"] = "SI"
    data["A_per_s"] = [[0.0]]
    file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="strictly stable"):
        load_thermalport(file)


def test_thermalport_consumes_a_physical_heat_source(tmp_path):
    config = ThermalPort.Params(
        sources={"source": "load.power"},
        file=str(_thermalport_file(tmp_path / "thermal.json")),
    )
    element = ThermalPort("thermal", config, {}, None)
    assert element.connections() == {(element.model, "heat_source_W"): "load.power"}
    assert element.subsystems() == {"thermal": element.model}
    assert element.signals()["thermal.temperature_junction_K"] == pytest.approx(300.0)


def test_thermal_component_types_share_the_component_registry():
    import peslite

    # Standard and full addon types use the same component registry.
    from peslite.components.network import ELEMENT_TYPES
    for name in ("shared_thermal_network", "thermalport"):
        assert name in ELEMENT_TYPES
    assert callable(peslite.Simulation)


def test_magnetic_and_thermal_ports_couple_without_owning_each_other(tmp_path):
    import peslite
    from peslite.assembly.params import from_dict

    model_file = _thermalport_file(tmp_path / "thermal.json")
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
                "temperature_source": "thermal.temperature_junction_K",
                "temperature_coefficient_per_K": 0.004,
                "temperature_reference_K": 300.0,
            },
            "load": {"type": "resistor", "bus": "node", "r": 1.0},
            "thermal": {
                "type": "thermalport",
                "sources": {"source": "magnet.loss_W"},
                "file": str(model_file),
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
    assert result.plant["thermal.temperature_junction_K"][-1] > 300.0
