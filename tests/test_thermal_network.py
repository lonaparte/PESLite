"""Analytical RC references, restartable thermal states and circuit attachments (Issue #40)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from peslite.addons.components.electrothermal import ElectroThermal, ElectroThermalModel
from peslite.addons.components.thermal_network import (
    FosterImpedance, LumpedThermalNetwork, ThermalLink,
)
from peslite.addons.functions.semiconductor_losses import (
    LossTable, SemiconductorLossModel, SwitchingEvent,
)
from peslite.addons.functions.thermal_analysis import (
    thermal_report, compare_thermal_reports,
)
from peslite.solver.model import Model


def test_single_node_cauer_matches_analytical_step_and_energy_balance():
    network = LumpedThermalNetwork.cauer([2.0], [4.0], ambient_K=300.0)
    # d(T-300)/dt = (P-(T-300)/R)/C, with P=10 and RC=8 s.
    for t in (0.0, 0.5, 4.0, 16.0):
        temperature = 300.0 + 40.0 * (1.0 - math.exp(-t / 8.0))
        assert network.rates([temperature], {"node_0": 10.0})[0] == pytest.approx(
            5.0 * math.exp(-t / 8.0))
    initial = np.array([320.0])
    derivative = network.rates(initial, {"node_0": 10.0})[0]
    assert 2.0 * derivative == pytest.approx(
        10.0 - network.cooling_power_W(initial))
    assert network.stored_excess_heat_J(initial) == pytest.approx(40.0)


def test_shared_heatsink_conserves_total_energy_and_rejects_invalid_graphs():
    network = LumpedThermalNetwork(
        {"junction_a": 1.0, "junction_b": 2.0, "heatsink": 10.0},
        [ThermalLink("junction_a", "heatsink", 1),
         ThermalLink("junction_b", "heatsink", 2),
         ThermalLink("heatsink", None, 5)],
        ambient_K=300,
    )
    state = np.array([320.0, 330.0, 310.0])
    derivative = network.rates(state, {"junction_a": 10, "junction_b": 5})
    assert float(np.dot(network.capacities_J_K, derivative)) == pytest.approx(
        15 - network.cooling_power_W(state))
    heated = network.committed_impulse(state, {"junction_a": 2.0})
    assert heated[0] == pytest.approx(322)
    assert network.stored_excess_heat_J(heated) - network.stored_excess_heat_J(state) == pytest.approx(2)
    with pytest.raises(ValueError, match="path to ambient"):
        LumpedThermalNetwork({"a": 1, "b": 1}, [ThermalLink("a", None, 1)])
    with pytest.raises(ValueError, match="at least one link to ambient"):
        LumpedThermalNetwork({"a": 1}, [])


def test_foster_step_response_and_restricted_temperature_outputs():
    network = FosterImpedance([1.0, 2.0], [2.0, 4.0], ambient_K=300)
    assert network.rates([0, 0], 10) == pytest.approx([5, 2.5])
    assert network.junction_K([10, 20]) == pytest.approx(330)
    assert network.committed_impulse([0, 0], 2) == pytest.approx([1, 0.5])
    component = ElectroThermalModel(network)
    assert math.isnan(component.output.case_K)
    assert component.state_names == ("rise_0_K", "rise_1_K", "last_transition_id")


def test_temperature_dependent_loss_updates_model_rhs_without_thermal_solver():
    table = LossTable({"temperature_K": [300, 400]}, [1, 2])
    model = SemiconductorLossModel(conduction_voltage_V=table)
    component = ElectroThermalModel(
        LumpedThermalNetwork.cauer([1, 1], [1, 1], ambient_K=300),
        loss_model=model, initial_temperature_K=300,
    )
    component.input.current_A = 10
    assert component.output.estimated_heat_W == pytest.approx(0)
    component.set_outputs(0)
    assert component.output.estimated_heat_W == pytest.approx(10)
    derivative = component.rhs(0)
    assert derivative[0] == pytest.approx(10)
    assert derivative[-1] == 0.0
    component.state.T_node_0_K = 400
    component.set_outputs(0)
    assert component.output.estimated_heat_W == pytest.approx(20)


def test_confirmed_impulse_is_exact_and_restart_id_is_a_registered_state():
    model = SemiconductorLossModel(turn_on_energy_J=0.2)
    component = ElectroThermalModel(
        LumpedThermalNetwork.cauer([0.1, 0.5], [1, 1], ambient_K=300),
        loss_model=model,
    )
    circuit = Model({"thermal": component}, {})
    initial = circuit.get_initial_values()
    assert len(initial) == 3
    energy = component.apply_committed_transition(
        SwitchingEvent(0.001, "turn_on", 2, 400), event_id=10)
    assert energy == pytest.approx(0.2)
    current = circuit.get_initial_values()
    assert current[0] == pytest.approx(302)
    assert current[2] == pytest.approx(10)
    with pytest.raises(ValueError, match="already been committed"):
        component.apply_committed_transition(
            SwitchingEvent(0.001, "turn_on", 2, 400), event_id=10)
    # A cold reconstruction can restore exactly the same continuous + event-ID state.
    resumed = ElectroThermalModel(
        LumpedThermalNetwork.cauer([0.1, 0.5], [1, 1], ambient_K=300),
        loss_model=model,
    )
    second = Model({"thermal": resumed}, {})
    second.set_states(current)
    with pytest.raises(ValueError, match="already been committed"):
        resumed.apply_committed_transition(
            SwitchingEvent(0.001, "turn_on", 2, 400), event_id=10)
    np.testing.assert_allclose(initial, [300, 300, -1])


def test_electrothermal_element_uses_existing_source_not_an_electrical_netlist():
    cfg = ElectroThermal.Params(
        c_j_J_K=0.25, r_jc_K_W=1.0, c_c_J_K=0.5, r_ca_K_W=2.0,
        heat_source="load.power",
    )
    element = ElectroThermal("thermal_load", cfg, {}, None)
    assert element.port_bindings() == ()
    assert element.subsystems()["thermal_load"] is element.model
    assert element.connections() == {(element.model, "accounted_heat_W"): "load.power"}
    assert math.isfinite(element.model.output.case_K)
    with pytest.raises(ValueError, match="exactly one"):
        ElectroThermal.Params(c_j_J_K=1, r_jc_K_W=1, c_c_J_K=1, r_ca_K_W=1)
    with pytest.raises(ValueError, match="overlap"):
        ElectroThermal.Params(
            c_j_J_K=1, r_jc_K_W=1, c_c_J_K=1, r_ca_K_W=1,
            heat_source="load.power", conduction_voltage_V=1,
        )


def test_model_runs_thermal_dynamics_from_a_connected_power_source():
    class Source:
        state_names = ()
        outputs_need_inputs = False
        observer = True

        def __init__(self):
            self.state = SimpleNamespace()
            self.input = SimpleNamespace()
            self.output = SimpleNamespace(power=10.0)

        def set_outputs(self, _t):
            self.output.power = 10.0

        def rhs(self, _t):
            return ()

    source = Source()
    thermal = ElectroThermalModel(
        LumpedThermalNetwork.cauer([1, 1], [1, 1], ambient_K=300))
    model = Model({"source": source, "thermal": thermal},
                  {(thermal, "accounted_heat_W"): (source, "power")})
    state = model.get_initial_values()
    assert model.rhs(0, state)[0] == pytest.approx(10.0)
    assert model.rhs(0, state)[1] == pytest.approx(0.0)
    assert "thermal.T_node_0_K" in model.state_labels()
    assert model.energy_specs["thermal"].kind == "observer"


def test_temperature_reports_and_device_comparison():
    t = np.array([0, 1, 2], dtype=float)
    result = SimpleNamespace(t=t, plant={
        "thermal.junction_K": np.array([300, 315, 320]),
        "thermal.case_K": np.array([300, 305, 310]),
        "thermal.heat_W": np.array([5, 5, 5]),
        "thermal.cooling_W": np.array([0, 1, 2]),
    }, states={})
    report = thermal_report(result, "thermal", maximum_junction_K=325,
                            output_power_W=95)
    assert report["junction_peak_K"] == pytest.approx(320)
    assert report["safety_margin_K"] == pytest.approx(5)
    assert report["within_rating"] is True
    assert report["heat_integrated_J"] == pytest.approx(10)
    assert report["estimated_device_efficiency"] == pytest.approx(95 / 100)
    second = {**report, "safety_margin_K": 0, "junction_peak_K": 325}
    assert list(compare_thermal_reports({"B": second, "A": report})) == ["A", "B"]


def test_end_to_end_attachment_in_existing_buck_circuit(tmp_path):
    import peslite
    from pathlib import Path

    examples = Path(__file__).resolve().parents[1] / "examples"
    params = peslite.load(examples / "buck-open-loop-example.pes", **{
        "simulation.t_end": 2.0e-4,
        "simulation.solver.linearisations": 0,
        "simulation.output.signals": 1,
    })
    params = params.replace(**{
        "elements.thermal_load": {
            "type": "electrothermal",
            "heat_source": "load.power",
            "c_j_J_K": 0.2,
            "r_jc_K_W": 1.5,
            "c_c_J_K": 1.0,
            "r_ca_K_W": 3.0,
            "ambient_K": 300.0,
            "initial_K": 300.0,
        }
    })
    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path)
    assert "thermal_load.T_node_0_K" in result.states
    assert "thermal_load.last_transition_id" in result.states
    assert result.plant["thermal_load.junction_K"][-1] >= 300.0
    assert np.max(result.plant["thermal_load.heat_W"]) >= 0.0
    assert result.summary["status"] == "completed"
