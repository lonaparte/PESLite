"""Datasheet lookup, heat attribution and conservative event accounting (issue #40)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from peslite.addons.functions import (
    LossTable, SemiconductorLossModel, SwitchingEvent, circuit_heat_report,
    compare_devices, device_loss_report, recorded_device_loss,
)


def _table():
    currents = np.array([0.0, 10.0])
    voltages = np.array([0.0, 400.0])
    temperatures = np.array([300.0, 400.0])
    data = (0.0001 * currents[:, None, None]
            + 0.000001 * voltages[None, :, None]
            + 0.00001 * temperatures[None, None, :])
    return LossTable({"current_A": currents, "voltage_V": voltages,
                      "temperature_K": temperatures}, data)


def test_datasheet_table_interpolates_in_three_dimensions_and_broadcasts():
    table = _table()
    assert table.evaluate(current_A=5, voltage_V=200, temperature_K=350) == pytest.approx(
        0.0005 + 0.0002 + 0.0035)
    np.testing.assert_allclose(table.evaluate(
        current_A=[-5, 10], voltage_V=200, temperature_K=350
    ), [0.0042, 0.0047])
    assert not table.values.flags.writeable


def test_out_of_range_policy_is_explicit_and_clamping_is_not_extrapolation():
    table = LossTable({"current_A": [0, 10]}, [0, 0.1])
    with pytest.raises(ValueError, match="outside"):
        table.evaluate(current_A=11)
    clamped = LossTable({"current_A": [0, 10]}, [0, 0.1], out_of_range="clamp")
    assert clamped.evaluate(current_A=11) == pytest.approx(0.1)
    assert clamped.evaluate(current_A=-11) == pytest.approx(0.1)


@pytest.mark.parametrize("axes,values", [
    ({}, [0, 1]),
    ({"power_W": [0, 1]}, [0, 1]),
    ({"current_A": [0, 0]}, [0, 1]),
    ({"current_A": [0, 2, 1]}, [0, 1, 2]),
    ({"voltage_V": [-1, 0]}, [1, 2]),
    ({"temperature_K": [0, 300]}, [1, 2]),
    ({"current_A": [0, 1]}, [1]),
    ({"current_A": [0, 1]}, [1, -1]),
    ({"current_A": [0, 1]}, [1, float("nan")]),
])
def test_datasheet_tables_reject_invalid_data(axes, values):
    with pytest.raises(ValueError):
        LossTable(axes, values)


def test_fixed_temperature_conduction_and_transition_energies_keep_watts_and_joules_separate():
    model = SemiconductorLossModel(
        conduction_voltage_V=1.2,
        turn_on_energy_J=0.002,
        turn_off_energy_J=0.003,
        reverse_recovery_energy_J=0.001,
    )
    events = [
        SwitchingEvent(0.25, "turn_on", 10, 300),
        SwitchingEvent(0.75, "turn_off", 10, 300),
        SwitchingEvent(1.25, "reverse_recovery", 10, 300),
    ]
    report = device_loss_report([0, 1, 2], [-10, 10, 10], model,
                                conduction_fraction=0.5, events=events)
    np.testing.assert_allclose(report.instantaneous_conduction_W, [6, 6, 6])
    data = report.to_dict()
    assert data["estimated_conduction_J"] == pytest.approx(12)
    assert data["turn_on_J"] == pytest.approx(0.002)
    assert data["turn_off_J"] == pytest.approx(0.003)
    assert data["reverse_recovery_J"] == pytest.approx(0.001)
    assert data["additional_estimated_J"] == pytest.approx(12.006)
    assert data["attributed_heat_W"] == pytest.approx(6.003)
    assert data["event_count"] == 3
    json.dumps(data, allow_nan=False)


def test_temperature_dependent_loss_is_only_a_fixed_reference_lookup():
    model = SemiconductorLossModel(
        conduction_voltage_V=LossTable({"temperature_K": [300, 400]}, [1, 2]),
        turn_on_energy_J=LossTable({"current_A": [0, 10], "temperature_K": [300, 400]},
                                   [[0, 0], [0.001, 0.002]]),
    )
    event = SwitchingEvent(0.5, "turn_on", 10, 400)
    cold = device_loss_report([0, 1], [10, 10], model, temperature_K=300,
                              events=[event]).to_dict()
    hot = device_loss_report([0, 1], [10, 10], model, temperature_K=400,
                             events=[event]).to_dict()
    assert cold["estimated_conduction_J"] == pytest.approx(10)
    assert hot["estimated_conduction_J"] == pytest.approx(20)
    assert hot["turn_on_J"] == pytest.approx(0.002)
    assert cold["turn_on_J"] == pytest.approx(0.001)


def test_electrically_realized_conduction_is_not_reestimated():
    model = SemiconductorLossModel(turn_on_energy_J=0.001)
    report = device_loss_report([0, 1], [5, 5], model,
                                accounted_conduction_power_W=[4, 4],
                                events=[SwitchingEvent(0.5, "turn_on", 5, 400)])
    assert report.to_dict()["accounted_conduction_J"] == pytest.approx(4)
    assert report.to_dict()["additional_estimated_J"] == pytest.approx(0.001)
    with pytest.raises(ValueError, match="overlap"):
        device_loss_report([0, 1], [5, 5], SemiconductorLossModel(
            conduction_voltage_V=1), accounted_conduction_power_W=[4, 4])


def test_missing_event_curve_duplicate_and_recovery_overlap_are_rejected():
    event = SwitchingEvent(0.5, "turn_on", 2, 400)
    with pytest.raises(ValueError, match="no turn_on"):
        device_loss_report([0, 1], [2, 2], SemiconductorLossModel(), events=[event])
    with pytest.raises(ValueError, match="duplicate"):
        device_loss_report([0, 1], [2, 2], SemiconductorLossModel(turn_on_energy_J=1),
                           events=[event, event])
    with pytest.raises(ValueError, match="reverse-recovery"):
        device_loss_report([0, 1], [2, 2], SemiconductorLossModel(
            reverse_recovery_energy_J=1, turn_on_includes_reverse_recovery=True),
            events=[SwitchingEvent(0.5, "reverse_recovery", 2, 400)])


def test_invalid_traces_and_events_fail_without_silent_loss_extrapolation():
    model = SemiconductorLossModel(conduction_voltage_V=1, turn_on_energy_J=0.1)
    for time, current in [([0], [1]), ([0, 0], [1, 2]), ([0, 1], [1]),
                          ([0, 1], [float("nan"), 2])]:
        with pytest.raises(ValueError):
            device_loss_report(time, current, model)
    for fraction in (-0.1, 1.1):
        with pytest.raises(ValueError, match="conduction_fraction"):
            device_loss_report([0, 1], [1, 1], model, conduction_fraction=fraction)
    with pytest.raises(ValueError, match="within"):
        device_loss_report([0, 1], [1, 1], model,
                           events=[SwitchingEvent(1.5, "turn_on", 1, 2)])
    with pytest.raises(ValueError, match="positive"):
        device_loss_report([0, 1], [1, 1], model, temperature_K=0)


def test_candidate_comparison_and_circuit_report_keep_accounted_energy_separate():
    candidates = {
        "device_a": SemiconductorLossModel(conduction_voltage_V=1, turn_on_energy_J=0.2),
        "device_b": SemiconductorLossModel(conduction_voltage_V=0.5, turn_on_energy_J=0.1),
    }
    reports = compare_devices([0, 1], [10, 10], candidates,
                              events=[SwitchingEvent(0.5, "turn_on", 10, 400)])
    assert reports["device_b"].to_dict()["attributed_heat_J"] < reports["device_a"].to_dict()["attributed_heat_J"]
    circuit = circuit_heat_report({"switch": reports["device_b"]},
                                  energy_closure={"dissipated": 8, "impulsive": 2})
    assert circuit["totals"]["circuit_electrical_dissipation_J"] == pytest.approx(10)
    assert circuit["totals"]["circuit_additional_estimate_J"] == pytest.approx(5.1)
    assert "combined_circuit_heat_J" not in circuit["totals"]
    with pytest.raises(ValueError, match="same time"):
        circuit_heat_report({"x": reports["device_b"], "y": device_loss_report(
            [0, 2], [10, 10], candidates["device_b"])})


def test_existing_streamed_result_columns_can_feed_the_heat_analysis():
    result = SimpleNamespace(t=np.array([0.0, 1.0]), plant={
        "device.current": np.array([4.0, 4.0]),
        "device.voltage": np.array([300.0, 300.0]),
        "device.on": np.array([1.0, 0.5]),
    })
    model = SemiconductorLossModel(conduction_voltage_V=1)
    report = recorded_device_loss(result, "device.current", model,
                                  voltage_column="device.voltage",
                                  conduction_fraction_column="device.on")
    assert report.to_dict()["estimated_conduction_J"] == pytest.approx(3)
    assert result.t.flags.writeable  # Reports must not mutate the caller's trace.
    with pytest.raises(ValueError, match="plant column"):
        recorded_device_loss(result, "missing.current", model)
