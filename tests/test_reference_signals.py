import math
from dataclasses import replace

import numpy as np
import pytest

import peslite
from peslite.addons.functions import reference_csv, reference_expression, reference_table
from peslite.assembly.cpp import _validate_native_control_graph
from peslite.assembly.params import from_dict, read_initial, read_tree, to_dict
from peslite.control import TableSignal
from peslite.control.controller import ConverterMeasurement
from peslite.solver import ConfigError


def test_native_table_interpolation_repeat_and_extrapolation():
    linear = TableSignal(time=[0.0, 1.0, 2.0], value=[0.0, 2.0, 0.0])
    held = TableSignal(
        time=[0.0, 1.0, 2.0], value=[0.0, 2.0, 0.0],
        interpolation="zero_order_hold", repeat=True,
    )
    extrapolated = TableSignal(
        time=[1.0, 2.0], value=[2.0, 4.0], extrapolation="linear",
    )

    assert linear(0.25) == pytest.approx(0.5)
    assert linear(-1.0) == 0.0
    assert linear(3.0) == 0.0
    assert held(0.99) == 0.0
    assert held(1.01) == 2.0
    assert held(2.5) == 0.0
    assert extrapolated(0.0) == pytest.approx(0.0)
    assert extrapolated(3.0) == pytest.approx(6.0)

    periodic = TableSignal(
        time=[0.0, 1.0], value=[2.0, 3.0], repeat=True,
        interpolation="zero_order_hold",
    )
    assert periodic(1.0) == periodic(0.0) == 2.0


@pytest.mark.parametrize("method", ("cubic", "pchip"))
def test_advanced_interpolation_is_constructed_once_and_evaluated_as_a_table(method):
    signal = TableSignal(
        time=[0.0, 0.5, 1.0], value=[0.0, 1.0, 0.0], interpolation=method,
    )
    assert signal(0.0) == pytest.approx(0.0)
    assert signal(0.5) == pytest.approx(1.0)
    assert signal(1.0) == pytest.approx(0.0)
    assert math.isfinite(signal(0.25))


@pytest.mark.parametrize("change", [
    {"time": []},
    {"time": [0.0, 0.0], "value": [0.0, 1.0]},
    {"time": [0.0], "value": [0.0, 1.0]},
    {"time": [0.0], "value": [0.0], "repeat": True},
    {"time": [False], "value": [0.0]},
    {"time": [0.0], "value": [0.0], "repeat": "yes"},
    {"time": [0.0, 1.0], "value": [0.0, math.inf]},
])
def test_invalid_native_tables_are_rejected(change):
    values = {"time": [0.0], "value": [0.0], **change}
    with pytest.raises(ConfigError):
        TableSignal(**values)


@pytest.mark.parametrize("tree", [[], {"base": {"s_base": 1.0}, "simulation": []}])
def test_signal_preparation_preserves_the_standard_mapping_errors(tree):
    with pytest.raises(ConfigError, match="expected a mapping"):
        from_dict(tree)


def _signal_case(examples, *, reference="duty", signal="duty_profile"):
    tree = read_tree(examples / "buck-open-loop-example.pes")
    tree["events"] = {}
    tree["signals"] = {
        signal: {
            "type": "table",
            "time": [0.0, 0.002, 0.004],
            "value": [0.2, 0.6, 0.4],
        },
    }
    tree["units"]["converter"]["ctrl"]["references"] = {
        reference: f"signals.{signal}",
    }
    return tree


def test_inline_signal_uses_the_existing_sampled_reference_boundary(examples):
    params = from_dict(_signal_case(examples))
    simulation = peslite.Simulation(params)
    ctrl = simulation.unit("converter").ctrl
    measurement = ConverterMeasurement(current=0.0, voltage=0.0, vin=48.0)

    ctrl.graph.sample(0.0, measurement)
    first = ctrl.graph.references["duty"]
    ctrl.graph.sample(0.001, measurement)
    middle = ctrl.graph.references["duty"]
    ctrl.graph.sample(0.003, measurement)
    last = ctrl.graph.references["duty"]

    assert (first, middle, last) == pytest.approx((0.2, 0.4, 0.5))
    assert to_dict(params)["units"]["converter"]["ctrl"]["references"] == {
        "duty": "signals.duty_profile",
    }


def test_signal_binding_uses_the_normal_reference_unit_conversion(examples):
    tree = read_tree(examples / "gfm-psc-example.pes")
    tree["signals"] = {
        "power": {"type": "table", "time": [0.0, 1.0], "value": [0.0, 1.0e6]},
    }
    references = tree["units"]["vsc"]["ctrl"]["references"]
    references.pop("p_ref_pu")
    references["p_ref"] = "signals.power"
    tree["events"] = {
        "constant": {
            "type": "set",
            "t": 0.01,
            "set": {"units.vsc.ctrl.references.p_ref_pu": 0.25},
        },
    }
    params = from_dict(tree)
    binding = params.unit("vsc").ctrl._reference_bindings["p_ref_pu"]

    assert binding(0.5) == pytest.approx(0.5e6 / params.unit("vsc").base.s_base)
    assert to_dict(params)["units"]["vsc"]["ctrl"]["references"]["p_ref"] == "signals.power"
    changed = params.changes[0].params.unit("vsc").ctrl
    assert changed.references.p_ref_pu == 0.25
    assert not changed._reference_bindings


def test_table_values_reuse_the_control_type_reference_validation(examples):
    tree = _signal_case(examples)
    tree["signals"]["duty_profile"]["value"] = [0.2, 1.1, 0.4]

    with pytest.raises(ConfigError, match=r"signals\.duty_profile\.value\[1\]\.duty"):
        from_dict(tree)


def test_integer_references_require_held_integer_tables(examples):
    tree = read_tree(examples / "wpt-phase-shift-example.pes")
    tree["events"] = {}
    tree["signals"] = {
        "modes": {"time": [0.0, 0.001], "value": [0, 1]},
    }
    references = tree["units"]["charger"]["primary"]["ctrl"]["references"]
    references["mode"] = "signals.modes"

    with pytest.raises(ConfigError, match="integer reference requires zero_order_hold"):
        from_dict(tree)

    tree["signals"]["modes"]["interpolation"] = "zero_order_hold"
    binding = from_dict(tree).unit("charger.primary").ctrl._reference_bindings["mode"]
    assert binding(0.001) == 1
    assert isinstance(binding(0.001), int)

    tree["signals"]["modes"]["extrapolation"] = "linear"
    with pytest.raises(ConfigError, match="integer reference cannot use linear extrapolation"):
        from_dict(tree)


def test_static_reference_errors_are_not_attributed_to_a_bound_table(examples):
    tree = read_tree(examples / "wpt-phase-shift-example.pes")
    tree["events"] = {}
    tree["signals"] = {"widths": {"time": [0.0, 0.001], "value": [0.2, 0.4]}}
    references = tree["units"]["charger"]["primary"]["ctrl"]["references"]
    references["width"] = "signals.widths"
    references["frequency"] = -1.0

    with pytest.raises(
        ConfigError,
        match=r"units\.charger\.primary\.ctrl\.references\.frequency",
    ):
        from_dict(tree)


def test_initial_time_and_set_events_follow_one_reference_path(examples):
    tree = _signal_case(examples)
    tree["simulation"]["initial"] = {"t": 0.001}
    tree["events"] = {
        "constant": {
            "type": "set",
            "t": 0.003,
            "set": {"units.converter.ctrl.references.duty": 0.3},
        },
    }
    params = from_dict(tree)

    assert params.unit("converter").ctrl.references.duty == pytest.approx(0.4)
    changed = params.changes[0].params.unit("converter").ctrl
    assert changed.references.duty == pytest.approx(0.3)
    assert not changed._reference_bindings


def test_sampled_simulation_and_event_use_the_same_reference_binding(examples, tmp_path):
    tree = _signal_case(examples)
    tree["simulation"]["t_end"] = 0.003
    tree["simulation"]["output"]["signals"] = 1
    tree["events"] = {
        "constant": {
            "type": "set",
            "t": 0.002,
            "set": {"units.converter.ctrl.references.duty": 0.3},
        },
    }
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    times = result.ctrl["converter.t"]
    reference = result.ctrl["converter.duty_reference"]
    before = times < 0.002 - 1e-12
    after = times >= 0.002 - 1e-12

    np.testing.assert_allclose(reference[before], np.interp(
        times[before], [0.0, 0.002, 0.004], [0.2, 0.6, 0.4],
    ), atol=1e-12)
    np.testing.assert_allclose(reference[after], 0.3, atol=1e-12)


def test_reference_playback_is_restart_safe(examples, tmp_path):
    tree = _signal_case(examples)
    tree["simulation"]["t_end"] = 0.004
    params = from_dict(tree)
    whole = peslite.Simulation(params).run(out_dir=tmp_path / "whole")
    first = peslite.Simulation(params.replace(**{
        "simulation.t_end": 0.002,
    })).run(out_dir=tmp_path / "first")
    resumed = peslite.Simulation(params.replace(**{
        "simulation.initial": read_initial(tmp_path / "first" / "states.csv"),
    })).run(out_dir=tmp_path / "resumed")

    assert first.final_states()["t"] == pytest.approx(0.002)
    assert resumed.final_states()["t"] == pytest.approx(0.004)
    for name, value in whole.final_states().items():
        assert resumed.final_states()[name] == pytest.approx(value, rel=1e-10, abs=1e-12)


def test_continuous_graph_reads_the_table_at_its_existing_evaluation_time(examples):
    tree = _signal_case(examples)
    tree["units"]["converter"]["bridge"] = {"model": "averaging"}
    simulation = peslite.Simulation(from_dict(tree))
    ctrl = simulation.unit("converter").ctrl
    measurement = ConverterMeasurement(current=0.0, voltage=0.0, vin=48.0)

    assert ctrl.graph.continuous
    ctrl.graph.flow(0.001, measurement)
    assert ctrl.graph.references["duty"] == pytest.approx(0.4)


def test_runtime_playback_error_names_the_bound_reference(examples):
    tree = _signal_case(examples)
    tree["signals"]["duty_profile"]["extrapolation"] = "error"
    ctrl = peslite.Simulation(from_dict(tree)).unit("converter").ctrl

    with pytest.raises(FloatingPointError, match="controller reference 'duty' failed"):
        ctrl.graph.sample(0.005, ConverterMeasurement(vin=48.0))


def test_runtime_validation_catches_interpolation_overshoot(examples):
    tree = _signal_case(examples)
    tree["signals"]["duty_profile"] = {
        "time": [0.0, 0.001, 0.002, 0.003],
        "value": [0.0, 1.0, 0.0, 1.0],
        "interpolation": "cubic",
    }
    ctrl = peslite.Simulation(from_dict(tree)).unit("converter").ctrl

    with pytest.raises(FloatingPointError, match=r"duty must be in \[0, 1\]"):
        ctrl.graph.sample(0.00075, ConverterMeasurement(vin=48.0))


def test_signal_configuration_survives_dump_and_load(examples, tmp_path):
    params = from_dict(_signal_case(examples))
    path = tmp_path / "signal.pes"
    peslite.dump(params, path)
    loaded = peslite.load(path)

    assert to_dict(loaded)["signals"] == to_dict(params)["signals"]
    assert loaded.unit("converter").ctrl._reference_bindings["duty"](0.001) == pytest.approx(0.4)


def test_reference_binding_is_part_of_configuration_equality(examples):
    tree = _signal_case(examples)
    tree["signals"]["other"] = dict(tree["signals"]["duty_profile"])
    first = from_dict(tree)
    tree["units"]["converter"]["ctrl"]["references"]["duty"] = "signals.other"
    second = from_dict(tree)

    assert first != second
    assert replace(first.unit("converter").ctrl, period=1.0e-4)._reference_bindings


def test_addon_csv_and_expression_helpers_generate_native_tables(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("time,current\n0,1\n0.5,2\n1,1\n", encoding="utf-8")
    csv_table = reference_csv(
        path, time_column="time", value_column="current", time_scale=2.0,
        time_offset=0.1, value_scale=3.0, value_offset=-1.0, interpolation="pchip",
    )
    sine = reference_expression(
        "2*sin(2*pi*50*t)", start=0.0, stop=0.02, period=0.001,
    )
    chirp = reference_expression(
        "sin(2*pi*(20*t + 0.5*100*t**2))", start=0.0, stop=0.1, period=0.01,
    )

    assert TableSignal(**csv_table)(0.1) == pytest.approx(2.0)
    assert csv_table["time"] == pytest.approx([0.1, 1.1, 2.1])
    assert TableSignal(**sine)(0.005) == pytest.approx(2.0, abs=1e-12)
    assert all(math.isfinite(value) for value in chirp["value"])
    assert reference_table([0.0, 1.0], [1.0, 2.0], repeat=True)["repeat"]


@pytest.mark.parametrize(("time", "value"), [
    ([False], [0.0]),
    ([0.0], [True]),
    ("01", [0.0, 1.0]),
])
def test_addon_table_rejects_non_numeric_sequences(time, value):
    with pytest.raises(ValueError, match="must contain numbers"):
        reference_table(time, value)


def test_expression_helper_rejects_non_mathematical_python():
    with pytest.raises(ValueError, match="not available"):
        reference_expression(
            "__import__('os').getcwd()", start=0.0, stop=1.0, period=0.1,
        )


def test_unknown_signal_and_native_export_fail_explicitly(examples):
    tree = _signal_case(examples)
    tree["units"]["converter"]["ctrl"]["references"]["duty"] = "signals.missing"
    with pytest.raises(ConfigError, match="unknown signal"):
        from_dict(tree)

    params = from_dict(_signal_case(examples))
    ctrl = peslite.Simulation(params).unit("converter").ctrl
    with pytest.raises(TypeError, match="time-varying controller reference signals"):
        _validate_native_control_graph(ctrl)
