import math
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import peslite
from peslite.addons.functions import reference_csv, reference_expression, reference_table
from peslite.assembly.cpp import _validate_native_control_graph
from peslite.assembly.params import from_dict, read_initial, read_tree, to_dict
from peslite.control import ControlMeasurement, TableSignal
from peslite.control.controller import ConverterMeasurement
from peslite.solver import ConfigError
from peslite.solver.timing import TIME_EPS


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
    assert held(1.0 - 0.5 * TIME_EPS) == 2.0
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


def test_table_bounds_cover_internal_extrema_and_only_the_repeated_phase_window():
    linear = TableSignal(time=[0.0, 1.0, 2.0], value=[0.0, 2.0, 0.0])
    cubic = TableSignal(
        time=[0.0, 1.0, 2.0, 3.0], value=[0.0, 1.0, 0.0, 1.0],
        interpolation="cubic",
    )
    repeated = TableSignal(
        time=[0.0, 1.0, 2.0], value=[0.0, 2.0, 0.0], repeat=True,
    )
    discontinuous = TableSignal(time=[0.0, 1.0], value=[2.0, 5.0], repeat=True)

    assert linear.bounds(0.25, 1.75) == pytest.approx((0.5, 2.0))
    low, high = cubic.bounds(0.0, 3.0)
    assert low < 0.0 and high > 1.0
    assert repeated.bounds(2.25, 2.75) == pytest.approx((0.5, 1.5))
    assert repeated.bounds(3.75, 4.25) == pytest.approx((0.0, 0.5))
    assert repeated.bounds(0.0, 2.0) == pytest.approx((0.0, 2.0))
    assert discontinuous.bounds(0.0, 1.0) == pytest.approx((2.0, 5.0))


def test_table_bounds_follow_extrapolation_and_reject_invalid_windows():
    linear = TableSignal(
        time=[1.0, 2.0], value=[2.0, 4.0], extrapolation="linear",
    )
    strict = TableSignal(
        time=[1.0, 2.0], value=[2.0, 4.0], extrapolation="error",
    )

    assert linear.bounds(0.0, 3.0) == pytest.approx((0.0, 6.0))
    with pytest.raises(ValueError, match="before the table interval"):
        strict.bounds(0.0, 1.5)
    with pytest.raises(ValueError, match="finite start <= stop"):
        linear.bounds(2.0, 1.0)


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


def _unit_section(tree, path):
    unit = tree["units"]
    for name in path.split("."):
        unit = unit[name]
    return unit


def _bind_table_reference(
        tree, unit, reference, values, *, stop=0.001, times=None,
        interpolation="linear"):
    tree["signals"] = {
        "reference_profile": {
            "type": "table",
            "time": [0.0, stop] if times is None else list(times),
            "value": list(values),
            "interpolation": interpolation,
        },
    }
    references = _unit_section(tree, unit)["ctrl"].setdefault("references", {})
    references[reference] = "signals.reference_profile"
    return tree


def _evaluate_control_graph(ctrl, t, measurement):
    if ctrl.continuous:
        ctrl.startup.at(t)
        ctrl.graph.flow(t, measurement)
    else:
        ctrl.startup.advance()
        ctrl.graph.sample(t, measurement)
    return {
        name: ctrl.graph._gather(ctrl.graph._wiring[name], measurement)
        for name in ctrl.graph.nodes
    }


@pytest.mark.parametrize("model", ("pwm_averaging", "averaging"))
@pytest.mark.parametrize("values", ((24.0, 24.24), (12.0, 36.0)), ids=("small", "large"))
def test_voltage_reference_signal_crosses_cascaded_dc_loops(examples, model, values):
    tree = read_tree(examples / "buck-example.pes")
    tree["events"] = {}
    converter = tree["units"]["converter"]
    converter.setdefault("bridge", {})["model"] = model
    _bind_table_reference(tree, "converter", "voltage", values)
    ctrl = peslite.Simulation(from_dict(tree)).unit("converter").ctrl
    ctrl.command(True)
    measurement = ConverterMeasurement(current=4.0, voltage=24.0, vin=48.0)

    for t, expected in zip((0.0, 0.001), values):
        inputs = _evaluate_control_graph(ctrl, t, measurement)
        assert inputs["DCVoltageControl"]["reference"] == pytest.approx(expected)
        assert inputs["InductorCurrentControl"]["reference"] == pytest.approx(
            ctrl.graph.values["DCVoltageControl.current_reference"]
        )


@pytest.mark.parametrize("model", ("pwm_averaging", "averaging"))
@pytest.mark.parametrize("size", ("small", "large"))
@pytest.mark.parametrize("level", ("inner", "outer"))
def test_vsc_reference_signal_reaches_inner_and_outer_control_levels(
        examples, model, size, level):
    tree = read_tree(examples / "gfl-example.pes")
    tree["events"] = {}
    vsc = tree["units"]["vsc"]
    vsc.setdefault("bridge", {})["model"] = model
    if level == "inner":
        vsc["ctrl"]["loops"].pop("DCVoltageControl")
        reference = "id_ref_pu"
        values = (0.2, 0.202) if size == "small" else (-0.4, 0.8)
    else:
        reference = "vdc_ref_pu"
        values = (1.0, 1.01) if size == "small" else (0.6, 1.4)
    _bind_table_reference(tree, "vsc", reference, values)
    ctrl = peslite.Simulation(from_dict(tree)).unit("vsc").ctrl
    ctrl.command(True)
    measurement = ControlMeasurement(0.0, 1.0 + 0j, 0j, 1.0, np.zeros(3))

    for t, expected in zip((0.0, 0.001), values):
        measurement.t = t
        inputs = _evaluate_control_graph(ctrl, t, measurement)
        if level == "inner":
            assert inputs["CurrentLoop"]["id_ref"] == pytest.approx(expected)
        else:
            assert inputs["DCVoltageControl"]["reference"] == pytest.approx(expected)
            assert inputs["CurrentLoop"]["id_ref"] == pytest.approx(
                ctrl.graph.values["DCVoltageControl.current_reference"]
            )


@pytest.mark.parametrize("values", ((4.0, 4.04), (1.0, 7.0)), ids=("small", "large"))
def test_reference_signal_reaches_a_predictive_current_loop(examples, values):
    tree = read_tree(examples / "mpc-buck-example.pes")
    tree["events"] = {}
    _bind_table_reference(tree, "converter", "current", values)
    ctrl = peslite.Simulation(from_dict(tree)).unit("converter").ctrl
    ctrl.command(True)
    measurement = ConverterMeasurement(current=4.0, voltage=24.0, vin=48.0)

    for t, expected in zip((0.0, 0.001), values):
        inputs = _evaluate_control_graph(ctrl, t, measurement)
        assert inputs["PredictiveCurrentControl"]["reference"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "example,unit,reference,outer,inner,inner_port,small,large",
    (
        ("wpt-ss-example.pes", "charger.primary", "power", "PowerControl",
         "ResonantModulation", "modulation", (1200.0, 1212.0), (200.0, 2500.0)),
        ("wpt-lcc-example.pes", "charger.secondary", "power", "PhasePowerControl",
         "ResonantModulation", "phase", (-500.0, -505.0), (-100.0, -1000.0)),
    ),
    ids=("primary", "secondary"),
)
@pytest.mark.parametrize("size", ("small", "large"))
def test_reference_signal_crosses_nested_wpt_control_levels(
        examples, example, unit, reference, outer, inner, inner_port, small, large, size):
    tree = read_tree(examples / example)
    tree["events"] = {}
    values = small if size == "small" else large
    _bind_table_reference(tree, unit, reference, values)
    ctrl = peslite.Simulation(from_dict(tree)).unit(unit).ctrl
    ctrl.command(True)
    measurement = ConverterMeasurement(current=0.0, voltage=0.0, vin=400.0, power=0.0)

    for t, expected in zip((0.0, 0.001), values):
        inputs = _evaluate_control_graph(ctrl, t, measurement)
        assert inputs[outer]["reference"] == pytest.approx(expected)
        assert inputs[inner][inner_port] == pytest.approx(
            ctrl.graph.values[f"{outer}.{inner_port}"]
        )


def _reference_event_equivalence(
        tree, unit, reference, values, step, end, log_name, output_dir):
    tree["simulation"]["t_end"] = end
    tree["simulation"]["solver"]["linearisations"] = 0
    output = tree["simulation"].setdefault("output", {})
    output.update({
        "states": True, "signals": True, "energy": False,
        "record_every": 1, "precision": 17,
    })
    tree["simulation"]["progress"] = {"enable": False}
    tree["simulation"]["energy_check"] = "off"

    signal_tree = deepcopy(tree)
    _bind_table_reference(
        signal_tree, unit, reference, values, stop=step,
        interpolation="zero_order_hold",
    )
    event_tree = deepcopy(tree)
    _unit_section(event_tree, unit)["ctrl"].setdefault("references", {})[reference] = values[0]
    event_tree.setdefault("events", {})["reference_step"] = {
        "type": "set",
        "t": step,
        "set": {f"units.{unit}.ctrl.references.{reference}": values[1]},
    }

    signal = peslite.Simulation(from_dict(signal_tree)).run(out_dir=output_dir / "signal")
    event = peslite.Simulation(from_dict(event_tree)).run(out_dir=output_dir / "event")

    np.testing.assert_array_equal(signal.t, event.t)
    assert signal.states.keys() == event.states.keys()
    for name in signal.states:
        np.testing.assert_allclose(
            signal.states[name], event.states[name], rtol=2e-10, atol=2e-11,
            err_msg=name,
        )
    np.testing.assert_allclose(
        signal.ctrl[log_name], event.ctrl[log_name], rtol=0.0, atol=1e-12,
    )
    selected = signal.ctrl[f"{unit}.t"] >= step - 1e-12
    np.testing.assert_allclose(signal.ctrl[log_name][selected], values[1], atol=1e-12)


@pytest.mark.parametrize("size", ("small", "large"))
@pytest.mark.parametrize("case", ("dc_voltage", "predictive_current", "vsc_current"))
def test_reference_signal_matches_set_event_across_converter_control_levels(
        examples, case, size, tmp_path):
    if case == "dc_voltage":
        tree = read_tree(examples / "buck-example.pes")
        tree["events"] = {}
        values = (24.0, 24.24) if size == "small" else (12.0, 36.0)
        arguments = ("converter", "voltage", values, "converter.voltage_reference")
    elif case == "predictive_current":
        tree = read_tree(examples / "mpc-buck-example.pes")
        tree["events"] = {}
        values = (4.0, 4.04) if size == "small" else (1.0, 7.0)
        arguments = ("converter", "current", values, "converter.current_reference")
    else:
        tree = read_tree(examples / "gfl-example.pes")
        tree["units"]["vsc"]["ctrl"]["loops"].pop("DCVoltageControl")
        tree["events"] = {
            "connect_vsc": {"type": "connect", "target": "vsc", "t": 0.0, "ramp": 0.0},
        }
        values = (0.2, 0.202) if size == "small" else (-0.4, 0.8)
        arguments = ("vsc", "id_ref_pu", values, "vsc.id_ref_pu")
    _reference_event_equivalence(
        tree, *arguments[:3], 0.003, 0.006, arguments[3], tmp_path,
    )


@pytest.mark.parametrize(
    "example,unit,values,log_name",
    (
        ("wpt-ss-example.pes", "charger.primary", (1200.0, 1212.0),
         "charger.primary.power_reference"),
        ("wpt-lcc-example.pes", "charger.secondary", (-500.0, -505.0),
         "charger.secondary.power_reference"),
        ("wpt-ss-example.pes", "charger.primary", (200.0, 2500.0),
         "charger.primary.power_reference"),
        ("wpt-lcc-example.pes", "charger.secondary", (-100.0, -1000.0),
         "charger.secondary.power_reference"),
    ),
    ids=("primary-small", "secondary-small", "primary-large", "secondary-large"),
)
def test_reference_signal_matches_set_event_in_nested_wpt_control(
        examples, example, unit, values, log_name, tmp_path):
    tree = read_tree(examples / example)
    tree["events"] = {}
    tree["simulation"]["solver"]["dt"] = 100.0e-9
    tree["simulation"].setdefault("output", {})["period"] = 1.0e-6
    _reference_event_equivalence(
        tree, unit, "power", values, 40.0e-6, 80.0e-6, log_name, tmp_path,
    )


@pytest.mark.full
@pytest.mark.parametrize(
    "example,unit,loop,targets",
    (
        ("wpt-ss-example.pes", "charger.primary", "PowerControl",
         (1200.0, 1260.0, 1800.0)),
        ("wpt-lcc-example.pes", "charger.secondary", "PhasePowerControl",
         (-500.0, -525.0, -800.0)),
    ),
    ids=("primary", "secondary"),
)
def test_nested_wpt_tracks_small_and_large_arbitrary_reference_signals(
        examples, example, unit, loop, targets, tmp_path):
    tree = read_tree(examples / example)
    tree["events"] = {}
    for side in ("primary", "secondary"):
        _unit_section(tree, f"charger.{side}").setdefault("bridge", {})["model"] = "averaging"
    _bind_table_reference(
        tree, unit, "power", targets, times=(0.0, 0.003, 0.006),
        interpolation="zero_order_hold",
    )
    tree["simulation"]["t_end"] = 0.009
    tree["simulation"]["solver"].update({"dt": 250.0e-9, "linearisations": 0})
    tree["simulation"].update({"energy_check": "off", "progress": {"enable": False}})
    tree["simulation"]["output"].update({
        "states": False, "signals": True, "energy": False, "record_every": 1,
    })
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)

    for begin, end, target in zip((0.0024, 0.0054, 0.0084),
                                  (0.0028, 0.0058, 0.0088), targets):
        control_time = result.ctrl[f"{unit}.t"]
        control_selected = (control_time >= begin) & (control_time <= end)
        np.testing.assert_allclose(
            result.ctrl[f"{unit}.power_reference"][control_selected], target,
            rtol=0.0, atol=1e-12,
        )
        selected = (result.t >= begin) & (result.t <= end)
        time = result.t[selected]
        power = result.plant[f"{unit}.{loop}.power"][selected]
        mean = np.sum(0.5 * (power[1:] + power[:-1]) * np.diff(time)) / (end - begin)
        assert mean == pytest.approx(target, rel=0.005)


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

    with pytest.raises(ConfigError, match=r"signals\.duty_profile reaches 1\.1 during the run:.*duty"):
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
    with pytest.raises(ConfigError, match=r"references\.duty: signals\.duty_profile: time 0\.008"):
        from_dict(tree)
    tree["simulation"]["t_end"] = 0.004
    ctrl = peslite.Simulation(from_dict(tree)).unit("converter").ctrl

    with pytest.raises(FloatingPointError, match="controller reference 'duty' failed"):
        ctrl.graph.sample(0.005, ConverterMeasurement(vin=48.0))


def test_parse_validation_catches_interpolation_overshoot(examples):
    tree = _signal_case(examples)
    tree["signals"]["duty_profile"] = {
        "time": [0.0, 0.001, 0.002, 0.003],
        "value": [0.0, 1.0, 0.0, 1.0],
        "interpolation": "cubic",
    }
    with pytest.raises(ConfigError, match=r"signals\.duty_profile reaches -?\d.*duty must be in \[0, 1\]"):
        from_dict(tree)
    tree["signals"]["duty_profile"]["interpolation"] = "pchip"
    assert from_dict(tree).unit("converter").ctrl._reference_bindings


def test_repeated_signal_validation_uses_only_the_phases_reached_by_the_run(examples):
    tree = _signal_case(examples)
    tree["signals"]["duty_profile"] = {
        "time": [0.0, 1.0, 2.0], "value": [0.2, 0.4, 1.2], "repeat": True,
    }
    tree["simulation"]["t_end"] = 0.5
    assert from_dict(tree).unit("converter").ctrl._reference_bindings

    tree["simulation"]["t_end"] = 2.1
    with pytest.raises(ConfigError, match=r"signals\.duty_profile reaches 1\.2"):
        from_dict(tree)


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
