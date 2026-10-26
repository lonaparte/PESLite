"""One compiled model uses the ordinary parameter tree for physical and control changes."""

import csv
import subprocess

import numpy as np
import pytest

import peslite
from peslite.assembly.params import _written_out
from conftest import EXAMPLES
from test_cpp_export import _compiler


def _case(name, mode):
    p = peslite.load(EXAMPLES / f"{name}-example.pes")
    from peslite.assembly.params import _unit_items

    values = {f"units.{unit}.bridge.model": mode for unit, _ in _unit_items(p.units)}
    values.update({
        "simulation.t_end": 30e-6 if name.startswith("wpt") else 0.003,
        "simulation.solver.subsystems": {},
        "simulation.solver.linearisations": 0,
        "simulation.solver.type": "adaptive" if mode == "averaging" else "fixed",
        "simulation.solver.method": "DP45" if mode == "averaging" else "rk4",
        "simulation.energy_check": "warn",
        "simulation.output.states": True,
        "simulation.output.energy": True,
    })
    if name == "gfl":
        values.update({"events.connect_vsc.t": 0.0, "events.connect_vsc.ramp": 0.0})
    return p.replace(**values)


def _compile(p, directory, variables=("all",)):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    peslite.export(p, "cpp", directory, variables=variables)
    binary = directory / "run"
    subprocess.run([compiler, "-O2", "-std=c++17", str(directory / "peslite.cpp"),
                    "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


def _table(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        keys = reader.fieldnames
        return keys, np.array([[float(row[key]) for key in keys] for row in reader])


def _compare_run(p, binary, changes, directory, *, rtol=3e-6, atol=3e-5, parts=None):
    native, python = directory / "native", directory / "python"
    args = [str(binary), "--out", str(native)]
    for key, value in changes.items():
        args.extend(["--set", f"{key}={value}"])
    subprocess.run(args, check=True, capture_output=True, text=True)
    expected = p.replace(**changes)
    simulation = peslite.Simulation(expected, parts=parts)
    simulation.run(out_dir=python)
    filenames = ("states.csv", "energy.csv") if p.simulation.energy_check != "off" else ("states.csv",)
    for filename in filenames:
        keys, actual = _table(native / filename)
        expected_keys, reference = _table(python / filename)
        assert keys == expected_keys
        if filename == "states.csv":
            for index, name in enumerate(keys):
                if name in simulation._states.angle_names:
                    actual[:, index] = reference[:, index] + (
                        actual[:, index] - reference[:, index] + np.pi) % (2 * np.pi) - np.pi
        compared = list(range(len(keys)))
        if filename == "energy.csv":
            # Conservation residuals are zero identities. Compare them against the circuit's
            # power flow: iterative algebraic solves and direct elimination leave different
            # small residuals even when their physical states agree.
            residuals = [i for i, key in enumerate(keys)
                         if key in ("tellegen_W", "balance_residual_W")]
            watts = [i for i, key in enumerate(keys) if key.endswith("_W")]
            scale = np.maximum(1.0, np.max(np.abs(reference[:, watts]), axis=1))[:, None]
            np.testing.assert_allclose(actual[:, residuals] / scale, reference[:, residuals] / scale,
                                       rtol=rtol, atol=rtol)
            compared = [i for i in compared if i not in residuals]
        np.testing.assert_allclose(actual[:, compared], reference[:, compared], rtol=rtol, atol=atol)
    assert _written_out(peslite.load(native / "simulation.pes")) == _written_out(expected)


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
@pytest.mark.parametrize("name", ["buck", "gfl", "wpt", "wpt-lcc"])
def test_compiled_parameters_reinitialize_physics_control_and_energy(name, mode, tmp_path):
    p = _case(name, mode)
    if name == "buck":
        changes = {
            "elements.load.r": 8.0,
            "buses.output.c": 600e-6,
            "units.converter.filter.l_f": 1.4e-3,
            "units.converter.filter.r_f": 0.08,
            "units.converter.dclink.source.v": 52.0,
            "units.converter.ctrl.loops.DCVoltageControl.kp": 0.4,
            "units.converter.ctrl.loops.InductorCurrentControl.ki": 40.0,
        }
    elif name == "gfl":
        changes = {
            "buses.pcc.c": p.buses["pcc"].c * 1.2,
            "sources.grid.r": p.sources["grid"].r * 1.3,
            "units.vsc.ac_filter.l_f": p.unit().ac_filter.l_f * 1.1,
            "units.vsc.dclink.capacitor.c": p.unit().dclink.capacitor.c * 1.2,
            "units.vsc.ctrl.loops.CurrentLoop.bandwidth": 250.0,
        }
    else:
        changes = {
            "units.charger.link.primary_inductance": 110e-6,
            "units.charger.link.mutual_inductance": 25e-6,
            "units.charger.link.primary_resistance": 0.7,
            "units.charger.primary.dclink.source.v": 420.0,
        }
        if name == "wpt-lcc":
            changes["units.charger.link.primary_parallel_resistance"] = 0.05
            changes["units.charger.secondary.ctrl.loops.PhasePowerControl.ki"] = 0.1
            changes["units.charger.secondary.ctrl.loops.PhasePowerControl.power_filter.bandwidth"] = 1500.0
    binary = _compile(p, tmp_path / "model")
    for index, replacements in enumerate(({}, changes)):
        _compare_run(p, binary, replacements, tmp_path / str(index))


@pytest.mark.parametrize("at", [0.0, 0.001])
def test_runtime_load_binding_is_shared_with_a_scheduled_set_event(at, tmp_path):
    p = _case("buck", "pwm_averaging").replace(**{
        "events.load": {"type": "set", "t": at, "set": {"elements.load.r": 5.0}},
    })
    binary = _compile(p, tmp_path / "model", ("elements.load",))
    native, python = tmp_path / "native", tmp_path / "python"
    subprocess.run([binary, "--set", "elements.load.r=9", "--out", native], check=True)
    peslite.Simulation(p.replace(**{"elements.load.r": 9.0})).run(out_dir=python)
    np.testing.assert_allclose(_table(native / "states.csv")[1],
                               _table(python / "states.csv")[1], rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("name, unit, reference", [
    ("buck", "converter", {"voltage": 12.0}),
    ("gfl", "vsc", {"vdc_ref_pu": 1.02}),
])
@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
@pytest.mark.parametrize("variables", [(), ("common",), ("all",)])
def test_initial_control_events_match_direct_configuration_and_python(
        name, unit, reference, mode, variables, tmp_path):
    changes = {f"units.{unit}.ctrl.references.{key}": value for key, value in reference.items()}
    changes[f"units.{unit}.ctrl.loops.DCVoltageControl.ki"] = 160.0
    p = _case(name, mode).replace(**{
        "simulation.t_end": 0.001,
        "simulation.energy_check": "off",
        "events.initial_control": {"type": "set", "t": 0.0, "set": changes},
    })
    binary = _compile(p, tmp_path / "model", variables)
    _compare_run(p, binary, {}, tmp_path / "run")
    events = {key: value for key, value in p.events.items() if key != "initial_control"}
    direct = p.replace(**changes, events=events)
    peslite.Simulation(direct).run(out_dir=tmp_path / "direct")
    np.testing.assert_allclose(_table(tmp_path / "run" / "python" / "states.csv")[1],
                               _table(tmp_path / "direct" / "states.csv")[1],
                               rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("variables", [(), ("all",)])
@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
def test_initial_control_events_preserve_explicit_restart_state(variables, mode, tmp_path):
    p = _case("buck", mode).replace(**{
        "simulation.t_end": 0.001,
        "simulation.initial.t": 0.0004,
        "simulation.initial.states.converter.ctrl.DCVoltageControl.integral": 0.25,
        "simulation.energy_check": "off",
        "events": {
            "first": {"type": "set", "t": 0.0, "set": {
                "units.converter.ctrl.references.voltage": 18.0,
            }},
            "second": {"type": "set", "t": 0.0002, "set": {
                "units.converter.ctrl.references.voltage": 12.0,
                "units.converter.ctrl.loops.DCVoltageControl.ki": 160.0,
            }},
        },
    })
    binary = _compile(p, tmp_path / "model", variables)
    _compare_run(p, binary, {}, tmp_path / "run")
    columns, states = _table(tmp_path / "run" / "native" / "states.csv")
    assert states[0, columns.index("converter.ctrl.DCVoltageControl.integral")] == 0.25


@pytest.mark.parametrize("name, target", [("buck", "converter"), ("gfl", "vsc"), ("wpt", "charger")])
@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
def test_explicit_connection_variable_events_match_python(name, target, mode, tmp_path):
    p = _case(name, mode)
    end = p.simulation.t_end
    p = p.replace(**{"events": {
        "start": {"type": "set", "t": end * 0.2,
                  "set": {f"units.{target}.connected": True}},
        "stop": {"type": "set", "t": end * 0.8,
                 "set": {f"units.{target}.connected": False}},
    }})
    binary = _compile(p, tmp_path / "model")
    _compare_run(p, binary, {}, tmp_path / "run")


@pytest.mark.parametrize("name, target", [("buck", "converter"), ("gfl", "vsc"), ("gfm-dvoc", "vsc"),
                                         ("wpt", "charger.primary")])
@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
def test_initial_connection_uses_the_shared_set_interface(name, target, mode, tmp_path):
    p = _case(name, mode).replace(events={})
    if name in ("gfl", "gfm-dvoc"):
        p = p.replace(**{"sources.grid.angle": 0.2})
    path = f"units.{target}.connected"
    binary = _compile(p, tmp_path / "model", (path,))
    for on in (0, 1):
        _compare_run(p, binary, {path: on}, tmp_path / str(on))
    rejected = subprocess.run([binary, "--set", f"{path}=2"], capture_output=True, text=True)
    assert rejected.returncode != 0 and f"invalid value for {path}" in rejected.stderr


@pytest.mark.parametrize("at", [0.0, 0.001])
def test_initial_connection_does_not_replace_a_scheduled_command(at, tmp_path):
    p = _case("buck", "pwm_averaging").replace(events={
        "start": {"type": "connect", "target": "converter", "t": at},
        "stop": {"type": "disconnect", "target": "converter", "t": 0.002},
    })
    path = "units.converter.connected"
    binary = _compile(p, tmp_path / "model", (path,))
    for on in (0, 1):
        _compare_run(p, binary, {path: on}, tmp_path / str(on))


def test_connection_config_template_keeps_json_values_typed_without_yaml(monkeypatch):
    import builtins
    import json
    from peslite.assembly.cpp import _RuntimeParameters

    runtime = _RuntimeParameters(_case("buck", "pwm_averaging"), ("units.converter.connected",))
    original_import = builtins.__import__

    def without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("YAML is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_yaml)
    body, replacements = runtime.resolved_template()
    assert len(replacements) == 1
    marker, _expression = replacements[0]
    assert json.loads(body.replace(marker, "0"))["units"]["converter"]["connected"] == 0
    assert json.loads(body.replace(marker, "null"))["units"]["converter"]["connected"] is None


def test_source_ramp_keeps_its_command_state_when_controller_startup_is_restored(tmp_path):
    p = _case("gfl", "pwm_averaging").replace(**{
        "events.connect_vsc.ramp": 0.001,
        "simulation.initial.states.vsc.ctrl.startup.ramp": 0.002,
    })
    binary = _compile(p, tmp_path / "model")
    _compare_run(p, binary, {}, tmp_path / "run")


@pytest.mark.parametrize("name", ["gfm-psc", "gfm-dvoc", "gfm-vsg", "gfm-matching"])
@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
@pytest.mark.parametrize("variables", [(), ("all",)])
def test_cold_start_and_reconnect_use_the_same_tracking_law(name, mode, variables, tmp_path):
    p = _case(name, mode).replace(**{
        "sources.grid.angle": 0.2,
        "units.vsc.protection.overcurrent.enable": 0,
        "events": {
            "stop": {"type": "disconnect", "target": "vsc", "t": 0.001},
            "restart": {"type": "connect", "target": "vsc", "t": 0.002},
        },
    })
    binary = _compile(p, tmp_path / "model", variables)
    _compare_run(p, binary, {}, tmp_path / "run")


@pytest.mark.parametrize("voltage", [0.05, 1.0])
@pytest.mark.parametrize("variables", [(), ("all",)])
def test_custom_tracking_callback_is_lowered_without_a_sync_law_adapter(voltage, variables, tmp_path):
    from test_custom_parts import _TrackingOscillator

    p = _case("gfl", "averaging").replace(**{
        "sources.grid.angle": 0.2,
        "sources.grid.v_pu": voltage,
        "units.vsc.protection.overcurrent.enable": 0,
        "units.vsc.ctrl": {
            "type": "custom",
            "loops": {_TrackingOscillator.__name__: {}},
            "connections": {f"{_TrackingOscillator.__name__}.voltage": "meas.u_g"},
            "outputs": {"theta": f"{_TrackingOscillator.__name__}.theta",
                        "omega": f"{_TrackingOscillator.__name__}.omega",
                        "u_dq": f"{_TrackingOscillator.__name__}.command"},
        },
        "events": {
            "stop": {"type": "disconnect", "target": "vsc", "t": 0.001},
            "restart": {"type": "connect", "target": "vsc", "t": 0.002},
        },
    })
    binary = _compile(p, tmp_path / "model", variables)
    _compare_run(p, binary, {}, tmp_path / "run")


@pytest.mark.parametrize("target", ["python", "cpp"])
def test_tracking_callbacks_keep_the_same_output_contract(target, monkeypatch, tmp_path):
    from peslite.control.loops import PSC

    p = _case("gfm-psc", "averaging").replace(events={})
    monkeypatch.setattr(PSC, "track", lambda self, inputs: {})
    with pytest.raises(KeyError, match="theta"):
        if target == "python":
            peslite.Simulation(p).run(out_dir=tmp_path)
        else:
            peslite.export(p, "cpp", tmp_path)


@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
@pytest.mark.parametrize("variables", [(), ("all",)])
def test_restored_control_state_is_not_tracked_again(mode, variables, tmp_path):
    p = _case("gfm-dvoc", mode).replace(**{
        "sources.grid.angle": 0.2,
        "simulation.initial.states.vsc.ctrl.DVOC.theta": 0.7,
        "simulation.initial.states.vsc.ctrl.DVOC.v_mag_pu": 0.8,
        "events": {},
    })
    binary = _compile(p, tmp_path / "model", variables)
    _compare_run(p, binary, {}, tmp_path / "run")


def test_custom_event_variable_bindings_keep_retained_parameter_dependencies(tmp_path):
    from test_custom_parts import _LoadStep

    p = _case("buck", "pwm_averaging").replace(**{"events": {
        "load": {"type": _LoadStep.type, "target": "load", "t": 0.001, "factor": 0.5},
    }})
    binary = _compile(p, tmp_path / "model", ("elements.load.r",))
    _compare_run(p, binary, {"elements.load.r": 9.0}, tmp_path / "run")


@pytest.mark.parametrize("changes", [
    {"units.converter.pwm.f_sw": 25000.0},
    {"units.converter.ctrl.period": 100e-6,
     "units.converter.ctrl.loops.DCVoltageControl.period": 400e-6},
])
def test_runtime_timing_updates_derived_periods_and_loop_strides(changes, tmp_path):
    p = _case("buck", "pwm_averaging")
    binary = _compile(p, tmp_path / "model")
    native, python = tmp_path / "native", tmp_path / "python"
    args = [str(binary), "--out", str(native)]
    for key, value in changes.items():
        args.extend(["--set", f"{key}={value}"])
    subprocess.run(args, check=True)
    expected = p.replace(**changes)
    peslite.Simulation(expected).run(out_dir=python)
    np.testing.assert_allclose(_table(native / "states.csv")[1],
                               _table(python / "states.csv")[1], rtol=2e-6, atol=1e-6)
    recorded = peslite.load(native / "simulation.pes")
    assert recorded.unit().ctrl.period == expected.unit().ctrl.period


def test_runtime_source_parameters_share_the_normal_initial_bus_preset(tmp_path):
    p = _case("gfl", "pwm_averaging")
    binary = _compile(p, tmp_path / "model", ("sources.grid",))
    changes = {"sources.grid.v": 500.0, "sources.grid.angle": 0.2, "sources.grid.f": 51.0}
    native, python = tmp_path / "native", tmp_path / "python"
    args = [str(binary), "--out", str(native)]
    for key, value in changes.items():
        args.extend(["--set", f"{key}={value}"])
    subprocess.run(args, check=True)
    peslite.Simulation(p.replace(**changes)).run(out_dir=python)
    np.testing.assert_allclose(_table(native / "states.csv")[1],
                               _table(python / "states.csv")[1], rtol=3e-6, atol=3e-5)


def test_vsc_measurement_bases_remain_adjustable_after_export(tmp_path):
    p = _case("gfl", "pwm_averaging")
    binary = _compile(p, tmp_path / "model")
    _compare_run(p, binary, {"base.v_ll_rms": 720.0, "units.vsc.dclink.vdc_ref": 1600.0},
                 tmp_path / "run")


def test_runtime_reuses_parameter_validation_instead_of_a_cpp_device_whitelist(tmp_path):
    p = _case("buck", "pwm_averaging")
    binary = _compile(p, tmp_path / "model")
    for path, value in (("elements.load.r", 0), ("buses.output.c", -1),
                        ("units.converter.filter.l_f", 0), ("simulation.solver.dt", 0),
                        ("units.converter.ctrl.loops.DCVoltageControl.minimum", 100)):
        with pytest.raises(ValueError):
            p.replace(**{path: value})
        native = tmp_path / "invalid"
        result = subprocess.run([binary, "--set", f"{path}={value}", "--out", native],
                                capture_output=True, text=True)
        assert result.returncode == 1
        assert not native.exists()



@pytest.mark.parametrize("selection", ["all", "common"])
def test_zero_parallel_capacitance_keeps_the_compiled_structure(selection, tmp_path):
    params = _case("wpt", "averaging")
    binary = _compile(params, tmp_path / "model", (selection,))
    path = "units.charger.primary.bridge.parallel_capacitance"
    baseline = subprocess.run([binary, "--set", f"{path}=0", "--out", tmp_path / "zero"],
                              capture_output=True, text=True)
    assert baseline.returncode == 0
    changed = subprocess.run([binary, "--set", f"{path}=1e-9", "--out", tmp_path / "changed"],
                             capture_output=True, text=True)
    assert changed.returncode == 1
    assert "changed structural specialization" in changed.stderr
    assert not (tmp_path / "changed").exists()


@pytest.mark.parametrize("name,mode,selection", [
    ("buck", "switching", "all"),
    ("wpt", "averaging", "all"),
    ("gfl", "pwm_averaging", "common"),
])
def test_finite_bridge_parameters_use_the_shared_native_equations(name, mode, selection, tmp_path):
    from peslite.assembly.params import _unit_items

    p = _case(name, mode)
    names = tuple(unit for unit, _ in _unit_items(p.units))
    changes = {"simulation.t_end": 30e-6, "simulation.solver.dt": 0.25e-6}
    for unit in names:
        changes.update({
            f"units.{unit}.bridge.parallel_capacitance": 1e-5,
            f"units.{unit}.bridge.on_resistance": 1.0,
            f"units.{unit}.protection.overcurrent.enable": False,
        })
    p = p.replace(**changes)
    binary = _compile(p, tmp_path / "model", (selection,))
    unit = names[0]
    _compare_run(p, binary, {f"units.{unit}.bridge.parallel_capacitance": 1.2e-5,
                             f"units.{unit}.bridge.on_resistance": 1.1}, tmp_path / "run")


def test_parameter_selection_rejects_structural_paths_before_writing(tmp_path):
    p = _case("buck", "pwm_averaging")
    for path in ("units.converter.type", "units.converter.bridge.model", "buses.output.phases"):
        with pytest.raises(ValueError, match="not variable"):
            peslite.export(p, "cpp", tmp_path / "invalid", variables=[path])
        assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("per_unit", [False, True])
def test_initial_override_preserves_originally_redundant_load_assignment(per_unit, tmp_path):
    p = _case("buck", "pwm_averaging")
    path = "elements.load.r_pu" if per_unit else "elements.load.r"
    value = p.elements["load"].r / p.base.z_base if per_unit else p.elements["load"].r
    p = p.replace(**{
        "events.load": {"type": "set", "t": 0.001, "set": {path: value}},
    })
    binary = _compile(p, tmp_path / "model")
    changes = {"elements.load.r": 9.0}
    if per_unit:
        changes["base.s_base"] = 1.2 * p.base.s_base
    _compare_run(p, binary, changes, tmp_path / "run")


def test_source_assignments_preserve_unassigned_fields_and_phase_continuity(tmp_path):
    p = _case("gfl", "pwm_averaging")
    p = p.replace(**{
        "events.voltage": {"type": "set", "t": 0.001,
                           "set": {"sources.grid.v": p.sources["grid"].v}},
        "events.frequency": {"type": "set", "t": 0.002,
                             "set": {"sources.grid.f": 49.0}},
    })
    binary = _compile(p, tmp_path / "model")
    _compare_run(p, binary, {"sources.grid.v": 500.0, "sources.grid.f": 51.0,
                             "sources.grid.angle": 0.2}, tmp_path / "run")


def test_retained_source_preserves_prestart_phase_history_and_explicit_bus_state(tmp_path):
    p = _case("gfl", "pwm_averaging").replace(**{
        "simulation.initial.t": 0.0015,
        "simulation.initial.states.pcc.u_C": 400.0,
        "events.frequency": {"type": "set", "t": 0.001,
                             "set": {"sources.grid.f": 49.0}},
    })
    binary = _compile(p, tmp_path / "model")
    _compare_run(p, binary, {"sources.grid.f": 51.0, "sources.grid.angle": 0.2},
                 tmp_path / "run")


@pytest.mark.parametrize("name", ["boost", "buck-boost"])
@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_other_dc_connections_share_retained_physical_parameters(name, mode, tmp_path):
    p = _case(name, mode)
    changes = {"units.converter.filter.l_f": 700e-6,
               "units.converter.filter.r_f": 0.08,
               "units.converter.ctrl.loops.InductorCurrentControl.ki": 30.0}
    if name == "boost":
        changes.update({"elements.supply.v": 26.0,
                        "units.converter.dclink.capacitor.c": 600e-6})
    else:
        changes.update({"elements.load.r": 15.0, "buses.output.c": 600e-6})
    binary = _compile(p, tmp_path / "model")
    _compare_run(p, binary, changes, tmp_path / "run")


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging"])
def test_retained_carrier_phase_uses_normal_startup_clock(mode, tmp_path):
    p = _case("buck", mode)
    binary = _compile(p, tmp_path / "model")
    _compare_run(p, binary, {"units.converter.pwm.carrier_phase": 0.25}, tmp_path / "run")


def test_retained_clock_and_supply_reinitialize_the_adc_window_before_first_sample(tmp_path):
    p = _case("buck", "pwm_averaging").replace(**{"units.converter.meas.average": ["vin"]})
    binary = _compile(p, tmp_path / "model")
    _compare_run(p, binary, {"units.converter.pwm.carrier_phase": 0.25,
                             "units.converter.dclink.source.v": 52.0}, tmp_path / "run")


def test_retained_rms_periods_preserve_fixed_history_dimensions(tmp_path):
    from peslite.assembly.params import from_dict
    from test_measurement_network import load_case

    tree = load_case()
    tree["simulation"]["output"].update(states=True, energy=True)
    p = from_dict(tree)
    binary = _compile(p, tmp_path / "model")
    changes = {"units.charger.primary.pwm.f_sw": 1e6 / 3,
               "units.charger.primary.meas.rms.current": 36e-6}
    _compare_run(p, binary, changes, tmp_path / "run")
    output = tmp_path / "invalid"
    rejected = subprocess.run([binary, "--set", "units.charger.primary.meas.rms.current=24e-6",
                               "--out", output], capture_output=True, text=True)
    assert rejected.returncode == 1
    assert not output.exists()


def test_registered_element_uses_schema_and_ordinary_constructor_without_binding_table(tmp_path):
    from test_cpp_export import _ConditionalElement

    p = _case("gfl", "pwm_averaging").replace(**{
        "simulation.energy_check": "off",
        "elements.custom": {"type": _ConditionalElement.type, "bus": "pcc",
                            "inductance": 0.01, "resistance_low": 1.0,
                            "resistance_high": 2.0, "threshold": 5.0},
    })
    binary = _compile(p, tmp_path / "model", ("elements.custom",))
    _compare_run(p, binary, {"elements.custom.inductance": 0.02,
                             "elements.custom.threshold": 0.1}, tmp_path / "run")


def test_unrecorded_factory_mutation_cannot_be_silently_discarded(tmp_path):
    def make_system(params):
        system = peslite.System(params)
        branch = system.units["converter"].branch_f
        branch.retune(1.8 * branch.L, branch.R)
        return system

    simulation = peslite.Simulation(_case("buck", "pwm_averaging"), parts={"system": make_system})
    destination = tmp_path / "invalid"
    with pytest.raises(TypeError, match="factory changed this initialized parameter"):
        peslite.export(simulation, "cpp", destination, variables=["all"])
    assert not destination.exists()


def test_controller_factory_shares_parameterized_construction(tmp_path):
    class InheritedController(peslite.VSCController):
        pass

    p = _case("gfl", "pwm_averaging")
    simulation = peslite.Simulation(p, parts={"vsc.ctrl": InheritedController})
    binary = _compile(simulation, tmp_path / "model")
    _compare_run(p, binary, {"units.vsc.ctrl.loops.CurrentLoop.bandwidth": 250.0},
                 tmp_path / "run")


def test_modulator_factory_remains_parameterized_inside_the_shared_bridge(tmp_path):
    from peslite.components import CarrierComparison

    p = _case("gfl", "switching")
    simulation = peslite.Simulation(p, parts={
        "vsc.modulator": lambda cfg: CarrierComparison(cfg.pwm.f_sw),
    })
    binary = _compile(simulation, tmp_path / "model")
    _compare_run(p, binary, {"units.vsc.pwm.f_sw": 25000.0,
                             "units.vsc.protection.rocof.window": 0.08}, tmp_path / "run")


def test_dc_current_supply_shares_retained_values_with_its_runtime_ramp(tmp_path):
    p = _case("gfl", "pwm_averaging")
    binary = _compile(p, tmp_path / "model")
    _compare_run(p, binary, {"units.vsc.dclink.source.i": 0.8 * p.unit().dclink.source.i,
                             "units.vsc.dclink.source.k": 0.1}, tmp_path / "run")


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_zero_coefficient_fast_path_keeps_real_feedback_when_parameters_change(mode, tmp_path):
    p = _case("buck", mode)
    binary = _compile(p, tmp_path / "model")
    source = (tmp_path / "model/peslite.cpp").read_text(encoding="utf-8")
    direct = source.split("inline void evaluate(", 1)[1].split(
        "PESLITE_NOINLINE void evaluate_feedback(", 1)[0]
    assert "solve_algebraic(" not in direct
    assert direct.index("evaluate_feedback(t, y, dy)") < direct.index("= y[")
    # Every independent combination must retain the shared iterative path when needed.
    for index, changes in enumerate(({}, {"buses.output.r_d": 0.05},
                                     {"units.converter.dclink.source.r": 0.1},
                                     {"buses.output.r_d": 0.05,
                                      "units.converter.dclink.source.r": 0.1})):
        _compare_run(p, binary, changes, tmp_path / str(index))


def test_retained_recording_intervals_do_not_fix_the_export_time_maximum_branch(tmp_path):
    p = _case("buck", "pwm_averaging")
    binary = _compile(p, tmp_path / "model")
    changes = {"simulation.output.record_every": 0, "simulation.solver.phs_check_step": 1}
    _compare_run(p, binary, changes, tmp_path / "run")
    filename = "ctrl.converter.csv"
    np.testing.assert_allclose(_table(tmp_path / "run/native" / filename)[1],
                               _table(tmp_path / "run/python" / filename)[1],
                               rtol=2e-8, atol=2e-10)


def test_retained_clock_cannot_silently_reuse_a_restored_switching_sequence(tmp_path):
    p = _case("gfl", "switching").replace(**{
        "simulation.initial.states.vsc.pwm.d_a": 0.7,
        "simulation.initial.states.vsc.pwm.d_b": 0.4,
        "simulation.initial.states.vsc.pwm.d_c": 0.3,
        "simulation.initial.states.vsc.pwm.on": 1,
    })
    binary = _compile(p, tmp_path / "model", ("units.vsc.pwm",))
    output = tmp_path / "invalid"
    rejected = subprocess.run([binary, "--set", "units.vsc.pwm.f_sw=15000", "--out", output],
                              capture_output=True, text=True)
    assert rejected.returncode == 1
    assert not output.exists()
    _compare_run(p, binary, {}, tmp_path / "unchanged")
