"""Per-unit exact-switching, PWM-period-averaged and ideal averaged bridge models."""

import io
import warnings
from contextlib import redirect_stdout

import numpy as np
import pytest
import yaml

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import ConfigError


SHORT = ["--set", "simulation.t_end=0.004",
         "--set", "simulation.solver.linearisations=0"]


def _resolved(*args):
    """Return the resolved tree after applying command-line arguments."""
    output = io.StringIO()
    with redirect_stdout(output):
        assert peslite.main([*args, "--resolved"]) == 0
    return yaml.safe_load(output.getvalue())


def _models(*args):
    """Return ``{unit: bridge model}`` after resolving command-line arguments."""
    units = _resolved(*args)["units"]
    return {name: unit["bridge"]["model"] for name, unit in units.items()}


def test_pwm_averaging_is_the_default_bridge_model():
    assert _models("gfl-example") == {"vsc": "pwm_averaging"}
    assert _models("gfm-psc-example") == {"vsc": "pwm_averaging"}


@pytest.mark.parametrize("name", ["gfl-example", "gfm-psc-example", "two-converters-example"])
def test_switching_flag_selects_every_unit(name):
    assert set(_models(name, "--switching").values()) == {"switching"}


@pytest.mark.parametrize("name", ["gfl-example", "gfm-psc-example", "two-converters-example"])
def test_pwm_averaging_flag_selects_every_unit(name):
    assert set(_models(name, "--pwm-averaging").values()) == {"pwm_averaging"}


@pytest.mark.parametrize("name", ["gfl-example", "gfm-psc-example", "two-converters-example"])
def test_averaging_flag_selects_every_unit(name):
    assert set(_models(name, "--averaging").values()) == {"averaging"}


@pytest.mark.parametrize(("flag", "expected"), [
    (None, ("fixed", "rk4")),
    ("--switching", ("fixed", "rk4")),
    ("--pwm-averaging", ("fixed", "rk4")),
    ("--averaging", ("adaptive", "DP45")),
])
def test_bridge_flags_are_solver_presets(flag, expected):
    solver = _resolved("gfl-example", *([flag] if flag else []))["simulation"]["solver"]
    assert (solver["type"], solver["method"]) == expected
    assert solver["rtol"] == 1e-6


def test_averaging_flag_keeps_an_explicit_solver_choice():
    solver = _resolved(
        "gfl-example", "--averaging",
        "--set", "simulation.solver.type=fixed",
        "--set", "simulation.solver.method=heun",
    )["simulation"]["solver"]
    assert (solver["type"], solver["method"]) == ("fixed", "heun")


@pytest.mark.parametrize("flag", ["--switching", "--pwm-averaging"])
def test_set_can_override_fixed_solver_presets(flag):
    solver = _resolved(
        "gfl-example", flag,
        "--set", "simulation.solver.type=adaptive",
        "--set", "simulation.solver.method=DP45",
    )["simulation"]["solver"]
    assert (solver["type"], solver["method"]) == ("adaptive", "DP45")


def test_averaging_preset_overrides_file_solver_and_fixed_only_settings(tmp_path):
    tree = yaml.safe_load((EXAMPLES / "gfl-example.pes").read_text(encoding="utf-8"))
    tree["simulation"]["solver"].update({
        "type": "fixed", "method": "heun", "subsystems": {"vsc.dclink": 2},
    })
    path = tmp_path / "configured.pes"
    path.write_text(yaml.safe_dump(tree), encoding="utf-8")

    solver = _resolved(str(path), "--averaging")["simulation"]["solver"]
    assert (solver["type"], solver["method"], solver["subsystems"]) == (
        "adaptive", "DP45", {},
    )


def test_bridge_flags_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit):
        peslite.main(["gfl-example", "--switching", "--averaging", "--resolved"])
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.full
@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*-example.pes")), ids=lambda path: path.stem)
def test_every_bundled_file_runs_with_its_configured_model(path, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        params = peslite.load(path, **{"simulation.solver.linearisations": 0})
        duration = min(0.004, params.simulation.t_end)
        params = params.replace(**{"simulation.t_end": duration,
                                   "simulation.output.states": True})
    result = peslite.Simulation(params).run(out_dir=tmp_path / path.stem)
    assert np.isfinite(result.states["t"]).all()
    assert result.states["t"][-1] == pytest.approx(duration)


def test_units_can_have_different_models(tmp_path):
    params = peslite.load(
        EXAMPLES / "two-converters-example.pes",
        **{"simulation.t_end": 0.004,
           "simulation.solver.linearisations": 0,
           "units.vsc_1.bridge.model": "pwm_averaging",
           "units.vsc_2.bridge.model": "switching"},
    )
    simulation = peslite.Simulation(params)
    models = tuple(type(simulation.units[name].bridge.modulator).__name__
                   for name in ("vsc_1", "vsc_2"))
    assert models == ("ZOH", "CarrierComparison")
    simulation.run(out_dir=tmp_path)


def test_ideal_averaging_is_a_bridge_implementation_without_a_pwm(tmp_path):
    params = peslite.load(
        EXAMPLES / "two-converters-example.pes",
        **{"simulation.t_end": 0.004,
           "simulation.solver.linearisations": 0,
           "units.vsc_1.bridge.model": "averaging",
           "units.vsc_2.bridge.model": "switching"},
    )
    simulation = peslite.Simulation(params)
    assert not isinstance(simulation.units["vsc_1"].bridge, peslite.PWM)
    assert isinstance(simulation.units["vsc_1"].bridge, peslite.AveragingModel)
    assert not hasattr(simulation.units["vsc_1"], "pwm")
    assert not hasattr(simulation.units["vsc_1"], "averaging")
    assert type(simulation.units["vsc_2"].bridge.modulator).__name__ == "CarrierComparison"
    simulation.run(out_dir=tmp_path)


def test_changing_bridge_model_keeps_the_same_bridge_power_boundary(gfl):
    switching = peslite.Simulation(gfl()).unit()
    averaging = peslite.Simulation(
        gfl(**{"units.vsc.bridge.model": "averaging"})
    ).unit()

    assert tuple(switching.bridge.inp.__slots__) == tuple(averaging.bridge.inp.__slots__)
    assert set(switching.bridge.inp.__slots__) == {
        "q", "u_dc", "i_c",
    }
    assert [port for _component, port in switching.zoh_connections()] == ["q"]
    assert averaging.zoh_connections() == {}
    assert (averaging.bridge, "q") in averaging.connections()
    assert averaging.connections()[(averaging.ctrl, "u_dc")] == (averaging.dclink, "u_dc")
    for unit in (switching, averaging):
        assert unit.bridge.dc_terminal in [terminal for _bus, terminal in unit.terminals()]
        assert unit.bridge.power_terminal in [terminal for _bus, terminal in unit.terminals()]


def test_only_exact_switching_bridge_enters_the_switching_event_loop(gfl):
    bridges = {
        model: peslite.Simulation(gfl(**{"units.vsc.bridge.model": model})).unit().bridge
        for model in ("switching", "pwm_averaging", "averaging")
    }
    assert bridges["switching"].has_switching_events
    assert not bridges["pwm_averaging"].has_switching_events
    assert not bridges["averaging"].has_switching_events
    assert not hasattr(bridges["averaging"], "next_switch")


def test_averaging_silently_ignores_all_sampled_timing_semantics(tmp_path):
    common = {
        "units.vsc.bridge.model": "averaging",
        "events.connect_vsc.t": 0.0,
        "events.connect_vsc.ramp": 0.0,
        "simulation.t_end": 0.001,
        "simulation.solver.linearisations": 0,
        "units.vsc.meas.period": 7e-6,  # not ignored: the unit's protection clock
    }
    baseline = peslite.load(EXAMPLES / "gfl-example.pes", **common)
    configured = peslite.load(EXAMPLES / "gfl-example.pes", **common, **{
        "units.vsc.ctrl.period": 37e-6,
        "units.vsc.ctrl.computation": 1.0,
        "units.vsc.ctrl.loops.SRFPLL.period": 31e-6,
        "units.vsc.ctrl.loops.CurrentLoop.period": 43e-6,
        "units.vsc.ctrl.loops.DCVoltageControl.period": 59e-6,
        "units.vsc.meas.window": 0.2,
        "units.vsc.meas.average": ["u_g", "i_c", "u_dc"],
        "units.vsc.pwm.f_sw": 19_999.0,
        "units.vsc.pwm.method": "svpwm",
        "units.vsc.pwm.modulation_limit": 0.2,
        "units.vsc.pwm.sync": "synchronous",
        "units.vsc.pwm.update": "double",
        "units.vsc.pwm.carrier_phase": 0.37,
    })

    first = peslite.Simulation(baseline).run(out_dir=tmp_path / "baseline")
    second_sim = peslite.Simulation(configured)
    assert second_sim.unit().adc is None
    assert second_sim.unit().clock.period == 7e-6
    assert second_sim.unit().periods == []
    assert second_sim.unit().ctrl.periods == {
        "SRFPLL": 31e-6,
        "CurrentLoop": 43e-6,
        "DCVoltageControl": 59e-6,
    }
    second = second_sim.run(out_dir=tmp_path / "configured")
    for name, value in first.final_states().items():
        assert second.final_states()[name] == pytest.approx(value, rel=1e-13, abs=1e-13)


def test_pwm_averaging_flag_is_the_same_as_setting_the_file_value(tmp_path):
    out_flag, out_set = tmp_path / "flag", tmp_path / "set"
    with redirect_stdout(io.StringIO()):
        assert peslite.main(["gfl-example", "--pwm-averaging", *SHORT,
                             "--out", str(out_flag)]) == 0
        assert peslite.main(["gfl-example", "--set", "units.vsc.bridge.model=pwm_averaging",
                             *SHORT, "--out", str(out_set)]) == 0
    assert (out_flag / "states.csv").read_text() == (out_set / "states.csv").read_text()
    assert peslite.load(out_flag / "simulation.pes").unit("vsc").bridge.model == "pwm_averaging"


def test_set_wins_over_pwm_averaging_flag():
    assert _models("gfl-example", "--pwm-averaging",
                   "--set", "units.vsc.bridge.model=switching") == {
                       "vsc": "switching"}


def test_set_wins_over_switching_flag():
    assert _models("gfl-example", "--switching",
                   "--set", "units.vsc.bridge.model=averaging") == {
                       "vsc": "averaging"}


def test_set_wins_over_averaging_flag():
    assert _models("gfl-example", "--averaging",
                   "--set", "units.vsc.bridge.model=switching") == {
                       "vsc": "switching"}


def test_pwm_averaging_flag_results_use_their_own_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(peslite.solver.simulation, "_RESULTS", tmp_path)
    with redirect_stdout(io.StringIO()):
        assert peslite.main(["gfl-example", *SHORT]) == 0
        assert peslite.main(["gfl-example", "--switching", *SHORT]) == 0
        assert peslite.main(["gfl-example", "--pwm-averaging", *SHORT]) == 0
        assert peslite.main(["gfl-example", "--averaging", *SHORT]) == 0
        assert peslite.main(["gfm-psc-example", "--pwm-averaging", *SHORT]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "gfl-example", "gfl-example-averaging", "gfl-example-pwm-averaging",
        "gfl-example-switching", "gfm-psc-example-pwm-averaging"]
    assert (peslite.load(tmp_path / "gfl-example" / "simulation.pes")
            .unit("vsc").bridge.model == "pwm_averaging")
    assert (peslite.load(tmp_path / "gfl-example-switching" / "simulation.pes")
            .unit("vsc").bridge.model == "switching")
    assert (peslite.load(tmp_path / "gfl-example-pwm-averaging" / "simulation.pes")
            .unit("vsc").bridge.model == "pwm_averaging")
    assert (peslite.load(tmp_path / "gfl-example-averaging" / "simulation.pes")
            .unit("vsc").bridge.model == "averaging")


@pytest.mark.parametrize("solver", [
    {},
    {"simulation.solver.type": "adaptive", "simulation.solver.method": "DP45"},
], ids=["fixed", "adaptive"])
def test_pwm_averaging_supports_fixed_and_adaptive_solvers(gfl, solver, tmp_path):
    params = gfl(**solver)
    result = peslite.Simulation(params).run(out_dir=tmp_path / ("adaptive" if solver else "fixed"))
    assert result.t[-1] == pytest.approx(params.simulation.t_end)


def test_pwm_averaging_accepts_synchronous_pwm(gfl, tmp_path):
    params = gfl(**{"units.vsc.pwm.sync": "synchronous",
                    "units.vsc.pwm.carrier_phase": 0.3})
    result = peslite.Simulation(params).run(out_dir=tmp_path)
    assert result.t[-1] == pytest.approx(params.simulation.t_end)


def test_ideal_averaging_has_continuous_control_states_and_no_sampled_hardware_states(gfl):
    simulation = peslite.Simulation(gfl(**{"units.vsc.bridge.model": "averaging"}))
    names = set(simulation.state_names())
    assert {"vsc.ctrl.SRFPLL.theta", "vsc.ctrl.SRFPLL.integral_pu",
            "vsc.ctrl.CurrentLoop.integral_pu.re",
            "vsc.ctrl.CurrentLoop.integral_pu.im"} <= names
    assert not any(name.startswith(("vsc.pwm.", "vsc.bridge.", "vsc.meas."))
                   for name in names)


def test_ideal_averaging_algebraic_loop_runs_with_a_multirate_group(gfl, tmp_path):
    params = gfl(**{
        "units.vsc.bridge.model": "averaging",
        "simulation.t_end": 0.001,
        "simulation.solver.subsystems": {"vsc.dclink": 2},
    })
    simulation = peslite.Simulation(params)

    result = simulation.run(out_dir=tmp_path)

    assert len(simulation.system.model.algebraic_loops) == 1
    assert result.summary["t_stop"] == pytest.approx(params.simulation.t_end)
    assert all(np.isfinite(value) for value in result.final_states().values())


@pytest.mark.parametrize("solver", [
    {},
    {"simulation.solver.type": "adaptive", "simulation.solver.method": "DP45"},
], ids=["fixed", "adaptive"])
def test_ideal_averaging_continues_exactly_with_its_continuous_control_states(gfl, solver, tmp_path):
    params = gfl(**{"units.vsc.bridge.model": "averaging",
                    "simulation.t_end": 0.004,
                    "simulation.output.period": 5e-6,
                    **solver})
    whole_dir = tmp_path / "whole"
    whole = peslite.Simulation(params).run(out_dir=whole_dir)
    continued_params = peslite.load(whole_dir / "simulation.pes",
                                    initial=whole_dir / "states.csv",
                                    initial_time=0.003005)
    continued = peslite.Simulation(continued_params).run(out_dir=tmp_path / "continued")
    start = int(np.argmin(abs(whole.t - 0.003005)))
    assert np.array_equal(continued.t, whole.t[start:])
    for name, values in continued.states.items():
        np.testing.assert_allclose(values, whole.states[name][start:], rtol=1e-13, atol=1e-13,
                                   err_msg=name)
