"""Output names and values follow the simulation-file SI/pu and null rules."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import peslite
from peslite.solver.model import ConfigError


UNIT_SUFFIX = re.compile(r"_(s|hz|hz_s|J|rad)$")


def _run(params, out):
    return peslite.Simulation(params).run(out_dir=out)


def test_run_uses_a_default_output_directory(gfl, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = peslite.Simulation(gfl()).run()
    assert result.out_dir == Path("output/run")
    assert (tmp_path / result.out_dir / "states.csv").is_file()
    assert (tmp_path / result.out_dir / "summary.json").is_file()


def test_run_without_trip_uses_none_and_lists_for_absent_results(gfl, tmp_path):
    result = _run(gfl(), tmp_path)
    summary = result.summary
    assert (summary["tripped"], summary["vsc.tripped"]) == (0, 0)
    assert summary["vsc.trip_time"] is None and summary["vsc.trip_cause"] is None
    assert summary["vsc.alarms"] == []
    assert summary["vsc.modulation_saturation_first_t"] is None
    for criterion in ("overcurrent", "undervoltage", "overvoltage", "frequency",
                      "dc_voltage", "rocof"):
        assert summary[f"vsc.{criterion}_first_t"] is None
    assert summary["t_start"] == 0.0 and summary["t_stop"] == pytest.approx(0.004)

    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["vsc.trip_time"] is None
    assert isinstance(saved["vsc.alarms"], list)
    assert isinstance(saved["energy_problems"], list)
    assert "wall_time" in saved and "wall_time_s" not in saved


def test_trip_reports_time_cause_alarm_and_first_crossing(gfl, tmp_path):
    result = _run(gfl(**{
        "units.vsc.protection.overcurrent.limit_pu": 1e-3,
        "events.connect_vsc.ramp": 0,
        "simulation.t_end": 0.002,
    }), tmp_path)
    summary = result.summary
    assert (summary["tripped"], summary["vsc.tripped"], summary["vsc.trip_cause"]) == (
        1, 1, "overcurrent"
    )
    assert summary["vsc.trip_time"] == pytest.approx(summary["vsc.overcurrent_first_t"])
    assert "OVERCURRENT" in summary["vsc.alarms"]


def test_state_control_and_summary_names_have_no_unit_suffix(gfl, tmp_path):
    result = _run(gfl(**{"simulation.output.signals": 1}), tmp_path)
    names = [*result.summary, *result.states, *result.ctrl, *result.columns()]
    assert [name for name in names if UNIT_SUFFIX.search(name)] == []


def test_controller_states_in_pu_are_named_as_pu(examples):
    gfl_names = peslite.Simulation(peslite.load(examples / "gfl-example.pes")).state_names()
    psc_names = peslite.Simulation(peslite.load(
        examples / "gfm-psc-example.pes",
        **{"units.vsc.ctrl.loops.power.filter.enable": 1,
           "units.vsc.ctrl.loops.power.filter.bandwidth": 5.0},
    )).state_names()
    assert {
        "vsc.ctrl.pll.integral_pu",
        "vsc.ctrl.dvc.integral",
        "vsc.ctrl.cc.integral_pu.re",
        "vsc.tripped",
    } <= set(gfl_names)
    assert {
        "vsc.ctrl.power.p_pu",
        "vsc.ctrl.power.q_pu",
        "vsc.ctrl.sync.v_int_pu",
    } <= set(psc_names)
    assert not {
        "vsc.ctrl.pll.integral",
        "vsc.ctrl.power.p",
        "vsc.breaker_open",
    } & set(gfl_names + psc_names)
    assert not any(name.startswith(("plant.", "ctrl.", "pwm.", "meas."))
                   for name in gfl_names + psc_names)


def test_grid_following_and_grid_forming_logs_share_names(gfl, examples, tmp_path):
    result_gfl = _run(gfl(**{"simulation.output.signals": 1}), tmp_path / "gfl")
    result_gfm = _run(peslite.load(
        examples / "gfm-psc-example.pes",
        **{"simulation.t_end": 0.004, "events.connect_vsc.t": 0.0,
           "simulation.solver.linearisations": 0, "simulation.output.signals": 1},
    ), tmp_path / "gfm")
    for result in (result_gfl, result_gfm):
        assert {"vsc.freq_dev", "vsc.angle_rel", "vsc.vac_pu"} <= set(result.ctrl)
    assert "vsc.id_ref_pu" in result_gfl.ctrl
    assert "vsc.v_mag_pu" not in result_gfm.ctrl
    assert not hasattr(result_gfl, "control")


def test_ctrl_output_uses_only_the_abbreviated_name(gfl, tmp_path):
    result = _run(gfl(**{"simulation.output.signals": 1}), tmp_path)
    assert tmp_path / "ctrl.vsc.csv" in result.files
    assert (tmp_path / "ctrl.vsc.csv").is_file()
    assert not (tmp_path / "control.vsc.csv").exists()


def test_continuation_uses_the_renamed_states(gfl, tmp_path):
    first_dir = tmp_path / "first"
    result = _run(gfl(**{"simulation.t_end": 0.004}), first_dir)
    restarted = peslite.load(
        first_dir / "simulation.pes",
        initial=first_dir / "states.csv",
        initial_time=0.002,
    )
    continued = _run(restarted, tmp_path / "continued")
    index = int(np.argmin(abs(result.states["t"] - 0.003)))
    continued_index = int(np.argmin(abs(continued.states["t"] - 0.003)))
    assert continued.states["vsc.ctrl.pll.integral_pu"][continued_index] == pytest.approx(
        result.states["vsc.ctrl.pll.integral_pu"][index], rel=1e-9
    )


def test_disabled_state_output_collects_only_the_final_row(gfl, tmp_path):
    simulation = peslite.Simulation(gfl(**{
        "simulation.output.period": 5e-5,
        "simulation.output.states": 0,
    }))
    read_flat = simulation._states.read_flat
    calls = 0

    def counted():
        nonlocal calls
        calls += 1
        return read_flat()

    simulation._states.read_flat = counted
    result = simulation.run(out_dir=tmp_path)
    assert calls == 1
    assert result.states == {}
    assert result.final_states()["t"] == pytest.approx(result.summary["t_stop"])

    assert tmp_path / "states.csv" not in result.files
    assert not (tmp_path / "states.csv").exists()


def test_numeric_histories_are_streamed_and_energy_history_is_optional(gfl, tmp_path):
    without_energy = _run(gfl(**{
        "simulation.output.period": 5e-5,
        "simulation.output.signals": 1,
        "simulation.output.energy": 0,
        "simulation.solver.write_length": 3,
    }), tmp_path / "without-energy")
    assert without_energy.energy == {}
    assert "energy_balance_max_rel" in without_energy.summary
    histories = [without_energy.states, without_energy.plant, without_energy.ctrl]
    assert all(isinstance(values, np.ndarray) for history in histories for values in history.values())
    assert without_energy._records.plant_table.path.is_file()
    assert without_energy._records.states_table.path.is_file()
    assert without_energy._records.plant_table.batch_rows == 3
    assert without_energy._records.states_table.batch_rows == 3
    assert without_energy._records.plant_table._pending == []
    assert without_energy._records.states_table._pending == []

    with_energy = _run(gfl(**{
        "simulation.output.period": 5e-5,
        "simulation.output.signals": 1,
        "simulation.output.energy": 1,
        "simulation.solver.phs_check_step": 1,
    }), tmp_path / "with-energy")
    assert len(with_energy.energy["t"]) == len(with_energy.t)
    assert all(isinstance(values, np.ndarray) for values in with_energy.energy.values())
    assert with_energy._records.plant_table.batch_rows == 1000

    default_energy_checks = _run(gfl(**{
        "simulation.t_end": 0.002,
        "simulation.output.period": 1e-6,
        "simulation.output.energy": 1,
    }), tmp_path / "default-energy-checks")
    assert default_energy_checks.energy["t"] == pytest.approx([0.0, 0.001, 0.002])

    with pytest.raises(ConfigError, match="solver.write_length must be >= 1"):
        gfl(**{"simulation.solver.write_length": 0})
    with pytest.raises(ConfigError, match="solver.phs_check_step must be >= 1"):
        gfl(**{"simulation.solver.phs_check_step": 0})


def test_phs_check_step_does_not_change_the_simulation(gfl, tmp_path):
    base = {"simulation.t_end": 0.002, "simulation.output.period": 5e-6}
    dense = _run(gfl(**base, **{"simulation.solver.phs_check_step": 1}),
                 tmp_path / "dense-phs-checks")
    sparse = _run(gfl(**base, **{"simulation.solver.phs_check_step": 1000}),
                  tmp_path / "sparse-phs-checks")
    assert dense.n_rhs == sparse.n_rhs
    for name, values in dense.states.items():
        assert np.array_equal(values, sparse.states[name]), name
