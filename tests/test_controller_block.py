"""Sampled controller, converter lifecycle and protection boundaries."""

import cmath
from dataclasses import replace

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import (DCVoltageLimitParams, FrequencyLimitParams, OvercurrentParams,
                                     ProtectionParams, VoltageLimitParams)
from peslite.control import Measurement, Startup
from peslite.control.blocks import smoothstep


T = 5e-5


def _run(params, path):
    return peslite.Simulation(params).run(out_dir=path)


def _event(name, **event):
    return {f"events.{name}": event}


def _current(result):
    return np.abs(result.states["vsc.branch_f.i.re"]
                  + 1j * result.states["vsc.branch_f.i.im"])


def test_startup_counts_only_controller_interrupts():
    startup = Startup(T=0.25, run=False)
    startup.advance()
    assert (startup.active, startup.value, startup.complete) == (False, 0.0, False)
    startup.command(True, 1.0)
    seen = []
    for _ in range(6):
        startup.advance()
        seen.append((startup.active, startup.value, startup.complete, startup.steps))
    assert seen == [(True, smoothstep(x), x >= 1.0, min(4, n + 1))
                    for n, x in enumerate((0.0, 0.25, 0.5, 0.75, 1.0, 1.0))]
    startup.set_state({"steps": 100})
    assert startup.steps == 4
    startup.command(False)
    assert (startup.active, startup.steps, startup.value, startup.complete) == (False, 0, 0.0, False)


def test_completed_startup_is_not_advanced_again():
    startup = Startup(T=0.25, run=True, ramp=0.5)
    while not startup.complete:
        startup.advance()
    held = startup.get_state(), startup.value, startup.complete
    assert not startup.in_progress
    for _ in range(10):
        startup.advance()
    assert (startup.get_state(), startup.value, startup.complete) == held


def test_controller_needs_only_samples_and_host_commands(gfl):
    cfg = gfl().unit("vsc")
    controller = peslite.VSCController(cfg)
    assert not hasattr(controller, "scenario")
    assert not hasattr(controller, "protection")
    voltage, w0 = cfg.base.v_phase_peak, cfg.base.w0

    def sample(k):
        return Measurement(k * T, voltage * cmath.exp(1j * w0 * k * T), 0j,
                           cfg.dclink.vdc_ref, np.zeros(3))

    controller.command(False)
    output = controller(0.0, sample(0))
    assert not output.gates and not output.startup_complete and output.log["in_service"] == 0.0
    assert not hasattr(output, "tripped") and not hasattr(output, "trip_cause")
    controller.command(True, 4 * T)
    armed = []
    for k in range(1, 8):
        output = controller(k * T, sample(k))
        armed.append(controller.startup.complete)
        assert output.gates and np.all((0.0 <= output.d_abc) & (output.d_abc <= 1.0))
    assert armed == [False, False, False, False, True, True, True]
    controller.command(False)
    assert not controller(8 * T, sample(8)).gates


def test_pwm_start_clears_error_integrators_once(gfl):
    controller = peslite.VSCController(gfl().unit("vsc"))
    pll = controller.graph.nodes["SRFPLL"]
    dvc = controller.graph.nodes["DCVoltageControl"]
    cc = controller.graph.nodes["CurrentLoop"]
    pll.integral, dvc.integral, cc.integral = 1.0, 2.0, 3.0 + 4.0j

    controller.command(False)
    assert (pll.integral, dvc.integral, cc.integral) == (1.0, 2.0, 3.0 + 4.0j)
    controller.command(True)
    assert (pll.integral, dvc.integral, cc.integral) == (0.0, 0.0, 0.0j)

    pll.integral, dvc.integral, cc.integral = 5.0, 6.0, 7.0 + 8.0j
    controller.command(True)
    assert (pll.integral, dvc.integral, cc.integral) == (5.0, 6.0, 7.0 + 8.0j)


def test_presynchronisation_does_not_seed_a_pwm_start_integrator():
    params = peslite.load(
        EXAMPLES / "gfm-psc-example.pes",
        **{"simulation.solver.linearisations": 0, "simulation.progress.enable": 0},
    )
    cfg = params.unit("vsc")
    controller = peslite.VSCController(cfg)
    sync = controller.graph.nodes["PSC"]
    sync.v_int = 4.0
    controller.track(Measurement(0.0, cfg.base.v_phase_peak, 0j, cfg.dclink.vdc_ref, np.zeros(3)))
    assert sync.v_int == 0.0


def test_pwm_stays_blocked_until_first_controller_word_is_loaded(gfl, tmp_path):
    params = gfl(**{
        "events.connect_vsc.t": 0.001,
        "events.connect_vsc.ramp": 0.0,
        "simulation.t_end": 0.0012,
        "simulation.output.period": 2.5e-5,
        "simulation.output.signals": 1,
    })
    result = _run(params, tmp_path)
    t = result.states["t"]
    active, shadow = result.states["vsc.pwm.on"], result.states["vsc.pwm.shadow.on"]
    release = 0.001 + T
    assert np.all(active[t < release - 1e-9] == 0.0)
    assert np.all(active[t > release - 1e-9] == 1.0)
    assert shadow[np.argmin(abs(t - 0.001))] == 1.0
    assert np.all(_current(result)[t <= release + 1e-9] == 0.0)


def test_disconnect_stops_startup_and_connect_restarts_ramp(gfl, tmp_path):
    params = gfl(
        **{"events.connect_vsc.ramp": 0.0, "simulation.t_end": 0.003,
           "simulation.output.period": T},
        **_event("off", type="disconnect", target="vsc", t=0.001),
        **_event("on", type="connect", target="vsc", t=0.002, ramp=0.0005),
    )
    result = _run(params, tmp_path)
    t, steps = result.states["t"], result.states["vsc.ctrl.startup.steps"]
    assert np.all(steps[(t > 0.001 + 1e-9) & (t < 0.002 - 1e-9)] == 0.0)
    after = t > 0.002 - 1e-9
    end = round(0.0005 / T)
    assert np.array_equal(steps[after], np.minimum(np.arange(1, after.sum() + 1), end))


def test_fast_unit_protection_trips_during_unarmed_ramp(gfl, tmp_path):
    params = gfl(**{
        "units.vsc.protection.overcurrent.limit_pu": 0.01,
        "events.connect_vsc.ramp": 1.0,
        "simulation.t_end": 0.002,
        "simulation.stop_on_trip": 0,
        "simulation.output.period": 2.5e-5,
    })
    result = _run(params, tmp_path)
    summary = result.summary
    assert summary["vsc.trip_cause"] == "overcurrent"
    assert summary["vsc.trip_time"] == summary["vsc.overcurrent_first_t"] < 1.0
    after = result.states["t"] > summary["vsc.trip_time"] + 1e-9
    assert np.all(_current(result)[after] == 0.0)
    assert np.all(result.states["vsc.fault"][after] == 1.0)
    assert {"OVERCURRENT", "TRIP_OVERCURRENT"} <= set(summary["vsc.alarms"])


@pytest.mark.parametrize("criterion, cause, alarm, inputs", [
    ("dc_voltage", "vdc", "VDC_BAND", (1.0, 0.0, 0.2)),
    ("frequency", "freq", "FREQ_BAND", (1.0, 2.0, 0.0)),
    ("undervoltage", "voltage_uv", "VOLTAGE_UNDER", (0.5, 0.0, 0.0)),
    ("overvoltage", "voltage_ov", "VOLTAGE_OVER", (1.5, 0.0, 0.0)),
])
def test_sampled_protection_arms_and_holds(criterion, cause, alarm, inputs):
    cfg = ProtectionParams(
        hold=0.02,
        overcurrent=OvercurrentParams(limit_pu=1.0),
        dc_voltage=DCVoltageLimitParams(limit_pu=0.1),
        frequency=FrequencyLimitParams(limit=1.0),
        undervoltage=VoltageLimitParams(limit_pu=0.8),
        overvoltage=VoltageLimitParams(limit_pu=1.2),
    )
    cfg = replace(cfg, **{criterion: replace(getattr(cfg, criterion), enable=True)})
    protection = peslite.Protection(cfg, 0.01, 1.0)
    protection.check_sampled(0.0, peslite.ProtectionSample(
        terminal_voltage_pu=inputs[0],
        frequency_deviation=inputs[1],
        dc_voltage_error_pu=inputs[2],
        startup_complete=False,
    ))
    assert not protection.tripped and protection.stats.alarms == []
    sample = peslite.ProtectionSample(
        terminal_voltage_pu=inputs[0],
        frequency_deviation=inputs[1],
        dc_voltage_error_pu=inputs[2],
    )
    protection.check_sampled(0.01, sample)
    assert not protection.tripped and protection.stats.alarms == [alarm]
    protection.check_sampled(0.02, sample)
    protection.check_sampled(0.03, sample)
    assert protection.trip is not None and protection.trip.cause == cause


def test_fast_and_sampled_paths_share_one_trip_latch():
    cfg = ProtectionParams(
        hold=0.0,
        overcurrent=OvercurrentParams(enable=True, limit_pu=0.5),
        dc_voltage=DCVoltageLimitParams(enable=True, limit_pu=0.1),
        frequency=FrequencyLimitParams(limit=1.0),
        undervoltage=VoltageLimitParams(limit_pu=0.8),
        overvoltage=VoltageLimitParams(limit_pu=1.2),
    )
    protection = peslite.Protection(cfg, 0.01, 1.0)
    assert protection.check_fast(0.01, (0.6, 0.0, 0.0))
    first = protection.trip
    assert not protection.check_sampled(
        0.02,
        peslite.ProtectionSample(
            terminal_voltage_pu=1.0,
            frequency_deviation=0.0,
            dc_voltage_error_pu=0.2,
        ),
    )
    assert protection.trip is first and protection.trip.cause == "overcurrent"
    assert protection.stats.alarms == ["OVERCURRENT", "TRIP_OVERCURRENT"]


def test_controller_block_states_use_entity_first_names(gfl):
    names = set(peslite.Simulation(gfl()).state_names())
    assert {"vsc.ctrl.startup.run", "vsc.ctrl.startup.ramp", "vsc.ctrl.startup.steps",
            "vsc.tripped", "vsc.fault", "vsc.pwm.on", "vsc.pwm.shadow.on"} <= names
    assert "vsc.prot.hold_uv" in names
    assert not any(".ctrl.prot." in name for name in names)
    assert not any(".pending." in name for name in names)
    # Every graph controller restores its start-up entries strictly, like its loop states.
    for model in ("pwm_averaging", "averaging"):
        controller = peslite.VSCController(gfl(**{"units.vsc.bridge.model": model}).unit("vsc"))
        with pytest.raises(KeyError, match="start-up sequence has no state"):
            controller.set_state({"startup.stpes": 1})


@pytest.mark.parametrize("name", ["gfm-psc-example.pes", "gfm-vsg-example.pes"])
def test_grid_forming_controller_presynchronises_to_terminal_voltage(name, tmp_path):
    params = peslite.load(EXAMPLES / name, **{
        "simulation.t_end": 0.03,
        "simulation.progress.enable": 0,
        "simulation.solver.linearisations": 0,
        "sources.grid.angle": 1.0,
        "sources.grid.v_pu": 1.05,
        "events.connect_vsc.t": 0.0,
    })
    result = _run(params, tmp_path)
    assert not result.tripped and result.summary["vsc.max_current_pu"] < 0.3


def test_split_bound_holds_startup_and_pwm_modes(gfl, tmp_path):
    params = gfl(**{
        "events.connect_vsc.t": 0.003,
        "simulation.solver.linearisations": 2,
        "simulation.solver.subsystems.vsc.dclink": 10,
    })
    summary = _run(params, tmp_path).summary
    assert summary["split_bound"].startswith("linearised loop at t = 0.002, 0.004 s")
    # windows closed, but without the state history there is no row to linearise at
    summary = _run(params.replace(**{"simulation.output.states": 0}), tmp_path / "no-states").summary
    assert summary["interface_windows"] > 0
    assert summary["split_bound"].startswith("no recorded state row to linearise at")


def test_startup_and_registers_continue_from_a_saved_row(gfl, tmp_path):
    params = gfl(**{
        "events.connect_vsc.t": 0.001,
        "events.connect_vsc.ramp": 0.001,
        "simulation.t_end": 0.002,
        "simulation.output.period": 1e-5,
    })
    whole_dir = tmp_path / "whole"
    whole = _run(params, whole_dir)
    t0 = 0.00137
    continued_params = peslite.load(
        whole_dir / "simulation.pes", initial=whole_dir / "states.csv", initial_time=t0)
    continued = _run(continued_params, tmp_path / "continued")
    start = int(np.argmin(abs(whole.t - t0)))
    assert np.array_equal(continued.t, whole.t[start:])
    for name, values in continued.states.items():
        assert np.array_equal(values, whole.states[name][start:]), name
