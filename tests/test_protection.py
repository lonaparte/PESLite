"""Protection runtime behavior and output semantics."""

import peslite
from dataclasses import replace
from peslite.assembly.params import ProtectionParams, RocofParams


def test_frequency_history_is_lazy_and_retunes_without_changing_window_timing():
    cfg = ProtectionParams()
    protection = peslite.Protection(cfg, 1e-6, 10.0)
    for index in range(10):
        protection.check_sampled(index * 1e-6, peslite.ProtectionSample())
    assert protection._rocof is None
    cfg = replace(cfg, rocof=RocofParams(enable=True, limit=1.0, window=4e-6))
    protection.retune(cfg)
    assert protection._rocof is None
    for index in range(5):
        protection.check_sampled(index * 1e-6, peslite.ProtectionSample(
            frequency_deviation=float(index)))
    assert protection._rocof.n == 4
    assert protection.stats.rocof_first_t == 4e-6
    assert protection.stats.rocof_max == 1e6


def _run(params, out):
    return peslite.Simulation(params).run(out_dir=out)


def test_disabled_overcurrent_never_trips(gfl, tmp_path):
    settings = {
        "units.vsc.protection.overcurrent.limit_pu": 1e-3,
        "events.connect_vsc.ramp": 0,
        "simulation.t_end": 0.002,
    }
    assert _run(gfl(**settings), tmp_path / "enabled").tripped
    assert not _run(gfl(**settings, **{"units.vsc.protection.overcurrent.enable": 0}),
                    tmp_path / "disabled").tripped


def test_hold_zero_trips_only_while_the_criterion_is_met(gfl, tmp_path):
    settings = {
        "units.vsc.protection.dc_voltage.enable": 1,
        "units.vsc.protection.dc_voltage.limit_pu": 0.05,
        "units.vsc.protection.hold": 0.0,
        "events.connect_vsc.ramp": 0,
        "simulation.t_end": 0.003,
        **{f"units.vsc.protection.{criterion}.enable": 0
           for criterion in ("overcurrent", "undervoltage", "overvoltage", "frequency")},
    }
    assert not _run(gfl(**settings), tmp_path / "plain").tripped

    with_event = gfl(**settings, **{
        "events.vdc_reference": {
            "type": "set",
            "t": 0.001,
            "set": {"units.vsc.ctrl.references.vdc_ref_pu": 1.1},
        }
    })
    result = _run(with_event, tmp_path / "event")
    assert result.summary["vsc.trip_cause"] == "vdc"
    assert result.summary["vsc.trip_time"] >= 0.001
