"""ADC window integration on an optional bounded measurement grid."""

import warnings

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES


def _power(interval, period, tmp_path, **changes):
    settings = {
        "simulation.solver.linearisations": 0,
        "simulation.t_end": 2.0e-4,
        "simulation.output.period": period,
        "simulation.output.record_every": 1,
        "simulation.solver.measurement_interval": interval,
        "units.charger.primary.bridge.model": "pwm_averaging",
        "units.charger.secondary.bridge.model": "pwm_averaging",
        **changes,
    }
    params = peslite.load(EXAMPLES / "wpt-ss-example.pes", **settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        label = "events" if interval is None else f"interval-{interval}"
        result = peslite.Simulation(params).run(out_dir=tmp_path / f"{label}-{period}")
    times = np.round(np.asarray(result.t), 12)
    return dict(zip(times, np.asarray(result.columns()["charger.primary.power"])))


def test_bounded_measurements_are_independent_of_the_output_grid(tmp_path):
    errors = {}
    for name, interval in (("events", None), ("bounded", 1.0e-7)):
        coarse = _power(interval, 2.0e-6, tmp_path)
        fine = _power(interval, 1.0e-7, tmp_path)
        common = [time for time in coarse.keys() & fine.keys()
                  if np.isfinite(coarse[time]) and np.isfinite(fine[time])]
        errors[name] = max(abs(coarse[time] - fine[time]) for time in common)
    assert errors["events"] > 50.0
    assert errors["bounded"] < 2.0e-3 * errors["events"]


def test_bounded_measurements_work_with_multirate_subsystems(tmp_path):
    result = _power(
        1.0e-7,
        2.0e-6,
        tmp_path,
        **{"simulation.solver.subsystems": {"charger.primary.dclink": 4}},
    )
    assert result
    values = np.asarray(list(result.values()))
    assert np.isfinite(values[1:]).all()  # the first window has not closed yet


def test_default_event_grid_does_not_create_measurement_subintervals(monkeypatch, tmp_path):
    from peslite.solver import simulation as simulation_module

    def unexpected(*_args):
        raise AssertionError("the default measurement grid must not subdivide integration")

    monkeypatch.setattr(simulation_module, "subintervals", unexpected)
    params = peslite.load(EXAMPLES / "wpt-ss-example.pes", **{
        "simulation.t_end": 20.0e-6,
        "simulation.solver.linearisations": 0,
        "simulation.output.signals": False,
        "simulation.output.states": False,
    })
    peslite.Simulation(params).run(out_dir=tmp_path)


@pytest.mark.parametrize("interval", [0.0, -1.0e-6, float("inf")])
def test_measurement_interval_must_be_finite_and_positive(interval):
    with pytest.raises(ValueError, match="measurement_interval must be finite and positive"):
        peslite.load(EXAMPLES / "buck-example.pes", **{
            "simulation.solver.measurement_interval": interval,
        })


def test_generated_backends_reject_unimplemented_measurement_semantics(tmp_path):
    params = peslite.load(EXAMPLES / "buck-example.pes", **{
        "simulation.solver.measurement_interval": 1.0e-6,
        "simulation.solver.linearisations": 0,
    })
    with pytest.raises(ValueError, match="additional ADC integration grid"):
        peslite.Simulation(params).export("cpp", tmp_path)
