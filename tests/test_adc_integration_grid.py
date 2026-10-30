"""ADC window integration follows physical sampling events, not output snapshots."""

import warnings

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES


def _power(output_period, tmp_path, **changes):
    settings = {
        "simulation.solver.linearisations": 0,
        "simulation.t_end": 2.0e-4,
        "simulation.output.period": output_period,
        "simulation.output.record_every": 1,
        "units.charger.primary.bridge.model": "pwm_averaging",
        "units.charger.primary.meas.period": 1.0e-7,
        "units.charger.secondary.bridge.model": "pwm_averaging",
        **changes,
    }
    params = peslite.load(EXAMPLES / "wpt-ss-example.pes", **settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = peslite.Simulation(params).run(out_dir=tmp_path / f"output-{output_period}")
    times = np.round(np.asarray(result.t), 12)
    return dict(zip(times, np.asarray(result.columns()["charger.primary.power"])))


@pytest.mark.parametrize("changes", [
    {},
    # Neither the output grid nor a continuous unit's own clock adds window nodes.
    {"units.charger.primary.meas.period": None,
     "units.charger.secondary.bridge.model": "averaging"},
])
def test_adc_sampling_grid_is_independent_of_the_output_grid(tmp_path, changes):
    coarse = _power(2.0e-6, tmp_path, **changes)
    fine = _power(1.0e-7, tmp_path, **changes)
    common = [time for time in coarse.keys() & fine.keys()
              if np.isfinite(coarse[time]) and np.isfinite(fine[time])]
    scale = max(1.0, *(abs(fine[time]) for time in common))

    assert max(abs(coarse[time] - fine[time]) for time in common) <= 1.0e-9 * scale


def test_adc_sampling_grid_works_with_multirate_subsystems(tmp_path):
    result = _power(
        2.0e-6,
        tmp_path,
        **{"simulation.solver.subsystems": {"charger.primary.dclink": 4}},
    )
    values = np.asarray(list(result.values()))

    assert np.isfinite(values[1:]).all()
