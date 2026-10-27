"""Post-run energy closure uses the public streamed energy table."""

import pytest

import peslite
from peslite.addons.functions import energy_closure
from conftest import EXAMPLES


def test_energy_closure_of_open_loop_buck(tmp_path):
    params = peslite.load(EXAMPLES / "buck-open-loop-example.pes", **{
        "simulation.t_end": 2.0e-3,
        "simulation.output.energy": True,
        "simulation.solver.phs_check_step": 1,
    })
    result = peslite.Simulation(params).run(out_dir=tmp_path)
    closure = energy_closure(result)

    assert closure["supplied"] > 0.0
    assert closure["dissipated"] > 0.0
    assert abs(closure["relative"]) < 1.0e-4


def test_energy_closure_checks_its_input():
    with pytest.raises(ValueError, match="energy table"):
        energy_closure({"t": [0.0, 1.0]})
