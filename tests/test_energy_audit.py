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


def test_energy_closure_includes_supply_and_loss_at_state_jumps():
    closure = energy_closure({
        "t": [0.0, 1.0],
        "energy_J": [0.0, 1.0],
        "power_supplied_W": [0.0, 0.0],
        "power_dissipated_W": [0.0, 0.0],
        "circuit.supplied_impulsive_J": [0.0, 3.0],
        "circuit.dissipated_impulsive_J": [0.0, 2.0],
    })
    assert closure == {
        "stored": 1.0,
        "supplied": 3.0,
        "dissipated": 0.0,
        "impulsive": 2.0,
        "impulsive_supply": 3.0,
        "residual": 0.0,
        "relative": 0.0,
    }
