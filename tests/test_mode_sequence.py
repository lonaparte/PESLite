"""ModeSequence emits a reusable phase-shift bridge pattern."""

import math

import pytest

import peslite
from peslite.assembly.params import ConfigError, from_dict
from test_phase_shift import _phase_shift, _ps_tree
from test_wpt_rectifier import F, _rectifier


def _sequence(modes, **extra):
    primary = _phase_shift(0.8, phase=0.5 * math.pi, sync="pulse")
    primary["ctrl"]["loops"] = {
        "ModeSequence": {"modes": modes, **extra},
        "PhaseShiftModulation": {},
    }
    primary["ctrl"]["connections"] = {
        "ModeSequence.width": "references.width",
        "ModeSequence.phase": "references.phase",
        "PhaseShiftModulation.mode": "ModeSequence.mode",
        "PhaseShiftModulation.width": "ModeSequence.width",
        "PhaseShiftModulation.phase": "ModeSequence.phase",
    }
    return _ps_tree(primary, _rectifier(150.0), 4.0 / F)


def test_sequence_advances_and_restores_its_named_state():
    simulation = peslite.Simulation(from_dict(_sequence([1, 0, 4])))
    node = simulation.system.units["charger.primary"].ctrl.graph.nodes["ModeSequence"]
    inputs = {"width": 0.7, "phase": 0.2}

    assert [node.sample(k / F, inputs)["mode"] for k in range(4)] == [1.0, 0.0, 4.0, 1.0]
    node.set_state({"index": 2})
    assert node.sample(0.0, inputs)["mode"] == 4.0


def test_sequence_parameters_and_period_are_checked():
    with pytest.raises(ConfigError, match="unknown mode"):
        from_dict(_sequence([1, 2]))
    with pytest.raises(ConfigError, match="one entry per mode"):
        from_dict(_sequence([1, 0], widths=[0.5]))
    slow = _sequence([1, 0, 4])
    slow["units"]["charger"]["primary"]["ctrl"]["loops"]["ModeSequence"][
        "period"
    ] = 2.0 / F
    graph = peslite.Simulation(from_dict(slow)).system.units["charger.primary"].ctrl.graph
    assert "ModeSequence" in graph.due(0.0)
    assert "ModeSequence" not in graph.due(1.0 / F)
    assert "ModeSequence" in graph.due(2.0 / F)


def test_sequence_timing_is_independent_of_the_modulation_frequency():
    tree = _sequence([1, 0, 4])
    tree["events"] = {"frequency": {"type": "set", "t": 1.0 / F, "set": {
        "units.charger.primary.ctrl.references.frequency": 0.8 * F,
    }}}
    simulation = peslite.Simulation(from_dict(tree))
    node = simulation.system.units["charger.primary"].ctrl.graph.nodes["ModeSequence"]
    assert node.block_period == pytest.approx(1.0 / F)
