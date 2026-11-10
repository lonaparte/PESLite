"""Operating-point checks preserve periodicity and discrete-state semantics."""

import math
from types import SimpleNamespace

import pytest

from peslite.addons.functions import detect_operating_point
from peslite.addons.functions import operating_point as operating_module


class _Result:
    summary = {}
    tripped = False

    def final_states(self):
        return {"t": 0.0, "plant.x": 0.0, "mode": 0.0}


class _Simulation:
    def __init__(self, response="static", source_frequency=0.0):
        self.response = response
        self.result = _Result()
        self._rebuildable = True
        self.units = {}
        self.p = SimpleNamespace(changes=[], simulation=SimpleNamespace(
            output=SimpleNamespace(period=0.01)))
        source = SimpleNamespace(cfg=SimpleNamespace(f=source_frequency),
                                 emf=SimpleNamespace(e_peak=10.0))
        self.system = SimpleNamespace(sources={"grid": source})


class _Loop:
    def __init__(self, sim, *, period=None, frame_frequency=0, repeat_forcing=False):
        self.sim = sim
        self.period = 1.0 if period is None else period
        self.periods = [1.0]

    def advance(self, row, t):
        result = dict(row)
        if self.sim.response == "sine":
            result["plant.x"] = math.sin(math.tau * (t + self.period))
        elif self.sim.response == "drifting":
            result["plant.x"] = 0.2 * self.period
        elif self.sim.response == "mode":
            result["mode"] = 1.0
        return result

    def periodic_state(self, row):
        return row

    def coordinates(self, names, *, linearization=True):
        return [("scalar", ["plant.x"], 1.0), ("mode", ["mode"], 1.0)]

    def to_frame(self, row, t, cols):
        import numpy as np
        return np.asarray([row[name] for _, names, _ in cols for name in names])


@pytest.fixture(autouse=True)
def _use_fake_period_map(monkeypatch):
    monkeypatch.setattr(operating_module, "SystemLoop", _Loop)


def test_detects_dc_periodic_ac_and_unconverged_states():
    dc = detect_operating_point(_Simulation())
    assert dc.kind == "dc" and dc.valid
    assert dc.periodic_error == pytest.approx(0)
    assert dc.evaluations == 3
    ac = detect_operating_point(_Simulation("sine", source_frequency=50))
    assert ac.kind == "ac_periodic" and ac.valid
    assert ac.intra_period_error > 0.9
    periodic = detect_operating_point(_Simulation("sine"))
    assert periodic.kind == "periodic"
    drift = detect_operating_point(_Simulation("drifting"), tolerance=0.01)
    assert drift.kind == "unconverged" and not drift.valid
    assert "nonrepeating" in drift.diagnostics[0]
    discrete = detect_operating_point(_Simulation("mode"))
    assert not discrete.valid
    assert "discrete" in discrete.diagnostics[0]
    assert dc.to_dict()["at"]["unit"] == "s"


def test_stopped_and_tripped_runs_are_not_reported_as_steady():
    simulation = _Simulation()
    simulation.result = SimpleNamespace(final_states=simulation.result.final_states,
                                        summary={"stop_reason": "diverged"}, tripped=False)
    assert not detect_operating_point(simulation).valid
    simulation.result.summary.clear()
    simulation.result.tripped = True
    assert not detect_operating_point(simulation).valid


def test_rejects_invalid_options_and_unrun_simulations():
    sim = _Simulation()
    with pytest.raises(ValueError, match="tolerance"):
        detect_operating_point(sim, tolerance=-1)
    with pytest.raises(ValueError, match="period"):
        detect_operating_point(sim, period=0)
    sim.result = None
    with pytest.raises(ValueError, match="Simulation.run"):
        detect_operating_point(sim)
