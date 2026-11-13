"""Operating-point checks preserve periodicity and discrete-state semantics."""

import math
from types import SimpleNamespace

import peslite
import pytest

from peslite.addons.functions import detect_operating_point
from peslite.addons.functions import operating_point as operating_module
from peslite.solver.model import Empty, Model


class _Result:
    summary = {}
    tripped = False

    def final_states(self):
        return {"t": 0.0, "plant.x": 0.0, "mode": 0.0}


class _Simulation:
    def __init__(self, response="static"):
        self.response = response
        self.result = _Result()
        self._rebuildable = True
        self.units = {}
        self.p = SimpleNamespace(changes=[], simulation=SimpleNamespace(
            output=SimpleNamespace(period=0.01)))
        self.system = SimpleNamespace()


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


@pytest.fixture
def _use_fake_period_map(monkeypatch):
    monkeypatch.setattr(operating_module, "SystemLoop", _Loop)


def test_detects_dc_periodic_and_unconverged_states(_use_fake_period_map):
    dc = detect_operating_point(_Simulation())
    assert dc.kind == "dc" and dc.valid
    assert dc.periodic_error == pytest.approx(0)
    assert dc.evaluations == 3
    periodic = detect_operating_point(_Simulation("sine"))
    assert periodic.kind == "periodic" and periodic.valid
    assert periodic.intra_period_error > 0.9
    drift = detect_operating_point(_Simulation("drifting"), tolerance=0.01)
    assert drift.kind == "unconverged" and not drift.valid
    assert "nonrepeating" in drift.diagnostics[0]
    discrete = detect_operating_point(_Simulation("mode"))
    assert not discrete.valid
    assert "discrete" in discrete.diagnostics[0]
    assert dc.to_dict()["at"]["unit"] == "s"


def test_stopped_and_tripped_runs_are_not_reported_as_steady(_use_fake_period_map):
    simulation = _Simulation()
    simulation.result = SimpleNamespace(final_states=simulation.result.final_states,
                                        summary={"stop_reason": "diverged"}, tripped=False)
    assert not detect_operating_point(simulation).valid
    simulation.result.summary.clear()
    simulation.result.tripped = True
    assert not detect_operating_point(simulation).valid


def test_rejects_invalid_options_and_unrun_simulations(_use_fake_period_map):
    sim = _Simulation()
    with pytest.raises(ValueError, match="tolerance"):
        detect_operating_point(sim, tolerance=-1)
    with pytest.raises(ValueError, match="period"):
        detect_operating_point(sim, period=0)
    sim.result = None
    with pytest.raises(ValueError, match="Simulation.run"):
        detect_operating_point(sim)


class _LinearPlant:
    state_names = ("x", "y")
    outputs_need_inputs = False

    def __init__(self, frequency):
        self.frequency = frequency
        self.state = SimpleNamespace(x=1.0, y=0.0)
        self.input = Empty()
        self.output = Empty()

    def set_outputs(self, _t):
        pass

    def rhs(self, _t):
        omega = math.tau * self.frequency
        return -omega * self.state.y, omega * self.state.x


def _completed_simulation(examples, frequency):
    def factory(_params):
        plant = _LinearPlant(frequency)
        model = Model({"plant": plant}, {})
        return SimpleNamespace(
            units={}, event_subsystems=(), model=model, _unit_port_bindings={},
            state_parts=lambda: [("", model)],
        )

    period = 0.01 if frequency else 0.02
    params = peslite.load(examples / "buck-example.pes", **{
        "simulation.t_end": period * 2,
        "simulation.energy_check": "off",
        "simulation.solver.dt": period / 100,
        "simulation.solver.linearisations": 0,
        "simulation.output.period": period / 10,
        "simulation.output.states": False,
        "simulation.output.signals": False,
        "simulation.output.energy": False,
        "simulation.progress.enable": False,
    })
    simulation = peslite.Simulation(params, parts={"system": factory})
    simulation.run(out_dir=None)
    return simulation, period


@pytest.mark.parametrize("frequency, expected", [(0.0, "dc"), (100.0, "periodic")])
def test_completed_simulation_uses_the_real_system_loop(examples, frequency, expected):
    simulation, period = _completed_simulation(examples, frequency)
    result = detect_operating_point(simulation, period=period, tolerance=1e-4)
    assert result.kind == expected, result.diagnostics
    assert result.valid
    assert result.periodic_error < 1e-5
