"""Proven overwrites versus exhaustive differentiation through the same complete loop."""

import dataclasses

import numpy as np
import peslite
import pytest

from conftest import EXAMPLES
from peslite.control import DUTY_RATIO, Loop, register_loop_type
from peslite.solver.linearization import linearize_loop
from peslite.solver.model import StateRegistry
from peslite.solver.simulation import SystemLoop
from peslite.solver.splitbound import _period_map
from test_small_signal import _AffineMap


def _simulation(**changes):
    params = peslite.load(EXAMPLES / "buck-example.pes", **{
        "simulation.t_end": 0.003, "simulation.energy_check": "off",
        "simulation.output.states": False, "simulation.output.signals": False,
        "simulation.output.energy": False, "simulation.progress.enable": False,
        "simulation.solver.linearisations": 0, **changes,
    })
    sim = peslite.Simulation(params)
    sim.run(out_dir=None)
    assert not sim.result.tripped and not sim.result.summary.get("stop_reason")
    return sim


def _compare(sim, exhaustive_loop, *, period=None):
    loop = SystemLoop(sim, period=period, frame_frequency=0.0, repeat_forcing=True)
    row = sim.result.final_states()
    start = row.pop("t")
    cols = loop.coordinates(list(row))
    before = sim.result.final_states()
    optimized = linearize_loop(loop, row, start, cols)
    full = linearize_loop(exhaustive_loop(loop), row, start, cols)
    np.testing.assert_array_equal(optimized.phi, full.phi)
    np.testing.assert_array_equal(optimized.initial, full.initial)
    np.testing.assert_array_equal(optimized.final, full.final)
    assert optimized.active == full.active
    assert optimized.changed_modes == full.changed_modes
    assert optimized.evaluations <= full.evaluations
    assert sim.result.final_states() == before
    return loop, optimized, full


@pytest.mark.parametrize("model", ["averaging", "pwm_averaging", "switching"])
@pytest.mark.parametrize("rates", [(1, 1), (4, 1), (1, 4), (2, 3), (3, 2)])
@pytest.mark.parametrize("carrier_phase", [0.0, 0.25])
def test_converter_matrices_match_exhaustive_differences_across_clocks_and_bridges(
        model, rates, carrier_phase, exhaustive_loop):
    control_period = 50e-6
    sim = _simulation(**{
        "units.converter.bridge.model": model,
        "units.converter.pwm.carrier_phase": carrier_phase,
        "units.converter.ctrl.loops.DCVoltageControl.period": rates[0] * control_period,
        "units.converter.ctrl.loops.InductorCurrentControl.period": rates[1] * control_period,
    })
    _compare(sim, exhaustive_loop, period=300e-6 if model == "averaging" else None)


@pytest.mark.parametrize("rates", [(1, 1), (4, 1), (1, 4), (2, 3), (3, 2)])
@pytest.mark.parametrize("offset", [0.0, 12.5e-6, 25e-6])
def test_graph_retains_values_consumed_before_their_producer_updates(rates, offset, exhaustive_loop):
    sim = _simulation(**{
        "simulation.t_end": 0.003 + offset,
        "units.converter.ctrl.loops.DCVoltageControl.period": rates[0] * 50e-6,
        "units.converter.ctrl.loops.InductorCurrentControl.period": rates[1] * 50e-6,
    })
    loop, optimized, full = _compare(sim, exhaustive_loop)
    graph = sim.unit().ctrl.graph
    producer = next(name for name, node in graph.nodes.items() if node.role == "dc_voltage")
    held = f"held.{producer}.current_reference"
    declared = graph.overwritten_state_names(loop.period)
    assert (held in declared) == (rates[1] % rates[0] == 0)
    assert optimized.evaluations < full.evaluations
    # A short horizon cannot assume even an unused stored output was overwritten.
    assert not graph.overwritten_state_names(min(graph.periods.values()) / 2.0)


def test_pending_retuning_and_future_events_keep_exhaustive_evaluation(exhaustive_loop):
    sim = _simulation(**{"events": {
        "retune": {"type": "set", "t": 0.003025,
                   "set": {"units.converter.ctrl.references.voltage": 23.0}},
    }})
    loop, optimized, full = _compare(sim, exhaustive_loop)
    assert not loop.overwritten_columns(0.003)
    assert optimized.evaluations == full.evaluations
    graph = sim.unit().ctrl.graph
    graph.schedule(graph.cfg)
    assert not graph.overwritten_state_names(loop.period)


@pytest.mark.parametrize("model", ["pwm_averaging", "switching"])
@pytest.mark.parametrize("resistance", [6.0, 300.0])
def test_natural_ccm_and_dcm_maps_match_exhaustive_differences(model, resistance, exhaustive_loop):
    sim = _simulation(**{
        "simulation.t_end": 0.05, "units.converter.bridge.model": model,
        "elements.load.r": resistance,
    })
    _compare(sim, exhaustive_loop)
    light_load = resistance == 300.0
    if model == "pwm_averaging":
        # A cycle-averaged DCM current remains positive; the conduction interval identifies DCM.
        assert bool(sim.unit().conduction.out.in_dcm) == light_load
    else:
        row = sim.result.final_states()
        start = row.pop("t")
        period = sim.unit().bridge.carrier_period
        settings = dataclasses.replace(sim.p.simulation, t_end=start + period,
                                      initial=dataclasses.replace(sim.p.simulation.initial,
                                                                  t=start, states=row))
        continued = peslite.Simulation(dataclasses.replace(sim.p, simulation=settings))
        currents = []

        def observe(t, states, read_values, *, final, stop_reason):
            currents.append(states["converter.branch_f.i"])

        continued.run(out_dir=None, observe=observe,
                      times=[start + i * period / 20 for i in range(21)])
        assert (min(currents) < 1e-8) == light_load


def test_ordinary_simulation_does_not_run_the_proof_analysis(monkeypatch):
    def unexpected(*_args):
        pytest.fail("ordinary simulation must not inspect linearization overwrites")

    monkeypatch.setattr(StateRegistry, "overwritten_columns", unexpected)
    _simulation()


@register_loop_type
class _HeldMemory(Loop):
    """A custom sampled block with genuine internal memory and an unused output."""

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Params:
        period: float
        gain: float = 0.1
        type: str = "test_pruning_memory"

    type = "test_pruning_memory"
    inputs = {"value": DUTY_RATIO}
    outputs = {"duty": DUTY_RATIO, "unused": DUTY_RATIO}
    state_names = ("memory",)

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.memory = 0.0

    def initial_outputs(self):
        return {"duty": 0.5, "unused": 0.0}

    def sample(self, t, inputs):
        self.memory = 0.2 * self.memory + self.cfg.gain * inputs["value"]
        return {"duty": 0.5 + self.memory, "unused": 2.0 * self.memory}


@pytest.mark.parametrize("rates", [(1, 1), (4, 1), (2, 3)])
def test_custom_loop_memory_and_boundary_outputs_remain_in_the_matrix(rates, exhaustive_loop):
    sim = _simulation(**{
        "units.converter.ctrl.loops": {
            "producer": {"type": _HeldMemory.type, "period": rates[0] * 50e-6},
            "consumer": {"type": _HeldMemory.type, "period": rates[1] * 50e-6},
        },
        "units.converter.ctrl.connections": {
            "producer.value": "meas.duty_feedforward", "consumer.value": "producer.duty",
        },
        "units.converter.ctrl.outputs": {"duty": "consumer.duty"},
    })
    loop, optimized, full = _compare(sim, exhaustive_loop)
    columns = loop.overwritten_columns(sim.result.final_states()["t"])
    assert "converter.ctrl.held.consumer.duty" not in columns
    assert not any(name.endswith(".memory") for name in columns)
    assert optimized.evaluations < full.evaluations


def test_registered_overwrites_preserve_complex_columns_and_ignore_unknown_names():
    class Owner:
        def get_state(self):
            return {"cache": 1.0j, "memory": 2.0}

        def overwritten_state_names(self, period):
            return {"cache", "not_registered"}

    registry = StateRegistry({"nested.ctrl": Owner()})
    assert registry.overwritten_columns(0.01) == {"nested.ctrl.cache.re", "nested.ctrl.cache.im"}
    assert registry.columns == ("nested.ctrl.cache.re", "nested.ctrl.cache.im", "nested.ctrl.memory")


def test_proven_zero_columns_keep_rows_zero_modes_and_split_input_responses(exhaustive_loop):
    class Map(_AffineMap):
        matrix = np.array([[0.9, 0.0], [0.3, 0.0]])

        def overwritten_columns(self, t):
            return {"y"}

    loop, row = Map(), {"x": 0.0, "y": 2.0, "gate": 0.0}
    optimized = linearize_loop(loop, row, 0.0, loop.cols)
    full = linearize_loop(exhaustive_loop(loop), row, 0.0, loop.cols)
    np.testing.assert_array_equal(optimized.phi, full.phi)
    assert optimized.active == (0, 1)
    assert optimized.evaluations == 3 and full.evaluations == 5
    assert 0.0 in np.linalg.eigvals(optimized.phi[:2, :2])
    actual = _period_map(loop, row, 0.0, loop.cols, [("x", 1.0)])
    expected = _period_map(exhaustive_loop(loop), row, 0.0, loop.cols, [("x", 1.0)])
    for a, b in zip(actual[:3], expected[:3]):
        np.testing.assert_array_equal(a, b)
    assert actual[3] == expected[3] - 2


def test_unknown_maps_do_not_numerically_prune_small_but_nonzero_responses():
    loop = _AffineMap()
    loop.matrix = np.array([[0.9, 1e-15], [0.0, 0.8]])
    linear = linearize_loop(loop, {"x": 0.0, "y": 0.0, "gate": 0.0}, 0.0, loop.cols)
    assert linear.evaluations == 5
    assert linear.phi[0, 1] == pytest.approx(1e-15, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("declared", [{"z.re"}, {"z.re", "z.im"}])
def test_transformed_coordinate_groups_require_a_complete_proof(declared, exhaustive_loop):
    class Map(_AffineMap):
        cols = [("fixed", ["z.re", "z.im"], 1.0)]

        def overwritten_columns(self, t):
            return declared

        def advance(self, row, t, view=None):
            self.calls += 1
            return {"z.re": 0.2, "z.im": 0.3}

    loop, row = Map(), {"z.re": 0.1, "z.im": 0.2}
    optimized = linearize_loop(loop, row, 0.0, loop.cols)
    full = linearize_loop(exhaustive_loop(loop), row, 0.0, loop.cols)
    np.testing.assert_array_equal(optimized.phi, full.phi)
    assert optimized.evaluations == (1 if len(declared) == 2 else 5)
