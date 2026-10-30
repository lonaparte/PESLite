"""The solver kernel and its integration with the existing assembly layer."""

import ast
from pathlib import Path

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.solver import (AdaptiveSolver, Bag, DormandPrince45, Empty, FixedStepSolver,
                            Model, MultirateSolver, OutputMethod, make_solver)
from peslite.solver.multirate import parse_step

QUIET = {"simulation.progress.enable": 0, "simulation.solver.linearisations": 0}
FAST = {"events.connect_vsc.t": 0.0, "events.connect_vsc.ramp": 0.01}
SOLVER = Path(peslite.solver.__file__).parent


def test_solver_and_control_output_stage_names_do_not_collide():
    assert peslite.solver.OutputMethod is OutputMethod
    assert not hasattr(peslite.solver, "OutputStage")
    assert peslite.OutputStage is peslite.control.OutputStage


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*-example.pes")), ids=lambda p: p.stem)
def test_every_bundled_example_runs_with_the_solver(path, tmp_path):
    sim = peslite.Simulation(peslite.load(path, **QUIET, **{"simulation.t_end": 0.002}))
    result = sim.run(out_dir=tmp_path / path.stem)
    assert sim.ph_report.verdict == "port-hamiltonian"
    assert result.summary["t_stop"] == pytest.approx(0.002)
    assert not result.tripped


def test_make_solver_builds_the_configured_integrator():
    params = peslite.load(EXAMPLES / "gfl-example.pes")
    model = peslite.System(params).model
    settings = params.simulation.solver
    cases = {
        FixedStepSolver: {},
        DormandPrince45: {"type": "adaptive", "method": "DP45"},
        AdaptiveSolver: {"type": "adaptive", "method": "RK45"},
        MultirateSolver: {"subsystems": {"vsc.dclink": 4}},
    }
    for expected, changes in cases.items():
        configured = settings.__class__(**{**vars(settings), **changes})
        assert type(make_solver(configured, model)) is expected


def test_dp45_float_model_path_matches_the_array_path():
    class Oscillator:
        @staticmethod
        def rhs_list(t, y):
            return [y[1], -4.0 * y[0]]

        def rhs(self, t, y):
            return np.asarray(self.rhs_list(t, y))

    model = Oscillator()
    y0 = np.asarray([1.0, 0.0])
    fast = DormandPrince45(rtol=1e-8, atol=1e-11)(model.rhs, 0.0, 1.0, y0)
    array = DormandPrince45(rtol=1e-8, atol=1e-11)(
        lambda t, y: model.rhs(t, y), 0.0, 1.0, y0
    )
    np.testing.assert_allclose(fast.y, array.y, rtol=1e-13, atol=1e-14)


def test_model_automatically_solves_an_algebraic_output_loop():
    class In(Bag):
        __slots__ = ("x",)

    class Out(Bag):
        __slots__ = ("y",)

    class Feedback:
        state_names = ()
        outputs_need_inputs = True

        def __init__(self):
            self.state, self.inp, self.out = Empty(), In(), Out()

        def set_outputs(self, _t):
            self.out.y = 1.0 + 0.5 * self.inp.x

        def rhs(self, _t):
            return ()

    block = Feedback()
    model = Model({"feedback": block}, {(block, "x"): (block, "y")})
    model.sync(0.0, np.empty(0))

    assert len(model.algebraic_loops) == 1
    assert block.inp.x == pytest.approx(2.0)
    assert block.out.y == pytest.approx(2.0)


def test_observational_output_stage_is_pruned_or_promoted_by_rhs_dependencies():
    class State(Bag):
        __slots__ = ("x",)

    class Input(Bag):
        __slots__ = ("x",)

    class Output(Bag):
        __slots__ = ("x",)

    class Producer:
        state_names = ("x",)
        outputs_need_inputs = False
        output_stages = (OutputMethod("observe", (), ("x",), rhs=False),)

        def __init__(self):
            self.state, self.inp, self.out = State(x=2.0), Empty(), Output()
            self.observations = 0

        def observe(self, _t):
            self.observations += 1
            self.out.x = self.state.x

        def set_outputs(self, t):
            self.observe(t)

        def rhs(self, _t):
            return (1.0,)

    class Consumer:
        state_names = ("x",)
        outputs_need_inputs = False

        def __init__(self):
            self.state, self.inp, self.out = State(), Input(), State()

        def set_outputs(self, _t):
            self.out.x = self.state.x

        def rhs(self, _t):
            return (self.inp.x,)

    unconnected = Producer()
    isolated = Model({"producer": unconnected}, {})
    assert isolated.rhs(0.0, np.array([2.0])).tolist() == [1.0]
    assert unconnected.observations == 0
    isolated.sync(0.0, np.array([2.0]))
    assert unconnected.observations == 1

    producer, consumer = Producer(), Consumer()
    connected = Model(
        {"producer": producer, "consumer": consumer},
        {(consumer, "x"): (producer, "x")},
    )
    assert connected.rhs(0.0, np.array([2.0, 0.0])).tolist() == [1.0, 2.0]
    assert producer.observations == 1


@pytest.mark.parametrize("split", [{}, {"simulation.solver.subsystems": {"pcc": 3}}],
                         ids=["single-rate", "mid-window"])  # 2 ms lies inside a 3 dt window
def test_a_run_continues_exactly_from_a_saved_state(tmp_path, split):
    def run(t_end, out, **changes):
        params = peslite.load(EXAMPLES / "gfl-example.pes", **QUIET, **FAST, **split,
                              **{"simulation.t_end": t_end}, **changes)
        return peslite.Simulation(params).run(out_dir=out)

    whole = run(0.004, tmp_path / "whole")
    run(0.002, tmp_path / "first")
    continued = run(0.004, tmp_path / "continued", initial=tmp_path / "first" / "states.csv")
    continued_states = continued.final_states()
    whole_states = whole.final_states()
    assert continued_states.keys() == whole_states.keys()
    np.testing.assert_allclose(
        list(continued_states.values()), list(whole_states.values()), rtol=1e-13, atol=1e-13
    )


@pytest.mark.parametrize("value, expected", [(2, (2.0, None)), (0.25, (0.25, None)),
                                               ({"step": 4, "method": "BDF"}, (4.0, "BDF"))])
def test_parse_step(value, expected):
    assert parse_step(value) == expected


@pytest.mark.parametrize("value", [True, 0, -1, 0.3, 1.5, float("inf")])
def test_parse_step_rejects_invalid_ratios(value):
    with pytest.raises(ValueError):
        parse_step(value)


def test_solver_kernel_does_not_import_application_layers():
    forbidden = {"assembly", "components", "control", "firmware", "modulation", "params", "power",
                 "protection", "results", "sensing", "simulation"}
    for path in SOLVER.glob("*.py"):
        if path.name == "simulation.py":
            continue
        imported = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                package = list(path.relative_to(SOLVER.parent).parts[:-1])
                target = package[:len(package) - (node.level - 1)]
                if node.module:
                    target += node.module.split(".")
                if target:
                    imported.add(target[0])
        assert not imported.intersection(forbidden), f"{path.name} imports {sorted(imported & forbidden)}"
