"""The solver kernel and its integration with the existing assembly layer."""

import ast
from pathlib import Path

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.solver import (AdaptiveSolver, Bag, DormandPrince45, Empty, FixedStepSolver,
                            Model, MultirateSolver, OutputMethod, make_solver)
from peslite.solver.integrators import TABLEAUS, combine
from peslite.solver.multirate import parse_step

QUIET = {"simulation.progress.enable": 0, "simulation.solver.linearisations": 0}
FAST = {"events.connect_vsc.t": 0.0, "events.connect_vsc.ramp": 0.01}
SOLVER = Path(peslite.solver.__file__).parent


def test_solver_and_control_output_stage_names_do_not_collide():
    assert peslite.solver.OutputMethod is OutputMethod
    assert not hasattr(peslite.solver, "OutputStage")
    assert peslite.OutputStage is peslite.control.OutputStage


def test_fixed_explicit_solver_uses_implicit_step_for_an_unstable_linear_mode():
    class StiffMode:
        @staticmethod
        def linear_modes():
            return (("stiff", lambda: np.array([[-1e8]])),)

    solver = FixedStepSolver(1e-5, model=StiffMode())
    result = solver(lambda _t, y: -1e8 * y, 0.0, 1e-5, np.array([1.0]))

    assert solver._implicit is not None
    assert abs(result.y[0]) < 0.01


def test_fixed_explicit_stability_is_reused_by_equation_mode():
    class Modes:
        mode = "a"
        builds = 0

        def linear_mode_key(self):
            return self.mode

        def linear_modes(self):
            def build():
                self.builds += 1
                return np.array([[-1.0]])
            return ((self.mode, build),)

    model = Modes()
    solver = FixedStepSolver(1e-5, model=model)
    rhs = lambda _t, y: -y
    state = np.array([1.0])

    solver(rhs, 0.0, 1e-5, state)
    solver(rhs, 1e-5, 2e-5, state)
    assert model.builds == 1

    model.mode = "b"
    solver(rhs, 2e-5, 3e-5, state)
    assert model.builds == 2

    solver.parameters_changed()
    solver(rhs, 3e-5, 4e-5, state)
    assert model.builds == 3


def test_continuous_linear_modes_recheck_explicit_stability():
    class DynamicMode:
        mode = 0
        builds = 0

        def linear_modes(self):
            def build():
                self.builds += 1
                return np.array([[-1.0]])
            return ((self.mode, build),)

    model = DynamicMode()
    solver = FixedStepSolver(1e-5, model=model)
    state = np.array([1.0])
    solver(lambda _t, y: -y, 0.0, 1e-5, state)
    model.mode = 1
    solver(lambda _t, y: -y, 1e-5, 2e-5, state)
    assert model.builds == 2


@pytest.mark.parametrize("method", ["euler", "heun", "RK4"])
@pytest.mark.parametrize("step", [1e-4, 1e-2])
def test_vectorized_stability_matches_the_scalar_tableau(method, step):
    eigenvalues = np.array([-1.0, -100.0, -1.0 + 50.0j, -1.0 - 50.0j, 2.0])
    solver = FixedStepSolver(step, method)
    solver._mode_eigenvalues["mode"] = eigenvalues

    _c, rows, weights = TABLEAUS[method]
    expected = True
    for eigenvalue in eigenvalues:
        if eigenvalue.real > 1e-12 * max(1.0, abs(eigenvalue)):
            continue
        z = step * eigenvalue
        stages = [1.0 + 0.0j]
        for row in rows:
            stages.append(1.0 + z * sum(
                coefficient * value for coefficient, value in zip(row, stages)
            ))
        amplification = 1.0 + z * sum(
            coefficient * value for coefficient, value in zip(weights, stages)
        )
        expected &= abs(amplification) <= 1.0 + 1e-10

    assert solver._explicit_stable(
        step, (("mode", lambda: np.empty((0, 0))),)
    ) is bool(expected)


def test_linear_model_view_projects_solver_metadata():
    class State(Bag):
        __slots__ = ("x",)

    class Decay:
        state_names = ("x",)
        outputs_need_inputs = False

        def __init__(self, rate):
            self.rate = rate
            self.state, self.input, self.output = State(x=1.0), Empty(), Empty()

        def set_outputs(self, _t):
            pass

        def rhs(self, _t):
            return (-self.rate * self.state.x,)

    matrix = np.diag([-2.0, -4.0])
    model = Model(
        {"first": Decay(2.0), "second": Decay(4.0)}, {},
        linear_mode_key=lambda: "mode",
        linear_blocks=lambda: ((np.array([0, 1]), matrix, np.array([0, 1]), np.array([0, 1])),),
        linear_modes=lambda: (("mode", lambda: matrix),),
    )
    view = model.linear_view(np.array([False, True]))

    block, = view.linear_blocks()
    mode, = view.linear_modes()
    np.testing.assert_array_equal(block[0], [0])
    np.testing.assert_array_equal(block[1], [[-4.0]])
    np.testing.assert_array_equal(block[2], [0])
    np.testing.assert_array_equal(block[3], [0])
    assert mode[0] == "mode"
    np.testing.assert_array_equal(mode[1](), [[-4.0]])
    assert view.linear_mode_key() == "mode"


def test_custom_inner_explicit_method_uses_the_projected_stability_model():
    class State(Bag):
        __slots__ = ("x",)

    class Decay:
        state_names = ("x",)
        outputs_need_inputs = False

        def __init__(self, rate):
            self.rate = rate
            self.state, self.input, self.output = State(x=1.0), Empty(), Empty()

        def set_outputs(self, _t):
            pass

        def rhs(self, _t):
            return (-self.rate * self.state.x,)

    matrix = np.diag([-1.0, -1e5])
    model = Model(
        {"slow": Decay(1.0), "stiff": Decay(1e5)}, {},
        linear_mode_key=lambda: "mode",
        linear_blocks=lambda: ((np.array([0, 1]), matrix, np.array([0, 1]), np.array([0, 1])),),
        linear_modes=lambda: (("mode", lambda: matrix),),
    )
    solver = MultirateSolver(
        model, {"stiff": {"step": 1, "method": "heun"}}, 1e-3, "RK4", sweeps=2
    )
    result = solver(model.rhs, 0.0, 1e-3, np.ones(2))

    assert solver._inner_solver._implicit is not None
    assert np.all(np.isfinite(result.y))
    assert abs(result.y[1]) < 0.1


@pytest.mark.full
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
    # A group's subsystems share one step count, as they share one method.
    for subsystems in ({"vsc.dclink": 4, "pcc": 2}, {"vsc.dclink": 0.5, "pcc": 0.25}):
        configured = settings.__class__(**{**vars(settings), "subsystems": subsystems})
        with pytest.raises(ValueError, match="share one ratio"):
            make_solver(configured, model)


@pytest.mark.full
@pytest.mark.parametrize("method", ["radau", "trapezoidal"])
def test_a_stiff_inner_group_can_use_an_implicit_method(method, tmp_path):
    common = {
        **QUIET,
        "elements.cable1.sections": 8,
        "elements.cable1.length_km": 5.0,
        "simulation.t_end": 0.002,
    }
    reference = peslite.Simulation(peslite.load(
        EXAMPLES / "cable-gfl-example.pes",
        **common,
        **{
            "simulation.solver.type": "adaptive",
            "simulation.solver.method": "radau",
            "simulation.solver.rtol": 1e-10,
            "simulation.solver.atol": 1e-10,
        },
    )).run(out_dir=tmp_path / "reference").final_states()
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / "cable-gfl-example.pes",
        **common,
        **{"simulation.solver.subsystems": {
            "cable1": {"step": 1, "method": method},
        }},
    ))
    result = simulation.run(out_dir=tmp_path / "split")

    assert isinstance(simulation.solver, MultirateSolver)
    assert simulation.solver.inner_method == method
    assert not result.tripped
    assert result.summary["t_stop"] == pytest.approx(0.002)
    actual = result.final_states()
    assert max(
        abs(actual[name] - value) / max(1.0, abs(value))
        for name, value in reference.items()
    ) < 3e-2


def test_one_inner_group_uses_one_method():
    params = peslite.load(EXAMPLES / "gfl-example.pes")
    model = peslite.System(params).model
    with pytest.raises(ValueError, match="share one method"):
        MultirateSolver(
            model,
            {
                "pcc": {"step": 1, "method": "radau"},
                "vsc.branch_f": {"step": 1, "method": "trapezoidal"},
            },
            25e-6,
        )


def test_fixed_step_combination_uses_the_registered_tableau_weights(monkeypatch):
    monkeypatch.setitem(TABLEAUS, "midpoint", ((0.0, 0.5), ((0.5,),), (0.0, 1.0)))
    state = np.array([1.0])
    derivatives = [np.array([3.0]), np.array([5.0])]
    np.testing.assert_allclose(combine("midpoint", state, 0.1, derivatives), [1.5])


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


def test_fixed_solver_observes_a_guard_inside_its_internal_step_loop():
    class Ramp:
        calls = 0

        def rhs_list(self, _t, _y):
            self.calls += 1
            return [1.0]

        def rhs(self, t, y):
            return np.asarray(self.rhs_list(t, y))

    ramp = Ramp()
    solver = FixedStepSolver(0.1, "RK4")
    result = solver.advance_guarded(
        ramp.rhs, 0.0, 1.0, np.zeros(1),
        lambda _t, y: 0.55 - y[0],
        synchronized_guard=lambda _t, y: 0.55 - y[0],
    )

    assert result.crossed
    assert result.previous_t == pytest.approx(0.5)
    assert result.previous_y[0] == pytest.approx(0.5)
    assert result.t == pytest.approx(0.6)
    assert result.y[0] == pytest.approx(0.6)
    # Six complete RK4 steps and only the first stage of the next step are needed.
    assert ramp.calls == result.n_rhs == 25


def test_stiff_fixed_solver_uses_the_full_guard_at_the_interval_end():
    class State(Bag):
        __slots__ = ("x",)

    class Decay:
        state_names = ("x",)
        outputs_need_inputs = False

        def __init__(self):
            self.state, self.input, self.output = State(x=1.0), Empty(), Empty()

        def set_outputs(self, _t):
            pass

        def rhs(self, _t):
            return (-1e5 * self.state.x,)

    matrix = np.array([[-1e5]])
    model = Model(
        {"decay": Decay()}, {},
        linear_blocks=lambda: ((np.array([0]), matrix, np.array([0]), np.array([0])),),
        linear_modes=lambda: (("mode", lambda: matrix),),
    )
    result = FixedStepSolver(1e-3, "heun", model=model).advance_guarded(
        model.rhs, 0.0, 1e-3, np.ones(1),
        lambda t, _y: 5e-4 - t,
        synchronized_guard=lambda _t, _y: 1.0,
    )

    assert result.crossed
    assert result.t == pytest.approx(1e-3)
    assert result.margin < 0.0


def test_model_automatically_solves_an_algebraic_output_loop():
    class In(Bag):
        __slots__ = ("x",)

    class Output(Bag):
        __slots__ = ("y",)

    class Feedback:
        state_names = ()
        outputs_need_inputs = True

        def __init__(self):
            self.state, self.input, self.output = Empty(), In(), Output()

        def set_outputs(self, _t):
            self.output.y = 1.0 + 0.5 * self.input.x

        def rhs(self, _t):
            return ()

    block = Feedback()
    model = Model({"feedback": block}, {(block, "x"): (block, "y")})
    model.sync(0.0, np.empty(0))

    assert len(model.algebraic_loops) == 1
    assert block.input.x == pytest.approx(2.0)
    assert block.output.y == pytest.approx(2.0)


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
            self.state, self.input, self.output = State(x=2.0), Empty(), Output()
            self.observations = 0

        def observe(self, _t):
            self.observations += 1
            self.output.x = self.state.x

        def set_outputs(self, t):
            self.observe(t)

        def rhs(self, _t):
            return (1.0,)

    class Consumer:
        state_names = ("x",)
        outputs_need_inputs = False

        def __init__(self):
            self.state, self.input, self.output = State(), Input(), State()

        def set_outputs(self, _t):
            self.output.x = self.state.x

        def rhs(self, _t):
            return (self.input.x,)

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


def test_multistep_methods_report_lightly_damped_circuit_modes(tmp_path):
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / "gfl-example.pes",
        **QUIET,
        **{
            "simulation.t_end": 5e-4,
            "simulation.solver.type": "adaptive",
            "simulation.solver.method": "BDF",
        },
    ))
    with pytest.warns(RuntimeWarning, match="lightly damped circuit modes"):
        result = simulation.run(out_dir=tmp_path)
    assert "lightly damped" in " ".join(result.summary["solver_notes"])


@pytest.mark.full
def test_device_equations_follow_commands_changed_at_a_commutation(tmp_path):
    counts = {}
    for circuit_model in ("device", "statewise"):
        params = peslite.load(EXAMPLES / "wpt-scc-example.pes", **QUIET, **{
            "simulation.solver.circuit_model": circuit_model,
            "simulation.t_end": 1e-4,
            "simulation.solver.subsystems": {},
            "simulation.solver.type": "fixed",
            "simulation.solver.method": "RK4",
            "simulation.energy_check": "off",
        })
        result = peslite.Simulation(params).run(out_dir=tmp_path / circuit_model)
        counts[circuit_model] = result.n_rhs
    assert counts["device"] == counts["statewise"]
