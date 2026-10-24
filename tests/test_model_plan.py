"""Basic dependency detection and its use by the shared execution planner."""

import gc
import linecache
import random
import weakref

import pytest

from peslite.solver import Bag, Empty, Model, OutputMethod
from peslite.solver import model as kernel


def _successors(dependencies):
    successors = [[] for _ in dependencies]
    for node, deps in enumerate(dependencies):
        for dep in deps:
            successors[dep].append(node)
    return successors


def test_residual_components_exclude_blocked_downstream_nodes_from_the_loop():
    dependencies = [set(), {0}, {1, 3}, {2}, {3}, {4}, set(), {7}]
    successors = _successors(dependencies)
    order, residual = kernel._topological_order(dependencies, successors)
    assert set(order) == {0, 1, 6}
    assert set(residual) == {2, 3, 4, 5, 7}
    assert set(kernel._strong_components(successors, residual)) == {(2, 3), (4,), (5,), (7,)}


def test_dependency_analysis_matches_mutual_reachability():
    """Check SCC membership independently, including self edges and disconnected graphs."""
    rng = random.Random(317)
    for size in range(9):
        for _ in range(40):
            dependencies = [{dep for dep in range(size) if rng.random() < 0.2}
                            for _ in range(size)]
            successors = _successors(dependencies)
            order, residual = kernel._topological_order(dependencies, successors)
            position = {node: i for i, node in enumerate(order)}
            for node in order:
                assert all(dep in position and position[dep] < position[node]
                           for dep in dependencies[node])
            reach = []
            for node in range(size):
                seen, pending = {node}, [node]
                while pending:
                    for nxt in successors[pending.pop()]:
                        if nxt not in seen:
                            seen.add(nxt)
                            pending.append(nxt)
                reach.append(seen)
            expected = {tuple(member for member in range(size)
                              if member in reach[node] and node in reach[member])
                        for node in range(size)}
            actual = {(node,) for node in order}
            actual.update(kernel._strong_components(successors, residual))
            assert actual == expected
            cyclic = {node for component in expected
                      if len(component) > 1 or component[0] in dependencies[component[0]]
                      for node in component}
            assert set(residual) == {node for member in cyclic for node in reach[member]}


def test_residual_analysis_does_not_depend_on_the_python_recursion_limit():
    dependencies = [{0}] + [{node - 1} for node in range(1, 2500)]
    successors = _successors(dependencies)
    order, residual = kernel._topological_order(dependencies, successors)
    assert not order
    assert set(kernel._strong_components(successors, residual)) == {
        (node,) for node in range(len(dependencies))}


class _Input(Bag):
    __slots__ = ("x", "feedback")


class _Signal(Bag):
    __slots__ = ("x",)


class _Feedthrough:
    state_names = ()
    outputs_need_inputs = True

    def __init__(self, offset=1.0, gain=0.5):
        self.state, self.inp, self.out = Empty(), _Input(), _Signal()
        self.offset, self.gain, self.calls = offset, gain, 0

    def set_outputs(self, _t):
        self.calls += 1
        self.out.x = self.offset + self.gain * (self.inp.x + self.inp.feedback)

    def rhs(self, _t):
        return ()


@pytest.mark.parametrize("cyclic", [False, True])
def test_generated_source_cache_follows_model_lifetime(cyclic):
    def cached_models():
        return {name for name in linecache.cache if name.startswith("<peslite model ")}

    gc.collect()
    before = cached_models()
    live = Model({"live": _Feedthrough()}, {})
    filename = live.rhs_list.__code__.co_filename
    assert linecache.getlines(filename) == live.compiled_source.splitlines(keepends=True)
    released = []
    for _ in range(32):
        part = _Feedthrough()
        temporary = Model({"temporary": part}, {})
        if cyclic:
            part.owner = temporary
        released.append(weakref.ref(temporary))
        assert linecache.getlines(temporary.rhs_list.__code__.co_filename)
        del temporary, part
    gc.collect()
    assert all(ref() is None for ref in released)
    assert cached_models() == before | {filename}
    live.evaluate_outputs(0.0)
    assert live.subsystems[0].out.x == pytest.approx(1.0)
    assert linecache.getlines(filename) == live.compiled_source.splitlines(keepends=True)
    del live
    gc.collect()
    assert cached_models() == before


def test_acyclic_model_skips_scc_even_for_a_deep_signal_chain(monkeypatch):
    def unexpected_scc(*_args):
        pytest.fail("SCC analysis must not run on an acyclic model")

    monkeypatch.setattr(kernel, "_strong_components", unexpected_scc)
    blocks = [_Feedthrough(gain=1.0) for _ in range(1200)]
    connections = {(dst, "x"): (src, "x") for src, dst in zip(blocks, blocks[1:])}
    model = Model(blocks, connections)
    model.evaluate_outputs(0.0)
    assert not model.algebraic_loops
    assert blocks[-1].out.x == 1200.0
    assert len(model._plan) == 2399


def test_model_only_analyzes_residual_and_solves_distinct_loops_in_order(monkeypatch):
    observed = []
    analyze = kernel._strong_components

    def capture(successors, nodes):
        observed.append(set(nodes))
        return analyze(successors, nodes)

    monkeypatch.setattr(kernel, "_strong_components", capture)
    source = _Feedthrough(offset=2.0)
    first, second, downstream = (_Feedthrough() for _ in range(3))
    model = Model(
        {"source": source, "first": first, "second": second, "downstream": downstream},
        {(first, "x"): (source, "x"), (first, "feedback"): (first, "x"),
         (second, "x"): (first, "x"), (second, "feedback"): (second, "x"),
         (downstream, "x"): (second, "x")},
    )
    # Source output (0) and the first input copy (4) are topologically sortable.
    assert observed == [set(range(9)) - {0, 4}]
    assert [set(loop.labels) for loop in model.algebraic_loops] == [
        {"first.set_outputs", "first.inp.feedback"},
        {"second.set_outputs", "second.inp.feedback"},
    ]
    model.evaluate_outputs(0.0)
    assert first.out.x == pytest.approx(4.0)
    assert second.out.x == pytest.approx(6.0)
    assert downstream.out.x == pytest.approx(4.0)


def test_state_feedback_is_not_direct_feedthrough():
    class Integrator(_Feedthrough):
        state_names = ("x",)
        outputs_need_inputs = False

        def __init__(self):
            super().__init__()
            self.state = self.out = _Signal(x=3.0)

        def rhs(self, _t):
            return (self.inp.x,)

    state, feedback = Integrator(), _Feedthrough()
    model = Model([state, feedback], {(feedback, "x"): (state, "x"),
                                     (state, "x"): (feedback, "x")})
    assert not model.algebraic_loops
    assert model.rhs_list(0.0, [3.0]) == [2.5]


def test_output_stage_dependencies_do_not_create_an_artificial_loop():
    class Outputs(Bag):
        __slots__ = ("value", "driven")

    class Staged(_Feedthrough):
        state_names = ("x",)
        output_stages = (OutputMethod("observe", (), ("value",)),
                         OutputMethod("drive", ("x",), ("driven",)))

        def __init__(self):
            super().__init__()
            self.state, self.out = _Signal(x=3.0), Outputs()

        def observe(self, _t):
            self.out.value = self.state.x

        def drive(self, _t):
            self.out.driven = 2.0 * self.inp.x

        def rhs(self, _t):
            return (self.inp.x,)

    staged, feedback = Staged(), _Feedthrough()
    model = Model([staged, feedback], {(feedback, "x"): (staged, "value"),
                                      (staged, "x"): (feedback, "x")})
    assert not model.algebraic_loops
    assert model.rhs_list(0.0, [3.0]) == [2.5]
    assert staged.out.driven == 5.0


@pytest.mark.parametrize("cyclic", [False, True])
def test_held_input_refresh_uses_the_same_dependency_plan(cyclic):
    driven, consumer, independent = (_Feedthrough() for _ in range(3))
    connections = {(consumer, "x"): (driven, "x")}
    if cyclic:
        connections[driven, "feedback"] = driven, "x"
    model = Model([independent, consumer, driven], connections,
                  {(driven, "x"): "held"})
    model.evaluate_outputs(0.0)
    calls = independent.calls
    model.set_zoh_input("held", 4.0)
    model.sync_zoh(0.0)
    expected = 6.0 if cyclic else 3.0
    assert driven.out.x == pytest.approx(expected)
    assert consumer.out.x == pytest.approx(1.0 + 0.5 * expected)
    assert independent.calls == calls
