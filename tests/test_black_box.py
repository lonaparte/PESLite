"""Array-callable black-box subsystems on the ordinary model kernel."""

import numpy as np
import pytest

from peslite import BlackBoxSubsystem, Simulation
from peslite.assembly.params import from_dict
from peslite.solver import Model


class _System:
    def __init__(self, model):
        self.model = model
        self.units = {}
        self.sources = {}
        self._port_bindings = {}

    def state_parts(self):
        return [("", self.model), *self.model.discrete_updates]

    def signals(self, _t):
        return {f"{name}.output": subsystem.out.y
                for name, subsystem in self.model.discrete_updates}


def test_stateless_black_box_needs_only_input_and_output_callables():
    block = BlackBoxSubsystem(
        lambda t, inputs: (2.0 * inputs[0] + t,),
        inputs=("u",),
        outputs=("y",),
    )
    block.inp.u = 3.0
    model = Model({"block": block}, {})

    model.evaluate_outputs(0.5)

    assert block.state_names == ()
    assert model.n_states == 0
    assert block.out.y == pytest.approx(6.5)


def test_named_input_promotes_to_complex_when_connected_signal_is_complex():
    block = BlackBoxSubsystem(
        lambda _t, inputs: inputs,
        inputs=("u",),
        outputs=("y",),
    )
    block.inp.u = 1.0 + 2.0j

    Model({"block": block}, {}).evaluate_outputs(0.0)

    assert block.out.y == 1.0 + 2.0j


def test_stateful_black_box_state_is_owned_by_model():
    block = BlackBoxSubsystem(
        lambda _t, state, _inputs: state,
        inputs=("u",),
        outputs=("y",),
        state={"filtered": 1.0},
        rhs=lambda _t, state, inputs: (inputs[0] - state[0],),
        feedthrough=(),
    )
    block.inp.u = 4.0
    model = Model({"block": block}, {})

    assert model.state_labels() == ["block.filtered"]
    assert model.rhs(0.0, np.array([1.0])) == pytest.approx([3.0])
    model.set_states(np.array([2.5]))
    model.evaluate_outputs(0.0)
    assert block.out.y == pytest.approx(2.5)


def test_vector_state_receives_stable_registered_names():
    block = BlackBoxSubsystem(
        lambda _t, state, _inputs: (state.sum(),),
        inputs=(),
        outputs=("total",),
        state=np.array([1.0, 2.0]),
        rhs=lambda _t, state, _inputs: -state,
    )
    model = Model({"block": block}, {})

    assert model.state_labels() == ["block.state_0", "block.state_1"]
    assert model.rhs(0.0, np.array([1.0, 2.0])) == pytest.approx([-1.0, -2.0])


def test_stateless_feedthrough_uses_the_existing_algebraic_loop_solver():
    block = BlackBoxSubsystem(
        lambda _t, inputs: (1.0 + 0.5 * inputs[0],),
        inputs=("u",),
        outputs=("y",),
    )
    model = Model({"block": block}, {(block, "u"): (block, "y")})

    assert len(model.algebraic_loops) == 1
    model.evaluate_outputs(0.0)
    assert block.out.y == pytest.approx(2.0)


def test_state_output_breaks_feedback_dependency():
    block = BlackBoxSubsystem(
        lambda _t, state, _inputs: state,
        inputs=("u",),
        outputs=("y",),
        state={"x": 3.0},
        rhs=lambda _t, _state, inputs: inputs,
        feedthrough=(),
    )
    model = Model({"block": block}, {(block, "u"): (block, "y")})

    assert not model.algebraic_loops
    assert model.rhs(0.0, np.array([3.0])) == pytest.approx([3.0])


def test_discrete_black_box_updates_on_the_shared_simulation_event_grid(tmp_path):
    block = BlackBoxSubsystem(
        lambda _t, state, _inputs: state,
        inputs=(),
        outputs=("y",),
        state={"hidden": 0.0},
        update=lambda _t, state, _inputs: state + 1.0,
        period=1.0e-3,
        feedthrough=(),
    )
    model = Model({"block": block}, {})
    params = from_dict({
        "base": {"s_base": 1.0},
        "simulation": {
            "t_end": 2.1e-3,
            "solver": {"dt": 1.0e-4},
            "output": {"period": 1.0e-3},
            "energy_check": "off",
        },
    })

    result = Simulation(params, system=_System(model)).run(out_dir=tmp_path)

    assert model.n_states == 0
    assert result.states["block.hidden"][-1] == pytest.approx(3.0)
    assert block.out.y == pytest.approx(3.0)


def test_restored_discrete_state_does_not_repeat_the_restart_instant(tmp_path):
    block = BlackBoxSubsystem(
        lambda _t, state, _inputs: state,
        inputs=(),
        outputs=("y",),
        state={"hidden": 0.0},
        update=lambda _t, state, _inputs: state + 1.0,
        period=1.0e-3,
        feedthrough=(),
    )
    model = Model({"block": block}, {})
    params = from_dict({
        "base": {"s_base": 1.0},
        "simulation": {
            "t_end": 3.1e-3,
            "initial": {"t": 2.0e-3, "states": {"block.hidden": 7.0}},
            "solver": {"dt": 1.0e-4},
            "output": {"period": 1.0e-3},
            "energy_check": "off",
        },
    })

    Simulation(params, system=_System(model)).run(out_dir=tmp_path)

    assert block.out.y == pytest.approx(8.0)


def test_coincident_discrete_updates_sample_before_any_state_is_committed(tmp_path):
    def block(initial):
        return BlackBoxSubsystem(
            lambda _t, state, _inputs: state,
            inputs=("other",),
            outputs=("y",),
            state={"hidden": initial},
            update=lambda _t, _state, inputs: inputs,
            period=1.0e-3,
            feedthrough=(),
        )

    first, second = block(1.0), block(10.0)
    model = Model(
        {"first": first, "second": second},
        {(first, "other"): (second, "y"), (second, "other"): (first, "y")},
    )
    params = from_dict({
        "base": {"s_base": 1.0},
        "simulation": {
            "t_end": 1.0e-4,
            "solver": {"dt": 1.0e-5},
            "output": {"period": 1.0e-4},
            "energy_check": "off",
        },
    })

    Simulation(params, system=_System(model)).run(out_dir=tmp_path)

    assert first.state.hidden == pytest.approx(10.0)
    assert second.state.hidden == pytest.approx(1.0)


def test_black_box_validates_declared_dimensions_and_dependencies():
    with pytest.raises(ValueError, match="unknown inputs"):
        BlackBoxSubsystem(
            lambda _t, inputs: inputs,
            inputs=("u",),
            outputs=("y",),
            feedthrough=("missing",),
        )

    block = BlackBoxSubsystem(
        lambda _t, _inputs: (1.0, 2.0),
        inputs=(),
        outputs=("y",),
    )
    with pytest.raises(ValueError, match="returned 2 values; expected 1"):
        Model({"block": block}, {}).evaluate_outputs(0.0)
