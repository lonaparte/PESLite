"""Array-callable black-box subsystems on the ordinary model kernel."""

from dataclasses import dataclass

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite import BlackBoxSubsystem, Simulation
from peslite.assembly.params import from_dict
from peslite.components import Element, Terminal, register_element_type
from peslite.solver import Bag, Model
from peslite.solver.energy import PowerPort


@register_element_type
class _Counter(Element):
    """A discrete model ``name`` counting its updates and a continuous lag ``name.lag``."""

    type = "test_black_box_counter"

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float
        type: str = "test_black_box_counter"

    def __init__(self, name, cfg, buses, p):
        self.name = name
        self.counter = BlackBoxSubsystem(
            lambda _t, state, _inputs: state, inputs=(), outputs=("y",),
            state={"hidden": 0.0}, update=lambda _t, state, _inputs: state + 1.0,
            period=cfg.period, feedthrough=(), observer=True,
        )
        self.lag = BlackBoxSubsystem(
            lambda _t, state, _inputs: state, inputs=("y",), outputs=("z",),
            state={"z": 0.0}, rhs=lambda _t, state, inputs: inputs - state,
            feedthrough=(), observer=True,
        )

    def subsystems(self):
        return {self.name: self.counter, f"{self.name}.lag": self.lag}

    def connections(self):
        return {(self.lag, "y"): (self.counter, "y")}

    def terminals(self):
        return ()


@register_element_type
class _Swap(Element):
    """Two discrete models taking each other's output at the same instants."""

    type = "test_black_box_swap"

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "test_black_box_swap"

    def __init__(self, name, cfg, buses, p):
        def block(initial):
            return BlackBoxSubsystem(
                lambda _t, state, _inputs: state, inputs=("other",), outputs=("y",),
                state={"hidden": initial}, update=lambda _t, _state, inputs: inputs,
                period=1.0e-3, feedthrough=(), observer=True,
            )

        self.name, self.first, self.second = name, block(1.0), block(10.0)

    def subsystems(self):
        return {f"{self.name}.first": self.first, f"{self.name}.second": self.second}

    def connections(self):
        return {(self.first, "other"): (self.second, "y"),
                (self.second, "other"): (self.first, "y")}

    def terminals(self):
        return ()


@register_element_type
class _Load(Element):
    """A stateless black-box resistor declaring its port and dissipation."""

    type = "test_black_box_load"

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bus: str
        r: float
        type: str = "test_black_box_load"

    def __init__(self, name, cfg, buses, p):
        self.name, self.cfg = name, cfg
        self.model = BlackBoxSubsystem(
            lambda _t, inputs: inputs / cfg.r,
            inputs=("u",),
            outputs=("i",),
            ports=(PowerPort("inp.u", "out.i"),),
            dissipated_power=lambda load: load.inp.u * load.out.i,
        )

    def subsystems(self):
        return {self.name: self.model}

    def connections(self):
        return {}

    def terminals(self):
        return ((self.cfg.bus, Terminal(self.model, "u", "i", 1)),)


@register_element_type
class _WindowedSampler(Element):
    """A clocked model sampling a continuous neighbour inside its solver window."""

    type = "test_black_box_windowed_sampler"

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "test_black_box_windowed_sampler"

    def __init__(self, name, cfg, buses, p):
        self.name = name
        self.ramp = BlackBoxSubsystem(
            lambda _t, state, _inputs: state,
            inputs=(), outputs=("x",), state={"x": 1.0},
            rhs=lambda _t, state, _inputs: state, feedthrough=(), observer=True,
        )
        self.sampler = BlackBoxSubsystem(
            lambda _t, state, _inputs: state,
            inputs=("x",), outputs=("sample",), state={"sample": 0.0},
            update=lambda _t, _state, inputs: inputs, period=0.15,
            feedthrough=(), observer=True,
        )

    def subsystems(self):
        return {f"{self.name}.ramp": self.ramp, f"{self.name}.sampler": self.sampler}

    def connections(self):
        return {(self.sampler, "x"): (self.ramp, "x")}

    def terminals(self):
        return ()


def _elements(elements, t_end, **simulation):
    return from_dict({"base": {"s_base": 1.0}, "elements": elements, "simulation": {
        "t_end": t_end, "solver": {"dt": 1.0e-4}, "output": {"period": 1.0e-3},
        "energy_check": "strict", **simulation,
    }})


def test_stateless_black_box_needs_only_input_and_output_callables(tmp_path):
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
    assert all(isinstance(record, Bag) for record in (block.inp, block.out, block.state))

    def bus_voltage(load):
        params = peslite.load(EXAMPLES / "buck-example.pes").replace(**{
            "units.converter.bridge.model": "switching",
            "simulation.t_end": 2.0e-3,
            "simulation.energy_check": "strict",
            "elements.load": load,
        })
        result = Simulation(params).run(out_dir=tmp_path / load["type"])
        return result.states["output.u_C"][-1]

    declared = {"type": "test_black_box_load", "bus": "output", "r": 6.0}
    built_in = {"type": "resistor", "bus": "output", "r": 6.0}
    assert bus_voltage(declared) == pytest.approx(bus_voltage(built_in), rel=1e-12)


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
    simulation = Simulation(_elements(
        {"block": {"type": "test_black_box_counter", "period": 1.0e-3}}, 2.1e-3,
    ))

    result = simulation.run(out_dir=tmp_path)

    assert simulation.system.model.state_labels() == ["block.lag.z"]
    assert result.states["block.hidden"][-1] == pytest.approx(3.0)
    assert simulation.system.named_elements["block"].counter.out.y == pytest.approx(3.0)


@pytest.mark.parametrize(("initial", "t_end", "expected"), [
    ({"t": 2.0e-3, "states": {"block.hidden": 7.0}}, 3.1e-3, 8.0),
    ({"t": 0.0, "states": {"block.lag.z": 0.5}}, 2.1e-3, 3.0),
])
def test_restored_discrete_state_does_not_repeat_the_restart_instant(
    tmp_path, initial, t_end, expected,
):
    params = _elements(
        {"block": {"type": "test_black_box_counter", "period": 1.0e-3}},
        t_end,
        initial=initial,
    )

    result = Simulation(params).run(out_dir=tmp_path)

    assert result.states["block.hidden"][-1] == pytest.approx(expected)


def test_coincident_discrete_updates_sample_before_any_state_is_committed(tmp_path):
    result = Simulation(_elements(
        {"pair": {"type": "test_black_box_swap"}},
        1.0e-4,
        solver={"dt": 1.0e-5},
        output={"period": 1.0e-4},
    )).run(out_dir=tmp_path / "pair")

    assert result.states["pair.first.hidden"][-1] == pytest.approx(10.0)
    assert result.states["pair.second.hidden"][-1] == pytest.approx(1.0)

    params = peslite.load(EXAMPLES / "buck-example.pes").replace(**{
        "elements.block": {"type": "test_black_box_counter", "period": 5.0e-5},
        "units.converter.meas.sources": {"voltage": "block.y"},
        "simulation.t_end": 3.01e-4,
        "simulation.output.signals": True,
        "simulation.output.record_every": 1,
        "simulation.energy_check": "off",
    })

    log = Simulation(params).run(out_dir=tmp_path / "measured").ctrl

    np.testing.assert_array_equal(log["converter.voltage"], np.arange(7.0))


def test_discrete_update_samples_a_settled_multirate_state(tmp_path):
    def params(multirate):
        solver = {"type": "fixed", "method": "rk4", "dt": 0.1,
                  "linearisations": 0}
        if multirate:
            solver["subsystems"] = {"block.ramp": 4}
        return _elements(
            {"block": {"type": "test_black_box_windowed_sampler"}},
            0.16,
            solver=solver,
            output={"period": 0.05},
            energy_check="off",
        )

    regular = Simulation(params(False)).run(out_dir=tmp_path / "regular")
    windowed_simulation = Simulation(params(True))
    windowed = windowed_simulation.run(out_dir=tmp_path / "windowed")

    assert windowed.states["block.sampler.sample"][-1] == pytest.approx(
        regular.states["block.sampler.sample"][-1], rel=1e-6, abs=1e-9,
    )
    assert any(abs(end - 0.15) < 1e-12
               for end, *_rest in windowed_simulation.solver.window_log)


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

    with pytest.raises(ValueError, match="identifiers"):
        BlackBoxSubsystem(lambda _t, inputs: inputs, inputs=("a.b",), outputs=("y",))
    with pytest.raises(ValueError, match=r"offset must be in \[0, period\)"):
        BlackBoxSubsystem(
            lambda _t, state, _inputs: state,
            inputs=(),
            outputs=("y",),
            state={"x": 0.0},
            update=lambda _t, state, _inputs: state,
            period=1.0e-3,
            offset=1.0e-3,
        )
    with pytest.raises(ValueError, match="observer"):
        BlackBoxSubsystem(
            lambda _t, inputs: inputs,
            inputs=("u",),
            outputs=("y",),
            observer=True,
            ports=(PowerPort("inp.u", "out.y"),),
        )
