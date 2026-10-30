"""One conduction locator for bounds, margin crossings and scheduled boundaries."""

import math

import numpy as np
import pytest

from peslite.assembly.conduction import CommutationError, DiodeConduction
from peslite.solver.model import Bag, Model


class _Ramp:
    """x' = 1 with a discrete flag: a margin event at x = 1 and scheduled toggles."""

    state_names = ("x",)
    outputs_need_inputs = False

    def __init__(self, scheduled=(), zeno=False):
        self.state = type("_S", (Bag,), {"__slots__": ("x",)})()
        self.inp = type("_I", (Bag,), {"__slots__": ()})()
        self.out = type("_O", (Bag,), {"__slots__": ("x",)})()
        self.flag = 0
        self.at = []
        self.scheduled = list(scheduled)
        self.zeno = zeno

    def set_outputs(self, t):
        self.out.x = self.state.x

    def rhs(self, t):
        return (1.0,)

    def commutation_margin(self):
        if self.zeno:
            return -1.0
        return (1.0 - self.state.x) if self.flag == 0 else 1.0

    def commutate(self, t):
        self.at.append(t)
        if self.scheduled and abs(self.scheduled[0] - t) < 1e-12:
            self.scheduled.pop(0)
            return
        self.flag = 1
        self.state.x = 0.0  # reinitialized state, written back by the handler

    def next_commutation_time(self):
        return self.scheduled[0] if self.scheduled else math.inf


def _handler(part, dt=0.1):
    from peslite.solver.integrators import FixedStepSolver
    model = Model({"ramp": part}, {})
    return model, DiodeConduction(model, FixedStepSolver(dt, "rk4"))


def test_margin_commutation_is_located_and_its_state_written_back():
    part = _Ramp()
    model, handler = _handler(part)
    y = handler.advance(0.0, 3.0, np.zeros(1))
    assert part.at == [pytest.approx(1.0, abs=1e-14)]
    assert y[0] == pytest.approx(2.0, abs=1e-12)  # reset to 0 at t = 1, then 2 s of ramp


def test_zero_margin_plateau_does_not_move_the_commutation_to_its_start():
    part = _Ramp()
    part.commutation_margin = lambda: (
        min(0.0, 1.0 - part.state.x) if part.flag == 0 else 1.0)
    _model, handler = _handler(part, dt=0.3)
    y = handler.advance(0.0, 2.0, np.zeros(1))
    assert part.at == [pytest.approx(1.0, abs=1e-14)]
    assert y[0] == pytest.approx(1.0, abs=1e-12)


def test_commutation_uses_the_models_existing_complex_state_packing():
    part = _Ramp()
    part.state.x = 0j
    part.rhs = lambda t: (1.0 + 2.0j,)
    part.commutation_margin = lambda: 1.0 - part.state.x.real if part.flag == 0 else 1.0

    def reset(t):
        part.at.append(t)
        part.flag = 1
        part.state.x = -1.0 + 3.0j

    part.commutate = reset
    model, handler = _handler(part)
    y = handler.advance(0.0, 2.0, model.get_initial_values())
    assert part.at == [pytest.approx(1.0, abs=1e-14)]
    np.testing.assert_allclose(y, [0.0, 5.0], rtol=0.0, atol=1e-12)


def test_scheduled_commutations_end_steps_exactly():
    part = _Ramp(scheduled=[0.3333333, 0.71])
    _model, handler = _handler(part, dt=0.25)
    handler.advance(0.0, 0.9, np.zeros(1))
    assert part.at == [0.3333333, 0.71]


def test_zeno_commutation_is_rejected():
    _model, handler = _handler(_Ramp(zeno=True))
    with pytest.raises(CommutationError, match="without time advancing"):
        handler.advance(0.0, 1.0, np.zeros(1))


@pytest.mark.parametrize("start", [0.0, 0.25])
def test_initial_boundary_dispatches_due_events_and_the_following_schedule(start):
    part = _Ramp(scheduled=[start, start + 0.2])
    part.flag = 1
    _model, handler = _handler(part)
    handler.advance(start, start + 0.5, np.zeros(1))
    assert part.at == pytest.approx([start, start + 0.2], abs=1e-14)


def test_located_and_scheduled_events_on_distinct_owners_share_the_boundary():
    from peslite.solver.integrators import FixedStepSolver

    crossing = _Ramp()
    crossing.commutation_margin = lambda: (
        np.nextafter(1.0, 0.0) - crossing.state.x if crossing.flag == 0 else 1.0)
    clock = _Ramp(scheduled=[1.0])
    clock.flag = 1
    model = Model({"crossing": crossing, "clock": clock}, {})
    handler = DiodeConduction(model, FixedStepSolver(0.25, "rk4"))
    handler.advance(0.0, 1.5, np.zeros(2))
    assert crossing.at == pytest.approx([1.0], abs=1e-14)
    assert clock.at == pytest.approx([1.0], abs=1e-14)


def test_bounded_current_and_margin_event_use_the_same_advance():
    from peslite.components import RLBranch
    from peslite.solver.integrators import FixedStepSolver

    branch = RLBranch(1.0, 0.0, i0=0.5, phases=1, unilateral=True)
    branch.inp.u1 = -1.0
    ramp = _Ramp()
    model = Model({"branch": branch, "ramp": ramp}, {})
    handler = DiodeConduction(model, FixedStepSolver(0.1, "rk4"))
    state = handler.advance(0.0, 2.0, model.get_initial_values())
    assert state[model.state_slice(branch, "i")][0] == 0.0
    assert state[model.state_slice(ramp, "x")][0] == pytest.approx(1.0, abs=1e-12)
    assert ramp.at == pytest.approx([1.0], abs=1e-14)


def test_current_zero_locator_preserves_plain_float_solver_times():
    from peslite.components import RLBranch
    from peslite.solver.integrators import FixedStepSolver

    class CheckingSolver(FixedStepSolver):
        def advance(self, rhs, t0, t1, state, **kwargs):
            assert type(t0) is float and type(t1) is float
            return super().advance(rhs, t0, t1, state, **kwargs)

    branch = RLBranch(1.0, 0.0, i0=0.5, phases=1, unilateral=True)
    branch.inp.u1 = -1.0
    model = Model({"branch": branch}, {})
    handler = DiodeConduction(model, CheckingSolver(1.0, "rk4"))
    state = handler.advance(0.0, 1.0, model.get_initial_values())
    assert state[0] == 0.0
