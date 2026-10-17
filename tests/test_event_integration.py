"""Event-time integration semantics and optional solver hooks."""

import math
from dataclasses import dataclass

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.assembly import Event, register_event_type
from peslite.components.network import RCNode
from peslite.solver import MultirateSolver, SolverStep
from peslite.solver.model import Bag, Empty, Model


def _gfl(**changes):
    return peslite.load(
        EXAMPLES / "gfl-example.pes",
        **{
            "simulation.energy_check": "off",
            "simulation.solver.linearisations": 0,
            **changes,
        },
    )


def _set(name, t, **values):
    return {f"events.{name}": {"type": "set", "t": t, "set": values}}


RUNS = []


@register_event_type
class _MarkAtIntegrationTime(Event):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "test_integration_mark"
        t: float
        label: str = "mark"

    type = "test_integration_mark"

    @staticmethod
    def apply(event, system, t):
        RUNS.append((event.label, system, t))


class _HookedHeun:
    """Small user solver which records interval ends and the optional hook order."""

    def __init__(self, dt):
        self.dt = dt
        self.n_rhs = 0
        self.ends = []
        self.stages = []
        self.hooks = []
        self.source = None

    def __call__(self, f, t0, t1, y):
        n = max(1, math.ceil((t1 - t0) / self.dt - 1e-9))
        h = (t1 - t0) / n
        for k in range(n):
            t = t0 + k * h
            self.stages.append((t, self.source.branch.R))
            k1 = f(t, y)
            self.stages.append((t + h, self.source.branch.R))
            y = y + 0.5 * h * (k1 + f(t + h, y + h * k1))
        self.n_rhs += 2 * n
        self.ends.append(t1)
        return SolverStep(t1, y, self.n_rhs)

    def settle(self, t, y):
        self.hooks.append(("settle", t))
        return y

    def parameters_changed(self):
        self.hooks.append(("parameters", None))


class _PlainHeun:
    """User solver without optional hooks; its last stage is at each interval end."""

    def __init__(self, dt):
        self.dt, self.n_rhs = dt, 0

    def __call__(self, f, t0, t1, y):
        steps = max(1, math.ceil((t1 - t0) / self.dt - 1e-9))
        step = (t1 - t0) / steps
        for index in range(steps):
            t = t0 + index * step
            first = f(t, y)
            y = y + 0.5 * step * (first + f(t + step, y + step * first))
        self.n_rhs += 2 * steps
        return SolverStep(t1, y, self.n_rhs)


def test_event_is_an_exact_stop_and_solver_hooks_bracket_the_model_change(tmp_path):
    event_t = 0.000137
    p = _gfl(**{"simulation.t_end": 0.0003},
             **_set("retune", event_t, **{"sources.grid.r_pu": 0.08}))
    solver = _HookedHeun(p.simulation.solver.dt)
    sim = peslite.Simulation(p, solver=solver)
    solver.source = sim.system.sources["grid"]
    old_r = solver.source.branch.R
    apply = sim.system.apply

    def recorded_apply(change):
        solver.hooks.append(("apply", change.t))
        apply(change)

    sim.system.apply = recorded_apply
    sim.run(out_dir=tmp_path)

    assert any(abs(end - event_t) < 1e-14 for end in solver.ends)
    assert solver.hooks == [
        ("parameters", None),
        ("settle", event_t),
        ("apply", event_t),
        ("parameters", None),
    ]
    at_event = [resistance for t, resistance in solver.stages if abs(t - event_t) < 1e-14]
    before_event = [resistance for t, resistance in solver.stages if t < event_t - 1e-14]
    after_event = [resistance for t, resistance in solver.stages if t > event_t + 1e-14]
    assert before_event and all(value == old_r for value in before_event)
    assert at_event == pytest.approx([old_r, 2 * old_r])
    assert after_event and all(value == pytest.approx(2 * old_r) for value in after_event)


def test_registered_custom_event_runs_once_at_its_exact_time(tmp_path):
    event_t = 0.000137
    p = _gfl(**{
        "simulation.t_end": 0.0003,
        "events.mark": {"type": "test_integration_mark", "t": event_t, "label": "seen"},
    })
    sim = peslite.Simulation(p)
    RUNS.clear()
    sim.run(out_dir=tmp_path)
    assert RUNS == [("seen", sim.system, event_t)]


def test_set_event_before_restart_time_is_in_force_before_the_first_stage(tmp_path):
    p = _gfl(**{
        "simulation.initial.t": 0.0002,
        "simulation.t_end": 0.0004,
        **_set("retune", 0.0001, **{"sources.grid.r_pu": 0.08}),
    })
    solver = _HookedHeun(p.simulation.solver.dt)
    sim = peslite.Simulation(p, solver=solver)
    solver.source = sim.system.sources["grid"]
    old_r = solver.source.branch.R
    sim.run(out_dir=tmp_path)
    assert solver.stages[0][0] == pytest.approx(0.0002)
    assert solver.stages[0][1] == pytest.approx(2 * old_r)


class _ConstantFlow:
    class Out(Bag):
        __slots__ = ("flow",)

    state_names = ()
    outputs_need_inputs = False

    def __init__(self, flow):
        self.flow = flow
        self.state, self.inp, self.out = Empty(), Empty(), self.Out()

    def set_outputs(self, _t):
        self.out.flow = self.flow

    def rhs(self, _t):
        return ()


@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "DP45"])
def test_shortened_multirate_window_uses_its_length_and_keeps_the_grid(method):
    def build():
        flow, capacitor = _ConstantFlow(2.0), RCNode(1.0, 0.0)
        model = Model({"flow": flow, "capacitor": capacitor},
                      {(capacitor, "i"): (flow, "flow")})
        solver = MultirateSolver(
            model, {"capacitor": {"step": 4, "method": method}}, dt=0.001
        )
        return flow, model, solver

    flow, model, solver = build()
    y = solver(model.rhs, 0.0, 0.0015, model.get_initial_values()).y
    y = solver.settle(0.0015, y)
    assert y[0] == pytest.approx(0.003, abs=1e-14)
    assert solver.get_state()["window_end"] == pytest.approx(0.004)
    saved = solver.get_state()
    state_at_event = y.copy()

    flow.flow = 5.0
    y = solver(model.rhs, 0.0015, 0.0055, y).y
    final = solver.settle(0.0055, y)
    assert final[0] == pytest.approx(0.023, abs=1e-14)
    assert [entry[0] for entry in solver.window_log] == pytest.approx([0.0015, 0.004, 0.0055])

    other_flow, other_model, other = build()
    other_flow.flow = 5.0
    other.set_state(saved)
    continued = other(other_model.rhs, 0.0015, 0.0055, state_at_event).y
    assert np.array_equal(final, other.settle(0.0055, continued))


def test_multirate_refreshes_storage_parameters_after_a_set_event(tmp_path):
    event_t = 0.000137
    p = _gfl(**{
        "simulation.t_end": 0.00015,
        "simulation.solver.subsystems": {"grid.branch": 4},
        **_set("retune", event_t, **{"sources.grid.x_pu": 0.8}),
    })
    sim = peslite.Simulation(p)
    old_inductance = sim.system.sources["grid"].branch.L
    sim.run(out_dir=tmp_path)
    held = {storage.label: storage for storage in sim.solver.held_storages}
    assert held["grid.branch.i"].value == pytest.approx(2 * old_inductance)


@pytest.mark.parametrize("kind", ["rk4", "DP45", "multirate", "user"])
def test_fixed_adaptive_multirate_and_user_solver_use_the_right_model_at_event(kind, tmp_path):
    event_t = 0.00137
    solver_settings = {
        "rk4": {},
        "DP45": {"simulation.solver.type": "adaptive", "simulation.solver.method": "DP45"},
        "multirate": {"simulation.solver.subsystems": {"grid.branch": 4}},
        "user": {},
    }[kind]
    params = _gfl(
        **{"simulation.t_end": 0.002, **solver_settings},
        **_set("jump", event_t, **{"sources.grid.angle": 0.1, "sources.grid.r_pu": 0.08}),
    )
    simulation = peslite.Simulation(
        params,
        solver=_PlainHeun(params.simulation.solver.dt) if kind == "user" else None,
    )
    model, source = simulation.system.model, simulation.system.sources["grid"]
    old_resistance = source.branch.R
    apply, seen = simulation.system.apply, []
    changed = False

    def recorded_apply(change):
        nonlocal changed
        apply(change)
        changed = True

    def watch(rhs):
        def watched(t, *args, **kwargs):
            value = rhs(t, *args, **kwargs)
            if changed:
                assert t >= event_t - 1e-14
            else:
                assert source.branch.R == old_resistance
                assert source.emf.out.phi == 0.0
            seen.append((t, changed))
            return value
        return watched

    simulation.system.apply = recorded_apply
    model.rhs_list = watch(model.rhs_list)
    model.rhs_group = watch(model.rhs_group)
    simulation.run(out_dir=tmp_path)

    assert any(abs(t - event_t) < 1e-14 and not after for t, after in seen)
    assert source.emf.out.phi == 0.1
    assert source.branch.R == pytest.approx(2 * old_resistance)
    if kind == "multirate":
        ends = np.array([entry[0] for entry in simulation.solver.window_log])
        assert np.any(abs(ends - event_t) < 1e-14)
        regular = ends[abs(ends - event_t) >= 1e-14] / simulation.solver.window
        assert np.max(abs(regular - np.round(regular))) < 1e-10


def test_run_continued_across_an_event_matches_the_whole_run(tmp_path):
    event_t = 0.00137
    changes = {
        "simulation.t_end": 0.003,
        "simulation.solver.subsystems": {"grid.branch": 4},
        **_set("jump", event_t, **{"sources.grid.angle": 0.1, "sources.grid.r_pu": 0.08}),
    }
    whole = peslite.Simulation(_gfl(**changes)).run(out_dir=tmp_path / "whole")
    peslite.Simulation(_gfl(**changes)).run(event_t, out_dir=tmp_path / "first")
    continued_params = peslite.load(
        EXAMPLES / "gfl-example.pes",
        initial=tmp_path / "first" / "states.csv",
        **{"simulation.energy_check": "off", "simulation.solver.linearisations": 0, **changes},
    )
    continued = peslite.Simulation(continued_params).run(out_dir=tmp_path / "continued")
    assert continued.final_states() == whole.final_states()


def test_disconnected_converter_pauses_control_and_a_trip_cannot_reconnect(tmp_path):
    protections_off = {
        f"units.vsc.protection.{name}.enable": 0
        for name in ("overcurrent", "undervoltage", "overvoltage", "frequency", "dc_voltage", "rocof")
    }
    p = _gfl(**protections_off, **{
        "simulation.t_end": 0.0012,
        "simulation.output.period": 25e-6,
        "events.connect_vsc.t": 0.0002,
        "events.connect_vsc.ramp": 0.0,
        "events.off": {"type": "disconnect", "target": "vsc", "t": 0.0006},
        "events.on": {"type": "connect", "target": "vsc", "t": 0.0009},
    })
    result = peslite.Simulation(p).run(out_dir=tmp_path / "disconnected")
    t = result.states["t"]
    integral = result.states["vsc.ctrl.CurrentLoop.integral_pu.re"]
    assert np.ptp(integral[t < 0.0002 - 1e-12]) == 0.0
    assert np.ptp(integral[(t > 0.0006 + 1e-12) & (t < 0.0009 - 1e-12)]) == 0.0

    tripping = _gfl(**{
        "simulation.t_end": 0.004,
        "simulation.stop_on_trip": 0,
        "events.connect_vsc.t": 0.0,
        "events.connect_vsc.ramp": 0.0,
        "units.vsc.protection.overcurrent.limit_pu": 1e-3,
        "events.off": {"type": "disconnect", "target": "vsc", "t": 0.002},
        "events.on": {"type": "connect", "target": "vsc", "t": 0.003},
    })
    with pytest.warns(UserWarning, match="has tripped.*leaves it disconnected"):
        sim = peslite.Simulation(tripping)
        result = sim.run(out_dir=tmp_path / "tripped")
    assert result.tripped
    assert sim.unit().tripped and sim.unit().branch_f.breaker_open
