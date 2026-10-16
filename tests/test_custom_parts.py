"""Custom loop, element and event example code lives in tests, not examples/."""

from dataclasses import dataclass

import numpy as np
import pytest

import peslite
from peslite.assembly import Event, register_event_type
from peslite.assembly.params import ConfigError, from_dict
from peslite.components import Element, RLBranch, register_element_type
from peslite.control import SyncLaw, register_loop_type
from peslite.solver import SolverStep


@register_loop_type
class _LaggedPSC(SyncLaw):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float
        k_p_pu: float
        tau: float
        type: str = "test_lagged_psc"

        _quantities = {"k_p_pu": "1/power"}

        def __post_init__(self):
            if not self.tau > 0.0:
                raise ConfigError(f"tau must be > 0, got {self.tau}")

    type = "test_lagged_psc"
    state_names = {"theta": "theta", "p_f_pu": "p_f"}

    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        self.p_f = 0.0

    def _sample(self, period, p_pu, q_pu, v_mag_pu, v_dc_pu,
                p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.p_f += period / self.cfg.tau * (p_pu - self.p_f)
        self.omega = self.w0 + self.cfg.k_p_pu * (p_ref_pu - self.p_f)
        self.theta += period * self.omega
        self.v_mag = v_ref_pu


@register_element_type
class _ShuntRL(Element):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "test_shunt_rl"
        bus: str
        r: float
        l: float

        _quantities = {"r": "resistance", "l": "inductance"}
        _input_aliases = {"x_pu": "l_pu"}

        def __post_init__(self):
            if not (self.l > 0.0 and self.r >= 0.0):
                raise ConfigError(f"l must be > 0 and r >= 0, got l = {self.l}, r = {self.r}")

    type = "test_shunt_rl"

    def __init__(self, name, cfg, buses, _params):
        self.name, self.cfg = name, cfg
        self.branch = RLBranch(cfg.l, cfg.r)

    def subsystems(self):
        return {f"{self.name}.branch": self.branch}

    def connections(self):
        return {}

    def terminals(self):
        return ((self.cfg.bus, self.branch.terminal1),)

    def retune(self, cfg):
        self.cfg = cfg
        self.branch.retune(cfg.l, cfg.r)


@register_event_type
class _LoadStep(Event):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "test_load_step"
        target: str
        t: float
        factor: float

        def __post_init__(self):
            if not self.factor > 0.0:
                raise ConfigError(f"factor must be > 0, got {self.factor}")

    type = "test_load_step"

    @staticmethod
    def apply(event, system, _t):
        branch = system.named_elements[event.target].branch
        branch.retune(branch.L, branch.R * event.factor)


class _Midpoint:
    """User solver without optional event hooks."""

    def __init__(self, dt):
        self.dt, self.n_rhs, self.ends = dt, 0, []

    def __call__(self, f, t0, t1, y0):
        steps = max(1, int(np.ceil((t1 - t0) / self.dt - 1e-9)))
        step, state, t = (t1 - t0) / steps, y0, t0
        for _ in range(steps):
            first = f(t, state)
            state = state + step * f(t + 0.5 * step, state + 0.5 * step * first)
            t += step
        self.n_rhs += 2 * steps
        self.ends.append(t1)
        return SolverStep(t1, state, self.n_rhs)


def _custom_case(case):
    tree = case()
    unit = tree["units"]["vsc"]
    unit["dclink"] = {"vdc_ref": 1500.0, "source": {"type": "voltage"}}
    unit["ctrl"] = {
        "type": "gfm",
        "loops": {
            "sync": {"type": "test_lagged_psc", "period": 5e-5,
                     "k_p_pu": 6.2832, "tau": 0.02},
            "vi": {"type": "virtual_impedance"},
            "damp": {"type": "active_damping", "period": 5e-5,
                     "r_a_pu": 0.2, "alpha_d": 40.0},
        },
        "references": {"p_ref_pu": 0.5},
    }
    tree["elements"] = {
        "load": {"type": "test_shunt_rl", "bus": "pcc", "r_pu": 5.0, "x_pu": 1.0}
    }
    tree["events"] = {
        "connect_vsc": {"type": "connect", "target": "vsc", "t": 0.0},
        "load_step": {"type": "test_load_step", "target": "load", "t": 0.002, "factor": 0.5},
    }
    tree["simulation"].update(t_end=0.003, energy_check="off")
    return tree


def test_registered_custom_parts_and_user_solver_work_together(case, tmp_path):
    params = from_dict(_custom_case(case))
    solver = _Midpoint(params.simulation.solver.dt)
    simulation = peslite.Simulation(params, solver=solver)
    load = simulation.system.named_elements["load"].branch
    original_resistance = load.R
    result = simulation.run(out_dir=tmp_path)

    assert "vsc.ctrl.sync.p_f_pu" in result.states
    assert load.R == pytest.approx(0.5 * original_resistance)
    assert any(abs(end - 0.002) < 1e-14 for end in solver.ends)
    assert result.n_rhs > 0


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("units.vsc.ctrl.loops.sync.tau", 0.0, "tau must be > 0"),
        ("elements.load.bus", "missing", "unknown bus"),
        ("events.load_step.factor", -1.0, "factor must be > 0"),
    ],
)
def test_custom_part_parameters_are_validated(case, path, value, message):
    tree = _custom_case(case)
    params = from_dict(tree)
    with pytest.raises(ConfigError, match=message):
        params.replace(**{path: value})
