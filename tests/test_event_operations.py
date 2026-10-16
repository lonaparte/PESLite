"""Component switching, retuning and registered circuit elements for issue #2."""

from dataclasses import dataclass

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import ConfigError, from_dict, read_tree, to_dict
from peslite.components import Element, RLBranch, register_element_type
from peslite.control import Measurement


def _tree():
    return read_tree(EXAMPLES / "gfl-example.pes")


def _load(tree, name="load", **values):
    tree["elements"] = {
        name: {"type": "load", "bus": "pcc", "x_pu": 0.5, "r_pu": 2.0, **values}
    }
    return tree


def _line(tree):
    tree["buses"]["remote"] = {"c_pu": 0.02, "r_d_pu": 0.5}
    tree.setdefault("branches", {})["line"] = {
        "bus1": "pcc", "bus2": "remote", "x_pu": 0.1, "r_pu": 0.01,
    }
    return tree


def _measurement(p, t):
    return Measurement(t, 0j, 0j, p.unit("vsc").dclink.vdc_ref, np.zeros(3))


@register_element_type
class _PassiveShunt(Element):
    """Test element without switching or retuning support."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        type: str = "test_passive_shunt"
        bus: str
        r: float
        l: float

        _quantities = {"r": "resistance", "l": "inductance"}
        _input_aliases = {"x_pu": "l_pu"}

    type = "test_passive_shunt"

    def __init__(self, name, cfg, buses, p):
        self.name, self.cfg = name, cfg
        self.branch = RLBranch(cfg.l, cfg.r)

    def subsystems(self):
        return {f"{self.name}.branch": self.branch}

    def connections(self):
        return {}

    def terminals(self):
        return ((self.cfg.bus, self.branch.terminal1),)


def test_builtin_load_is_typed_scaled_built_and_written(tmp_path):
    p = from_dict(_load(_tree()))
    cfg = p.elements["load"]
    assert (cfg.r, cfg.l) == pytest.approx(
        (2.0 * p.base.z_base, 0.5 * p.base.z_base / p.base.w0)
    )
    assert to_dict(p)["elements"]["load"]["type"] == "load"
    assert from_dict(to_dict(p)).elements == p.elements
    assert peslite.register_element_type is register_element_type

    system = peslite.System(p)
    assert system.named_elements["load"].branch in system.model.subsystems
    assert "load.branch.i" in system.model.get_state()

    quick = p.replace(**{
        "events.connect_vsc.t": 0.0,
        "events.connect_vsc.ramp": 0.0,
        "simulation.t_end": 2e-4,
        "simulation.energy_check": "off",
    })
    result = peslite.Simulation(quick).run(out_dir=tmp_path)
    assert "load.branch.i.re" in result.states


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"type": "unknown"}, "Register a custom element type"),
        ({"bus": "nowhere"}, "elements.load.bus: unknown bus"),
        ({"x_pu": -0.1}, "l must be > 0"),
        ({"r_pu": float("inf")}, "elements.load.r must be finite"),
    ],
)
def test_builtin_element_parameters_are_checked(change, message):
    tree = _load(_tree())
    tree["elements"]["load"].update(change)
    with pytest.raises(ConfigError, match=message):
        from_dict(tree)


def test_element_names_and_registrations_are_checked():
    tree = _load(_tree(), name="pcc")
    with pytest.raises(ConfigError, match="already used by buses.pcc"):
        from_dict(tree)

    class NoType(Element):
        @dataclass(frozen=True)
        class Params:
            bus: str

    with pytest.raises(TypeError, match="needs a name"):
        register_element_type(NoType)
    with pytest.raises(ValueError, match="already registered"):
        register_element_type(_PassiveShunt)


def test_system_checks_optional_custom_element_operations():
    tree = _tree()
    tree["elements"] = {"shunt": {
        "type": "test_passive_shunt", "bus": "pcc", "r_pu": 2.0, "x_pu": 0.5,
    }}
    tree["events"]["off"] = {"type": "disconnect", "target": "shunt", "t": 0.4}
    p = from_dict(tree)
    with pytest.raises(ConfigError, match="has no breakers and no connect"):
        peslite.Simulation(p)

    tree["events"].pop("off")
    tree["events"]["retune"] = {
        "type": "set", "t": 0.4, "set": {"elements.shunt.r_pu": 1.0},
    }
    p = from_dict(tree)
    with pytest.raises(ConfigError, match=r"elements\.shunt has no retune"):
        peslite.Simulation(p)


def test_system_switches_units_sources_and_loads():
    tree = _line(_load(_tree()))
    tree["units"]["vsc"]["bridge"] = {"model": "pwm_averaging"}
    tree["events"]["connect_vsc"].update(t=0.0, ramp=0.0)
    tree["events"].update({
        "unit_off": {"type": "disconnect", "target": "vsc", "t": 0.1},
        "unit_on": {"type": "connect", "target": "vsc", "t": 0.2, "ramp": 0.1},
        "grid_off": {"type": "disconnect", "target": "grid", "t": 0.1},
        "grid_on": {"type": "connect", "target": "grid", "t": 0.2},
        "load_off": {"type": "disconnect", "target": "load", "t": 0.1},
        "load_on": {"type": "connect", "target": "load", "t": 0.2},
        "line_off": {"type": "disconnect", "target": "line", "t": 0.1},
        "line_on": {"type": "connect", "target": "line", "t": 0.2},
    })
    p = from_dict(tree)
    system = peslite.System(p)
    system.check_events(p)

    for name, branch in (("grid", system.sources["grid"].branch),
                         ("load", system.named_elements["load"].branch),
                         ("line", system.branches["line"])):
        branch.state.i = 2 + 3j
        system.switch(name, False, 0.1)
        assert branch.breaker_open and branch.state.i == 0j
        system.switch(name, True, 0.2)
        assert not branch.breaker_open

    unit = system.units["vsc"]
    source = unit.dclink.source
    system.switch("vsc", False, 0.1)
    assert unit.branch_f.breaker_open and source(0.15, source.u_ref) == 0.0
    cc = unit.ctrl.graph.nodes["CurrentLoop"]
    dvc = unit.ctrl.graph.nodes["DCVoltageControl"]
    cc.integral, dvc.integral = 0.2 + 0.1j, 0.3
    unit.ctrl(0.15, _measurement(p, 0.15))
    assert not unit.ctrl.startup.active
    assert cc.integral == 0.2 + 0.1j and dvc.integral == 0.3

    system.switch("vsc", True, 0.2, 0.1)
    assert unit.breaker and unit.branch_f.breaker_open  # gates stay blocked until a PWM load enables them
    assert 0.0 < source(0.25, source.u_ref) < source.i_nom
    unit.trip(0.25, "test")
    system.switch("vsc", True, 0.3)
    assert unit.tripped and unit.branch_f.breaker_open


def test_system_applies_retunes_and_control_loops_keep_their_state():
    tree = _line(_load(_tree()))
    tree["units"]["vsc"]["bridge"] = {"model": "pwm_averaging"}
    tree["units"]["vsc"]["ctrl"]["loops"]["DCVoltageControl"]["antiwindup"] = 0
    tree["events"]["connect_vsc"].update(t=0.0, ramp=0.0)
    tree["events"]["retune"] = {"type": "set", "t": 0.001, "set": {
        "buses.pcc.c_pu": 0.03,
        "buses.pcc.r_d_pu": 0.25,
        "sources.grid.x_pu": 0.8,
        "sources.grid.r_pu": 0.08,
        "sources.grid.v_pu": 0.9,
        "sources.grid.f": 51.0,
        "sources.grid.angle": 0.1,
        "branches.line.x_pu": 0.2,
        "branches.line.r_pu": 0.02,
        "elements.load.r_pu": 1.0,
        "units.vsc.dclink.source.i_pu": 0.5,
        "units.vsc.ctrl.references.p_ref_pu": 0.4,
        "units.vsc.ctrl.loops.SRFPLL.kp_pu": 10.0,
        "units.vsc.ctrl.loops.DCVoltageControl.kp": 0.4,
        "units.vsc.protection.hold": 0.01,
        "units.vsc.protection.rocof.window": 0.2,
    }}
    p = from_dict(tree)
    change = p.changes[0]
    system = peslite.System(p)
    system.check_events(p)

    bus = system.buses["pcc"]
    source = system.sources["grid"]
    line = system.branches["line"]
    load = system.named_elements["load"]
    unit = system.units["vsc"]
    bus.state.u_C = 3 + 4j
    source.branch.state.i = 5 + 6j
    line.state.i = 6 + 7j
    load.branch.state.i = 7 + 8j
    old_pll = unit.ctrl.graph.nodes["SRFPLL"]
    old_pll.theta, old_pll.integral = 1.2, 0.3
    old_dvc = unit.ctrl.graph.nodes["DCVoltageControl"]
    old_dvc.integral, old_dvc.n_updates, old_dvc.n_limit_exceeded = 0.2, 5, 2
    old_dvc.n_reverse, old_dvc.first_limit_t = 1, 4e-4
    system.apply(change)
    current = change.params
    assert bus.state.u_C == 3 + 4j
    assert (bus.C, bus.R_d) == (current.buses["pcc"].c, current.buses["pcc"].r_d)
    assert source.branch.state.i == 5 + 6j
    assert (source.branch.L, source.branch.R, source.emf.e_peak) == pytest.approx(
        (current.sources["grid"].l, current.sources["grid"].r,
         current.sources["grid"].v)
    )
    assert source.emf.phi(change.t) == pytest.approx(0.1)
    assert source.emf.phi(change.t + 0.001) == pytest.approx(0.1 + 2 * np.pi * 0.001)
    assert line.state.i == 6 + 7j
    assert (line.L, line.R) == (current.branches["line"].l, current.branches["line"].r)
    assert load.branch.state.i == 7 + 8j and load.branch.R == current.elements["load"].r
    assert unit.dclink.source.i_nom == current.unit("vsc").dclink.source.i
    assert unit.protection.cfg.hold == 0.01
    assert unit.protection._rocof_window == pytest.approx(0.2)
    assert unit.ctrl.graph.nodes["SRFPLL"] is old_pll  # queued until the next controller update

    unit.ctrl(change.t, _measurement(p, change.t))
    pll = unit.ctrl.graph.nodes["SRFPLL"]
    assert pll is not old_pll and pll.kp == 10.0
    # The retuned loop first takes over the old state, then performs this interrupt's update.
    assert pll.integral == pytest.approx(0.3)
    assert pll.theta == pytest.approx(1.2 + pll.T * (pll.w0 + pll.ki * 0.3))
    dvc = unit.ctrl.graph.nodes["DCVoltageControl"]
    assert dvc is not old_dvc and dvc.cfg.kp == 0.4 and dvc.kp == -0.4
    assert dvc.integral == 0.2
    assert (dvc.n_updates, dvc.n_limit_exceeded, dvc.n_reverse, dvc.first_limit_t) == (6, 3, 2, 4e-4)
    assert unit.ctrl.graph.references["p_ref_pu"] == 0.4

    specs = system.model.energy_specs
    assert specs["pcc"].storage[0].value == current.buses["pcc"].c
    assert specs["grid.branch"].storage[0].value == current.sources["grid"].l
    assert specs["line"].storage[0].value == current.branches["line"].l
    assert specs["load.branch"].storage[0].value == current.elements["load"].l


def test_continuous_controller_retune_rebinds_its_ode_states(tmp_path):
    tree = _tree()
    tree["events"]["connect_vsc"].update(t=0.0, ramp=0.0)
    tree["events"]["retune"] = {"type": "set", "t": 1e-4, "set": {
        "units.vsc.ctrl.loops.SRFPLL.kp_pu": 10.0,
        "units.vsc.ctrl.loops.DCVoltageControl.kp": 0.4,
    }}
    tree["simulation"].update(t_end=2e-4, energy_check="off")
    tree["simulation"]["progress"] = {"enable": 0}
    simulation = peslite.Simulation(from_dict(tree))

    result = simulation.run(out_dir=tmp_path)

    assert simulation.unit().ctrl.graph.nodes["SRFPLL"].kp == 10.0
    assert simulation.unit().ctrl.graph.nodes["DCVoltageControl"].kp == -0.4
    assert all(np.isfinite(values[-1]) for values in result.states.values())
