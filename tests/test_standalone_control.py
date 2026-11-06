"""Standalone controllers use the normal configuration and sampled-unit runtime."""

from dataclasses import dataclass, field

import numpy as np
import pytest

import peslite
from peslite.assembly.params import ConfigError, from_dict, runtime_changeable, to_dict
from peslite.control import ControlInterface, ControlOutput, Startup
from peslite.control.profiles import register_control_type


@dataclass(frozen=True)
class _References:
    duty: float = 0.2


@dataclass(frozen=True)
class _FixedModel:
    coefficient: float = 1.0


@dataclass(frozen=True)
class _Options:
    gain: float = 0.5
    size: int = field(default=1, metadata={"structural": True})
    model: _FixedModel = field(default_factory=_FixedModel, metadata={"runtime": False})


def _validate_control(ctrl, where):
    if not 0.0 <= ctrl.references.duty * ctrl.options.gain <= 1.0:
        raise ConfigError(f"{where}.options.gain: commanded duty must be in [0, 1]")


class _StandaloneController:
    continuous = False

    def __init__(self, cfg):
        self.p = cfg
        self.startup = Startup(cfg.ctrl.period, run=False)
        self.duty = 0.0

    def __call__(self, t, measurement):
        self.startup.advance()
        self.duty = self.p.ctrl.references.duty * self.p.ctrl.options.gain
        return ControlOutput(
            np.array([self.duty, 0.0, 0.0]), gates=self.startup.active,
            startup_complete=self.startup.complete,
        )

    def command(self, run, ramp):
        self.startup.command(run, ramp)

    def retune(self, cfg, paths, t):
        self.p = cfg

    def get_state(self):
        return {"duty": self.duty,
                **{f"startup.{key}": value for key, value in self.startup.get_state().items()}}

    def set_state(self, values):
        self.duty = values.get("duty", self.duty)
        self.startup.set_state({key[8:]: value for key, value in values.items()
                                if key.startswith("startup.")})


register_control_type(
    "test_standalone", references=_References, options=_Options,
    unit_types=frozenset(("vsc",)), default_bridge="three_phase_bridge",
    interface=ControlInterface(inputs={}, outputs={}),
    controller_factory=_StandaloneController, requires_loops=False,
    supported_models=frozenset(("switching", "pwm_averaging")), validator=_validate_control,
)


def _case(case):
    tree = case()
    tree["units"]["vsc"]["ctrl"] = {"type": "test_standalone"}
    tree["units"]["vsc"]["connected"] = True
    tree["simulation"]["t_end"] = 0.0002
    return tree


@pytest.mark.static
def test_registered_standalone_schema_without_loops_round_trips(case):
    params = from_dict(_case(case))
    ctrl = params.unit("vsc").ctrl
    assert ctrl.loops == {}
    assert ctrl.options == _Options()
    assert ctrl.references == _References()
    assert from_dict(to_dict(params)).unit("vsc").ctrl == ctrl


@pytest.mark.static
def test_controller_options_preserve_runtime_and_structural_boundaries(case):
    params = from_dict(_case(case))
    assert runtime_changeable("units.vsc.ctrl.options.gain", params)
    assert runtime_changeable("units.vsc.ctrl.references.duty", params)
    for path in ("options.size", "options.model.coefficient", "options.unknown", "period",
                 "type", "connections", "outputs"):
        assert not runtime_changeable(f"units.vsc.ctrl.{path}", params), path
    tree = _case(case)
    tree["events"] = {"resize": {"type": "set", "t": 0.0001,
                                  "set": {"units.vsc.ctrl.options.size": 2}}}
    with pytest.raises(ConfigError, match="options.size"):
        from_dict(tree)


@pytest.mark.static
def test_standalone_schema_rejects_unknown_options_and_invalid_combinations(case):
    tree = _case(case)
    tree["units"]["vsc"]["ctrl"]["options"] = {"unknown": 1}
    with pytest.raises(ConfigError, match="options.*unknown"):
        from_dict(tree)
    tree["units"]["vsc"]["ctrl"]["options"] = {"gain": 8.0}
    with pytest.raises(ConfigError, match="commanded duty"):
        from_dict(tree)


@pytest.mark.static
def test_standalone_type_rejects_unsupported_bridge_model_before_assembly(case):
    tree = _case(case)
    tree["units"]["vsc"]["bridge"]["model"] = "averaging"
    with pytest.raises(ConfigError, match="bridge.model.*supports only"):
        from_dict(tree)


@pytest.mark.static
def test_existing_graph_controller_still_requires_loops_and_has_no_options(case):
    tree = case()
    del tree["units"]["vsc"]["ctrl"]["loops"]
    with pytest.raises(ConfigError, match="loops must contain at least one"):
        from_dict(tree)
    tree = case()
    tree["units"]["vsc"]["ctrl"]["options"] = {"gain": 1.0}
    with pytest.raises(ConfigError, match="does not accept options"):
        from_dict(tree)


@pytest.mark.parametrize("bridge_model", ["switching", "pwm_averaging"])
def test_standalone_controller_runs_and_retunes_through_existing_unit(case, tmp_path, bridge_model):
    tree = _case(case)
    tree["units"]["vsc"]["bridge"]["model"] = bridge_model
    tree["events"] = {"retune": {"type": "set", "t": 0.0001, "set": {
        "units.vsc.ctrl.options.gain": 1.5,
        "units.vsc.ctrl.references.duty": 0.4,
    }}}
    simulation = peslite.Simulation(from_dict(tree))
    simulation.run(out_dir=tmp_path)
    controller = simulation.system.units["vsc"].ctrl
    assert controller.duty == pytest.approx(0.6)
    assert controller.startup.complete
    assert simulation.system.get_state()["vsc.ctrl.duty"] == pytest.approx(0.6)
