"""Bridge topologies and their simulation forms."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import peslite
import peslite.components
from peslite.components import (
    AveragingModel,
    FullBridge,
    HalfBridge,
    PWMAveragingModel,
    SwitchingModel,
    ThreePhaseBridge,
    make_bridge,
)


def test_bridge_topologies_share_the_existing_converter_module():
    components = Path(peslite.components.__file__).parent
    structures = (HalfBridge, FullBridge, ThreePhaseBridge)

    assert all(structure.__module__ == "peslite.components.converter" for structure in structures)
    assert not hasattr(peslite, "Bridge")
    assert not (components / "semiconductors.py").exists()
    for obsolete in (
        "Bridge",
        "ConverterStage",
        "BuckStage",
        "BoostStage",
        "InvertingStage",
        "FullBridgeStage",
        "SinglePhaseBridge",
        "Buck",
        "Boost",
        "BuckBoost",
        "DiodeFullBridge",
        "PassiveBridge",
    ):
        assert not hasattr(peslite.components, obsolete)


def _bridge_configuration(model):
    return SimpleNamespace(
        bridge=SimpleNamespace(model=model, parallel_capacitance=0.0, on_resistance=0.0),
        ctrl=SimpleNamespace(type="custom", period=1e-4, offset=None, computation=0.0, outputs={},
                             references=None),
        pwm=SimpleNamespace(
            load_period=1e-4,
            grid_offset=0.0,
            switching_period=1e-4,
            f_sw=10_000.0,
            sync="asynchronous",
            carrier_phase=0.0,
            pattern_sync="interval",
            dead_time=0.0,
            dead_time_compensation=False,
        ),
        base=SimpleNamespace(f0=50.0),
    )


@pytest.mark.parametrize("topology,duties,expected", [
    (HalfBridge(), np.array([0.75]), 0.75),
    (FullBridge(), np.array([0.75, 0.25]), 0.5),
    (
        ThreePhaseBridge(),
        np.array([0.75, 0.25, 0.5]),
        complex(0.25, -0.14433756729740643),
    ),
])
def test_each_bridge_topology_has_switching_pwm_and_averaging_forms(
        topology, duties, expected):
    switching = make_bridge(_bridge_configuration("switching"), topology=topology)
    pwm = make_bridge(_bridge_configuration("pwm_averaging"), topology=topology)
    averaging = make_bridge(_bridge_configuration("averaging"), topology=topology)

    assert isinstance(switching, SwitchingModel)
    assert isinstance(pwm, PWMAveragingModel)
    assert isinstance(averaging, AveragingModel)
    assert switching.topology is pwm.topology is averaging.topology is topology
    assert switching.duty.shape == pwm.duty.shape == duties.shape

    pwm.reset(duties, on=True)
    assert pwm.start(0.0, (None, None), continued=False) == pytest.approx(expected)

    averaging.inp.q = expected
    averaging.inp.u_dc = 48.0
    three_phase = isinstance(topology, ThreePhaseBridge)
    averaging.inp.i_c = 2.0 + 0.0j if three_phase else 2.0
    averaging.set_outputs(0.0)
    factor = 1.5 if three_phase else 1.0
    assert averaging.out.u_c == pytest.approx(48.0 * expected)
    assert averaging.out.i_dc == pytest.approx(factor * (expected * 2.0).real)


@pytest.mark.parametrize(("topology", "duties", "current", "expected"), [
    (HalfBridge(), np.array([0.75]), 2.0, 0.65),
    (HalfBridge(), np.array([1.00]), 2.0, 1.00),
    (HalfBridge(), np.array([0.00]), -2.0, 0.00),
    (HalfBridge(reverse_port=True), np.array([0.75]), 2.0, 0.15),
    (FullBridge(), np.array([0.75, 0.25]), 2.0, 0.30),
    (
        ThreePhaseBridge(),
        np.array([0.75, 0.25, 0.50]),
        2.0 + 0.0j,
        ThreePhaseBridge.pwm_state((0.65, 0.35, 0.60)),
    ),
])
def test_pwm_average_dead_time_uses_each_topologys_pole_current_mapping(
        topology, duties, current, expected):
    cfg = _bridge_configuration("pwm_averaging")
    cfg.pwm.dead_time = 1.0e-5
    bridge = make_bridge(cfg, topology=topology)
    bridge.inp.i_c = current
    bridge.reset(duties, on=True)

    assert bridge.start(0.0, (None, None), continued=False) == pytest.approx(expected)

    cfg.pwm.dead_time_compensation = True
    compensated = make_bridge(cfg, topology=topology)
    compensated.inp.i_c = current
    compensated.reset(duties, on=True)
    assert compensated.start(0.0, (None, None), continued=False) == pytest.approx(
        topology.pwm_state(duties)
    )
    assert not any(name.startswith("dead_time.") for name in compensated.get_state())


def test_pwm_dead_time_requires_controlled_devices():
    cfg = _bridge_configuration("switching")
    cfg.pwm.dead_time = 1.0e-6

    with pytest.raises(ValueError, match="requires a controlled bridge"):
        make_bridge(cfg, topology=FullBridge(upper_controlled=False, lower_controlled=False))


def test_ideal_dead_time_compensation_preserves_the_switching_sequence_without_state():
    topology = HalfBridge()
    cfg = _bridge_configuration("switching")
    cfg.pwm.dead_time = 1.0e-5
    cfg.pwm.dead_time_compensation = True
    compensated = make_bridge(cfg, topology=topology)
    baseline_cfg = _bridge_configuration("switching")
    baseline = make_bridge(baseline_cfg, topology=topology)
    for bridge in (baseline, compensated):
        bridge.inp.i_c = 2.0
        bridge.reset(np.array([0.75]), on=True)

    assert compensated.start(0.0, (None, None), False) == pytest.approx(
        baseline.start(0.0, (None, None), False)
    )
    assert compensated.dead_time == pytest.approx(1.0e-5)
    assert compensated._dead_time_sequence is None
    assert not any(name.startswith("dead_time.") for name in compensated.get_state())


def test_full_bridge_device_attributes_select_rectification_on_the_same_structure():
    controlled = FullBridge()
    rectifier = FullBridge(upper_controlled=False, lower_controlled=False)
    assert type(controlled) is type(rectifier)
    assert controlled.controlled and not rectifier.controlled
    cfg = _bridge_configuration("switching")
    cfg.bridge.parallel_capacitance = 0.0
    model = make_bridge(cfg, topology=rectifier)
    assert model.topology is rectifier and model.modulator is None
    model.inp.u_dc, model.inp.i_c, model.inp.u_hold = 100.0, 2.0, 0.0
    assert model.commutation_margin() < 0.0
    model.commutate(0.0)
    model.set_outputs(0.0)
    assert model.out.u_c == -100.0 and model.out.i_dc == -2.0


def test_half_bridge_fixes_device_placement_at_construction():
    upper_switched_leg = HalfBridge(upper_controlled=True, lower_controlled=False)
    lower_switched_leg = HalfBridge(
        upper_controlled=False, lower_controlled=True, reverse_port=True,
    )
    fully_controlled_leg = HalfBridge()

    assert upper_switched_leg.upper_controlled
    assert not upper_switched_leg.lower_controlled
    assert not lower_switched_leg.upper_controlled
    assert lower_switched_leg.lower_controlled
    assert fully_controlled_leg.upper_controlled
    assert fully_controlled_leg.lower_controlled
    assert not fully_controlled_leg.reverse_port

    reverse_fully_controlled_leg = HalfBridge(reverse_port=True)
    assert reverse_fully_controlled_leg.reverse_port
    assert reverse_fully_controlled_leg.pwm_state((0.25,)) == pytest.approx(0.75)


def test_bridge_topologies_need_no_external_device_solver():
    source = Path(__import__("peslite.components.converter", fromlist=["_"]).__file__)
    imports = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "scipy" not in imports


def test_full_bridge_produces_bipolar_switching_and_pwm_states():
    bridge = FullBridge()

    assert bridge.switching_state((1, 0)) == 1.0
    assert bridge.switching_state((0, 1)) == -1.0
    assert bridge.pwm_state((0.75, 0.25)) == pytest.approx(0.5)
    assert bridge.state_from_duty(0.75) == pytest.approx(0.5)
