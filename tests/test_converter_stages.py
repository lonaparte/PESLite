"""Bridge topologies and their simulation forms."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import peslite
import peslite.components
from peslite.components import (
    AveragingBridge,
    DeviceParameters,
    FullBridge,
    HalfBridge,
    PWMBridge,
    SwitchingBridge,
    ThreePhaseBridge,
    make_bridge,
)
from peslite.solver.model import ConfigError


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
    ):
        assert not hasattr(peslite.components, obsolete)


def _bridge_configuration(model):
    return SimpleNamespace(
        bridge=SimpleNamespace(model=model),
        ctrl=SimpleNamespace(period=1e-4, computation=0.0),
        pwm=SimpleNamespace(
            load_period=1e-4,
            grid_offset=0.0,
            switching_period=1e-4,
            f_sw=10_000.0,
            sync="asynchronous",
            carrier_phase=0.0,
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

    assert isinstance(switching, SwitchingBridge)
    assert isinstance(pwm, PWMBridge)
    assert isinstance(averaging, AveragingBridge)
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


def test_half_bridge_fixes_device_placement_at_construction():
    with pytest.raises(ConfigError, match="finite and nonnegative"):
        DeviceParameters(on_resistance=-1.0)

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

    assert upper_switched_leg.resolve((1, 0), 2.0, 48.0).path == "upper.channel"
    assert upper_switched_leg.resolve((0, 0), 2.0, 48.0).path == "lower.diode"
    assert lower_switched_leg.resolve((0, 1), 2.0, 48.0).path == "lower.channel"
    assert lower_switched_leg.resolve((0, 0), -2.0, 48.0).path == "upper.diode"
    assert fully_controlled_leg.resolve((0, 1), 2.0, 48.0).path == "lower.channel"

    with pytest.raises(ConfigError, match="diode-only"):
        upper_switched_leg.resolve((0, 1), 0.0, 48.0)
    with pytest.raises(ConfigError, match="shoot-through"):
        fully_controlled_leg.resolve((1, 1), 0.0, 48.0)


def test_piecewise_linear_current_sharing_needs_no_local_solver():
    devices = DeviceParameters(
        on_resistance=1.0,
        diode_drop=0.2,
        diode_resistance=0.1,
    )
    result = HalfBridge(devices).resolve((1, 0), -1.0, 48.0)
    channel = abs(result.channel_current)
    diode = abs(result.diode_current)

    assert channel + diode == pytest.approx(1.0)
    assert diode > 0.0
    assert devices.on_resistance * channel == pytest.approx(
        devices.diode_drop + devices.diode_resistance * diode
    )

    source = Path(__import__("peslite.components.converter", fromlist=["_"]).__file__)
    imports = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "scipy" not in imports


def test_full_bridge_is_two_controlled_half_bridges():
    bridge = FullBridge()
    positive = bridge.resolve("positive", 2.0, 48.0)
    negative = bridge.resolve("negative", 2.0, 48.0)
    zero = bridge.resolve("zero_low", 2.0, 48.0)
    blocked = bridge.resolve("blocked", 0.0, 48.0, floating=12.0)

    assert positive.voltage == 48.0
    assert positive.dc_current == 2.0
    assert positive.paths == ("upper.channel", "lower.channel")
    assert negative.voltage == -48.0
    assert zero.voltage == 0.0
    assert blocked.voltage == 12.0 and blocked.blocked
