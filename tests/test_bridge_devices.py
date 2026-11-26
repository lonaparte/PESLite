"""Independent switch-channel and natural-diode bridge resistance."""

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.components import FullBridge, HalfBridge, ThreePhaseBridge, make_bridge
from peslite.solver import netlist as nl
from peslite.solver.model import ConfigError, Model
from test_converter_stages import _bridge_configuration


def _custom_case():
    return peslite.load(EXAMPLES / "buck-open-loop-example.pes").replace(**{
        "units.converter.bridge.model": "switching",
        "units.converter.bridge.lower_controlled": False,
        "units.converter.bridge.switch_resistance": 0.035,
        "units.converter.bridge.diode_resistance": 0.045,
        "simulation.t_end": 0.0001,
        "simulation.output.period": 2e-6,
        "simulation.output.precision": 15,
        "simulation.output.energy": True,
        "simulation.solver.dt": 2.5e-7,
        "simulation.solver.linearisations": 0,
    })


def _devices(bridge):
    return () if bridge._netlist_devices is None else bridge._netlist_devices[1]


def test_bridge_uses_independent_switch_and_diode_resistance():
    bridge = peslite.Simulation(_custom_case()).unit("converter").bridge
    high, low = _devices(bridge)
    assert high.R_gated == pytest.approx(0.035)
    assert high.R_on == pytest.approx(0.045)
    assert low.gate is None
    assert low.R_on == low.R_gated == pytest.approx(0.045)


@pytest.mark.parametrize("topology", [HalfBridge(), FullBridge(), ThreePhaseBridge()])
def test_bridge_topologies_share_independent_device_resistance(topology):
    cfg = _bridge_configuration("switching")
    cfg.bridge.switch_resistance = 0.035
    cfg.bridge.diode_resistance = 0.045
    bridge = make_bridge(cfg, topology=topology)
    Model({"bridge": bridge}, {})

    assert _devices(bridge)
    for device in _devices(bridge):
        assert device.R_on == pytest.approx(0.045)
        assert device.R_gated == pytest.approx(
            0.035 if device.gate is not None else 0.045
        )


def test_device_resistance_defaults_are_ideal():
    params = peslite.load(EXAMPLES / "buck-open-loop-example.pes").replace(**{
        "units.converter.bridge.model": "switching",
        "units.converter.bridge.lower_controlled": False,
    })
    assert params.units["converter"].bridge.switch_resistance == 0.0
    assert params.units["converter"].bridge.diode_resistance == 0.0
    high, low = _devices(peslite.Simulation(params).unit("converter").bridge)
    assert high.R_on == high.R_gated == pytest.approx(0.0)
    assert low.R_on == low.R_gated == pytest.approx(0.0)


def test_equal_explicit_resistances_reuse_the_shared_bridge_equation():
    params = _custom_case().replace(**{
        "units.converter.bridge.lower_controlled": True,
        "units.converter.bridge.switch_resistance": 0.04,
        "units.converter.bridge.diode_resistance": 0.04,
    })
    bridge = peslite.Simulation(params).unit("converter").bridge
    assert bridge.resistance == pytest.approx(0.04)
    assert not _devices(bridge)


@pytest.mark.parametrize("form", ["device", "statewise", "nodal"])
def test_gate_resistance_selects_network_mode(form):
    state = {"gate": False}
    net = nl.Netlist(form)
    source, output = nl.Node(), nl.Node()
    device = nl.Diode(
        source, output, R_on=0.045,
        gate=lambda: state["gate"], gated_R_on=0.035,
    )
    net.add(nl.VoltageSource(source, None, lambda _t: 12.0), device,
            nl.Resistor(output, None, 1.0))
    net.set_conducting(device, True)

    diode_mode = net.mode()
    net.evaluate(0.0, np.zeros(0))
    diode_current = float(net.branch_current(device)[0])
    assert diode_current == pytest.approx(12.0 / 1.045, rel=1e-9)

    state["gate"] = True
    net.release_topology()
    switch_mode = net.mode()
    net.evaluate(0.0, np.zeros(0))
    switch_current = float(net.branch_current(device)[0])
    assert diode_mode.conducting == switch_mode.conducting == (True,)
    assert diode_mode.structure_key != switch_mode.structure_key
    assert switch_current == pytest.approx(12.0 / 1.035, rel=1e-9)


def test_equal_resistance_retains_existing_mode_identity():
    state = {"gate": False}
    net = nl.Netlist("device")
    source, output = nl.Node(), nl.Node()
    device = nl.Diode(source, output, R_on=0.05, gate=lambda: state["gate"])
    net.add(nl.VoltageSource(source, None, lambda _t: 12.0), device,
            nl.Resistor(output, None, 1.0))
    net.set_conducting(device, True)
    first = net.mode()
    state["gate"] = True
    net.release_topology()
    second = net.mode()
    assert first is second
    assert not first.gated


@pytest.mark.parametrize("model", ["pwm_averaging", "averaging"])
def test_independent_resistance_requires_switching_model(model):
    with pytest.raises(ConfigError, match="requires bridge.model: switching"):
        _custom_case().replace(**{"units.converter.bridge.model": model})


@pytest.mark.long
def test_independent_resistance_python_cpp_parity(tmp_path):
    from test_cpp_parameters import _compare_run, _compile

    params = _custom_case()
    binary = _compile(params, tmp_path / "model", variables=("all",))
    changes = {
        "units.converter.bridge.switch_resistance": 0.04,
        "units.converter.bridge.diode_resistance": 0.055,
    }
    # Runtime symbolic coefficients and a rebuilt numerical netlist may locate a
    # natural-commutation zero at slightly different floating-point instants.
    _compare_run(
        params, binary, changes, tmp_path / "comparison", rtol=5e-4, atol=1e-5
    )
