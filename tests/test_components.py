"""The refactored power-circuit and converter-hardware components."""

import ast
from pathlib import Path

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.components import (
    ADC,
    AveragingBridge,
    MeasurementPorts,
    PWM,
    RCNode,
    RLBranch,
    Terminal,
    ZOH,
)
from peslite.control.blocks import abc2complex

COMPONENTS = Path(peslite.components.__file__).parent
PACKAGE = COMPONENTS.parent


def test_electrical_terminals_use_the_energy_direction_and_numbered_voltage_ports():
    branch = RLBranch(0.1, 0.01)

    assert (branch.terminal1.subsystem, branch.terminal1.voltage,
            branch.terminal1.current, branch.terminal1.direction) == (
                branch, "u1", "i", 1,
            )
    assert (branch.terminal2.subsystem, branch.terminal2.voltage,
            branch.terminal2.current, branch.terminal2.direction) == (
                branch, "u2", "i", -1,
            )
    assert [port.sign for port in branch.ports] == [
        branch.terminal1.direction, branch.terminal2.direction,
    ]

    node = RCNode(0.1, 0.01)
    assert node.inp.__slots__ == ("i",)
    with pytest.raises(ValueError, match="direction must be"):
        Terminal(branch, "u1", "i", 0)
    with pytest.raises(ValueError, match="does not match the energy-port direction"):
        Terminal(branch, "u1", "i", -1)


def test_unit_delegates_sampling_to_adc_and_actuation_to_bridge():
    p = peslite.load(
        EXAMPLES / "gfl-example.pes",
        **{"units.vsc.bridge.model": "pwm_averaging"},
    )
    unit = peslite.Simulation(p).unit()

    assert type(p.unit()) is peslite.assembly.UnitParams
    assert p.unit().type == "vsc"
    assert isinstance(unit, peslite.Unit)
    assert isinstance(unit, peslite.VSC)
    assert isinstance(unit.adc, ADC)
    assert isinstance(unit.bridge, PWM)
    assert not hasattr(unit, "pwm")
    assert isinstance(unit.protection, peslite.Protection)
    assert peslite.ADC is ADC and peslite.PWM is PWM
    assert not hasattr(unit, "pwm")
    assert not hasattr(unit, "sampler")
    assert not hasattr(unit, "window")
    assert not hasattr(unit, "modulator")
    assert not hasattr(unit, "computation_delay")


def test_adc_owns_oversampling_and_averaging_window_state():
    values = {"v": 1 + 0j, "i": 2 + 0j, "dc": 10.0}
    ports = MeasurementPorts(
        u_g=lambda: values["v"],
        i_c=lambda: values["i"],
        i_c_state=lambda: values["i"],
        u_dc=lambda: values["dc"],
    )
    adc = ADC(ports, period=1.0, sample_period=0.5, length=1.0,
              channels={"u_g": 0j, "i_c": 0j})
    adc.seed()
    values.update(v=3 + 0j, i=4 + 0j)
    adc.accumulate(0.5)
    adc.peek(0.5)
    values.update(v=5 + 0j, i=6 + 0j)
    adc.accumulate(0.5)
    sample = adc.sample(1.0)

    assert sample.u_g == 3 + 0j
    assert sample.i_c == 4 + 0j
    assert sample.u_dc == 10.0
    assert [s.t for s in sample.samples] == [0.5, 1.0]
    assert set(adc.get_state()) == {"x_u_g", "x_i_c"}


def test_adc_reconstructs_its_sampling_position_from_the_start_time():
    ports = MeasurementPorts(
        u_g=lambda: 1 + 0j,
        i_c=lambda: 2 + 0j,
        i_c_state=lambda: 2 + 0j,
        u_dc=lambda: 3.0,
    )
    adc = ADC(ports, period=1.0, sample_period=0.25)
    adc.start(0.6, 0.0, 1.0, held=True)
    assert adc.n_samp == 3
    assert [sample.t for sample in adc.peeks] == [0.25, 0.5]
    assert adc.t_sample(0.0) == 0.75


def test_pwm_owns_timer_registers_and_switching_schedule():
    initial = np.array([0.2, 0.4, 0.6])
    pwm = PWM(period=1e-3, load_period=1e-3, offset=0.0, computation=2e-4,
              carrier_period=1e-3, modulator=ZOH())
    pwm.reset(initial)
    assert pwm.get_state() == {
        "d_a": 0.2, "d_b": 0.4, "d_c": 0.6, "on": False,
        "shadow.d_a": 0.2, "shadow.d_b": 0.4, "shadow.d_c": 0.6,
        "shadow.on": False,
    }

    pwm.write(0.0, np.array([0.8, 0.7, 0.6]))
    q = pwm.load(0.0)
    assert q == abc2complex(initial)
    assert pwm.k == 1
    assert pwm.t_next == 1e-3
    assert pwm.next_switch == float("inf")
    pwm.load(1e-3)
    assert pwm.active[0] == 0.8
    pwm.set_state({"shadow.d_a": 0.1})
    assert pwm.get_state()["d_a"] == 0.8
    assert pwm.get_state()["shadow.d_a"] == 0.1


def test_ideal_averaging_bridge_has_no_timer_register_or_delay_state():
    bridge = AveragingBridge()
    bridge.reset(np.array([0.2, 0.4, 0.6]), on=True)

    assert bridge.on
    assert bridge.event_periods == []
    assert bridge.next_time() == float("inf")
    assert bridge.get_state() == {}
    assert bridge.zoh_connections("vsc.q") == {}
    assert not hasattr(bridge, "active")
    assert not hasattr(bridge, "history")
    assert not hasattr(bridge, "next_switch")


def test_duty_register_states_are_owned_by_pwm():
    params = peslite.load(
        EXAMPLES / "gfl-example.pes",
        **{"units.vsc.bridge.model": "switching"},
    )
    names = set(peslite.Simulation(params).state_names())
    for part in ("", "shadow."):
        assert {f"vsc.pwm.{part}d_{phase}" for phase in "abc"} <= names
    assert not any(".clock." in name or "pending." in name or "computation_delay" in name
                   for name in names)


def test_components_import_only_control_solver_and_themselves():
    allowed = {"components", "control", "solver"}
    for path in COMPONENTS.glob("*.py"):
        imported = set()
        rel = path.relative_to(COMPONENTS.parent).with_suffix("")
        here = list(rel.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = here[:len(here) - (node.level - 1)]
                target = base + (node.module.split(".") if node.module else [])
                if target:
                    imported.add(target[0])
        assert imported <= allowed, f"{path.name} imports {sorted(imported - allowed)}"


def test_absorbed_hardware_packages_are_removed():
    for name in ("power", "sensing", "firmware", "modulation", "protection"):
        assert not (PACKAGE / name).exists()
