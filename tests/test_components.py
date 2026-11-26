"""The refactored power-circuit and converter-hardware components."""

import ast
from pathlib import Path

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.components import (
    ADC,
    AveragingModel,
    Capacitor,
    MeasurementPorts,
    PWM,
    PWMAveragingModel,
    Bus,
    Element,
    ParallelRLC,
    Passive,
    Resistor,
    RLBranch,
    SeriesRLC,
    Port,
    ThreePhaseBridge,
    ZOH,
)
from peslite.components.registry import ELEMENT_TYPES

COMPONENTS = Path(peslite.components.__file__).parent


def test_builtin_element_configuration_and_assembly_are_separate_from_physics():
    physical_types = {
        "series_rlc": SeriesRLC,
        "parallel_rlc": ParallelRLC,
        "capacitor": Capacitor,
        "resistor": Resistor,
    }

    for name, physical_type in physical_types.items():
        definition = ELEMENT_TYPES[name]
        assert definition.params.__module__ == "peslite.assembly.elements"
        assert definition.constructor.__module__ == "peslite.assembly.elements"
        assert "Params" not in physical_type.__dict__


def test_passive_elements_share_the_general_element_boundary():
    assert issubclass(Passive, Element)
    assert SeriesRLC.__bases__ == (Passive,)
    assert ParallelRLC.__bases__ == (Passive,)


def test_passive_initialization_keeps_series_and_parallel_current_meanings():
    series = SeriesRLC(R=1.0, L=2.0, C=3.0, i0=4.0, u0=5.0, phases=1)
    parallel = ParallelRLC(R=1.0, L=2.0, C=3.0, i0=4.0, u0=5.0, phases=1)

    assert (series.state.i, series.state.u_C, series.output.i) == (4.0, 5.0, 4.0)
    assert (parallel.state.i_L, parallel.state.u_C, parallel.output.i) == (4.0, 5.0, 0.0)


def test_zero_contribution_passive_values_use_the_absent_representation():
    for zero, absent in (
        (SeriesRLC(R=0.0, L=0.0, C=0.0, phases=1), SeriesRLC(phases=1)),
        (ParallelRLC(R=0.0, L=0.0, C=0.0, phases=1), ParallelRLC(phases=1)),
    ):
        assert (zero.R, zero.L, zero.C) == (absent.R, absent.L, absent.C)
        assert zero.state_names == absent.state_names
        zero_model = peslite.Model({"component": zero}, {})
        absent_model = peslite.Model({"component": absent}, {})
        assert zero_model.state_labels() == absent_model.state_labels()
        assert tuple(type(item).__name__ for item in zero._device_circuit.netlist.elements) == tuple(
            type(item).__name__ for item in absent._device_circuit.netlist.elements
        )


def test_zero_passive_values_stay_absent_during_updates_and_specialization():
    series = SeriesRLC(phases=1)
    parallel = ParallelRLC(phases=1)
    assert not series.set_parameters(r=0.0, l=0.0, c=0.0)
    assert not parallel.set_parameters(r=0.0, l=0.0, c=0.0)
    for component in (RLBranch(), Capacitor(), Resistor(0.0)):
        assert (component.R, component.L, component.C) == (None, None, None)


def test_electrical_ports_store_inward_current_and_numbered_names():
    branch = RLBranch(L=0.1, R=0.01)

    assert (branch.port1.component, branch.port1.name,
            branch.port1.voltage, branch.port1.current) == (
                branch, "1", 0j, 0j,
            )
    assert (branch.port2.component, branch.port2.name,
            branch.port2.voltage, branch.port2.current) == (
                branch, "2", 0j, 0j,
            )

    node = Bus(0.1, 0.01)
    assert node.port.component is node
    assert node.port.name == "power"
    with pytest.raises(ValueError, match="scale must be positive"):
        Port(branch, "1", scale=0.0)


def test_inherited_branch_behavior_uses_the_declared_netlist_equations():
    class DerivedBranch(RLBranch):
        pass

    branch = DerivedBranch(L=1.0, R=0.0, i0=1.0, phases=1)
    branch.port1.voltage, branch.port2.voltage = 0.0, 1.0
    branch.set_outputs(0.0)

    assert any(type(element).__name__ == "Inductor"
               for element in branch._device_circuit.netlist.elements)
    assert branch.output.i == 1.0
    assert branch.rhs(0.0) == (-1.0,)
    branch.open_breaker()
    assert branch.rhs(0.0) == (0.0,)
    branch.close_breaker()
    branch.state.i = 1.0
    assert branch.rhs(0.0) == (-1.0,)


def test_model_prepares_a_standalone_component_before_planning():
    branch = RLBranch(L=1.0, R=2.0, i0=3.0, phases=1)
    branch.port1.voltage, branch.port2.voltage = 5.0, 1.0

    model = peslite.Model({"branch": branch}, {})

    assert branch._device_circuit is not None
    assert model.rhs_list(0.0, [3.0]) == [-2.0]


@pytest.mark.parametrize(
    ("component", "states"),
    [
        (SeriesRLC(R=2.0, phases=1), ()),
        (SeriesRLC(C=1.0e-3, phases=1), ("u_C",)),
        (SeriesRLC(R=2.0, L=1.0e-3, C=2.0e-3, phases=1), ("i", "u_C")),
        (ParallelRLC(R=2.0, phases=1), ()),
        (ParallelRLC(L=1.0e-3, phases=1), ("i_L",)),
        (ParallelRLC(R=2.0, L=1.0e-3, C=2.0e-3, phases=1), ("i_L", "u_C")),
    ],
)
def test_series_and_parallel_rlc_share_the_netlist_model(component, states):
    model = peslite.Model({"rlc": component}, {})

    assert component.state_names == states
    assert model.state_labels() == [f"rlc.{name}" for name in states]
    assert tuple(item.state for item in component.storage) == states
    model.rhs(0.0, model.get_initial_values())


def test_rlc_constructors_accept_values_in_resistance_inductance_capacitance_order():
    series = SeriesRLC(2.0, 1.0e-3, 2.0e-3)
    parallel = ParallelRLC(3.0, 4.0e-3, 5.0e-3)
    branch = RLBranch(6.0, 7.0e-3)
    capacitor = Capacitor(8.0, 9.0e-3)
    resistor = Resistor(10.0)

    assert (series.R, series.L, series.C) == (2.0, 1.0e-3, 2.0e-3)
    assert (parallel.R, parallel.L, parallel.C) == (3.0, 4.0e-3, 5.0e-3)
    assert (branch.R, branch.L, branch.C) == (6.0, 7.0e-3, None)
    assert (capacitor.R, capacitor.L, capacitor.C) == (8.0, None, 9.0e-3)
    assert (resistor.R, resistor.L, resistor.C) == (10.0, None, None)


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


@pytest.mark.parametrize("phases,voltage,current,expected", [
    (1, 24.0, 5.0, 120.0), (3, 3.0 + 4.0j, 2.0 - 1.0j, 3.0),
])
def test_port_power_reuses_the_declared_energy_convention(phases, voltage, current, expected):
    branch = RLBranch(L=0.1, R=0.01, phases=phases)
    branch.port1.voltage = branch.port2.voltage = voltage
    branch.port1.current = current
    branch.port2.current = -current
    assert branch.port1.power() == pytest.approx(expected)
    assert branch.port2.power() == pytest.approx(-expected)


def test_adc_owns_oversampling_and_averaging_window_state():
    values = {"v": 1 + 0j, "i": 2 + 0j, "dc": 10.0}
    ports = MeasurementPorts({
        "u_g": lambda: values["v"],
        "i_c": lambda: values["i"],
        "u_dc": lambda: values["dc"],
    })
    adc = ADC(ports, period=1.0, sample_period=0.5, length=1.0,
              channels={"u_g": 0j, "i_c": 0j})
    adc.seed()
    values.update(v=3 + 0j, i=4 + 0j)
    adc.accumulate(0.5)
    adc.peek(0.5)
    values.update(v=5 + 0j, i=6 + 0j)
    adc.accumulate(0.5)
    sample = adc.sample(1.0)

    assert sample.values["u_g"] == 3 + 0j
    assert sample.values["i_c"] == 4 + 0j
    assert sample.values["u_dc"] == 10.0
    assert [s.t for s in sample.samples] == [0.5, 1.0]
    assert sample.samples[-1].values == {
        "u_g": 5 + 0j,
        "i_c": 6 + 0j,
        "u_dc": 10.0,
    }
    assert not hasattr(sample, "raw")
    assert set(adc.get_state()) == {
        "x_u_g", "x_i_c", "integration_time", "last_u_g", "last_i_c",
    }


def test_adc_reconstructs_its_sampling_position_from_the_start_time():
    ports = MeasurementPorts({
        "u_g": lambda: 1 + 0j,
        "i_c": lambda: 2 + 0j,
        "u_dc": lambda: 3.0,
    })
    adc = ADC(ports, period=1.0, sample_period=0.25)
    adc.start(0.6, 0.0, 1.0, held=True)
    assert adc.n_samp == 3
    assert [sample.t for sample in adc.peeks] == [0.25, 0.5]
    assert adc.t_sample(0.0) == 0.75
    # An explicit offset on the control grid is aligned within the clock tolerance
    # (3e-5 % 1e-5 is 9.999999999999999e-06, not 0.0).
    assert not ADC(ports, period=4e-5, sample_period=1e-5, control_offset=3e-5).independent
    assert ADC(ports, period=4e-5, sample_period=1e-5, control_offset=3e-5,
               sample_offset=5e-6).independent


def test_pwm_owns_timer_registers_and_switching_schedule():
    initial = np.array([0.2, 0.4, 0.6])
    pwm = PWMAveragingModel(period=1e-3, load_period=1e-3, control_offset=0.0, load_offset=0.0,
                            computation=2e-4, carrier_period=1e-3, modulator=ZOH(),
                            topology=ThreePhaseBridge())
    pwm.reset(initial)
    assert pwm.get_state() == {
        "d_a": 0.2, "d_b": 0.4, "d_c": 0.6, "on": False,
        "shadow.d_a": 0.2, "shadow.d_b": 0.4, "shadow.d_c": 0.6,
        "shadow.on": False,
    }

    pwm.write(0.0, np.array([0.8, 0.7, 0.6]))
    q = pwm.load(0.0)
    assert q == ThreePhaseBridge.switching_state(initial)
    assert pwm.k == 1
    assert pwm.t_interrupt == 1e-3
    assert pwm.next_switch == float("inf")
    pwm.load(1e-3)
    assert pwm.active[0] == 0.8
    pwm.start(1e-3, (None, None), True)
    completed = pwm.completed
    pwm.set_state({"shadow.d_a": 0.1})
    assert pwm.get_state()["d_a"] == 0.8
    assert pwm.get_state()["shadow.d_a"] == 0.1
    assert pwm.completed is completed
    assert pwm.completed[0] == 0.8


def test_ideal_averaging_bridge_has_no_timer_register_or_delay_state():
    bridge = AveragingModel()
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
    allowed = {"components", "control", "defaults", "solver"}
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
