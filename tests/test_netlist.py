"""Electrical netlist primitives and assembled circuit-model parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

import peslite
from conftest import EXAMPLES, QUIET
from peslite.assembly.circuit import Circuit, NetworkView, bind_device, declare_netlist
from peslite.components.network import ACVoltageSource
from peslite.solver import FixedStepSolver
from peslite.solver.netlist import (Capacitor, CurrentSource, Diode, Inductor, Netlist, Node,
                                    Resistor, StateSpace, Switch, Transformer, VoltageSource)


FORMS = ("statewise", "mna")
CIRCUIT_MODELS = ("statewise", "mna")


def _affine(A, b, initial, t):
    size = len(initial)
    augmented = np.zeros((size + 1, size + 1))
    augmented[:size, :size] = A
    augmented[:size, size] = b
    return (expm(augmented * t) @ np.append(initial, 1.0))[:size]


@pytest.mark.parametrize("form", FORMS)
def test_netlist_rc_rl_and_rlc_match_closed_state_equations(form):
    voltage, resistance, inductance, capacitance = 10.0, 2.0, 1e-3, 1e-4
    cases = []

    net = Netlist(form)
    source, output = Node(), Node()
    net.add(VoltageSource(source, None, lambda _t: voltage),
            Resistor(source, output, resistance),
            Capacitor(output, None, capacitance, [0]))
    cases.append((net, np.array([[-1 / (resistance * capacitance)]]),
                  np.array([voltage / (resistance * capacitance)]), np.array([1.0])))

    net = Netlist(form)
    source, output = Node(), Node()
    net.add(VoltageSource(source, None, lambda _t: voltage),
            Resistor(source, output, resistance),
            Inductor(output, None, inductance, [0]))
    cases.append((net, np.array([[-resistance / inductance]]),
                  np.array([voltage / inductance]), np.array([0.5])))

    net = Netlist(form)
    source, middle, output = Node(), Node(), Node()
    net.add(VoltageSource(source, None, lambda _t: voltage),
            Resistor(source, middle, resistance),
            Inductor(middle, output, inductance, [0]),
            Capacitor(output, None, capacitance, [1]))
    cases.append((
        net,
        np.array([[-resistance / inductance, -1 / inductance], [1 / capacitance, 0.0]]),
        np.array([voltage / inductance, 0.0]),
        np.array([1.0, -2.0]),
    ))

    for net, A, b, initial in cases:
        np.testing.assert_allclose(net.state_space().A, A, rtol=1e-13, atol=1e-13)
        solver = FixedStepSolver(1e-6, "RK4")
        t_end = 5e-4
        actual = solver(net.evaluate, 0.0, t_end, initial).y
        np.testing.assert_allclose(actual, _affine(A, b, initial, t_end), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("form", FORMS)
def test_netlist_node_without_capacitance_is_algebraic(form):
    net = Netlist(form)
    source, middle, output = Node(), Node(), Node()
    net.add(VoltageSource(source, None, lambda _t: 12.0),
            Resistor(source, middle, 3.0), Resistor(middle, output, 5.0),
            Capacitor(output, None, 1e-3, [0]))
    derivative = net.evaluate(0.0, np.array([4.0]))
    assert derivative[0] == pytest.approx((12.0 - 4.0) / (8.0e-3), rel=1e-14)
    assert net.node_voltage(middle)[0] == pytest.approx(9.0, rel=1e-14)


@pytest.mark.parametrize("form", FORMS)
def test_selected_netlist_outputs_match_the_full_equations(form):
    net = Netlist(form)
    source, output = Node(), Node()
    resistor = Resistor(source, output, 2.0)
    net.add(VoltageSource(source, None, lambda _t: 10.0), resistor,
            Capacitor(output, None, 1e-3, [0]))
    selected = [net.output_slice("node_voltage", output).start,
                net.output_slice("branch_current", resistor).start]
    derivative, values = net.evaluate_with_outputs(0.0, np.array([4.0]), selected)
    np.testing.assert_allclose(derivative, [3_000.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(values, net.outputs()[selected], rtol=0.0, atol=1e-12)


def test_selected_scalar_outputs_follow_a_dynamic_transformer_mode():
    ratio = [0.5]
    net = Netlist("device")
    primary, secondary, common = Node(), Node(), Node()
    transformer = Transformer(primary, common, secondary, common, lambda: ratio[0])
    voltage = VoltageSource(secondary, None, lambda _t: 12.0)
    net.add(transformer, voltage, CurrentSource(primary, None, lambda _t: 5.0),
            VoltageSource(common, None, lambda _t: 0.0), Resistor(primary, None, 4.0))
    selected = [net.output_slice("node_voltage", primary).start,
                net.output_slice("branch_current", voltage).start]
    evaluate = net.scalar_evaluators()[2](selected)
    np.testing.assert_allclose(evaluate(0.0, ()),
                               np.asarray(net.evaluate_scalar_outputs(0.0, ()))[selected],
                               rtol=0.0, atol=1e-12)
    ratio[0] = 0.25
    np.testing.assert_allclose(evaluate(0.0, ()),
                               np.asarray(net.evaluate_scalar_outputs(0.0, ()))[selected],
                               rtol=0.0, atol=1e-12)


def test_causal_transformer_reduction_supports_a_shared_voltage_reference():
    ratio = [0.5]
    net = Netlist("device")
    primary, secondary, common = Node(), Node(), Node()
    transformer = Transformer(primary, common, secondary, common, lambda: ratio[0])
    voltage = VoltageSource(secondary, None, lambda _t: 12.0)
    reference = VoltageSource(common, None, lambda _t: 2.0)
    net.add(transformer, voltage, CurrentSource(primary, None, lambda _t: 5.0), reference)
    selected = [net.output_slice("node_voltage", primary).start,
                net.output_slice("branch_current", voltage).start,
                net.output_slice("branch_current", reference).start]
    evaluate = net.scalar_evaluators()[2](selected)
    np.testing.assert_allclose(evaluate(0.0, ()), [7.0, -2.5, -2.5], rtol=0.0, atol=1e-12)
    ratio[0] = 0.25
    np.testing.assert_allclose(evaluate(0.0, ()), [4.5, -1.25, -3.75], rtol=0.0, atol=1e-12)


def test_custom_ac_waveform_remains_the_source_of_its_declared_netlist():
    class CustomSource(ACVoltageSource):
        def set_outputs(self, t):
            self.output.e_g = 40.0 + t

    source = CustomSource(2.0 * np.pi * 50.0, 325.0, phases=1)
    declaration = declare_netlist(source)
    assert declaration is not None
    bind_device("source", source, declaration)
    source.set_outputs(2.0)
    assert source.output.e_g == 42.0


@pytest.mark.parametrize("form", FORMS)
def test_dynamic_transformer_update_matches_a_directly_factorized_ratio(form):
    def circuit(ratio):
        net = Netlist(form)
        primary, middle, secondary = Node(), Node(), Node()
        net.add(Transformer(primary, None, secondary, None, ratio),
                VoltageSource(secondary, None, lambda _t: 12.0),
                Resistor(primary, middle, 2.0), Inductor(middle, None, 1e-3, [0]))
        return net

    ratio = [0.2]
    dynamic = circuit(lambda: ratio[0])
    dynamic.state_space()
    ratio[0] = 0.7
    updated = dynamic.state_space()
    direct = circuit(((0.7,),)).state_space()
    for field in StateSpace.__dataclass_fields__:
        np.testing.assert_allclose(getattr(updated, field), getattr(direct, field),
                                   rtol=1e-13, atol=1e-13)


def test_scalar_reduction_does_not_treat_parallel_branches_as_a_series_path():
    net = Netlist("device")
    source, middle, return_node = Node(), Node(), Node()
    net.add(
        VoltageSource(source, None, lambda _t: 10.0),
        VoltageSource(return_node, None, lambda _t: 0.0),
        Resistor(source, middle, 2.0),
        Resistor(source, middle, 2.0),
        Inductor(middle, return_node, 1e-3, [0]),
    )
    derivatives, _outputs, _selected_outputs = net.scalar_evaluators()
    assert derivatives(0.0, (1.0,))[0] == pytest.approx(9_000.0)


def test_diode_state_relation_is_proved_from_compiled_network_equations():
    net = Netlist("device")
    source, middle, winding, switched, load = Node(), Node(), Node(), Node(), Node()
    diode = Diode(source, middle)
    closed = False
    net.add(
        VoltageSource(source, None, lambda _t: 10.0),
        diode,
        Resistor(middle, winding, 2.0),
        Inductor(winding, None, 1e-3, [0]),
        VoltageSource(switched, None, lambda _t: 1.0),
        Switch(switched, load, lambda: closed),
        Resistor(load, None, 1.0),
    )

    net.set_conducting(diode, True)
    relations = net.state_diode_relations()
    assert relations == ((0, ((0, 1.0),)),)
    assert net.state_diode_relations() is relations

    closed = True
    assert net.state_diode_relations() is relations
    assert len(net._state_diode_structures[0]) == 2

    parallel = Netlist("device")
    source, middle, winding = Node(), Node(), Node()
    parallel.add(
        VoltageSource(source, None, lambda _t: 10.0),
        Diode(source, middle),
        Resistor(middle, winding, 2.0),
        Inductor(winding, None, 1e-3, [0]),
    )
    parallel.add(CurrentSource(middle, None, lambda _t: 1.0))
    parallel.set_conducting(parallel._diodes[0], True)
    assert parallel.state_diode_relations() == ()


@pytest.mark.parametrize("circuit_model, frozen", [("device", False), ("statewise", True)])
def test_circuit_freezes_only_when_its_runtime_has_not_bound_the_topology(
        circuit_model, frozen):
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / "buck-example.pes",
        **QUIET,
        **{
            "simulation.solver.circuit_model": circuit_model,
            "units.converter.bridge.model": "switching",
        },
    ))
    circuit, = simulation.system.circuits

    circuit.freeze_topology()
    assert (circuit.netlist._frozen_topology is not None) is frozen
    circuit.release_topology()


def _final(example: str, circuit_model: str, out_dir: Path) -> tuple[peslite.Simulation, dict]:
    changes = ({"events.connect_vsc.t": 0.0, "events.connect_vsc.ramp": 5e-3}
               if example == "gfl-example" else {})
    params = peslite.load(
        EXAMPLES / f"{example}.pes",
        **QUIET,
        **{"simulation.t_end": 2e-4,
           "simulation.solver.circuit_model": circuit_model,
           **changes},
    )
    simulation = peslite.Simulation(params)
    return simulation, simulation.run(out_dir=out_dir).final_states()


@pytest.mark.parametrize("example", ["gfl-example", "buck-open-loop-example", "wpt-example"])
def test_rk4_netlist_circuit_models_match_device_equations(example, tmp_path):
    reference_sim, reference = _final(example, "device", tmp_path / example / "device")
    expected = np.asarray(list(reference.values()), dtype=float)
    scale = np.maximum(np.abs(expected), max(np.max(np.abs(expected)), 1.0) * 1e-12)
    for circuit_model in CIRCUIT_MODELS:
        simulation, result = _final(example, circuit_model, tmp_path / example / circuit_model)
        np.testing.assert_array_equal(simulation.system.model.state_labels(),
                                      reference_sim.system.model.state_labels())
        actual = np.asarray(list(result.values()), dtype=float)
        assert np.max(np.abs(actual - expected) / scale) < 1e-9


def test_device_netlist_equations_are_inlined_into_the_model_program():
    params = peslite.load(
        EXAMPLES / "gfl-example.pes", **QUIET,
        **{"events.connect_vsc.t": 0.0, "events.connect_vsc.ramp": 5e-3},
    )
    source = peslite.Simulation(params).system.model.compiled_source

    assert ".rhs(t)  # pcc" not in source
    assert ".rhs(t)  # grid.branch" not in source
    assert "# pcc" in source and "# grid.branch" in source


@pytest.mark.full
@pytest.mark.parametrize("example", ["wpt-phase-shift-example", "wpt-scc-example"])
def test_switching_wpt_netlists_match_device_equations(example, tmp_path):
    def run(circuit_model):
        params = peslite.load(
            EXAMPLES / f"{example}.pes",
            **QUIET,
            **{"simulation.t_end": 1e-4,
               "simulation.solver.circuit_model": circuit_model},
        )
        result = peslite.Simulation(params).run(out_dir=tmp_path / circuit_model)
        return np.asarray(list(result.final_states().values()), dtype=float)

    reference = run("device")
    for circuit_model in CIRCUIT_MODELS:
        np.testing.assert_allclose(run(circuit_model), reference, rtol=1e-9, atol=2e-6)


@pytest.mark.parametrize("circuit_model", CIRCUIT_MODELS)
def test_switching_bridge_with_device_capacitance_is_native(circuit_model, tmp_path):
    changes = {
        "simulation.t_end": 1e-8,
        "simulation.solver.dt": 1e-10,
        "simulation.progress.enable": 0,
        "simulation.solver.linearisations": 0,
        "units.vsc.bridge.model": "switching",
        "units.vsc.bridge.parallel_capacitance": 1e-7,
        "units.vsc.bridge.on_resistance": 0.01,
    }
    reference = peslite.Simulation(peslite.load(
        EXAMPLES / "gfl-example.pes", **changes
    )).run(out_dir=tmp_path / "device").final_states()
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / "gfl-example.pes",
        **changes,
        **{"simulation.solver.circuit_model": circuit_model},
    ))
    bridge = simulation.system.model.subsystems[
        simulation.system.model.names.index("vsc.bridge")
    ]
    assert isinstance(bridge, NetworkView)
    actual = simulation.run(out_dir=tmp_path / circuit_model).final_states()
    np.testing.assert_allclose(list(actual.values()), list(reference.values()), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("circuit_model", CIRCUIT_MODELS)
def test_assembled_netlist_preserves_component_views(circuit_model):
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / "two-converters-example.pes",
        **QUIET,
        **{"simulation.solver.circuit_model": circuit_model},
    ))
    circuit, = simulation.system.circuits
    assert isinstance(circuit, Circuit) and circuit.netlist.form == circuit_model
    views = [subsystem for subsystem in simulation.system.model.subsystems
             if isinstance(subsystem, NetworkView)]
    assert views and all(view.state is view.component.state for view in views)
    assert simulation.ph_report.verdict == "port-hamiltonian"


@pytest.mark.parametrize("circuit_model", CIRCUIT_MODELS)
@pytest.mark.parametrize("example", [path.stem for path in sorted(EXAMPLES.glob("*.pes"))])
def test_every_example_builds_with_an_electrical_netlist(example, circuit_model):
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / f"{example}.pes",
        **QUIET,
        **{"simulation.solver.circuit_model": circuit_model},
    ))
    assert simulation.system.circuits
