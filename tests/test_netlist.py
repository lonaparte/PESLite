"""Electrical netlist primitives and assembled circuit-model parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

import peslite
from conftest import EXAMPLES, QUIET
from peslite.assembly.circuit import (Circuit, NetworkView, _live_statements, _ModeStructure,
                                      bind_device, compile_network)
from peslite.components.network import ACVoltageSource
from peslite.solver import FixedStepSolver
from peslite.solver import netlist as nl
from peslite.solver.netlist import (Capacitor, CurrentSource, Diode, Inductor, Netlist, Node,
                                    Resistor, StateSpace, Switch, Transformer, VoltageSource)


FORMS = ("statewise", "nodal")
CIRCUIT_MODELS = ("statewise", "nodal")


def test_symbolic_solve_uses_a_free_reference_only_for_a_proven_gauge():
    matrix = [[1.0, -1.0], [2.0, -2.0]]
    right = [3.0, 6.0]
    reference = np.asarray(matrix)

    with pytest.raises(ValueError, match="singular at column 1"):
        nl._symbolic_solve(matrix, right, reference)

    solution = nl._symbolic_solve(matrix, right, reference, gauge=True)
    np.testing.assert_allclose(np.asarray(matrix) @ solution, right)
    assert solution[1] == 0.0


def test_symbolic_solve_keeps_pivots_off_declared_varying_entries():
    ratio, parameter, first, second = (
        nl._Symbolic.value(name) for name in ("ratio", "parameter", "first", "second")
    )
    solution = nl._symbolic_solve(
        [[ratio, 1.0], [parameter, 0.0]],
        [first, second],
        np.array([[10.0, 1.0], [1.0, 0.0]]),
        {(0, 0)},
    )
    _shared, expressions = nl._symbolic_source(solution)
    assert expressions[0] == "(second / parameter)"
    assert "/ ratio" not in " ".join(expressions)


def test_symbolic_solve_propagates_varying_entries_through_elimination():
    ratio, parameter, first, second, third = (
        nl._Symbolic.value(name)
        for name in ("ratio", "parameter", "first", "second", "third")
    )
    solution = nl._symbolic_solve(
        [[1.0, ratio, 0.0], [1.0, 1.0, 1.0], [0.0, parameter, 0.0]],
        [first, second, third],
        np.array([[1.0, -9.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),
        {(0, 1)},
    )
    shared, expressions = nl._symbolic_source(solution)
    source = "\n".join((*shared, *expressions))
    assert "third / parameter" in source
    assert "/ (1 + (-(ratio)))" not in source


def _affine(A, b, initial, t):
    size = len(initial)
    augmented = np.zeros((size + 1, size + 1))
    augmented[:size, :size] = A
    augmented[:size, size] = b
    return (expm(augmented * t) @ np.append(initial, 1.0))[:size]


@pytest.mark.parametrize("form", ("device", *FORMS))
def test_behavioral_current_state_is_one_physical_branch(form):
    net = Netlist(form)
    node = Node()
    state = net.add(nl.CurrentState(node, None, [0], lambda _t: 2.0))
    net.add(Resistor(node, None, 4.0), nl.Dissipation(lambda _t: 3.0))

    derivative = net.evaluate(0.0, np.array([1.5]))

    np.testing.assert_allclose(derivative, [2.0])
    np.testing.assert_allclose(net.branch_current(state), [1.5])
    np.testing.assert_allclose(net.branch_voltage(state), [-6.0])
    assert net.dissipated_power() == pytest.approx(12.0)


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


@pytest.mark.parametrize("form", ("device", *FORMS))
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
    np.testing.assert_allclose(
        net.evaluate_outputs(0.0, np.array([4.0]), selected), values,
        rtol=0.0, atol=1e-12,
    )


def test_grouped_nodal_selected_outputs_reuse_the_network_factorization():
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / "two-converters-example.pes",
        **QUIET,
        **{"simulation.solver.circuit_model": "nodal"},
    ))
    circuit, = simulation.system.circuits
    circuit.receive()
    net, states, selected = circuit.netlist, circuit._states(), circuit.output_indices
    topology = net.topology()

    assert topology._layout is not None
    expected = net.evaluate_with_outputs(0.0, states, selected)[1]
    np.testing.assert_allclose(
        net.evaluate_outputs(0.0, states, selected), expected,
        rtol=1e-12, atol=1e-12,
    )


def test_a_small_nodal_network_is_factorized_whole():
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / "wpt-ss-example.pes",
        **QUIET,
        **{"simulation.solver.circuit_model": "nodal"},
    ))
    circuit, = simulation.system.circuits
    circuit.receive()
    topology = circuit.netlist.topology()

    assert circuit.netlist._grouping_used()
    assert topology.n_z <= nl._DENSE_GROUP
    assert topology._layout is None


def _numeric_input_reach(topology):
    """Measure which source and ratio values affect derivatives and output rows."""
    net, state_space = topology._store[0], topology.state_space
    effects = []
    state_ratios = []
    for source, offset in net._sources:
        columns = slice(offset, offset + net._branch(source).dim)
        effects.append((
            np.hstack((state_space.B[:, columns], state_space.B1[:, columns])),
            np.hstack((state_space.D[:, columns], state_space.D1[:, columns])),
        ))
    rng = np.random.default_rng(7)
    for index, ratio in enumerate(topology.mode.ratios):
        ratios = list(topology.mode.ratios)
        ratios[index] = np.asarray(ratio) + 0.1 + 0.05 * rng.standard_normal(np.shape(ratio))
        other = topology.with_ratios(nl.Mode(
            topology.mode.closed, topology.mode.conducting, ratios,
        ))
        matrix = other.M.toarray() if hasattr(other.M, "toarray") else np.asarray(other.M)
        other.lu = nl._DenseSolve(matrix, "network")
        changed = other.state_space
        effects.append((
            np.hstack((changed.A - state_space.A, changed.B - state_space.B,
                       changed.B1 - state_space.B1)),
            np.hstack((changed.C - state_space.C, changed.D - state_space.D,
                       changed.D1 - state_space.D1)),
        ))
        state_ratios.append(np.abs(changed.A - state_space.A).max(initial=0.0))
    scale = max(
        np.abs(matrix).max(initial=0.0)
        for matrix in (state_space.A, state_space.B, state_space.C, state_space.D)
    )
    threshold = 1e-9 * max(scale, 1.0)
    return (
        np.asarray([np.abs(rows).max(initial=0.0) > threshold
                    for rows, _outputs in effects]),
        np.asarray([np.abs(rows).max(axis=1, initial=0.0) > threshold
                    for _derivatives, rows in effects]),
        np.asarray(state_ratios) > threshold,
    )


@pytest.mark.full
@pytest.mark.parametrize("form", FORMS)
def test_large_numeric_stage_dependencies_follow_the_network_structure(
        form, monkeypatch):
    symbolic_calls = 0
    original = nl._symbolic_solve

    def counted(*args, **kwargs):
        nonlocal symbolic_calls
        symbolic_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(nl, "_symbolic_solve", counted)
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / "cable-gfl-example.pes",
        **QUIET,
        **{
            "elements.cable1.sections": 24,
            "simulation.solver.circuit_model": form,
        },
    ))
    assert symbolic_calls == 0
    for circuit in simulation.system.circuits:
        topology = circuit.netlist.topology()
        assert topology.n_z > nl._DENSE_GROUP
        structural = topology.input_reach()
        assert structural is not None
        derivatives, outputs, state_ratios = structural
        numeric_derivatives, numeric_outputs, numeric_state_ratios = _numeric_input_reach(topology)
        np.testing.assert_array_equal(derivatives, numeric_derivatives)
        assert np.all(outputs | ~numeric_outputs)
        assert np.all(state_ratios | ~numeric_state_ratios)
    assert not simulation.system.model.algebraic_loops


def test_generated_equations_keep_only_statements_used_by_the_requested_values():
    statements = ["    a = f(t)", "    b = g(t)", "    c = 2.0 * a", "    d = b + c"]
    assert _live_statements(statements, ["c + 1.0"]) == [
        "    a = f(t)", "    c = 2.0 * a",
    ]
    assert _live_statements(statements, ["d"]) == statements
    assert _live_statements(["    x, y = pair(t)"], ["1.0"]) == [
        "    x, y = pair(t)",
    ]


def test_selected_scalar_outputs_follow_a_dynamic_transformer_mode():
    ratio = [np.float64(0.5)]
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
    ratio[0] = np.float64(0.25)
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
    declaration = compile_network(source)
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
    base = dynamic.topology()
    selected = base.selected_outputs((0,))
    ratio[0] = 0.7
    updated = dynamic.state_space()
    topology = dynamic.topology()
    assert isinstance(topology.lu, nl._RatioUpdate)
    assert topology.selected_outputs((0,)) is selected
    right = np.arange(1.0, topology.n_z + 1.0)
    np.testing.assert_allclose(
        topology.M @ topology.lu.solve(right), right, rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        topology.M.T @ topology.lu.solve_transpose(right), right,
        rtol=1e-12, atol=1e-12,
    )
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


def _final(example: str, changes: dict, circuit_model: str,
           out_dir: Path) -> tuple[peslite.Simulation, dict]:
    params = peslite.load(
        EXAMPLES / f"{example}.pes",
        **QUIET,
        **{"simulation.t_end": 2e-4, "simulation.solver.circuit_model": circuit_model},
    )
    # Every converter connects at once, so the short run exercises its circuit.
    params = params.replace(**{
        f"events.{name}.{key}": value for name, event in params.events.items()
        if event.type == "connect" for key, value in (("t", 0.0), ("ramp", 5e-3))
    }, **changes)
    simulation = peslite.Simulation(params)
    return simulation, simulation.run(out_dir=out_dir).final_states()


_SWITCHING = {"simulation.t_end": 1e-3, "simulation.solver.subsystems": {}}


@pytest.mark.parametrize("example, changes", [
    ("gfl-example", {}), ("buck-open-loop-example", {}), ("boost-example", {}),
    ("buck-boost-example", {}), ("wpt-example", {}), ("wpt-scc-example", {}),
    ("cable-gfl-example", {}), ("two-converters-example", {}), ("gfm-vsg-example", {}),
    ("boost-example", {**_SWITCHING, "units.converter.bridge.model": "switching"}),
    ("buck-example", {**_SWITCHING, "simulation.t_end": 2e-3, "elements.load.r": 200.0,
                      "units.converter.bridge.model": "switching"}),
    ("gfl-example", {**_SWITCHING, "units.vsc.bridge.model": "switching"}),
    ("two-converters-example", {**_SWITCHING, "simulation.t_end": 5e-4,
                                "units.vsc_1.bridge.model": "switching",
                                "units.vsc_2.bridge.model": "switching"}),
], ids=lambda value: value if isinstance(value, str) else
       "-".join(sorted({str(v) for k, v in value.items() if k.endswith("bridge.model")})) or "default")
def test_rk4_netlist_circuit_models_match_device_equations(example, changes, tmp_path):
    reference_sim, reference = _final(example, changes, "device", tmp_path / "device")
    expected = np.asarray(list(reference.values()), dtype=float)
    scale = np.maximum(np.abs(expected), max(np.max(np.abs(expected)), 1.0) * 1e-12)
    for circuit_model in CIRCUIT_MODELS:
        simulation, result = _final(example, changes, circuit_model, tmp_path / circuit_model)
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


@pytest.mark.parametrize("circuit_model", CIRCUIT_MODELS)
def test_numeric_circuit_stages_keep_the_declared_equation_dependencies(circuit_model):
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / "buck-example.pes",
        **QUIET,
        **{"simulation.solver.circuit_model": circuit_model},
    ))
    circuit, = simulation.system.circuits

    assert all(stage.method != "solve" for stage in circuit.output_stages)
    assert not simulation.system.model.algebraic_loops
    assert "_circuit_numeric_output" not in simulation.system.model.compiled_source


@pytest.mark.parametrize("example", [
    "buck-example", "boost-example", "buck-boost-example",
])
def test_continuously_varying_device_ratios_do_not_create_algebraic_loops(example):
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / f"{example}.pes", **QUIET,
    ))

    assert not simulation.system.model.algebraic_loops


def _derivative_row(net, element):
    return net._branches[net._ports[id(element)][0]].x


def _owned_devices(structure, owners, row):
    return sorted(variable for cluster in owners[row] for variable in structure.clusters[cluster])


def test_binary_mode_clusters_include_switches_and_diodes_from_connections():
    net = Netlist("statewise")
    bus, left, right, source = Node(), Node(), Node(), Node()
    capacitor = Capacitor(bus, None, 1e-4, [0])
    inductors = [
        Inductor(source, left, 1e-3, [1]),
        Inductor(source, right, 2e-3, [2]),
    ]
    net.add(
        VoltageSource(source, None, lambda _t: 48.0), capacitor, *inductors,
        Switch(left, bus, lambda: True), Diode(right, bus), Resistor(bus, None, 4.0),
    )
    structure = _ModeStructure(
        net, (("closed", 0), ("conducting", 0)), nl.Mode((True,), (True,), ()),
    )
    owners = structure.equation_owners(net)

    assert structure.clusters == [[0], [1]]
    assert _owned_devices(structure, owners, _derivative_row(net, capacitor)) == [0, 1]
    assert _owned_devices(structure, owners, _derivative_row(net, inductors[0])) == [0]
    assert _owned_devices(structure, owners, _derivative_row(net, inductors[1])) == [1]


def test_short_device_loop_owns_every_changed_equation_row():
    net = Netlist("statewise")
    bus, middle, source = Node(), Node(), Node()
    net.add(
        VoltageSource(source, None, lambda _t: 48.0),
        Capacitor(bus, None, 1e-4, [0]),
        Inductor(source, middle, 1e-3, [1]),
        Diode(middle, bus),
        Diode(None, middle),
        Resistor(bus, None, 4.0),
    )
    structure = _ModeStructure(
        net, (("conducting", 0), ("conducting", 1)),
        nl.Mode((), (False, False), ()),
    )
    owners = structure.equation_owners(net)

    def rows(conducting):
        space = net._compiled(nl.Mode((), conducting, ())).state_space
        return np.vstack((
            np.hstack((space.A, space.B, space.B1)),
            np.hstack((space.C, space.D, space.D1)),
        ))

    changed = np.flatnonzero(np.any(
        rows((True, True)) != rows((False, False)), axis=1,
    ))
    assert structure.clusters == [[0, 1]]
    assert all(owners[row] == [0] for row in changed)


def test_singular_diode_recovery_preserves_the_gated_device():
    net = Netlist("device")
    source, middle = Node(), Node()
    net.add(
        VoltageSource(source, None, lambda _t: 1.0),
        Diode(source, middle),
        Diode(middle, None),
    )
    mode = nl.Mode((), (True, True), ())

    assert net.resolved_topology(mode, (True, False)).mode.conducting == (True, False)
    assert net.resolved_topology(mode, (False, True)).mode.conducting == (False, True)


def test_export_planning_covers_default_device_circuits_and_diode_modes():
    direct = peslite.Simulation(peslite.load(
        EXAMPLES / "gfl-example.pes", **QUIET,
        **{"units.vsc.bridge.model": "pwm_averaging"},
    ))
    direct.system.plan_export()
    circuit, = direct.system.circuits
    decompositions = tuple(value for value in circuit._mode_decompositions.values() if value)
    assert decompositions
    assert decompositions[-1].variables == (("closed", 0), ("closed", 1))

    diode = peslite.Simulation(peslite.load(
        EXAMPLES / "buck-example.pes", **QUIET,
        **{"units.converter.bridge.model": "switching"},
    ))
    diode.system.plan_export()
    circuit, = diode.system.circuits
    variables, = (key[0] for key in circuit._mode_decompositions)
    assert set(variables) == {
        ("closed", 0), ("conducting", 0), ("conducting", 1),
    }
    solve = next(stage for stage in circuit.output_stages if stage.rhs)
    assert set(solve.inputs) == set(circuit.inputs)

    numeric = peslite.Simulation(peslite.load(
        EXAMPLES / "buck-example.pes", **QUIET,
        **{
            "units.converter.bridge.model": "switching",
            "simulation.solver.circuit_model": "statewise",
        },
    ))
    numeric.system.plan_export()
    circuit, = numeric.system.circuits
    solve = next(stage for stage in circuit.output_stages if stage.rhs)
    assert set(solve.inputs) == set(circuit.inputs)


def test_device_ratio_stability_certificate_avoids_exact_mode_spectra(
        monkeypatch, tmp_path):
    count = 0
    original = nl._DenseSolve.__init__

    def counting(self, *args, **kwargs):
        nonlocal count
        count += 1
        original(self, *args, **kwargs)

    monkeypatch.setattr(nl._DenseSolve, "__init__", counting)
    params = peslite.load(EXAMPLES / "gfl-example.pes", **QUIET, **{
        "events.connect_vsc.t": 0.0,
        "events.connect_vsc.ramp": 1e-3,
        "simulation.t_end": 0.002,
        "simulation.solver.circuit_model": "device",
    })
    simulation = peslite.Simulation(params)
    initial = count
    initial_structures = sum(
        len(circuit.netlist._structures) for circuit in simulation.system.circuits
    )
    simulation.run(out_dir=tmp_path)
    modes = simulation.solver._mode_eigenvalues
    new_structures = sum(
        len(circuit.netlist._structures) for circuit in simulation.system.circuits
    ) - initial_structures
    assert not modes
    assert simulation.solver._stability_results
    assert count - initial <= new_structures


@pytest.mark.parametrize("example", [
    "gfl-example", "buck-example", "wpt-scc-example",
])
def test_device_stability_matrix_matches_the_current_netlist_mode(example):
    simulation = peslite.Simulation(peslite.load(
        EXAMPLES / f"{example}.pes", **QUIET,
    ))
    for circuit in simulation.system.circuits:
        circuit.receive()
        _key, build, _bound = circuit.linear_mode()
        np.testing.assert_allclose(
            build(), circuit.netlist.topology().state_space.A,
            rtol=1e-10, atol=1e-9,
        )


@pytest.mark.full
def test_transformed_device_runtime_is_stable_at_grazing_commutations(tmp_path):
    params = peslite.load(
        EXAMPLES / "wpt-phase-shift-example.pes",
        **QUIET,
        **{"simulation.t_end": 6e-4},
    )
    result = peslite.Simulation(params).run(out_dir=tmp_path)
    assert all(np.isfinite(value) for value in result.final_states().values())


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
def test_averaged_bridge_mixture_publishes_the_same_outputs(circuit_model, tmp_path):
    changes = {
        "simulation.t_end": 1e-8,
        "simulation.solver.dt": 1e-10,
        "simulation.solver.method": "radau",
        "simulation.solver.rtol": 1e-9,
        "simulation.solver.atol": 1e-11,
        "simulation.progress.enable": 0,
        "simulation.solver.linearisations": 0,
        "units.vsc.bridge.model": "averaging",
        "units.vsc.bridge.parallel_capacitance": 1e-7,
        "units.vsc.bridge.on_resistance": 0.01,
    }

    def run(form):
        simulation = peslite.Simulation(peslite.load(
            EXAMPLES / "gfl-example.pes",
            **changes,
            **{"simulation.solver.circuit_model": form},
        ))
        simulation.run(out_dir=tmp_path / form)
        bridge = simulation.unit().bridge.output
        return bridge.u_c, bridge.i_dc, bridge.i_device

    np.testing.assert_allclose(
        run(circuit_model), run("device"), rtol=1e-9, atol=1e-9
    )


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
