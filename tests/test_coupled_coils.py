"""Independent coupled windings, compensation elements and physical checks."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from scipy.linalg import expm

import peslite
from peslite.assembly.params import from_dict, to_dict
from peslite.components.network import ELEMENT_TYPES, element_buses
from peslite.components.magnetic import MagComponent, WindingParams
from peslite.addons.components.coupled_coils import CoupledCoils


def _windings(n, *, r=0.0):
    return {
        f"w{k}": {"positive": f"p{k}", "negative": "ground", "r": r}
        for k in range(n)
    }


def _matrix(n, sign=1.0):
    return (np.eye(n) * 0.01 + sign * (np.ones((n, n)) - np.eye(n)) * 0.001).tolist()


def _case(n=3, sign=1.0):
    return {
        "base": {"s_base": 100.0},
        "buses": {f"p{k}": {"phases": 1} for k in range(n)},
        "elements": {
            "coils": {"type": "coupled_coils", "windings": _windings(n, r=0.1),
                      "inductance": _matrix(n, sign)},
            **{f"load{k}": {"type": "resistor", "bus": f"p{k}", "r": 2.0 + k}
               for k in range(n)},
        },
        "simulation": {"t_end": 1.0e-3,
                       "solver": {"dt": 5.0e-6, "linearisations": 0},
                       "output": {"period": 1.0e-4, "precision": 17,
                                  "states": True, "signals": True, "energy": True}},
    }


def test_coupled_coil_schema_round_trip_and_independent_terminals():
    case = _case(3, sign=-1)
    case["buses"]["return"] = {"phases": 1}
    case["elements"]["coils"]["windings"]["w1"]["negative"] = "return"
    params = from_dict(case)
    assert from_dict(to_dict(params)).elements["coils"] == params.elements["coils"]
    assert element_buses(params.elements["coils"])["windings.w1.negative"] == "return"

    assembly = CoupledCoils("coils", params.elements["coils"],
                            {name: type("Bus", (), {"phases": 1})()
                             for name in params.buses}, None)
    bindings = assembly.port_bindings()
    assert len(bindings) == 4
    assert {name for name, _port in bindings} == {"p0", "p1", "p2", "return"}
    assert isinstance(assembly.model, MagComponent)
    assert tuple(assembly.model.state_names) == ("i_mode_1", "i_mode_2", "i_mode_3")
    assert sum(port.grounded for port in assembly.model.ports) == 1


def test_magnetic_components_follow_one_component_convention(tmp_path):
    """Enforce the magnetic-component convention deliberately agreed within the project.

    The general configuration and its concise coupled-coils preset must produce the same ordinary
    component structure and physical result without maintaining separate equations. This internal
    design contract may change only through an explicit design decision; it is not a published
    architecture requirement or external interface standard.
    """
    concise = _case(2, sign=-1.0)
    general = deepcopy(concise)
    coils = general["elements"]["coils"]
    inductance = coils.pop("inductance")
    coils["type"] = "magnetic_component"
    coils["model"] = {"type": "matrix", "inductance": inductance}

    general_params = from_dict(general)
    rebuilt = from_dict(to_dict(general_params))
    assert rebuilt.elements["coils"] == general_params.elements["coils"]
    assert rebuilt.elements["coils"].model.type == "matrix"

    concise_simulation = peslite.Simulation(from_dict(concise))
    general_simulation = peslite.Simulation(general_params)
    concise_component = concise_simulation.system.named_elements["coils"].model
    general_component = general_simulation.system.named_elements["coils"].model
    assert type(concise_component) is type(general_component) is MagComponent
    assert concise_component.state_names == general_component.state_names
    np.testing.assert_array_equal(
        concise_component.inductance,
        general_component.inductance,
    )

    concise_result = concise_simulation.run(out_dir=tmp_path / "concise")
    general_result = general_simulation.run(out_dir=tmp_path / "general")
    assert concise_result.columns().keys() == general_result.columns().keys()
    for name, values in concise_result.columns().items():
        np.testing.assert_allclose(values, general_result.columns()[name], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("n,sign", [(2, 1.0), (3, -1.0), (4, 1.0)])
def test_coupled_coils_match_independent_full_matrix_reference(n, sign, tmp_path):
    case = _case(n, sign)
    initial_modes = np.linspace(0.2, 0.2 * n, n)
    case["simulation"]["initial"] = {
        "states": {f"coils.i_mode_{j + 1}": float(v)
                   for j, v in enumerate(initial_modes)}
    }
    params = from_dict(case)
    simulation = peslite.Simulation(params)
    component = simulation.system.named_elements["coils"].model
    assert not simulation.energy_problems
    result = simulation.run(out_dir=tmp_path)
    columns = result.columns()

    matrix = np.asarray(case["elements"]["coils"]["inductance"])
    resistances = np.diag([2.1 + k for k in range(n)])
    initial = np.linalg.eigh(matrix)[1] @ initial_modes
    target = expm(-np.linalg.solve(matrix, resistances) * result.t[-1]) @ initial
    actual = np.array([columns[f"coils.i_w{k}"][-1] for k in range(n)])
    np.testing.assert_allclose(actual, target, rtol=2e-5, atol=1e-8)

    expected_energy = 0.5 * actual @ matrix @ actual
    modal = np.array([result.final_states()[f"coils.i_mode_{k + 1}"] for k in range(n)])
    calculated_energy = sum(0.5 * storage.value * v**2
                            for storage, v in zip(component.storage, modal))
    assert calculated_energy == pytest.approx(expected_energy, rel=1e-9, abs=1e-10)


@pytest.mark.parametrize("matrix,error", [
    ([[1.0, 0.0]], "finite 2 x 2"),
    ([[1.0, 0.4], [0.3, 1.0]], "symmetric"),
    ([[1.0, 1.1], [1.1, 1.0]], "positive definite"),
    ([[1.0, 0.0], [0.0, float("nan")]], "finite 2 x 2"),
])
def test_coupled_coils_reject_invalid_inductance_matrix(matrix, error):
    params = {name: WindingParams(positive=f"p{idx}", negative="ground")
              for idx, name in enumerate(("tx", "rx"))}
    with pytest.raises(ValueError, match=error):
        CoupledCoils.Params(windings=params, inductance=matrix)


def test_coupled_coils_reject_bad_winding_and_terminal_reference():
    with pytest.raises(ValueError, match="distinct"):
        WindingParams(positive="p", negative="p")
    with pytest.raises(ValueError, match="r must be finite"):
        WindingParams(positive="p", negative="ground", r=-1.0)
    bad = _case(2)
    bad["elements"]["coils"]["windings"]["w1"]["positive"] = "missing"
    with pytest.raises(ValueError, match="unknown bus.*missing"):
        from_dict(bad)


def test_compensation_capacitor_is_a_separate_registered_element():
    assert "coupled_coils" in ELEMENT_TYPES
    assert "magnetic_component" in ELEMENT_TYPES
    assert "capacitor" in ELEMENT_TYPES
    c = ELEMENT_TYPES["capacitor"].params(bus1="p", bus2="q", c=1e-6, r=0.02)
    assert c.type == "capacitor"
    with pytest.raises(ValueError, match="distinct"):
        ELEMENT_TYPES["capacitor"].params(bus1="p", bus2="p", c=1e-6)
    with pytest.raises(ValueError, match="c must be finite"):
        ELEMENT_TYPES["capacitor"].params(bus1="p", bus2="q", c=-1)


def test_compensation_capacitor_joins_independent_coil_network(tmp_path):
    case = _case(2)
    case["elements"]["compensation"] = {
        "type": "capacitor", "bus1": "p0", "bus2": "p1",
        "c": 2.0e-4, "r": 0.01,
    }
    simulation = peslite.Simulation(from_dict(case))
    capacitor = simulation.system.named_elements["compensation"].model
    assert capacitor.state_names == ("u",)
    assert not simulation.energy_problems
    result = simulation.run(out_dir=tmp_path)
    assert result.summary["t_stop"] == pytest.approx(case["simulation"]["t_end"])


def test_existing_wpt_configuration_is_unchanged():
    from conftest import EXAMPLES

    params = peslite.load(EXAMPLES / "wpt-ss-example.pes")
    simulation = peslite.Simulation(params)
    assert "charger.link" in simulation.system.model.names
    assert not simulation.energy_problems


def test_distinct_winding_terminals_match_full_matrix_circuit(tmp_path):
    inductance = np.array([[0.01, -0.002], [-0.002, 0.012]])
    series_resistance = (0.15, 0.25)
    positive_loads = (1.0, 1.5)
    negative_loads = (2.0, 2.5)
    case = _case(2)
    case["buses"].update({f"n{k}": {"phases": 1} for k in range(2)})
    case["elements"]["coils"]["inductance"] = inductance.tolist()
    for k in range(2):
        case["elements"]["coils"]["windings"][f"w{k}"].update(
            positive=f"p{k}", negative=f"n{k}", r=series_resistance[k]
        )
        case["elements"][f"load{k}"]["r"] = positive_loads[k]
        case["elements"][f"return_load{k}"] = {
            "type": "resistor", "bus": f"n{k}", "r": negative_loads[k]
        }
    initial_modes = np.array([0.5, -0.3])
    case["simulation"]["initial"] = {
        "states": {f"coils.i_mode_{k + 1}": float(value)
                   for k, value in enumerate(initial_modes)}
    }

    simulation = peslite.Simulation(from_dict(case))
    component = simulation.system.named_elements["coils"].model
    assert len(component.ports) == 4
    assert not any(port.grounded for port in component.ports)
    assert not simulation.energy_problems
    result = simulation.run(out_dir=tmp_path)
    currents = np.array([result.columns()[f"coils.i_w{k}"][-1] for k in range(2)])
    effective_resistance = np.diag(np.array(series_resistance) + positive_loads + np.array(negative_loads))
    starting_currents = np.linalg.eigh(inductance)[1] @ initial_modes
    expected = expm(-np.linalg.solve(inductance, effective_resistance) * result.t[-1]) @ starting_currents
    np.testing.assert_allclose(currents, expected, rtol=2e-5, atol=1e-8)
    for k in range(2):
        voltage = result.columns()[f"coils.u_w{k}"][-1]
        assert voltage == pytest.approx(
            -(positive_loads[k] + negative_loads[k]) * currents[k],
            rel=1e-6, abs=1e-8,
        )


def test_coupled_coils_restart_reproduces_modal_currents(tmp_path):
    case = _case(2, sign=-1.0)
    case["simulation"]["t_end"] = 6.0e-4
    case["simulation"]["initial"] = {
        "states": {"coils.i_mode_1": 0.4, "coils.i_mode_2": -0.2}
    }
    whole_dir = tmp_path / "whole"
    whole = peslite.Simulation(from_dict(case)).run(out_dir=whole_dir)
    start_time = 3.0e-4
    restarted = peslite.load(
        whole_dir / "simulation.pes",
        initial=whole_dir / "states.csv",
        initial_time=start_time,
    )
    continued = peslite.Simulation(restarted).run(out_dir=tmp_path / "continued")
    start = int(np.argmin(abs(whole.t - start_time)))
    np.testing.assert_array_equal(continued.t, whole.t[start:])
    for name, values in continued.states.items():
        np.testing.assert_allclose(values, whole.states[name][start:],
                                   rtol=1e-11, atol=1e-12, err_msg=name)
