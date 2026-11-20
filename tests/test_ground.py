"""Shared explicit and implicit electrical-ground semantics."""

from copy import deepcopy

import numpy as np
import pytest

import peslite
from peslite.assembly.params import ConfigError, from_dict


def _network(phases: int = 1) -> dict:
    return {
        "base": {"s_base": 1000.0, "v_ll_rms": 100.0, "f0": 50.0},
        "buses": {"node": {"phases": phases, "c": 1.0e-3}},
        "sources": {
            "source": {"bus": "node", "l": 1.0e-3, "r": 0.2, "v": 10.0},
        },
        "simulation": {
            "t_end": 1.0e-3,
            "solver": {"dt": 1.0e-5, "linearisations": 0},
            "progress": {"enable": False},
        },
    }


def test_explicit_ground_branch_uses_the_existing_grounded_port():
    tree = _network()
    tree["branches"] = {
        "shunt": {"bus1": "node", "bus2": "ground", "l": 2.0e-3, "r": 2.0},
    }

    simulation = peslite.Simulation(from_dict(tree))
    branch = simulation.system.branches["shunt"]

    assert "ground" not in simulation.system.buses
    assert not branch.port1.grounded
    assert branch.port2.grounded


@pytest.mark.parametrize("phases", (1, 3))
def test_explicit_ground_matches_the_existing_implicit_load(tmp_path, phases):
    explicit = _network(phases)
    explicit["branches"] = {
        "shunt": {"bus1": "node", "bus2": "ground", "l": 2.0e-3, "r": 2.0},
    }
    implicit = deepcopy(explicit)
    implicit.pop("branches")
    implicit["elements"] = {
        "shunt": {"type": "load", "bus": "node", "l": 2.0e-3, "r": 2.0},
    }

    branch_simulation = peslite.Simulation(from_dict(explicit))
    load_simulation = peslite.Simulation(from_dict(implicit))
    branch_simulation.run(out_dir=tmp_path / "branch")
    load_simulation.run(out_dir=tmp_path / "load")

    branch_current = branch_simulation.system.branches["shunt"].state.i
    load_current = load_simulation.system.named_elements["shunt"].branch.state.i
    np.testing.assert_allclose(branch_current, load_current, rtol=1e-12, atol=1e-12)


def test_declarative_two_terminal_component_accepts_ground():
    tree = _network()
    tree["elements"] = {
        "capacitor": {
            "type": "capacitor", "bus1": "node", "bus2": "ground", "c": 2.0e-3,
        },
    }

    system = peslite.System(from_dict(tree))
    component = system.named_elements["capacitor"].model

    assert not component.port_bus1.grounded
    assert component.port_bus2.grounded
    assert component.port_bus2 in component.ports


def test_network_without_ground_connection_remains_floating():
    tree = _network()
    tree.pop("sources")
    tree["buses"]["other"] = {"phases": 1, "c": 2.0e-3}
    tree["branches"] = {
        "link": {"bus1": "node", "bus2": "other", "l": 2.0e-3, "r": 2.0},
    }

    system = peslite.System(from_dict(tree))
    branch = system.branches["link"]

    assert not branch.port1.grounded
    assert not branch.port2.grounded


def test_ground_is_reserved_and_cannot_be_the_only_component_endpoint():
    conflict = _network()
    conflict["buses"]["ground"] = {"phases": 1}
    with pytest.raises(ConfigError, match="reserved electrical reference"):
        from_dict(conflict)

    isolated = _network()
    isolated["elements"] = {"load": {"type": "load", "bus": "ground", "l": 1.0e-3}}
    with pytest.raises(ConfigError, match="cannot connect only to ground"):
        from_dict(isolated)
