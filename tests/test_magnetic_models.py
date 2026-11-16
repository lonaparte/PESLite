"""Portable magnetic results and the project magnetic-component convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import peslite
from peslite.addons.components import register_component_type
from peslite.addons.components.magcircuit import MagCircuitLaw
from peslite.addons.components.magport import MagPortElement, MagPortModel, _load
from peslite.addons.functions import (
    export_magnetic_circuit,
    magnetic_field,
    reluctance_inductance,
    write_magport,
)
from peslite.assembly.params import from_dict
from peslite.components import (MagneticBranchDescription, MagneticDescription,
                                MagneticParams)
from peslite.components.network import ELEMENT_TYPES
from peslite.solver import ConfigError
from peslite.solver.energy import spec_of


def test_external_magnetic_model_round_trip_and_strict_schema(tmp_path):
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    basis = np.arange(24, dtype=float).reshape(4, 3, 2) * 1e-6
    path = write_magport(
        tmp_path / "model.npz",
        winding_names=("primary", "secondary"),
        inductance=[[4e-3, 1e-3], [1e-3, 3e-3]],
        physical={"source": "external", "temperature": 293.15},
        magnetic={
            "branches": {
                "core": {
                    "area": 1.2e-4,
                    "path_length": 0.08,
                    "mu_r": 1800.0,
                    "core_loss": {"provider": "external"},
                },
            },
            "coupling": {
                "core": {"primary": 20.0, "secondary": -12.0},
            },
        },
        field_points=points,
        field_basis=basis,
    )

    loaded = _load(path)
    assert loaded.winding_names == ("primary", "secondary")
    np.testing.assert_array_equal(loaded.inductance, [[4e-3, 1e-3], [1e-3, 3e-3]])
    np.testing.assert_array_equal(loaded.field_points, points)
    np.testing.assert_array_equal(loaded.field_basis, basis)
    assert dict(loaded.physical) == {"source": "external", "temperature": 293.15}
    assert loaded.magnetic.coupling["core"] == {"primary": 20.0, "secondary": -12.0}
    assert loaded.magnetic.branches["core"].core_loss == {"provider": "external"}
    assert not loaded.inductance.flags.writeable

    invalid = tmp_path / "invalid.npz"
    np.savez(
        invalid,
        format="peslite-magport",
        version=1,
        winding_names=np.array(["primary"]),
        inductance=np.array([[1.0]]),
        unexpected=np.array(1.0),
    )
    with pytest.raises(ConfigError, match="unknown entries.*unexpected"):
        _load(invalid)


def test_simulation_file_resolves_external_magnetic_model_relative_to_itself(tmp_path):
    model_dir = tmp_path / "models"
    path = write_magport(
        model_dir / "coil.npz",
        winding_names=("coil",),
        inductance=[[2e-3]],
    )
    case = tmp_path / "case.pes"
    case.write_text(
        """base: {s_base: 1.0}
buses:
  node: {phases: 1}
elements:
  coil:
    type: magport
    windings:
      coil: {positive: node, negative: ground}
    file: models/coil.npz
  load: {type: resistor, bus: node, r: 2.0}
simulation: {t_end: 1.0e-4}
""",
        encoding="utf-8",
    )

    params = peslite.load(case)
    simulation = peslite.Simulation(params)
    assert params.elements["coil"].file == str(path)
    assert simulation.system.named_elements["coil"].model.names == ("coil",)


def test_linear_reluctance_network_reduces_to_reciprocal_inductance(tmp_path):
    reluctance = np.array([2.0e6, 3.0e6, 4.0e6])
    coupling = np.array([[20.0, 0.0], [0.0, 30.0], [10.0, -15.0]])
    expected = coupling.T @ np.diag(1.0 / reluctance) @ coupling

    actual = reluctance_inductance(reluctance, coupling)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)

    path = export_magnetic_circuit(
        tmp_path / "reluctance.npz",
        winding_names=("p", "s"),
        reluctance=reluctance,
        coupling=coupling,
    )
    loaded = _load(path)
    np.testing.assert_allclose(loaded.inductance, expected, rtol=1e-14, atol=0.0)
    network = dict(loaded.physical)["reluctance_network"]
    assert network["coupling"] == coupling.tolist()


def test_reluctance_network_respects_flux_continuity_constraints():
    reluctance = np.diag([2.0, 3.0])
    coupling = np.array([[4.0], [5.0]])
    incidence = np.array([[1.0, -1.0]])
    # Flux continuity forces both branch fluxes to one common value.
    expected = np.array([[(4.0 + 5.0) ** 2 / (2.0 + 3.0)]])
    np.testing.assert_allclose(
        reluctance_inductance(reluctance, coupling, incidence), expected,
        rtol=1e-14, atol=0.0,
    )


class _Result:
    def __init__(self, time, columns):
        self.t = np.asarray(time)
        self._columns = columns

    def columns(self):
        return self._columns


def test_external_field_map_reconstructs_flux_density(tmp_path):
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    basis = np.zeros((4, 3, 2))
    basis[:, 0, 0] = 2.0
    basis[:, 1, 1] = -3.0
    path = write_magport(
        tmp_path / "field.npz",
        winding_names=("p", "s"),
        inductance=[[2.0, 0.1], [0.1, 3.0]],
        field_points=points,
        field_basis=basis,
    )
    result = _Result(
        [0.0, 1.0],
        {"coil.i_p": np.array([1.0, 2.0]), "coil.i_s": np.array([4.0, 5.0])},
    )

    field = magnetic_field(result, path, element="coil")
    assert field.flux_density.shape == (2, 4, 3)
    np.testing.assert_array_equal(field.flux_density[:, 0], [[2.0, -12.0, 0.0],
                                                             [4.0, -15.0, 0.0]])
    interpolated = magnetic_field(
        result, path, element="coil", points=[[0.25, 0.25, 0.25]]
    )
    np.testing.assert_allclose(interpolated.flux_density[:, 0],
                               [[2.0, -12.0, 0.0], [4.0, -15.0, 0.0]])


def test_magport_owns_nonlinear_and_internal_magnetic_states_without_black_box():
    class StatefulLaw:
        winding_names = ("coil",)
        initial_state = {"memory": 0.25}

        @staticmethod
        def currents(_t, flux_linkage, state):
            return [flux_linkage[0] + state["memory"] * flux_linkage[0] ** 3]

        @staticmethod
        def derivatives(_t, flux_linkage, currents, state):
            return {"memory": currents[0] - state["memory"] + flux_linkage[0]}

    model = MagPortModel(StatefulLaw(), resistance=(0.2,), terminals=((False, True),))
    model.state.lambda_coil = 0.5
    model.input.u_positive_coil = 3.0
    model.set_outputs(0.0)

    assert model.state_names == ("lambda_coil", "memory")
    assert model.ports[0].behavioral and model.ports[1].grounded
    assert model.output.i_positive_coil == pytest.approx(0.53125)
    assert model.output.i_negative_coil == pytest.approx(-0.53125)
    assert model.rhs(0.0) == pytest.approx((2.89375, 0.78125))
    assert spec_of(model).kind == "default"


def test_project_addon_replaces_the_magnetic_law_through_public_interfaces(tmp_path):
    """A project addon only defines its parameters and law; common assembly stays shared."""

    class ProjectLaw:
        initial_state = {}

        def __init__(self, names, gain):
            self.winding_names = tuple(names)
            self.gain = gain

        def currents(self, _t, flux_linkage, _state):
            return self.gain * flux_linkage

        def stored_energy(self, _t, flux_linkage, _state):
            return 0.5 * self.gain * float(flux_linkage @ flux_linkage)

        @staticmethod
        def stored_energy_rate(
            _t, _flux_linkage, currents, _state, flux_rate, _state_rate
        ):
            return float(currents @ flux_rate)

    @dataclass(frozen=True, kw_only=True)
    class ProjectParams(MagneticParams):
        gain: float
        type: str = "test_project_magnet"

    class ProjectMagnet(MagPortElement):
        type = "test_project_magnet"
        Params = ProjectParams

        def __init__(self, name, cfg, buses, _params):
            super().__init__(name, cfg, buses, ProjectLaw(cfg.windings, cfg.gain))

    register_component_type(ProjectMagnet)
    try:
        case = {
            "base": {"s_base": 1.0},
            "buses": {"node": {"phases": 1}},
            "elements": {
                "magnet": {
                    "type": "test_project_magnet",
                    "windings": {
                        "coil": {"positive": "node", "negative": "ground", "r": 0.1}
                    },
                    "gain": 250.0,
                },
                "load": {"type": "resistor", "bus": "node", "r": 1.0},
            },
            "simulation": {
                "t_end": 2.0e-5,
                "energy_check": "strict",
                "initial": {"states": {"magnet.lambda_coil": 1.0e-3}},
                "solver": {"dt": 1.0e-6},
                "output": {"period": 1.0e-5, "signals": True},
            },
        }
        simulation = peslite.Simulation(from_dict(case))
        result = simulation.run(out_dir=tmp_path / "project_magnet")
        assert not simulation.energy_problems
        assert "magnet.i_coil" in result.columns()
    finally:
        ELEMENT_TYPES.pop("test_project_magnet", None)


def _linear_equivalence_case(inductance):
    return {
        "base": {"s_base": 100.0},
        "buses": {"p": {"phases": 1}, "s": {"phases": 1}},
        "elements": {
            "coils": {
                "type": "coupled_coils",
                "windings": {
                    "primary": {"positive": "p", "negative": "ground", "r": 0.15},
                    "secondary": {"positive": "s", "negative": "ground", "r": 0.2},
                },
                "inductance": np.asarray(inductance).tolist(),
            },
            "p_load": {"type": "resistor", "bus": "p", "r": 1.2},
            "s_load": {"type": "resistor", "bus": "s", "r": 1.7},
        },
        "simulation": {
            "t_end": 4.0e-4,
            "solver": {"dt": 2.0e-6},
            "output": {"period": 2.0e-5, "precision": 17,
                       "states": True, "signals": True},
        },
    }


def test_core_inductance_magport_and_magcircuit_are_numerically_equivalent(tmp_path):
    """The optimized and magnetic formulations must agree for the same complete linear model."""
    inductance = np.array([[8.0e-3, -1.5e-3], [-1.5e-3, 6.0e-3]])
    _modes, shapes = np.linalg.eigh(inductance)
    modal_initial = np.array([0.35, -0.2])
    physical_initial = shapes @ modal_initial
    linkage_initial = inductance @ physical_initial

    core = _linear_equivalence_case(inductance)
    core["simulation"]["initial"] = {
        "states": {
            "coils.i_mode_1": float(modal_initial[0]),
            "coils.i_mode_2": float(modal_initial[1]),
        }
    }

    port = _linear_equivalence_case(inductance)
    file = write_magport(
        tmp_path / "linear.npz",
        winding_names=("primary", "secondary"),
        inductance=inductance,
    )
    port["elements"]["coils"].pop("inductance")
    port["elements"]["coils"].update(type="magport", file=str(file))
    port["simulation"]["initial"] = {
        "states": {
            "coils.lambda_primary": float(linkage_initial[0]),
            "coils.lambda_secondary": float(linkage_initial[1]),
        }
    }

    circuit = _linear_equivalence_case(inductance)
    turns = np.linalg.cholesky(inductance).T
    circuit["elements"]["coils"].pop("inductance")
    circuit["elements"]["coils"].update(
        type="magcircuit",
        magnetic={
            "branches": {
                "path_1": {"reluctance": 1.0},
                "path_2": {"reluctance": 1.0},
            },
            "coupling": {
                "path_1": {
                    "primary": float(turns[0, 0]),
                    "secondary": float(turns[0, 1]),
                },
                "path_2": {
                    "primary": float(turns[1, 0]),
                    "secondary": float(turns[1, 1]),
                },
            },
        },
    )
    circuit["simulation"]["initial"] = port["simulation"]["initial"]

    results = [
        peslite.Simulation(from_dict(case)).run(out_dir=tmp_path / label)
        for label, case in (("core", core), ("port", port), ("circuit", circuit))
    ]
    columns = [result.columns() for result in results]
    for signal in (
        "coils.i_primary", "coils.i_secondary", "coils.u_primary", "coils.u_secondary"
    ):
        np.testing.assert_allclose(columns[1][signal], columns[0][signal], rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(columns[2][signal], columns[0][signal], rtol=2e-12, atol=2e-13)


def test_nonlinear_magport_file_and_magcircuit_branch_law(tmp_path):
    flux = np.linspace(-3.0e-3, 3.0e-3, 13)
    current = 120.0 * flux + 2.0e6 * flux**3
    file = write_magport(
        tmp_path / "nonlinear.npz",
        winding_names=("coil",),
        flux_linkage=flux,
        current=current,
    )
    data = _load(file)
    assert data.inductance is None
    np.testing.assert_array_equal(data.flux_linkage[:, 0], flux)
    np.testing.assert_array_equal(data.current[:, 0], current)

    law = MagCircuitLaw(
        ("coil",),
        [[4.0]],
        mmf=lambda _t, branch_flux, _state: (
            20.0 * branch_flux + 3.0e4 * branch_flux**3
        ),
        differential=lambda _t, branch_flux, _state: (
            20.0 + 9.0e4 * branch_flux**2
        ),
    )
    linkage = np.array([0.08])
    # For one branch, phi=lambda/N and i=F(phi)/N.
    expected = (20.0 * 0.02 + 3.0e4 * 0.02**3) / 4.0
    assert law.currents(0.0, linkage, {}) == pytest.approx([expected])

    base = {
        "base": {"s_base": 1.0},
        "buses": {"node": {"phases": 1}},
        "elements": {
            "magnet": {
                "type": "magport",
                "file": str(file),
                "windings": {
                    "coil": {"positive": "node", "negative": "ground", "r": 0.1}
                },
            },
            "load": {"type": "resistor", "bus": "node", "r": 1.0},
        },
        "simulation": {
            "t_end": 2.0e-4,
            "energy_check": "strict",
            "initial": {"states": {"magnet.lambda_coil": 2.0e-3}},
            "solver": {"dt": 1.0e-6, "linearisations": 0},
            "output": {"period": 1.0e-5, "precision": 17,
                       "states": True, "signals": True},
        },
    }
    circuit = {
        **base,
        "elements": {
            **base["elements"],
            "magnet": {
                "type": "magcircuit",
                "windings": base["elements"]["magnet"]["windings"],
                "magnetic": {
                    "branches": {
                        "core": {"flux": flux.tolist(), "mmf": current.tolist()},
                    },
                    "coupling": {"core": {"coil": 1.0}},
                },
            },
        },
    }
    port_simulation = peslite.Simulation(from_dict(base))
    circuit_simulation = peslite.Simulation(from_dict(circuit))
    port_result = port_simulation.run(out_dir=tmp_path / "nonlinear_port")
    circuit_result = circuit_simulation.run(out_dir=tmp_path / "nonlinear_circuit")
    assert not port_simulation.energy_problems
    assert not circuit_simulation.energy_problems
    for signal in ("magnet.i_coil", "magnet.u_coil"):
        np.testing.assert_allclose(
            circuit_result.columns()[signal], port_result.columns()[signal],
            rtol=3e-11, atol=2e-13,
        )


def test_magnetic_domain_uses_named_topology_and_material_geometry():
    branch = MagneticBranchDescription(
        area=2.0e-4,
        path_length=0.1,
        bh_curve=[[-1.0, -800.0], [0.0, 0.0], [1.0, 800.0]],
        hysteresis={"provider": "measured"},
    )
    reluctance, flux, mmf = branch.constitutive_data()
    assert reluctance is None
    np.testing.assert_allclose(flux, [-2.0e-4, 0.0, 2.0e-4])
    np.testing.assert_allclose(mmf, [-80.0, 0.0, 80.0])

    magnetic = MagneticDescription(
        branches={"core": branch, "gap": MagneticBranchDescription(reluctance=2.0e6)},
        nodes={"continuity": {"core": 1.0, "gap": -1.0}},
        coupling={"core": {"primary": 20.0}, "gap": {"secondary": 12.0}},
    )
    coupling, incidence = magnetic.matrices(("primary", "secondary"))
    np.testing.assert_array_equal(coupling, [[20.0, 0.0], [0.0, 12.0]])
    np.testing.assert_array_equal(incidence, [[1.0, -1.0]])
