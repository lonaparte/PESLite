"""Portable magnetic results and the project magnetic-component convention."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.linalg import expm
from scipy.optimize import brentq

import peslite
from peslite.addons.components import register_component_type
from peslite.addons.components.magcircuit import MagCircuitLaw
from peslite.addons.components.magport import (
    MagPort,
    MagPortModel,
    TabulatedEnergyLaw,
    TabulatedMagneticLaw,
    _load,
)
from peslite.addons.functions import (
    export_magnetic_circuit,
    magnetic_field,
    reluctance_inductance,
    write_magport,
)
from peslite.assembly.params import from_dict
from peslite.addons.components.magcomponent import (
    CoupledCoilsModel,
    MagneticBranchDescription,
    MagneticDescription,
    MagneticParams,
)
from peslite.components.network import ELEMENT_TYPES, LinearEnergyLaw
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

    grid = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    multiple = TabulatedMagneticLaw(("p", "s"), grid, 2.0 * grid)
    np.testing.assert_allclose(
        multiple.currents(0.0, np.array([0.25, 0.25]), {}), [0.5, 0.5]
    )
    with pytest.raises(ConfigError, match="outside the sampled magnetic current map"):
        multiple.currents(0.0, np.array([0.75, 0.75]), {})


def test_multiwinding_energy_samples_define_one_conservative_current_map(tmp_path):
    axes = (np.array([-1.0, 0.0, 1.0]), np.array([-2.0, 0.0, 2.0]))
    flux = np.asarray([(x, y) for x in axes[0] for y in axes[1]])
    # This bilinear potential is represented exactly by multilinear interpolation.
    energy = 2.0 * flux[:, 0] + 3.0 * flux[:, 1] + 0.4 * flux[:, 0] * flux[:, 1]
    law = TabulatedEnergyLaw(("primary", "secondary"), flux, energy)
    point = np.array([0.25, -0.5])
    direction = np.array([0.7, -0.2])
    currents = law.currents(0.0, point, {})
    np.testing.assert_allclose(currents, [2.0 + 0.4 * point[1], 3.0 + 0.4 * point[0]])
    step = 1e-7
    numerical_rate = (
        law.stored_energy(0.0, point + step * direction, {})
        - law.stored_energy(0.0, point - step * direction, {})
    ) / (2.0 * step)
    assert numerical_rate == pytest.approx(currents @ direction, rel=1e-7)
    assert law.stored_energy_rate(0.0, point, currents, {}, direction, {}) == pytest.approx(
        numerical_rate, rel=1e-7
    )

    model = MagPortModel(
        law,
        resistance=(0.0, 0.0),
        terminals=((False, True), (False, True)),
    )
    model.state.lambda_primary, model.state.lambda_secondary = point
    model.set_outputs(0.0)
    np.testing.assert_allclose(
        [model.output.i_positive_primary, model.output.i_positive_secondary], currents
    )
    specification = spec_of(model)
    assert specification.stored_energy is not None
    assert specification.stored_energy_rate is not None
    assert model.stored_energy(0.0) == pytest.approx(law.stored_energy(0.0, point, {}))

    file = write_magport(
        tmp_path / "energy-map.npz",
        winding_names=("primary", "secondary"),
        flux_linkage=flux,
        stored_energy=energy,
    )
    data = _load(file)
    assert data.current is None
    np.testing.assert_array_equal(data.stored_energy, energy)


def test_linear_inductance_uses_the_same_energy_law_contract():
    inductance = np.array([[4.0, 1.0], [1.0, 3.0]])
    law = LinearEnergyLaw(("primary", "secondary"), inductance)
    flux = np.array([0.7, -0.2])
    flux_rate = np.array([0.3, 0.4])
    current = np.linalg.solve(inductance, flux)

    np.testing.assert_allclose(law.currents(0.0, flux, {}), current)
    assert law.stored_energy(0.0, flux, {}) == pytest.approx(0.5 * flux @ current)
    assert law.stored_energy_rate(
        0.0, flux, current, {}, flux_rate, {}
    ) == pytest.approx(current @ flux_rate)

    model = CoupledCoilsModel(
        law.winding_names,
        inductance,
        resistance=(0.0, 0.0),
        terminals=((False, True), (False, True)),
    )
    modal_current = np.array([0.2, -0.35])
    for name, value in zip(model.state_names, modal_current):
        setattr(model.state, name, value)
    winding_current = law.shapes @ modal_current
    np.testing.assert_allclose(model._winding_currents(), winding_current)
    linkage = inductance @ winding_current
    assert law.stored_energy(0.0, linkage, {}) == pytest.approx(
        0.5 * np.dot(law.modes, modal_current * modal_current)
    )


def test_multiwinding_energy_samples_require_a_complete_cartesian_grid():
    with pytest.raises(ConfigError, match="complete Cartesian flux grid"):
        TabulatedEnergyLaw(
            ("primary", "secondary"),
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [0.0, 0.5, 0.5],
        )


def test_magport_matches_magnetic_loss_and_temperature_coupling_signals():
    class LinearLaw:
        winding_names = ("coil",)
        initial_state = {}

        @staticmethod
        def currents(_t, flux_linkage, _state):
            return flux_linkage

    model = MagPortModel(
        LinearLaw(),
        resistance=(2.0,),
        terminals=((False, True),),
        temperature_coefficient_per_K=0.01,
        temperature_reference_K=300.0,
    )
    model.state.lambda_coil = 1.0
    model.input.u_positive_coil = 10.0
    model.set_outputs(0.0)
    assert model.rhs(0.0) == pytest.approx((8.0,))
    assert model.output.copper_loss_W == pytest.approx(2.0)
    assert model.output.core_loss_W == 0.0

    model.input.temperature_K = 400.0
    model.set_outputs(0.0)
    assert model.rhs(0.0) == pytest.approx((6.0,))
    assert model.output.loss_W == pytest.approx(4.0)


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

    class ProjectMagnet(MagPort):
        type = "test_project_magnet"
        Params = ProjectParams

        def __init__(self, name, cfg, buses, _params):
            super().__init__(
                name,
                cfg,
                buses,
                _params,
                law=ProjectLaw(cfg.windings, cfg.gain),
            )

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
        "buses": {
            "p": {"phases": 1},
            "s": {"phases": 1},
            "n": {"phases": 1},
        },
        "elements": {
            "coils": {
                "type": "coupled_coils",
                "windings": {
                    "primary": {"positive": "p", "negative": "n", "r": 0.15},
                    "secondary": {"positive": "s", "negative": "ground", "r": 0.2},
                },
                "inductance": np.asarray(inductance).tolist(),
            },
            "p_load": {"type": "resistor", "bus": "p", "r": 1.2},
            "return": {"type": "load", "bus": "n", "l": 2.0e-3, "r": 0.4},
            "s_load": {"type": "resistor", "bus": "s", "r": 1.7},
        },
        "simulation": {
            "t_end": 4.0e-4,
            "solver": {"dt": 2.0e-6},
            "output": {"period": 2.0e-5, "precision": 17,
                       "states": True, "signals": True},
        },
    }


def test_base_inductance_magport_and_magcircuit_are_numerically_equivalent(tmp_path):
    """The optimized and magnetic formulations must agree for the same complete linear model."""
    inductance = np.array([[8.0e-3, -1.5e-3], [-1.5e-3, 6.0e-3]])
    _modes, shapes = np.linalg.eigh(inductance)
    modal_initial = np.array([0.35, -0.2])
    physical_initial = shapes @ modal_initial

    core = _linear_equivalence_case(inductance)
    core["simulation"]["initial"] = {
        "states": {
            "coils.i_mode_1": float(modal_initial[0]),
            "coils.i_mode_2": float(modal_initial[1]),
            "return.branch.i": float(physical_initial[0]),
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
    port["simulation"]["initial"] = core["simulation"]["initial"]

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

    simulations = [peslite.Simulation(from_dict(case)) for case in (core, port, circuit)]
    models = [simulation.system.named_elements["coils"].model for simulation in simulations]
    assert {type(model) for model in models} == {type(models[0])}
    results = [simulation.run(out_dir=tmp_path / label)
               for label, simulation in zip(("core", "port", "circuit"), simulations)]
    columns = [result.columns() for result in results]
    for signal in (
        "coils.i_primary", "coils.i_secondary", "coils.u_primary", "coils.u_secondary"
    ):
        np.testing.assert_allclose(columns[1][signal], columns[0][signal], rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(columns[2][signal], columns[0][signal], rtol=2e-12, atol=2e-13)

    loop_inductance = inductance + np.diag([2.0e-3, 0.0])
    loop_resistance = np.diag([0.15 + 1.2 + 0.4, 0.2 + 1.7])
    expected = expm(
        -np.linalg.solve(loop_inductance, loop_resistance) * results[0].t[-1]
    ) @ physical_initial
    actual = [columns[0]["coils.i_primary"][-1], columns[0]["coils.i_secondary"][-1]]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-8)


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

    legs = np.array([[2.0e5, 4.0e14], [3.0e5, 1.0e14], [1.0e5, 2.0e14]])
    three_leg = MagCircuitLaw(
        ("coil",),
        [[0.0], [0.0], [50.0]],
        incidence=[[1.0, 1.0, -1.0]],
        mmf=lambda _t, phi, _state: legs[:, 0] * phi + legs[:, 1] * phi**3,
        differential=lambda _t, phi, _state: (
            legs[:, 0] + 3.0 * legs[:, 1] * phi**2
        ),
    )

    def leg(phi, index):
        return legs[index, 0] * phi + legs[index, 1] * phi**3

    for value in np.linspace(-2.0e-2, 2.0e-2, 41):
        centre = value / 50.0
        outer = brentq(
            lambda phi: leg(phi, 0) - leg(centre - phi, 1), -1.0, 1.0
        )
        reference = (leg(outer, 0) + leg(centre, 2)) / 50.0
        assert three_leg.currents(0.0, np.array([value]), {}) == pytest.approx(
            [reference], rel=1e-8, abs=1e-12
        )

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

    interpolator = PchipInterpolator(flux, current)
    slope = interpolator.derivative()
    initial_current = float(interpolator(2.0e-3))
    series_results = []
    for label, case in (("port", base), ("circuit", circuit)):
        series = deepcopy(case)
        series["buses"]["return"] = {"phases": 1}
        series["elements"]["magnet"]["windings"]["coil"]["negative"] = "return"
        series["elements"]["return"] = {
            "type": "load", "bus": "return", "l": 1.0e-3, "r": 0.5,
        }
        series["simulation"]["initial"]["states"]["return.branch.i"] = initial_current
        simulation = peslite.Simulation(from_dict(series))
        result = simulation.run(out_dir=tmp_path / f"nonlinear_{label}_series")
        assert not simulation.energy_problems
        series_results.append(result)

    reference = solve_ivp(
        lambda _t, linkage: (
            -1.6 * interpolator(linkage[0])
            / (1.0 + 1.0e-3 * slope(linkage[0])),
        ),
        (series_results[0].t[0], series_results[0].t[-1]),
        (2.0e-3,),
        t_eval=series_results[0].t,
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    expected_current = interpolator(reference.y[0])
    for result in series_results:
        np.testing.assert_allclose(
            result.columns()["magnet.i_coil"], expected_current,
            rtol=3.0e-6, atol=2.0e-10,
        )
    np.testing.assert_allclose(
        series_results[1].columns()["magnet.i_coil"],
        series_results[0].columns()["magnet.i_coil"],
        rtol=3.0e-8, atol=2.0e-11,
    )


def test_magcircuit_material_loss_uses_solved_branch_flux_and_rate():
    observed = {}

    def loss(t, branch_flux, branch_rate, winding_current, state, temperature_K):
        observed.update(
            t=t,
            flux=branch_flux.copy(),
            rate=branch_rate.copy(),
            current=winding_current.copy(),
            state=dict(state),
            temperature=temperature_K,
        )
        return 3.0 * branch_rate[0] ** 2

    law = MagCircuitLaw(
        ("coil",),
        [[4.0]],
        mmf=lambda _t, branch_flux, _state: 20.0 * branch_flux,
        differential=lambda _t, branch_flux, _state: np.full_like(branch_flux, 20.0),
        dissipated_power=loss,
    )
    model = MagPortModel(law, resistance=(2.0,), terminals=((False, True),))
    model.state.lambda_coil = 0.08
    model.input.u_positive_coil = 4.2
    model.set_outputs(0.25)

    np.testing.assert_allclose(observed["flux"], [0.02])
    np.testing.assert_allclose(observed["rate"], [1.0])
    np.testing.assert_allclose(observed["current"], [0.1])
    assert observed["t"] == 0.25
    assert observed["state"] == {}
    assert observed["temperature"] == 298.15
    assert model.output.copper_loss_W == pytest.approx(0.02)
    assert model.output.core_loss_W == pytest.approx(3.0)
    assert model.output.loss_W == pytest.approx(3.02)
    assert model.dissipated_power() == pytest.approx(3.02)

    observed.clear()
    reluctance = np.array([2.0, 3.0, 5.0])
    constrained = MagCircuitLaw(
        ("coil",),
        [[0.0], [0.0], [5.0]],
        incidence=[[1.0, 1.0, -1.0]],
        mmf=lambda _t, flux, _state: reluctance * flux,
        differential=lambda _t, _flux, _state: reluctance,
        dissipated_power=loss,
    )
    model = MagPortModel(constrained, resistance=(0.0,), terminals=((False, True),))
    model.state.lambda_coil = 0.1
    model.input.u_positive_coil = 0.5
    model.set_outputs(0.0)
    np.testing.assert_allclose(observed["flux"], [0.012, 0.008, 0.02])
    np.testing.assert_allclose(observed["rate"], [0.06, 0.04, 0.1])
    assert model.output.core_loss_W == pytest.approx(3.0 * 0.06**2)


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

    providers = {
        "hysteresis": {"provider": "measured"},
        "core_loss": {"provider": "measured"},
        "eddy_current": {"provider": "measured"},
    }
    case = {
        "base": {"s_base": 1.0},
        "buses": {"p": {"phases": 1}, "s": {"phases": 1}},
        "elements": {
            "magnet": {
                "type": "magcircuit",
                "windings": {
                    "primary": {"positive": "p", "negative": "ground"},
                    "secondary": {"positive": "s", "negative": "ground"},
                },
                "magnetic": {
                    "branches": {
                        "core": {
                            "area": 2.0e-4,
                            "path_length": 0.1,
                            "bh_curve": [[-1.0, -800.0], [0.0, 0.0], [1.0, 800.0]],
                            **providers,
                        },
                        "gap": {"reluctance": 2.0e6},
                    },
                    "coupling": {
                        "core": {"primary": 20.0},
                        "gap": {"secondary": 12.0},
                    },
                },
            },
            "p_load": {"type": "resistor", "bus": "p", "r": 1.0},
            "s_load": {"type": "resistor", "bus": "s", "r": 1.0},
        },
        "simulation": {"t_end": 1.0e-6},
    }
    with pytest.warns(UserWarning, match="retains but does not evaluate") as caught:
        peslite.Simulation(from_dict(case))
    message = str(caught[0].message)
    for field in providers:
        assert f"core.{field}" in message
