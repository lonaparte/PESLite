"""Portable magnetic results and the project magnetic-component convention."""

from __future__ import annotations

import numpy as np
import pytest

import peslite
from peslite.addons.functions import (
    build_reluctance_model,
    import_magnetic_model,
    magnetic_field,
    reluctance_inductance,
)
from peslite.components.magnetic import MagneticModel
from peslite.solver import BlackBoxSubsystem, ConfigError, Model
from peslite.solver.energy import Port


def test_external_magnetic_model_round_trip_and_strict_schema(tmp_path):
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    basis = np.arange(24, dtype=float).reshape(4, 3, 2) * 1e-6
    path = import_magnetic_model(
        tmp_path / "model.npz",
        winding_names=("primary", "secondary"),
        inductance=[[4e-3, 1e-3], [1e-3, 3e-3]],
        physical={"source": "external", "temperature": 293.15},
        field_points=points,
        field_basis=basis,
    )

    loaded = MagneticModel.load(path)
    assert loaded.winding_names == ("primary", "secondary")
    np.testing.assert_array_equal(loaded.inductance, [[4e-3, 1e-3], [1e-3, 3e-3]])
    np.testing.assert_array_equal(loaded.field_points, points)
    np.testing.assert_array_equal(loaded.field_basis, basis)
    assert dict(loaded.physical) == {"source": "external", "temperature": 293.15}
    assert not loaded.inductance.flags.writeable

    invalid = tmp_path / "invalid.npz"
    np.savez(
        invalid,
        format="peslite-magnetic-model",
        version=1,
        winding_names=np.array(["primary"]),
        inductance=np.array([[1.0]]),
        unexpected=np.array(1.0),
    )
    with pytest.raises(ConfigError, match="unknown entries.*unexpected"):
        MagneticModel.load(invalid)


def test_simulation_file_resolves_external_magnetic_model_relative_to_itself(tmp_path):
    model_dir = tmp_path / "models"
    path = import_magnetic_model(
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
    type: magnetic_component
    windings:
      coil: {positive: node, negative: ground}
    model: {file: models/coil.npz}
  load: {type: resistor, bus: node, r: 2.0}
simulation: {t_end: 1.0e-4}
""",
        encoding="utf-8",
    )

    params = peslite.load(case)
    simulation = peslite.Simulation(params)
    assert params.elements["coil"].model.file == str(path)
    assert simulation.system.named_elements["coil"].model.magnetic.winding_names == ("coil",)


def test_linear_reluctance_network_reduces_to_reciprocal_inductance(tmp_path):
    reluctance = np.array([2.0e6, 3.0e6, 4.0e6])
    coupling = np.array([[20.0, 0.0], [0.0, 30.0], [10.0, -15.0]])
    expected = coupling.T @ np.diag(1.0 / reluctance) @ coupling

    actual = reluctance_inductance(reluctance, coupling)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)

    path = build_reluctance_model(
        tmp_path / "reluctance.npz",
        winding_names=("p", "s"),
        reluctance=reluctance,
        coupling=coupling,
    )
    loaded = MagneticModel.load(path)
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
    path = import_magnetic_model(
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


def test_external_nonlinear_constitutive_result_uses_plain_black_box_state():
    """A nonlinear magnetic result uses the existing state and physical-port contracts."""
    port = Port(name="winding", effort="input.voltage", flow="output.current")

    def current(flux):
        return flux[0] + 0.2 * flux[0] ** 3

    model = BlackBoxSubsystem(
        output=lambda _t, state, _inputs: (current(state),),
        rhs=lambda _t, _state, inputs: (inputs[0],),
        inputs={"voltage": 0.0},
        outputs={"current": 0.0},
        state={"flux_linkage": 0.0},
        feedthrough=(),
        ports=(port,),
    )
    port.bind(model)
    model.input.voltage = 3.0
    assembled = Model({"magnetics": model}, {})
    state = np.array([0.5])

    assembled.set_states(state)
    assembled.evaluate_outputs(0.0)
    assert assembled.state_labels() == ["magnetics.flux_linkage"]
    assert assembled.rhs(0.0, state) == pytest.approx([3.0])
    assert model.output.current == pytest.approx(0.525)
