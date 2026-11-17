"""Temperature affects actual electrical equations through a registered element."""

import numpy as np
import pytest

from peslite.addons.components.thermal import (
    ThermalResistor, ThermalResistorModel,
)
from peslite.solver.model import Model


def test_r_of_temperature_changes_real_electrical_voltage_equation():
    resistor = ThermalResistorModel(
        r_reference_Ohm=2, alpha_per_K=0.01, reference_K=300
    )
    resistor.input.voltage = 10.0
    resistor.set_outputs(0.0)
    cold = resistor.output.current
    resistor.input.temperature_K = 400.0
    resistor.set_outputs(0.0)
    hot = resistor.output.current
    assert cold == pytest.approx(5.0)
    assert hot == pytest.approx(2.5)
    assert resistor.resistance() == pytest.approx(4.0)
    assert resistor.dissipated_power() == pytest.approx(25.0)


def test_electrical_temperature_connection_uses_existing_model_graph():
    from types import SimpleNamespace

    class Sensor:
        state_names = ()
        outputs_need_inputs = False
        observer = True

        def __init__(self):
            self.state = SimpleNamespace()
            self.input = SimpleNamespace()
            self.output = SimpleNamespace(temperature_K=350.0)

        def set_outputs(self, _t):
            pass

        def rhs(self, _t):
            return ()

    sensor = Sensor()
    resistor = ThermalResistorModel(2, 0.01, 300)
    resistor.input.voltage = 10
    model = Model(
        {"sensor": sensor, "resistor": resistor},
        {(resistor, "temperature_K"): (sensor, "temperature_K")},
    )
    model.sync(0, np.array([]))
    assert resistor.input.temperature_K == pytest.approx(350)
    assert resistor.output.current == pytest.approx(10 / 3)


def test_registered_thermal_resistor_uses_the_bus_power_convention():
    from types import SimpleNamespace
    cfg = ThermalResistor.Params(
        bus="dc", temperature_source="thermal.temperature_K",
        r_reference_Ohm=2.0, alpha_per_K=0.005,
    )
    element = ThermalResistor(
        "load", cfg,
        {"dc": SimpleNamespace(phases=1)}, None
    )
    assert element.connections() == {
        (element.model, "temperature_K"): "thermal.temperature_K"}
    assert element.port_bindings() == (("dc", element.model.port),)
    three_phase = ThermalResistor(
        "load", cfg, {"dc": SimpleNamespace(phases=3)}, None
    )
    three_phase.model.input.voltage = 10 + 5j
    three_phase.model.set_outputs(0.0)
    assert three_phase.model.output.current == pytest.approx((10 + 5j) / 2)
    assert three_phase.model.output.heat_W == pytest.approx(1.5 * 125 / 2)
    with pytest.raises(ValueError, match="positive"):
        ThermalResistor.Params(
            bus="dc", temperature_source="thermal.temperature_K",
            r_reference_Ohm=0.0,
        )


def test_temperature_dependent_resistor_is_memoryless_in_a_complete_circuit(tmp_path):
    from pathlib import Path

    import peslite

    params = peslite.load(
        Path(__file__).resolve().parents[1] / "examples" / "buck-open-loop-example.pes",
        **{
            "simulation.t_end": 2.0e-4,
            "simulation.output.period": 1.0e-5,
            "simulation.output.precision": 17,
            "simulation.solver.linearisations": 0,
            "simulation.energy_check": "off",
        },
    ).replace(**{
        "elements.load": {
            "type": "thermal_resistor",
            "bus": "output",
            "temperature_source": "thermal.temperature_K",
            "r_reference_Ohm": 2.0,
            "alpha_per_K": 0.004,
            "reference_K": 300.0,
        },
        "elements.thermal": {
            "type": "electrothermal",
            "heat_source": "load.heat_W",
            "c_j_J_K": 1.0,
            "r_jc_K_W": 1.0,
            "c_c_J_K": 1.0,
            "r_ca_K_W": 1.0,
            "ambient_K": 300.0,
            "initial_K": 300.0,
        },
    })
    simulation = peslite.Simulation(params)
    assert not any(
        label.startswith("load.") for label in simulation.system.model.state_labels()
    )
    result = simulation.run(out_dir=tmp_path)
    assert np.max(result.plant["load.heat_W"]) > 0.0
