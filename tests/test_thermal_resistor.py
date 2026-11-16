"""Temperature affects actual electrical equations through a registered element."""

import numpy as np
import pytest

from peslite.addons.components.thermal_resistor import (
    ThermalResistor, ThermalResistorModel,
)
from peslite.solver.model import Model


def test_r_of_temperature_changes_real_electrical_voltage_equation():
    resistor = ThermalResistorModel(
        r_reference_Ohm=2, alpha_per_K=0.01, l_H=1, reference_K=300
    )
    resistor.port.voltage = 10.0
    model = Model({"ohmic": resistor}, {})
    current = np.array([1.0])
    cold = model.rhs(0.0, current)[0]
    resistor.input.temperature_K = 400.0
    hot = model.rhs(0.0, current)[0]
    assert cold == pytest.approx(8.0)
    assert hot == pytest.approx(6.0)
    assert resistor.resistance() == pytest.approx(4.0)
    model.sync(0.0, current)
    assert resistor.dissipated_power() == pytest.approx(4.0)


def test_electrical_temperature_connection_uses_existing_model_graph():
    from types import SimpleNamespace

    class Sensor:
        state_names = ()
        outputs_need_inputs = False
        observer = True

        def __init__(self):
            self.state = SimpleNamespace()
            self.input = SimpleNamespace()
            self.output = SimpleNamespace(junction_K=350.0)

        def set_outputs(self, _t):
            pass

        def rhs(self, _t):
            return ()

    sensor = Sensor()
    resistor = ThermalResistorModel(2, 0.01, 1, 300)
    resistor.port.voltage = 10
    model = Model(
        {"sensor": sensor, "resistor": resistor},
        {(resistor, "temperature_K"): (sensor, "junction_K")},
    )
    dy = model.rhs(0, np.array([1.0]))
    assert resistor.input.temperature_K == pytest.approx(350)
    assert dy[0] == pytest.approx(7.0)


def test_registered_thermal_resistor_requires_scalar_bus_and_temperature_source():
    from types import SimpleNamespace
    cfg = ThermalResistor.Params(
        bus="dc", temperature_source="thermal.junction_K",
        r_reference_Ohm=2.0, l_H=0.1, alpha_per_K=0.005,
    )
    element = ThermalResistor(
        "load", cfg,
        {"dc": SimpleNamespace(phases=1)}, None
    )
    assert element.connections() == {
        (element.model, "temperature_K"): "thermal.junction_K"}
    assert element.port_bindings() == (("dc", element.model.port),)
    with pytest.raises(ValueError, match="scalar/DC"):
        ThermalResistor("load", cfg, {"dc": SimpleNamespace(phases=3)}, None)
    with pytest.raises(ValueError, match="positive"):
        ThermalResistor.Params(
            bus="dc", temperature_source="thermal.junction_K",
            r_reference_Ohm=0.0, l_H=0.1,
        )
