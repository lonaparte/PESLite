"""Physical port perturbations preserve native circuit and power orientations."""

import math
from types import SimpleNamespace

import numpy as np
import peslite
import pytest

from peslite.addons.components.port_injection import PortInjection
from peslite.assembly.params import from_dict, read_tree
from peslite.solver import ConfigError, netlist as nl


def _injector(**kwargs):
    cfg = PortInjection.Params(bus="pcc", **kwargs)
    return PortInjection("stimulus", cfg, {"pcc": SimpleNamespace(phases=1)}, None)


@pytest.mark.parametrize("kind,primitive", [
    ("current", nl.CurrentSource), ("voltage", nl.VoltageSource),
    ("load", nl.CurrentSource),
])
def test_injectors_use_the_correct_shunt_or_series_connection(kind, primitive):
    options = {"nominal_voltage": 100.0} if kind == "load" else {}
    element = _injector(kind=kind, amplitude=2.0, frequency=25.0, **options)
    assert element.model.state_names == ()
    assert (len(element.model.ports) == 2) == (kind == "voltage")
    if kind == "voltage":
        assert element.series_connections() == {"pcc": "stimulus.bus.in"}
        assert element.port_bindings()[0][0] == "stimulus.bus.in"
        assert element.model.injection_at(0.0) == pytest.approx(2.0)
    elif kind == "current":
        assert element.series_connections() == {}
        assert element.model.injection_at(0.0) == pytest.approx(2.0)
    else:
        assert element.model.injection_at(0.0) == pytest.approx(-200.0)

    class Builder:
        def __init__(self):
            self.nodes = {}
            self.elements = []
            self.outputs = {}
            self.supplier = False

        def node(self, port):
            return self.nodes.setdefault(id(port), nl.Node(1, name=port.name))

        def function(self, callback):
            return callback

        def add(self, *elements):
            self.elements.extend(elements)

        def output(self, _model, field, quantity):
            self.outputs[field] = quantity

        def port_output(self, _model, *_args):
            pass

        def supplies(self, _model):
            self.supplier = True

        def losses(self, _model):
            pass

    builder = Builder()
    element.model.network(builder)
    source = builder.elements[0]
    assert isinstance(source, primitive)
    assert builder.supplier
    if kind == "voltage":
        assert source.ports == ((builder.node(element.model.port), builder.node(element.model.inlet)),)
        assert builder.outputs["injection"][0] == "branch_voltage"
    else:
        assert source.ports == ((None, builder.node(element.model.port)),)
        assert builder.outputs["injection"][0] == "branch_current"


def test_three_phase_dq_injection_rotates_at_the_configured_frame_rate():
    cfg = PortInjection.Params(bus="pcc", kind="current", amplitude=3.0,
                               frequency=10.0, frame_frequency=50.0, axis="q")
    element = PortInjection("stimulus", cfg, {"pcc": SimpleNamespace(phases=3)}, None)
    assert element.model.injection_at(0.0) == pytest.approx(3j)
    t = 0.005
    expected = 3 * math.cos(math.tau * 10 * t) * 1j * complex(
        math.cos(math.tau * 50 * t), math.sin(math.tau * 50 * t))
    assert element.model.injection_at(t) == pytest.approx(expected)
    assert element.model.port.scale == pytest.approx(1.5)


def test_sequence_injection_preserves_phase_and_rejects_bias():
    settings = dict(bus="pcc", kind="current", amplitude=2.0,
                    frequency=20.0, frame_frequency=50.0,
                    phase=math.pi / 2, axis="negative")
    cfg = PortInjection.Params(**settings)
    source = PortInjection("stimulus", cfg, {"pcc": SimpleNamespace(phases=3)}, None)
    assert source.model.injection_at(0.0) == pytest.approx(2j)
    with pytest.raises(ConfigError, match="zero bias"):
        PortInjection.Params(**{**settings, "bias": 0.2})


def test_explicit_voltage_inlet_handles_arbitrary_connection_nodes():
    cfg = PortInjection.Params(bus="out", inlet="in", kind="voltage", bias=2.0)
    buses = {"in": SimpleNamespace(phases=1), "out": SimpleNamespace(phases=1)}
    source = PortInjection("series", cfg, buses, None)
    assert source.series_connections() == {}
    assert tuple(bus for bus, _ in source.port_bindings()) == ("in", "out")
    with pytest.raises(ConfigError, match="matching phases"):
        PortInjection("bad", cfg, {"in": SimpleNamespace(phases=3),
                                   "out": SimpleNamespace(phases=1)}, None)


@pytest.mark.parametrize("change", [
    {"kind": "short"}, {"amplitude": float("nan")}, {"frequency": -1.0},
    {"tone_amplitudes": [2.0]}, {"kind": "load"},
    {"kind": "voltage", "inlet": "pcc"}, {"nominal_voltage": 1.0},
])
def test_bad_physical_injections_fail_during_schema_validation(change):
    with pytest.raises(ConfigError):
        PortInjection.Params(bus="pcc", **change)


def test_shunt_current_run_exports_negative_port_current_and_injected_current(examples, tmp_path):
    tree = read_tree(examples / "buck-open-loop-example.pes")
    tree["events"] = {}
    tree["elements"]["injection"] = {
        "type": "port_injection", "kind": "current", "bus": "output",
        "frequency": 400.0, "amplitude": 0.2,
    }
    tree["simulation"].update(t_end=0.005, energy_check="off")
    tree["simulation"]["output"].update(period=10e-6, signals=True, states=False, energy=False)
    sim = peslite.Simulation(from_dict(tree))
    result = sim.run(out_dir=tmp_path)
    current = result.plant["injection.injection"]
    into_source = result.plant["injection.current"]
    assert not result.tripped
    assert np.ptp(current) == pytest.approx(0.4, rel=0.03)
    np.testing.assert_allclose(into_source, -current, rtol=1e-8, atol=1e-9)


def test_series_voltage_source_across_two_named_ports_changes_physical_voltage(tmp_path):
    tree = {
        "base": {"s_base": 1000.0, "v_ll_rms": 230.0, "f0": 50.0},
        "buses": {"input": {"phases": 1}, "output": {"phases": 1, "c": 0.001}},
        "elements": {
            "supply": {"type": "voltage_source", "bus": "input", "v": 10.0, "r": 1.0},
            "series": {"type": "port_injection", "kind": "voltage", "inlet": "input",
                       "bus": "output", "bias": 2.0},
            "load": {"type": "resistor", "bus": "output", "r": 9.0},
        },
        "simulation": {"t_end": 0.2, "solver": {"dt": 1e-4, "linearisations": 0},
                       "output": {"period": 0.001, "signals": 1, "energy": 0}},
    }
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    assert not result.tripped
    v_out = result.plant["output.u"][-1]
    v_in = result.plant["input.u"][-1]
    assert v_out - v_in == pytest.approx(2.0, abs=1e-5)
    assert result.plant["series.injection"][-1] == pytest.approx(2.0)
    assert result.plant["series.current"][-1] == pytest.approx(
        result.plant["series.source_current"][-1])


def test_injection_accepts_named_converter_electrical_ports(examples):
    tree = read_tree(examples / "buck-open-loop-example.pes")
    tree["elements"]["port_stimulus"] = {
        "type": "port_injection", "kind": "current",
        "bus": "converter.power", "amplitude": 0.01, "frequency": 100.0,
    }
    sim = peslite.Simulation(from_dict(tree))
    injector = sim.system.named_elements["port_stimulus"]
    assert injector.model.phases == 1
    assert injector.model.port.scale == 1.0
