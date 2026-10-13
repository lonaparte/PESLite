"""Scalar SI converter loops on the shared control graph."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import peslite.control
from peslite.control import (
    DUTY_RATIO,
    LOOP_TYPES,
    SI_CURRENT,
    SI_VOLTAGE,
    CapacitorVoltageControl,
    ControlGraph,
    ControlInterface,
    InductorCurrentControl,
    OutputLimit,
    PI,
)


@dataclass(frozen=True)
class _References:
    voltage: float = 24.0


def _graph(*, continuous=False):
    period = 0.0001
    loops = {
        "voltage": CapacitorVoltageControl.Params(
            kp=0.18,
            ki=35.0,
            minimum=0.0,
            maximum=18.0,
            period=period,
        ),
        "current": InductorCurrentControl.Params(
            kp=0.025,
            ki=5.0,
            minimum=0.0,
            maximum=0.85,
            period=period,
        ),
    }
    ctrl = SimpleNamespace(
        loops=loops,
        references=_References(),
        period=period,
        connections={},
        outputs={},
    )
    unit = SimpleNamespace(
        ctrl=ctrl,
        pwm=SimpleNamespace(grid_offset=0.0),
        bridge=SimpleNamespace(model="averaging" if continuous else "pwm_averaging"),
    )
    interface = ControlInterface(
        MappingProxyType({
            "meas.current": SI_CURRENT,
            "meas.voltage": SI_VOLTAGE,
            "meas.vin": SI_VOLTAGE,
            "meas.duty_feedforward": DUTY_RATIO,
            "references.voltage": SI_VOLTAGE,
        }),
        MappingProxyType({"duty": DUTY_RATIO}),
    )
    connections = {
        "voltage.reference": "references.voltage",
        "voltage.voltage": "meas.voltage",
        "current.reference": "voltage.current_reference",
        "current.current": "meas.current",
        "current.feedforward": "meas.duty_feedforward",
    }
    return ControlGraph(
        unit,
        None,
        connections,
        {"duty": "current.duty"},
        interface=interface,
    )


def test_scalar_control_uses_the_shared_control_package():
    control = Path(peslite.control.__file__).parent

    assert not (control / "dc.py").exists()
    assert not hasattr(peslite.control, "make_scalar_controller")
    assert CapacitorVoltageControl.__module__ == "peslite.control.loops"
    assert InductorCurrentControl.__module__ == "peslite.control.loops"
    assert {"capacitor_voltage_control", "inductor_current_control"} <= set(LOOP_TYPES)
    assert LOOP_TYPES["dc_voltage_pi"].__name__ == "DCLinkVoltageLoop"


def test_scalar_control_graph_has_typed_si_ports_and_bounded_output():
    graph = _graph()
    measurement = SimpleNamespace(
        current=0.0, voltage=12.0, vin=48.0, duty_feedforward=0.25,
    )

    assert graph.interface.inputs["meas.current"] is SI_CURRENT
    assert graph.interface.inputs["meas.voltage"] is SI_VOLTAGE
    assert graph.interface.outputs["duty"] is DUTY_RATIO
    assert graph.sample(0.0, measurement) == {"voltage", "current"}
    assert 0.0 <= graph.output("duty", measurement) <= 0.85


def test_scalar_control_graph_runs_the_same_loops_continuously():
    graph = _graph(continuous=True)
    measurement = SimpleNamespace(
        current=0.0, voltage=12.0, vin=48.0, duty_feedforward=0.25,
    )

    derivatives = graph.flow(0.0, measurement)

    assert set(derivatives) == {"voltage.integral", "current.integral"}
    assert 0.0 <= graph.output("duty", measurement) <= 0.85


@pytest.mark.parametrize("feedforward", [0.0, 0.2, 0.25])
def test_inductor_current_control_accepts_topology_independent_feedforward(feedforward):
    graph = _graph()
    current = graph.nodes["current"]

    assert not hasattr(current.unit, "topology")
    assert current.equation(0.0, 0.0, feedforward)[0] == pytest.approx(feedforward)


def test_output_limit_tracks_sampled_and_continuous_pi_states():
    sampled = PI(2.0, 4.0, 0.1)
    requested = sampled(2.0, 0.0)
    before = sampled.integral
    limiter = OutputLimit(1.0, lower=0.0, pi=sampled)

    assert limiter.sample(0.0, requested) == 1.0
    assert limiter.saturated
    assert sampled.integral == pytest.approx(
        before + 0.1 * (1.0 - requested) / 2.0
    )

    continuous = PI(2.0, 4.0, None)
    requested = continuous(2.0, 0.0)
    limiter = OutputLimit(1.0, lower=0.0, pi=continuous)
    assert limiter.flow(requested) == 1.0
    assert continuous.derivative == pytest.approx(0.5)
