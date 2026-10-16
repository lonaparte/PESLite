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
    DCVoltageControl,
    ControlGraph,
    ControlInterface,
    InductorCurrentControl,
    PowerControl,
    PhasePowerControl,
    ResonantModulation,
    OutputLimit,
    PI,
)


@dataclass(frozen=True)
class _References:
    voltage: float = 24.0


def _graph(*, continuous=False):
    period = 0.0001
    loops = {
        "voltage": DCVoltageControl.Params(
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
    assert not hasattr(peslite.control, "ScalarController")
    assert not hasattr(peslite.control, "ResonantController")
    assert DCVoltageControl.__module__ == "peslite.control.loops"
    assert InductorCurrentControl.__module__ == "peslite.control.loops"
    assert {"dc_voltage_control", "inductor_current_control"} <= set(LOOP_TYPES)
    assert LOOP_TYPES["dc_voltage_control"] is DCVoltageControl
    assert LOOP_TYPES["resonant_modulation"] is ResonantModulation


def test_control_profiles_declare_one_static_interface_contract():
    from peslite.control.profiles import control_type

    for name in ("gfl", "gfm", "custom", "dc", "resonant"):
        assert isinstance(control_type(name).interface, ControlInterface)


def test_direct_measured_controller_output_declares_its_input_dependency():
    import peslite
    from conftest import EXAMPLES

    params = peslite.load(EXAMPLES / "buck-example.pes").replace(**{
        "units.converter.bridge.model": "averaging",
        "units.converter.ctrl.outputs": {"duty": "meas.duty_feedforward"},
    })
    controller = peslite.System(params).units["converter"].ctrl
    assert not controller.graph.nodes
    assert controller.outputs_need_inputs


@pytest.mark.parametrize(
    "polarity,reference,voltage,error",
    [(1.0, 2.0, 1.0, 1.0), (-1.0, 1.0, 2.0, -1.0)],
)
def test_dc_voltage_control_always_uses_reference_minus_voltage(
        polarity, reference, voltage, error):
    unit = SimpleNamespace(bridge=SimpleNamespace(model="pwm_averaging"))
    startup = SimpleNamespace(active=True, value=1.0)
    loop = DCVoltageControl(
        DCVoltageControl.Params(
            kp=2.0, ki=0.0, minimum=-10.0, maximum=10.0,
            output_polarity=polarity, period=0.1,
        ),
        unit,
        startup,
    )

    output = loop.sample(0.0, {"reference": reference, "voltage": voltage})

    assert loop.error == error
    assert output == {"current_reference": pytest.approx(2.0)}
    assert tuple(loop.outputs) == ("current_reference",)


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


@pytest.mark.parametrize("loop_class,command", [
    (PowerControl, "modulation"), (PhasePowerControl, "phase"),
])
@pytest.mark.parametrize("continuous", [False, True])
def test_power_feedback_reuses_the_same_equation_and_dynamic_blocks(
        loop_class, command, continuous):
    unit = SimpleNamespace(bridge=SimpleNamespace(
        model="averaging" if continuous else "pwm_averaging",
    ))
    loop = loop_class(
        loop_class.Params(kp=0.01, ki=0.0, minimum=-1.0, maximum=1.0, period=0.001),
        unit, SimpleNamespace(active=True, value=1.0),
    )
    inputs = {"reference": 150.0, "power": 100.0,
              "feedforward": 0.1}
    output = loop.flow(0.0, inputs)[0] if continuous else loop.sample(0.0, inputs)

    assert isinstance(loop.pi, PI)
    assert isinstance(loop.limit, OutputLimit)
    assert output == {command: pytest.approx(0.6), "power": pytest.approx(100.0)}
    assert loop.get_state().keys() == {"integral"}
    assert PowerControl.equation is PhasePowerControl.equation


@pytest.mark.parametrize("loop_class", [PowerControl, PhasePowerControl])
def test_power_feedback_filters_signed_instantaneous_product_and_restores_state(loop_class):
    from peslite.control.loops import PowerFilterParams
    from peslite.control import Filter

    cfg = loop_class.Params(
        kp=0.001, ki=0.1, minimum=-1.0, maximum=1.0, period=0.001,
        power_filter=PowerFilterParams(enable=True, bandwidth=100.0),
    )
    unit = SimpleNamespace(bridge=SimpleNamespace(model="pwm_averaging"))
    startup = SimpleNamespace(active=True, value=1.0)
    loop = loop_class(cfg, unit, startup)
    inputs = {"reference": -100.0, "power": -100.0,
              "feedforward": 0.0}
    for n in range(10):
        output = loop.sample(n * cfg.period, inputs)
    assert isinstance(loop.power_filter, Filter)
    assert -100.0 < output["power"] < -99.0
    restored = loop_class(cfg, unit, startup)
    restored.set_state(loop.get_state())
    assert restored.sample(0.01, inputs) == loop.sample(0.01, inputs)


@pytest.mark.parametrize("wired", [False, True])
def test_declared_power_loop_outside_the_active_graph_is_not_constructed(wired):
    import peslite
    from conftest import EXAMPLES

    params = peslite.load(EXAMPLES / "wpt-example.pes").replace(**{
        "units.charger.primary.ctrl.loops.PowerControl": {
            "kp": 0.001, "ki": 0.1, "minimum": 0.0, "maximum": 1.0,
        },
    })
    if wired:
        params = params.replace(**{
            "units.charger.primary.ctrl.connections": {
                "PowerControl.reference": "references.power",
                "PowerControl.power": "meas.power",
                "PowerControl.feedforward": "references.modulation",
            },
        })
    unit = peslite.System(params).units["charger.primary"]
    controller = unit.ctrl
    assert set(controller.graph.nodes) == {"ResonantModulation"}
    assert "meas.power" not in controller.graph.interface.inputs
    assert "power" not in unit.ports.readers


@pytest.mark.parametrize("lower,upper", [(-1.2, 0.1), (0.0, 1.3), (0.1, 1.2)])
def test_scalar_output_limits_keep_the_original_positive_upper_contract(lower, upper):
    limiter = OutputLimit(upper, lower=lower)
    for value in (lower - 1.0, lower, (lower + upper) / 2.0, upper, upper + 1.0):
        expected = min(upper, max(lower, value))
        assert limiter.sample(0.0, value) == expected
        assert limiter.flow(value) == expected


@pytest.mark.parametrize("lower,upper", [(-1.0, float("inf")), (float("nan"), 1.0),
                                          (-1.2, 0.0), (-1.2, -0.1), (1.0, 1.0)])
def test_scalar_output_limits_reject_invalid_intervals(lower, upper):
    from peslite.assembly.params import ConfigError

    with pytest.raises(ConfigError, match="limit"):
        OutputLimit(upper, lower=lower)


@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
@pytest.mark.parametrize("minimum,maximum", [(-1.2, 0.1), (-1.2, 0.0), (-1.2, -0.1),
                                             (0.2, 0.8)])
def test_phase_constraint_translates_the_coordinate_without_wrapping(mode, minimum, maximum):
    unit = SimpleNamespace(bridge=SimpleNamespace(model=mode))
    loop = PhasePowerControl(
        PhasePowerControl.Params(kp=0.01, ki=0.1, minimum=minimum, maximum=maximum,
                                 period=0.001),
        unit, SimpleNamespace(active=True, value=1.0),
    )
    assert loop.limit.lower == 0.0
    assert loop.limit.limit == pytest.approx(maximum - minimum)
    received = []
    apply = loop.limit._apply

    def observe(value):
        received.append(value)
        return apply(value)

    loop.limit._apply = observe
    for phase in (minimum - 7.0, minimum, (minimum + maximum) / 2.0, maximum, maximum + 7.0):
        result = loop.constrain(phase)
        assert received[-1] == pytest.approx(phase - minimum)
        assert result == pytest.approx(min(maximum, max(minimum, phase)))
        if minimum <= phase <= maximum:
            assert received[-1] >= 0.0
    # Crossing a whole turn remains a saturation, never a modulo bypass.
    assert loop.constrain(maximum + 2.0 * 3.141592653589793) == pytest.approx(maximum)


@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
@pytest.mark.parametrize("reference", [-1000.0, 1000.0])
def test_phase_coordinate_translation_preserves_pi_tracking(mode, reference):
    unit = SimpleNamespace(bridge=SimpleNamespace(model=mode))
    loop = PhasePowerControl(
        PhasePowerControl.Params(kp=0.01, ki=0.1, minimum=-1.2, maximum=0.1,
                                 period=0.001),
        unit, SimpleNamespace(active=True, value=1.0),
    )
    inputs = {"reference": reference, "power": 0.0, "feedforward": -0.3}
    requested = -0.3 + loop.pi.kp * reference
    if mode != "averaging":
        requested += loop.pi.ki * 0.5 * loop.cfg.period * reference
    expected = min(0.1, max(-1.2, requested))
    result = loop.flow(0.0, inputs)[0] if mode == "averaging" else loop.sample(0.0, inputs)
    assert result["phase"] == pytest.approx(expected)
    assert loop.limit.saturated
    correction = (expected - requested) / loop.pi.kp
    if mode == "averaging":
        assert loop.pi.derivative == pytest.approx(reference + correction)
    else:
        assert loop.pi.integral == pytest.approx(loop.cfg.period * (reference + correction))


def test_converter_control_entry_points_share_explicit_port_power():
    import peslite
    from conftest import EXAMPLES
    from peslite.control import ConverterMeasurement

    params = peslite.load(EXAMPLES / "wpt-ss-example.pes")
    controllers = [peslite.System(params).units["charger.primary"].ctrl for _ in range(2)]
    for controller in controllers:
        controller.command(True)
    direct = controllers[0].sample(0.0, 5.0, 48.0, 24.0, power=0.0)
    protocol = controllers[1](0.0, ConverterMeasurement(
        current=5.0, voltage=48.0, vin=24.0, power=0.0,
    ))
    assert direct == protocol
    assert direct.log["power"] == 0.0
    assert controllers[0].control_inputs(5.0, 48.0, 24.0, power=-120.0)["meas.power"] == -120.0
    for measurement in (ConverterMeasurement(current=5.0, voltage=48.0, vin=24.0),
                        SimpleNamespace(current=5.0, voltage=48.0, vin=24.0)):
        with pytest.raises(ValueError, match="electrical-port power"):
            controllers[0](0.0, measurement)
    with pytest.raises(ValueError, match="electrical-port power"):
        controllers[0].sample(0.0, 5.0, 48.0, 24.0)
    open_loop = peslite.System(peslite.load(EXAMPLES / "wpt-example.pes")).units[
        "charger.primary"
    ].ctrl
    open_loop(0.0, SimpleNamespace(current=5.0, voltage=48.0, vin=24.0))
