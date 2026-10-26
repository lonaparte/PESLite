"""Finite bridge conductance, port capacitance and the unchanged ideal specialization."""

import math

import numpy as np
import pytest

from peslite.components import FullBridge, HalfBridge, ThreePhaseBridge, make_bridge
from peslite.solver.integrators import FixedStepSolver
from peslite.solver.model import ConfigError, Model
from peslite.components.pwm import CarrierComparison
from test_converter_stages import _bridge_configuration


def _model(topology, mode="switching", *, capacitance=1e-4, resistance=0.5):
    cfg = _bridge_configuration(mode)
    cfg.bridge.parallel_capacitance = capacitance
    cfg.bridge.on_resistance = resistance
    bridge = make_bridge(cfg, topology=topology)
    bridge.enabled = True
    bridge.inp.u_dc = 48.0
    bridge.inp.q = topology.switching_state((1,) + (0,) * (len(topology.duty_names) - 1))
    return bridge, Model({"bridge": bridge}, {})


@pytest.mark.parametrize("topology", [HalfBridge(), FullBridge(), ThreePhaseBridge()])
@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_zero_impedance_keeps_the_original_state_and_equation_layout(topology, mode):
    bridge, model = _model(topology, mode, capacitance=0.0, resistance=0.0)
    assert model.n_states == 0
    assert not model.algebraic_loops
    assert not bridge.storage
    assert not hasattr(bridge, "capacitor")
    assert not hasattr(bridge, "on_resistance")


@pytest.mark.parametrize("topology", [HalfBridge(), FullBridge(), ThreePhaseBridge()])
def test_capacitive_switching_step_matches_the_analytic_rc_response(topology):
    capacitance, resistance = 1e-4, 0.5
    bridge, model = _model(topology, capacitance=capacitance, resistance=resistance)
    solver = FixedStepSolver(0.25e-6, "rk4")
    end = 100e-6
    y = solver(model.rhs, 0.0, end, model.get_initial_values()).y
    model.sync(end, y)
    expected = bridge.inp.q * bridge.inp.u_dc * (
        1.0 - math.exp(-end / (topology.series_devices * resistance * capacitance)))
    assert bridge.out.u_c == pytest.approx(expected, rel=1e-9, abs=1e-10)
    report = model.energy_balance(end, y)
    part, = report.subsystems
    assert abs(part.residual) < 1e-8
    assert part.dissipated > 0.0 and part.supplied == 0.0


@pytest.mark.parametrize("topology", [HalfBridge(), FullBridge(), ThreePhaseBridge()])
@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
def test_capacitor_and_device_energy_balance_for_every_form(topology, mode):
    bridge, model = _model(topology, mode)
    for enabled in (False, True):
        bridge.enabled = enabled
        for voltage in (0.0, 25.0, -130.0):
            bridge.state.u_C = voltage + (12.0j if isinstance(topology.zero, complex) else 0.0)
            bridge.inp.i_c = 1.3 + (0.7j if isinstance(topology.zero, complex) else 0.0)
            y = model.get_initial_values()
            report = model.energy_balance(0.0, y)
            part, = report.subsystems
            assert part.dissipated >= 0.0
            assert abs(part.residual) < 1e-7 * max(1.0, report.scale)


def test_controlled_voltage_reversal_has_finite_current_and_continuous_capacitor_voltage():
    bridge, model = _model(FullBridge())
    bridge.state.u_C = 48.0
    bridge.inp.q = -1.0
    y = model.get_initial_values()
    model.sync(0.0, y)
    assert bridge.out.u_c == 48.0
    assert bridge.out.i_device == pytest.approx(-96.0)
    assert model.rhs(0.0, y)[0] == pytest.approx(-96.0 / 1e-4)


def test_diode_leg_blocks_device_current_without_clamping_its_capacitor_port_current():
    bridge, model = _model(HalfBridge(lower_controlled=False))
    bridge.inp.q = 0.0
    bridge.state.u_C = 10.0
    bridge.inp.i_c = -2.0
    y = model.get_initial_values()
    model.sync(0.0, y)
    assert not bridge.diode_blocks_reverse_current()
    assert bridge.out.i_device == 0.0
    assert model.rhs(0.0, y)[0] == pytest.approx(2.0 / 1e-4)


def test_full_bridge_diodes_do_not_become_gated_devices_when_the_unit_is_enabled():
    topology = FullBridge(upper_controlled=False, lower_controlled=False)
    bridge, model = _model(topology)
    bridge.state.u_C = 60.0
    model.sync(0.0, model.get_initial_values())
    assert bridge.enabled
    assert bridge.out.i_device == pytest.approx(-12.0)
    assert bridge.out.i_dc == pytest.approx(-12.0)
    assert bridge.dissipated_power() == pytest.approx(144.0)


@pytest.mark.parametrize("voltage", [-200.0, -25.0, 0.0, 25.0, 200.0])
def test_full_bridge_zero_vector_eliminates_the_common_voltage_without_a_loop(voltage):
    bridge, model = _model(FullBridge())
    bridge.inp.q = 0.0
    bridge.state.u_C = voltage
    y = model.get_initial_values()
    part, = model.energy_balance(0.0, y).subsystems
    assert not model.algebraic_loops
    assert abs(part.residual) < 1e-8
    assert part.dissipated >= 0.0


@pytest.mark.parametrize("topology", [FullBridge(), ThreePhaseBridge()])
@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
def test_averaged_zero_vector_does_not_invent_capacitor_charging_loss(topology, mode):
    bridge, model = _model(topology, mode)
    bridge.inp.q = topology.zero
    model.sync(0.0, model.get_initial_values())
    assert bridge.out.i_device == pytest.approx(0.0, abs=1e-10)
    assert bridge.out.i_dc == pytest.approx(0.0, abs=1e-10)
    assert bridge.dissipated_power() == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("mode", ["pwm_averaging", "averaging"])
def test_half_bridge_averages_actual_charging_current_and_loss(mode):
    bridge, model = _model(HalfBridge(), mode)
    bridge.inp.q = 0.5
    bridge.state.u_C = 24.0
    model.sync(0.0, model.get_initial_values())
    assert bridge.out.i_device == pytest.approx(0.0)
    assert bridge.out.i_dc == pytest.approx(24.0)
    assert bridge.dissipated_power() == pytest.approx(1152.0)


@pytest.mark.parametrize("width", [1, 2, 3])
def test_averaged_state_fractions_match_the_existing_exact_carrier(width):
    for duties in (np.linspace(0.0, 1.0, width), np.full(width, 0.5),
                   np.random.default_rng(41).uniform(size=width)):
        exact = CarrierComparison(10_000.0)(0.0, 1e-4, duties)
        totals = {}
        for dt, state in zip(exact.dt, exact.q_abc):
            key = tuple(state)
            totals[key] = totals.get(key, 0.0) + dt / 1e-4
        averaged = {}
        for weight, state in CarrierComparison.state_fractions(duties):
            if weight > 1e-12:
                averaged[state] = averaged.get(state, 0.0) + weight
        assert averaged == pytest.approx(totals)


@pytest.mark.parametrize("mode", ["switching", "pwm_averaging", "averaging"])
@pytest.mark.parametrize("resistance", [1.0, 75.0])
def test_three_phase_device_equations_leave_one_dc_port_feedback(mode, resistance):
    from peslite.components.converter import DCLink, DCVoltageSource

    bridge, _ = _model(ThreePhaseBridge(), mode)
    link = DCLink(None, DCVoltageSource(48.0, resistance))
    model = Model({"bridge": bridge, "link": link}, {
        (bridge, "u_dc"): (link, "u_dc"), (link, "i_dc"): (bridge, "i_dc")})
    model.sync(0.0, model.get_initial_values())
    expected = 48.0 / (1.0 + resistance * (2.0 / 3.0) / bridge.on_resistance)
    assert link.out.u_dc == pytest.approx(expected, rel=1e-8)
    assert [len(loop.tears) for loop in model.algebraic_loops] == [1]


def test_floating_three_phase_reference_matches_an_independent_nodal_solution():
    from itertools import product
    from scipy.optimize import brentq
    from peslite.control.blocks import phases

    bridge, _ = _model(ThreePhaseBridge())
    voltages = np.random.default_rng(15).normal(size=(10, 2)) * 200.0
    for enabled in (False, True):
        bridge.enabled = enabled
        for real, imag in voltages:
            bridge.state.u_C = complex(real, imag)
            offsets = phases(bridge.state.u_C)
            for gates in product((0, 1), repeat=3):
                def currents(reference):
                    result = []
                    for voltage, gate in zip(offsets, gates):
                        pole = reference + voltage
                        upper = 48.0 - pole
                        lower = -pole
                        result.extend((upper if enabled and gate else min(upper, 0.0),
                                       lower if enabled and not gate else max(lower, 0.0)))
                    return np.array(result)

                bound = max(map(abs, offsets)) + 100.0
                expected = brentq(lambda value: sum(currents(value)), -bound, bound)
                actual = bridge._pole_reference(gates)
                np.testing.assert_allclose(currents(actual), currents(expected), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("topology", [HalfBridge(), FullBridge(), ThreePhaseBridge()])
def test_resistance_without_capacitance_uses_the_same_conduction_path(topology):
    bridge, model = _model(topology, capacitance=0.0)
    bridge.inp.i_c = 2.0 + 0j if isinstance(topology.zero, complex) else 2.0
    model.sync(0.0, model.get_initial_values())
    expected = bridge.inp.q * 48.0 - topology.series_devices * 0.5 * bridge.inp.i_c
    assert bridge.out.u_c == pytest.approx(expected)
    assert bridge.dissipated_power() == pytest.approx(topology.power_scale * topology.series_devices * 2.0)


@pytest.mark.parametrize("topology", [HalfBridge(), FullBridge(), ThreePhaseBridge()])
def test_nonzero_capacitance_requires_explicit_positive_device_resistance(topology):
    with pytest.raises(ConfigError, match="positive on_resistance"):
        _model(topology, resistance=0.0)
