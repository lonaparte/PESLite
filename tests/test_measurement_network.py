"""Shared named-port loads, measurement bindings and RMS protection."""

import copy
import math
from unittest.mock import Mock

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import ConfigError, from_dict, read_tree
from peslite.components import ADC, MeasurementPorts


def load_case(model="pwm_averaging", link="ss", rms=True):
    tree = read_tree(EXAMPLES / ("wpt-example.pes" if link == "ss" else "wpt-lcc-example.pes"))
    charger = tree["units"]["charger"]
    for side, window in (("primary", 12e-6), ("secondary", 15e-6)):
        unit = charger[side]
        unit["bridge"] = {"model": model}
        unit["meas"] = {"rms": {"current": window}} if rms else {}
    tree["elements"] = {
        "load": {"type": "resistor", "bus": "charger.secondary.power", "r": 2.0, "series": True},
    }
    tree["events"] = {}
    tree["simulation"].update(t_end=30e-6)
    tree["simulation"]["solver"].update(dt=50e-9, linearisations=0)
    tree["simulation"]["output"].update(period=1e-6, states=False, signals=False, energy=False)
    return tree


@pytest.mark.parametrize("link", ["ss", "lcc"])
def test_series_resistor_matches_physical_series_loss_without_adding_states(link):
    explicit = load_case(link=link, rms=False)
    lumped = copy.deepcopy(explicit)
    lumped.pop("elements")
    resistance = "secondary_resistance" if link == "ss" else "secondary_compensation_resistance"
    lumped["units"]["charger"]["link"][resistance] += 2.0
    a, b = (peslite.Simulation(from_dict(tree)) for tree in (explicit, lumped))
    assert a.system.model.state_labels() == b.system.model.state_labels()
    assert not a.energy_problems
    # Compare the actual physical differential equations, including nonzero stored energy.
    y = np.linspace(0.01, 0.1, len(a.system.model.get_initial_values()))
    np.testing.assert_allclose(a.system.model.rhs(0.0, y), b.system.model.rhs(0.0, y),
                               rtol=1e-12, atol=1e-8)
    a.run(out_dir=None)
    load = a.system.named_elements["load"]
    assert load.resistor.out.power == pytest.approx(2.0 * load.resistor.out.i ** 2)


@pytest.mark.parametrize("endpoint", ["charger", "charger.primary", "charger.primary.dc"])
def test_shunt_load_resolves_unit_default_and_named_ports(endpoint):
    tree = load_case(rms=False)
    tree["elements"]["load"].update(bus=endpoint, series=False)
    sim = peslite.Simulation(from_dict(tree))
    sim.system.model.sync(0.0, sim.system.model.get_initial_values())
    load = sim.system.named_elements["load"]
    primary = sim.units["charger.primary"]
    assert load.resistor.inp.u == primary.dclink.out.u_dc
    assert load.resistor.out.i == pytest.approx(primary.dclink.out.u_dc / 2.0)


def test_series_elements_share_the_same_generic_insertion_path_in_declaration_order():
    tree = load_case(rms=False)
    tree["elements"]["extra"] = {
        "type": "resistor", "bus": "charger.secondary.power", "r": 3.0, "series": True,
    }
    sim = peslite.Simulation(from_dict(tree))
    sim.run(out_dir=None)
    first = sim.system.named_elements["load"].resistor
    second = sim.system.named_elements["extra"].resistor
    np.testing.assert_allclose(first.out.i, second.out.i)
    assert second.inp.u == first.out.u
    assert second.out.power == pytest.approx(first.out.power * 1.5)


@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_load_power_drives_the_existing_graph_and_load_step_keeps_state(model):
    tree = load_case(model, link="lcc")
    tree["units"]["charger"]["secondary"]["meas"].update(
        sources={"power": "load.power"}, average=["power"],
    )
    tree["events"] = {"step": {"type": "set", "t": 15e-6, "set": {"elements.load.r": 3.0}}}
    p = from_dict(tree)
    sim = peslite.Simulation(p)
    unit = sim.units["charger.secondary"]
    before = sim.system.model.get_initial_values().copy()
    sim.system.apply(p.changes[0])
    np.testing.assert_array_equal(before, sim.system.model.get_initial_values())
    assert sim.system.named_elements["load"].resistor.resistance == 3.0
    sim = peslite.Simulation(p)
    result = sim.run(out_dir=None)
    unit = sim.units["charger.secondary"]
    load = sim.system.named_elements["load"].resistor
    assert unit.ports.value("power") == load.out.power
    assert type(unit.ctrl).__name__ == "GraphController"
    assert result.summary["ph_verdict"] == "port-hamiltonian"
    assert unit.adc.rms["current"].value > 0.0


@pytest.mark.parametrize("model", ["pwm_averaging", "averaging"])
def test_named_port_power_observer_is_nonloading_and_native_graph_compatible(model):
    tree = load_case(model, link="lcc", rms=False)
    unit = tree["units"]["charger"]["secondary"]
    unit["meas"] = {"sources": {"power": "charger.primary.dc.power"}}
    sim = peslite.Simulation(from_dict(tree))
    sim.system.model.sync(0.0, sim.system.model.get_initial_values())
    primary = sim.units["charger.primary"]
    assert sim.units["charger.secondary"].ports.value("power") == pytest.approx(
        -primary.dclink.terminal.power()
    )
    assert not sim.energy_problems
    sim.run(out_dir=None)


def test_rms_integrates_square_before_sampling_and_preserves_window_state():
    value = [0.0]
    adc = ADC(MeasurementPorts({"current": lambda: value[0]}), 0.1, 0.1,
              rms={"current": 1.0})
    adc.seed()
    adc.sample(0.0)
    for index in range(1, 10001):
        t = index / 10000
        value[0] = 3.0 * np.sin(2.0 * np.pi * t)
        adc.accumulate(0.0001)
        if index % 1000 == 0:
            adc.sample(t)
            if index < 10000:
                assert adc.rms["current"].value is None
    assert adc.rms["current"].value == pytest.approx(3.0 / np.sqrt(2.0), rel=1e-12)
    state = adc.get_state()
    restored = ADC(adc.ports, 0.1, 0.1, rms={"current": 1.0})
    restored.set_state(state)
    assert restored.get_state() == state
    for instance in (adc, restored):
        instance.seed()
        instance.accumulate(0.1)
        instance.sample(1.1)
    assert restored.get_state() == adc.get_state()


def test_rms_value_is_cached_until_the_window_changes_or_state_is_restored(monkeypatch):
    sqrt = Mock(wraps=math.sqrt)
    monkeypatch.setattr("peslite.components.adc.math.sqrt", sqrt)
    adc = ADC(MeasurementPorts({"current": lambda: 2.0}), 1.0, 1.0,
              rms={"current": 2.0})
    window = adc.rms["current"]
    adc.seed()
    adc.accumulate(1.0)
    adc.sample_rms()
    assert window.value is None
    assert sqrt.call_count == 0
    adc.accumulate(1.0)
    adc.sample_rms()
    assert sqrt.call_count == 0  # Do not calculate an unused statistic.
    assert window.value == window.value == 2.0
    assert sqrt.call_count == 1
    adc.seed()
    adc.accumulate(0.5)
    assert window.value == 2.0
    assert sqrt.call_count == 1
    adc.accumulate(0.5)
    adc.sample_rms()
    assert window.value == 2.0
    assert sqrt.call_count == 2
    state = window.get_state()
    state["total"] = 18.0
    window.set_state(state)
    assert window.value == 3.0
    assert sqrt.call_count == 3
    assert window.get_state() == state


def test_mean_and_rms_share_one_bound_read_per_interval_and_reseed_discontinuities():
    value = [2.0]
    readers = {"current": lambda: 0.0}
    adc = ADC(MeasurementPorts(readers), 1.0, 1.0, length=1.0,
              channels={"current": 0.0}, rms={"current": 1.0})
    read = Mock(side_effect=lambda: value[0])
    readers["current"] = read  # Assembly may replace a binding after creating the ADC.
    adc.seed()
    value[0] = 4.0
    adc.accumulate(0.5)
    value[0] = 10.0
    adc.seed()
    value[0] = 12.0
    adc.accumulate(0.5)
    assert read.call_count == 4
    sample = adc.sample(1.0)
    assert sample.values["current"] == 7.0
    assert adc.rms["current"].value == pytest.approx(math.sqrt(66.0))


def test_off_grid_rms_sampling_keeps_the_pending_integral_and_window():
    def run(events):
        tree = load_case("averaging")
        tree["units"]["charger"]["primary"]["protection"] = {
            "overvoltage": {"enable": True, "limit_pu": 1e-6}, "hold": 1.0,
        }
        tree["events"] = events
        sim = peslite.Simulation(from_dict(tree))
        sim.run(out_dir=None)
        unit = sim.units["charger.primary"]
        window = unit.adc.rms["current"].get_state()
        return unit.protection.get_state(), (window["count"], window["index"])

    plain = run({})
    # An off-clock file event and the other unit's samples neither advance the hold timers
    # nor close an RMS period of this continuous unit.
    off_grid = run({"off": {"type": "set", "t": 10.5e-6,
                            "set": {"units.charger.primary.protection.hold": 1.0}}})
    assert plain[0]["hold_ov"] == pytest.approx(30e-6)  # the 1 us clock from 1 us to 30 us
    assert off_grid == plain


@pytest.mark.parametrize("rms", [True, False])
@pytest.mark.parametrize("link", ["ss", "lcc"])
def test_continuous_rms_seeds_actual_initial_outputs_not_structure_probe_values(link, rms,
                                                                                monkeypatch):
    initial, protected = {}, {}
    original, check_fast = ADC.seed, peslite.Protection.check_fast

    def seed(adc, t=None):
        initial.setdefault(id(adc), adc.ports.read())
        original(adc, t)

    def first_check(protection, t, currents):
        protected.setdefault(id(protection), tuple(currents))
        return check_fast(protection, t, currents)

    monkeypatch.setattr(ADC, "seed", seed)
    monkeypatch.setattr(peslite.Protection, "check_fast", first_check)
    sim = peslite.Simulation(from_dict(load_case("averaging", link=link, rms=rms)))
    sim.run(out_dir=None)
    for unit in sim.units.values():
        if rms:
            assert initial[id(unit.adc)]["current"] == 0.0
        # The first protection check at t0 reads the same refreshed outputs, with or without an ADC.
        assert protected[id(unit.protection)] == (0.0,)


@pytest.mark.parametrize("side", ["primary", "secondary"])
@pytest.mark.parametrize("model", ["switching", "pwm_averaging", "averaging"])
def test_each_side_rms_uses_the_common_trip_latch_after_a_full_window(side, model):
    tree = load_case(model)
    tree["units"]["charger"][side]["protection"] = {
        "rms_current": {"enable": True, "limit_pu": 0.001}, "hold": 0.0,
    }
    sim = peslite.Simulation(from_dict(tree))
    result = sim.run(out_dir=None)
    unit = sim.units[f"charger.{side}"]
    window = tree["units"]["charger"][side]["meas"]["rms"]["current"]
    assert result.summary["t_stop"] >= window - 1e-12
    assert unit.protection.trip.cause == "rms_current"
    assert not unit.gates and not unit.breaker
    assert "RMS_OVERCURRENT" in unit.protection.stats.alarms


def test_unused_rms_has_no_window_allocation_and_invalid_sources_fail_at_assembly():
    tree = load_case(rms=False)
    sim = peslite.Simulation(from_dict(tree))
    assert all(not unit.adc.rms for unit in sim.units.values())
    assert all(unit.protection._rocof is None for unit in sim.units.values())
    tree["units"]["charger"]["primary"]["meas"] = {"sources": {"power": "missing.power"}}
    with pytest.raises(ConfigError, match="unknown physical output"):
        peslite.Simulation(from_dict(tree))


def test_instantaneous_measurement_has_no_duplicate_raw_conversion(monkeypatch):
    import peslite.assembly.unit as unit_module

    sim = peslite.Simulation(from_dict(load_case()))
    sim.system.model.sync(0.0, sim.system.model.get_initial_values())
    constructor = Mock(wraps=unit_module.ConverterMeasurement)
    monkeypatch.setattr(unit_module, "ConverterMeasurement", constructor)
    sample = sim.units["charger.primary"]._instantaneous_measurement(0.0)
    assert constructor.call_count == 1
    assert sample.samples == ()


def test_continuous_rms_hold_uses_its_own_existing_protection_clock():
    tree = load_case()
    primary = tree["units"]["charger"]["primary"]
    primary["bridge"]["model"] = "averaging"
    primary["protection"] = {
        "rms_current": {"enable": True, "limit_pu": 0.001}, "hold": 4.5e-6,
    }
    sim = peslite.Simulation(from_dict(tree))
    sim.run(out_dir=None)
    # The secondary's 1.25 us interrupts must not advance the primary's 1 us hold timer.
    assert sim.units["charger.primary"].protection.trip.t == pytest.approx(16e-6)


def test_rms_protection_can_be_enabled_by_an_existing_set_event_without_new_states():
    tree = load_case()
    tree["events"] = {"arm": {"type": "set", "t": 16e-6, "set": {
        "units.charger.primary.protection.rms_current.enable": True,
        "units.charger.primary.protection.rms_current.limit_pu": 0.001,
        "units.charger.primary.protection.hold": 0.0,
    }}}
    sim = peslite.Simulation(from_dict(tree))
    states = sim.state_names()
    sim.run(out_dir=None)
    assert sim.state_names() == states
    assert sim.units["charger.primary"].protection.trip.t == pytest.approx(16e-6)


def test_measurement_windows_and_series_connections_cannot_be_retuned_as_values():
    tree = load_case()
    tree["events"] = {"bad": {"type": "set", "t": 10e-6, "set": {"elements.load.series": False}}}
    with pytest.raises(ConfigError, match="cannot change during a run"):
        from_dict(tree)
    tree["events"] = {}
    tree["units"]["charger"]["primary"]["meas"]["rms"]["current"] = 0.5e-6
    with pytest.raises(ConfigError, match="whole sample periods"):
        from_dict(tree)
