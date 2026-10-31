"""PWM dead-time and compensation comparisons against motulator 0.21.0."""

from __future__ import annotations

import importlib.metadata
from time import perf_counter

import numpy as np
import pytest

pytest.importorskip(
    "motulator", reason="install the reference dependencies with peslite[reference]"
)
from motulator.common.control import InverterNonlinearity, PWM as MotulatorPWM
from motulator.common.model import CarrierComparison as MotulatorCarrier
from motulator.common.model import ZOH as MotulatorZOH
from motulator.common.utils import abc2complex as motulator_abc2complex

from peslite.components.converter import PWMAveragingModel, ThreePhaseBridge
from peslite.components.pwm import CarrierComparison, _DeadTimeSequence, ZOH
from peslite.control.blocks import abc2complex


MOTULATOR_VERSION = "0.21.0"
PERIOD = 100e-6
DEAD_TIME = 2e-6
pytestmark = [pytest.mark.full, pytest.mark.motulator]


def _record_difference(record_property, name, actual, expected):
    difference = np.abs(np.asarray(actual) - np.asarray(expected))
    record_property(f"{name}_mean_absolute_difference", float(np.mean(difference)))
    record_property(f"{name}_maximum_absolute_difference", float(np.max(difference)))


def _record_timing(record_property, peslite_seconds, motulator_seconds):
    record_property("peslite_evaluation_seconds", peslite_seconds)
    record_property("motulator_evaluation_seconds", motulator_seconds)
    record_property("evaluation_time_difference_seconds", peslite_seconds - motulator_seconds)


def _canonical(durations, states):
    """Merge adjacent intervals with the same effective leg states."""
    merged_durations = []
    merged_states = []
    for duration, state in zip(durations, states):
        if merged_states and np.array_equal(state, merged_states[-1]):
            merged_durations[-1] += duration
        else:
            merged_durations.append(duration)
            merged_states.append(np.asarray(state, dtype=float))
    return np.asarray(merged_durations), np.asarray(merged_states)


def _effective_states(gates, currents):
    """Resolve motulator's explicit blanking state through its current-directed diode law."""
    upper, blanked = gates[:, 0], gates[:, 1]
    return upper + 0.5 * blanked * (1.0 - np.sign(currents))


def _averaged_bridge(duties, currents, *, compensated=False):
    bridge = PWMAveragingModel(
        PERIOD, PERIOD, 0.0, 0.0, 0.0, PERIOD,
        ZOH(), ThreePhaseBridge(), dead_time=DEAD_TIME,
        dead_time_compensation=compensated,
    )
    bridge.inp.i_c = abc2complex(currents)
    bridge.reset(np.asarray(duties), on=True)
    return bridge


@pytest.mark.parametrize(("duties", "currents"), [
    ([0.25, 0.50, 0.75], [2.0, -1.0, -1.0]),
    ([0.01, 0.99, 0.50], [-1.0, 2.0, -1.0]),
    ([0.00, 1.00, 0.50], [-1.0, -1.0, 2.0]),
])
def test_switching_dead_time_matches_motulator(
        duties, currents, record_property):
    assert importlib.metadata.version("motulator") == MOTULATOR_VERSION
    duties, currents = np.asarray(duties), np.asarray(currents)

    modulator = CarrierComparison(1.0 / PERIOD, phase=0.5, min_interval=0.0)
    dead_time = _DeadTimeSequence(DEAD_TIME, 3, min_interval=0.0)
    # Align the preceding carrier cycle: motulator assumes its first command already
    # conducts, while PESLite also models the initial transition from blocked gates.
    warmup = modulator(-PERIOD, PERIOD, duties)
    dead_time(-PERIOD, PERIOD, warmup, currents, enabled=True)
    started = perf_counter()
    actual = dead_time(
        0.0, PERIOD, modulator(0.0, PERIOD, duties), currents, enabled=True,
    )
    peslite_seconds = perf_counter() - started

    reference = MotulatorCarrier(
        t_d=DEAD_TIME, delay=0, single_update=True, N=10_000,
    )
    reference(PERIOD, duties)
    started = perf_counter()
    expected_durations, gates = reference(PERIOD, duties)
    motulator_seconds = perf_counter() - started
    expected_states = _effective_states(gates, currents)

    actual_durations, actual_states = _canonical(actual.dt, actual.q_abc)
    expected_durations, expected_states = _canonical(expected_durations, expected_states)
    _record_difference(record_property, "interval_duration", actual_durations, expected_durations)
    _record_difference(record_property, "effective_leg_state", actual_states, expected_states)
    _record_timing(record_property, peslite_seconds, motulator_seconds)
    np.testing.assert_allclose(actual_durations, expected_durations, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(actual_states, expected_states)


@pytest.mark.parametrize(("duties", "currents"), [
    ([0.25, 0.50, 0.75], [2.0, -1.0, -1.0]),
    ([0.01, 0.99, 0.50], [-1.0, 2.0, -1.0]),
    ([0.00, 1.00, 0.50], [-1.0, -1.0, 2.0]),
])
def test_averaged_dead_time_matches_motulator(
        duties, currents, record_property):
    assert importlib.metadata.version("motulator") == MOTULATOR_VERSION
    duties, currents = np.asarray(duties), np.asarray(currents)
    bridge = _averaged_bridge(duties, currents)
    started = perf_counter()
    actual = bridge.start(0.0, (None, None), continued=False)
    peslite_seconds = perf_counter() - started

    reference = MotulatorZOH(t_d=DEAD_TIME, delay=0, single_update=True)
    started = perf_counter()
    durations, gates = reference(PERIOD, duties)
    motulator_seconds = perf_counter() - started
    effective = _effective_states(gates, currents)
    expected = motulator_abc2complex(np.average(effective, axis=0, weights=durations))

    _record_difference(record_property, "averaged_switching_state", actual, expected)
    _record_timing(record_property, peslite_seconds, motulator_seconds)
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-15)


@pytest.mark.parametrize(("voltage_reference", "currents"), [
    (130.0 + 50.0j, [2.0, -1.0, -1.0]),
    (-80.0 + 140.0j, [-1.0, 2.0, -1.0]),
    (500.0 + 0.0j, [2.0, -1.0, -1.0]),
])
def test_ideal_dead_time_compensation_matches_motulator(
        voltage_reference, currents, record_property):
    assert importlib.metadata.version("motulator") == MOTULATOR_VERSION
    u_dc = 650.0
    currents = np.asarray(currents)
    nonlinearity = InverterNonlinearity(t_d=DEAD_TIME)
    reference = MotulatorPWM(
        d_err=nonlinearity, feedforward=True, k_pred=0.0, single_update=True,
    )
    reference.get_realized_voltage(
        motulator_abc2complex(currents), u_dc, average=False,
    )
    duties = np.asarray(reference.duty_ratios(voltage_reference, u_dc))
    bridge = _averaged_bridge(duties, currents, compensated=True)

    started = perf_counter()
    actual = bridge.start(0.0, (None, None), continued=False)
    peslite_seconds = perf_counter() - started
    started = perf_counter()
    compensated, _ = reference.compute_output(PERIOD, voltage_reference, u_dc, 0.0)
    realized = np.asarray(compensated) - nonlinearity(
        currents, np.asarray(compensated), u_dc, PERIOD / 2.0,
    )
    expected = motulator_abc2complex(realized)
    motulator_seconds = perf_counter() - started

    _record_difference(record_property, "compensated_switching_state", actual, expected)
    _record_timing(record_property, peslite_seconds, motulator_seconds)
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-15)
