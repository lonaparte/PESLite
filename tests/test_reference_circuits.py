"""Open-loop bridge models against an independently propagated RLC circuit."""

import math

import numpy as np
import pytest
from scipy.linalg import expm

import peslite
from peslite.assembly.params import from_dict

F_SW, VDC = 20.0e3, 48.0
L_F, R_F, C_BUS, R_LOAD = 0.5e-3, 0.05, 100.0e-6, 2.0


def _case(bridge, model, duty, t_end=2.0e-3, period=1.0e-6):
    return {
        "base": {"s_base": 500.0, "v_ll_rms": 230.0, "f0": 50.0},
        "buses": {"output": {"phases": 1, "c": C_BUS}},
        "elements": {"load": {"type": "resistor", "bus": "output", "r": R_LOAD}},
        "units": {"converter": {
            "type": "dc_converter", "bus": "output",
            "filter": {"l_f": L_F, "r_f": R_F},
            "dclink": {"vdc_ref": VDC, "source": {"type": "voltage"}},
            "pwm": {"f_sw": F_SW},
            "bridge": {"type": bridge, "model": model},
            "ctrl": {"type": "open_loop", "computation": 0.0,
                     "loops": {"DutyModulation": {}}, "references": {"duty": duty}},
        }},
        "simulation": {"t_end": t_end,
                       "solver": {"dt": 2.5e-6, "linearisations": 0},
                       "output": {"period": period, "signals": True, "states": False}},
    }


def _leg(duty, time):
    phase = (time * F_SW) % 1.0
    return float(phase < 0.5 * duty or phase >= 1.0 - 0.5 * duty)


def _level(bridge, duty, time):
    return _leg(duty, time) if bridge == "half_bridge" else (
        _leg(duty, time) - _leg(1.0 - duty, time)
    )


def _edges(bridge, duty, start, stop):
    period = 1.0 / F_SW
    edges = {start}
    cycle = math.floor(start / period)
    while cycle * period < stop:
        duties = (duty,) if bridge == "half_bridge" else (duty, 1.0 - duty)
        for value in duties:
            for offset in (0.0, 0.5 * value * period, period - 0.5 * value * period):
                edge = cycle * period + offset
                if start < edge < stop:
                    edges.add(edge)
        cycle += 1
    return sorted(edges)


def _exact_response(drive, start, stop, samples, edges):
    matrix = np.array([
        [-R_F / L_F, -1.0 / L_F],
        [1.0 / C_BUS, -1.0 / (R_LOAD * C_BUS)],
    ])
    input_vector = np.array([1.0 / L_F, 0.0])
    inverse = np.linalg.inv(matrix)
    state = np.zeros(2)
    result = np.zeros((samples.size, 2))
    bounds = [edge for edge in edges if edge >= start] + [stop]
    for left, right in zip(bounds[:-1], bounds[1:]):
        forced = -inverse @ input_vector * drive(0.5 * (left + right))
        transition = expm(matrix * (right - left))
        for index in np.flatnonzero((samples >= left) & (samples <= right)):
            result[index] = expm(matrix * (samples[index] - left)) @ (state - forced) + forced
        state = transition @ (state - forced) + forced
    return result


@pytest.mark.parametrize("bridge", ["half_bridge", "full_bridge"])
@pytest.mark.parametrize("model", ["switching", "averaging"])
def test_open_loop_bridge_matches_the_exact_linear_response(bridge, model, tmp_path):
    duty = 0.37 if bridge == "half_bridge" else 0.81
    tree = _case(bridge, model, duty)
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    columns = result.columns()
    time = np.asarray(result.t)
    service = np.asarray(columns["converter.in_service"])
    start = 0.0 if model == "averaging" else time[np.argmax(service == 1.0)]
    if model == "averaging":
        average = duty if bridge == "half_bridge" else 2.0 * duty - 1.0
        drive, edges = lambda _time: VDC * average, [0.0]
    else:
        drive = lambda at: VDC * _level(bridge, duty, at)
        edges = _edges(bridge, duty, start, tree["simulation"]["t_end"])
    keep = time < tree["simulation"]["t_end"] - 1.0e-9
    expected = _exact_response(drive, start, tree["simulation"]["t_end"], time[keep], edges)
    current = np.asarray(columns["converter.i_c"])[keep]
    voltage = np.asarray(columns["output.u"])[keep]

    assert np.max(np.abs(current - expected[:, 0])) < 1.0e-6 * np.max(np.abs(expected[:, 0]))
    assert np.max(np.abs(voltage - expected[:, 1])) < 1.0e-6 * np.max(np.abs(expected[:, 1]))
