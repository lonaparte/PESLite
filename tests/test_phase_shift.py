"""Square-wave phase-shift modulation: modulator edges, register path, control and the circuit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from peslite.components.pwm import PhaseShiftModulator, PulseSequence, phase_shift_level

from phase_shift_reference import level as reference_level, waveform

pytestmark = pytest.mark.full


def _phase_modulator(frequency, update):
    modulator = PhaseShiftModulator(frequency)
    return PulseSequence(modulator, 2) if update == "pulse" else modulator


def _levels(modulator, loads, t_end):
    """Run the modulator over consecutive loads; return its edges [(t, level)]."""
    edges = []
    for k, (t_l, registers) in enumerate(loads):
        t_next = loads[k + 1][0] if k + 1 < len(loads) else t_end
        seq = modulator(t_l, t_next - t_l, np.asarray(registers, dtype=float))
        t = t_l
        for dt, legs in zip(seq.dt, seq.q_abc):
            value = int(round(legs[0] - legs[1]))
            if not edges or edges[-1][1] != value:
                edges.append((t, value))
            t += dt
        assert t == pytest.approx(t_next, rel=0, abs=1e-15)
    return edges


def _sample(edges, times):
    starts = np.array([t for t, _ in edges])
    values = np.array([v for _, v in edges])
    index = np.searchsorted(starts, times, side="right") - 1
    return values[index]


def _compare(edges, times, expected, guard):
    got = _sample(edges, times)
    near = np.zeros(times.size, dtype=bool)
    switching = [t for (t, v), (_t0, v0) in zip(edges[1:], edges[:-1]) if v != v0]
    switching += list(times[1:][np.diff(expected) != 0])
    for t in switching:
        near |= np.abs(times - t) <= guard
    mismatch = (got != expected) & ~near
    return int(mismatch.sum()), int((~near).sum())


def test_level_matches_reference_definition():
    rng = np.random.default_rng(1)
    for _ in range(20000):
        mode = int(rng.choice([0, 1, 3, 4]))
        width = float(rng.uniform(-0.2, 1.2))
        y = float(rng.uniform(-20.0, 20.0))
        assert phase_shift_level(min(1.0, max(0.0, width)), mode, y) == reference_level(mode, width, y)


@pytest.mark.parametrize("pulse", [False, True])
def test_random_register_sequences_match_time_stepped_reference(pulse):
    rng = np.random.default_rng(7 if pulse else 3)
    f_nominal = 85e3
    load = 0.5 / f_nominal                  # double update
    mismatches = checked = 0
    for case in range(8):
        modulator = _phase_modulator(f_nominal, "pulse" if pulse else "interval")
        loads, t = [], 0.0
        registers = [0.8, 0.0, f_nominal, 0]
        for k in range(40):
            if k == 0 or rng.random() < 0.35:
                registers = [float(rng.choice([rng.uniform(0.0, 1.0), 1.0, 0.0],
                                              p=[0.8, 0.15, 0.05])),
                             float(rng.uniform(-math.pi, math.pi)),
                             f_nominal if pulse else f_nominal * float(rng.uniform(0.9, 1.1)),
                             int(rng.choice([0, 1, 3, 4], p=[0.55, 0.15, 0.1, 0.2]))]
            loads.append((t, tuple(registers)))
            t += load
        t_end = t
        dt = 1.0 / f_nominal / 4000.0
        times, expected = waveform(loads, t_end - dt, dt, pulse)
        edges = _levels(modulator, loads, t_end)
        bad, n = _compare(edges, times, expected, 2.5 * dt)
        mismatches += bad
        checked += n
    assert checked > 150000
    assert mismatches == 0


def test_frequency_register_reanchors_the_angle_to_absolute_time():
    modulator = PhaseShiftModulator(100e3)
    at = 7e-6
    slow = modulator.phase_clock(at, registers=np.array([1.0, 0.0, 50e3, 0.0]))
    fast = modulator.phase_clock(at, registers=np.array([1.0, 0.0, 100e3, 0.0]))
    assert slow.at(at) == pytest.approx(2.0 * math.pi * 50e3 * at)
    assert fast.at(at) == pytest.approx(2.0 * math.pi * 100e3 * at)


def test_register_load_on_a_switching_edge_is_applied_once():
    f = 100e3
    for offset in (-1e-15, 0.0, 1e-15, 1e-10):
        modulator = PhaseShiftModulator(f)
        # the positive pulse of width 0.5 starts at angle -pi/4: t = -1.25 us + k 10 us
        t_l = 8.75e-6 + offset
        loads = [(0.0, (0.5, 0.0, f, 0)), (t_l, (0.5, 0.0, f, 0)), (20e-6, (0.5, 0.0, f, 0))]
        edges = _levels(modulator, loads, 30e-6)
        rises = [t for (t, v), (_t0, v0) in zip(edges[1:], edges[:-1]) if v == 1 and v0 != 1]
        assert len(rises) == 3
        assert rises[0] == pytest.approx(8.75e-6, abs=2e-15)


@pytest.mark.parametrize("mode, expected", [(0, {1, -1, 0}), (1, {1, 0}), (3, {0}), (4, {-1, 0})])
def test_mode_codes(mode, expected):
    modulator = PhaseShiftModulator(100e3)
    edges = _levels(modulator, [(0.0, (0.6, 0.3, 100e3, mode))], 40e-6)
    assert {v for _t, v in edges} == expected


def test_pulse_sync_large_phase_step_leaves_no_fragment():
    f = 100e3
    period = 1.0 / f
    width = 5.0 / 6.0                      # 150 degree pulses
    modulator = _phase_modulator(f, "pulse")
    loads = [(k * 0.5 * period, (width, 0.0 if k < 7 else 2.5, f, 0)) for k in range(16)]
    edges = _levels(modulator, loads, 8 * period)
    pulses = [(t1 - t0, v) for (t0, v), (t1, _v) in zip(edges[:-1], edges[1:]) if v != 0]
    for length, _v in pulses[1:-1]:
        assert length == pytest.approx(width * 0.5 * period, rel=1e-9)
    interval = _phase_modulator(f, "interval")
    edges = _levels(interval, loads, 8 * period)
    lengths = [t1 - t0 for (t0, v), (t1, _v) in zip(edges[:-1], edges[1:]) if v != 0]
    assert min(lengths[1:-1]) < 0.9 * width * 0.5 * period   # the interval rule cuts a pulse


def test_recalling_an_interval_reproduces_it_and_restored_states_continue():
    """A continued run re-calls the modulator for the interval it was saved in, here in the
    middle of a pulse-sync transition (old pulse running, then the zero gap)."""
    f = 100e3
    period = 1.0 / f
    loads = [(k * 0.5 * period, (5.0 / 6.0, 0.0 if k < 7 else 2.5, f * (1.0 if k < 9 else 0.93), 0))
             for k in range(16)]
    whole = _phase_modulator(f, "pulse")
    expected = _levels(whole, loads, 8 * period)
    for split in (7, 8, 9):
        first = _phase_modulator(f, "pulse")
        _levels(first, loads[:split + 1], loads[split + 1][0])
        again = first(loads[split][0], 0.5 * period, np.asarray(loads[split][1], dtype=float))
        restored = _phase_modulator(f, "pulse")
        restored.set_state(first.get_state())
        edges = _levels(restored, loads[split:], 8 * period)
        recalled = _phase_modulator(f, "pulse")
        _levels(recalled, loads[:split + 1], loads[split + 1][0])
        assert again.dt.tolist() == pytest.approx(
            recalled(loads[split][0], 0.5 * period,
                     np.asarray(loads[split][1], dtype=float)).dt.tolist(), abs=1e-18)
        times = np.linspace(loads[split][0], 8 * period, 40001)[1:-1]
        assert np.array_equal(_sample(edges, times), _sample(expected, times))
        changes = [t for (t, v), (_t, v0) in zip(edges[1:], edges[:-1]) if v != v0]
        assert changes == pytest.approx(
            [t for (t, v), (_t, v0) in zip(expected[1:], expected[:-1])
             if v != v0 and t > loads[split][0] + 1e-15], abs=1e-15)


def test_carried_registers_pass_through_without_shaping_the_pattern():
    plain = PhaseShiftModulator(100e3)
    carrying = PhaseShiftModulator(100e3, carried=("scc_angle",))
    assert carrying.registers == plain.registers + ("scc_angle",)
    assert carrying.neutral_registers() == plain.neutral_registers() + (0.0,)
    loads = [(0.0, (0.6, 0.3, 100e3, 0)), (10e-6, (0.4, -0.2, 90e3, 0))]
    assert _levels(carrying, [(t, r + (1.7,)) for t, r in loads], 30e-6) == _levels(
        plain, loads, 30e-6)


def test_registers_are_validated():
    modulator = PhaseShiftModulator(100e3)
    with pytest.raises(ValueError, match="mode"):
        modulator(0.0, 1e-6, np.array([0.5, 0.0, 100e3, 2.0]))
    with pytest.raises(ValueError, match="frequency"):
        modulator(0.0, 1e-6, np.array([0.5, 0.0, 0.0, 0.0]))
    for k, name in enumerate(("width", "phase", "frequency")):
        values = np.array([0.5, 0.0, 100e3, 0.0])
        values[k] = math.nan
        with pytest.raises(ValueError, match=f"{name} register must be finite"):
            modulator(0.0, 1e-6, values)


# ------------------------------------------------------------------ the bridge in a circuit

import copy  # noqa: E402

import peslite  # noqa: E402
from peslite.assembly.params import ConfigError, from_dict, read_initial  # noqa: E402
from peslite.control.loops import LOOP_TYPES  # noqa: E402

from test_wpt_rectifier import F, SS, V1, _rectifier, _reference  # noqa: E402
from wpt_reference import rectify  # noqa: E402


def _phase_shift(width, phase=0.0, sync="interval", computation=0.0, loops=None,
                 connections=None, **references):
    ctrl = {"type": "phase_shift", "computation": computation,
            "loops": loops or {"PhaseShiftModulation": {}},
            "references": {"width": width, "frequency": F, "phase": phase, **references}}
    if connections:
        ctrl["connections"] = connections
    return {"type": "dc_converter",
            "dclink": {"vdc_ref": V1, "source": {"type": "voltage", "v": V1}},
            "pwm": {"f_sw": F, "pattern_sync": sync}, "bridge": {"model": "switching"},
            "ctrl": ctrl}


def _ps_tree(primary, secondary, t_end, *, states=False, events=None):
    tree = {"base": {"s_base": 5000.0, "v_ll_rms": 400.0, "f0": 50.0},
            "units": {"charger": {"type": "wpt", "primary": primary, "secondary": secondary,
                                  "link": SS}},
            "simulation": {"t_end": t_end,
                           "solver": {"type": "fixed", "method": "rk4", "dt": 20e-9,
                                      "linearisations": 0},
                           "output": {"period": 1e-7, "record_every": 1, "states": states,
                                      "signals": True, "energy": False}}}
    if events:
        tree["events"] = events
    return tree


def _square_wave_breaks(width, t_on, t_end):
    h = 0.5 * math.pi * width
    breaks = [t_on]
    for k in range(int(t_end * F) + 2):
        for angle in (-h, h, math.pi - h, math.pi + h):
            at = (angle + 2.0 * math.pi * k) / (2.0 * math.pi * F)
            if t_on < at < t_end:
                breaks.append(at)
    return breaks


@pytest.mark.parametrize("v2, width", [(150.0, 0.7), (220.0, 0.6), (260.0, 1.0)])
def test_square_wave_into_a_diode_rectifier_matches_the_circuit_reference(v2, width, tmp_path):
    """The hold voltage of the blocked rectifier jumps at every bridge edge; the margin check
    after each outer event keeps discontinuous conduction exact."""
    t_end = 0.4e-3
    tree = _ps_tree(_phase_shift(width), _rectifier(v2), t_end)
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    columns = result.columns()
    t = np.asarray(result.t)
    keep = (t > 0.0) & (t < t_end - 1e-9)
    t = t[keep]
    t_on = 0.0                                        # the fresh-run interrupt enables gates

    def source(at):
        return 0.0 if at < t_on else V1 * reference_level(0, width, 2.0 * math.pi * F * at)

    states, i_dc, log = rectify(_reference(SS), source, v2, t_end, t,
                                breaks=_square_wave_breaks(width, t_on, t_end), max_step=1e-7)
    scale = np.max(np.abs(states[:, 0]))
    for k, column in enumerate(("charger.link.i_primary_coil", "charger.link.i_secondary_coil")):
        assert np.max(np.abs(np.asarray(columns[column])[keep] - states[:, k])) < 2e-9 * scale
    got = np.asarray(columns["charger.secondary.i_dc"])[keep]
    assert got.mean() == pytest.approx(i_dc.mean(), rel=1e-6)
    blocked = np.mean(np.abs(states[:, 1]) < 1e-9 * scale)
    assert blocked > (0.4 if v2 > 200.0 else 0.0)


def _first_interrupt_after(t):
    return math.ceil(t * F - 1e-9) / F


@pytest.mark.parametrize("sync", ["interval", "pulse"])
def test_register_steps_through_events_follow_the_reference_waveform(sync, tmp_path):
    t_end = 0.2e-3
    t_event = 61e-6
    width, phase_step = 5.0 / 6.0, 2.5
    events = {"step": {"type": "set", "t": t_event,
                       "set": {"units.charger.primary.ctrl.references.phase": phase_step}}}
    tree = _ps_tree(_phase_shift(width, sync=sync), _rectifier(200.0), t_end, events=events)
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    t = np.asarray(result.t)
    u = np.asarray(result.columns()["charger.primary.u_g"])
    t_on, t_step = 0.0, _first_interrupt_after(t_event)
    loads = [(t_on, (width, 0.0, F, 0)), (t_step, (width, phase_step, F, 0))]
    dt = 1.0 / F / 2000.0
    times, levels = waveform(loads, t_end, dt, sync == "pulse")
    keep = (t > t_on + 1e-7) & (t < t_end - 1e-6)
    expected = levels[np.clip(np.round((t[keep] - t_on) / dt).astype(int), 0, levels.size - 1)]
    edges = times[1:][np.diff(levels) != 0]
    near = np.array([np.min(np.abs(edges - at)) <= 2.0 * dt for at in t[keep]])
    assert np.all(np.abs(u[keep][~near] - V1 * expected[~near]) < 1e-9)
    pulses = np.diff(np.flatnonzero(np.diff(np.sign(u[keep])) != 0))
    if sync == "pulse":     # no pulse fragment: every complete pulse keeps its width
        full = width * 0.5 / F / 1e-7
        lengths = [n for n, value in zip(pulses, np.sign(u[keep])[np.flatnonzero(
            np.diff(np.sign(u[keep])) != 0)[1:]]) if value != 0]
        assert all(abs(n - full) <= 1.01 for n in lengths)


def test_saved_states_continue_registers_and_pulses(tmp_path):
    whole = _ps_tree(_phase_shift(0.8, sync="pulse"), _rectifier(220.0), 0.2e-3, states=True,
                     events={"f": {"type": "set", "t": 40e-6, "set": {
                         "units.charger.primary.ctrl.references.frequency": 0.93 * F}}})
    first = copy.deepcopy(whole)
    first["simulation"]["t_end"] = 0.1e-3
    reference = peslite.Simulation(from_dict(copy.deepcopy(whole))).run(out_dir=tmp_path / "whole")
    peslite.Simulation(from_dict(first)).run(out_dir=tmp_path / "first")
    second = copy.deepcopy(whole)
    second["simulation"]["initial"] = read_initial(tmp_path / "first" / "states.csv")
    continued = peslite.Simulation(from_dict(second)).run(out_dir=tmp_path / "second")
    for column in ("charger.link.i_secondary_coil", "charger.primary.u_g"):
        expected = dict(zip(np.round(np.asarray(reference.t), 12),
                            np.asarray(reference.columns()[column])))
        got = dict(zip(np.round(np.asarray(continued.t), 12),
                       np.asarray(continued.columns()[column])))
        common = sorted(at for at in set(expected) & set(got) if at > 0.1e-3)
        assert len(common) > 500
        assert max(abs(expected[at] - got[at]) for at in common) < 1e-6 * max(
            map(abs, expected.values()))


def test_phase_shift_states_are_declared_for_loop_maps():
    simulation = peslite.Simulation(from_dict(_ps_tree(_phase_shift(0.8, sync="pulse"),
                                                       _rectifier(200.0), 1e-5)))
    angles = simulation._states.angle_names
    assert {"charger.primary.pwm.phase", "charger.primary.pwm.shadow.phase"} <= angles
    assert "charger.primary.pwm.angle" not in angles
    assert {"charger.primary.pwm.mode", "charger.primary.pwm.pulse.q_0",
            "charger.primary.pwm.pulse.q_1"} <= simulation._states.discrete_names
    assert "charger.primary.pwm.pulse.remaining_0" in simulation._states.linearization_columns


def test_power_feedback_drives_the_width(tmp_path):
    loops = {"PowerControl": {"kp": 2e-4, "ki": 3.0, "minimum": 0.05, "maximum": 1.0,
                              "power_filter": {"enable": True, "bandwidth": 1000.0}},
             "PhaseShiftModulation": {}}
    connections = {"PowerControl.reference": "references.power",
                   "PowerControl.power": "meas.power",
                   "PowerControl.feedforward": "references.width",
                   "PhaseShiftModulation.width": "PowerControl.modulation"}
    primary = _phase_shift(0.3, loops=loops, connections=connections, power=800.0)
    primary["ctrl"]["computation"] = 1e-6
    # The window integrates on its ADC grid, not on output rows: oversample the resonant power.
    primary["meas"] = {"average": ["power"], "period": 1.0 / (8 * F)}
    tree = _ps_tree(primary, _rectifier(200.0), 3e-3)
    tree["simulation"]["output"]["period"] = 1e-6
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    columns = result.columns()
    t = np.asarray(result.t)
    late = t > 2.5e-3
    width = np.asarray(columns["charger.primary.width"])[late]
    assert 0.3 < width.mean() < 1.0
    assert np.ptp(width) < 0.002
    measured = np.asarray(columns["charger.primary.power"])[late]
    assert measured.mean() == pytest.approx(800.0, rel=0.002)


@pytest.mark.parametrize("name", ["PowerControl", "PhasePowerControl", "FrequencyPowerControl",
                                  "VoltageControl", "PhaseVoltageControl", "FrequencyVoltageControl",
                                  "CurrentControl", "PhaseCurrentControl", "FrequencyCurrentControl"])
def test_feedback_family_shares_one_law(name):
    from types import SimpleNamespace

    cls = next(loop for loop in LOOP_TYPES.values() if loop.__name__ == name)
    unit = SimpleNamespace(bridge=SimpleNamespace(model="pwm_averaging"))
    minimum = -0.5 if cls.command == "phase" else 0.1
    loop = cls(cls.Params(kp=0.01, ki=0.0, minimum=minimum, maximum=2.0, period=1e-3),
               unit, SimpleNamespace(active=True, value=1.0))
    out = loop.sample(0.0, {"reference": 50.0, cls.measured: 20.0, "feedforward": 0.4})
    assert out == {cls.command: pytest.approx(0.7), cls.measured: pytest.approx(20.0)}
    assert cls.equation is LOOP_TYPES["power_control"].equation
    if cls.command == "frequency":
        with pytest.raises(ConfigError, match="minimum must be > 0"):
            cls(cls.Params(kp=0.01, ki=0.0, minimum=0.0, maximum=2.0, period=1e-3),
                unit, SimpleNamespace(active=True, value=1.0))


def test_phase_shift_configuration_is_checked():
    base = _ps_tree(_phase_shift(0.5), _rectifier(200.0), 1e-5)
    for path, value, message in [
        (("bridge", "model"), "averaging", "resolved switching edges"),
        (("bridge", "model"), "pwm_averaging", "resolved switching edges"),
        (("bridge", "type"), "half_bridge", "requires 2 switching legs"),
        (("ctrl", "references", "mode"), 2, "mode must be one of"),
        (("ctrl", "references", "width"), 1.2, "width must be in"),
    ]:
        tree = copy.deepcopy(base)
        node = tree["units"]["charger"]["primary"]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ConfigError, match=message):
            from_dict(tree)

    tree = copy.deepcopy(base)
    tree["units"]["charger"]["primary"]["pwm"]["carrier_phase"] = 0.25
    with pytest.warns(UserWarning, match="carrier_phase is ignored"):
        from_dict(tree)
    resonant = copy.deepcopy(base)
    resonant["units"]["charger"]["primary"]["ctrl"] = {
        "type": "resonant", "loops": {"ResonantModulation": {}},
        "references": {"modulation": 0.5, "frequency": F}}
    resonant["units"]["charger"]["primary"]["pwm"] = {"f_sw": 1e6, "pattern_sync": "pulse"}
    resonant["units"]["charger"]["primary"]["ctrl"]["computation"] = 0.0
    from_dict(resonant)


def test_other_exporters_receive_the_unmodified_phase_shift_model(monkeypatch, tmp_path):
    """A backend's capabilities do not constrain independently registered exporters."""
    from peslite.assembly.exporter import EXPORTERS, ExportResult

    received = []
    def custom(simulation, destination, name, variables, reusable):
        received.append(simulation)
        return ExportResult("test_phase_shift", destination, ())
    monkeypatch.setitem(EXPORTERS, "test_phase_shift", custom)
    simulation = peslite.Simulation(from_dict(_ps_tree(_phase_shift(0.5), _phase_shift(0.5), 1e-5)))
    result = simulation.export("test_phase_shift", tmp_path)
    assert received == [simulation]
    assert result.format == "test_phase_shift"


def test_imperix_reports_the_missing_hardware_modulator_adapter(tmp_path):
    simulation = peslite.Simulation(from_dict(_ps_tree(_phase_shift(.5), _phase_shift(.5), 1e-5)))
    config = {"units": {name: {} for name in simulation.system.units}}
    with pytest.raises(ValueError, match="hardware modulation adapter is not implemented"):
        simulation.export("imperix", tmp_path, target_config=config, build=False)


@pytest.mark.parametrize("sync", ["interval", "pulse"])
def test_native_phase_shift_dead_time_matches_python(sync, tmp_path):
    import csv
    import shutil
    import subprocess

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    primary, secondary = _phase_shift(.7, sync=sync), _phase_shift(.8)
    primary["pwm"]["dead_time"] = secondary["pwm"]["dead_time"] = 0.5e-6
    params = from_dict(_ps_tree(primary, secondary, 30e-6, states=True))
    peslite.Simulation(params).export("cpp", tmp_path / "model")
    subprocess.run([compiler, "-std=c++17", "-O2", str(tmp_path / "model/peslite.cpp"),
                    "-o", str(tmp_path / "run")], check=True)
    subprocess.run([str(tmp_path / "run"), "--out", str(tmp_path / "native")], check=True)
    peslite.Simulation(params).run(out_dir=tmp_path / "python")

    for filename in ("plant.csv", "states.csv"):
        with (tmp_path / "native" / filename).open() as stream:
            native = list(csv.DictReader(stream))
        with (tmp_path / "python" / filename).open() as stream:
            python = list(csv.DictReader(stream))
        assert native[0].keys() == python[0].keys()
        for key in native[0]:
            actual = np.array([float(row[key]) for row in native])
            expected = np.array([float(row[key]) for row in python])
            np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7,
                                       equal_nan=True, err_msg=f"{sync}:{filename}:{key}")


@pytest.mark.parametrize("sync", ["interval", "pulse"])
@pytest.mark.parametrize("variables", [(), ("common",), ("all",)])
@pytest.mark.parametrize("restart", [False, True])
def test_native_phase_shift_preserves_events_registers_and_pulses(sync, variables, restart, tmp_path):
    import csv
    import shutil
    import subprocess

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    tree = _ps_tree(_phase_shift(.7, sync=sync, computation=1.0e-6), _phase_shift(.8), 110e-6,
                    states=True)
    prefix = "units.charger.primary.ctrl.references."
    tree["events"] = {
        "phase": {"type": "set", "t": 35e-6, "set": {prefix + "phase": -.4}},
        "frequency": {"type": "set", "t": 57e-6, "set": {prefix + "frequency": .91 * F}},
        "mode": {"type": "set", "t": 79e-6, "set": {prefix + "mode": 4}},
    }
    if restart:
        first = copy.deepcopy(tree)
        first["simulation"]["t_end"] = 63.2e-6
        peslite.Simulation(from_dict(first)).run(out_dir=tmp_path / "first")
        tree["simulation"]["initial"] = read_initial(tmp_path / "first/states.csv")
    params = from_dict(tree)
    simulation = peslite.Simulation(params)
    simulation.export("cpp", tmp_path / "model", variables=variables)
    subprocess.run([compiler, "-std=c++17", "-O2", str(tmp_path / "model/peslite.cpp"),
                    "-o", str(tmp_path / "run")], check=True, capture_output=True, text=True)
    command = [str(tmp_path / "run"), "--out", str(tmp_path / "native")]
    if variables:
        path = prefix + "width"
        changes = {path: .6, prefix + "frequency": 1.07 * F, prefix + "phase": .3,
                   prefix + "mode": 1}
        for name, value in changes.items():
            command += ["--set", f"{name}={value}"]
        params = params.replace(**changes)
    subprocess.run(command, check=True, capture_output=True, text=True)
    reference = peslite.Simulation(params)
    reference.run(out_dir=tmp_path / "python")
    for filename in ("plant.csv", "states.csv", "ctrl.charger.primary.csv"):
        with (tmp_path / "native" / filename).open() as stream:
            native = list(csv.DictReader(stream))
        with (tmp_path / "python" / filename).open() as stream:
            python = list(csv.DictReader(stream))
        assert len(native) == len(python)
        assert native[0].keys() == python[0].keys()
        for key in native[0]:
            actual = np.array([float(row[key]) for row in native])
            expected = np.array([float(row[key]) for row in python])
            assert np.array_equal(np.isnan(actual), np.isnan(expected)), key
            valid = np.isfinite(expected)
            actual, expected = actual[valid], expected[valid]
            if not len(expected):
                continue
            error = actual - expected
            if key in reference._states.angle_names:
                error = (error + math.pi) % (2 * math.pi) - math.pi
            assert np.max(np.abs(error)) < 2e-7 * max(1.0, np.max(np.abs(expected))), key
