"""Switch-controlled capacitors on compensated links, against an independent circuit reference.

The reference integrates a finite-resistance MOSFET and diode circuit with implicit Radau.
It schedules the two gate waveforms independently and keeps capacitor voltage continuous.
"""

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

import peslite
from peslite.assembly.params import ConfigError, from_dict, read_initial
from peslite.components.wpt import SwitchControlledCapacitor
from phase_shift_reference import level
from test_phase_shift import _phase_shift, _ps_tree, _square_wave_breaks
from test_wpt_rectifier import F, L1, L2, LCC, LCC_VALUES, SS, V1, C_60K, _rectifier
from wpt_reference import Circuit, switched

pytestmark = pytest.mark.full

V2 = 150.0
T_END = 0.3e-3
T_ON = 0.0              # the fresh-run interrupt enables the bridges


def _circuit(link, coil, cx):
    if link["type"] == "ss":
        m = link["mutual_inductance"]
        return Circuit([[L1, m], [m, L2]], [C_60K, C_60K], [0.3, 0.3], ("primary", "secondary"),
                       scc={coil: cx})
    v = LCC_VALUES
    return Circuit([[v["L1"], v["M"]], [v["M"], v["L2"]]], [v["C1"], v["C2"]], [0.8, 0.8],
                   ("primary", "secondary"),
                   lcc={"primary": (v["Lf1"], 0.3, v["Cf1"], 0.0),
                        "secondary": (v["Lf2"], 0.3, v["Cf2"], 0.05)}, scc={coil: cx})


def _case(link, side, scc, secondary):
    """Primary and secondary phase-shift (or rectifier) converters; the SCC side's bridge keeps
    phase 0, the other side carries the relative phase."""
    link = dict(link, **{f"{side}_scc": scc})
    p1, p2 = (0.0, -0.5 * math.pi) if side == "primary" else (0.5 * math.pi, 0.0)
    if secondary == "rectifier":
        second = _rectifier(V2)
    else:
        second = _phase_shift(1.0, phase=p2)
        second["dclink"] = {"vdc_ref": V2, "source": {"type": "voltage", "v": V2}}
    tree = _ps_tree(_phase_shift(1.0, phase=p1), second, T_END)
    tree["units"]["charger"]["link"] = link
    tree["simulation"]["output"]["energy"] = True

    def u1(at):
        return 0.0 if at < T_ON else V1 * level(0, 1.0, 2.0 * math.pi * F * at - p1)

    def u2(at):
        return 0.0 if at < T_ON else V2 * level(0, 1.0, 2.0 * math.pi * F * at - p2)

    breaks = [T_ON] + [b + p1 / (2.0 * math.pi * F) for b in _square_wave_breaks(1.0, T_ON, T_END)]
    if secondary != "rectifier":
        breaks += [b + p2 / (2.0 * math.pi * F) for b in _square_wave_breaks(1.0, T_ON, T_END)]
    return tree, u1, (None if secondary == "rectifier" else u2), breaks


CASES = {
    "ss_bridge_common": (SS, "primary", {"capacitance": 60e-9, "angle": 2.0, "sync": "bridge",
                                       "sync_angle": 0.5 * math.pi}, "square"),
    "ss_bridge_alternating": (SS, "primary", {"capacitance": 60e-9, "angle": 1.6, "sync": "bridge",
                                          "drive": "alternating", "sync_angle": 0.5 * math.pi},
                          "square"),
    "ss_current_common": (SS, "primary", {"capacitance": 60e-9, "angle": 1.2}, "square"),
    "ss_secondary_bridge": (SS, "secondary", {"capacitance": 60e-9, "angle": 1.4,
                                              "sync": "bridge", "sync_angle": 0.5 * math.pi},
                            "square"),
    "ss_rectifier_alternating": (SS, "primary", {"capacitance": 80e-9, "angle": 1.8, "sync": "bridge",
                                             "drive": "alternating", "sync_angle": 0.5 * math.pi},
                             "rectifier"),
    "lcc_rectifier_alternating": (LCC, "primary", {"capacitance": 40e-9, "angle": 1.5,
                                               "sync": "bridge", "drive": "alternating",
                                               "sync_angle": 0.0}, "rectifier"),
}


@pytest.mark.parametrize("case", [name for name in sorted(CASES) if "current" not in name])
def test_switch_controlled_capacitor_matches_independent_reference(case, tmp_path):
    link, side, scc, secondary = CASES[case]
    tree, u1, u2, breaks = _case(link, side, scc, secondary)
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    columns = result.columns()
    t = np.asarray(result.t)
    keep = (t > 0.0) & (t < T_END - 1e-9)
    t = t[keep]
    coil = 0 if side == "primary" else 1
    circuit = _circuit(link, coil, scc["capacitance"])
    states, energy, conduction_energy = switched(
        circuit, u1, T_END, t, coil=coil, frequency=F, conduction=scc["angle"],
        sync_angle=scc.get("sync_angle", 0.0), drive=scc.get("drive", "common"), breaks=breaks, u_secondary=u2,
        V2=V2 if secondary == "rectifier" else None)
    offset = 2 * len(circuit.front)
    reference = states[:, offset:offset + 2]
    scale = np.max(np.abs(reference))
    got = np.column_stack([np.asarray(columns["charger.link.i_primary_coil"])[keep],
                           np.asarray(columns["charger.link.i_secondary_coil"])[keep]])
    assert np.max(np.abs(got - reference)) < 2e-4 * scale
    summary = result.summary
    prefix = f"charger.link.{side}_scc."
    assert summary[prefix + "switchings"] > 0
    # The finite reference also dissipates 2*Ron*I**2 during ordinary conduction.
    assert summary[prefix + "turn_on_energy"] + conduction_energy == pytest.approx(
        energy, rel=5e-3, abs=1e-7)
    energy_table = result.energy
    column = "charger.link.dissipated_impulsive_J"
    assert np.asarray(energy_table[column])[-1] == pytest.approx(summary[prefix + "turn_on_energy"])


def test_saved_states_continue_the_switch_schedule(tmp_path):
    link, side, scc, secondary = CASES["ss_current_common"]
    tree, *_ = _case(link, side, scc, secondary)
    tree["simulation"]["output"]["states"] = True
    first = copy.deepcopy(tree)
    first["simulation"]["t_end"] = 0.15e-3
    whole = peslite.Simulation(from_dict(copy.deepcopy(tree))).run(out_dir=tmp_path / "whole")
    peslite.Simulation(from_dict(first)).run(out_dir=tmp_path / "first")
    second = copy.deepcopy(tree)
    second["simulation"]["initial"] = read_initial(tmp_path / "first" / "states.csv")
    continued = peslite.Simulation(from_dict(second)).run(out_dir=tmp_path / "second")
    column = "charger.link.i_primary_coil"
    expected = dict(zip(np.round(np.asarray(whole.t), 12), np.asarray(whole.columns()[column])))
    got = dict(zip(np.round(np.asarray(continued.t), 12),
                   np.asarray(continued.columns()[column])))
    common = sorted(at for at in set(expected) & set(got) if at > 0.15e-3)
    assert len(common) > 1000
    assert max(abs(expected[at] - got[at]) for at in common) < 1e-7 * max(
        map(abs, expected.values()))


def test_phase_and_frequency_steps_reschedule_the_bridge_references(tmp_path):
    """References follow the active phase register; no stale instant stalls the run."""
    link, side, scc, secondary = CASES["ss_bridge_common"]
    tree, *_ = _case(link, side, scc, secondary)
    path = "units.charger.primary.ctrl.references"
    tree["events"] = {f"s{k}": {"type": "set", "t": at, "set": {f"{path}.phase": value}}
                      for k, (at, value) in enumerate([(50e-6, 0.4), (90.3e-6, -0.7),
                                                       (130e-6, 2.9), (170.1e-6, 0.0)])}
    tree["events"]["f"] = {"type": "set", "t": 210e-6, "set": {f"{path}.frequency": 0.9 * F}}
    tree["simulation"]["output"]["period"] = 2e-8
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    assert result.summary["t_stop"] == pytest.approx(T_END)
    t = np.asarray(result.t)
    assert np.all(np.isfinite(result.columns()["charger.link.i_primary_coil"]))
    refined = copy.deepcopy(tree)
    refined["simulation"]["solver"]["dt"] *= 0.5
    reference = peslite.Simulation(from_dict(refined)).run(out_dir=tmp_path / "refined")
    for key in ("charger.link.i_primary_coil", "charger.link.i_secondary_coil"):
        expected = np.interp(result.t, reference.t, reference.columns()[key])
        assert np.asarray(result.columns()[key]) == pytest.approx(expected, rel=2e-5, abs=2e-5)


def test_command_input_closes_a_power_loop_through_the_scc_angle(tmp_path):
    """Power falls with angle on this independently checked branch; a phase-output PI drives it."""
    link = dict(SS, primary_scc={"capacitance": 120e-9, "sync": "bridge",
                                 "drive": "common", "sync_angle": 0.5 * math.pi,
                                 "command": True})
    primary = _phase_shift(1.0)
    primary["meas"] = {"average": ["power"]}
    primary["ctrl"]["computation"] = 1e-6
    primary["ctrl"]["references"]["power"] = 8000.0
    primary["ctrl"]["loops"] = {
        "PhaseShiftModulation": {},
        "PhasePowerControl": {"kp": -2e-5, "ki": -0.3, "minimum": 0.3, "maximum": 1.8,
                              "power_filter": {"enable": True, "bandwidth": 1000.0}}}
    primary["ctrl"]["connections"] = {"PhasePowerControl.reference": "references.power",
                                      "PhasePowerControl.power": "meas.power",
                                      "PhasePowerControl.feedforward": "references.phase"}
    primary["ctrl"]["outputs"] = {"scc_angle": "PhasePowerControl.phase"}
    tree = _ps_tree(primary, _rectifier(V2), 4e-3)
    tree["units"]["charger"]["link"] = link
    tree["simulation"]["output"]["period"] = 1e-6
    result = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path)
    t = np.asarray(result.t)
    power = np.asarray(result.columns()["charger.primary.power"])
    assert np.mean(power[t > 3.5e-3]) == pytest.approx(8000.0, rel=0.01)
    assert result.summary["charger.link.primary_scc.hard_turn_ons"] > 0


def test_scc_configuration_is_checked():
    link, side, scc, secondary = CASES["ss_bridge_common"]
    base, *_ = _case(link, side, scc, secondary)
    for change, message in [
        ({"capacitance": -1e-9}, "capacitance must be finite and positive"),
        ({"angle": 4.0}, "angle must be in"),
        ({"command": True}, "controller output scc_angle"),
    ]:
        tree = copy.deepcopy(base)
        tree["units"]["charger"]["link"]["primary_scc"].update(change)
        with pytest.raises(ConfigError, match=message):
            from_dict(tree)
    commanded = _commanded(1e-5)
    commanded["units"]["charger"]["link"]["primary_scc"]["angle"] = 1.0
    with pytest.raises(ConfigError, match="not used with command"):
        from_dict(commanded)
    rectified = copy.deepcopy(base)
    rectified["units"]["charger"]["link"]["secondary_scc"] = {"capacitance": 1e-8,
                                                              "sync": "bridge"}
    rectified["units"]["charger"]["secondary"] = _rectifier(V2)
    with pytest.raises(ConfigError, match="clocked bridge"):
        from_dict(rectified)
    alternating = copy.deepcopy(base)
    alternating["units"]["charger"]["link"]["primary_scc"].update(sync="current", drive="alternating")
    from_dict(alternating)
    multi = {"type": "ss_multi", "inductance": [[1e-4, 2e-5], [2e-5, 1e-4]],
             "primary_capacitance": [1e-7], "secondary_capacitance": [1e-7],
             "primary_scc": {"capacitance": 1e-8}}
    tree = copy.deepcopy(base)
    tree["units"]["charger"]["link"] = multi
    with pytest.raises(ConfigError, match="primary_scc"):
        from_dict(tree)


def test_switched_link_uses_the_native_commutation_dispatcher(tmp_path):
    link, side, scc, secondary = CASES["ss_bridge_common"]
    tree, *_ = _case(link, side, scc, secondary)
    result = peslite.Simulation(from_dict(tree)).export("cpp", tmp_path)
    assert result.files[0].is_file()


def _commanded(t_end):
    link = dict(SS, primary_scc={"capacitance": 120e-9, "sync": "bridge",
                                 "sync_angle": 0.5 * math.pi, "command": True})
    primary = _phase_shift(1.0)
    primary["meas"] = {"average": ["power"]}
    primary["ctrl"]["computation"] = 2e-6
    primary["ctrl"]["references"]["power"] = 8000.0
    primary["ctrl"]["loops"] = {
        "PhaseShiftModulation": {},
        "PhasePowerControl": {"kp": -2e-5, "ki": -0.3, "minimum": 0.3, "maximum": 1.8}}
    primary["ctrl"]["connections"] = {"PhasePowerControl.reference": "references.power",
                                      "PhasePowerControl.power": "meas.power",
                                      "PhasePowerControl.feedforward": "references.phase"}
    primary["ctrl"]["outputs"] = {"scc_angle": "PhasePowerControl.phase"}
    tree = _ps_tree(primary, _rectifier(V2), t_end, states=True)
    tree["units"]["charger"]["link"] = link
    return tree


@pytest.mark.parametrize("drive", ["common", "alternating"])
def test_commanded_angle_takes_the_register_path_and_continues(drive, tmp_path):
    whole = _commanded(0.3e-3)
    whole["units"]["charger"]["link"]["primary_scc"]["drive"] = drive
    simulation = peslite.Simulation(from_dict(copy.deepcopy(whole)))
    names = set(simulation._states.read())
    assert {"charger.primary.pwm.scc_angle", "charger.primary.pwm.shadow.scc_angle"} <= names
    reference = simulation.run(out_dir=tmp_path / "whole")
    first = copy.deepcopy(whole)
    first["simulation"]["t_end"] = 0.17e-3
    peslite.Simulation(from_dict(first)).run(out_dir=tmp_path / "first")
    rows = (tmp_path / "first" / "states.csv").read_text().splitlines()
    assert "nan" not in rows[-1].lower()
    initial = read_initial(tmp_path / "first" / "states.csv")
    second = copy.deepcopy(whole)
    second["simulation"]["initial"] = initial
    continued = peslite.Simulation(from_dict(second)).run(out_dir=tmp_path / "second")
    for column in ("charger.link.i_primary_coil", "charger.link.primary_scc_on"):
        expected = dict(zip(np.round(np.asarray(reference.t), 12),
                            np.asarray(reference.columns()[column])))
        got = dict(zip(np.round(np.asarray(continued.t), 12),
                       np.asarray(continued.columns()[column])))
        common = sorted(at for at in set(expected) & set(got) if at > 0.17e-3)
        assert len(common) > 500
        assert max(abs(expected[at] - got[at]) for at in common) < 1e-7 * max(
            map(abs, expected.values()))


def test_clock_memory_is_retained_in_the_linearized_map():
    link, side, config, secondary = CASES["ss_current_common"]
    tree, *_ = _case(link, side, config, secondary)
    states = peslite.Simulation(from_dict(tree))._states
    assert "charger.link.primary_scc.half" in states.linearization_columns
    assert "charger.link.primary_scc.sign" in states.discrete_names


@pytest.mark.parametrize("drive", ["common", "alternating"])
def test_trial_margins_leave_switch_state_and_diagnostics_unchanged(drive):
    scc = SwitchControlledCapacitor("primary", 0, 60e-9, 1.2, drive=drive)
    scc.started = True
    scc.zero.sign = 1.0
    before = scc.get_state(0.0), scc.diagnostics()
    for at, current, voltage in [(1e-6, .2, 30), (.5e-6, -.1, -40), (0, 0, 0)]:
        scc.margin(at, current, voltage)
    assert (scc.get_state(0.0), scc.diagnostics()) == before


def test_system_state_interface_includes_switch_schedules():
    link, side, scc, secondary = CASES["ss_current_common"]
    tree, *_ = _case(link, side, scc, secondary)
    system = peslite.Simulation(from_dict(tree)).system
    prefix = "charger.link.primary_scc."
    assert prefix + "theta" in system.get_state()
    system.set_state({prefix + "theta": 1.3, prefix + "since": 0.2})
    saved = system.get_state()
    assert saved[prefix + "theta"] == 1.3
    assert saved[prefix + "since"] == 0.2
    system.set_state(saved)
    assert system.get_state() == saved
    with pytest.raises(KeyError, match="unknown system states"):
        system.set_state({prefix + "unknown": 0.0})


def test_scc_and_phase_shift_reuse_the_capacitive_bridge_equations(tmp_path):
    link, side, scc, secondary = CASES["ss_bridge_common"]
    tree, *_ = _case(link, side, scc, secondary)
    tree["simulation"]["t_end"] = 30e-6
    tree["simulation"]["output"]["states"] = True
    for side in ("primary", "secondary"):
        tree["units"]["charger"][side]["bridge"].update(
            parallel_capacitance=20e-9, on_resistance=1.0)
    results = []
    for dt in (5e-9, 2.5e-9):
        tree["simulation"]["solver"]["dt"] = dt
        sim = peslite.Simulation(from_dict(copy.deepcopy(tree)))
        result = sim.run(out_dir=tmp_path / str(dt))
        assert result.summary["t_stop"] == pytest.approx(30e-6)
        balance = sim.system.model.energy_balance(result.t[-1], sim.system.model.get_initial_values())
        assert balance.max_residual < 1e-8 * max(1.0, balance.scale)
        results.append(result)
    first, refined = results
    for key in ("charger.link.i_primary_coil", "charger.link.i_secondary_coil"):
        expected = np.interp(first.t, refined.t, refined.columns()[key])
        error = np.max(np.abs(np.asarray(first.columns()[key]) - expected))
        assert error < 5e-4 * max(1.0, np.max(np.abs(expected)))


class _Clock:
    def __init__(self, frequency=100e3):
        self.frequency = frequency
        self.phase = 0.0

    def phase_clock(self):
        from peslite.components.pwm import PhaseClock
        return PhaseClock(0.0, 0.0, 2 * math.pi * self.frequency)

    def phase_reference(self):
        return self.phase


def _capacitor(current, *, angle=1.2, drive="alternating", voltage=0.0):
    """Use the existing capacitor and real commutation dispatcher for device tests."""
    from peslite.components.network import RCNode
    from peslite.solver.model import Model
    from peslite.assembly.conduction import DiodeConduction
    from peslite.solver.integrators import FixedStepSolver

    switch = SwitchControlledCapacitor("primary", 0, 60e-9, angle, drive, "bridge")
    switch.state_name = "u_C"
    switch.bind_bridge(_Clock())

    class Capacitor(RCNode):
        def set_outputs(self, t):
            self.t = t
            self.inp.i = 0.0 if switch.on else current(t)
            super().set_outputs(t)

        def commutation_margin(self):
            return switch.margin(self.t, current(self.t), self.state.u_C)

        def next_commutation_time(self):
            return switch.next_time(self.t)

        def commutate(self, t):
            switch.commutate(self, t, current(t), self.state.u_C)

    capacitor = Capacitor(60e-9, 0.0, u0=voltage, phases=1)
    model = Model({"capacitor": capacitor}, {})
    handler = DiodeConduction(model, FixedStepSolver(2e-9, "rk4"))
    return switch, capacitor, model, handler


@pytest.mark.parametrize("delta", [-5e-16, 0.0, 5e-16])
def test_gate_edge_is_consumed_once_by_the_real_dispatcher(delta):
    switch, capacitor, model, handler = _capacitor(lambda t: 1.0)
    state = model.get_initial_values()
    handler.synchronize(0.0, state)
    edge = switch.next_time(0.0)
    handler.synchronize(edge + delta, state)
    assert switch.gates == (True, True)
    assert switch.next_time(edge + delta) > edge + 1e-9
    count = switch.switchings
    handler.synchronize(edge + delta, state)
    assert switch.switchings == count


def test_body_diode_conducts_outside_the_two_gate_overlap():
    switch, capacitor, model, handler = _capacitor(lambda t: 1.0, angle=0.0, voltage=2.0)
    state = model.get_initial_values()
    handler.synchronize(0.0, state)
    assert switch.gates == (True, False)
    assert switch.on
    assert capacitor.state.u_C == 0.0
    assert switch.impulsive_energy == pytest.approx(0.5 * capacitor.C * 2.0**2)


def test_clock_and_carried_commands_do_not_depend_on_phase_shift_modulation():
    from peslite.components.converter import HalfBridge, SwitchingModel
    from peslite.components.pwm import CarrierComparison
    from peslite.control.controller import BridgeControlOutput

    modulator = CarrierComparison(100e3, phase=0.25)
    modulator.carried = ("scc_angle",)
    bridge = SwitchingModel(1e-5, 1e-5, 0.0, 0.0, 0.0, 1e-5, modulator, HalfBridge())
    output = BridgeControlOutput(commands={"duty": 0.4, "scc_angle": 1.2})
    bridge.reset(bridge.command_registers(output), on=True)
    sequence = bridge.load(0.0)
    assert bridge.modulation_width == 1
    assert np.isfinite(sequence)
    switch = SwitchControlledCapacitor("primary", 0, 60e-9, 0.0, sync="bridge", command=True)
    switch.bind_bridge(bridge)
    assert switch._command() == 1.2
    assert switch._clock().at(0.0) == pytest.approx(math.pi / 2)
    assert switch._gate_state(0.0) == (True, True)
    assert not {"since", "half", "theta", "referenced"} & switch.get_state(0.0).keys()


def test_two_ungated_devices_block_both_current_directions():
    for current in (-1.0, 1.0):
        switch, capacitor, model, handler = _capacitor(lambda t: current, drive="common", angle=0.0,
                                                      voltage=2.0 * current)
        state = model.get_initial_values()
        handler.synchronize(0.0, state)
        assert switch.gates == (False, False)
        assert not switch.on
        assert capacitor.state.u_C == 2.0 * current


@pytest.mark.parametrize("drive", ["common", "alternating"])
def test_constant_bypass_does_not_need_current_clock_acquisition(drive):
    switch = SwitchControlledCapacitor("primary", 0, 60e-9, math.pi, drive)
    owner = SimpleNamespace(state=SimpleNamespace(u_primary_scc=3.0))
    switch.commutate(owner, 0.0, 0.0, 3.0)
    assert switch.gates == (True, True)
    assert switch.on
    assert owner.state.u_primary_scc == 0.0
    assert switch.next_time(0.0) == math.inf


def test_current_reference_accepts_each_located_zero_without_fixed_debounce():
    switch = SwitchControlledCapacitor("primary", 0, 60e-9, 1.2)
    switch.started = switch.referenced = True
    switch.half = 5e-6
    switch.zero.sign = 1.0
    owner = SimpleNamespace(state=SimpleNamespace(u_primary_scc=0.0))
    switch.commutate(owner, 1e-6, -2e-9, 0.0)
    assert switch._reference == 1e-6
    assert switch.half == 1e-6
    assert switch.zero.sign == -1.0


@pytest.mark.parametrize("drive, expected, phase, after", [
    ("common", [(False, False), (True, True), (False, False), (False, False), (True, True)],
     math.pi / 2, (True, True)),
    ("alternating", [(True, False), (True, True), (False, True), (True, False), (True, True)],
     math.pi, (True, False)),
])
def test_gate_schedule_remains_valid_for_backwards_locator_probes(drive, expected, phase, after):
    switch = SwitchControlledCapacitor("primary", 0, 60e-9, math.pi / 2, drive, "bridge")
    clock = _Clock()
    switch.bind_bridge(clock)
    times = [9e-6, 2e-6, 4e-6, .25e-6, 7e-6]
    assert [switch._gate_state(at) for at in times] == expected
    clock.phase = phase
    assert switch._gate_state(4e-6) == after
    switch.angle = math.pi
    assert switch._gate_state(4e-6) == (True, True)
