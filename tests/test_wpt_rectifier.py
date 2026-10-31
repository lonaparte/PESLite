"""Diode full-bridge rectifiers on compensated links, against an independent circuit reference.

The reference (``wpt_reference.py``) integrates the circuit equations in coil coordinates with the
full inductance matrix and SciPy DOP853, locating every diode transition with its own events.
"""

import copy
import math

import numpy as np
import pytest

import peslite
from peslite.assembly.params import ConfigError, from_dict, read_initial
from peslite.components import FullBridge
from wpt_reference import Circuit, rectify

pytestmark = pytest.mark.slow

F = 85e3
V1 = 400.0


def _primary():
    """An ideal sinusoidal port voltage V1 sin(2 pi F t) from the averaging bridge."""
    return {"type": "dc_converter",
            "dclink": {"vdc_ref": V1, "source": {"type": "voltage", "v": V1}},
            "pwm": {"f_sw": 1e6}, "bridge": {"model": "averaging"},
            "ctrl": {"type": "resonant", "loops": {"ResonantModulation": {}},
                     "references": {"modulation": 1.0, "frequency": F, "phase": 0.0}}}


def _rectifier(v2, capacitance=0.0):
    bridge = {"type": "full_bridge", "upper_controlled": False, "lower_controlled": False,
              "model": "switching"}
    if capacitance:
        bridge["parallel_capacitance"] = capacitance
        bridge["on_resistance"] = 10.0
    return {"type": "dc_converter",
            "dclink": {"vdc_ref": v2, "source": {"type": "voltage", "v": v2}},
            "bridge": bridge}


def _tree(link, v2, t_end, capacitance=0.0, dt=20e-9):
    return {"base": {"s_base": 5000.0, "v_ll_rms": 400.0, "f0": 50.0},
            "units": {"charger": {"type": "wpt", "primary": _primary(),
                                  "secondary": _rectifier(v2, capacitance), "link": link}},
            "simulation": {"t_end": t_end,
                           "solver": {"type": "fixed", "method": "rk4", "dt": dt,
                                      "linearisations": 0},
                           "output": {"period": 1e-7, "record_every": 1, "states": False,
                                      "signals": True, "energy": False}}}


def test_bridge_model_selection_does_not_change_natural_rectifier_commutation(tmp_path):
    results = {}
    implementations = []
    for model in ("switching", "pwm_averaging", "averaging"):
        tree = _tree(SS, 220.0, 30e-6)
        tree["units"]["charger"]["secondary"]["bridge"]["model"] = model
        simulation = peslite.Simulation(from_dict(tree))
        bridge = simulation.unit("charger.secondary").bridge
        implementations.append((bridge.set_dc_current.__func__, bridge.set_ac_voltage.__func__))
        result = simulation.run(out_dir=tmp_path / model)
        columns = result.columns()
        results[model] = (
            np.asarray(result.t),
            np.asarray(columns["charger.secondary.i_dc"]),
            np.asarray(columns["charger.link.i_secondary_coil"]),
        )

    assert implementations[1:] == implementations[:1] * 2
    for model in ("pwm_averaging", "averaging"):
        for actual, expected in zip(results[model], results["switching"]):
            np.testing.assert_array_equal(actual, expected)


L1 = L2 = 100e-6
M = 30e-6
C_60K = 1.0 / ((2 * math.pi * 60e3) ** 2 * L1)
SS = {"type": "ss", "primary_inductance": L1, "secondary_inductance": L2,
      "mutual_inductance": M, "primary_capacitance": C_60K, "secondary_capacitance": C_60K,
      "primary_resistance": 0.3, "secondary_resistance": 0.3}
LCC_VALUES = dict(L1=200.7e-6, L2=203.5e-6, M=30e-6, C1=26.87e-9, C2=28.98e-9, Lf1=66.39e-6,
                  Lf2=68.97e-6, Cf1=54.83e-9, Cf2=51.67e-9)
LCC = {"type": "lcc", "primary_inductance": 200.7e-6, "secondary_inductance": 203.5e-6,
       "mutual_inductance": 30e-6, "primary_series_capacitance": 26.87e-9,
       "secondary_series_capacitance": 28.98e-9, "primary_compensation_inductance": 66.39e-6,
       "secondary_compensation_inductance": 68.97e-6, "primary_parallel_capacitance": 54.83e-9,
       "secondary_parallel_capacitance": 51.67e-9, "primary_resistance": 0.8,
       "secondary_resistance": 0.8, "primary_compensation_resistance": 0.3,
       "secondary_compensation_resistance": 0.3, "secondary_parallel_resistance": 0.05}
C_S = 1.0 / ((2 * math.pi * F) ** 2 * 203.5e-6)
LCC_S = {"type": "lcc_s", "primary_inductance": 200.7e-6, "secondary_inductance": 203.5e-6,
         "mutual_inductance": 30e-6, "primary_series_capacitance": 26.87e-9,
         "primary_compensation_inductance": 66.39e-6, "primary_parallel_capacitance": 54.83e-9,
         "secondary_capacitance": C_S, "primary_resistance": 0.8, "secondary_resistance": 0.8,
         "primary_compensation_resistance": 0.3, "primary_parallel_resistance": 0.02}
C_M = 1.0 / ((2 * math.pi * F) ** 2 * 120e-6)
L_MULTI = [[120e-6, 6e-6, 20e-6, 3e-6], [6e-6, 120e-6, 2e-6, 18e-6],
           [20e-6, 2e-6, 120e-6, 5e-6], [3e-6, 18e-6, 5e-6, 120e-6]]
MULTI = {"type": "ss_multi", "inductance": L_MULTI,
         "primary_capacitance": [C_M, 1.05 * C_M], "secondary_capacitance": [0.97 * C_M, C_M],
         "primary_resistance": [0.05, 0.06], "secondary_resistance": [0.04, 0.05]}


def _reference(link, series=0.0):
    two = (("primary", "secondary"))
    if link["type"] == "ss":
        m = link["mutual_inductance"]
        return Circuit([[L1, m], [m, L2]], [C_60K, C_60K], [0.3, 0.3], two,
                       primary_series_resistance=series)
    if link["type"] == "lcc":
        v = LCC_VALUES
        return Circuit([[v["L1"], v["M"]], [v["M"], v["L2"]]], [v["C1"], v["C2"]], [0.8, 0.8], two,
                       lcc={"primary": (v["Lf1"], 0.3, v["Cf1"], 0.0),
                            "secondary": (v["Lf2"], 0.3, v["Cf2"], 0.05)})
    if link["type"] == "lcc_s":
        v = LCC_VALUES
        return Circuit([[v["L1"], v["M"]], [v["M"], v["L2"]]], [v["C1"], C_S], [0.8, 0.8], two,
                       lcc={"primary": (v["Lf1"], 0.3, v["Cf1"], 0.02)})
    return Circuit(L_MULTI, [C_M, 1.05 * C_M, 0.97 * C_M, C_M], [0.05, 0.06, 0.04, 0.05],
                   ("primary", "primary", "secondary", "secondary"))


def _coil_columns(link):
    if link["type"] == "ss_multi":
        return ["charger.link.i_primary_coil_1", "charger.link.i_primary_coil_2",
                "charger.link.i_secondary_coil_1", "charger.link.i_secondary_coil_2"]
    return ["charger.link.i_primary_coil", "charger.link.i_secondary_coil"]


def _compare(tree, reference, out_dir, capacitance=0.0):
    simulation = peslite.Simulation(from_dict(copy.deepcopy(tree)))
    result = simulation.run(out_dir=out_dir)
    t_end = tree["simulation"]["t_end"]
    t = np.asarray(result.t)
    cols = result.columns()
    keep = (t > 0.0) & (t < t_end - 1e-9)
    t = t[keep]
    states, i_dc_ref, log = rectify(
        reference, lambda tt: V1 * math.sin(2 * math.pi * F * tt), tree["units"]["charger"][
            "secondary"]["dclink"]["source"]["v"], t_end, t, parallel_capacitance=capacitance,
        on_resistance=tree["units"]["charger"]["secondary"]["bridge"].get("on_resistance", 0.0))
    offset = 2 * len(reference.front)
    coil_errors = []
    for j, column in enumerate(_coil_columns(tree["units"]["charger"]["link"])):
        expected = states[:, offset + j]
        coil_errors.append(np.max(np.abs(np.asarray(cols[column])[keep] - expected))
                           / np.max(np.abs(expected)))
    i_dc = np.asarray(cols["charger.secondary.i_dc"])[keep]
    bridge, = (sub for sub in simulation.system.model.subsystems if isinstance(getattr(sub, "topology", None), FullBridge) and not sub.topology.controlled)
    return simulation, result, coil_errors, i_dc, i_dc_ref, log, bridge


CASES = {
    "ss_dcm": (SS, 200.0, 0.0),
    "ss_ccm": (dict(SS, primary_capacitance=1 / ((2 * math.pi * F) ** 2 * L1),
                    secondary_capacitance=1 / ((2 * math.pi * F) ** 2 * L2)), 60.0, 0.0),
    "ss_negative_mutual": (dict(SS, mutual_inductance=-M), 200.0, 0.0),
    "lcc_dcm": (LCC, 600.0, 0.0),
    "lcc_s": (LCC_S, 50.0, 0.0),
    "ss_multi": (MULTI, 500.0, 0.0),
    "lcc_parallel_capacitance": (dict(LCC, secondary_parallel_resistance=0.0), 150.0, 4.7e-9),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_diode_bridge_matches_independent_circuit_reference(case, tmp_path):
    link, v2, capacitance = CASES[case]
    if case == "ss_ccm":
        reference = Circuit([[L1, M], [M, L2]], [link["primary_capacitance"]] * 2, [0.3, 0.3],
                            ("primary", "secondary"))
    elif case == "lcc_parallel_capacitance":
        v = LCC_VALUES
        reference = Circuit([[v["L1"], v["M"]], [v["M"], v["L2"]]], [v["C1"], v["C2"]],
                            [0.8, 0.8], ("primary", "secondary"),
                            lcc={"primary": (v["Lf1"], 0.3, v["Cf1"], 0.0),
                                 "secondary": (v["Lf2"], 0.3, v["Cf2"], 0.0)})
    else:
        reference = _reference(link)
    _sim, result, errors, i_dc, i_dc_ref, log, bridge = _compare(
        _tree(link, v2, 0.3e-3, capacitance, dt=5e-9 if capacitance else 20e-9),
        reference, tmp_path, capacitance)

    assert max(errors) < 1e-6
    tail = slice(-1000, None)  # the last 0.1 ms
    assert np.mean(i_dc[tail]) == pytest.approx(np.mean(i_dc_ref[tail]), rel=1e-6)
    if not capacitance:
        assert bridge.commutations == len(log)
    assert np.array_equal(np.abs(i_dc) < 1e-9, i_dc_ref == 0.0)
    assert result.summary["ph_verdict"] == "port-hamiltonian"


def test_discontinuous_conduction_occurs_in_the_reference_cases():
    """The DCM cases really block: a hold-voltage error could not hide in continuous conduction."""
    for case in ("ss_dcm", "lcc_dcm", "lcc_parallel_capacitance"):
        link, v2, capacitance = CASES[case]
        reference = _reference(link) if case != "lcc_parallel_capacitance" else Circuit(
            [[200.7e-6, 30e-6], [30e-6, 203.5e-6]], [26.87e-9, 28.98e-9], [0.8, 0.8],
            ("primary", "secondary"), lcc={"primary": (66.39e-6, 0.3, 54.83e-9, 0.0),
                                           "secondary": (68.97e-6, 0.3, 51.67e-9, 0.0)})
        t = np.linspace(0.2e-3, 0.3e-3, 1001)
        _states, i_dc, _log = rectify(reference, lambda tt: V1 * math.sin(2 * math.pi * F * tt),
                                      v2, 0.3e-3, t, parallel_capacitance=capacitance,
                                      on_resistance=10.0 if capacitance else 0.0)
        assert 0.05 < np.mean(i_dc == 0.0) < 0.95


def test_hold_voltage_uses_the_link_terminal_behind_a_series_element(tmp_path):
    tree = _tree(SS, 200.0, 0.3e-3)
    tree["elements"] = {"rs": {"type": "resistor", "bus": "charger.primary.power", "r": 2.0,
                               "series": True}}
    _sim, _result, errors, i_dc, i_dc_ref, log, bridge = _compare(
        tree, _reference(SS, series=2.0), tmp_path)
    assert max(errors) < 1e-6
    assert np.mean(i_dc[-1000:]) == pytest.approx(np.mean(i_dc_ref[-1000:]), rel=1e-6)
    assert bridge.commutations == len(log)


def test_adaptive_solver_locates_the_same_discontinuous_conduction(tmp_path):
    """The margins are checked every ``max_step``, whatever the output grid: a coarse output
    period (the only outer events here) must not hide commutations."""
    fixed = _tree(SS, 200.0, 0.2e-3)
    fixed["simulation"]["output"]["period"] = 2e-5
    adaptive = copy.deepcopy(fixed)
    adaptive["simulation"]["solver"] = {"type": "adaptive", "method": "DOP853", "rtol": 1e-10,
                                        "atol": 1e-12, "max_step": 50e-9, "linearisations": 0}
    fine = copy.deepcopy(adaptive)
    fine["simulation"]["output"]["period"] = 1e-7
    results = []
    for k, tree in enumerate((fixed, adaptive, fine)):
        simulation = peslite.Simulation(from_dict(copy.deepcopy(tree)))
        result = simulation.run(out_dir=tmp_path / str(k))
        bridge, = (sub for sub in simulation.system.model.subsystems
                   if isinstance(getattr(sub, "topology", None), FullBridge) and not sub.topology.controlled)
        columns = result.columns()
        values = dict(zip(np.round(np.asarray(result.t), 12),
                          np.asarray(columns["charger.link.i_secondary_coil"])))
        results.append((values, bridge.commutations))
    (fixed_i, fixed_n), (coarse_i, coarse_n), (fine_i, fine_n) = results
    assert fixed_n == coarse_n == fine_n > 50
    scale = max(map(abs, fixed_i.values()))
    for other in (coarse_i, fine_i):
        assert max(abs(fixed_i[t] - other[t]) for t in fixed_i) < 1e-6 * scale


def test_adaptive_solver_needs_a_finite_check_interval():
    tree = _tree(SS, 200.0, 1e-5)
    tree["simulation"]["solver"] = {"type": "adaptive", "method": "DOP853", "linearisations": 0}
    tree["simulation"]["output"]["signals"] = False
    with pytest.raises(ConfigError, match="max_step must be finite"):
        peslite.Simulation(from_dict(tree)).run(out_dir=None)


def test_continuation_from_saved_states_reproduces_the_run(tmp_path):
    whole = _tree(SS, 200.0, 0.2e-3)
    whole["simulation"]["output"]["states"] = True
    first = copy.deepcopy(whole)
    first["simulation"]["t_end"] = 0.1e-3
    reference = peslite.Simulation(from_dict(copy.deepcopy(whole))).run(out_dir=tmp_path / "whole")
    peslite.Simulation(from_dict(first)).run(out_dir=tmp_path / "first")
    second = copy.deepcopy(whole)
    second["simulation"]["initial"] = read_initial(tmp_path / "first" / "states.csv")
    continued = peslite.Simulation(from_dict(second)).run(out_dir=tmp_path / "second")
    column = "charger.link.i_secondary_coil"
    expected = dict(zip(np.round(np.asarray(reference.t), 12),
                        np.asarray(reference.columns()[column])))
    got = dict(zip(np.round(np.asarray(continued.t), 12), np.asarray(continued.columns()[column])))
    common = sorted(set(expected) & set(got))
    assert len(common) > 500
    assert max(abs(expected[t] - got[t]) for t in common) < 1e-9 * max(map(abs, expected.values()))


def test_diode_full_bridge_configuration_is_checked():
    tree = _tree(SS, 200.0, 1e-5)
    default_model = copy.deepcopy(tree)
    default_model["units"]["charger"]["secondary"]["bridge"].pop("model")
    params = from_dict(default_model)
    assert params.unit("charger.secondary").bridge.model == "pwm_averaging"
    assert peslite.Simulation(params).unit("charger.secondary").bridge.describe().startswith(
        "diode full-bridge rectifier"
    )
    with_ctrl = copy.deepcopy(tree)
    with_ctrl["units"]["charger"]["secondary"]["ctrl"] = {"type": "resonant", "loops": {}}
    with pytest.raises(ConfigError, match="no ctrl"):
        from_dict(with_ctrl)
    mixed = copy.deepcopy(tree)
    mixed["units"]["charger"]["secondary"]["bridge"]["upper_controlled"] = True
    mixed["units"]["charger"]["secondary"]["pwm"] = {"f_sw": 1e5}
    mixed["units"]["charger"]["secondary"]["ctrl"] = {
        "type": "resonant", "loops": {"ResonantModulation": {}},
        "references": {"modulation": 0.5, "frequency": F}}
    with pytest.raises(ConfigError, match="four controlled devices or none"):
        from_dict(mixed)
    soft = _tree(LCC, 200.0, 1e-5, capacitance=1e-9)
    soft["units"]["charger"]["secondary"]["dclink"]["capacitor"] = {"c": 1e-6}
    soft["units"]["charger"]["secondary"]["dclink"]["source"]["r"] = 75.0
    assert from_dict(soft).unit("charger.secondary").dclink.capacitor is not None
    resistive = _tree(LCC, 200.0, 1e-5, capacitance=1e-9)
    resistive["units"]["charger"]["secondary"]["dclink"]["source"]["r"] = 5.0
    assert from_dict(resistive).unit("charger.secondary").dclink.source.r == 5.0
    loaded = _tree(LCC, 200.0, 1e-5, capacitance=1e-9)
    loaded["elements"] = {"rs": {"type": "resistor", "bus": "charger.secondary.dc", "r": 2.0,
                                 "series": True}}
    peslite.Simulation(from_dict(loaded))
    for name in ("overcurrent", "dc_voltage"):
        protected = copy.deepcopy(tree)
        protected["units"]["charger"]["secondary"]["protection"] = {
            name: {"enable": True, "limit_pu": 2.0}}
        with pytest.raises(ConfigError, match="no gates to block"):
            from_dict(protected)
    for bridge, message in (({"type": "half_bridge"}, "at least one controlled device"),
                            ({}, "required when no devices are controlled")):
        leg = copy.deepcopy(tree)
        leg["units"]["charger"]["secondary"]["bridge"] = dict(
            bridge, upper_controlled=False, lower_controlled=False, model="switching")
        with pytest.raises(ConfigError, match=message):
            from_dict(leg)


def test_diode_full_bridge_is_exported_without_a_controller(tmp_path):
    result = peslite.export(from_dict(_tree(SS, 200.0, 1e-5)), "cpp", tmp_path)
    assert result.files[0].is_file()


def test_capacitive_diode_bridge_is_exported_without_a_controller(tmp_path):
    result = peslite.export(from_dict(_tree(LCC, 200.0, 1e-5, capacitance=4.7e-9)), "cpp", tmp_path)
    assert result.files[0].is_file()


def test_windowed_solver_groups_are_rejected():
    tree = _tree(SS, 200.0, 1e-5)
    tree["simulation"]["solver"]["subsystems"] = {"charger.primary.dclink": 4}
    tree["simulation"]["output"]["signals"] = False
    simulation = peslite.Simulation(from_dict(tree))
    with pytest.raises(ConfigError, match="single-rate solver"):
        simulation.run(out_dir=None)


def test_parameter_files_round_trip_a_diode_full_bridge(tmp_path):
    params = from_dict(_tree(LCC_S, 50.0, 1e-5))
    path = tmp_path / "rect.pes"
    peslite.dump(params, path)
    again = peslite.load(path)
    assert again.unit("charger.secondary").ctrl is None
    assert not again.unit("charger.secondary").bridge.controlled
    assert again.unit("charger").link.type == "lcc_s"


def test_two_blocked_ports_use_the_existing_algebraic_solver_without_dummy_controls():
    tree = _tree(SS, 200.0, 5e-6)
    tree["units"]["charger"]["primary"] = _rectifier(400.0)
    tree["simulation"]["output"]["signals"] = False
    sim = peslite.Simulation(from_dict(tree))
    result = sim.run(out_dir=None)
    assert all(unit.ctrl is None for unit in sim.units.values())
    assert all(unit.bridge.get_state()["direction"] == 0.0 for unit in sim.units.values())
    assert result.summary["t_stop"] == pytest.approx(5e-6)
    np.testing.assert_allclose(sim.system.model.get_initial_values(), 0.0, atol=1e-10)


@pytest.mark.parametrize("link", [SS, LCC, LCC_S, MULTI])
def test_blocked_coupled_ports_project_their_currents_jointly_without_internal_supply(link):
    tree = _tree(link, 200.0, 1e-6)
    tree["units"]["charger"]["primary"] = _rectifier(400.0)
    simulation = peslite.Simulation(from_dict(tree))
    model = simulation.system.model
    y = np.random.default_rng(42).normal(size=model.n_states)
    model.project_states(y)
    model.sync(0.0, y)
    coupled = dict(zip(model.names, model.subsystems))["charger.link"]
    assert coupled.out.i_primary == pytest.approx(0.0, abs=1e-13)
    assert coupled.out.i_secondary == pytest.approx(0.0, abs=1e-13)
    for unit in simulation.units.values():
        assert unit.bridge.supplied_power() == 0.0
    assert not simulation.ph_report.problems
    assert all(status.role == "passive" for status in simulation.ph_report.subsystems
               if status.name.endswith(".bridge"))


@pytest.mark.parametrize("link", [SS, LCC, LCC_S, MULTI])
@pytest.mark.parametrize("active", [("primary",), ("secondary",), ("primary", "secondary")])
def test_port_current_projection_matches_weighted_constraint_solve(link, active):
    tree = _tree(link, 200.0, 1e-6)
    tree["units"]["charger"]["primary"] = _rectifier(400.0)
    simulation = peslite.Simulation(from_dict(tree))
    coupled = dict(zip(simulation.system.model.names, simulation.system.model.subsystems))["charger.link"]
    targets = {side: value for side, value in zip(active, (.3, -.7))}
    coupled._current_constraints = {side: (lambda value=value: value) for side, value in targets.items()}
    initial = np.random.default_rng(42).normal(size=len(coupled._mode_names))
    for name, value in zip(coupled._mode_names, initial):
        setattr(coupled.state, name, value)
    modal = [side for side in active if not coupled.sides[side].lcc]
    expected = initial.copy()
    if modal:
        rows = np.array([np.sum([coupled._shapes[k] for k in coupled.sides[side].coils], axis=0)
                         for side in modal])
        weighted = rows / np.asarray(coupled._inductance)
        expected += weighted.T @ np.linalg.solve(
            weighted @ rows.T, np.array([targets[side] for side in modal]) - rows @ initial)
    coupled.project_state()
    np.testing.assert_allclose([getattr(coupled.state, name) for name in coupled._mode_names], expected,
                               rtol=1e-13, atol=1e-13)
    for side in active:
        if coupled.sides[side].lcc:
            assert getattr(coupled.state, f"i_{side}_comp") == targets[side]


def test_nonzero_initial_current_establishes_conduction_before_state_projection():
    tree = _tree(SS, 200.0, 1e-9, dt=1e-10)
    tree["simulation"]["output"]["signals"] = False
    tree["simulation"]["initial"] = {"states": {"charger.link.i_mode_1": 1.0}}
    simulation = peslite.Simulation(from_dict(tree))
    simulation.run(out_dir=None)
    model = simulation.system.model
    link = dict(zip(model.names, model.subsystems))["charger.link"]
    assert abs(link.out.i_secondary) > 0.5
    assert simulation.unit("charger.secondary").bridge.get_state()["direction"] != 0.0


@pytest.mark.parametrize("case", ["ss_dcm", "lcc_dcm", "ss_multi"])
def test_diode_rectification_passes_strict_energy_checks_without_internal_supply(case):
    link, voltage, capacitance = CASES[case]
    tree = _tree(link, voltage, 40e-6, capacitance)
    tree["simulation"]["energy_check"] = "strict"
    tree["simulation"]["solver"]["phs_check_step"] = 1
    tree["simulation"]["output"]["signals"] = False
    simulation = peslite.Simulation(from_dict(tree))
    result = simulation.run(out_dir=None)
    assert "stop_reason" not in result.summary
    assert result.summary["t_stop"] == pytest.approx(40e-6)
    for unit in simulation.units.values():
        if not unit.bridge.topology.controlled:
            assert unit.bridge.supplied_power() == 0.0


def test_two_rectifying_ports_match_an_independent_unforced_coupled_circuit():
    from scipy.linalg import expm

    tree = _tree(SS, 200.0, 20e-6)
    for side in ("primary", "secondary"):
        unit = _rectifier(200.0)
        unit["dclink"]["source"]["v"] = 0.0
        tree["units"]["charger"][side] = unit
    tree["simulation"]["output"]["signals"] = False
    tree["simulation"]["initial"] = {"states": {
        "charger.link.u_primary_cap": 1.0, "charger.link.u_secondary_cap": 2.0}}
    sim = peslite.Simulation(from_dict(tree))
    sim.run(out_dir=None)
    inverse = np.linalg.inv([[L1, M], [M, L2]])
    matrix = np.block([[-inverse * 0.3, -inverse],
                       [np.eye(2) / C_60K, np.zeros((2, 2))]])
    expected = expm(matrix * 20e-6) @ np.array([0.0, 0.0, 1.0, 2.0])
    model = sim.system.model
    link = dict(zip(model.names, model.subsystems))["charger.link"]
    actual = (*link.coil_currents(), link.state.u_primary_cap, link.state.u_secondary_cap)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("endpoint", ["charger", "charger.primary", "charger.primary.dc"])
def test_capacitive_bridge_accepts_series_impedance_at_canonical_terminal_connections(endpoint):
    tree = _tree(SS, 200.0, 1e-6)
    tree["units"]["charger"]["primary"] = _rectifier(400.0, 4.7e-9)
    tree["units"]["charger"]["secondary"] = _primary()
    tree["elements"] = {"resistance": {"type": "resistor", "bus": endpoint,
                                       "r": 2.0, "series": True}}
    simulation = peslite.Simulation(from_dict(tree))
    assert simulation.unit("charger.primary").bridge.state_names == ("u_C",)


def test_shunt_load_does_not_make_a_prescribed_dc_voltage_nonstiff():
    tree = _tree(SS, 200.0, 1e-6)
    tree["units"]["charger"]["primary"] = _rectifier(400.0, 4.7e-9)
    tree["units"]["charger"]["secondary"] = _primary()
    tree["elements"] = {"load": {"type": "resistor", "bus": "charger", "r": 100.0}}
    tree["simulation"]["output"]["signals"] = False
    sim = peslite.Simulation(from_dict(tree))
    sim.run(out_dir=None)
    assert sim.unit("charger.primary").dclink.out.u_dc == 400.0


@pytest.mark.parametrize("local_filter", [True, False])
def test_standalone_rectifier_uses_the_common_rl_terminal_law(local_filter):
    from scipy.linalg import expm

    l, r, c = 1e-3, 0.3, 1e-4
    unit = _rectifier(200.0)
    tree = {"base": _tree(SS, 200.0, 1e-6)["base"],
            "buses": {"storage": {"c": c, "phases": 1}},
            "units": {"rect": unit},
            "simulation": {"t_end": 100e-6,
                           "solver": {"dt": 20e-9, "linearisations": 0},
                           "output": {"period": 1e-6, "states": False,
                                      "signals": False, "energy": False},
                           "initial": {"states": {"storage.u_C": 300.0}}}}
    if local_filter:
        unit.update(bus="storage", filter={"l_f": l, "r_f": r})
        branch_name = "rect.branch_f"
    else:
        tree["branches"] = {"line": {"bus1": "rect.power", "bus2": "storage",
                                     "l": l, "r": r}}
        branch_name = "line"
    sim = peslite.Simulation(from_dict(tree))
    result = sim.run(out_dir=None)
    expected = np.array([0.0, 200.0]) + expm(
        np.array([[-r / l, -1.0 / l], [1.0 / c, 0.0]]) * 100e-6) @ np.array([0.0, 100.0])
    states = result.final_states()
    np.testing.assert_allclose([states[branch_name + ".i"], states["storage.u_C"]],
                               expected, rtol=1e-9, atol=1e-9)
    assert sim.unit().ctrl is None


@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("rms", [True, False])
def test_diode_full_bridge_converter_reuses_adc_sources_means_rms_and_restart(tmp_path, average, rms):
    tree = _tree(SS, 200.0, 12e-6)
    tree["units"]["charger"]["secondary"]["meas"] = {
        "period": 0.5e-6, "window": 0.5e-6, "average": ["current"],
        "rms": {"current": 2e-6}, "sources": {"voltage": "charger.link.u_secondary_cap"}}
    if not rms:
        tree["units"]["charger"]["secondary"]["meas"].pop("rms")
    if not average:
        tree["units"]["charger"]["secondary"]["meas"].pop("window")
        tree["units"]["charger"]["secondary"]["meas"]["average"] = []
    tree["simulation"]["output"].update(states=True, signals=True)
    sim = peslite.Simulation(from_dict(copy.deepcopy(tree)))
    whole = sim.run(out_dir=tmp_path / "whole")
    unit = sim.unit("charger.secondary")
    assert unit.ctrl is None and unit.adc.clocked
    if rms:
        assert np.isfinite(unit.adc.rms["current"].value)
    assert unit.adc.values["current"] != 0.0
    signal = "current_mean" if average else "current"
    assert f"charger.secondary.meas.{signal}" in whole.columns()
    model = sim.system.model
    link = dict(zip(model.names, model.subsystems))["charger.link"]
    assert unit.ports.value("voltage") == link.out.u_secondary_cap
    first = copy.deepcopy(tree)
    first["simulation"]["t_end"] = 8e-6
    peslite.Simulation(from_dict(first)).run(out_dir=tmp_path / "first")
    tree["simulation"]["initial"] = read_initial(tmp_path / "first" / "states.csv")
    continued = peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path / "continued")
    for name, value in whole.final_states().items():
        assert continued.final_states()[name] == pytest.approx(value, rel=1e-8, abs=1e-8)


def test_controllerless_adc_consumes_a_short_averaging_window_boundary():
    tree = _tree(SS, 200.0, 1.9e-6)
    tree["units"]["charger"]["secondary"]["meas"] = {
        "period": 0.5e-6, "window": 0.25e-6, "average": ["current"],
    }
    tree["simulation"]["output"].update(states=False, signals=False, energy=False)
    simulation = peslite.Simulation(from_dict(tree))
    result = simulation.run(out_dir=None)
    adc = simulation.unit("charger.secondary").adc
    assert result.summary["t_stop"] == pytest.approx(1.9e-6)
    assert adc.window_open
    assert adc.next_time() == pytest.approx(2.0e-6)


def test_controllerless_continuation_does_not_repeat_the_saved_clock_edge(tmp_path):
    tree = _tree(SS, 200.0, 1.0e-6)
    tree["simulation"]["output"].update(
        period=0.5e-6, states=True, signals=False, energy=False,
    )
    peslite.Simulation(from_dict(tree)).run(out_dir=tmp_path / "first")

    tree["simulation"]["t_end"] = 1.2e-6
    tree["simulation"]["initial"] = read_initial(tmp_path / "first" / "states.csv")
    simulation = peslite.Simulation(from_dict(tree))
    protection = simulation.unit("charger.secondary").protection
    calls = []
    check_sampled = protection.check_sampled

    def record(t, sample):
        calls.append(t)
        return check_sampled(t, sample)

    protection.check_sampled = record
    simulation.run(out_dir=tmp_path / "continued")
    assert calls == []


@pytest.mark.parametrize("period", [None, 0.5e-6])
def test_diode_full_bridge_adc_window_cannot_exceed_its_actual_sampling_period(period):
    tree = _tree(SS, 200.0, 12e-6)
    meas = {"average": ["current"], "window": 4e-6}
    if period is not None:
        meas["period"] = period
    tree["units"]["charger"]["secondary"]["meas"] = meas
    with pytest.raises(ConfigError, match="longer than the sampling period"):
        from_dict(tree)


def test_capacitor_voltage_stays_continuous_when_the_dc_clamp_increases():
    traces = []
    for stepped in (False, True):
        tree = _tree(dict(LCC, secondary_parallel_resistance=0.0), 150.0, 80e-6,
                     capacitance=4.7e-9)
        tree["simulation"]["output"].update(signals=False, states=False, energy=False)
        if stepped:
            tree["events"] = {"dc_step": {"type": "set", "t": 60e-6,
                              "set": {"units.charger.secondary.dclink.source.v": 180.0}}}
        sim = peslite.Simulation(from_dict(tree))
        bridge = sim.unit("charger.secondary").bridge
        trace = {}

        def observe(t, states, read_values, *, final, stop_reason):
            trace[round(t, 12)] = bridge.state.u_C
            assert bridge.out.u_c == bridge.state.u_C

        sim.run(out_dir=None, observe=observe)
        traces.append(trace)
    assert traces[0][60e-6] == pytest.approx(traces[1][60e-6], abs=1e-8)


def test_lower_dc_voltage_discharges_the_capacitor_through_finite_device_resistance():
    tree = _tree(dict(LCC, secondary_parallel_resistance=0.0), 150.0, 80e-6,
                 capacitance=4.7e-9)
    tree["simulation"]["output"]["signals"] = False
    tree["events"] = {"dc_step": {"type": "set", "t": 60e-6,
                      "set": {"units.charger.secondary.dclink.source.v": 120.0}}}
    simulation = peslite.Simulation(from_dict(tree))
    result = simulation.run(out_dir=None)
    assert "stop_reason" not in result.summary
    assert simulation.unit("charger.secondary").bridge.state.u_C < 150.0


@pytest.mark.parametrize("resistance,stored", [(1.0, False), (75.0, False), (75.0, True)])
def test_capacitive_rectifier_runs_with_ordinary_dc_source_impedance_and_storage(resistance, stored):
    tree = _tree(dict(LCC, secondary_parallel_resistance=0.0), 150.0, 80e-6,
                 capacitance=4.7e-9)
    link = tree["units"]["charger"]["secondary"]["dclink"]
    link["source"]["r"] = resistance
    if stored:
        link["capacitor"] = {"c": 1e-6}
    tree["simulation"]["energy_check"] = "strict"
    tree["simulation"]["solver"]["phs_check_step"] = 1
    tree["simulation"]["output"]["signals"] = False
    simulation = peslite.Simulation(from_dict(tree))
    result = simulation.run(out_dir=None)
    assert "stop_reason" not in result.summary
    assert result.summary["t_stop"] == pytest.approx(80e-6)


def test_custom_export_backend_receives_commutating_models(tmp_path):
    from peslite.assembly.exporter import EXPORTERS, ExportResult

    def snapshot(simulation, destination, name, variables, reusable):
        return ExportResult("test_snapshot", destination, ())

    EXPORTERS["test_snapshot"] = snapshot
    try:
        result = peslite.export(from_dict(_tree(SS, 200.0, 1e-6)), "test_snapshot", tmp_path)
        assert result.format == "test_snapshot"
    finally:
        EXPORTERS.pop("test_snapshot")


def test_blocked_controlled_bridge_carrying_current_is_reported():
    tree = _tree(SS, 200.0, 0.1e-3)
    tree["units"]["charger"]["secondary"] = {
        "type": "dc_converter", "dclink": {"vdc_ref": 200.0, "source": {"type": "voltage"}},
        "pwm": {"f_sw": 1e6}, "bridge": {"model": "pwm_averaging"},
        "ctrl": {"type": "resonant", "computation": 0.0, "loops": {"ResonantModulation": {}},
                 "references": {"modulation": 0.5, "frequency": F, "phase": 1.0}}}
    tree["events"] = {"off": {"type": "disconnect", "target": "charger.secondary", "t": 50e-6}}
    tree["simulation"]["output"]["signals"] = False
    with pytest.warns(UserWarning, match="blocked gates while its port carries"):
        peslite.Simulation(from_dict(tree)).run(out_dir=None)


