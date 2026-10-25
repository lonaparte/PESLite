"""Full-system AC checks use the shipped converters and their ordinary simulation loop.

The scenarios have constant commands after a 50 ms connection ramp. GFL injects 0.3 pu DC
power, PSC requests 0.5 pu, VSG requests 0.2 pu, and the shared-bus pair requests 0.3 pu per
converter. These are specified test operating points, not claims that every example settles
under arbitrary loading. The original 20 kHz PWM, computation delay and electrical network
are retained; no separate small-signal plant is constructed.
"""

import copy
import dataclasses
from itertools import combinations

import numpy as np
import peslite
import pytest
from scipy.optimize import linear_sum_assignment

from conftest import EXAMPLES
from peslite.addons.functions.small_signal import small_signal, small_signal_report
from peslite.assembly.params import from_dict, to_dict
from peslite.solver.linearization import (active_coordinates, coordinate_difference, coordinate_width,
                                         linearize_loop)
from peslite.solver.simulation import SystemLoop


@pytest.fixture(scope="module")
def ac_parameters():
    common = {
        "simulation.t_end": 4.0, "simulation.energy_check": "off",
        "simulation.solver.linearisations": 0, "simulation.solver.subsystems": {},
        "simulation.output.states": False, "simulation.output.signals": False,
        "simulation.output.energy": False, "simulation.progress.enable": False,
        "units.vsc.bridge.model": "pwm_averaging",
    }
    specifications = {
        "gfl": ("gfl-example.pes", {"units.vsc.dclink.source.i_pu": 0.3,
                                    "units.vsc.ctrl.loops.DCVoltageControl.feedforward": 0.3}),
        "psc": ("gfm-psc-example.pes", {}),
        "vsg": ("gfm-vsg-example.pes", {"units.vsc.ctrl.references.p_ref_pu": 0.2}),
    }
    trees = {name: to_dict(peslite.load(EXAMPLES / filename, **{**common, **changes}))
             for name, (filename, changes) in specifications.items()}
    mixed = copy.deepcopy(trees["gfl"])
    mixed["units"] = {"gfl": mixed["units"]["vsc"],
                      "gfm": copy.deepcopy(trees["psc"]["units"]["vsc"])}
    mixed["units"]["gfm"]["ctrl"]["references"]["p_ref_pu"] = 0.3
    trees["mixed"] = mixed
    for tree in trees.values():
        tree["events"] = {f"connect_{name}": {"type": "connect", "target": name,
                                             "t": 0.0, "ramp": 0.05}
                          for name in tree["units"]}
    return {name: from_dict(tree) for name, tree in trees.items()}


@pytest.fixture(scope="module")
def ac_points(ac_parameters):
    cache = {}

    def point(name):
        if name not in cache:
            sim = peslite.Simulation(ac_parameters[name])
            sim.run(out_dir=None)
            assert not sim.result.tripped and not sim.result.summary.get("stop_reason")
            before = sim.result.final_states()
            result = small_signal(sim)
            assert sim.result.final_states() == before
            assert result.stability == "stable"
            cache[name] = sim, result
        return cache[name]

    return point


@pytest.fixture(scope="module", params=("gfl", "psc", "vsg", "mixed"))
def ac_point(request, ac_points):
    return ac_points(request.param)


def _restart(sim, duration, **changes):
    """Exercise the public named-state continuation path, not a test-specific solver."""
    row = sim.result.final_states()
    start = row.pop("t")
    row = {name: value for name, value in row.items() if not name.startswith("solver.")}
    params = sim.p.replace(**changes) if changes else sim.p
    initial = dataclasses.replace(params.simulation.initial, t=start, states=row)
    params = dataclasses.replace(params, simulation=dataclasses.replace(
        params.simulation, initial=initial, t_end=start + duration))
    return peslite.Simulation(params)


def _mode_error(a, b):
    """Compare an unordered spectrum without assuming a conjugate-pair ordering."""
    distance = np.abs(a[:, None] - b[None, :])
    rows, columns = linear_sum_assignment(distance)
    return float(np.max(distance[rows, columns]))


def test_complete_ac_map_retains_control_and_pwm_memory(ac_point, record_property):
    sim, analysis = ac_point
    record_property("periodic_error", analysis.periodic_error)
    record_property("spectral_radius", float(np.max(np.abs(analysis.multipliers))))
    record_property("coordinates", len(analysis.coordinates))
    record_property("evaluations", analysis.evaluations)
    assert analysis.period == pytest.approx(1.0 / sim.p.sources["grid"].f)
    assert analysis.periodic_error <= 1e-3
    assert analysis.evaluations < 1 + 2 * len(analysis.coordinates)
    assert np.all(np.isfinite(analysis.phi))
    for name, unit in sim.units.items():
        assert unit.cfg.ctrl.computation > 0.0
        assert any(key.startswith(f"{name}.ctrl.") for key in analysis.coordinates)
        assert any(key.startswith(f"{name}.pwm.shadow.") for key in analysis.coordinates)
    assert not any(".prot." in key or ".startup." in key for key in analysis.coordinates)
    for a, b in combinations(sim.units, 2):
        first = [i for i, name in enumerate(analysis.coordinates) if name.startswith(a + ".")]
        second = [i for i, name in enumerate(analysis.coordinates) if name.startswith(b + ".")]
        assert np.linalg.norm(analysis.phi[np.ix_(first, second)]) > 1e-3
        assert np.linalg.norm(analysis.phi[np.ix_(second, first)]) > 1e-3


def test_ac_pruning_preserves_the_entire_exhaustive_matrix(ac_point, exhaustive_loop, record_property):
    sim, analysis = ac_point
    loop = SystemLoop(sim, frame_frequency=0.0, repeat_forcing=True)
    row = sim.result.final_states()
    start = row.pop("t")
    full = linearize_loop(exhaustive_loop(loop), row, start, loop.coordinates(list(row)))
    expected = full.phi[np.ix_(full.active, full.active)]
    np.testing.assert_array_equal(analysis.phi, expected)
    modes = np.linalg.eigvals(expected).astype(complex)
    modes = modes[np.argsort(-np.abs(modes), kind="stable")]
    np.testing.assert_array_equal(analysis.multipliers, modes)
    record_property("exhaustive_evaluations", full.evaluations)
    record_property("optimized_evaluations", analysis.evaluations)
    assert analysis.evaluations < full.evaluations
    skipped = loop.overwritten_columns(start)
    for name, unit in sim.units.items():
        graph = unit.ctrl.graph
        for port in ("theta", "omega"):
            source = graph.outputs[port]
            if source in graph.values:
                assert f"{name}.ctrl.held.{graph._state_port(source)}" not in skipped


@pytest.mark.parametrize("changes,offset", [
    ({"units.vsc.ctrl.loops.DCVoltageControl.period": 200e-6}, 0.02),
    ({"units.vsc.ctrl.loops.CurrentLoop.period": 200e-6}, 0.02),
    ({"units.vsc.ctrl.loops.SRFPLL.period": 200e-6}, 0.02),
    ({"units.vsc.pwm.carrier_phase": 0.25}, 12.5e-6),
    ({"units.vsc.pwm.sync": "synchronous"}, 0.02),
    ({"units.vsc.ctrl.computation": 0.0}, 0.02),
    ({"units.vsc.pwm.update": "double"}, 0.02),
], ids=("slow-voltage", "slow-current", "slow-pll", "shifted-clock", "synchronous",
        "zero-delay", "double-update"))
def test_ac_clock_variants_match_exhaustive_local_maps(ac_points, changes, offset,
                                                      exhaustive_loop):
    # These are local-map comparisons; no settled-stability claim is made after changing clocks.
    original, _analysis = ac_points("gfl")
    sim = _restart(original, offset, **changes)
    sim.run(out_dir=None)
    assert not sim.result.tripped and not sim.result.summary.get("stop_reason")
    loop = SystemLoop(sim, frame_frequency=0.0, repeat_forcing=True)
    row = sim.result.final_states()
    start = row.pop("t")
    cols = loop.coordinates(list(row))
    optimized = linearize_loop(loop, row, start, cols)
    full = linearize_loop(exhaustive_loop(loop), row, start, cols)
    np.testing.assert_array_equal(optimized.phi, full.phi)
    assert optimized.changed_modes == full.changed_modes
    assert optimized.evaluations < full.evaluations


def test_coupled_converters_with_different_clock_phases_match_exhaustive_maps(ac_points,
                                                                            exhaustive_loop):
    original, _analysis = ac_points("mixed")
    sim = _restart(original, 0.02, **{
        "units.gfl.ctrl.loops.DCVoltageControl.period": 200e-6,
        "units.gfm.pwm.carrier_phase": 0.25,
    })
    sim.run(out_dir=None)
    assert not sim.result.tripped and not sim.result.summary.get("stop_reason")
    loop = SystemLoop(sim, frame_frequency=0.0, repeat_forcing=True)
    row = sim.result.final_states()
    start = row.pop("t")
    cols = loop.coordinates(list(row))
    optimized = linearize_loop(loop, row, start, cols)
    full = linearize_loop(exhaustive_loop(loop), row, start, cols)
    np.testing.assert_array_equal(optimized.phi, full.phi)
    assert optimized.changed_modes == full.changed_modes
    assert optimized.evaluations < full.evaluations


def test_ac_period_restart_matches_the_main_time_domain_simulation(ac_point):
    sim, analysis = ac_point
    row = sim.result.final_states()
    start = row.pop("t")
    loop = SystemLoop(sim, frame_frequency=0.0, repeat_forcing=True)
    first = loop.advance(row, start)
    second = loop.advance(row, start)
    assert first == second
    continued = _restart(sim, analysis.period)
    continued.run(out_dir=None)
    final = continued.result.final_states()
    final.pop("t")
    assert first == final


def test_ac_modal_prediction_matches_an_independent_perturbation_direction(ac_point, record_property):
    sim, analysis = ac_point
    loop = SystemLoop(sim, frame_frequency=0.0, repeat_forcing=True)
    row = sim.result.final_states()
    start = row.pop("t")
    cols = loop.coordinates(list(row))
    full = loop.to_frame(row, start, cols)
    active = list(active_coordinates(cols))
    scales = np.maximum(1.0, np.abs(full))
    j = 0
    for kind, _names, floor in cols:
        if kind == "angle":
            scales[j] = floor
        j += coordinate_width(kind)
    scales = scales[active]
    direction = np.sin(np.arange(len(active)) + 0.5) * scales * 1e-6
    perturbed = full.copy()
    perturbed[active] += direction
    nominal = loop.to_frame(loop.advance(row, start), start + loop.period, cols)
    changed = loop.to_frame(loop.advance(loop.from_frame(perturbed, start, cols, row), start),
                            start + loop.period, cols)
    actual = coordinate_difference(changed, nominal, cols)[active]
    predicted = analysis.phi @ direction
    error = np.linalg.norm((actual - predicted) / scales) / np.linalg.norm(predicted / scales)
    record_property("directional_relative_error", float(error))
    assert error < 2e-3


def test_ac_online_report_uses_the_same_settled_operating_state(ac_point):
    sim, analysis = ac_point
    # A fresh run uses the same validated initial state; no stored CSV trajectory is read.
    online = _restart(sim, analysis.period)
    report = small_signal_report(online, out_dir=None)
    values = report.changes[0]["changes"]
    assert values["analysis.stability"] == analysis.stability
    assert values["analysis.period"] == analysis.period
    assert report.reused_baselines == 1
    assert not any(entry["changes"].get("analysis.stability") in ("transient", "unavailable")
                   for entry in report.changes)
    assert values["analysis.growth_rate"] == pytest.approx(analysis.exponents[0].real,
                                                         rel=2e-3, abs=2e-3)


def test_ac_spectrum_is_consistent_at_other_phases_and_periods(ac_point, record_property):
    sim, analysis = ac_point
    for offset in (analysis.period / 4.0, analysis.period):
        restarted = _restart(sim, offset)
        restarted.run(out_dir=None)
        shifted = small_signal(restarted)
        assert shifted.stability == analysis.stability
        error = _mode_error(analysis.multipliers, shifted.multipliers)
        record_property(f"mode_error_after_{offset:g}_s", error)
        assert error < 2e-3


def test_sampled_ac_map_converges_under_fixed_step_and_difference_refinement(ac_points, record_property):
    sim, analysis = ac_points("gfl")
    refined = []
    for dt in (12.5e-6, 6.25e-6):
        # Change future integration settings, not the operating point being differentiated.
        numerical_view = copy.copy(sim)
        numerical_view.p = sim.p.replace(**{"simulation.solver.dt": dt})
        refined.append(small_signal(numerical_view))
    coarse_error = np.linalg.norm(analysis.phi - refined[-1].phi)
    fine_error = np.linalg.norm(refined[0].phi - refined[-1].phi)
    record_property("coarse_phi_error_to_finest", float(coarse_error))
    record_property("fine_phi_error_to_finest", float(fine_error))
    assert fine_error < coarse_error / 4.0
    assert _mode_error(analysis.multipliers, refined[-1].multipliers) < 2e-3
    for step in (5e-5, 2.5e-5):
        smaller = small_signal(sim, relative_step=step)
        assert _mode_error(analysis.multipliers, smaller.multipliers) < 2e-3
        assert smaller.stability == analysis.stability
