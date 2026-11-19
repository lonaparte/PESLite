"""File-backed control-graph debugging through the ordinary PESLite runtime."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import peslite
import pytest

from conftest import EXAMPLES
from peslite.addons.functions.debug import (
    DebugOperation, DebugPlan, _analyses, execute_debug, parse_debug_text, prepare_debug,
    read_debug_plan,
)
from peslite.solver.model import ConfigError


MODEL = EXAMPLES / "gfl-example.pes"


def _bypass(*, policy="continue", probes=None, operations=()):
    return DebugPlan(
        model=str(MODEL.resolve()), state_policy=policy,
        operations=(
            DebugOperation("disconnect", source="vsc.CurrentLoop.u_dq",
                           target="vsc.ThreePhaseModulation.u_dq"),
            DebugOperation("connect", source="Test.out",
                           target="vsc.ThreePhaseModulation.u_dq"),
            *operations,
        ),
        sources={"Test": {"type": "sine", "amplitude": 0.01, "frequency": 500.0}},
        probes=probes or {"effective": "vsc.ThreePhaseModulation.u_dq"},
    )


def _short_params():
    return peslite.load(MODEL, **{
        "simulation.t_end": 0.0005,
        "simulation.energy_check": "off",
        "simulation.progress.enable": 0,
        "simulation.solver.linearisations": 0,
    })


def test_plan_round_trip_and_strict_sections():
    plan = _bypass(
        operations=(DebugOperation("set", path="simulation.t_end", value=0.002),),
        probes={"effective": "vsc.ThreePhaseModulation.u_dq",
                "integral": "state.vsc.ctrl.CurrentLoop.integral_pu"},
    )
    assert parse_debug_text(plan.to_text()) == plan

    with pytest.raises(ConfigError, match="duplicate @probe"):
        parse_debug_text("model: case.pes\n@probe\na: A.out\n@probe\nb: B.out\n")
    with pytest.raises(ConfigError, match="state_policy"):
        parse_debug_text("model: case.pes\nstate_policy: discard\n")


@pytest.mark.parametrize(
    "name", ("gfl-debug-example.pesdbg", "buck-control-injection-example.pesdbg"),
)
def test_bundled_debug_plans_use_the_generic_overlay(name):
    path = EXAMPLES / "debug" / name
    plan = read_debug_plan(path)
    model = (path.parent / plan.model).resolve()

    prepared = prepare_debug(peslite.load(model), plan, directory=path.parent)

    assert prepared.watch


def test_plan_normalization_keeps_only_the_latest_value_and_round_trips_matrices():
    matrix = [list(range(30)), list(range(30, 60))]
    plan = DebugPlan(operations=(
        DebugOperation("set", path="simulation.t_end", value=0.001),
        DebugOperation("set", path="simulation.t_end", value=0.002),
    ))
    prepared = prepare_debug(_short_params(), plan, directory=MODEL.parent)

    assert prepared.parameter_overrides["simulation.t_end"] == 0.002
    assert [item.path for item in prepared.operations].count("simulation.t_end") == 1
    saved = DebugPlan(operations=(
        DebugOperation("set", path="units.vsc.ctrl.loops.MPC.tracking_weight", value=matrix),
    )).to_text()
    assert parse_debug_text(saved).operations[-1].value == matrix


def test_programmatic_and_cli_plan_names_follow_file_validation():
    plan = replace(_bypass(), probes={"invalid.alias": "vsc.CurrentLoop.u_dq"})
    with pytest.raises(ConfigError, match="simple names"):
        prepare_debug(_short_params(), plan, directory=MODEL.parent)


def test_overlay_reuses_parameter_replacement_and_the_typed_control_graph():
    original = _short_params()
    plan = _bypass(
        policy="freeze",
        operations=(DebugOperation("set", path="simulation.t_end", value=0.0006),),
    )
    prepared = prepare_debug(original, plan, directory=MODEL.parent)

    assert original.simulation.t_end == pytest.approx(0.0005)
    assert prepared.params.simulation.t_end == pytest.approx(0.0006)
    assert prepared.parameter_overrides == {"simulation.t_end": 0.0006}
    assert original.unit("vsc").ctrl._control_overlay is None

    graph = peslite.Simulation(prepared.params).system.units["vsc"].ctrl.graph
    assert graph.connections["ThreePhaseModulation.u_dq"] == "debug.Test"
    assert graph._frozen == {"CurrentLoop"}
    assert graph.probe_names == ("effective.re", "effective.im", "effective.abs")


def test_invalid_overlay_is_rejected_before_simulation():
    plan = DebugPlan(operations=(
        DebugOperation("connect", source="Test.out",
                       target="vsc.ThreePhaseModulation.u_dq"),
    ), sources={"Test": 0.1})
    with pytest.raises(ConfigError, match="already connected"):
        prepare_debug(_short_params(), plan, directory=MODEL.parent)


def test_loop_gain_analysis_applies_the_feedback_sign_at_a_generic_cut():
    time = np.arange(0.0, 1.0, 0.001)
    after = np.cos(math.tau * 20.0 * time)
    before = -3.0 * after
    plan = DebugPlan(analyses={
        "return_ratio": {
            "type": "loop_gain", "input": "after", "output": "before",
            "frequency": 20.0,
        },
    })

    report = _analyses(plan, None, {
        "t": time.tolist(), "before": before.tolist(), "after": after.tolist(),
    })
    point = report["return_ratio"]["result"]["return_ratio"]["value"][0]

    assert point == {"real": pytest.approx(3.0), "imag": pytest.approx(0.0)}


def test_tracking_policy_uses_the_loop_tracking_law():
    model = EXAMPLES / "gfm-psc-example.pes"
    params = peslite.load(model, **{
        "simulation.t_end": 0.0005,
        "simulation.energy_check": "off",
        "simulation.progress.enable": 0,
    })
    plan = DebugPlan(
        state_policy="tracking",
        operations=(
            DebugOperation("disconnect", source="vsc.PSC.theta",
                           target="vsc.ThreePhaseModulation.theta"),
            DebugOperation("connect", source="Test.out",
                           target="vsc.ThreePhaseModulation.theta"),
        ),
        sources={"Test": 0.0},
    )
    prepared = prepare_debug(params, plan, directory=model.parent)
    graph = peslite.Simulation(prepared.params).system.units["vsc"].ctrl.graph
    assert graph._tracking == {"PSC"}
    assert graph.connections["ThreePhaseModulation.theta"] == "debug.Test"


def test_debug_run_records_control_signals_states_and_only_two_artifacts(tmp_path):
    plan = _bypass(
        probes={"original": "vsc.CurrentLoop.u_dq",
                "injected": "vsc.Test.out",
                "effective": "vsc.ThreePhaseModulation.u_dq",
                "integral": "state.vsc.ctrl.CurrentLoop.integral_pu"},
        operations=(DebugOperation("set", path="CurrentLoop.bandwidth", value=210.0),),
    )
    plan = replace(plan, checks={
        "bounded": {"signal": "effective.abs", "metric": "max",
                    "operator": "<=", "value": 0.02},
    })
    run, report = execute_debug(MODEL, _short_params(), "", plan, output=tmp_path)

    assert report["status"] == "passed"
    assert report["checks"]["bounded"]["passed"] is True
    assert report["metrics"]["bounded"]["metric"] == "max"
    assert report["evidence"]["sample_count"] == len(report["data"]["t"])
    assert {"original.re", "injected.re", "effective.re", "integral.re"} <= report["data"].keys()
    assert report["plan"]["effective_configuration"]["simulation"]["t_end"] == 0.0005
    assert {path.name for path in run.iterdir()} == {"debug.pesdbg", "report.json"}
    assert "~ units.vsc.ctrl.loops.CurrentLoop.bandwidth = 210.0" in (
        run / "debug.pesdbg"
    ).read_text(encoding="utf-8")
    assert json.loads((run / "report.json").read_text(encoding="utf-8"))["status"] == "passed"


def test_debug_cli_json_and_history_use_the_same_run(tmp_path, monkeypatch, capsys):
    from peslite.solver import simulation as cli

    monkeypatch.setattr(cli, "_RESULTS", tmp_path / "output")
    plan_path = tmp_path / "control.pesdbg"
    plan_path.write_text(_bypass().to_text(), encoding="utf-8")

    assert peslite.main(["debug", str(MODEL), "--plan", str(plan_path),
                         "--set", "simulation.t_end=0.0005",
                         "--set", "simulation.energy_check=off",
                         "--set", "simulation.solver.linearisations=0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "debug" and payload["debug"]["status"] == "passed"
    report = Path(payload["output"]["summary"])
    assert report.name == "report.json" and report.is_file()

    assert peslite.main(["debug", str(MODEL),
                         "--set", "simulation.t_end=0.0005",
                         "--set", "simulation.energy_check=off",
                         "--set", "simulation.solver.linearisations=0", "--json"]) == 0
    inherited = json.loads(capsys.readouterr().out)
    inherited_plan = Path(inherited["output"]["files"][0]).read_text(encoding="utf-8")
    assert "- vsc.CurrentLoop.u_dq -> vsc.ThreePhaseModulation.u_dq" in inherited_plan

    assert peslite.main(["debug", "ls", str(MODEL), "--json"]) == 0
    history = json.loads(capsys.readouterr().out)
    assert len(history["runs"]) == 2
    assert str(report) in {entry["report"] for entry in history["runs"]}
