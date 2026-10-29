"""The refactored controller, its loop registry and dependency boundary."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

import peslite
from conftest import EXAMPLES
from peslite.assembly.params import from_dict, read_tree, to_dict
from peslite.control import (ANGLE, CONTROL_INTERFACE, FREQUENCY, I, I_AB, I_DQ, LOOP_TYPES,
                             POWER, V, V_AB, V_DQ, Filter, Integrator, Loop, Measurement, PI,
                             SignalType, SyncLaw, complex_power, register_loop_type, unit_delay)
from peslite.solver.model import ConfigError

CONTROL = Path(peslite.control.__file__).parent
QUIET = {"simulation.progress.enable": 0, "simulation.solver.linearisations": 0}
FAST = {"events.connect_vsc.t": 0.0, "events.connect_vsc.ramp": 0.01}


@register_loop_type
class _FixedFrequency(SyncLaw):
    """A registered synchronization law at nominal frequency."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: float
        type: str = "test_fixed_frequency"

    type = "test_fixed_frequency"

    def _sample(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu,
                p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0
        self.theta += T * self.omega
        self.v_mag = v_ref_pu


@register_loop_type
class _AffineFeedback(Loop):
    """A stateless continuous loop used to exercise automatic loop-level feedback solving."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        gain: float
        bias: complex
        period: float | None = None
        type: str = "test_affine_feedback"

    type = "test_affine_feedback"
    inputs = {"x": V_DQ}
    outputs = {"y": V_DQ}

    def initial_outputs(self):
        return {"y": 0j}

    def sample(self, t, inputs):
        return {"y": self.cfg.bias + self.cfg.gain * inputs["x"]}

    def flow(self, t, inputs):
        return self.sample(t, inputs), {}


def test_registered_loop_owns_its_schema_and_uses_default_role_wiring(tmp_path):
    p = peslite.load(
        EXAMPLES / "gfm-psc-example.pes",
        **QUIET,
        **FAST,
        **{"simulation.t_end": 0.01},
    )
    loops = dict(p.unit().ctrl.loops)
    loops.pop("PSC")
    loops["_FixedFrequency"] = _FixedFrequency.Params(period=5e-5)
    p = p.replace(**{
        "units.vsc.bridge.model": "pwm_averaging",
        "units.vsc.ctrl.loops": loops,
    })
    sim = peslite.Simulation(p)
    ctrl = sim.unit().ctrl
    assert type(p.unit().ctrl.loops["_FixedFrequency"]) is _FixedFrequency.Params
    assert ctrl.graph.connections["VirtualImpedance.v_ref"] == "_FixedFrequency.v_ref"
    result = sim.run(out_dir=tmp_path)
    assert result.summary["vsc.law"] == "test_fixed_frequency"
    assert result.states["vsc.ctrl._FixedFrequency.theta"][-1] == pytest.approx(
        2 * np.pi * 50.0 * 0.01, rel=1e-9
    )


def test_sampled_custom_loop_must_supply_its_own_continuous_equations_for_averaging():
    p = peslite.load(
        EXAMPLES / "gfm-psc-example.pes", **QUIET,
        **{"units.vsc.bridge.model": "averaging"},
    )
    loops = dict(p.unit().ctrl.loops)
    loops.pop("PSC")
    loops["_FixedFrequency"] = _FixedFrequency.Params(period=5e-5)
    p = p.replace(**{"units.vsc.ctrl.loops": loops})
    with pytest.raises(ConfigError, match="no continuous-time flow implementation"):
        peslite.Simulation(p)


def test_continuous_control_graph_automatically_solves_feedback_between_loops():
    p = peslite.load(
        EXAMPLES / "gfl-example.pes", **QUIET,
        **{"units.vsc.bridge.model": "averaging"},
    )
    unit = peslite.Simulation(p).unit().cfg
    loops = {
        **unit.ctrl.loops,
        "_AffineFeedback[a]": _AffineFeedback.Params(gain=0.25, bias=1.0),
        "_AffineFeedback[b]": _AffineFeedback.Params(gain=0.5, bias=2.0),
    }
    connections = {
        **unit.ctrl.connections,
        "_AffineFeedback[a].x": "_AffineFeedback[b].y",
        "_AffineFeedback[b].x": "_AffineFeedback[a].y",
    }
    configured = replace(
        unit, ctrl=replace(
            unit.ctrl,
            loops=loops,
            connections=connections,
            outputs={"u_dq": "_AffineFeedback[a].y"},
        )
    )
    object.__setattr__(configured, "base", unit.base)
    controller = peslite.VSCController(configured)

    controller.set_outputs(0.0)

    assert controller.graph.values["_AffineFeedback[a].y"] == pytest.approx(12.0 / 7.0)
    assert controller.graph.values["_AffineFeedback[b].y"] == pytest.approx(20.0 / 7.0)


def test_repeated_loop_class_uses_qualified_names_without_type_fields():
    tree = read_tree(EXAMPLES / "gfl-example.pes")
    unit = tree["units"]["vsc"]
    unit["bridge"] = {"model": "averaging"}
    unit["ctrl"] = {
        "type": "custom",
        "loops": {
            "_AffineFeedback[a]": {"gain": 0.25, "bias": 1.0},
            "_AffineFeedback[b]": {"gain": 0.5, "bias": 2.0},
        },
        "connections": {
            "_AffineFeedback[a].x": "_AffineFeedback[b].y",
            "_AffineFeedback[b].x": "_AffineFeedback[a].y",
        },
        "outputs": {
            "u_dq": "_AffineFeedback[a].y",
            "theta": "references.theta",
            "omega": "references.omega",
        },
    }

    params = from_dict(tree)
    simulation = peslite.Simulation(params)

    assert set(simulation.unit().ctrl.graph.nodes) == {
        "_AffineFeedback[a]", "_AffineFeedback[b]",
    }
    assert all("type" not in entry
               for entry in simulation.unit().ctrl.graph.describe()["loops"].values())
    written = to_dict(params)["units"]["vsc"]["ctrl"]["loops"]
    assert all("type" not in entry for entry in written.values())


def test_a_declared_loop_outside_the_output_dependency_graph_is_not_constructed():
    tree = read_tree(EXAMPLES / "gfl-example.pes")
    tree["units"]["vsc"]["ctrl"]["loops"]["_AffineFeedback[unused]"] = {
        "gain": 0.0,
        "bias": 0.0,
    }
    simulation = peslite.Simulation(from_dict(tree))

    assert "_AffineFeedback[unused]" in simulation.unit().cfg.ctrl.loops
    assert "_AffineFeedback[unused]" not in simulation.unit().ctrl.graph.nodes
    assert not any(target.startswith("_AffineFeedback[unused].")
                   for target in simulation.unit().ctrl.graph.connections)


def test_power_and_delay_are_functions_with_caller_owned_state():
    assert complex_power(2 + 3j, 4 - 1j) == (2 + 3j) * (4 - 1j).conjugate()
    output, state = unit_delay(2.0, 1.0)
    assert (output, state) == (1.0, 2.0)
    output, state = unit_delay(3.0, state)
    assert (output, state) == (2.0, 3.0)
    assert {"power", "unit_delay"}.isdisjoint(LOOP_TYPES)


def test_enabled_sync_power_filter_requires_a_bandwidth():
    with pytest.raises(ConfigError, match="enabled power filter needs a bandwidth"):
        peslite.load(
            EXAMPLES / "gfm-psc-example.pes",
            **{"units.vsc.ctrl.loops.PSC.power_filter.enable": 1},
        )


def test_loop_registration_and_loop_owned_validation():
    class NoParams(Loop):
        type = "test_no_params"

    with pytest.raises(TypeError):
        register_loop_type(NoParams)
    with pytest.raises(ValueError):
        register_loop_type(LOOP_TYPES["psc"])
    params = peslite.load(EXAMPLES / "gfm-psc-example.pes")
    with pytest.raises(ConfigError, match="loops.vi.x_v_pu must be positive"):
        params.replace(**{"units.vsc.ctrl.loops.vi": {
            "type": "virtual_admittance", "x_v_pu": 0.0,
        }})


def test_adc_sample_and_controller_measurement_have_distinct_scopes():
    from peslite.components.adc import ADCSample

    assert peslite.ADCSample is ADCSample
    assert Measurement.__module__ == "peslite.control.controller"


def test_control_graph_has_one_typed_boundary_for_inputs_and_outputs(gfl):
    graph = peslite.Simulation(gfl()).unit().ctrl.graph

    assert graph.interface is CONTROL_INTERFACE
    assert set(CONTROL_INTERFACE.inputs) == {
        "meas.u_g", "meas.i_c", "meas.u_dc",
        "references.id_ref_pu", "references.iq_ref_pu", "references.theta",
        "references.omega", "references.p_ref_pu", "references.q_ref_pu",
        "references.v_ref_pu", "references.vdc_ref_pu", "references.zero_v_pu",
    }
    assert set(CONTROL_INTERFACE.outputs) == {"u_dq", "theta", "omega"}
    assert set(graph._boundary_sources) == set(CONTROL_INTERFACE.inputs)
    for port, source in graph.outputs.items():
        assert port in CONTROL_INTERFACE.outputs
        owner, _, source_port = source.partition(".")
        assert graph.nodes[owner].outputs[source_port] == CONTROL_INTERFACE.outputs[port]


@pytest.mark.parametrize("model", ["pwm_averaging", "averaging"])
def test_reference_changes_rewire_graph_outputs_and_frame(gfl, model):
    params = gfl(**{"units.vsc.bridge.model": model})
    loops = {
        name: loop
        for name, loop in params.unit("vsc").ctrl.loops.items()
        if name != "SRFPLL"
    }
    params = params.replace(**{"units.vsc.ctrl.loops": loops})
    cfg = params.unit("vsc")
    controller = peslite.VSCController(cfg)
    moved = params.replace(**{
        "units.vsc.ctrl.references.theta": 0.5,
        "units.vsc.ctrl.references.omega": 300.0,
    })

    controller.retune(
        moved.unit("vsc"),
        ["ctrl.references.theta", "ctrl.references.omega"],
        0.0,
    )
    if controller.continuous:
        controller.inp.u_g = cfg.base.v_phase_peak + 0j
        controller.set_outputs(0.0)
    else:
        controller(0.0, Measurement(
            0.0, cfg.base.v_phase_peak + 0j, 0j, cfg.dclink.vdc_ref, np.zeros(3),
        ))

    assert (controller.theta, controller.command_theta, controller.omega) == (0.5, 0.5, 300.0)
    assert controller.v_dq == pytest.approx(np.exp(-0.5j))


def test_control_ports_have_the_canonical_signal_types():
    signals = (I_AB, V_AB, I_DQ, V_DQ, I, V, ANGLE, FREQUENCY, POWER)

    assert all(isinstance(signal, SignalType) for signal in signals)
    assert len(set(signals)) == 9
    assert {signal for signal in signals if signal.complex_value} == {
        I_AB, V_AB, I_DQ, V_DQ,
    }
    assert not any(hasattr(peslite.control, old) for old in (
        "CURRENT", "VOLTAGE", "DC_VOLTAGE", "POWER_PU", "PQ", "SignalPort",
    ))


def test_pi_owns_tracking_antiwindup_but_not_the_output_limit():
    pi = PI(2.0, 4.0, 0.1)
    intended = pi(2.0, 0.0)
    integrated = pi.integral

    assert intended == pytest.approx(4.4)
    assert integrated == pytest.approx(0.2)
    pi.antiwindup(intended, 1.0)
    assert pi.integral == pytest.approx(integrated + 0.1 * (1.0 - intended) / 2.0)

    disabled = PI(2.0, 4.0, 0.1, antiwindup=False)
    intended = disabled(2.0, 0.0)
    integrated = disabled.integral
    disabled.antiwindup(intended, 1.0)
    assert disabled.integral == integrated


def test_integrator_binds_continuous_or_trapezoidal_sampled_semantics_at_construction():
    sampled = Integrator(0.1, initial=1.0)
    continuous = Integrator(None, initial=1.0)

    assert sampled(2.0) == pytest.approx(1.1)
    assert sampled(4.0) == pytest.approx(1.4)
    assert sampled.value == pytest.approx(1.6)
    assert continuous(2.0) == pytest.approx(1.0)
    assert continuous.value == pytest.approx(1.0)
    assert continuous.derivative == pytest.approx(2.0)


def test_filter_uses_descending_s_coefficients_and_tustin_for_sampled_control():
    # 10 / (s + 10): Tustin at T=0.1 gives y0=1/3 and y1=7/9 for a unit step.
    sampled = Filter((10.0,), (1.0, 10.0), 0.1)
    continuous = Filter((10.0,), (1.0, 10.0), None)

    assert sampled(1.0) == pytest.approx(1.0 / 3.0)
    assert sampled(1.0) == pytest.approx(7.0 / 9.0)
    assert continuous(1.0) == pytest.approx(0.0)
    assert continuous.derivative == pytest.approx((10.0,))

    with pytest.raises(ValueError, match="must be proper"):
        Filter((1.0, 0.0), (1.0,), 0.1)


def test_dc_voltage_control_owns_its_limit_in_both_execution_modes(gfl):
    sampled = peslite.Simulation(gfl()).unit().ctrl
    continuous = peslite.Simulation(gfl(**{"units.vsc.bridge.model": "averaging"})).unit().ctrl

    assert set(sampled.graph._limits) == {"CurrentLoop"}
    assert continuous.graph._limits == {}
    assert set(continuous._flow_limits) == {"DCVoltageControl"}
    assert (sampled.graph._limits["CurrentLoop"][1].pi
            is sampled.graph.nodes["CurrentLoop"].pi)
    assert (sampled.graph.nodes["DCVoltageControl"].limit.pi
            is sampled.graph.nodes["DCVoltageControl"].pi)
    assert (continuous.graph.nodes["DCVoltageControl"].limit.pi
            is continuous.graph.nodes["DCVoltageControl"].pi)


def test_averaging_clips_dc_current_reference_and_tracks_antiwindup(gfl):
    ctrl = peslite.Simulation(gfl(**{"units.vsc.bridge.model": "averaging"})).unit().ctrl
    dvc = ctrl.graph.nodes["DCVoltageControl"]
    ctrl.command_at(True, 0.0, 0.0)
    intended = 2.0 * dvc.maximum
    integral = (intended - dvc.cfg.feedforward) / dvc.ki
    setattr(ctrl.state, "DCVoltageControl.integral", integral)
    ctrl.inp.u_dc = ctrl.p.dclink.vdc_ref

    ctrl.set_outputs(0.0)

    assert dvc.raw == pytest.approx(intended)
    assert ctrl.graph.values["DCVoltageControl.current_reference"] == pytest.approx(dvc.maximum)
    derivative = ctrl.rhs(0.0)[ctrl.state_names.index("DCVoltageControl.integral")]
    assert derivative == pytest.approx((dvc.maximum - intended) / dvc.kp)
    assert ctrl.alarms == []
    ctrl.observe_limits(0.0)
    assert ctrl.alarms == ["ID_REF_LIMIT"]


def test_unknown_auxiliary_output_source_is_rejected(gfl):
    params = gfl().replace(**{"units.vsc.ctrl.outputs": {"status": "cc.u_dq"}})
    with pytest.raises(ConfigError, match="ctrl.outputs.status: missing or unknown source"):
        peslite.Simulation(params)


def test_control_imports_only_itself_and_the_solver_kernel():
    allowed = {"control", "solver"}
    for path in CONTROL.glob("*.py"):
        imported = set()
        rel = path.relative_to(CONTROL.parent).with_suffix("")
        here = list(rel.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = here[:len(here) - (node.level - 1)]
                target = base + (node.module.split(".") if node.module else [])
                if target:
                    imported.add(target[0])
        assert imported <= allowed, f"{path.name} imports {sorted(imported - allowed)}"
