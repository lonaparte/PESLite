"""Optional ONNX Runtime compatibility tests for the framework-neutral black-box boundary."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper

import peslite
from peslite import BlackBoxSubsystem
from peslite.solver import Model
from peslite.solver.simulation import SystemLoop


pytestmark = [pytest.mark.full, pytest.mark.onnx]


def _session(nodes, inputs, outputs, initializers=()):
    return ort.InferenceSession(
        _model(nodes, inputs, outputs, initializers).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )


def _model(nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, "peslite-black-box", inputs, outputs, initializer=initializers)
    model = helper.make_model(
        graph,
        ir_version=9,
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    return model


def _value(name, size):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [size])


def test_onnx_stateless_model_uses_the_plain_array_callable_boundary():
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [2], [2.0, -3.0])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [2], [0.5, 1.0])
    session = _session(
        [helper.make_node("Mul", ["input", "scale"], ["scaled"]),
         helper.make_node("Add", ["scaled", "bias"], ["output"])],
        [_value("input", 2)],
        [_value("output", 2)],
        [scale, bias],
    )
    block = BlackBoxSubsystem(
        lambda _t, inputs: session.run(None, {"input": inputs.astype(np.float32)})[0],
        inputs=("u0", "u1"),
        outputs=("y0", "y1"),
    )
    block.inp.u0, block.inp.u1 = 1.5, -2.0

    Model({"onnx": block}, {}).evaluate_outputs(0.0)

    assert (block.out.y0, block.out.y1) == pytest.approx((3.5, 7.0))


def test_onnx_continuous_model_receives_peslite_owned_state():
    session = _session(
        [helper.make_node("Add", ["state", "input"], ["output"]),
         helper.make_node("Sub", ["input", "state"], ["derivative"])],
        [_value("state", 2), _value("input", 2)],
        [_value("output", 2), _value("derivative", 2)],
    )

    def infer(state, inputs):
        return session.run(None, {
            "state": state.astype(np.float32),
            "input": inputs.astype(np.float32),
        })

    block = BlackBoxSubsystem(
        output=lambda _t, state, inputs: infer(state, inputs)[0],
        rhs=lambda _t, state, inputs: infer(state, inputs)[1],
        inputs=("u0", "u1"),
        outputs=("y0", "y1"),
        state={"hidden_0": 1.0, "hidden_1": -2.0},
    )
    block.inp.u0, block.inp.u1 = 4.0, 3.0
    model = Model({"onnx": block}, {})

    model.evaluate_outputs(0.0)

    assert model.state_labels() == ["onnx.hidden_0", "onnx.hidden_1"]
    assert (block.out.y0, block.out.y1) == pytest.approx((5.0, 1.0))
    assert model.rhs(0.0, np.array([1.0, -2.0])) == pytest.approx((3.0, 5.0))


def test_onnx_recurrent_update_is_committed_as_discrete_peslite_state():
    half = helper.make_tensor("half", TensorProto.FLOAT, [2], [0.5, 0.5])
    session = _session(
        [helper.make_node("Mul", ["state", "half"], ["memory"]),
         helper.make_node("Add", ["memory", "input"], ["next_state"])],
        [_value("state", 2), _value("input", 2)],
        [_value("next_state", 2)],
        [half],
    )
    block = BlackBoxSubsystem(
        output=lambda _t, state, _inputs: state,
        update=lambda _t, state, inputs: session.run(None, {
            "state": state.astype(np.float32),
            "input": inputs.astype(np.float32),
        })[0],
        period=1.0e-3,
        inputs=("u0", "u1"),
        outputs=("y0", "y1"),
        state={"hidden_0": 2.0, "hidden_1": -4.0},
        feedthrough=(),
    )
    block.inp.u0, block.inp.u1 = 3.0, 1.0

    pending = block.next_state(0.0)
    assert block.get_state() == {"hidden_0": 2.0, "hidden_1": -4.0}
    block.set_state(dict(zip(block.get_state(), pending)))
    block.set_outputs(0.0)

    assert block.get_state() == pytest.approx({"hidden_0": 4.0, "hidden_1": -1.0})
    assert (block.out.y0, block.out.y1) == pytest.approx((4.0, -1.0))


def test_configured_onnx_model_resolves_its_file_and_physical_input(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [2.0])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [0.5])
    model = _model(
        [helper.make_node("Mul", ["input.0", "scale"], ["scaled"]),
         helper.make_node("Add", ["scaled", "bias"], ["output.0"])],
        [_value("input.0", 1)],
        [_value("output.0", 1)],
        [scale, bias],
    )
    onnx.save(model, model_dir / "model.onnx")
    case = tmp_path / "case.pes"
    case.write_text(
        """base: {s_base: 1.0}
buses:
  node: {phases: 1, c: 1.0e-3}
elements:
  load: {type: resistor, bus: node, r: 2.0}
  onnx_model:
    type: onnx
    model: models/model.onnx
    inputs: {input.0: [load.power]}
    outputs: {output.0: [estimate]}
simulation: {t_end: 1.0e-4}
""",
        encoding="utf-8",
    )

    params = peslite.load(case)
    system = peslite.System(params)
    system.model.evaluate_outputs(0.0)

    assert params.elements["onnx_model"].model == str((model_dir / "model.onnx").resolve())
    assert system.signals(0.0)["onnx_model.estimate"] == pytest.approx(0.5)

    resolved = tmp_path / "resolved" / "simulation.pes"
    resolved.parent.mkdir()
    peslite.dump(params, resolved)
    reloaded = peslite.load(resolved)
    peslite.System(reloaded).model.evaluate_outputs(0.0)


def test_configured_recurrent_onnx_model_uses_registered_sampled_state(tmp_path):
    half = helper.make_tensor("half", TensorProto.FLOAT, [2], [0.5, 0.5])
    model = _model(
        [helper.make_node("Mul", ["hidden_in", "half"], ["memory"]),
         helper.make_node("Add", ["memory", "input"], ["hidden_out"]),
         helper.make_node("Identity", ["hidden_out"], ["output"])],
        [_value("hidden_in", 2), _value("input", 2)],
        [_value("hidden_out", 2), _value("output", 2)],
        [half],
    )
    path = tmp_path / "recurrent.onnx"
    onnx.save(model, path)
    params = peslite.assembly.params.from_dict({
        "base": {"s_base": 1.0},
        "buses": {"node": {"phases": 1, "c": 1.0e-3}},
        "elements": {
            "load": {"type": "resistor", "bus": "node", "r": 2.0},
            "onnx_model": {
                "type": "onnx",
                "model": str(path),
                "inputs": {"input": ["load.power", "load.power"]},
                "outputs": {"output": ["y0", "y1"]},
                "period": 1.0e-3,
                "state": {"hidden_in": {
                    "output": "hidden_out",
                    "initial": [2.0, -4.0],
                }},
            },
        },
        "simulation": {
            "t_end": 1.1e-3,
            "solver": {"dt": 1.0e-4, "linearisations": 0},
            "output": {"period": 1.0e-4},
            "energy_check": "off",
        },
    })

    simulation = peslite.Simulation(params)
    result = simulation.run(out_dir=tmp_path / "run")
    block = simulation.system.named_elements["onnx_model"].model

    assert SystemLoop(simulation).periods == [1.0e-3]
    assert "onnx_model.state_0" not in simulation._states.discrete_names
    assert block.get_state() == pytest.approx({
        "state_0": 0.5,
        "state_1": -1.0,
        "output_0": 0.5,
        "output_1": -1.0,
    })
    assert result.states["onnx_model.state_0"][-1] == pytest.approx(0.5)
    assert (block.out.y0, block.out.y1) == pytest.approx((0.5, -1.0))
