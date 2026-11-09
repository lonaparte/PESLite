"""The PESLite GUI protocol for simulation files."""

import io
import json
import os
import subprocess
import sys

import pytest

from conftest import EXAMPLES, ROOT
from peslite.gui import FrontendService, Topology
import peslite


def _serve(tmp_path, *requests):
    out = io.StringIO()
    service = FrontendService(out, root=tmp_path, output_root=tmp_path / "output")
    lines = [json.dumps({"id": index, **request}) + "\n" for index, request in enumerate(requests, 1)]
    service.serve(io.StringIO("".join(lines)))
    return [json.loads(line) for line in out.getvalue().splitlines()]


def _response(messages, request_id):
    return next(message for message in messages if message.get("id") == request_id)


@pytest.mark.parametrize("case", ["two-converters-example", "cable-gfl-example", "wpt-example",
                                  "wpt-load-example", "buck-example"])
def test_topology_references_only_existing_components(case):
    topology = Topology.build(case, EXAMPLES / f"{case}.pes", peslite.load(EXAMPLES / f"{case}.pes"))
    ids = {component["id"] for component in topology.components}
    assert len(ids) == len(topology.components)
    for connection in topology.connections:
        assert {connection["source"], connection["target"]} <= ids
    # Every editable path is accepted back as an override.
    values = topology.editable_parameters()
    peslite.load(EXAMPLES / f"{case}.pes", **values)


def test_protocol_loads_simulates_and_streams(tmp_path):
    overrides = {"simulation.t_end": 0.004, "simulation.energy_check": "off"}
    messages = _serve(
        tmp_path,
        {"method": "getSystems", "params": {}},
        {"method": "model.load", "params": {"systemName": "buck-example", "parameters": overrides}},
        {"method": "simulation.start",
         "params": {"systemName": "buck-example", "parameters": overrides}},
        {"method": "interaction.run", "params": {"systemName": "buck-example"}},
        {"method": "server.shutdown", "params": {}},
    )
    assert "buck-example" in _response(messages, 1)["result"]
    loaded = _response(messages, 2)["result"]
    assert loaded["parameters"]["simulation.t_end"] == 0.004
    descriptor = loaded["descriptor"]
    assert {"bus-output", "unit-converter", "element-load"} <= {
        component["id"] for component in descriptor["components"]}

    events = [message["type"] for message in messages if "type" in message]
    assert events[0] == "simulation.started" and events[-1] == "simulation.completed"
    assert "simulation.chunk" in events
    result = _response(messages, 3)
    assert result["ok"], result.get("error")
    signals = {signal["label"]: signal for signal in result["result"]["descriptor"]["signals"]}
    voltage = signals["|V|-output"]
    assert voltage["category"] == "output" and len(voltage["time"]) == len(voltage["values"]) > 10
    assert signals["converter.ctrl.DCVoltageControl.integral"]["category"] == "state"
    streamed = sum(len(signal["values"]) for message in messages
                   if message.get("type") == "simulation.chunk"
                   for signal in message["signals"] if signal["id"] == voltage["id"])
    assert streamed == len(voltage["values"])
    assert not _response(messages, 4)["ok"]
    assert (tmp_path / "output" / "buck-example" / "gui-snapshot.json").is_file()


@pytest.mark.full
def test_small_signal_modes_follow_the_frontend_schema(tmp_path):
    overrides = {"simulation.t_end": 0.050025, "simulation.energy_check": "off",
                 "simulation.solver.linearisations": 0}
    messages = _serve(tmp_path, {"method": "smallSignal.run",
                                 "params": {"systemName": "buck-example", "parameters": overrides}})
    response = _response(messages, 1)
    assert response["ok"], response.get("error")
    small = response["result"]["descriptor"]["smallSignal"]
    assert small["available"] and small["stability"] == "stable"
    assert small["unstablePoleCount"] == 0 and small["modes"]
    assert all(mode["realPart"] < 0 and mode["dominantState"] for mode in small["modes"])


def test_module_serves_stdio_like_the_extension_starts_it():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    process = subprocess.run(
        [sys.executable, "-u", "-m", "peslite.gui"], cwd=ROOT, capture_output=True,
        text=True, timeout=120, env=env,
        input='{"id":1,"method":"getSystems","params":{}}\n'
              '{"id":2,"method":"server.shutdown","params":{}}\n')
    lines = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.returncode == 0
    assert "gfl-example" in lines[0]["result"] and lines[1]["result"] == {"shutdown": True}
