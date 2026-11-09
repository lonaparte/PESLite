"""Line-delimited JSON service for the PESLite frontend.

Every stdin line is one request ``{"id", "method", "params"}``; every stdout line is either a
response ``{"id", "ok", "result" | "error"}`` or an event ``{"type", "requestId", ...}``.
Diagnostics, warnings and any console output from a run go to stderr, so stdout stays valid JSON.

Methods: ``getSystems``, ``model.load``, ``model.reset``, ``simulation.start``,
``smallSignal.run``, ``interaction.run``, ``transientVoltage.run`` and ``server.shutdown``.
A "system" is a PESLite simulation file; its parameters are dotted-path overrides.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, TextIO

from ..assembly.params import load
from ..solver.simulation import Simulation, _example_configs_dir
from .descriptor import SignalRecorder, Topology, json_safe, small_signal_descriptor

CASE_SUFFIXES = (".pes", ".yaml", ".yml")
CASES_ENVIRONMENT = "PESLITE_CASES"
SNAPSHOT_NAME = "gui-snapshot.json"
# Frontend run options and the parameter path each one sets.
OPTION_PATHS = {"EndTime_s": "simulation.t_end", "OutputStep_s": "simulation.output.period"}
RESULT_OPTIONS = ("RunTimeDomain", "RunSmallSignal", "RunInteractionAnalysis")


def discover_cases(root: Path) -> dict[str, Path]:
    """Simulation files by name: ``PESLITE_CASES`` entries, then ``root``, then the examples.

    ``PESLITE_CASES`` lists files or directories separated by ``os.pathsep``. An earlier location
    wins when two files share a name.
    """
    locations = [Path(item).expanduser() for item in
                 os.environ.get(CASES_ENVIRONMENT, "").split(os.pathsep) if item.strip()]
    locations += [root, _example_configs_dir()]
    cases: dict[str, Path] = {}
    for location in locations:
        if location.is_file():
            files = [location]
        elif location.is_dir():
            files = sorted(path for path in location.iterdir()
                           if path.is_file() and path.suffix.lower() in CASE_SUFFIXES)
        else:
            continue
        for path in files:
            if path.suffix.lower() in (".yaml", ".yml") and not _looks_like_case(path):
                continue
            cases.setdefault(path.stem, path.resolve())
    return cases


def _looks_like_case(path: Path) -> bool:
    """Accept generic YAML files only when they declare the required top-level sections."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "\nbase:" in f"\n{text}" and "\nsimulation:" in f"\n{text}"


class FrontendService:
    """Hold one loaded case, its latest completed run and its frontend snapshot."""

    def __init__(self, out: TextIO | None = None, root: str | Path | None = None,
                 output_root: str | Path | None = None) -> None:
        self.out = out or sys.stdout
        self.root = Path(root or Path.cwd()).resolve()
        self.output_root = Path(output_root) if output_root else self.root / "output"
        self.cases: dict[str, Path] = {}
        self.running = True
        self.reset()

    # ------------------------------------------------------------ state
    def reset(self) -> None:
        self.case = ""
        self.overrides: dict[str, Any] = {}
        self.model_key = ""
        self.topology: Topology | None = None
        self.simulation: Simulation | None = None  # the latest completed run of this model
        self.signals: list[dict[str, Any]] = []
        self.small_signal: dict[str, Any] = {}
        self.layout: dict[str, Any] | None = None

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.case

    @property
    def snapshot_file(self) -> Path:
        return self.output_dir / SNAPSHOT_NAME

    # ------------------------------------------------------------ transport
    def write(self, message: dict[str, Any]) -> None:
        json.dump(json_safe(message), self.out, allow_nan=False)
        self.out.write("\n")
        self.out.flush()

    def respond(self, request_id: Any, result: Any) -> None:
        self.write({"id": request_id, "ok": True, "result": result})

    def fail(self, request_id: Any, error: BaseException | str) -> None:
        self.write({"id": request_id, "ok": False, "error": str(error) or type(error).__name__})

    def event(self, kind: str, request_id: Any, **payload: Any) -> None:
        self.write({"type": kind, "requestId": request_id, **payload})

    def serve(self, lines: TextIO) -> None:
        for line in lines:
            if not line.strip():
                continue
            request = None
            try:
                request = json.loads(line)
                self.dispatch(request)
            except Exception as error:  # report every failure to the request that caused it
                traceback.print_exc(file=sys.stderr)
                self.fail(request.get("id") if isinstance(request, dict) else None, error)
            if not self.running:
                break

    def dispatch(self, request: dict[str, Any]) -> None:
        request_id, method = request.get("id"), request["method"]
        params = request.get("params") or {}
        if method == "getSystems":
            self.cases = discover_cases(self.root)
            self.respond(request_id, list(self.cases))
        elif method == "model.load":
            self.load_model(params)
            self.respond(request_id, {
                "systemName": self.case,
                "parameters": self.topology.editable_parameters(),
                "options": self.options(),
                **self.save(),
            })
        elif method == "model.reset":
            self.reset()
            self.respond(request_id, {"reset": True})
        elif method == "simulation.start":
            self.load_model(params)
            summary = self.run(request_id, stream=True)
            self.respond(request_id, {**self.save(), "summary": summary})
        elif method == "smallSignal.run":
            self.load_model(params)
            self.run_small_signal(request_id)
            self.respond(request_id, {**self.save(),
                                      "unstablePoleCount": self.small_signal["unstablePoleCount"]})
        elif method in ("interaction.run", "transientVoltage.run"):
            raise ValueError(f"{method} is unavailable; the PESLite frontend supports simulation "
                             "and small-signal analysis")
        elif method == "server.shutdown":
            self.respond(request_id, {"shutdown": True})
            self.running = False
        else:
            raise KeyError(f"unknown method: {method}")

    # ------------------------------------------------------------ model
    def options(self) -> dict[str, Any]:
        p = self.topology.params
        return {"EndTime_s": p.simulation.t_end, "OutputStep_s": p.simulation.output.period}

    def load_model(self, params: dict[str, Any]) -> None:
        """Load the named case with its overrides, reusing the current model when unchanged."""
        case = params.get("systemName") or self.case
        if not case:
            raise ValueError("select a simulation file first")
        overrides = dict(params.get("parameters") or {})
        options = {key: value for key, value in (params.get("options") or {}).items()
                   if key not in RESULT_OPTIONS and value is not None}
        unknown = set(options) - set(OPTION_PATHS)
        if unknown:
            raise KeyError("unsupported options (set them as parameters instead): "
                           + ", ".join(sorted(unknown)))
        overrides.update({OPTION_PATHS[key]: value for key, value in options.items()})
        key = json.dumps([case, overrides], sort_keys=True, default=str)
        if self.topology is not None and key == self.model_key:
            return
        if case not in self.cases:
            self.cases = discover_cases(self.root)
        if case not in self.cases:
            raise KeyError(f"unknown simulation file {case!r}; known: {sorted(self.cases)}")
        path = self.cases[case]
        topology = Topology.build(case, path, load(path, **overrides))
        self.reset()
        self.case, self.overrides, self.model_key, self.topology = case, overrides, key, topology
        self.restore()

    def restore(self) -> None:
        """Keep a saved layout, and saved results when they used the same parameter values."""
        saved = self._snapshot()
        if not saved:
            return
        if isinstance(saved.get("layout"), dict):
            self.layout = saved["layout"]
        current = self.topology.system()["parameters"]
        if saved["system"].get("parameters") == current:
            self.signals = saved.get("signals") or []
            small = saved.get("smallSignal") or {}
            self.small_signal = small if small.get("available") else {}

    def _snapshot(self) -> dict[str, Any]:
        """Read a snapshot belonging to the loaded system."""
        try:
            saved = json.loads(self.snapshot_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        system = saved.get("system") if isinstance(saved, dict) else None
        return (saved if isinstance(system, dict)
                and system.get("id") == self.topology.system_id else {})

    def save(self) -> dict[str, Any]:
        """Write the descriptor atomically, keeping a layout the extension saved in place."""
        saved = self._snapshot()
        if isinstance(saved.get("layout"), dict):
            self.layout = saved["layout"]
        descriptor = self.topology.descriptor(self.signals, self.small_signal, self.layout)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.output_dir,
                                         suffix=".tmp", delete=False) as handle:
            json.dump(json_safe(descriptor), handle, allow_nan=False)
            temporary = Path(handle.name)
        os.replace(temporary, self.snapshot_file)
        return {"descriptor": descriptor, "snapshotFile": str(self.snapshot_file)}

    # ------------------------------------------------------------ analyses
    def run(self, request_id: Any, *, stream: bool) -> dict[str, Any]:
        """Run the loaded case once, streaming its signals as frontend simulation chunks."""
        p = self.topology.params
        self.simulation, self.signals, self.small_signal = None, [], {}
        sim = Simulation(p)
        sequence = 0

        def emit(signals: list[dict[str, Any]]) -> None:
            nonlocal sequence
            self.event("simulation.chunk", request_id, sequence=sequence, signals=signals)
            sequence += 1

        recorder = SignalRecorder(self.topology, sim, emit=emit if stream else None)
        if stream:
            span = p.simulation.t_end - p.simulation.initial.t
            self.event("simulation.started", request_id, systemName=self.case,
                       sampleCount=int(math.floor(span / p.simulation.output.period + 1e-9)) + 1)
        result = sim.run(out_dir=self.output_dir, observe=recorder.observe,
                         observe_values=recorder.observe_values, name=self.case)
        recorder.flush()
        self.signals = recorder.signals()
        self.simulation = sim
        reason = result.summary.get("stop_reason")
        if reason:
            print(f"[peslite] simulation stopped early: {reason}", file=sys.stderr, flush=True)
        if stream:
            self.event("simulation.completed", request_id, systemName=self.case,
                       stopReason=reason or "", tripped=result.tripped)
        return {key: value for key, value in result.summary.items() if key != "ph_report"}

    def run_small_signal(self, request_id: Any) -> None:
        """Linearize the final state of the latest run, running the case first when needed."""
        from ..addons.functions.small_signal import small_signal

        if self.simulation is None:
            self.run(request_id, stream=False)
        analysis = small_signal(self.simulation)
        self.small_signal = small_signal_descriptor(analysis)
        if not self.small_signal.get("available"):
            raise ValueError("; ".join(analysis.diagnostics) or "no periodic operating state")
        print(f"[peslite] small-signal: {analysis.stability}, period {analysis.period:.6g} s, "
              f"{len(self.small_signal['modes'])} modes", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Serve the PESLite frontend protocol until shutdown or end of input."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m peslite.gui", description=main.__doc__)
    parser.add_argument("--root", help="directory searched for simulation files (default: cwd)")
    parser.add_argument("--output", help="run and snapshot directory (default: <root>/output)")
    args = parser.parse_args(argv)
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    # Only protocol messages may reach stdout; route ordinary prints (progress lines) to stderr.
    protocol, sys.stdout = sys.stdout, sys.stderr
    try:
        FrontendService(protocol, args.root, args.output).serve(sys.stdin)
    finally:
        sys.stdout = protocol
    return 0
