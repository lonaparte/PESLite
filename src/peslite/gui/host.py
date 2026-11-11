"""Small file/command contract shared by local Web and VS Code hosts.

The browser-only SVG renderer has no dependency on this module. The optional
legacy-YAML function addon is imported only for legacy buffers; new @circuit
syntax is parsed by the renderer itself. Nothing here changes the simulator.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any

from .workspace import Workspace, WorkspaceError


class SimulationProcess:
    """Launch the existing PESLite CLI, with bounded output and cancellation."""

    def __init__(self, workspace: Workspace, python: str = sys.executable):
        self.workspace = workspace
        self.python = python
        self.process: subprocess.Popen | None = None
        self.path = ""
        self.log: deque[str] = deque(maxlen=300)
        self._lock = threading.RLock()

    def start(self, name: str, *, trust_addons: bool = False) -> dict:
        path = self.workspace.path(name, suffixes={".pes"})
        if not path.is_file():
            raise WorkspaceError("select an existing .pes file")
        if self.workspace.has_project_addons(name) and not trust_addons:
            raise WorkspaceError("project-local PESaddons execute Python code; explicit trust is required")
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                raise WorkspaceError("another simulation is already running")
            self.log.clear()
            self.path = name
            # Use the normal CLI, its default output directory and addon discovery.
            self.process = subprocess.Popen(
                [self.python, "-m", "peslite.solver.simulation", str(path)],
                cwd=self.workspace.root, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1,
            )
            threading.Thread(target=self._collect, args=(self.process,), daemon=True).start()
        return self.status()

    def _collect(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with self._lock:
                self.log.append(line[-2000:])
        process.stdout.close()
        process.wait()

    def status(self) -> dict:
        with self._lock:
            process = self.process
            return {"running": bool(process is not None and process.poll() is None),
                    "exitCode": process.poll() if process else None,
                    "path": self.path, "log": "".join(self.log)[-30000:]}

    def stop(self) -> dict:
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                return self.status()
            self.process.terminate()
        return {"stopping": True}


class Host:
    """Dispatch transport-neutral, workspace-scoped GUI requests."""

    def __init__(self, root: str | Path, *, python: str = sys.executable):
        self.files = Workspace(root)
        self.runs = SimulationProcess(self.files, python)

    def dispatch(self, method: str, params: dict[str, Any]) -> dict | list:
        if not isinstance(params, dict):
            raise WorkspaceError("request parameters must be an object")
        name = params.get("path", "")
        if method == "tree":
            return {"files": self.files.tree(name)}
        if method == "file.read":
            return self.files.read(name)
        if method == "file.write":
            return self.files.write(name, params["text"], params.get("version"))
        if method == "file.create":
            return self.files.write(name, params.get("text", ""), None, create=True)
        if method == "file.rename":
            return self.files.rename(params["source"], params["destination"], params["version"])
        if method == "file.delete":
            return self.files.delete(name, params["version"])
        if method == "csv.read":
            return self.files.csv(name, offset=int(params.get("offset", 0)),
                                  stride=int(params.get("stride", 1)),
                                  limit=int(params.get("limit", 1000)),
                                  byte_offset=(int(params["byteOffset"])
                                               if params.get("byteOffset") is not None else None))
        if method == "summary.read":
            return self.files.summary(name)
        if method in {"preview", "parameter.patch", "component.add", "component.remove",
                      "connection.rewire"}:
            target = self.files.path(name, suffixes={".pes"})
            text = params.get("text")
            if not isinstance(text, str):
                raise WorkspaceError("preview requires source text")
            from ..assembly.simulation_file import is_native_source
            if is_native_source(text):
                if method != "preview":
                    raise WorkspaceError("new-syntax schematics must be edited in source text")
                return {"format": "native", "model": None, "fields": [], "diagnostics": []}
            if params.get("trustAddons") is True and self.files.has_project_addons(name):
                from ..addons import _discover_for
                # A workspace may place PESaddons next to a nested model or at
                # the workspace root. Both locations require the same approval.
                if (self.files.root / "PESaddons").is_dir():
                    _discover_for(self.files.root / "workspace.pes")
                _discover_for(target)
            from ..addons.functions import (
                legacy_yaml_add_component, legacy_yaml_patch, legacy_yaml_preview,
                legacy_yaml_remove_component, legacy_yaml_rewire,
            )
            if method == "parameter.patch":
                return {"edit": legacy_yaml_patch(text, target, params["key"], params["value"])}
            if method == "component.add":
                return {"edit": legacy_yaml_add_component(
                    text, target, params["section"], params["name"], params["definition"])}
            if method == "component.remove":
                return {"edit": legacy_yaml_remove_component(
                    text, target, params["section"], params["name"])}
            if method == "connection.rewire":
                return {"edit": legacy_yaml_rewire(
                    text, target, params["key"], params["target"])}
            return legacy_yaml_preview(text, target)
        if method == "run.start":
            return self.runs.start(name, trust_addons=params.get("trustAddons") is True)
        if method == "run.stop":
            return self.runs.stop()
        if method == "run.status":
            return self.runs.status()
        raise WorkspaceError(f"unknown GUI operation: {method}")


def invoke(root: str | Path, request: dict[str, Any]) -> dict:
    """One-shot JSON bridge used by the VS Code extension host process."""
    try:
        result = Host(root).dispatch(request["method"], request.get("params") or {})
        return {"ok": True, "result": result}
    except (WorkspaceError, KeyError, ValueError, OSError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}


def main(argv: list[str] | None = None) -> int:
    """Run one JSON request on stdin; do not alter the existing JSON-lines service."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m peslite.gui.host")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise WorkspaceError("request must be a JSON object")
        response = invoke(args.root, request)
    except (ValueError, OSError, TypeError) as exc:
        response = {"ok": False, "error": str(exc)}
    print(json.dumps(response, allow_nan=False))
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
