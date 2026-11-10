"""Serve the PESLite preview workbench for local files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .preview_cache import PreviewLayoutCache

PREVIEW = Path(__file__).resolve().parent / "preview"
STATIC = {
    "/": ("demo.html", "text/html; charset=utf-8"),
    "/demo.html": ("demo.html", "text/html; charset=utf-8"),
    "/workbench.css": ("workbench.css", "text/css; charset=utf-8"),
    "/workbench.js": ("workbench.js", "text/javascript; charset=utf-8"),
    "/dist/pes-preview.iife.js": ("dist/pes-preview.iife.js", "text/javascript; charset=utf-8"),
    "/dist/pes-preview.es.js": ("dist/pes-preview.es.js", "text/javascript; charset=utf-8"),
}
SKIPPED = {"node_modules", "__pycache__", "dist", "build", "test-output", "output"}
MAX_ENTRIES = 5000
MAX_DEPTH = 8
MAX_FILE = 2 << 20
MAX_BODY = 4 << 20


class Workspace:
    def __init__(self, root: str | Path, open_file: str | Path | None = None,
                 cache: PreviewLayoutCache | None = None) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.open_file = self.relative(Path(open_file).resolve()) if open_file else None
        self.cache = cache or PreviewLayoutCache()

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def resolve(self, relative: Any) -> Path:
        if not isinstance(relative, str) or not relative or "\0" in relative:
            raise PermissionError("a file path inside the workspace is required")
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise PermissionError(f"{relative!r} is outside the workspace")
        if not path.is_file():
            raise FileNotFoundError(relative)
        return path

    def tree(self) -> list[dict[str, Any]]:
        count = 0

        def walk(folder: Path, depth: int) -> list[dict[str, Any]]:
            nonlocal count
            entries = []
            try:
                children = sorted(folder.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            except OSError:
                return entries
            for child in children:
                if count >= MAX_ENTRIES or child.name.startswith(".") or child.name in SKIPPED:
                    continue
                resolved = child.resolve()
                if resolved != self.root and self.root not in resolved.parents:
                    continue
                count += 1
                entry: dict[str, Any] = {"name": child.name, "path": self.relative(child)}
                if child.is_dir():
                    entry.update(type="folder", children=walk(child, depth + 1) if depth < MAX_DEPTH else [])
                else:
                    entry["type"] = "file"
                entries.append(entry)
            return entries

        return walk(self.root, 0)

    def read(self, relative: Any) -> dict[str, Any]:
        path = self.resolve(relative)
        if path.stat().st_size > MAX_FILE:
            raise ValueError(f"{relative} is larger than {MAX_FILE >> 20} MB")
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{relative} is not UTF-8 text") from error
        key = self.relative(path)
        return {"path": key, "name": path.name, "source": source, "layout": self.cache.load(key)}

    def write(self, relative: Any, source: Any) -> dict[str, Any]:
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        path = self.resolve(relative)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False, newline="") as handle:
            handle.write(source)
            temporary = Path(handle.name)
        os.replace(temporary, path)
        return {"path": self.relative(path), "saved": True}

    def save_layout(self, relative: Any, layout: Any) -> dict[str, Any]:
        key = self.relative(self.resolve(relative))
        return {"layout": self.cache.save(key, layout)}


class PreviewHandler(BaseHTTPRequestHandler):
    server_version = "PESLitePreview/1"

    def __init__(self, *args: Any, workspace: Workspace, **kwargs: Any) -> None:
        self.workspace = workspace
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def allowed_host(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in {"127.0.0.1", "localhost", "::1", self.server.server_address[0]}

    def send_json(self, value: Any) -> None:
        body = json.dumps(value).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def fail(self, status: HTTPStatus, message: str) -> None:
        body = message.encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> Any:
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            raise ValueError("expected an application/json body")
        length = int(self.headers.get("Content-Length") or 0)
        if length < 0 or length > MAX_BODY:
            raise ValueError("invalid request size")
        return json.loads(self.rfile.read(length) or b"null")

    def route(self, action) -> None:
        if not self.allowed_host():
            self.fail(HTTPStatus.FORBIDDEN, "unexpected Host header")
            return
        try:
            action()
        except PermissionError as error:
            self.fail(HTTPStatus.FORBIDDEN, str(error))
        except FileNotFoundError as error:
            self.fail(HTTPStatus.NOT_FOUND, str(error))
        except (ValueError, json.JSONDecodeError) as error:
            self.fail(HTTPStatus.BAD_REQUEST, str(error))

    def do_GET(self) -> None:  # noqa: N802
        self.route(self.get)

    def do_PUT(self) -> None:  # noqa: N802
        self.route(self.put)

    def do_POST(self) -> None:  # noqa: N802
        self.route(self.post)

    def get(self) -> None:
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if url.path == "/api/workspace":
            self.send_json({"name": self.workspace.root.name, "open": self.workspace.open_file,
                            "openName": Path(self.workspace.open_file).name if self.workspace.open_file else None,
                            "files": self.workspace.tree()})
        elif url.path == "/api/file":
            self.send_json(self.workspace.read(query.get("path", [""])[0]))
        elif url.path in STATIC:
            name, content_type = STATIC[url.path]
            path = PREVIEW / name
            if not path.is_file():
                self.fail(HTTPStatus.SERVICE_UNAVAILABLE, f"missing preview asset: {name}")
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.fail(HTTPStatus.NOT_FOUND, "not found")

    def put(self) -> None:
        url = urlsplit(self.path)
        relative = parse_qs(url.query).get("path", [""])[0]
        if url.path == "/api/file":
            data = self.body()
            self.send_json(self.workspace.write(relative, (data or {}).get("source")))
        elif url.path == "/api/layout":
            self.send_json(self.workspace.save_layout(relative, self.body()))
        else:
            self.fail(HTTPStatus.NOT_FOUND, "not found")

    def post(self) -> None:
        if urlsplit(self.path).path != "/api/session/close":
            self.fail(HTTPStatus.NOT_FOUND, "not found")
            return
        self.workspace.cache.clear()
        self.send_json({"cleared": True})


class PreviewServer(ThreadingHTTPServer):
    def __init__(self, address, handler, workspace: Workspace) -> None:
        self.workspace = workspace
        super().__init__(address, handler)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self.workspace.cache.close()


def make_server(workspace: Workspace, host: str = "127.0.0.1", port: int = 8765) -> PreviewServer:
    return PreviewServer((host, port), partial(PreviewHandler, workspace=workspace), workspace)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m peslite.gui.preview_host")
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    target = Path(args.path).expanduser().resolve()
    workspace = Workspace(target.parent if target.is_file() else target,
                          open_file=target if target.is_file() else None)
    server = make_server(workspace, port=args.port)
    host, port = server.server_address[:2]
    print(f"PESLite preview at http://{host}:{port}/", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
