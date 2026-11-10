"""Loopback-only HTTP host for PESLite's shared, offline file-backed UI.

Usage: peslite-gui --root path/to/workspace --port 8765 --no-browser
This entry point is independent of the existing python -m peslite.gui
JSON-lines integration, which remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .host import Host
from .workspace import WorkspaceError

MAX_REQUEST = 4 * 1024 * 1024 + 4096


class LocalServer(ThreadingHTTPServer):
    """One bound workspace and session token, never accessible on the LAN."""

    daemon_threads = True

    def __init__(self, port: int, root: str | Path, *, assets: Path | None = None):
        super().__init__(("127.0.0.1", port), Handler)
        self.host_api = Host(root)
        self.token = secrets.token_urlsafe(32)
        self.assets = (assets or Path(__file__).resolve().parent / "web_dist").resolve()


class Handler(BaseHTTPRequestHandler):
    server: LocalServer
    protocol_version = "HTTP/1.1"

    def _origin_allowed(self) -> bool:
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return (self.headers.get("Host", "") in hosts and
                (origin is None or origin in {f"http://{host}" for host in hosts}))

    def _send(self, status: int, content: bytes, mime: str):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        # Denied POST requests may leave unread bytes; close rather than parse them as HTTP.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                         "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                         "font-src 'self'; img-src 'self' data:; object-src 'none'; "
                         "base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: int, value):
        self._send(status, json.dumps(value, allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if not self._origin_allowed():
            self._json(403, {"error": "invalid host or origin"})
            return
        path = urlsplit(self.path).path
        if path == "/":
            path = "/index.html"
        relative = path.removeprefix("/")
        if not relative or relative.startswith(".") or "\\" in relative or ".." in Path(relative).parts:
            self._json(404, {"error": "asset not found"})
            return
        selected = (self.server.assets / relative).resolve()
        if not selected.is_relative_to(self.server.assets) or not selected.is_file():
            self._json(404, {"error": "asset not found; build the frontend UI first"})
            return
        data = selected.read_bytes()
        if relative == "index.html":
            data = data.replace(b"__PESLITE_TOKEN__", self.server.token.encode("ascii"))
        self._send(200, data, mimetypes.guess_type(relative)[0] or "application/octet-stream")

    def do_POST(self):
        if not self._origin_allowed():
            self._json(403, {"error": "invalid host or origin"})
            return
        if self.headers.get("X-PESLite-Token") != self.server.token or (
                self.headers.get("Content-Type", "").split(";")[0] != "application/json"):
            self._json(403, {"error": "missing session token or invalid content type"})
            return
        if urlsplit(self.path).path != "/api":
            self._json(404, {"error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 2 or length > MAX_REQUEST:
                raise WorkspaceError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise WorkspaceError("request must be a JSON object")
            result = self.server.host_api.dispatch(payload["method"], payload.get("params") or {})
            self._json(200, {"ok": True, "result": result})
        except (WorkspaceError, ValueError, OSError, TypeError, KeyError) as exc:
            self._json(getattr(exc, "status", 400), {"ok": False, "error": str(exc)})
        except Exception as exc:
            # Do not disclose Python tracebacks or filesystem internals through HTTP.
            print(f"[peslite-gui] unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
            self._json(500, {"ok": False, "error": "internal frontend error"})

    def log_message(self, format: str, *args):
        print("[peslite-gui] " + format % args, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="peslite-gui", description=__doc__)
    parser.add_argument("--root", default=".", help="workspace directory")
    parser.add_argument("--port", type=int, default=8765,
                        help="loopback port (0 selects a free port)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.port < 0 or args.port > 65535:
        parser.error("port must be in the range 0..65535")
    with LocalServer(args.port, args.root) as server:
        address = f"http://127.0.0.1:{server.server_port}/"
        print(f"PESLite workspace: {address}", flush=True)
        if not args.no_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
