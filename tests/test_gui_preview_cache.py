"""Temporary layout storage and the local GUI preview host."""

import io
import json
import threading
import urllib.request

from peslite.gui.preview_cache import PreviewLayoutCache, empty_layout, normalize_layout
from peslite.gui.preview_host import Workspace, make_server
from peslite.gui.service import FrontendService
from peslite.gui.host import Host

LAYOUT = {"format": "peslite.preview-layout", "version": 1,
          "views": {"circuit": {"VSC": {"dx": 48, "dy": -24}}}}


def _request(url, method="GET", value=None):
    data = None if value is None else json.dumps(value).encode()
    request = urllib.request.Request(url, method=method, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_layout_cache_validates_keys_and_removes_its_file():
    cache = PreviewLayoutCache()
    path = None
    try:
        assert normalize_layout({"views": {"circuit": {
            "A": {"dx": 1, "dy": 2}, "bad": {"dx": "1", "dy": 2},
        }}})["views"] == {"circuit": {"A": {"dx": 1, "dy": 2}}}
        cache.save("folder-a/grid.pes", LAYOUT)
        cache.save("folder-b/grid.pes", {"views": {"circuit": {"B": {"dx": 8, "dy": 0}}}})
        path = cache.path
        assert path is not None and path.is_file()
        assert cache.load("folder-a/grid.pes") == LAYOUT
        assert "B" in cache.load("folder-b/grid.pes")["views"]["circuit"]
    finally:
        cache.close()
    assert path is not None and not path.exists()


def test_preview_host_reads_files_and_clears_layout_cache_on_close(tmp_path):
    case = tmp_path / "case.pes"
    case.write_text("@circuit\nA -- B\n", encoding="utf-8")
    workspace = Workspace(tmp_path, open_file=case)
    server = make_server(workspace, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert _request(f"{url}/api/workspace")["open"] == "case.pes"
        assert _request(f"{url}/api/file?path=case.pes")["source"].startswith("@circuit")
        _request(f"{url}/api/layout?path=case.pes", "PUT", LAYOUT)
        cache_file = workspace.cache.path
        assert cache_file is not None and cache_file.is_file()
        assert _request(f"{url}/api/session/close", "POST") == {"cleared": True}
        assert not cache_file.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gui_service_uses_the_same_session_cache(tmp_path):
    (tmp_path / "case.pes").write_text("@circuit\nA -- B\n", encoding="utf-8")
    output = io.StringIO()
    service = FrontendService(output, root=tmp_path)
    service.dispatch({"id": 1, "method": "preview.layout.save",
                      "params": {"systemName": "case", "layout": LAYOUT}})
    path = service.preview_cache.path
    service.dispatch({"id": 2, "method": "preview.layout.load",
                      "params": {"systemName": "case"}})
    service.dispatch({"id": 3, "method": "server.shutdown", "params": {}})
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[1]["result"]["layout"] == LAYOUT
    assert path is not None and not path.exists()
    assert service.preview_cache.load("missing") == empty_layout()


def test_file_backed_host_uses_the_shared_session_cache(tmp_path):
    (tmp_path / "case.pes").write_text("@circuit\nA -- B\n", encoding="utf-8")
    host = Host(tmp_path)
    try:
        assert host.dispatch("preview.layout.load", {"path": "case.pes"}) == {
            "layout": empty_layout(),
        }
        assert host.dispatch("preview.layout.save", {
            "path": "case.pes", "layout": LAYOUT,
        }) == {"layout": LAYOUT}
        path = host.preview_cache.path
        assert path is not None and path.is_file()
        assert host.dispatch("preview.layout.load", {"path": "case.pes"})["layout"] == LAYOUT
        assert host.dispatch("preview.layout.clear", {}) == {"cleared": True}
        assert not path.exists()
    finally:
        host.close()
