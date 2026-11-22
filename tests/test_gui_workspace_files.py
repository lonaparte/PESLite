"""File-backed GUI checks independent of the PESLite simulation runtime."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui

MODULE = Path(__file__).resolve().parents[1] / "src/peslite/gui/workspace.py"
spec = importlib.util.spec_from_file_location("peslite_workspace_standalone", MODULE)
workspace = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = workspace
spec.loader.exec_module(workspace)


def test_traversal_symlinks_hidden_and_disallowed_suffix(tmp_path):
    backend = workspace.Workspace(tmp_path)
    (tmp_path / "case.pes").write_text("simulation: {}")
    (tmp_path / "link.pes").symlink_to(tmp_path / "case.pes")
    for path in ("../wrong.pes", "..//wrong.pes", ".git/config.pes",
                 "/tmp/absolute.pes", "link.pes", "sub/../../wrong.pes",
                 "unsupported.py", "dir\\escape.pes"):
        with pytest.raises(workspace.WorkspaceError):
            backend.path(path)


def test_bounded_csv_windows_and_summary(tmp_path):
    backend = workspace.Workspace(tmp_path)
    (tmp_path / "plant.csv").write_text("t,v,i\n0,1,2\n1,NaN,\n2,3,4\n3,5,6\n")
    first = backend.csv("plant.csv", limit=2)
    assert first["rows"] == [[0, 1, 2], [1, None, None]]
    assert first["nextOffset"] == 2
    second = backend.csv("plant.csv", offset=first["nextOffset"], limit=2)
    assert second["rows"] == [[2, 3, 4], [3, 5, 6]]
    assert backend.csv("plant.csv", stride=2)["rows"] == [[0, 1, 2], [2, 3, 4]]
    (tmp_path / "simulation.pes").write_text("base: {}")
    (tmp_path / "summary.json").write_text(json.dumps({"ok": True}))
    assert backend.summary("summary.json") == {
        "summary": {"ok": True}, "simulationFile": "simulation.pes",
    }


def test_csv_window_on_file_growth(tmp_path):
    table = tmp_path / 'states.csv'
    table.write_text('t,a\n0,1\n1,NaN\n')
    backend = workspace.Workspace(tmp_path)
    first = backend.csv('states.csv', limit=10)
    assert first['eof'] and first['rows'] == [[0, 1], [1, None]]
    with table.open('a') as out:
        out.write('2,3\n3,4\n')
    second = backend.csv('states.csv', offset=first['nextOffset'], limit=10)
    assert second['rows'] == [[2, 3], [3, 4]]


def test_native_host_preview_does_not_execute_addons(tmp_path):
    from peslite.gui.host import Host, WorkspaceError

    (tmp_path / 'PESaddons').mkdir()
    (tmp_path / 'model.pes').write_text('@circuit\nA -- B\n')
    api = Host(tmp_path)
    text = '# comment\n\n@circuit\nA -- B\n'
    response = api.dispatch('preview', {'path': 'model.pes', 'text': text})
    assert response['format'] == 'native' and response['source'] == text
    assert response['diagnostics'][0]['message'] == 'top level must be a mapping'
    with pytest.raises(WorkspaceError, match='trust'):
        api.dispatch('run.start', {'path': 'model.pes'})


def test_native_preview_detection_does_not_depend_on_section_order(tmp_path):
    from peslite.gui.host import Host

    text = 'base: {s_base: 1}\n@circuit\nA -- B\nsimulation: {t_end: 1}\n'
    (tmp_path / 'model.pes').write_text(text)
    response = Host(tmp_path).dispatch('preview', {'path': 'model.pes', 'text': text})
    assert response == {
        'format': 'native', 'model': None, 'diagnostics': [], 'source': text,
    }


def test_debug_plan_is_a_workspace_file_and_previews_its_model(tmp_path):
    from shutil import copyfile
    from conftest import EXAMPLES
    from peslite.gui.host import Host

    copyfile(EXAMPLES / "gfl-example.pes", tmp_path / "case.pes")
    text = (
        "model: case.pes\n\n@debug\n"
        "- vsc.CurrentLoop.u_dq -> vsc.ThreePhaseModulation.u_dq\n"
        "+ Test.out -> vsc.ThreePhaseModulation.u_dq\n\n"
        "@sources\nTest: 0.1\n\n"
        "@probe\neffective: vsc.ThreePhaseModulation.u_dq\n"
    )
    (tmp_path / "control.pesdbg").write_text(text, encoding="utf-8")
    host = Host(tmp_path)
    try:
        assert "control.pesdbg" in {entry["name"] for entry in host.dispatch("tree", {})["files"]}
        result = host.dispatch("debug.inspect", {"path": "control.pesdbg", "text": text})
    finally:
        host.close()

    assert result["modelPath"] == "case.pes"
    assert result["preview"]["format"] == "native"
    assert [operation["kind"] for operation in result["operations"]] == [
        "disconnect", "connect",
    ]
    assert result["watch"] == ["vsc.effective.re", "vsc.effective.im", "vsc.effective.abs"]


def test_csv_byte_cursor_tails_without_rescanning_and_skips_partial_rows(tmp_path):
    file = tmp_path / 'plant.csv'
    file.write_bytes(b't,signal\n0,5\n1,NaN\n2,')
    backend = workspace.Workspace(tmp_path)
    first = backend.csv('plant.csv', limit=100)
    assert first['rows'] == [[0, 5], [1, None]]
    assert first['nextOffset'] == 2
    assert first['eof']
    assert first['nextByteOffset'] == len(b't,signal\n0,5\n1,NaN\n')
    with file.open('ab') as handle:
        handle.write(b'3\n3,4\n')
    second = backend.csv('plant.csv', byte_offset=first['nextByteOffset'],
                         offset=first['nextOffset'], limit=100)
    assert second['rows'] == [[2, 3], [3, 4]]
    assert second['nextOffset'] == 4
    assert second['nextByteOffset'] == file.stat().st_size
    with pytest.raises(workspace.FileConflict):
        backend.csv('plant.csv', byte_offset=file.stat().st_size + 1)


def test_large_csv_windows_scan_a_bounded_number_of_records(tmp_path):
    csv_file = tmp_path / 'states.csv'
    csv_file.write_text('t,v\n' + ''.join(f'{n},{n*2}\n' for n in range(40000)))
    backend = workspace.Workspace(tmp_path)
    first = backend.csv('states.csv', stride=1000, limit=100)
    assert first['rows'] == [[float(i * 1000), float(i * 2000)] for i in range(20)]
    assert first['nextOffset'] == workspace.MAX_SCANNED_ROWS
    assert not first['eof']
    second = backend.csv('states.csv', offset=first['nextOffset'],
                         byte_offset=first['nextByteOffset'], stride=1000, limit=100)
    assert second['rows'] == [[float(i * 1000), float(i * 2000)] for i in range(20, 40)]
    assert second['eof']
