"""File-backed frontend checks independent of the PESLite simulation runtime."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "src/peslite/frontend/workspace.py"
spec = importlib.util.spec_from_file_location("peslite_workspace_standalone", MODULE)
workspace = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = workspace
spec.loader.exec_module(workspace)


def test_yaml_patch_keeps_comments_and_flow_mapping():
    original = ('# project title\nbase: {s_base: 2.0e6, f0: 50} # same\n'
                'simulation:\n  t_end: 1.0 # preserve\n  solver: {dt: 25e-6}\n')
    edit = workspace.patch_scalar(original, "simulation.solver.dt", 0.00005)
    changed = workspace.patched_text(original, edit)
    assert changed == original.replace("25e-6", "5e-05")
    assert "# preserve" in changed and changed.startswith("# project title")
    assert edit["line"] == 5


def test_rejects_ambiguous_or_destructive_source_changes():
    with pytest.raises(workspace.WorkspaceError, match="duplicated"):
        workspace.patch_scalar("base: {s_base: 1, s_base: 2}\n", "base.s_base", 3)
    with pytest.raises(workspace.WorkspaceError, match="aliased"):
        workspace.patch_scalar("base: {s_base: &amount 1}\nsimulation: {t_end: *amount}\n", "base.s_base", 2)
    with pytest.raises(workspace.WorkspaceError, match="only explicit"):
        workspace.patch_scalar("simulation:\n  solver:\n    dt: {value: 1}\n", "simulation.solver.dt", 4)


def test_versioned_atomic_writes_and_conflicts(tmp_path):
    backend = workspace.Workspace(tmp_path)
    created = backend.write("case.pes", "base: {s_base: 10}\n", None, create=True)
    version = created["version"]
    assert backend.read("case.pes")["version"] == version
    changed = backend.write("case.pes", "base: {s_base: 20}\n", version)
    with pytest.raises(workspace.FileConflict):
        backend.write("case.pes", "base: {s_base: 30}\n", version)
    assert backend.read("case.pes")["text"] == "base: {s_base: 20}\n"
    backend.rename("case.pes", "renamed.pes", changed["version"])
    with pytest.raises(workspace.FileConflict):
        backend.delete("renamed.pes", version)
    backend.delete("renamed.pes", backend.read("renamed.pes")["version"])
    assert not (tmp_path / "renamed.pes").exists()


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


def test_source_positions_include_flow_values_and_survive_unicode():
    text = 'meta: {title: "⚡ power"}\n# untouched\nsimulation: {t_end: 0.05}\n'
    edit = workspace.patch_scalar(text, 'simulation.t_end', 0.1)
    result = workspace.patched_text(text, edit)
    assert result == text.replace('t_end: 0.05', 't_end: 0.1')
    assert edit['line'] == 3


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
    from peslite.frontend.host import Host, WorkspaceError

    (tmp_path / 'PESaddons').mkdir()
    (tmp_path / 'model.pes').write_text('@circuit\nA -- B\n')
    api = Host(tmp_path)
    response = api.dispatch('preview', { 'path': 'model.pes', 'text': '# comment\n\n@circuit\nA -- B\n' })
    assert response == {'format': 'native', 'model': None, 'fields': [], 'diagnostics': []}
    with pytest.raises(WorkspaceError, match='trust'):
        api.dispatch('run.start', {'path': 'model.pes'})


def test_component_insertion_is_a_single_source_range_edit():
    text = ('# keep the model header\nbase: {s_base: 2e6}\n'
            'buses:\n  pcc:\n    c_pu: 0.02 # preserve\n'
            'simulation:\n  t_end: 1\n')
    patch = workspace.add_component(text, 'buses', 'aux', '{phases: 3, c_pu: 0.05}')
    assert patch['start'] == patch['end']
    modified = workspace.patched_text(text, patch)
    assert modified.startswith('# keep the model header\nbase: {s_base: 2e6}\n')
    assert 'c_pu: 0.02 # preserve\n  aux:' in modified
    assert '  simulation:' not in modified
    assert modified.endswith('simulation:\n  t_end: 1\n')
    import yaml
    parsed = yaml.safe_load(modified)
    assert parsed['buses']['aux'] == {'phases': 3, 'c_pu': 0.05}


def test_component_insertion_creates_missing_section_in_file_order():
    source = 'base:\n  s_base: 2e6\nsimulation:\n  t_end: 1.0\n'
    patch = workspace.add_component(source, 'buses', 'pcc', '{phases: 3, c_pu: 0.02}')
    result = workspace.patched_text(source, patch)
    assert result.index('buses:') < result.index('simulation:')
    import yaml
    assert yaml.safe_load(result)['buses']['pcc']['phases'] == 3


def test_component_edit_refuses_ambiguous_and_destructive_changes():
    with pytest.raises(workspace.WorkspaceError, match='flow-style'):
        workspace.add_component('buses: {pcc: {phases: 3}}\n', 'buses', 'bus2', '{}')
    with pytest.raises(workspace.WorkspaceError, match='already exists'):
        workspace.add_component('buses:\n  pcc: {phases: 3}\n', 'buses', 'pcc', '{}')
    with pytest.raises(workspace.WorkspaceError, match='definition must'):
        workspace.add_component('buses:\n  pcc: {}\n', 'buses', 'other', '[1, 2]')
    with pytest.raises(workspace.WorkspaceError, match='comment'):
        workspace.remove_component('buses:\n  a: {phases: 3} # keep\n  b: {}\n', 'buses', 'a')
    with pytest.raises(workspace.WorkspaceError, match='last entry'):
        workspace.remove_component('buses:\n  pcc: {}\n', 'buses', 'pcc')


def test_removal_of_unreferenced_simple_component_is_minimal():
    text = 'buses:\n  a: {phases: 3}\n  b: {phases: 3}\nsimulation: {t_end: 1}\n'
    change = workspace.remove_component(text, 'buses', 'a')
    assert workspace.patched_text(text, change) == (
        'buses:\n  b: {phases: 3}\nsimulation: {t_end: 1}\n')


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
