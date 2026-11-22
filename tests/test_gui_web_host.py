"""Loopback HTTP security and conflict tests for the optional GUI host."""

import http.client
import json
import threading

import pytest

from peslite.gui.web_service import LocalServer


pytestmark = pytest.mark.ui


def test_local_http_requires_origin_token_and_keeps_workspace_read_only(tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'index.html').write_text('<meta content="__PESLITE_TOKEN__">')
    server = LocalServer(0, tmp_path, assets=assets)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        assert server.server_address[0] == '127.0.0.1'
        base = f'127.0.0.1:{server.server_port}'
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
        connection.request('GET', '/', headers={'Host': base})
        index = connection.getresponse()
        assert index.status == 200 and server.token in index.read().decode()

        def send(payload, *, token=server.token, origin=None):
            headers = {'Host': base, 'Content-Type': 'application/json',
                       'X-PESLite-Token': token}
            if origin:
                headers['Origin'] = origin
            connection.request('POST', '/api', body=json.dumps(payload), headers=headers)
            reply = connection.getresponse()
            return reply.status, json.loads(reply.read())

        write = {'method': 'file.write', 'params': {'path': 'model.pes', 'text': 'changed'}}
        assert send(write, token='bad')[0] == 403
        assert send(write, origin='https://example.com')[0] == 403
        (tmp_path / 'model.pes').write_text('@circuit\nA -- B\n')
        status, rejected = send(write)
        assert status == 400 and not rejected['ok']
        assert (tmp_path / 'model.pes').read_text() == '@circuit\nA -- B\n'
        status, _ = send({'method': 'file.read', 'params': {'path': '../wrong.pes'}})
        assert status == 400
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_run_reports_process_start_failure_without_executing_a_model(tmp_path):
    from peslite.gui.host import Host

    (tmp_path / 'case.pes').write_text('base: {s_base: 1}\n')
    host = Host(tmp_path, python='/nonexistent-peslite-python')
    try:
        host.dispatch('run.start', { 'path': 'case.pes' })
    except OSError as exc:
        assert 'nonexistent-peslite-python' in str(exc)
    else:
        raise AssertionError('The host must reject a missing Python interpreter')
    assert host.dispatch('run.status', {})['running'] is False


def test_web_host_exposes_bounded_csv_byte_cursor(tmp_path):
    from peslite.gui.host import Host

    csv_path = tmp_path / 'energy.csv'
    csv_path.write_text('t,total\n0,10\n1,NaN\n2,12\n')
    host = Host(tmp_path)
    first = host.dispatch('csv.read', {'path': 'energy.csv', 'limit': 2})
    assert first['rows'] == [[0, 10], [1, None]]
    assert first['nextByteOffset'] > 0
    tail = host.dispatch('csv.read', {'path': 'energy.csv',
                  'offset': first['nextOffset'], 'byteOffset': first['nextByteOffset']})
    assert tail['rows'] == [[2, 12]]


@pytest.mark.parametrize("method", [
    "file.write", "file.create", "file.rename", "file.delete", "editor.applyPatch",
    "parameter.patch", "component.add", "component.remove", "connection.rewire",
])
def test_gui_host_has_no_source_mutation_operation(tmp_path, method):
    from peslite.gui.host import Host
    from peslite.gui.workspace import WorkspaceError

    host = Host(tmp_path)
    with pytest.raises(WorkspaceError, match="unknown GUI operation"):
        host.dispatch(method, {})
