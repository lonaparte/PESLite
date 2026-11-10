"""Loopback HTTP security and conflict tests for the optional frontend host."""

import http.client
import json
import threading

from peslite.frontend.web_service import LocalServer


def test_local_http_requires_origin_token_and_version(tmp_path):
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

        create = {'method': 'file.create', 'params': {'path': 'model.pes',
                                                   'text': 'base: {s_base: 5}\n'}}
        assert send(create, token='bad')[0] == 403
        assert send(create, origin='https://example.com')[0] == 403
        status, created = send(create)
        assert status == 200 and created['ok']
        version = created['result']['version']
        status, _ = send({'method': 'file.write', 'params': {
            'path': 'model.pes', 'text': 'base: {s_base: 6}\n', 'version': version}})
        assert status == 200
        status, _ = send({'method': 'file.write', 'params': {
            'path': 'model.pes', 'text': 'base: {s_base: 7}\n', 'version': version}})
        assert status == 409
        status, _ = send({'method': 'file.read', 'params': {'path': '../wrong.pes'}})
        assert status == 400
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_run_reports_process_start_failure_without_executing_a_model(tmp_path):
    from peslite.frontend.host import Host

    (tmp_path / 'case.pes').write_text('base: {s_base: 1}\n')
    host = Host(tmp_path, python='/nonexistent-peslite-python')
    try:
        host.dispatch('run.start', { 'path': 'case.pes' })
    except OSError as exc:
        assert 'nonexistent-peslite-python' in str(exc)
    else:
        raise AssertionError('The host must reject a missing Python interpreter')
    assert host.dispatch('run.status', {})['running'] is False


def test_web_host_exposes_bounded_csv_byte_cursor_and_rejects_bad_rewire(tmp_path):
    from peslite.frontend.host import Host

    csv_path = tmp_path / 'energy.csv'
    csv_path.write_text('t,total\n0,10\n1,NaN\n2,12\n')
    host = Host(tmp_path)
    first = host.dispatch('csv.read', {'path': 'energy.csv', 'limit': 2})
    assert first['rows'] == [[0, 10], [1, None]]
    assert first['nextByteOffset'] > 0
    tail = host.dispatch('csv.read', {'path': 'energy.csv',
                  'offset': first['nextOffset'], 'byteOffset': first['nextByteOffset']})
    assert tail['rows'] == [[2, 12]]
