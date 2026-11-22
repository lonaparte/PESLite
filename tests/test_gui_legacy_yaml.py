"""Integration tests for the optional legacy-YAML GUI function addon."""

from pathlib import Path

import pytest
import yaml

from conftest import EXAMPLES
from peslite.assembly.syntax import parse_native_text
from peslite.addons.functions.legacy_yaml import read_simulation
from peslite.gui.legacy_yaml import preview as legacy_yaml_preview


pytestmark = pytest.mark.ui


def _legacy_source(path: Path) -> str:
    source = path.read_text(encoding='utf-8')
    tree = parse_native_text(source, path)[0]
    return source.splitlines()[0] + '\n' + yaml.safe_dump(tree, sort_keys=False)


@pytest.mark.parametrize('case', ['buck-example', 'two-converters-example', 'wpt-example'])
def test_legacy_samples_normalize_without_rewriting(case):
    path = EXAMPLES / f'{case}.pes'
    text = _legacy_source(path)
    response = legacy_yaml_preview(text, path)
    assert response['diagnostics'] == []
    model = response['model']
    assert model['defaultView'] == 'circuit'
    assert model['views'][0]['nodes']
    assert model['views'][0]['edges']
    assert _legacy_source(path) == text


def test_legacy_controller_scoping_and_event_boundaries():
    path = EXAMPLES / 'two-converters-example.pes'
    response = legacy_yaml_preview(_legacy_source(path), path)
    assert response['diagnostics'] == []
    views = {view['name']: view for view in response['model']['views']}
    assert {'circuit', 'vsc_1', 'vsc_2'} <= set(views)
    assert {'PSC', 'VirtualImpedance'} <= {node['name'] for node in views['vsc_1']['nodes']}
    assert {'CurrentLoop', 'SRFPLL'} <= {node['name'] for node in views['vsc_2']['nodes']}
    assert response['model']['timeline']
    assert [segment['start'] for segment in response['model']['timeline']] == [0, .05, .5, .75]
    assert response['model']['timeline'][0]['disconnected']


def test_invalid_yaml_is_a_diagnostic():
    path = Path('model.pes')
    text = 'base:\n  s_base: [\n'
    result = legacy_yaml_preview(text, path)
    assert result['model'] is None
    assert result['diagnostics'][0]['severity'] == 'error'
    assert result['diagnostics'][0]['range']['start']['line'] == 3


def test_legacy_reader_and_gui_preview_keep_separate_module_boundaries():
    assert read_simulation.__module__ == "peslite.addons.functions.legacy_yaml"
    assert legacy_yaml_preview.__module__ == "peslite.gui.legacy_yaml"
