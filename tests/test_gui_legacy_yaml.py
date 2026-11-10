"""Integration tests for the optional legacy-YAML GUI function addon."""

from pathlib import Path

import pytest

from conftest import EXAMPLES
from peslite.addons.functions import legacy_yaml_patch, legacy_yaml_preview
from peslite.gui.workspace import patched_text, WorkspaceError


@pytest.mark.parametrize('case', ['buck-example', 'two-converters-example', 'wpt-example'])
def test_legacy_samples_normalize_without_rewriting(case):
    path = EXAMPLES / f'{case}.pes'
    text = path.read_text(encoding='utf-8')
    response = legacy_yaml_preview(text, path)
    assert response['diagnostics'] == []
    model = response['model']
    assert model['defaultView'] == 'circuit'
    assert model['views'][0]['nodes']
    assert model['views'][0]['edges']
    assert any(field['path'] == 'simulation.t_end' for field in response['fields'])
    assert path.read_text(encoding='utf-8') == text


def test_legacy_controller_scoping_and_event_boundaries():
    path = EXAMPLES / 'two-converters-example.pes'
    response = legacy_yaml_preview(path.read_text(encoding='utf-8'), path)
    assert response['diagnostics'] == []
    views = {view['name']: view for view in response['model']['views']}
    assert {'circuit', 'vsc_1', 'vsc_2'} <= set(views)
    assert {'PSC', 'VirtualImpedance'} <= {node['name'] for node in views['vsc_1']['nodes']}
    assert {'CurrentLoop', 'SRFPLL'} <= {node['name'] for node in views['vsc_2']['nodes']}
    assert response['model']['timeline']
    assert [segment['start'] for segment in response['model']['timeline']] == [0, .05, .5, .75]
    assert response['model']['timeline'][0]['disconnected']


def test_legacy_validated_range_edit_preserves_original_syntax_and_comments():
    path = EXAMPLES / 'buck-example.pes'
    text = path.read_text(encoding='utf-8')
    edit = legacy_yaml_patch(text, path, 'simulation.t_end', 0.05)
    changed = patched_text(text, edit)
    assert text != changed
    assert changed.splitlines()[0] == text.splitlines()[0]
    assert 't_end: 0.05' in changed
    assert legacy_yaml_preview(changed, path)['diagnostics'] == []
    assert path.read_text(encoding='utf-8') == text


def test_invalid_yaml_is_diagnostic_and_rejected_without_writes():
    path = Path('model.pes')
    text = 'base:\n  s_base: [\n'
    result = legacy_yaml_preview(text, path)
    assert result['model'] is None
    assert result['diagnostics'][0]['severity'] == 'error'
    assert result['diagnostics'][0]['range']['start']['line'] == 3
    with pytest.raises(WorkspaceError):
        legacy_yaml_patch(text, path, 'base.s_base', 1)


def test_legacy_adapter_is_a_lazy_function_addon():
    from peslite.addons import functions

    assert 'legacy_yaml_preview' in functions.__all__
    assert 'legacy_yaml_patch' in functions.__all__
    assert 'legacy_yaml_preview' not in functions._ROOT_EXPORTS


def test_additional_component_preserves_text_and_validates_model():
    from peslite.addons.functions import (
        legacy_yaml_add_component, legacy_yaml_remove_component,
    )
    path = EXAMPLES / 'buck-example.pes'
    original = path.read_text(encoding='utf-8')
    added = legacy_yaml_add_component(
        original, path, 'elements', 'test_resistor',
        '{type: resistor, bus: output, r: 12.0}',
    )
    expanded = patched_text(original, added)
    assert expanded != original
    assert expanded.startswith(original[:original.index('elements:')])
    assert expanded.splitlines()[0] == original.splitlines()[0]
    assert '  test_resistor:' in expanded
    assert legacy_yaml_preview(expanded, path)['diagnostics'] == []
    removed = legacy_yaml_remove_component(expanded, path, 'elements', 'test_resistor')
    restored = patched_text(expanded, removed)
    assert legacy_yaml_preview(restored, path)['diagnostics'] == []
    assert restored == original
    assert path.read_text(encoding='utf-8') == original


def test_connection_rewire_checks_existing_buses_and_preserves_source():
    from peslite.addons.functions import legacy_yaml_rewire
    path = EXAMPLES / 'two-converters-example.pes'
    original = path.read_text(encoding='utf-8')
    with pytest.raises(WorkspaceError, match='unknown destination'):
        legacy_yaml_rewire(original, path, 'sources.grid.bus', 'not_a_bus')
    edit = legacy_yaml_rewire(original, path, 'sources.grid.bus', 'pcc')
    updated = patched_text(original, edit)
    assert legacy_yaml_preview(updated, path)['diagnostics'] == []
    assert 'bus: "pcc"' in updated
    assert path.read_text(encoding='utf-8') == original
    with pytest.raises(WorkspaceError, match='only component terminals'):
        legacy_yaml_rewire(original, path, 'simulation.t_end', 'pcc')


def test_removing_a_connected_bus_is_rejected():
    from peslite.addons.functions import legacy_yaml_remove_component
    path = EXAMPLES / 'two-converters-example.pes'
    original = path.read_text(encoding='utf-8')
    with pytest.raises(WorkspaceError):
        legacy_yaml_remove_component(original, path, 'buses', 'pcc')
    assert path.read_text(encoding='utf-8') == original


def test_addon_edits_are_public_lazy_functions():
    from peslite.addons import functions
    from peslite.addons.functions import (
        legacy_yaml_add_component, legacy_yaml_remove_component, legacy_yaml_rewire,
    )
    for function in (legacy_yaml_add_component, legacy_yaml_remove_component, legacy_yaml_rewire):
        assert callable(function)
    assert all(function in functions.__all__ for function in (
        'legacy_yaml_add_component', 'legacy_yaml_remove_component', 'legacy_yaml_rewire'))
    assert not any(function in functions._ROOT_EXPORTS for function in (
        'legacy_yaml_add_component', 'legacy_yaml_remove_component', 'legacy_yaml_rewire'))
