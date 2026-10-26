"""Native commutations follow component equations, state ownership and the common solver."""

import copy
import math
import subprocess
from functools import partial

import numpy as np
import pytest

import peslite
from peslite.assembly.params import from_dict, read_initial
from peslite.solver.model import Bag, Empty, Model
from test_cpp_parameters import _case as _example, _compare_run, _compile, _table
from test_phase_shift import _phase_shift, _ps_tree
from test_scc import CASES, _case, _commanded
from test_wpt_rectifier import LCC, LCC_S, MULTI, SS, _rectifier, _tree


def _short(tree, end=90e-6):
    tree = copy.deepcopy(tree)
    tree['simulation'].update(t_end=end, energy_check='warn')
    tree['simulation']['output'].update(period=250e-9, record_every=1, states=True,
                                        signals=True, energy=True)
    return tree


@pytest.mark.parametrize('link', [SS, LCC, LCC_S, MULTI], ids=['ss', 'lcc', 'lcc_s', 'ss_multi'])
@pytest.mark.parametrize('adaptive', [False, True])
def test_controllerless_rectifier_matches_python(link, adaptive, tmp_path):
    tree = _short(_tree(link, 200.0, 90e-6))
    if adaptive:
        tree['simulation']['solver'].update(type='adaptive', method='DP45', max_step=50e-9)
    params = from_dict(tree)
    binary = _compile(params, tmp_path / 'model', ())
    _compare_run(params, binary, {}, tmp_path / 'run', rtol=2e-8, atol=2e-8)
    names, actual = _table(tmp_path / 'run/native/plant.csv')
    expected_names, expected = _table(tmp_path / 'run/python/plant.csv')
    assert names == expected_names
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8, equal_nan=True)
    assert not (tmp_path / 'run/native/ctrl.charger.secondary.csv').exists()


@pytest.mark.parametrize('case', sorted(CASES))
def test_scc_modes_and_rectification_share_native_commutations(case, tmp_path):
    tree = _short(_case(*CASES[case])[0])
    params = from_dict(tree)
    binary = _compile(params, tmp_path / 'model', ())
    _compare_run(params, binary, {}, tmp_path / 'run', rtol=2e-7, atol=2e-6)
    import json
    native = json.loads((tmp_path / 'run/native/summary.json').read_text())
    python = json.loads((tmp_path / 'run/python/summary.json').read_text())
    for name, expected in python.items():
        if '_scc.' in name:
            assert native[name] == pytest.approx(expected, rel=2e-7, abs=1e-8)


@pytest.mark.parametrize('variables', [('common',), ('all',)])
@pytest.mark.parametrize('restart', [False, True])
def test_native_scc_parameters_events_and_restart(variables, restart, tmp_path):
    tree = _short(_case(*CASES['ss_bridge_alternating'])[0])
    prefix = 'units.charger.primary.ctrl.references.'
    tree['events'] = {
        'phase': {'type': 'set', 't': 47e-6, 'set': {prefix + 'phase': .4}},
        'frequency': {'type': 'set', 't': 61e-6, 'set': {prefix + 'frequency': 78e3}},
    }
    params = from_dict(tree)
    if restart:
        peslite.Simulation(params.replace(**{'simulation.t_end': 38.7e-6})).run(out_dir=tmp_path / 'first')
        params = params.replace(**{'simulation.initial': read_initial(tmp_path / 'first/states.csv')})
    binary = _compile(params, tmp_path / 'model', variables)
    changes = {} if restart else {
        'units.charger.link.primary_scc.angle': 1.3,
        'units.charger.link.primary_scc.sync_angle': 1.1,
        'units.charger.link.primary_scc.capacitance': 70e-9,
        'units.charger.link.mutual_inductance': 18e-6,
    }
    _compare_run(params, binary, changes, tmp_path / 'run', rtol=3e-7, atol=2e-6)


def test_both_sccs_and_carried_commands_use_the_same_native_path(tmp_path):
    tree = _short(_commanded(90e-6))
    primary = tree['units']['charger']['primary']
    primary['ctrl']['type'] = 'resonant'
    primary['ctrl']['loops']['ResonantModulation'] = primary['ctrl']['loops'].pop('PhaseShiftModulation')
    primary['ctrl']['references']['modulation'] = primary['ctrl']['references'].pop('width')
    primary['ctrl']['references'].pop('mode', None)
    secondary = _phase_shift(.8)
    tree['units']['charger']['secondary'] = secondary
    tree['units']['charger']['link']['secondary_scc'] = {
        'capacitance': 50e-9, 'angle': 1.4, 'sync': 'current', 'drive': 'alternating',
    }
    params = from_dict(tree)
    simulation = peslite.Simulation(params)
    assert 'scc_angle' in simulation.system.units['charger.primary'].bridge.register_names
    binary = _compile(params, tmp_path / 'model', ('common',))
    _compare_run(params, binary, {}, tmp_path / 'run', rtol=3e-7, atol=2e-6)


@pytest.mark.parametrize('example,unit,reference', [
    ('gfl', 'vsc', 'id_ref_pu'), ('buck', 'converter', 'voltage'),
    ('wpt', 'charger.primary', 'phase'),
])
def test_auxiliary_registers_follow_the_existing_converter_transport(example, unit, reference, tmp_path):
    params = _example(example, 'switching').replace(**{
        f'units.{unit}.ctrl.outputs': {'monitor': 'references.' + reference},
    })
    params = params.replace(**{'simulation.t_end': max(40e-6, 3 * params.unit(unit).pwm.load_period)})
    binary = _compile(params, tmp_path / 'model', ('all',))
    _compare_run(params, binary, {}, tmp_path / 'run', rtol=1e-7, atol=3e-6)
    names, states = _table(tmp_path / 'run/native/states.csv')
    assert states[-1, names.index(unit + '.pwm.monitor')] == pytest.approx(
        getattr(params.unit(unit).ctrl.references, reference))


def test_nonzero_rectifier_current_is_preserved_before_projection(tmp_path):
    tree = _short(_tree(SS, 200., 1e-6), 1e-6)
    tree['simulation']['initial'] = {'states': {'charger.link.i_mode_1': 1.0}}
    params = from_dict(tree)
    binary = _compile(params, tmp_path / 'model', ())
    _compare_run(params, binary, {}, tmp_path / 'run', rtol=2e-7, atol=3e-6)
    names, states = _table(tmp_path / 'run/native/states.csv')
    assert states[0, names.index('charger.link.i_mode_1')] == 1.0
    assert states[0, names.index('charger.secondary.bridge.direction')] != 0.


@pytest.mark.parametrize('restart', [False, True])
def test_capacitive_rectifier_retains_its_independent_adc(restart, tmp_path):
    tree = _short(_tree(LCC, 200.0, 50e-6, capacitance=4.7e-9), 50e-6)
    tree['simulation']['solver']['dt'] = 5e-9
    tree['units']['charger']['secondary']['meas'] = {
        'period': 1e-6, 'window': .5e-6, 'average': ['current', 'power', 'vin'],
        'rms': {'current': 2e-6},
    }
    params = from_dict(tree)
    if restart:
        peslite.Simulation(params.replace(**{'simulation.t_end': 23.7e-6})).run(out_dir=tmp_path / 'first')
        params = params.replace(**{'simulation.initial': read_initial(tmp_path / 'first/states.csv')})
    binary = _compile(params, tmp_path / 'model', ('all',))
    _compare_run(params, binary, {}, tmp_path / 'run', rtol=3e-7, atol=3e-6)
    names, actual = _table(tmp_path / 'run/native/plant.csv')
    expected_names, expected = _table(tmp_path / 'run/python/plant.csv')
    assert names == expected_names
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-6, equal_nan=True)


class _ResetState(Bag):
    __slots__ = ('x',)


class _ResetOutput(Bag):
    __slots__ = ('x',)


class _Reset:
    state_names = ('x',)
    outputs_need_inputs = False

    def __init__(self):
        self.state, self.inp, self.out = _ResetState(x=0j), Empty(), _ResetOutput(x=0j)
        self.located, self.scheduled = False, 0
        self.time, self.next = 0.0, .3333333

    def set_outputs(self, t):
        self.time, self.out.x = t, self.state.x

    def rhs(self, t):
        return (1.0 + 2.0j,)

    def commutation_margin(self):
        return math.inf if self.located else min(0.0, 1.0 - self.state.x.real)

    def next_commutation_time(self):
        return self.next

    def commutate(self, t):
        if abs(t - self.next) < 1e-12:
            self.scheduled += 1
            self.next += .4
        else:
            self.located = True
            self.state.x = -.5 + 1.0j

    def equation_state_bindings(self):
        # Binding labels describe storage; they must not become conflicting C++ identifiers.
        return {'clock.next': (self, 'next'), 'clock_next': (self, 'time'),
                **{name: (self, name) for name in ('located', 'scheduled')}}

    def get_state(self):
        return {'located': self.located, 'scheduled': self.scheduled, 'next': self.next}

    def set_state(self, values):
        for key, value in values.items():
            setattr(self, key, value)


def _reset_system(params, component=_Reset):
    system = peslite.System(params)
    original = system.model
    system.model = Model({**dict(zip(original.names, original.subsystems)), 'custom': component()},
                         original.connections)
    return system


@pytest.mark.parametrize('adaptive', [False, True])
def test_generic_commutation_locates_plateaus_and_resets_complex_states(adaptive, tmp_path):
    tree = _ps_tree(_rectifier(400.), _rectifier(200.), 1.6)
    tree['simulation'].update(energy_check='off')
    tree['simulation']['solver'].update(dt=.08, type='adaptive' if adaptive else 'fixed',
                                        method='DP45' if adaptive else 'rk4', max_step=.08)
    tree['simulation']['output'].update(period=.1, states=True)
    params = from_dict(tree)
    parts = {'system': _reset_system}
    simulation = peslite.Simulation(params, parts=parts)
    binary = _compile(simulation, tmp_path / 'model', ())
    _compare_run(params, binary, {}, tmp_path / 'run', rtol=1e-12, atol=1e-12, parts=parts)
    names, rows = _table(tmp_path / 'run/native/states.csv')
    assert rows[-1, names.index('custom.x.re')] == pytest.approx(.1, abs=1e-13)
    assert rows[-1, names.index('custom.scheduled')] == 4


class _Stalled(_Reset):
    def commutation_margin(self):
        return -1.0 if self.state.x.real > .4 else 1.0

    def next_commutation_time(self):
        return math.inf

    def commutate(self, t):
        pass


def test_native_commutation_reports_no_progress_instead_of_hanging(tmp_path):
    from peslite.assembly.conduction import CommutationError

    tree = _ps_tree(_rectifier(400.), _rectifier(200.), .8)
    tree['simulation'].update(energy_check='off')
    tree['simulation']['solver'].update(dt=.08, type='fixed', method='rk4')
    tree['simulation']['output'].update(period=.1, states=True)
    simulation = peslite.Simulation(from_dict(tree), parts={'system': partial(_reset_system, component=_Stalled)})
    binary = _compile(simulation, tmp_path / 'model', ())
    native = subprocess.run([str(binary), '--out', str(tmp_path / 'native')], capture_output=True, text=True,
                            timeout=10)
    assert native.returncode != 0
    assert 'without time advancing' in native.stderr
    with pytest.raises(CommutationError, match='without time advancing'):
        simulation.run(out_dir=tmp_path / 'python')
