"""Native lowering must follow the assembled equations rather than their class names."""

from types import SimpleNamespace
import subprocess

import numpy as np
import pytest

from peslite.assembly.cpp import (_ALGEBRAIC_SOLVER, _ControlGenerator, _ModelGenerator,
                                  _RuntimeParameters, _number, _uses_implementation)
from peslite.components import Bag
from peslite.components.converter import DCCapacitor, DCLink, DCVoltageSource
from peslite.components.network import ACVoltageSource, RLBranch, RCNode
from peslite.solver.model import Empty, Model
from peslite.control import Loop
from peslite.control.blocks import PI
from peslite.control.loops import DUTY_RATIO
from test_cpp_export import _compiler

pytestmark = pytest.mark.full


def test_parameter_identifier_collisions_are_rejected_by_the_shared_interface():
    with pytest.raises(ValueError, match="parameter names collide"):
        _RuntimeParameters({"same.name": 1.0, "same_name": 2.0}, ("all",))


def _generator(model):
    system = SimpleNamespace(model=model, units={}, sources={}, named_elements={})
    simulation = SimpleNamespace(system=system, p=SimpleNamespace(changes=()))
    return _ModelGenerator(simulation, None)


def _run_model(model, readings, tmp_path, *, times=(0.0,), initial_state=None,
               check_floating_exceptions=False):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++17 compiler installed")
    generator = _generator(model)
    body = generator.body(derivatives=False)
    expressions = [generator.value(record, field) for record, field in readings]
    source = tmp_path / "equations.cpp"
    initial = ", ".join(_number(value) for value in model.get_initial_values())
    source.write_text(
        '#include <algorithm>\n#include <array>\n#include <cmath>\n#include <complex>\n#include <cfenv>\n#include <cstdlib>\n'
        '#include <iomanip>\n#include <iostream>\n#include <limits>\n#include <stdexcept>\n'
        'using Complex = std::complex<double>;\n' + _ALGEBRAIC_SOLVER
        + '\nstruct Probe {\n' + '\n'.join(generator.members())
        + f'\nstd::array<double, {model.n_states}> y{{{{{initial}}}}};\n'
        + 'void run(double t) {\n' + '\n'.join(body)
        + '\nstd::cout << std::setprecision(17)'
        + ''.join(f" << std::real({item}) << ' ' << std::imag({item}) << ' '"
                  for item in expressions)
        + ';\n}\n};\nint main(int argc, char** argv) { try { Probe probe; '
        + ('if (argc > 1) probe.y[0] = std::strtod(argv[1], nullptr); ' if initial_state is not None else '')
        + ('std::feclearexcept(FE_ALL_EXCEPT); ' if check_floating_exceptions else '')
        + ''.join(f'probe.run({_number(t)});' for t in times)
        + ('if (std::fetestexcept(FE_INVALID | FE_DIVBYZERO)) return 3; ' if check_floating_exceptions else '') + '} '
        + 'catch(const std::exception& e) { std::cerr << e.what(); return 2; } }\n',
        encoding="utf-8",
    )
    executable = tmp_path / "equations"
    subprocess.run([compiler, "-O2", "-std=c++17", str(source), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable), *([] if initial_state is None else [str(initial_state)])],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    values = np.fromstring(result.stdout, sep=" ").reshape(-1, 2)
    return values[:, 0] + 1j * values[:, 1]


class _Input(Bag):
    __slots__ = ("x",)


class _Output(Bag):
    __slots__ = ("value",)


class _RepeatedEquation:
    state_names = ("x",)
    outputs_need_inputs = False

    def __init__(self):
        self.state = _Input(x=0.0)
        self.inp = Empty()
        self.out = _Output()

    def set_outputs(self, t):
        x = self.state.x
        common = (x + 1.25) * (x + 2.5) * (x + 3.75)
        self.out.value = common + common * (1.0 / x + np.sqrt(x)) if x > 0.0 else common

    def rhs(self, t):
        return (0.0,)


@pytest.mark.parametrize("value", [-1.0, 0.0, 2.0])
def test_shared_expressions_preserve_guarded_division_and_square_root(value, tmp_path):
    component = _RepeatedEquation()
    component.state.x = value
    model = Model({"component": component}, {})
    component.set_outputs(0.0)
    actual, = _run_model(model, [(component.out, "value")], tmp_path,
                         initial_state=value, check_floating_exceptions=True)
    assert actual == pytest.approx(component.out.value)


class _Feedback:
    state_names = ()
    outputs_need_inputs = True

    def __init__(self, gain, offset):
        self.gain, self.offset = gain, offset
        self.state = Empty()
        self.inp = _Input(x=0j if isinstance(gain, complex) else 0.0)
        self.out = _Output(value=self.inp.x)

    def set_outputs(self, t):
        self.out.value = self.gain * self.inp.x + self.offset

    def rhs(self, t):
        return ()


@pytest.mark.parametrize("gain", [2.0, -2.0, 0.99, 2.0 + 0.5j])
def test_native_algebraic_scalar_uses_the_same_secant_solve(gain, tmp_path):
    component = _Feedback(gain, 1.0)
    model = Model({"feedback": component}, {(component, "x"): (component, "value")})
    loop, = model.algebraic_loops
    loop.max_sweeps, loop.atol, loop.rtol = 25, 1e-13, 1e-12
    actual, = _run_model(model, [(component.out, "value")], tmp_path)
    model.sync(0.0, model.get_initial_values())
    assert actual == pytest.approx(component.out.value, rel=1e-12, abs=1e-12)
    assert actual == pytest.approx(1.0 / (1.0 - gain), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("gain", [0.0, 0.5])
def test_native_export_keeps_the_shared_plan_for_zero_and_nonzero_feedback(gain, tmp_path):
    component = _Feedback(gain, 3.0)
    model = Model({"feedback": component}, {(component, "x"): (component, "value")})
    generator = _generator(model)
    assert len(model.algebraic_loops) == 1
    assert any('solve_algebraic(' in line for line in generator.body(derivatives=False)) == bool(gain)
    assert not any('algebraic_value_' in line for line in generator.members())
    actual, = _run_model(model, [(component.out, "value")], tmp_path)
    assert actual == pytest.approx(3.0 / (1.0 - gain))


def test_static_dependencies_can_order_an_existing_native_execution_block():
    fixed, adapter = _Feedback(0.0, 3.0), _Feedback(0.5, 0.0)
    model = Model({'fixed': fixed, 'adapter': adapter}, {
        (fixed, 'x'): (adapter, 'value'), (adapter, 'x'): (fixed, 'value'),
    })
    generator = _generator(model)
    statement = (f'{generator.value(adapter.out, "value")} = '
                 f'0.5 * {generator.value(adapter.inp, "x")};')
    generator.continuous_units[id(adapter)] = SimpleNamespace(output_lines=lambda: [statement])
    generator._algebraic_plans.clear()
    plan = generator._algebraic_plan(model.algebraic_loops[0])
    assert plan is not None and plan.count(statement) == 1
    copy = (f'{generator.value(adapter.inp, "x")} = '
            f'{generator.value(fixed.out, "value")};')
    assert plan.index(copy) < plan.index(statement)


def test_native_algebraic_nonconvergence_is_an_error(tmp_path):
    component = _Feedback(1.0, 1.0)
    model = Model({"feedback": component}, {(component, "x"): (component, "value")})
    model.algebraic_loops[0].max_sweeps = 3
    with pytest.raises(FloatingPointError, match="did not converge"):
        model.sync(0.0, model.get_initial_values())
    with pytest.raises(RuntimeError, match="did not converge"):
        _run_model(model, [(component.out, "value")], tmp_path)


def test_native_multivariable_algebraic_loop_uses_its_configured_limit(tmp_path):
    class Pair(Bag):
        __slots__ = ("x", "y")

    class Coupled:
        state_names = ()
        outputs_need_inputs = True

        def __init__(self):
            self.state = Empty()
            self.inp, self.out = Pair(), Pair()

        def set_outputs(self, t):
            self.out.x = 0.4 * self.inp.x + 0.2 * self.inp.y + 1.0
            self.out.y = 0.1 * self.inp.x + 0.5 * self.inp.y + 1.0

        def rhs(self, t):
            return ()

    component = Coupled()
    model = Model({"coupled": component}, {
        (component, name): (component, name) for name in ("x", "y")
    })
    loop, = model.algebraic_loops
    assert len(loop.tears) == 2
    loop.max_sweeps, loop.rtol, loop.atol = 80, 1e-13, 1e-14
    readings = [(component.out, name) for name in ("x", "y")]
    actual = _run_model(model, readings, tmp_path)
    model.sync(0.0, model.get_initial_values())
    np.testing.assert_allclose(actual, [component.out.x, component.out.y], rtol=1e-12)


def test_native_multivariable_feedback_keeps_the_previous_output_between_calls(tmp_path):
    class Pair(Bag):
        __slots__ = ('x', 'y')

    class Coupled:
        state_names = ()
        outputs_need_inputs = True

        def __init__(self):
            self.state = Empty()
            self.inp, self.out = Pair(x=1.0, y=1.0), Pair(x=1.0, y=1.0)

        def set_outputs(self, t):
            self.out.x = 0.8 * self.inp.x + 0.05 * self.inp.y + 0.15 * (1.0 + t)
            self.out.y = 0.05 * self.inp.x + 0.8 * self.inp.y + 0.15 * (1.0 + t)

        def rhs(self, t):
            return ()

    component = Coupled()
    model = Model({'feedback': component}, {
        (component, name): (component, name) for name in ('x', 'y')
    })
    times = tuple(index * 1e-7 for index in range(101))
    readings = [(component.out, name) for name in ('x', 'y')]
    actual = _run_model(model, readings, tmp_path, times=times).reshape(-1, 2)
    expected = []
    for t in times:
        model.sync(t, model.get_initial_values())
        expected.append((component.out.x, component.out.y))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_native_inherited_source_keeps_the_existing_adapter():
    class AnnotatedSource(ACVoltageSource):
        pass

    source = AnnotatedSource(314.0, 325.0)
    generator = _generator(Model({'source': source}, {}))
    assert any('std::cos' in line for line in generator._output(source.set_outputs))


@pytest.mark.parametrize("phases", [1, 3])
def test_native_ac_source_preserves_the_connected_phase_representation(phases, tmp_path):
    source = ACVoltageSource(2.0 * np.pi * 50.0, 325.0, phases=phases)
    model = Model({"source": source}, {})
    times = (0.0, 1.0e-3, 2.5e-3)
    actual = _run_model(model, [(source.out, "e_g")], tmp_path, times=times)
    expected = []
    for time in times:
        source.set_outputs(time)
        expected.append(source.out.e_g)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


class _ModifiedBranch(RLBranch):
    def set_outputs(self, t):
        self.out.i = 2.0 * self.state.i

    def rhs(self, t):
        return (7.0,)


@pytest.mark.parametrize("instance_override", [False, True])
def test_native_branch_keeps_its_own_output_and_rhs(instance_override, tmp_path):
    cls = RLBranch if instance_override else _ModifiedBranch
    branch = cls(2.0, 1.0, i0=3.0, phases=1)
    if instance_override:
        branch.set_outputs = _ModifiedBranch.set_outputs.__get__(branch)
        branch.rhs = _ModifiedBranch.rhs.__get__(branch)
    model = Model({"branch": branch}, {})
    generator = _generator(model)
    assert generator._rhs(branch) == ["7.0"]
    actual, = _run_model(model, [(branch.out, "i")], tmp_path)
    assert actual == 6.0


def test_native_unilateral_output_clamps_a_negative_trial_state(tmp_path):
    branch = RLBranch(1.0, 0.0, i0=-1.0, phases=1, unilateral=True)
    model = Model({"branch": branch}, {})
    actual, = _run_model(model, [(branch.out, "i")], tmp_path)
    model.sync(0.0, model.get_initial_values())
    assert actual == branch.out.i == 0.0


def test_builtin_subclass_export_hooks_take_precedence():
    class HookedNode(RCNode):
        def cpp_outputs(self, generator):
            return ["custom_output;"]

        def cpp_rhs(self, generator):
            return ["custom_rhs"]

    node = HookedNode(1.0, 0.0, phases=1)
    generator = _generator(Model({"node": node}, {}))
    assert generator._output(node.set_outputs) == ["custom_output;"]
    assert generator._rhs(node) == ["custom_rhs"]


@pytest.mark.parametrize("capacitance", [None, 0.1])
def test_dclink_lowering_traces_the_shared_capacitor_state(capacitance, tmp_path):
    cap = DCCapacitor(12.0, capacitance, 0.25) if capacitance is not None else None
    link = DCLink(cap, DCVoltageSource(24.0, 2.0))
    link.inp.i_dc = 3.0
    model = Model({"link": link}, {})
    readings = [(link.out, name) for name in ("u_dc", "u_C", "i_src")]
    actual = _run_model(model, readings, tmp_path)
    model.sync(0.0, model.get_initial_values())
    np.testing.assert_allclose(actual, [getattr(record, name) for record, name in readings],
                               rtol=1e-14, atol=1e-14)


def _control_generator(node):
    generator = object.__new__(_ControlGenerator)
    generator.name, generator.prefix, generator.continuous = "test", "test_", False
    generator.graph = SimpleNamespace(nodes={"node": node}, _limits={}, periods={"node": 1.0})
    generator.cfg = SimpleNamespace(ctrl=SimpleNamespace(period=1.0))
    generator.ctrl = SimpleNamespace()
    generator.model = _generator(Model({}, {}))
    return generator


@pytest.mark.parametrize('initial', [None, 1.0])
def test_loop_lowering_rejects_non_numeric_carried_state_instead_of_guessing(initial):
    class TimeLatch(Loop):
        inputs = {}
        outputs = {"duty": DUTY_RATIO}
        carried = ("seen_at",)

        def __init__(self):
            self.startup = None
            self._state_blocks = {}
            self.seen_at = initial

        def sample(self, t, inputs):
            if t == 3.0:
                self.seen_at = None
            return {"duty": 0.0}

    node = TimeLatch()
    generator = _control_generator(node)
    with pytest.raises(TypeError, match='unsupported equation value None'):
        generator._loop_members()
        generator._loop_lines('node', node)


@pytest.mark.parametrize('initial, increment, kind', [
    (-2, 1, 'real'), (1.0 + 2j, 1j, 'complex'),
    (np.complex64(1.0 + 2j), 1j, 'complex'), (False, None, 'bool'),
])
def test_carried_values_use_the_same_scalar_types_as_equations(initial, increment, kind):
    class Accumulator(Loop):
        inputs = {}
        outputs = {'duty': DUTY_RATIO}
        carried = ('value',)

        def __init__(self):
            self.startup = None
            self._state_blocks = {}
            self.value = initial

        def sample(self, t, inputs):
            self.value = not self.value if increment is None else self.value + increment
            return {'duty': 0.0}

    node = Accumulator()
    generator = _control_generator(node)
    members = generator._loop_members()
    values, carried, _, _ = generator._trace_loop('node', node)
    assert carried == ('value',)
    assert values[-1].kind == kind
    assert members[0].startswith({'real': 'double', 'complex': 'Complex', 'bool': 'bool'}[kind])
    assert not any('std::size_t' in line for line in members)


def test_native_adapter_checks_follow_python_method_lookup():
    class Implementation:
        def advance(self):
            pass

        def __call__(self):
            self.advance()

        def describe(self):
            return "Implementation"

    class Inherited(Implementation):
        def describe(self):
            return "Display text does not affect execution"

    obj = Inherited()
    assert _uses_implementation(obj, Implementation)
    # Python does not use instance attributes for obj(), but does for obj.advance().
    obj.__call__ = lambda: None
    assert _uses_implementation(obj, Implementation)
    obj.advance = lambda: None
    assert not _uses_implementation(obj, Implementation)

    class Nested(Implementation):
        def __call__(self):
            def run():
                self.advance()
            run()

    obj = Nested()
    assert _uses_implementation(obj, Nested)
    obj.advance = lambda: None
    assert not _uses_implementation(obj, Nested)
    obj = Inherited()
    obj.optional_hook = lambda: None
    assert not _uses_implementation(obj, Implementation, 'optional_hook')


@pytest.mark.parametrize('continuous', [False, True])
@pytest.mark.parametrize('initial', [(1.0, 2.0), (1.0 + 3j, 2.0 + 4j)])
def test_loop_reset_updates_states_and_carried_values_simultaneously(continuous, initial, tmp_path):
    class Coupled(Loop):
        state_names = ('a', 'b')
        carried = ('armed',)

        def reset_integrator(self):
            self.a, self.b = self.b, self.a
            self.armed = False

    node = Coupled(SimpleNamespace(period=1.0),
                   SimpleNamespace(bridge=SimpleNamespace(model='averaging' if continuous
                                                         else 'pwm_averaging')), None)
    node.a, node.b, node.armed = *initial, True
    generator = _control_generator(node)
    generator.continuous = continuous
    generator.ctrl.state = SimpleNamespace()
    complex_state = isinstance(initial[0], complex)
    width = 2 if complex_state else 1
    generator.model.state_index = {
        (id(generator.ctrl.state), f'node.{name}'): (index * width, complex_state)
        for index, name in enumerate(('a', 'b'))
    }
    generator._state = lambda loop, local: generator.node(loop, local)
    members = generator._loop_members()
    reset = generator._reset_integrators()
    if continuous:
        assert not any('test_n_node_a =' in line for line in members)
        raw = [part for value in initial for part in ((value.real, value.imag)
                                                     if complex_state else (value,))]
        members += ['double y[] = {' + ', '.join(map(str, raw)) + '};']
        readings = [generator.model.state_value((id(generator.ctrl.state), f'node.{name}'))
                    for name in ('a', 'b')]
    else:
        readings = [generator.node('node', name) for name in ('a', 'b')]
    code = ('#include <complex>\n#include <iostream>\nusing Complex = std::complex<double>;\n'
            'int main() {\n' + '\n'.join(members + reset)
            + '\nstd::cout' + ''.join(f" << std::real({name}) << ' ' << std::imag({name}) << ' '"
                                     for name in readings)
            + f" << {generator.node('node', 'armed')};\n}}\n")
    compiler = _compiler()
    if compiler is None:
        pytest.skip('no C++17 compiler installed')
    source, executable = tmp_path / 'reset.cpp', tmp_path / 'reset'
    source.write_text(code, encoding='utf-8')
    subprocess.run([compiler, '-std=c++17', '-O2', str(source), '-o', str(executable)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    a_real, a_imag, b_real, b_imag, armed = map(float, result.stdout.split())
    node.reset_integrator()
    assert complex(a_real, a_imag) == node.a
    assert complex(b_real, b_imag) == node.b
    assert bool(armed) == node.armed is False


def test_loop_lowering_copies_each_registered_dynamic_block_once(monkeypatch):
    node = Loop(SimpleNamespace(period=1.0),
                SimpleNamespace(bridge=SimpleNamespace(model="pwm_averaging")), None)
    node.pi = node.state_block("integral", PI(1.0, 2.0, 1.0))
    generator = _control_generator(node)
    calls = []
    original = generator._copy_block

    def counted(block):
        calls.append(block)
        return original(block)

    generator._state = lambda *args: "state_value"
    monkeypatch.setattr(generator, "_copy_block", counted)
    cloned = generator._clone_loop("node", node)
    assert calls == [node.pi]
    assert cloned._state_blocks["integral"] is cloned.pi
