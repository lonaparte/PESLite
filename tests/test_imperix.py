"""Hardware boundary tests; host SDK doubles never count as target/hardware acceptance."""

import copy
import json
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import peslite
from peslite.assembly import imperix
from peslite.assembly.imperix_build import build_project
from peslite.control.blocks import phases
from peslite.control.controller import Measurement, VSCController
from peslite.solver.simulation import convert_main, main
from conftest import EXAMPLES
from test_cpp_export import _compiler, _run_generated


def _simulation(name="buck"):
    p = peslite.load(EXAMPLES / f"{name.removesuffix('-slow-terminal')}-example.pes")
    values = {"simulation.solver.linearisations": 0, "simulation.initial.states": {}}
    from peslite.assembly.params import _unit_items
    for key, _ in _unit_items(p.units):
        values.update({f"units.{key}.bridge.model": "pwm_averaging",
                       f"units.{key}.meas": {}, f"units.{key}.protection": {}})
        if name.endswith("-slow-terminal"):
            values[f"units.{key}.ctrl.loops.CurrentLoop.period"] = 2.0 * p.unit(key).ctrl.period
    return peslite.Simulation(p.replace(**values))


def _mapping(simulation):
    adc_index = pwm_index = 0
    bindings = {}
    for key, unit in simulation.units.items():
        adc = {}
        for channel, read in unit.ports.readers.items():
            width = 3 if isinstance(read(), complex) else 1
            items = list(range(adc_index, adc_index + width))
            adc[channel] = items if width == 3 else items[0]
            adc_index += width
        width = len(unit.bridge.topology.duty_names)
        bindings[key] = dict(adc=adc, pwm=list(range(pwm_index, pwm_index + width)),
                             dead_time=1e-7, startup_ramp=0.0,
                             hardware_overcurrent_configured=True)
        pwm_index += width
    return {"units": bindings}


def _export(sim, directory, config=None, **kwargs):
    return sim.export("imperix", directory, target_config=config or _mapping(sim), build=False, **kwargs)


def _compile(source, binary, include=None):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("a C++17 host compiler is required")
    args = [compiler, "-O2", "-std=c++17", "-fno-exceptions", str(source), "-o", str(binary)]
    if include:
        args += ["-I", str(include)]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[:6000]


@pytest.mark.parametrize("name", ["buck", "boost", "buck-boost", "gfl", "gfl-slow-terminal", "gfm-psc", "gfm-vsg",
                                  "gfm-dvoc", "gfm-matching", "wpt-ss", "wpt-lcc"])
@pytest.mark.parametrize("variables", [(), ("all",)])
def test_generated_controller_matches_python_loops_states_and_duties(name, variables, tmp_path):
    sim = _simulation(name)
    _export(sim, tmp_path, variables=variables)
    expected, lines = [], ['#include "peslite_control.hpp"', '#include <iostream>', '#include <iomanip>',
                           'int main() { std::cout << std::setprecision(17);']
    for key, unit in sim.units.items():
        tag = imperix._identifier(key)
        lines += [f"peslite_generated::Controller_{tag} c_{tag};", f"c_{tag}.command(true);"]
        unit.ctrl.command(True, 0.0)
        for index in range(24):
            t = index * unit.cfg.ctrl.period
            if isinstance(unit.ctrl, VSCController):
                rot = np.exp(1j * unit.cfg.base.w0 * t)
                values = dict(u_g=unit.cfg.base.v_phase_peak * rot,
                              i_c=0.05 * unit.cfg.base.i_phase_peak * rot,
                              u_dc=unit.cfg.dc_base.v)
                meas = Measurement(t, **values, i_abc=np.array(phases(values["i_c"])))
            else:
                values = dict(current=1.0 + index * 0.001, voltage=unit.cfg.dclink.vdc_ref * 0.8,
                              vin=unit.cfg.dclink.vdc_ref)
                if "power" in unit.ports.readers:
                    values["power"] = 150.0
                meas = SimpleNamespace(t=t, **values)
            output = unit.ctrl(t, meas)
            duties = unit.bridge.topology.command_duties(output)
            expected.append(tuple(duties))
            lines.append("{")
            lines.append(f"peslite_generated::Controller_{tag}::Measurements m;")
            for channel, value in values.items():
                expression = (f"peslite_generated::Complex{{{float(value.real)!r}, {float(value.imag)!r}}}"
                              if isinstance(value, complex) else repr(float(value)))
                lines.append(f"m.{channel} = {expression};")
            lines += [f"if (!c_{tag}.sample({t!r}, m)) return 2;",
                      f"for (double value : c_{tag}.duty) std::cout << value << ' ';",
                      "std::cout << '\\n';", "}"]
        # Compare loop-owned state too, including filters, PI integration and held outputs.
        for loop, node in unit.ctrl.graph.nodes.items():
            for local, value in node.get_state().items():
                member = f"c_{tag}.u_{tag}_n_" + imperix._identifier(f"{loop}_{local}")
                lines.append(f"std::cout << std::real({member}) << ' ' << std::imag({member}) << '\\n';")
                expected.append((float(np.real(value)), float(np.imag(value))))
    lines.append("}")
    source = tmp_path / "check.cpp"
    source.write_text("\n".join(lines), encoding="utf-8")
    binary = tmp_path / "check"
    _compile(source, binary)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    actual = [tuple(map(float, line.split())) for line in result.stdout.splitlines()]
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=5e-6, atol=3e-6)
    header = (tmp_path / "peslite_control.hpp").read_text()
    assert not any(text in header for text in ("std::vector", "std::string ", "new ", "rhs("))


def _sdk_double(directory):
    directory.mkdir()
    for sub in ("Core", "Driver"):
        (directory / sub).mkdir()
    for header in ("Core/core.h", "Core/interrupts.h", "Driver/peripherals.h"):
        (directory / header).write_text("#pragma once\n", encoding="utf-8")
    declarations = """
#pragma once
#include <cmath>
#define NUMBER_OF_DEVICES(n) constexpr int device_count = n
enum tUserSafe { SAFE, UNSAFE }; enum tErrorSource { TEST_FAULT };
enum tClock { CLOCK_0, CLOCK_1, CLOCK_2, CLOCK_3 };
enum tPwmCarrier { TRIANGLE }; enum tPwmRate { SINGLE_RATE, DOUBLE_RATE };
enum tPwmOutMode { COMPLEMENTARY, PWMH_ACTIVE };
inline double input[4][32]{}, sensitivity[4][32]{}, offset[4][32]{}, duty[4][32]{};
inline unsigned adc_reads[4][32]{}, history_reads[4][32]{}, history_depth[4][32]{};
inline double input_history[4][32][64]{};
inline unsigned cpu_postscaler = 0;
inline tClock pwm_clock[4][32]{};
inline bool enabled[4][32]{};
inline double frequency[4]{}, carrier_phase[4][32]{}, sampling_phase = -1;
inline void Clock_SetFrequency(tClock c, float value) { frequency[c] = value; }
inline void ConfigureMainInterrupt(tUserSafe (*)(), tClock, float phase, unsigned postscaler = 0) { sampling_phase = phase; cpu_postscaler = postscaler; }
inline void Adc_ConfigureSensor(unsigned c, float gain, float shift, unsigned d = 0) { sensitivity[d][c] = gain; offset[d][c] = shift; }
inline float Adc_GetValue(unsigned c, unsigned d = 0) { ++adc_reads[d][c]; return input[d][c] / sensitivity[d][c] + offset[d][c]; }
inline void Adc_EnableSynchronousAveraging(unsigned, unsigned = 0) {}
inline void Adc_ConfigureHistory(unsigned c, unsigned depth, unsigned d = 0) { history_depth[d][c] = depth; }
inline float Adc_GetHistory(unsigned c, unsigned n, unsigned d = 0) { ++history_reads[d][c]; return input_history[d][c][n] / sensitivity[d][c] + offset[d][c]; }
inline void CbPwm_ConfigureClock(tPwmOutput c, tClock clock, unsigned d = 0) { pwm_clock[d][c] = clock; }
inline void CbPwm_ConfigureCarrier(tPwmOutput, tPwmCarrier, unsigned = 0) {}
inline void CbPwm_ConfigureUpdateRate(tPwmOutput, tPwmRate, unsigned = 0) {}
inline void CbPwm_ConfigureOutputMode(tPwmOutput, tPwmOutMode, unsigned = 0) {}
inline void CbPwm_ConfigureDeadTime(tPwmOutput, float, unsigned = 0) {}
inline void CbPwm_SetPhase(tPwmOutput c, float value, unsigned d = 0) { carrier_phase[d][c] = value; }
inline void CbPwm_SetDutyCycle(tPwmOutput c, float value, unsigned d = 0) { duty[d][c] = value; }
inline void CbPwm_Activate(tPwmOutput c, unsigned d = 0) { enabled[d][c] = true; }
inline void CbPwm_Deactivate(tPwmOutput c, unsigned d = 0) { enabled[d][c] = false; }
"""
    enum = "enum tPwmOutput {" + ",".join(f"PWM_CHANNEL_{i}={i}" for i in range(32)) + "};\n"
    declarations = declarations.replace("inline double input", enum + "inline double input", 1)
    (directory / "extern_user.h").write_text(declarations, encoding="utf-8")


@pytest.mark.parametrize("name", ["buck", "boost", "buck-boost", "gfl", "wpt-ss", "wpt-lcc"])
@pytest.mark.parametrize("variables", [(), ("all",)])
def test_default_probes_publish_calibrated_interface_samples_without_extra_adc_reads(name, variables, tmp_path):
    sim = _simulation(name)
    config = _mapping(sim)
    expected = {}
    inputs = {}
    for unit_name, binding in config["units"].items():
        for key, specification in binding["adc"].items():
            channels = specification if isinstance(specification, list) else [specification]
            calibrated = []
            suffixes = ("_a", "_b", "_c") if len(channels) == 3 else ("",)
            for index, (channel, suffix) in enumerate(zip(channels, suffixes)):
                device = (len(unit_name) + index) % 2
                gain, offset, sign = 0.5 * (index + 1), 0.25 * index, -1 if index % 2 else 1
                calibrated.append(dict(device=device, channel=channel, sensitivity=gain, offset=offset, sign=sign))
                raw = 16.0 + device * 64 + channel
                inputs[device, channel] = raw
                path = f"{unit_name}.{key}{suffix}"
                expected[path] = (device, channel, gain, offset, sign)
            binding["adc"][key] = calibrated if isinstance(specification, list) else calibrated[0]
    out = tmp_path / "out"
    _export(sim, out, config, variables=variables)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["probes"] == {path: imperix._identifier(path) for path in expected}
    source = (out / "user.cpp").read_text()
    assert source.count("Adc_GetValue(") == len(inputs)
    assert all(source.count(f"volatile float {symbol} = 0.0f;") == 1
               for symbol in manifest["probes"].values())
    sdk = tmp_path / "sdk"
    _sdk_double(sdk)
    lines = ['#include "user.cpp"', '#include <cassert>', 'int main() {',
             'assert(UserInit() == SAFE); assert(peslite_run == 0);']
    for symbol in manifest["probes"].values():
        lines.append(f"assert({symbol} == 0.0f);")
    for sample in range(2):
        lines += [f"input[{device}][{channel}] = {raw + sample * 2.0!r};"
                  for (device, channel), raw in inputs.items()]
        lines.append('assert(UserInterrupt() == SAFE);')
        for path, (device, channel, gain, offset, sign) in expected.items():
            value = float(np.float32((inputs[device, channel] + sample * 2.0) / gain + offset)) * sign
            lines.append(f"assert({manifest['probes'][path]} == static_cast<float>({value!r}));")
        lines += [f"assert(adc_reads[{device}][{channel}] == {sample + 1});" for device, channel in inputs]
    lines.append('assert(UserInit() == SAFE);')
    lines += [f"assert({symbol} == 0.0f);" for symbol in manifest["probes"].values()]
    lines.append('}')
    check = out / "check.cpp"
    check.write_text("\n".join(lines), encoding="utf-8")
    _compile(check, tmp_path / "check", sdk)
    subprocess.run([str(tmp_path / "check")], check=True)


@pytest.mark.parametrize("name", ["gfl", "two-converters"])
@pytest.mark.parametrize("sign", [1, -1])
def test_shared_adc_channels_keep_distinct_logical_probes_without_extra_reads(name, sign, tmp_path):
    sim = _simulation(name)
    config = _mapping(sim)
    first, last = tuple(sim.units)[0], tuple(sim.units)[-1]
    voltage = config["units"][first]["adc"]["u_g"]
    current = config["units"][last]["adc"]["i_c"]
    unused = current[0]
    shared = dict(channel=voltage[0], sensitivity=0.5, offset=-1.0)
    voltage[0] = shared
    current[0] = {**shared, "sign": sign}
    out = tmp_path / "out"
    _export(sim, out, config)
    probes = json.loads((out / "manifest.json").read_text())["probes"]
    sdk = tmp_path / "sdk"
    _sdk_double(sdk)
    check = out / "check.cpp"
    check.write_text(f'''#include "user.cpp"
#include <cassert>
int main() {{
    assert(UserInit() == SAFE);
    input[0][{shared['channel']}] = 32.0;
    assert(UserInterrupt() == SAFE);
    assert({probes[first + '.u_g_a']} == 63.0f);
    assert({probes[last + '.i_c_a']} == {sign} * 63.0f);
    assert(sensitivity[0][{shared['channel']}] == 0.5 && offset[0][{shared['channel']}] == -1.0);
    assert(adc_reads[0][{shared['channel']}] == 1 && adc_reads[0][{unused}] == 0);
}}
''', encoding="utf-8")
    _compile(check, tmp_path / "check", sdk)
    subprocess.run([str(tmp_path / "check")], check=True)


@pytest.mark.parametrize("controls,samples,windows,expected", [
    ((50e-6,), (50e-6,), (None,), (50e-6, 50e-6, 1)),
    ((50e-6,), (12.5e-6,), (None,), (50e-6, 12.5e-6, 4)),
    ((50e-6, 75e-6), (50e-6, 75e-6), (None, None), (25e-6, 25e-6, 1)),
    ((100e-6, 150e-6), (25e-6, 50e-6), (None, None), (50e-6, 25e-6, 2)),
    ((50e-6,), (50e-6,), (25e-6,), (50e-6, 25e-6, 2)),
])
def test_timing_mapping_uses_existing_periods_without_a_tick_search(controls, samples, windows, expected):
    units = {str(index): SimpleNamespace(cfg=SimpleNamespace(
        ctrl=SimpleNamespace(period=control),
        meas=SimpleNamespace(period=sample, window=window, average=["voltage"] if window else [])))
        for index, (control, sample, window) in enumerate(zip(controls, samples, windows))}
    actual = imperix._timing(units, {})
    assert actual["interrupt_period"] == pytest.approx(expected[0])
    assert actual["sampling_period"] == pytest.approx(expected[1])
    assert actual["postscaler"] == expected[2]


@pytest.mark.parametrize("variables", [(), ("all",), ("common",)])
def test_explicit_auto_mode_keeps_the_existing_generated_code(variables, tmp_path):
    sim = _simulation()
    config = _mapping(sim)
    _export(sim, tmp_path / "default", config, variables=variables)
    config["clock_mode"] = "auto"
    _export(sim, tmp_path / "auto", config, variables=variables)
    for name in ("user.cpp", "peslite_control.hpp", "manifest.json"):
        assert (tmp_path / "default" / name).read_bytes() == (tmp_path / "auto" / name).read_bytes()


@pytest.mark.parametrize("variables", [(), ("all",), ("common",)])
@pytest.mark.parametrize("clock", ["CLK0", "CLK3"])
def test_manual_pwm_clock_keeps_default_adc_and_control_on_clk0(variables, clock, tmp_path):
    sim = _simulation()
    config = {**_mapping(sim), "clock_mode": "manual", "clocks": {"units.converter.pwm.f_sw": clock}}
    out, sdk = tmp_path / "out", tmp_path / "sdk"
    _export(sim, out, config, variables=variables)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["clock_mode"] == "manual"
    assert manifest["bindings"]["converter"]["clock"] == int(clock[-1])
    assert manifest["clocks"][clock]["frequency"] == 20000.0
    assert "units.converter.pwm.f_sw" in manifest["clocks"][clock]["bindings"]
    for path in ("interrupt_period", "units.converter.meas.period", "units.converter.ctrl.period",
                 "units.converter.ctrl.loops.DCVoltageControl.period"):
        assert path in manifest["clocks"]["CLK0"]["bindings"]
    assert set(manifest["clocks"]) == {"CLK0", clock}
    _sdk_double(sdk)
    check = out / "check.cpp"
    check.write_text(f'''#include "user.cpp"
#include <cassert>
int main() {{
    assert(UserInit() == SAFE);
    assert(pwm_clock[0][0] == CLOCK_{clock[-1]});
    assert(frequency[CLOCK_0] == 20000.0 && frequency[CLOCK_{clock[-1]}] == 20000.0);
    assert(cpu_postscaler == 1);
    input[0][0] = 1.0; input[0][1] = 24.0; input[0][2] = 48.0;
    for (unsigned k = 0; k < 8; ++k) {{
        assert(UserInterrupt() == SAFE);
        assert(control_converter.u_converter_pwm_k == k + 1);
    }}
}}
''', encoding="utf-8")
    _compile(check, tmp_path / "check", sdk)
    subprocess.run([str(tmp_path / "check")], check=True)


def test_manual_adc_cpu_and_slow_loops_use_the_existing_dividers(tmp_path):
    sim = peslite.Simulation(_simulation().p.replace(**{
        "units.converter.ctrl.period": 50e-6,
        "units.converter.meas.period": 12.5e-6,
        "units.converter.ctrl.loops.DCVoltageControl.period": 200e-6,
    }))
    config = {**_mapping(sim), "interrupt_period": 25e-6, "clock_mode": "manual", "clocks": {
        "interrupt_period": "CLK0", "units.converter.pwm.f_sw": "CLK3",
        "units.converter.ctrl.period": "CLK0", "units.converter.meas.period": "CLK0",
        "units.converter.ctrl.loops.DCVoltageControl.period": "CLK0",
    }}
    out, sdk = tmp_path / "out", tmp_path / "sdk"
    _export(sim, out, config)
    auto = {key: value for key, value in config.items() if key not in ("clock_mode", "clocks")}
    _export(sim, tmp_path / "auto", auto)
    assert (out / "peslite_control.hpp").read_bytes() == (tmp_path / "auto" / "peslite_control.hpp").read_bytes()
    manual_source = (out / "user.cpp").read_text().replace("CLOCK_3", "CLOCK_1")
    assert manual_source == (tmp_path / "auto" / "user.cpp").read_text()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["timing"] == {"sampling_period": 12.5e-6, "interrupt_period": 25e-6, "postscaler": 2}
    assert set(manifest["clocks"]) == {"CLK0", "CLK3"}
    _sdk_double(sdk)
    check = out / "check.cpp"
    check.write_text('''#include "user.cpp"
#include <cassert>
int main() {
    assert(UserInit() == SAFE);
    assert(frequency[CLOCK_0] == 80000.0 && frequency[CLOCK_3] == 20000.0);
    assert(cpu_postscaler == 2 && pwm_clock[0][0] == CLOCK_3);
    input[0][0] = 1.0; input[0][1] = 24.0; input[0][2] = 48.0;
    for (unsigned k = 0; k < 16; ++k) {
        assert(UserInterrupt() == SAFE);
        assert(control_converter.u_converter_pwm_k == k / 2 + 1);
    }
}
''', encoding="utf-8")
    _compile(check, tmp_path / "check", sdk)
    subprocess.run([str(tmp_path / "check")], check=True)


@pytest.mark.parametrize("path", ["interrupt_period", "units.converter.pwm.f_sw",
                                  "units.converter.meas.period", "units.converter.ctrl.period",
                                  "units.converter.ctrl.loops.DCVoltageControl.period"])
def test_manual_mode_requires_each_explicit_period_even_when_values_match(path, tmp_path):
    sim = peslite.Simulation(_simulation().p.replace(**{
        "units.converter.ctrl.period": 50e-6, "units.converter.meas.period": 50e-6,
        "units.converter.ctrl.loops.DCVoltageControl.period": 50e-6,
    }))
    clocks = {"interrupt_period": "CLK0", "units.converter.pwm.f_sw": "CLK0",
              "units.converter.meas.period": "CLK0", "units.converter.ctrl.period": "CLK0",
              "units.converter.ctrl.loops.DCVoltageControl.period": "CLK0"}
    clocks.pop(path)
    config = {**_mapping(sim), "clock_mode": "manual", "interrupt_period": 50e-6, "clocks": clocks}
    with pytest.raises(ValueError, match="missing explicit period bindings") as error:
        _export(sim, tmp_path / "out", config)
    assert path in str(error.value)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field", ["meas.period", "ctrl.period", "ctrl.loops.DCVoltageControl.period"])
@pytest.mark.parametrize("clock", ["CLK1", "CLK2", "CLK3"])
def test_adc_and_cpu_clock_bindings_cannot_use_pwm_only_clocks(field, clock, tmp_path):
    sim = _simulation()
    path = "units.converter." + field
    config = {**_mapping(sim), "clock_mode": "manual", "clocks": {
        "units.converter.pwm.f_sw": "CLK0", path: clock}}
    with pytest.raises(ValueError, match="ADC and CPU schedules must use CLK0") as error:
        _export(sim, tmp_path / "out", config)
    assert path in str(error.value)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mode,clocks,message", [
    ("automatic", {}, "clock_mode"), (None, {}, "clock_mode"),
    ("auto", {"units.converter.pwm.f_sw": "CLK0"}, "require clock_mode='manual'"),
    ("manual", {}, "missing explicit period bindings"),
    ("manual", None, "must be a mapping"),
    ("manual", {"units.converter.pwm.f_sw": "CLK4"}, "expected CLK0"),
    ("manual", {"units.converter.pwm.f_sw": 0}, "expected CLK0"),
    ("manual", {"units.missing.pwm.f_sw": "CLK1"}, "unknown fields"),
    ("manual", {"units.converter.meas.window": "CLK0"}, "unknown fields"),
    ("manual", {"units.converter.ctrl.computation": "CLK0"}, "unknown fields"),
    ("manual", {"units.converter.pwm.carrier_phase": "CLK0"}, "unknown fields"),
    ("manual", {"interrupt_period": "CLK1"}, "must use CLK0"),
])
def test_invalid_clock_configuration_is_rejected_before_writing_files(mode, clocks, message, tmp_path):
    sim = _simulation()
    config = {**_mapping(sim), "clock_mode": mode, "clocks": clocks}
    with pytest.raises(ValueError, match=message):
        _export(sim, tmp_path / "out", config)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("same_frequency,second_clock", [(True, "CLK1"), (True, "CLK3"), (False, "CLK3")])
def test_manual_pwm_clocks_can_share_or_separate_equal_frequencies(same_frequency, second_clock, tmp_path):
    sim = _simulation("two-converters")
    if same_frequency:
        sim = peslite.Simulation(sim.p.replace(**{"units.vsc_2.pwm.f_sw": 20000.0}))
    config = {**_mapping(sim), "clock_mode": "manual", "clocks": {
        "units.vsc_1.pwm.f_sw": "CLK1", "units.vsc_2.pwm.f_sw": second_clock,
    }}
    _export(sim, tmp_path, config)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["bindings"]["vsc_1"]["clock"] == 1
    assert manifest["bindings"]["vsc_2"]["clock"] == int(second_clock[-1])
    assert set(manifest["clocks"]) == {"CLK0", "CLK1", second_clock}
    assert manifest["clocks"][second_clock]["frequency"] == (20000.0 if same_frequency else 10000.0)


@pytest.mark.parametrize("name,settings,clocks", [
    ("buck", {"units.converter.meas.period": 12.5e-6},
     {"units.converter.pwm.f_sw": "CLK0", "units.converter.meas.period": "CLK0"}),
    ("two-converters", {}, {"units.vsc_1.pwm.f_sw": "CLK1", "units.vsc_2.pwm.f_sw": "CLK1"}),
])
def test_one_hardware_clock_cannot_have_conflicting_frequencies(name, settings, clocks, tmp_path):
    sim = peslite.Simulation(_simulation(name).p.replace(**settings))
    config = {**_mapping(sim), "clock_mode": "manual", "clocks": clocks}
    with pytest.raises(ValueError, match="already uses .* Hz; cannot also use .* Hz"):
        _export(sim, tmp_path / "out", config)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["wpt-ss", "wpt-lcc"])
def test_manual_nested_unit_clocks_use_the_existing_parameter_paths(name, tmp_path):
    sim = _simulation(name)
    clocks = {f"units.{key}.pwm.f_sw": f"CLK{index + 1}" for index, key in enumerate(sim.units)}
    # Explicit loop periods in these examples need bindings; inherited ones do not.
    for key, unit in sim.units.items():
        for loop, params in unit.cfg.ctrl.loops.items():
            field = f"ctrl.loops.{loop}.period"
            if field not in unit.cfg.derived_defaults:
                clocks[f"units.{key}.{field}"] = "CLK0"
    config = {**_mapping(sim), "clock_mode": "manual", "clocks": clocks}
    _export(sim, tmp_path, config)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "units.charger.primary.pwm.f_sw" in manifest["clocks"]["CLK1"]["bindings"]
    assert "units.charger.secondary.pwm.f_sw" in manifest["clocks"]["CLK2"]["bindings"]
    assert set(manifest["clocks"]) == {"CLK0", "CLK1", "CLK2"}


def test_automatic_clock_capacity_counts_frequencies_not_units():
    def unit(frequency):
        return SimpleNamespace(cfg=SimpleNamespace(
            pwm=SimpleNamespace(f_sw=frequency), meas=SimpleNamespace(period=None),
            ctrl=SimpleNamespace(period=1.0 / frequency, loops={}), derived_defaults={"ctrl.period"}))

    units = {f"unit{index}": unit(frequency)
             for index, frequency in enumerate((10000.0, 20000.0, 30000.0, 40000.0, 20000.0))}
    clocks, assigned = imperix._clock_bindings(units, {}, {"sampling_period": 100e-6})
    assert clocks == {0: 10000.0, 1: 20000.0, 2: 30000.0, 3: 40000.0}
    assert assigned["units.unit1.pwm.f_sw"] == assigned["units.unit4.pwm.f_sw"] == 1
    units["unit5"] = unit(50000.0)
    with pytest.raises(ValueError, match="exceed four hardware clocks"):
        imperix._clock_bindings(units, {}, {"sampling_period": 100e-6})


@pytest.mark.parametrize("variables", [(), ("all",)])
def test_fast_adc_uses_native_postscaler_without_extra_cpu_interrupts(variables, tmp_path):
    sim = peslite.Simulation(_simulation().p.replace(**{"units.converter.meas.period": 12.5e-6}))
    out, sdk = tmp_path / "out", tmp_path / "sdk"
    _export(sim, out, variables=variables)
    timing = json.loads((out / "manifest.json").read_text())["timing"]
    assert timing == {"interrupt_period": 50e-6, "sampling_period": 12.5e-6, "postscaler": 4}
    _sdk_double(sdk)
    check = out / "check.cpp"
    check.write_text('''#include "user.cpp"
#include <cassert>
int main() {
    assert(UserInit() == SAFE);
    assert(frequency[CLOCK_0] == 80000.0 && cpu_postscaler == 4);
    assert(frequency[pwm_clock[0][0]] == 20000.0);
    input[0][0] = 1.0; input[0][1] = 24.0; input[0][2] = 48.0;
    for (unsigned k = 0; k < 8; ++k) {
        assert(UserInterrupt() == SAFE);
        assert(control_converter.u_converter_pwm_k == k + 1);
    }
    assert(adc_reads[0][0] == 8 && history_reads[0][0] == 0);
    assert(history_depth[0][0] == 0);
}
''', encoding="utf-8")
    _compile(check, tmp_path / "check", sdk)
    subprocess.run([str(tmp_path / "check")], check=True)


def test_nonharmonic_units_and_slow_loops_run_on_the_automatic_common_interrupt(tmp_path):
    sim = peslite.Simulation(_simulation("two-converters").p.replace(**{
        "units.vsc_2.pwm.f_sw": 20000.0,
        "units.vsc_2.ctrl.period": 75e-6,
        "units.vsc_2.ctrl.loops.DCVoltageControl.period": 300e-6,
    }))
    out, sdk = tmp_path / "out", tmp_path / "sdk"
    _export(sim, out)
    timing = json.loads((out / "manifest.json").read_text())["timing"]
    assert timing == {"interrupt_period": 25e-6, "sampling_period": 25e-6, "postscaler": 1}
    _sdk_double(sdk)
    check = out / "check.cpp"
    check.write_text('''#include "user.cpp"
#include <cassert>
int main() {
    assert(UserInit() == SAFE);
    input[0][6] = 1500.0; input[0][13] = 1500.0;
    for (unsigned k = 0; k < 12; ++k) {
        assert(UserInterrupt() == SAFE);
        assert(control_vsc_1.u_vsc_1_pwm_k == k / 2 + 1);
        assert(control_vsc_2.u_vsc_2_pwm_k == k / 3 + 1);
    }
    assert(cpu_postscaler == 1 && frequency[CLOCK_0] == 40000.0);
}
''', encoding="utf-8")
    _compile(check, tmp_path / "check", sdk)
    subprocess.run([str(tmp_path / "check")], check=True)


@pytest.mark.parametrize("variables", [(), ("all",)])
def test_sdk_history_combines_native_window_means_and_waits_for_valid_history(variables, tmp_path):
    sim = peslite.Simulation(_simulation().p.replace(**{"units.converter.meas.average": ["voltage"]}))
    config = _mapping(sim)
    config["interrupt_period"] = 12.5e-6
    config["units"]["converter"]["adc"]["voltage"] = dict(channel=1, sensitivity=2.0, offset=1.0)
    out, sdk = tmp_path / "out", tmp_path / "sdk"
    _export(sim, out, config, variables=variables)
    _sdk_double(sdk)
    check = out / "check.cpp"
    check.write_text('''#include "user.cpp"
#include <cassert>
int main() {
    assert(UserInit() == SAFE);
    assert(cpu_postscaler == 1 && history_depth[0][1] == 4);
    input[0][0] = 1.0; input[0][1] = 8.0; input[0][2] = 48.0;
    input_history[0][1][1] = 6.0;
    input_history[0][1][2] = 4.0;
    input_history[0][1][3] = 2.0;
    for (unsigned k = 0; k < 3; ++k) {
        assert(UserInterrupt() == SAFE);
        assert(converter_voltage == 0.0f && history_reads[0][1] == 0);
        assert(control_converter.u_converter_pwm_k == 0 && !enabled[0][0]);
    }
    assert(UserInterrupt() == SAFE);
    assert(converter_voltage == 3.5f && history_reads[0][1] == 3);
    assert(UserInterrupt() == SAFE);
    assert(control_converter.u_converter_pwm_k == 1);
    assert(adc_reads[0][1] == 5 && history_reads[0][1] == 6);
}
''', encoding="utf-8")
    _compile(check, tmp_path / "check", sdk)
    subprocess.run([str(tmp_path / "check")], check=True)


def test_partial_average_window_is_mapped_to_native_adc_clock(tmp_path):
    sim = peslite.Simulation(_simulation().p.replace(**{
        "units.converter.meas.average": ["current"], "units.converter.meas.window": 25e-6}))
    _export(sim, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["timing"] == {
        "interrupt_period": 50e-6, "sampling_period": 25e-6, "postscaler": 2}
    assert manifest["bindings"]["converter"]["adc"]["current"][0]["history"] == 1


@pytest.mark.parametrize("sign", [1, -1])
def test_shared_adc_windows_are_calculated_only_once_per_interrupt(sign, tmp_path):
    sim = peslite.Simulation(_simulation("gfl").p.replace(**{
        "units.vsc.meas.average": ["u_g", "i_c"]}))
    config = _mapping(sim)
    config["interrupt_period"] = 25e-6
    config["units"]["vsc"]["adc"]["i_c"][0] = {
        "channel": config["units"]["vsc"]["adc"]["u_g"][0], "sign": sign}
    out, sdk = tmp_path / "out", tmp_path / "sdk"
    _export(sim, out, config)
    _sdk_double(sdk)
    check = out / "check.cpp"
    check.write_text(f'''#include "user.cpp"
#include <cassert>
int main() {{
    assert(UserInit() == SAFE && history_depth[0][0] == 2);
    input[0][0] = 32.0; input[0][6] = 1500.0;
    input_history[0][0][1] = 16.0;
    assert(UserInterrupt() == SAFE);
    assert(history_reads[0][0] == 0);
    assert(UserInterrupt() == SAFE);
    assert(vsc_u_g_a == 24.0f && vsc_i_c_a == {sign} * 24.0f);
    assert(adc_reads[0][0] == 2 && adc_reads[0][3] == 0);
    assert(history_reads[0][0] == 1);
}}
''', encoding="utf-8")
    _compile(check, tmp_path / "check", sdk)
    subprocess.run([str(tmp_path / "check")], check=True)


@pytest.mark.parametrize("selector", ["all", "common"])
def test_adc_window_is_fixed_before_the_history_layout_is_generated(selector, tmp_path):
    sim = peslite.Simulation(_simulation().p.replace(**{
        "units.converter.meas.average": ["voltage"], "units.converter.meas.window": 50e-6}))
    config = _mapping(sim)
    config["interrupt_period"] = 12.5e-6
    out, sdk = tmp_path / "out", tmp_path / "sdk"
    result = _export(sim, out, config, variables=[selector])
    fixed = next(item["parameters"] for item in result.diagnostics
                 if item["code"] == "hardware-timing-constants")
    assert "units.converter.meas.window" in fixed
    assert "param_units_converter_meas_window" not in (out / "user.cpp").read_text()
    _sdk_double(sdk)
    check = out / "check.cpp"
    check.write_text('''#include "user.cpp"
#include <cassert>
int main() {
    assert(UserInit() == SAFE && history_depth[0][1] == 4);
    assert(UserInterrupt() == SAFE && !peslite_run && !enabled[0][0]);
}
''', encoding="utf-8")
    _compile(check, tmp_path / "check", sdk)
    subprocess.run([str(tmp_path / "check")], check=True)


def test_incommensurate_periods_do_not_expand_into_a_per_tick_schedule():
    units = {str(index): SimpleNamespace(cfg=SimpleNamespace(
        ctrl=SimpleNamespace(period=period),
        meas=SimpleNamespace(period=None, window=None, average=[])))
        for index, period in enumerate((50e-6, 50e-6 + 1e-12))}
    with pytest.raises(ValueError, match="common timing period"):
        imperix._timing(units, {})


def test_incompatible_explicit_interrupt_is_rejected_before_writing_files(tmp_path):
    sim = _simulation()
    config = _mapping(sim)
    config["interrupt_period"] = 75e-6
    with pytest.raises(ValueError, match="whole multiples"):
        _export(sim, tmp_path / "out", config)
    assert not (tmp_path / "out").exists()


def test_native_history_limit_is_checked_before_writing_files(tmp_path):
    sim = peslite.Simulation(_simulation().p.replace(**{"units.converter.meas.average": ["voltage"]}))
    config = _mapping(sim)
    config["interrupt_period"] = 0.5e-6
    with pytest.raises(ValueError, match="at most 64"):
        _export(sim, tmp_path / "out", config)
    assert not (tmp_path / "out").exists()


def test_probe_names_reject_collisions_between_scalar_and_phase_channels():
    channels = tuple(dict(channel=index, device=0, sign=1) for index in range(3))
    unit = SimpleNamespace(name="custom", tag="custom", binding={
        "adc": {"voltage": channels, "voltage_a": channels[:1]},
    })
    with pytest.raises(ValueError, match="exported names collide"):
        imperix._source([unit], {}, {}, 50e-6, 0.5, SimpleNamespace(selected=()))


@pytest.mark.parametrize("unit_name, channel, variables", [
    ("peslite", "run", ()),
    ("param_base", "s_base", ("all",)),
    ("update", "parameters", ("all",)),
    ("stop", "all", ()),
])
def test_probe_names_cannot_shadow_other_exported_symbols(unit_name, channel, variables):
    from peslite.assembly.cpp import _RuntimeParameters

    runtime = _RuntimeParameters(_simulation().p, variables)
    unit = SimpleNamespace(name=unit_name, tag=unit_name, binding={
        "adc": {channel: (dict(channel=0, device=0, sign=1),)},
        "pwm": (), "dead_time": 0.0,
    }, cfg=SimpleNamespace(ctrl=SimpleNamespace(period=50e-6),
                           pwm=SimpleNamespace(f_sw=20000.0, carrier_phase=0.0)),
       graph=SimpleNamespace(period_stride=lambda _period, _interrupt: 1),
       topology=SimpleNamespace(unilateral=True), m=lambda field: "test_" + field)
    with pytest.raises(ValueError, match="exported names collide"):
        imperix._source([unit], {}, {}, 50e-6, 0.5, runtime)


def test_hardware_and_csv_use_the_same_canonical_phase_names(tmp_path):
    from peslite.solver.simulation import _plant_output_row

    sim = _simulation("gfl")
    _export(sim, tmp_path)
    probes = json.loads((tmp_path / "manifest.json").read_text())["probes"]
    row = _plant_output_row(sim.p, 0.0, sim.system.signals(0.0), {})
    names = {f"vsc.{channel}_{phase}" for channel in ("u_g", "i_c") for phase in "abc"}
    assert names <= probes.keys() and names <= row.keys()
    assert all(probes[path] == path.replace(".", "_") for path in names)


def test_sdk_boundary_calibration_default_stop_run_and_fail_safe(tmp_path):
    sim = _simulation()
    config = _mapping(sim)
    config["units"]["converter"]["adc"]["current"] = dict(channel=0, sensitivity=0.5, offset=-1.0, sign=-1)
    result = _export(sim, tmp_path / "export", config)
    sdk = tmp_path / "sdk"
    _sdk_double(sdk)
    source = tmp_path / "export" / "check.cpp"
    source.write_text('''#include "user.cpp"
#include <cassert>
#include <limits>
int main() {
    assert(UserInit() == SAFE); assert(peslite_run == 0); assert(!enabled[0][0]);
    input[0][0] = 0.5; input[0][1] = 24; input[0][2] = 48;
    assert(UserInterrupt() == SAFE); assert(!enabled[0][0]);
    peslite_run = 1; assert(UserInterrupt() == SAFE); assert(!enabled[0][0]);
    for (int i = 0; i < 4; ++i) assert(UserInterrupt() == SAFE);
    assert(enabled[0][0]);
    assert(duty[0][0] >= 0 && duty[0][0] <= 1);
    UserError(TEST_FAULT); assert(!enabled[0][0] && peslite_run == 0);
    peslite_run = 1; input[0][0] = std::numeric_limits<double>::quiet_NaN();
    assert(UserInterrupt() == UNSAFE); assert(!enabled[0][0] && peslite_run == 0);
}
''', encoding="utf-8")
    binary = tmp_path / "check"
    _compile(source, binary, sdk)
    subprocess.run([str(binary)], check=True)
    manifest = json.loads((result.out_dir / "manifest.json").read_text())
    assert manifest["status"] == "source_only" and manifest["hardware_tested"] is False
    assert not (result.out_dir / "run.elf").exists()


@pytest.mark.parametrize("name", ["gfl", "wpt-ss"])
def test_sdk_boundary_compiles_for_complex_and_nested_units(name, tmp_path):
    sim = _simulation(name)
    directory = tmp_path / "out"
    _export(sim, directory)
    sdk = tmp_path / "sdk"
    _sdk_double(sdk)
    source = directory / "check.cpp"
    source.write_text('#include "user.cpp"\nint main() { UserInit(); UserInterrupt(); UserError(TEST_FAULT); }\n',
                      encoding="utf-8")
    binary = tmp_path / "check"
    _compile(source, binary, sdk)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("settings, message", [
    ({"units.converter.bridge.model": "averaging"}, "sampled"),
    ({"units.converter.meas.rms": {"current": 200e-6}}, "ADC-history"),
    ({"simulation.initial.states": {"output.u_C": 24.0}}, "starts cold"),
])
def test_unsupported_hardware_semantics_are_not_silently_approximated(settings, message, tmp_path):
    sim = peslite.Simulation(_simulation().p.replace(**settings))
    with pytest.raises((TypeError, ValueError), match=message):
        _export(sim, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_fast_protection_requires_hardware_declaration(tmp_path):
    sim = peslite.Simulation(_simulation().p.replace(**{
        "units.converter.protection.overcurrent": {"enable": True, "limit_pu": 2.0}}))
    config = _mapping(sim)
    config["units"]["converter"].pop("hardware_overcurrent_configured")
    with pytest.raises(ValueError, match="hardware_overcurrent_configured"):
        _export(sim, tmp_path, config)


def test_replaced_controller_execution_is_not_silently_discarded(tmp_path):
    class Modified(VSCController):
        def __call__(self, t, measurement):
            result = super().__call__(t, measurement)
            result.d_abc *= 0.5
            return result

    p = _simulation("gfl").p
    sim = peslite.Simulation(p, parts={"vsc.ctrl": Modified})
    with pytest.raises(TypeError, match="controller.*no generator"):
        _export(sim, tmp_path)


def test_reporting_overrides_do_not_restrict_elf_export(tmp_path):
    class ReportingOnly(VSCController):
        def set_logging(self, enabled):
            super().set_logging(enabled)

        def summary(self):
            return {**super().summary(), "extra": 1}

    normal = _simulation("gfl")
    modified = peslite.Simulation(normal.p, parts={"vsc.ctrl": ReportingOnly})
    _export(normal, tmp_path / "normal")
    _export(modified, tmp_path / "modified")
    assert (tmp_path / "normal" / "peslite_control.hpp").read_bytes() == (
        tmp_path / "modified" / "peslite_control.hpp").read_bytes()
    # Simulator export still replaces logging and summary, so it must retain its checks.
    with pytest.raises(TypeError, match="controller.*no generator"):
        modified.export("cpp", tmp_path / "simulator")


def test_sampled_only_protection_does_not_read_fast_current(monkeypatch, tmp_path):
    sim = _simulation()

    def unused_fast_current(self):
        pytest.fail("sampled-only generation must not read the plant's fast current")

    monkeypatch.setattr(type(sim.units["converter"]), "_protection_currents", unused_fast_current)
    _export(sim, tmp_path)
    header = (tmp_path / "peslite_control.hpp").read_text()
    assert "bool sampled_protect_converter(" in header
    assert "bool protect_converter(" not in header


@pytest.mark.parametrize("name", ["buck", "gfl", "wpt-ss"])
def test_plant_algebraic_loop_warning_reuses_the_existing_model(name, tmp_path):
    sim = _simulation(name)
    loops = tuple(loop.labels for loop in sim.system.model.algebraic_loops)
    result = _export(sim, tmp_path)
    warnings = [item for item in result.diagnostics if item["code"] == "algebraic-loop"]
    assert bool(warnings) == bool(loops)
    if loops:
        assert len(warnings) == 1
        assert warnings[0]["loops"] == loops
        assert "not exported" in warnings[0]["message"]
        assert all(label in warnings[0]["message"] for loop in loops for label in loop)
    assert "solve_algebraic" not in (tmp_path / "peslite_control.hpp").read_text()


def test_single_transistor_needs_no_complementary_dead_time(tmp_path):
    sim = _simulation()
    config = _mapping(sim)
    config["units"]["converter"].pop("dead_time")
    _export(sim, tmp_path, config)
    source = (tmp_path / "user.cpp").read_text()
    assert "PWMH_ACTIVE" in source and "CbPwm_ConfigureDeadTime" not in source


def test_unused_hardware_frequency_history_has_no_storage(tmp_path):
    sim = _simulation("gfl")
    _export(sim, tmp_path / "off")
    assert "u_vsc_rocof_window" not in (tmp_path / "off" / "peslite_control.hpp").read_text()
    enabled = peslite.Simulation(sim.p.replace(**{
        "units.vsc.protection.rocof": {"enable": True, "limit": 2.0, "window": 0.001}}))
    _export(enabled, tmp_path / "on")
    assert "MovingWindow<20> u_vsc_rocof_window" in (tmp_path / "on" / "peslite_control.hpp").read_text()


@pytest.mark.parametrize("calibration, averages", [
    ({"sensitivity": 0.5}, []), ({"offset": 1.0}, []), ({}, ["u_g"]),
])
def test_compatible_adc_sharing_is_allowed_but_pwm_ownership_is_unique(calibration, averages, tmp_path):
    sim = _simulation("gfl")
    config = _mapping(sim)
    config["units"]["vsc"]["adc"]["i_c"][0] = config["units"]["vsc"]["adc"]["u_g"][0]
    _export(sim, tmp_path / "allowed", config)
    sim = peslite.Simulation(sim.p.replace(**{"units.vsc.meas.average": averages}))
    config["units"]["vsc"]["adc"]["i_c"][0] = {"channel": 0, "sign": -1, **calibration}
    with pytest.raises(ValueError, match="conflicting calibration"):
        _export(sim, tmp_path / "bad-adc", config)
    config = _mapping(sim)
    config["units"]["vsc"]["pwm"] = [0, 0, 2]
    with pytest.raises(ValueError, match="more than one bridge leg"):
        _export(sim, tmp_path / "bad-pwm", config)


@pytest.mark.parametrize("mutation, message", [
    (lambda c: c["units"]["converter"].pop("adc"), "bind exactly"),
    (lambda c: c["units"]["converter"].update(dead_time=-1.0), "nonnegative"),
    (lambda c: c["units"]["converter"].update(dead_time=1.0), "half a switching"),
    (lambda c: c["units"]["converter"].update(pwm=[]), "bind each"),
    (lambda c: c["units"]["converter"]["adc"].update(current={"channel": 0, "sign": 0}), "sign"),
    (lambda c: c["units"]["converter"]["adc"].update(current={"channel": True}), "integer"),
    (lambda c: c.update(sampling_phase=1.0), "sampling_phase"),
    (lambda c: c.update(unknown=True), "unknown fields"),
])
def test_invalid_binding_fails_before_creating_output(mutation, message, tmp_path):
    sim = _simulation()
    config = _mapping(sim)
    mutation(config)
    with pytest.raises(ValueError, match=message):
        _export(sim, tmp_path / "not-created", config)
    assert not (tmp_path / "not-created").exists()


def test_parameter_selection_and_caller_ownership(tmp_path):
    sim = _simulation()
    before = copy.deepcopy(sim.units["converter"].ctrl.get_state())
    selected = "units.converter.ctrl.references.voltage"
    _export(sim, tmp_path, variables=[selected])
    header = (tmp_path / "peslite_control.hpp").read_text()
    assert "parameters.param_units_converter_ctrl_references_voltage" in header
    assert "u_converter_reference_voltage" not in header
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["parameters"] == {selected: "param_units_converter_ctrl_references_voltage"}
    assert sim.units["converter"].ctrl.get_state() == before
    _export(sim, tmp_path / "solver", variables=["simulation.t_end"])
    assert "volatile float param_simulation_t_end" in (tmp_path / "solver" / "user.cpp").read_text()


@pytest.mark.parametrize("selector", ["all", "common"])
def test_hardware_parameter_defaults_follow_the_normal_export_interface(selector, tmp_path):
    sim = _simulation()
    _export(sim, tmp_path / "constant")
    assert "volatile float param_" not in (tmp_path / "constant" / "user.cpp").read_text()
    _export(sim, tmp_path / "variable", variables=[selector])
    assert "volatile float param_units_converter_ctrl_references_voltage" in (
        tmp_path / "variable" / "user.cpp").read_text()
    sdk = tmp_path / "sdk"
    _sdk_double(sdk)
    source = tmp_path / "variable" / "check.cpp"
    source.write_text('#include "user.cpp"\nint main() { return UserInit(); }\n', encoding="utf-8")
    _compile(source, tmp_path / "check", sdk)
    subprocess.run([str(tmp_path / "check")], check=True)


@pytest.mark.parametrize("selector", ["all", "common", "units.converter.ctrl.references", "simulation.solver"])
def test_parameter_range_and_symbols_match_standalone_cpp_except_fixed_hardware_timing(selector, tmp_path):
    import re

    sim = _simulation()
    result = _export(sim, tmp_path / "hardware", variables=[selector])
    sim.export("cpp", tmp_path / "simulation", variables=[selector])
    manifest = json.loads((tmp_path / "hardware" / "manifest.json").read_text())
    cpp = (tmp_path / "simulation" / "peslite.cpp").read_text()
    paths = re.findall(r'if \(path == "([^"]+)"\)', cpp)
    fixed = {path for item in result.diagnostics if item["code"] == "hardware-timing-constants"
             for path in item["parameters"]}
    assert manifest["parameters"] == {path: "param_" + imperix._identifier(path)
                                      for path in paths if path not in fixed}
    if selector in ("all", "common"):
        assert "units.converter.pwm.f_sw" in fixed
        assert "units.converter.pwm.f_sw" in paths  # Standalone C++ still allows --set.
    assert "ref_units_" not in (tmp_path / "hardware" / "user.cpp").read_text()


@pytest.mark.parametrize("variables", [("all",), (
    "units.converter.ctrl.loops.DCVoltageControl.kp",
    "units.converter.ctrl.loops.DCVoltageControl.ki",
    "units.converter.ctrl.loops.InductorCurrentControl.ki",
)])
def test_online_gains_refresh_coefficients_without_resetting_loop_state(variables, tmp_path):
    sim = _simulation()
    _export(sim, tmp_path / "export", variables=variables)
    sdk = tmp_path / "sdk"
    _sdk_double(sdk)
    unit = sim.units["converter"]
    unit.ctrl.command(True, 0.0)
    expected, lines = [], ['#include "user.cpp"', '#include <iostream>', '#include <iomanip>',
                           'int main() { std::cout << std::setprecision(17);',
                           'if (UserInit() != SAFE) return 1; peslite_run = 1;',
                           'input[0][0] = 1.0; input[0][1] = 23.0; input[0][2] = 48.0;']
    for index in range(24):
        if index == 8:
            changes = {"ctrl.loops.DCVoltageControl.kp": 0.6,
                       "ctrl.loops.DCVoltageControl.ki": 160.0,
                       "ctrl.loops.InductorCurrentControl.ki": 60.0}
            updated = sim.p.replace(**{"units.converter." + key: value for key, value in changes.items()})
            unit.ctrl.retune(updated.unit("converter"), tuple(changes), index * unit.cfg.ctrl.period)
            lines += [f"param_units_converter_{imperix._identifier(key)} = {value!r};"
                      for key, value in changes.items()]
        t = (index + 0.5) * unit.cfg.ctrl.period
        output = unit.ctrl(t, SimpleNamespace(t=t, current=1.0, voltage=23.0, vin=48.0))
        expected.append(unit.bridge.topology.command_duties(output))
        lines += ['if (UserInterrupt() != SAFE) return 2;',
                  "for (double value : control_converter.duty) std::cout << value << ' ';",
                  "std::cout << '\\n';"]
    lines += ['auto accepted = peslite_generated::parameters;',
              'param_units_converter_ctrl_loops_DCVoltageControl_ki = std::numeric_limits<float>::quiet_NaN();',
              'if (UserInterrupt() != UNSAFE || peslite_run != 0 || enabled[0][0]) return 3;',
              'if (peslite_generated::parameters.param_units_converter_ctrl_loops_DCVoltageControl_ki !=',
              '    accepted.param_units_converter_ctrl_loops_DCVoltageControl_ki) return 4;', '}']
    source = tmp_path / "export" / "check.cpp"
    source.write_text("\n".join(lines), encoding="utf-8")
    _compile(source, tmp_path / "check", sdk)
    result = subprocess.run([str(tmp_path / "check")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = [tuple(map(float, line.split())) for line in result.stdout.splitlines()]
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("variables", [(), ("all",), ("common",)])
@pytest.mark.parametrize("frequency, sample_period, expected_clock", [
    (20000.0, 50e-6, 0),  # ADC and both PWM outputs share CLK0.
    (20000.0, 12.5e-6, 1),  # Faster ADC, both PWM outputs share CLK1.
    (85000.0, 1.0 / 85000.0, 0),  # Reciprocal roundoff must not allocate another clock.
])
def test_equal_frequencies_share_clocks_independently_of_parameter_selection(
        variables, frequency, sample_period, expected_clock, tmp_path):
    base = _simulation("two-converters")
    settings = {f"units.{name}.{path}": value for name in base.units for path, value in {
        "pwm.f_sw": frequency, "ctrl.period": 4.0 * sample_period,
        "meas.period": sample_period,
    }.items()}
    settings["units.vsc_2.pwm.carrier_phase"] = 0.25
    sim = peslite.Simulation(base.p.replace(**settings))
    _export(sim, tmp_path / "export", variables=variables)
    source_text = (tmp_path / "export" / "user.cpp").read_text()
    assert source_text.count("Clock_SetFrequency(") == expected_clock + 1
    interrupt = source_text.split("tUserSafe UserInterrupt()", 1)[1]
    assert "Clock_SetFrequency" not in interrupt and "CbPwm_SetPhase" not in interrupt
    assert "Clock_ConfigureAsRealTimeTunable" not in source_text
    sdk = tmp_path / "sdk"
    _sdk_double(sdk)
    source = tmp_path / "export" / "check.cpp"
    source.write_text(f'''#include "user.cpp"
#include <cassert>
int main() {{
    assert(UserInit() == SAFE);
    assert(pwm_clock[0][0] == CLOCK_{expected_clock} && pwm_clock[0][3] == CLOCK_{expected_clock});
    assert(frequency[CLOCK_{expected_clock}] == {frequency!r});
    assert(carrier_phase[0][0] == 0.0 && carrier_phase[0][3] == 0.25);
    input[0][6] = 1500.0; input[0][13] = 1500.0;
    assert(UserInterrupt() == SAFE);
    assert(frequency[CLOCK_{expected_clock}] == {frequency!r});
}}
''', encoding="utf-8")
    _compile(source, tmp_path / "check", sdk)
    result = subprocess.run([str(tmp_path / "check")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name, key, changes", [
    ("gfl", "vsc", {"DCVoltageControl.kp": 0.6, "DCVoltageControl.ki": 60.0,
                     "CurrentLoop.bandwidth": 300.0}),
    ("wpt-ss", "charger.primary", {"PowerControl.kp": 0.0002, "PowerControl.ki": 0.9,
                                  "PowerControl.power_filter.bandwidth": 1500.0}),
])
def test_ac_and_nested_wpt_gains_retune_through_the_same_parameter_store(name, key, changes, tmp_path):
    sim = _simulation(name)
    _export(sim, tmp_path, variables=["all"])
    unit, tag = sim.units[key], imperix._identifier(key)
    unit.ctrl.command(True, 0.0)
    lines = ['#include "peslite_control.hpp"', '#include <iostream>', '#include <iomanip>',
             'using namespace peslite_generated;', 'int main() { std::cout << std::setprecision(17);',
             f'Controller_{tag} c; c.command(true);']
    expected = []
    for index in range(24):
        t = index * unit.cfg.ctrl.period
        if index == 8:
            paths = {f"ctrl.loops.{path}": value for path, value in changes.items()}
            updated = sim.p.replace(**{f"units.{key}.{path}": value for path, value in paths.items()})
            unit.ctrl.retune(updated.unit(key), tuple(paths), t)
            lines.append('auto candidate = parameters;')
            lines += [f"candidate.param_units_{tag}_{imperix._identifier(path)} = {value!r};"
                      for path, value in paths.items()]
            lines.append('if (!set_parameters(candidate)) return 1;')
        if isinstance(unit.ctrl, VSCController):
            values = dict(u_g=320.0 + 0j, i_c=1.0 + 0j, u_dc=unit.cfg.dc_base.v)
            meas = Measurement(t, **values, i_abc=np.array(phases(values["i_c"])))
        else:
            values = dict(current=1.0, voltage=320.0, vin=400.0, power=150.0)
            meas = SimpleNamespace(t=t, **values)
        expected.append(unit.bridge.topology.command_duties(unit.ctrl(t, meas)))
        lines += ['{', f'Controller_{tag}::Measurements m;']
        for channel, value in values.items():
            literal = (f"Complex{{{value.real!r}, {value.imag!r}}}"
                       if isinstance(value, complex) else repr(float(value)))
            lines.append(f"m.{channel} = {literal};")
        lines += [f'if (!c.sample({t!r}, m)) return 2;',
                  "for (double value : c.duty) std::cout << value << ' ';", "std::cout << '\\n';", '}']
    lines.append('}')
    source = tmp_path / 'check.cpp'
    source.write_text('\n'.join(lines), encoding='utf-8')
    _compile(source, tmp_path / 'check')
    result = subprocess.run([str(tmp_path / 'check')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose([tuple(map(float, line.split())) for line in result.stdout.splitlines()],
                               expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("prefix", [(), ("all",), ("common",)])
@pytest.mark.parametrize("field", ["pwm.f_sw", "pwm.carrier_phase", "ctrl.period", "ctrl.computation",
                                   "meas.period", "meas.window", "ctrl.loops.DCVoltageControl.period",
                                   "pwm", "ctrl.loops"])
def test_explicit_timing_selection_is_rejected_before_writing_files(prefix, field, tmp_path):
    sim = peslite.Simulation(_simulation().p.replace(**{
        "units.converter.meas.period": 50e-6, "units.converter.meas.window": 25e-6,
        "units.converter.meas.average": ["voltage"],
    }))
    path = "units.converter." + field
    with pytest.raises(ValueError, match="timing parameters must remain constant") as error:
        _export(sim, tmp_path / "out", variables=(*prefix, path))
    assert path in str(error.value)
    assert not (tmp_path / "out").exists()


def test_factory_derived_timing_traces_dependencies_instead_of_matching_names(tmp_path):
    def factory(cfg):
        ctrl = VSCController(cfg)
        loop = cfg.ctrl.loops["CurrentLoop"]
        ctrl.graph.periods["CurrentLoop"] = loop.period * loop.bandwidth / 200.0
        return ctrl

    sim = peslite.Simulation(_simulation("gfl").p, parts={"vsc.ctrl": factory})
    path = "units.vsc.ctrl.loops.CurrentLoop.bandwidth"
    with pytest.raises(ValueError, match=path):
        _export(sim, tmp_path / "explicit", variables=[path])
    result = _export(sim, tmp_path / "all", variables=["all"])
    fixed = next(item["parameters"] for item in result.diagnostics
                 if item["code"] == "hardware-timing-constants")
    assert path in fixed
    manifest = json.loads((tmp_path / "all" / "manifest.json").read_text())
    assert path not in manifest["parameters"]
    assert "units.vsc.ctrl.loops.DCVoltageControl.kp" in manifest["parameters"]


def test_constant_export_does_not_run_parameter_timing_analysis(monkeypatch, tmp_path):
    def unexpected(*args):
        pytest.fail("constant exports do not need parameter dependency analysis")

    monkeypatch.setattr(imperix, "_timing_parameters", unexpected)
    _export(_simulation(), tmp_path)


@pytest.mark.parametrize("selector", ["all", "common"])
def test_timing_and_common_constants_share_one_parameter_rebuild(selector, monkeypatch, tmp_path):
    from peslite.assembly import cpp

    original, exclusions = cpp._CppGenerator, []

    class Counted(original):
        def __init__(self, *args, **kwargs):
            exclusions.append(set(kwargs.get("exclude", ())))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cpp, "_CppGenerator", Counted)
    result = _export(_simulation(), tmp_path, variables=[selector])
    assert len(exclusions) == 2 and not exclusions[0]
    assert exclusions[1] == {path for item in result.diagnostics
                             if item["code"] in ("common-constants", "hardware-timing-constants")
                             for path in item["parameters"]}


def test_standalone_export_shares_the_slow_vsc_command_frame_fix(tmp_path):
    p = _simulation("gfl-slow-terminal").p.replace(**{
        "simulation.t_end": 0.001, "simulation.solver.subsystems": {},
        "simulation.solver.type": "fixed", "simulation.solver.method": "rk4",
        "events.connect_vsc.t": 0.0, "events.connect_vsc.ramp": 0.0,
        "simulation.output.states": True, "simulation.output.signals": True,
        "simulation.output.record_every": 1,
    })
    compiler = _compiler()
    if compiler is None:
        pytest.skip("a C++17 host compiler is required")
    _run_generated(p, tmp_path, compiler)


def test_cli_help_describes_imperix_elf_export(capsys):
    assert convert_main(["--help"]) == 0
    assert "Imperix ELF export" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "Imperix ELF export" in capsys.readouterr().out


def test_cli_sidecar_and_backend_option_routing(tmp_path, capsys):
    sim = _simulation()
    case = tmp_path / "case.pes"
    peslite.dump(sim.p, case)
    case.with_suffix(".imperix.json").write_text(json.dumps(_mapping(sim)), encoding="utf-8")
    assert convert_main([str(case), "--imperix", "--imperix-source-only", "--out", str(tmp_path / "out")]) == 0
    warnings = capsys.readouterr().err
    assert "hardware timing" in warnings
    assert "warning: The simulation contains algebraic loops" in warnings
    assert (tmp_path / "out" / "user.cpp").exists()
    with pytest.raises(SystemExit):
        main([str(case), "--imperix-source-only"])
    with pytest.raises(SystemExit):
        main([str(case), "--imperix", "--export", "cpp"])


def _target_elf():
    # A synthetic ARM ELF exercises artifact validation only; it is not a hardware binary.
    data = bytearray(320)
    data[:7] = b"\x7fELF\x01\x01\x01"
    struct.pack_into("<HHI", data, 16, 2, 40, 1)
    struct.pack_into("<III", data, 24, 0x1000, 52, 84)
    struct.pack_into("<HHHHHH", data, 40, 52, 32, 1, 40, 4, 3)
    struct.pack_into("<I", data, 52, 1)
    struct.pack_into("<I", data, 128, 2)
    names = b"\0.symtab\0.debug_info\0.shstrtab\0"
    data[244:244 + len(names)] = names
    for index, name in enumerate((b".symtab", b".debug_info", b".shstrtab"), 1):
        struct.pack_into("<II", data, 84 + index * 40, names.index(name), (2, 1, 3)[index - 1])
    struct.pack_into("<II", data, 84 + 3 * 40 + 16, 244, len(names))
    return bytes(data)


def test_sdk_build_is_explicit_fresh_and_portable(tmp_path, monkeypatch):
    sim = _simulation()
    directory = tmp_path / "out"
    _export(sim, directory)
    monkeypatch.delenv("PESLITE_IMPERIX_SDK", raising=False)
    with pytest.raises(RuntimeError, match="installed official"):
        build_project(directory)
    assert json.loads((directory / "manifest.json").read_text())["status"] == "build_failed"
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    fixture = tmp_path / "fixture.elf"
    fixture.write_bytes(_target_elf())
    profile = {"sdk": str(sdk), "elf_machine": 40, "name": "run",
               "command": [sys.executable, "-B", "-c",
                           "from pathlib import Path; import sys; Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())",
                           str(fixture), "{elf}"]}
    profile_path = directory / "imperix-build.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    # Standalone build.py has no dependency on the PESLite installation.
    subprocess.run([sys.executable, "-B", str(directory / "build.py")], cwd=tmp_path, check=True,
                   capture_output=True, text=True)
    old = (directory / "run.elf").read_bytes()
    assert old == _target_elf()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "built" and manifest["hardware_tested"] is False
    profile["command"] = [sys.executable, "-B", "-c", "import sys; sys.exit(0)", "{elf}"]
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fresh ELF"):
        build_project(directory)
    assert (directory / "run.elf").read_bytes() == old
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["artifact"] is None and manifest["sha256"] is None
    assert not list(directory.glob(".build-*"))


@pytest.mark.parametrize("change, message", [
    (lambda p: p.__setitem__(18, 62), "machine mismatch"),
    (lambda p: p.__setitem__(16, 1), "executable"),
    (lambda p: p.__setitem__(128, 0), "stripped"),
    (lambda p: p.__setitem__(52, 0), "loadable"),
    (lambda p: p.__setitem__(164, 0), "debug information"),
])
def test_host_or_invalid_elf_cannot_be_published(change, message, tmp_path):
    data = bytearray(_target_elf())
    change(data)
    path = tmp_path / "bad.elf"
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match=message):
        imperix.imperix_build._elf(path, 40)
