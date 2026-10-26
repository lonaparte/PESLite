"""Imperix ELF export using the existing sampled control equations and SDK peripheral bindings.

Only the hardware boundary is new. Loop equations, state updates, startup, output-stage equations
and sampled protection use the C++ lowering already used by standalone simulation export.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from ..control.controller import VSCController
from . import imperix_build
from .cpp import (_CONTROL_MATH, _MOVING_WINDOW, _ControlGenerator, _Expression,
                  _ModelGenerator, _ProtectionGenerator,
                  _controller_implementation, _identifier, _number,
                  _parameter_generator, _prepare_export)
from .exporter import ExportResult, _exporter, _signal_names
from .validate import _whole


def _keys(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"{where}: unknown fields {sorted(unknown)}")


def _positive(value, where, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError(f"{where} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def _channel(value, *, adc=False):
    value = {"channel": value} if isinstance(value, int) and not isinstance(value, bool) else value
    _keys(value, {"channel", "device", "sensitivity", "offset", "sign"} if adc
          else {"channel", "device"}, "ADC binding" if adc else "PWM binding")
    result = {"channel": value.get("channel"), "device": value.get("device", 0)}
    for name, index in result.items():
        if type(index) is not int or index < 0:
            raise ValueError(f"binding.{name} must be a nonnegative integer")
    if adc:
        result["sensitivity"] = _positive(value.get("sensitivity", 1.0), "ADC sensitivity")
        offset = value.get("offset", 0.0)
        if isinstance(offset, bool) or not isinstance(offset, (float, int)) or not math.isfinite(offset):
            raise ValueError("ADC offset must be finite")
        sign = value.get("sign", 1)
        if type(sign) is not int or sign not in (-1, 1):
            raise ValueError("ADC sign must be 1 or -1")
        result.update(offset=float(offset), sign=sign)
    return result


class _ImperixControlGenerator(_ProtectionGenerator, _ControlGenerator):
    """An allocation-free hardware boundary for the existing control-loop tracer."""

    continuous = False

    def __init__(self, name, unit, model, binding):
        super().__init__(name, unit, model, "u")
        self.topology = unit.bridge.topology
        self.binding = binding

    def _adc_value(self, channel):
        return "measured." + _identifier(channel)

    def source(self, path):
        if path.startswith("meas."):
            return "meas_" + _identifier(path[5:])
        if path.startswith("references."):
            return self._reference(path[11:])
        if path == "startup.active":
            return self.m("startup_run")
        if path in self.graph.interface.outputs:
            return self.source(self.graph.outputs[path])
        return self.g(path)

    def state_expression(self, _part, local):
        return self.node(*local.split(".", 1))

    def alarm_members(self):
        return []

    def _observes_rocof(self):
        return self.cfg.protection.rocof.enable and super()._observes_rocof()

    def _protection_helpers(self):
        # The hardware fault latch is fixed-size. No strings or alarm vectors enter the ISR.
        return [
            f"void alarm_{self.tag}(std::string_view) {{}}",
            f"bool latch_{self.tag}(double t, std::string_view, bool fault = false) {{",
            f"    {self.m('fault')} = {self.m('fault')} || fault;",
            f"    if ({self.m('tripped')}) return false;",
            f"    {self.m('tripped')} = true; {self.m('trip_time')} = t; return true;",
            "}",
        ]

    def _trip_body(self):
        return [f"{self.m('startup_run')} = false;"]

    def header(self):
        startup = self.ctrl.startup
        channels = self.model.unit_measurements(self.unit)
        measured = [f"{'Complex' if value.kind == 'complex' else 'double'} "
                    f"{_identifier(name)}{{}};" for name, value in channels.items()]
        members = [
            f"bool {self.m('tripped')} = false, {self.m('fault')} = false;",
            f"double {self.m('trip_time')} = std::numeric_limits<double>::quiet_NaN();",
            f"bool {self.m('startup_run')} = false, {self.m('startup_complete')} = false;",
            f"double {self.m('startup_value')} = 0.0;",
            f"double {self.m('startup_ramp')} = {_number(self.binding.get('startup_ramp', startup.ramp))};",
            f"long long {self.m('startup_steps')} = 0, {self.m('pwm_k')} = 0;",
            f"bool {self.m('command_saturated')} = false;",
            *[f"double {self.m('prot_' + key)} = 0.0;"
              for key in self.protection.get_state()],
            *self.protection_members(), *self._loop_members(),
            *[f"{'Complex' if isinstance(value, complex) else 'double'} "
              f"{self.g(key)} = {self._literal(value)};" for key, value in self.graph.values.items()],
            f"std::array<double, {len(self.topology.duty_names)}> duty{{}};",
        ]
        on = self._controller_command_lines(True, self.binding.get("startup_ramp", startup.ramp))
        off = self._controller_command_lines(False, 0.0)
        lines = self._sampled_measurement_lines()
        lines += self._sampled_loop_lines()
        if isinstance(self.ctrl, VSCController):
            members += [f"Complex {self.m('u_cmd')} = {self._literal(self.ctrl.u_cmd)};",
                        f"double {self.m('theta')} = {_number(self.ctrl.theta)};",
                        f"double {self.m('omega')} = {_number(self.ctrl.omega)};",
                        f"double {self.m('command_theta')} = {_number(self.ctrl.command_theta)};"]
            lines += self._sampled_vsc_command_lines()
            modulation, _ = self._modulation_values(
                self.m("u_cmd"), self.m("command_theta"), self._adc_value("u_dc"))
            output = SimpleNamespace(d_abc=tuple(0.5 * (1.0 + value) for value in modulation))
            voltage = "std::abs(meas_u_g)"
            frequency = f"({self.source(self.graph.outputs['omega'])} - {_number(self.cfg.base.w0)}) / (2.0 * pi)"
            vdc_error = f"meas_u_dc - {self._reference('vdc_ref_pu')}"
        else:
            output = SimpleNamespace(**{port: _Expression(self.source(port), "real")
                                        for port in self.graph.interface.outputs})
            voltage = f"std::abs(meas_voltage) / {_number(self.cfg.dc_base.v)}"
            frequency = "0.0"
            dc_value = self.model.value(self.unit.dclink.out, "u_dc")
            dc_channel = next((key for key, value in channels.items() if value.code == dc_value), None)
            if dc_channel is None and self.cfg.protection.dc_voltage.enable:
                raise ValueError(f"{self.name}: DC-link protection has no mapped DC-link measurement")
            vdc_error = (f"measured.{_identifier(dc_channel)} / {_number(self.cfg.dc_base.v)} - 1.0"
                         if dc_channel is not None else "std::numeric_limits<double>::quiet_NaN()")
        for index, value in enumerate(self.topology.command_duties(output)):
            lines.append(f"duty[{index}] = {_Expression.of(value).code};")
        lines += [f"sampled_protect_{self.tag}(t, {voltage}, {frequency}, {vdc_error}, "
                  f"{self.m('startup_complete')});",
                  f"++{self.m('pwm_k')};",
                  "for (double value : duty) if (!std::isfinite(value) || value < 0.0 || value > 1.0)",
                  f"    latch_{self.tag}(t, \"invalid duty\", true);",
                  f"return !{self.m('tripped')};"]
        checks = [f"std::isfinite(std::real(measured.{_identifier(key)})) && "
                  f"std::isfinite(std::imag(measured.{_identifier(key)}))" for key in channels]
        return (f"struct Controller_{self.tag} {{\n"
                + "\n".join("    " + line for line in members)
                + f"\n    struct Measurements {{ {' '.join(measured)} }};\n"
                + f"    void command(bool run) {{\n        if (run == {self.m('startup_run')}) return;\n"
                + "        if (run) {\n" + "\n".join("            " + line for line in on)
                + "\n        } else {\n" + "\n".join("            " + line for line in off)
                + "\n        }\n    }\n"
                + self.protection_method(sampled_only=True)
                + "\n    bool sample(double t, const Measurements& measured) {\n"
                + "        if (!(" + " && ".join(checks) + ")) {\n"
                + f"            latch_{self.tag}(t, \"invalid input\", true); return false;\n        }}\n"
                + "\n".join("        " + line for line in lines)
                + "\n    }\n};\n")


def _common_period(periods):
    """Find a common schedule without enumerating ticks or a hyperperiod."""
    values = [Fraction(value).limit_denominator(250_000_000) for value in periods]
    denominator = math.lcm(*(value.denominator for value in values))
    ticks = [value.numerator * (denominator // value.denominator) for value in values]
    period = math.gcd(*ticks) / denominator
    if period < 4e-9 or not all(_whole(value / period) for value in periods):
        raise ValueError("no common timing period can be mapped at the SDK's 4 ns clock resolution; "
                         "choose compatible sampling and control periods")
    return period


def _timing(units, config):
    """Map existing Unit periods to one ADC clock and a decimated CPU interrupt."""
    controls = [unit.cfg.ctrl.period for unit in units.values()]
    period = (_positive(config["interrupt_period"], "interrupt_period")
              if "interrupt_period" in config else _common_period(controls))
    samples = [unit.cfg.meas.period or unit.cfg.ctrl.period for unit in units.values()]
    windows = [unit.cfg.meas.window or unit.cfg.ctrl.period for unit in units.values()
               if unit.cfg.meas.average]
    sampling_period = _common_period([period, *samples, *windows])
    postscaler = round(period / sampling_period)
    return dict(interrupt_period=period, sampling_period=sampling_period, postscaler=postscaler)


def _unit_periods(cfg):
    """The existing Unit fields that define scheduled execution or a PWM carrier."""
    return {"pwm.f_sw": cfg.pwm.f_sw, "meas.period": cfg.meas.period,
            "ctrl.period": cfg.ctrl.period,
            **{f"ctrl.loops.{name}.period": loop.period for name, loop in cfg.ctrl.loops.items()}}


def _timing_parameters(units, runtime):
    """Trace timing inputs through the shared parameter and factory expression DAG."""
    expressions = []
    for unit in units.values():
        cfg = runtime.copy(unit.cfg)
        values = (*_unit_periods(cfg).values(), cfg.pwm.carrier_phase, cfg.ctrl.computation,
                  cfg.meas.window, *cfg.meas.rms.values(),
                  *runtime.object(unit.ctrl).graph.periods.values())
        expressions.extend(_number(value) for value in values if value is not None)
    dependencies = runtime.parameter_dependencies(expressions)
    if any(paths is None for paths in dependencies.values()):
        raise TypeError("Imperix timing must depend only on fixed configuration parameters")
    return {path for paths in dependencies.values() for path in paths}


def _clock_bindings(units, config, timing):
    """Resolve automatic or explicit clock IDs into one hardware frequency table."""
    mode = config.get("clock_mode", "auto")
    if mode not in ("auto", "manual"):
        raise ValueError("imperix.clock_mode must be 'auto' or 'manual'")
    assigned = {"interrupt_period": 0}
    required = {"interrupt_period"} if "interrupt_period" in config else set()
    carriers = {}
    for name, unit in units.items():
        prefix = f"units.{name}."
        for field, value in _unit_periods(unit.cfg).items():
            path = prefix + field
            assigned[path] = 0
            if value is not None and field not in unit.cfg.derived_defaults:
                required.add(path)
        carriers[prefix + "pwm.f_sw"] = unit.cfg.pwm.f_sw
    requested = config.get("clocks", {})
    _keys(requested, assigned.keys(), "imperix.clocks")
    if mode == "auto" and requested:
        raise ValueError("imperix.clocks bindings require clock_mode='manual'")
    for path, clock in requested.items():
        if clock not in ("CLK0", "CLK1", "CLK2", "CLK3"):
            raise ValueError(f"imperix.clocks.{path}: expected CLK0, CLK1, CLK2 or CLK3")
        if path not in carriers and clock != "CLK0":
            raise ValueError(f"imperix.clocks.{path}: ADC and CPU schedules must use CLK0")
        assigned[path] = int(clock[-1])
    if mode == "manual" and (missing := required - requested.keys()):
        raise ValueError("imperix.clocks is missing explicit period bindings: " + ", ".join(sorted(missing)))
    clocks = {0: 1.0 / timing["sampling_period"]}
    for path, frequency in carriers.items():
        clock = (assigned[path] if mode == "manual" else
                 next((key for key, value in clocks.items()
                       if math.isclose(value, frequency, rel_tol=1e-12)), len(clocks)))
        if clock >= 4:
            raise ValueError("the sampling clock and PWM frequencies exceed four hardware clocks")
        existing = clocks.setdefault(clock, frequency)
        if not math.isclose(existing, frequency, rel_tol=1e-12):
            raise ValueError(f"imperix.clocks.{path}: CLK{clock} already uses {existing:.12g} Hz; "
                             f"cannot also use {frequency:.12g} Hz")
        assigned[path] = clock
    return clocks, assigned


def _bindings(simulation, config, model, *, events=None):
    _keys(config, {"units", "interrupt_period", "sampling_phase", "clock_mode", "clocks", "build"}, "imperix")
    units = simulation.system.units
    bindings = config.get("units")
    if not isinstance(bindings, dict) or bindings.keys() != units.keys():
        raise ValueError(f"imperix.units must bind exactly the leaf Units: {sorted(units)}")
    if not units:
        raise ValueError("Imperix export requires at least one sampled Unit")
    for name, unit in units.items():
        if unit.continuous:
            raise TypeError(f"{name}: Imperix requires a supported sampled control graph")
        if unit.cfg.pwm.pattern_sync == "pulse":
            raise TypeError(f"{name}: the Imperix hardware PWM adapter does not implement "
                            "complete-pulse updates")
        names = unit.bridge.register_names[:unit.bridge.modulation_width]
        if names != unit.bridge.topology.duty_names:
            raise ValueError(f"Imperix export currently binds duty registers; {name!r} uses "
                             f"{names}. Its hardware modulation adapter is not implemented")
        _controller_implementation(unit.ctrl, "__call__", "command")
    timing = _timing(units, config)
    period, sampling_period = timing["interrupt_period"], timing["sampling_period"]
    phase = _positive(config.get("sampling_phase", 0.5), "sampling_phase", zero=True)
    if phase >= 1.0:
        raise ValueError("sampling_phase must be in [0, 1)")
    clocks, assigned = _clock_bindings(units, config, timing)
    adc, pwm, normalized = {}, set(), {}
    for name, unit in units.items():
        cfg, binding = unit.cfg, bindings[name]
        configured = model.runtime.copy(cfg)
        _keys(binding, {"adc", "pwm", "dead_time", "hardware_overcurrent_configured", "startup_ramp"},
              f"imperix.units.{name}")
        if cfg.pwm.sync != "asynchronous":
            raise ValueError(f"{name}: angle-locked simulation carriers have no hardware adapter")
        stride = unit.ctrl.graph.period_stride(configured.ctrl.period, period)
        if (configured.ctrl.period < period * (1.0 - 1e-6)
                or abs(stride * period - configured.ctrl.period) > 1e-6 * configured.ctrl.period):
            raise ValueError("Unit control periods must be whole multiples of interrupt_period")
        if cfg.meas.rms:
            raise ValueError(f"{name}: hardware RMS requires an ADC-history adapter, not sampled averages")
        sample_period = configured.meas.period or configured.ctrl.period
        sample_stride = unit.ctrl.graph.period_stride(sample_period, sampling_period)
        if (sample_period < sampling_period * (1.0 - 1e-6)
                or abs(sample_stride * sampling_period - sample_period) > 1e-6 * sample_period):
            raise ValueError(f"{name}: ADC sample period is not on the mapped CLOCK_0 grid")
        window = configured.meas.window or configured.ctrl.period
        history = int(round(window / sampling_period)) if cfg.meas.average else 1
        if cfg.meas.average and abs(window - history * sampling_period) > 1e-6 * window:
            raise ValueError(f"{name}: averaging window is not on the mapped CLOCK_0 grid")
        if history > 64:
            raise ValueError(f"{name}: the averaging window requires {history} ADC history samples; "
                             "the SDK supports at most 64")
        unilateral = unit.bridge.topology.unilateral
        dead_time = _positive(binding.get("dead_time", 0.0 if unilateral else None),
                              f"{name}.dead_time", zero=unilateral)
        if dead_time >= 0.5 / configured.pwm.f_sw:
            raise ValueError(f"{name}: dead_time must be shorter than half a switching period")
        if cfg.protection.overcurrent.enable and binding.get("hardware_overcurrent_configured") is not True:
            raise ValueError(f"{name}: enabled fast protection requires hardware_overcurrent_configured=true")
        default_ramp = next((event.ramp for event in (simulation.p.changes if events is None else events)
                             if event.values.get(f"units.{name}.connected") is True),
                            model.runtime.object(unit.ctrl).startup.ramp)
        ramp = binding.get("startup_ramp", default_ramp)
        _positive(ramp, f"{name}.startup_ramp", zero=True)
        readers = model.unit_measurements(unit)
        mapped = binding.get("adc")
        if not isinstance(mapped, dict) or mapped.keys() != readers.keys():
            raise ValueError(f"{name}.adc must bind exactly {sorted(readers)} in SI units")
        channels = {}
        for key, spec in readers.items():
            values = mapped[key]
            values = values if isinstance(values, list) else [values]
            width = 3 if spec.kind == "complex" else 1
            if len(values) != width:
                raise ValueError(f"{name}.adc.{key} requires {width} channel bindings")
            channels[key] = tuple(_channel(value, adc=True) for value in values)
            for item in channels[key]:
                identity = item["device"], item["channel"]
                calibration = {field: value for field, value in item.items() if field != "sign"}
                calibration["average"] = key in cfg.meas.average
                if identity in adc and any(adc[identity][field] != value
                                           for field, value in calibration.items()):
                    raise ValueError(f"ADC {identity} has conflicting calibration or averaging")
                item["history"] = history if calibration["average"] else 1
                depth = max(adc.get(identity, {}).get("history", 1), item["history"])
                adc[identity] = {**calibration, "history": depth}
        outputs = binding.get("pwm")
        if not isinstance(outputs, list) or len(outputs) != len(unit.bridge.topology.duty_names):
            raise ValueError(f"{name}.pwm must bind each of {unit.bridge.topology.duty_names}")
        outputs = tuple(_channel(value) for value in outputs)
        for item in outputs:
            identity = item["device"], item["channel"]
            if identity in pwm:
                raise ValueError(f"PWM {identity} is assigned to more than one bridge leg")
            pwm.add(identity)
        normalized[name] = dict(adc=channels, pwm=outputs, dead_time=dead_time,
                                startup_ramp=ramp, stride=stride, clock=assigned[f"units.{name}.pwm.f_sw"],
                                hardware_overcurrent_configured=binding.get("hardware_overcurrent_configured", False))
    return normalized, adc, clocks, timing, phase, assigned


def _parameter_header(runtime):
    """One shared parameter snapshot; retuning never reconstructs controller state."""
    if not runtime.selected:
        return ""
    members = runtime.initialization_members()
    checks = [runtime.invalid_value(item, "parameters." + item.member)
              for item in runtime.selected]
    checks += [condition for _, condition in runtime.validation_conditions()]
    commit = ["peslite_generated::parameters = parameters;"]
    commit += [f"peslite_generated::{member} = {member};"
               for member in runtime.initializers.values()]
    return (runtime.declarations() + "\ninline RuntimeParameters parameters;\n"
            + "\n".join("inline " + line for line in members)
            + "\ninline bool set_parameters(const RuntimeParameters& parameters) {\n"
            + "\n".join("    " + line for line in members)
            + "\n" + "\n".join(f"    if ({condition}) return false;" for condition in checks)
            + "\n" + "\n".join("    " + line for line in commit)
            + "\n    return true;\n}\n")


def _parameter_source(runtime, declare):
    """Use the SDK's native global variables, with the ordinary C++ member names."""
    if not runtime.selected:
        return
    read, changed, assign, publish = [], [], [], []
    for item in runtime.selected:
        kind = "int" if isinstance(item.default, int) else "float"
        value = f"value_{item.member}"
        effective = runtime.value(item.path, item.default)
        previous = f"static_cast<{kind}>({effective})"
        declare(item.member, f"volatile {kind} {item.member} = {_number(item.default)};")
        read.append(f"const {kind} {value} = {item.member};")
        changed.append(f"{value} != {previous}")
        assign.append(f"if ({value} != {previous}) {{")
        assign.append(f"    candidate.{item.member} = {value};")
        if item.derived or isinstance(item.default, bool):
            assign.append(f"    candidate.given_{item.member} = true;")
        if item.derived:
            publish += [f"if (!peslite_generated::parameters.given_{item.member})",
                        f"    {item.member} = static_cast<{kind}>({effective});"]
        assign.append("}")
    function = ("static bool update_parameters() {\n"
                + "    using namespace peslite_generated;\n"
                + "\n".join("    " + line for line in read)
                + "\n    if (!(" + " || ".join(changed) + ")) return true;\n"
                + "    auto candidate = peslite_generated::parameters;\n"
                + "\n".join("    " + line for line in assign)
                + "\n    if (!peslite_generated::set_parameters(candidate)) return false;\n"
                + "\n".join("    " + line for line in publish) + "\n"
                + "    return true;\n}\n")
    declare("update_parameters", function)


def _source(generators, adc, clocks, period, phase, runtime, *, sampling_period=None, postscaler=1):
    globals_, probes = {}, {}
    init, sample, compute, write, stop = [], [], [], [], []
    sampling_period = period if sampling_period is None else sampling_period
    readings = {}

    def reading(item):
        device, channel, depth = item["device"], item["channel"], item.get("history", 1)
        raw = f"adc_{device}_{channel}"
        if depth == 1:
            return raw
        key = device, channel, depth
        if key not in readings:
            value = f"{raw}_mean_{depth}"
            warmup = (depth - 1 + postscaler - 1) // postscaler
            sample.extend([
                f"double {value} = 0.0;",
                f"if (tick >= {warmup}) {{",
                f"    {value} = {raw};",
                f"    for (unsigned n = 1; n < {depth}; ++n)",
                f"        {value} += Adc_GetHistory({channel}, n, {device});",
                f"    {value} /= {depth};",
                "}",
            ])
            readings[key] = value
        return readings[key]

    def declare(symbol, code):
        if symbol in globals_:
            raise ValueError(f"Imperix exported names collide after C++ identifier normalization: {symbol}")
        globals_[symbol] = code

    _parameter_source(runtime, declare)
    declare("tick", "namespace { unsigned long long tick = 0; }")
    devices = {item["device"] for item in adc.values()}
    for clock, frequency in sorted(clocks.items()):
        init.append(f"Clock_SetFrequency(CLOCK_{clock}, {frequency});")
    init.append(f"ConfigureMainInterrupt(UserInterrupt, CLOCK_0, {_number(phase)}, {postscaler});")
    for (device, channel), item in adc.items():
        init.append(f"Adc_ConfigureSensor({channel}, {_number(item['sensitivity'])}, "
                    f"{_number(item['offset'])}, {device});")
        if item["average"]:
            init.append(f"Adc_EnableSynchronousAveraging({channel}, {device});")
        if item["history"] > 1:
            init.append(f"Adc_ConfigureHistory({channel}, {item['history']}, {device});")
        sample.append(f"const double adc_{device}_{channel} = Adc_GetValue({channel}, {device});")
    for gen in generators:
        name, tag, binding = gen.name, gen.tag, gen.binding
        instance = f"control_{tag}"
        command = runtime.by_path[f"units.{name}.connected"].member
        run = f"peslite_generated::parameters.{command} != 0"
        ready, active = f"ready_{tag}", f"active_{tag}"
        declare(instance, f"peslite_generated::Controller_{tag} {instance};")
        declare(ready, f"unsigned long long {ready} = std::numeric_limits<unsigned long long>::max();")
        declare(active, f"bool {active} = false;")
        init.append(f"{instance} = peslite_generated::Controller_{tag}{{}};")
        init += [f"{ready} = std::numeric_limits<unsigned long long>::max();", f"{active} = false;"]
        init.append(f"{command} = 0; peslite_generated::parameters.{command} = 0;")
        stop += [f"{command} = 0; peslite_generated::parameters.{command} = 0;",
                 f"{instance}.command(false);",
                 f"{ready} = std::numeric_limits<unsigned long long>::max();", f"{active} = false;"]
        read = []
        for key, channels in binding["adc"].items():
            values = [f"({_number(item['sign'])} * {reading(item)})"
                      for item in channels]
            for path, measured in zip(_signal_names(f"{name}.{key}", len(values)), values):
                symbol = _identifier(path)
                declare(symbol, f"volatile float {symbol} = 0.0f;")
                probes[path] = symbol
                init.append(f"{symbol} = 0.0f;")
                sample.append(f"{symbol} = static_cast<float>({measured});")
            value = ("peslite_generated::abc_to_complex(" + ", ".join(values) + ")"
                     if len(values) == 3 else values[0])
            read.append(f"measured.{_identifier(key)} = {value};")
        stride = f"stride_{tag}"
        frequency = _number(gen.cfg.pwm.f_sw)
        depth = max(item.get("history", 1) for channels in binding["adc"].values() for item in channels)
        warmup = (depth - 1 + postscaler - 1) // postscaler
        due = f"tick % {stride} == 0"
        if warmup:
            due += f" && tick >= {warmup}"
        compute.append(f"constexpr unsigned long long {stride} = {binding['stride']};")
        compute += [f"{instance}.command({run});",
                    f"if (!({run})) {ready} = std::numeric_limits<unsigned long long>::max();",
                    f"if ({due}) {{",
                    f"    peslite_generated::Controller_{tag}::Measurements measured;",
                    *["    " + line for line in read],
                    f"    if (!{instance}.sample(t, measured)) {{ stop_all(); return UNSAFE; }}",
                    f"    if (({run}) && {ready} == std::numeric_limits<unsigned long long>::max())",
                    f"        {ready} = tick + static_cast<unsigned long long>(std::ceil(1.0 / ({frequency}) / {_number(period)})) + 1;",
                    "}"]
        mode = "PWMH_ACTIVE" if gen.topology.unilateral else "COMPLEMENTARY"
        duties, activate, deactivate_all = [], [], []
        carrier_phase = _number(gen.cfg.pwm.carrier_phase % 1.0)
        for index, output in enumerate(binding["pwm"]):
            channel, device = output["channel"], output["device"]
            devices.add(device)
            address = f"PWM_CHANNEL_{channel}, "
            init += [f"CbPwm_ConfigureClock({address}CLOCK_{binding['clock']}, {device});",
                     f"CbPwm_ConfigureCarrier({address}TRIANGLE, {device});",
                     f"CbPwm_ConfigureUpdateRate({address}{gen.cfg.pwm.update.upper()}_RATE, {device});",
                     f"CbPwm_ConfigureOutputMode({address}{mode}, {device});",
                     *([f"CbPwm_ConfigureDeadTime({address}{_number(binding['dead_time'])}, {device});"]
                       if mode == "COMPLEMENTARY" else []),
                     f"CbPwm_SetPhase({address}{carrier_phase}, {device});"]
            deactivate = f"CbPwm_Deactivate(PWM_CHANNEL_{channel}, {device});"
            init.append(deactivate)
            stop.append(deactivate)
            duties.append(f"CbPwm_SetDutyCycle({address}{instance}.duty[{index}], {device});")
            activate.append(f"CbPwm_Activate(PWM_CHANNEL_{channel}, {device});")
            deactivate_all.append(deactivate)
        write += [f"if ({due}) {{",
                  *["    " + line for line in duties], "}",
                  f"const bool enable_{tag} = ({run}) && !{instance}.{gen.m('tripped')} && tick >= {ready};",
                  f"if (enable_{tag} != {active}) {{", f"    {active} = enable_{tag};",
                  f"    if ({active}) {{", *["        " + line for line in activate],
                  "    } else {", *["        " + line for line in deactivate_all], "    }", "}"]
    declare("stop_all", "static void stop_all() {\n"
            + "\n".join("    " + line for line in stop) + "\n}\n")
    declare("UserInit", "tUserSafe UserInit() {\n    tick = 0;\n"
            + ("    if (!update_parameters()) return UNSAFE;\n" if runtime.selected else "")
            + "\n".join("    " + line for line in init) + "\n    return SAFE;\n}\n")
    declare("UserInterrupt", "tUserSafe UserInterrupt() {\n"
            + "    using namespace peslite_generated;\n"
            + ("    if (!update_parameters()) { stop_all(); return UNSAFE; }\n" if runtime.selected else "")
            + f"    const double t = tick * {_number(period)} + {_number(phase * sampling_period)};\n"
            + "\n".join("    " + line for line in sample + compute + write)
            + "\n    ++tick; return SAFE;\n}\n")
    declare("UserError", "void UserError(tErrorSource) { stop_all(); }\n")
    return ("#include \"user.h\"\n#include \"peslite_control.hpp\"\n\n"
            + f"NUMBER_OF_DEVICES({max(devices) + 1});\n\n"
            + "\n".join(globals_.values()) + "\n", probes)


@_exporter("imperix")
def _export_imperix(simulation, out_dir, name, variables, reusable, *,
                    target_config=None, sdk=None, build=True):
    """Generate Imperix ELF sources and optionally build the executable with the configured SDK."""
    if not simulation._rebuildable:
        raise TypeError("Imperix export cannot reconstruct supplied instances; use parts factories")
    from .conduction import DiodeConduction
    if DiodeConduction(simulation.system.model).commutating:
        raise ValueError("Imperix hardware adapters do not implement plant commutations")
    if not name or name in (".", "..") or any(char in name for char in "/\\"):
        raise ValueError("Imperix export name must be a single filename stem")
    if target_config is None:
        raise ValueError("Imperix export requires explicit target_config ADC/PWM bindings")
    origin = Path.cwd()
    if isinstance(target_config, (str, Path)):
        path = Path(target_config)
        config = json.loads(path.read_text(encoding="utf-8"))
        origin = path.resolve().parent
    else:
        config = target_config
    if simulation.p.simulation.initial.states or simulation.p.simulation.initial.t != 0.0:
        raise ValueError("Imperix starts cold; simulation restart states cannot be deployed")
    # Retain event write targets, not their schedules or assigned values. Hardware and
    # simulation write the same variables through the existing parameter interface.
    event_targets = tuple(sorted({path for change in simulation.p.changes
                                 for path in change.paths}))
    configurable = bool(variables or event_targets)
    commands = {f"units.{name}.connected": False for name in simulation.system.units}
    params = simulation.p.replace(events={}, **commands)
    variables = tuple(dict.fromkeys((*variables, *event_targets, *commands)))
    prepared, initializations = _prepare_export(simulation, variables, reusable, params=params)
    fixed = set()

    def timing_constants(runtime):
        fixed.update(_timing_parameters(prepared.system.units, runtime))
        explicit = sorted({path for path in fixed for prefix in variables
                           if prefix not in ("all", "common")
                           and (path == prefix or path.startswith(prefix + "."))})
        if explicit:
            raise ValueError("Imperix timing parameters must remain constant: " + ", ".join(explicit))
        return fixed

    generator, model, diagnostics = _parameter_generator(
        prepared, name, variables, initializations, constants=timing_constants if configurable else None)
    if fixed:
        diagnostics.append({
            "code": "hardware-timing-constants", "parameters": tuple(sorted(fixed)),
            "message": "Imperix keeps timing parameters constant: " + ", ".join(sorted(fixed)),
        })
    runtime = generator.runtime
    if missing := set(event_targets) - runtime.by_path.keys():
        raise ValueError("Imperix event target variables must remain tunable: "
                         + ", ".join(sorted(missing)))
    model = model if model is not None else _ModelGenerator(prepared, runtime)
    bindings, adc, clocks, timing, phase, assigned = _bindings(
        prepared, config, model, events=simulation.p.changes)
    period = timing["interrupt_period"]
    unit_ids = [_identifier(key) for key in simulation.system.units]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("Imperix exported names collide after C++ identifier normalization")
    generators = [_ImperixControlGenerator(key, unit, model, bindings[key])
                  for key, unit in prepared.system.units.items()]
    controllers = "\n".join(gen.header() for gen in generators)
    source, probes = _source(generators, adc, clocks, period, phase, runtime,
                            sampling_period=timing["sampling_period"], postscaler=timing["postscaler"])
    header = ("#pragma once\n#include <algorithm>\n#include <array>\n#include <cmath>\n"
              "#include <complex>\n#include <limits>\n#include <string_view>\n"
              "namespace peslite_generated {\nusing Complex = std::complex<double>;\n"
              "constexpr double pi = 3.141592653589793238462643383279502884;\n"
              + _MOVING_WINDOW + "\n" + _CONTROL_MATH + "\n"
              + _parameter_header(runtime) + controllers + "}\n")
    profile = config.get("build", {})
    if not isinstance(profile, dict):
        raise ValueError("imperix.build must be a mapping")
    profile = dict(profile)
    if sdk is not None:
        profile["sdk"] = str(Path(sdk).resolve())
    elif profile.get("sdk"):
        profile["sdk"] = str((origin / profile["sdk"]).resolve())
    diagnostics = tuple(diagnostics) + ({"code": "hardware-timing", "message":
                    "Imperix uses ADC/CPU/PWM hardware timing; ctrl.computation is not a software delay. "
                    "Simulation plant, source, breaker and scheduled events are not deployed. "
                    "Fast overcurrent protection and external interlocks require hardware setup."},)
    loops = tuple(loop.labels for loop in model.model.algebraic_loops)
    if loops:
        diagnostics += ({"code": "algebraic-loop", "loops": loops, "message":
                         "The simulation contains algebraic loops: "
                         + "; ".join(", ".join(labels) for labels in loops)
                         + ". The Imperix ELF includes control code only; the plant equations "
                         "and their algebraic-loop solvers are not exported."},)
    files = {
        "user.cpp": source,
        "user.h": '#pragma once\n#include "extern_user.h"\n#include "Core/core.h"\n'
                  '#include "Core/interrupts.h"\n#include "Driver/peripherals.h"\n'
                  'tUserSafe UserInit();\ntUserSafe UserInterrupt();\nvoid UserError(tErrorSource);\n',
        "peslite_control.hpp": header,
        "imperix-build.json": json.dumps({**profile, "name": name}, indent=2) + "\n",
        "build.py": Path(imperix_build.__file__).read_text(encoding="utf-8"),
        "manifest.json": json.dumps({"format": "imperix", "name": name, "status": "source_only",
                                     "hardware_tested": False, "period": period,
                                     "timing": timing,
                                     "clock_mode": config.get("clock_mode", "auto"),
                                     "clocks": {f"CLK{clock}": {
                                         "frequency": frequency,
                                         "bindings": [path for path, key in assigned.items() if key == clock],
                                     } for clock, frequency in sorted(clocks.items())},
                                     "sampling_phase": phase, "bindings": bindings,
                                     "probes": probes,
                                     "parameters": {item.path: item.member for item in runtime.selected}},
                                    indent=2) + "\n",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (out_dir / filename).write_text(content, encoding="utf-8")
    paths = tuple(out_dir / filename for filename in files)
    if build:
        artifact = imperix_build.build_project(out_dir)
        paths += (artifact, out_dir / "build-command.json", out_dir / "build.log")
    return ExportResult("imperix", out_dir, paths, diagnostics)
