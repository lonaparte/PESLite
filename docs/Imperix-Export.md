# Imperix ELF Export (Experimental)

> Experimental: Interfaces and generated output may change. Host-side tests do not establish
> SDK or hardware acceptance; validate the generated code with the target SDK and hardware before use.

The `imperix` backend exports an ELF executable for Imperix hardware using the official CPP SDK.
The executable runs the configured sampled control logic, not the simulated power circuit.
Source-only mode generates the files needed to build the ELF later.

The exporter reuses the existing control graph, loop equations, PI/filter state, output constraints,
startup and sampled-protection generation. The SDK supplies ADC, CPU interrupts, clocks and PWM.
Neither the plant integrator nor a second set of controller equations runs on the hardware.
Custom loops use the existing equation/state contract. Replaced controller execution callbacks
without a lowering adapter are rejected rather than replaced by the built-in controller.
`set_logging()` and `summary()` overrides do not restrict ELF export because they are not exported.

Algebraic loops detected in the assembled simulation are reported as export warnings, including
the participating computations. The plant equations and their algebraic-loop solvers are not
included in the ELF; these warnings do not imply that an iterative solver runs on the hardware.

## Export and build

Place `case.imperix.json` next to `case.pes`, then export:

```bash
peslite-convert case.pes all --imperix
```

Without `--imperix-config`, the CLI reads the resolved PES file's same-directory sidecar:
`case.pes` uses `case.imperix.json`. The configuration is not read from the export output
directory. A missing configuration file stops export with a file-not-found error, including
in source-only mode; no hardware bindings are inferred or created automatically.

This writes to `export/case/` and runs the configured SDK build. An explicit source-only export is:

```bash
peslite-convert case.pes all --imperix --imperix-source-only
peslite case.pes --export imperix --imperix-config bindings.json --imperix-source-only
```

The generated package contains `user.cpp`, `user.h`, `peslite_control.hpp`, `manifest.json`,
`imperix-build.json` and a standalone `build.py`. It does not require PESLite on the SDK machine:

```bash
python build.py
```

A successful SDK build adds `<name>.elf`, `build-command.json` and `build.log`. Missing SDK,
compiler/linker errors, missing fresh output or a host/stripped ELF are errors, never reported as
successful ELF export. Failed builds preserve an older ELF but mark the manifest
`build_failed` with no current artifact. Existing Cockpit project/acceptance files are not deleted.
Create the Cockpit project from the target ELF; PESLite does not invent a proprietary `.ixproj`.

## Explicit hardware bindings

Copy [template.imperix.json](template.imperix.json) as a starting point for a single three-phase
Unit named `vsc`. Rename it to match the PES file or select it with `--imperix-config`. Replace
the Unit name, channel assignments, calibration, dead time and SDK build placeholders for the
actual setup. The template's unit sensitivities are placeholders, not sensor calibration.
The build wrapper path is also a placeholder for the user's SDK project integration, not a
command supplied by the SDK; it must accept the shown SDK/source/output arguments and produce
the requested ELF using the installed SDK. `elf_machine` must match that target architecture.
The `build` mapping may be omitted when only generating sources.

The template deliberately leaves `hardware_overcurrent_configured` false. If fast overcurrent
protection is enabled in the PES file, export is rejected until the operator has configured
hardware protection and explicitly confirms it by setting this flag to true.

Example bindings for a three-phase VSC follow. These channel numbers and sensitivities illustrate
syntax, not a validated wiring diagram or sensor calibration:

```json
{
  "sampling_phase": 0.5,
  "units": {
    "vsc": {
      "adc": {
        "u_g": [
          {"channel": 0, "sensitivity": 0.01},
          {"channel": 1, "sensitivity": 0.01},
          {"channel": 2, "sensitivity": 0.01}
        ],
        "i_c": [
          {"channel": 3, "sensitivity": 0.1},
          {"channel": 4, "sensitivity": 0.1},
          {"channel": 5, "sensitivity": 0.1}
        ],
        "u_dc": {"channel": 6, "sensitivity": 0.005}
      },
      "pwm": [0, 1, 2],
      "dead_time": 0.000001,
      "startup_ramp": 1.0,
      "hardware_overcurrent_configured": true
    }
  }
}
```

`units` binds every assembled leaf Unit by its existing name, including names such as
`charger.primary`. Complex channels require three ADC bindings in a/b/c order; scalar channels
require one. VSC channels are `u_g`, `i_c`, `u_dc`; scalar converters use `current`, `voltage`,
`vin` and, when requested by their graph, signed port `power` in watts. A power binding must
represent that physical power signal; a voltage/current product or remote measurement is not
silently inferred. `meas.sources` names describe simulation observations, not physical wiring.

ADC bindings accept `channel`, optional `device` (default zero), positive `sensitivity`, finite
`offset` and `sign` (`+1` or `-1`). An integer binding uses sensitivity 1 and offset 0. SDK values
are `analog_input / sensitivity + offset`, followed by `sign`. They must be SI values, not pu;
the normal controller boundary performs its own scaling. Compatible ADC bindings may share an
input with different logical signs; each reader applies its own sign after the shared sampling
and calibration. Conflicting calibration/averaging is rejected. PWM device/channel pairs cannot
be shared by different bridge legs. Physical channel ranges must match the installed controller and SDK.

`pwm` follows the bridge's existing duty-register order: three entries for a three-phase bridge,
two for a full bridge and one for a half bridge. Fully controlled legs use complementary outputs.
A unilateral half bridge uses `PWMH_ACTIVE`; wire that active output to its controlled transistor,
including the lower transistor in a Boost circuit. An unused complementary lane is not a diode.
Complementary legs require an explicit positive `dead_time`, shorter than half the carrier period.
A single controlled transistor does not use complementary dead time and may omit it.

## Timing and run commands

Timing is mapped at export from the existing PES periods; no second set of Unit or loop periods
is needed in the binding file. The CPU interrupt period is the greatest common divisor of the
Unit control periods. For example, Units at 50 us and 75 us use a 25 us interrupt, rather than
requiring a manually selected common period. Optional `interrupt_period` overrides this choice
and must still divide every Unit period. Loop subrates retain the normal graph periods.

The ADC period is a common divisor of that interrupt, the requested `meas.period` values
(defaulting to each Unit's control period), and any averaging windows. The mandatory CLK0 drives
this sampling grid. The SDK's `ConfigureMainInterrupt` postscaler decimates it to the CPU rate:
an ADC period of 12.5 us with a 50 us controller uses 80 kHz CLK0 and postscaler 4, not an 80 kHz
CPU interrupt. All channels physically sample on this common grid; a slower logical channel
does not acquire an independent physical clock. Instantaneous controller inputs use the current
sample. The CPU reads all bound inputs before controller updates and writes PWM duties afterward.
With the default `clock_mode: "auto"`, equal PWM carrier frequencies share a hardware clock,
including CLK0 when they match the physical ADC rate. This sharing is the same for constant,
`all` and `common` exports. Different carrier phases remain per-output settings and do not require
different clocks. Up to four distinct clocks,
including CLK0, are available for sampling and PWM carrier frequencies.
`manifest.json` records the mapped `sampling_period`, `interrupt_period` and `postscaler` in `timing`.
The SDK may round clock settings to its hardware resolution. Select representable periods and
verify the actual rates: controller coefficients and the modulation time base use configured periods.

For manual assignment, set `clock_mode` to `"manual"` and add `clocks`. Keys are the existing
full parameter paths; values are `"CLK0"`, `"CLK1"`, `"CLK2"` or `"CLK3"`. No second set of
frequencies or periods is defined in the binding file. For example, these top-level fields bind
a Unit whose PWM, ADC, control and voltage-loop periods are explicitly configured in the PES file:

```json
{
  "clock_mode": "manual",
  "clocks": {
    "units.vsc.pwm.f_sw": "CLK1",
    "units.vsc.meas.period": "CLK0",
    "units.vsc.ctrl.period": "CLK0",
    "units.vsc.ctrl.loops.DCVoltageControl.period": "CLK0"
  }
}
```

This is a configuration fragment; the normal `units` channel bindings remain required.
Manual mode requires a binding for every explicitly configured scheduling period, even when two
periods have equal values. Every Unit's `pwm.f_sw` needs a binding. If `interrupt_period` is
explicitly configured in the sidecar, bind that exact key to `"CLK0"` as well. Omitted or null
control/loop periods retain the existing default-following behavior and do not need entries;
an unspecified `meas.period` likewise follows the Unit control period. Optional entries for
these defaulted fields are accepted. Windows, computation delays, dead times and carrier phases
are durations or offsets, not separate clocks, so they are not `clocks` keys. Nested Units use
their normal paths, such as `units.charger.primary.pwm.f_sw`.

ADC and CPU schedules must bind to CLK0. Unit and loop periods are implemented by the existing
CPU postscaler and software dividers, not additional hardware clocks. A control period that
defaults to the PWM period follows its duration, not its hardware clock ID: a PWM on CLK1 still
has its control task executed from CLK0. All ADC channels keep the common physical sampling grid.
PWM may bind to any CLK, but a clock can have only one frequency. In particular, a PWM on CLK0
must match the physical ADC rate, not merely a slower CPU or logical sample rate. Same-frequency
PWMs can be deliberately bound to different clocks in manual mode. Missing bindings, unknown
paths, invalid IDs, incompatible frequencies and ADC/CPU assignments outside CLK0 fail before
files are written. Nonempty `clocks` bindings require manual mode; they are never silently ignored.

Both modes use the same timing and code-generation path. The manifest records `clock_mode` and
a resolved `clocks` table containing each used clock's frequency and full-path bindings,
including defaults. Assignment happens only during export; no runtime clock allocator is added.

`sampling_phase` is in `[0, 1)` relative to CLK0. PWM uses the configured carrier phase,
frequency and single/double register-update selection. The hardware adapter uses
`pattern_sync: interval`; complete-pulse software updates require a hardware adapter and are
rejected rather than approximated. Only asynchronous carriers are supported;
an angle-locked simulation carrier needs a separate hardware timing adapter. Here
`pwm.sync: asynchronous` refers to the simulated electrical angle, not independently clocked ADC
channels. Hardware acquisition, execution time and register latency replace numerical timing:
`ctrl.computation` is not reproduced by busy-waiting. Measure execution time and PWM latch margins
on the target.

Instantaneous ADC inputs can sample faster than the CPU. Averaged channels use native synchronous
averaging over one CLK0 period. A longer configured window combines consecutive native window
means through `Adc_ConfigureHistory` / `Adc_GetHistory`, without a second ADC integrator or a
software history buffer. A physical channel is read once per interrupt; identical logical windows
share the calculated mean. The SDK's 64-sample history limit is checked before files are written.
Probes start at zero and control/PWM activation waits until the required history is known to be
valid after startup. Ordinary single-rate channels need no history and retain their existing path.
RMS windows are still rejected: averaged samples cannot reconstruct a mean-square waveform.
Continuous `averaging` controllers and saved simulation initial states are also rejected.
Use a sampled bridge configuration and a cold start.

The hardware adapters bind duty registers only, so phase-shift (non-duty) modulator registers are
rejected. Plants with commutating subsystems, such as diode rectifiers and switch-controlled
capacitors, are rejected because no hardware adapter implements their commutations. An explicit
`ctrl.offset` or `meas.offset` is rejected before files are written: hardware timing comes from
`sampling_phase` on the CLK0 grid, and no adapter places a controller or ADC clock at a
configured position.

Period mapping uses rational common divisors, not a per-tick search or an expanded schedule.
It rejects a common period below the SDK's 4 ns clock resolution. Periods still need to be
representable on the target, and the resulting CPU rate and execution deadline must be checked
on hardware; the compiler does not infer a target's ADC throughput or CPU execution budget.

Every leaf converter exports its `units.<name>.connected` command through the shared parameter
interface, even when no numeric parameters are selected. Its SDK symbol is
`param_units_<name>_connected`, using the normal path normalization. Commands start at zero,
regardless of simulation initial commands; PWM outputs remain deactivated. Set a command to `1`
from Cockpit only after calibration and interlock checks, and to `0` to block that converter.
Nested converters have independent commands, listed in `manifest.json`; no global run alias is
generated. A new run resets the normal loop integrators. Activation waits
until a freshly computed duty word has had a full carrier period plus one interrupt to latch;
rapid stop/start cannot immediately re-enable an old compare word. Duty writes occur only on
controller updates; activation calls occur only on enable transitions. `startup_ramp`
overrides the first configured connect ramp; absent both, it is zero. Simulation event times,
plant/source initial conditions, solver choices and breaker/source actions do not execute on hardware.

Sampled voltage/frequency/DC-link protection uses the existing held criteria and trip latch.
Enabled fast overcurrent protection requires `hardware_overcurrent_configured: true`: this is
the operator's declaration of prior hardware setup, not automatic threshold configuration.
Any generated trip, nonfinite measurement/reference, invalid duty or `UserError`
deactivates all mapped PWM outputs and clears all connection commands. This does not operate external
contactors, guarantee behavior after an unhandled CPU/network failure or replace physical interlocks.
Fault latches require reinitialization; no automatic trip reset is generated.
The hardware ISR uses fixed-size state without heap allocation or C++ exception handling.
Disabled frequency-history monitoring allocates no ROCOF buffer.

## Default interface probes

Every bound ADC interface channel is exposed as a global `volatile float` probe without extra
configuration, independently of tunable-parameter selection. Probes use the same calibrated SI
samples and channel signs as the controller. Existing power channels are included as well.
No extra ADC reads, sampling channels, history buffers or plant calculations are added.

Scalar paths such as `converter.voltage` become `converter_voltage`. Three-phase channels
use the same canonical names as CSV output, with `_a`, `_b` and `_c` suffixes, for example
`vsc_u_g_a`, `vsc_u_g_b`, `vsc_u_g_c` and `vsc_i_c_a`, `vsc_i_c_b`, `vsc_i_c_c`.
Each phase comes directly from its ADC sample, before the complex space-vector conversion, so
unbalanced and zero-sequence components are not discarded. Nested Unit names follow the same
rule, for example `charger.primary.voltage` becomes `charger_primary_voltage`.
`manifest.json` records the exact logical-path-to-symbol mapping under `probes`.
Only path punctuation is normalized for C++ identifiers; no probe-specific prefix or display-name
aliases are added. Collisions with other generated variables or functions are rejected at export.

Probes start at zero, are reset by `UserInit()` and are refreshed on every main interrupt,
including while PWM is stopped and between slower Unit controller updates. Synchronously averaged
channels expose the SDK's averaged samples; these are not automatically oversampled Cockpit probes.
Probe variables are observation copies only and are not read back into control calculations.
Unbound physical quantities and controller-internal signals are not automatically added.

## Tunable parameters and Python API

Parameter selection uses the standalone C++ interface: omitted `variables` keeps constants except
event targets and leaf-unit connection commands, which are always tunable. Individual paths or
sections select additional numeric parameters and leaf-unit connection commands;
`all` selects all available leaves,
and `common` also keeps automatically detected structure-sensitive parameters constant. References
and controller gains use the same interface, without a separate reference whitelist or naming rule.

Imperix additionally keeps hardware timing fixed at export: PWM frequency/carrier phase, control
and loop periods, ADC periods/windows, and the configured computation delay are not tunable.
Explicitly selecting a timing parameter, including through a section, fails with its full path.
`all` and `common` omit timing parameters and report the omitted paths. Dependencies introduced by
custom construction factories are traced through the shared parameter expressions, not a separate
parameter-name blacklist. Standalone C++ parameter selection and `--set` are unchanged.

The exporter uses the official SDK's support for global `float` and `int` variables. Names are the
same `param_<parameter_path>` members used by standalone C++, with punctuation replaced by
underscores. For example, `units.vsc.ctrl.references.p_ref_pu` becomes
`param_units_vsc_ctrl_references_p_ref_pu`. `manifest.json` records the exact path-to-symbol mapping.
Cockpit values use SDK-supported 32-bit types; control calculations retain double precision.
The SDK executable must retain its symbols and debug information.

At the start of an interrupt, changed values are checked using the shared C++ parameter validation
and structural-specialization guards, then construction coefficients are refreshed together.
Controller states are preserved. Unchanged values do not trigger coefficient recomputation.
Invalid updates leave the last accepted configuration intact, stop PWM and return `UNSAFE`.
Derived defaults keep following their source parameters until explicitly changed; their SDK
variables are refreshed to show the effective values after an accepted update.

Simulation events and Cockpit parameter changes are alternative writers to the same target
variables. Every normalized event target is automatically included in the ordinary tunable
parameter selection, even without `all` or an explicit `variables` selection. Repeated events
writing one target share one variable. Unit composition and SI/pu conversion use the existing
event bindings; no event-specific naming table is maintained.

The full target path is the variable name. The SDK symbol uses the same
`param_<path_with_punctuation_replaced_by_underscores>` rule as standalone C++, for example:

| Target variable | SDK symbol | Operation |
|---|---|---|
| `units.vsc.connected` | `param_units_vsc_connected` | `1`: start; `0`: stop |
| `units.vsc.ctrl.references.p_ref_pu` | `param_units_vsc_ctrl_references_p_ref_pu` | Set the power reference |
| `units.vsc.ctrl.loops.CurrentLoop.bandwidth` | `param_units_vsc_ctrl_loops_CurrentLoop_bandwidth` | Retune the current loop |

`manifest.json` lists the actual paths and symbols under `parameters`. There are no
`events.<name>.trigger` variables. Event times and event-assigned values are not executed in the
ELF, including at time zero. Original parameter defaults remain in force until Cockpit writes a
new value; connection commands still start at zero. Writing a target applies that value through
the existing checked parameter update or connection lifecycle. It does not replay an event's
other assignments. Local Python and standalone C++ simulation keep their timed event schedules.

Hardware timing remains fixed. An event target that is not available through the shared parameter
interface, is a hardware-timing dependency, or would be frozen by `common` fails export rather than
being silently ignored. Simulated source and branch switching does not create an external hardware
actuator. Connection commands accept only `0` and `1`.

The ELF contains control code, not the simulated plant or solver. Their selected parameters remain
exposed, but changing a simulated resistor does not change a physical resistor. SDK hardware limits
still apply: controller periods must be whole multiples of the main interrupt period, and
fixed-size history layouts cannot change. All clock frequencies and carrier phases are configured
only by `UserInit()`; parameter updates never reconfigure them. Changing hardware timing requires
a new export and build. Modulation commands produced by control loops remain ordinary outputs;
fixing the carrier timing does not freeze a full bridge's commanded phase shift or duty cycle.

```python
import peslite

project = peslite.export(
    peslite.load("case.pes"), "imperix", "export/case",
    target_config="case.imperix.json",
    variables=["common"],
    build=False,
)
```

## Official SDK build profile

The optional top-level `build` mapping contains `sdk`, `command`, `elf_machine` and `timeout`.
`--imperix-sdk` overrides `sdk`; `PESLITE_IMPERIX_SDK` is a fallback. Relative SDK paths resolve
against the binding file. `command` is an explicit argument list, executed without a shell,
using `{sdk}`, `{source}` and `{elf}` placeholders. It must write its fresh target executable to
`{elf}`. `{source}` is the generated package, containing the user source/header files for the
official SDK project. `timeout` is in (0, 3600] seconds and defaults to 300.

Use the actual compiler, C++17 flags, include paths, target ABI, linker script/specs, startup and
BBOS libraries from the installed official SDK project. The exporter deliberately does not guess
these from a generation number or substitute a host compiler. Set `elf_machine` to the project's
ELF machine (ARM 40 or AArch64 183). A project-specific SDK build wrapper may be used as the
command. Build staging is private and removed after success or failure.

## Validation boundary

Host tests compare generated control duties and loop states against Python controllers and
exercise peripheral calls through SDK test doubles. They are not SDK/linker or physical-hardware
acceptance. A manifest always starts with `hardware_tested: false`; even a valid built ELF needs
target timing, scaling/polarity, duty/phase, hardware protection and stop/fault checks before use.

SDK contracts: [getting started](https://imperix.com/doc/help/getting-started-with-cpp-sdk),
[control task](https://imperix.com/doc/software/config-control-task-configuration),
[ADC](https://imperix.com/doc/software/analog-data-acquisition),
[PWM](https://imperix.com/doc/software/carrier-based-pwm),
[probes](https://imperix.com/doc/software/probe-variable),
[tunable parameters](https://imperix.com/doc/software/tunable-parameter),
[clocks](https://imperix.com/doc/software/clock-generators).
