# Semiconductor Loss Estimates

The optional `peslite` add-on `peslite.addons.functions.semiconductor_losses` provides the
**Level 1 circuit heat/loss analysis** for [Issue #40](https://github.com/lonaparte/PESLite/issues/40).
It operates on existing electrical simulation traces and does **not** create a second
network, change solver equations, allocate thermal states, or require `peslite-full`.
Core-only simulations continue to work without this add-on.

## Loss contract

Use `LossTable` for measured conduction voltage in **V** or switching energy in **J**.
Axes must be named `current_A`, `voltage_V`, or `temperature_K` and contain strictly
increasing SI points. Multilinear interpolation applies inside the grid. The default
out-of-range policy is **error**; `out_of_range="clamp"` explicitly uses endpoint values.
No unconstrained extrapolation is performed, and all table values must be nonnegative.
Current and voltage lookups use magnitudes.

`SemiconductorLossModel` supports four independent quantities:

| Field | Meaning | Units |
|---|---|---|
| `conduction_voltage_V` | On-state voltage drop, including MOSFET/IGBT/diode characteristics | V |
| `turn_on_energy_J` | Energy dissipated on an independently confirmed turn-on | J |
| `turn_off_energy_J` | Energy dissipated on an independently confirmed turn-off | J |
| `reverse_recovery_energy_J` | Additional diode reverse-recovery energy | J |

Each field accepts a nonnegative scalar, a `LossTable`, or `None`. For instance,
manufacturer tabulations may depend on current, DC-link voltage and junction temperature.
A table can be assembled from manufacturer-provided arrays after converting all units to SI;
there is no particular CSV layout requirement. Missing data is **not** a claim that a
physical loss mechanism is zero. Reported totals are losses *attributed by the supplied
model and observations*, not automatically comprehensive device loss certification.

## Example: compare devices under identical circuit conditions

```python
import numpy as np
from peslite.addons.functions import compare_devices, circuit_heat_report
from peslite.addons.functions.semiconductor_losses import (
    LossTable, SemiconductorLossModel, SwitchingEvent,
)

# Example circuit operating trace (use result.t and result.plant for actual runs).
t = np.linspace(0.0, 0.01, 101)
i_device = np.full_like(t, 10.0)
# These events must be confirmed by the simulation or an independent event log.
committed_events = [
    SwitchingEvent(0.002, "turn_on", current_A=10.0, voltage_V=400.0),
    SwitchingEvent(0.007, "turn_off", current_A=10.0, voltage_V=400.0),
]

candidates = {
    "MOSFET A": SemiconductorLossModel(
        conduction_voltage_V=LossTable(
            {"current_A": [0.0, 20.0], "temperature_K": [300.0, 400.0]},
            [[0.0, 0.0], [0.4, 0.6]],
        ),
        turn_on_energy_J=0.001,
        turn_off_energy_J=0.0015,
    ),
    "MOSFET B": SemiconductorLossModel(
        conduction_voltage_V=0.25,
        turn_on_energy_J=0.0012,
        turn_off_energy_J=0.001,
    ),
}

comparison = compare_devices(
    t, i_device, candidates, temperature_K=350.0,
    events=committed_events,
)
for name, report in comparison.items():
    print(name, report.to_dict()["attributed_heat_W"])
print(circuit_heat_report({"chosen": comparison["MOSFET A"]})["totals"])
```

For an actual PESLite run, enable `simulation.output.signals: 1`, inspect the recorded
columns, and pass the verified **device** current column to
`recorded_device_loss(result, current_column, model, ...)`. Optional
`voltage_column` and `conduction_fraction_column` map other observed signals.
Line or inductor current is not automatically semiconductor current for every bridge
and topology. Use the [Output and Results](Output-and-Results.md) guide to inspect columns.

The lower-level `device_loss_report(time_s, current_A, model, ...)` also accepts
`voltage_V` arrays, explicit `conduction_fraction` between zero and one, and
`accounted_conduction_power_W` when a resistive/conduction loss is **already**
represented in the electrical equations. Supplying that accounted power **together
with** a conduction-voltage curve is rejected to prevent double counting.

## What gets reported

`DeviceLossReport.instantaneous_conduction_W` contains the sampled continuous
conduction power. `event_losses` contains each transition's separate impulse in **J**.
`to_dict()` includes accounted vs estimated conduction, turn-on/turn-off/recovery
energy, average power over the trace duration, attributed heat and event count.
Continuous energy is integrated by the trapezoidal rule on the supplied observation
grid. For precise circuit-wide *electrically realized* losses, use the accepted-step
`energy_closure(result)` output; the CSV sampling resolution is not equivalent.

`circuit_heat_report({"device_name": report, ...}, energy_closure=closure)` adds device
attributions and returns the electrically realized circuit dissipation and
additional semiconductor estimates **in separate fields**. It deliberately does
not sum them into an unqualified physical circuit total: some electrically realized
losses can overlap with estimates. The user must check which mechanisms each input
covers before combining budgets.

## Switching, averaging and current limitations

- Exact switching energy requires externally supplied *committed* `SwitchingEvent`
  records. Neither sparse waveform snapshots nor rejected solver steps provide
  reliable transition counts. This phase does **not** introduce a simulator event
  hook or infer events from sampled gate commands.
- The `turn_on_includes_reverse_recovery=True` flag rejects separate
  reverse-recovery events, avoiding a common datasheet double count.
- Averaged/PWM-averaged runs can use a documented `conduction_fraction` estimate,
  but do not invent switching events. If the trace does not represent actual
  semiconductor currents or RMS effects, precise semiconductor losses are
  **unsupported** until the electrical model supplies that information.
- `temperature_K` is a **fixed loss-evaluation temperature**, not a predicted
  thermal state. Junction/case RC transients, loss-temperature feedback, shared
  heatsinks and magnetic/thermal coupling are subsequent phases of Issue #40.
- This post-processing is Python-only; no silent change is made to standalone
  C++ electrical energy accounting or `.pes` configurations.

The staged architecture and division between standard `peslite` and advanced
`peslite-full` integrations are recorded in the
[Issue #40 implementation plan](https://github.com/lonaparte/PESLite/issues/40#issuecomment-6911919041).
