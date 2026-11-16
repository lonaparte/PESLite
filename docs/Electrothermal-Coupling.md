# Electrothermal Coupling

The standard `peslite` add-on integrates heat dynamics into the **existing**
`Model`, numerical solvers, result columns and restartable state registry.
`peslite-core` remains unchanged; there is no thermal-only solver or
duplicated electrical topology.

## Unit conventions and supported networks

- `LumpedThermalNetwork`: physical thermal nodes in kelvin (K), heat capacity
  in joules per kelvin (J/K), resistance in kelvin per watt (K/W), continuous
  heat power in watts (W), and committed heat impulses in joules (J).
  Arbitrary node links support **shared heat sinks** and cooling to a fixed
  ambient. `LumpedThermalNetwork.cauer(C, R)` produces a serial
  junction-to-case-to-ambient ladder; the last resistance is to ambient.
- `FosterImpedance`: parallel transient-thermal-impedance branches relative
  to a **fixed** ambient. It provides junction temperature only. Foster
  branch states are *not* physically composable case or heat-sink nodes;
  Cauer is required for shared cooling or physical heat balance.
- `ElectroThermalModel`: a normal PESLite continuous subsystem with a
  `state`/`input`/`output` interface and automatic registration in
  `Model.state_labels()` and `states.csv`. Its separate thermal-energy
  bookkeeping does not silently modify the electrical port-Hamiltonian audit.

Thermal dynamics use the same numerical integration steps as electrical
dynamics. Results can be compared against the analytical first-order
response `T(t) = T_amb + P R (1 - exp(-t/(R C)))`.

## Attach a Cauer junction/case network to an existing circuit

For example, supplement the [open-loop Buck](https://github.com/lonaparte/PESLite/blob/main/examples/buck-open-loop-example.pes)
configuration with the following element:

```yaml
elements:
  load:
    type: resistor
    bus: output
    r: 2.0

  thermal_load:
    type: electrothermal
    heat_source: load.power
    c_j_J_K: 0.2
    r_jc_K_W: 1.5
    c_c_J_K: 1.0
    r_ca_K_W: 3.0
    ambient_K: 300.0
    initial_K: 300.0
```

`heat_source` must identify an **already existing scalar physical output**.
This example observes the electrical dissipation of the same `load`
element; it does not create another electrical resistor or subtract the
same energy twice. The additional thermal subsystem publishes
`thermal_load.temperature_K`, `thermal_load.case_K`,
`thermal_load.heat_W` and `thermal_load.cooling_W`. Its stored
temperature states and last committed transition ID are restartable.
Enable `simulation.output.signals: 1` for the public outputs.

For an electrical circuit modeled as ideal, choose instead
`current_source` and a nonnegative `conduction_voltage_V` **without**
`heat_source`. This is explicitly **estimated** conduction heat. The
two alternative modes are mutually exclusive because their conduction
losses may overlap.

You can construct a richer loss provider programmatically with
`SemiconductorLossModel` and `LossTable` from
`peslite.addons.functions.semiconductor_losses` and pass it to
`ElectroThermalModel`. Junction temperature feeds back into these
tables automatically. A reference temperature is not a substitute for
a solved thermal state.

## Temperature feedback into actual electrical equations

`thermal_resistor` implements a memoryless scalar electrical resistance whose actual current is
`u / R(T)` and whose electrically dissipated power is `R(T) * i²`. It does not add an artificial
electrical state. This is different from estimating heat on an ideal
electrical switch. Connect its temperature input to the thermal
element's physical output:

```yaml
elements:
  load:
    type: thermal_resistor
    bus: output
    temperature_source: thermal_load.temperature_K
    r_reference_Ohm: 2.0
    alpha_per_K: 0.004
    reference_K: 300.0

  thermal_load:
    type: electrothermal
    heat_source: load.heat_W
    c_j_J_K: 0.2
    r_jc_K_W: 1.5
    c_c_J_K: 1.0
    r_ca_K_W: 3.0
    ambient_K: 300.0
    initial_K: 300.0
```

Both components use the one acausal circuit and the existing Model
connection plan. A nonpositive resistance at a simulated temperature
is rejected. Temperature-dependent behavior is **not** automatically
applied to arbitrary built-in switch models, unobserved semiconductor
devices or imported black boxes. Unsupported combinations must be
identified explicitly; Python-specific thermal callbacks are not
silently approximated for standalone C++ generation.

## Confirmed switching impulses

```python
from peslite.addons.functions.semiconductor_losses import (
    SemiconductorLossModel, SwitchingEvent,
)
from peslite.addons.components.electrothermal import ElectroThermalModel
from peslite.addons.components.thermal_network import LumpedThermalNetwork

network = LumpedThermalNetwork.cauer(
    [0.2, 1.0], [1.5, 3.0], ambient_K=300.0
)
component = ElectroThermalModel(
    network,
    loss_model=SemiconductorLossModel(
        turn_on_energy_J=0.001, turn_off_energy_J=0.0015
    ),
)
event = SwitchingEvent(0.002, "turn_on", current_A=10, voltage_V=400)
heat_J = component.apply_committed_transition(event, event_id=1)
```

Only use a **confirmed** device transition (not a proposed RHS step,
rejected integrator step or coarse sampled PWM edge). Each per-device
`event_id` must be monotonically increasing. The state
`last_transition_id` is saved as a discrete state alongside the temperature states, so continuation
rejects a repeated impulse without adding another differential state. After applying a committed event at a
simulation boundary, repack the same model vector with
`model.get_initial_values()` before advancing. This explicit
observer interface does not imply that every bridge already exposes
a semiconductor-specific event-current/voltage stream.

`turn_on_includes_reverse_recovery` prevents double-counting overlapping
datasheet definitions.

## Reports and device selection

`peslite.addons.functions.thermal_analysis.thermal_report(result, name,
maximum_junction_K=...)` reads a completed Cauer run and reports
peak/final junction and case temperature, peak time, thermal headroom,
continuous heat and cooling energy. An optional *explicit* delivered
output power enables an estimated efficiency metric.

Use `compare_thermal_reports` after running candidates under the same
converter, heat provider, ambient, initial conditions and observation
window. Candidate ranking alone cannot establish identical operating
conditions.

## Numerical and energy limitations

- Thermal storage `sum(C_i * (T_i - T_ambient))` plus ambient cooling
  satisfies `dQ/dt = sum(P_heat) - P_cooling` for constant ambient.
  Committed impulses update `Q` by exactly their energy (J).
- The electrical energy closure remains the source of truth for
  electrically realized dissipation. **Estimated** heat is not a
  second electrical sink.
- Validated simulation modes should be run at a converged single
  integration rate first. Core's existing multirate grouping may be
  used only for supported combinations; located commutators retain
  the solver's single-rate restrictions.
- Foster temperature states do not have physical case/heat-sink
  identity. The `thermal_report` function intentionally requires
  a physical Cauer case/cooling result.

Advanced magnetic loss exchange, external field models and reduced-order
thermal coupling are documented in
[Advanced Thermal Coupling](Advanced-Thermal-Coupling.md).
