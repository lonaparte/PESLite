# Thermal Analysis

The standard PESLite installation includes the `thermal` addon. `peslite-full` additionally
includes the `thermalport` addon for external thermal models.

Thermal states use the existing simulation model, solvers, result files and restart mechanism;
there is no separate thermal solver or duplicated electrical network.

## Heat sources

A thermal model can consume either:

- electrically realized dissipation published by an existing component; or
- an explicitly estimated loss that is not already represented in the electrical equations.

Do not count the same mechanism in both groups. Electrically realized dissipation remains part of
the electrical energy balance, while an estimated loss is reported separately.

`SemiconductorLossModel` accepts nonnegative scalar values or `LossTable` data for:

| Field | Quantity |
|---|---|
| `conduction_voltage_V` | on-state voltage drop |
| `turn_on_energy_J` | confirmed turn-on energy |
| `turn_off_energy_J` | confirmed turn-off energy |
| `reverse_recovery_energy_J` | confirmed reverse-recovery energy |

Table axes may be `current_A`, `voltage_V` and `temperature_K`. Points must be strictly increasing
and use SI units. Values are interpolated inside the supplied range; the default out-of-range policy
is an error, while `out_of_range="clamp"` uses the nearest endpoint.

Use `recorded_device_loss(...)` for a completed PESLite result or `device_loss_report(...)` for
explicit arrays. The current must represent the semiconductor current being evaluated; a line or
inductor current is not automatically a device current for every topology. `compare_devices(...)`
compares several loss models under the same operating trace, and `circuit_heat_report(...)` keeps
realized circuit dissipation separate from additional estimates.

Exact switching losses require confirmed `SwitchingEvent` records. Rejected solver stages and
sparsely recorded gate commands are not switching events. `turn_on_includes_reverse_recovery=True`
prevents the same recovery energy from being counted again.

## Thermal networks

Thermal quantities use kelvin (K), watts (W), joules (J), joules per kelvin (J/K) and kelvin per
watt (K/W).

- `LumpedThermalNetwork` represents physical capacitive nodes and conductive links. Its Cauer helper
  builds a junction-to-case-to-ambient ladder and can participate in shared cooling networks.
- `FosterImpedance` represents parallel transient-impedance branches relative to fixed ambient. Its
  branches do not represent physical case or heat-sink nodes.
- `thermalport` evaluates a stable continuous-time model imported from an external file.
- `shared_thermal_network` assigns several named heat inputs to nodes in one thermal network.

Thermal dynamics use the configured PESLite integrator. A first-order RC network follows

$$
T(t) = T_{\mathrm{ambient}} + PR\left(1-e^{-t/(RC)}\right).
$$

## Temperature simulation

Attach a Cauer network to an existing scalar loss signal:

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

`heat_source` identifies an existing physical output and does not create another electrical load.
The element publishes `temperature_K`, `case_K`, `heat_W` and `cooling_W`. Enable
`simulation.output.signals: 1` to record public signals.

For an ideal electrical model without realized conduction loss, use `current_source` together with
`conduction_voltage_V`. It is an estimated heat source and cannot be combined with `heat_source` on
the same thermal element.

The reduced-order thermal model uses

$$
\dot{\boldsymbol{x}}=\boldsymbol{A}\boldsymbol{x}+\boldsymbol{B}P,
\qquad
T=T_{\mathrm{ambient}}+\boldsymbol{C}\boldsymbol{x}+DP.
$$

Store the model in a `thermalport` file:

```json
{
  "format": "peslite-thermalport",
  "version": 1,
  "units": "SI",
  "heat_names": ["loss"],
  "temperature_names": ["junction"],
  "A_per_s": [[-0.125]],
  "B_K_per_J": [[0.5]],
  "C": [[1.0]],
  "D_K_W": [[0.0]],
  "ambient_K": [300.0],
  "initial": [0.0],
  "signals": {}
}
```

Connect it to an existing heat signal:

```yaml
elements:
  thermal:
    type: thermalport
    sources: {loss: magnet.loss_W}
    file: models/magnet-thermal.json
```

Use `load_thermalport()` to load the same file through the Python API.

For multiple heat inputs or temperature outputs, list their names in `heat_names` and
`temperature_names`; the component publishes `heat_<name>_W` and `temperature_<name>_K`.
The `signals` mapping may add linear outputs such as `cooling_W` without changing the thermal
state interface.

`shared_thermal_network` maps every configured source path to a named thermal node:

```yaml
elements:
  cooling:
    type: shared_thermal_network
    capacities_J_K: {inverter: 0.2, winding: 0.5, sink: 10.0}
    links:
      - {node: inverter, other: sink, resistance_K_W: 0.5}
      - {node: winding, other: sink, resistance_K_W: 1.2}
      - {node: sink, other: null, resistance_K_W: 4.0}
    sources:
      electric: {path: load.power, node: inverter}
      magnetic: {path: magnet.loss_W, node: winding}
    ambient_K: 300.0
    initial_K: 300.0
```

It publishes `temperature_<node>_K`, total incoming heat and heat rejected to ambient.

## Confirmed heat impulses

`ElectroThermalModel.apply_committed_transition(...)` applies a confirmed switching-energy impulse
at an accepted simulation boundary. The nonnegative event ID must increase monotonically. It is
stored as a discrete restart state, preventing continuation from applying the same event twice
without adding a differential state.

## Reports and limits

`thermal_report(result, name, maximum_junction_K=...)` reports peak and final temperature, thermal
headroom, continuous heat and cooling energy for a completed Cauer run. Use
`compare_thermal_reports(...)` only for candidates evaluated under the same circuit, ambient,
initial conditions and observation interval.

- Foster states cannot be reported as physical case or heat-sink temperatures.
- Averaged models require a justified conduction fraction and do not provide exact switching events.
- Imported reduced-order models contain only the declared temperature dynamics; they do not imply a
  field reconstruction or a general thermal PDE solver.
- Python callbacks that are unsupported by standalone export are rejected rather than approximated.

Connections between thermal, electrical and magnetic models are described in
[Multiphysics Analysis](Multiphysics-Analysis.md).
