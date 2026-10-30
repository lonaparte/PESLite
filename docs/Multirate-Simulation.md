# Multirate Simulation

Multirate integration lets named subsystems advance on a step different from the system step. It is
useful when one part of a connected model is much faster or slower than the rest and a single global
step would waste evaluations.

Multirate schedules currently require the fixed-step solver.

## Step definition

`simulation.solver.dt` is the maximum system step. Entries under
`simulation.solver.subsystems` are ratios relative to `dt`:

- `0.1` means `dt / 10`: ten subsystem steps per system step;
- `1` means the normal system step;
- `10` means `10 * dt`: one subsystem update over a ten-step window.

A ratio below one must be `1/N`; a ratio above one must be an integer `N`. All entries below one
form one fast group and all entries above one form one windowed group; each group uses one ratio,
and building the simulation rejects different ratios in one group. The mapping key is the
assembled subsystem name, such as `pcc`, `grid.branch` or `vsc.dclink`.

```yaml
simulation:
  solver:
    type: fixed
    method: rk4
    dt: 25e-6
    subsystems:
      pcc: 0.1
      vsc.dclink: 10
    sweeps: 2
```

Here the bus is treated as the fast group and the DC link as the slow, windowed group. Unlisted
subsystems remain on the system step.

## Fast subsystems

A subsystem with ratio `1/N` is sub-stepped inside each system interval. It shares the system's
fixed integration method. `sweeps` controls coupling sweeps between the fast group and the rest of
the system; increasing it can reduce split error but increases RHS evaluations.

Fast sub-stepping helps only when it replaces an otherwise smaller global `dt`. Assigning many
strongly coupled subsystems to the fast group may cost more than a single-rate run.

## Slow subsystems and windows

A subsystem with ratio `N` advances over a window of `N * dt`. During that window, the rest of the
system sees extrapolated slow states, while the slow subsystem receives window-averaged interface
inputs. All windowed subsystems share one window method; an entry may specify it explicitly:

```yaml
subsystems:
  vsc.dclink: {step: 10, method: rk4}
```

Windowing is most effective for storage dynamics whose state changes little over one window. Large
windows across strong coupling can reduce accuracy or erase dynamics that matter to the study.

## Events and exact boundaries

Event, output and end times remain exact integration boundaries; they are not rounded to a
multirate, controller or PWM grid. An event inside a slow window is handled as described in
[Exact event semantics](Events-Timeline.md#exact-event-semantics).

## Complete example

This case puts the DC link on a ten-system-step window while the remaining system uses RK4 with a
maximum step of 25 microseconds:

```yaml
base: {s_base: 2.0e6, v_ll_rms: 690.0, f0: 50.0}

buses:
  pcc: {c_pu: 0.02, r_d_pu: 0.5}

sources:
  grid:
    bus: pcc
    x_pu: 0.4
    r_pu: 0.04

units:
  vsc:
    bus: pcc
    ac_filter: {l_f_pu: 0.2, r_f_pu: 0.01}
    dclink:
      vdc_ref: 1500.0
      source: {type: voltage}
    pwm: {f_sw: 20000.0, modulation_limit: 0.95}
    ctrl:
      type: gfm
      loops:
        PSC: {k_p_pu: 6.2832, k_vi: 20.0}
        VirtualImpedance: {}
        ActiveDamping: {r_a_pu: 0.2, alpha_d: 40.0}
      references: {p_ref_pu: 0.5}
    protection: {overcurrent: {enable: 1, limit_pu: 1.5}}

events:
  connect_vsc: {type: connect, target: vsc, t: 0.2, ramp: 0.2}
  p_step: {type: set, t: 0.6, set: {units.vsc.ctrl.references.p_ref_pu: 0.7}}

simulation:
  t_end: 1.0
  solver:
    type: fixed
    method: rk4
    dt: 25e-6
    subsystems:
      vsc.dclink: 10
    sweeps: 2
    linearisations: 3
  output:
    period: 0.5e-3
    record_every: 10
```

Save it as `multirate.pes` and run:

```bash
peslite multirate.pes
```

Do not add `--averaging` to this command unless the solver is deliberately overridden: that preset
selects adaptive DP45, while subsystem schedules require a fixed solver.

## Accuracy and speed workflow

Always compare a proposed split with a single-rate baseline:

1. run with `subsystems: {}` at a converged `dt`;
2. add one fast or slow group;
3. compare important states at common output times and inspect the summary;
4. reduce the ratio or increase `sweeps` if the split error is too large;
5. measure wall time only after accuracy is acceptable.

`linearisations` controls points used for the post-run split-error bound; `0` disables that bound.
The multirate solver also reports interface and window indicators in the run summary. Treat these as
diagnostics, not as a replacement for convergence against a tighter reference run.

## Limitations

- Multirate schedules are supported only with `simulation.solver.type: fixed`.
- A fast group uses the system fixed method; a windowed group uses one method.
- A system with commutating subsystems, such as diode rectifiers or switch-controlled capacitors,
  cannot use a windowed group because it cannot locate their commutation instants.
- The filter of an averaged unilateral half bridge requests a state-dependent local step in DCM
  ([DC Converters and Wireless Power Transfer](DC-Converters-and-WPT.md#ccm-and-dcm))
  and therefore cannot be windowed.
- C++ export does not support multirate subsystem schedules
  ([C++ Export](Cpp-Export.md#current-limitations)).

For bridge-dependent periods and event boundaries, see
[Bridge and PWM](Bridge-and-PWM.md).
