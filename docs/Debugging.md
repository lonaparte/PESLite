# Debugging

The `debug` addon provides `peslite debug`, which runs a temporary control-graph overlay without
changing the simulation file. One plain-text `.pesdbg` file defines the overlay, observations and
checks.

## Run a plan

```bash
peslite debug case.pes --plan control.pesdbg
peslite debug case.pes CurrentLoop.bandwidth=250.0
peslite debug case.pes --probe command=vsc.CurrentLoop.u_dq
```

Each run creates `output/<case>/debug/YYMMDDTHHMM-NNN/` containing the effective
`debug.pesdbg` and `report.json`. With no `--plan`, PESLite
inherits the latest run for that case; use `--fresh` to start with an empty overlay, pass an ID
printed by `debug ls` to `--from`, or list the history with:

```bash
peslite debug ls case.pes
peslite debug rm case.pes 261120T1030-000
peslite debug rm case.pes --all
```

`debug ls` prints the run IDs accepted by `debug rm`; for example, `261120T1030-000` names the
directory `output/<case>/debug/261120T1030-000/`. `rm` removes that complete directory, while
`--all` removes every run directory for the case. Both commands are limited to `debug/` under the
current working directory, retain the `debug/` directory itself and do not remove ordinary results.

## Plan syntax

The preamble selects the model and the state policy for a bypassed loop. `continue` keeps its
state evolution, `freeze` holds it, and `tracking` follows its inputs when the loop supplies a
tracking law.

```yaml
model: case.pes
state_policy: continue

@debug
Test: sine
  amplitude: 0.02
  frequency: 500.0
  offset: 0.1

@vsc
- CurrentLoop.u_dq -> ThreePhaseModulation.u_dq
+ Test.out -> ThreePhaseModulation.u_dq
~ CurrentLoop.bandwidth = 250.0

@probe
original: vsc.CurrentLoop.u_dq
injected: vsc.Test.out
effective: vsc.ThreePhaseModulation.u_dq

@analysis
response:
  type: frequency_response
  input: injected.re
  output: effective.re
  frequency: 500.0

@checks
command_limit:
  signal: effective.abs
  metric: max
  operator: <=
  value: 0.2
```

A Unit view such as `@vsc` uses the same loop, port and `Name: Type` declaration syntax as a
simulation-file view. A disconnect must match an existing connection, a connect requires a free
compatible input, and all replacements pass the ordinary typed-port validation before assembly.
Parameter paths may be complete or an unambiguous suffix; the CLI also accepts
`CurrentLoop.bandwidth=250.0` directly.

Source types are `constant`, `step`, `ramp`, `sine`, `pulse`, `chirp`, `prbs`, `multitone`, `table`
and `csv`. Complex probes expose `.re`, `.im` and `.abs`; scalar probes retain their alias. Prefix
a simulation-state path with `state.` to observe an internal controller state, for example
`current_integral: state.vsc.ctrl.CurrentLoop.integral_pu`.

Analyses reuse the standard `small_signal`, `frequency_response`, `impedance`, `loop_gain` and
`dynamic` functions. Checks support `final`, `min`, `max`, `mean`, `rms` and `max_abs` with the
operators `<`, `<=`, `>`, `>=`, `==` and `!=`.

## Graphical interface

The graphical interface opens `.pesdbg` as its authoritative source, validates it against the
referenced `.pes` file and overlays added and removed control connections on the model view. Running
the file invokes the same `peslite debug` command; previous runs and their JSON reports remain
available in the workspace file tree.

## Report

`report.json` records the PESLite version, model hash, effective configuration, termination status,
probe samples, analyses, metrics, checks, evidence and diagnostics. A passed check set exits with
code `0`; a failed check set exits with code `6`. Configuration, assembly, simulation, analysis and
check failures use the same report structure and identify their stage. Successful execution confirms
the requested computation, not the physical validity of the model.
