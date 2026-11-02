# Black-box Models

`BlackBoxSubsystem` connects an array-valued Python callable to the normal model graph without
requiring a model base class or file format. A registered element places it through the ordinary
`subsystems()`, `connections()` and `port_bindings()` interfaces.

## Stateless models

A stateless model declares named inputs and outputs. Its callable receives time and the ordered
input vector:

```python
from peslite import BlackBoxSubsystem

model = BlackBoxSubsystem(
    lambda t, inputs: external_model(inputs),
    inputs=("voltage",),
    outputs=("current",),
)
```

Input and output mappings may provide complex initial values when required:

```python
model = BlackBoxSubsystem(
    output=lambda t, inputs: external_model(inputs),
    inputs={"voltage": 0j},
    outputs={"current": 0j},
)
```

## Stateful models

Supplying initial `state` values registers them as PESLite states. A continuous model provides
`rhs`, from which PESLite obtains the state derivative:

```python
model = BlackBoxSubsystem(
    output=lambda t, state, inputs: external_output(state, inputs),
    rhs=lambda t, state, inputs: external_rhs(state, inputs),
    inputs=("voltage",),
    outputs=("current",),
    state={"filter": 0.0},
    feedthrough=(),
)
```

A state vector may be supplied directly and receives the stable names `state_0`, `state_1`, and so
on. Dynamic memory, including filter histories and recurrent hidden values, must be registered state
rather than mutable data retained by the callable.

## Discrete state updates

A discrete model replaces `rhs` with `update` and a sampling `period`:

```python
model = BlackBoxSubsystem(
    output=lambda t, state, inputs: state,
    update=lambda t, state, inputs: recurrent_model(state, inputs),
    period=100e-6,
    inputs=("signal",),
    outputs=("estimate",),
    state={"hidden": 0.0},
    feedthrough=(),
)
```

`offset`, in [0, `period`), selects the update grid position. At one instant, all due units and
clocked models sample before any commits; restored discrete states resume after the restart instant.
The update period joins post-run loop-map schedules, where sampled floating-point state remains a
smooth coordinate rather than a lifecycle mode.

## Energy declaration

Use `observer=True` for a non-loading signal model. A loading model may use the ordinary energy
members `ports`, `storage`, `dissipated_power` and `supplied_power`; without a declaration, the
energy check applies its default black-box accounting.

```python
from peslite.solver.energy import Port

port = Port(name="power", effort="input.u", flow="output.i")

load = BlackBoxSubsystem(
    lambda _t, inputs: inputs / resistance,
    inputs=("u",),
    outputs=("i",),
    ports=(port,),
    dissipated_power=lambda model: model.input.u * model.output.i,
)
port.bind(load)
```

The physical port remains acausal; `effort` and `flow` only adapt the causal callable to the
electrical network. Use `Port.signals(effort, flow)` for an energy pair that is not a physical
connection.

## Callable boundary

Inputs, outputs and states have identifier names and use ordered one-dimensional NumPy-compatible
values. `feedthrough` lists the current inputs read while calculating outputs and defaults to all
inputs; the solver uses it for dependency ordering and algebraic-loop detection.

PyTorch and other runtimes can use this array boundary directly. Standalone C++ export supports
callables that the scalar equation tracer can lower, but does not embed external runtime sessions.

## ONNX files

Install `peslite[onnx]`, then add an ONNX file as a non-loading element. Tensor names are mapping
keys; each list maps its flattened values to PESLite signal paths or public scalar outputs:

```yaml
elements:
  onnx_model:
    type: onnx
    model: models/model.onnx
    inputs:
      input: [load.power]
    outputs:
      output: [estimate]
```

Relative model paths start beside the `.pes` file. The output above is available as
`onnx_model.estimate` for recording, measurement routing and other configured model inputs.

Set `period` to sample the inputs and hold the outputs. Recurrent tensor state is registered as
PESLite state by mapping each state input to its next-state output:

```yaml
    period: 100.0e-6
    state:
      hidden_in:
        output: hidden_out
        initial: [0.0, 0.0]
```
