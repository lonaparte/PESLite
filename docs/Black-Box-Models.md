# Black-box Models

`BlackBoxSubsystem` connects an array-valued Python callable to the normal model graph without
requiring a model base class or file format. It can be used inside any component assembled from
PESLite subsystems and connections.

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

A discrete model replaces `rhs` with `update` and a sampling `period`. PESLite samples all models
due at the same instant before committing any new state:

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

`offset` selects the position of the periodic update grid. Restored discrete states resume after the
restart instant instead of repeating its update.

## Callable boundary

Inputs, outputs and states use ordered one-dimensional NumPy-compatible values. A small callable
adapter can therefore reshape tensors or invoke any external runtime without adding that runtime to
PESLite. `feedthrough` lists the current inputs read while calculating outputs and defaults to all
inputs; the solver uses it for dependency ordering and algebraic-loop detection.

ONNX Runtime, PyTorch and other runtimes remain outside the core package. Their sessions are called
from the same array boundary; install `peslite[onnx]` to use the optional ONNX dependencies.
Standalone C++ export supports callables that the existing scalar equation tracer can lower. An
external runtime session is not embedded into the generated C++ executable.
