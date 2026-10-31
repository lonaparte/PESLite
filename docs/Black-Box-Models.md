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

Supplying initial `state` values registers them as PESLite states. PESLite passes the current state
vector to both functions and obtains its derivative from `rhs`:

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
rather than mutable data retained by the callable. Sampled models follow the same rule through a
registered control `Loop`.

## Callable boundary

Inputs, outputs and states use ordered one-dimensional NumPy-compatible values. A small callable
adapter can therefore reshape tensors or invoke any external runtime without adding that runtime to
PESLite. `feedthrough` lists the current inputs read while calculating outputs and defaults to all
inputs; the solver uses it for dependency ordering and algebraic-loop detection.
