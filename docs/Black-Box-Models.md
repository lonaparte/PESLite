# Black-box Models

`BlackBoxSubsystem` connects an array-valued Python callable to the normal model graph without
requiring a model base class or file format. A registered element places it through the ordinary
`subsystems()`, `connections()` and `terminals()` interfaces.

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

## Energy declaration

Use `observer=True` for a non-loading signal model. A loading model may use the ordinary energy
members `ports`, `storage`, `dissipated_power` and `supplied_power`; without a declaration, the
energy check applies its default black-box accounting.

```python
from peslite.solver.energy import PowerPort

load = BlackBoxSubsystem(
    lambda _t, inputs: inputs / resistance,
    inputs=("u",),
    outputs=("i",),
    ports=(PowerPort("inp.u", "out.i"),),
    dissipated_power=lambda model: model.inp.u * model.out.i,
)
```

## Callable boundary

Inputs, outputs and states have identifier names and use ordered one-dimensional NumPy-compatible
values. `feedthrough` lists the current inputs read while calculating outputs and defaults to all
inputs; the solver uses it for dependency ordering and algebraic-loop detection.

ONNX Runtime, PyTorch and other runtimes remain outside the core package. Their sessions are called
through the same array boundary; install `peslite[onnx]` for the optional ONNX dependencies.
Standalone C++ export supports callables that the scalar equation tracer can lower, but does not
embed external runtime sessions.
