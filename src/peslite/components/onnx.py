"""Configured ONNX inference through the ordinary black-box subsystem boundary."""

from __future__ import annotations

import keyword
import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..solver.model import BlackBoxSubsystem, ConfigError
from .network import Element, register_component_type

__all__ = ["ONNXModel", "ONNXStateParams"]


_TENSOR_DTYPES = {
    "tensor(bool)": np.dtype(bool),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint64)": np.dtype(np.uint64),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
    "tensor(complex64)": np.dtype(np.complex64),
    "tensor(complex128)": np.dtype(np.complex128),
}


def _names(mapping: dict[str, list[str]], label: str) -> tuple[str, ...]:
    """Validate tensor-to-scalar mappings and return their flattened scalar names."""
    flattened = []
    for tensor, names in mapping.items():
        if not isinstance(tensor, str) or not tensor:
            raise ConfigError(f"{label} tensor names must be nonempty strings")
        if not isinstance(names, list) or not names:
            raise ConfigError(f"{label}.{tensor} must be a nonempty list")
        if any(not isinstance(name, str) or not name for name in names):
            raise ConfigError(f"{label}.{tensor} names must be nonempty strings")
        flattened.extend(names)
    return tuple(flattened)


@dataclass(frozen=True, kw_only=True)
class ONNXStateParams:
    """One recurrent ONNX input, its next-state output and initial flattened value."""

    output: str
    initial: list[complex | float | int | bool]

    def __post_init__(self) -> None:
        if not isinstance(self.output, str) or not self.output:
            raise ConfigError("output must be a nonempty ONNX tensor name")
        if not isinstance(self.initial, list) or not self.initial:
            raise ConfigError("initial must be a nonempty list")
        if any(not isinstance(value, numbers.Number) or not np.isfinite(value)
               for value in self.initial):
            raise ConfigError("initial values must be finite numeric scalars")


@dataclass(frozen=True)
class _Tensor:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    size: int

    def array(self, values) -> np.ndarray:
        return np.asarray(values, dtype=self.dtype).reshape(self.shape)


def _tensor(meta: Any, size: int, where: str) -> _Tensor:
    """Bind one configured flattened width to an ONNX tensor shape and dtype."""
    dtype = _TENSOR_DTYPES.get(meta.type)
    if dtype is None:
        raise ConfigError(f"{where}: unsupported ONNX type {meta.type!r}")
    shape = list(meta.shape)
    dynamic = [index for index, dimension in enumerate(shape)
               if not isinstance(dimension, int) or dimension <= 0]
    known = math.prod(dimension for dimension in shape
                      if isinstance(dimension, int) and dimension > 0)
    if len(dynamic) > 1 or (dynamic and (known == 0 or size % known)):
        raise ConfigError(
            f"{where}: shape {meta.shape!r} cannot be inferred from {size} configured values"
        )
    if dynamic:
        shape[dynamic[0]] = size // known
    expected = math.prod(shape) if shape else 1
    if expected != size:
        raise ConfigError(f"{where}: tensor has {expected} values, but {size} are configured")
    return _Tensor(meta.name, dtype, tuple(shape), size)


def _initial(dtype: np.dtype):
    if dtype.kind == "c":
        return 0j
    if dtype.kind == "b":
        return False
    return 0.0


class _ONNXInference:
    """One validated ONNX Runtime session with flattened PESLite boundaries."""

    def __init__(self, cfg: Any) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ConfigError(
                "ONNX models require the optional dependencies: pip install peslite[onnx]"
            ) from exc

        path = Path(cfg.model).expanduser().resolve()
        if not path.is_file():
            raise ConfigError(f"model file does not exist: {path}")
        try:
            self.session = ort.InferenceSession(str(path))
        except Exception as exc:
            raise ConfigError(f"cannot load ONNX model {path}: {exc}") from exc

        inputs = {meta.name: meta for meta in self.session.get_inputs()}
        outputs = {meta.name: meta for meta in self.session.get_outputs()}
        overlap = set(cfg.inputs) & set(cfg.state)
        if overlap:
            raise ConfigError(f"tensors cannot be both external inputs and state: {sorted(overlap)}")
        configured_inputs = set(cfg.inputs) | set(cfg.state)
        missing = set(inputs) - configured_inputs
        unknown = configured_inputs - set(inputs)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"unconfigured inputs {sorted(missing)}")
            if unknown:
                details.append(f"unknown inputs {sorted(unknown)}")
            raise ConfigError("; ".join(details))

        requested_outputs = set(cfg.outputs) | {state.output for state in cfg.state.values()}
        unknown_outputs = requested_outputs - set(outputs)
        if unknown_outputs:
            raise ConfigError(f"unknown outputs {sorted(unknown_outputs)}")

        self.external = tuple(
            _tensor(inputs[name], len(names), f"inputs.{name}")
            for name, names in cfg.inputs.items()
        )
        self.hidden = tuple(
            _tensor(inputs[name], len(state.initial), f"state.{name}.initial")
            for name, state in cfg.state.items()
        )
        self.public = tuple(
            _tensor(outputs[name], len(names), f"outputs.{name}")
            for name, names in cfg.outputs.items()
        )
        self.next_hidden = tuple(
            _tensor(outputs[state.output], len(state.initial), f"state.{name}.output")
            for name, state in cfg.state.items()
        )
        self.output_names = tuple(dict.fromkeys(
            [tensor.name for tensor in self.public]
            + [tensor.name for tensor in self.next_hidden]
        ))

    @staticmethod
    def _parts(values, tensors: tuple[_Tensor, ...]):
        offset = 0
        for tensor in tensors:
            yield tensor, values[offset:offset + tensor.size]
            offset += tensor.size

    def run(self, inputs, hidden=()) -> tuple[list[Any], list[Any]]:
        feeds = {}
        for tensor, values in self._parts(inputs, self.external):
            feeds[tensor.name] = tensor.array(values)
        for tensor, values in self._parts(hidden, self.hidden):
            feeds[tensor.name] = tensor.array(values)
        result = dict(zip(self.output_names, self.session.run(self.output_names, feeds)))

        def flattened(tensors: tuple[_Tensor, ...]) -> list[Any]:
            values = []
            for tensor in tensors:
                array = np.asarray(result[tensor.name]).reshape(-1)
                if array.size != tensor.size:
                    raise ValueError(
                        f"ONNX output {tensor.name!r} returned {array.size} values; "
                        f"expected {tensor.size}"
                    )
                values.extend(item.item() for item in array)
            return values

        return flattened(self.public), flattened(self.next_hidden)


@register_component_type
class ONNXModel(Element):
    """A configured, non-loading ONNX model connected to physical subsystem outputs."""

    type = "onnx"

    @dataclass(frozen=True, kw_only=True)
    class Params:
        model: str = field(metadata={"asset": True, "runtime": False})
        outputs: dict[str, list[str]] = field(
            metadata={"literal_keys": True, "runtime": False}
        )
        inputs: dict[str, list[str]] = field(
            default_factory=dict, metadata={"literal_keys": True, "runtime": False}
        )
        period: float | None = field(default=None, metadata={"runtime": False})
        offset: float = field(default=0.0, metadata={"runtime": False})
        state: dict[str, ONNXStateParams] = field(
            default_factory=dict,
            metadata={"entries": ONNXStateParams, "literal_keys": True, "runtime": False},
        )
        type: str = "onnx"

        def __post_init__(self) -> None:
            if not isinstance(self.model, str) or not self.model:
                raise ConfigError("model must be a nonempty path")
            _names(self.inputs, "inputs")
            outputs = _names(self.outputs, "outputs")
            if not outputs:
                raise ConfigError("outputs must contain at least one tensor mapping")
            if any(not name.isidentifier() or keyword.iskeyword(name) for name in outputs):
                raise ConfigError("output names must be valid identifiers")
            if len(set(outputs)) != len(outputs):
                raise ConfigError("output names must be unique")
            if self.state and self.period is None:
                raise ConfigError("recurrent state requires period")
            if self.period is not None and (
                    isinstance(self.period, bool) or not math.isfinite(self.period)
                    or self.period <= 0.0):
                raise ConfigError(f"period must be finite and positive, got {self.period}")
            if (isinstance(self.offset, bool) or not math.isfinite(self.offset)
                    or self.offset < 0.0
                    or (self.period is not None and self.offset >= self.period)):
                raise ConfigError("offset must be in [0, period)")
            if self.period is None and self.offset != 0.0:
                raise ConfigError("offset requires period")

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        runtime = _ONNXInference(cfg)
        source_paths = tuple(
            source for sources in cfg.inputs.values() for source in sources
        )
        input_names = tuple(f"input_{index}" for index in range(len(source_paths)))
        input_initial = {
            port: _initial(tensor.dtype)
            for port, tensor in zip(
                input_names,
                (tensor for tensor in runtime.external for _ in range(tensor.size)),
            )
        }
        output_names = tuple(name for names in cfg.outputs.values() for name in names)
        output_initial = {
            name: _initial(tensor.dtype)
            for name, tensor in zip(
                output_names,
                (tensor for tensor in runtime.public for _ in range(tensor.size)),
            )
        }
        self._sources = dict(zip(input_names, source_paths))

        if cfg.period is None:
            self.model = BlackBoxSubsystem(
                lambda _t, inputs: runtime.run(inputs)[0],
                inputs=input_initial,
                outputs=output_initial,
                observer=True,
            )
            return

        hidden_initial = [value for state in cfg.state.values() for value in state.initial]
        held_initial = list(output_initial.values())
        hidden_size = len(hidden_initial)
        state = {
            **{f"state_{index}": value for index, value in enumerate(hidden_initial)},
            **{f"output_{index}": value for index, value in enumerate(held_initial)},
        }

        def output(_t, values, _inputs):
            return values[hidden_size:]

        def update(_t, values, inputs):
            public, hidden = runtime.run(inputs, values[:hidden_size])
            return (*hidden, *public)

        self.model = BlackBoxSubsystem(
            output,
            inputs=input_initial,
            outputs=output_initial,
            state=state,
            update=update,
            period=cfg.period,
            offset=cfg.offset,
            feedthrough=(),
            observer=True,
        )

    def subsystems(self) -> dict[str, Any]:
        return {self.name: self.model}

    def connections(self) -> dict:
        return {(self.model, name): source for name, source in self._sources.items()}

    @staticmethod
    def port_bindings() -> tuple:
        return ()

    def signals(self) -> dict[str, Any]:
        return {
            f"{self.name}.{name}": getattr(self.model.output, name)
            for name in self.model.output_names
        }
