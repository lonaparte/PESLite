"""Thermal networks, loss models and temperature-dependent components."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Protocol, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ...components.network import Element
from ...components.registry import register_component_type
from ...solver import Bag, ConfigError, Empty, OutputMethod, Port
from ...solver import record_type

__all__ = [
    "ThermalLink", "LumpedThermalNetwork", "FosterImpedance",
    "SharedThermalModel", "SharedThermalNetwork",
    "LossTable", "SemiconductorLossModel", "SwitchingEvent",
    "ThermalComponent", "ThermalParams", "ThermalPropertyLaw",
    "ThermalResistor", "ThermalResistorModel",
    "ElectroThermal", "ElectroThermalModel",
]


def _positive(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a finite positive number") from None
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return number


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite") from None
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _record(prefix: str, names: tuple[str, ...]) -> type[Bag]:
    """Return one cached state/input/output record type for a field layout."""
    return record_type(prefix, names)


@dataclass(frozen=True)
class ThermalLink:
    """Thermal R in K/W between two named nodes, or a node and ambient (other=None)."""

    node: str
    other: str | None
    resistance_K_W: float

    def __post_init__(self):
        object.__setattr__(
            self, "resistance_K_W", _positive(self.resistance_K_W, "resistance_K_W")
        )
        if self.other is not None and self.node == self.other:
            raise ValueError("thermal link endpoints must differ")


class LumpedThermalNetwork:
    """Temperature dynamics of named capacitances and conductive/convective links.

    An ambient link is required for physical steady state. A node can represent
    a junction, case, shared heat sink or cooling plate. Multiple sources sharing
    the same sink are represented by ONE network, not duplicated electrical parts.
    """

    def __init__(self, capacities_J_K: Mapping[str, float], links: Sequence[ThermalLink],
                 *, ambient_K: float = 298.15) -> None:
        if not capacities_J_K:
            raise ValueError("a thermal network needs at least one capacitive node")
        if any(not isinstance(name, str) or not name.isidentifier()
               for name in capacities_J_K):
            raise ValueError("thermal node names must be identifiers")
        self.names = tuple(capacities_J_K)
        self.capacities_J_K = np.array([
            _positive(capacity, f"{name}.capacity_J_K")
            for name, capacity in capacities_J_K.items()
        ], dtype=float)
        self.ambient_K = _positive(ambient_K, "ambient_K")
        self.links = tuple(links)
        positions = {name: i for i, name in enumerate(self.names)}
        self._connections: list[tuple[int, int, float]] = []
        self._ambient: list[tuple[int, float]] = []
        for link in self.links:
            if not isinstance(link, ThermalLink):
                raise TypeError("links must contain ThermalLink entries")
            if link.node not in positions:
                raise ValueError(f"unknown thermal node {link.node!r}")
            conductance = 1.0 / link.resistance_K_W
            a = positions[link.node]
            if link.other is None:
                self._ambient.append((a, conductance))
            else:
                if link.other not in positions:
                    raise ValueError(f"unknown thermal node {link.other!r}")
                self._connections.append((a, positions[link.other], conductance))
        if not self._ambient:
            raise ValueError("at least one link to ambient is required")
        # Every node must conduct heat to ambient through the resistance graph.
        reachable = {idx for idx, _ in self._ambient}
        for _ in self.names:
            previous = len(reachable)
            for a, b, _g in self._connections:
                if a in reachable or b in reachable:
                    reachable.update((a, b))
            if len(reachable) == previous:
                break
        if len(reachable) != len(self.names):
            raise ValueError("each thermal node must have a path to ambient")
        self._positions = positions
        self.capacities_J_K.setflags(write=False)

    @classmethod
    def cauer(cls, capacities_J_K: Sequence[float], resistances_K_W: Sequence[float],
              *, ambient_K: float = 298.15) -> "LumpedThermalNetwork":
        """Physical junction-to-ambient ladder with N capacities and N resistances.

        The first N-1 resistances link neighboring nodes; the last attaches the
        final case/heatsink node to the fixed ambient temperature.
        """
        if not capacities_J_K or len(capacities_J_K) != len(resistances_K_W):
            raise ValueError("Cauer ladder needs equally sized, nonempty C and R lists")
        names = tuple(f"node_{index}" for index in range(len(capacities_J_K)))
        edges = [
            ThermalLink(names[index], names[index + 1], resistance)
            for index, resistance in enumerate(resistances_K_W[:-1])
        ]
        edges.append(ThermalLink(names[-1], None, resistances_K_W[-1]))
        return cls(dict(zip(names, capacities_J_K)), edges, ambient_K=ambient_K)

    def _array(self, temperatures_K: Sequence[float]) -> np.ndarray:
        x = np.asarray(temperatures_K, dtype=float)
        # Numerical/energy-structure probes may evaluate trial states outside the
        # physical temperature range. Only committed impulses enforce positive K;
        # otherwise the solver's speculative RHS would reject audit probes.
        if x.shape != (len(self.names),) or not np.isfinite(x).all():
            raise ValueError("temperatures_K need one finite K per node")
        return x

    def rates(self, temperatures_K: Sequence[float],
              heat_W: Mapping[str, float]) -> np.ndarray:
        """Return Kelvin/s with positive heat entering named nodes."""
        x = self._array(temperatures_K)
        unknown = set(heat_W) - self._positions.keys()
        if unknown:
            raise ValueError(f"heat sources name unknown thermal nodes: {sorted(unknown)}")
        q = np.zeros_like(x)
        for name, value in heat_W.items():
            q[self._positions[name]] += _finite(value, f"{name}.heat_W")
        for a, b, conductance in self._connections:
            flow = conductance * (x[b] - x[a])
            q[a] += flow
            q[b] -= flow
        for a, conductance in self._ambient:
            q[a] += conductance * (self.ambient_K - x[a])
        return q / self.capacities_J_K

    def cooling_power_W(self, temperatures_K: Sequence[float]) -> float:
        """Net heat rejected to the ambient, positive for hot components."""
        x = self._array(temperatures_K)
        return float(sum(g * (x[a] - self.ambient_K) for a, g in self._ambient))

    def stored_excess_heat_J(self, temperatures_K: Sequence[float]) -> float:
        """Thermal energy relative to constant ambient, in J."""
        x = self._array(temperatures_K)
        return float(np.dot(self.capacities_J_K, x - self.ambient_K))

    def committed_impulse(self, temperatures_K: Sequence[float],
                          energy_J: Mapping[str, float]) -> np.ndarray:
        """Apply a confirmed energy impulse to the owning capacitive node exactly."""
        x = self._array(temperatures_K).copy()
        for name, energy in energy_J.items():
            if name not in self._positions:
                raise ValueError(f"unknown thermal node {name!r}")
            x[self._positions[name]] += (
                _finite(energy, f"{name}.energy_J") / self.capacities_J_K[self._positions[name]]
            )
        if np.any(x <= 0):
            raise ValueError("heat impulse produced nonphysical temperature")
        return x


class FosterImpedance:
    """Parallel Foster thermal impedance referenced to a *fixed* ambient.

    Each branch has a temperature rise x_i and obeys
    d(x_i)/dt = (P - x_i/R_i)/C_i; T_j = T_ambient + sum(x_i).
    These are behavioral impedance states, not physical case or sink nodes:
    Foster branches cannot be directly joined into a shared heat-sink network.
    """

    def __init__(self, resistances_K_W: Sequence[float], capacitances_J_K: Sequence[float],
                 *, ambient_K: float = 298.15):
        if not resistances_K_W or len(resistances_K_W) != len(capacitances_J_K):
            raise ValueError("Foster impedance needs equally sized nonempty R and C lists")
        self.resistances_K_W = np.array([
            _positive(v, "resistance_K_W") for v in resistances_K_W], dtype=float)
        self.capacitances_J_K = np.array([
            _positive(v, "capacitance_J_K") for v in capacitances_J_K], dtype=float)
        self.ambient_K = _positive(ambient_K, "ambient_K")
        self.resistances_K_W.setflags(write=False)
        self.capacitances_J_K.setflags(write=False)

    def rates(self, rises_K: Sequence[float], heat_W: float) -> np.ndarray:
        x = np.asarray(rises_K, dtype=float)
        if x.shape != self.resistances_K_W.shape or not np.isfinite(x).all():
            raise ValueError("Foster rises must match the branch count")
        q = _finite(heat_W, "heat_W")
        return (q - x / self.resistances_K_W) / self.capacitances_J_K

    def junction_K(self, rises_K: Sequence[float]) -> float:
        x = np.asarray(rises_K, dtype=float)
        if x.shape != self.resistances_K_W.shape or not np.isfinite(x).all():
            raise ValueError("Foster rises must match the branch count")
        return float(self.ambient_K + np.sum(x))

    def committed_impulse(self, rises_K: Sequence[float], energy_J: float) -> np.ndarray:
        """Advance every branch's rise by E/C at an accepted impulse."""
        x = np.asarray(rises_K, dtype=float)
        if x.shape != self.resistances_K_W.shape or not np.isfinite(x).all():
            raise ValueError("Foster rises must match the branch count")
        return x + _finite(energy_J, "energy_J") / self.capacitances_J_K


class SharedThermalModel:
    """Couple independent heat inputs to one shared RC graph."""

    outputs_need_inputs: ClassVar[bool] = True
    observer: ClassVar[bool] = True

    def __init__(
        self,
        network: LumpedThermalNetwork,
        sources: Mapping[str, str],
        *,
        initial_K: float | None = None,
    ) -> None:
        if not isinstance(network, LumpedThermalNetwork):
            raise TypeError("shared thermal backend requires a physical lumped network")
        if not sources:
            raise ValueError("shared heat model needs at least one source")
        if (
            any(not isinstance(name, str) or not name.isidentifier() for name in sources)
            or any(node not in network.names for node in sources.values())
        ):
            raise ValueError("heat sources need identifier names and existing thermal nodes")
        self.network = network
        self.source_nodes = dict(sources)
        self.state_names = tuple(f"T_{node}_K" for node in network.names)
        self.input_names = tuple(f"heat_{name}_W" for name in sources)
        self.temperature_names = tuple(f"temperature_{node}_K" for node in network.names)
        self.output_names = (*self.temperature_names, "heat_total_W", "cooling_W")
        initial = network.ambient_K if initial_K is None else _finite(initial_K, "initial_K")
        if initial <= 0:
            raise ValueError("initial_K must be positive")
        self.state = _record("_SharedThermalState", self.state_names)(
            **dict.fromkeys(self.state_names, initial)
        )
        self.input = _record("_SharedThermalInput", self.input_names)(
            **dict.fromkeys(self.input_names, 0.0)
        )
        self.output = _record("_SharedThermalOutput", self.output_names)()
        self.set_outputs(0.0)

    def _temperatures(self) -> np.ndarray:
        return np.fromiter(
            (getattr(self.state, name) for name in self.state_names),
            dtype=float,
            count=len(self.state_names),
        )

    def _heat(self) -> dict[str, float]:
        heat = {name: 0.0 for name in self.network.names}
        for source, node in self.source_nodes.items():
            value = _finite(getattr(self.input, f"heat_{source}_W"), f"{source}.heat_W")
            if value < 0:
                raise ValueError("passive thermal sources require nonnegative losses")
            heat[node] += value
        return heat

    def set_outputs(self, _t: float) -> None:
        temperatures = self._temperatures()
        for name, temperature in zip(self.temperature_names, temperatures):
            setattr(self.output, name, float(temperature))
        self.output.heat_total_W = sum(self._heat().values())
        self.output.cooling_W = self.network.cooling_power_W(temperatures)

    def rhs(self, _t: float) -> tuple[float, ...]:
        return tuple(self.network.rates(self._temperatures(), self._heat()))


@register_component_type
class SharedThermalNetwork(Element):
    """Connect several heat signals to one configured thermal network."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        capacities_J_K: dict[str, float] = field(metadata={"structural": True})
        links: list[dict] = field(metadata={"structural": True})
        sources: dict[str, dict[str, str]] = field(metadata={"structural": True})
        ambient_K: float = 298.15
        initial_K: float = 298.15
        type: str = "shared_thermal_network"

        def __post_init__(self) -> None:
            if not isinstance(self.sources, Mapping) or not self.sources:
                raise ConfigError("shared thermal network requires heat sources")
            for name, source in self.sources.items():
                if (
                    not isinstance(name, str)
                    or not name.isidentifier()
                    or not isinstance(source, Mapping)
                    or set(source) != {"path", "node"}
                    or not all(isinstance(value, str) and value for value in source.values())
                ):
                    raise ConfigError("each shared heat source needs a named path and node")
            try:
                links = [ThermalLink(**item) for item in self.links]
                LumpedThermalNetwork(
                    self.capacities_J_K,
                    links,
                    ambient_K=self.ambient_K,
                )
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"invalid shared thermal network: {exc}") from None

    type = "shared_thermal_network"

    def __init__(self, name: str, cfg: Any, _buses: dict, _params: Any) -> None:
        self.name, self.cfg = name, cfg
        network = LumpedThermalNetwork(
            cfg.capacities_J_K,
            [ThermalLink(**item) for item in cfg.links],
            ambient_K=cfg.ambient_K,
        )
        self.model = SharedThermalModel(
            network,
            {source: item["node"] for source, item in cfg.sources.items()},
            initial_K=cfg.initial_K,
        )

    def connections(self) -> dict:
        return {
            (self.model, f"heat_{name}_W"): source["path"]
            for name, source in self.cfg.sources.items()
        }

    def signals(self) -> dict[str, float]:
        return {
            f"{self.name}.{field}": getattr(self.model.output, field)
            for field in self.model.output_names
        }


_AXES = frozenset(("current_A", "voltage_V", "temperature_K"))
_EVENT_KINDS = frozenset(("turn_on", "turn_off", "reverse_recovery"))


def _values(name: str, values: Any, count: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a one-dimensional array of finite numbers")
    if count is not None and result.size != count:
        raise ValueError(f"{name} must have {count} samples")
    return result


def _scalar(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be {'nonnegative and ' if nonnegative else ''}finite")
    return result


@dataclass(frozen=True)
class LossTable:
    """Nonnegative loss data on named, strictly increasing SI axes.

    Axes may be any nonempty subset of ``current_A``, ``voltage_V`` and
    ``temperature_K``. Multilinear interpolation is used inside the grid.
    Outside it, the default ``out_of_range='error'`` rejects the lookup;
    ``'clamp'`` uses the nearest boundary rather than extrapolating.
    Current and voltage are treated as magnitudes.
    """

    axes: Mapping[str, Sequence[float]]
    values: Any
    out_of_range: str = "error"
    _names: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _interpolator: Any = field(init=False, repr=False, compare=False)
    _grid: tuple[np.ndarray, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.axes or any(name not in _AXES for name in self.axes):
            raise ValueError("loss table needs named current_A, voltage_V or temperature_K axes")
        if self.out_of_range not in ("error", "clamp"):
            raise ValueError("out_of_range must be 'error' or 'clamp'")
        grid = []
        for name, points in self.axes.items():
            axis = _values(f"axis {name}", points)
            if axis.size < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError(f"axis {name} needs at least two strictly increasing points")
            if (np.any(axis < 0) if name != "temperature_K" else np.any(axis <= 0)):
                raise ValueError(f"axis {name} must use physical nonnegative SI values")
            axis = axis.copy()
            axis.setflags(write=False)
            grid.append(axis)
        try:
            values = np.array(self.values, dtype=float, copy=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("loss table values must be finite and nonnegative") from exc
        if values.shape != tuple(len(axis) for axis in grid):
            raise ValueError("loss table values must match the axis grid shape")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("loss table values must be finite and nonnegative")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "_names", tuple(self.axes))
        object.__setattr__(self, "_grid", tuple(grid))
        object.__setattr__(self, "_interpolator", RegularGridInterpolator(
            grid, values, method="linear", bounds_error=self.out_of_range == "error",
            fill_value=None,
        ))

    def evaluate(self, *, current_A: Any = 0.0, voltage_V: Any = 0.0,
                 temperature_K: Any = 298.15) -> float | np.ndarray:
        """Evaluate one or broadcasted lookup points using SI quantities."""
        sources = {"current_A": current_A, "voltage_V": voltage_V,
                   "temperature_K": temperature_K}
        try:
            columns = np.broadcast_arrays(*(
                np.asarray(sources[name], dtype=float) for name in self._names
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError("loss table coordinates must be broadcastable numeric arrays") from exc
        coordinates = []
        for name, column, grid in zip(self._names, columns, self._grid):
            if not np.isfinite(column).all():
                raise ValueError(f"{name} lookup coordinates must be finite")
            if name == "temperature_K" and np.any(column <= 0):
                raise ValueError("temperature_K must be positive")
            magnitude = column if name == "temperature_K" else np.abs(column)
            if self.out_of_range == "clamp":
                magnitude = np.clip(magnitude, grid[0], grid[-1])
            coordinates.append(magnitude.ravel())
        points = np.stack(coordinates, axis=-1)
        try:
            lookup = self._interpolator(points).reshape(columns[0].shape)
        except ValueError as exc:
            raise ValueError("loss table lookup is outside the declared axes") from exc
        return float(lookup) if lookup.ndim == 0 else lookup


@dataclass(frozen=True)
class SemiconductorLossModel:
    """Optional conduction voltage (V) and transition energies (J) for one device.

    Missing fields mean that loss mechanism is unavailable, not that it is known
    to be zero. The caller supplies known conduction intervals and transitions.
    """

    conduction_voltage_V: float | LossTable | None = None
    turn_on_energy_J: float | LossTable | None = None
    turn_off_energy_J: float | LossTable | None = None
    reverse_recovery_energy_J: float | LossTable | None = None
    turn_on_includes_reverse_recovery: bool = False

    def __post_init__(self) -> None:
        for name in ("conduction_voltage_V", "turn_on_energy_J", "turn_off_energy_J",
                     "reverse_recovery_energy_J"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, LossTable):
                object.__setattr__(self, name, _scalar(name, value, nonnegative=True))
        if type(self.turn_on_includes_reverse_recovery) is not bool:
            raise ValueError("turn_on_includes_reverse_recovery must be bool")


def _lookup(value: float | LossTable | None, current: Any, voltage: Any,
            temperature: float) -> Any:
    return (value.evaluate(current_A=current, voltage_V=voltage,
                           temperature_K=temperature)
            if isinstance(value, LossTable) else 0.0 if value is None else value)


@dataclass(frozen=True)
class SwitchingEvent:
    """One confirmed device transition, with pre-transition absolute operating values.

    These events must originate from accepted/committed simulation transitions or
    an independent verified event log. Snapshot-detected edges are not sufficient.
    """

    time_s: float
    kind: str
    current_A: float
    voltage_V: float
    device: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _EVENT_KINDS:
            raise ValueError(f"unknown switching event kind: {self.kind!r}")
        if self.device is not None and (
                not isinstance(self.device, str) or not self.device
                or any(not part.isidentifier() for part in self.device.split("."))):
            raise ValueError("device must contain dot-separated identifiers")
        for name in ("time_s", "current_A", "voltage_V"):
            object.__setattr__(self, name, _scalar(name, getattr(self, name)))


class ThermalPropertyLaw(Protocol):
    """Return named physical properties at one component temperature."""

    property_names: tuple[str, ...]

    def properties(self, temperature_K: float) -> Sequence[float]: ...


@dataclass(frozen=True, kw_only=True)
class ThermalParams:
    """Common temperature connection for thermal component adapters."""

    temperature_source: str = field(metadata={"runtime": False})

    def __post_init__(self) -> None:
        if not isinstance(self.temperature_source, str) or not self.temperature_source:
            raise ConfigError("temperature_source must name a physical output")


class ThermalComponent(Element):
    """Physical component whose constitutive properties depend on temperature.

    Concrete components retain the ordinary network-component interface. The thermal addon only
    adds a named property law, a temperature input and a realized-loss output.
    """

    def __init__(self, thermal_law: ThermalPropertyLaw) -> None:
        names = tuple(getattr(thermal_law, "property_names", ()))
        if (
            not names
            or any(not isinstance(name, str) or not name.isidentifier() for name in names)
            or len(set(names)) != len(names)
            or not callable(getattr(thermal_law, "properties", None))
        ):
            raise TypeError(
                "a thermal property law needs unique property_names and properties(temperature_K)"
            )
        self.thermal_law = thermal_law
        self.thermal_property_names = names

    def temperature_K(self) -> float:
        """Return the connected physical temperature."""
        try:
            temperature = float(self.input.temperature_K)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("a thermal component needs a numeric temperature_K input") from None
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("thermal component temperature_K must be finite and positive")
        return temperature

    def _thermal_property_values(self) -> tuple[float, ...]:
        """Evaluate and validate the current constitutive-property vector."""
        try:
            values = tuple(self.thermal_law.properties(self.temperature_K()))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"cannot evaluate thermal component properties: {exc}") from None
        if len(values) != len(self.thermal_property_names):
            raise ValueError("thermal property values must match property_names")
        try:
            values = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("thermal component properties must be numeric") from None
        if not all(math.isfinite(value) for value in values):
            raise ValueError("thermal component properties must be finite")
        return values

    def thermal_properties(self) -> Mapping[str, float]:
        """Return the component's current constitutive properties by name."""
        return dict(zip(self.thermal_property_names, self._thermal_property_values()))

    def dissipated_power(self) -> float:
        """Return the realized loss published by the concrete component."""
        try:
            loss = float(self.output.loss_W)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("a thermal component must publish numeric loss_W") from None
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError("thermal component loss_W must be finite and nonnegative")
        return loss

    @staticmethod
    def supplied_power() -> float:
        return 0.0


class _Input(Bag):
    __slots__ = ("voltage", "temperature_K")


class _Output(Bag):
    __slots__ = ("current", "loss_W", "resistance_Ohm")


@dataclass(frozen=True)
class _LinearResistance:
    property_names = ("resistance_Ohm",)

    reference_Ohm: float
    coefficient_per_K: float
    reference_K: float

    def properties(self, temperature_K: float) -> tuple[float]:
        return (self.reference_Ohm * (
            1.0 + self.coefficient_per_K * (temperature_K - self.reference_K)
        ),)


class ThermalResistorModel(ThermalComponent):
    """Memoryless scalar resistor with temperature supplied as a separate signal."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = True
    storage: ClassVar[tuple] = ()
    output_stages: ClassVar[tuple[OutputMethod, ...]] = (
        OutputMethod(
            "set_outputs",
            inputs=("voltage", "temperature_K"),
            outputs=("current", "loss_W", "resistance_Ohm"),
        ),
    )

    def __init__(self, r_reference_Ohm: float, alpha_per_K: float,
                 reference_K: float, phases: int = 1, *,
                 thermal_law: ThermalPropertyLaw | None = None) -> None:
        if phases not in (1, 3):
            raise ValueError("phases must be 1 or 3")
        super().__init__(thermal_law or _LinearResistance(
            r_reference_Ohm, alpha_per_K, reference_K
        ))
        if self.thermal_property_names != ("resistance_Ohm",):
            raise ValueError("a thermal resistor law must provide resistance_Ohm")
        self.power_scale = 1.5 if phases == 3 else 1.0
        zero = 0j if phases == 3 else 0.0
        self.state = Empty()
        self.input = _Input(voltage=zero, temperature_K=reference_K)
        self.output = _Output(
            current=zero,
            loss_W=0.0,
            resistance_Ohm=r_reference_Ohm,
        )
        self.port = Port(
            self,
            "power",
            scale=self.power_scale,
            effort="input.voltage",
            flow="output.current",
        )
        self.ports = (self.port,)

    def resistance(self) -> float:
        value = self._thermal_property_values()[0]
        if value <= 0.0:
            raise ValueError("thermal resistance must remain positive and finite")
        return value

    def set_outputs(self, _t: float) -> None:
        resistance = self.resistance()
        current = self.input.voltage / resistance
        self.output.resistance_Ohm = resistance
        self.output.current = current
        self.output.loss_W = self.power_scale * resistance * abs(current) ** 2

    @staticmethod
    def rhs(_t: float) -> tuple:
        return ()

@register_component_type
class ThermalResistor(Element):
    """Bind an electrical resistor to an electrothermal output.

    Use an ``elements`` entry of type ``thermal_resistor``.
    The temperature path is a scalar physical output, e.g. ``chip.temperature_K``.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params(ThermalParams):
        bus: str
        r_reference_Ohm: float
        alpha_per_K: float = 0.0
        reference_K: float = 298.15
        type: str = "thermal_resistor"

        def __post_init__(self):
            super().__post_init__()
            for label in ("r_reference_Ohm", "reference_K"):
                value = getattr(self, label)
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                    raise ConfigError(f"{label} must be finite and positive")
            if not math.isfinite(self.alpha_per_K):
                raise ConfigError("alpha_per_K must be finite")

    type = "thermal_resistor"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        self.model = ThermalResistorModel(
            cfg.r_reference_Ohm,
            cfg.alpha_per_K,
            cfg.reference_K,
            buses[cfg.bus].phases,
        )

    def connections(self) -> dict:
        return {(self.model, "temperature_K"): self.cfg.temperature_source}

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return ((self.cfg.bus, self.model.port),)

    def signals(self) -> dict[str, complex | float]:
        return {f"{self.name}.i": self.model.output.current,
                f"{self.name}.loss_W": self.model.output.loss_W,
                f"{self.name}.r_Ohm": self.model.output.resistance_Ohm}

    def retune(self, cfg: Any) -> bool:
        if cfg.bus != self.cfg.bus or cfg.temperature_source != self.cfg.temperature_source:
            raise ConfigError("thermal resistor wiring cannot change during a run")
        self.cfg = cfg
        self.model.thermal_law = _LinearResistance(
            cfg.r_reference_Ohm, cfg.alpha_per_K, cfg.reference_K
        )
        return False


class _ThermalInput(Bag):
    __slots__ = ("heat_W", "current_A", "voltage_V", "conduction_fraction")


class _ThermalOutput(Bag):
    __slots__ = ("temperature_K", "case_K", "sink_K", "heat_W", "accounted_heat_W",
                 "estimated_heat_W", "switching_energy_J", "cooling_W")


class ElectroThermalModel:
    """A two-node Cauer junction/case or fixed-ambient Foster temperature model.

    The external heat power comes from a REAL circuit output via Model.connections.
    An optional semiconductor loss model supplies ADDITIONAL, estimated conduction
    heat using the current and the **current junction temperature**. No estimated
    heat is silently subtracted from electrical supply.
    """

    outputs_need_inputs: ClassVar[bool] = True
    observer: ClassVar[bool] = True  # Exclude separate thermal energy from electrical P-H closure.

    def __init__(
        self, network: LumpedThermalNetwork | FosterImpedance, *,
        loss_model: SemiconductorLossModel | None = None,
        initial_temperature_K: float | None = None,
        switching_devices: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(network, (LumpedThermalNetwork, FosterImpedance)):
            raise TypeError("network must be a Cauer/lumped network or Foster impedance")
        if loss_model is not None and not isinstance(loss_model, SemiconductorLossModel):
            raise TypeError("loss_model must be a SemiconductorLossModel")
        self.network = network
        self.loss_model = loss_model
        self.switching_devices = frozenset(switching_devices)
        self._foster = isinstance(network, FosterImpedance)
        self._temperature_names = (
            tuple(f"rise_{k}_K" for k in range(len(network.capacitances_J_K)))
            if self._foster else
            tuple(f"T_{name}_K" for name in network.names)
        )
        self.state_names = self._temperature_names
        self.discrete_state_names = ("last_transition_id", "switching_energy_J")
        reference = (network.ambient_K if initial_temperature_K is None
                     else _positive(initial_temperature_K, "initial_temperature_K"))
        values = ([reference - network.ambient_K] * len(self._temperature_names)
                  if self._foster else [reference] * len(self._temperature_names))
        self.state = _record("_ThermalState", self.state_names)(
            **dict(zip(self._temperature_names, values))
        )
        self.last_transition_id = -1
        self.switching_energy_J = 0.0
        self.input = _ThermalInput(heat_W=0.0, current_A=0.0,
                                   voltage_V=0.0, conduction_fraction=1.0)
        self.output = _ThermalOutput()
        self.set_outputs(0.0)

    def _states(self) -> np.ndarray:
        return np.asarray([getattr(self.state, name) for name in self._temperature_names],
                          dtype=float)

    def temperature_K(self) -> float:
        values = self._states()
        return (self.network.junction_K(values) if self._foster else float(values[0]))

    def _heat(self) -> tuple[float, float]:
        realized = float(self.input.heat_W)
        current = float(self.input.current_A)
        voltage = float(self.input.voltage_V)
        fraction = float(self.input.conduction_fraction)
        if (not all(math.isfinite(v) for v in (realized, current, voltage, fraction))
                or realized < 0.0 or not 0.0 <= fraction <= 1.0):
            raise ValueError("thermal heat/current/voltage inputs must be finite and physical")
        curve = None if self.loss_model is None else self.loss_model.conduction_voltage_V
        additional = float(fraction * abs(current) * _lookup(
            curve, current, voltage, self.temperature_K()
        ))
        return realized, additional

    def set_outputs(self, t: float) -> None:
        values = self._states()
        realized, additional = self._heat()
        self.output.temperature_K = self.temperature_K()
        self.output.case_K = (math.nan if self._foster or len(values) < 2
                              else float(values[1]))
        self.output.sink_K = (math.nan if self._foster or len(values) < 3
                              else float(values[-1]))
        self.output.accounted_heat_W = realized
        self.output.estimated_heat_W = additional
        self.output.heat_W = realized + additional
        self.output.switching_energy_J = self.switching_energy_J
        self.output.cooling_W = (math.nan if self._foster
                                 else self.network.cooling_power_W(values))

    def rhs(self, t: float) -> tuple[float, ...]:
        realized, additional = self._heat()
        p = realized + additional
        state = self._states()
        derivatives = (self.network.rates(state, p) if self._foster
                       else self.network.rates(state, {self.network.names[0]: p}))
        return tuple(float(value) for value in derivatives)

    def get_state(self) -> dict[str, int | float]:
        return {
            "last_transition_id": self.last_transition_id,
            "switching_energy_J": self.switching_energy_J,
        }

    def set_state(self, values: Mapping[str, Any]) -> None:
        unknown = set(values) - {"last_transition_id", "switching_energy_J"}
        if unknown:
            raise KeyError(f"unknown electrothermal state {sorted(unknown)}")
        if "last_transition_id" in values:
            value = values["last_transition_id"]
            if isinstance(value, bool) or not isinstance(value, int) or value < -1:
                raise ValueError("last_transition_id must be an integer >= -1")
            self.last_transition_id = value
        if "switching_energy_J" in values:
            value = float(values["switching_energy_J"])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("switching_energy_J must be finite and nonnegative")
            self.switching_energy_J = value

    def _transition_energy(
        self, event: Any, *, strict: bool, temperature_K: float | None = None
    ) -> float | None:
        if self.loss_model is None:
            if strict:
                raise ValueError("switching events require a semiconductor loss model")
            return None
        if (event.kind == "reverse_recovery"
                and self.loss_model.turn_on_includes_reverse_recovery):
            if strict:
                raise ValueError("reverse recovery is already included in turn-on energy")
            return None
        field = {"turn_on": "turn_on_energy_J", "turn_off": "turn_off_energy_J",
                 "reverse_recovery": "reverse_recovery_energy_J"}.get(event.kind)
        if field is None:
            raise ValueError(f"unknown switching event kind: {event.kind!r}")
        curve = getattr(self.loss_model, field)
        if curve is None:
            if strict:
                raise ValueError(f"no {event.kind} energy curve was provided")
            return None
        temperature = self.temperature_K() if temperature_K is None else temperature_K
        return float(_lookup(curve, event.current_A, event.voltage_V, temperature))

    def _apply_transition_energy(
        self, energy: float, time_s: float, *, refresh_outputs: bool = True
    ) -> None:
        states = self._states()
        if self._foster:
            updated = self.network.committed_impulse(states, energy)
        else:
            updated = self.network.committed_impulse(
                states, {self.network.names[0]: energy}
            )
        for name, value in zip(self._temperature_names, updated):
            setattr(self.state, name, float(value))
        self.switching_energy_J += energy
        self.output.switching_energy_J = self.switching_energy_J
        if refresh_outputs:
            self.set_outputs(time_s)

    def apply_committed_transition(self, event: SwitchingEvent, *, event_id: int) -> float:
        """Apply one accepted event in J, once per monotonic device event sequence.

        Invoke only at an actual model boundary, then repack the existing model
        state with `model.get_initial_values()`. The monotonic event ID is kept in
        a restartable state so continuation cannot reapply an already counted
        event. Event IDs must increase monotonically per attachment.
        """
        if not isinstance(event, SwitchingEvent):
            raise TypeError("event must be a confirmed SwitchingEvent")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
            raise ValueError("event_id must be a nonnegative integer")
        if event_id <= self.last_transition_id:
            raise ValueError("event_id has already been committed or is out of order")
        energy = self._transition_energy(event, strict=True)
        assert energy is not None
        self._apply_transition_energy(energy, event.time_s)
        self.last_transition_id = event_id
        return energy

    def apply_committed_transitions(self, events: Any) -> float:
        """Apply selected, available device energies from one committed circuit boundary."""
        temperature = self.temperature_K()
        energy = 0.0
        count = 0
        time_s = 0.0
        for event in events:
            if self.switching_devices and event.device not in self.switching_devices:
                continue
            value = self._transition_energy(
                event, strict=False, temperature_K=temperature
            )
            if value is not None:
                energy += value
                count += 1
                time_s = event.time_s
        if count == 0:
            return 0.0
        self._apply_transition_energy(energy, time_s, refresh_outputs=False)
        self.last_transition_id += count
        return energy


@register_component_type
class ElectroThermal(Element):
    """Configure a passive, two-node electrothermal attachment in ``elements``.

    ``heat_source`` names a pre-existing scalar physical output (in W).
    This is the circuit's realized dissipation, never an extra electrical load.
    Optional ``current_source`` selects the actual device current for an ideal
    electrical model with a separate estimated conduction voltage. Those modes
    cannot be combined without proven disjoint heat mechanisms.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        c_j_J_K: float
        r_jc_K_W: float
        c_c_J_K: float
        r_ca_K_W: float
        ambient_K: float = 298.15
        initial_K: float = 298.15
        heat_source: str | None = field(default=None, metadata={"runtime": False})
        current_source: str | None = field(default=None, metadata={"runtime": False})
        switching_source: str | None = field(default=None, metadata={"runtime": False})
        switching_devices: tuple[str, ...] = field(default=(), metadata={"runtime": False})
        conduction_voltage_V: float | None = None
        turn_on_energy_J: float | None = None
        turn_off_energy_J: float | None = None
        reverse_recovery_energy_J: float | None = None
        turn_on_includes_reverse_recovery: bool = False
        type: str = "electrothermal"

        def __post_init__(self):
            try:
                for name in ("c_j_J_K", "r_jc_K_W", "c_c_J_K", "r_ca_K_W",
                             "ambient_K", "initial_K"):
                    _positive(getattr(self, name), name)
            except ValueError as exc:
                raise ConfigError(str(exc)) from None
            if self.conduction_voltage_V is not None:
                if (not math.isfinite(self.conduction_voltage_V)
                        or self.conduction_voltage_V < 0):
                    raise ConfigError("conduction_voltage_V must be nonnegative")
            if self.heat_source is not None and self.current_source is not None:
                raise ConfigError("heat_source and current_source are mutually exclusive")
            if (self.heat_source is None and self.current_source is None
                    and self.switching_source is None):
                raise ConfigError(
                    "electrothermal requires a heat, current or switching source"
                )
            if (self.heat_source is not None and
                    self.conduction_voltage_V is not None):
                raise ConfigError(
                    "electrically accounted and estimated conduction must not overlap"
                )
            if (self.current_source is not None and self.conduction_voltage_V is None):
                raise ConfigError("current_source requires a conduction voltage estimate")
            energies = (
                self.turn_on_energy_J,
                self.turn_off_energy_J,
                self.reverse_recovery_energy_J,
            )
            if self.switching_source is None and (
                    any(value is not None for value in energies)
                    or self.switching_devices or self.turn_on_includes_reverse_recovery):
                raise ConfigError("switching-loss fields require switching_source")
            if self.switching_source is not None and not any(
                    value is not None for value in energies):
                raise ConfigError("switching_source requires at least one transition energy")
            for name, value in zip((
                    "turn_on_energy_J", "turn_off_energy_J", "reverse_recovery_energy_J"
            ), energies):
                if value is not None and (not math.isfinite(value) or value < 0.0):
                    raise ConfigError(f"{name} must be finite and nonnegative")
            if type(self.turn_on_includes_reverse_recovery) is not bool:
                raise ConfigError("turn_on_includes_reverse_recovery must be bool")
            if (len(set(self.switching_devices)) != len(self.switching_devices)
                    or any(not name or any(
                        not part.isidentifier() for part in name.split(".")
                    ) for name in self.switching_devices)):
                raise ConfigError("switching_devices must contain unique device paths")

    type = "electrothermal"

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        self.model = self._new_model(cfg)

    @staticmethod
    def _new_model(cfg: Any) -> ElectroThermalModel:
        loss_values = (
            cfg.conduction_voltage_V,
            cfg.turn_on_energy_J,
            cfg.turn_off_energy_J,
            cfg.reverse_recovery_energy_J,
        )
        model = (SemiconductorLossModel(
            conduction_voltage_V=cfg.conduction_voltage_V,
            turn_on_energy_J=cfg.turn_on_energy_J,
            turn_off_energy_J=cfg.turn_off_energy_J,
            reverse_recovery_energy_J=cfg.reverse_recovery_energy_J,
            turn_on_includes_reverse_recovery=cfg.turn_on_includes_reverse_recovery,
        ) if any(value is not None for value in loss_values) else None)
        network = LumpedThermalNetwork.cauer(
            [cfg.c_j_J_K, cfg.c_c_J_K], [cfg.r_jc_K_W, cfg.r_ca_K_W],
            ambient_K=cfg.ambient_K,
        )
        return ElectroThermalModel(network, loss_model=model,
                                  initial_temperature_K=cfg.initial_K,
                                  switching_devices=cfg.switching_devices)

    def connections(self) -> dict:
        source = (self.cfg.heat_source if self.cfg.heat_source is not None
                  else self.cfg.current_source)
        if source is None:
            return {}
        field = "heat_W" if self.cfg.heat_source is not None else "current_A"
        return {(self.model, field): source}

    def transition_bindings(self) -> tuple[tuple[str, ElectroThermalModel], ...]:
        return (() if self.cfg.switching_source is None else
                ((self.cfg.switching_source, self.model),))

    def signals(self) -> dict[str, float]:
        return {f"{self.name}.temperature_K": self.model.output.temperature_K,
                f"{self.name}.case_K": self.model.output.case_K,
                f"{self.name}.heat_W": self.model.output.heat_W,
                f"{self.name}.switching_energy_J": self.model.output.switching_energy_J,
                f"{self.name}.cooling_W": self.model.output.cooling_W}

    def retune(self, cfg: Any) -> bool:
        if (cfg.heat_source != self.cfg.heat_source
                or cfg.current_source != self.cfg.current_source
                or cfg.switching_source != self.cfg.switching_source
                or cfg.switching_devices != self.cfg.switching_devices):
            raise ConfigError("cannot retune electrothermal source connections during a run")
        if cfg == self.cfg:
            return False
        state = self.model._states()
        replacement = self._new_model(cfg)
        if len(replacement._temperature_names) != len(self.model._temperature_names):
            raise ConfigError("electrothermal retune cannot change the state layout")
        self.model.network = replacement.network
        self.model.loss_model = replacement.loss_model
        for name, value in zip(self.model._temperature_names, state):
            setattr(self.model.state, name, float(value))
        self.cfg = cfg
        return False
