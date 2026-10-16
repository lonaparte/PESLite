"""What each control loop computes: the loop types, their typed signals and their registry.

A loop type is one class: its parameters (``Params``, the schema of its class-named
``ctrl.loops`` entry), its typed input and output ports, its role in the default wiring, and its
sampled/continuous law.
How loops are connected is not theirs to know (:mod:`.controller`). Built-in loops also expose
their continuous-time outputs and state derivatives for the ideal averaged converter model.
Grid-control signals use pu and scalar converter signals use SI. Unit-system-independent loops
declare generic scalar voltage/current ports that bind to either control-graph boundary.
Time is in s, angles in rad and frequencies in rad/s.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field, is_dataclass
from typing import Any, ClassVar, Mapping, Optional

import numpy as np

from ..solver.model import ConfigError, assign, gather, join, scatter
from .blocks import Filter, Integrator, OutputLimit, PI, complex_power

__all__ = ["SignalType", "I_AB", "V_AB", "I_DQ", "V_DQ", "I", "V",
           "ANGLE", "FREQUENCY", "POWER",
           "ROLES", "Loop", "LOOP_TYPES", "register_loop_type", "PowerFilterParams",
           "CurrentLimitParams",
           "SRFPLL", "CurrentLoop", "DCVoltageControl", "SyncLawParams", "SyncLaw",
           "PSC", "Droop", "VSG", "DVOC",
           "Matching", "VirtualImpedance", "VirtualAdmittance", "ActiveDamping",
           "SI_CURRENT", "SI_VOLTAGE", "DUTY_RATIO", "MODULATION", "HERTZ",
           "InductorCurrentControl", "ResonantModulation"]


# ------------------------------------------------------------------ signals

@dataclass(frozen=True)
class SignalType:
    """The physical quantity and representation carried by one directed control port."""

    unit: str
    frame: str = "scalar"
    complex_value: bool = False

    def compatible(self, other: "SignalType") -> bool:
        """Return whether two ports carry the same signal in a compatible representation."""
        if self == other:
            return True
        if self.frame != other.frame or self.complex_value != other.complex_value:
            return False
        families = {
            "current": frozenset(("current", "pu_current", "A")),
            "voltage": frozenset(("voltage", "pu_voltage", "V")),
        }
        return any(
            self.unit in family and other.unit in family
            and (self.unit == quantity or other.unit == quantity)
            for quantity, family in families.items()
        )


I_AB = SignalType("pu_current", "alpha_beta", True)
V_AB = SignalType("pu_voltage", "alpha_beta", True)
I_DQ = SignalType("pu_current", "dq", True)
V_DQ = SignalType("pu_voltage", "dq", True)
I = SignalType("pu_current")
V = SignalType("pu_voltage")
ANGLE = SignalType("rad")
FREQUENCY = SignalType("rad/s")
POWER = SignalType("pu_power")
GENERIC_CURRENT = SignalType("current")
GENERIC_VOLTAGE = SignalType("voltage")


# ------------------------------------------------------------------ the base class and the registry

# The parts a loop can play in the default GFL/GFM wiring assembled by ControllerGraph.
ROLES = ("pll", "sync", "current", "dc_voltage", "impedance", "admittance",
         "damping", "modulation")


class Loop:
    """A control loop with typed ports, built as ``cls(cfg, unit, startup)``.

    A loop type sets ``type`` (its internal registry name), ``Params`` (a frozen dataclass with
    ``type`` and ``period`` fields; its ``_quantities`` name the fields given in SI or pu),
    ``inputs`` / ``outputs`` (port name -> :class:`SignalType`) and ``role`` (one of
    :data:`ROLES`, or ``None``: wired only by ``ctrl.connections``). With ``outputs_from_state``
    its held outputs are set from its states (``initial_outputs()``) when states are loaded.
    A loop made from standard dynamic blocks registers them with ``state_block()`` and implements
    one positional ``equation()``. The base class then supplies sampled and continuous execution.
    A specialized loop may instead provide ``sample()`` and the allocation-free ``flow_path()``.
    Its named states are the attributes in ``state_names`` plus its registered blocks unless it
    overrides ``get_state`` / ``set_state``. A loop rebuilt after retuning continues from those
    states and from attributes listed in ``carried``.
    ``cfg``: the loop's parameters; ``unit``: the unit's parameters; ``startup``: the controller's
    current run/ramp state.
    """

    type: ClassVar[str]
    Params: ClassVar[type]
    inputs: ClassVar[Mapping[str, SignalType]] = {}
    outputs: ClassVar[Mapping[str, SignalType]] = {}
    role: ClassVar[Optional[str]] = None
    outputs_from_state: ClassVar[bool] = False  # held outputs = initial_outputs() after loading states
    flow_inputs: ClassVar[tuple[str, ...] | None] = None
    state_names: tuple[str, ...] | Mapping[str, str] = ()
    carried: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: Any, unit: Any, startup: Any) -> None:
        self.cfg, self.unit, self.startup = cfg, unit, startup
        self.block_period = None if unit.bridge.model == "averaging" else cfg.period
        self._state_blocks: dict[str, Any] = {}

    def state_block(self, name: str, block):
        """Register a standard dynamic block and return it for use in one equation.

        Its state, continuous derivative and direct ODE binding are then owned by this loop. A
        first-order block uses ``name``; a higher-order filter uses ``name.0``, ``name.1``, ...
        """
        if not isinstance(name, str) or not name or name in self._state_blocks:
            raise ValueError(f"loop state block needs a unique non-empty name, got {name!r}")
        required = ("get_state", "set_state", "state_derivatives", "state_bindings")
        if any(not callable(getattr(block, method, None)) for method in required):
            raise TypeError(f"{type(block).__name__} is not a dynamic control block")
        names = self.state_names
        existing = set(names) | set(gather(self._state_blocks))
        proposed = {join(name, local) for local in block.get_state()}
        overlap = existing & proposed
        if overlap:
            raise ValueError(f"loop state block {name!r} duplicates states {sorted(overlap)}")
        self._state_blocks[name] = block
        return block

    def _block_state_derivatives(self) -> tuple[Any, ...]:
        """Return registered block derivatives in public-state order."""
        return tuple(
            value
            for block in self._state_blocks.values()
            for value in block.state_derivatives().values()
        )

    def equation(self, *inputs):
        """Return outputs from one mode-independent equation using registered state blocks."""
        raise NotImplementedError

    def initial_outputs(self) -> dict[str, Any]:
        raise NotImplementedError

    def sample(self, t: float, inputs: Mapping[str, Any]) -> dict[str, Any]:
        if self.flow_inputs is None:
            raise NotImplementedError
        result = self.equation(*(inputs[port] for port in self.flow_inputs))
        return dict(zip(self.outputs, result))

    def flow(self, t: float, inputs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return continuous outputs and named state derivatives for ideal averaging."""
        if self.flow_inputs is None:
            raise NotImplementedError(
                f"loop type {self.type!r} has no continuous-time implementation"
            )
        result = self.flow_path(*(inputs[port] for port in self.flow_inputs))
        n = len(self.outputs)
        return dict(zip(self.outputs, result[:n])), dict(zip(self.flow_state(), result[n:]))

    def flow_into(self, t: float, inputs: Mapping[str, Any],
                  outputs: dict[str, Any], derivatives: dict[str, Any]) -> None:
        """Evaluate continuous outputs and derivatives into reusable mappings."""
        if self.flow_inputs is None:
            fresh_outputs, fresh_derivatives = self.flow(t, inputs)
            outputs.update(fresh_outputs)
            derivatives.update(fresh_derivatives)
            return
        result = self.flow_path(*(inputs[port] for port in self.flow_inputs))
        n = len(self.outputs)
        for name, value in zip(self.outputs, result[:n]):
            outputs[name] = value
        for name, value in zip(self.flow_state(), result[n:]):
            derivatives[name] = value

    def flow_path(self, *inputs):
        """Allocation-free positional continuous law used by the compiled graph.

        A loop that registers standard dynamic blocks only implements :meth:`equation`; their
        derivatives are appended automatically in the same order as :meth:`flow_state`.
        """
        if type(self).equation is Loop.equation:
            raise NotImplementedError(
                f"loop type {self.type!r} has no continuous-time implementation"
            )
        if self.state_names:
            raise NotImplementedError(
                f"loop type {self.type!r}: equation-based loops must register every dynamic "
                "state with state_block()"
            )
        outputs = tuple(self.equation(*inputs))
        return outputs + self._block_state_derivatives()

    def flow_state(self) -> dict[str, Any]:
        """States integrated with the plant in ideal averaging mode."""
        return self.get_state()

    def flow_state_bindings(self):
        """Optional direct ``state name -> (object, attribute)`` bindings for the flow hot path.

        The default covers loops whose public state names map directly to their attributes. A
        loop with a custom/nested state representation may return ``None`` and keep using
        :meth:`set_state`, or override this method with equivalent direct bindings.
        """
        names = self.state_names
        attrs = names if isinstance(names, Mapping) else {name: name for name in names}
        bindings = {name: (self, attr) for name, attr in attrs.items()}
        for prefix, block in self._state_blocks.items():
            own = block.state_bindings()
            if own is None:
                return None
            bindings.update({join(prefix, name): binding for name, binding in own.items()})
        if set(self.flow_state()) != set(bindings):
            return None
        return bindings

    def get_state(self) -> dict[str, Any]:
        names = self.state_names
        attrs = names if isinstance(names, Mapping) else {name: name for name in names}
        state = {name: getattr(self, attr) for name, attr in attrs.items()}
        state.update(gather(self._state_blocks))
        return state

    def set_state(self, values: Mapping[str, Any]) -> None:
        unknown = set(values) - set(self.get_state())
        if unknown:
            raise KeyError(f"{type(self).__name__} has no state(s) {sorted(unknown)}")
        block_states = set(gather(self._state_blocks))
        scatter(self._state_blocks, {name: value for name, value in values.items()
                                     if name in block_states})
        assign(self, {name: value for name, value in values.items() if name not in block_states},
               self.state_names)

    def reset_integrator(self) -> None:
        """Clear this loop's error integrator, if it has one, before PWM start-up."""

    def retuned(self, old: "Loop") -> None:
        """Take non-state runtime attributes from the loop instance being replaced."""
        for name in self.carried:
            setattr(self, name, getattr(old, name))


LOOP_TYPES: dict[str, type] = {}


def loop_type(name: str, where: str = "ctrl.loops") -> type:
    """Return a registered loop class by internal type or public Python class name.

    ``LOOP_TYPES`` remains the one authoritative registry. Class-name lookup is needed only while
    loading configuration and deliberately does not add a second registry or any runtime lookup.
    """
    if isinstance(name, str):
        cls = LOOP_TYPES.get(name)
        if cls is not None:
            return cls
        cls = next((candidate for candidate in LOOP_TYPES.values()
                    if candidate.__name__ == name), None)
        if cls is not None:
            return cls
    known = sorted({*LOOP_TYPES, *(cls.__name__ for cls in LOOP_TYPES.values())})
    raise ConfigError(f"{where}: unknown loop {name!r}; known: {known}")


def loop_key_type(name: str, where: str = "ctrl.loops") -> type | None:
    """Resolve ``ClassName`` or ``ClassName[label]``; return ``None`` for a legacy node name."""
    class_name = name
    if "[" in name or "]" in name:
        if not name.endswith("]") or name.count("[") != 1 or name.count("]") != 1:
            raise ConfigError(f"{where}: malformed class-qualified loop name {name!r}")
        class_name, label = name[:-1].split("[", 1)
        if not class_name or not label:
            raise ConfigError(f"{where}: malformed class-qualified loop name {name!r}")
    try:
        cls = loop_type(class_name, where)
    except ConfigError:
        return None
    return cls if cls.__name__ == class_name else None


def register_loop_type(cls: type) -> type:
    """Register ``cls`` (see :class:`Loop`) under its internal ``cls.type`` name.

    Simulation files select the implementation with the Python class name. Its parameters are
    then checked like those of the built-in loops. The class is returned so this function can be
    used as a decorator.
    """
    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not name:
        raise TypeError("a loop type needs a name: the class attribute 'type'")
    if name in LOOP_TYPES:
        raise ValueError(f"loop type {name!r} is already registered")
    if any(registered.__name__ == cls.__name__ for registered in LOOP_TYPES.values()):
        raise ValueError(f"loop class name {cls.__name__!r} is already registered")
    params = getattr(cls, "Params", None)
    if not is_dataclass(params) or not {"type", "period"} <= set(params.__dataclass_fields__):
        raise TypeError(f"loop type {name!r}: Params must be a dataclass with type and period")
    if params.__dataclass_fields__["type"].default != name:
        raise TypeError(f"loop type {name!r}: Params.type must default to {name!r}")
    if cls.role is not None and cls.role not in ROLES:
        raise ValueError(f"loop type {name!r}: role {cls.role!r} is not one of {ROLES}")
    LOOP_TYPES[name] = cls
    return cls


def _into(frame: float):
    """The rotation from alpha-beta into the dq frame of angle ``frame`` (rad)."""
    return np.exp(-1j * frame)


@dataclass(frozen=True)
class PowerFilterParams:
    """Optional first-order low-pass filter on measured active and reactive power."""

    enable: bool = False
    bandwidth: Optional[float] = None  # Hz

    def __post_init__(self) -> None:
        if self.bandwidth is not None and (
                not math.isfinite(self.bandwidth) or self.bandwidth <= 0.0):
            raise ConfigError("power-filter bandwidth must be finite and positive")
        if self.enable and self.bandwidth is None:
            raise ConfigError("enabled power filter needs a bandwidth")


@dataclass(frozen=True)
class CurrentLimitParams:
    """Optional current-reference magnitude limit."""

    enable: bool = False
    limit_pu: Optional[float] = None

    _quantities = {"limit_pu": "current"}


# ------------------------------------------------------------------ grid following

@register_loop_type
class SRFPLL(Loop):
    """SRF-PLL: ``omega = w0 + kp*eps + ki*integral``, ``theta += T*omega``.

    Input: the voltage, rotated into the frame of the previous angle. ``normalisation``:
    ``"rated"`` (eps = v_q) or ``"amplitude"`` (eps = v_q / u_g, u_g a filtered v_d estimate).
    Named states: ``theta`` (rad, unwrapped), ``integral`` (pu*s), ``u_g`` (pu, amplitude only).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        kp_pu: float
        ki_pu: float
        normalisation: str = "rated"
        period: Optional[float] = None
        type: str = "srf_pll"

        _choices = {"normalisation": ("rated", "amplitude")}
        _quantities = {"kp_pu": "1/voltage", "ki_pu": "1/voltage"}

    type = "srf_pll"
    role = "pll"
    outputs_from_state = True
    inputs = {"v": V_AB}
    outputs = {"theta": ANGLE, "frame": ANGLE, "omega": FREQUENCY}
    flow_inputs = ("v",)

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.kp, self.ki, self.w0, self.T = cfg.kp_pu, cfg.ki_pu, unit.base.w0, cfg.period
        self.pi = PI(self.kp, self.ki, self.block_period)
        self.theta, self.omega = 0.0, self.w0
        self.u_g = 1.0 if cfg.normalisation == "amplitude" else None
        self.state_names = {"theta": "theta", "integral_pu": "integral",
                            **({} if self.u_g is None else {"u_g_pu": "u_g"})}

    @property
    def integral(self):
        return self.pi.integral

    @integral.setter
    def integral(self, value):
        self.pi.integral = value

    def initial_outputs(self):
        return {"theta": self.theta, "frame": self.theta, "omega": self.omega}

    def sample(self, t, inputs):
        frame = self.theta
        v = inputs["v"] * _into(frame)
        u_g = self.u_g
        if u_g is None:                                    # rated
            eps = v.imag
        else:                                              # amplitude
            eps = v.imag / u_g if u_g > 0.0 else 0.0
        self.omega = self.pi(eps, 0.0, self.w0)
        self.theta += self.T * self.omega
        if u_g is not None:
            self.u_g = u_g + self.T * self.kp * (v.real - u_g)
        return {"theta": self.theta, "frame": frame, "omega": self.omega}

    def flow_path(self, voltage):
        """Positional form used by the construction-time compiled continuous graph."""
        v = voltage * _into(self.theta)
        if self.u_g is None:
            eps = v.imag
        else:
            eps = v.imag / self.u_g if self.u_g > 0.0 else 0.0
        self.omega = self.pi(eps, 0.0, self.w0)
        d_integral = self.pi.derivative
        if self.u_g is not None:
            return (self.theta, self.theta, self.omega, self.omega, d_integral,
                    self.kp * (v.real - self.u_g))
        return self.theta, self.theta, self.omega, self.omega, d_integral

    def flow_state_bindings(self):
        bindings = {"theta": (self, "theta"), "integral_pu": (self.pi, "integral")}
        if self.u_g is not None:
            bindings["u_g_pu"] = (self, "u_g")
        return bindings

    def reset_integrator(self):
        self.pi.reset()


@register_loop_type
class CurrentLoop(Loop):
    """dq PI current loop: ``u = u_ff + (r_pu + j*omega/w0*x_pu)*i + kp*e + ki*integral`` (+ ``extra``).

    Gains default from ``bandwidth`` (Hz) and the unit's filter. The controller may constrain
    ``u_dq`` and feed the applied output back to this PI's tracking anti-windup. Named state:
    ``integral_pu`` (complex dq, pu*s).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bandwidth: float  # Hz
        kp_pu: Optional[float] = None  # pu voltage / pu current
        ki_pu: Optional[float] = None  # pu voltage / (pu current * s)
        decoupling: bool = True
        feedforward: bool = True
        antiwindup: bool = True
        period: Optional[float] = None
        type: str = "dq_current_pi"

        _quantities = {"kp_pu": "resistance", "ki_pu": "resistance"}

    type = "dq_current_pi"
    role = "current"
    inputs = {"id_ref": I, "iq_ref": I,
              "v": V_AB,
              "i": I_AB,
              "frame": ANGLE, "omega": FREQUENCY,
              "extra": V_DQ}
    outputs = {"u_dq": V_DQ, "extra": V_DQ}
    flow_inputs = ("id_ref", "iq_ref", "v", "i", "frame", "omega", "extra")

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        x = unit.ac_filter.l_f * unit.base.w0 / unit.base.z_base
        r = unit.ac_filter.r_f / unit.base.z_base
        self.kp = cfg.kp_pu if cfg.kp_pu is not None else x / unit.base.w0 * 2 * math.pi * cfg.bandwidth
        self.ki = cfg.ki_pu if cfg.ki_pu is not None else r * 2 * math.pi * cfg.bandwidth
        self.x_pu, self.r_pu, self.w0, self.T = x, r, unit.base.w0, cfg.period
        self.decoupling, self.feedforward = cfg.decoupling, cfg.feedforward
        self.antiwindup = cfg.antiwindup
        self.pi = PI(self.kp, self.ki, self.block_period,
                     antiwindup=cfg.antiwindup, initial=0j)
        self.extra = 0j

    @property
    def integral(self):
        return self.pi.integral

    @integral.setter
    def integral(self, value):
        self.pi.integral = complex(value)

    def initial_outputs(self):
        return {"u_dq": 0j, "extra": 0j}

    def sample(self, t, inputs):
        rot = _into(inputs["frame"])
        self.extra = inputs["extra"]
        i, u_ff, omega = inputs["i"] * rot, inputs["v"] * rot, inputs["omega"]
        reference = i if not self.startup.active else inputs["id_ref"] + 1j * inputs["iq_ref"]
        ff = self.feedforward_voltage(i, u_ff, omega, self.extra)
        return {"u_dq": self.pi(reference, i, ff), "extra": self.extra}

    def flow_path(self, id_ref, iq_ref, voltage, current, frame, omega, extra):
        """Positional form used by the construction-time compiled continuous graph."""
        rot = _into(frame)
        self.extra = extra
        i, u_ff = current * rot, voltage * rot
        reference = i if not self.startup.active else id_ref + 1j * iq_ref
        ff = self.feedforward_voltage(i, u_ff, omega, self.extra)
        output = self.pi(reference, i, ff)
        return output, self.extra, self.pi.derivative

    def feedforward_voltage(self, i: complex, u_ff: complex, omega: float,
                            extra: complex = 0j) -> complex:
        """Feed-forward and decoupling part of the voltage command."""
        u = extra
        if self.feedforward:
            u += u_ff
        if self.decoupling:
            u += (self.r_pu + 1j * omega / self.w0 * self.x_pu) * i
        return u

    def get_state(self):
        return {"integral_pu": self.integral}

    def flow_state_bindings(self):
        return {"integral_pu": (self.pi, "integral")}

    def set_state(self, values):
        unknown = set(values) - {"integral_pu"}
        if unknown:
            raise KeyError(f"dq_current_pi has no state(s) {sorted(unknown)}")
        if "integral_pu" in values:
            self.integral = values["integral_pu"]

    def reset_integrator(self):
        self.pi.reset()


@register_loop_type
class DCVoltageControl(Loop):
    """Unit-system-independent voltage PI producing a bounded current reference.

    The error is always ``reference - voltage``. ``output_polarity`` is ``+1`` when positive
    commanded current raises the controlled voltage and ``-1`` when it lowers the voltage, as for
    a VSC whose positive d-axis current exports power from its DC side. The feed-forward follows
    the common converter start-up ramp.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        kp: float
        ki: float
        minimum: float
        maximum: float
        feedforward: float = 0.0
        output_polarity: float = 1.0
        antiwindup: bool = True
        period: Optional[float] = None
        type: str = "dc_voltage_control"

        _choices = {"output_polarity": (-1.0, 1.0)}

    type = "dc_voltage_control"
    role = "dc_voltage"
    inputs = {"reference": GENERIC_VOLTAGE, "voltage": GENERIC_VOLTAGE}
    outputs = {"current_reference": GENERIC_CURRENT}
    flow_inputs = ("reference", "voltage")
    carried = ("n_updates", "n_limit_exceeded", "n_reverse", "first_limit_t")

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.kp = cfg.output_polarity * cfg.kp
        self.ki = cfg.output_polarity * cfg.ki
        self.T = cfg.period
        self.minimum, self.maximum = cfg.minimum, cfg.maximum
        self.antiwindup = cfg.antiwindup
        self.pi = self.state_block(
            "integral", PI(self.kp, self.ki, self.block_period,
                           antiwindup=cfg.antiwindup)
        )
        self.limit = OutputLimit(cfg.maximum, lower=cfg.minimum, pi=self.pi)
        self.error = self.raw = 0.0
        self.limit_exceeded = False
        self.n_updates = self.n_limit_exceeded = 0
        self.n_reverse = 0
        self.first_limit_t: float | None = None

    @property
    def integral(self):
        return self.pi.integral

    @integral.setter
    def integral(self, value):
        self.pi.integral = float(value)

    def initial_outputs(self):
        return {"current_reference": 0.0}

    def _requested(self, reference, voltage):
        self.error = reference - voltage
        if self.startup is not None and not self.startup.active:
            self.pi.derivative = 0.0
            return 0.0
        ramp = 1.0 if self.startup is None else self.startup.value
        return self.pi(reference, voltage, ramp * self.cfg.feedforward)

    def equation(self, reference, voltage):
        self.raw = self._requested(reference, voltage)
        current = (self.limit.flow(self.raw) if self.block_period is None
                   else self.limit.sample(0.0, self.raw))
        return (current,)

    def sample(self, t, inputs):
        current, = self.equation(inputs["reference"], inputs["voltage"])
        self.n_updates += 1
        self.observe_limit(t, self.raw, current, count=True)
        if self.raw < 0.0:
            self.n_reverse += 1
        return {"current_reference": current}

    def observe_limit(self, t, intended, output, *, count):
        """Record whether the requested current lies outside the configured range."""
        self.limit_exceeded = output != intended
        if count and self.limit_exceeded:
            self.n_limit_exceeded += 1
            if self.first_limit_t is None:
                self.first_limit_t = t

    def reset_integrator(self):
        self.pi.reset()
        self.limit_exceeded = False


# ------------------------------------------------------------------ grid forming


@dataclass(frozen=True, kw_only=True)
class SyncLawParams:
    """Parameters shared by grid-forming synchronization laws."""

    period: Optional[float] = None
    power_filter: PowerFilterParams = field(default_factory=PowerFilterParams)


class SyncLaw(Loop):
    """Base of the grid-forming synchronization laws: angle (rad), frequency (rad/s) and voltage
    magnitude reference (pu) from the power feedback, the setpoints and the measured voltage.

    A law implements ``step(T, p, q, v_mag, v_dc, p_ref, q_ref, v_ref, i_dq)``, which advances
    ``theta``, ``omega`` and ``v_mag`` by one period ``T``; its inputs are in pu (the voltage
    magnitude on the ac base, ``v_dc`` on the dc base) and ``i_dq`` is the current in the law's frame.
    The active-power setpoint follows the controller's start-up ramp. Before it runs, a grid-forming
    law follows the measured terminal voltage so that it starts in phase with the grid.
    """

    role = "sync"
    outputs_from_state = True
    inputs = {"v": V_AB, "i": I_AB, "u_dc": V,
              "p_ref": POWER, "q_ref": POWER, "v_ref": V}
    outputs = {"theta": ANGLE, "frame": ANGLE, "omega": FREQUENCY, "v_ref": V,
               "p": POWER, "q": POWER}
    state_names = ("theta",)
    flow_inputs = ("v", "i", "u_dc", "p_ref", "q_ref", "v_ref")

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.w0, self.T = unit.base.w0, cfg.period
        self.theta, self.omega = 0.0, self.w0
        self.v_mag = unit.ctrl.references.v_ref_pu
        power_filter = getattr(cfg, "power_filter", None)
        if power_filter is not None and power_filter.enable:
            pole = 2.0 * math.pi * power_filter.bandwidth
            self.lpf_p = self.state_block(
                "p_pu", Filter((pole,), (1.0, pole), self.block_period)
            )
            self.lpf_q = self.state_block(
                "q_pu", Filter((pole,), (1.0, pole), self.block_period)
            )
            self._measure_power = self._filtered_power
        else:
            self.lpf_p = self.lpf_q = None
            self._measure_power = self._instantaneous_power

    @staticmethod
    def _instantaneous_power(voltage, current):
        power = complex_power(voltage, current)
        return power.real, power.imag

    def _filtered_power(self, voltage, current):
        power = complex_power(voltage, current)
        return self.lpf_p(power.real), self.lpf_q(power.imag)

    def initial_outputs(self):
        return {"theta": self.theta, "frame": self.theta, "omega": self.omega,
                "v_ref": self.unit.ctrl.references.v_ref_pu, "p": 0.0, "q": 0.0}

    def sample(self, t, inputs):
        if not self.startup.active:
            out = self.track(inputs)
            if out is not None:
                return out
            p, q = self._measure_power(inputs["v"], inputs["i"])
            return {"theta": self.theta, "frame": self.theta, "omega": self.omega,
                    "v_ref": self.v_mag, "p": p, "q": q}
        p, q = self._measure_power(inputs["v"], inputs["i"])
        frame = self.theta
        rot = _into(frame)
        v, i = inputs["v"] * rot, inputs["i"] * rot
        p_ref = self.startup.value * inputs["p_ref"]
        q_ref, v_ref = inputs["q_ref"], inputs["v_ref"]
        self._sample(
            self.T, p, q, abs(v), inputs["u_dc"],
            p_ref, q_ref, v_ref, i,
        )
        return {"theta": self.theta, "frame": frame, "omega": self.omega,
                "v_ref": self.v_mag, "p": p, "q": q}

    def _sample(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu,
                p_ref_pu, q_ref_pu, v_ref_pu, i_dq) -> None:
        raise NotImplementedError

    def flow_path(self, voltage, current, u_dc, p_ref, q_ref, v_ref):
        """Shared continuous evaluation for all synchronization laws."""
        p, q = self._measure_power(voltage, current)
        block_derivatives = self._block_state_derivatives()
        frame = self.theta
        rot = _into(frame)
        v, i = voltage * rot, current * rot
        if not self.startup.active:
            self.omega = self.w0
            magnitude = abs(v)
            if magnitude > 0.1:
                self.v_mag = magnitude
            derivatives = getattr(self, "_inactive_flow", None)
            if derivatives is None:
                derivatives = (self.w0,) + (0.0,) * (len(self.state_names) - 1)
                self._inactive_flow = derivatives
            return (self.theta, frame, self.omega, self.v_mag, p, q,
                    *derivatives, *block_derivatives)
        derivatives = self._flow(
            p, q, abs(v), u_dc, self.startup.value * p_ref, q_ref, v_ref, i
        )
        return (self.theta, frame, self.omega, self.v_mag, p, q,
                *derivatives, *block_derivatives)

    def _flow(self, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu,
              q_ref_pu, v_ref_pu, i_dq) -> tuple[Any, ...]:
        raise NotImplementedError

    def track(self, inputs):
        """Hold the law on a usable measured voltage before start-up."""
        voltage = inputs["v"]
        if abs(voltage) <= 0.1:
            return None
        p, q = self._measure_power(voltage, inputs["i"])
        theta = self.theta
        rotated = voltage * np.exp(-1j * theta)
        delta = np.arctan2(rotated.imag, rotated.real)
        theta += delta
        self.follow(theta, abs(voltage), inputs["v_ref"])
        return {"theta": theta, "frame": theta, "omega": self.omega,
                "v_ref": self.v_mag, "p": p, "q": q}

    def follow(self, theta: float, v_mag_pu: float, v_ref_pu: float) -> None:
        self.theta, self.omega, self.v_mag = theta, self.w0, v_mag_pu


@register_loop_type
class PSC(SyncLaw):
    """Power-synchronization control ``w = w0 + k_p (P* - P)`` with a PI voltage-magnitude loop.

    Named states: ``theta`` (rad), ``v_int`` (pu*s).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params(SyncLawParams):
        k_p_pu: float  # rad/s per pu
        k_v: float = 0.0  # AVR proportional gain, pu/pu (0: open-loop magnitude)
        k_vi: float = 0.0  # AVR integral gain, 1/s
        type: str = "psc"

        _quantities = {"k_p_pu": "1/power"}

    type = "psc"
    state_names = {"theta": "theta", "v_int_pu": "v_int"}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.k_p, self.k_v, self.k_vi = cfg.k_p_pu, cfg.k_v, cfg.k_vi
        self.voltage_pi = PI(self.k_v, self.k_vi, self.block_period)

    @property
    def v_int(self):
        return self.voltage_pi.integral

    @v_int.setter
    def v_int(self, value):
        self.voltage_pi.integral = value

    def _sample(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu,
                p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0 + self.k_p * (p_ref_pu - p_pu)
        self.theta += T * self.omega
        self.v_mag = self.voltage_pi(v_ref_pu, v_mag_pu, v_ref_pu)

    def _flow(self, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu,
              q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0 + self.k_p * (p_ref_pu - p_pu)
        self.v_mag = self.voltage_pi(v_ref_pu, v_mag_pu, v_ref_pu)
        return self.omega, self.voltage_pi.derivative

    def flow_state_bindings(self):
        bindings = super().flow_state_bindings()
        if bindings is not None:
            bindings["v_int_pu"] = (self.voltage_pi, "integral")
        return bindings

    def follow(self, theta, v_mag_pu, v_ref_pu):
        super().follow(theta, v_mag_pu, v_ref_pu)
        error = v_ref_pu - v_mag_pu
        self.v_int = ((v_mag_pu - v_ref_pu - self.k_v * error) / self.k_vi
                      if self.k_vi > 0.0 else 0.0)

    def reset_integrator(self):
        self.voltage_pi.reset()


@register_loop_type
class Droop(SyncLaw):
    """P-f / Q-V droop control. Named state: ``theta`` (rad)."""

    @dataclass(frozen=True, kw_only=True)
    class Params(SyncLawParams):
        m_p_pu: float  # rad/s per pu (P-f droop)
        n_q_pu: float  # pu/pu (Q-V droop)
        type: str = "droop"

        _quantities = {"m_p_pu": "1/power", "n_q_pu": "voltage/power"}

    type = "droop"

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.m_p, self.n_q = cfg.m_p_pu, cfg.n_q_pu

    def _sample(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu,
                p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0 + self.m_p * (p_ref_pu - p_pu)
        self.theta += T * self.omega
        self.v_mag = v_ref_pu + self.n_q * (q_ref_pu - q_pu)

    def _flow(self, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu,
              q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.w0 + self.m_p * (p_ref_pu - p_pu)
        self.v_mag = v_ref_pu + self.n_q * (q_ref_pu - q_pu)
        return (self.omega,)


@register_loop_type
class VSG(SyncLaw):
    """Virtual synchronous generator: ``2H dw/dt = p* - p - d_p (w - w0)/w0`` and a Q-V droop.

    Named states: ``theta`` (rad), ``dw_pu`` and, if ``t_q > 0``, ``v_mag`` (pu).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params(SyncLawParams):
        h: float  # inertia constant, s
        d_p_pu: float  # damping, pu power per pu speed deviation
        k_q_pu: float  # Q-V droop, pu/pu
        t_q: float = 0.0  # voltage-magnitude lag time constant, s (0: none)
        type: str = "vsg"

        _quantities = {"d_p_pu": "power/frequency", "k_q_pu": "voltage/power"}

    type = "vsg"

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.h, self.d_p, self.k_q, self.t_q = cfg.h, cfg.d_p_pu, cfg.k_q_pu, cfg.t_q
        self.dw_pu = 0.0
        self.state_names = {"theta": "theta", "dw_pu": "dw_pu",
                            **({"v_mag_pu": "v_mag"} if self.t_q > 0.0 else {})}

    def _sample(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu,
                p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.dw_pu += T / (2.0 * self.h) * (p_ref_pu - p_pu - self.d_p * self.dw_pu)
        self.omega = self.w0 * (1.0 + self.dw_pu)
        self.theta += T * self.omega
        v_cmd = v_ref_pu + self.k_q * (q_ref_pu - q_pu)
        if self.t_q > 0.0:
            self.v_mag += T / self.t_q * (v_cmd - self.v_mag)
        else:
            self.v_mag = v_cmd

    def _flow(self, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu,
              q_ref_pu, v_ref_pu, i_dq):
        d_dw = (p_ref_pu - p_pu - self.d_p * self.dw_pu) / (2.0 * self.h)
        self.omega = self.w0 * (1.0 + self.dw_pu)
        v_cmd = v_ref_pu + self.k_q * (q_ref_pu - q_pu)
        if self.t_q > 0.0:
            return self.omega, d_dw, (v_cmd - self.v_mag) / self.t_q
        self.v_mag = v_cmd
        return self.omega, d_dw

    def follow(self, theta, v_mag_pu, v_ref_pu):
        super().follow(theta, v_mag_pu, v_ref_pu)
        self.dw_pu = 0.0


@register_loop_type
class DVOC(SyncLaw):
    """Dispatchable virtual oscillator control, integrated in polar form.

    ``dv/dt = j w0 v + eta e^{j kappa} ((p* - j q*) v / V*^2 - i) + eta alpha (1 - |v|^2 / V*^2) v``
    with oscillator voltage ``v`` and current ``i`` in pu. Named states: ``theta`` (rad), ``v_mag`` (pu).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params(SyncLawParams):
        eta_pu: float  # rad/s
        alpha_pu: float  # magnitude regulation gain
        kappa: float  # rotation of the current error, rad
        type: str = "dvoc"

        _quantities = {"eta_pu": "resistance", "alpha_pu": "1/resistance"}

    type = "dvoc"
    state_names = {"theta": "theta", "v_mag_pu": "v_mag"}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.eta, self.alpha = cfg.eta_pu, cfg.alpha_pu
        self.rot = cmath.exp(1j * cfg.kappa)

    def _sample(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu,
                p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        V = self.v_mag
        i_star = (p_ref_pu - 1j * q_ref_pu) * V / (v_ref_pu * v_ref_pu)
        w = self.eta * self.rot * (i_star - i_dq) + self.eta * self.alpha * (1.0 - V * V / (v_ref_pu * v_ref_pu)) * V
        self.omega = self.w0 + w.imag / V
        self.v_mag = V + T * w.real
        self.theta += T * self.omega

    def _flow(self, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu,
              q_ref_pu, v_ref_pu, i_dq):
        V = max(self.v_mag, 1e-12)
        i_star = (p_ref_pu - 1j * q_ref_pu) * V / (v_ref_pu * v_ref_pu)
        w = (self.eta * self.rot * (i_star - i_dq)
             + self.eta * self.alpha * (1.0 - V * V / (v_ref_pu * v_ref_pu)) * V)
        self.omega = self.w0 + w.imag / V
        return self.omega, w.real


@register_loop_type
class Matching(SyncLaw):
    """Matching control ``w = k_theta * v_dc``; ``k_theta_pu`` defaults to ``w0 / vdc_ref_pu``.

    Named state: ``theta`` (rad).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params(SyncLawParams):
        k_theta_pu: Optional[float] = None  # rad/s per pu dc voltage (dc base)
        k_q_pu: float = 0.0  # Q-V droop, pu/pu
        type: str = "matching"

        _quantities = {"k_theta_pu": "1/dc_voltage", "k_q_pu": "voltage/power"}

    type = "matching"

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        vdc_ref_pu = unit.ctrl.references.vdc_ref_pu
        self.k_theta = cfg.k_theta_pu if cfg.k_theta_pu is not None else self.w0 / vdc_ref_pu
        self.k_q = cfg.k_q_pu

    def _sample(self, T, p_pu, q_pu, v_mag_pu, v_dc_pu,
                p_ref_pu, q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.k_theta * v_dc_pu
        self.theta += T * self.omega
        self.v_mag = v_ref_pu + self.k_q * (q_ref_pu - q_pu)

    def _flow(self, p_pu, q_pu, v_mag_pu, v_dc_pu, p_ref_pu,
              q_ref_pu, v_ref_pu, i_dq):
        self.omega = self.k_theta * v_dc_pu
        self.v_mag = v_ref_pu + self.k_q * (q_ref_pu - q_pu)
        return (self.omega,)


# ------------------------------------------------------------------ voltage-reference shaping

@register_loop_type
class VirtualImpedance(Loop):
    """Voltage command ``v_ref - (r_pu + j omega/w0 x_pu) i`` (+ ``extra``) behind a virtual impedance.

    Without a ``period`` it runs whenever the loops before it update.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        r_v_pu: float = 0.0
        x_v_pu: float = 0.0
        period: Optional[float] = None  # None: runs on upstream updates
        type: str = "virtual_impedance"

        _quantities = {"r_v_pu": "resistance", "x_v_pu": "resistance"}

    type = "virtual_impedance"
    role = "impedance"
    inputs = {"v_ref": V, "i": I_AB, "frame": ANGLE, "omega": FREQUENCY,
              "extra": V_DQ}
    outputs = {"u_dq": V_DQ}
    flow_inputs = ("v_ref", "i", "frame", "omega", "extra")

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.r_pu, self.x_pu, self.w0 = cfg.r_v_pu, cfg.x_v_pu, unit.base.w0

    def initial_outputs(self):
        return {"u_dq": 0j}

    def sample(self, t, inputs):
        i = inputs["i"] * _into(inputs["frame"])
        drop = (self.r_pu + 1j * inputs["omega"] / self.w0 * self.x_pu) * i
        return {"u_dq": inputs["v_ref"] - drop + inputs["extra"]}

    def flow_path(self, v_ref, current, frame, omega, extra):
        i = current * _into(frame)
        drop = (self.r_pu + 1j * omega / self.w0 * self.x_pu) * i
        return (v_ref - drop + extra,)


@register_loop_type
class VirtualAdmittance(Loop):
    """Current reference from ``x_pu/w0 di/dt = v_ref - v - (r_pu + j omega/w0 x_pu) i``,
    optionally magnitude-limited. Named state: ``i_ref_pu`` (dq, pu)."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: Optional[float] = None
        x_v_pu: float
        r_v_pu: float = 0.0
        current_limit: CurrentLimitParams = field(default_factory=CurrentLimitParams)
        type: str = "virtual_admittance"

        _quantities = {"r_v_pu": "resistance", "x_v_pu": "resistance"}

        def __post_init__(self) -> None:
            if not self.x_v_pu > 0.0:
                raise ConfigError("x_v_pu must be positive")

    type = "virtual_admittance"
    role = "admittance"
    inputs = {"v_ref": V, "v": V_AB, "frame": ANGLE, "omega": FREQUENCY}
    outputs = {"id_ref": I, "iq_ref": I}
    flow_inputs = ("v_ref", "v", "frame", "omega")

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.r_pu, self.x_pu, self.w0, self.T = cfg.r_v_pu, cfg.x_v_pu, unit.base.w0, cfg.period
        self.i_limit = cfg.current_limit.limit_pu if cfg.current_limit.enable else math.inf
        self.i_ref = 0j

    def initial_outputs(self):
        return {"id_ref": 0.0, "iq_ref": 0.0}

    def sample(self, t, inputs):
        v_ref_dq, v_dq, omega = inputs["v_ref"], inputs["v"] * _into(inputs["frame"]), inputs["omega"]
        i_ref = self.i_ref + self.T * self.w0 / self.x_pu * (
            v_ref_dq - v_dq - (self.r_pu + 1j * omega / self.w0 * self.x_pu) * self.i_ref)
        mag = abs(i_ref)
        if mag > self.i_limit:
            i_ref *= self.i_limit / mag
        self.i_ref = i_ref
        return {"id_ref": i_ref.real, "iq_ref": i_ref.imag}

    def flow_path(self, v_ref, voltage, frame, omega):
        """Positional form used by the construction-time compiled continuous graph."""
        v_ref_dq = v_ref
        v_dq = voltage * _into(frame)
        i_ref = self.i_ref
        mag = abs(i_ref)
        i_out = i_ref if mag <= self.i_limit else i_ref * (self.i_limit / mag)
        derivative = self.w0 / self.x_pu * (
            v_ref_dq - v_dq
            - (self.r_pu + 1j * omega / self.w0 * self.x_pu) * i_out
        )
        if mag >= self.i_limit:
            radial = (derivative * i_out.conjugate()).real
            if radial > 0.0:
                derivative -= radial / (self.i_limit * self.i_limit) * i_out
        return i_out.real, i_out.imag, derivative

    def get_state(self):
        return {"i_ref_pu": self.i_ref}

    def flow_state_bindings(self):
        return {"i_ref_pu": (self, "i_ref")}

    def set_state(self, values):
        if set(values) - {"i_ref_pu"}:
            raise KeyError("virtual_admittance has the state i_ref_pu (current pu)")
        if "i_ref_pu" in values:
            self.i_ref = complex(values["i_ref_pu"])


@register_loop_type
class ActiveDamping(Loop):
    """Damping voltage ``-r_a`` times the current high-pass filtered at ``alpha_d`` (rad/s).

    Named state: ``hpf_pu`` (the low-passed current, pu; starts at zero).
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: Optional[float] = None
        r_a_pu: float
        alpha_d: float
        type: str = "active_damping"

        _quantities = {"r_a_pu": "resistance"}

    type = "active_damping"
    role = "damping"
    inputs = {"i": I_AB, "frame": ANGLE}
    outputs = {"extra": V_DQ}
    flow_inputs = ("i", "frame")

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.r_a = cfg.r_a_pu
        self.alpha_d = cfg.alpha_d
        self.lpf = Filter((cfg.alpha_d,), (1.0, cfg.alpha_d),
                          self.block_period, initial=0j)

    def initial_outputs(self):
        return {"extra": 0j}

    def sample(self, t, inputs):
        current = inputs["i"] * _into(inputs["frame"])
        return {"extra": -self.r_a * (current - self.lpf(current))}

    def flow_path(self, current, frame):
        current_dq = current * _into(frame)
        delta = current_dq - self.lpf(current_dq)
        return -self.r_a * delta, self.lpf.derivative[0]

    def get_state(self):
        return gather({"hpf_pu": self.lpf})

    def flow_state_bindings(self):
        return {"hpf_pu": (self.lpf, "x0")}

    def set_state(self, values):
        scatter({"hpf_pu": self.lpf}, values)


# ------------------------------------------------------------------ scalar SI converter loops

SI_CURRENT = SignalType("A", "scalar")
SI_VOLTAGE = SignalType("V", "scalar")
DUTY_RATIO = SignalType("1", "duty_ratio")
MODULATION = SignalType("1", "modulation")
HERTZ = SignalType("Hz")


@dataclass(frozen=True, kw_only=True)
class _ScalarPIParams:
    kp: float
    ki: float
    minimum: float
    maximum: float
    period: Optional[float] = None


class _LimitedScalarPI(Loop):
    """One registered PI state shared by sampled and continuous execution."""

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.pi = self.state_block("integral", PI(cfg.kp, cfg.ki, self.block_period))
        self.limit = OutputLimit(cfg.maximum, lower=cfg.minimum, pi=self.pi)

    def constrain(self, requested):
        if self.block_period is None:
            return self.limit.flow(requested)
        return self.limit.sample(0.0, requested)

    def initial_outputs(self):
        return {port: self.cfg.minimum for port in self.outputs}

    def reset_integrator(self):
        self.pi.reset()


@register_loop_type
class InductorCurrentControl(_LimitedScalarPI):
    """Topology-independent inductor-current PI with an external duty feedforward."""

    @dataclass(frozen=True, kw_only=True)
    class Params(_ScalarPIParams):
        type: str = "inductor_current_control"

    type = "inductor_current_control"
    role = "current"
    inputs = {"reference": SI_CURRENT, "current": SI_CURRENT,
              "feedforward": DUTY_RATIO}
    outputs = {"duty": DUTY_RATIO, "requested_duty": DUTY_RATIO}
    flow_inputs = ("reference", "current", "feedforward")

    def equation(self, reference, current, feedforward):
        requested = self.pi(reference, current, feedforward)
        return self.constrain(requested), requested


@register_loop_type
class ResonantModulation(Loop):
    """Temporary open-loop sinusoidal modulation for a resonant converter application.

    It is an ordinary registered loop: the DC converter still uses the same control graph,
    scheduling and bridge boundary as feedback-controlled applications. A later feedback loop can
    replace this loop without changing the converter implementation.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        period: Optional[float] = None
        type: str = "resonant_modulation"

    type = "resonant_modulation"
    role = "modulation"
    inputs = {"modulation": MODULATION, "frequency": HERTZ, "phase": ANGLE}
    outputs = {"duty": DUTY_RATIO}

    def initial_outputs(self):
        return {"duty": 0.5}

    def _duty(self, t, inputs):
        q = self.startup.value * inputs["modulation"] * np.sin(
            2.0 * math.pi * inputs["frequency"] * t + inputs["phase"]
        )
        return 0.5 * (1.0 + q)

    def sample(self, t, inputs):
        return {"duty": self._duty(t, inputs)}

    def flow(self, t, inputs):
        return {"duty": self._duty(t, inputs)}, {}
