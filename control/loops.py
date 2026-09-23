"""Typed loop definitions, the loop-type registry and adapters for the built-in control loops."""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, is_dataclass
from typing import Any, Callable, Mapping

from ..params.loops import LOOP_SCHEMAS
from .current import CurrentController
from .dc_voltage import DCVoltageController
from .pll import SRFPLL
from .power import PowerCalculator
from .shaping import ActiveDamping, VirtualAdmittance, VirtualImpedance
from .sync import PSC, Droop, VSG, DVOC, Matching


@dataclass(frozen=True)
class SignalType:
    unit: str
    frame: str = "scalar"
    complex_value: bool = False


V_AB = SignalType("pu_voltage_ac", "alpha_beta", True)
I_AB = SignalType("pu_current_ac", "alpha_beta", True)
V_DQ = SignalType("pu_voltage_ac", "dq", True)
VOLTAGE = SignalType("pu_voltage_ac")
DC_VOLTAGE = SignalType("pu_voltage_dc")
CURRENT = SignalType("pu_current_ac")
ANGLE = SignalType("rad")
FREQUENCY = SignalType("rad/s")
POWER_PU = SignalType("pu_power")


@dataclass(frozen=True)
class LoopDefinition:
    parameters: type
    inputs: Mapping[str, SignalType]
    outputs: Mapping[str, SignalType]
    factory: Callable  # (loop parameters, unit parameters, scenario) -> loop
    delayed_inputs: frozenset[str] = frozenset()


LOOP_TYPES: dict[str, LoopDefinition] = {}


def register_loop_type(name: str, definition: LoopDefinition) -> None:
    """Register a loop type under ``name`` so that cases can use it.

    The loop object provides ``initial_outputs()``, ``update(t, inputs)``, ``get_state()``,
    ``set_state()`` and, for delayed inputs, ``latch(inputs)``. Parameters must be a
    dataclass with ``type`` and ``period`` fields.
    """
    if name in LOOP_TYPES:
        raise ValueError(f"loop type {name!r} is already registered")
    if not is_dataclass(definition.parameters) or not {"type", "period"} <= set(definition.parameters.__dataclass_fields__):
        raise TypeError("loop parameters must be a dataclass with type and period")
    LOOP_SCHEMAS[name] = definition.parameters
    LOOP_TYPES[name] = definition


class AlgorithmLoop:
    def __init__(self, cfg, unit, scenario):
        self.cfg, self.unit, self.scenario = cfg, unit, scenario
        self.obj: Any = None

    def get_state(self):
        return self.obj.get_state() if hasattr(self.obj, "get_state") else {}

    def set_state(self, values):
        if hasattr(self.obj, "set_state"):
            self.obj.set_state(values)
        elif values:
            raise KeyError(f"{self.cfg.type} has no algorithm state")


class PLLLoop(AlgorithmLoop):
    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        self.obj = SRFPLL(cfg.kp_pu, cfg.ki_pu, unit.base.w0, cfg.period, normalisation=cfg.normalisation)

    def initial_outputs(self):
        return {"theta": self.obj.theta, "frame": self.obj.theta, "omega": self.obj.omega}

    def update(self, t, inputs):
        frame = self.obj.theta
        kp = self.scenario.pll_kp(t)
        if kp is not None:
            self.obj.kp = kp
        v = inputs["v"] * cmath.exp(-1j * frame)
        theta, omega = self.obj.update(v)
        return {"theta": theta, "frame": frame, "omega": omega}


class CCLoop(AlgorithmLoop):
    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        x = unit.ac_filter.l_f * unit.base.w0 / unit.base.z_base
        r = unit.ac_filter.r_f / unit.base.z_base
        kp = cfg.kp_pu if cfg.kp_pu is not None else x / unit.base.w0 * 2 * math.pi * cfg.bw_hz
        ki = cfg.ki_pu if cfg.ki_pu is not None else r * 2 * math.pi * cfg.bw_hz
        self.obj = CurrentController(x, r, unit.base.w0, cfg.period, kp, ki, cfg.decoupling, cfg.feedforward, cfg.antiwindup)
        self.extra = 0j

    def initial_outputs(self):
        return {"u_dq": 0j, "extra": 0j}

    def update(self, t, inputs):
        rot = cmath.exp(-1j * inputs["frame"])
        self.extra = inputs["extra"]
        u = self.obj.update(complex(inputs["id_ref"], inputs["iq_ref"]),
                            inputs["i"] * rot, inputs["v"] * rot, inputs["omega"])
        return {"u_dq": u + self.extra, "extra": self.extra}

    def after_rollback(self):
        return self.obj.command(*self.obj._last) + self.extra


class DVCLoop(AlgorithmLoop):
    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        self.obj = DCVoltageController(cfg.kp_pu, cfg.ki_pu, cfg.period, cfg.limit_pu, cfg.bidirectional, cfg.antiwindup)
        self.frozen = False
        self.n_updates = 0

    def initial_outputs(self):
        return {"id_ref": 0.0}

    def update(self, t, inputs):
        self.n_updates += 1
        value = self.obj.update(t, inputs["u_dc"], inputs["vdc_ref"], self.scenario.startup(t) * self.cfg.id0_export_pu,
                                frozen=self.frozen)
        return {"id_ref": value}


class PowerLoop(AlgorithmLoop):
    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        self.obj = PowerCalculator(cfg.bw_hz, cfg.period)

    def initial_outputs(self):
        return {"p": 0.0, "q": 0.0}

    def update(self, t, inputs):
        p, q = self.obj.update(inputs["v"], inputs["i"])
        return {"p": p, "q": q}


class SyncLoop(AlgorithmLoop):
    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        cls = {"psc": PSC, "droop": Droop, "vsg": VSG, "dvoc": DVOC, "matching": Matching}[cfg.type]
        v0 = unit.control.references.v_ref_pu
        if cfg.type == "matching":
            self.obj = cls(cfg, unit.base.w0, cfg.period, v0, unit.control.references.vdc_ref_pu)
        else:
            self.obj = cls(cfg, unit.base.w0, cfg.period, v0)

    def initial_outputs(self):
        return {"theta": float(getattr(self.obj, "theta", 0.0)), "frame": float(getattr(self.obj, "theta", 0.0)),
                "omega": float(getattr(self.obj, "omega", self.unit.base.w0)),
                "v_ref": self.unit.control.references.v_ref_pu}

    def update(self, t, inputs):
        frame = float(getattr(self.obj, "theta", inputs["theta_seed"]))
        v = inputs["v"] * cmath.exp(-1j * frame)
        i = inputs["i"] * cmath.exp(-1j * frame)
        p_ref, q_ref, v_ref = self.scenario.setpoints(t, inputs["p_ref"], inputs["q_ref"], inputs["v_ref"])
        out = self.obj.update(self.cfg.period, inputs["p"], inputs["q"], abs(v),
                              inputs["u_dc"], p_ref, q_ref, v_ref,
                              i_dq=i)
        return {"theta": out.theta, "frame": frame, "omega": out.omega,
                "v_ref": out.v_mag}


class ImpedanceLoop(AlgorithmLoop):
    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        self.obj = VirtualImpedance(cfg.r_v_pu, cfg.x_v_pu, unit.base.w0)

    def initial_outputs(self):
        return {"u_dq": 0j}

    def update(self, t, inputs):
        i = inputs["i"] * cmath.exp(-1j * inputs["frame"])
        return {"u_dq": complex(inputs["v_ref"], 0.0) - self.obj.drop(i, inputs["omega"]) + inputs["extra"]}


class AdmittanceLoop(AlgorithmLoop):
    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        limit = cfg.current_limit_pu if cfg.current_limit_pu > 0 else math.inf
        self.obj = VirtualAdmittance(cfg.r_v_pu, cfg.x_v_pu, unit.base.w0, cfg.period, limit)

    def initial_outputs(self):
        return {"id_ref": 0.0, "iq_ref": 0.0}

    def update(self, t, inputs):
        value = self.obj.update(complex(inputs["v_ref"], 0.0),
                                inputs["v"] * cmath.exp(-1j * inputs["frame"]), inputs["omega"])
        return {"id_ref": value.real, "iq_ref": value.imag}


class DampingLoop(AlgorithmLoop):
    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        self.obj = ActiveDamping(cfg.r_a_pu, cfg.alpha_d, cfg.period)

    def initial_outputs(self):
        return {"extra": 0j}

    def update(self, t, inputs):
        return {"extra": -self.obj(inputs["i"] * cmath.exp(-1j * inputs["frame"]))}


class DelayLoop(AlgorithmLoop):
    def __init__(self, cfg, unit, scenario):
        super().__init__(cfg, unit, scenario)
        initial = cfg.initial_pu if cfg.signal.endswith("_pu") else cfg.initial
        self.value = 0.0 if initial is None else initial
        self.state_name = "value_pu" if cfg.signal.endswith("_pu") else "value"

    def initial_outputs(self):
        return {"value": self.value}

    def update(self, t, inputs):
        return {"value": self.value}

    def latch(self, inputs):
        self.value = inputs["value"]

    def get_state(self):
        return {self.state_name: self.value}

    def set_state(self, values):
        if set(values) - {self.state_name}:
            raise KeyError(f"unit_delay only has {self.state_name} state")
        self.value = values.get(self.state_name, self.value)


def _register(name, inputs, outputs, cls, delayed=frozenset()):
    register_loop_type(name, LoopDefinition(LOOP_SCHEMAS[name], inputs, outputs, cls, delayed))


_register("srf_pll", {"v": V_AB}, {"theta": ANGLE, "frame": ANGLE, "omega": FREQUENCY}, PLLLoop)
_register("dq_current_pi", {"id_ref": CURRENT, "iq_ref": CURRENT, "v": V_AB, "i": I_AB,
                           "frame": ANGLE, "omega": FREQUENCY, "extra": V_DQ},
          {"u_dq": V_DQ, "extra": V_DQ}, CCLoop)
_register("dc_voltage_pi", {"u_dc": DC_VOLTAGE, "vdc_ref": DC_VOLTAGE}, {"id_ref": CURRENT}, DVCLoop)
_register("power", {"v": V_AB, "i": I_AB}, {"p": POWER_PU, "q": POWER_PU}, PowerLoop)
for _name in ("psc", "droop", "vsg", "dvoc", "matching"):
    _register(_name, {"p": POWER_PU, "q": POWER_PU, "v": V_AB, "i": I_AB, "u_dc": DC_VOLTAGE,
                      "p_ref": POWER_PU, "q_ref": POWER_PU, "v_ref": VOLTAGE, "theta_seed": ANGLE},
              {"theta": ANGLE, "frame": ANGLE, "omega": FREQUENCY, "v_ref": VOLTAGE}, SyncLoop)
_register("virtual_impedance", {"v_ref": VOLTAGE, "i": I_AB, "frame": ANGLE, "omega": FREQUENCY,
                                "extra": V_DQ}, {"u_dq": V_DQ}, ImpedanceLoop)
_register("virtual_admittance", {"v_ref": VOLTAGE, "v": V_AB, "frame": ANGLE, "omega": FREQUENCY},
          {"id_ref": CURRENT, "iq_ref": CURRENT}, AdmittanceLoop)
_register("active_damping", {"i": I_AB, "frame": ANGLE}, {"extra": V_DQ}, DampingLoop)
_register("unit_delay", {"value": CURRENT}, {"value": CURRENT}, DelayLoop, frozenset({"value"}))
