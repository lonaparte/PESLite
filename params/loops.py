"""Control-loop parameter schemas and the ``LOOP_SCHEMAS`` registry keyed by ``type``.

``period`` is a loop's execution period (s). Custom loop types register a schema
in ``LOOP_SCHEMAS`` before a configuration using them is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..phs.protocols import ConfigError
from .schema import (PLLParams, CurrentLoopParams, DCVoltageLoopParams, PSCParams,
                     DroopParams, VSGParams, DVOCParams, MatchingParams, build)


@dataclass(frozen=True, kw_only=True)
class PLLLoopParams(PLLParams):
    period: float
    type: str = "srf_pll"


@dataclass(frozen=True, kw_only=True)
class CCLoopParams(CurrentLoopParams):
    period: float
    type: str = "dq_current_pi"


@dataclass(frozen=True, kw_only=True)
class DVCLoopParams(DCVoltageLoopParams):
    period: float
    limit_pu: float = 1.0
    antiwindup: str = "none"
    type: str = "dc_voltage_pi"

    _quantities = {"limit_pu": "current"}

    _choices = {"antiwindup": ("none", "conditional")}


@dataclass(frozen=True, kw_only=True)
class PowerLoopParams:
    period: float
    bw_hz: float = 0.0
    type: str = "power"


@dataclass(frozen=True, kw_only=True)
class PSCLoopParams:
    period: float
    k_p_pu: float
    k_v: float = 0.0
    k_vi: float = 0.0
    type: str = "psc"

    _quantities = {"k_p_pu": "1/power"}


@dataclass(frozen=True, kw_only=True)
class DroopLoopParams(DroopParams):
    period: float
    type: str = "droop"


@dataclass(frozen=True, kw_only=True)
class VSGLoopParams(VSGParams):
    period: float
    type: str = "vsg"


@dataclass(frozen=True, kw_only=True)
class DVOCLoopParams(DVOCParams):
    period: float
    type: str = "dvoc"


@dataclass(frozen=True, kw_only=True)
class MatchingLoopParams(MatchingParams):
    period: float
    type: str = "matching"


@dataclass(frozen=True, kw_only=True)
class ImpedanceLoopParams:
    r_v_pu: float = 0.0
    x_v_pu: float = 0.0
    period: Optional[float] = None  # None: runs on upstream updates
    type: str = "virtual_impedance"

    _quantities = {"r_v_pu": "resistance", "x_v_pu": "resistance"}


@dataclass(frozen=True, kw_only=True)
class AdmittanceLoopParams:
    period: float
    x_v_pu: float
    r_v_pu: float = 0.0
    current_limit_pu: float = 0.0
    type: str = "virtual_admittance"

    _quantities = {"r_v_pu": "resistance", "x_v_pu": "resistance", "current_limit_pu": "current"}


@dataclass(frozen=True, kw_only=True)
class DampingLoopParams:
    period: float
    r_a_pu: float
    alpha_d: float
    type: str = "active_damping"

    _quantities = {"r_a_pu": "resistance"}


@dataclass(frozen=True, kw_only=True)
class DelayLoopParams:
    period: float
    initial: Optional[float] = None  # angle (rad) or frequency (rad/s)
    initial_pu: Optional[float] = None  # electrical signals
    signal: str = "current_pu"
    type: str = "unit_delay"

    _choices = {"signal": ("current_pu", "voltage_pu", "dc_voltage_pu", "voltage_dq_pu", "voltage_ab_pu", "current_ab_pu",
                            "angle", "frequency", "power_pu")}

    @staticmethod
    def _signal_quantities(values):
        scale = {"current_pu": "current", "current_ab_pu": "current",
                 "voltage_pu": "voltage", "voltage_dq_pu": "voltage", "voltage_ab_pu": "voltage",
                 "dc_voltage_pu": "dc_voltage", "power_pu": "power"}.get(values.get("signal", "current_pu"))
        return {"initial_pu": scale} if scale else {}


LOOP_SCHEMAS = {cls.__dataclass_fields__["type"].default: cls for cls in (
    PLLLoopParams, CCLoopParams, DVCLoopParams, PowerLoopParams, PSCLoopParams,
    DroopLoopParams, VSGLoopParams, DVOCLoopParams, MatchingLoopParams,
    ImpedanceLoopParams, AdmittanceLoopParams, DampingLoopParams, DelayLoopParams)}


def build_loop(data, where):
    from dataclasses import is_dataclass
    if is_dataclass(data):
        from .schema import to_dict
        data = to_dict(data)
    kind = data.get("type") if isinstance(data, dict) else None
    if kind not in LOOP_SCHEMAS:
        raise ConfigError(f"{where}.type: unknown loop type {kind!r}; known: {sorted(LOOP_SCHEMAS)}")
    try:
        return build(LOOP_SCHEMAS[kind], data)
    except ConfigError as exc:
        raise ConfigError(f"{where}.{exc}") from exc
