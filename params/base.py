"""AC/DC per-unit bases and SI/pu conversion of electrical inputs.

AC: Vb = sqrt(2/3)*v_ll_rms (phase peak), Ib = S/(1.5*Vb), Zb = Vb/Ib, w0 = 2*pi*f0.
DC: Vdc = vdc_ref, Idc = S/Vdc, Zdc = Vdc**2/S. L base = Z/w0, C base = 1/(w0*Z).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..phs.protocols import ConfigError

__all__ = ["BaseValues", "DCBase"]


@dataclass(frozen=True)
class DCBase:
    """DC-side bases derived from the rated DC voltage and the power base."""

    v: float  # rated dc voltage (V)
    i: float  # s_base / v (A)
    z: float  # v / i (ohm)
    w0: float

    # pu -> SI
    def L(self, x_pu: float) -> float:
        return x_pu * self.z / self.w0

    def R(self, r_pu: float) -> float:
        return r_pu * self.z

    def C(self, c_pu: float) -> float:
        return c_pu / (self.w0 * self.z)


@dataclass(frozen=True)
class BaseValues:
    """System base (``s_base`` VA, ``v_ll_rms`` V, ``f0`` Hz) and derived phase-peak SI bases."""

    s_base: float
    v_ll_rms: float
    f0: float
    v_phase_peak: float = field(init=False)
    i_phase_peak: float = field(init=False)
    z_base: float = field(init=False)
    w0: float = field(init=False)

    def __post_init__(self) -> None:
        for name in ("s_base", "v_ll_rms", "f0"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"base.{name} must be finite and positive")
        v = math.sqrt(2.0 / 3.0) * self.v_ll_rms
        i = self.s_base / (1.5 * v)
        object.__setattr__(self, "v_phase_peak", v)
        object.__setattr__(self, "i_phase_peak", i)
        object.__setattr__(self, "z_base", v / i)
        object.__setattr__(self, "w0", 2.0 * math.pi * self.f0)

    # pu -> SI
    def L(self, x_pu: float) -> float:
        return x_pu * self.z_base / self.w0

    def R(self, r_pu: float) -> float:
        return r_pu * self.z_base

    def C(self, c_pu: float) -> float:
        return c_pu / (self.w0 * self.z_base)

    def dc(self, vdc_ref: float) -> DCBase:
        """Return the DC base for the rated DC voltage ``vdc_ref`` (V)."""
        i = self.s_base / vdc_ref if vdc_ref > 0.0 else 0.0
        return DCBase(v=vdc_ref, i=i, z=vdc_ref / i if i > 0.0 else 0.0, w0=self.w0)

    def parameter_scales(self, vdc_ref: float = 1.0) -> dict[str, float]:
        """Return the SI value of one pu for each quantity name."""
        dc = self.dc(vdc_ref)
        return {"1": 1.0, "voltage": self.v_phase_peak, "current": self.i_phase_peak,
                "power": self.s_base, "frequency": self.w0,
                "resistance": self.z_base, "inductance": self.L(1.0), "capacitance": self.C(1.0),
                "dc_voltage": dc.v, "dc_current": dc.i, "dc_resistance": dc.z,
                "dc_capacitance": dc.C(1.0)}


def quantity_metadata(cls, name: str) -> dict:
    """Merge the class attribute ``name`` over the class hierarchy into one dict."""
    result = {}
    for parent in reversed(cls.__mro__):
        result.update(parent.__dict__.get(name, {}))
    return result


def quantity_fields(cls, values: dict | None = None) -> dict:
    quantities = quantity_metadata(cls, "_quantities")
    if hasattr(cls, "_signal_quantities"):
        quantities.update(cls._signal_quantities(values or {}))
    return quantities


def quantity_name(cls, name: str, values: dict | None = None) -> str:
    """Return the stored field name for the SI or ``_pu`` spelling of a parameter."""
    name = quantity_metadata(cls, "_input_aliases").get(name, name)
    for target in quantity_fields(cls, values):
        si = target.removesuffix("_pu")
        if name in (si, si + "_pu"):
            return target
    return name


def convert_quantities(data: dict, cls, scales: dict[str, float], where: str) -> dict:
    """Convert SI/pu inputs of one section to its storage units and fill pu defaults.

    Raises ConfigError if both the SI and the pu spelling of a parameter are given.
    """
    aliases = quantity_metadata(cls, "_input_aliases")
    quantities = quantity_fields(cls, data)
    out, seen = {}, {}
    for name, value in data.items():
        spelling = aliases.get(name, name)
        target = quantity_name(cls, name, data)
        if target in seen:
            raise ConfigError(f"{where}.{target}: both {seen[target]!r} and {name!r} specify the same parameter")
        seen[target] = name
        if target in quantities and value is not None:
            if isinstance(value, bool):
                raise ConfigError(f"{where}.{name}: expected a number")
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ConfigError(f"{where}.{name}: expected a number, got {value!r}") from None
            if spelling.endswith("_pu") != target.endswith("_pu"):
                scale = _scale(quantities[target], scales)
                value = value * scale if spelling.endswith("_pu") else value / scale
        out[target] = value
    for name, value in quantity_metadata(cls, "_defaults_pu").items():
        if name not in out:
            out[name] = value if name.endswith("_pu") else value * _scale(quantities[name], scales)
    return out


def _scale(expression: str, scales: dict[str, float]) -> float:
    numerator, *denominators = expression.split("/")
    value = math.prod(scales[name] for name in numerator.split("*"))
    for denominator in denominators:
        value /= math.prod(scales[name] for name in denominator.split("*"))
    return value
