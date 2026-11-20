"""Bergeron and segmented lumped-parameter cable models behind one cable interface."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar

from .network import Element, NetworkComponent, register_component_type
from ..solver import Bag, ConfigError, Empty, Port, StoragePort
from ..solver.history import AcceptedSignal, DelayedSignal, DelayHistory

__all__ = ["BergeronCableModel", "Cable", "LumpedCableModel"]


class _CableOutput(Bag):
    __slots__ = ("i1", "i2")


@lru_cache(maxsize=None)
def _state_record(sections: int) -> type[Bag]:
    names = tuple(f"iL{k}" for k in range(1, sections + 2)) + tuple(
        f"uC{k}" for k in range(1, sections + 1)
    )
    return type(f"_LumpedCableState{sections}", (Bag,), {"__slots__": names})


class LumpedCableModel(NetworkComponent):
    """Segmented two-port R-L/G-C cable; both currents are positive into the cable."""

    outputs_need_inputs: ClassVar[bool] = False

    def __init__(self, *, length_km: float, r_per_km: float, l_per_km: float,
                 g_per_km: float, c_per_km: float, sections: int,
                 voltage0: complex) -> None:
        self.sections = sections
        self.state_names = _state_record(sections).__slots__
        self.state_dimensions = dict.fromkeys(self.state_names, 2)
        initial = {name: 0j for name in self.state_names}
        initial.update({f"uC{k}": complex(voltage0) for k in range(1, sections + 1)})
        self.state = _state_record(sections)(**initial)
        self.input = Empty()
        self.output = _CableOutput(i1=0j, i2=0j)
        self.port1 = Port(
            self, "1", voltage=complex(voltage0), current=0j, scale=1.5,
            electrical_domain="ac", standalone="voltage"
        )
        self.port2 = Port(
            self, "2", voltage=complex(voltage0), current=0j, scale=1.5,
            electrical_domain="ac", standalone="voltage"
        )
        self.ports = (self.port1, self.port2)
        self.set_parameters(
            length_km=length_km,
            r_per_km=r_per_km,
            l_per_km=l_per_km,
            g_per_km=g_per_km,
            c_per_km=c_per_km,
        )

    def set_parameters(self, *, length_km: float, r_per_km: float, l_per_km: float,
                       g_per_km: float, c_per_km: float) -> None:
        """Set the segmented values and their stored-energy declarations."""
        cell = length_km / self.sections
        end = 0.5 * cell
        lengths = (end, *(cell for _ in range(self.sections - 1)), end)
        self.series_r = tuple(r_per_km * span for span in lengths)
        self.series_l = tuple(l_per_km * span for span in lengths)
        self.shunt_g = g_per_km * cell
        self.shunt_c = c_per_km * cell
        self.storage = tuple(
            StoragePort(
                f"iL{k + 1}", inductance, 1.5,
                ((self.port1, "voltage", 1.0),) if k == 0 else
                ((self.port2, "voltage", -1.0),) if k == self.sections else (),
                "inductor",
            )
            for k, inductance in enumerate(self.series_l)
        ) + tuple(
            StoragePort(f"uC{k + 1}", self.shunt_c, 1.5, kind="capacitor")
            for k in range(self.sections)
        )

    def network(self, nb):
        """Declare the segmented R-L/G-C network."""
        from ..solver import netlist as nl

        nodes = [
            nb.node(self.port1),
            *(nb.internal(2, 1.5) for _ in range(self.sections)),
            nb.node(self.port2),
        ]
        nb.port_output(self, "i1", self.port1, "current")
        nb.port_output(self, "i2", self.port2, "current")
        for index, (resistance, inductance) in enumerate(
                zip(self.series_r, self.series_l)):
            start = nodes[index]
            if resistance:
                inner = nb.internal(2, 1.5)
                nb.add(nl.Resistor(start, inner, resistance))
                start = inner
            nb.add(nl.Inductor(
                start,
                nodes[index + 1],
                inductance,
                nb.state(self, f"iL{index + 1}"),
            ))
        for index in range(1, self.sections + 1):
            nb.add(nl.Capacitor(
                nodes[index], None, self.shunt_c, nb.state(self, f"uC{index}")
            ))
            if self.shunt_g:
                nb.add(nl.Resistor(nodes[index], None, 1.0 / self.shunt_g))
        nb.losses(self)


class _BergeronOutput(Bag):
    __slots__ = ("i1", "i2", "i_mid", "wave_1_left", "wave_1_right",
                 "wave_2_left", "wave_2_right")


class BergeronCableModel(NetworkComponent):
    """PSCAD-style Bergeron line: distributed LC delay with lumped series resistance."""

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    energy_states: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, length_km: float, r_per_km: float, l_per_km: float,
                 c_per_km: float, voltage0: complex) -> None:
        self.state = Empty()
        self.input = Empty()
        initial = complex(voltage0)
        self.output = _BergeronOutput(
            i1=0j, i2=0j, i_mid=0j,
            wave_1_left=initial, wave_1_right=initial,
            wave_2_left=initial, wave_2_right=initial,
        )
        self.port1 = Port(
            self, "1", voltage=initial, current=0j, scale=1.5,
            electrical_domain="ac", standalone="voltage",
        )
        self.port2 = Port(
            self, "2", voltage=initial, current=0j, scale=1.5,
            electrical_domain="ac", standalone="voltage",
        )
        self.ports = (self.port1, self.port2)
        self.length_km = float(length_km)
        self.total_resistance = float(r_per_km * length_km)
        self.surge_impedance = math.sqrt(l_per_km / c_per_km)
        self.travel_time = length_km * math.sqrt(l_per_km * c_per_km)
        self.segment_delay = 0.5 * self.travel_time
        self._histories = tuple(
            DelayHistory(self.segment_delay, initial) for _ in range(4)
        )
        self.accepted_histories = tuple(
            AcceptedSignal(history, self.output, field)
            for history, field in zip(self._histories, (
                "wave_1_left", "wave_1_right", "wave_2_left", "wave_2_right",
            ))
        )
        self.storage = ()

    def solver_maximum_step(self) -> float:
        """Keep delayed reads behind the latest committed solver state."""
        return self.segment_delay

    def stored_energy(self, t: float) -> float:
        """Energy in the two lossless distributed-LC halves."""
        return 1.5 * sum(
            history.integral_squared(t) for history in self._histories
        ) / (4.0 * self.surge_impedance)

    def stored_energy_rate(self, _t: float, _derivatives: dict) -> float:
        """Power entering the distributed storage after its lumped resistive loss."""
        return self.port1.power() + self.port2.power() - self.dissipated_power()

    def dissipated_power(self) -> float:
        """Return only the physical quarter-half-quarter series-resistance loss."""
        resistance = self.total_resistance
        return 1.5 * resistance * (
            0.25 * abs(self.output.i1) ** 2
            + 0.5 * abs(self.output.i_mid) ** 2
            + 0.25 * abs(self.output.i2) ** 2
        )

    def _line_segment(self, nb, left, right, left_history, right_history, prefix: str):
        """Declare one lossless delayed two-port in Norton form."""
        from ..solver import netlist as nl

        impedance = self.surge_impedance
        left_signal = DelayedSignal(right_history, -1.0 / impedance)
        left_resistor = nb.add(nl.Resistor(left, None, impedance, name=f"{prefix}_left_z"))
        left_source = nb.add(nl.CurrentSource(
            left, None, left_signal, left_signal.derivative,
            name=f"{prefix}_left_history",
        ))
        right_signal = DelayedSignal(left_history, -1.0 / impedance)
        right_resistor = nb.add(nl.Resistor(right, None, impedance, name=f"{prefix}_right_z"))
        right_source = nb.add(nl.CurrentSource(
            right, None, right_signal, right_signal.derivative,
            name=f"{prefix}_right_history",
        ))
        nb.linear_output(self, f"wave_{prefix}_left", (
            (1.0, ("node_voltage", left)),
            (impedance, ("branch_current", left_resistor)),
            (impedance, ("branch_current", left_source)),
        ))
        nb.linear_output(self, f"wave_{prefix}_right", (
            (1.0, ("node_voltage", right)),
            (impedance, ("branch_current", right_resistor)),
            (impedance, ("branch_current", right_source)),
        ))
        return left_resistor, left_source

    def network(self, nb):
        """Declare the two delayed LC halves and PSCAD's quarter-half-quarter resistance."""
        from ..solver import netlist as nl

        first, last = nb.node(self.port1), nb.node(self.port2)
        resistance = self.total_resistance
        line_1_left = nb.internal(2, 1.5) if resistance else first
        line_1_right = nb.internal(2, 1.5)
        line_2_left = nb.internal(2, 1.5) if resistance else line_1_right
        line_2_right = nb.internal(2, 1.5) if resistance else last
        if resistance:
            nb.add(nl.Resistor(first, line_1_left, 0.25 * resistance))
            nb.add(nl.Resistor(
                line_1_right, line_2_left, 0.5 * resistance,
            ))
            nb.add(nl.Resistor(line_2_right, last, 0.25 * resistance))
        self._line_segment(
            nb, line_1_left, line_1_right,
            self._histories[0], self._histories[1], "1",
        )
        second_left = self._line_segment(
            nb, line_2_left, line_2_right,
            self._histories[2], self._histories[3], "2",
        )
        nb.linear_output(self, "i_mid", tuple(
            (1.0, ("branch_current", branch)) for branch in second_left
        ))
        nb.port_output(self, "i1", self.port1, "current")
        nb.port_output(self, "i2", self.port2, "current")


@register_component_type
class Cable(Element):
    """Registered two-port cable with Bergeron and segmented lumped models."""

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bus1: str
        bus2: str
        length_km: float
        r_per_km: float
        l_per_km: float
        c_per_km: float
        g_per_km: float = 0.0
        model: str = field(default="bergeron", metadata={"structural": True})
        sections: int = field(default=4, metadata={"structural": True})
        type: str = "cable"

        def __post_init__(self) -> None:
            positive = {
                "length_km": self.length_km,
                "l_per_km": self.l_per_km,
                "c_per_km": self.c_per_km,
            }
            nonnegative = {"r_per_km": self.r_per_km, "g_per_km": self.g_per_km}
            for name, value in positive.items():
                if not math.isfinite(value) or value <= 0.0:
                    raise ConfigError(f"{name} must be finite and > 0, got {value}")
            for name, value in nonnegative.items():
                if not math.isfinite(value) or value < 0.0:
                    raise ConfigError(f"{name} must be finite and >= 0, got {value}")
            if self.model not in ("bergeron", "lumped"):
                raise ConfigError(
                    f"model must be 'bergeron' or 'lumped', got {self.model!r}"
                )
            if self.sections < 1:
                raise ConfigError(f"sections must be >= 1, got {self.sections}")
            if self.model == "bergeron" and self.g_per_km != 0.0:
                raise ConfigError(
                    "model='bergeron' requires g_per_km=0 because the Bergeron loss "
                    "approximation uses lumped series resistance"
                )
            if self.bus1 == self.bus2:
                raise ConfigError("bus1 and bus2 must name different buses")

    type = "cable"
    domain_transparent = True

    def __init__(self, name: str, cfg: Any, buses: dict, p: Any) -> None:
        self.name, self.cfg = name, cfg
        common = dict(
            length_km=cfg.length_km,
            r_per_km=cfg.r_per_km,
            l_per_km=cfg.l_per_km,
            c_per_km=cfg.c_per_km,
            voltage0=complex(p.base.v_phase_peak),
        )
        self.model = (
            BergeronCableModel(**common)
            if cfg.model == "bergeron" else
            LumpedCableModel(
                **common, g_per_km=cfg.g_per_km, sections=cfg.sections,
            )
        )

    def port_bindings(self) -> tuple[tuple[str, Port], ...]:
        return ((self.cfg.bus1, self.model.port1),
                (self.cfg.bus2, self.model.port2))

    def signals(self) -> dict[str, complex]:
        return {f"{self.name}.i1": self.model.output.i1,
                f"{self.name}.i2": self.model.output.i2}

    def retune(self, cfg: Any) -> bool:
        """Apply numerical changes without replacing the selected physical model."""
        if self.cfg == cfg:
            return False
        if isinstance(self.model, LumpedCableModel):
            self.cfg = cfg
            self.model.set_parameters(
                length_km=cfg.length_km,
                r_per_km=cfg.r_per_km,
                l_per_km=cfg.l_per_km,
                g_per_km=cfg.g_per_km,
                c_per_km=cfg.c_per_km,
            )
            return True
        if any(getattr(self.cfg, name) != getattr(cfg, name)
               for name in ("length_km", "r_per_km", "l_per_km", "c_per_km")):
            raise ConfigError(
                "Bergeron cable parameters define its delay history and cannot change during a run"
            )
        self.cfg = cfg
        return False
