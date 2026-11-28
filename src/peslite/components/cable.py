"""Bergeron and segmented lumped-parameter cable models behind one cable interface."""

from __future__ import annotations

import math
from typing import Any, ClassVar

from .network import Element
from ..solver import Bag, Empty, Port, StoragePort
from ..solver.history import AcceptedSignal, DelayedSignal, DelayHistory
from ..solver.model import ConfigError, record_type

__all__ = ["BergeronCableModel", "LumpedCableModel"]


def _state_record(sections: int) -> type[Bag]:
    names = tuple(f"iL{k}" for k in range(1, sections + 2)) + tuple(
        f"uC{k}" for k in range(1, sections + 1)
    )
    return record_type(f"_LumpedCableState{sections}", names)


class _CableModel(Element):
    """Common two-port boundary of every cable formulation."""

    Output = record_type("CableOutput", ("i1", "i2"))
    configuration_fields: ClassVar[tuple[str, ...]] = ()
    retunable_fields: ClassVar[tuple[str, ...]] = ()
    zero_fields: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = False
    domain_transparent = True
    signal_fields = {"i1": "i1", "i2": "i2"}

    def __init__(self, output: Bag, voltage0: complex) -> None:
        initial = complex(voltage0)
        self.input = Empty()
        self.output = output
        self.port1 = Port(
            self, "1", voltage=initial, current=0j, scale=1.5,
            electrical_domain="ac", standalone="voltage",
        )
        self.port2 = Port(
            self, "2", voltage=initial, current=0j, scale=1.5,
            electrical_domain="ac", standalone="voltage",
        )
        self.ports = (self.port1, self.port2)

    @classmethod
    def from_config(cls, name: str, cfg: Any, _buses: dict, params: Any) -> "_CableModel":
        """Construct and connect one selected cable model."""
        for field in cls.zero_fields:
            if getattr(cfg, field) != 0.0:
                raise ConfigError(
                    f"model={cfg.model!r} requires {field}=0 because that branch "
                    "is not represented by the selected cable model"
                )
        values = {field: getattr(cfg, field) for field in cls.configuration_fields}
        cable = cls(**values, voltage0=complex(params.base.v_phase_peak))
        cable.name, cable.cfg, cable.model = name, cfg, cable
        cable.assembled_port_bindings = ((cfg.bus1, cable.port1), (cfg.bus2, cable.port2))
        return cable

    def retune(self, cfg: Any) -> bool:
        """Update coefficients without reselecting or rebuilding the cable model."""
        changed = tuple(
            field for field in self.configuration_fields
            if getattr(self.cfg, field) != getattr(cfg, field)
        )
        fixed = tuple(field for field in changed if field not in self.retunable_fields)
        if fixed:
            raise ConfigError(
                f"{type(self).__name__} parameters {', '.join(fixed)} "
                "cannot change during a run"
            )
        if changed:
            self.set_parameters(**{
                field: getattr(cfg, field) for field in self.retunable_fields
            })
        self.cfg = cfg
        return bool(changed)


class LumpedCableModel(_CableModel):
    """Segmented two-port R-L/G-C cable; both currents are positive into the cable."""

    configuration_fields = (
        "length_km", "r_per_km", "l_per_km", "g_per_km", "c_per_km", "sections",
    )
    retunable_fields = (
        "length_km", "r_per_km", "l_per_km", "g_per_km", "c_per_km",
    )

    def __init__(self, *, length_km: float, r_per_km: float, l_per_km: float,
                 g_per_km: float, c_per_km: float, sections: int,
                 voltage0: complex) -> None:
        self.sections = sections
        state_type = _state_record(sections)
        self.state_names = state_type.__slots__
        self.state_dimensions = dict.fromkeys(self.state_names, 2)
        initial = {name: 0j for name in self.state_names}
        initial.update({f"uC{k}": complex(voltage0) for k in range(1, sections + 1)})
        self.state = state_type(**initial)
        super().__init__(self.Output(i1=0j, i2=0j), voltage0)
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


class BergeronCableModel(_CableModel):
    """PSCAD-style Bergeron line: distributed LC delay with lumped series resistance."""

    configuration_fields = ("length_km", "r_per_km", "l_per_km", "c_per_km")
    zero_fields = ("g_per_km",)
    Output = record_type(
        "BergeronCableOutput",
        ("i1", "i2", "i_mid", "wave_1_left", "wave_1_right",
         "wave_2_left", "wave_2_right"),
    )
    state_names: ClassVar[tuple[str, ...]] = ()
    energy_states: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, length_km: float, r_per_km: float, l_per_km: float,
                 c_per_km: float, voltage0: complex) -> None:
        self.state = Empty()
        initial = complex(voltage0)
        super().__init__(self.Output(
            i1=0j, i2=0j, i_mid=0j,
            wave_1_left=initial, wave_1_right=initial,
            wave_2_left=initial, wave_2_right=initial,
        ), initial)
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


CABLE_MODELS = {
    "bergeron": BergeronCableModel,
    "lumped": LumpedCableModel,
}
