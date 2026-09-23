"""Ideal lossless two-level bridge.

``q`` is the space vector of the phase switching states (or duty ratios).
"""

from __future__ import annotations

from typing import ClassVar

from ..phs.containers import Bag, Empty
from ..phs.protocols import OutputStage

__all__ = ["Bridge"]


class _BridgeInp(Bag):
    __slots__ = ("q", "u_dc", "i_c")


class _BridgeOut(Bag):
    __slots__ = ("u_c", "i_dc")


class Bridge:
    """Lossless bridge: ``u_c = q u_dc``, ``i_dc = 1.5 Re(q conj(i_c))`` (SI).

    ``connections`` and ``zoh_connections`` return the power and switching-state wiring.
    """

    state_names: ClassVar[tuple[str, ...]] = ()
    outputs_need_inputs: ClassVar[bool] = True
    dirac: ClassVar[bool] = True
    output_stages: ClassVar[tuple[OutputStage, ...]] = (
        OutputStage("set_dc_current", inputs=("q", "i_c"), outputs=("i_dc",)),
        OutputStage("set_ac_voltage", inputs=("q", "u_dc"), outputs=("u_c",)),
    )

    def __init__(self) -> None:
        self.state = Empty()
        self.inp = _BridgeInp(q=0j, u_dc=0.0, i_c=0j)
        self.out = _BridgeOut(u_c=0j, i_dc=0.0)

    def set_dc_current(self, t: float) -> None:
        self.out.i_dc = 1.5 * (self.inp.q * self.inp.i_c.conjugate()).real

    def set_ac_voltage(self, t: float) -> None:
        self.out.u_c = self.inp.q * self.inp.u_dc

    def set_outputs(self, t: float) -> None:
        """Evaluate dc current and ac voltage from the current inputs."""
        self.set_dc_current(t)
        self.set_ac_voltage(t)

    def rhs(self, t: float):
        return ()

    def connections(self, dc_link, ac_branch) -> dict:
        """Return the connections of the bridge to ``dc_link`` and ``ac_branch``."""
        return {
            (ac_branch, "u_from"): (self, "u_c"),
            (self, "i_c"): (ac_branch, "i"),
            (dc_link, "i_dc"): (self, "i_dc"),
            (self, "u_dc"): (dc_link, "u_dc"),
        }

    def zoh_connections(self, label: str) -> dict:
        """Return the connection of the held switching state ``label`` to ``q``."""
        return {(self, "q"): label}
