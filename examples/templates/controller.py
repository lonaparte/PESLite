# PESLite 0.5.0
"""Voltage PI control-loop template using the shared sampled and continuous PI block."""

from dataclasses import dataclass
from typing import Optional

from peslite.addons.control import Loop, PI, SignalType, register_loop_type


@register_loop_type
class TemplateController(Loop):
    @dataclass(frozen=True, kw_only=True)
    class Params:
        kp: float = 1.0
        ki: float = 10.0
        period: Optional[float] = None
        type: str = "template_controller"

    type = "template_controller"
    inputs = {"reference": SignalType("voltage"), "voltage": SignalType("voltage")}
    outputs = {"current": SignalType("current")}
    flow_inputs = ("reference", "voltage")

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        self.pi = self.state_block("integral", PI(cfg.kp, cfg.ki, self.block_period))

    def initial_outputs(self):
        return {"current": 0.0}

    def equation(self, reference, voltage):
        return (self.pi(reference, voltage),)
