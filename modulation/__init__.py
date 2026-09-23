"""PWM methods and modulators that turn duty ratios into piecewise-constant switching sequences."""

from .methods import PWM_METHODS, make_pwm_method, spwm, svpwm
from .modulators import (ZOH, CarrierComparison, SampledCarrier, StepAveragedCarrier,
                         SynchronousCarrier, carrier, carrier_position, duty_fraction,
                         make_modulator)

__all__ = ["PWM_METHODS", "make_pwm_method", "spwm", "svpwm",
           "CarrierComparison", "SynchronousCarrier", "ZOH", "StepAveragedCarrier", "SampledCarrier",
           "carrier", "carrier_position", "duty_fraction", "make_modulator"]
