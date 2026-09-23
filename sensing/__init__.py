"""Sensing: measurement ports, the sampler and the averaging ADC window.

The sampling mode of a converter is set by ``units.<u>.measurement``.
"""

from .sampler import MeasurementPorts, Sampler
from .window import SamplingWindow

__all__ = ["MeasurementPorts", "Sampler", "SamplingWindow"]
