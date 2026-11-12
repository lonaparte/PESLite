"""Entry point for optional simulation-file format addons."""

from __future__ import annotations

from types import ModuleType

from ...assembly.simulation_file import register_simulation_reader
from .._discovery import discover_modules

__all__ = ["discover", "register_simulation_reader"]


def discover() -> tuple[ModuleType, ...]:
    """Import format addons found on this package's search path."""
    return discover_modules(__name__, __path__)
