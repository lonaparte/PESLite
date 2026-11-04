"""Entry point for optional and custom PESLite component modules.

Place modules in this package and register their parameter dataclasses with
``register_component_type``. PESLite imports each module during package initialisation, so its
types enter the same registry and configuration search path as built-in circuit elements.
"""

from __future__ import annotations

from types import ModuleType

from ... import components as _components
from .._discovery import discover_modules

# Add-on components use the exact public component API and the exact built-in element registry.
globals().update({name: getattr(_components, name) for name in _components.__all__})
__all__ = [*_components.__all__, "discover"]


def discover() -> tuple[ModuleType, ...]:
    """Import component add-ons found on this package's search path."""
    return discover_modules(__name__, __path__)
