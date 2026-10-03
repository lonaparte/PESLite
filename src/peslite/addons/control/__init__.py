"""Entry point for optional and custom PESLite control-loop modules.

Place modules in this package and register their loop classes with
``register_loop_type``. PESLite imports each module during package initialisation, so its types
enter the same registry and configuration search path as built-in control loops.
"""

from __future__ import annotations

from types import ModuleType

from ... import control as _control
from .._discovery import discover_modules

# An add-on control module gets the same public construction surface as a built-in control module.
# Keeping this list tied to ``control.__all__`` prevents the two entry points drifting.
globals().update({name: getattr(_control, name) for name in _control.__all__})
__all__ = [*_control.__all__, "discover"]


def discover() -> tuple[ModuleType, ...]:
    """Import control add-ons found on this package's search path."""
    return discover_modules(__name__, __path__)
