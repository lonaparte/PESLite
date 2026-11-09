"""Import modules placed in one add-on search package."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import Iterable


def discover_modules(package: str, paths: Iterable[str]) -> tuple[ModuleType, ...]:
    """Import every module directly contained in ``paths`` in deterministic order.

    Importing a controller or component module executes its normal PESLite registration
    decorators.  Python's module cache makes repeated discovery safe and inexpensive.
    """
    prefix = f"{package}."
    names = sorted(
        info.name for info in iter_modules(paths, prefix)
        if not info.name.rsplit(".", 1)[-1].startswith("_")
    )
    return tuple(import_module(name) for name in names)
