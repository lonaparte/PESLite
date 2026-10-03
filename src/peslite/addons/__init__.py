"""Optional functions and automatically discovered custom PESLite types."""

from __future__ import annotations

from importlib import import_module, invalidate_caches
from pathlib import Path
from typing import Any

from . import components, control
from ._discovery import discover_modules

# These imports happen once when PESLite starts.  They populate the existing registries; no
# directory search or add-on dispatch remains in the simulation hot path.
control.discover()
components.discover()

__all__ = ["control", "components", "functions", "plot_csv", "plot_result"]


def _merge_path(package, directory: Path) -> bool:
    """Append one real directory to a package search path, once."""
    if not directory.is_dir():
        return False
    resolved = str(directory.resolve())
    if resolved not in package.__path__:
        package.__path__.append(resolved)
    return True


def _discover_for(simulation_file: str | Path) -> tuple[Any, ...]:
    """Merge and discover ``PESaddons`` beside one simulation file.

    The installed add-on packages remain the first search locations.  A sibling ``PESaddons``
    contributes the same ``control``, ``components`` and ``functions`` package paths without
    changing the process-wide ``sys.path``.  Controller and component decorators must run before
    the simulation file is parsed; functions remain lazy and are imported only when requested.
    """
    root = Path(simulation_file).resolve().parent / "PESaddons"
    if not root.is_dir():
        return ()

    _merge_path(import_module(__name__), root)
    locations = []
    for package, name in ((control, "control"), (components, "components")):
        directory = root / name
        if _merge_path(package, directory):
            locations.append((package, directory))

    # Import the existing functions package only when this project contributes that namespace.
    # Optional third-party implementations remain lazy; this merely makes project-local function
    # modules importable from the same path.
    function_directory = root / "functions"
    if function_directory.is_dir():
        functions = import_module(f"{__name__}.functions")
        _merge_path(functions, function_directory)
    invalidate_caches()

    imported = []
    for package, directory in locations:
        imported.extend(discover_modules(package.__name__, (str(directory),)))
    return tuple(imported)


def __getattr__(name: str) -> Any:
    """Load optional function implementations only when their API is requested."""
    if name == "functions":
        module = import_module(f"{__name__}.functions")
        globals()[name] = module
        return module
    if name in {"plot_csv", "plot_result"}:
        module = import_module(f"{__name__}.functions")
        globals().update({public: getattr(module, public) for public in ("plot_csv", "plot_result")})
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
