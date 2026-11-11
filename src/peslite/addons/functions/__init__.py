"""Standard namespace for lazily loaded built-in and project-local functions."""

from importlib import import_module

_EXPORTS = {
    "energy_closure": ("energy_audit", "energy_closure", True),
    "plot_csv": ("plotting", "plot_csv", True),
    "plot_result": ("plotting", "plot_result", True),
    "reference_csv": ("reference_signals", "reference_csv", False),
    "reference_expression": ("reference_signals", "reference_expression", False),
    "reference_table": ("reference_signals", "reference_table", False),
    "legacy_yaml_preview": ("peslite.gui.legacy_yaml", "preview", False),
    "legacy_yaml_patch": ("peslite.gui.legacy_yaml", "validated_patch", False),
    "legacy_yaml_add_component": ("peslite.gui.legacy_yaml", "validated_add_component", False),
    "legacy_yaml_remove_component": ("peslite.gui.legacy_yaml", "validated_remove_component", False),
    "legacy_yaml_rewire": ("peslite.gui.legacy_yaml", "validated_rewire", False),
}
__all__ = list(_EXPORTS)
_ROOT_EXPORTS = tuple(name for name, target in _EXPORTS.items() if target[2])


def _resolve(name):
    """Resolve one public function through the authoritative lazy-export registry."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute, _root = _EXPORTS[name]
    qualified = module_name if "." in module_name else f"{__name__}.{module_name}"
    try:
        value = getattr(import_module(qualified), attribute)
    except ModuleNotFoundError as exc:
        if exc.name == qualified:
            raise ModuleNotFoundError(
                f"add-on function {name!r} is unavailable in this installation",
                name=qualified,
            ) from None
        raise
    globals()[name] = value
    return value


def __getattr__(name):
    """Load a public function from its implementation module when requested."""
    return _resolve(name)
