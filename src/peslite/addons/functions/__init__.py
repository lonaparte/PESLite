"""Standard namespace for lazily loaded built-in and project-local functions."""

from importlib import import_module

_EXPORTS = {
    "energy_closure": ("energy_audit", "energy_closure"),
    "plot_csv": ("plotting", "plot_csv"),
    "plot_result": ("plotting", "plot_result"),
    "reference_csv": ("reference_signals", "reference_csv"),
    "reference_expression": ("reference_signals", "reference_expression"),
    "reference_table": ("reference_signals", "reference_table"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    """Load a public function from its implementation module when requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
