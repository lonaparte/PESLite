"""Standard namespace for lazily loaded built-in and project-local functions."""

from importlib import import_module

_EXPORTS = {
    "energy_closure": ("energy_audit", "energy_closure", True),
    "plot_csv": ("plotting", "plot_csv", True),
    "plot_result": ("plotting", "plot_result", True),
    "reference_csv": ("reference_signals", "reference_csv", False),
    "reference_expression": ("reference_signals", "reference_expression", False),
    "reference_table": ("reference_signals", "reference_table", False),
    "reference_prbs": ("identification", "reference_prbs", False),
    "reference_multitone": ("identification", "reference_multitone", False),
    "reference_chirp": ("identification", "reference_chirp", False),
    "detect_operating_point": ("operating_point", "detect_operating_point", False),
    "dynamic_performance": ("dynamic", "dynamic_performance", False),
    "frequency_response": ("frequency", "frequency_response", False),
    "frequency_sweep": ("frequency", "frequency_sweep", False),
    "impedance_matrix": ("frequency", "impedance_matrix", False),
}
__all__ = list(_EXPORTS)
_ROOT_EXPORTS = tuple(name for name, target in _EXPORTS.items() if target[2])


def _resolve(name):
    """Resolve one public function through the authoritative lazy-export registry."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute, _root = _EXPORTS[name]
    qualified = f"{__name__}.{module_name}"
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
