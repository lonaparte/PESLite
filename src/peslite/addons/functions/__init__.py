"""Standard namespace for lazily loaded built-in and project-local functions."""

from importlib import import_module

from ...assembly.syntax import register_simulation_reader


class FunctionUnavailableError(ImportError):
    """A cataloged function's implementation provider is not installed."""


_EXPORTS = {
    "energy_closure": ("energy_audit", "energy_closure"),
    "thermal_report": ("thermal", "thermal_report"),
    "compare_thermal_reports": ("thermal", "compare_thermal_reports"),
    "foster_step_response": ("thermal", "foster_step_response"),
    "load_thermalport": ("peslite.addons.components.thermalport", "load_thermalport"),
    "device_loss_report": ("thermal", "device_loss_report"),
    "recorded_device_loss": ("thermal", "recorded_device_loss"),
    "compare_devices": ("thermal", "compare_devices"),
    "circuit_heat_report": ("thermal", "circuit_heat_report"),
    "plot_csv": ("plotting", "plot_csv"),
    "plot_result": ("plotting", "plot_result"),
    "reference_csv": ("reference_signals", "reference_csv"),
    "reference_expression": ("reference_signals", "reference_expression"),
    "reference_table": ("reference_signals", "reference_table"),
    "read_legacy_yaml": ("legacy_yaml", "read_legacy_yaml"),
    "legacy_yaml_preview": ("peslite.gui.legacy_yaml", "preview"),
    "legacy_yaml_patch": ("peslite.gui.legacy_yaml", "validated_patch"),
    "legacy_yaml_add_component": ("peslite.gui.legacy_yaml", "validated_add_component"),
    "legacy_yaml_remove_component": ("peslite.gui.legacy_yaml", "validated_remove_component"),
    "legacy_yaml_rewire": ("peslite.gui.legacy_yaml", "validated_rewire"),
    "reference_prbs": ("identification", "reference_prbs"),
    "reference_multitone": ("identification", "reference_multitone"),
    "reference_chirp": ("identification", "reference_chirp"),
    "detect_operating_point": ("operating_point", "detect_operating_point"),
    "dynamic_performance": ("dynamic", "dynamic_performance"),
    "frequency_response": ("frequency", "frequency_response"),
    "frequency_sweep": ("frequency", "frequency_sweep"),
    "impedance_matrix": ("frequency", "impedance_matrix"),
    "recorded_signal": ("port_measurement", "recorded_signal"),
    "measure_recorded_transfer": ("port_measurement", "measure_recorded_transfer"),
    "measure_recorded_sweep": ("port_measurement", "measure_recorded_sweep"),
    "measure_multitone": ("port_measurement", "measure_multitone"),
    "measure_port_matrix": ("port_measurement", "measure_port_matrix"),
    "run_port_sweep": ("port_measurement", "run_port_sweep"),
    "run_port_matrix": ("port_measurement", "run_port_matrix"),
    "measure_return_ratio": ("loop_gain", "measure_return_ratio"),
    "loop_gain_sweep": ("loop_gain", "loop_gain_sweep"),
    "small_signal": ("stability", "small_signal"),
    "small_signal_report": ("stability", "small_signal_report"),
    "debug_history": ("debug", "debug_history"),
    "debug_root": ("debug", "debug_root"),
    "execute_debug": ("debug", "execute_debug"),
    "parse_debug_text": ("debug", "parse_debug_text"),
    "parse_operation": ("debug", "parse_operation"),
    "prepare_debug": ("debug", "prepare_debug"),
    "read_debug_plan": ("debug", "read_debug_plan"),
    "export_magnetic_circuit": ("magcircuit", "export_magnetic_circuit"),
    "reluctance_inductance": ("magcircuit", "reluctance_inductance"),
    "write_magport": ("magport", "write_magport"),
    "magnetic_field": ("magport", "magnetic_field"),
}
__all__ = list(_EXPORTS)


def _resolve(name):
    """Resolve one public function through the authoritative lazy-export registry."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    qualified = module_name if "." in module_name else f"{__name__}.{module_name}"
    try:
        value = getattr(import_module(qualified), attribute)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == qualified or qualified.startswith(exc.name + ".")):
            raise FunctionUnavailableError(
                f"add-on function {name!r} is unavailable in this installation"
            ) from None
        raise
    globals()[name] = value
    return value


def __getattr__(name):
    """Load a public function from its implementation module when requested."""
    return _resolve(name)


_SIMULATION_READERS = ("read_legacy_yaml",)


@register_simulation_reader
def _read_compatible_simulation(text, path):
    """Try installed syntax-compatibility functions without loading them at startup."""
    for name in _SIMULATION_READERS:
        try:
            reader = _resolve(name)
        except FunctionUnavailableError:
            continue
        parameters = reader(text, path)
        if parameters is not None:
            return parameters
    return None
