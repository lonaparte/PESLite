"""Shared numerical and timing metrics for external reference simulators."""

from __future__ import annotations

import numpy as np


def normalized_errors(actual, expected):
    """Return per-signal mean and maximum errors normalized by reference magnitude."""
    errors = {}
    for name, values in actual.items():
        scale = max(1.0, float(np.max(np.abs(expected[name]))))
        difference = np.abs(values - expected[name]) / scale
        errors[name] = {
            "mean": float(np.mean(difference)),
            "maximum": float(np.max(difference)),
        }
    return errors


def check_comparison(errors, limit, timings, reference, record_property):
    """Record the common accuracy/runtime contract and enforce its maximum error."""
    for signal, values in errors.items():
        record_property(f"{signal}_mean_normalized_difference", values["mean"])
        record_property(f"{signal}_maximum_normalized_difference", values["maximum"])
    mean = float(np.mean([values["mean"] for values in errors.values()]))
    maximum = max(values["maximum"] for values in errors.values())
    record_property("mean_normalized_difference", mean)
    record_property("maximum_normalized_difference", maximum)
    record_property("peslite_simulation_seconds", timings["peslite"])
    record_property(f"{reference}_simulation_seconds", timings[reference])
    record_property(
        "simulation_time_difference_seconds",
        timings["peslite"] - timings[reference],
    )
    record_property(
        f"peslite_to_{reference}_simulation_time_ratio",
        timings["peslite"] / timings[reference],
    )
    assert maximum <= limit, errors
