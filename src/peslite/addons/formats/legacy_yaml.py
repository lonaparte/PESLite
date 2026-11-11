"""Compatibility reader for parameter-only YAML simulation files."""

from __future__ import annotations

from pathlib import Path

from ...assembly.simulation_file import parse_parameter_mapping, register_simulation_reader


@register_simulation_reader
def read(text: str, path: Path):
    """Read the parameter mapping used before native interface views were introduced."""
    return parse_parameter_mapping(text, path)
