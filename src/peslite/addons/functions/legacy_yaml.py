"""Compatibility function for parameter-only YAML simulation files."""

from __future__ import annotations

from pathlib import Path

from ...assembly.syntax import parse_parameter_mapping

__all__ = ["read_simulation"]


def read_simulation(text: str, path: Path):
    """Read the parameter mapping used before native interface views were introduced."""
    return parse_parameter_mapping(text, path)
