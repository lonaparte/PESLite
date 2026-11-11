"""Parse the native simulation document and registered compatibility formats."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_VIEW_HEADER = re.compile(r"^@([A-Za-z_][\w-]*)\s*$")
_PARAMETER_HEADER = re.compile(r"^[A-Za-z_][\w-]*:\s*(?:#.*)?$")
_VIEW_DEFINITION = re.compile(
    r"^[A-Za-z_][\w-]*\s*:\s*[A-Za-z_][\w.-]*(?:\s*,\s*[A-Za-z_][\w.-]*)?\s*$"
)
_READERS: list[Callable[[str, Path], Mapping[str, Any] | None]] = []


class SimulationFileError(ValueError):
    """Invalid native syntax or an unavailable source-format addon."""


@dataclass(frozen=True)
class InterfaceView:
    """One source-located graphical view embedded in a simulation file."""

    name: str
    line: int
    source: str


@dataclass(frozen=True)
class SimulationDocument:
    """The parameter tree and graphical views parsed from one source buffer."""

    parameters: dict[str, Any]
    views: tuple[InterfaceView, ...]
    source_format: str = "native"


def register_simulation_reader(
        reader: Callable[[str, Path], Mapping[str, Any] | None],
) -> Callable[[str, Path], Mapping[str, Any] | None]:
    """Register one fallback reader for a non-native simulation-file format."""
    if reader not in _READERS:
        _READERS.append(reader)
    return reader


def is_native_source(text: str) -> bool:
    """Whether a source buffer contains a top-level native view header."""
    return any(line.startswith("@") for line in text.removeprefix("\ufeff").splitlines())


def parse_parameter_mapping(text: str, path: str | Path = "<string>") -> dict[str, Any]:
    """Parse one YAML parameter mapping with source-located errors."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = (f" at line {mark.line + 1}, column {mark.column + 1}"
                    if mark is not None else "")
        message = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        raise SimulationFileError(f"{path}: {message}{location}") from None
    if not isinstance(data, dict):
        raise SimulationFileError(f"{path}: top level must be a mapping")
    return data


def _blank(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith(("\n", "\r")):
        return line[-1]
    return ""


def _starts_parameters(statement: str) -> bool:
    """Distinguish a top-level parameter domain from a typed view declaration."""
    content = statement.split("#", 1)[0].rstrip()
    return bool(_PARAMETER_HEADER.fullmatch(statement) or (
        re.match(r"^[A-Za-z_][\w-]*\s*:", content) and
        not _VIEW_DEFINITION.fullmatch(content)
    ))


def parse_native_document(text: str, path: str | Path = "<string>") -> SimulationDocument:
    """Parse native ``@view`` sections and the remaining YAML parameter domains."""
    source = text.removeprefix("\ufeff")
    lines = source.splitlines(keepends=True)
    parameters = list(lines)
    views: list[InterfaceView] = []
    names: set[str] = set()
    active_name: str | None = None
    active_line = 0
    active_start = 0

    def finish(end: int) -> None:
        nonlocal active_name
        if active_name is None:
            return
        views.append(InterfaceView(active_name, active_line,
                                   "".join(lines[active_start:end]).rstrip("\r\n")))
        active_name = None

    for index, line in enumerate(lines):
        statement = line.rstrip("\r\n")
        if statement.startswith("@"):
            finish(index)
            match = _VIEW_HEADER.fullmatch(statement)
            if match is None:
                raise SimulationFileError(
                    f"{path}: invalid interface view header at line {index + 1}")
            name = match.group(1)
            if name in names:
                raise SimulationFileError(
                    f"{path}: duplicate @{name} view at line {index + 1}")
            names.add(name)
            active_name, active_line, active_start = name, index + 1, index
            parameters[index] = _blank(line)
        elif active_name is not None and _starts_parameters(statement):
            finish(index)
        elif active_name is not None:
            parameters[index] = _blank(line)
    finish(len(lines))
    if "circuit" not in names:
        raise SimulationFileError(f"{path}: native simulation files require an @circuit view")
    tree = parse_parameter_mapping("".join(parameters), path)
    return SimulationDocument(tree, tuple(views))


def parse_simulation_document(text: str, path: str | Path = "<string>") -> SimulationDocument:
    """Parse native syntax, or delegate a non-native buffer to an installed addon."""
    source = Path(path)
    if is_native_source(text):
        return parse_native_document(text, source)
    errors: list[SimulationFileError] = []
    for reader in _READERS:
        try:
            parameters = reader(text, source)
        except SimulationFileError as exc:
            errors.append(exc)
            continue
        if parameters is not None:
            return SimulationDocument(dict(parameters), (), reader.__module__)
    if errors:
        raise errors[0]
    raise SimulationFileError(
        f"{path}: native simulation files require an @circuit view; "
        "this source needs a compatible format addon"
    )


def read_simulation_document(path: str | Path) -> SimulationDocument:
    """Read and parse one native or addon-provided simulation file."""
    source = Path(path)
    return parse_simulation_document(source.read_text(encoding="utf-8"), source)


__all__ = [
    "InterfaceView", "SimulationDocument", "SimulationFileError", "is_native_source",
    "parse_native_document", "parse_parameter_mapping", "parse_simulation_document",
    "read_simulation_document", "register_simulation_reader",
]
